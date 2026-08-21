from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Mapping, Sequence

from curve_fx_harness_client.models import CandidateResult

from ..artifacts.io import atomic_write_bytes
from ..evaluation.grouping import CompiledEvaluation, group_evaluations
from ..evaluation.selected import SelectedEvaluator
from ..specs.common import canonical_json_bytes
from ..specs.scenario import ScenarioSpec
from .adapter import SSHProcessAdapter
from .grouped_remote import GroupedRemoteError, GroupedWorkReceipt, GroupedWorkRequest
from .shared_nfs import package_identity_sha256
from .site import SiteProfile


@dataclass(frozen=True, slots=True)
class GroupedDispatch:
    """Results and attestations collected from grouped remote jobs."""
    results_by_evaluation_id: Mapping[str, CandidateResult]
    attestations_by_session_group_id: Mapping[str, Mapping[str, str]]
    requests: int


def dispatch_grouped_evaluations(
    *,
    run_root: Path, run_id: str, selected: SelectedEvaluator,
    evaluations: Sequence[CompiledEvaluation], scenarios: Sequence[ScenarioSpec],
    evaluation_ids_by_blade: Mapping[str, Sequence[str]], repository: Path,
    site: SiteProfile, chunk_size: int, evaluator_workers: int,
    request_namespace: str, ssh: SSHProcessAdapter,
) -> GroupedDispatch:
    """Dispatch exact grouped evaluation coverage across remote blades."""
    values = tuple(evaluations)
    if not values and not any(evaluation_ids_by_blade.values()):
        return GroupedDispatch({}, {}, 0)
    pairs = tuple((item.ordinal, item.evaluation_id) for item in values)
    identifiers = tuple(item.evaluation_id for item in values)
    ordinals = tuple(item.ordinal for item in values)
    if (
        any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
        or any(
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            for ordinal in ordinals
        )
        or len(set(identifiers)) != len(identifiers)
        or len(set(ordinals)) != len(ordinals)
    ):
        raise GroupedRemoteError("grouped evaluations have invalid or duplicate IDs or ordinals")
    site.validate_blades(tuple(evaluation_ids_by_blade))
    active = tuple(blade for blade, assigned in evaluation_ids_by_blade.items() if assigned)
    flattened = tuple(
        identifier
        for blade in active
        for identifier in evaluation_ids_by_blade[blade]
    )
    if len(flattened) != len(set(flattened)) or set(flattened) != set(identifiers):
        raise GroupedRemoteError("grouped blade assignments overlap or lack exact coverage")
    groups = group_evaluations(
        values,
        artifact_sha256=selected.artifact_sha256,
        parameter_schema=selected.compiler.schema,
    )
    group_by_id = {
        item.evaluation_id: group.key.sha256
        for group in groups
        for item in group.evaluations
    }
    package_sha = package_identity_sha256(repository)
    jobs = []
    for blade in active:
        assigned = set(evaluation_ids_by_blade[blade])
        request_values = tuple(item for item in values if item.evaluation_id in assigned)
        request_id = f"{request_namespace}_{blade}"
        request = GroupedWorkRequest(
            run_id, request_id, request_values, tuple(scenarios),
            canonical_json_bytes(selected.provenance), chunk_size, evaluator_workers, package_sha,
        ).validated()
        request_path = run_root / "grouped_requests" / f"{request_id}.json"
        receipt_path = run_root / "grouped_receipts" / f"{request_id}.json"
        atomic_write_bytes(request_path, request.canonical_json)
        jobs.append((blade, request, request_path, receipt_path))

    def dispatch(job: tuple) -> GroupedWorkReceipt:
        blade, request, request_path, receipt_path = job
        argv = [site.cluster.worker_command, "--project-root",
                str(site.cluster.repository_root), "worker", "grouped",
                str(request_path), "--out", str(receipt_path),
                "--remote-run-root", str(site.cluster.remote_run_root),
                "--blade", blade]
        executed = ssh.run_ssh(blade, shlex.join(argv), timeout=site.harness.timeout_seconds)
        if not executed.ok:
            raise GroupedRemoteError(f"grouped worker failed on {blade}: {executed.stderr}")
        receipt = GroupedWorkReceipt.from_json(receipt_path)
        expected = tuple((item.ordinal, item.evaluation_id) for item in request.evaluations)
        observed = tuple((item.ordinal, item.candidate_id) for item in receipt.results)
        expected_groups = {group_by_id[item.evaluation_id] for item in request.evaluations}
        if (
            receipt.request_sha256 != request.sha256
            or receipt.blade != blade
            or receipt.artifact_sha256 != selected.artifact_sha256
            or observed != expected
            or set(receipt.group_session_attestations) != expected_groups
        ):
            raise GroupedRemoteError(f"grouped receipt validation failed on {blade}")
        return receipt

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = tuple(pool.submit(dispatch, job) for job in jobs)
        receipts = tuple(future.result() for future in futures)
    observed = [
        (item.ordinal, item.candidate_id)
        for receipt in receipts
        for item in receipt.results
    ]
    if len(observed) != len(set(observed)) or set(observed) != set(pairs):
        raise GroupedRemoteError("grouped receipts have duplicate or incomplete global coverage")
    results = {item.candidate_id: item for receipt in receipts for item in receipt.results}
    if len(results) != len(values):
        raise GroupedRemoteError("grouped receipts contain duplicate evaluation IDs")
    attestations = {}
    for receipt in receipts:
        for group_id, attestation in receipt.group_session_attestations.items():
            previous = attestations.setdefault(group_id, attestation)
            if previous != attestation:
                raise GroupedRemoteError("one SessionGroup has unequal cross-blade attestations")
    return GroupedDispatch({identifier: results[identifier] for identifier in identifiers},
                           attestations, len(jobs))
