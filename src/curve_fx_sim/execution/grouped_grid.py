"""Bounded plan-backed grid execution and shared-NFS dispatch."""

from __future__ import annotations

import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from curve_fx_harness_client import EvaluatorClient
from curve_fx_harness_client.models import BatchResultFrame, CandidateSpec, ObservationSpec

from ..artifacts.io import atomic_write_bytes, sha256_path
from ..evaluation.grouping import bind_local_session_group, group_evaluations
from ..evaluation.selected import SelectedEvaluator
from ..evaluation.session import LocalSessionMaterialization
from ..grids.model import CartesianGridPlan
from ..specs.scenario import ScenarioSpec
from .adapter import SSHProcessAdapter
from .collection import (
    load_grid_shard_receipt,
    normalize_session_attestation,
    write_grid_shard_result,
)
from .site import SiteProfile


@dataclass(frozen=True, slots=True)
class GridExecution:
    receipt_paths: tuple[Path, ...]
    evaluator_sessions: int


@dataclass(slots=True)
class _OpenSession:
    client: Any
    session_id: str
    session_key: Any
    attestation: Mapping[str, str]


def _new_client(selected: SelectedEvaluator, work_dir: Path, workers: int, timeout: float) -> EvaluatorClient:
    identity = selected.verified_evaluator.identity
    return EvaluatorClient(
        selected.binary_path, work_dir=work_dir,
        expected_policy_id=identity.policy_id,
        expected_policy_source_sha256=identity.policy_source_sha256,
        expected_policy_abi=identity.policy_abi,
        expected_policy_parameter_count=identity.policy_parameter_count,
        launch_argv=[selected.binary_path, "serve", "--workers", str(workers)],
        timeout=timeout,
    )


def execute_grid_plan(
    *, manifest: Mapping[str, Any], plan: CartesianGridPlan,
    selected: SelectedEvaluator, scenario: ScenarioSpec,
    shards: Sequence[Mapping[str, Any]], repository: Path, work_dir: Path,
    blade: str, work_request_sha256: str, evaluator_workers: int,
    timeout_seconds: float,
    client_factory: Callable[[SelectedEvaluator, Path], Any] | None = None,
) -> GridExecution:
    """Compile one shard at a time while retaining one client per session group."""
    sessions: dict[str, _OpenSession] = {}
    receipts: list[Path] = []
    primary: BaseException | None = None
    try:
        for descriptor in shards:
            ranges = tuple((int(start), int(end)) for start, end in descriptor["ranges"])
            points = tuple(plan.iter_points(ranges))
            groups = group_evaluations(
                tuple(point.evaluation for point in points),
                artifact_sha256=selected.artifact_sha256,
                parameter_schema=selected.compiler.schema,
            )
            rows, shard_attestations = [], {}
            for group in groups:
                group_id = group.key.sha256
                opened = sessions.get(group_id)
                if opened is None:
                    materialized = LocalSessionMaterialization.from_scenario(
                        scenario, repository=repository,
                        manifest_root=work_dir / "session_transport" / group_id,
                        session_id=f"sess_{manifest['run_id']}_{group_id[:12]}",
                    ).validated()
                    binding = bind_local_session_group(group, materialized)
                    client = (
                        client_factory(selected, work_dir)
                        if client_factory is not None else
                        _new_client(selected, work_dir, evaluator_workers, timeout_seconds)
                    )
                    hello = client.start()
                    identity = selected.verified_evaluator.identity
                    if (hello.evaluator_identity.binary_sha256 != selected.binary_sha256
                            or hello.evaluator_identity.numeric_mode != identity.numeric_mode):
                        raise RuntimeError("grid evaluator hello differs from selected artifact")
                    ready = client.open_session(**json.loads(binding.session_request_json))
                    opened = _OpenSession(
                        client, binding.transport_receipt.session_id, group.session_key,
                        normalize_session_attestation(
                            ready, expected_session_id=binding.transport_receipt.session_id),
                    )
                    sessions[group_id] = opened
                elif opened.session_key != group.session_key:
                    raise RuntimeError("grid session group identity changed between shards")
                shard_attestations[group_id] = opened.attestation
                for observation in group.observation_groups:
                    fragment = observation.key.request_fragment(selected.compiler.schema)
                    observation_spec = ObservationSpec.model_validate(fragment.pop("observation", {}))
                    projection = fragment.pop("metric_projection", "summary")
                    if fragment:
                        raise RuntimeError("unsupported grid observation fields")
                    candidates = [
                        CandidateSpec(
                            ordinal=item.ordinal, candidate_id=str(item.evaluation_id),
                            policy_params=list(item.candidate.policy_params),
                            pool_overrides=json.loads(item.candidate.pool_overrides_json),
                        )
                        for item in observation.evaluations
                    ]
                    response = opened.client.evaluate_batch(
                        candidates, observation=observation_spec,
                        metric_projection=projection,
                    )
                    expected = sorted((item.ordinal, item.candidate_id) for item in candidates)
                    observed = (
                        [(item.ordinal, item.candidate_id) for item in response.results]
                        if isinstance(response, BatchResultFrame) else []
                    )
                    if (not isinstance(response, BatchResultFrame)
                            or response.status != "complete"
                            or response.session_id != opened.session_id
                            or observed != expected or len(observed) != len(set(observed))):
                        raise RuntimeError("grid evaluator batch coverage or session mismatch")
                    rows.extend(response.results)
            shard_id = str(descriptor["shard_id"])
            npz_path = work_dir / "results" / f"{shard_id}.npz"
            receipt_path = work_dir / "results" / f"{shard_id}.receipt.json"
            write_grid_shard_result(
                npz_path, receipt_path, manifest=manifest, plan=plan,
                shard=descriptor, blade=blade,
                work_request_sha256=work_request_sha256,
                session_attestations=shard_attestations, results=rows,
            )
            receipts.append(receipt_path)
    except BaseException as exc:  # noqa: BLE001
        primary = exc
    cleanup: list[BaseException] = []
    for opened in sessions.values():
        try:
            opened.client.close_session(opened.session_id)
        except BaseException as exc:  # noqa: BLE001
            cleanup.append(exc)
        try:
            opened.client.shutdown()
        except BaseException as exc:  # noqa: BLE001
            cleanup.append(exc)
    failures = ([primary] if primary is not None else []) + cleanup
    if len(failures) > 1:
        raise BaseExceptionGroup("grid execution and cleanup failed", failures)
    if failures:
        raise failures[0].with_traceback(failures[0].__traceback__)
    return GridExecution(tuple(receipts), len(sessions))


def dispatch_grid_requests(
    *, run_root: Path, requests_by_blade: Mapping[str, Any],
    manifest: Mapping[str, Any], plan: CartesianGridPlan,
    repository: Path, site: SiteProfile, ssh: SSHProcessAdapter,
) -> int:
    """Publish one compact request per active blade and verify shard receipts."""
    from .grouped_remote import GridWorkReceipt

    site.validate_blades(tuple(requests_by_blade))
    jobs = []
    for blade, request in requests_by_blade.items():
        request_path = run_root / "grouped_requests" / f"{request.request_id}.json"
        receipt_path = run_root / "grouped_receipts" / f"{request.request_id}.json"
        atomic_write_bytes(request_path, request.canonical_json)
        jobs.append((blade, request, request_path, receipt_path))

    def dispatch(job: tuple[Any, ...]) -> None:
        blade, request, request_path, receipt_path = job
        argv = [
            site.cluster.worker_command, "--project-root", str(site.cluster.repository_root),
            "worker", "grid", str(request_path), "--out", str(receipt_path),
            "--remote-run-root", str(site.cluster.remote_run_root), "--blade", blade,
        ]
        result = ssh.run_ssh(blade, shlex.join(argv), timeout=site.harness.timeout_seconds)
        if not result.ok:
            raise RuntimeError(f"grid worker failed on {blade}: {result.stderr}")
        receipt = GridWorkReceipt.from_json(receipt_path)
        expected_ids = tuple(item["shard_id"] for item in request.shards)
        if (receipt.request_sha256 != request.sha256 or receipt.run_id != request.run_id
                or receipt.request_id != request.request_id or receipt.blade != blade
                or receipt.plan_sha256 != request.plan_sha256
                or receipt.artifact_sha256 != request.artifact_sha256
                or tuple(item["shard_id"] for item in receipt.shards) != expected_ids):
            raise RuntimeError(f"grid receipt validation failed on {blade}")
        by_id = {item["shard_id"]: item for item in request.shards}
        for item in receipt.shards:
            path = run_root / item["receipt_path"]
            if sha256_path(path) != item["receipt_sha256"]:
                raise RuntimeError(f"grid shard receipt hash mismatch on {blade}")
            load_grid_shard_receipt(
                path, manifest=manifest, plan=plan, shard=by_id[item["shard_id"]],
            )

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        for future in (pool.submit(dispatch, job) for job in jobs):
            future.result()
    return len(jobs)


__all__ = ["GridExecution", "dispatch_grid_requests", "execute_grid_plan"]
