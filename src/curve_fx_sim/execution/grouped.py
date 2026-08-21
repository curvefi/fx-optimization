"""Bounded local execution of portable session groups."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from curve_fx_harness_client import EvaluatorClient
from curve_fx_harness_client.models import (BatchResultFrame, CandidateResult,
                                            CandidateSpec, ObservationSpec)

from ..evaluation.grouping import (LocalSessionGroupBinding, ObservationGroup,
                                   SessionGroup, group_evaluations)
from ..evaluation.selected import SelectedEvaluator
from .collection import normalize_session_attestation


class GroupedExecutionError(RuntimeError):
    """A grouped execution request or result is invalid."""
    pass


@dataclass(frozen=True, slots=True)
class GroupExecutionReceipt:
    """Attestation returned for one executed session group."""
    session_group_id: str
    session_attestation: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class GroupedExecutionResult:
    """Results and attestations returned by grouped local execution."""
    results_by_evaluation_id: Mapping[str, CandidateResult]
    receipts_by_session_group_id: Mapping[str, GroupExecutionReceipt]
    workers: int


def _default_client(selected: SelectedEvaluator, work_dir: Path) -> EvaluatorClient:
    identity = selected.verified_evaluator.identity
    return EvaluatorClient(
        selected.binary_path,
        work_dir=work_dir,
        expected_policy_id=identity.policy_id,
        expected_policy_source_sha256=identity.policy_source_sha256,
        expected_policy_abi=identity.policy_abi,
        expected_policy_parameter_count=identity.policy_parameter_count,
    )


def execute_local_groups(
    selected: SelectedEvaluator,
    groups: Sequence[SessionGroup],
    bind: Callable[[SessionGroup], LocalSessionGroupBinding],
    evaluation_ids: Sequence[str],
    *,
    work_dir: Path,
    chunk_size: int,
    max_workers: int,
    client_factory: Callable[[SelectedEvaluator, Path], Any] = _default_client,
) -> GroupedExecutionResult:
    """Evaluate homogeneous chunks through persistent group-owned client lanes."""
    values = tuple(groups)
    evaluations = tuple(row for group in values for row in group.evaluations)
    if group_evaluations(
        evaluations,
        artifact_sha256=selected.artifact_sha256,
        parameter_schema=selected.compiler.schema,
    ) != values:
        raise GroupedExecutionError("session groups are not canonical")

    ordered = tuple(evaluation_ids)
    by_id = {row.evaluation_id: row for row in evaluations}
    if (
        chunk_size < 1
        or max_workers < 1
        or len(ordered) != len(set(ordered))
        or any(not value or value not in by_id for value in ordered)
    ):
        raise GroupedExecutionError("chunk size, workers, or evaluation IDs are invalid")

    requested = set(ordered)
    active: list[tuple[SessionGroup, list[tuple[ObservationGroup, tuple[Any, ...]]]]] = []
    for group in values:
        chunks: list[tuple[ObservationGroup, tuple[Any, ...]]] = []
        for observation in group.observation_groups:
            rows = [row for row in observation.evaluations if row.evaluation_id in requested]
            chunks.extend(
                (observation, tuple(rows[start : start + chunk_size]))
                for start in range(0, len(rows), chunk_size)
            )
        if chunks:
            active.append((group, chunks))

    bindings = {group.key.sha256: bind(group) for group, _ in active}
    lane_counts = [1] * len(active)
    extra_slots = max(0, max_workers - len(active))
    while extra_slots:
        eligible = [
            index
            for index, (_, chunks) in enumerate(active)
            if len(chunks) >= 2 * (lane_counts[index] + 1)
        ]
        if not eligible:
            break
        index = max(eligible, key=lambda item: len(active[item][1]) / lane_counts[item])
        lane_counts[index] += 1
        extra_slots -= 1

    lanes = [
        (group, chunks[lane::lane_count])
        for (group, chunks), lane_count in zip(active, lane_counts, strict=True)
        for lane in range(lane_count)
    ]
    workers = min(max_workers, len(lanes)) if lanes else 0

    def run_lane(lane):
        group, chunks = lane
        binding = bindings[group.key.sha256]
        if binding.group != group:
            raise GroupedExecutionError("binding callback returned the wrong group")
        if (
            hashlib.sha256(binding.session_request_json).hexdigest()
            != binding.session_request_sha256
        ):
            raise GroupedExecutionError("binding session request hash mismatch")
        request = json.loads(binding.session_request_json)
        client = client_factory(selected, Path(work_dir))
        primary = None
        cleanup = []
        opened = False
        outcome = None
        try:
            hello = client.start()
            identity = selected.verified_evaluator.identity
            if (
                hello.evaluator_identity.binary_sha256 != selected.binary_sha256
                or hello.evaluator_identity.numeric_mode != identity.numeric_mode
            ):
                raise GroupedExecutionError("evaluator hello differs from selected artifact")
            ready = client.open_session(**request)
            opened = True
            attestation = normalize_session_attestation(
                ready, expected_session_id=binding.transport_receipt.session_id
            )
            results = []
            for observation, rows in chunks:
                fragment = observation.key.request_fragment(selected.compiler.schema)
                observation_spec = ObservationSpec.model_validate(fragment.pop("observation", {}))
                projection = fragment.pop("metric_projection", "summary")
                if fragment:
                    raise GroupedExecutionError("unsupported evaluate_batch observation fields")
                candidates = [
                    CandidateSpec(
                        ordinal=row.ordinal,
                        candidate_id=str(row.evaluation_id),
                        policy_params=list(row.candidate.policy_params),
                        pool_overrides=json.loads(row.candidate.pool_overrides_json),
                    )
                    for row in rows
                ]
                response = client.evaluate_batch(
                    candidates,
                    observation=observation_spec,
                    metric_projection=projection,
                )
                _check_batch(response, candidates, binding.transport_receipt.session_id)
                results.extend(response.results)
            outcome = group.key.sha256, attestation, tuple(results)
        except BaseException as exc:  # noqa: BLE001
            primary = exc
        finally:
            if opened:
                try:
                    client.close_session(binding.transport_receipt.session_id)
                except BaseException as exc:  # noqa: BLE001
                    cleanup.append(exc)
            try:
                client.shutdown()
            except BaseException as exc:  # noqa: BLE001
                cleanup.append(exc)
        failures = ([primary] if primary is not None else []) + cleanup
        if len(failures) > 1:
            raise BaseExceptionGroup("group execution and cleanup failed", failures)
        if failures:
            raise failures[0].with_traceback(failures[0].__traceback__)
        return outcome

    outcomes, failures = [], []
    if lanes:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(run_lane, lane) for lane in lanes]:
                try:
                    outcomes.append(future.result())
                except BaseException as exc:  # noqa: BLE001
                    failures.append(exc)
    if failures:
        raise BaseExceptionGroup("grouped evaluation failed: " + "; ".join(map(str, failures)), failures)

    results, attestations = {}, {}
    for group_id, attestation, batch in outcomes:
        previous = attestations.setdefault(group_id, attestation)
        if previous != attestation:
            raise GroupedExecutionError("one session group returned different attestations")
        for row in batch:
            if row.candidate_id in results:
                raise GroupedExecutionError("duplicate evaluation result")
            results[row.candidate_id] = row
    if set(results) != requested:
        raise GroupedExecutionError("group execution did not return exact coverage")
    receipts = {
        group_id: GroupExecutionReceipt(group_id, attestation)
        for group_id, attestation in attestations.items()
    }
    return GroupedExecutionResult({value: results[value] for value in ordered}, receipts, workers)


def _check_batch(response: Any, candidates: Sequence[CandidateSpec], session_id: str) -> None:
    expected = sorted((row.ordinal, row.candidate_id) for row in candidates)
    actual = (
        [(row.ordinal, row.candidate_id) for row in response.results]
        if isinstance(response, BatchResultFrame)
        else []
    )
    if (
        not isinstance(response, BatchResultFrame)
        or response.status != "complete"
        or response.session_id != session_id
        or actual != expected
        or len(actual) != len(set(actual))
    ):
        raise GroupedExecutionError("evaluator batch session or coverage/order mismatch")


__all__ = [
    "GroupExecutionReceipt",
    "GroupedExecutionError",
    "GroupedExecutionResult",
    "execute_local_groups"]
