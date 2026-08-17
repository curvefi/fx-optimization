"""Exact grid-result coverage and common evaluation-table collection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..artifacts.tables import EvaluationRow, EvaluationTable, MetricProjection
from curve_fx_harness_client.models import BatchResultFrame, CandidateResult
from .model import GridPoint


class GridCoverageError(ValueError):
    """Raised when evaluator results are not an exact cover of grid requests."""


def _flatten_results(
    batches: Sequence[BatchResultFrame | CandidateResult]
    | BatchResultFrame,
) -> tuple[CandidateResult, ...]:
    if isinstance(batches, BatchResultFrame):
        if batches.status != "complete":
            raise GridCoverageError(f"evaluation batch status is {batches.status!r}")
        return tuple(batches.results)
    output: list[CandidateResult] = []
    for item in batches:
        if isinstance(item, BatchResultFrame):
            if item.status != "complete":
                raise GridCoverageError(f"evaluation batch status is {item.status!r}")
            output.extend(item.results)
        elif isinstance(item, CandidateResult):
            output.append(item)
        else:
            raise TypeError(f"unsupported evaluator result {type(item).__name__}")
    return tuple(output)


def collect_evaluations(
    points: Sequence[GridPoint],
    batches: Sequence[BatchResultFrame | CandidateResult] | BatchResultFrame,
    *,
    metric_projection: MetricProjection,
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationTable:
    """Collect exactly one evaluator result per grid point.

    Coverage is checked by both candidate identity and canonical ordinal.  A
    failed candidate remains a row; missing, duplicated, foreign, or reordered
    identities are rejected rather than inferred from position.
    """
    if metric_projection is None:
        raise GridCoverageError("grid collection requires an explicit MetricProjection")
    expected_by_id = {point.candidate_id: point for point in points}
    if len(expected_by_id) != len(points):
        raise GridCoverageError("grid point candidate ids are not unique")
    expected_ordinals = {point.ordinal for point in points}
    if expected_ordinals != set(range(len(points))):
        raise GridCoverageError("grid point ordinals are not the complete canonical range")

    results = _flatten_results(batches)
    observed: dict[str, CandidateResult] = {}
    observed_ordinals: set[int] = set()
    for result in results:
        point = expected_by_id.get(result.candidate_id)
        if point is None:
            raise GridCoverageError(f"foreign candidate result {result.candidate_id!r}")
        if result.candidate_id in observed:
            raise GridCoverageError(f"duplicate candidate result {result.candidate_id!r}")
        if result.ordinal != point.ordinal:
            raise GridCoverageError(
                f"candidate {result.candidate_id!r} returned ordinal {result.ordinal}, "
                f"expected {point.ordinal}"
            )
        if result.ordinal in observed_ordinals:
            raise GridCoverageError(f"duplicate result ordinal {result.ordinal}")
        if result.status == "ok" and not result.economic_fingerprint:
            raise GridCoverageError(
                f"successful candidate {result.candidate_id!r} has no economic fingerprint"
            )
        observed[result.candidate_id] = result
        observed_ordinals.add(result.ordinal)

    missing = [point.candidate_id for point in points if point.candidate_id not in observed]
    if missing:
        preview = ", ".join(missing[:5])
        raise GridCoverageError(
            f"grid evaluation is missing {len(missing)} candidates: {preview}"
        )
    if len(results) != len(points):
        raise GridCoverageError(
            f"grid result count {len(results)} does not equal point count {len(points)}"
        )

    rows: list[EvaluationRow] = []
    coordinate_indices: dict[str, list[int]] = {}
    for point in sorted(points, key=lambda item: item.ordinal):
        result = observed[point.candidate_id]
        artifacts = result.artifacts
        projected_metrics = metric_projection.project_metrics(result.metrics)
        if result.status == "ok":
            missing_metrics = [
                field for field in metric_projection.fields if field not in projected_metrics
            ]
            if missing_metrics:
                raise GridCoverageError(
                    f"candidate {point.candidate_id!r} is missing projected metrics: {missing_metrics!r}"
                )
        rows.append(
            EvaluationRow(
                candidate_id=point.candidate_id,
                ordinal=point.ordinal,
                coordinates=dict(point.coordinates),
                params={"vector": list(point.policy_params)},
                pool_overrides=dict(point.pool_overrides),
                metrics=projected_metrics,
                status=result.status,
                economic_fingerprint=result.economic_fingerprint or None,
                trace_path=artifacts.trace_path if artifacts else None,
                actions_path=artifacts.actions_path if artifacts else None,
                tags=("grid",),
            )
        )
        coordinate_indices[point.candidate_id] = list(point.coordinate_indices)
    table_metadata = dict(metadata or {})
    table_metadata.update(
        {
            "source_kind": "grid",
            "coordinate_indices": coordinate_indices,
            "coverage": {
                "expected": len(points),
                "observed": len(results),
                "complete": True,
            },
        }
    )
    return EvaluationTable(
        rows=rows,
        metadata=table_metadata,
        metric_projection=metric_projection,
    )


def merge_evaluation_tables(
    tables: Sequence[EvaluationTable],
    *,
    expected_candidate_ids: Sequence[str],
    metric_projection: MetricProjection,
) -> EvaluationTable:
    """Merge disjoint shard tables with exact candidate and projection coverage."""
    expected = tuple(expected_candidate_ids)
    if len(set(expected)) != len(expected):
        raise GridCoverageError("expected candidate ids are duplicated")
    rows: dict[str, EvaluationRow] = {}
    coordinate_indices: dict[str, Any] = {}
    shared_metadata: dict[str, Any] | None = None
    for table in tables:
        if table.metric_projection != metric_projection:
            raise GridCoverageError("shard MetricProjection differs from grid projection")
        raw_indices = table.metadata.get("coordinate_indices", {})
        if isinstance(raw_indices, Mapping):
            coordinate_indices.update(raw_indices)
        shard_metadata = {
            key: value
            for key, value in table.metadata.items()
            if key not in {"coordinate_indices", "coverage"}
        }
        if shared_metadata is None:
            shared_metadata = shard_metadata
        elif shard_metadata != shared_metadata:
            raise GridCoverageError("shard table metadata differs across grid shards")
        for row in table.rows:
            if row.candidate_id in rows:
                raise GridCoverageError(f"candidate {row.candidate_id!r} appears in multiple shards")
            rows[row.candidate_id] = row
    missing_indices = [candidate_id for candidate_id in expected if candidate_id not in coordinate_indices]
    if missing_indices:
        raise GridCoverageError(
            f"shard coordinate coverage is missing {missing_indices[:5]!r}"
        )
    unknown = sorted(set(rows) - set(expected))
    missing = [candidate_id for candidate_id in expected if candidate_id not in rows]
    if unknown or missing:
        raise GridCoverageError(
            f"shard coverage mismatch: missing={missing[:5]!r}, unknown={unknown[:5]!r}"
        )
    ordered = [rows[candidate_id] for candidate_id in expected]
    metadata = dict(shared_metadata or {})
    metadata.update(
        {
            "source_kind": "grid",
            "coordinate_indices": coordinate_indices,
            "coverage": {
                "expected": len(expected),
                "observed": len(ordered),
                "complete": True,
            },
        }
    )
    return EvaluationTable(
        rows=ordered,
        metadata=metadata,
        metric_projection=metric_projection,
    )


__all__ = [
    "GridCoverageError",
    "collect_evaluations",
    "merge_evaluation_tables",
]
