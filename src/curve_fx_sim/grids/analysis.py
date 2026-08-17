"""Deterministic ranking and exact dense-grid inference over evaluation tables."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts.tables import EvaluationRow, EvaluationTable


class GridAnalysisError(ValueError):
    """Raised when table metadata cannot support exact grid analysis."""


def _exact_value_equal(left: Any, right: Any) -> bool:
    """Exact coordinate equality: numeric spellings such as ``1`` and ``"1.00"`` match."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return left == right


@dataclass(frozen=True)
class DenseAxis:
    """One exact dense grid axis inferred from per-row coordinates.

    ``names`` always has length one for coordinates-inferred axes; ``values``
    preserves the declaration order found in the table rows.
    """

    names: tuple[str, ...]
    values: tuple[Any, ...]

    @property
    def key(self) -> str:
        return "/".join(self.names)


def dense_grid_axes(table: EvaluationTable) -> tuple[DenseAxis, ...]:
    """Infer exact dense axes from the table's per-row coordinates (coordinates_json).

    This is the metadata-free counterpart of declared axis metadata: axis names
    are the coordinate keys in first-appearance order and each axis' values are
    the distinct coordinate values in first-appearance order, compared exactly
    (``1`` and ``"1.00"`` denote the same cell).  Raises GridAnalysisError when
    the rows do not share one coordinate namespace.
    """
    ordered_names: list[str] = []
    axes_by_name: dict[str, list[Any]] = {}
    for row in table.rows:
        coordinates = {str(key): value for key, value in (row.coordinates or {}).items()}
        if not ordered_names:
            ordered_names = list(coordinates)
        elif set(ordered_names) != set(coordinates):
            raise GridAnalysisError(
                "evaluation table rows do not share one grid coordinate namespace"
            )
        for name in ordered_names:
            value = coordinates.get(name)
            axis_values = axes_by_name.setdefault(name, [])
            if not any(_exact_value_equal(value, existing) for existing in axis_values):
                axis_values.append(value)
    if not ordered_names:
        raise GridAnalysisError("evaluation table rows declare no grid coordinates")
    return tuple(
        DenseAxis(names=(name,), values=tuple(axes_by_name[name]))
        for name in ordered_names
    )


def dense_grid_indices(
    table: EvaluationTable,
    axes: Sequence[DenseAxis],
) -> Mapping[str, tuple[int, ...]]:
    """Map each candidate id to its exact per-axis index tuple.

    Matches every row coordinate exactly against the inferred axis values;
    raises GridAnalysisError when a coordinate has no declared cell.
    """
    indices: dict[str, tuple[int, ...]] = {}
    for row in table.rows:
        coordinates = {str(key): value for key, value in (row.coordinates or {}).items()}
        location: list[int] = []
        for axis in axes:
            name = axis.names[0]
            value = coordinates.get(name)
            index = next(
                (
                    position
                    for position, candidate in enumerate(axis.values)
                    if _exact_value_equal(candidate, value)
                ),
                None,
            )
            if index is None:
                raise GridAnalysisError(
                    f"candidate {row.candidate_id!r} coordinate {name!r} has no dense axis value"
                )
            location.append(index)
        indices[row.candidate_id] = tuple(location)
    return indices


def first_numeric_metric(metrics: Mapping[str, np.ndarray]) -> str:
    """First metric with at least one finite value, in declaration order."""
    for name, values in metrics.items():
        if bool(np.isfinite(np.asarray(values, dtype=float)).any()):
            return name
    raise GridAnalysisError("no finite numeric metric available")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dense_ranks(values: Sequence[Any], *, descending: bool) -> list[int]:
    parsed = [_number(value) for value in values]
    valid = [(index, value) for index, value in enumerate(parsed) if value is not None]
    valid.sort(key=lambda item: (-item[1] if descending else item[1], item[0]))
    result: dict[int, int] = {}
    last: float | None = None
    rank = -1
    for index, value in valid:
        if last is None or value != last:
            rank += 1
            last = value
        result[index] = rank
    worst = len(valid) + 1
    return [result.get(index, worst) for index in range(len(values))]


@dataclass(frozen=True)
class RankedEvaluation:
    row: EvaluationRow
    weighted_rank: float
    metric_ranks: Mapping[str, int]


def rank_evaluations(
    table: EvaluationTable,
    *,
    descending: Sequence[str] = (),
    ascending: Sequence[str] = (),
    weights: Mapping[str, float] | None = None,
    top: int | None = None,
) -> tuple[RankedEvaluation, ...]:
    """Return stable weighted ranks; non-finite and failed rows rank worst."""
    specifications = [
        *((name, True) for name in descending),
        *((name, False) for name in ascending),
    ]
    if not specifications:
        raise GridAnalysisError("ranking requires at least one metric")
    names = [name for name, _ in specifications]
    if len(set(names)) != len(names):
        raise GridAnalysisError("a ranking metric may be declared only once")
    if top is not None and top < 0:
        raise GridAnalysisError("top must be nonnegative")
    resolved_weights = {name: float((weights or {}).get(name, 1.0)) for name in names}
    if any(not math.isfinite(value) or value < 0 for value in resolved_weights.values()):
        raise GridAnalysisError("ranking weights must be finite and nonnegative")
    if not any(value > 0 for value in resolved_weights.values()):
        raise GridAnalysisError("at least one ranking weight must be positive")

    rank_columns = {
        name: _dense_ranks(
            [row.metrics.get(name) if row.status == "ok" else None for row in table.rows],
            descending=descending_metric,
        )
        for name, descending_metric in specifications
    }
    ranked: list[tuple[float, int, RankedEvaluation]] = []
    for position, row in enumerate(table.rows):
        metric_ranks = {name: rank_columns[name][position] for name in names}
        score = sum(resolved_weights[name] * metric_ranks[name] for name in names)
        ranked.append(
            (
                score,
                row.ordinal,
                RankedEvaluation(row=row, weighted_rank=score, metric_ranks=metric_ranks),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2].row.candidate_id))
    output = tuple(item[2] for item in ranked)
    return output if top is None else output[:top]


__all__ = [
    "DenseAxis",
    "GridAnalysisError",
    "RankedEvaluation",
    "dense_grid_axes",
    "dense_grid_indices",
    "first_numeric_metric",
    "rank_evaluations",
]
