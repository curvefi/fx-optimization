"""Local maxima enumeration and weighted multi-metric ranked maxima over grids.

Semantics are ported from the arb_sim analysis tools
(``find_local_maxima_orjson.py`` / ``find_ranked_maxima.py``) onto the common
:class:`~curve_fx_sim.plotting.heatmap.HeatmapDataset` grid:

* :func:`find_local_maxima` reports plateau-deduplicated local maxima over the
  radius-1 grid neighborhood, optionally restricted to a subset of axes, with
  ``axis`` (cardinal neighbors only) or ``full`` (all adjacent cells)
  connectivity.
* :func:`ranked_maxima` ranks every cell by weighted standard ranks across
  several metrics, honoring per-metric "good" thresholds.

Masked (NaN / failed) cells never count as maxima, and they are treated as
absent when evaluating neighbors: a finite peak next to a failed cell is still
a local maximum.  This intentionally deviates from scipy's NaN-propagating
``maximum_filter`` used by the original tool, which silently discarded such
peaks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from ..plotting.heatmap import HeatmapDataset, MaskSpec


class MaximaError(ValueError):
    """Raised when a grid cannot support local maxima or ranked maxima."""


def _neighborhood_footprint(
    ndim: int,
    connectivity: str,
    radius: int,
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Boolean footprint covering the neighborhood around the center cell."""
    if radius < 1:
        raise MaximaError("neighborhood radius must be >= 1")
    if connectivity not in {"axis", "full"}:
        raise MaximaError("connectivity must be 'axis' or 'full'")
    active = set(range(ndim) if axes is None else axes)
    shape = (2 * radius + 1,) * ndim
    foot = np.zeros(shape, dtype=bool)
    for offset in np.ndindex(shape):
        delta = tuple(index - radius for index in offset)
        if any(d != 0 for axis, d in enumerate(delta) if axis not in active):
            continue
        nonzero_active = sum(
            1 for axis, d in enumerate(delta) if axis in active and d != 0
        )
        include = nonzero_active <= 1 if connectivity == "axis" else True
        if include:
            foot[offset] = True
    return foot


def _finite_neighborhood_maximum(
    values: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    """Neighborhood maximum with masked cells treated as absent (-inf)."""
    finite = np.where(np.isfinite(values), values, -np.inf)
    return ndi.maximum_filter(finite, footprint=footprint, mode="nearest")


def find_local_maxima_candidates(
    metric: Any,
    *,
    axes: Sequence[int] | None = None,
    connectivity: str = "full",
) -> tuple[tuple[int, ...], ...]:
    """Every cell that dominates its neighborhood; plateaus stay unmerged.

    ``axes`` restricts the neighborhood to the listed grid dimensions (default
    all).  Masked cells are excluded and do not disqualify finite neighbors.
    """
    values = np.asarray(metric, dtype=float)
    if values.ndim == 0:
        raise MaximaError("local maxima require a grid metric array")
    if connectivity == "axis":
        finite = np.where(np.isfinite(values), values, -np.inf)
        max_mask = np.isfinite(values)
        active_axes = range(values.ndim) if axes is None else axes
        for axis in active_axes:
            lower_dst = [slice(None)] * values.ndim
            lower_ref = [slice(None)] * values.ndim
            lower_dst[axis] = slice(1, None)
            lower_ref[axis] = slice(None, -1)
            max_mask[tuple(lower_dst)] &= (
                finite[tuple(lower_dst)] >= finite[tuple(lower_ref)]
            )
            upper_dst = [slice(None)] * values.ndim
            upper_ref = [slice(None)] * values.ndim
            upper_dst[axis] = slice(None, -1)
            upper_ref[axis] = slice(1, None)
            max_mask[tuple(upper_dst)] &= (
                finite[tuple(upper_dst)] >= finite[tuple(upper_ref)]
            )
    elif connectivity == "full":
        footprint = _neighborhood_footprint(values.ndim, connectivity, 1, axes)
        maxima = _finite_neighborhood_maximum(values, footprint)
        max_mask = np.isfinite(values) & (values == maxima)
    else:
        raise MaximaError("connectivity must be 'axis' or 'full'")
    return tuple(tuple(int(index) for index in coord) for coord in np.argwhere(max_mask))


def find_local_maxima(
    metric: Any,
    *,
    axes: Sequence[int] | None = None,
    connectivity: str = "full",
) -> tuple[tuple[int, ...], ...]:
    """Plateau-deduplicated local maxima over the grid neighborhood.

    Cells tied at the same value within one connected maximum plateau collapse
    to a single representative: the highest-valued point, with ties broken by
    row-major index order.  ``axes`` restricts the neighborhood to the listed
    grid dimensions; ``connectivity`` selects cardinal-only ("axis") or full
    ("full") adjacency.
    """
    values = np.asarray(metric, dtype=float)
    if values.ndim == 0:
        raise MaximaError("local maxima require a grid metric array")
    footprint = _neighborhood_footprint(values.ndim, connectivity, 1, axes)
    maxima = _finite_neighborhood_maximum(values, footprint)
    max_mask = np.isfinite(values) & (values == maxima)
    labels, count = ndi.label(max_mask, structure=footprint)
    coords: list[tuple[int, ...]] = []
    for label in range(1, count + 1):
        points = np.argwhere(labels == label)
        plateau_values = values[tuple(points.T)]
        best = points[int(np.argmax(plateau_values))]
        coords.append(tuple(int(index) for index in best))
    return tuple(coords)


def rank_values(
    values: Sequence[Any],
    *,
    descending: bool,
    good_threshold: float | None = None,
    good_when_low: bool = False,
) -> list[int]:
    """Standard 1-based ranks with threshold-good and non-finite handling.

    Ported from ``find_ranked_maxima._rank_values``: cells meeting
    ``good_threshold`` receive rank 0 (best), remaining finite cells receive
    consecutive 1-based ranks in value order (ties keep input order and
    consume distinct ranks), and non-finite cells receive the worst rank.
    """
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        parsed.append(number if math.isfinite(number) else math.nan)
    ranks = [0] * len(parsed)
    good_flags = [False] * len(parsed)
    if good_threshold is not None:
        for index, value in enumerate(parsed):
            if not math.isfinite(value):
                continue
            if good_when_low and value <= good_threshold:
                good_flags[index] = True
            if not good_when_low and value >= good_threshold:
                good_flags[index] = True
    valid = [
        (index, value)
        for index, value in enumerate(parsed)
        if math.isfinite(value) and not good_flags[index]
    ]
    valid.sort(key=lambda item: item[1], reverse=descending)
    current = 1
    for index, _ in valid:
        ranks[index] = current
        current += 1
    worst = current if current > 1 else 1
    for index, value in enumerate(parsed):
        if not math.isfinite(value):
            ranks[index] = worst
    return ranks


def coordinates_at(dataset: HeatmapDataset, location: Sequence[int]) -> dict[str, Any]:
    """Combine per-axis coordinate declarations for one grid cell."""
    coordinate: dict[str, Any] = {}
    for axis, index in zip(dataset.axes, location, strict=True):
        coordinate.update(axis.coordinate(int(index)))
    return coordinate


@dataclass(frozen=True)
class RankedMaxima:
    """One grid cell with its weighted multi-metric rank."""

    rank: int
    weighted_score: float
    grid_indices: tuple[int, ...]
    candidate_id: str
    ordinal: int
    coordinates: Mapping[str, Any]
    metrics: Mapping[str, float]
    metric_ranks: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "weighted_score": self.weighted_score,
            "grid_indices": list(self.grid_indices),
            "candidate_id": self.candidate_id,
            "ordinal": self.ordinal,
            "coordinates": dict(self.coordinates),
            "metrics": dict(self.metrics),
            "metric_ranks": dict(self.metric_ranks),
        }


def ranked_maxima(
    dataset: HeatmapDataset,
    *,
    descending: Sequence[str] = (),
    ascending: Sequence[str] = (),
    weights: Mapping[str, float] | None = None,
    thresholds: Mapping[str, float] | None = None,
    top: int | None = None,
    mask: MaskSpec = MaskSpec(),
) -> tuple[RankedMaxima, ...]:
    """Rank every grid cell by weighted standard ranks across several metrics.

    ``descending`` metrics are maximized and ``ascending`` metrics minimized;
    each metric contributes ``weight * rank`` to the score, lower being better.
    Per-metric ``thresholds`` mark cells meeting them as "good" (rank 0).
    Masked (NaN / failed) cells rank worst per metric.  Output is ordered by
    weighted score with ties in row-major index order; ``top`` truncates.
    """
    specifications = [
        *((name, True) for name in descending),
        *((name, False) for name in ascending),
    ]
    if not specifications:
        raise MaximaError("ranked maxima require at least one metric")
    names = [name for name, _ in specifications]
    if len(set(names)) != len(names):
        raise MaximaError("a ranking metric may be declared only once")
    for name in names:
        if name not in dataset.metrics:
            raise MaximaError(f"unknown grid metric {name!r}")
    resolved_weights = {name: float((weights or {}).get(name, 1.0)) for name in names}
    if any(not math.isfinite(value) or value < 0 for value in resolved_weights.values()):
        raise MaximaError("ranking weights must be finite and nonnegative")
    if not any(value > 0 for value in resolved_weights.values()):
        raise MaximaError("at least one ranking weight must be positive")
    resolved_thresholds = {
        name: float((thresholds or {}).get(name, math.nan)) for name in names
    }
    if any(
        not math.isnan(value) and not math.isfinite(value)
        for value in resolved_thresholds.values()
    ):
        raise MaximaError("ranking thresholds must be finite")
    arrays = {name: dataset.metric_array(name, mask) for name in names}
    rank_columns = {
        name: rank_values(
            arrays[name].reshape(-1).tolist(),
            descending=descending_metric,
            good_threshold=resolved_thresholds[name],
            good_when_low=not descending_metric,
        )
        for name, descending_metric in specifications
    }
    entries: list[tuple[float, tuple[int, ...], dict[str, int], dict[str, float]]] = []
    for position, indices in enumerate(np.ndindex(dataset.shape)):
        location = tuple(int(index) for index in indices)
        metric_ranks = {name: rank_columns[name][position] for name in names}
        score = sum(resolved_weights[name] * metric_ranks[name] for name in names)
        metrics = {name: float(arrays[name][location]) for name in names}
        entries.append((score, location, metric_ranks, metrics))
    entries.sort(key=lambda item: item[0])
    output = tuple(
        RankedMaxima(
            rank=position,
            weighted_score=score,
            grid_indices=location,
            candidate_id=str(dataset.candidate_ids[location]),
            ordinal=int(dataset.ordinals[location]),
            coordinates=coordinates_at(dataset, location),
            metrics=metrics,
            metric_ranks=metric_ranks,
        )
        for position, (score, location, metric_ranks, metrics) in enumerate(entries, start=1)
    )
    return output if top is None else output[:top]


__all__ = [
    "MaximaError",
    "RankedMaxima",
    "coordinates_at",
    "find_local_maxima",
    "find_local_maxima_candidates",
    "rank_values",
    "ranked_maxima",
]
