"""Exact evaluation-table data and the maintained tiled heatmap view."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .masked_metrics import (
    is_masked_metric,
    masked_metric_slippage_source,
    masked_metric_source,
    masked_metric_uses_detach,
    masked_metric_uses_slippage,
)

@dataclass(frozen=True)
class SelectionRef:
    run_id: str
    kind: str
    index: int
    coordinate: Mapping[str, Any]
    candidate_id: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "kind": self.kind, "index": self.index,
            "coordinate": dict(self.coordinate), "candidate_id": self.candidate_id,
            "tags": list(self.tags),
        }


def atomic_write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

AxisScale = Literal["linear", "log", "categorical"]

class HeatmapValidationError(ValueError):
    """Raised when table coordinates cannot form one exact dense heatmap."""


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else format(value, "f")
    if isinstance(value, tuple):
        return [_python_value(item) for item in value]
    if isinstance(value, list):
        return [_python_value(item) for item in value]
    return value


def _labels(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        " / ".join(str(_python_value(item)) for item in value)
        if isinstance(value, (tuple, list))
        else str(_python_value(value))
        for value in values
    )


def auto_log(values: Sequence[Any]) -> bool:
    if len(values) < 3:
        return False
    try:
        numbers = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    if np.any(~np.isfinite(numbers)) or np.any(numbers <= 0):
        return False
    log_diffs = np.diff(np.log(numbers))
    linear_diffs = np.diff(numbers)
    if np.any(log_diffs <= 0) or np.any(linear_diffs <= 0):
        return False
    log_mean = float(np.mean(log_diffs))
    linear_mean = float(np.mean(linear_diffs))
    if log_mean <= 0 or linear_mean <= 0:
        return False
    log_cv = float(np.std(log_diffs) / log_mean)
    linear_cv = float(np.std(linear_diffs) / linear_mean)
    if log_cv < 0.05 and log_cv < linear_cv:
        return True
    # A forced-included value may perturb two gaps in an otherwise geometric grid.
    median = float(np.median(log_diffs))
    if median <= 0:
        return False
    close_share = float(np.mean(np.abs(log_diffs / median - 1.0) < 0.15))
    return close_share >= 0.85 and log_cv < linear_cv


def _inferred_scale(values: Sequence[Any]) -> AxisScale:
    """Linear/log/categorical detection for coordinates-inferred axes.

    Mirrors the metadata detection rules: non-numeric values are categorical,
    positive geometric progressions are logarithmic, everything else linear.
    """
    try:
        numbers = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return "categorical"
    if np.any(~np.isfinite(numbers)):
        return "categorical"
    if auto_log(values):
        return "log"
    return "linear"


@dataclass(frozen=True)
class HeatmapAxis:
    names: tuple[str, ...]
    values: tuple[Any, ...]
    scale: AxisScale = "linear"
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.names or any(not name for name in self.names):
            raise HeatmapValidationError("heatmap axis requires non-empty names")
        if not self.values:
            raise HeatmapValidationError(f"heatmap axis {self.key!r} is empty")
        if len(set(self.names)) != len(self.names):
            raise HeatmapValidationError(f"heatmap axis names are duplicated: {self.names!r}")
        if self.labels and len(self.labels) != len(self.values):
            raise HeatmapValidationError(f"heatmap axis {self.key!r} label count is invalid")
        if len(self.names) > 1:
            for row in self.values:
                if not isinstance(row, (tuple, list)) or len(row) != len(self.names):
                    raise HeatmapValidationError(
                        f"coupled axis {self.key!r} row does not match its names"
                    )
        elif self.scale in {"linear", "log"}:
            try:
                values = np.asarray(self.values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise HeatmapValidationError(
                    f"numeric axis {self.key!r} contains non-numeric values"
                ) from exc
            if np.any(~np.isfinite(values)):
                raise HeatmapValidationError(f"numeric axis {self.key!r} must be finite")
            exact_values = {Decimal(str(value)) for value in self.values}
            if len(exact_values) != len(self.values):
                raise HeatmapValidationError(f"numeric axis {self.key!r} must be unique")
            if self.scale == "log" and np.any(values <= 0):
                raise HeatmapValidationError(f"log axis {self.key!r} must be finite and positive")
        if self.scale not in {"linear", "log", "categorical"}:
            raise HeatmapValidationError(f"unsupported axis scale {self.scale!r}")

    @property
    def key(self) -> str:
        return "/".join(self.names)

    @property
    def is_coupled(self) -> bool:
        return len(self.names) > 1

    @property
    def is_singleton(self) -> bool:
        return len(self.values) == 1

    @property
    def display_labels(self) -> tuple[str, ...]:
        return self.labels or _labels(self.values)

    def coordinate(self, index: int) -> dict[str, Any]:
        value = self.values[index]
        if self.is_coupled:
            return {
                name: _python_value(item)
                for name, item in zip(self.names, value, strict=True)
            }
        return {self.names[0]: _python_value(value)}

    @classmethod
    def from_metadata(cls, raw: Mapping[str, Any]) -> "HeatmapAxis":
        if not isinstance(raw, Mapping):
            raise HeatmapValidationError("heatmap axis metadata must be an object")
        names_raw = raw.get("names") or ()
        if not names_raw:
            name = raw.get("name")
            names_raw = (name,) if name else ()
        if isinstance(names_raw, str):
            names_raw = (names_raw,)
        if not isinstance(names_raw, Sequence) or isinstance(names_raw, (bytes, bytearray)):
            raise HeatmapValidationError("heatmap axis names metadata must be an array")
        names = tuple(str(name) for name in names_raw)
        rows = raw.get("rows")
        values = raw.get("values")
        # AxisSpec serialization retains the inactive representation as an
        # empty array. Treat only populated representations as declarations.
        rows = None if rows == [] else rows
        values = None if values == [] else values
        if rows is not None and values is not None:
            raise HeatmapValidationError("heatmap axis metadata cannot declare both rows and values")
        if rows is not None:
            if len(names) <= 1:
                raise HeatmapValidationError("heatmap axis rows require coupled axis names")
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
                raise HeatmapValidationError("heatmap axis rows metadata must be an array")
            axis_values = tuple(
                tuple(row)
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray))
                else row
                for row in rows
            )
        else:
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise HeatmapValidationError("heatmap axis values metadata must be an array")
            axis_values = tuple(values)
        generation = raw.get("generation", {})
        if not isinstance(generation, Mapping):
            raise HeatmapValidationError("heatmap axis generation metadata must be an object")
        declared_scale = generation.get("scale")
        if declared_scale not in {None, "linear", "log", "logarithmic", "categorical"}:
            raise HeatmapValidationError(f"unsupported axis scale metadata {declared_scale!r}")
        if len(names) > 1 or rows is not None or declared_scale == "categorical":
            scale: AxisScale = "categorical"
        elif declared_scale in {"log", "logarithmic"}:
            scale = "log"
        elif declared_scale == "linear":
            scale = "linear"
        else:
            scale = _inferred_scale(axis_values)
        return cls(names=names, values=axis_values, scale=scale)


@dataclass(frozen=True)
class MaskSpec:
    max_price_diff_bps: float | None = None
    max_detach_energy: float | None = None
    max_final_price_diff_bps: float | None = None
    slippage_thr_bps: float | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise HeatmapValidationError(f"{name} must be finite and nonnegative")

    @property
    def enabled(self) -> bool:
        return any(value is not None for value in self.__dict__.values())

    def to_dict(self) -> dict[str, float | None]:
        # Core visual filters are always explicit in saved explorer state.
        # Slippage fields remain conditional because most grids do not load
        # those diagnostics.
        payload: dict[str, float | None] = {
            "max_price_diff_bps": self.max_price_diff_bps,
            "max_detach_energy": self.max_detach_energy,
            "max_final_price_diff_bps": self.max_final_price_diff_bps,
        }
        if self.slippage_thr_bps is not None:
            payload["slippage_thr_bps"] = self.slippage_thr_bps
        return payload


@dataclass(frozen=True)
class HeatmapSelection:
    candidate_id: str
    ordinal: int
    coordinates: Mapping[str, Any]
    metrics: Mapping[str, float | None]
    grid_indices: tuple[int, ...]

    def to_selection_ref(self, run_id: str) -> SelectionRef:
        """Represent this exact table cell for replay."""
        return SelectionRef(
            run_id=run_id,
            kind="grid_point",
            index=self.ordinal,
            coordinate=dict(self.coordinates),
            candidate_id=self.candidate_id,
            tags=("heatmap",),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "ordinal": self.ordinal,
            "coordinates": dict(self.coordinates),
            "metrics": dict(self.metrics),
            "grid_indices": list(self.grid_indices),
        }


@dataclass(frozen=True)
class HeatmapDataset:
    axes: tuple[HeatmapAxis, ...]
    metrics: Mapping[str, np.ndarray]
    candidate_ids: np.ndarray
    ordinals: np.ndarray
    valid: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.axes) < 2:
            raise HeatmapValidationError("a heatmap requires at least two grid axes")
        if len(set(axis.key for axis in self.axes)) != len(self.axes):
            raise HeatmapValidationError("heatmap axis keys are duplicated")
        shape = self.shape
        for name, raw in self.metrics.items():
            values = np.asarray(raw, dtype=float)
            if values.shape != shape:
                raise HeatmapValidationError(
                    f"metric {name!r} has shape {values.shape}, expected {shape}"
                )
        for name, raw in (("candidate_ids", self.candidate_ids),
                          ("ordinals", self.ordinals)):
            if np.asarray(raw).shape != shape:
                raise HeatmapValidationError(f"{name} does not match heatmap shape {shape}")
        if np.asarray(self.valid).shape != shape:
            raise HeatmapValidationError(f"valid does not match heatmap shape {shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis.values) for axis in self.axes)

    @property
    def axis_keys(self) -> tuple[str, ...]:
        return tuple(axis.key for axis in self.axes)

    def axis(self, key: str) -> HeatmapAxis:
        for axis in self.axes:
            if axis.key == key:
                return axis
        raise HeatmapValidationError(f"unknown heatmap axis {key!r}")

    def axis_index(self, key: str) -> int:
        return self.axis_keys.index(self.axis(key).key)

    def _masked_metric_array(self, name: str, mask: MaskSpec) -> np.ndarray:
        """Apply every enabled diagnostic threshold to ``X_masked``."""
        source = masked_metric_source(name, self.metrics)
        if source is None:
            raise HeatmapValidationError(f"unknown masked metric {name!r}")
        if source not in self.metrics:
            raise HeatmapValidationError(
                f"masked metric {name!r} requires {source!r} in the evaluation table"
            )
        values = np.asarray(self.metrics[source], dtype=float).copy()
        values[~np.asarray(self.valid, dtype=bool)] = np.nan
        if mask.max_price_diff_bps is not None:
            pdiff = next(
                (
                    np.asarray(self.metrics[n], dtype=float)
                    for n in ("max_7d_rel_price_diff", "max_rel_price_diff")
                    if n in self.metrics
                ),
                None,
            )
            if pdiff is None:
                raise HeatmapValidationError("price-difference mask metric is unavailable")
            values[
                ~np.isfinite(pdiff)
                | (pdiff < 0.0)
                | (np.abs(pdiff) > mask.max_price_diff_bps / 10_000.0)
            ] = np.nan
        uses_detach = masked_metric_uses_detach(name, self.metrics)
        if mask.max_detach_energy is not None and uses_detach:
            if "detach_energy_ungated" not in self.metrics:
                raise HeatmapValidationError("detachment mask metric is unavailable")
            detach = np.asarray(self.metrics["detach_energy_ungated"], dtype=float)
            values[
                ~np.isfinite(detach)
                | (detach < 0.0)
                | (detach > mask.max_detach_energy)
            ] = np.nan
        if mask.max_final_price_diff_bps is not None and uses_detach:
            if "final_rel_price_diff" not in self.metrics:
                raise HeatmapValidationError("final price-difference mask metric is unavailable")
            final_diff = np.asarray(self.metrics["final_rel_price_diff"], dtype=float)
            values[
                ~np.isfinite(final_diff)
                | (final_diff < 0.0)
                | (np.abs(final_diff) > mask.max_final_price_diff_bps / 10_000.0)
            ] = np.nan
        if mask.slippage_thr_bps is not None and masked_metric_uses_slippage(
            name, self.metrics
        ):
            slippage_name = (
                masked_metric_slippage_source(name) or "tw_real_slippage_1pct"
            )
            if slippage_name not in self.metrics:
                raise HeatmapValidationError(
                    f"slippage mask metric {slippage_name!r} is unavailable"
                )
            slippage = np.asarray(self.metrics[slippage_name], dtype=float)
            values[
                ~np.isfinite(slippage)
                | (slippage == -1.0)
                | (slippage > mask.slippage_thr_bps / 10_000.0)
            ] = np.nan
        return values

    def metric_array(self, name: str, mask: MaskSpec = MaskSpec()) -> np.ndarray:
        if is_masked_metric(name, self.metrics):
            return self._masked_metric_array(name, mask)
        if name not in self.metrics:
            raise HeatmapValidationError(f"unknown heatmap metric {name!r}")
        values = np.asarray(self.metrics[name], dtype=float).copy()
        values[~np.asarray(self.valid, dtype=bool)] = np.nan
        return values

    def slice_metric(
        self,
        name: str,
        *,
        x_axis: str,
        y_axis: str,
        fixed_indices: Mapping[str, int],
        mask: MaskSpec = MaskSpec(),
    ) -> np.ndarray:
        x_index = self.axis_index(x_axis)
        y_index = self.axis_index(y_axis)
        if x_index == y_index:
            raise HeatmapValidationError("heatmap x and y axes must differ")
        selection: list[Any] = []
        for index, axis in enumerate(self.axes):
            if index in {x_index, y_index}:
                selection.append(slice(None))
            else:
                fixed = int(fixed_indices.get(axis.key, 0))
                if fixed < 0 or fixed >= len(axis.values):
                    raise HeatmapValidationError(f"fixed index is outside axis {axis.key!r}")
                selection.append(fixed)
        result = self.metric_array(name, mask)[tuple(selection)]
        if x_index < y_index:
            result = result.T
        return np.asarray(result, dtype=float)

    def point(self, indices: Sequence[int]) -> HeatmapSelection:
        if len(indices) != len(self.axes):
            raise HeatmapValidationError("heatmap point dimensionality is invalid")
        location = tuple(int(index) for index in indices)
        coordinate: dict[str, Any] = {}
        for axis, index in zip(self.axes, location, strict=True):
            if index < 0 or index >= len(axis.values):
                raise HeatmapValidationError(f"point index is outside axis {axis.key!r}")
            coordinate.update(axis.coordinate(index))
        candidate_id = str(self.candidate_ids[location])
        ordinal = int(self.ordinals[location])
        if not candidate_id:
            raise HeatmapValidationError("selected heatmap cell has no exact candidate")
        if not bool(self.valid[location]):
            raise HeatmapValidationError("selected heatmap cell is invalid")
        metrics: dict[str, float | None] = {}
        for name in self.metrics:
            value = float(self.metric_array(name)[location])
            metrics[name] = value if math.isfinite(value) else None
        return HeatmapSelection(
            candidate_id=candidate_id,
            ordinal=ordinal,
            coordinates=coordinate,
            metrics=metrics,
            grid_indices=location,
        )

@dataclass
class HeatmapTilesState:
    axes: tuple[HeatmapAxis, ...]
    metrics: tuple[str, ...]
    tiles: tuple[str, ...]
    x_axis: str
    y_axis: str
    slider_indices: dict[str, int] = field(default_factory=dict)
    mask: MaskSpec = field(default_factory=MaskSpec)
    ncol: int = 2
    log_axes: tuple[str, ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        keys = tuple(axis.key for axis in self.axes)
        if self.x_axis == self.y_axis or self.x_axis not in keys or self.y_axis not in keys:
            raise HeatmapValidationError("heatmap tiles state requires two distinct declared axes")
        if not self.tiles:
            raise HeatmapValidationError("tiled heatmap requires at least one metric")
        for tile in self.tiles:
            if tile not in self.metrics and not is_masked_metric(tile, self.metrics):
                raise HeatmapValidationError(
                    f"unknown heatmap metric {tile!r}; available metrics: {', '.join(self.metrics)}"
                )
        if self.ncol < 1:
            raise HeatmapValidationError("tiled heatmap ncol must be >= 1")
        for name in self.log_axes:
            if name not in keys:
                raise HeatmapValidationError(
                    f"--log-axis {name!r} is not a heatmap axis (available: {', '.join(keys)})"
                )
            axis = next(axis for axis in self.axes if axis.key == name)
            if axis.scale == "categorical":
                raise HeatmapValidationError(f"cannot log-scale categorical axis {name!r}")
            if not axis.is_coupled and np.any(np.asarray(axis.values, dtype=float) <= 0):
                raise HeatmapValidationError(f"--log-axis {name!r} requires all axis values to be > 0")
        self.slider_indices = {
            axis.key: int(self.slider_indices.get(axis.key, 0))
            for axis in self.axes
            if axis.key not in {self.x_axis, self.y_axis}
        }
        for axis in self.axes:
            if axis.key in self.slider_indices:
                index = self.slider_indices[axis.key]
                if index < 0 or index >= len(axis.values):
                    raise HeatmapValidationError(f"slider index is outside axis {axis.key!r}")

    @property
    def singleton_axes(self) -> tuple[str, ...]:
        return tuple(axis.key for axis in self.axes if axis.is_singleton)

    @property
    def slider_axes(self) -> tuple[HeatmapAxis, ...]:
        return tuple(
            axis
            for axis in self.axes
            if axis.key not in {self.x_axis, self.y_axis} and not axis.is_singleton
        )

    def to_dict(self) -> dict[str, Any]:
        tile_data = {
            "source": self.source,
            "shape": [len(axis.values) for axis in self.axes],
            "axis_keys": [axis.key for axis in self.axes],
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "slider_indices": dict(self.slider_indices),
        }
        return {
            "schema_version": "fxopt_heatmap_state_v1",
            "metric": self.tiles[0],
            "metrics": list(self.metrics),
            "tiles": [
                {"metric": tile, "data": dict(tile_data)}
                for tile in self.tiles
            ],
            "ncol": self.ncol,
            "log_axes": list(self.log_axes),
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "slider_indices": dict(self.slider_indices),
            "slider_coordinates": {
                axis.key: _python_value(axis.values[self.slider_indices[axis.key]])
                for axis in self.axes
                if axis.key in self.slider_indices
            },
            "singleton_axes": list(self.singleton_axes),
            "mask": self.mask.to_dict(),
            "axes": [
                {
                    "key": axis.key,
                    "names": list(axis.names),
                    "values": [_python_value(value) for value in axis.values],
                    "scale": axis.scale,
                }
                for axis in self.axes
            ],
            "data": {
                "source": self.source,
                "shape": [len(axis.values) for axis in self.axes],
                "axis_keys": [axis.key for axis in self.axes],
            },
        }

    @classmethod
    def default(
        cls,
        dataset: HeatmapDataset,
        *,
        tiles: Sequence[str],
        x_axis: str | None = None,
        y_axis: str | None = None,
        mask: MaskSpec = MaskSpec(),
        ncol: int = 2,
        log_axes: Sequence[str] = (),
    ) -> "HeatmapTilesState":
        active = [axis.key for axis in dataset.axes if not axis.is_singleton]
        candidates = active + [axis.key for axis in dataset.axes if axis.key not in active]
        if len(candidates) < 2:
            raise HeatmapValidationError("heatmap needs two axes")
        canonical = {"donation_apy", "reserved_profit_fraction"}
        if x_axis is None and y_axis is None and canonical <= set(active):
            selected_x, selected_y = "donation_apy", "reserved_profit_fraction"
        else:
            selected_x = x_axis or candidates[0]
            selected_y = y_axis or next(key for key in candidates if key != selected_x)
        return cls(
            axes=dataset.axes,
            metrics=tuple(dataset.metrics),
            tiles=tuple(tiles),
            x_axis=selected_x,
            y_axis=selected_y,
            mask=mask,
            ncol=ncol,
            log_axes=tuple(log_axes),
        )


def _is_positional(axis: HeatmapAxis) -> bool:
    if axis.scale == "categorical" or axis.is_coupled:
        return True
    values = np.asarray(axis.values, dtype=float)
    return len(values) > 1 and not bool(np.all(np.diff(values) > 0))


def _centers(axis: HeatmapAxis) -> np.ndarray:
    if _is_positional(axis):
        return np.arange(len(axis.values), dtype=float)
    return np.asarray(axis.values, dtype=float)


def edges(centers: np.ndarray, *, logarithmic: bool) -> np.ndarray:
    if len(centers) == 1:
        center = float(centers[0])
        if logarithmic:
            factor = math.sqrt(2.0)
            return np.asarray([center / factor, center * factor])
        width = abs(center) * 0.5 if center else 0.5
        return np.asarray([center - width, center + width])
    if logarithmic:
        logs = np.log(centers)
        middle = (logs[:-1] + logs[1:]) / 2.0
        edges = np.empty(len(centers) + 1)
        edges[1:-1] = np.exp(middle)
        edges[0] = math.exp(2.0 * logs[0] - middle[0])
        edges[-1] = math.exp(2.0 * logs[-1] - middle[-1])
        return edges
    middle = (centers[:-1] + centers[1:]) / 2.0
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = middle
    edges[0] = centers[0] - (middle[0] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - middle[-1])
    return edges


__all__ = [
    "auto_log",
    "edges",
    "HeatmapAxis",
    "HeatmapDataset",
    "HeatmapSelection",
    "HeatmapTilesState",
    "HeatmapValidationError",
    "MaskSpec",
]
