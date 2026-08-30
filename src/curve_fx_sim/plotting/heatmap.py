"""Interactive and headless heatmaps over exact common evaluation tables."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcsetup
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.widgets import RadioButtons, Slider

from .masked_metrics import (
    SKEW_MASKED_METRICS,
    SLIPPAGE_APY_MASK_SOURCES,
    is_masked_metric,
    masked_metric_source,
    masked_metric_uses_detach,
)
from .theme import DEFAULT_THEME, PlotTheme, apply_theme

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

_MAIN_FIGURE_SIZE = (12.0, 7.4)
_MAIN_AXES_RECT = (0.08, 0.19, 0.67, 0.73)
_RADIO_RECT = (0.79, 0.58, 0.19, 0.34)
_SLIDER_LEFT = 0.16
_MAX_VISIBLE_METRIC_CONTROLS = 12
_SLIDER_WIDTH = 0.56
_SLIDER_HEIGHT = 0.025
_SLIDER_BOTTOM = 0.045
_SLIDER_GAP = 0.038


def interactive_backend_active() -> bool:
    """True when matplotlib selected a display-capable interactive backend.

    The module no longer forces Agg: matplotlib auto-selects a GUI backend
    when a display is available and falls back to Agg headlessly.  Callers
    save first and then ``show()`` only when this reports True.
    """
    current = matplotlib.get_backend().lower()
    return current in _interactive_backend_names()


def _interactive_backend_names() -> set[str]:
    try:
        # matplotlib >= 3.9 registry API.
        from matplotlib.backends.registry import BackendFilter, backend_registry

        return {
            name.lower()
            for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
    except Exception:  # matplotlib 3.8: rcsetup.interactive_bk
        return {name.lower() for name in rcsetup.interactive_bk}


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


def _exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return left == right


def _labels(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        " / ".join(str(_python_value(item)) for item in value)
        if isinstance(value, (tuple, list))
        else str(_python_value(value))
        for value in values
    )


def _auto_log(values: Sequence[Any]) -> bool:
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
    if _auto_log(values):
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
    max_skew_percent: float | None = None
    max_final_price_diff_bps: float | None = None
    slippage_thr_bps: float | None = None
    slippage_thr_max_bps: float | None = None

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
            "max_skew_percent": self.max_skew_percent,
            "max_final_price_diff_bps": self.max_final_price_diff_bps,
        }
        if self.slippage_thr_bps is not None:
            payload["slippage_thr_bps"] = self.slippage_thr_bps
        if self.slippage_thr_max_bps is not None:
            payload["slippage_thr_max_bps"] = self.slippage_thr_max_bps
        return payload


@dataclass(frozen=True)
class HeatmapSelection:
    candidate_id: str
    ordinal: int
    coordinates: Mapping[str, Any]
    metrics: Mapping[str, float | None]
    grid_indices: tuple[int, ...]

    def to_selection_ref(self, run_id: str) -> SelectionRef:
        """Adapt this exact table cell to Foundation's sole replay boundary."""
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
        """Apply conditional legacy filters to a masked metric.

        Raw tiles are observations, not pass/fail views: their values are only
        invalidated by a failed evaluation row.  The threshold controls belong
        to the derived ``*_masked`` family and are intentionally conditional
        on that metric's semantics.
        """
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
            values[np.abs(pdiff) > mask.max_price_diff_bps / 10_000.0] = np.nan
        if masked_metric_uses_detach(name, self.metrics) and mask.max_detach_energy is not None:
            if "detach_energy_ungated" not in self.metrics:
                raise HeatmapValidationError("detachment mask metric is unavailable")
            detach = np.asarray(self.metrics["detach_energy_ungated"], dtype=float)
            values[
                ~np.isfinite(detach) | (detach > mask.max_detach_energy)
            ] = np.nan
        slippage_source = SLIPPAGE_APY_MASK_SOURCES.get(name)
        if slippage_source is not None and mask.slippage_thr_bps is not None:
            if slippage_source not in self.metrics:
                raise HeatmapValidationError(
                    f"masked metric {name!r} requires {slippage_source!r} in the evaluation table"
                )
            slippage = np.asarray(self.metrics[slippage_source], dtype=float)
            values[~np.isfinite(slippage) | (slippage > mask.slippage_thr_bps / 10_000.0)] = np.nan
        if name in SKEW_MASKED_METRICS:
            if mask.max_skew_percent is not None:
                if "max_7d_skew" not in self.metrics:
                    raise HeatmapValidationError("skew mask metric is unavailable")
                skew = np.asarray(self.metrics["max_7d_skew"], dtype=float)
                values[~np.isfinite(skew) | (np.abs(skew) > mask.max_skew_percent / 100.0)] = np.nan
            if mask.max_final_price_diff_bps is not None:
                if "final_rel_price_diff" not in self.metrics:
                    raise HeatmapValidationError("final price-difference mask metric is unavailable")
                final_diff = np.asarray(self.metrics["final_rel_price_diff"], dtype=float)
                values[
                    ~np.isfinite(final_diff)
                    | (np.abs(final_diff) > mask.max_final_price_diff_bps / 10_000.0)
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
class HeatmapState:
    axes: tuple[HeatmapAxis, ...]
    metrics: tuple[str, ...]
    metric: str
    x_axis: str
    y_axis: str
    slider_indices: dict[str, int] = field(default_factory=dict)
    mask: MaskSpec = field(default_factory=MaskSpec)
    source: str | None = None

    def __post_init__(self) -> None:
        keys = tuple(axis.key for axis in self.axes)
        if self.x_axis == self.y_axis or self.x_axis not in keys or self.y_axis not in keys:
            raise HeatmapValidationError("heatmap state requires two distinct declared axes")
        if self.metric not in self.metrics and not is_masked_metric(self.metric, self.metrics):
            raise HeatmapValidationError("heatmap state metric is unavailable")
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
        return {
            "schema_version": "fxopt_heatmap_state_v1",
            "metric": self.metric,
            "metrics": list(self.metrics),
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
        metric: str | None = None,
        x_axis: str | None = None,
        y_axis: str | None = None,
        mask: MaskSpec = MaskSpec(),
    ) -> "HeatmapState":
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
        metrics = tuple(dataset.metrics)
        resolved_metric = metric
        if resolved_metric is None:
            if not metrics:
                raise HeatmapValidationError("heatmap has no metrics")
            resolved_metric = metrics[0]
        return cls(
            axes=dataset.axes,
            metrics=metrics,
            metric=resolved_metric,
            x_axis=selected_x,
            y_axis=selected_y,
            mask=mask,
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


def _edges(centers: np.ndarray, *, logarithmic: bool) -> np.ndarray:
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


def _cell_index(axis: HeatmapAxis, coordinate: float) -> int | None:
    """Return the containing declared cell, never a nearest coordinate."""
    edges = _edges(
        _centers(axis),
        logarithmic=axis.scale == "log" and not _is_positional(axis),
    )
    if coordinate < edges[0] or coordinate > edges[-1]:
        return None
    index = int(np.searchsorted(edges, coordinate, side="right") - 1)
    if index == len(axis.values) and coordinate == edges[-1]:
        index -= 1
    return index if 0 <= index < len(axis.values) else None


def _atomic_save(figure: object, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable heatmap artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format=path.suffix.lstrip("."), bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class MatplotlibHeatmapView:
    """Live N-dimensional heatmap with exact cell inspection."""

    def __init__(
        self,
        data: HeatmapDataset,
        state: HeatmapState | None = None,
        *,
        on_select: Callable[[HeatmapSelection], None] | None = None,
        theme: PlotTheme = DEFAULT_THEME,
    ) -> None:
        self.dataset = data
        self.state = state or HeatmapState.default(self.dataset)
        if self.state.axes != self.dataset.axes:
            raise HeatmapValidationError("heatmap state axes differ from the dataset")
        self.theme = theme
        self.on_select = on_select
        self.last_selection: HeatmapSelection | None = None
        show_metric_controls = len(self.state.metrics) <= _MAX_VISIBLE_METRIC_CONTROLS
        axes_rect = _MAIN_AXES_RECT if show_metric_controls else (0.08, 0.12, 0.82, 0.80)
        self.figure = plt.figure(figsize=_MAIN_FIGURE_SIZE)
        self.axis = self.figure.add_axes(axes_rect)
        self._metric_axis = self.figure.add_axes(_RADIO_RECT, facecolor=theme.surface)
        self._metric_axis.set_title("Metric", loc="left", fontfamily=theme.font_family)
        self._metric_radio = RadioButtons(
            self._metric_axis,
            self.state.metrics,
            active=self.state.metrics.index(self.state.metric),
            activecolor=theme.accent,
        )
        self._metric_axis.set_visible(show_metric_controls)
        self._metric_radio.on_clicked(self._on_metric)
        self._sliders: dict[str, Slider] = {}
        self._image: object | None = None
        self._colorbar: object | None = None
        self._rebuild_sliders()
        self._draw()
        self._click_connection = self.figure.canvas.mpl_connect(
            "button_press_event", self._on_click
        )

    def _rebuild_sliders(self) -> None:
        for slider in self._sliders.values():
            slider.ax.remove()
        self._sliders.clear()
        for row, axis in enumerate(self.state.slider_axes):
            slider_axis = self.figure.add_axes(
                (
                    _SLIDER_LEFT,
                    _SLIDER_BOTTOM + row * _SLIDER_GAP,
                    _SLIDER_WIDTH,
                    _SLIDER_HEIGHT,
                ),
                facecolor=self.theme.surface,
            )
            slider = Slider(
                slider_axis,
                axis.key,
                0,
                len(axis.values) - 1,
                valinit=self.state.slider_indices[axis.key],
                valstep=1,
                color=self.theme.accent,
            )
            slider.valtext.set_text(axis.display_labels[self.state.slider_indices[axis.key]])
            slider.on_changed(lambda value, key=axis.key: self._on_slider(key, value))
            self._sliders[axis.key] = slider

    def _on_slider(self, key: str, value: float) -> None:
        index = int(value)
        self.state.slider_indices[key] = index
        self._sliders[key].valtext.set_text(self.dataset.axis(key).display_labels[index])
        self._draw()

    def _on_metric(self, metric: str) -> None:
        self.state.metric = metric
        self._draw()

    def _draw(self) -> None:
        self.axis.clear()
        x_axis = self.dataset.axis(self.state.x_axis)
        y_axis = self.dataset.axis(self.state.y_axis)
        values = self.dataset.slice_metric(
            self.state.metric,
            x_axis=x_axis.key,
            y_axis=y_axis.key,
            fixed_indices=self.state.slider_indices,
            mask=self.state.mask,
        )
        x_centers = _centers(x_axis)
        y_centers = _centers(y_axis)
        self._image = self.axis.pcolormesh(
            _edges(
                x_centers,
                logarithmic=x_axis.scale == "log" and not _is_positional(x_axis),
            ),
            _edges(
                y_centers,
                logarithmic=y_axis.scale == "log" and not _is_positional(y_axis),
            ),
            np.ma.masked_invalid(values),
            shading="flat",
            cmap=self.theme.sequential_cmap,
        )
        if self._colorbar is None:
            self._colorbar = self.figure.colorbar(self._image, ax=self.axis, pad=0.02)
        else:
            self._colorbar.update_normal(self._image)
        self._configure_axis(x_axis, orientation="x")
        self._configure_axis(y_axis, orientation="y")
        self.axis.set_title(
            self.state.metric,
            loc="left",
            fontfamily=self.theme.font_family,
            fontsize=self.theme.title_size,
        )
        apply_theme(self.figure, np.asarray([self.axis, self._metric_axis], dtype=object), self.theme)
        self.figure.canvas.draw_idle()

    def _configure_axis(self, axis: HeatmapAxis, *, orientation: Literal["x", "y"]) -> None:
        label = self.axis.set_xlabel if orientation == "x" else self.axis.set_ylabel
        scale = self.axis.set_xscale if orientation == "x" else self.axis.set_yscale
        ticks = self.axis.set_xticks if orientation == "x" else self.axis.set_yticks
        ticklabels = self.axis.set_xticklabels if orientation == "x" else self.axis.set_yticklabels
        label(axis.key, fontfamily=self.theme.font_family, fontsize=self.theme.label_size)
        positional = _is_positional(axis)
        if axis.scale == "log" and not positional:
            scale("log")
        if positional:
            ticks(_centers(axis))
            ticklabels(
                axis.display_labels,
                rotation=35 if orientation == "x" else 0,
                ha="right" if orientation == "x" else "center",
            )

    def selection_from_event(self, event: MouseEvent) -> HeatmapSelection | None:
        if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:
            return None
        x_axis = self.dataset.axis(self.state.x_axis)
        y_axis = self.dataset.axis(self.state.y_axis)
        x_index = _cell_index(x_axis, float(event.xdata))
        y_index = _cell_index(y_axis, float(event.ydata))
        if x_index is None or y_index is None:
            return None
        indices = [self.state.slider_indices.get(axis.key, 0) for axis in self.dataset.axes]
        indices[self.dataset.axis_index(x_axis.key)] = x_index
        indices[self.dataset.axis_index(y_axis.key)] = y_index
        return self.dataset.point(indices)

    def _on_click(self, event: MouseEvent) -> None:
        if event.button not in {MouseButton.LEFT, 1}:
            return
        selection = self.selection_from_event(event)
        if selection is None:
            return
        self.last_selection = selection
        if self.on_select is not None:
            self.on_select(selection)

    def show(self, *, block: bool = True) -> None:
        plt.show(block=block)

    def save(
        self,
        output: Path | str,
        *,
        state_path: Path | str | None = None,
    ) -> tuple[Path, Path]:
        image_path = Path(output)
        sidecar_path = Path(state_path) if state_path else image_path.with_suffix(".state.json")
        if sidecar_path.exists():
            raise FileExistsError(f"immutable heatmap state already exists: {sidecar_path}")
        _atomic_save(self.figure, image_path)
        atomic_write_json(sidecar_path, self.state.to_dict())
        return image_path, sidecar_path

    def close(self) -> None:
        plt.close(self.figure)


def render_heatmap(
    data: HeatmapDataset,
    output: Path | str,
    *,
    state: HeatmapState | None = None,
    metric: str | None = None,
    x_axis: str | None = None,
    y_axis: str | None = None,
    mask: MaskSpec = MaskSpec(),
    theme: PlotTheme = DEFAULT_THEME,
    source: str | None = None,
) -> tuple[Path, Path]:
    """Headlessly render a heatmap and deterministic complete state sidecar.

    ``source`` is an opaque pointer to the N-D data (typically the run-relative
    evaluation-table name) recorded in the state sidecar so frontends can
    re-render any slice.  The figure is saved unconditionally; callers that
    also want the interactive widget figure call ``show()`` themselves when
    :func:`interactive_backend_active` reports a display-capable backend.
    """
    dataset = data
    resolved_state = state or HeatmapState.default(
        dataset,
        metric=metric,
        x_axis=x_axis,
        y_axis=y_axis,
        mask=mask,
    )
    if source is not None:
        resolved_state.source = source
    view = MatplotlibHeatmapView(dataset, resolved_state, theme=theme)
    try:
        return view.save(output)
    finally:
        view.close()


_TILE_FIG_WIDTH = 6.0
_TILE_FIG_HEIGHT = 5.0
_TILE_SLIDER_LEFT = 0.12
_TILE_SLIDER_WIDTH = 0.56
_TILE_SLIDER_HEIGHT = 0.025
_TILE_SLIDER_BOTTOM = 0.06
_TILE_SLIDER_GAP = 0.04


class MatplotlibHeatmapTilesView:
    """Multi-metric tiled heatmap sharing one N-D slice and one validity mask."""

    def __init__(
        self,
        data: HeatmapDataset,
        *,
        tiles: Sequence[str],
        x_axis: str | None = None,
        y_axis: str | None = None,
        ncol: int = 2,
        log_axes: Sequence[str] = (),
        mask: MaskSpec = MaskSpec(),
        theme: PlotTheme = DEFAULT_THEME,
    ) -> None:
        self.dataset = data
        self.state = HeatmapTilesState.default(
            self.dataset,
            tiles=tiles,
            x_axis=x_axis,
            y_axis=y_axis,
            mask=mask,
            ncol=ncol,
            log_axes=log_axes,
        )
        if self.state.axes != self.dataset.axes:
            raise HeatmapValidationError("heatmap tiles state axes differ from the dataset")
        self.theme = theme
        n = len(self.state.tiles)
        cols = min(self.state.ncol, n)
        rows = max(1, math.ceil(n / cols)) if n else 1
        grid_bottom = 0.12 if self.state.slider_axes else 0.06
        self.figure = plt.figure(figsize=(_TILE_FIG_WIDTH * cols, _TILE_FIG_HEIGHT * rows))
        self._axes = [self.figure.add_subplot(rows, cols, index + 1) for index in range(rows * cols)]
        self._images: dict[int, object] = {}
        self._colorbars: list[object] = []
        self._sliders: dict[str, Slider] = {}
        self._rebuild_sliders()
        self._draw()
        self.figure.subplots_adjust(bottom=grid_bottom, wspace=0.28, hspace=0.32)

    def _rebuild_sliders(self) -> None:
        for slider in self._sliders.values():
            slider.ax.remove()
        self._sliders.clear()
        for row, axis in enumerate(self.state.slider_axes):
            slider_axis = self.figure.add_axes(
                (
                    _TILE_SLIDER_LEFT,
                    _TILE_SLIDER_BOTTOM + row * _TILE_SLIDER_GAP,
                    _TILE_SLIDER_WIDTH,
                    _TILE_SLIDER_HEIGHT,
                ),
                facecolor=self.theme.surface,
            )
            slider = Slider(
                slider_axis,
                axis.key,
                0,
                len(axis.values) - 1,
                valinit=self.state.slider_indices[axis.key],
                valstep=1,
                color=self.theme.accent,
            )
            slider.valtext.set_text(axis.display_labels[self.state.slider_indices[axis.key]])
            slider.on_changed(lambda value, key=axis.key: self._on_slider(key, value))
            self._sliders[axis.key] = slider

    def _on_slider(self, key: str, value: float) -> None:
        index = int(value)
        self.state.slider_indices[key] = index
        if key in self._sliders:
            self._sliders[key].valtext.set_text(self.dataset.axis(key).display_labels[index])
        self._draw()

    def _draw(self) -> None:
        for colorbar in self._colorbars:
            try:
                colorbar.remove()
            except Exception:
                pass
        self._colorbars = []
        self._images = {}
        x_axis = self.dataset.axis(self.state.x_axis)
        y_axis = self.dataset.axis(self.state.y_axis)
        log_x = x_axis.key in self.state.log_axes
        log_y = y_axis.key in self.state.log_axes
        x_positional = _is_positional(x_axis)
        y_positional = _is_positional(y_axis)
        x_centers = _centers(x_axis)
        y_centers = _centers(y_axis)
        x_edges = _edges(x_centers, logarithmic=log_x and not x_positional)
        y_edges = _edges(y_centers, logarithmic=log_y and not y_positional)
        n = len(self.state.tiles)
        cols = min(self.state.ncol, n)
        for index, axis in enumerate(self._axes):
            axis.clear()
            if index >= n:
                axis.axis("off")
                continue
            metric = self.state.tiles[index]
            # Legacy plot_heatmap_nd_opt.py builds metrics in two passes: raw
            # metrics keep every cell (only success-masked); the *_masked
            # family alone receives the pdiff/slippage caps. Scope the mask to
            # masked tiles so raw panels keep full coverage.
            tile_mask = (
                self.state.mask
                if is_masked_metric(metric, self.dataset.metrics)
                else MaskSpec()
            )
            values = self.dataset.slice_metric(
                metric,
                x_axis=x_axis.key,
                y_axis=y_axis.key,
                fixed_indices=self.state.slider_indices,
                mask=tile_mask,
            )
            image = axis.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(values),
                shading="flat",
                cmap=self.theme.sequential_cmap,
            )
            self._images[index] = image
            if log_x and not x_positional:
                axis.set_xscale("log")
            if log_y and not y_positional:
                axis.set_yscale("log")
            self._configure_tile_ranges(image, values)
            self._tile_axial(axis, x_axis, orientation="x", logarithmic=log_x)
            self._tile_axial(axis, y_axis, orientation="y", logarithmic=log_y)
            axis.set_title(metric, fontfamily=self.theme.font_family, fontsize=self.theme.title_size)
            colorbar = self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.05)
            self._colorbars.append(colorbar)
        apply_theme(self.figure, np.asarray(self._axes[:max(1, n)], dtype=object) if n else np.asarray([], dtype=object), self.theme)
        self.figure.canvas.draw_idle()

    def _configure_tile_ranges(self, image: object, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        zmin = float(np.min(finite))
        zmax = float(np.max(finite))
        if zmin == zmax:
            eps = 1e-12 if zmax == 0 else abs(zmax) * 1e-12
            zmin, zmax = zmin - eps, zmax + eps
        image.set_clim(zmin, zmax)

    def _tile_axial(
        self,
        axis: object,
        heat_axis: HeatmapAxis,
        *,
        orientation: Literal["x", "y"],
        logarithmic: bool,
    ) -> None:
        label = axis.set_xlabel if orientation == "x" else axis.set_ylabel
        scale = axis.set_xscale if orientation == "x" else axis.set_yscale
        ticks = axis.set_xticks if orientation == "x" else axis.set_yticks
        ticklabels = axis.set_xticklabels if orientation == "x" else axis.set_yticklabels
        label(heat_axis.key, fontfamily=self.theme.font_family, fontsize=self.theme.label_size)
        positional = _is_positional(heat_axis)
        if logarithmic and not positional:
            scale("log")
        if positional:
            centers = _centers(heat_axis)
            ticks(centers)
            ticklabels(
                heat_axis.display_labels,
                rotation=35 if orientation == "x" else 0,
                ha="right" if orientation == "x" else "center",
            )

    def show(self, *, block: bool = True) -> None:
        plt.show(block=block)

    def save(
        self,
        output: Path | str,
        *,
        state_path: Path | str | None = None,
    ) -> tuple[Path, Path]:
        image_path = Path(output)
        sidecar_path = Path(state_path) if state_path else image_path.with_suffix(".state.json")
        if sidecar_path.exists():
            raise FileExistsError(f"immutable heatmap state already exists: {sidecar_path}")
        _atomic_save(self.figure, image_path)
        atomic_write_json(sidecar_path, self.state.to_dict())
        return image_path, sidecar_path

    def close(self) -> None:
        plt.close(self.figure)


def render_heatmap_tiles(
    data: HeatmapDataset,
    output: Path | str,
    *,
    tiles: Sequence[str],
    x_axis: str | None = None,
    y_axis: str | None = None,
    ncol: int = 2,
    log_axes: Sequence[str] = (),
    mask: MaskSpec = MaskSpec(),
    theme: PlotTheme = DEFAULT_THEME,
    source: str | None = None,
) -> tuple[Path, Path]:
    """Headlessly render the multi-metric tiled heatmap and its state sidecar.

    Each tile shares the same x/y axes, the same fixed slice for extra
    dimensions, and the same validity mask: a masked cell is NaN on every
    tile.  ``source`` is recorded per tile so frontends can re-render any
    slice.  The figure is saved unconditionally; callers that also want the
    interactive widget figure call ``show()`` themselves when
    :func:`interactive_backend_active` reports a display-capable backend.
    """
    dataset = data
    view = MatplotlibHeatmapTilesView(
        dataset,
        tiles=tiles,
        x_axis=x_axis,
        y_axis=y_axis,
        ncol=ncol,
        log_axes=log_axes,
        mask=mask,
        theme=theme,
    )
    if source is not None:
        view.state.source = source
    try:
        return view.save(output)
    finally:
        view.close()


__all__ = [
    "HeatmapAxis",
    "HeatmapDataset",
    "HeatmapSelection",
    "HeatmapState",
    "HeatmapTilesState",
    "HeatmapValidationError",
    "MaskSpec",
    "MatplotlibHeatmapTilesView",
    "MatplotlibHeatmapView",
    "interactive_backend_active",
    "render_heatmap",
    "render_heatmap_tiles",
]
