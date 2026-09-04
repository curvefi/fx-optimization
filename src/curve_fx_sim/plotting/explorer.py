"""Interactive N-D heatmap explorer over a prepared :class:`HeatmapDataset`."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    FormatStrFormatter,
    NullFormatter,
    NullLocator,
)
from matplotlib.widgets import RadioButtons, Slider

from .heatmap import (
    HeatmapAxis,
    HeatmapDataset,
    HeatmapSelection,
    HeatmapTilesState,
    MaskSpec,
    SelectionRef,
    atomic_write_json,
    auto_log,
    edges,
)

from .masked_metrics import (
    is_masked_metric,
    masked_metric_slippage_sources,
    masked_metric_uses_detach,
)

_LN2 = math.log(2.0)
_MAX_TICKS = 12
_CONTROL_ROW_PITCH = 0.10


@dataclass(frozen=True)
class _AxisView:
    key: str
    name: str
    centers: np.ndarray
    labels: tuple[str, ...]
    positional: bool
    logarithmic: bool


def _format_duration_short(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d" if not hours else f"{days}d{hours}h"
    if hours:
        return f"{hours}h" if not minutes else f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m" if not seconds else f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _axis_name_and_labels(name: str, values: Sequence[float]) -> tuple[str, tuple[str, ...]]:
    key = name.lower()
    leaf = key.rsplit(".", 1)[-1]
    display_name = name
    if leaf == "a":
        return display_name, tuple(f"{value / 1e4:.4g}" for value in values)
    if leaf in {"reserved_profit_fraction", "admin_fee"}:
        scaled = bool(values) and max(abs(value) for value in values) > 1.0
        if scaled:
            return f"{name} (÷1e10)", tuple(f"{value / 1e10:.2f}" for value in values)
        return display_name, tuple(f"{value:.4g}" for value in values)
    if "fee_bps" in leaf:
        return f"{name} (bps)", tuple(f"{value:.2f}" for value in values)
    if "fee" in leaf and "gamma" not in leaf:
        scale = 1e4 if values and max(abs(value) for value in values) <= 1 else 1e-6
        return f"{name} (bps)", tuple(f"{value * scale:.2f}" for value in values)
    if "ma_time" in key or "price_source_ema_half_time" in key:
        suffix = "" if "hrs" in key else " (hrs)"
        return display_name + suffix, tuple(_format_duration_short(value * _LN2) for value in values)
    if key.endswith("_wad"):
        return f"{name} (/1e18)", tuple(f"{value / 1e18:.6g}" for value in values)
    if "gamma" in key:
        return display_name, tuple(f"{value / 1e18:.5f}" for value in values)
    if "liquidity" in key or "balance" in key:
        return f"{name} (/1e18)", tuple(f"{value / 1e18:.2f}" for value in values)
    return display_name, tuple(f"{value:.4g}" for value in values)


def _format_slider_value(name: str, value: float) -> str:
    key = name.lower()
    leaf = key.rsplit(".", 1)[-1]
    if "fee_bps" in leaf:
        return f"{value:.1f} bps"
    if leaf in {"reserved_profit_fraction", "admin_fee"}:
        return f"{(value / 1e10 if abs(value) > 1 else value):.4f}"
    if "fee" in leaf and "gamma" not in leaf:
        return f"{value * (1e4 if abs(value) <= 1 else 1e-6):.1f} bps"
    if "ma_time" in key or "price_source_ema_half_time" in key:
        return _format_duration_short(value * _LN2)
    if leaf == "a":
        return f"{value / 1e4:.2f}"
    if key.endswith("_wad"):
        return f"{value / 1e18:.6g}"
    if "gamma" in key:
        return f"{value / 1e18:.6f}"
    if "apy" in key or "ratio" in key:
        return f"{value:.4f}"
    return f"{value:.4g}"


def _select_ticks(length: int, maximum: int = _MAX_TICKS) -> tuple[int, ...]:
    if length <= maximum:
        return tuple(range(length))
    return tuple(sorted(set(int(index) for index in np.linspace(0, length - 1, maximum, dtype=int))))


def _apply_fixed_ticks(target: Any, values: Sequence[float], labels: Sequence[str]) -> None:
    target.set_major_locator(FixedLocator(values))
    target.set_major_formatter(FixedFormatter(labels or [""] * len(values)))
    target.set_minor_locator(NullLocator())
    target.set_minor_formatter(NullFormatter())
    target.get_offset_text().set_visible(False)


def _auto_font_size(nx: int, ny: int) -> int:
    grid = max(1, nx, ny)
    if grid <= 4:
        return 6
    if grid <= 8:
        return 7
    if grid <= 16:
        return 8
    if grid >= 32:
        return 10
    return int(round(8 + (grid - 16) / 16 * 2))


def _finite_clim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return (0.0, 1.0)
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        epsilon = 1e-12 if upper == 0 else abs(upper) * 1e-12
        lower, upper = lower - epsilon, upper + epsilon
    return lower, upper


def _axis_view(
    axis: HeatmapAxis,
    explicit_logs: set[str],
) -> _AxisView:
    if axis.is_coupled:
        fee_axis = all(
            "fee" in name.lower().rsplit(".", 1)[-1]
            and "gamma" not in name.lower().rsplit(".", 1)[-1]
            for name in axis.names
        )
        labels: list[str] = []
        for row in axis.values:
            parts = []
            for name, value in zip(axis.names, row, strict=True):
                try:
                    formatted = _format_slider_value(name, float(value))
                    parts.append(formatted.removesuffix(" bps") if fee_axis else formatted)
                except (TypeError, ValueError, ZeroDivisionError):
                    parts.append(str(value))
            labels.append("(" + "/".join(parts) + ")")
        return _AxisView(
            key=axis.key,
            name=f"{axis.key}, bps" if fee_axis else axis.key,
            centers=np.arange(len(axis.values), dtype=float),
            labels=tuple(labels),
            positional=True,
            logarithmic=False,
        )
    try:
        numeric = np.asarray(axis.values, dtype=float)
    except (TypeError, ValueError):
        return _AxisView(
            key=axis.key,
            name=axis.key,
            centers=np.arange(len(axis.values), dtype=float),
            labels=axis.display_labels,
            positional=True,
            logarithmic=False,
        )
    name = axis.names[0]
    positional = len(numeric) > 1 and not bool(np.all(np.diff(numeric) > 0))
    display_name, labels = _axis_name_and_labels(name, numeric.tolist())
    centers = np.arange(len(numeric), dtype=float) if positional else numeric
    logarithmic = not positional and (
        axis.key in explicit_logs or axis.scale == "log" or auto_log(numeric)
    )
    return _AxisView(axis.key, display_name, centers, labels, positional, logarithmic)


def _main_figure_size(nx: int, ny: int, rows: int, cols: int) -> tuple[float, float]:
    cell_aspect = ny / max(1, nx)
    cell_width = 22.0 / cols
    cell_height = cell_width * cell_aspect
    height = cell_height * rows
    if height > 12.0:
        height = 12.0
        cell_height = height / rows
        cell_width = cell_height / cell_aspect
        width = cell_width * cols
    else:
        width = 22.0
    return max(10.0, min(22.0, width)), max(6.0, min(12.0, height))


def _cell_index_from_view(view: _AxisView, coordinate: float) -> int | None:
    cell_edges = edges(view.centers, logarithmic=view.logarithmic and not view.positional)
    if coordinate < cell_edges[0] or coordinate > cell_edges[-1]:
        return None
    index = int(np.searchsorted(cell_edges, coordinate, side="right") - 1)
    if index == len(view.centers) and coordinate == cell_edges[-1]:
        index -= 1
    return index if 0 <= index < len(view.centers) else None


def format_axis_value(value: Any) -> str:
    """Format one exact coordinate without changing its stored value."""
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(format_axis_value(item) for item in value) + ")"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.6g}"


def format_metric_value(metric: str, value: Any) -> str:
    """Use units that make fractions and price errors readable in output."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a" if value is None else str(value)
    if not math.isfinite(number):
        return "n/a"
    scale, suffix = _metric_scale_info(metric)
    return f"{number * scale:.6g}{'%' if suffix else ''}"


def _window_title(figure: Any, title: str) -> None:
    manager = getattr(figure.canvas, "manager", None)
    setter = getattr(manager, "set_window_title", None)
    if callable(setter):
        setter(title)


def _metric_scale_info(metric: str) -> tuple[float, str]:
    key = metric.lower()
    scale = (
        1e-18
        if metric in {"virtual_price", "xcp_profit", "price_scale", "D", "totalSupply"}
        else 1.0
    )
    percent = (
        key in {"vpminusone", "apy"}
        or "apy" in key
        or "tw_real_slippage" in key
        or "geom_mean" in key
        or "rel_price_diff" in key
        or "pool_fee" in key
        or "imbalance" in key
        or "skew" in key
    )
    if percent:
        scale *= 100.0
    return scale, " (%)" if percent else ""


def _finite_max(dataset: HeatmapDataset, metric: str, scale: float = 1.0) -> float | None:
    raw = dataset.metrics.get(metric)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    maximum = float(np.nanmax(np.abs(values)) * scale)
    return maximum if math.isfinite(maximum) else None


class HeatmapExplorer:
    """Two-window heatmap and controls explorer with exact-cell replay."""

    def __init__(
        self,
        data: HeatmapDataset,
        *,
        metrics: Sequence[str] | None = None,
        ncol: int = 3,
        log_axes: Sequence[str] = (),
        x_axis: str | None = None,
        y_axis: str | None = None,
        max_pricethr: float | None = 100.0,
        max_detach_energy: float | None = None,
        slipthr: float | None = None,
        final_pdiffthr: float | None = None,
        run_id: str | None = None,
        on_replay: Callable[[SelectionRef, str], Any] | None = None,
    ) -> None:
        if not isinstance(data, HeatmapDataset):
            raise TypeError("HeatmapExplorer requires a HeatmapDataset")
        self.dataset = data
        self.metrics = tuple(metrics) if metrics is not None else tuple(self.dataset.metrics)
        missing = [
            name
            for name in self.metrics
            if name not in self.dataset.metrics
            and not is_masked_metric(name, self.dataset.metrics)
        ]
        if missing:
            raise ValueError(f"unknown explorer metric(s): {', '.join(missing)}")
        if not self.metrics:
            raise ValueError("explorer requires at least one metric")
        if ncol < 1:
            raise ValueError("ncol must be positive")
        slippage_sources = masked_metric_slippage_sources(
            self.metrics, self.dataset.metrics
        )
        if slipthr is None and slippage_sources:
            slipthr = 20.0
        self.run_id = run_id or ""
        self.on_replay = on_replay
        self._updating_radios = False
        self.last_selection: HeatmapSelection | None = None
        mask = MaskSpec(
            max_price_diff_bps=max_pricethr,
            max_detach_energy=max_detach_energy,
            max_final_price_diff_bps=final_pdiffthr,
            slippage_thr_bps=slipthr,
        )
        self.state = HeatmapTilesState.default(
            self.dataset,
            tiles=self.metrics,
            ncol=ncol,
            log_axes=log_axes,
            x_axis=x_axis,
            y_axis=y_axis,
            mask=mask,
        )
        explicit_logs = set(log_axes)
        self._axis_views = {
            axis.key: _axis_view(axis, explicit_logs)
            for axis in self.dataset.axes
        }
        self.state.log_axes = tuple(
            axis.key for axis in self.dataset.axes if self._axis_views[axis.key].logarithmic
        )
        self.fig_main = plt.figure(figsize=(13.0, 8.0), layout="constrained", num="Heatmaps")
        self.fig_controls = plt.figure(figsize=(3.2, 6.0), num="Controls")
        _window_title(self.fig_main, "Heatmaps")
        _window_title(self.fig_controls, "Controls")
        self.meshes: list[Any] = []
        self.colorbars: list[Any] = []
        self.axes: list[Any] = []
        self.sliders: list[tuple[str, Slider]] = []
        self._threshold_sliders: dict[str, Slider] = {}
        self._metric_axes: list[Any] = []
        self._slider_value_texts: dict[str, Any] = {}
        self._threshold_value_texts: dict[str, Any] = {}
        self._main_size_initialized = False
        self._controls_size_initialized = False
        self._click_connection = self.fig_main.canvas.mpl_connect(
            "button_press_event", self._on_click
        )
        self._rebuild_controls()
        self._draw_heatmaps()

    def _get_slider_dims(self) -> list[tuple[int, str]]:
        return [
            (index, axis.key)
            for index, axis in enumerate(self.dataset.axes)
            if axis.key in self.state.slider_indices and not axis.is_singleton
        ]

    def _disconnect_controls(self) -> None:
        for widget in (
            getattr(self, "x_radio", None),
            getattr(self, "y_radio", None),
            *(slider for _, slider in getattr(self, "sliders", ())),
            *getattr(self, "_threshold_sliders", {}).values(),
        ):
            if widget is not None:
                widget.disconnect_events()

    def _rebuild_controls(self) -> None:
        self._disconnect_controls()
        self.fig_controls.clear()
        _window_title(self.fig_controls, "Controls")
        active_keys = [axis.key for axis in self.dataset.axes if not axis.is_singleton]
        keys = active_keys if len(active_keys) >= 2 else list(self.dataset.axis_keys)
        self._radio_keys = keys
        radio_labels = [self._axis_views[key].name for key in keys]
        self._radio_label_to_key = dict(zip(radio_labels, keys, strict=True))
        selected = set(self.state.tiles)
        masked = {
            name for name in selected
            if is_masked_metric(name, self.dataset.metrics)
        }
        detach_masked = any(
            masked_metric_uses_detach(name, self.dataset.metrics)
            for name in masked
        )
        slippage_sources = masked_metric_slippage_sources(
            tuple(masked), self.dataset.metrics
        )
        price_source = self._mask_source("max_7d_rel_price_diff", "max_rel_price_diff")
        threshold_count = sum((
            bool(masked and price_source),
            bool(detach_masked and "detach_energy_ungated" in self.dataset.metrics),
            bool(slippage_sources),
        ))
        slider_count = len(self._get_slider_dims()) + threshold_count
        radio_height = len(keys) * 0.03 + 0.01
        content_height = (
            0.03 + 0.01 + radio_height + 0.02
            + slider_count * _CONTROL_ROW_PITCH + 0.02
        )
        height_multiplier = 11.0 + max(0, len(keys) - 5) * 0.3
        if not self._controls_size_initialized:
            self.fig_controls.set_size_inches(3.2, max(6.0, content_height * height_multiplier + 1.0))
            self._controls_size_initialized = True
        self.fig_controls.text(
            0.5, 0.98, "Dimension Controls", ha="center", va="top", fontsize=10, fontweight="bold"
        )
        radio_top = 0.94
        radio_bottom = radio_top - radio_height
        radio_columns = (("X", 0.05), ("Y", 0.23))
        radio_boxes = []
        for title, left in radio_columns:
            self.fig_controls.text(left, 0.95, f"{title} axis:", ha="left", va="top", fontsize=9)
            box = self.fig_controls.add_axes((left, radio_bottom, 0.16, radio_height))
            box.set_frame_on(False)
            radio_boxes.append(box)
        self.fig_controls.text(0.43, 0.95, "Dimension", ha="left", va="top", fontsize=9)
        label_rows = np.linspace(
            1.0,
            0.0,
            len(keys) + 2,
        )
        for row_y, label in zip(label_rows[1:-1], radio_labels, strict=True):
            self.fig_controls.text(
                0.43,
                radio_bottom + radio_height * row_y,
                label,
                ha="left",
                va="center",
                fontsize=8,
            )
        x_box, y_box = radio_boxes
        self.x_radio = RadioButtons(x_box, radio_labels, active=keys.index(self.state.x_axis))
        self.y_radio = RadioButtons(y_box, radio_labels, active=keys.index(self.state.y_axis))
        self.x_radio.on_clicked(self._on_x_changed)
        self.y_radio.on_clicked(self._on_y_changed)
        for radio in (self.x_radio, self.y_radio):
            for label in radio.labels:
                label.set_visible(False)
        self.sliders = []
        self._slider_value_texts = {}
        y = radio_bottom - 0.02
        for _, key in self._get_slider_dims():
            axis = self.dataset.axis(key)
            view = self._axis_views[key]
            self.fig_controls.text(0.05, y, f"{view.name}:", ha="left", va="bottom", fontsize=8)
            control_ax = self.fig_controls.add_axes((0.20, y - 0.04, 0.62, 0.025))
            slider = Slider(
                control_ax, "", 0, len(axis.values) - 1,
                valinit=self.state.slider_indices[key], valstep=1,
            )
            slider.valtext.set_visible(False)
            self._slider_value_texts[key] = self.fig_controls.text(
                0.20,
                y - 0.05,
                view.labels[self.state.slider_indices[key]],
                ha="left",
                va="top",
                fontsize=8,
            )
            slider.on_changed(lambda value, name=key: self._on_dimension_slider(name, value))
            self.sliders.append((key, slider))
            y -= _CONTROL_ROW_PITCH
        self._threshold_sliders = {}
        self._threshold_value_texts = {}
        if masked and price_source:
            y = self._add_threshold_slider(
                "max_pricethr", "max 7d pdiff thr (bps)", y,
                max_pricethr=self.state.mask.max_price_diff_bps,
                source=price_source,
                scale=10_000.0,
            )
        if slippage_sources:
            y = self._add_threshold_slider(
                "slipthr", "slippage cap (bps)", y,
                max_pricethr=self.state.mask.slippage_thr_bps,
                minimum_maximum=100.0,
            )
        if detach_masked and "detach_energy_ungated" in self.dataset.metrics:
            y = self._add_threshold_slider(
                "detachthr", "detach energy max", y,
                max_pricethr=self.state.mask.max_detach_energy,
                source="detach_energy_ungated",
            )
        self.fig_controls.text(0.08, max(0.02, y - 0.02), "Shift+click / right-click: exact replay", fontsize=8)
        self.fig_controls.canvas.draw_idle()

    def _mask_source(self, *names: str) -> str | None:
        return next((name for name in names if name in self.dataset.metrics), None)

    def _add_threshold_slider(
        self,
        key: str,
        label: str,
        y: float,
        *,
        max_pricethr: float | None,
        source: str | None = None,
        scale: float = 1.0,
        minimum_maximum: float = 1.0,
    ) -> float:
        maximum = _finite_max(self.dataset, source, scale) if source else None
        maximum = max(
            minimum_maximum,
            float(maximum or 0.0),
            float(max_pricethr or 0.0),
        )
        initial = maximum if max_pricethr is None else min(maximum, max(0.0, float(max_pricethr)))
        self.fig_controls.text(0.05, y, f"{label}:", ha="left", va="bottom", fontsize=8)
        control_ax = self.fig_controls.add_axes((0.20, y - 0.04, 0.62, 0.025))
        slider = Slider(control_ax, "", 0.0, maximum, valinit=initial)
        slider.valtext.set_visible(False)
        self._threshold_value_texts[key] = self.fig_controls.text(
            0.20, y - 0.05, f"{initial:.4g}", ha="left", va="top", fontsize=8
        )
        slider.on_changed(lambda value, name=key: self._on_threshold_slider(name, value))
        self._threshold_sliders[key] = slider
        return y - _CONTROL_ROW_PITCH

    def _on_dimension_slider(self, key: str, value: float) -> None:
        self.state.slider_indices[key] = int(value)
        if key in self._slider_value_texts:
            self._slider_value_texts[key].set_text(self._axis_views[key].labels[int(value)])
        self._draw_heatmaps()

    def _on_threshold_slider(self, key: str, value: float) -> None:
        mask = self.state.mask
        values = {
            "max_price_diff_bps": mask.max_price_diff_bps,
            "max_detach_energy": mask.max_detach_energy,
            "max_final_price_diff_bps": mask.max_final_price_diff_bps,
            "slippage_thr_bps": mask.slippage_thr_bps,
        }
        values[
            {
                "max_pricethr": "max_price_diff_bps",
                "detachthr": "max_detach_energy",
                "slipthr": "slippage_thr_bps",
                "final_pdiffthr": "max_final_price_diff_bps",
            }[key]
        ] = float(value)
        self.state.mask = MaskSpec(**values)
        if key in self._threshold_value_texts:
            self._threshold_value_texts[key].set_text(f"{value:.4g}")
        self._draw_heatmaps()

    def _swap_axes(self, axis: str, value: str) -> None:
        if self._updating_radios:
            return
        if value == (self.state.x_axis if axis == "x" else self.state.y_axis):
            return
        if axis == "x":
            if value == self.state.y_axis:
                old = self.state.x_axis
                self.state.x_axis, self.state.y_axis = value, old
                self._updating_radios = True
                self.y_radio.set_active(self._radio_keys.index(old))
                self._updating_radios = False
            else:
                self.state.x_axis = value
        else:
            if value == self.state.x_axis:
                old = self.state.y_axis
                self.state.y_axis, self.state.x_axis = value, old
                self._updating_radios = True
                self.x_radio.set_active(self._radio_keys.index(old))
                self._updating_radios = False
            else:
                self.state.y_axis = value
        old_indices = dict(self.state.slider_indices)
        self.state.slider_indices = {
            key: min(old_indices.get(key, 0), len(self.dataset.axis(key).values) - 1)
            for key in self.dataset.axis_keys
            if key not in {self.state.x_axis, self.state.y_axis}
        }
        self._rebuild_controls()
        self._draw_heatmaps()

    def _on_x_changed(self, value: str) -> None:
        self._swap_axes("x", self._radio_label_to_key[value])

    def _on_y_changed(self, value: str) -> None:
        self._swap_axes("y", self._radio_label_to_key[value])

    def _draw_heatmaps(self) -> None:
        self.fig_main.clear()
        _window_title(self.fig_main, "Heatmaps")
        n = len(self.state.tiles)
        cols = min(self.state.ncol, n)
        rows = max(1, math.ceil(n / cols))
        x_view = self._axis_views[self.state.x_axis]
        y_view = self._axis_views[self.state.y_axis]
        if not self._main_size_initialized:
            self.fig_main.set_size_inches(
                *_main_figure_size(len(x_view.centers), len(y_view.centers), rows, cols)
            )
            self._main_size_initialized = True
        self._metric_axes = [self.fig_main.add_subplot(rows, cols, i + 1) for i in range(n)]
        self.axes = self._metric_axes
        self.meshes = []
        self.colorbars = []
        x_axis = self.dataset.axis(self.state.x_axis)
        y_axis = self.dataset.axis(self.state.y_axis)
        x_edges = edges(x_view.centers, logarithmic=x_view.logarithmic)
        y_edges = edges(y_view.centers, logarithmic=y_view.logarithmic)
        x_tick_indices = _select_ticks(len(x_view.centers))
        y_tick_indices = _select_ticks(len(y_view.centers))
        tick_font = _auto_font_size(len(x_view.centers), len(y_view.centers))
        label_font = max(8, tick_font + 2)
        title_font = max(label_font, tick_font + 4)
        for index, (axis, metric) in enumerate(zip(self._metric_axes, self.state.tiles, strict=True)):
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
            metric_scale, metric_suffix = _metric_scale_info(metric)
            display_values = np.ma.masked_invalid(values) * metric_scale
            mesh = axis.pcolormesh(x_edges, y_edges, display_values, shading="auto", cmap="turbo")
            mesh.set_clim(*_finite_clim(np.asarray(values, dtype=float) * metric_scale))
            self.meshes.append(mesh)
            if x_view.logarithmic:
                axis.set_xscale("log")
            if y_view.logarithmic:
                axis.set_yscale("log")
            ny, nx = display_values.shape
            try:
                axis.set_box_aspect(ny / nx)
            except AttributeError:
                axis.set_aspect("equal", adjustable="box")
            _apply_fixed_ticks(
                axis.xaxis,
                [x_view.centers[i] for i in x_tick_indices],
                [x_view.labels[i] for i in x_tick_indices],
            )
            axis.tick_params(axis="x", labelrotation=45, labelsize=tick_font)
            for tick_label in axis.get_xticklabels():
                tick_label.set_ha("right")
            first_column = index % cols == 0
            _apply_fixed_ticks(
                axis.yaxis,
                [y_view.centers[i] for i in y_tick_indices],
                [y_view.labels[i] for i in y_tick_indices] if first_column else [],
            )
            axis.tick_params(axis="y", labelsize=tick_font)
            axis.set_xlabel(x_view.name, fontsize=label_font)
            if first_column:
                axis.set_ylabel(y_view.name, fontsize=label_font)
            axis.set_title(f"{metric}{metric_suffix}", fontsize=title_font)
            colorbar = self.fig_main.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(f"{metric}{metric_suffix}", fontsize=tick_font)
            colorbar.ax.tick_params(labelsize=tick_font)
            colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
            self.colorbars.append(colorbar)
            axis.format_coord = self._format_coord(metric)
        for index in range(n, rows * cols):
            self.fig_main.add_subplot(rows, cols, index + 1).axis("off")
        self.fig_main.canvas.draw_idle()

    def _format_coord(self, metric: str) -> Callable[[float, float], str]:
        x_axis = self.dataset.axis(self.state.x_axis)
        y_axis = self.dataset.axis(self.state.y_axis)
        x_view = self._axis_views[x_axis.key]
        y_view = self._axis_views[y_axis.key]

        def format_coord(x: float, y: float) -> str:
            xi = _cell_index_from_view(x_view, float(x))
            yi = _cell_index_from_view(y_view, float(y))
            if xi is None or yi is None:
                return ""
            indices = [self.state.slider_indices.get(axis.key, 0) for axis in self.dataset.axes]
            indices[self.dataset.axis_index(x_axis.key)] = xi
            indices[self.dataset.axis_index(y_axis.key)] = yi
            try:
                self.dataset.point(indices)
            except ValueError:
                return ""
            tile_mask = (
                self.state.mask
                if is_masked_metric(metric, self.dataset.metrics)
                else MaskSpec()
            )
            value = self.dataset.metric_array(metric, tile_mask)[tuple(indices)]
            return (
                f"x={x_view.labels[xi]}, y={y_view.labels[yi]}, "
                f"{metric}={format_metric_value(metric, value)}"
            )

        return format_coord

    def _selection_from_event(self, event: MouseEvent) -> HeatmapSelection | None:
        if event.inaxes not in self._metric_axes or event.xdata is None or event.ydata is None:
            return None
        x_axis = self.dataset.axis(self.state.x_axis)
        y_axis = self.dataset.axis(self.state.y_axis)
        x_index = _cell_index_from_view(self._axis_views[x_axis.key], float(event.xdata))
        y_index = _cell_index_from_view(self._axis_views[y_axis.key], float(event.ydata))
        if x_index is None or y_index is None:
            return None
        indices = [self.state.slider_indices.get(axis.key, 0) for axis in self.dataset.axes]
        indices[self.dataset.axis_index(x_axis.key)] = x_index
        indices[self.dataset.axis_index(y_axis.key)] = y_index
        try:
            return self.dataset.point(indices)
        except ValueError:
            return None

    def _on_click(self, event: MouseEvent) -> None:
        selection = self._selection_from_event(event)
        if selection is None:
            return
        self.last_selection = selection
        key = str(getattr(event, "key", "") or "").lower()
        right = event.button in {MouseButton.RIGHT, 3}
        shifted = event.button in {MouseButton.LEFT, 1} and "shift" in key
        if right or shifted:
            self.replay(selection, "right" if right else "shift")
        elif event.button in {MouseButton.LEFT, 1}:
            coordinates = ", ".join(
                f"{name}={format_axis_value(value)}"
                for name, value in selection.coordinates.items()
            )
            metrics = []
            for metric in self.state.tiles:
                tile_mask = (
                    self.state.mask
                    if is_masked_metric(metric, self.dataset.metrics)
                    else MaskSpec()
                )
                value = self.dataset.metric_array(metric, tile_mask, selection=selection.grid_indices)
                metrics.append(f"{metric}={format_metric_value(metric, value)}")
            print(
                f"selection candidate={selection.candidate_id} ordinal={selection.ordinal} "
                f"({coordinates}) | " + " | ".join(metrics)
            )

    def replay(self, selection: HeatmapSelection, mode: str = "shift") -> Any:
        """Forward an exact selection to the caller's local replay callback."""
        selection_ref = selection.to_selection_ref(self.run_id)
        if self.on_replay is not None:
            return self.on_replay(selection_ref, mode)
        raise RuntimeError("replay requires an on_replay callback")

    def save(self, output: Path | str, *, state_path: Path | str | None = None) -> tuple[Path, Path]:
        image_path = Path(output)
        sidecar = Path(state_path) if state_path is not None else image_path.with_suffix(".state.json")
        if image_path.exists() or sidecar.exists():
            raise FileExistsError("interactive explorer outputs are immutable")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{image_path.name}.", suffix=image_path.suffix, dir=image_path.parent)
        os.close(fd)
        try:
            self.fig_main.savefig(temporary_name, format=image_path.suffix.lstrip("."), dpi=150, bbox_inches="tight")
            os.replace(temporary_name, image_path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        payload = self.state.to_dict()
        payload["explorer"] = {
            "window_titles": ["Heatmaps", "Controls"],
            "run_id": self.run_id,
            "selection": self.last_selection.to_selection_ref(self.run_id).to_dict() if self.last_selection else None,
        }
        atomic_write_json(sidecar, payload)
        return image_path, sidecar

    def show(self, *, block: bool = True) -> None:
        plt.show(block=block)

    def close(self) -> None:
        self._disconnect_controls()
        self.fig_main.canvas.mpl_disconnect(self._click_connection)
        for figure in (self.fig_main, self.fig_controls):
            plt.close(figure)


__all__ = [
    "HeatmapExplorer",
    "format_axis_value",
    "format_metric_value",
]
