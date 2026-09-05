"""Interactive mature heatmaps backed directly by fxopt's two result files."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from curve_fx_sim.plotting.explorer import HeatmapExplorer
from curve_fx_sim.plotting.heatmap import (
    HeatmapAxis,
    HeatmapDataset,
    HeatmapSelection,
    infer_scale,
)
from curve_fx_sim.plotting.masked_metrics import (
    masked_metric_slippage_source,
    masked_metric_source,
    masked_metric_uses_slippage,
)
from .results import ResultColumns, read_result_columns
from .shiftclick import shiftclick_figure, trace_stored_candidate


def _metric_columns(
    requested: Sequence[str],
    available: Sequence[str],
    *,
    need_final_price_diff: bool = False,
    need_slippage: bool = False,
) -> tuple[str, ...]:
    available_set = set(available)
    needed: set[str] = set()
    for name in requested:
        source = masked_metric_source(name, available_set)
        needed.add(source or name)
        if source is None:
            continue
        price_diff = next(
            (item for item in ("max_7d_rel_price_diff", "max_rel_price_diff")
             if item in available_set),
            None,
        )
        if price_diff is not None:
            needed.add(price_diff)
        if "detach_energy_ungated" in available_set:
            needed.add("detach_energy_ungated")
        if need_final_price_diff and "final_rel_price_diff" in available_set:
            needed.add("final_rel_price_diff")
        slippage_source = masked_metric_slippage_source(name)
        if slippage_source is not None or (
            need_slippage and masked_metric_uses_slippage(name, available_set)
        ):
            slippage_source = slippage_source or "tw_real_slippage_1pct"
            if slippage_source in available_set:
                needed.add(slippage_source)
    return tuple(name for name in available if name in needed)


def _heatmap_axis(name: str, raw_values: object) -> HeatmapAxis:
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"run axis {name!r} has invalid values")
    values = tuple(raw_values)
    grouped = isinstance(values[0], dict)
    if any(isinstance(value, dict) != grouped for value in values):
        raise ValueError(f"run axis {name!r} mixes scalar and grouped values")
    if not grouped:
        return HeatmapAxis(names=(name,), values=values, scale=infer_scale(values))

    member_names = tuple(sorted(values[0]))
    if not member_names or any(tuple(sorted(value)) != member_names for value in values):
        raise ValueError(f"run axis {name!r} grouped values have inconsistent members")
    rows = tuple(tuple(value[member] for member in member_names) for value in values)
    if len(member_names) == 1:
        scalar_values = tuple(row[0] for row in rows)
        return HeatmapAxis(
            names=member_names,
            values=scalar_values,
            scale=infer_scale(scalar_values),
        )
    return HeatmapAxis(names=member_names, values=rows, scale="categorical")


def _dataset(columns: ResultColumns) -> HeatmapDataset:
    raw_axes = columns.metadata.get("axes")
    raw_shape = columns.metadata.get("shape")
    if not isinstance(raw_axes, dict) or not isinstance(raw_shape, list):
        raise ValueError("run has no Cartesian axis metadata")
    names = tuple(sorted(raw_axes))
    axes = tuple(_heatmap_axis(name, raw_axes[name]) for name in names)
    shape = tuple(int(value) for value in raw_shape)
    if shape != tuple(len(axis.values) for axis in axes):
        raise ValueError("run axis metadata and shape disagree")
    if int(np.prod(shape)) != columns.row_count:
        raise ValueError("interactive heatmaps require a complete Cartesian grid")

    metrics = {
        name: np.asarray(values, dtype=float).reshape(shape)
        for name, values in columns.metrics.items()
    }
    return HeatmapDataset(
        axes=axes,
        metrics=metrics,
        valid=columns.ok_mask.reshape(shape),
        metadata=dict(columns.metadata),
    )


def open_fxopt_explorer(
    run_dir: str | Path,
    *,
    metrics: Sequence[str],
    columns: int = 3,
    x_axis: str | None = None,
    y_axis: str | None = None,
    log_axes: Sequence[str] = (),
    max_price_diff_bps: float | None = 100.0,
    max_detach_energy: float | None = None,
    slippage_bps: float | None = None,
    final_price_diff_bps: float | None = None,
    shiftclick_yb_mode: str = "active_2l",
    shiftclick_yb_cash_multiplier: float = 3.0,
) -> HeatmapExplorer:
    """Open the interactive UI from columnar run results."""
    root = Path(run_dir).expanduser().resolve()
    try:
        available_metrics = tuple(
            json.loads((root / "run.json").read_bytes())["metric_names"]
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("run has no readable metric manifest") from exc
    selected_columns = _metric_columns(
        metrics,
        available_metrics,
        need_final_price_diff=final_price_diff_bps is not None,
        need_slippage=slippage_bps is not None,
    )
    results = read_result_columns(root, metrics=selected_columns)

    def replay(selection: HeatmapSelection, mode: str):
        ordinal = int(selection.ordinal)
        candidate = results.candidate_at(ordinal)
        if candidate.candidate_id != selection.candidate_id:
            raise ValueError("selected candidate does not match the stored result row")
        coordinates = ", ".join(
            f"{name}={value}" for name, value in selection.coordinates.items()
        )
        print(f"replay ordinal={ordinal} ({coordinates})", flush=True)
        with tempfile.TemporaryDirectory(prefix="fxopt-replay-") as output:
            summary = trace_stored_candidate(
                results.run_id, results.metadata,
                candidate=candidate, ordinal=ordinal, output_dir=output,
                trace_interval=200, trace_actions=False,
                yb_mode=shiftclick_yb_mode if mode == "shift" else "off",
                yb_cash_multiplier=shiftclick_yb_cash_multiplier if mode == "shift" else None,
            )
            figure = shiftclick_figure(summary, title=f"{results.run_id}: {ordinal}")
        figure.show()
        return figure

    dataset = _dataset(results)
    raw_axes = results.metadata["axes"]
    aliases = {
        name: axis.key
        for name, axis in zip(sorted(raw_axes), dataset.axes, strict=True)
    }
    return HeatmapExplorer(
        dataset,
        metrics=tuple(metrics),
        ncol=columns,
        x_axis=aliases.get(x_axis, x_axis),
        y_axis=aliases.get(y_axis, y_axis),
        log_axes=tuple(aliases.get(name, name) for name in log_axes),
        max_pricethr=max_price_diff_bps,
        max_detach_energy=max_detach_energy,
        slipthr=slippage_bps,
        final_pdiffthr=final_price_diff_bps,
        run_id=results.run_id,
        on_replay=replay,
    )


__all__ = ["open_fxopt_explorer"]
