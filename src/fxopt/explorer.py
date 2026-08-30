"""Interactive mature heatmaps backed directly by fxopt's two result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from curve_fx_sim.plotting.explorer import HeatmapExplorer
from curve_fx_sim.plotting.heatmap import HeatmapAxis, HeatmapDataset
from curve_fx_sim.plotting.masked_metrics import (
    SKEW_MASKED_METRICS,
    SLIPPAGE_APY_MASK_SOURCES,
    masked_metric_source,
    masked_metric_uses_detach,
)
from .results import ResultColumns, read_result_columns
from .shiftclick import shiftclick_figure, trace_stored_candidate


def _metric_columns(requested: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
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
        if masked_metric_uses_detach(name, available_set) and "detach_energy_ungated" in available_set:
            needed.add("detach_energy_ungated")
        if name in SKEW_MASKED_METRICS:
            needed.update(available_set.intersection({"max_7d_skew", "final_rel_price_diff"}))
        slippage = SLIPPAGE_APY_MASK_SOURCES.get(name)
        if slippage is not None:
            needed.add(slippage)
    return tuple(name for name in available if name in needed)


def _heatmap_axis(name: str, raw_values: object) -> HeatmapAxis:
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"run axis {name!r} has invalid values")
    values = tuple(raw_values)
    grouped = isinstance(values[0], dict)
    if any(isinstance(value, dict) != grouped for value in values):
        raise ValueError(f"run axis {name!r} mixes scalar and grouped values")
    if not grouped:
        return HeatmapAxis.from_metadata({"name": name, "values": values})

    member_names = tuple(sorted(values[0]))
    if not member_names or any(tuple(sorted(value)) != member_names for value in values):
        raise ValueError(f"run axis {name!r} grouped values have inconsistent members")
    rows = tuple(tuple(value[member] for member in member_names) for value in values)
    if len(member_names) == 1:
        return HeatmapAxis.from_metadata({
            "name": member_names[0],
            "values": tuple(row[0] for row in rows),
        })
    return HeatmapAxis.from_metadata({"names": member_names, "rows": rows})


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
        candidate_ids=columns.candidate_ids_array().reshape(shape),
        ordinals=columns.ordinals.reshape(shape),
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
    max_skew_percent: float | None = None,
    slippage_bps: float | None = None,
    final_price_diff_bps: float | None = None,
) -> HeatmapExplorer:
    """Open the interactive UI from columnar run results."""
    root = Path(run_dir).expanduser().resolve()
    try:
        available_metrics = tuple(
            json.loads((root / "run.json").read_bytes())["metric_names"]
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("run has no readable metric manifest") from exc
    selected_columns = _metric_columns(metrics, available_metrics)
    results = read_result_columns(root, metrics=selected_columns)

    def replay(selection: Any, mode: str) -> Path:
        ordinal = int(selection.index)
        candidate = results.candidate_at(ordinal)
        if candidate.candidate_id != selection.candidate_id:
            raise ValueError("selected candidate does not match the stored result row")
        output = root / "inspections" / f"ordinal-{ordinal}"
        summary = trace_stored_candidate(
            results.run_id,
            results.metadata,
            candidate=candidate,
            ordinal=ordinal,
            output_dir=output,
            trace_interval=200,
            trace_actions=False,
            yb_mode=None if mode == "shift" else "off",
        )
        figure = shiftclick_figure(summary, title=f"{results.run_id}: {ordinal}")
        figure.show()
        return summary

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
        skewthr=max_skew_percent,
        slipthr=slippage_bps,
        final_pdiffthr=final_price_diff_bps,
        run_id=results.run_id,
        run_dir=root,
        on_replay=replay,
    )


__all__ = ["open_fxopt_explorer"]
