#!/usr/bin/env python3
"""
N-dimensional heatmap explorer (fxsim port of plot_heatmap_nd_opt.py).

Near-verbatim port of the legacy interactive explorer: separate Heatmaps /
Controls / Metrics windows, X/Y radio buttons, per-dimension and threshold
sliders, turbo colormap, per-tile colorbars, masked metrics, hover coordinate
readout, and click-to-inspect. The data layer is a thin adapter
(plotting/viewer_data.py) over a collected fxsim run directory
(manifest.json + evaluation_table.npz).

Key improvements vs legacy:
- NPZ-only data path (no legacy arb_run JSON support)
- Inspect clicks replay the exact cell through the unified harness
  (curve_fx_eval_v1) and open the attested trajectory, instead of the legacy
  arb_sim.py + plot_price_scale.py subprocess pipeline
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib

HAS_OUT_ARG = any(arg == "--out" or arg.startswith("--out=") for arg in sys.argv[1:])
if HAS_OUT_ARG:
    matplotlib.use("Agg")
else:
    # Force interactive backend for the normal explorer workflow.
    for backend in ["macosx", "TkAgg", "Qt5Agg"]:
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.ticker import FixedFormatter, FixedLocator, FormatStrFormatter, NullFormatter, NullLocator

from .masked_metrics import (
    MASKED_METRICS,
    MASKED_METRIC_SOURCES,
    PDIFF_ONLY_MASKED_METRICS,
    SKEW_MASKED_METRICS,
    SLIPPAGE_APY_MASK_SOURCES,
)
from .viewer_data import load_viewer_data

LN2 = math.log(2)
SECONDS_PER_HOUR = 3600.0
MA_TIME_TO_HOURS = LN2 / SECONDS_PER_HOUR
MASK_MAX_PRICE_METRIC = "max_7d_rel_price_diff"
MASK_MAX_PRICE_FALLBACK_METRIC = "max_rel_price_diff"
MASK_FINAL_PRICE_METRIC = "final_rel_price_diff"
MASK_SKEW_METRIC = "max_7d_skew"

# ==================== LAYOUT CONSTANTS ====================
CTRL_FIG_WIDTH = 3.2
CTRL_MIN_HEIGHT = 6.0
CTRL_HEIGHT_MULT = 11.0
CTRL_HEIGHT_PAD = 1.0

CTRL_TITLE_Y = 0.98
CTRL_TITLE_FONTSIZE = 10

RADIO_ITEM_HEIGHT = 0.03
RADIO_BOX_PADDING = 0.01
RADIO_FONTSIZE = 8
RADIO_X_LABEL_Y = 0.95
RADIO_LABEL_GAP = 0.01
RADIO_GROUP_GAP = 0.02

SLIDER_HEIGHT = 0.075
SLIDER_TOP_GAP = 0.02
SLIDER_LABEL_OFFSET = 0.015
SLIDER_BOX_LEFT = 0.20
SLIDER_BOX_WIDTH = 0.62
SLIDER_BOX_HEIGHT = 0.03
SLIDER_BOX_Y_OFFSET = 0.02
SLIDER_VALUE_X = SLIDER_BOX_LEFT
SLIDER_VALUE_Y_OFFSET = -0.035
SLIDER_FONTSIZE = 8
# ==========================================================

DEFAULT_METRICS = [
    "apy_net",
    "apy_corr",
    "tw_real_slippage",
    "rel_price_diff_geom_mean",
    "virtual_price",
    "xcp_profit",
]

DEFAULT_COSTS = {
    "arb_fee_bps": 10.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1,
}

def _parse_log_axes(raw_values: List[str]) -> set[str]:
    return {
        name.strip()
        for raw in raw_values
        for name in raw.split(",")
        if name.strip()
    }


def _load(path: Path) -> Dict[str, Any]:
    return load_viewer_data(path)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _max_mask_price_diff(metrics: Dict[str, Any]) -> float:
    value = _to_float(metrics.get(MASK_MAX_PRICE_METRIC))
    if math.isfinite(value):
        return value
    return _to_float(metrics.get(MASK_MAX_PRICE_FALLBACK_METRIC))


def _bps_to_raw_ratio(bps: float) -> float:
    return float(bps) / 10_000.0


def _bps_to_display_percent(bps: float) -> float:
    return float(bps) / 100.0


def _parse_grid_dims(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Parse grid dimensions from metadata.
    Returns list of (dim_key, dim_name) tuples sorted by dim_key (x1, x2, ...).
    """
    grid = data.get("metadata", {}).get("grid", {})
    dims = []
    for key, val in grid.items():
        if isinstance(key, str) and key.lower().startswith("x") and key[1:].isdigit():
            idx = int(key[1:])
            name = val.get("name") if isinstance(val, dict) else None
            if name:
                dims.append((idx, key, name))
    dims.sort(key=lambda t: t[0])
    return [(key, name) for _, key, name in dims]


def _axis_normalization(name: str) -> Tuple[float, str]:
    key = (name or "").lower()
    if name == "A" or key == "a":
        return 1e4, " (÷1e4)"
    if key in {"reserved_profit_fraction", "admin_fee"}:
        return 1e10, " (÷1e10)"
    if "fee_bps" in key:
        return 1.0, " (bps)"
    if "fee" in key and "gamma" not in key:
        return 0.0, " (bps)"
    if "ma_time" in key or "price_source_ema_half_time" in key:
        return 1.0, " (hrs)" if "hrs" not in key else ""
    if key.endswith("_wad"):
        return 1e18, " (/1e18)"
    if "gamma" in key:
        return 1e18, " (/1e18)"
    if "liquidity" in key or "balance" in key:
        return 1e18, " (/1e18)"
    return 1.0, ""


def _is_time_axis(name: str) -> bool:
    key = (name or "").lower()
    return "ma_time" in key or "price_source_ema_half_time" in key


def _format_duration_short(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d" if not hours else f"{days}d{hours}h"
    if hours:
        return f"{hours}h" if not minutes else f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m" if not seconds else f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _format_axis_labels(name: str, values: List[float]) -> Tuple[List[str], str]:
    scale, suffix = _axis_normalization(name)
    key = (name or "").lower()
    display_name = name or ""

    if key in {"reserved_profit_fraction", "admin_fee"}:
        finite = [abs(float(v)) for v in values if math.isfinite(float(v))]
        is_scaled = bool(finite) and max(finite) > 1.0
        if is_scaled:
            return [f"{(v / 1e10):.2f}" for v in values], f"{name} (÷1e10)"
        return [f"{v:.4g}" for v in values], display_name

    if suffix and suffix not in display_name:
        display_name = f"{display_name}{suffix}"

    if "fee_bps" in key:
        labels = [f"{v:.2f}" for v in values]
        return labels, display_name
    if scale == 0.0 and "fee" in key and "gamma" not in key:
        labels = [f"{(v / 1e10 * 1e4):.2f}" for v in values]
        return labels, f"{name} (bps)"
    if _is_time_axis(name):
        labels = [_format_duration_short(v * LN2) for v in values]
        return labels, display_name
    if key.endswith("_wad"):
        labels = [f"{(v / 1e18):.6g}" for v in values]
        return labels, display_name
    if "gamma" in key:
        labels = [f"{(v / 1e18):.5f}" for v in values]
        return labels, name
    if scale != 1.0:
        labels = [f"{(v / scale):.2f}" for v in values]
        return labels, display_name
    labels = [f"{v:.4g}" for v in values]
    return labels, display_name


def _apply_fixed_ticks(axis, values: List[float], labels: List[str]) -> None:
    if not labels:
        labels = [""] * len(values)
    axis.set_major_locator(FixedLocator(values))
    axis.set_major_formatter(FixedFormatter(labels))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())
    axis.get_offset_text().set_visible(False)


def _format_slider_value(name: str, value: float) -> str:
    scale, _ = _axis_normalization(name)
    key = (name or "").lower()
    if "fee_bps" in key:
        return f"{value:.1f} bps"
    if key in {"reserved_profit_fraction", "admin_fee"}:
        return f"{(value / 1e10 if abs(value) > 1.0 else value):.4f}"
    if scale == 0.0 and "fee" in key and "gamma" not in key:
        return f"{(value / 1e10 * 1e4):.1f} bps"
    if _is_time_axis(name):
        return _format_duration_short(value * LN2)
    if name == "A" or key == "a":
        return f"{value / 1e4:.2f}"
    if key.endswith("_wad"):
        return f"{value / 1e18:.6g}"
    if "gamma" in key:
        return f"{value / 1e18:.6f}"
    if "apy" in key or "ratio" in key:
        return f"{value:.4f}"
    return f"{value:.4g}"


def _format_coupled_slider_value(name: str, value: float) -> str:
    key = (name or "").lower()
    if key in {"mid_fee", "out_fee"}:
        bps = value / 1e10 * 1e4
        return f"{bps:.0f}" if abs(bps - round(bps)) < 1e-9 else f"{bps:.1f}"
    if key == "fee_gamma":
        return f"{value / 1e18:.6g}"
    return _format_slider_value(name, value)


def _is_auto_log_axis(values: List[float]) -> bool:
    if len(values) < 3:
        return False
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        return False
    log_diffs = np.diff(np.log(arr))
    lin_diffs = np.diff(arr)
    if np.any(log_diffs <= 0) or np.any(lin_diffs <= 0):
        return False
    log_mean = float(np.mean(log_diffs))
    lin_mean = float(np.mean(lin_diffs))
    if log_mean <= 0.0 or lin_mean <= 0.0:
        return False
    log_cv = float(np.std(log_diffs) / log_mean)
    lin_cv = float(np.std(lin_diffs) / lin_mean)
    if log_cv < 0.05 and log_cv < lin_cv:
        return True

    # Tolerate one forced-included grid value inside an otherwise log-spaced
    # axis. That perturbs the two adjacent log gaps but should not make the
    # axis render linearly.
    log_median = float(np.median(log_diffs))
    if log_median <= 0.0:
        return False
    close_share = float(np.mean(np.abs(log_diffs / log_median - 1.0) < 0.15))
    return close_share >= 0.85 and log_cv < lin_cv


def _auto_log_axes(dim_values: Dict[str, List[float]]) -> set[str]:
    return {name for name, values in dim_values.items() if _is_auto_log_axis(values)}


def _coupled_axis_value_labels(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    grid = metadata.get("grid", {}) if isinstance(metadata, dict) else {}
    if not isinstance(grid, dict):
        return {}
    labels: Dict[str, List[str]] = {}
    for key in sorted(
        (k for k in grid if isinstance(k, str) and k.startswith("x") and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    ):
        axis = grid.get(key)
        if not isinstance(axis, dict) or "names" not in axis or "values" not in axis:
            continue
        names = [str(name) for name in axis.get("names", [])]
        axis_name = "/".join(names)
        axis_labels = []
        for row in axis.get("values", []):
            if not isinstance(row, list) or len(row) != len(names):
                axis_labels.append(str(row))
                continue
            parts = [
                _format_coupled_slider_value(name, float(value))
                for name, value in zip(names, row)
            ]
            axis_labels.append("(" + "/".join(parts) + ")")
        labels[axis_name] = axis_labels
    return labels


def _categorical_axis_values(metadata: Dict[str, Any]) -> Dict[str, List[Any]]:
    grid = metadata.get("grid", {}) if isinstance(metadata, dict) else {}
    if not isinstance(grid, dict):
        return {}
    values_by_axis: Dict[str, List[Any]] = {}
    for key in sorted(
        (k for k in grid if isinstance(k, str) and k.startswith("x") and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    ):
        axis = grid.get(key)
        if not isinstance(axis, dict) or "name" not in axis or "values" not in axis:
            continue
        raw_values = list(axis.get("values", []))
        try:
            [float(value) for value in raw_values]
        except (TypeError, ValueError):
            values_by_axis[str(axis["name"])] = raw_values
    return values_by_axis


def _categorical_axis_value_labels(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        name: [str(value) for value in values]
        for name, values in _categorical_axis_values(metadata).items()
    }


def _grid_has_chainlink_source(metadata: Dict[str, Any]) -> bool:
    grid = metadata.get("grid", {}) if isinstance(metadata, dict) else {}
    if not isinstance(grid, dict):
        return False
    for axis in grid.values():
        if not isinstance(axis, dict) or axis.get("name") != "policy.price_source":
            continue
        values = axis.get("values")
        if isinstance(values, list) and any(str(value) == "chainlink" for value in values):
            return True
    return False


def _coupled_axis_value_rows(metadata: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grid = metadata.get("grid", {}) if isinstance(metadata, dict) else {}
    if not isinstance(grid, dict):
        return {}
    rows_by_axis: Dict[str, List[Dict[str, Any]]] = {}
    for key in sorted(
        (k for k in grid if isinstance(k, str) and k.startswith("x") and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    ):
        axis = grid.get(key)
        if not isinstance(axis, dict) or "names" not in axis or "values" not in axis:
            continue
        names = [str(name) for name in axis.get("names", [])]
        axis_name = "/".join(names)
        rows: List[Dict[str, Any]] = []
        for row in axis.get("values", []):
            if isinstance(row, list) and len(row) == len(names):
                rows.append({name: value for name, value in zip(names, row)})
        rows_by_axis[axis_name] = rows
    return rows_by_axis


def _stringify_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_stringify_value(v) for v in value]
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _stringify_pool(pool: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _stringify_value(val) for key, val in pool.items()}


def _set_dotted(obj: Dict[str, Any], path: str, value: Any) -> None:
    cur = obj
    parts = path.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _nearest_index(values: List[float], coord: float) -> int:
    if not values or not math.isfinite(coord):
        return 0
    arr = np.asarray(values, dtype=float)
    idx = int(np.clip(np.searchsorted(arr, coord), 0, len(arr) - 1))
    if idx > 0 and abs(arr[idx - 1] - coord) < abs(arr[idx] - coord):
        idx -= 1
    return idx


def _edges_from_centers(centers: List[float], log_scale: bool = False) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    if c.size == 0:
        return np.array([0, 1])
    if c.size == 1:
        x0 = float(c[0])
        if log_scale:
            if x0 <= 0:
                raise ValueError("log-scale edges require positive centers")
            return np.array([x0 / math.sqrt(2.0), x0 * math.sqrt(2.0)])
        d = abs(x0) * 0.5 if x0 != 0 else 0.5
        return np.array([x0 - d, x0 + d])
    if log_scale:
        if np.any(c <= 0):
            raise ValueError("log-scale edges require all centers > 0")
        logs = np.log(c)
        mids = (logs[:-1] + logs[1:]) / 2.0
        first = logs[0] - (mids[0] - logs[0])
        last = logs[-1] + (logs[-1] - mids[-1])
        return np.exp(np.concatenate([[first], mids, [last]]))
    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (mids[0] - c[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _select_ticks(values: List[float], max_ticks: int) -> List[int]:
    n = len(values)
    if n == 0:
        return []
    if max_ticks <= 0 or n <= max_ticks:
        return list(range(n))
    idxs = np.linspace(0, n - 1, num=max_ticks, dtype=int)
    uniq = sorted(set(int(i) for i in idxs))
    if uniq[-1] != n - 1:
        uniq[-1] = n - 1
    return uniq


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
    span = 32 - 16
    t = (grid - 16) / span
    size = 8 + t * (10 - 8)
    return int(round(size))


def _metric_scale_flags(metric: str) -> Tuple[bool, bool]:
    mlow = (metric or "").lower()
    scale_1e18 = metric in {
        "virtual_price",
        "xcp_profit",
        "price_scale",
        "D",
        "totalSupply",
    }
    scale_percent = (
        mlow in {"vpminusone", "apy"}
        or "apy" in mlow
        or "tw_real_slippage" in mlow
        or "geom_mean" in mlow
        or "rel_price_diff" in mlow
        or "pool_fee" in mlow
        or "imbalance" in mlow
        or "skew" in mlow
    )
    return scale_1e18, scale_percent


def _metric_scale_info(metric: str) -> Tuple[float, str]:
    scale_1e18, scale_percent = _metric_scale_flags(metric)
    factor = 1.0
    if scale_1e18:
        factor /= 1e18
    if scale_percent:
        factor *= 100.0
    suffix = " (%)" if scale_percent else ""
    return factor, suffix


def _metrics_to_build(metrics: List[str]) -> List[str]:
    metrics_to_build = list(metrics)
    masked_requested = [m for m in metrics_to_build if m in MASKED_METRICS]
    if masked_requested:
        for req in (
            MASK_MAX_PRICE_METRIC,
            MASK_MAX_PRICE_FALLBACK_METRIC,
            MASK_FINAL_PRICE_METRIC,
            MASK_SKEW_METRIC,
            *(MASKED_METRIC_SOURCES[m] for m in masked_requested),
            *(
                SLIPPAGE_APY_MASK_SOURCES[m]
                for m in masked_requested
                if m in SLIPPAGE_APY_MASK_SOURCES
            ),
        ):
            if req not in metrics_to_build:
                metrics_to_build.append(req)
    return metrics_to_build


def _ordered_grid(metadata: Dict[str, Any]) -> Tuple[List[str], List[List[float]], List[str]]:
    """Legacy ordered_grid equivalent over viewer_data metadata.grid."""
    grid = metadata.get("grid", {})
    if not isinstance(grid, dict):
        return [], [], []
    keys = sorted(
        (k for k in grid if isinstance(k, str) and k.startswith("x") and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )
    names: List[str] = []
    values: List[List[float]] = []
    for key in keys:
        axis = grid.get(key)
        if not isinstance(axis, dict) or "values" not in axis:
            continue
        raw_values = axis["values"]
        if "name" in axis:
            names.append(str(axis["name"]))
            try:
                values.append([float(v) for v in raw_values])
            except (TypeError, ValueError):
                values.append([float(i) for i, _ in enumerate(raw_values)])
            continue
        raw_names = axis.get("names")
        if not isinstance(raw_names, list):
            continue
        names.append("/".join(str(name) for name in raw_names))
        values.append([float(i) for i, _ in enumerate(raw_values)])
    return names, values, keys[: len(names)]


def _grid_shape(metadata: Dict[str, Any], n_pools: int) -> Tuple[int, ...]:
    _, values, _ = _ordered_grid(metadata)
    if not values:
        return (n_pools,)
    shape = tuple(len(axis) for axis in values)
    count = 1
    for n in shape:
        count *= n
    if count != n_pools:
        return (n_pools,)
    return shape


def _extract_nd_arrays_npz(
    npz_run: Any,
    metadata: Dict[str, Any],
    metrics: List[str],
    skew_thr_pct: float,
    max_price_thr_bps: float,
    slippage_thr_bps: float,
) -> Tuple[
    List[str],
    Dict[str, List[float]],
    Dict[str, np.ndarray],
    Dict[str, float],
    Dict[Tuple[int, ...], Dict[str, Any]],
    Dict[Tuple[int, ...], Dict[str, Any]],
]:
    dim_names, grid_values, _ = _ordered_grid(metadata)
    if not dim_names:
        raise SystemExit("fxsim run requires metadata.grid")

    shape = _grid_shape(metadata, npz_run.n_pools)
    if len(shape) != len(dim_names):
        raise SystemExit("NPZ grid shape does not match metadata.grid")

    dim_values_sorted = {
        name: values for name, values in zip(dim_names, grid_values)
    }
    metrics_to_build = _metrics_to_build(metrics)
    metric_scale = {
        m: _metric_scale_info(m)[0] for m in metrics_to_build
    }
    metric_arrays: Dict[str, np.ndarray] = {}

    success_mask: np.ndarray | None = None
    try:
        success_mask = npz_run.load_array("success").astype(bool).reshape(shape)
    except KeyError:
        pass

    for metric in metrics_to_build:
        if metric in MASKED_METRICS:
            continue
        try:
            raw = npz_run.load_array(metric).astype(float, copy=False).reshape(shape)
        except KeyError:
            metric_arrays[metric] = np.full(shape, np.nan, dtype=float)
            continue
        arr = raw * metric_scale[metric]
        if success_mask is not None:
            arr = np.where(success_mask, arr, np.nan)
        metric_arrays[metric] = arr

    for metric in metrics_to_build:
        if metric not in MASKED_METRICS:
            continue
        apy = metric_arrays.get(MASKED_METRIC_SOURCES[metric])
        max_rel = metric_arrays.get(MASK_MAX_PRICE_METRIC)
        if max_rel is None or not np.isfinite(max_rel).any():
            max_rel = metric_arrays.get(MASK_MAX_PRICE_FALLBACK_METRIC)
        max_skew = metric_arrays.get(MASK_SKEW_METRIC)
        if apy is None or max_rel is None:
            metric_arrays[metric] = np.full(shape, np.nan, dtype=float)
            continue
        masked = np.array(apy, copy=True)
        masked[max_rel > _bps_to_display_percent(max_price_thr_bps)] = np.nan
        slippage_source = SLIPPAGE_APY_MASK_SOURCES.get(metric)
        if slippage_source is not None:
            slippage = metric_arrays.get(slippage_source)
            if slippage is None:
                masked[:] = np.nan
            else:
                masked[
                    ~np.isfinite(slippage)
                    | (slippage > _bps_to_display_percent(slippage_thr_bps))
                ] = np.nan
        if metric in SKEW_MASKED_METRICS and skew_thr_pct > 0.0:
            if max_skew is None:
                masked[:] = np.nan
            else:
                masked[~np.isfinite(max_skew) | (max_skew > skew_thr_pct)] = np.nan
        metric_arrays[metric] = masked

    pool_configs = dict(getattr(npz_run, "pool_configs", {}))
    metrics_lookup = dict(getattr(npz_run, "metrics_lookup", {}))
    return (
        dim_names,
        dim_values_sorted,
        metric_arrays,
        metric_scale,
        pool_configs,
        metrics_lookup,
    )


def _extract_nd_arrays(
    data: Dict[str, Any],
    metrics: List[str],
    price_thr_bps: float = 0.0,
    max_price_thr_bps: float | None = None,
    skew_thr_pct: float | None = None,
    slippage_thr_bps: float = 20.0,
    **_compat: Any,
) -> Tuple[
    List[str],
    Dict[str, List[float]],
    Dict[str, np.ndarray],
    Dict[str, float],
    Dict[Tuple[int, ...], Dict[str, Any]],
    Dict[Tuple[int, ...], Dict[str, Any]],
]:
    """Extract dense arrays from a collected fxsim run (NPZ-only)."""
    if max_price_thr_bps is None:
        max_price_thr_bps = 100.0
    if skew_thr_pct is None:
        skew_thr_pct = float(_compat.get("imbalance_thr_pct", price_thr_bps))
    npz_run = data.get("_npz_run")
    if npz_run is None or not hasattr(npz_run, "load_array"):
        raise SystemExit("fxsim view requires a collected run directory (manifest.json + evaluation_table.npz)")
    return _extract_nd_arrays_npz(
        npz_run,
        data.get("metadata", {}),
        metrics,
        skew_thr_pct,
        max_price_thr_bps,
        slippage_thr_bps,
    )


def _compute_global_clims(
    metric_arrays: Dict[str, np.ndarray],
    clamp: bool = False,
) -> Dict[str, Tuple[float, float]]:
    clims: Dict[str, Tuple[float, float]] = {}
    for metric, arr in metric_arrays.items():
        if arr.size == 0:
            clims[metric] = (0.0, 1.0)
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            clims[metric] = (0.0, 1.0)
            continue
        if clamp:
            finite = np.where(finite < 0.0, 0.0, finite)
        try:
            zmin = float(np.min(finite))
            zmax = float(np.max(finite))
        except ValueError:
            clims[metric] = (0.0, 1.0)
            continue
        if not math.isfinite(zmin) or not math.isfinite(zmax):
            clims[metric] = (0.0, 1.0)
            continue
        if zmin == zmax:
            eps = 1e-12 if zmax == 0 else abs(zmax) * 1e-12
            zmin, zmax = zmin - eps, zmax + eps
        clims[metric] = (zmin, zmax)
    return clims


class NDHeatmapExplorerOpt:
    """Optimized N-dimensional heatmap explorer with fast slicing."""

    def __init__(
        self,
        data: Dict[str, Any],
        metrics: List[str],
        ncol: int,
        cmap: str,
        max_ticks: int,
        clamp: bool,
        price_thr_bps: float,
        max_price_thr_bps: float,
        final_price_thr_bps: float = 0.0,
        skew_thr_pct: float | None = None,
        slippage_thr_bps: float = 20.0,
        slippage_thr_max_bps: float = 100.0,
        log_axes: set[str] | None = None,
        start_time: str | None = None,
    ):
        self.data = data
        self.metrics = metrics
        self.ncol = ncol
        self.cmap = cmap
        self.max_ticks = max_ticks
        self.clamp = clamp
        self.skew_thr_pct = max(
            0.0,
            float(0.0 if skew_thr_pct is None else skew_thr_pct),
        )
        self.max_price_thr_bps = max(1.0, float(max_price_thr_bps))
        self.final_price_thr_bps = max(0.0, float(final_price_thr_bps))
        self.slippage_thr_max_bps = max(0.0, float(slippage_thr_max_bps))
        self.slippage_thr_bps = min(
            self.slippage_thr_max_bps, max(0.0, float(slippage_thr_bps))
        )
        self.log_axes = set(log_axes or set())
        self.start_time = start_time
        self.has_skew_thr_slider = any(m in SKEW_MASKED_METRICS for m in self.metrics)
        self.has_max_price_thr_slider = any(m in MASKED_METRICS for m in self.metrics)
        self.has_slippage_thr_slider = any(
            m in SLIPPAGE_APY_MASK_SOURCES for m in self.metrics
        )
        self.has_final_price_thr_slider = False

        (
            self.dim_names,
            self.dim_values,
            self.metric_arrays,
            self.metric_scale,
            self.pool_configs,
            self.metrics_lookup,
        ) = _extract_nd_arrays(
            data,
            metrics,
            price_thr_bps,
            max_price_thr_bps,
            skew_thr_pct=self.skew_thr_pct,
            slippage_thr_bps=self.slippage_thr_bps,
        )
        final_price_arr = self.metric_arrays.get(MASK_FINAL_PRICE_METRIC)
        self.has_final_price_thr_slider = (
            any(m in SKEW_MASKED_METRICS for m in self.metrics)
            and final_price_arr is not None
            and np.isfinite(final_price_arr).any()
        )
        self.n_dims = len(self.dim_names)
        self.axis_dim_names = [
            name for name in self.dim_names if len(self.dim_values[name]) > 1
        ]
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        self.axis_value_labels = _coupled_axis_value_labels(metadata)
        self.axis_value_labels.update(_categorical_axis_value_labels(metadata))
        self.categorical_axis_values = _categorical_axis_values(metadata)
        self.coupled_axis_rows = _coupled_axis_value_rows(metadata)
        self.log_axes |= _auto_log_axes(self.dim_values)

        self.skew_thr_max_pct = (
            self._compute_skew_thr_max_pct()
            if self.has_skew_thr_slider
            else max(100.0, self.skew_thr_pct)
        )
        self.skew_thr_min_pct = (
            self._compute_skew_thr_min_pct()
            if self.has_skew_thr_slider
            else 50.0
        )
        if self.has_skew_thr_slider and self.skew_thr_pct <= 0.0:
            self.skew_thr_pct = self.skew_thr_max_pct
        elif 0.0 < self.skew_thr_pct < self.skew_thr_min_pct:
            self.skew_thr_pct = self.skew_thr_min_pct

        self.max_price_thr_max_bps = (
            self._compute_max_price_thr_max_bps()
            if self.has_max_price_thr_slider
            else max(1.0, self.max_price_thr_bps)
        )
        if self.has_max_price_thr_slider:
            self.max_price_thr_bps = min(
                self.max_price_thr_bps, self.max_price_thr_max_bps
            )


        self.final_price_thr_max_bps = (
            self._compute_final_price_thr_max_bps()
            if self.has_final_price_thr_slider
            else max(1.0, self.final_price_thr_bps)
        )
        if self.has_final_price_thr_slider and self.final_price_thr_bps <= 0.0:
            self.final_price_thr_bps = self.final_price_thr_max_bps

        if len(self.axis_dim_names) < 2:
            raise SystemExit("Need at least 2 non-singleton dimensions for a heatmap")

        self.x_name = self.axis_dim_names[0]
        self.y_name = self.axis_dim_names[1]

        unknown_log_axes = sorted(self.log_axes - set(self.dim_names))
        if unknown_log_axes:
            raise SystemExit(
                "Unknown --log-axis value(s): "
                + ", ".join(unknown_log_axes)
                + f". Available axes: {', '.join(self.dim_names)}"
            )
        for name in sorted(self.log_axes):
            if any(v <= 0 for v in self.dim_values[name]):
                raise SystemExit(f"--log-axis {name} requires all axis values to be > 0")

        self.slider_indices: Dict[str, int] = {}
        self._init_slider_indices()

        self.global_clim = _compute_global_clims(self.metric_arrays, clamp=self.clamp)
        for metric, source_key in MASKED_METRIC_SOURCES.items():
            if metric in self.global_clim and source_key in self.global_clim:
                self.global_clim[metric] = self.global_clim[source_key]

        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        self.base_pool = meta.get("base_pool") if isinstance(meta, dict) else None
        if not isinstance(self.base_pool, dict):
            self.base_pool = {}
        self.base_costs = meta.get("base_costs") if isinstance(meta, dict) else None
        if not isinstance(self.base_costs, dict):
            self.base_costs = {}
        self.fee_equalize = bool(meta.get("fee_equalize")) if isinstance(meta, dict) else False
        self.candles_file = None
        self.cowswap_file = None
        self.config_cowswap_fee_bps = 0.0
        self.chainlink_feed = None
        self.config_start_time = None
        self.config_real = "longdouble"
        self.config_dustswapfreq = 3600
        self.config_disable_slippage_probes = False
        if isinstance(meta, dict):
            harness_args = meta.get("harness_args")
            if isinstance(harness_args, dict):
                if harness_args.get("dustswapfreq") is not None:
                    self.config_dustswapfreq = int(harness_args["dustswapfreq"])
                if harness_args.get("disable_slippage_probes") is not None:
                    self.config_disable_slippage_probes = bool(
                        harness_args["disable_slippage_probes"]
                    )
            self.config_start_time = meta.get("start_time")
            if not self.config_start_time and isinstance(harness_args, dict):
                self.config_start_time = harness_args.get("start_time")
            self.candles_file = (
                meta.get("candles_file")
                or meta.get("datafile")
                or meta.get("remote_candles")
            )
            self.chainlink_feed = meta.get("chainlink_feed") or meta.get("chainlink_file")
            self.config_real = str(meta.get("real", "longdouble"))

        self._inspect_running = False
        self.harness_binary: str | None = None

        self.fig_main = None
        self.fig_controls = None
        self.fig_metrics = None
        self.metrics_text = None
        self.axes = []
        self.meshes = []
        self.colorbars = []
        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []
        self.slider_value_texts = []
        self.skew_thr_slider = None
        self.skew_thr_label = None
        self.skew_thr_value_text = None
        self.max_price_thr_slider = None
        self.max_price_thr_label = None
        self.max_price_thr_value_text = None
        self.slippage_thr_slider = None
        self.slippage_thr_label = None
        self.slippage_thr_value_text = None
        self.final_price_thr_slider = None
        self.final_price_thr_label = None
        self.final_price_thr_value_text = None
        self.x_radio = None
        self.y_radio = None
        self._updating_radios = False

        self._setup_figures()

    def _finite_clim(self, values: np.ndarray) -> Tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 1.0)
        if self.clamp:
            finite = np.where(finite < 0.0, 0.0, finite)
        zmin = float(np.min(finite))
        zmax = float(np.max(finite))
        if not math.isfinite(zmin) or not math.isfinite(zmax):
            return (0.0, 1.0)
        if zmin == zmax:
            eps = 1e-12 if zmax == 0 else abs(zmax) * 1e-12
            zmin, zmax = zmin - eps, zmax + eps
        return zmin, zmax

    def _clim_for_masked_metric(self, metric: str) -> Tuple[float, float]:
        source = self.metric_arrays.get(MASKED_METRIC_SOURCES[metric])
        max_rel = self.metric_arrays.get(MASK_MAX_PRICE_METRIC)
        if max_rel is None or not np.isfinite(max_rel).any():
            max_rel = self.metric_arrays.get(MASK_MAX_PRICE_FALLBACK_METRIC)
        if source is None or max_rel is None:
            return (0.0, 1.0)
        mask = max_rel <= _bps_to_display_percent(self.max_price_thr_bps)
        slippage_source = SLIPPAGE_APY_MASK_SOURCES.get(metric)
        if slippage_source is not None:
            slippage = self.metric_arrays.get(slippage_source)
            if slippage is None:
                return (0.0, 1.0)
            mask &= np.isfinite(slippage) & (
                slippage <= _bps_to_display_percent(self.slippage_thr_bps)
            )
        if metric in SKEW_MASKED_METRICS and self.skew_thr_pct > 0.0:
            max_skew = self.metric_arrays.get(MASK_SKEW_METRIC)
            if max_skew is None:
                return (0.0, 1.0)
            mask &= np.isfinite(max_skew) & (max_skew <= self.skew_thr_pct)
        if metric in SKEW_MASKED_METRICS and self.has_final_price_thr_slider:
            final_rel = self.metric_arrays.get(MASK_FINAL_PRICE_METRIC)
            if final_rel is None:
                return (0.0, 1.0)
            mask &= (
                np.isfinite(final_rel)
                & (final_rel <= _bps_to_display_percent(self.final_price_thr_bps))
            )
        return self._finite_clim(source[mask])

    def _clim_for_metric_slice(self, metric: str, Z: np.ndarray) -> Tuple[float, float] | None:
        if metric not in MASKED_METRICS:
            return self.global_clim.get(metric)
        return self._finite_clim(Z)

    def _format_axis_value(self, name: str, value: float) -> str:
        labels = self.axis_value_labels.get(name)
        if labels is not None:
            idx = int(round(value))
            if 0 <= idx < len(labels) and abs(value - idx) < 1e-9:
                return labels[idx]
        return _format_slider_value(name, value)

    def _raw_axis_value(self, name: str, value: float) -> Any:
        raw_values = self.categorical_axis_values.get(name)
        if raw_values is None:
            return value
        idx = int(round(value))
        if 0 <= idx < len(raw_values) and abs(value - idx) < 1e-9:
            return raw_values[idx]
        return value

    def _format_axis_labels(self, name: str, values: List[float]) -> Tuple[List[str], str]:
        labels = self.axis_value_labels.get(name)
        if labels is not None:
            return [
                labels[int(round(v))]
                if 0 <= int(round(v)) < len(labels) and abs(v - round(v)) < 1e-9
                else _format_slider_value(name, v)
                for v in values
            ], name
        return _format_axis_labels(name, values)
        self._setup_metrics_window()

    def _init_slider_indices(self):
        self.slider_indices = {}
        for name in self.dim_names:
            if name not in (self.x_name, self.y_name) and len(self.dim_values[name]) > 1:
                self.slider_indices[name] = self._default_slider_index(name)

    def _default_slider_index(self, name: str) -> int:
        key = (name or "").lower()
        if key in {"reserved_profit_fraction", "reserved_profit_ratio"}:
            return max(0, len(self.dim_values.get(name, [])) - 1)
        return 0

    def _get_slider_dims(self) -> List[Tuple[int, str]]:
        return [
            (i, name)
            for i, name in enumerate(self.dim_names)
            if name not in (self.x_name, self.y_name) and len(self.dim_values[name]) > 1
        ]

    def _slice_raw(self, metric: str) -> np.ndarray | None:
        arr = self.metric_arrays.get(metric)
        if arr is None:
            return None
        x_idx = self.dim_names.index(self.x_name)
        y_idx = self.dim_names.index(self.y_name)

        slicer: List[Any] = []
        for name in self.dim_names:
            if name == self.x_name or name == self.y_name:
                slicer.append(slice(None))
            else:
                slicer.append(self.slider_indices.get(name, 0))

        slice_arr = arr[tuple(slicer)]

        if x_idx < y_idx:
            slice_arr = slice_arr.T

        return slice_arr

    def _slice_metric(self, metric: str) -> np.ndarray:
        if metric in MASKED_METRICS:
            apy_slice = self._slice_raw(MASKED_METRIC_SOURCES[metric])
            max_rel_slice = self._slice_raw(MASK_MAX_PRICE_METRIC)
            if max_rel_slice is None or not np.isfinite(max_rel_slice).any():
                max_rel_slice = self._slice_raw(MASK_MAX_PRICE_FALLBACK_METRIC)
            max_skew_slice = self._slice_raw(MASK_SKEW_METRIC)
            if apy_slice is None or max_rel_slice is None:
                xs = self.dim_values[self.x_name]
                ys = self.dim_values[self.y_name]
                return np.full((len(ys), len(xs)), np.nan, dtype=float)
            thr_max_rel_pct = _bps_to_display_percent(self.max_price_thr_bps)
            masked = np.array(apy_slice, copy=True)
            masked[max_rel_slice > thr_max_rel_pct] = np.nan
            slippage_source = SLIPPAGE_APY_MASK_SOURCES.get(metric)
            if slippage_source is not None:
                slippage_slice = self._slice_raw(slippage_source)
                if slippage_slice is None:
                    masked[:] = np.nan
                else:
                    masked[
                        ~np.isfinite(slippage_slice)
                        | (
                            slippage_slice
                            > _bps_to_display_percent(self.slippage_thr_bps)
                        )
                    ] = np.nan
            if metric in SKEW_MASKED_METRICS and self.skew_thr_pct > 0.0:
                if max_skew_slice is None:
                    masked[:] = np.nan
                else:
                    masked[
                        ~np.isfinite(max_skew_slice)
                        | (max_skew_slice > self.skew_thr_pct)
                    ] = np.nan
            if metric in SKEW_MASKED_METRICS and self.has_final_price_thr_slider:
                final_rel_slice = self._slice_raw(MASK_FINAL_PRICE_METRIC)
                if final_rel_slice is None:
                    masked[:] = np.nan
                else:
                    masked[
                        ~np.isfinite(final_rel_slice)
                        | (final_rel_slice > _bps_to_display_percent(self.final_price_thr_bps))
                    ] = np.nan
            return masked

        slice_arr = self._slice_raw(metric)
        if slice_arr is None:
            xs = self.dim_values[self.x_name]
            ys = self.dim_values[self.y_name]
            return np.full((len(ys), len(xs)), np.nan, dtype=float)
        return slice_arr

    def _attach_format_coord(self, ax, xs, ys, mesh):
        """Attach a format_coord function to ax that shows x, y, z on hover."""
        xs_arr = np.array(xs)
        ys_arr = np.array(ys)

        def format_coord(x, y):
            if len(xs_arr) == 0 or len(ys_arr) == 0:
                return ""
            j = int(np.clip(np.searchsorted(xs_arr, x) - 0.5, 0, len(xs_arr) - 1))
            i = int(np.clip(np.searchsorted(ys_arr, y) - 0.5, 0, len(ys_arr) - 1))
            if j < len(xs_arr) - 1 and abs(x - xs_arr[j + 1]) < abs(x - xs_arr[j]):
                j += 1
            if i < len(ys_arr) - 1 and abs(y - ys_arr[i + 1]) < abs(y - ys_arr[i]):
                i += 1
            Z_arr = mesh.get_array()
            if Z_arr is not None:
                Z_2d = Z_arr.reshape(len(ys_arr), len(xs_arr))
                z_val = (
                    Z_2d[i, j]
                    if 0 <= i < Z_2d.shape[0] and 0 <= j < Z_2d.shape[1]
                    else float("nan")
                )
            else:
                z_val = float("nan")
            x_val = self._format_axis_value(self.x_name, float(xs_arr[j]))
            y_val = self._format_axis_value(self.y_name, float(ys_arr[i]))
            return f"x={x_val}, y={y_val}, z={z_val:.4g}"

        ax.format_coord = format_coord

    def _setup_figures(self):
        n = len(self.metrics)
        cols = min(self.ncol, n)
        rows = int(np.ceil(n / cols)) if n > 0 else 1

        max_fig_w = 22.0
        max_fig_h = 12.0

        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]

        cell_aspect = len(ys) / max(1, len(xs))
        cell_w = max_fig_w / cols
        cell_h = cell_w * cell_aspect
        fig_h = cell_h * rows

        if fig_h > max_fig_h:
            fig_h = max_fig_h
            cell_h = fig_h / rows
            cell_w = cell_h / cell_aspect
            fig_w = cell_w * cols
        else:
            fig_w = max_fig_w

        fig_w = max(10.0, min(max_fig_w, fig_w))
        fig_h = max(6.0, min(max_fig_h, fig_h))

        self.fig_main, axes_grid = plt.subplots(
            rows, cols, figsize=(fig_w, fig_h), constrained_layout=True, num="Heatmaps"
        )
        axes_grid = np.atleast_1d(axes_grid).reshape(rows, cols)

        base_font = _auto_font_size(len(xs), len(ys))
        tick_font = base_font
        label_font = max(8, base_font + 2)
        title_font = max(label_font, base_font + 4)
        colorbar_font = base_font

        xticks = _select_ticks(xs, self.max_ticks)
        yticks = _select_ticks(ys, self.max_ticks)
        xlab_full, xlabel = self._format_axis_labels(self.x_name, xs)
        ylab_full, ylabel = self._format_axis_labels(self.y_name, ys)
        xlabels = [xlab_full[i] for i in xticks]
        ylabels = [ylab_full[i] for i in yticks]

        log_x = self.x_name in self.log_axes
        log_y = self.y_name in self.log_axes
        Xedges = _edges_from_centers(xs, log_x)
        Yedges = _edges_from_centers(ys, log_y)

        self.axes = []
        self.meshes = []
        self.colorbars = []

        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes_grid[r, c]
                if idx >= n:
                    ax.axis("off")
                    continue

                metric = self.metrics[idx]
                Z = self._slice_metric(metric)

                mesh = ax.pcolormesh(Xedges, Yedges, Z, cmap=self.cmap, shading="auto")
                if log_x:
                    ax.set_xscale("log")
                if log_y:
                    ax.set_yscale("log")

                ny, nx = Z.shape
                try:
                    ax.set_box_aspect(ny / nx)
                except Exception:
                    ax.set_aspect("equal", adjustable="box")

                _apply_fixed_ticks(ax.xaxis, [xs[i] for i in xticks], xlabels)
                ax.tick_params(axis="x", labelrotation=45, labelsize=tick_font)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
                if c == 0:
                    _apply_fixed_ticks(ax.yaxis, [ys[i] for i in yticks], ylabels)
                    ax.tick_params(axis="y", labelsize=tick_font)
                    ax.set_ylabel(ylabel, fontsize=label_font)
                else:
                    _apply_fixed_ticks(ax.yaxis, [ys[i] for i in yticks], [])
                    ax.tick_params(axis="y", labelsize=tick_font)
                ax.set_xlabel(xlabel, fontsize=label_font)

                _, title_suffix = _metric_scale_info(metric)
                ax.set_title(f"{metric}{title_suffix}", fontsize=title_font)

                clim = self._clim_for_metric_slice(metric, Z)
                if clim is not None:
                    zmin, zmax = clim
                    mesh.set_clim(zmin, zmax)

                cb = self.fig_main.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label(metric + title_suffix, fontsize=colorbar_font)
                cb.ax.tick_params(labelsize=tick_font)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

                self._attach_format_coord(ax, xs, ys, mesh)

                self.axes.append(ax)
                self.meshes.append(mesh)
                self.colorbars.append(cb)
                idx += 1

        self._setup_controls()
        self.fig_main.canvas.mpl_connect("button_press_event", self._on_click)

    def _setup_controls(self):
        slider_dims = self._get_slider_dims()
        n_sliders = len(slider_dims)
        n_dims = len(self.axis_dim_names)
        extra_sliders = 0
        if self.has_skew_thr_slider:
            extra_sliders += 1
        if self.has_max_price_thr_slider:
            extra_sliders += 1
        if self.has_slippage_thr_slider:
            extra_sliders += 1
        if self.has_final_price_thr_slider:
            extra_sliders += 1
        n_sliders_total = n_sliders + extra_sliders

        radio_box_height = n_dims * RADIO_ITEM_HEIGHT + RADIO_BOX_PADDING

        total_content = (
            0.03
            + 2 * (RADIO_LABEL_GAP + radio_box_height + RADIO_GROUP_GAP)
            + n_sliders_total * SLIDER_HEIGHT
            + SLIDER_TOP_GAP
        )
        height_mult = CTRL_HEIGHT_MULT + max(0, n_dims - 5) * 0.3
        fig_height = max(CTRL_MIN_HEIGHT, total_content * height_mult + CTRL_HEIGHT_PAD)

        self.fig_controls = plt.figure(
            figsize=(CTRL_FIG_WIDTH, fig_height), num="Controls"
        )

        self.fig_controls.text(
            0.5,
            CTRL_TITLE_Y,
            "Dimension Controls",
            ha="center",
            va="top",
            fontsize=CTRL_TITLE_FONTSIZE,
            fontweight="bold",
        )

        x_label_y = RADIO_X_LABEL_Y
        x_box_top = x_label_y - RADIO_LABEL_GAP
        x_box_height = radio_box_height

        self.fig_controls.text(
            0.05, x_label_y, "X axis:", ha="left", va="top", fontsize=RADIO_FONTSIZE + 1
        )
        x_ax = self.fig_controls.add_axes(
            [SLIDER_BOX_LEFT, x_box_top - x_box_height, 0.75, x_box_height]
        )
        x_ax.set_frame_on(False)
        self.x_radio = RadioButtons(
            x_ax, self.axis_dim_names, active=self.axis_dim_names.index(self.x_name)
        )
        for label in self.x_radio.labels:
            label.set_fontsize(RADIO_FONTSIZE)
        self.x_radio.on_clicked(self._on_x_changed)

        y_label_y = x_box_top - x_box_height - RADIO_GROUP_GAP
        y_box_top = y_label_y - RADIO_LABEL_GAP
        y_box_height = radio_box_height

        self.fig_controls.text(
            0.05, y_label_y, "Y axis:", ha="left", va="top", fontsize=RADIO_FONTSIZE + 1
        )
        y_ax = self.fig_controls.add_axes(
            [SLIDER_BOX_LEFT, y_box_top - y_box_height, 0.75, y_box_height]
        )
        y_ax.set_frame_on(False)
        self.y_radio = RadioButtons(
            y_ax, self.axis_dim_names, active=self.axis_dim_names.index(self.y_name)
        )
        for label in self.y_radio.labels:
            label.set_fontsize(RADIO_FONTSIZE)
        self.y_radio.on_clicked(self._on_y_changed)

        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []
        self.slider_value_texts = []

        slider_start_y = y_box_top - y_box_height - SLIDER_TOP_GAP

        for i, (_, dim_name) in enumerate(slider_dims):
            vals = self.dim_values[dim_name]
            slider_y = slider_start_y - i * SLIDER_HEIGHT

            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                f"{dim_name}:",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                0,
                len(vals) - 1,
                valinit=self.slider_indices.get(dim_name, 0),
                valstep=1,
            )
            slider.valtext.set_visible(False)

            current_idx = int(slider.val)
            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y + SLIDER_VALUE_Y_OFFSET,
                self._format_axis_value(dim_name, vals[current_idx]),
                ha="left",
                va="top",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_value_texts.append(val_text)

            def make_update(name, vals_list, val_txt):
                def update(idx):
                    idx = int(idx)
                    self.slider_indices[name] = idx
                    val_txt.set_text(self._format_axis_value(name, vals_list[idx]))
                    self._refresh_heatmaps()

                return update

            slider.on_changed(make_update(dim_name, vals, val_text))
            self.sliders.append((dim_name, slider))

        self._add_filter_sliders(slider_start_y, n_sliders)

        self.fig_controls.canvas.draw_idle()

    def _add_filter_sliders(self, slider_start_y: float, slider_offset: int) -> None:
        if self.has_skew_thr_slider:
            slider_y = slider_start_y - slider_offset * SLIDER_HEIGHT
            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                "7d skew max (%):",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.skew_thr_label = lbl
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                1,
                self._skew_thr_slider_max(),
                valinit=self._skew_thr_to_slider_value(self.skew_thr_pct),
                valstep=1,
            )
            slider.valtext.set_visible(False)
            self.skew_thr_slider = slider

            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y + SLIDER_VALUE_Y_OFFSET,
                self._format_skew_thr_value(),
                ha="left",
                va="top",
                fontsize=SLIDER_FONTSIZE,
            )
            self.skew_thr_value_text = val_text
            self.slider_value_texts.append(val_text)

            def update_skew_thr(val):
                self.skew_thr_pct = self._skew_slider_value_to_thr(float(val))
                if self.skew_thr_value_text is not None:
                    self.skew_thr_value_text.set_text(self._format_skew_thr_value())
                self._refresh_heatmaps()

            slider.on_changed(update_skew_thr)
            slider_offset += 1

        if self.has_max_price_thr_slider:
            slider_y = slider_start_y - slider_offset * SLIDER_HEIGHT
            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                "max 7d pdiff thr (bps):",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.max_price_thr_label = lbl
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                1,
                max(1.0, self.max_price_thr_max_bps),
                valinit=self.max_price_thr_bps,
                valstep=1,
            )
            slider.valtext.set_visible(False)
            self.max_price_thr_slider = slider

            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y + SLIDER_VALUE_Y_OFFSET,
                f"{int(self.max_price_thr_bps)}",
                ha="left",
                va="top",
                fontsize=SLIDER_FONTSIZE,
            )
            self.max_price_thr_value_text = val_text
            self.slider_value_texts.append(val_text)

            def update_max_price_thr(val):
                self.max_price_thr_bps = float(val)
                if self.max_price_thr_value_text is not None:
                    self.max_price_thr_value_text.set_text(
                        f"{int(self.max_price_thr_bps)}"
                    )
                self._refresh_heatmaps()

            slider.on_changed(update_max_price_thr)
            slider_offset += 1
        if self.has_slippage_thr_slider:
            slider_y = slider_start_y - slider_offset * SLIDER_HEIGHT
            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                "slippage max (bps):",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slippage_thr_label = lbl
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                0,
                self.slippage_thr_max_bps,
                valinit=self.slippage_thr_bps,
                valstep=1,
            )
            slider.valtext.set_visible(False)
            self.slippage_thr_slider = slider

            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y + SLIDER_VALUE_Y_OFFSET,
                f"{int(self.slippage_thr_bps)}",
                ha="left",
                va="top",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slippage_thr_value_text = val_text
            self.slider_value_texts.append(val_text)

            def update_slippage_thr(val):
                self.slippage_thr_bps = float(val)
                if self.slippage_thr_value_text is not None:
                    self.slippage_thr_value_text.set_text(
                        f"{int(self.slippage_thr_bps)}"
                    )
                self._refresh_heatmaps()

            slider.on_changed(update_slippage_thr)
            slider_offset += 1

        if self.has_final_price_thr_slider:
            slider_y = slider_start_y - slider_offset * SLIDER_HEIGHT
            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                "last pdiff thr (bps):",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.final_price_thr_label = lbl
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                1,
                max(1.0, self.final_price_thr_max_bps),
                valinit=self.final_price_thr_bps,
                valstep=1,
            )
            slider.valtext.set_visible(False)
            self.final_price_thr_slider = slider

            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y + SLIDER_VALUE_Y_OFFSET,
                f"{int(self.final_price_thr_bps)}",
                ha="left",
                va="top",
                fontsize=SLIDER_FONTSIZE,
            )
            self.final_price_thr_value_text = val_text
            self.slider_value_texts.append(val_text)

            def update_final_price_thr(val):
                self.final_price_thr_bps = float(val)
                if self.final_price_thr_value_text is not None:
                    self.final_price_thr_value_text.set_text(
                        f"{int(self.final_price_thr_bps)}"
                    )
                self._refresh_heatmaps()

            slider.on_changed(update_final_price_thr)

    def _setup_metrics_window(self):
        if self.fig_metrics is not None:
            return
        self.fig_metrics = plt.figure(figsize=(6.5, 9.0), num="Metrics")
        self.fig_metrics.text(
            0.02,
            0.98,
            "Metrics (left click a heatmap cell)",
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        self.metrics_text = self.fig_metrics.text(
            0.02,
            0.94,
            "",
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
        )
        self.fig_metrics.canvas.draw_idle()

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.6g}"
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "[" + ", ".join(self._format_metric_value(v) for v in value) + "]"
        if isinstance(value, dict):
            try:
                return json.dumps(value, sort_keys=True)
            except Exception:
                return str(value)
        return str(value)

    def _metric_display_value(self, metric: str, metrics: Dict[str, Any] | None) -> str:
        if metrics is None:
            return "missing"

        if metric in MASKED_METRICS:
            source_key = MASKED_METRIC_SOURCES[metric]
            apy_net = _to_float(metrics.get(source_key))
            max_rel = _max_mask_price_diff(metrics)
            if (
                not math.isfinite(apy_net)
                or not math.isfinite(max_rel)
            ):
                return "nan"
            thr_max_rel = _bps_to_raw_ratio(self.max_price_thr_bps)
            if max_rel > thr_max_rel:
                return "nan"
            slippage_source = SLIPPAGE_APY_MASK_SOURCES.get(metric)
            if slippage_source is not None:
                slippage = _to_float(metrics.get(slippage_source))
                if (
                    not math.isfinite(slippage)
                    or slippage > _bps_to_raw_ratio(self.slippage_thr_bps)
                ):
                    return "nan"
            if metric in SKEW_MASKED_METRICS and self.skew_thr_pct > 0.0:
                max_skew = _to_float(metrics.get(MASK_SKEW_METRIC))
                if not math.isfinite(max_skew) or max_skew > self.skew_thr_pct / 100.0:
                    return "nan"
            if metric in SKEW_MASKED_METRICS and self.has_final_price_thr_slider:
                final_rel = _to_float(metrics.get(MASK_FINAL_PRICE_METRIC))
                if (
                    not math.isfinite(final_rel)
                    or final_rel > _bps_to_raw_ratio(self.final_price_thr_bps)
                ):
                    return "nan"
            scale, _ = _metric_scale_info(metric)
            return self._format_metric_value(apy_net * scale)

        raw = metrics.get(metric)
        if raw is None:
            return "missing"

        if isinstance(raw, (int, float, np.floating)):
            scale, _ = _metric_scale_info(metric)
            return self._format_metric_value(float(raw) * scale)
        return self._format_metric_value(raw)

    def _metric_array_display_value(self, metric: str, idx_tuple: Tuple[int, ...]) -> str:
        if metric in MASKED_METRICS:
            apy = self.metric_arrays.get(MASKED_METRIC_SOURCES[metric])
            max_rel = self.metric_arrays.get(MASK_MAX_PRICE_METRIC)
            if max_rel is None or not np.isfinite(max_rel).any():
                max_rel = self.metric_arrays.get(MASK_MAX_PRICE_FALLBACK_METRIC)
            max_skew = self.metric_arrays.get(MASK_SKEW_METRIC)
            final_rel = self.metric_arrays.get(MASK_FINAL_PRICE_METRIC)
            if apy is None or max_rel is None:
                return "missing"
            value = float(apy[idx_tuple])
            mx = float(max_rel[idx_tuple])
            if (
                not math.isfinite(value)
                or not math.isfinite(mx)
                or mx > _bps_to_display_percent(self.max_price_thr_bps)
            ):
                return "nan"
            slippage_source = SLIPPAGE_APY_MASK_SOURCES.get(metric)
            if slippage_source is not None:
                slippage = self.metric_arrays.get(slippage_source)
                if slippage is None:
                    return "nan"
                slippage_value = float(slippage[idx_tuple])
                if (
                    not math.isfinite(slippage_value)
                    or slippage_value
                    > _bps_to_display_percent(self.slippage_thr_bps)
                ):
                    return "nan"
            if metric in SKEW_MASKED_METRICS and self.skew_thr_pct > 0.0:
                if max_skew is None:
                    return "nan"
                skew = float(max_skew[idx_tuple])
                if not math.isfinite(skew) or skew > self.skew_thr_pct:
                    return "nan"
            if metric in SKEW_MASKED_METRICS and self.has_final_price_thr_slider:
                if final_rel is None:
                    return "nan"
                final_value = float(final_rel[idx_tuple])
                if (
                    not math.isfinite(final_value)
                    or final_value > _bps_to_display_percent(self.final_price_thr_bps)
                ):
                    return "nan"
            return self._format_metric_value(value)

        arr = self.metric_arrays.get(metric)
        if arr is None:
            return "missing"
        value = float(arr[idx_tuple])
        if not math.isfinite(value):
            return "nan"
        return self._format_metric_value(value)

    def _update_metrics_window(
        self, idx_tuple: Tuple[int, ...], coords: Dict[str, float]
    ):
        if self.metrics_text is None or self.fig_metrics is None:
            return
        metrics = self.metrics_lookup.get(idx_tuple)
        lines = ["coords:"]
        for name in self.dim_names:
            lines.append(f"  {name}: {self._format_axis_value(name, coords[name])}")
        lines.append("")
        lines.append("metrics:")
        for metric in self.metrics:
            _, suffix = _metric_scale_info(metric)
            label = f"{metric}{suffix}" if suffix else metric
            val = (
                self._metric_display_value(metric, metrics)
                if metrics is not None
                else self._metric_array_display_value(metric, idx_tuple)
            )
            lines.append(f"  {label}: {val}")
        self.metrics_text.set_text("\n".join(lines))
        self.fig_metrics.canvas.draw_idle()

    def _format_skew_thr_value(self) -> str:
        return f"{int(self.skew_thr_pct)}"

    def _skew_thr_slider_max(self) -> float:
        return max(1.0, math.ceil(self.skew_thr_max_pct) - math.floor(self.skew_thr_min_pct) + 1.0)

    def _skew_thr_to_slider_value(self, pct: float) -> float:
        return max(1.0, round(pct - math.floor(self.skew_thr_min_pct) + 1.0))

    def _skew_slider_value_to_thr(self, val: float) -> float:
        return math.floor(self.skew_thr_min_pct) + float(val) - 1.0

    def _compute_skew_thr_min_pct(self) -> float:
        arr = self.metric_arrays.get(MASK_SKEW_METRIC)
        if arr is None:
            return 50.0
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 50.0
        min_percent = float(np.nanmin(finite))
        if not math.isfinite(min_percent) or min_percent <= 0.0:
            return 50.0
        return max(0.0, math.floor(min_percent))

    def _compute_skew_thr_max_pct(self) -> float:
        arr = self.metric_arrays.get(MASK_SKEW_METRIC)
        if arr is None:
            return max(100.0, self.skew_thr_pct)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return max(100.0, self.skew_thr_pct)
        max_percent = float(np.nanmax(finite))
        if not math.isfinite(max_percent) or max_percent <= 0.0:
            return max(100.0, self.skew_thr_pct)
        return math.ceil(max_percent)

    def _compute_max_price_thr_max_bps(self) -> float:
        arr = self.metric_arrays.get(MASK_MAX_PRICE_METRIC)
        if arr is None or not np.isfinite(arr).any():
            arr = self.metric_arrays.get(MASK_MAX_PRICE_FALLBACK_METRIC)
        if arr is None:
            return max(1.0, self.max_price_thr_bps)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return max(1.0, self.max_price_thr_bps)
        max_percent = float(np.nanmax(finite))
        max_bps = max_percent * 100.0
        if not math.isfinite(max_bps) or max_bps <= 0.0:
            return max(1.0, self.max_price_thr_bps)
        return max_bps


    def _compute_final_price_thr_max_bps(self) -> float:
        arr = self.metric_arrays.get(MASK_FINAL_PRICE_METRIC)
        if arr is None:
            return max(1.0, self.final_price_thr_bps)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return max(1.0, self.final_price_thr_bps)
        max_percent = float(np.nanmax(finite))
        max_bps = max_percent * 100.0
        if not math.isfinite(max_bps) or max_bps <= 0.0:
            return max(1.0, self.final_price_thr_bps)
        return max_bps

    def _on_x_changed(self, label: str):
        if self._updating_radios:
            return
        if label == self.x_name:
            return

        if label == self.y_name:
            self._updating_radios = True
            old_x = self.x_name
            self.x_name = label
            self.y_name = old_x
            y_idx = self.axis_dim_names.index(self.y_name)
            self.y_radio.set_active(y_idx)
            self._updating_radios = False
        else:
            self.x_name = label

        self._rebuild_sliders()
        self._rebuild_heatmaps()

    def _on_y_changed(self, label: str):
        if self._updating_radios:
            return
        if label == self.y_name:
            return

        if label == self.x_name:
            self._updating_radios = True
            old_y = self.y_name
            self.y_name = label
            self.x_name = old_y
            x_idx = self.axis_dim_names.index(self.x_name)
            self.x_radio.set_active(x_idx)
            self._updating_radios = False
        else:
            self.y_name = label

        self._rebuild_sliders()
        self._rebuild_heatmaps()

    def _rebuild_sliders(self):
        for slider_ax in self.slider_axes:
            slider_ax.remove()
        for lbl in self.slider_labels:
            lbl.remove()
        for val_txt in self.slider_value_texts:
            val_txt.remove()
        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []
        self.slider_value_texts = []
        self.skew_thr_slider = None
        self.skew_thr_label = None
        self.skew_thr_value_text = None
        self.max_price_thr_slider = None
        self.max_price_thr_label = None
        self.max_price_thr_value_text = None
        self.slippage_thr_slider = None
        self.slippage_thr_label = None
        self.slippage_thr_value_text = None
        self.final_price_thr_slider = None
        self.final_price_thr_label = None
        self.final_price_thr_value_text = None

        new_slider_indices = {}
        for name in self.dim_names:
            if name not in (self.x_name, self.y_name) and len(self.dim_values[name]) > 1:
                default_idx = self._default_slider_index(name)
                new_slider_indices[name] = min(
                    self.slider_indices.get(name, default_idx),
                    len(self.dim_values[name]) - 1,
                )
        self.slider_indices = new_slider_indices

        slider_dims = self._get_slider_dims()
        n_dims = len(self.axis_dim_names)
        n_sliders = len(slider_dims)
        radio_box_height = n_dims * RADIO_ITEM_HEIGHT + RADIO_BOX_PADDING

        x_label_y = RADIO_X_LABEL_Y
        x_box_top = x_label_y - RADIO_LABEL_GAP
        y_label_y = x_box_top - radio_box_height - RADIO_GROUP_GAP
        y_box_top = y_label_y - RADIO_LABEL_GAP
        slider_start_y = y_box_top - radio_box_height - SLIDER_TOP_GAP

        for i, (_, dim_name) in enumerate(slider_dims):
            vals = self.dim_values[dim_name]
            slider_y = slider_start_y - i * SLIDER_HEIGHT
            current_idx = self.slider_indices.get(dim_name, 0)

            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                f"{dim_name}:",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                0,
                len(vals) - 1,
                valinit=current_idx,
                valstep=1,
            )
            slider.valtext.set_visible(False)

            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y + SLIDER_VALUE_Y_OFFSET,
                self._format_axis_value(dim_name, vals[current_idx]),
                ha="left",
                va="top",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_value_texts.append(val_text)

            def make_update(name, vals_list, val_txt):
                def update(idx):
                    idx = int(idx)
                    self.slider_indices[name] = idx
                    val_txt.set_text(self._format_axis_value(name, vals_list[idx]))
                    self._refresh_heatmaps()

                return update

            slider.on_changed(make_update(dim_name, vals, val_text))
            self.sliders.append((dim_name, slider))

        self._add_filter_sliders(slider_start_y, n_sliders)

        self.fig_controls.canvas.draw_idle()

    def _rebuild_heatmaps(self):
        for cb in self.colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self.colorbars = []
        self.meshes = []

        for ax in self.axes:
            ax.clear()

        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]

        base_font = _auto_font_size(len(xs), len(ys))
        tick_font = base_font
        label_font = max(8, base_font + 2)
        title_font = max(label_font, base_font + 4)
        colorbar_font = base_font

        xticks = _select_ticks(xs, self.max_ticks)
        yticks = _select_ticks(ys, self.max_ticks)
        xlab_full, xlabel = self._format_axis_labels(self.x_name, xs)
        ylab_full, ylabel = self._format_axis_labels(self.y_name, ys)
        xlabels = [xlab_full[i] for i in xticks]
        ylabels = [ylab_full[i] for i in yticks]

        log_x = self.x_name in self.log_axes
        log_y = self.y_name in self.log_axes
        Xedges = _edges_from_centers(xs, log_x)
        Yedges = _edges_from_centers(ys, log_y)

        n = len(self.metrics)
        cols = min(self.ncol, n)

        for idx, ax in enumerate(self.axes):
            if idx >= n:
                ax.axis("off")
                continue

            metric = self.metrics[idx]
            Z = self._slice_metric(metric)

            mesh = ax.pcolormesh(Xedges, Yedges, Z, cmap=self.cmap, shading="auto")
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

            ny, nx = Z.shape
            try:
                ax.set_box_aspect(ny / nx)
            except Exception:
                ax.set_aspect("equal", adjustable="box")

            _apply_fixed_ticks(ax.xaxis, [xs[i] for i in xticks], xlabels)
            ax.tick_params(axis="x", labelrotation=45, labelsize=tick_font)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            c = idx % cols
            if c == 0:
                _apply_fixed_ticks(ax.yaxis, [ys[i] for i in yticks], ylabels)
                ax.tick_params(axis="y", labelsize=tick_font)
                ax.set_ylabel(ylabel, fontsize=label_font)
            else:
                _apply_fixed_ticks(ax.yaxis, [ys[i] for i in yticks], [])
                ax.tick_params(axis="y", labelsize=tick_font)
            ax.set_xlabel(xlabel, fontsize=label_font)

            _, title_suffix = _metric_scale_info(metric)
            ax.set_title(f"{metric}{title_suffix}", fontsize=title_font)

            clim = self._clim_for_metric_slice(metric, Z)
            if clim is not None:
                zmin, zmax = clim
                mesh.set_clim(zmin, zmax)

            cb = self.fig_main.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(metric + title_suffix, fontsize=colorbar_font)
            cb.ax.tick_params(labelsize=tick_font)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

            self._attach_format_coord(ax, xs, ys, mesh)

            self.meshes.append(mesh)
            self.colorbars.append(cb)

        self.fig_main.canvas.draw_idle()

    def _refresh_heatmaps(self):
        for idx, mesh in enumerate(self.meshes):
            if idx >= len(self.metrics):
                continue
            metric = self.metrics[idx]
            Z = self._slice_metric(metric)
            mesh.set_array(Z.ravel())
            clim = self._clim_for_metric_slice(metric, Z)
            if clim is not None:
                mesh.set_clim(*clim)
            if idx < len(self.colorbars):
                self.colorbars[idx].update_normal(mesh)
        self.fig_main.canvas.draw_idle()

    def _build_inspect_pool_config(
        self, coords: Dict[str, float], idx_tuple: Tuple[int, ...]
    ) -> Dict[str, Any] | None:
        config = self.pool_configs.get(idx_tuple)
        if config and "pool" in config:
            pool = _stringify_pool(config.get("pool", {}))
            costs = config.get("costs") or dict(DEFAULT_COSTS)
            return {"tag": "inspect", "pool": pool, "costs": costs}

        if not self.base_pool:
            return None

        pool = dict(self.base_pool)
        costs = dict(self.base_costs) if self.base_costs else dict(DEFAULT_COSTS)
        touches_fee = False
        for name, val in coords.items():
            coupled_rows = self.coupled_axis_rows.get(name)
            if coupled_rows is not None:
                idx = int(round(val))
                if 0 <= idx < len(coupled_rows):
                    for item_name, item_val in coupled_rows[idx].items():
                        if item_name == "policy.fee_bps":
                            pool["policy"] = {"kind": "fixed_fee", "fee_bps": item_val}
                        elif item_name.startswith("costs."):
                            _set_dotted(costs, item_name.removeprefix("costs."), item_val)
                        elif "." in item_name:
                            _set_dotted(pool, item_name, item_val)
                        else:
                            pool[item_name] = item_val
                        touches_fee = touches_fee or item_name in {"mid_fee", "out_fee"}
                continue
            if name == "policy.fee_bps":
                pool["policy"] = {"kind": "fixed_fee", "fee_bps": val}
            elif name.startswith("costs."):
                _set_dotted(costs, name.removeprefix("costs."), val)
            elif "." in name:
                _set_dotted(pool, name, self._raw_axis_value(name, val))
            else:
                pool[name] = self._raw_axis_value(name, val)
            touches_fee = touches_fee or name in {"mid_fee", "out_fee"}
        if self.fee_equalize and touches_fee and "mid_fee" in pool:
            pool["out_fee"] = pool["mid_fee"]
        elif touches_fee and "mid_fee" in pool and "out_fee" in pool:
            if float(pool["out_fee"]) < float(pool["mid_fee"]):
                pool["out_fee"] = pool["mid_fee"]
        return {
            "tag": "inspect",
            "pool": _stringify_pool(pool),
            "costs": costs,
        }

    def _run_inspect_simulation(
        self,
        idx_tuple: Tuple[int, ...],
        run_yb_releverage: bool = False,
    ):
        """Replay the exact clicked cell through the unified harness.

        The legacy inspect ran a second, detailed arb_sim subprocess; the
        ported viewer replays the identical cell through the same
        curve_fx_eval_v1 harness with full-trace observation (shiftclick) and
        opens the attested trajectory. ``run_yb_releverage`` is accepted for
        click-handler parity; the scenario's own yb_mode governs the replay.
        """
        if self._inspect_running:
            print("Inspect run already in progress; ignoring click.")
            return
        harness = self.harness_binary or self._discover_harness_binary()
        if not harness:
            print(
                "Inspect requires --harness <path to arb_evaluator_ld>; "
                "no locally built binary matching the run's attested SHA-256 was found."
            )
            return
        self.harness_binary = harness
        npz_run = self.data.get("_npz_run")
        coords = getattr(npz_run, "row_coordinates", {}).get(idx_tuple)
        if not coords:
            print("No exact row coordinates for this cell; skipping inspect.")
            return
        meta = self.data.get("metadata", {}) if isinstance(self.data, dict) else {}
        run_id = str(meta.get("run_id", ""))
        if not run_id:
            print("Run manifest carries no run_id; skipping inspect.")
            return

        self._inspect_running = True
        try:
            import shutil

            from ..artifacts.store import RunStore
            from ..evaluation.client import SubprocessHarnessClient
            from ..shiftclick import run_shiftclick
            from ..specs.common import repository_root
            from ..specs.shiftclick import ShiftclickSpec

            # Legacy click semantics: right-click = sparse trace, shift+click
            # = full trace. Sparse targets one sample per trading hour:
            # ~8.7k records for a year of 1-minute candles (two trace events
            # per candle, weekend gaps included via the scenario candle
            # count), staying under the 10k-point plot cap. Replays replace
            # one ephemeral viewer-inspect run instead of accumulating runs.
            n_candles = int(meta.get("n_candles", 0) or 0)
            trace_interval = 1
            if not run_yb_releverage:
                trace_interval = max(1, int(2 * n_candles / 8760)) if n_candles else 120
            spec = ShiftclickSpec(
                id="viewer-inspect",
                source_kind="grid",
                source_run_id=run_id,
                selection_kind="coordinates",
                selection_value=dict(coords),
                pair_id=str(meta.get("pair_id", "")),
                scenario_id=str(meta.get("scenario_id", "")),
                policy_id=str(meta.get("policy_id", "twocrypto_native")),
                trace_interval=trace_interval,
                trace_actions=True,
            )
            store = RunStore(repository_root())
            inspect_out = store.runs_dir / f"shiftclick_{spec.id}"
            if inspect_out.exists():
                shutil.rmtree(inspect_out)
            client = SubprocessHarnessClient(self.harness_binary, repository=store.root_dir)
            print(
                f"\nReplaying exact cell through the harness "
                f"(trace_interval={trace_interval})..."
            )
            result = run_shiftclick(
                spec, store=store, client=client, output_dir=inspect_out
            )
            print("Replay complete:", result.run_dir)
            self._show_trajectory_window(result.run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Inspect replay failed: {exc}")
        finally:
            self._inspect_running = False

    def _discover_harness_binary(self) -> str | None:
        """Find a locally built evaluator matching the run's attested SHA-256.

        The run manifest records the exact binary digest that produced the
        grid; only a binary with that digest can replay the cell identically.
        """
        import hashlib

        meta = self.data.get("metadata", {}) if isinstance(self.data, dict) else {}
        expected = str(meta.get("binary_sha256", ""))
        if len(expected) != 64:
            return None
        repo_root = Path(__file__).resolve().parents[3]
        harness_root = repo_root.parent / "curve-fx-arb-harness"
        candidates = [
            harness_root / "build" / "native" / "arb_evaluator_ld",
            harness_root / "build" / "compiled" / "arb_evaluator_ld",
            Path.home() / "arb" / "bin" / "arb_evaluator_ld",
        ]
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve() if candidate.exists() else candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.is_file():
                continue
            try:
                digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            except OSError:
                continue
            if digest.lower() == expected.lower():
                print(f"Discovered attested evaluator: {resolved}")
                return str(resolved)
        return None

    def _show_trajectory_window(self, run_dir: Path) -> None:
        """Open the legacy multi-panel shiftclick figure for the replay sidecars."""
        from ..plotting.shiftclick_view import render_shiftclick_figure

        trace_files = sorted(run_dir.glob("trace/*.trace.json"))
        if not trace_files:
            print(f"No trace sidecars found in {run_dir / 'trace'}")
            return
        action_files = sorted(run_dir.glob("trace/*.actions.json"))
        figure = render_shiftclick_figure(
            trace_files[0],
            action_files[0] if action_files else None,
            title=Path(run_dir).name,
            fee_source="both",
        )
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Shiftclick inspect")
        print("Opened shiftclick window; close it to continue.")
        plt.show(block=False)

    def _on_click(self, event):
        key = str(getattr(event, "key", "") or "").lower()
        is_shift_left_click = event.button == 1 and "shift" in key
        is_right_click = event.button == 3
        is_plain_left_click = event.button == 1 and not is_shift_left_click
        is_inspect_click = is_shift_left_click or is_right_click
        if not (is_plain_left_click or is_inspect_click):
            return
        if event.inaxes not in self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]
        x_idx = _nearest_index(xs, event.xdata)
        y_idx = _nearest_index(ys, event.ydata)

        indices: List[int] = []
        coords: Dict[str, float] = {}
        for name in self.dim_names:
            if name == self.x_name:
                idx = x_idx
                val = xs[x_idx]
            elif name == self.y_name:
                idx = y_idx
                val = ys[y_idx]
            else:
                idx = self.slider_indices.get(name, 0)
                val = self.dim_values[name][idx]
            indices.append(idx)
            coords[name] = val

        idx_tuple = tuple(indices)
        if is_plain_left_click:
            self._update_metrics_window(idx_tuple, coords)
            return

        pool_config = self._build_inspect_pool_config(coords, idx_tuple)

        print("\nSelected point:")
        for name in self.dim_names:
            print(f"  {name}: {self._format_axis_value(name, coords[name])}")

        if pool_config:
            print("Pool config:")
            print(json.dumps(pool_config, indent=2))
        self._run_inspect_simulation(
            idx_tuple,
            run_yb_releverage=is_shift_left_click,
        )

    def show(self):
        print(f"\nGrid dimensions: {self.dim_names}")
        for name in self.dim_names:
            vals = self.dim_values[name]
            print(
                f"  {name}: {len(vals)} values "
                f"({self._format_axis_value(name, vals[0])} .. {self._format_axis_value(name, vals[-1])})"
            )
        print(f"\nMetrics: {self.metrics}")
        print(f"X axis: {self.x_name}, Y axis: {self.y_name}")
        slider_dims = self._get_slider_dims()
        if slider_dims:
            print(f"Sliders: {[name for _, name in slider_dims]}")
        print("\nLeft click updates metrics window.")
        print("Shift+click / right-click replays the exact cell through the harness")
        print("  (full-trace shiftclick) and opens the attested trajectory window.")
        print("Close all windows to exit.")
        plt.show()

    def save(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig_main.savefig(out_path, dpi=150)
        print(f"Saved optimized N-D heatmap to {out_path}")


def open_viewer(
    run_dir: Path,
    *,
    metrics: List[str] | None = None,
    ncol: int = 3,
    cmap: str = "turbo",
    max_ticks: int = 12,
    clamp: bool = False,
    skewthr: float = 0.0,
    max_pricethr: float = 100.0,
    slipthr: float = 20.0,
    slipthr_max: float = 100.0,
    last_pdifthr: float = 0.0,
    log_axes: List[str] | None = None,
    start_time: str | None = None,
    out: Path | None = None,
    harness: str | None = None,
) -> int:
    """Open the N-D heatmap explorer for one collected fxsim run directory."""
    data = _load(run_dir)
    if metrics:
        selected = [m.strip() for m in metrics if m.strip()]
    else:
        available = set(data.get("_npz_run", {}).metric_names() or set())
        selected = [m for m in DEFAULT_METRICS if m in available] or ["apy_net"]
    explorer = NDHeatmapExplorerOpt(
        data=data,
        metrics=selected,
        ncol=ncol,
        cmap=cmap,
        max_ticks=max_ticks,
        clamp=clamp,
        price_thr_bps=skewthr,
        max_price_thr_bps=max_pricethr,
        final_price_thr_bps=last_pdifthr,
        skew_thr_pct=skewthr,
        slippage_thr_bps=slipthr,
        slippage_thr_max_bps=slipthr_max,
        log_axes=set(log_axes or set()),
        start_time=start_time,
    )
    explorer.harness_binary = harness
    if out is not None:
        explorer.save(out)
        plt.close("all")
    else:
        explorer.show()
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="N-D heatmap explorer for fxsim run dirs")
    ap.add_argument("run_dir", type=str, help="Collected fxsim run directory (manifest.json + evaluation_table.npz)")
    ap.add_argument("--metrics", type=str, default=None, help="Comma-separated metrics")
    ap.add_argument("--cmap", type=str, default="turbo")
    ap.add_argument("--max-ticks", type=int, default=12)
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--clamp", action="store_true", default=False)
    ap.add_argument("--skewthr", "--pricethr", "--imbalancethr", dest="skewthr", type=float, default=0.0,
                    help="Max max_7d_skew percent for apy_masked; defaults to the data max.")
    ap.add_argument("--max-pricethr", "--maxpdiffthr", type=float, default=100.0)
    ap.add_argument("--slipthr", "--slippagethr", type=float, default=20.0,
                    help="Initial slippage cap for apy_1/5/10_masked metrics.")
    ap.add_argument("--slipthr-max", type=float, default=100.0,
                    help="Maximum slippage-slider value in bps (default: 100).")
    ap.add_argument("--last-pdifthr", "--final-pdifthr", type=float, default=0.0,
                    help="Max final_rel_price_diff in bps for masked metrics; 0 defaults to the data max.")
    ap.add_argument("--log-axis", action="append", default=[],
                    help="Plot axes on a log scale. Comma-separated values and repeated flags are accepted.")
    ap.add_argument("--start-time", type=str, default=None,
                    help="Forward Unix timestamp or DD-MM-YYYY (legacy inspect start).")
    ap.add_argument("--harness", type=str, default=None,
                    help="Path to arb_evaluator_ld; enables shift/right-click exact-cell replay + trajectory.")
    ap.add_argument("--out", type=str, default=None,
                    help="Save current heatmap slice to this image instead of opening windows.")
    args = ap.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None
    return open_viewer(
        Path(args.run_dir),
        metrics=metrics,
        ncol=args.ncol,
        cmap=args.cmap,
        max_ticks=args.max_ticks,
        clamp=args.clamp,
        skewthr=args.skewthr,
        max_pricethr=args.max_pricethr,
        slipthr=args.slipthr,
        slipthr_max=args.slipthr_max,
        last_pdifthr=args.last_pdifthr,
        log_axes=_parse_log_axes(args.log_axis),
        start_time=args.start_time,
        out=Path(args.out) if args.out else None,
        harness=args.harness,
    )


if __name__ == "__main__":
    raise SystemExit(main())
