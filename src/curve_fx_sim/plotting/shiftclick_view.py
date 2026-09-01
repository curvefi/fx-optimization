"""Multi-panel Shift-click view for raw evaluator traces.

The view contains stacked shared-x panels for prices, rolling 90d annualized LP
net APY with GM floor shading, YB releverage APY and balance sheet, pool skew
with binned pool fee, and 1% TVL slippage over a datetime axis. Input is the
raw evaluator JSON trace produced by Shift-click.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

SEC_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
ROLLING_APY_WINDOW_DAYS = 90
ROLLING_APY_WINDOW_S = float(ROLLING_APY_WINDOW_DAYS) * 24.0 * 60.0 * 60.0
ROLLING_APY_SAMPLE_S = 60.0 * 60.0
ROLLING_APY_FLOOR = 1e-20
DEFAULT_MAX_POINTS = 10_000


def _series(records: Sequence[Mapping[str, Any]], name: str, default: Any = None) -> np.ndarray:
    values = []
    for record in records:
        value = record.get(name)
        if value is None:
            if default is None:
                values.append(float("nan"))
            else:
                values.append(default)
        else:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _trace_to_detailed(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Map harness trace records onto the detailed plotting field set."""
    n = len(records)
    return {
        "timestamps": _series(records, "t"),
        "price_scale": _series(records, "price_scale"),
        "p_cex": _series(records, "p_cex"),
        "token0": _series(records, "token0"),
        "token1": _series(records, "token1"),
        "fee": _series(records, "fee", float("nan")),
        "slippage_1pct_0to1": _series(records, "slippage_1pct_0to1", float("nan")),
        "slippage_1pct_1to0": _series(records, "slippage_1pct_1to0", float("nan")),
        "lp_xcp_profit": _series(records, "lp_xcp_profit", float("nan")),
        "donation_apy": _series(records, "donation_apy", 0.0),
        "yb_initialized": _series(records, "yb_initialized", 0.0),
        "yb_growth": _series(records, "yb_growth", float("nan")),
        "yb_fee": _series(records, "yb_fee", float("nan")),
        "yb_releverage_trades": _series(records, "yb_releverage_trades", 0.0),
        "yb_stable_balance": _series(records, "yb_stable_balance", float("nan")),
        "yb_debt": _series(records, "yb_debt", float("nan")),
        "yb_collateral_lp": _series(records, "yb_collateral_lp", float("nan")),
        "yb_lp_oracle": _series(records, "yb_lp_oracle", float("nan")),
        "yb_lp_fair": _series(records, "yb_lp_fair", float("nan")),
    }


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100.0:.3g}%"


def _donation_growth(timestamps: np.ndarray, donation_apy: np.ndarray, donation_frequency: float) -> np.ndarray:
    elapsed = timestamps - timestamps[0]
    donation_apy = np.nan_to_num(donation_apy, nan=0.0)
    if donation_frequency and donation_frequency > 0.0:
        period_rate = donation_apy * donation_frequency / SEC_PER_YEAR
        return np.power(1.0 + period_rate, elapsed / donation_frequency)
    return np.power(1.0 + donation_apy, elapsed / SEC_PER_YEAR)


def _annualized_growth_apy(timestamps: np.ndarray, growth: np.ndarray) -> float:
    if len(timestamps) < 2 or not (growth[-1] > 0.0) or not (growth[0] > 0.0):
        return np.nan
    dt = timestamps[-1] - timestamps[0]
    if not (dt > 0.0):
        return np.nan
    return np.expm1(np.log(growth[-1] / growth[0]) * SEC_PER_YEAR / dt)


def _rolling_window_growth_gm(timestamps: np.ndarray, growth: np.ndarray) -> float:
    if len(timestamps) == 0:
        return np.nan
    samples: list[tuple[float, float]] = []
    last_sample_ts: float | None = None
    sum_log_apy = 0.0
    n_windows = 0
    for ts, value in zip(timestamps, growth):
        if not np.isfinite(value) or not (value > 0.0):
            continue
        if last_sample_ts is not None and ts < last_sample_ts + ROLLING_APY_SAMPLE_S:
            continue
        samples.append((float(ts), float(value)))
        last_sample_ts = float(ts)
        cutoff = ts - ROLLING_APY_WINDOW_S if ts > ROLLING_APY_WINDOW_S else 0.0
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.pop(0)
        first_ts, first_value = samples[0]
        if ts < first_ts + ROLLING_APY_WINDOW_S:
            continue
        dt = ts - first_ts
        if not (dt > 0.0) or not (first_value > 0.0):
            continue
        window_growth = value / first_value
        annualized = ROLLING_APY_FLOOR
        if np.isfinite(window_growth) and window_growth > 0.0:
            annualized = np.power(window_growth, SEC_PER_YEAR / dt) - 1.0
            if not np.isfinite(annualized) or annualized < ROLLING_APY_FLOOR:
                annualized = ROLLING_APY_FLOOR
        sum_log_apy += float(np.log(annualized))
        n_windows += 1
    if n_windows == 0:
        return np.nan
    return float(np.exp(sum_log_apy / n_windows))


def _rolling_window_net_apy(
    timestamps: np.ndarray,
    profit_growth: np.ndarray,
    donation_apy: np.ndarray,
    donation_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamps) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    donation_growth = _donation_growth(timestamps, donation_apy, donation_frequency)
    with np.errstate(invalid="ignore", divide="ignore"):
        net_profit_growth = profit_growth / donation_growth
    rolling = np.full(len(timestamps), np.nan, dtype=float)
    floored = np.zeros(len(timestamps), dtype=bool)
    start = 0
    for i, ts in enumerate(timestamps):
        cutoff = ts - ROLLING_APY_WINDOW_S
        while start + 1 < len(timestamps) and timestamps[start + 1] <= cutoff:
            start += 1
        dt = ts - timestamps[start]
        if dt < ROLLING_APY_WINDOW_S or not (net_profit_growth[start] > 0.0):
            continue
        growth = net_profit_growth[i] / net_profit_growth[start]
        if not np.isfinite(growth) or growth <= 0.0:
            rolling[i] = 0.0
            floored[i] = True
            continue
        apy = np.power(growth, SEC_PER_YEAR / dt) - 1.0
        rolling[i] = apy
        floored[i] = bool(np.isfinite(apy) and apy < 0.0)
    return rolling, floored


def _rolling_window_growth_apy(timestamps: np.ndarray, growth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rolling = np.full(len(timestamps), np.nan, dtype=float)
    floored = np.zeros(len(timestamps), dtype=bool)
    start = 0
    for i, ts in enumerate(timestamps):
        cutoff = ts - ROLLING_APY_WINDOW_S
        while start + 1 < len(timestamps) and timestamps[start + 1] <= cutoff:
            start += 1
        dt = ts - timestamps[start]
        if dt < ROLLING_APY_WINDOW_S or not (growth[start] > 0.0):
            continue
        window_growth = growth[i] / growth[start]
        if not np.isfinite(window_growth) or window_growth <= 0.0:
            rolling[i] = 0.0
            floored[i] = True
            continue
        apy = np.power(window_growth, SEC_PER_YEAR / dt) - 1.0
        rolling[i] = apy
        floored[i] = bool(np.isfinite(apy) and apy < 0.0)
    return rolling, floored


def _make_yb_path(
    t: np.ndarray,
    growth: np.ndarray,
    summary: Mapping[str, Any],
    source: str,
    max_points: int,
) -> dict[str, Any] | None:
    if len(t) == 0:
        return None
    rolling_apy, rolling_floored = _rolling_window_growth_apy(t, growth)
    gm_apy = _rolling_window_growth_gm(t, growth)
    if max_points > 0 and len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points, dtype=int)
        t = t[idx]
        growth = growth[idx]
        rolling_apy = rolling_apy[idx]
        rolling_floored = rolling_floored[idx]
    elapsed = t - t[0]
    apy = np.full(len(t), np.nan, dtype=np.float64)
    mask = (elapsed > 0.0) & (growth > 0.0)
    apy[mask] = np.expm1(np.log(growth[mask]) * SEC_PER_YEAR / elapsed[mask])
    return {
        "t": t,
        "growth": growth,
        "apy": apy,
        "rolling_apy": rolling_apy,
        "rolling_floored": rolling_floored,
        "gm_apy": gm_apy,
        "summary": summary,
        "source": source,
    }


def _load_embedded_yb_path(detailed: Mapping[str, np.ndarray], max_points: int) -> dict[str, Any] | None:
    initialized = detailed.get("yb_initialized")
    growth = detailed.get("yb_growth")
    if initialized is None or growth is None:
        return None
    mask = (initialized > 0.5) & np.isfinite(growth)
    if not np.any(mask):
        return None
    t = detailed["timestamps"][mask]
    growth = growth[mask]
    fees = detailed.get("yb_fee")
    trades = detailed.get("yb_releverage_trades")
    finite_fees = fees[mask][np.isfinite(fees[mask])] if fees is not None else []
    summary = {
        "fee": float(finite_fees[-1]) if len(finite_fees) else None,
        "apy": _annualized_growth_apy(t, growth),
        "apy_gm": _rolling_window_growth_gm(t, growth),
        "n_releverage_trades": int(trades[mask][-1]) if trades is not None else None,
    }
    return _make_yb_path(t, growth, summary, "pool trace", max_points)


def _load_embedded_yb_balance_path(detailed: Mapping[str, np.ndarray], max_points: int) -> dict[str, Any] | None:
    names = (
        "yb_stable_balance",
        "yb_debt",
        "yb_collateral_lp",
        "yb_lp_oracle",
        "yb_lp_fair",
    )
    series = [detailed.get(name) for name in names]
    if any(values is None for values in series):
        return None
    initialized = detailed.get("yb_initialized")
    mask = np.ones(len(detailed["timestamps"]), dtype=bool)
    if initialized is not None:
        mask &= initialized > 0.5
    for values in series:
        mask &= np.isfinite(values)
    if not np.any(mask):
        return None
    result: dict[str, Any] = {"t": detailed["timestamps"][mask]}
    for name, values in zip(names, series, strict=True):
        result[name] = values[mask]
    if max_points > 0 and len(result["t"]) > max_points:
        idx = np.linspace(0, len(result["t"]) - 1, max_points, dtype=int)
        for name in tuple(result):
            result[name] = result[name][idx]
    return result


def _plot_yb_axis(ax, yb_path):
    yb_dates = [datetime.fromtimestamp(t, timezone.utc) for t in yb_path["t"]]
    growth_pct = (yb_path["growth"] - 1.0) * 100.0
    rolling_apy_pct = yb_path["rolling_apy"] * 100.0
    rolling_floored = yb_path["rolling_floored"] & np.isfinite(rolling_apy_pct)
    summary = yb_path["summary"]

    ax.plot(
        yb_dates,
        rolling_apy_pct,
        linewidth=0.9,
        color="purple",
        alpha=0.85,
        label=f"rolling {ROLLING_APY_WINDOW_DAYS}d annualized YB APY",
    )
    if np.any(rolling_floored):
        ax.fill_between(
            yb_dates,
            rolling_apy_pct,
            0.0,
            where=rolling_floored,
            color="red",
            alpha=0.25,
            label="floored in GM",
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel(f"{ROLLING_APY_WINDOW_DAYS}d YB APY (%)", color="purple")
    ax.tick_params(axis="y", labelcolor="purple")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax_growth = ax.twinx()
    ax_growth.plot(yb_dates, growth_pct, linewidth=1.0, color="tab:blue", alpha=0.65, label="YB deposit growth")
    ax_growth.set_ylabel("YB growth (%)", color="tab:blue")
    ax_growth.tick_params(axis="y", labelcolor="tab:blue")
    ax_growth.legend(loc="upper center")

    fee = summary.get("fee")
    final_apy = summary.get("apy")
    summary_gm_apy = summary.get("apy_gm")
    gm_apy = summary_gm_apy if isinstance(summary_gm_apy, (int, float)) else yb_path.get("gm_apy", np.nan)
    trades = summary.get("n_releverage_trades")
    source = yb_path.get("source", "unknown source")
    title_parts = []
    if isinstance(final_apy, (int, float)):
        title_parts.append(f"APY={final_apy * 100.0:.3g}% (GM={_format_pct(gm_apy)})")
    if isinstance(fee, (int, float)):
        title_parts.append(f"fee={fee * 100.0:.3g}%")
    if isinstance(trades, (int, float)):
        title_parts.append(f"trades={int(trades)}")
    if title_parts:
        ax.set_title(f"YB releverage [{source}]: " + ", ".join(title_parts), fontsize=10)


def _plot_yb_balance_axis(ax, balance_path):
    dates = [datetime.fromtimestamp(t, timezone.utc) for t in balance_path["t"]]
    cash = balance_path["yb_stable_balance"]
    debt = balance_path["yb_debt"]
    collateral = balance_path["yb_collateral_lp"]
    oracle_value = collateral * balance_path["yb_lp_oracle"]
    fair_value = collateral * balance_path["yb_lp_fair"]

    ax.plot(dates, cash, color="tab:blue", linewidth=0.9, label="LevAMM stable cash")
    ax.plot(dates, debt, color="tab:red", linewidth=0.9, label="LevAMM debt")
    ax.plot(dates, oracle_value, color="tab:orange", linestyle="--", linewidth=0.9, label="LP collateral @ LP oracle")
    ax.plot(dates, fair_value, color="tab:green", linestyle="--", linewidth=0.9, label="LP collateral @ fair NAV")
    ax.set_ylabel("stable units")
    ax.tick_params(axis="y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title("YB AMM balance sheet", fontsize=10)

    ax_lp = ax.twinx()
    ax_lp.plot(dates, collateral, color="purple", linewidth=1.0, alpha=0.75, label="LevAMM LP collateral")
    ax_lp.set_ylabel("LP shares", color="purple")
    ax_lp.tick_params(axis="y", labelcolor="purple")
    ax_lp.legend(loc="upper right", fontsize=8)


def _plot_pool_fee_axis(ax, dates, pool_fee, *, spine_offset=None):
    if pool_fee is None:
        return None
    fee_pct = pool_fee * 100.0
    if not np.any(np.isfinite(fee_pct)):
        return None
    if spine_offset is not None:
        ax.spines["right"].set_position(("axes", spine_offset))
        ax.spines["right"].set_visible(True)
    ax.plot(dates, fee_pct, linewidth=0.75, color="tab:green", alpha=0.45, label="Raw sampled pool fee")
    ax.set_ylabel("Pool fee (%)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.legend(loc="lower right")
    return ax


def _plot_binned_pool_fee_axis(ax, timestamps, pool_fee, *, bin_hours: float = 24.0, spine_offset=None):
    if pool_fee is None:
        return None
    fee_pct = pool_fee * 100.0
    mask = np.isfinite(timestamps) & np.isfinite(fee_pct)
    if not np.any(mask):
        return None
    t = timestamps[mask]
    fee = fee_pct[mask]
    if t.size == 0:
        return None
    if spine_offset is not None:
        ax.spines["right"].set_position(("axes", spine_offset))
        ax.spines["right"].set_visible(True)
    bin_seconds = max(float(bin_hours) * 3600.0, 1.0)
    start = np.floor(float(t[0]) / bin_seconds) * bin_seconds
    stop = np.ceil(float(t[-1]) / bin_seconds) * bin_seconds + bin_seconds
    edges = np.arange(start, stop + bin_seconds * 0.5, bin_seconds, dtype=np.float64)
    if edges.size < 2:
        return _plot_pool_fee_axis(
            ax,
            [datetime.fromtimestamp(ts, timezone.utc) for ts in t],
            fee / 100.0,
            spine_offset=spine_offset,
        )
    lo = np.searchsorted(t, edges[:-1], side="left")
    hi = np.searchsorted(t, edges[1:], side="left")
    centers, median, p10, p90 = [], [], [], []
    for left, right, edge_left, edge_right in zip(lo, hi, edges[:-1], edges[1:], strict=False):
        if right <= left:
            continue
        values = fee[left:right]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        centers.append((edge_left + edge_right) * 0.5)
        p10_v, med_v, p90_v = np.percentile(values, [10, 50, 90])
        p10.append(p10_v)
        median.append(med_v)
        p90.append(p90_v)
    if not centers:
        return None
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in centers]
    median_arr = np.asarray(median, dtype=np.float64)
    p10_arr = np.asarray(p10, dtype=np.float64)
    p90_arr = np.asarray(p90, dtype=np.float64)
    ax.fill_between(dates, p10_arr, p90_arr, color="tab:green", alpha=0.14, linewidth=0.0,
                    label=f"Sampled pool fee p10-p90 ({bin_hours:g}h)")
    ax.plot(dates, median_arr, linewidth=0.9, color="tab:green", alpha=0.9,
            label=f"Sampled pool fee median ({bin_hours:g}h)")
    ax.set_ylabel("Pool fee (%)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.legend(loc="lower right")
    return ax


def _plot_slippage_axis(ax, timestamps, slippage_0to1, slippage_1to0, *, bin_hours: float = 24.0):
    slip01_bps = np.asarray(slippage_0to1, dtype=np.float64) * 10_000.0
    slip10_bps = np.asarray(slippage_1to0, dtype=np.float64) * 10_000.0
    timestamps = np.asarray(timestamps, dtype=np.float64)
    mask = np.isfinite(timestamps) & np.isfinite(slip01_bps) & np.isfinite(slip10_bps)
    if not np.any(mask):
        return None
    t = timestamps[mask]
    s01 = slip01_bps[mask]
    s10 = slip10_bps[mask]
    mean = 0.5 * (s01 + s10)
    bin_seconds = max(float(bin_hours) * 3600.0, 1.0)
    start = np.floor(float(t[0]) / bin_seconds) * bin_seconds
    stop = np.ceil(float(t[-1]) / bin_seconds) * bin_seconds + bin_seconds
    edges = np.arange(start, stop + bin_seconds * 0.5, bin_seconds, dtype=np.float64)
    lo = np.searchsorted(t, edges[:-1], side="left")
    hi = np.searchsorted(t, edges[1:], side="left")
    centers, mean_p10, mean_median, mean_p90, med01, med10 = [], [], [], [], [], []
    for left, right, edge_left, edge_right in zip(lo, hi, edges[:-1], edges[1:], strict=False):
        if right <= left:
            continue
        centers.append((edge_left + edge_right) * 0.5)
        p10, median, p90 = np.percentile(mean[left:right], [10, 50, 90])
        mean_p10.append(p10)
        mean_median.append(median)
        mean_p90.append(p90)
        med01.append(np.mean(s01[left:right]))
        med10.append(np.mean(s10[left:right]))
    if not centers:
        return None
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in centers]
    ax.fill_between(dates, mean_p10, mean_p90, color="gray", alpha=0.18, linewidth=0.0,
                    label=f"two-way mean p10-p90 ({bin_hours:g}h)")
    ax.plot(dates, mean_median, color="black", linewidth=0.9, label="two-way mean median")
    ax.plot(dates, med01, color="tab:blue", linewidth=0.65, label="coin0→coin1 daily mean")
    ax.plot(dates, med10, color="tab:orange", linewidth=0.65, label="coin1→coin0 daily mean")
    ax.axhline(10.0, color="red", linestyle="--", linewidth=0.9, label="10 bps target")
    ax.set_ylabel("1% TVL slippage (bps)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def _plot_pool_skew_axis(ax, dates, pool_skew):
    ax.plot(dates, pool_skew, linewidth=0.8, color="blue", alpha=0.75, label="Pool skew")
    ax.axhline(50, color="blue", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Pool skew (%)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def render_shiftclick_figure(
    trace_path: Path,
    *,
    title: str | None = None,
    donation_frequency: float | None = None,
):
    """Build the multi-panel view from a raw evaluator trace."""
    bin_hours = 6.0
    max_points = DEFAULT_MAX_POINTS
    payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(row, Mapping) for row in payload):
        raise ValueError("raw evaluator trace must be an array of objects")
    records = tuple(payload)
    detailed = _trace_to_detailed(records)
    timestamps = detailed["timestamps"]
    price_scale = detailed["price_scale"]
    p_cex = detailed["p_cex"]
    token0 = detailed["token0"]
    token1 = detailed["token1"]
    pool_fee = detailed["fee"]
    slippage_1pct_0to1 = detailed["slippage_1pct_0to1"]
    slippage_1pct_1to0 = detailed["slippage_1pct_1to0"]
    have_slippage_panel = bool(
        np.any(np.isfinite(slippage_1pct_0to1)) or np.any(np.isfinite(slippage_1pct_1to0))
    )
    sampled_fee_timestamps = timestamps
    sampled_pool_fee = pool_fee
    lp_xcp_profit = detailed["lp_xcp_profit"]
    donation_apy = detailed["donation_apy"]
    if donation_frequency is None:
        donation_frequency = 0.0
    else:
        try:
            donation_frequency = float(donation_frequency)
        except (TypeError, ValueError):
            donation_frequency = 0.0
    donation_growth = _donation_growth(timestamps, donation_apy, donation_frequency)
    with np.errstate(invalid="ignore", divide="ignore"):
        net_lp_growth = lp_xcp_profit / donation_growth
    lp_net_apy = _annualized_growth_apy(timestamps, net_lp_growth)
    lp_gm_apy = _rolling_window_growth_gm(timestamps, net_lp_growth)
    rolling_apy, rolling_floored = _rolling_window_net_apy(
        timestamps, lp_xcp_profit, donation_apy, donation_frequency
    )
    val0 = token0
    val1 = token1 * p_cex
    denom = val0 + val1
    pool_skew = np.zeros_like(denom, dtype=float)
    mask = denom > 0
    pool_skew[mask] = np.maximum(val0[mask], val1[mask]) / denom[mask]
    pool_skew *= 100.0
    rel_diff = (price_scale / p_cex - 1) * 100

    n = len(timestamps)
    if n > max_points:
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        timestamps = timestamps[indices]
        price_scale = price_scale[indices]
        p_cex = p_cex[indices]
        pool_fee = pool_fee[indices]
        pool_skew = pool_skew[indices]
        rel_diff = rel_diff[indices]
        rolling_apy = rolling_apy[indices]
        rolling_floored = rolling_floored[indices]
        net_lp_growth = net_lp_growth[indices]
        slippage_1pct_0to1 = slippage_1pct_0to1[indices]
        slippage_1pct_1to0 = slippage_1pct_1to0[indices]

    slippage_timestamps = timestamps
    dates = [datetime.fromtimestamp(t, timezone.utc) for t in timestamps]

    yb_path = _load_embedded_yb_path(detailed, max_points)
    yb_balance_path = _load_embedded_yb_balance_path(detailed, max_points)
    panel_count = 3 + int(yb_balance_path is not None) + int(yb_path is not None) + int(
        have_slippage_panel
    )
    fig, panel_array = plt.subplots(
        panel_count,
        1,
        figsize=(14, 3 * panel_count + 1),
        sharex=True,
        squeeze=False,
    )
    panels = list(panel_array[:, 0])
    ax1, ax_apy, ax2 = panels[:3]
    panel_index = 3
    ax_yb_balance = None
    if yb_balance_path is not None:
        ax_yb_balance = panels[panel_index]
        panel_index += 1
    ax_skew = None
    if yb_path is not None:
        ax_skew = panels[panel_index]
        panel_index += 1
    ax_slippage = None
    if have_slippage_panel:
        ax_slippage = panels[panel_index]

    ax1.plot(dates, p_cex, label="CEX price", alpha=0.7, linewidth=1)
    ax1.plot(dates, price_scale, label="price_scale", alpha=0.7, linewidth=1)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.set_title(f"Price Scale vs CEX Price\n{title or Path(trace_path).name}")
    ax1.grid(True, alpha=0.3)

    rolling_apy_pct = rolling_apy * 100.0
    net_lp_growth_pct = (net_lp_growth - 1.0) * 100.0
    ax_apy.plot(
        dates,
        rolling_apy_pct,
        linewidth=0.8,
        color="green",
        alpha=0.85,
        label=f"rolling {ROLLING_APY_WINDOW_DAYS}d annualized lp_xcp net APY",
    )
    if np.any(rolling_floored & np.isfinite(rolling_apy_pct)):
        ax_apy.fill_between(
            dates,
            rolling_apy_pct,
            0.0,
            where=rolling_floored & np.isfinite(rolling_apy_pct),
            color="red",
            alpha=0.25,
            label="floored in GM",
        )
    ax_apy.axhline(0, color="black", linewidth=0.7)
    ax_apy.set_ylabel(f"{ROLLING_APY_WINDOW_DAYS}d lp_xcp net APY (%)")
    ax_apy.set_title(f"LP net APY={_format_pct(lp_net_apy)} (GM={_format_pct(lp_gm_apy)})", fontsize=10)
    ax_apy.legend(loc="upper left")
    ax_apy.grid(True, alpha=0.3)
    ax_apy_growth = ax_apy.twinx()
    ax_apy_growth.plot(dates, net_lp_growth_pct, linewidth=1.0, color="tab:blue", alpha=0.65, label="LP net growth")
    ax_apy_growth.set_ylabel("LP net growth (%)", color="tab:blue")
    ax_apy_growth.tick_params(axis="y", labelcolor="tab:blue")
    ax_apy_growth.legend(loc="upper center")

    if yb_path is not None:
        _plot_yb_axis(ax2, yb_path)
        if ax_yb_balance is not None:
            _plot_yb_balance_axis(ax_yb_balance, yb_balance_path)
        _plot_pool_skew_axis(ax_skew, dates, pool_skew)
        fee_axis = ax_skew.twinx()
        if _plot_binned_pool_fee_axis(
            fee_axis,
            sampled_fee_timestamps,
            sampled_pool_fee,
            bin_hours=bin_hours,
        ) is None:
            fee_axis.remove()
    else:
        ax2.plot(dates, rel_diff, linewidth=0.5, color="red", alpha=0.7, label="Price deviation")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axhline(1, color="gray", linestyle="--", linewidth=0.5, label="±1%")
        ax2.axhline(-1, color="gray", linestyle="--", linewidth=0.5)
        ax2.axhline(5, color="orange", linestyle="--", linewidth=0.5, label="±5%")
        ax2.axhline(-5, color="orange", linestyle="--", linewidth=0.5)
        ax2.set_ylabel("Price deviation (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        ax3 = ax2.twinx()
        ax3.plot(dates, pool_skew, linewidth=0.8, color="blue", alpha=0.6, label="Pool skew")
        ax3.axhline(50, color="blue", linestyle=":", linewidth=0.5, alpha=0.5)
        ax3.set_ylabel("Pool skew (%)", color="blue")
        ax3.tick_params(axis="y", labelcolor="blue")
        ax3.set_ylim(50, 100)
        ax3.legend(loc="upper right")
        fee_axis = ax2.twinx()
        if _plot_binned_pool_fee_axis(
            fee_axis,
            sampled_fee_timestamps,
            sampled_pool_fee,
            bin_hours=bin_hours,
            spine_offset=1.08,
        ) is None:
            fee_axis.remove()

    if ax_slippage is not None:
        _plot_slippage_axis(
            ax_slippage,
            slippage_timestamps,
            slippage_1pct_0to1,
            slippage_1pct_1to0,
        )

    bottom_ax = ax_slippage or (ax_skew if ax_skew is not None else ax2)
    bottom_ax.set_xlabel("Date")
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    bottom_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


__all__ = ["render_shiftclick_figure"]
