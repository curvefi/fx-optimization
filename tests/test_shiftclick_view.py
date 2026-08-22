"""Raw evaluator trace coverage for the multi-panel Shift-click view."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from curve_fx_sim.plotting.shiftclick_view import render_shiftclick_figure


def _trace(n: int, *, with_yb: bool) -> list[dict[str, object]]:
    start = 1_747_947_480
    return [
        {
            "t": start + index * 600,
            "price_scale": 1.1278 * (1 + math.sin(index / 60) * 0.002),
            "p_cex": 1.1278 * (1 + math.sin(index / 60) * 0.002 + 0.0002 * math.sin(index / 20)),
            "token0": 500_000.0,
            "token1": 443_300.0 * (1 + 0.01 * math.sin(index / 40)),
            "fee": 0.001 + 0.0005 * (index % 24) / 24,
            "slippage_1pct_0to1": 0.0005 + 0.0002 * (index % 12) / 12,
            "slippage_1pct_1to0": 0.0004 + 0.0003 * (index % 12) / 12,
            "lp_xcp_profit": 1.0 + 1e-4 * index / n + 0.001 * (index / n),
            "donation_apy": 0.02,
            "yb_initialized": 1.0 if with_yb else 0.0,
            "yb_growth": 1.001 + 1e-5 * index if with_yb else None,
            "yb_fee": 0.012 if with_yb else None,
            "yb_releverage_trades": index if with_yb else 0.0,
            "yb_stable_balance": 1000.0 * (1 + 0.01 * index / n) if with_yb else None,
            "yb_debt": 800.0 * (1 + 0.02 * index / n) if with_yb else None,
            "yb_collateral_lp": 1200.0 if with_yb else None,
            "yb_lp_oracle": 1.0 if with_yb else None,
            "yb_lp_fair": 1.01 if with_yb else None,
        }
        for index in range(n)
    ]


def _write_trace(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_raw_yb_trace_renders_all_shiftclick_panels(tmp_path: Path) -> None:
    trace = _write_trace(tmp_path / "evaluator.json", _trace(300, with_yb=True))
    figure = render_shiftclick_figure(trace, title="raw", donation_frequency=25_200.0)
    try:
        labels = {line.get_label() for axis in figure.axes for line in axis.lines}
        titles = {axis.get_title() for axis in figure.axes}
        assert len(figure.axes) >= 6
        assert "LP net growth" in labels
        assert "Pool skew" in labels
        assert "coin0→coin1 daily mean" in labels
        assert any(title.startswith("Price Scale vs CEX Price") for title in titles)
    finally:
        plt.close(figure)


def test_raw_non_yb_trace_keeps_price_and_slippage_panels(tmp_path: Path) -> None:
    trace = _write_trace(tmp_path / "evaluator.json", _trace(120, with_yb=False))
    figure = render_shiftclick_figure(trace, title="raw non-yb")
    try:
        labels = {line.get_label() for axis in figure.axes for line in axis.lines}
        assert "LP net growth" in labels
        assert "coin1→coin0 daily mean" in labels
        assert all("YB" not in axis.get_ylabel() for axis in figure.axes)
    finally:
        plt.close(figure)
