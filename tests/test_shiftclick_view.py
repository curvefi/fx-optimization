"""Tests for the precise shiftclick multi-panel view (plot_price_scale port)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from curve_fx_sim.plotting.shiftclick_view import (
    _format_pct,
    _load_executed_fee_actions,
    _rolling_window_growth_gm,
    render_shiftclick_figure,
)


def _synthetic_trace(n: int = 500, *, with_yb: bool = False) -> list[dict[str, object]]:
    t0 = 1_747_947_480
    records = []
    for i in range(n):
        t = t0 + i * 600  # 10-minute cadence
        drift = math.sin(i / 60.0) * 0.002
        records.append(
            {
                "t": t,
                "price_scale": 1.1278 * (1 + drift),
                "p_cex": 1.1278 * (1 + drift + 0.0002 * math.sin(i / 20.0)),
                "token0": 500_000.0,
                "token1": 443_300.0 * (1 + 0.01 * math.sin(i / 40.0)),
                "fee": 0.001 + 0.0005 * (i % 24) / 24.0,
                "slippage_1pct_0to1": 0.0005 + 0.0002 * (i % 12) / 12.0,
                "slippage_1pct_1to0": 0.0004 + 0.0003 * (i % 12) / 12.0,
                "lp_xcp_profit": 1.0 + 1e-4 * i / n + 0.001 * (i / n),
                "donation_apy": 0.02,
                "yb_initialized": 1.0 if with_yb else 0.0,
                "yb_growth": (1.001 + 1e-5 * i) if with_yb else float("nan"),
                "yb_fee": 0.012 if with_yb else float("nan"),
                "yb_releverage_trades": i if with_yb else 0.0,
                "yb_stable_balance": 1000.0 * (1 + 0.01 * i / n) if with_yb else float("nan"),
                "yb_debt": 800.0 * (1 + 0.02 * i / n) if with_yb else float("nan"),
                "yb_collateral_lp": 1200.0 if with_yb else float("nan"),
                "yb_lp_oracle": 1.0 if with_yb else float("nan"),
                "yb_lp_fair": 1.01 if with_yb else float("nan"),
            }
        )
    return records


def _write_sidecars(tmp_path: Path, records: list[dict[str, object]]) -> tuple[Path, Path]:
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(records), encoding="utf-8")
    actions = tmp_path / "actions.json"
    actions.write_text(
        json.dumps(
            [
                {"type": "exchange", "ts": records[0]["t"] + 1, "dy_after_fee": 90.0, "fee_tokens": 0.009},
                {"type": "donation", "ts": records[0]["t"] + 2},
                {"type": "exchange", "ts": records[1]["t"] + 1, "dy_after_fee": 50.0, "fee_tokens": 0.005},
            ]
        ),
        encoding="utf-8",
    )
    return trace, actions


def test_format_pct() -> None:
    assert _format_pct(0.1234) == "12.3%"
    assert _format_pct(float("nan")) == "n/a"


def test_rolling_gm_flat_growth() -> None:
    # The rolling GM window is 90 days; the span must exceed it.
    t = np.arange(0.0, 200 * 24.0 * 3600.0, 3600.0)
    growth = np.full_like(t, 1.0 + 1e-5)
    gm = _rolling_window_growth_gm(t, growth)
    assert np.isfinite(gm) and gm > 0.0


def test_executed_fee_actions(tmp_path: Path) -> None:
    trace, actions = _write_sidecars(tmp_path, _synthetic_trace(10))
    parsed = _load_executed_fee_actions(actions)
    assert parsed is not None
    assert len(parsed["timestamps"]) == 2
    # fee = fee_tokens / (dy_after_fee + fee_tokens)
    assert float(parsed["fee"][0]) == pytest.approx(0.009 / 90.009)


def test_render_non_yb_panels(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    trace, actions = _write_sidecars(tmp_path, _synthetic_trace(200))
    fig = render_shiftclick_figure(trace, actions, title="test")
    # prices + APY(+growth twin) + deviation/skew(+fee twin) + slippage
    assert len(fig.axes) == 7
    out = tmp_path / "out.png"
    fig.savefig(out, dpi=100)
    assert out.is_file() and out.stat().st_size > 0


def test_render_yb_panels(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    trace, actions = _write_sidecars(tmp_path, _synthetic_trace(300, with_yb=True))
    fig = render_shiftclick_figure(trace, actions, title="yb-test")
    # prices + APY + YB(+growth twin) + YB balance(+lp twin) + skew(+fee twin) + slippage
    assert len(fig.axes) == 10
    out = tmp_path / "yb.png"
    fig.savefig(out, dpi=100)
    assert out.is_file() and out.stat().st_size > 0
