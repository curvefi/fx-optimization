"""Tests for the precise shiftclick multi-panel view (plot_price_scale port)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from curve_fx_sim.plotting.shiftclick_view import (
    _donation_growth,
    render_shiftclick_figure,
)
from curve_fx_sim.artifacts.io import atomic_write_json, sha256_path
from curve_fx_sim.shiftclick.archive import ACTION_COLUMNS, TRACE_COLUMNS


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


def _write_archive(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    arrays = {"trace_000": np.asarray([[row.get(name, np.nan) for name in TRACE_COLUMNS] for row in records], dtype="<f8")}
    for kind, columns in ACTION_COLUMNS.items():
        arrays[f"{kind}_000"] = np.empty((0, len(columns)), dtype="<f8")
    archive = tmp_path / "replay_trace.npz"
    np.savez_compressed(archive, **arrays)
    atomic_write_json(tmp_path / "replay_trace.json", {
        "schema_version": "curve_fx_replay_trace_v1",
        "source_run_id": "source", "candidate_id": "candidate", "ordinal": 0,
        "columns": {"trace": list(TRACE_COLUMNS), **{kind: list(columns) for kind, columns in ACTION_COLUMNS.items()}},
        "scenarios": [{"index": 0, "id": "scenario", "evaluation_id": "evaluation",
                       "economic_fingerprint": "fingerprint", "source_sidecars": [],
                       "row_counts": {"trace": len(records), **{kind: 0 for kind in ACTION_COLUMNS}}}],
        "npz": {"path": archive.name, "sha256": sha256_path(archive), "bytes": archive.stat().st_size},
    })
    return archive
def test_render_active_2l_uses_attested_donation_frequency(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    records = _synthetic_trace(300, with_yb=True)
    trace = _write_archive(tmp_path, records)
    fields = (
        "yb_initialized", "yb_growth", "yb_fee", "yb_releverage_trades",
        "yb_stable_balance", "yb_debt", "yb_collateral_lp", "yb_lp_oracle", "yb_lp_fair",
    )
    assert all(field in records[0] for field in fields)
    frequency = 25200.0
    timestamps = np.asarray([record["t"] for record in records], dtype=float)
    donation_apy = np.full(len(records), 0.02)
    expected_growth = _donation_growth(timestamps, donation_apy, frequency)
    expected_net_growth = np.asarray(
        [record["lp_xcp_profit"] for record in records], dtype=float
    ) / expected_growth
    fig = render_shiftclick_figure(
        trace, companion_path=trace.with_name("replay_trace.json"),
        title="active-2l", donation_frequency=frequency
    )
    growth_axis = next(
        axis for axis in fig.axes
        if any(line.get_label() == "LP net growth" for line in axis.lines)
    )
    growth_line = next(line for line in growth_axis.lines if line.get_label() == "LP net growth")
    assert growth_line.get_ydata()[-1] == pytest.approx((expected_net_growth[-1] - 1.0) * 100.0)
def test_render_non_yb_panels(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    trace = _write_archive(tmp_path, _synthetic_trace(200))
    render_shiftclick_figure(
        trace, companion_path=trace.with_name("replay_trace.json"), title="test")
