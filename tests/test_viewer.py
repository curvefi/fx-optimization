"""Tests for the legacy-style N-D heatmap explorer (plotting/viewer + viewer_data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable
from curve_fx_sim.plotting.viewer_data import TableRun


def _core(policy_id: str = "twocrypto_native") -> dict[str, object]:
    return {
        "schema_version": "curve_fx_sim_identity_v2",
        "binary": "arb_evaluator_ld",
        "sha256": "a" * 64,
        "harness_version": "1.0.0",
        "pool_version": "0.1.0",
        "policy_id": policy_id,
        "policy_source_sha256": "none",
        "policy_abi": "twocrypto_policy_v1",
        "policy_parameter_count": 0,
        "numeric_mode": "longdouble",
        "metric_fields": ["apy_net", "max_7d_rel_price_diff", "tw_real_slippage_1pct"],
    }


def _make_run(tmp_path: Path, run_id: str = "viewer_run") -> Path:
    """Build a minimal collected run: manifest + evaluation_table.npz."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    axes = [
        {
            "name": "A",
            "targets": [{"path": ["A"], "scale": "10000", "display_scale": "1", "kind": "integer"}],
            "values": ["1", "10"],
        },
        {
            "name": "donation_apy",
            "targets": [{"path": ["donation_apy"], "scale": "1", "display_scale": "1", "kind": "decimal"}],
            "values": ["0", "0.05"],
        },
    ]
    coords = [
        (0, 0, "1", "0"),
        (0, 1, "1", "0.05"),
        (1, 0, "10", "0"),
        (1, 1, "10", "0.05"),
    ]
    pools = []
    rows = []
    for ordinal, (i, j, a, d) in enumerate(coords):
        pools.append(
            {
                "id": f"p{ordinal:06d}",
                "ordinal": ordinal,
                "coordinates": {"A": a, "donation_apy": d},
                "coordinate_indices": [i, j],
                "policy_params": [],
                "pool_overrides": {
                    "A": 10000 * (10 ** i),
                    "donation_apy": 0.05 * j,
                },
            }
        )
        rows.append(
            EvaluationRow(
                candidate_id=f"p{ordinal:06d}",
                ordinal=ordinal,
                coordinates={"A": a, "donation_apy": d},
                params={},
                pool_overrides={"A": 10000 * (10 ** i), "donation_apy": 0.05 * j},
                metrics={
                    "apy_net": 0.001 + 0.0001 * (i + j),
                    "max_7d_rel_price_diff": 0.005 + 0.001 * (i + j),
                    "tw_real_slippage_1pct": 0.0005 + 0.0002 * (i + j),
                },
                status="ok",
                economic_fingerprint=f"fp_{ordinal}",
            )
        )
    manifest = {
        "schema_version": "fxsim_manifest_v1",
        "run_kind": "grid",
        "run_id": run_id,
        "core": _core(),
        "resolved_spec": {
            "pair": {"id": "eurusd"},
            "scenario": {"id": "eurusd-2025"},
        },
        "grid": {
            "grid_id": "test_grid",
            "pool_count": 4,
            "resolved_axes": axes,
            "pools": pools,
        },
        "attempt_history": [],
        "artifacts": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    table = EvaluationTable(rows, metric_projection=None)
    table.to_npz(run_dir / "evaluation_table.npz")
    return run_dir


def test_adapter_axes_raw_units_and_cell_views(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    run = TableRun(run_dir)
    grid = run.metadata["grid"]
    assert grid["x1"]["name"] == "A"
    assert grid["x1"]["values"] == [10000.0, 100000.0]
    assert grid["x2"]["name"] == "donation_apy"
    assert grid["x2"]["values"] == [0.0, 0.05]
    assert run.metadata["run_id"] == "viewer_run"
    assert run.metadata["pair_id"] == "eurusd"
    assert run.metadata["scenario_id"] == "eurusd-2025"
    assert run.metadata["policy_id"] == "twocrypto_native"

    assert len(run.pool_configs) == 4
    assert len(run.metrics_lookup) == 4
    assert len(run.row_coordinates) == 4
    top = run.pool_configs[(1, 1)]
    assert top["pool"]["A"] == 100000
    assert top["pool"]["donation_apy"] == 0.05
    assert run.row_coordinates[(1, 1)] == {"A": "10", "donation_apy": "0.05"}

    apy = run.load_array("apy_net")
    assert apy.shape == (4,)
    assert float(apy[3]) == pytest.approx(0.0012)
    assert int(run.load_array("success").sum()) == 4
    with pytest.raises(KeyError):
        run.load_array("missing_metric")


def test_open_viewer_saves_slice(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from curve_fx_sim.plotting.viewer import open_viewer

    run_dir = _make_run(tmp_path)
    out = tmp_path / "slice.png"
    exit_code = open_viewer(
        run_dir,
        metrics=["apy_net", "max_7d_rel_price_diff"],
        ncol=2,
        max_ticks=12,
        max_pricethr=100.0,
        slipthr=20.0,
        slipthr_max=100.0,
        out=out,
    )
    assert exit_code == 0
    assert out.is_file() and out.stat().st_size > 0


def test_open_viewer_ui_structure(tmp_path: Path) -> None:
    """Legacy UI replica: three titled figures, X/Y radios, dimension sliders."""
    import matplotlib

    matplotlib.use("Agg")
    from curve_fx_sim.plotting.viewer import NDHeatmapExplorerOpt, _load

    data = _load(_make_run(tmp_path))
    explorer = NDHeatmapExplorerOpt(
        data=data,
        metrics=["apy_net", "max_7d_rel_price_diff"],
        ncol=2,
        cmap="turbo",
        max_ticks=12,
        clamp=False,
        price_thr_bps=0.0,
        max_price_thr_bps=100.0,
        slippage_thr_bps=20.0,
        slippage_thr_max_bps=100.0,
    )
    assert explorer.fig_main.canvas.manager.get_window_title() == "Heatmaps"
    assert explorer.fig_controls.canvas.manager.get_window_title() == "Controls"
    assert explorer.x_radio is not None and explorer.y_radio is not None
    assert [label.get_text() for label in explorer.x_radio.labels] == ["A", "donation_apy"]
    # 2x2 grid: no extra dims, so no dimension sliders; filter sliders only
    # appear when masked metrics are selected.
    assert [name for _, name in explorer._get_slider_dims()] == []
    assert len(explorer.meshes) == 2
    assert len(explorer.colorbars) == 2
    import matplotlib.pyplot as plt

    plt.close("all")
    # Masked metrics add the threshold filter sliders.
    explorer2 = NDHeatmapExplorerOpt(
        data=data,
        metrics=["apy_masked"],
        ncol=1,
        cmap="turbo",
        max_ticks=12,
        clamp=False,
        price_thr_bps=0.0,
        max_price_thr_bps=100.0,
        slippage_thr_bps=20.0,
        slippage_thr_max_bps=100.0,
    )
    assert explorer2.max_price_thr_slider is not None
    assert explorer2.slippage_thr_slider is None  # apy_masked has no slippage source
    plt.close("all")
    explorer3 = NDHeatmapExplorerOpt(
        data=data,
        metrics=["apy_1_masked"],
        ncol=1,
        cmap="turbo",
        max_ticks=12,
        clamp=False,
        price_thr_bps=0.0,
        max_price_thr_bps=100.0,
        slippage_thr_bps=20.0,
        slippage_thr_max_bps=100.0,
    )
    assert explorer3.slippage_thr_slider is not None
    plt.close("all")


def test_open_viewer_masked_metrics(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from curve_fx_sim.plotting.viewer import open_viewer

    run_dir = _make_run(tmp_path)
    out = tmp_path / "masked.png"
    exit_code = open_viewer(
        run_dir,
        metrics=["apy_masked", "apy_1_masked"],
        ncol=2,
        max_pricethr=100.0,
        slipthr=20.0,
        slipthr_max=100.0,
        out=out,
    )
    assert exit_code == 0
    assert out.is_file()
