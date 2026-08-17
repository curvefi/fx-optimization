"""Heatmap metadata compatibility, coordinates inference, and state tests."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.manifest import new_grid_manifest, write_manifest_atomic
from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable
from curve_fx_sim.cli import main
from curve_fx_sim.plotting.heatmap import (
    HeatmapAxis,
    HeatmapDataset,
    HeatmapState,
    HeatmapValidationError,
    MaskSpec,
    _cell_index,
    render_heatmap,
)


def test_axis_metadata_ignores_empty_inactive_representation() -> None:
    axis = HeatmapAxis.from_metadata(
        {
            "name": "kappa",
            "names": [],
            "values": ["0.5", "1.0"],
            "rows": [],
            "generation": {},
        }
    )

    assert axis.names == ("kappa",)
    assert axis.values == ("0.5", "1.0")


def test_axis_metadata_rejects_two_populated_representations() -> None:
    with pytest.raises(HeatmapValidationError, match="both rows and values"):
        HeatmapAxis.from_metadata(
            {
                "names": ["x", "y"],
                "values": [1],
                "rows": [[1, 2]],
                "generation": {},
            }
        )


def test_numeric_axis_preserves_descending_declaration_order() -> None:
    axis = HeatmapAxis(names=("weight",), values=("3", "2", "1"))
    assert axis.values == ("3", "2", "1")
    assert _cell_index(axis, 0.0) == 0
    assert _cell_index(axis, 1.0) == 1
    assert _cell_index(axis, 2.0) == 2


def test_numeric_axis_rejects_exact_duplicate_values() -> None:
    with pytest.raises(HeatmapValidationError, match="must be unique"):
        HeatmapAxis(names=("weight",), values=("1", "1.0"))


def _three_axis_table(*, with_metadata: bool = True) -> EvaluationTable:
    """3x3x2 coordinates grid with declared metadata (grid-style) or bare rows."""
    kappa = ("0.5", "1.0", "2.0")
    weight = ("3", "2", "1")
    mode = ("a", "b")
    rows: list[EvaluationRow] = []
    indices: dict[str, list[int]] = {}
    for ordinal, (ki, wi, mi) in enumerate(
        itertools.product(range(3), range(3), range(2))
    ):
        candidate_id = f"cand_{ordinal:04d}"
        rows.append(
            EvaluationRow(
                candidate_id=candidate_id,
                ordinal=ordinal,
                coordinates={"kappa": kappa[ki], "weight": weight[wi], "mode": mode[mi]},
                metrics={"apy": 0.01 * (ki + 1) + 0.001 * wi},
            )
        )
        indices[candidate_id] = [ki, wi, mi]
    metadata: dict[str, object] = {}
    if with_metadata:
        metadata.update(
            {
                "axes": [
                    {"name": "kappa", "values": list(kappa), "generation": {}},
                    {"name": "weight", "values": list(weight), "generation": {}},
                    {"name": "mode", "values": list(mode), "generation": {}},
                ],
                "coordinate_indices": indices,
            }
        )
    return EvaluationTable(rows=rows, metadata=metadata)


def test_dataset_infers_exact_dense_axes_from_coordinates_json() -> None:
    dataset = HeatmapDataset.from_table(_three_axis_table(with_metadata=False))
    assert dataset.axis_keys == ("kappa", "weight", "mode")
    kappa, weight, mode = dataset.axes
    assert kappa.values == ("0.5", "1.0", "2.0")
    assert kappa.scale == "log"
    assert weight.values == ("3", "2", "1")
    assert weight.scale == "linear"
    assert mode.values == ("a", "b")
    assert mode.scale == "categorical"
    assert dataset.shape == (3, 3, 2)
    assert dataset.metrics["apy"].shape == (3, 3, 2)
    assert dataset.candidate_ids[1, 1, 0] == "cand_0008"


def test_dataset_metadata_path_preferred_when_declared() -> None:
    table = _three_axis_table(with_metadata=True)
    dataset = HeatmapDataset.from_table(table)
    assert dataset.axis_keys == ("kappa", "weight", "mode")
    assert dataset.candidate_ids[0, 0, 0] == "cand_0000"
    # String-valued metadata axes auto-detect as categorical.
    assert dataset.axis("mode").scale == "categorical"


def test_axis_metadata_declared_scale_overrides_autodetection() -> None:
    axis = HeatmapAxis.from_metadata(
        {"name": "ratio", "values": ["0.5", "1.0", "2.0"], "generation": {"scale": "linear"}}
    )
    assert axis.scale == "linear"
    axis = HeatmapAxis.from_metadata(
        {"name": "mode", "values": ["a", "b"], "generation": {"scale": "categorical"}}
    )
    assert axis.scale == "categorical"


def test_dataset_rejects_inconsistent_coordinate_namespaces() -> None:
    rows = [
        EvaluationRow(
            candidate_id=f"cand_{index}",
            ordinal=index,
            coordinates=(
                {"a": 1, "b": 2} if index == 0 else {"a": 1, "c": 2}
            ),
        )
        for index in range(2)
    ]
    with pytest.raises(HeatmapValidationError, match="coordinate namespace"):
        HeatmapDataset.from_table(EvaluationTable(rows=rows))


def test_masked_metric_arrays_follow_legacy_semantics() -> None:
    rows = []
    for ordinal, (x, y) in enumerate(itertools.product(("1", "2"), ("1", "2"))):
        pdiff = 0.004 + 0.005 * ordinal  # 0.4% .. 1.9%
        slippage = 0.0015 + 0.001 * ordinal  # 0.15% .. 0.45%
        rows.append(
            EvaluationRow(
                candidate_id=f"cand_{ordinal}",
                ordinal=ordinal,
                coordinates={"x": x, "y": y},
                metrics={
                    "apy_net": 0.05 + 0.01 * ordinal,
                    "max_7d_rel_price_diff": pdiff,
                    "tw_real_slippage_1pct": slippage,
                    "tw_real_slippage_5pct": slippage,
                    "tw_real_slippage_10pct": slippage,
                    "apy_net_gm": 0.04 + 0.01 * ordinal,
                },
            )
        )
    dataset = HeatmapDataset.from_table(EvaluationTable(rows=rows))

    mask = MaskSpec(max_price_diff_bps=100, slippage_thr_bps=20)
    masked = dataset.metric_array("apy_1_masked", mask)
    raw = dataset.metric_array("apy_net")
    pdiff = dataset.metric_array("max_7d_rel_price_diff")
    slippage = dataset.metric_array("tw_real_slippage_1pct")
    expected = np.isfinite(raw) & (pdiff <= 0.01) & np.isfinite(slippage) & (slippage <= 0.002)
    assert np.array_equal(np.isfinite(masked), expected)

    # slippage-masked names are price-diff masked only (no self-window).
    sl_masked = dataset.metric_array("tw_real_slippage_1pct_masked", mask)
    assert np.array_equal(
        np.isfinite(sl_masked),
        np.isfinite(slippage) & (pdiff <= 0.01),
    )

    # No threshold set: no mask applied.
    assert np.array_equal(
        np.isfinite(dataset.metric_array("apy_1_masked", MaskSpec())),
        np.isfinite(raw),
    )

    with pytest.raises(HeatmapValidationError, match="unknown heatmap metric"):
        dataset.metric_array("apy_99_masked", MaskSpec())


def test_default_metric_is_first_numeric_metric() -> None:
    rows = []
    for ordinal, (x, y) in enumerate(itertools.product(("1", "2"), ("1", "2"))):
        rows.append(
            EvaluationRow(
                candidate_id=f"cand_{ordinal}",
                ordinal=ordinal,
                coordinates={"x": x, "y": y},
                metrics={"a_note": "text", "apy": 0.1 + 0.01 * ordinal},
            )
        )
    dataset = HeatmapDataset.from_table(EvaluationTable(rows=rows))
    assert dataset.metrics["a_note"].shape == (2, 2)
    state = HeatmapState.default(dataset)
    assert state.metric == "apy"


def test_render_heatmap_slider_state_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "heatmap.png"
    image, state = render_heatmap(
        _three_axis_table(with_metadata=True),
        output,
        metric="apy",
        source="evaluation_table.npz",
    )
    assert image.exists()
    payload = json.loads(state.read_text())
    assert payload["schema_version"] == "fxsim_heatmap_state_v1"
    assert payload["data"] == {
        "source": "evaluation_table.npz",
        "shape": [3, 3, 2],
        "axis_keys": ["kappa", "weight", "mode"],
    }
    assert payload["metric"] == "apy"
    assert payload["x_axis"] == "kappa"
    assert payload["y_axis"] == "weight"
    assert payload["slider_indices"] == {"mode": 0}
    assert payload["slider_coordinates"] == {"mode": "a"}
    assert payload["singleton_axes"] == []
    assert payload["mask"] == {
        "max_price_diff_bps": None,
        "max_skew_percent": None,
        "max_final_price_diff_bps": None,
    }
    assert [axis["key"] for axis in payload["axes"]] == ["kappa", "weight", "mode"]
    assert payload["axes"][0]["scale"] == "log"


_CORE = {
    "schema_version": "curve_fx_sim_identity_v2",
    "binary": "arb_evaluator_ld",
    "sha256": "a" * 64,
    "harness_version": "1.0.0",
    "pool_version": "0.1.0",
    "policy_id": "policy_v1",
    "policy_source_sha256": "b" * 64,
    "policy_abi": "twocrypto_policy_v1",
    "policy_parameter_count": 1,
    "numeric_mode": "double",
    "real_type": "double",
    "compiler": "clang++",
    "build_target": "arb_evaluator_ld",
    "metric_schema": "twocrypto-summary-v1",
    "metric_fields": ["apy"],
}


def test_cli_plot_heatmap_renders_npz_slice(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Guard against a display-capable backend blocking on show().
    monkeypatch.setattr(
        "curve_fx_sim.plotting.heatmap.interactive_backend_active", lambda: False
    )
    run_dir = tmp_path / "heatmap_run"
    run_dir.mkdir()
    table = _three_axis_table(with_metadata=True)
    table_path = table.to_npz(run_dir / "evaluation_table.npz")
    table_ref = {
        "path": "evaluation_table.npz",
        "sha256": sha256_path(table_path),
        "bytes": table_path.stat().st_size,
        "row_count": len(table.rows),
    }
    manifest = new_grid_manifest(
        run_id="heatmap_run",
        grid_id="test-grid",
        pool_count=len(table.rows),
        resolved_spec={},
        resolved_axes=[
            {"name": "kappa", "values": ["0.5", "1.0", "2.0"]},
            {"name": "weight", "values": ["3", "2", "1"]},
            {"name": "mode", "values": ["a", "b"]},
        ],
        pools=[
            {
                "id": row.candidate_id,
                "ordinal": row.ordinal,
                "coordinates": row.coordinates,
                "policy_params": [],
                "pool_overrides": {},
            }
            for row in table.rows
        ],
        shards=[],
        core=_CORE,
        table_ref=table_ref,
    )
    write_manifest_atomic(run_dir / "manifest.json", manifest, expected_kind="grid")

    result = CliRunner().invoke(
        main, ["plot", "heatmap", str(run_dir), "--metric", "apy"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["metric"] == "apy"
    assert data["results_source"].endswith("evaluation_table.npz")
    image = Path(data["output"])
    state = Path(data["state"])
    assert image.exists()
    payload = json.loads(state.read_text())
    assert payload["data"]["source"] == "evaluation_table.npz"
    assert payload["data"]["shape"] == [3, 3, 2]
