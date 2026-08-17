"""Local maxima and weighted multi-metric ranked maxima over evaluation-table grids."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from curve_fx_sim.analysis.maxima import (
    MaximaError,
    find_local_maxima,
    find_local_maxima_candidates,
    rank_values,
    ranked_maxima,
)
from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.manifest import new_grid_manifest, write_manifest_atomic
from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable
from curve_fx_sim.cli import main
from curve_fx_sim.grids.analysis import rank_evaluations
from curve_fx_sim.plotting.heatmap import HeatmapDataset

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
    "metric_fields": ["apy", "slip"],
}


def _two_axis_table() -> EvaluationTable:
    """3x3 coordinates-only grid; peak (1, 1); cell (2, 2) failed."""
    xs = ("0.5", "1.0", "2.0")
    ys = ("1", "2", "3")
    apy = {
        (0, 0): 1.0, (0, 1): 2.0, (0, 2): 1.0,
        (1, 0): 2.0, (1, 1): 5.0, (1, 2): 2.0,
        (2, 0): 1.0, (2, 1): 2.0, (2, 2): 1.0,
    }
    rows = []
    for ordinal, (xi, yi) in enumerate(itertools.product(range(3), range(3))):
        rows.append(
            EvaluationRow(
                candidate_id=f"cand_{ordinal:03d}",
                ordinal=ordinal,
                coordinates={"x": xs[xi], "y": ys[yi]},
                metrics={"apy": apy[(xi, yi)], "slip": 0.01 * (xi + 1)},
                status="failed" if (xi, yi) == (2, 2) else "ok",
            )
        )
    return EvaluationTable(rows=rows)


def _write_run_dir(tmp_path: Path, table: EvaluationTable, *, run_id: str) -> Path:
    """Write a manifest-attested NPZ run directory mirroring the grid flow."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    table_path = table.to_npz(run_dir / "evaluation_table.npz")
    table_ref = {
        "path": "evaluation_table.npz",
        "sha256": sha256_path(table_path),
        "bytes": table_path.stat().st_size,
        "row_count": len(table.rows),
    }
    manifest = new_grid_manifest(
        run_id=run_id,
        grid_id="test-grid",
        pool_count=len(table.rows),
        resolved_spec={},
        resolved_axes=[
            {"name": "x", "values": ["0.5", "1.0", "2.0"]},
            {"name": "y", "values": ["1", "2", "3"]},
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
    return run_dir


# ---------------------------------------------------------------------------
# find_local_maxima / find_local_maxima_candidates
# ---------------------------------------------------------------------------


def test_local_maxima_single_peak() -> None:
    grid = np.array([[1.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 1.0]])
    assert find_local_maxima(grid) == ((1, 1),)


def test_local_maxima_plateau_collapses_to_one_representative() -> None:
    grid = np.array([[1.0, 2.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    # Tied plateau cells merge into one component; row-major first wins.
    assert find_local_maxima(grid) == ((0, 1),)
    # Without plateau dedup every plateau cell is reported.
    assert find_local_maxima_candidates(grid) == ((0, 1), (1, 0), (1, 1))


def test_local_maxima_peak_adjacent_to_masked_cell_is_found() -> None:
    grid = np.array([[1.0, 3.0, 1.0], [2.0, np.nan, 1.0], [1.0, 2.0, 1.0]])
    # A masked cell is absent, not evidence against its finite neighbors.
    assert find_local_maxima(grid, connectivity="full") == ((0, 1), (2, 1))
    assert find_local_maxima(grid, connectivity="axis") == (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    )
    assert find_local_maxima_candidates(grid, connectivity="axis") == (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    )


def test_local_maxima_all_masked_is_empty() -> None:
    grid = np.full((3, 3), np.nan)
    assert find_local_maxima(grid) == ()


def test_local_maxima_axes_restriction() -> None:
    grid = np.array([[3.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    # Column-wise maxima (ties along the active axis stay distinct).
    assert find_local_maxima(grid, axes=(0,)) == ((0, 0), (2, 0), (2, 1))
    # Row-wise maxima.
    assert find_local_maxima(grid, axes=(1,)) == ((0, 0), (1, 1), (2, 1))
    # Full 2-D neighborhood: diagonal neighbors suppress the plateaus.
    assert find_local_maxima(grid) == ((0, 0), (2, 1))


def test_local_maxima_connectivity_axis_vs_full() -> None:
    grid = np.array([[2.0, 4.0], [4.0, 1.0]])
    # Diagonal ties are one plateau under full connectivity, two maxima under axis.
    assert find_local_maxima(grid, connectivity="full") == ((0, 1),)
    assert find_local_maxima(grid, connectivity="axis") == ((0, 1), (1, 0))


def test_local_maxima_rejects_invalid_input() -> None:
    with pytest.raises(MaximaError, match="grid metric array"):
        find_local_maxima(np.asarray(1.0))
    with pytest.raises(MaximaError, match="connectivity"):
        find_local_maxima(np.zeros((2, 2)), connectivity="diagonal")


# ---------------------------------------------------------------------------
# rank_values
# ---------------------------------------------------------------------------


def test_rank_values_descending_with_ties() -> None:
    assert rank_values([3.0, 1.0, 2.0], descending=True) == [1, 3, 2]
    # Ties keep input order and consume consecutive ranks (not dense).
    assert rank_values([5.0, 5.0, 3.0], descending=True) == [1, 2, 3]


def test_rank_values_non_finite_ranks_worst() -> None:
    assert rank_values([5.0, np.nan, 3.0], descending=True) == [1, 3, 2]


def test_rank_values_threshold_good_cells_rank_zero() -> None:
    assert rank_values([5.0, 3.0, 1.0], descending=True, good_threshold=4.0) == [0, 1, 2]
    assert rank_values([5.0, 3.0, 1.0], descending=False, good_threshold=2.0, good_when_low=True) == [2, 1, 0]


# ---------------------------------------------------------------------------
# ranked_maxima
# ---------------------------------------------------------------------------


def test_ranked_maxima_weighted_ordering() -> None:
    dataset = HeatmapDataset.from_table(_two_axis_table())
    ranked = ranked_maxima(
        dataset,
        descending=("apy",),
        ascending=("slip",),
    )
    assert [item.rank for item in ranked] == list(range(1, 10))
    first = ranked[0]
    assert first.grid_indices == (0, 1)
    assert first.candidate_id == "cand_001"
    assert first.ordinal == 1
    assert first.coordinates == {"x": "0.5", "y": "2"}
    assert first.weighted_score == pytest.approx(4.0)
    assert first.metric_ranks == {"apy": 2, "slip": 2}
    # The failed cell (2, 2) ranks worst on both metrics.
    assert ranked[-1].grid_indices == (2, 2)
    assert ranked[-1].weighted_score == pytest.approx(18.0)
    assert ranked[-1].metric_ranks == {"apy": 9, "slip": 9}


def test_ranked_maxima_weights_and_top() -> None:
    dataset = HeatmapDataset.from_table(_two_axis_table())
    ranked = ranked_maxima(
        dataset,
        descending=("apy",),
        ascending=("slip",),
        weights={"apy": 2.0},
        top=3,
    )
    assert len(ranked) == 3
    assert [item.weighted_score for item in ranked] == pytest.approx([6.0, 7.0, 10.0])


def test_ranked_maxima_threshold_good_cells() -> None:
    dataset = HeatmapDataset.from_table(_two_axis_table())
    ranked = ranked_maxima(
        dataset,
        descending=("apy",),
        ascending=("slip",),
        thresholds={"apy": 2.0},
    )
    # (0,0) and (0,1) tie at score 2; row-major order decides.
    assert ranked[0].grid_indices == (0, 0)
    assert ranked[0].weighted_score == pytest.approx(2.0)
    assert ranked[1].grid_indices == (0, 1)
    assert ranked[1].metric_ranks["apy"] == 0
    assert ranked[1].weighted_score == pytest.approx(2.0)


def test_ranked_maxima_requires_a_metric() -> None:
    dataset = HeatmapDataset.from_table(_two_axis_table())
    with pytest.raises(MaximaError, match="at least one metric"):
        ranked_maxima(dataset)


def test_ranked_maxima_rejects_unknown_metric() -> None:
    dataset = HeatmapDataset.from_table(_two_axis_table())
    with pytest.raises(MaximaError, match="unknown grid metric"):
        ranked_maxima(dataset, descending=("missing",))


# ---------------------------------------------------------------------------
# NPZ provenance: analyze rank and maxima keep working on from_npz tables
# ---------------------------------------------------------------------------


def test_rank_evaluations_works_on_npz_table(tmp_path: Path) -> None:
    table = _two_axis_table()
    table_path = table.to_npz(tmp_path / "roundtrip.npz")
    loaded = EvaluationTable.from_npz(table_path)
    ranked = rank_evaluations(loaded, descending=("apy",), top=3)
    assert len(ranked) == 3
    assert ranked[0].row.candidate_id == "cand_004"
    assert ranked[0].metric_ranks["apy"] == 0
    assert ranked[0].weighted_rank == 0


def test_maxima_work_on_npz_table(tmp_path: Path) -> None:
    table = _two_axis_table()
    table_path = table.to_npz(tmp_path / "roundtrip.npz")
    loaded = EvaluationTable.from_npz(table_path)
    dataset = HeatmapDataset.from_table(loaded)
    assert find_local_maxima(dataset.metric_array("apy")) == ((1, 1),)
    ranked = ranked_maxima(dataset, descending=("apy",), top=1)
    assert ranked[0].grid_indices == (1, 1)
    assert ranked[0].metrics["apy"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# fxsim analyze maxima CLI
# ---------------------------------------------------------------------------


def test_cli_maxima_local(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, _two_axis_table(), run_id="maxima_local")
    result = CliRunner().invoke(
        main,
        ["analyze", "maxima", str(run_dir), "--local", "--metric", "apy"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["mode"] == "local"
    assert data["local_maxima_count"] == 1
    point = data["local_maxima"][0]
    assert point["grid_indices"] == [1, 1]
    assert point["candidate_id"] == "cand_004"
    assert point["coordinates"] == {"x": "1.0", "y": "2"}
    assert point["metric"] == 5.0


def test_cli_maxima_local_axis_restriction(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, _two_axis_table(), run_id="maxima_axis")
    result = CliRunner().invoke(
        main,
        [
            "analyze", "maxima", str(run_dir),
            "--local", "--metric", "apy",
            "--axis", "x", "--connectivity", "axis",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["local_maxima_count"] == 3
    assert data["axes"] == ["x"]


def test_cli_maxima_ranked(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, _two_axis_table(), run_id="maxima_ranked")
    result = CliRunner().invoke(
        main,
        [
            "analyze", "maxima", str(run_dir),
            "--desc-metrics", "apy",
            "--asc-metrics", "slip",
            "--weights", "apy=2",
            "--top", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["mode"] == "ranked"
    assert data["ranked_count"] == 3
    assert data["weights"] == {"apy": 2.0, "slip": 1.0}
    first = data["ranked"][0]
    assert first["coordinates"] == {"x": "0.5", "y": "2"}
    assert first["metrics"]["slip"] == pytest.approx(0.01)
    assert first["weighted_score"] == pytest.approx(6.0)


def test_cli_maxima_default_ranked_mode(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, _two_axis_table(), run_id="maxima_default")
    result = CliRunner().invoke(main, ["analyze", "maxima", str(run_dir)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["mode"] == "ranked"
    assert data["descending"] == ["apy"]
    assert data["ranked_count"] == 9
    first = data["ranked"][0]
    assert first["rank"] == 1
    assert first["grid_indices"] == [1, 1]
    assert first["coordinates"] == {"x": "1.0", "y": "2"}
    assert first["metrics"]["apy"] == pytest.approx(5.0)
    assert first["metric_ranks"]["apy"] == 1
