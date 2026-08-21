"""Local maxima and weighted multi-metric ranked maxima over evaluation-table grids."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from curve_fx_sim.analysis.maxima import (
    find_local_maxima,
    find_local_maxima_candidates,
    ranked_maxima,
)
from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable
from curve_fx_sim.grids.analysis import rank_evaluations
from curve_fx_sim.plotting.heatmap import HeatmapDataset


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
def test_local_maxima_peak_plateau_and_masking() -> None:
    grid = np.array([[1.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 1.0]])
    assert find_local_maxima(grid) == ((1, 1),)

    plateau = np.array([[1.0, 2.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    assert find_local_maxima(plateau) == ((0, 1),)
    assert find_local_maxima_candidates(plateau) == ((0, 1), (1, 0), (1, 1))

    masked = np.array([[1.0, 3.0, 1.0], [2.0, np.nan, 1.0], [1.0, 2.0, 1.0]])
    assert find_local_maxima(masked, connectivity="full") == ((0, 1), (2, 1))
    assert find_local_maxima(masked, connectivity="axis") == (
        (0, 1), (1, 0), (1, 2), (2, 1)
    )
def test_ranked_maxima_orders_weights_and_thresholds() -> None:
    dataset = HeatmapDataset.from_table(_two_axis_table())
    ranked = ranked_maxima(dataset, descending=("apy",), ascending=("slip",))
    assert [item.rank for item in ranked] == list(range(1, 10))
    assert ranked[0].grid_indices == (0, 1)
    assert ranked[0].candidate_id == "cand_001"
    assert ranked[0].coordinates == {"x": "0.5", "y": "2"}
    assert ranked[0].weighted_score == pytest.approx(4.0)
    assert ranked[0].metric_ranks == {"apy": 2, "slip": 2}
    assert ranked[-1].grid_indices == (2, 2)
    assert ranked[-1].weighted_score == pytest.approx(18.0)

    weighted = ranked_maxima(
        dataset,
        descending=("apy",),
        ascending=("slip",),
        weights={"apy": 2.0},
        top=3,
    )
    assert [item.weighted_score for item in weighted] == pytest.approx([6.0, 7.0, 10.0])

    thresholded = ranked_maxima(
        dataset,
        descending=("apy",),
        ascending=("slip",),
        thresholds={"apy": 2.0},
    )
    assert thresholded[0].grid_indices == (0, 0)
    assert thresholded[0].weighted_score == pytest.approx(2.0)
    assert thresholded[1].grid_indices == (0, 1)
    assert thresholded[1].metric_ranks["apy"] == 0


def test_maxima_and_ranking_work_on_npz_table(tmp_path: Path) -> None:
    table_path = _two_axis_table().to_npz(tmp_path / "roundtrip.npz")
    loaded = EvaluationTable.from_npz(table_path)

    ranked = rank_evaluations(loaded, descending=("apy",), top=3)
    assert len(ranked) == 3
    assert ranked[0].row.candidate_id == "cand_004"
    assert ranked[0].metric_ranks["apy"] == 0

    dataset = HeatmapDataset.from_table(loaded)
    assert find_local_maxima(dataset.metric_array("apy")) == ((1, 1),)
    top = ranked_maxima(dataset, descending=("apy",), top=1)
    assert top[0].grid_indices == (1, 1)
    assert top[0].metrics["apy"] == pytest.approx(5.0)
