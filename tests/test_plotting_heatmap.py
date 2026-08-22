"""Direct checks for the active HeatmapDataset plotting seam."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from curve_fx_sim.plotting.heatmap import (
    HeatmapAxis,
    HeatmapDataset,
    HeatmapValidationError,
    MaskSpec,
    render_heatmap,
)


def _dataset() -> HeatmapDataset:
    shape = (2, 2, 2)
    return HeatmapDataset(
        axes=(
            HeatmapAxis(("x",), (1, 2)),
            HeatmapAxis(("y",), (10, 20)),
            HeatmapAxis(("scenario",), ("base", "stress"), scale="categorical"),
        ),
        metrics={
            "apy_net": np.arange(8, dtype=float).reshape(shape) / 100,
            "apy_net_gm": np.arange(8, dtype=float).reshape(shape) / 200,
            "max_7d_rel_price_diff": np.array(
                [[[0.001, 0.002], [0.003, 0.004]], [[0.005, 0.006], [0.007, 0.008]]]
            ),
            "max_7d_skew": np.array(
                [[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]]
            ),
            "final_rel_price_diff": np.array(
                [[[0.001, 0.002], [0.003, 0.004]], [[0.005, 0.006], [0.007, 0.008]]]
            ),
            "tw_real_slippage_1pct": np.array(
                [[[0.001, 0.002], [0.003, 0.004]], [[0.005, 0.006], [0.007, 0.008]]]
            ),
            "tw_real_slippage_5pct": np.full(shape, 0.001),
            "tw_real_slippage_10pct": np.full(shape, 0.001),
        },
        candidate_ids=np.asarray(
            [f"candidate-{index}" for index in range(8)], dtype=object
        ).reshape(shape),
        ordinals=np.arange(8, dtype=np.int64).reshape(shape),
        valid=np.ones(shape, dtype=bool),
    )


def test_dataset_masks_filter_price_skew_final_diff_and_slippage() -> None:
    dataset = _dataset()
    mask = MaskSpec(
        max_price_diff_bps=40,
        max_skew_percent=5,
        max_final_price_diff_bps=40,
        slippage_thr_bps=20,
    )

    apy = dataset.metric_array("apy_1_masked", mask)
    expected = np.array([[[True, True], [False, False]], [[False, False], [False, False]]])
    assert np.array_equal(np.isfinite(apy), expected)
    assert np.array_equal(
        np.isfinite(dataset.metric_array("tw_real_slippage_1pct_masked", mask)),
        np.array([[[True, True], [True, True]], [[False, False], [False, False]]]),
    )
    gm_expected = np.array([[[True, True], [True, True]], [[False, False], [False, False]]])
    assert np.array_equal(np.isfinite(dataset.metric_array("apy_gm_masked", mask)), gm_expected)
    with pytest.raises(HeatmapValidationError, match="unknown heatmap metric"):
        dataset.metric_array("missing", mask)


def test_render_heatmap_writes_png_and_complete_state(tmp_path: Path) -> None:
    image, state = render_heatmap(
        _dataset(), tmp_path / "heatmap.png", metric="apy_net", source="raw-evaluator.json"
    )

    assert image.is_file() and image.stat().st_size > 0
    payload = json.loads(state.read_text())
    assert payload["data"]["shape"] == [2, 2, 2]
    assert payload["data"]["axis_keys"] == ["x", "y", "scenario"]
    assert payload["slider_indices"] == {"scenario": 0}
    assert payload["slider_coordinates"] == {"scenario": "base"}
    assert payload["metric"] == "apy_net"
    assert payload["data"]["source"] == "raw-evaluator.json"
