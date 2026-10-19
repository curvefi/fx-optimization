"""Dimension slider behavior of the heatmap explorer: 1-based index label and colorbar limits."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from curve_fx_sim.plotting.explorer import HeatmapExplorer  # noqa: E402
from curve_fx_sim.plotting.heatmap import HeatmapAxis, HeatmapDataset  # noqa: E402


def _lattice_dataset() -> HeatmapDataset:
    """2x3x4 grid where m = 12*A + 4*B + C and the price diff in bps equals m."""
    axes = (
        HeatmapAxis(("A",), (1.0, 2.0)),
        HeatmapAxis(("B",), (10.0, 20.0, 30.0)),
        HeatmapAxis(("C",), (0.25, 0.5, 0.75, 1.0)),
    )
    shape = tuple(len(axis.values) for axis in axes)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return HeatmapDataset(
        axes=axes,
        metrics={"m": values, "max_7d_rel_price_diff": values / 10_000.0},
        valid=np.ones(shape, dtype=bool),
    )


def _other_index(start: int) -> int:
    return 3 if start != 3 else 0


def test_colorbar_follows_slice_by_default() -> None:
    explorer = HeatmapExplorer(
        _lattice_dataset(), metrics=("m",), x_axis="A", y_axis="B", max_pricethr=None,
    )
    try:
        start = explorer.state.slider_indices["C"]
        assert explorer.meshes[0].get_clim() == (start, 20.0 + start)
        target = _other_index(start)
        dict(explorer.sliders)["C"].set_val(target)
        assert explorer.meshes[0].get_clim() == (target, 20.0 + target)
    finally:
        plt.close("all")


def test_frozen_colorbar_spans_all_slices_and_tracks_mask_thresholds() -> None:
    explorer = HeatmapExplorer(
        _lattice_dataset(), metrics=("m_masked",), x_axis="A", y_axis="B",
        max_pricethr=100.0, freeze_colorbar=True,
    )
    try:
        assert explorer.meshes[0].get_clim() == (0.0, 23.0)
        start = explorer.state.slider_indices["C"]
        dict(explorer.sliders)["C"].set_val(_other_index(start))
        assert explorer.meshes[0].get_clim() == (0.0, 23.0)

        explorer._threshold_sliders["max_pricethr"].set_val(10.0)
        assert explorer.meshes[0].get_clim() == (0.0, 10.0)
        dict(explorer.sliders)["C"].set_val(start)
        assert explorer.meshes[0].get_clim() == (0.0, 10.0)
    finally:
        plt.close("all")


def test_dimension_slider_index_is_one_based_and_separate_from_value_label() -> None:
    axes = (
        HeatmapAxis(("A",), (1.0, 2.0)),
        HeatmapAxis(("B",), (10.0, 20.0, 30.0)),
        HeatmapAxis(("C",), (0.25, 0.5, 0.75, 1.0)),
    )
    shape = tuple(len(axis.values) for axis in axes)
    dataset = HeatmapDataset(
        axes=axes,
        metrics={"m": np.arange(np.prod(shape), dtype=float).reshape(shape)},
        valid=np.ones(shape, dtype=bool),
    )
    explorer = HeatmapExplorer(dataset, x_axis="A", y_axis="B", max_pricethr=None)
    try:
        index_text = explorer._slider_index_texts["C"]
        value_text = explorer._slider_value_texts["C"]
        start = explorer.state.slider_indices["C"]
        assert index_text.get_text() == f"{start + 1}/4"
        assert value_text.get_text() == explorer._axis_views["C"].labels[start]

        slider = dict(explorer.sliders)["C"]
        slider.set_val(3)
        assert explorer.state.slider_indices["C"] == 3
        assert index_text.get_text() == "4/4"
        assert value_text.get_text() == explorer._axis_views["C"].labels[3]

        # The index sits left of the bar, so it never overlaps the handle or the value label.
        index_box = index_text.get_window_extent()
        assert index_box.x1 <= slider.ax.get_window_extent().x0
        assert not index_box.overlaps(value_text.get_window_extent())
    finally:
        plt.close("all")
