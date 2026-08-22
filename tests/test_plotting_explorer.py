"""Focused functional checks for the maintained N-D explorer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backend_bases import MouseEvent

from curve_fx_sim.plotting.explorer import HeatmapExplorer
from curve_fx_sim.plotting.heatmap import HeatmapAxis, HeatmapDataset


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
            "max_7d_rel_price_diff": np.full(shape, 0.001),
            "tw_real_slippage_1pct": np.full(shape, 0.001),
        },
        candidate_ids=np.asarray(
            [f"candidate-{index}" for index in range(8)], dtype=object
        ).reshape(shape),
        ordinals=np.arange(8, dtype=np.int64).reshape(shape),
        valid=np.ones(shape, dtype=bool),
    )


def _event(explorer: HeatmapExplorer, *, button: int = 1, key: str | None = None) -> MouseEvent:
    event = MouseEvent("button_press_event", explorer.fig_main.canvas, 0, 0, button=button)
    event.inaxes = explorer.axes[0]
    event.xdata = 1.0
    event.ydata = 10.0
    event.key = key
    return event


def test_dimension_and_axis_sliders_keep_global_color_limits() -> None:
    explorer = HeatmapExplorer(_dataset(), metrics=["apy_net"], ncol=1)
    try:
        clims = [mesh.get_clim() for mesh in explorer.meshes]
        assert dict(explorer.sliders)["scenario"].val == 0
        dict(explorer.sliders)["scenario"].set_val(1)
        assert explorer.state.slider_indices == {"scenario": 1}
        assert [mesh.get_clim() for mesh in explorer.meshes] == clims

        explorer._swap_axes("x", "scenario")
        assert explorer.state.x_axis == "scenario"
        assert explorer.state.y_axis == "y"
        assert {key for key, _ in explorer.sliders} == {"x"}
    finally:
        explorer.close()


def test_left_shift_and_right_clicks_forward_exact_selection_ref() -> None:
    callbacks: list[tuple[object, str]] = []
    explorer = HeatmapExplorer(
        _dataset(), metrics=["apy_net"], run_id="grid-run",
        on_replay=lambda selection, mode: callbacks.append((selection, mode)),
    )
    try:
        explorer._on_click(_event(explorer))
        assert explorer.last_selection is not None
        assert callbacks == []

        explorer._on_click(_event(explorer, key="shift"))
        explorer._on_click(_event(explorer, button=3))
        assert [mode for _, mode in callbacks] == ["shift", "right"]
        assert callbacks[0][0] == callbacks[1][0]
        selection = callbacks[0][0]
        assert selection.run_id == "grid-run"
        assert selection.candidate_id == "candidate-0"
        assert selection.index == 0
        assert selection.coordinate == {"x": 1, "y": 10, "scenario": "base"}
    finally:
        explorer.close()


def test_save_exports_png_and_state_with_selection(tmp_path: Path) -> None:
    explorer = HeatmapExplorer(_dataset(), metrics=["apy_net"], run_id="grid-run")
    try:
        explorer._on_click(_event(explorer))
        image, state = explorer.save(tmp_path / "explorer.png")
    finally:
        explorer.close()

    assert image.is_file() and image.stat().st_size > 0
    payload = json.loads(state.read_text())
    assert payload["explorer"]["window_titles"] == ["Heatmaps", "Controls", "Metrics"]
    assert payload["explorer"]["selection"]["candidate_id"] == "candidate-0"
    assert payload["slider_indices"] == {"scenario": 0}
