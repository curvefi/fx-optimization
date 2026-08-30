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
            "apy": (np.arange(8, dtype=float).reshape(shape) + 10) / 100,
            "apy_net": np.arange(8, dtype=float).reshape(shape) / 100,
            "apy_net_robust_90d": (
                np.arange(8, dtype=float).reshape(shape) + 20
            ) / 100,
            "detach_energy_ungated": np.arange(8, dtype=float).reshape(shape),
            "max_7d_rel_price_diff": (
                np.arange(1, 9, dtype=float).reshape(shape) / 1_000
            ),
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


def test_dimension_and_axis_sliders_rescale_visible_slice() -> None:
    explorer = HeatmapExplorer(_dataset(), metrics=["apy_net"], ncol=1)
    try:
        assert not hasattr(explorer, "fig_metrics")
        assert all(not label.get_visible() for label in explorer.x_radio.labels)
        assert all(not label.get_visible() for label in explorer.y_radio.labels)
        x_position = explorer.x_radio.ax.get_position()
        y_position = explorer.y_radio.ax.get_position()
        assert x_position.y0 == y_position.y0
        assert x_position.x0 < y_position.x0
        shared_labels = {
            text.get_text(): text
            for text in explorer.fig_controls.texts
            if text.get_text() in {"x", "y", "scenario"}
        }
        assert set(shared_labels) == {"x", "y", "scenario"}
        expected_rows = np.linspace(1.0, 0.0, 5)[1:-1]
        assert np.allclose(
            [shared_labels[name].get_position()[1] for name in ("x", "y", "scenario")],
            x_position.y0 + x_position.height * expected_rows,
        )
        assert explorer.meshes[0].get_clim() == (0.0, 6.0)
        assert dict(explorer.sliders)["scenario"].val == 0
        dict(explorer.sliders)["scenario"].set_val(1)
        assert explorer.state.slider_indices == {"scenario": 1}
        assert np.allclose(explorer.meshes[0].get_clim(), (1.0, 7.0))

        explorer._swap_axes("x", "scenario")
        assert explorer.state.x_axis == "scenario"
        assert explorer.state.y_axis == "y"
        assert {key for key, _ in explorer.sliders} == {"x"}
    finally:
        explorer.close()


def test_controls_size_survives_axis_rebuild() -> None:
    explorer = HeatmapExplorer(
        _dataset(), metrics=["apy_net_robust_90d_masked"],
        ncol=1, max_pricethr=40,
    )
    try:
        initial_size = tuple(explorer.fig_controls.get_size_inches())
        assert initial_size[0] == 3.2 and initial_size[1] >= 6.0
        row_labels = {
            text.get_text(): text.get_position()[1]
            for text in explorer.fig_controls.texts
            if text.get_text() in {
                "scenario:", "max 7d pdiff thr (bps):", "detach energy max:"
            }
        }
        assert row_labels["scenario:"] - row_labels["max 7d pdiff thr (bps):"] >= 0.09
        assert (
            row_labels["max 7d pdiff thr (bps):"]
            - row_labels["detach energy max:"]
            >= 0.09
        )
        assert explorer.max_price_thr_slider.val == 40
        assert explorer.max_price_thr_slider.valmax == 80
        user_size = (8.5, 4.25)
        explorer.fig_controls.set_size_inches(*user_size)
        explorer.x_radio.set_active(explorer._radio_keys.index("scenario"))
        assert tuple(explorer.fig_controls.get_size_inches()) == user_size
    finally:
        explorer.close()


def test_main_size_survives_axis_rebuild() -> None:
    explorer = HeatmapExplorer(_dataset(), metrics=["apy_net"], ncol=1)
    try:
        user_size = (8.5, 4.25)
        explorer.fig_main.set_size_inches(*user_size)
        explorer.x_radio.set_active(explorer._radio_keys.index("scenario"))
        assert tuple(explorer.fig_main.get_size_inches()) == user_size
    finally:
        explorer.close()


def test_left_shift_and_right_clicks_forward_exact_selection_ref(capsys) -> None:
    callbacks: list[tuple[object, str]] = []
    explorer = HeatmapExplorer(
        _dataset(),
        metrics=["apy_net", "max_7d_rel_price_diff", "tw_real_slippage_1pct"],
        run_id="grid-run",
        on_replay=lambda selection, mode: callbacks.append((selection, mode)),
    )
    try:
        explorer._on_click(_event(explorer))
        assert explorer.last_selection is not None
        assert callbacks == []
        output = capsys.readouterr().out
        assert "candidate=candidate-0 ordinal=0" in output
        assert "x=1, y=10, scenario=base" in output
        assert "apy_net=0%" in output
        assert "max_7d_rel_price_diff=0.1%" in output
        assert "tw_real_slippage_1pct=0.1%" in output

        explorer._on_click(_event(explorer, key="shift"))
        explorer._on_click(_event(explorer, button=3))
        assert capsys.readouterr().out == ""
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
    assert payload["explorer"]["window_titles"] == ["Heatmaps", "Controls"]
    assert payload["explorer"]["selection"]["candidate_id"] == "candidate-0"
    assert payload["slider_indices"] == {"scenario": 0}
