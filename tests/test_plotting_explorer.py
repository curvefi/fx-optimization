"""Focused functional checks for the maintained N-D explorer."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
import pytest
matplotlib.use("Agg")

from matplotlib.backend_bases import MouseEvent
from matplotlib.ticker import FixedLocator, FormatStrFormatter

from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable
from curve_fx_sim.plotting.explorer import (
    HeatmapExplorer,
    _select_ticks,
)


def _table() -> EvaluationTable:
    rows = []
    for ordinal, (x, y) in enumerate(itertools.product(("1", "2"), ("1", "2"))):
        rows.append(
            EvaluationRow(
                candidate_id=f"candidate-{ordinal}",
                ordinal=ordinal,
                coordinates={"x": x, "y": y},
                metrics={
                    "apy_net": 0.01 + ordinal * 0.01,
                    "max_7d_rel_price_diff": 0.001 + ordinal * 0.001,
                    "max_7d_skew": 0.5 + ordinal * 0.01,
                    "final_rel_price_diff": 0.001 + ordinal * 0.001,
                    "tw_real_slippage_1pct": 0.001 + ordinal * 0.001,
                },
            )
        )
    return EvaluationTable(rows)


def _three_dimensional_table() -> EvaluationTable:
    axes = (
        (
            "A",
            ("1", "2", "4"),
            [{"path": ["A"], "scale": "10000", "display_scale": "1", "kind": "integer"}],
        ),
        (
            "fee_bps",
            ("10", "20", "40"),
            [
                {
                    "path": ["mid_fee"],
                    "scale": "1000000",
                    "display_scale": "1",
                    "kind": "decimal",
                }
            ],
        ),
        ("donation_apy", ("0.01", "0.02"), []),
    )
    rows = []
    locations = {}
    for ordinal, location in enumerate(itertools.product(range(3), range(3), range(2))):
        ai, fi, di = location
        candidate_id = f"candidate-{ordinal}"
        rows.append(
            EvaluationRow(
                candidate_id=candidate_id,
                ordinal=ordinal,
                coordinates={
                    "A": axes[0][1][ai],
                    "fee_bps": axes[1][1][fi],
                    "donation_apy": axes[2][1][di],
                },
                metrics={
                    "apy_net": 0.01 * (ai + 2 * fi + 10 * di),
                    "max_7d_rel_price_diff": 0.001 * (1 + ai + fi + di),
                },
            )
        )
        locations[candidate_id] = list(location)
    metadata = {
        "axes": [
            {
                "name": name,
                "values": list(values),
                "generation": {},
                "targets": targets,
            }
            for name, values, targets in axes
        ],
        "coordinate_indices": locations,
    }
    return EvaluationTable(rows, metadata=metadata)

def test_legacy_visual_contract_and_global_clims_survive_dimension_slider() -> None:
    explorer = HeatmapExplorer(
        _three_dimensional_table(),
        metrics=["apy_net", "max_7d_rel_price_diff"],
        ncol=2,
    )
    try:
        assert explorer.fig_main.get_size_inches() == pytest.approx((22.0, 11.0))
        assert explorer.fig_controls.get_size_inches()[0] == pytest.approx(3.2)
        assert explorer.fig_metrics.get_size_inches() == pytest.approx((6.5, 9.0))
        assert explorer.fig_metrics.texts[0].get_fontweight() == "bold"
        assert explorer.fig_main.texts == []  # format_coord feeds the toolbar without a footer.

        first, second = explorer.axes
        assert first.get_box_aspect() == pytest.approx(1.0)
        assert first.get_xscale() == first.get_yscale() == "log"
        assert first.get_xlabel() == "A (÷1e4)"
        assert first.get_ylabel() == "mid_fee (bps)"
        assert second.get_ylabel() == ""
        assert [label.get_text() for label in first.get_xticklabels()] == ["1.00", "2.00", "4.00"]
        assert [label.get_text() for label in first.get_yticklabels()] == ["10.00", "20.00", "40.00"]
        assert isinstance(first.xaxis.get_major_locator(), FixedLocator)
        assert isinstance(explorer.colorbars[0].formatter, FormatStrFormatter)
        assert len(_select_ticks(64)) == 12
        assert (_select_ticks(64)[0], _select_ticks(64)[-1]) == (0, 63)

        event = MouseEvent("button_press_event", explorer.fig_main.canvas, 0, 0, button=1)
        event.inaxes = first
        event.xdata = 10_000.0
        event.ydata = 10_000_000.0
        selection = explorer._selection_from_event(event)
        assert selection is not None
        assert selection.candidate_id == "candidate-0"

        clims = [mesh.get_clim() for mesh in explorer.meshes]
        tick_locations = tuple(first.xaxis.get_majorticklocs())
        dict(explorer.sliders)["donation_apy"].set_val(1)
        assert [mesh.get_clim() for mesh in explorer.meshes] == clims
        assert tuple(explorer.axes[0].xaxis.get_majorticklocs()) == tick_locations
        assert explorer.axes[0].get_box_aspect() == pytest.approx(1.0)
    finally:
        explorer.close()

def test_plain_and_shift_click_keep_exact_selection_ref() -> None:
    selections = []
    explorer = HeatmapExplorer(
        _table(),
        metrics=["apy_net"],
        run_id="grid-run",
        on_replay=lambda selection, mode: selections.append((selection, mode)),
    )
    try:
        axis = explorer.axes[0]
        event = MouseEvent("button_press_event", explorer.fig_main.canvas, 0, 0, button=1)
        event.inaxes = axis
        event.xdata = 1.0
        event.ydata = 1.0
        explorer._on_click(event)
        assert explorer.last_selection is not None
        assert selections == []
        event.key = "shift"
        explorer._on_click(event)
        assert len(selections) == 1
        selection, mode = selections[0]
        assert mode == "shift"
        assert selection.run_id == "grid-run"
        assert selection.candidate_id == explorer.last_selection.candidate_id
        assert selection.index == explorer.last_selection.ordinal
        assert selection.coordinate == dict(explorer.last_selection.coordinates)
        event.key = None
        event.button = 3
        explorer._on_click(event)
        assert len(selections) == 2
        right_selection, mode = selections[1]
        assert mode == "right"
        assert right_selection == selection
    finally:
        explorer.close()
def test_source_rows_keep_full_identity_despite_duplicate_candidate_and_ordinal() -> None:
    rows = [
        EvaluationRow("same", ordinal=7, coordinates={"x": 1, "y": 1}, metrics={"apy_net": 1}),
        EvaluationRow("same", ordinal=7, coordinates={"x": 2, "y": 1}, metrics={"apy_net": 2}),
    ]
    explorer = HeatmapExplorer(EvaluationTable(rows), metrics=["apy_net"])
    try:
        assert explorer._source_row is not None
        assert len(explorer._source_row) == 2
        assert all(any(value is row for value in explorer._source_row.values()) for row in rows)
    finally:
        explorer.close()

def test_export_is_immutable_and_sidecar_has_window_state(tmp_path: Path) -> None:
    explorer = HeatmapExplorer(_table(), metrics=["apy_net"], run_id="grid-run")
    try:
        _image, sidecar = explorer.save(tmp_path / "view.png")
    finally:
        explorer.close()
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["explorer"]["window_titles"] == ["Heatmaps", "Controls", "Metrics"]
    with pytest.raises(FileExistsError):
        explorer.save(_image)
