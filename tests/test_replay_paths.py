"""A run's replay inputs, recorded on another machine or checkout, resolve to the same files in this workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from fxopt import shiftclick


def test_replay_inputs_move_onto_this_checkout() -> None:
    repo = Path(shiftclick.__file__).resolve().parents[2]
    if repo.name not in shiftclick._REPOSITORIES:
        pytest.skip("this checkout's directory is not named after its repository")
    elsewhere = Path("/elsewhere/workspace") / repo.name
    template = Path("configs") / "templates" / "ethusd-yb-weth-live-20250324.json"
    metadata = {
        "replay": {
            "evaluator": str(elsewhere / "pyproject.toml"),
            "work_dir": str(elsewhere / "configs" / "autoresearch"),
            "open_session": {
                "template_path": str(elsewhere / template),
                "price_feed_path": "/elsewhere/missing.npz",
                "event_cursor": "scalar",
            },
        },
        shiftclick.EVALUATOR_POLICY_METADATA_KEY: {
            "policy_id": "report_lean", "policy_abi": "abi", "policy_parameter_count": 9,
        },
    }
    spec = shiftclick._replay_from_metadata("run", metadata)
    assert spec.evaluator == repo / "pyproject.toml"
    assert spec.work_dir == repo / "configs" / "autoresearch"
    assert spec.open_session == {
        "template_path": str(repo / template),
        "price_feed_path": "/elsewhere/missing.npz",  # no such file in this workspace either: kept as stored
        "event_cursor": "scalar",
    }
