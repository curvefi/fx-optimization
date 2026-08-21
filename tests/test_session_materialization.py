"""Focused coverage for verified local session materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

from curve_fx_sim.evaluation.session import LocalSessionMaterialization
from curve_fx_sim.specs.scenario import MarketFileRef, ScenarioSpec


def _scenario(root: Path) -> ScenarioSpec:
    template = root / "template.json"
    market = root / "market.json"
    template.write_bytes(b'{"pool":"fixture"}\n')
    market.write_bytes(b"[]\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return ScenarioSpec(
        id="scenario",
        pair_id="pair",
        name="Scenario",
        template_path=Path("template.json"),
        template_sha256=digest(template),
        market_files=(
            MarketFileRef(path=Path("market.json"), sha256=digest(market)),
        ),
        n_candles=10,
        end_time=12345,
        yb_mode="active_2l",
        yb_releverage_fee=0.013,
    )


def test_materialization_has_path_independent_closure_and_path_specific_receipt(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = LocalSessionMaterialization.from_scenario(
        _scenario(first_root),
        repository=first_root,
        manifest_root=first_root / "sessions",
        session_id="session-1",
    )
    second = LocalSessionMaterialization.from_scenario(
        _scenario(second_root),
        repository=second_root,
        manifest_root=second_root / "sessions",
        session_id="session-1",
    )

    assert first.closure == second.closure
    assert first.scenario_key == second.scenario_key
    assert first.transport_receipt.sha256 != second.transport_receipt.sha256
