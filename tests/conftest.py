"""Shared pytest fixtures for curve-fx-optimization tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from curve_fx_sim.artifacts.manifest import (
    new_grid_manifest,
    new_optimization_manifest,
    new_shiftclick_manifest,
)
from curve_fx_sim.artifacts.store import RunStore
from curve_fx_sim.artifacts.tables import (
    EvaluationRow,
    EvaluationTable,
    MetricProjection,
)
from curve_fx_sim.specs.common import (
    canonical_dict,
    canonical_json_bytes,
    format_exact_decimal,
)
from curve_fx_sim.specs.grid import (
    AxisSpec,
    AxisTarget,
    GridSpec,
)
from curve_fx_sim.specs.pair import PairSpec
from curve_fx_sim.specs.policy import (
    PolicyParameter,
    PolicySpec,
)
from curve_fx_sim.specs.scenario import (
    MarketFileRef,
    ScenarioSpec,
)
from curve_fx_sim.specs.shiftclick import ShiftclickSpec


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Fixture providing a mock repository root with standard directories."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
    (tmp_path / "configs" / "pairs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "scenarios").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "policies").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "grids").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "optimizations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def sample_pair(repo_root: Path) -> PairSpec:
    return PairSpec(
        id="chfusd",
        name="CHF/USD",
        base_token="CHF",
        quote_token="USD",
        base_decimals=18,
        quote_decimals=18,
    )


@pytest.fixture
def sample_scenario(repo_root: Path) -> ScenarioSpec:
    return ScenarioSpec(
        id="scenario_jan",
        pair_id="chfusd",
        name="January Test",
        start_time=1000,
        end_time=2000,
        n_candles=100,
        market_files=(MarketFileRef(path=Path("data/feed.csv"), sha256="abc" * 21 + "a"),),
    )


@pytest.fixture
def sample_grid() -> GridSpec:
    return GridSpec(
        id="test_grid",
        pair_id="chfusd",
        axes=(
            AxisSpec(
                name="mid_fee",
                values=(Decimal("0.0001"), Decimal("0.0005")),
                targets=(AxisTarget(path=("mid_fee",), kind="decimal"),),
            ),
            AxisSpec(
                name="out_fee",
                values=(Decimal("0.0010"), Decimal("0.0040")),
                targets=(AxisTarget(path=("out_fee",), kind="decimal"),),
            ),
        ),
    )


@pytest.fixture
def sample_table() -> EvaluationTable:
    rows = [
        EvaluationRow(
            candidate_id="cand_001",
            ordinal=0,
            coordinates={"mid_fee": 0.0001, "out_fee": 0.0010},
            params={"param_a": 10.0},
            metrics={"apy": 0.12, "vp": 1.05, "trades": 150},
            status="ok",
            economic_fingerprint="fp001",
        ),
        EvaluationRow(
            candidate_id="cand_002",
            ordinal=1,
            coordinates={"mid_fee": 0.0005, "out_fee": 0.0040},
            params={"param_a": 20.0},
            metrics={"apy": 0.18, "vp": 1.08, "trades": 220},
            status="ok",
            economic_fingerprint="fp002",
        ),
    ]
    return EvaluationTable(rows=rows, metadata={"test": True})


@pytest.fixture
def run_store(repo_root: Path) -> RunStore:
    return RunStore(repo_root)
