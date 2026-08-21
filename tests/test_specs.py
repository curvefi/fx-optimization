"""Tests for specifications, Decimal canonicalization, coordinate expansion, and loaders."""
from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import pytest
from curve_fx_sim.grids.model import expand_grid
from curve_fx_sim.specs.common import (
    SpecError,
    canonical_decimal,
    canonical_json_bytes,
    canonical_primitive,
    format_exact_decimal,
)
from curve_fx_sim.specs.grid import AxisSpec, AxisTarget, GridSpec
from curve_fx_sim.specs.pair import load_pair_spec
from curve_fx_sim.specs.policy import PolicyParameter, PolicySpec
from curve_fx_sim.specs.scenario import ScenarioSpec, load_scenario_spec


def test_canonical_values_and_json_bytes() -> None:
    assert canonical_decimal(10) == Decimal("10")
    assert canonical_decimal("0.000000000000000001") == Decimal("0.000000000000000001")
    assert canonical_decimal(0.0005) == Decimal("0.0005")
    for value in (True, float("nan"), float("inf")):
        with pytest.raises(SpecError):
            canonical_decimal(value)
    assert [format_exact_decimal(value) for value in (
        Decimal("10"), Decimal("0.0001"), Decimal("-0.05"), Decimal("0")
    )] == ["10", "0.0001", "-0.05", "0"]
    data = {"b": Decimal("0.0001"), "a": 10, "c": [Decimal("1.5"), Decimal("2.0")]}
    primitive = canonical_primitive(data)
    assert primitive == {"a": 10, "b": "0.0001", "c": ["1.5", 2]}
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}'


def test_pair_and_scenario_spec_loaders(repo_root: Path) -> None:
    (repo_root / "configs" / "pairs" / "chfusd.toml").write_text(
        """
[pair]
id = "chfusd"
name = "CHF/USD Pair"
base_token = "CHF"
quote_token = "USD"
base_decimals = 18
quote_decimals = 18
""",
        encoding="utf-8",
    )
    pair = load_pair_spec("chfusd", repository=repo_root)
    assert pair.to_dict()["base_token"] == "CHF"
    (repo_root / "configs" / "scenarios" / "scenario_chf_01.toml").write_text(
        """
[scenario]
id = "scenario_chf_01"
pair_id = "chfusd"
name = "CHF January"
start_time = 1000
end_time = 5000
n_candles = 50
dustswap_freq_s = 1800
yb_releverage = true
yb_releverage_fee = 0.013
yb_cash_multiplier = 1.25
market_files = ["data/feed.csv"]
""",
        encoding="utf-8",
    )
    (repo_root / "data" / "feed.csv").write_text("timestamp,price\n", encoding="utf-8")
    scenario = load_scenario_spec("scenario_chf_01", repository=repo_root)
    assert scenario.pair_id == "chfusd"
    assert scenario.market_files[0].path == Path("data/feed.csv")
    assert scenario.scenario_fingerprint() == scenario.scenario_fingerprint()


def test_spec_validation_table() -> None:
    parameter = PolicyParameter(
        name="step_bps", type="int", default=10, min_val=1, max_val=100, step=1
    )
    assert parameter.validate_value(15) == 15
    assert parameter.validate_value(None) == 10
    for value in (200, 1.5, True):
        with pytest.raises(SpecError):
            parameter.validate_value(value)
    spec = PolicySpec(
        id="test_policy", header_file="test.hpp", source_sha256="a" * 64, parameters=(parameter,)
    )
    assert spec.validate_params({"step_bps": 20}) == {"step_bps": 20}
    with pytest.raises(SpecError, match="undeclared parameters"):
        spec.validate_params({"ma_time": 300})
    with pytest.raises(SpecError, match="off lattice"):
        PolicyParameter(name="weight", default=0.15, min_val=0.0, max_val=1.0, step=0.1)
    with pytest.raises(SpecError, match="unsupported type"):
        PolicyParameter(name="choice", type="choice", default=1, min_val=0, max_val=2, step=1)
    parameter = PolicyParameter(name="weight", default=0.5, min_val=0.0, max_val=1.0, step=0.1)
    with pytest.raises(SpecError, match="source_sha256"):
        PolicySpec(id="bad", header_file="bad.hpp", source_sha256="not-a-digest", parameters=(parameter,))
    with pytest.raises(SpecError, match="duplicate parameter"):
        PolicySpec(id="bad", header_file="bad.hpp", source_sha256="a" * 64, parameters=(parameter, parameter))
    base = {"id": "s", "pair_id": "p", "name": "S"}
    with pytest.raises(SpecError, match="start_time must be an integer"):
        ScenarioSpec.from_dict({**base, "start_time": 1.5})
    with pytest.raises(SpecError, match="unsupported scenario fields"):
        ScenarioSpec.from_dict({**base, "economic_defaults": {"A": 5}})
    with pytest.raises(SpecError, match="end_time must be greater"):
        ScenarioSpec.from_dict({**base, "start_time": 10, "end_time": 9})
    with pytest.raises(SpecError, match="yb_cash_multiplier must be positive"):
        ScenarioSpec.from_dict({**base, "yb_cash_multiplier": 0})
    grid = GridSpec(
        id="demo_grid",
        pair_id="chfusd",
        axes=(
            AxisSpec(name="A", values=(Decimal("100"), Decimal("200")), targets=(AxisTarget(path=("A",), kind="integer"),)),
            AxisSpec(name="fee_bps", values=(Decimal("5"), Decimal("10"), Decimal("20")), targets=(AxisTarget(path=("mid_fee",), display_scale=Decimal("10000"), kind="decimal"),)),
        ),
        static_overrides={"gamma": 0.0001},
    )
    points = expand_grid(grid)
    assert grid.pool_count == len(points) == 6
    assert points[0].coordinates == {"A": 100, "fee_bps": 5}
    assert points[0].pool_overrides == {"A": 100, "mid_fee": 0.0005, "gamma": 0.0001}
    with pytest.raises(SpecError, match="targets but 2 names"):
        AxisSpec(name="coupled", names=("col1", "col2"), targets=(AxisTarget(path=("p1",)),), rows=((Decimal("1"), Decimal("2")),))
