"""Tests for specifications, Decimal canonicalization, coordinate expansion, and spec loaders."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from curve_fx_sim.specs.common import (
    SpecError,
    canonical_decimal,
    canonical_dict,
    canonical_json_bytes,
    canonical_primitive,
    format_exact_decimal,
    serializable,
)
from curve_fx_sim.grids.model import expand_grid
from curve_fx_sim.specs.grid import (
    AxisSpec,
    AxisTarget,
    GridSpec,
)
from curve_fx_sim.specs.pair import PairSpec, load_pair_spec
from curve_fx_sim.specs.parameters import (
    POOL_SCENARIO_IDENTITY_PATHS,
    build_parameter_registry,
    validate_parameter_space_names,
)
from curve_fx_sim.specs.policy import (
    PolicyParameter,
    PolicySpec,
    load_policy_spec,
)
from curve_fx_sim.specs.scenario import (
    MarketFileRef,
    ScenarioSpec,
    load_scenario_spec,
)
from curve_fx_sim.specs.shiftclick import (
    ShiftclickSpec,
    load_shiftclick_spec,
)


def test_canonical_decimal_exactness() -> None:
    # Exact integer and decimal values
    assert canonical_decimal(10) == Decimal("10")
    assert canonical_decimal("0.000000000000000001") == Decimal("0.000000000000000001")
    assert canonical_decimal(0.0005) == Decimal("0.0005")

    # Booleans are rejected
    with pytest.raises(SpecError, match="cannot be a boolean"):
        canonical_decimal(True)

    # Non-finite values are rejected
    with pytest.raises(SpecError, match="must be a finite decimal"):
        canonical_decimal(float("nan"))
    with pytest.raises(SpecError, match="must be a finite decimal"):
        canonical_decimal(float("inf"))


def test_format_exact_decimal() -> None:
    assert format_exact_decimal(Decimal("10")) == "10"
    assert format_exact_decimal(Decimal("0.0001")) == "0.0001"
    assert format_exact_decimal(Decimal("123.456")) == "123.456"
    assert format_exact_decimal(Decimal("-0.05")) == "-0.05"
    assert format_exact_decimal(Decimal("0")) == "0"


def test_canonical_primitive_and_json_bytes() -> None:
    data = {
        "b": Decimal("0.0001"),
        "a": 10,
        "c": [Decimal("1.5"), Decimal("2.0")],
    }
    prim = canonical_primitive(data)
    assert prim["a"] == 10
    assert prim["b"] == "0.0001"
    assert prim["c"] == ["1.5", 2]

    # JSON bytes deterministic ordering
    bytes1 = canonical_json_bytes({"z": 1, "a": 2})
    bytes2 = canonical_json_bytes({"a": 2, "z": 1})
    assert bytes1 == bytes2
    assert bytes1 == b'{"a":2,"z":1}'


def test_pair_spec_lifecycle(repo_root: Path) -> None:
    toml_content = """
[pair]
id = "chfusd"
name = "CHF/USD Pair"
base_token = "CHF"
quote_token = "USD"
base_decimals = 18
quote_decimals = 18
"""
    pair_file = repo_root / "configs" / "pairs" / "chfusd.toml"
    pair_file.write_text(toml_content, encoding="utf-8")
    spec = load_pair_spec("chfusd", repository=repo_root)
    assert spec.id == "chfusd"
    assert spec.name == "CHF/USD Pair"
    as_dict = spec.to_dict()
    assert as_dict["id"] == "chfusd"
    assert as_dict["base_token"] == "CHF"


def test_scenario_spec_lifecycle(repo_root: Path) -> None:
    toml_content = """
[scenario]
id = "scenario_chf_01"
pair_id = "chfusd"
name = "CHF January"
start_time = 1000
end_time = 5000
n_candles = 50
candle_filter = 98.5
min_swap = 0.00001
dustswap_freq_s = 1800
yb_releverage = true
yb_releverage_fee = 0.013
yb_cash_multiplier = 1.25
market_files = ["data/feed.csv"]
"""
    scen_file = repo_root / "configs" / "scenarios" / "scenario_chf_01.toml"
    scen_file.write_text(toml_content, encoding="utf-8")
    feed_file = repo_root / "data" / "feed.csv"
    feed_file.write_text("timestamp,price\n", encoding="utf-8")

    spec = load_scenario_spec("scenario_chf_01", repository=repo_root)
    assert spec.id == "scenario_chf_01"
    assert spec.pair_id == "chfusd"
    assert spec.dustswap_freq_s == 1800
    assert spec.yb_releverage is True
    assert spec.yb_releverage_fee == 0.013
    assert spec.yb_cash_multiplier == 1.25
    assert len(spec.market_files) == 1

    fp1 = spec.scenario_fingerprint()
    assert len(fp1) == 64
    assert fp1 == spec.scenario_fingerprint()


def test_policy_spec_validation(repo_root: Path) -> None:
    param = PolicyParameter(
        name="step_bps",
        type="int",
        default=10,
        min_val=1,
        max_val=100,
        step=1,
    )
    assert param.validate_value(15) == 15
    assert param.validate_value(None) == 10
    with pytest.raises(SpecError):
        param.validate_value(200)
    with pytest.raises(SpecError, match="must be an integer"):
        param.validate_value(1.5)
    with pytest.raises(SpecError, match="must be numeric"):
        param.validate_value(True)

    spec = PolicySpec(
        id="test_policy",
        header_file="test.hpp",
        source_sha256="a" * 64,
        parameters=(param,),
    )
    validated = spec.validate_params({"step_bps": 20})
    assert validated == {"step_bps": 20}
    with pytest.raises(SpecError, match="undeclared parameters"):
        spec.validate_params({"ma_time": 300})


def test_policy_spec_rejects_ambiguous_parameter_contracts() -> None:
    with pytest.raises(SpecError, match="off lattice"):
        PolicyParameter(
            name="weight",
            default=0.15,
            min_val=0.0,
            max_val=1.0,
            step=0.1,
        )
    with pytest.raises(SpecError, match="unsupported type"):
        PolicyParameter(
            name="choice",
            type="choice",
            default=1,
            min_val=0,
            max_val=2,
            step=1,
        )
    parameter = PolicyParameter(
        name="weight",
        default=0.5,
        min_val=0.0,
        max_val=1.0,
        step=0.1,
    )
    with pytest.raises(SpecError, match="source_sha256"):
        PolicySpec(
            id="bad",
            header_file="bad.hpp",
            source_sha256="not-a-digest",
            parameters=(parameter,),
        )
    with pytest.raises(SpecError, match="duplicate parameter"):
        PolicySpec(
            id="bad",
            header_file="bad.hpp",
            source_sha256="a" * 64,
            parameters=(parameter, parameter),
        )


def test_scenario_spec_rejects_lossy_or_ignored_fields() -> None:
    base = {"id": "s", "pair_id": "p", "name": "S"}
    with pytest.raises(SpecError, match="start_time must be an integer"):
        ScenarioSpec.from_dict({**base, "start_time": 1.5})
    with pytest.raises(SpecError, match="unsupported scenario fields"):
        ScenarioSpec.from_dict({**base, "economic_defaults": {"A": 5}})
    with pytest.raises(SpecError, match="end_time must be greater"):
        ScenarioSpec.from_dict({**base, "start_time": 10, "end_time": 9})
    with pytest.raises(SpecError, match="yb_cash_multiplier must be positive"):
        ScenarioSpec.from_dict({**base, "yb_cash_multiplier": 0})


def test_grid_spec_ranges_and_expansion() -> None:
    grid = GridSpec(
        id="demo_grid",
        pair_id="chfusd",
        axes=(
            AxisSpec(
                name="A",
                values=(Decimal("100"), Decimal("200")),
                targets=(AxisTarget(path=("A",), kind="integer"),),
            ),
            AxisSpec(
                name="fee_bps",
                values=(Decimal("5"), Decimal("10"), Decimal("20")),
                targets=(AxisTarget(path=("mid_fee",), scale=Decimal("1"), display_scale=Decimal("10000"), kind="decimal"),),
            ),
        ),
        static_overrides={"gamma": 0.0001},
    )
    assert grid.pool_count == 6
    assert grid.coordinate_shape == (2, 3)

    points = expand_grid(grid)
    assert len(points) == 6
    first = points[0]
    assert first.ordinal == 0
    assert first.coordinates == {"A": 100, "fee_bps": 5}
    assert first.pool_overrides["A"] == 100
    assert first.pool_overrides["mid_fee"] == 0.0005
    assert first.pool_overrides["gamma"] == 0.0001

def test_grid_log_range_and_coupled_validation() -> None:
    # Log range test
    axis_data = {
        "name": "log_axis",
        "range": {"start": "1", "stop": "100", "num": 3, "scale": "log"},
    }
    from curve_fx_sim.specs.grid import _parse_axis
    axis = _parse_axis(axis_data, 0)
    assert len(axis.values) == 3
    assert axis.values[0] == Decimal("1")
    assert axis.values[1] == Decimal("10")
    assert axis.values[2] == Decimal("100")

    # Coupled axis validation: mismatched names and targets
    with pytest.raises(SpecError, match="targets but 2 names"):
        AxisSpec(
            name="coupled",
            names=("col1", "col2"),
            targets=(AxisTarget(path=("p1",)),),
            rows=((Decimal("1"), Decimal("2")),),
        )


# --- Parameter registry -----------------------------------------------------

CHFUSD_TEMPLATE = {
    "pools": [
        {
            "tag": "chfusd_native_fx",
            "pool": {
                "initial_liquidity": [
                    "500000000000000000000000",
                    "441501923509893000000000",
                ],
                "A": "543000.0",
                "gamma": "100000000000000",
                "mid_fee": "1000000.0",
                "out_fee": "20000000.0",
                "fee_gamma": "100000000000000000",
                "adjustment_step_min": "100000000",
                "adjustment_step_max": "5000000000000000",
                "ma_time": "5194",
                "reserved_profit_fraction": "5000000000.0",
                "admin_fee": "0",
                "policy": {"kind": "none"},
                "initial_price": "1132528483091349800",
                "start_timestamp": "1609711200",
                "donation_apy": "0.049",
                "donation_frequency": "3600",
                "donation_duration": "604800",
                "donation_coins_ratio": "0.5",
                "user_swap_size_frac": "0.01",
            },
            "costs": {
                "arb_fee_bps": 10,
                "gas_coin0": 0.0,
                "use_volume_cap": False,
                "volume_cap_mult": 1,
            },
        }
    ]
}


def _registry_policy() -> PolicySpec:
    return PolicySpec(
        id="compiled_policy",
        header_file="policies/compiled_policy.hpp",
        source_sha256="b" * 64,
        parameters=(
            PolicyParameter("p0", default=1.5, min_val=1.0, max_val=2.0, step=0.1),
            PolicyParameter("p1", default=5.0, min_val=0.0, max_val=10.0, step=1.0),
        ),
    )


def test_parameter_registry_policy_abi_order() -> None:
    registry = build_parameter_registry(_registry_policy(), CHFUSD_TEMPLATE, {})

    assert list(registry) == ["p0", "p1"]
    assert registry["p0"].kind == "policy"
    assert registry["p0"].abi_index == 0
    assert registry["p0"].target_path is None
    assert registry["p1"].abi_index == 1
    assert registry["p0"].default == 1.5
    assert registry["p0"].min_val == 1.0
    assert registry["p0"].step == 0.1


POOL_SPACE = {
    "A": {"min": 100000.0, "max": 1000000.0, "step": 1000.0, "transform": "log"},
    "gamma": {"min": 1e13, "max": 1e15, "step": 1e12, "transform": "log"},
    "mid_fee": {"min": 0.00001, "max": 0.001, "step": 0.00001},
    "out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005},
    "fee_gamma": {"min": 0.001, "max": 1.0, "step": 0.001},
    "adjustment_step_min": {"min": 1e-12, "max": 1e-6, "step": 1e-12, "transform": "log"},
    "adjustment_step_max": {"min": 1e-5, "max": 0.1, "step": 1e-5, "transform": "log"},
    "ma_time": {"min": 60.0, "max": 604800.0, "step": 60.0, "transform": "log"},
    "reserved_profit_fraction": {"min": 0.0, "max": 0.5, "step": 0.01},
    "admin_fee": {"min": 0.0, "max": 0.5, "step": 0.01},
    "donation_apy": {"min": 0.0, "max": 0.2, "step": 0.001},
    "donation_frequency": {"min": 3600.0, "max": 86400.0, "step": 3600.0},
    "donation_duration": {"min": 604800.0, "max": 2592000.0, "step": 604800.0},
    "donation_coins_ratio": {"min": 0.0, "max": 1.0, "step": 0.1},
    "arb_fee_bps": {"min": 0.0, "max": 100.0, "step": 1.0},
    "gas_coin0": {"min": 0.0, "max": 1.0, "step": 0.1},
    "volume_cap_mult": {"min": 0.0, "max": 10.0, "step": 0.1},
}


def _pool_registry():
    return build_parameter_registry(_registry_policy(), CHFUSD_TEMPLATE, POOL_SPACE)


def test_parameter_registry_pool_dims_only_from_parameter_space() -> None:
    registry = _pool_registry()
    pool_names = [name for name, spec in registry.items() if spec.kind == "pool"]

    assert pool_names == list(POOL_SPACE)
    for spec in registry.values():
        if spec.kind == "pool":
            assert spec.abi_index is None
            assert spec.target_path
    assert registry["arb_fee_bps"].target_path == ("costs", "arb_fee_bps")
    assert registry["donation_coins_ratio"].target_path == ("donation_coins_ratio",)


def test_parameter_registry_without_pool_entries_is_policy_only() -> None:
    registry = build_parameter_registry(_registry_policy(), CHFUSD_TEMPLATE, {})

    assert list(registry) == ["p0", "p1"]
    assert [name for name, spec in registry.items() if spec.kind == "pool"] == []


def test_parameter_registry_defaults_and_scale_from_template() -> None:
    registry = _pool_registry()

    assert registry["A"].default == 543000.0
    assert registry["gamma"].default == 100000000000000.0
    assert registry["mid_fee"].default == 0.0001
    assert registry["out_fee"].default == 0.002
    assert registry["fee_gamma"].default == 0.1
    assert registry["adjustment_step_min"].default == 1e-10
    assert registry["adjustment_step_max"].default == 0.005
    assert registry["ma_time"].default == 5194.0
    assert registry["reserved_profit_fraction"].default == 0.5
    assert registry["admin_fee"].default == 0.0
    assert registry["donation_apy"].default == 0.049
    assert registry["donation_frequency"].default == 3600.0
    assert registry["donation_duration"].default == 604800.0
    assert registry["donation_coins_ratio"].default == 0.5
    assert registry["arb_fee_bps"].default == 10.0
    assert registry["gas_coin0"].default == 0.0
    assert registry["volume_cap_mult"].default == 1.0

    # override_scale is the per-field raw-unit multiplier (harness parse unit).
    assert registry["out_fee"].override_scale == Decimal("10000000000")
    assert registry["mid_fee"].override_scale == Decimal("10000000000")
    assert registry["fee_gamma"].override_scale == Decimal("1000000000000000000")
    assert registry["adjustment_step_min"].override_scale == Decimal("1000000000000000000")
    assert registry["A"].override_scale == Decimal("1")
    assert registry["donation_apy"].override_scale == Decimal("1")
    assert registry["arb_fee_bps"].override_scale == Decimal("1")


def test_parameter_registry_bounds_and_transform_from_parameter_space() -> None:
    registry = _pool_registry()

    assert registry["out_fee"].bounds == (Decimal("0.001"), Decimal("0.005"))
    assert registry["out_fee"].step == Decimal("0.0005")
    assert registry["out_fee"].transform == "linear"
    assert registry["ma_time"].transform == "log"
    assert registry["A"].transform == "log"
    assert registry["donation_apy"].transform == "linear"


def test_parameter_registry_pool_values_list() -> None:
    registry = build_parameter_registry(
        _registry_policy(),
        CHFUSD_TEMPLATE,
        {"donation_apy": [0.0, 0.05, 0.1]},
    )
    dim = registry["donation_apy"]

    assert dim.kind == "pool"
    assert dim.bounds == (Decimal("0.0"), Decimal("0.1"))
    assert dim.step == Decimal("0.05")
    assert dim.transform == "linear"
    assert dim.default == 0.049


def test_parameter_registry_rejects_scenario_identity_and_unknown() -> None:
    for name in POOL_SCENARIO_IDENTITY_PATHS:
        with pytest.raises(SpecError, match="scenario-identity"):
            build_parameter_registry(
                _registry_policy(),
                CHFUSD_TEMPLATE,
                {name: {"min": 0, "max": 1, "step": 1}},
            )
    # A dotted path rooted in an identity field is identity too.
    with pytest.raises(SpecError, match="scenario-identity"):
        build_parameter_registry(
            _registry_policy(),
            CHFUSD_TEMPLATE,
            {"policy.params": {"min": 0, "max": 1, "step": 1}},
        )
    with pytest.raises(SpecError, match="undeclared dimensions: nope"):
        build_parameter_registry(
            _registry_policy(),
            CHFUSD_TEMPLATE,
            {"nope": {"min": 0, "max": 1, "step": 1}},
        )


def test_parameter_registry_accepts_user_swap_size_frac() -> None:
    registry = build_parameter_registry(
        _registry_policy(),
        CHFUSD_TEMPLATE,
        {"user_swap_size_frac": {"min": 0.0, "max": 0.08, "step": 0.01}},
    )

    dim = registry["user_swap_size_frac"]
    assert dim.kind == "pool"
    assert dim.target_path == ("user_swap_size_frac",)
    assert dim.override_scale == Decimal("1")


def test_parameter_registry_policy_only_rejects_pool_names() -> None:
    registry = build_parameter_registry(_registry_policy(), None, {})

    assert list(registry) == ["p0", "p1"]
    with pytest.raises(SpecError, match="undeclared dimensions: out_fee"):
        build_parameter_registry(
            _registry_policy(),
            None,
            {"out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005}},
        )


def test_parameter_registry_requires_template_field_for_pool_default() -> None:
    sparse = {"pools": [{"pool": {"A": "543000.0"}}]}
    with pytest.raises(SpecError, match="no field out_fee"):
        build_parameter_registry(
            _registry_policy(),
            sparse,
            {"out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005}},
        )


def test_validate_parameter_space_names_through_registry() -> None:
    registry = _pool_registry()

    validate_parameter_space_names({"p0": {}, "out_fee": {}}, registry)
    with pytest.raises(SpecError, match="undeclared dimensions: nope"):
        validate_parameter_space_names({"nope": {}}, registry)
