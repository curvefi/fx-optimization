"""PolicySpec and the parameter registry are the optimizer parameter-space authorities."""

import pytest

from decimal import Decimal

from curve_fx_sim.grids.model import GridValidationError, expand_grid
from curve_fx_sim.optimization.profiles import (
    create_lattice_spec,
    profile_from_policy_spec,
    quantized,
)
from curve_fx_sim.optimization.requests import split_request
from curve_fx_sim.specs.common import SpecError
from curve_fx_sim.specs.grid import AxisSpec, AxisTarget, GridSpec
from curve_fx_sim.specs.optimization import OptimizationSpec
from curve_fx_sim.specs.parameters import build_parameter_registry
from curve_fx_sim.specs.policy import PolicyParameter, PolicySpec


def _policy() -> PolicySpec:
    return PolicySpec(
        id="compiled_policy",
        header_file="policies/compiled_policy.hpp",
        source_sha256="b" * 64,
        parameters=(
            PolicyParameter("p0", default=1.5, min_val=1.0, max_val=2.0, step=0.1),
            PolicyParameter("p1", default=5.0, min_val=0.0, max_val=10.0, step=1.0),
        ),
    )


def test_parameter_space_narrows_and_freezes_dense_policy_vector() -> None:
    profile = profile_from_policy_spec(
        _policy(),
        {"p0": {"min": 1.2, "max": 1.8, "step": 0.2}},
    )
    lattice = create_lattice_spec(profile)

    assert profile.parameter_names == ("p0", "p1")
    assert profile.fixed_params == {1: 5.0}
    assert [axis.name for axis in lattice.axes] == ["p0"]
    assert lattice.decode(lattice.encode(profile.initial_seed)) == [1.6, 5.0]


def test_parameter_space_cannot_invent_or_widen_policy_parameters() -> None:
    with pytest.raises(SpecError, match="undeclared dimensions"):
        profile_from_policy_spec(_policy(), {"pool_A": {"min": 1, "max": 2, "step": 1}})
    with pytest.raises(SpecError, match="widens"):
        profile_from_policy_spec(_policy(), {"p0": {"min": 0, "max": 2, "step": 0.1}})


def test_grid_resolves_named_axes_to_policy_dense_order() -> None:
    grid = GridSpec(
        id="policy_grid",
        pair_id="chfusd",
        policy_id="compiled_policy",
        axes=(
            AxisSpec(
                name="p1",
                values=(Decimal("4"), Decimal("6")),
                targets=(AxisTarget(path=("policy_params", "p1")),),
            ),
        ),
    )

    points = expand_grid(grid, policy_spec=_policy())

    assert [point.policy_params for point in points] == [(1.5, 4), (1.5, 6)]
    assert all(not point.pool_overrides for point in points)


def test_compiled_policy_grid_rejects_pool_and_unknown_policy_targets() -> None:
    pool_grid = GridSpec(
        id="pool_grid",
        pair_id="chfusd",
        policy_id="compiled_policy",
        axes=(
            AxisSpec(
                name="mid_fee",
                values=(Decimal("1"),),
                targets=(AxisTarget(path=("pool_overrides", "mid_fee")),),
            ),
        ),
    )
    with pytest.raises(GridValidationError, match="PolicySpec parameter name"):
        expand_grid(pool_grid, policy_spec=_policy())

    unknown_grid = GridSpec(
        id="unknown_grid",
        pair_id="chfusd",
        policy_id="compiled_policy",
        axes=(
            AxisSpec(
                name="unknown",
                values=(Decimal("1"),),
                targets=(AxisTarget(path=("policy_params", "unknown")),),
            ),
        ),
    )
    with pytest.raises(GridValidationError, match="PolicySpec parameter name"):
        expand_grid(unknown_grid, policy_spec=_policy())


def test_optimization_spec_rejects_ignored_generic_config_fields() -> None:
    with pytest.raises(SpecError, match="unsupported optimizer_config fields"):
        OptimizationSpec(
            id="bad-optimizer",
            pair_id="chfusd",
            policy_id="compiled_policy",
            optimizer_config={"population": 16},
        )
    with pytest.raises(SpecError, match="unsupported scoring_config fields"):
        OptimizationSpec(
            id="bad-scoring",
            pair_id="chfusd",
            policy_id="compiled_policy",
            scoring_config={"gate_enabled": True},
        )


# --- Registry-backed pool dimensions ----------------------------------------

_TEMPLATE = {
    "pools": [
        {
            "pool": {
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


def test_profile_with_pool_dims_splits_request_into_policy_and_overrides() -> None:
    profile = profile_from_policy_spec(
        _policy(),
        {
            "p0": {"min": 1.2, "max": 1.8, "step": 0.2},
            "out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005},
        },
        template_json=_TEMPLATE,
    )
    lattice = create_lattice_spec(profile)

    assert profile.n_params() == 2
    assert profile.dense_dim() == 3
    assert profile.initial_seed == (1.5, 5.0, 0.002)
    assert [dim.name for dim in profile.pool_dims] == ["out_fee"]
    assert profile.pool_dims[0].index == 2
    assert profile.pool_dims[0].target_path == ("out_fee",)
    assert [axis.name for axis in lattice.axes] == ["p0", "out_fee"]
    assert lattice.dim == 2
    assert lattice.n_params == 3

    policy_params, pool_overrides = split_request(profile, profile.initial_seed)
    assert policy_params == [1.5, 5.0]
    assert pool_overrides == {"out_fee": 20000000}

    decoded = lattice.decode(lattice.encode(profile.initial_seed))
    quant = quantized(profile, decoded)
    policy_params, pool_overrides = split_request(profile, quant)
    assert policy_params == [1.6, 5.0]
    assert pool_overrides == {"out_fee": 20000000}


def test_pool_log_dimension_propagates_to_lattice_axis() -> None:
    profile = profile_from_policy_spec(
        _policy(),
        {
            "ma_time": {"min": 60, "max": 604800, "step": 60, "transform": "log"},
            "out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005},
        },
        template_json=_TEMPLATE,
    )
    lattice = create_lattice_spec(profile)
    axis_by_name = {axis.name: axis for axis in lattice.axes}

    assert axis_by_name["ma_time"].is_log is True
    assert axis_by_name["out_fee"].is_log is False
    dim_by_name = {dim.name: dim for dim in profile.pool_dims}
    assert dim_by_name["ma_time"].transform == "log"
    assert dim_by_name["out_fee"].transform == "linear"


def test_parameter_space_unknown_name_rejected_through_registry() -> None:
    with pytest.raises(SpecError, match="undeclared dimensions: nope"):
        profile_from_policy_spec(
            _policy(),
            {"nope": {"min": 0, "max": 1, "step": 0.1}},
            template_json=_TEMPLATE,
        )


def test_optimization_spec_validate_parameter_space_through_registry() -> None:
    space = {
        "p1": {"min": 1.0, "max": 9.0, "step": 1.0},
        "out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005},
    }
    registry = build_parameter_registry(_policy(), _TEMPLATE, space)
    spec = OptimizationSpec(
        id="registry-opt",
        pair_id="chfusd",
        policy_id="compiled_policy",
        scenarios=("chfusd-smoke",),
        parameter_space=space,
    )
    spec.validate_parameter_space(registry)

    bad = OptimizationSpec(
        id="registry-bad",
        pair_id="chfusd",
        policy_id="compiled_policy",
        scenarios=("chfusd-smoke",),
        parameter_space={"initial_price": {"min": 1, "max": 2, "step": 1}},
    )
    with pytest.raises(SpecError, match="undeclared dimensions: initial_price"):
        bad.validate_parameter_space(registry)


def test_build_parameter_registry_rejects_multi_pool_template() -> None:
    multi_pool_template = {
        "pools": [_TEMPLATE["pools"][0], _TEMPLATE["pools"][0]],
    }
    with pytest.raises(SpecError, match="single-pool"):
        build_parameter_registry(
            _policy(),
            multi_pool_template,
            {"out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005}},
        )


def test_build_parameter_registry_rejects_scenario_identity_names() -> None:
    with pytest.raises(SpecError, match="scenario-identity"):
        build_parameter_registry(
            _policy(),
            _TEMPLATE,
            {"initial_price": {"min": 1, "max": 2, "step": 1}},
        )
    with pytest.raises(SpecError, match="scenario-identity"):
        build_parameter_registry(
            _policy(),
            _TEMPLATE,
            {"use_volume_cap": {"min": 0, "max": 1, "step": 1}},
        )


def _registry():
    return build_parameter_registry(
        _policy(),
        _TEMPLATE,
        {
            "out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005},
            "arb_fee_bps": {"min": 0, "max": 100, "step": 1},
        },
    )


def test_registry_grid_allows_pool_axes_and_unknown_paths() -> None:
    registry = _registry()
    grid = GridSpec(
        id="registry_grid",
        pair_id="chfusd",
        policy_id="compiled_policy",
        axes=(
            AxisSpec(
                name="out_fee",
                values=(Decimal("0.002"), Decimal("0.004")),
                targets=(AxisTarget(path=("out_fee",)),),
            ),
            AxisSpec(
                name="custom",
                values=(Decimal("1"),),
                targets=(AxisTarget(path=("custom", "x"), kind="integer"),),
            ),
        ),
    )

    points = expand_grid(grid, policy_spec=_policy(), registry=registry)

    assert [point.policy_params for point in points] == [(1.5, 5.0), (1.5, 5.0)]
    assert [point.pool_overrides for point in points] == [
        {"out_fee": 0.002, "custom": {"x": 1}},
        {"out_fee": 0.004, "custom": {"x": 1}},
    ]


def test_registry_grid_rejects_unregistered_policy_target() -> None:
    registry = _registry()
    unknown_grid = GridSpec(
        id="unknown_grid",
        pair_id="chfusd",
        policy_id="compiled_policy",
        axes=(
            AxisSpec(
                name="unknown",
                values=(Decimal("1"),),
                targets=(AxisTarget(path=("policy_params", "unknown")),),
            ),
        ),
    )
    with pytest.raises(GridValidationError, match="registered policy parameter"):
        expand_grid(unknown_grid, policy_spec=_policy(), registry=registry)


def test_registry_grid_rejects_registered_dim_on_wrong_path() -> None:
    registry = _registry()
    wrong_path_grid = GridSpec(
        id="wrong_path",
        pair_id="chfusd",
        policy_id="compiled_policy",
        axes=(
            AxisSpec(
                name="arb_fee_bps",
                values=(Decimal("10"),),
                targets=(AxisTarget(path=("arb_fee_bps",)),),
            ),
        ),
    )
    with pytest.raises(GridValidationError, match="must target costs.arb_fee_bps"):
        expand_grid(wrong_path_grid, policy_spec=_policy(), registry=registry)


def test_registry_grid_accepts_costs_path_for_registered_dim() -> None:
    registry = _registry()
    grid = GridSpec(
        id="costs_path",
        pair_id="chfusd",
        policy_id="compiled_policy",
        axes=(
            AxisSpec(
                name="arb_fee_bps",
                values=(Decimal("10"), Decimal("20")),
                targets=(AxisTarget(path=("costs", "arb_fee_bps"), kind="integer"),),
            ),
        ),
    )
    points = expand_grid(grid, policy_spec=_policy(), registry=registry)

    assert [point.pool_overrides for point in points] == [
        {"costs": {"arb_fee_bps": 10}},
        {"costs": {"arb_fee_bps": 20}},
    ]
    assert [point.policy_params for point in points] == [(1.5, 5.0), (1.5, 5.0)]
