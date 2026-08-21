from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from curve_fx_sim.evaluation.plans import CandidateSchema
from curve_fx_sim.optimization.search import SearchLayout, SearchLayoutError
from curve_fx_sim.specs.common import canonical_json_bytes


def _descriptor(
    name: str,
    path: str,
    value_type: str,
    unit: str,
    wire: str,
    classification: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "name": name,
        "lowering_path": path,
        "type": value_type,
        "unit": unit,
        "wire_representation": wire,
        "classification": classification,
        **extra,
    }


def _schema(*, with_policy: bool = True) -> CandidateSchema:
    policy = [
        _descriptor(
            "policy.b",
            "evaluate_batch.candidates[].policy_params[1]",
            "integer",
            "count",
            "finite_binary64",
            "candidate",
            order=1,
            default=4,
            minimum=0,
            maximum=10,
            quantum=2,
        ),
        _descriptor(
            "policy.a",
            "evaluate_batch.candidates[].policy_params[0]",
            "real",
            "dimensionless",
            "finite_binary64",
            "candidate",
            order=0,
            default=1.0,
            minimum=0.5,
            maximum=2.0,
            quantum=0.25,
        ),
    ]
    parameters = (policy if with_policy else []) + [
        _descriptor(
            "pool.out_fee",
            "pool_overrides.pool.out_fee",
            "real",
            "fee_fraction",
            "binary64_fraction_or_1e10",
            "candidate",
        ),
        _descriptor(
            "pool.costs.gas_coin0",
            "pool_overrides.costs.gas_coin0",
            "real",
            "coin0",
            "binary64",
            "candidate",
        ),
        _descriptor(
            "pool.absent",
            "pool_overrides.pool.absent",
            "real",
            "dimensionless",
            "binary64",
            "candidate",
        ),
        _descriptor(
            "run.yb_releverage_fee",
            "open_session.yb_releverage_fee",
            "real",
            "fraction",
            "binary64",
            "session",
            default=0.01,
        ),
        _descriptor(
            "run.samples",
            "open_session.samples",
            "integer",
            "count",
            "uint64",
            "session",
            default=10,
        ),
        _descriptor(
            "run.observed",
            "open_session.observed",
            "real",
            "dimensionless",
            "binary64",
            "observation",
            default=0.0,
        ),
        _descriptor(
            "run.legacy",
            "open_session.legacy",
            "boolean",
            "legacy_alias",
            "json_boolean",
            "session",
            default=False,
        ),
    ]
    unsupported = (
        ("flag", "boolean", "flag", "json_boolean", False, {}),
        ("mode", "enum", "mode", "utf8", "off", {"choices": ["off", "on"]}),
        ("label", "string", "identifier", "utf8", "x", {}),
        ("options", "object", "settings", "json_object", {}, {}),
        ("pair", "real_pair", "wad_pair", "binary64_from_wad_1e18", [1.0, 1.0], {}),
    )
    parameters.extend(
        _descriptor(
            f"run.{name}", f"open_session.{name}", kind, unit, wire, "session",
            default=default, **extra,
        )
        for name, kind, unit, wire, default, extra in unsupported
    )
    parameter_schema = {
        "schema_version": "curve_fx_parameter_schema_v1",
        "parameters": parameters,
    }
    canonical = canonical_json_bytes(parameter_schema).decode()
    description = {
        "schema_version": "curve_fx_evaluator_description_v1",
        "policy": {
            "id": "compiled" if with_policy else "native_passthrough",
            "parameter_count": 2 if with_policy else 0,
            "descriptor_abi_version": 1,
        },
        "parameter_schema": parameter_schema,
        "parameter_schema_canonical_json": canonical,
        "parameter_schema_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    return CandidateSchema.from_description(description)


TEMPLATE = {"pools": [{
    "pool": {"out_fee": "20000000"},
    "costs": {"gas_coin0": "0.004"},
}]}
POOL_RUN_SPACE = {
    "run.yb_releverage_fee": {"min": 0.01, "max": 0.02, "step": 0.001},
    "pool.out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005},
}


def test_empty_space_searches_policy_in_schema_order_on_exact_domains() -> None:
    layout = SearchLayout.from_schema(_schema(), {}, TEMPLATE, {})
    assert [dimension.name for dimension in layout.dimensions] == ["policy.b", "policy.a"]
    assert layout.default_vector == (4, 1)
    assert layout.dimensions[1].minimum == Decimal("0.5")
    assert layout.dimensions[1].maximum == Decimal("2.0")
    assert layout.dimensions[1].step == Decimal("0.25")
    lattice = layout.create_lattice_spec()
    assert [(axis.name, axis.min_tick, axis.max_tick) for axis in lattice.axes] == [
        ("policy.b", 0, 5),
        ("policy.a", 2, 8),
    ]
    assert layout.to_proposal(lattice.decode((3, 5))) == {"policy.b": 6, "policy.a": 1.25}

def test_pool_and_run_layout_is_named_order_independent_and_human_scaled() -> None:
    layout = SearchLayout.from_schema(
        _schema(),
        POOL_RUN_SPACE,
        TEMPLATE,
        {"yb_releverage_fee": 0.013},
    )
    assert [dimension.name for dimension in layout.dimensions] == [
        "pool.out_fee",
        "run.yb_releverage_fee",
    ]
    assert layout.dimensions[0].default == Decimal("0.002")
    assert layout.dimensions[1].default == Decimal("0.013")
    assert layout.default_vector == (0.002, 0.013)
    assert layout.to_proposal([0.003, 0.014]) == {
        "pool.out_fee": 0.003, "run.yb_releverage_fee": 0.014}
    reordered = SearchLayout.from_schema(
        _schema(),
        dict(reversed(list(POOL_RUN_SPACE.items()))),
        TEMPLATE,
        {"yb_releverage_fee": 0.013},
    )
    assert reordered.sha256 == layout.sha256
    assert layout.create_lattice_spec().profile_name == layout.sha256


def test_invalid_search_dimensions_fail_closed_by_category() -> None:
    cases = (
        ({"nope": {}}, "unknown canonical"),
        ({"run.observed": {}}, "observation"),
        ({"run.legacy": {}}, "legacy alias"),
        ({"run.flag": {}}, "unsupported optimizer type boolean"),
        ({"run.mode": {}}, "unsupported optimizer type enum"),
        ({"run.label": {}}, "unsupported optimizer type string"),
        ({"run.options": {}}, "unsupported optimizer type object"),
        ({"run.pair": {}}, "unsupported optimizer type real_pair"),
        ({"policy.a": {"min": 0.25}}, "widens"),
        ({"policy.a": {"step": 0.3}}, "integer multiple"),
        ({"policy.a": {"min": 0.6}}, "off descriptor lattice"),
        ({"policy.a": {"step": 0}}, "positive"),
        ({"policy.a": {"values": [0.5, 1.0, 1.25]}}, "uniformly spaced"),
        ({"policy.a": {"min": 1.25, "max": 1.5, "step": 0.25}}, "base default.*outside"),
        ({"policy.a": {"min": float("nan")}}, "finite"),
        ({"run.samples": {"min": 8, "max": 12, "step": 0.5}}, "numeric lattice"),
        ({"pool.costs.gas_coin0": {"min": 0, "max": 0.008, "step": 0.004, "transform": "log"}}, "positive bounds/default"),
    )
    for space, message in cases:
        with pytest.raises(SearchLayoutError, match=message):
            SearchLayout.from_schema(_schema(), space, TEMPLATE, {})

    layout = SearchLayout.from_schema(
        _schema(),
        {"policy.a": {"min": 0.5, "max": 1.5, "step": 0.25}},
        TEMPLATE,
        {},
    )
    with pytest.raises(SearchLayoutError, match="finite"):
        layout.to_proposal([float("inf")])
    with pytest.raises(SearchLayoutError, match="off the exact step"):
        layout.to_proposal([0.6])
    with pytest.raises(SearchLayoutError, match="length"):
        layout.to_proposal([])
    with pytest.raises(SearchLayoutError, match="exactly one pool"):
        SearchLayout.from_schema(
            _schema(with_policy=False),
            {"pool.out_fee": {"min": 0.001, "max": 0.003, "step": 0.001}},
            {"pools": [TEMPLATE["pools"][0], TEMPLATE["pools"][0]]},
            {},
        )
