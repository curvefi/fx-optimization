from __future__ import annotations

import copy
from decimal import Decimal
import hashlib

import pytest

from curve_fx_sim.evaluation.plans import (
    CandidateCompiler,
    CandidatePlanError,
    CandidateSchema,
    ScenarioKey,
)
from curve_fx_sim.specs.common import canonical_json_bytes
from curve_fx_sim.specs.scenario import ScenarioClosure, ScenarioMarketInput


POLICY = (
    ("fast_half_life_s", 3600.0, 60.0, 86400.0, 10.0),
    ("slow_half_life_s", 86400.0, 60.0, 604800.0, 10.0),
    ("kappa", 1.0, 0.0, 5.0, 0.05),
    ("min_cap_bps", 10.0, 0.0, 250.0, 0.5),
    ("deadband_bps", 0.0, 0.0, 100.0, 0.5),
)
HASH_A = "a" * 64
HASH_B = "b" * 64


def _seal(description: dict[str, object]) -> dict[str, object]:
    canonical = canonical_json_bytes(description["parameter_schema"]).decode("utf-8")
    description["parameter_schema_canonical_json"] = canonical
    description["parameter_schema_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return description


def _parameter(
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


def _description(*, with_policy: bool = True) -> dict[str, object]:
    parameters: list[dict[str, object]] = []
    if with_policy:
        for order, (name, default, minimum, maximum, quantum) in enumerate(POLICY):
            parameters.append(
                _parameter(
                    f"policy.{name}",
                    f"evaluate_batch.candidates[].policy_params[{order}]",
                    "real",
                    "seconds" if order < 2 else "dimensionless",
                    "finite_binary64",
                    "candidate",
                    order=order,
                    default=default,
                    minimum=minimum,
                    maximum=maximum,
                    quantum=quantum,
                )
            )
    parameters.extend(
        [
            _parameter(
                "pool.out_fee",
                "pool_overrides.pool.out_fee",
                "real",
                "fee_fraction",
                "binary64_fraction_or_1e10",
                "candidate",
            ),
            _parameter("run.session_id", "open_session.session_id", "string", "identifier", "utf8", "session"),
            _parameter("run.template_path", "open_session.template_path", "string", "path", "utf8", "session"),
            _parameter(
                "run.template_sha256", "open_session.template_sha256", "string", "sha256", "lower_hex_64", "session"
            ),
            _parameter("run.manifest_path", "open_session.manifest_path", "string", "path", "utf8", "session"),
            _parameter(
                "run.manifest_sha256", "open_session.manifest_sha256", "string", "sha256", "lower_hex_64", "session"
            ),
            _parameter("run.n_candles", "open_session.n_candles", "integer", "count", "uint64", "session", default=0),
            _parameter(
                "run.candle_filter", "open_session.candle_filter", "real", "percent", "binary64", "session", default=0.0
            ),
            _parameter(
                "run.yb_mode", "open_session.yb_mode", "enum", "yb_mode", "utf8", "session",
                default="off", choices=["off", "passive", "active_2l"],
            ),
            _parameter(
                "run.yb_releverage", "open_session.yb_releverage", "boolean", "legacy_alias",
                "json_boolean", "session", default=False,
            ),
            _parameter(
                "run.disable_slippage_probes",
                "open_session.disable_slippage_probes",
                "boolean",
                "flag",
                "json_boolean",
                "observation",
                default=False,
            ),
            _parameter(
                "run.metric_projection",
                "evaluate_batch.metric_projection",
                "enum",
                "projection",
                "utf8",
                "observation",
                choices=["summary", "full"],
            ),
        ]
    )
    schema: dict[str, object] = {
        "schema_version": "curve_fx_parameter_schema_v1",
        "parameters": parameters,
    }
    return _seal({
        "schema_version": "curve_fx_evaluator_description_v1",
        "policy": {
            "id": "compiled" if with_policy else "native_passthrough",
            "parameter_count": len(POLICY) if with_policy else 0,
            "descriptor_abi_version": 1,
        },
        "parameter_schema": schema,
    })


def _session(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "session_id": "session-1",
        "template_path": "/local/template.json",
        "template_sha256": HASH_A,
        "manifest_path": "/local/manifest.json",
        "manifest_sha256": HASH_B,
    }
    result.update(updates)
    return result


def _scenario(*, template_sha256: str = "c" * 64) -> ScenarioClosure:
    return ScenarioClosure(
        scenario_id="ethusd",
        pair_id="eth-usd",
        template_sha256=template_sha256,
        market_inputs=(ScenarioMarketInput(kind="market", sha256="d" * 64),),
    )


def _compile(
    compiler: CandidateCompiler,
    proposal: dict[str, object] | None = None,
    *,
    session: dict[str, object] | None = None,
    scenario: ScenarioClosure | ScenarioKey | None = None,
):
    return compiler.compile(
        proposal or {},
        open_session=session or _session(),
        scenario=scenario or _scenario(),
    )


def test_order_independent_lowering_and_candidate_session_separation() -> None:
    compiler = CandidateCompiler.from_description(_description())
    first = _compile(
        compiler,
        {"policy.kappa": 1.5, "pool.out_fee": 0.002},
        session=_session(candle_filter=99.9),
    )
    second = compiler.compile(
        {"pool.out_fee": 0.002, "policy.kappa": 1.5},
        open_session=dict(reversed(list(_session(candle_filter=99.9).items()))),
        scenario=_scenario(),
    )
    assert first.policy_params == (3600.0, 86400.0, 1.5, 10.0, 0.0)
    assert first.pool_overrides == {"pool": {"out_fee": 20_000_000}}
    assert first.candidate_json == second.candidate_json
    assert first.candidate_sha256 == second.candidate_sha256
    assert first.session_key == second.session_key
    assert first.named_values == (
        ("policy.deadband_bps", 0.0),
        ("policy.fast_half_life_s", 3600.0),
        ("policy.kappa", 1.5),
        ("policy.min_cap_bps", 10.0),
        ("policy.slow_half_life_s", 86400.0),
        ("pool.out_fee", 0.002),
    )

    different_session = _compile(
        compiler,
        {"policy.kappa": 1.5, "pool.out_fee": 0.002},
        session=_session(n_candles=12),
    )
    assert different_session.session_key != first.session_key
    assert different_session.candidate_json == first.candidate_json

    different_candidate = _compile(compiler, {"policy.kappa": 2.0, "pool.out_fee": 0.002})
    baseline_candidate = _compile(compiler, {"policy.kappa": 1.5, "pool.out_fee": 0.002})
    assert different_candidate.session_key == baseline_candidate.session_key
    assert different_candidate.candidate_json != baseline_candidate.candidate_json


def test_scenario_session_transport_identity_and_relocation() -> None:
    compiler = CandidateCompiler.from_description(_description())
    baseline = _compile(compiler)
    relocated = _compile(
        compiler,
        session=_session(
            session_id="remote-session",
            template_path="/nfs/template.json",
            template_sha256="d" * 64,
            manifest_path="/nfs/manifest.json",
            manifest_sha256="e" * 64,
        ),
    )
    assert relocated.session_request != baseline.session_request
    assert relocated.scenario_key == baseline.scenario_key
    assert relocated.session_key == baseline.session_key

    changed_session = _compile(compiler, session=_session(candle_filter=42.0))
    changed_scenario = _compile(compiler, scenario=_scenario(template_sha256="f" * 64))
    assert changed_session.session_key != baseline.session_key
    assert changed_session.scenario_key == baseline.scenario_key
    assert changed_scenario.scenario_key != baseline.scenario_key
    assert changed_scenario.session_key == baseline.session_key

    transport = _compile(compiler, {"run.disable_slippage_probes": True})
    assert transport.session_request["disable_slippage_probes"] is True
    assert transport.session_key != baseline.session_key


    closure = _scenario()
    key = ScenarioKey.from_closure(closure)
    assert key.identity_json == canonical_json_bytes(closure.to_identity())
    assert key.sha256 == closure.sha256
    with pytest.raises(CandidatePlanError, match="ScenarioClosure or validated ScenarioKey"):
        compiler.compile({}, open_session=_session(), scenario={"forged": True})

    forged_json = canonical_json_bytes({"forged": True})
    forged = ScenarioKey(forged_json, hashlib.sha256(forged_json).hexdigest())
    with pytest.raises(CandidatePlanError, match="ScenarioClosure identity"):
        compiler.compile({}, open_session=_session(), scenario=forged)


def test_compiler_finalizes_yb_legacy_alias_table() -> None:
    compiler = CandidateCompiler.from_description(_description())
    for mode, alias in (("off", False), ("passive", True), ("active_2l", True)):
        plan = _compile(compiler, {"run.yb_mode": mode}, session=_session(yb_releverage=not alias))
        assert plan.session_request["yb_mode"] == mode
        assert plan.session_request["yb_releverage"] is alias
        assert plan.session_request_json == canonical_json_bytes(plan.session_request)


def test_invalid_proposals_fail_by_category() -> None:
    compiler = CandidateCompiler.from_description(_description())
    cases = (
        ({"unknown": 1}, "unknown proposal key"),
        ({"run.template_path": "/forged/template.json"}, "transport materialization field"),
        ({"run.yb_releverage": True}, "legacy alias"),
        ({"policy.fast_half_life_s": 65.0}, "off quantum"),
        ({"policy.kappa": 6.0}, "outside"),
        ({"pool.out_fee": float("inf")}, "finite"),
        ({"run.yb_mode": "automatic"}, "must be one of"),
    )
    for proposal, message in cases:
        with pytest.raises(CandidatePlanError, match=message):
            _compile(compiler, proposal)


def test_schema_corruption_table() -> None:
    mutations = (
        lambda ps: ps[1].update(name=ps[0]["name"]),
        lambda ps: ps[1].update(lowering_path=ps[0]["lowering_path"]),
        lambda ps: ps[1].update(order=ps[0]["order"]),
        lambda ps: ps[0].update(type="decimal128"),
        lambda ps: ps[0].update(wire_representation="decimal_string"),
        lambda ps: ps[-1].update(choices=["summary", "summary"]),
    )
    for mutation in mutations:
        description = _description()
        mutation(description["parameter_schema"]["parameters"])
        with pytest.raises(CandidatePlanError):
            CandidateSchema.from_description(_seal(description))

    malformed = _description()
    malformed["parameter_schema_sha256"] = "not-a-hash"
    with pytest.raises(CandidatePlanError, match="lowercase SHA-256"):
        CandidateSchema.from_description(malformed)
    mismatched = copy.deepcopy(_description())
    mismatched["parameter_schema_sha256"] = "0" * 64
    with pytest.raises(CandidatePlanError, match="does not match canonical"):
        CandidateSchema.from_description(mismatched)


def test_scaled_wad_values_canonicalize_at_the_binary64_boundary() -> None:
    description = _description()
    description["parameter_schema"]["parameters"].insert(
        len(POLICY),
        _parameter(
            "pool.initial_price", "pool_overrides.pool.initial_price", "real", "price",
            "binary64_from_wad_1e18", "candidate",
        ),
    )
    compiler = CandidateCompiler.from_description(_seal(description))
    first = _compile(compiler, {"pool.initial_price": Decimal("0.009007199254740995")})
    second = _compile(compiler, {"pool.initial_price": Decimal("0.009007199254740996")})
    assert first.candidate_payload == second.candidate_payload
    assert first.candidate_sha256 == second.candidate_sha256

    with pytest.raises(CandidatePlanError, match="binary64 domain"):
        _compile(compiler, {"pool.initial_price": Decimal("1e999999")})
