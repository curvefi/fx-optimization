"""Focused coverage for verified local compiled-session admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from curve_fx_harness_client.models import SessionReadyFrame

from curve_fx_sim.evaluation import client as client_module
from curve_fx_sim.evaluation.client import SubprocessHarnessClient
from curve_fx_sim.evaluation.plans import CandidatePlan, ScenarioKey, SessionKey
from curve_fx_sim.evaluation.session import (
    LocalSessionMaterialization,
    SessionMaterializationError,
)
from curve_fx_sim.specs.common import canonical_json_bytes
from curve_fx_sim.specs.scenario import MarketFileRef, ScenarioSpec


def _scenario(root: Path, *, declared_hashes: bool = True) -> ScenarioSpec:
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
        template_sha256=digest(template) if declared_hashes else None,
        market_files=(
            MarketFileRef(
                path=Path("market.json"),
                sha256=digest(market) if declared_hashes else None,
            ),
        ),
        n_candles=10,
        end_time=12345,
        yb_mode="active_2l",
        yb_releverage_fee=0.013,
    )


def _session_key(request: dict[str, object]) -> SessionKey:
    identity_json = canonical_json_bytes(
        {
            "version": "curve_fx_session_key_v1",
            "parameter_schema_sha256": "f" * 64,
            "open_session": {"run.n_candles": request["n_candles"]},
        }
    )
    return SessionKey(identity_json, hashlib.sha256(identity_json).hexdigest())


def _plan(
    materialization: LocalSessionMaterialization,
    request: dict[str, object],
    *,
    scenario_key: ScenarioKey | None = None,
) -> CandidatePlan:
    candidate_json = canonical_json_bytes({"policy_params": [], "pool_overrides": {}})
    return CandidatePlan(
        scenario_key=scenario_key or materialization.scenario_key,
        session_key=_session_key(request),
        session_request_json=canonical_json_bytes(request),
        policy_params=(),
        pool_overrides_json=b"{}",
        candidate_json=candidate_json,
        candidate_sha256=hashlib.sha256(candidate_json).hexdigest(),
        named_values=(),
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

def test_compiled_inlet_verifies_exact_request_and_reopens_on_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "evaluator"
    binary.write_bytes(b"binary")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    calls: list[dict[str, object]] = []
    clients: list[object] = []

    class FakeEvaluatorClient:
        def __init__(self, **kwargs: object) -> None:
            self.shutdown_called = False
            clients.append(self)

        def open_session(self, **kwargs: object) -> SessionReadyFrame:
            calls.append(kwargs)
            return SessionReadyFrame(
                request_id="request",
                session_id=str(kwargs["session_id"]),
                scenarios=[],
                scenario_set_sha256="a" * 64,
                session_fingerprint="b" * 64,
                session_config_sha256="c" * 64,
                metric_schema_sha256="d" * 64,
            )

        def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(client_module, "EvaluatorClient", FakeEvaluatorClient)
    client = SubprocessHarnessClient(
        binary,
        repository=tmp_path,
        work_dir=work_dir,
    )
    monkeypatch.setattr(client, "prepare", lambda: object())
    materialization = client.materialize_session(_scenario(tmp_path), session_id="session-1")
    baseline = materialization.baseline_open_session_fields
    baseline_plan = _plan(materialization, baseline)

    first = client.open_compiled_session(baseline_plan, materialization)
    assert first.session_id == "session-1"
    assert calls == [json.loads(baseline_plan.session_request_json)]
    assert client.open_compiled_session(baseline_plan, materialization) is first
    assert len(calls) == 1

    changed = dict(baseline)
    changed["n_candles"] = 11
    client.open_compiled_session(_plan(materialization, changed), materialization)
    assert len(calls) == 2
    assert calls[-1]["n_candles"] == 11
    assert len(clients) == 2
    assert clients[0].shutdown_called is True


def test_forged_closure_transport_and_request_fail_before_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "evaluator"
    binary.write_bytes(b"binary")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    protocol_calls = 0

    class FakeEvaluatorClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def open_session(self, **kwargs: object) -> SessionReadyFrame:
            nonlocal protocol_calls
            protocol_calls += 1
            raise AssertionError("forged input reached protocol")

    monkeypatch.setattr(client_module, "EvaluatorClient", FakeEvaluatorClient)
    client = SubprocessHarnessClient(binary, repository=tmp_path, work_dir=work_dir)
    monkeypatch.setattr(client, "prepare", lambda: object())
    materialization = client.materialize_session(_scenario(tmp_path), session_id="session-1")
    baseline = materialization.baseline_open_session_fields

    forged_closure = replace(materialization.closure, pair_id="forged-pair")
    forged_transport = dict(baseline)
    forged_transport["template_path"] = "/forged/template.json"
    cases = (
        _plan(
            materialization,
            baseline,
            scenario_key=ScenarioKey.from_closure(forged_closure),
        ),
        _plan(materialization, forged_transport),
    )
    for plan in cases:
        with pytest.raises(SessionMaterializationError):
            client.open_compiled_session(plan, materialization)
    assert protocol_calls == 0
