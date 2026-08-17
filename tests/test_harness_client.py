"""Behavioral coverage for the orchestrator's canonical harness adapter."""

from __future__ import annotations

import json
from pathlib import Path
from curve_fx_harness_client.models import (
    BatchResultFrame,
    CandidateResult,
    CandidateSpec,
    EvaluatorIdentity as HarnessIdentity,
    HelloFrame,
    ObservationSpec,
    SessionReadyFrame,
)

from curve_fx_sim.evaluation import client as client_module
from curve_fx_sim.evaluation.client import HarnessClient, ScenarioHarnessClient, SubprocessHarnessClient
from curve_fx_sim.evaluation.identity import VerifiedEvaluator
from curve_fx_sim.specs.scenario import MarketFileRef, ScenarioSpec


def test_subprocess_adapter_attests_session_batch_and_close(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "arb_evaluator_ld"
    binary.write_bytes(b"evaluator")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.toml").write_text("[manifest]\n", encoding="utf-8")
    (tmp_path / "template.json").write_text("{}", encoding="utf-8")
    (tmp_path / "market.json").write_text("[]", encoding="utf-8")
    digest = "a" * 64
    identity = VerifiedEvaluator(
        path=str(binary),
        hello=HelloFrame(
            evaluator_identity=HarnessIdentity(
                binary_sha256=digest,
                harness_version="1",
                pool_version="1",
                policy_id="policy",
                policy_source_sha256="b" * 64,
                policy_abi="v1",
                policy_parameter_count=1,
                numeric_mode="double",
                real_type="double",
                compiler="clang",
                build_target="arb_evaluator_ld",
                ipo_enabled=False,
                native_tuning=False,
            ),
            metric_fields=["apy"],
        ),
    )

    class FakeEvaluatorClient:
        def __init__(self, **kwargs):
            self.closed = False
            self.opened = False
            self.open_count = 0
            self.evaluated = False
        def start(self):
            return HelloFrame(
                evaluator_identity=HarnessIdentity(
                    binary_sha256=digest,
                    harness_version="1",
                    pool_version="1",
                    policy_id="policy",
                    policy_source_sha256="b" * 64,
                    policy_abi="v1",
                    policy_parameter_count=1,
                    numeric_mode="double",
                    real_type="double",
                    compiler="clang",
                    build_target="arb_evaluator_ld",
                    ipo_enabled=False,
                    native_tuning=False,
                )
            )

        def open_session(self, **kwargs):
            self.opened = True
            self.open_count += 1
            self.open_kwargs = kwargs
            return SessionReadyFrame(
                request_id="req",
                session_id="sess",
                scenarios=[],
                scenario_set_sha256="c" * 64,
                    session_fingerprint="d" * 64,
                    session_config_sha256="f" * 64,
                    metric_schema_sha256="e" * 64,
            )

        def evaluate_batch(self, candidates, observation=None):
            self.evaluated = True
            return BatchResultFrame(
                request_id="batch",
                session_id="sess",
                status="complete",
                results=[
                    CandidateResult(
                        ordinal=candidates[0].ordinal,
                        candidate_id=candidates[0].candidate_id,
                        economic_fingerprint="f" * 64,
                    )
                ],
                elapsed_ms=1.0,
            )

        def close_session(self, session_id):
            self.closed = True

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(client_module, "EvaluatorClient", FakeEvaluatorClient)
    monkeypatch.setattr(client_module, "inspect_binary_identity", lambda path: identity)
    adapter = SubprocessHarnessClient(
        binary,
        repository=tmp_path,
        expected_policy_id="policy",
        expected_policy_source_sha256="b" * 64,
        expected_policy_abi="v1",
        expected_policy_parameter_count=1,
    )
    assert adapter.prepare().sha256 == digest

    scenario = ScenarioSpec(
        id="scenario",
        pair_id="pair",
        name="Scenario",
        template_path=Path("template.json"),
        market_files=(MarketFileRef(path=Path("market.json")),),
        end_time=12345,
        yb_releverage=True,
        yb_releverage_fee=0.013,
        yb_cash_multiplier=1.25,
    )
    ready = adapter.open_session(scenario)
    assert ready.session_id == "sess"
    cached_ready = adapter.open_session(scenario)
    assert cached_ready is ready
    assert adapter._client.open_count == 1
    open_kwargs = adapter._client.open_kwargs
    assert "market_files" not in open_kwargs
    assert "economic_defaults" not in open_kwargs
    assert open_kwargs["end_time"] == 12345
    assert open_kwargs["yb_releverage"] is True
    assert open_kwargs["yb_releverage_fee"] == 0.013
    assert open_kwargs["yb_cash_multiplier"] == 1.25
    manifest = json.loads(Path(open_kwargs["manifest_path"]).read_text(encoding="utf-8"))
    resolved = manifest["resolved_spec"]["scenario"]
    assert set(resolved) == {
        "id",
        "start_time",
        "end_time",
        "n_candles",
        "candle_filter",
        "market_files",
    }
    assert Path(resolved["market_files"][0]["path"]).is_absolute()
    assert len(resolved["market_files"][0]["sha256"]) == 64
    response = adapter.evaluate_batch(
        [CandidateSpec(ordinal=0, candidate_id="candidate", policy_params=[1.0])],
        observation=ObservationSpec(),
    )
    assert response.results[0].candidate_id == "candidate"
    adapter.close()
    assert adapter._client.closed


def test_scenario_client_reuses_one_child_per_scenario() -> None:
    scenarios = (
        ScenarioSpec(id="s1", pair_id="p", name="S1"),
        ScenarioSpec(id="s2", pair_id="p", name="S2"),
    )

    class StubClient(HarnessClient):
        def __init__(self) -> None:
            self.open_count = 0
            self.batch_count = 0

        def prepare(self):
            return type("Identity", (), {"sha256": "a" * 64})()

        def open_session(self, scenario_spec, session_id=None):
            self.open_count += 1
            return SessionReadyFrame(
                request_id="req",
                session_id=session_id or scenario_spec.id,
                scenarios=[],
                scenario_set_sha256="c" * 64,
                session_fingerprint="d" * 64,
                session_config_sha256="f" * 64,
                metric_schema_sha256="e" * 64,
            )

        def evaluate_batch(self, candidates, observation=None):
            self.batch_count += 1
            return BatchResultFrame(
                request_id="batch",
                session_id="sess",
                status="complete",
                results=[],
                elapsed_ms=0.0,
            )

        def close(self):
            return None

    created = []

    def factory(scenario):
        client = StubClient()
        created.append(client)
        return client

    client = ScenarioHarnessClient(scenarios, factory)
    client.prepare()
    client.open_session(scenarios[0])
    client.evaluate_batch([])
    client.open_session(scenarios[1])
    client.evaluate_batch([])
    client.open_session(scenarios[0])
    client.evaluate_batch([])

    assert len(created) == 2
    assert [item.batch_count for item in created] == [2, 1]
    client.close()
