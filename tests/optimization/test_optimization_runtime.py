"""Deterministic tests for optimization runtime, status, resume lifecycle, and artifact generation."""

import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import pytest

from curve_fx_sim.artifacts.store import RunStore
from curve_fx_sim.artifacts.io import atomic_write_json, sha256_path
from curve_fx_sim.artifacts.tables import EvaluationTable
from curve_fx_sim.execution.adapter import ProcessResult
from curve_fx_sim.execution.site import ClusterConfig, SiteProfile
from curve_fx_sim.evaluation.client import HarnessClient
from curve_fx_sim.evaluation.identity import VerifiedEvaluator
from curve_fx_harness_client.models import (
    BatchResultFrame,
    CandidateResult,
    CandidateSpec,
    EvaluatorIdentity as ProtocolEvaluatorIdentity,
    HelloFrame,
    ObservationSpec,
    SessionReadyFrame,
)
from curve_fx_sim.optimization.runtime import (
    EVALUATION_JOURNAL_FILENAME,
    _dispatch_remote_bundle,
    collect_optimization,
    run_optimization,
    status_optimization,
)
from curve_fx_sim.optimization.profiles import profile_from_policy_spec
from curve_fx_sim.optimization.worker import (
    BUNDLE_RESULT_SCHEMA_VERSION,
    OptimizationBundleResult,
    create_work_bundle,
    evaluate_work_bundle,
)
from curve_fx_sim.specs.common import SpecError
from curve_fx_sim.specs.optimization import OptimizationSpec
from curve_fx_sim.specs.pair import PairSpec
from curve_fx_sim.specs.policy import PolicyParameter, PolicySpec
from curve_fx_sim.specs.scenario import ScenarioSpec
class MockHarnessClient(HarnessClient):
    """Deterministic in-memory test double for the curve_fx_eval_v1 harness client."""

    def __init__(self, policy: PolicySpec) -> None:
        self.policy = policy
        self.prepared = False
        self.session_open = False
        self.closed = False
        self.evaluated_batches: list[list[CandidateSpec]] = []

    def prepare(self) -> VerifiedEvaluator:
        self.prepared = True
        return VerifiedEvaluator(
            path="mock_arb_evaluator_ld",
            hello=HelloFrame(
                evaluator_identity=ProtocolEvaluatorIdentity(
                    binary_sha256="0" * 64,
                    harness_version="0.1.0",
                    pool_version="0.1.0",
                    policy_id=self.policy.id,
                    policy_source_sha256=self.policy.source_sha256 or "0" * 64,
                    policy_abi=self.policy.policy_abi,
                    policy_parameter_count=len(self.policy.parameters),
                    compiler="clang++",
                    numeric_mode="double",
                    real_type="double",
                    build_target="arb_evaluator_ld",
                    ipo_enabled=False,
                    native_tuning=False,
                ),
                metric_fields=[
                    "apy_net",
                    "apy_net_gm",
                    "avg_rel_price_diff",
                    "detach_energy_ungated",
                    "duration_s",
                    "max_7d_rel_price_diff",
                    "trades",
                    "tw_real_slippage_1pct",
                ],
            ),
        )

    def open_session(
        self,
        scenario_spec: ScenarioSpec,
        pair_spec: PairSpec | None = None,
        session_id: str | None = None,
    ) -> SessionReadyFrame:
        self.session_open = True
        return SessionReadyFrame(
            request_id="mock_req_1",
            session_id=session_id or "mock_sess_1",
            scenarios=[{"id": scenario_spec.id, "events_count": 0}],
            scenario_set_sha256="1" * 64,
            session_fingerprint="2" * 64,
            session_config_sha256="4" * 64,
            metric_schema_sha256="3" * 64,
        )

    def evaluate_batch(
        self,
        candidates: Sequence[CandidateSpec],
        observation: ObservationSpec | None = None,
    ) -> BatchResultFrame:
        self.evaluated_batches.append(list(candidates))
        results = []
        for cand in candidates:
            p_val = 0.0
            if isinstance(cand.policy_params, (list, tuple)) and cand.policy_params:
                p_val = float(cand.policy_params[0])
            elif isinstance(cand.policy_params, dict) and cand.policy_params:
                p_val = float(next(iter(cand.policy_params.values()), 0.0))

            metrics = {
                "apy_net": 0.05 + 0.001 * p_val,
                "apy_net_gm": 0.04 + 0.0008 * p_val,
                "max_7d_rel_price_diff": 0.02,
                "avg_rel_price_diff": 0.005,
                "tw_real_slippage_1pct": 0.0005,
                "detach_energy_ungated": 0.001,
                "duration_s": 86400.0,
                "trades": 100,
            }
            fingerprint = f"fp_{cand.candidate_id}"
            results.append(
                CandidateResult(
                    ordinal=cand.ordinal,
                    candidate_id=cand.candidate_id,
                    status="ok",
                    economic_fingerprint=fingerprint,
                    metrics=metrics,
                )
            )

        return BatchResultFrame(
            request_id="mock_batch_1",
            session_id="mock_sess_1",
            status="complete",
            results=results,
            elapsed_ms=1.0,
        )
    def close(self) -> None:
        self.closed = True

@pytest.fixture
def mock_specs(tmp_path: Path):
    store = RunStore(tmp_path)

    candles_file = tmp_path / "data" / "candles.json"
    candles_file.parent.mkdir(parents=True, exist_ok=True)
    candles_file.write_text("[]")
    pair = PairSpec(
        id="chfusd",
        name="CHF/USD Test",
        base_token="CHF",
        quote_token="USD",
    )
    scenario = ScenarioSpec(
        id="chfusd-smoke",
        pair_id="chfusd",
        name="CHF/USD Smoke",
    )
    opt_spec = OptimizationSpec(
        id="test-chfusd-opt",
        pair_id="chfusd",
        policy_id="compiled_test_policy",
        algorithm="tmrbcd",
        scenarios=("chfusd-smoke",),
        optimizer_config={"budget": 8, "batch_size": 4, "seed": 42},
        scoring_config={"score_key": "score_fx_lp_e15_slippage_v1"},
    )
    policy_header = tmp_path / "policies" / "compiled_test_policy.hpp"
    policy_header.parent.mkdir(parents=True, exist_ok=True)
    policy_header.write_text("// compiled test policy\n", encoding="utf-8")
    policy = PolicySpec(
        id="compiled_test_policy",
        header_file=Path("policies/compiled_test_policy.hpp"),
        source_sha256=sha256_path(policy_header),
        parameters=(
            PolicyParameter("weight", default=1.0, min_val=0.0, max_val=10.0, step=0.5),
            PolicyParameter("fixed", default=2.0, min_val=1.0, max_val=3.0, step=0.5),
        ),
    )
    return store, pair, scenario, policy, opt_spec
def test_run_optimization_full_lifecycle(tmp_path: Path, mock_specs, monkeypatch):
    store, pair, scenario, policy, opt_spec = mock_specs
    mock_client = MockHarnessClient(policy)

    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_pair_spec", lambda *args, **kwargs: pair)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_scenario_spec", lambda *args, **kwargs: scenario)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_policy_spec", lambda *args, **kwargs: policy)

    result = run_optimization(
        opt_spec,
        store=store,
        client=mock_client,
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )

    assert result.candidates_evaluated == 8
    assert len(result.table.rows) == 8
    assert result.manifest_path.is_file()
    assert result.table_path.is_file()
    assert result.winner_path.is_file()
    assert result.topk_path.is_file()
    journal_path = result.manifest_path.parent / "evaluation_journal.jsonl"
    checkpoint_path = result.manifest_path.parent / "checkpoint.json"
    assert journal_path.is_file()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 5
    assert "eval_rows" not in checkpoint
    assert "journal" not in checkpoint
    assert checkpoint["step"] == 8
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 8

    meta = result.table.metadata
    assert meta["yb_settings"]["score_key"] == "score_fx_lp_e15_slippage_v1"

    status = status_optimization(result.run_id, store=store, repository=tmp_path)
    assert status.status == "completed"
    assert status.candidates_evaluated == 8

    collected = collect_optimization(result.run_id, store=store, repository=tmp_path)
    assert collected.run_id == result.run_id
    assert len(collected.table.rows) == 8
    assert collected.winner.candidate_id == result.winner.candidate_id
    for artifact_path in (result.table_path, result.winner_path, result.topk_path):
        original = artifact_path.read_bytes()
        artifact_path.write_bytes(original + b"\n")
        with pytest.raises(SpecError, match="byte size"):
            collect_optimization(result.run_id, store=store, repository=tmp_path)
        artifact_path.write_bytes(original)

def test_resume_replays_journaled_overhang_without_reevaluation(
    tmp_path: Path,
    mock_specs,
    monkeypatch,
) -> None:
    """A crash between a journal append and the checkpoint save leaves the journal
    one batch ahead of the restored optimizer; resume re-drives that journaled
    tail instead of re-evaluating it, and reproduces the no-crash result."""
    store, pair, scenario, policy, opt_spec = mock_specs
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_pair_spec", lambda *args, **kwargs: pair)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_scenario_spec", lambda *args, **kwargs: scenario)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_policy_spec", lambda *args, **kwargs: policy)

    crashed = run_optimization(
        opt_spec,
        store=store,
        client=MockHarnessClient(policy),
        run_id="overhang_journal",
        budget=4,
        batch_size=4,
        repository=tmp_path,
    )
    journal_path = crashed.manifest_path.parent / "evaluation_journal.jsonl"
    checkpoint_path = crashed.manifest_path.parent / "checkpoint.json"
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 4
    for name in ("manifest.json", "evaluation_table.npz", "winner.json", "topk.json"):
        (crashed.manifest_path.parent / name).unlink()

    reference = run_optimization(
        opt_spec,
        store=store,
        client=MockHarnessClient(policy),
        run_id="overhang_reference",
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )
    ref_journal = reference.manifest_path.parent / "evaluation_journal.jsonl"
    ref_lines = ref_journal.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ordinal"] for line in ref_lines] == list(range(8))
    with journal_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(ref_lines[4:]) + "\n")

    resumed = run_optimization(
        opt_spec,
        store=store,
        client=MockHarnessClient(policy),
        run_id="overhang_journal",
        resume=True,
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )
    assert resumed.candidates_evaluated == 8
    assert [row.ordinal for row in resumed.table.rows] == list(range(8))
    assert [row.to_dict() for row in resumed.table.rows] == [
        row.to_dict() for row in reference.table.rows
    ]
    final_lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ordinal"] for line in final_lines] == list(range(8))
    final_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert final_checkpoint["step"] == 8
@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda checkpoint: checkpoint.pop("optimizer_state"), "optimizer_state must be an object"),
        (
            lambda checkpoint: checkpoint["optimizer_state"].pop("cache"),
            "optimizer_state fields mismatch",
        ),
        (
            lambda checkpoint: checkpoint["optimizer_state"].__setitem__(
                "schema_version", 2
            ),
            "optimizer_state schema_version",
        ),
        (
            lambda checkpoint: checkpoint["optimizer_state"].__setitem__(
                "step", checkpoint["optimizer_state"]["step"] - 1
            ),
            "evaluation cache",
        ),
    ],
)
def test_optimization_resume_rejects_invalid_optimizer_state(
    tmp_path: Path,
    mock_specs,
    monkeypatch,
    mutate,
    match: str,
) -> None:
    store, pair, scenario, policy, opt_spec = mock_specs
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_pair_spec", lambda *args, **kwargs: pair)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_scenario_spec", lambda *args, **kwargs: scenario)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_policy_spec", lambda *args, **kwargs: policy)
    run_id = "invalid_optimizer_state"
    result = run_optimization(
        opt_spec,
        store=store,
        client=MockHarnessClient(policy),
        run_id=run_id,
        budget=4,
        batch_size=4,
        repository=tmp_path,
    )
    checkpoint_path = result.manifest_path.parent / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    mutate(checkpoint)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        run_optimization(
            opt_spec,
            store=store,
            client=MockHarnessClient(policy),
            run_id=run_id,
            resume=True,
            budget=8,
            batch_size=4,
            repository=tmp_path,
        )


def test_blade_work_bundle_preserves_primary_metrics_and_evaluation_status(
    tmp_path: Path,
    mock_specs,
) -> None:
    _store, pair, scenario, policy, opt_spec = mock_specs
    client = MockHarnessClient(policy)
    identity = client.prepare()
    profile = profile_from_policy_spec(policy)
    bundle = create_work_bundle(
        run_id="worker-contract",
        optimization_id=opt_spec.id,
        island_id="blade-b6",
        step=0,
        profile=profile,
        pair_spec=pair,
        scenarios=(scenario,),
        evaluator_identity=identity.to_dict(),
        yb_settings={"require_yb": False},
        score_key="score_fx_lp_e15_slippage_v1",
        proposals=({"ordinal": 0, "ask_id": "ask_000000", "params": [1.0, 2.0]},),
    )

    result = evaluate_work_bundle(bundle, client)

    assert result.result_version == BUNDLE_RESULT_SCHEMA_VERSION
    assert result.results[0]["status"] == "ok"
    assert result.results[0]["primary_metrics"]["apy_net"] == pytest.approx(0.051)
    result_path = tmp_path / "bundle-result.json"
    result.to_json(result_path)
    assert OptimizationBundleResult.from_json(result_path) == result
def test_shared_nfs_bundle_skips_per_bundle_transfers(tmp_path: Path, mock_specs, monkeypatch) -> None:
    store, pair, scenario, policy, opt_spec = mock_specs
    identity = MockHarnessClient(policy).prepare()
    profile = profile_from_policy_spec(policy)
    bundle = create_work_bundle(
        run_id="shared-opt", optimization_id=opt_spec.id, island_id="blade-a1", step=0,
        profile=profile, pair_spec=pair, scenarios=(scenario,),
        evaluator_identity=identity.to_dict(), yb_settings={"require_yb": False},
        score_key="score_fx_lp_e15_slippage_v1",
        proposals=({"ordinal": 0, "ask_id": "ask_0", "params": [1.0, 2.0]},),
    )
    site = SiteProfile(
        name="nfs", site_type="ssh",
        cluster=ClusterConfig(
            coordinator="blade-b6", transport="shared_nfs",
            remote_base=PurePosixPath(tmp_path),
            remote_run_root=PurePosixPath(store.runs_dir),
            repository_root=PurePosixPath("/srv/fx"), worker_command="/srv/fx/bin/worker",
            blades=("blade-b6", "blade-a1"),
        ),
    )
    class FakeSSH:
        uploads: list[object] = []
        downloads: list[object] = []
        def __init__(self, **_kwargs: object) -> None: pass
        def run_ssh(self, blade: str, command: str, **_kwargs: object) -> ProcessResult:
            out = Path(command.split("--out ", 1)[1].split()[0].strip("'\""))
            OptimizationBundleResult(
                bundle_id=bundle.bundle_id, bundle_sha256=bundle.bundle_sha256,
                island_id=bundle.island_id,
                results=({"ask_id": "ask_0"},), elapsed_ms=0.0,
            ).to_json(out)
            return ProcessResult(0, "", "")
        def rsync_upload(self, *args: object, **_kwargs: object) -> ProcessResult:
            self.uploads.append(args); return ProcessResult(0, "", "")
        def rsync_download(self, *args: object, **_kwargs: object) -> ProcessResult:
            self.downloads.append(args); return ProcessResult(0, "", "")
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.SSHProcessAdapter", FakeSSH)
    result = _dispatch_remote_bundle(
        bundle, run_dir=store.runs_dir / bundle.run_id, site=site, blade="blade-a1",
        expected_ask_ids=frozenset({"ask_0"}),
    )
    assert result.results[0]["ask_id"] == "ask_0"
    assert not FakeSSH.uploads and not FakeSSH.downloads
def test_resume_crash_boundary_checkpoint_never_ahead_of_journal(
    tmp_path: Path,
    mock_specs,
    monkeypatch,
) -> None:
    """Crash boundary between a journal append and its checkpoint save.

    Two crash states must both resume correctly:
      * journal bytes persisted but checkpoint stale -- the journal is one batch
        ahead and resume re-drives the journaled tail without re-evaluation;
      * checkpoint written after a flushed journal -- durable journal rows and
        checkpoint step agree, so resume accepts it (the flush ordering
        guarantees the checkpoint is never ahead of the journal).
    Every checkpoint write must observe journal rows >= its step, and a
    checkpoint must never exceed the durable journal rows after the flush.
    """
    store, pair, scenario, policy, opt_spec = mock_specs
    monkeypatch.setattr(
        "curve_fx_sim.optimization.runtime.load_pair_spec", lambda *args, **kwargs: pair
    )
    monkeypatch.setattr(
        "curve_fx_sim.optimization.runtime.load_scenario_spec",
        lambda *args, **kwargs: scenario,
    )
    monkeypatch.setattr(
        "curve_fx_sim.optimization.runtime.load_policy_spec",
        lambda *args, **kwargs: policy,
    )

    def _guarded_atomic_write_json(path, payload, **kwargs):
        dest = Path(path)
        if dest.name == "checkpoint.json" and isinstance(payload, Mapping) and "step" in payload:
            journal_path = dest.parent / EVALUATION_JOURNAL_FILENAME
            durable_rows = 0
            if journal_path.is_file():
                durable_rows = len(
                    [
                        line
                        for line in journal_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                )
            assert payload["step"] <= durable_rows, (
                f"checkpoint step {payload['step']} exceeds durable journal rows {durable_rows}"
            )
        return atomic_write_json(path, payload, **kwargs)

    monkeypatch.setattr(
        "curve_fx_sim.optimization.runtime.atomic_write_json",
        _guarded_atomic_write_json,
    )

    reference = run_optimization(
        opt_spec,
        store=store,
        client=MockHarnessClient(policy),
        run_id="crash_boundary_reference",
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )
    ref_lines = (reference.manifest_path.parent / EVALUATION_JOURNAL_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line)["ordinal"] for line in ref_lines] == list(range(8))

    def _crashed_run(run_id: str, budget: int) -> Path:
        result = run_optimization(
            opt_spec,
            store=store,
            client=MockHarnessClient(policy),
            run_id=run_id,
            budget=budget,
            batch_size=4,
            repository=tmp_path,
        )
        run_dir = result.manifest_path.parent
        for name in ("manifest.json", "evaluation_table.npz", "winner.json", "topk.json"):
            (run_dir / name).unlink()
        return run_dir

    run_dir = _crashed_run("crash_boundary_a", budget=4)
    journal_path = run_dir / EVALUATION_JOURNAL_FILENAME
    checkpoint_path = run_dir / "checkpoint.json"
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["step"] == 4
    with journal_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(ref_lines[4:]) + "\n")
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 8

    resumed_a_client = MockHarnessClient(policy)
    resumed_a = run_optimization(
        opt_spec,
        store=store,
        client=resumed_a_client,
        run_id="crash_boundary_a",
        resume=True,
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )
    assert resumed_a.candidates_evaluated == 8
    assert [row.ordinal for row in resumed_a.table.rows] == list(range(8))
    assert resumed_a_client.evaluated_batches == []
    assert [row.to_dict() for row in resumed_a.table.rows] == [
        row.to_dict() for row in reference.table.rows
    ]
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["step"] == 8
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 8
