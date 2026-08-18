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
from curve_fx_sim.evaluation.client import HarnessClient, ScenarioHarnessClient
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
from curve_fx_sim.evaluation.selection import SelectionRef
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
            # Deterministic synthetic response derived from parameters
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
            # Fingerprint includes candidate_id
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


class FailingOwnedHarnessClient(MockHarnessClient, ScenarioHarnessClient):
    def evaluate_batch(
        self,
        candidates: Sequence[CandidateSpec],
        observation: ObservationSpec | None = None,
    ) -> BatchResultFrame:
        raise RuntimeError("injected evaluation failure")


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


def test_owned_scenario_client_closes_after_failure(
    tmp_path: Path, mock_specs, monkeypatch
) -> None:
    store, pair, scenario, policy, opt_spec = mock_specs
    mock_client = FailingOwnedHarnessClient(policy)
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

    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        run_optimization(
            opt_spec,
            store=store,
            client=mock_client,
            budget=4,
            batch_size=4,
            repository=tmp_path,
        )
    assert mock_client.closed



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
    # The journal is a plain JSONL row log: one row per evaluated candidate.
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 8
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["ordinal"] in range(8)

    # Verify table metadata has complete immutable replay inputs
    meta = result.table.metadata
    assert meta["policy_id"] == "compiled_test_policy"
    assert "policy_source_sha256" in meta
    assert "evaluator_identity" in meta
    assert "yb_settings" in meta
    assert meta["yb_settings"]["score_key"] == "score_fx_lp_e15_slippage_v1"
    assert result.table.metric_projection is not None
    assert result.table.metric_projection.projection_id == "optimization-primary-scenario"
    assert result.table.rows[0].metrics["apy_net"] > 0.0

    # Verify winner SelectionRef structure, exact candidate_id, and lineage
    assert result.winner.kind == "optimizer_winner"
    assert result.winner.candidate_id is not None
    assert result.winner.candidate_id.endswith(scenario.id)
    assert result.winner.run_id == result.run_id
    assert result.winner.coordinate is not None
    assert "algorithm" in result.winner.coordinate

    # Verify status and collect queries
    status = status_optimization(result.run_id, store=store, repository=tmp_path)
    assert status.status == "completed"
    assert status.candidates_evaluated == 8

    collected = collect_optimization(result.run_id, store=store, repository=tmp_path)
    assert collected.run_id == result.run_id
    assert len(collected.table.rows) == 8
    assert collected.winner.candidate_id == result.winner.candidate_id
    assert collected.spec.scenarios == opt_spec.scenarios
    assert collected.spec.optimizer_config == opt_spec.optimizer_config
    assert collected.spec.scoring_config == opt_spec.scoring_config

    for artifact_path in (result.table_path, result.winner_path, result.topk_path):
        original = artifact_path.read_bytes()
        artifact_path.write_bytes(original + b"\n")
        with pytest.raises(SpecError, match="byte size"):
            collect_optimization(result.run_id, store=store, repository=tmp_path)
        artifact_path.write_bytes(original)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["resolved_spec"]["scenarios"] == list(opt_spec.scenarios)
    assert manifest["resolved_spec"]["scenario_specs"] == [scenario.to_dict()]

    run_dir_path = result.manifest_path.parent
    assert status_optimization(run_dir_path, store=store, repository=tmp_path).run_id == result.run_id
    assert collect_optimization(run_dir_path, store=store, repository=tmp_path).run_id == result.run_id


def test_optimization_resume_full_lifecycle(tmp_path: Path, mock_specs, monkeypatch):
    """Verify that an interrupted run resumed with resume=True preserves all earlier rows and produces complete table & winner."""
    store, pair, scenario, policy, opt_spec = mock_specs
    mock_client = MockHarnessClient(policy)

    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_pair_spec", lambda *args, **kwargs: pair)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_scenario_spec", lambda *args, **kwargs: scenario)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_policy_spec", lambda *args, **kwargs: policy)

    run_id = "test_resume_run_01"

    # 1. Step 1: Run with budget=4 (interrupts after 4 steps)
    res_part1 = run_optimization(
        opt_spec,
        store=store,
        client=mock_client,
        run_id=run_id,
        budget=4,
        batch_size=4,
        repository=tmp_path,
    )
    assert res_part1.candidates_evaluated == 4
    assert len(res_part1.table.rows) == 4

    # 2. Step 2: Resume with budget=8 (should evaluate remaining 4 steps and retain all 8 rows)
    res_resumed = run_optimization(
        opt_spec,
        store=store,
        client=mock_client,
        run_id=run_id,
        resume=True,
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )

    assert res_resumed.candidates_evaluated == 8
    assert len(res_resumed.table.rows) == 8

    # All candidate IDs from c_00000_chfusd-smoke to c_00007_chfusd-smoke are present in table
    candidate_ids = [r.candidate_id for r in res_resumed.table.rows]
    for i in range(8):
        assert f"c_{i:05d}_{scenario.id}" in candidate_ids

    # Winner is chosen from all 8 rows and has exact replay candidate_id
    assert res_resumed.winner.candidate_id in candidate_ids
    assert len(res_resumed.top_k) == min(8, len(res_resumed.table.rows))
    assert res_resumed.winner.coordinate is not None

    artifact_bytes = {
        path: path.read_bytes()
        for path in (
            res_resumed.manifest_path,
            res_resumed.table_path,
            res_resumed.winner_path,
            res_resumed.topk_path,
            res_resumed.manifest_path.parent / "checkpoint.json",
            res_resumed.manifest_path.parent / "evaluation_journal.jsonl",
        )
    }
    complete_resume = run_optimization(
        opt_spec,
        store=store,
        client=mock_client,
        run_id=run_id,
        resume=True,
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )
    assert complete_resume.candidates_evaluated == 8
    assert {path: path.read_bytes() for path in artifact_bytes} == artifact_bytes

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
    # A run that crashes after a journal append but before its checkpoint save
    # has not reached finalization, so none of the published artifacts exist.
    for name in ("manifest.json", "evaluation_table.npz", "winner.json", "topk.json"):
        (crashed.manifest_path.parent / name).unlink()

    # A sibling full run shares the deterministic ask trajectory, so its batch
    # with ordinals 4-7 is exactly what the crashed run journaled but failed to
    # checkpoint.  Append it to simulate a journal append whose checkpoint save
    # was lost to the crash.
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
    # The overhang rows were skipped, not re-appended, and the recovered run
    # reproduces the no-crash table exactly.
    assert [row.to_dict() for row in resumed.table.rows] == [
        row.to_dict() for row in reference.table.rows
    ]
    final_lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ordinal"] for line in final_lines] == list(range(8))
    # The replayed tail exhausted the budget without the loop running, so the
    # checkpoint must be refreshed to the caught-up step before publication.
    final_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert final_checkpoint["step"] == 8


def test_resume_rejects_partially_written_journal_row(
    tmp_path: Path,
    mock_specs,
    monkeypatch,
) -> None:
    """A journal row cut mid-write fails naturally on resume (plain JSONL)."""
    store, pair, scenario, policy, opt_spec = mock_specs
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_pair_spec", lambda *args, **kwargs: pair)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_scenario_spec", lambda *args, **kwargs: scenario)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_policy_spec", lambda *args, **kwargs: policy)
    result = run_optimization(
        opt_spec,
        store=store,
        client=MockHarnessClient(policy),
        run_id="broken_journal",
        budget=4,
        batch_size=4,
        repository=tmp_path,
    )
    journal_path = result.manifest_path.parent / "evaluation_journal.jsonl"
    journal_path.write_bytes(journal_path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="malformed row"):
        run_optimization(
            opt_spec,
            store=store,
            client=MockHarnessClient(policy),
            run_id="broken_journal",
            resume=True,
            budget=8,
            batch_size=4,
            repository=tmp_path,
        )



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


def test_objective_gate_does_not_mark_successful_evaluation_failed(
    tmp_path: Path,
    mock_specs,
    monkeypatch,
) -> None:
    store, pair, scenario, policy, opt_spec = mock_specs
    mock_client = MockHarnessClient(policy)

    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_pair_spec", lambda *args, **kwargs: pair)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_scenario_spec", lambda *args, **kwargs: scenario)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_policy_spec", lambda *args, **kwargs: policy)
    monkeypatch.setattr(
        "curve_fx_sim.optimization.runtime.score_scenarios",
        lambda *args, **kwargs: {
            "gate": False,
            "score_fx_lp_e15_slippage_v1": -1.0,
            "n_failed": 0,
        },
    )

    result = run_optimization(
        opt_spec,
        store=store,
        client=mock_client,
        budget=1,
        batch_size=1,
        repository=tmp_path,
    )

    assert result.table.rows[0].metrics["gate"] is False
    assert result.table.rows[0].status == "ok"


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


def test_blade_work_bundle_splits_policy_and_pool_dims_per_candidate(
    tmp_path: Path,
    mock_specs,
) -> None:
    """Distributed seam: per-proposal pool dims reach the evaluator as
    pool_overrides while policy_params stays the exact ABI vector."""
    _store, pair, scenario, _policy, opt_spec = mock_specs
    policy = PolicySpec(
        id="pool_dim_policy",
        header_file="policies/pool_dim_policy.hpp",
        source_sha256="a" * 64,
        parameters=(
            PolicyParameter("p0", default=1.5, min_val=1.0, max_val=2.0, step=0.1),
            PolicyParameter("p1", default=5.0, min_val=0.0, max_val=10.0, step=1.0),
        ),
    )
    template_json = {
        "pools": [
            {
                "pool": {
                    "out_fee": "20000000",
                    "donation_apy": "0.049",
                    "donation_frequency": "3600",
                    "donation_duration": "604800",
                    "donation_coins_ratio": "0.5",
                }
            }
        ]
    }
    profile = profile_from_policy_spec(
        policy,
        {
            "out_fee": {"min": 0.001, "max": 0.005, "step": 0.0005},
        },
        template_json=template_json,
    )
    client = MockHarnessClient(policy)
    identity = client.prepare()
    bundle = create_work_bundle(
        run_id="worker-pool-dims",
        optimization_id=opt_spec.id,
        island_id="blade-b6",
        step=0,
        profile=profile,
        pair_spec=pair,
        scenarios=(scenario,),
        evaluator_identity=identity.to_dict(),
        yb_settings={"require_yb": False},
        score_key="score_fx_lp_e15_slippage_v1",
        proposals=(
            {"ordinal": 0, "ask_id": "ask_000000", "params": [1.5, 5.0, 0.0015]},
            {"ordinal": 1, "ask_id": "ask_000001", "params": [1.5, 5.0, 0.0045]},
        ),
        pool_overrides={},
    )

    result = evaluate_work_bundle(bundle, client)

    assert result.results[0]["status"] == "ok"
    assert result.results[1]["status"] == "ok"
    # Bundle-level default is not shadowed by the proposal loop.
    assert bundle.pool_overrides == {}
    batches = client.evaluated_batches
    assert len(batches) == 1
    candidates = batches[0]
    assert [c.policy_params for c in candidates] == [[1.5, 5.0], [1.5, 5.0]]
    assert [c.pool_overrides for c in candidates] == [
        {"out_fee": 15000000},
        {"out_fee": 45000000},
    ]


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

    # Guard every checkpoint write: after the journal flush, the durable JSONL
    # must carry at least as many rows as the checkpoint's step.
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

    # Deterministic sibling run supplies the exact rows a crashed run journals.
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
        # Crash before finalization: no published artifacts survive.
        for name in ("manifest.json", "evaluation_table.npz", "winner.json", "topk.json"):
            (run_dir / name).unlink()
        return run_dir

    # Boundary A: journal bytes persisted (8 rows), checkpoint stale (step=4).
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
    # The tail was replayed from the journal, not re-evaluated.
    assert resumed_a_client.evaluated_batches == []
    assert [row.to_dict() for row in resumed_a.table.rows] == [
        row.to_dict() for row in reference.table.rows
    ]
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["step"] == 8
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 8

    # Boundary B: checkpoint written after a flushed journal -- durable journal
    # rows and checkpoint step agree (never ahead), then resume.
    run_dir_b = _crashed_run("crash_boundary_b", budget=8)
    journal_path_b = run_dir_b / EVALUATION_JOURNAL_FILENAME
    checkpoint_path_b = run_dir_b / "checkpoint.json"
    assert len(journal_path_b.read_text(encoding="utf-8").splitlines()) == 8
    assert json.loads(checkpoint_path_b.read_text(encoding="utf-8"))["step"] == 8

    resumed_b_client = MockHarnessClient(policy)
    resumed_b = run_optimization(
        opt_spec,
        store=store,
        client=resumed_b_client,
        run_id="crash_boundary_b",
        resume=True,
        budget=8,
        batch_size=4,
        repository=tmp_path,
    )
    assert resumed_b.candidates_evaluated == 8
    assert [row.ordinal for row in resumed_b.table.rows] == list(range(8))
    # Nothing to replay and nothing to re-evaluate.
    assert resumed_b_client.evaluated_batches == []
    assert [row.to_dict() for row in resumed_b.table.rows] == [
        row.to_dict() for row in reference.table.rows
    ]
    assert json.loads(checkpoint_path_b.read_text(encoding="utf-8"))["step"] == 8
    assert len(journal_path_b.read_text(encoding="utf-8").splitlines()) == 8
