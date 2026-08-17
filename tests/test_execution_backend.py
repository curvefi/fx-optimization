"""Tests for unified ExecutionBackend, persistent evaluator reuse, and resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
from curve_fx_harness_client.models import (
    BatchResultFrame,
    CandidateResult,
    CandidateSpec,
    HelloFrame,
    SessionReadyFrame,
)

from curve_fx_sim.artifacts.manifest import new_grid_manifest, write_manifest_atomic
from curve_fx_sim.execution.backend import ExecutionBackend, ExecutionBackendError
from curve_fx_sim.execution.collection import (
    CollectionError,
    grid_request_set_sha256,
    validate_shards,
    write_shard_result,
)
from curve_fx_sim.execution.sharding import make_assignments
from curve_fx_sim.execution.site import SiteProfile


class MockBatchEvaluatorClient:
    def __init__(self, blade: str, policy_identity: str) -> None:
        self.blade = blade
        self.policy_identity = policy_identity
        self.session_opened = False
        self.current_session_id: str | None = None
        self.batches_evaluated: list[list[dict[str, Any]]] = []
        self.open_session_kwargs: dict[str, Any] = {}
        self.shutdown_called = False
        self.hello = HelloFrame(
            evaluator_identity={
                "binary_sha256": "a" * 64,
                "harness_version": "1.0.0",
                "pool_version": "0.1.0",
                "policy_id": policy_identity,
                "policy_source_sha256": "b" * 64,
                "policy_abi": "twocrypto_policy_v1",
                "policy_parameter_count": 0,
                "numeric_mode": "double",
                "real_type": "double",
                "compiler": "clang++",
                "build_target": "arb_evaluator_ld",
                "ipo_enabled": False,
                "native_tuning": False,
            },
            metric_fields=["score"],
        )

    def start(self) -> HelloFrame:
        return self.hello

    def shutdown(self) -> None:
        self.shutdown_called = True

    def open_session(self, session_id: str, **kwargs: Any) -> SessionReadyFrame:
        self.session_opened = True
        self.current_session_id = session_id
        self.open_session_kwargs = dict(kwargs)
        return SessionReadyFrame(
            request_id="req-test",
            session_id=session_id,
            scenarios=[],
            scenario_set_sha256="1" * 64,
            session_fingerprint="2" * 64,
            session_config_sha256="3" * 64,
            metric_schema_sha256="4" * 64,
        )

    def evaluate_batch(
        self,
        candidates: Sequence[CandidateSpec],
    ) -> BatchResultFrame:
        cand_list = list(candidates)
        self.batches_evaluated.append(
            [candidate.model_dump() for candidate in cand_list]
        )
        assert self.current_session_id is not None
        return BatchResultFrame(
            request_id="batch-test",
            session_id=self.current_session_id,
            status="complete",
            results=[
                CandidateResult(
                    ordinal=candidate.ordinal,
                    candidate_id=candidate.candidate_id,
                    status="ok",
                    economic_fingerprint=f"fp_{candidate.ordinal}",
                    metrics={"score": 100.0 + float(candidate.ordinal)},
                )
                for candidate in cand_list
            ],
            elapsed_ms=0.0,
        )


def _manifest(run_id: str, count: int, policy: str = "test_policy_v1") -> dict[str, Any]:
    return new_grid_manifest(
        run_id=run_id,
        grid_id="test_grid",
        pool_count=count,
        resolved_spec={
            "policy": {"id": policy},
            "scenario": {
                "id": "scenario",
                "pair_id": "pair",
                "template_path": "template.json",
                "market_files": [{"path": "data/candles.json"}],
                "dustswap_freq_s": 777,
                "yb_releverage": True,
                "yb_cash_multiplier": 3.0,
            },
            "metric_projection": {"fields": ["score"], "projection_id": "grid"},
        },
        resolved_axes=[],
        pools=[{"id": f"pool_{i}", "ordinal": i, "policy_params": [], "pool_overrides": {}} for i in range(count)],
        core={
            "schema_version": "curve_fx_sim_identity_v2",
            "binary": "arb_evaluator_ld",
            "sha256": "a" * 64,
            "harness_version": "1.0.0",
            "pool_version": "0.1.0",
            "policy_id": policy,
            "policy_source_sha256": "b" * 64,
            "policy_abi": "twocrypto_policy_v1",
            "policy_parameter_count": 0,
            "numeric_mode": "double",
            "real_type": "double",
            "compiler": "clang++",
            "build_target": "arb_evaluator_ld",
            "metric_schema": "twocrypto-summary-v1",
            "ipo_enabled": False,
            "native_tuning": False,
            "metric_fields": ["score"],
        },
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    root = path.parent.parent
    (root / "data").mkdir(exist_ok=True)
    (root / "template.json").write_text("{}", encoding="utf-8")
    (root / "data" / "candles.json").write_text("[]", encoding="utf-8")
    write_manifest_atomic(path, manifest, expected_kind="grid")


def test_execution_backend_persistent_evaluator_reuse(tmp_path: Path) -> None:
    run_dir = tmp_path / "test_grid_persistent"
    run_dir.mkdir()
    manifest_file = run_dir / "manifest.json"
    _write_manifest(manifest_file, _manifest("test_grid_persistent", 10))
    created_clients: list[MockBatchEvaluatorClient] = []

    def mock_factory(blade: str, policy_id: str) -> MockBatchEvaluatorClient:
        client = MockBatchEvaluatorClient(blade, policy_id)
        created_clients.append(client)
        return client

    backend = ExecutionBackend(site_profile=SiteProfile(name="local", site_type="local"), client_factory=mock_factory)
    summary = backend.run_grid(manifest_file, chunk_size=2)
    assert summary.status == "succeeded"
    assert summary.total_pools == 10
    assert summary.executed_shards == 5
    assert len(created_clients) == 5
    assert sum(len(client.batches_evaluated) for client in created_clients) == 5
    assert all(client.open_session_kwargs["dustswap_freq_s"] == 777 for client in created_clients)
    assert all(client.open_session_kwargs["yb_releverage"] is True for client in created_clients)
    assert all(client.open_session_kwargs["yb_cash_multiplier"] == 3.0 for client in created_clients)
    disk_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert len(disk_manifest["grid"]["shards"]) == 5
    assert disk_manifest["scope"] == "local"
    assert len(disk_manifest["attempt_history"]) == 1
    assert disk_manifest["attempt_history"][0]["status"] == "succeeded"


def test_execution_backend_resume_skips_completed_shards(tmp_path: Path) -> None:
    run_dir = tmp_path / "test_grid_resume"
    run_dir.mkdir()
    (run_dir / "results").mkdir()
    manifest = _manifest("test_grid_resume", 6)
    assignments = make_assignments(6, ["local"], chunk_size=2, run_id="test_grid_resume")
    manifest["grid"]["shards"] = [assignment.to_dict() for assignment in assignments]
    _write_manifest(run_dir / "manifest.json", manifest)
    session_attestation = {
        "session_id": "sess_test_grid_resume_local",
        "scenario_set_sha256": "1" * 64,
        "session_fingerprint": "2" * 64,
        "session_config_sha256": "3" * 64,
        "metric_schema_sha256": "4" * 64,
    }
    request_set_sha256 = grid_request_set_sha256(manifest, session_attestation)
    write_shard_result(
        run_dir / "results" / f"{assignments[0].shard_id}.json",
        run_id="test_grid_resume",
        shard_id=assignments[0].shard_id,
        shard_index=assignments[0].shard_index,
        ranges=assignments[0].ranges,
        rows=[
            {"pool_index": 0, "ordinal": 0, "candidate_id": "pool_0", "status": "ok", "metrics": {"score": 100.0}, "economic_fingerprint": "fp0"},
            {"pool_index": 1, "ordinal": 1, "candidate_id": "pool_1", "status": "ok", "metrics": {"score": 101.0}, "economic_fingerprint": "fp1"},
        ],
        request_set_sha256=request_set_sha256,
        session_attestation=session_attestation,
    )
    created_clients: list[MockBatchEvaluatorClient] = []

    def mock_factory(blade: str, policy_id: str) -> MockBatchEvaluatorClient:
        client = MockBatchEvaluatorClient(blade, policy_id)
        created_clients.append(client)
        return client

    backend = ExecutionBackend(site_profile=SiteProfile(name="local", site_type="local"), client_factory=mock_factory)
    summary = backend.run_grid(run_dir / "manifest.json", chunk_size=2, resume=True)
    assert summary.skipped_shards == 1
    assert summary.executed_shards == 2
    assert [
        candidate["ordinal"]
        for client in created_clients
        for batch in client.batches_evaluated
        for candidate in batch
    ] == [2, 3, 4, 5]


def test_execution_backend_never_reuses_process_across_runs(tmp_path: Path) -> None:
    created_clients: list[MockBatchEvaluatorClient] = []

    def mock_factory(blade: str, policy_id: str) -> MockBatchEvaluatorClient:
        client = MockBatchEvaluatorClient(blade, policy_id)
        created_clients.append(client)
        return client

    backend = ExecutionBackend(
        site_profile=SiteProfile(name="local", site_type="local"),
        client_factory=mock_factory,
    )
    for run_id, scenario_id in (("run_A", "scenario_A"), ("run_B", "scenario_B")):
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        manifest = _manifest(run_id, 1)
        manifest["resolved_spec"]["scenario"]["id"] = scenario_id
        manifest_file = run_dir / "manifest.json"
        _write_manifest(manifest_file, manifest)
        backend.run_grid(manifest_file, chunk_size=1)

    assert len(created_clients) == 2
    assert created_clients[0] is not created_clients[1]
    assert created_clients[0].current_session_id == "sess_run_A_local"
    assert created_clients[1].current_session_id == "sess_run_B_local"


def test_is_shard_complete_checks_row_count_only(tmp_path: Path) -> None:
    """_is_shard_complete treats a receipt as complete by row count alone,
    without per-shard digest/identity re-verification."""
    from curve_fx_sim.execution.backend import _is_shard_complete

    run_dir = tmp_path / "count_only"
    run_dir.mkdir()
    (run_dir / "results").mkdir()
    manifest = _manifest("count_only", 2)
    assignments = make_assignments(2, ["local"], chunk_size=2, run_id="count_only")
    manifest["grid"]["shards"] = [assignment.to_dict() for assignment in assignments]
    _write_manifest(run_dir / "manifest.json", manifest)
    session_attestation = {
        "session_id": "sess_count_only_local",
        "scenario_set_sha256": "1" * 64,
        "session_fingerprint": "2" * 64,
        "session_config_sha256": "3" * 64,
        "metric_schema_sha256": "4" * 64,
    }
    receipt = run_dir / "results" / f"{assignments[0].shard_id}.json"
    write_shard_result(
        receipt,
        run_id="count_only",
        shard_id=assignments[0].shard_id,
        shard_index=assignments[0].shard_index,
        ranges=assignments[0].ranges,
        rows=[
            {"pool_index": 0, "ordinal": 0, "candidate_id": "pool_0", "status": "ok", "metrics": {"score": 100.0}, "economic_fingerprint": "fp0"},
            {"pool_index": 1, "ordinal": 1, "candidate_id": "pool_1", "status": "ok", "metrics": {"score": 101.0}, "economic_fingerprint": "fp1"},
        ],
        request_set_sha256=grid_request_set_sha256(manifest, session_attestation),
        session_attestation=session_attestation,
    )
    assert _is_shard_complete(receipt, assignments[0].ranges) is True
    # The candidate set changed after the receipt was written; the count-only
    # check still considers the shard complete.
    manifest["grid"]["pools"][0]["policy_params"] = [1.0]
    assert _is_shard_complete(receipt, assignments[0].ranges) is True
    # A missing receipt or a mismatched row count is not complete.
    assert _is_shard_complete(receipt.with_name("missing.json"), assignments[0].ranges) is False
    with receipt.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["rows"].pop()
    payload["row_count"] = len(payload["rows"])
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    assert _is_shard_complete(receipt, assignments[0].ranges) is False


def test_resume_reexecutes_shard_with_mismatched_row_count(tmp_path: Path) -> None:
    """A shard receipt whose row count does not match the shard ranges is not
    complete and is re-executed on resume."""
    run_dir = tmp_path / "incomplete_resume"
    run_dir.mkdir()
    (run_dir / "results").mkdir()
    manifest = _manifest("incomplete_resume", 2)
    assignments = make_assignments(2, ["local"], chunk_size=2, run_id="incomplete_resume")
    manifest["grid"]["shards"] = [assignment.to_dict() for assignment in assignments]
    _write_manifest(run_dir / "manifest.json", manifest)
    session_attestation = {
        "session_id": "sess_incomplete_resume_local",
        "scenario_set_sha256": "1" * 64,
        "session_fingerprint": "2" * 64,
        "session_config_sha256": "3" * 64,
        "metric_schema_sha256": "4" * 64,
    }
    # Only one of the two expected rows survived; the receipt is incomplete.
    write_shard_result(
        run_dir / "results" / f"{assignments[0].shard_id}.json",
        run_id="incomplete_resume",
        shard_id=assignments[0].shard_id,
        shard_index=assignments[0].shard_index,
        ranges=assignments[0].ranges,
        rows=[
            {"pool_index": 0, "ordinal": 0, "candidate_id": "pool_0", "status": "ok", "metrics": {"score": 100.0}, "economic_fingerprint": "fp0"},
        ],
        request_set_sha256=grid_request_set_sha256(manifest, session_attestation),
        session_attestation=session_attestation,
    )
    created_clients: list[MockBatchEvaluatorClient] = []
    backend = ExecutionBackend(
        site_profile=SiteProfile(name="local", site_type="local"),
        client_factory=lambda blade, policy: created_clients.append(
            MockBatchEvaluatorClient(blade, policy)
        )
        or created_clients[-1],
    )
    summary = backend.run_grid(run_dir / "manifest.json", chunk_size=2, resume=True)
    assert summary.skipped_shards == 0
    assert summary.executed_shards == 1
    assert [
        candidate["ordinal"]
        for client in created_clients
        for batch in client.batches_evaluated
        for candidate in batch
    ] == [0, 1]


def test_same_run_cannot_reuse_session_after_scenario_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "immutable_session"
    run_dir.mkdir()
    manifest_file = run_dir / "manifest.json"
    _write_manifest(manifest_file, _manifest("immutable_session", 1))
    backend = ExecutionBackend(
        client_factory=lambda blade, policy: MockBatchEvaluatorClient(blade, policy)
    )
    backend.run_grid(manifest_file, chunk_size=1)

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["resolved_spec"]["scenario"]["dustswap_freq_s"] = 778
    write_manifest_atomic(manifest_file, manifest, expected_kind="grid")
    with pytest.raises(ExecutionBackendError, match="opened for different inputs"):
        backend.run_grid(manifest_file, chunk_size=1, resume=True)


def test_collection_rejects_cross_blade_session_drift(tmp_path: Path) -> None:
    manifest = _manifest("cross_blade", 2)
    assignments = make_assignments(
        2,
        ["blade-a1", "blade-a2"],
        chunk_size=1,
        run_id="cross_blade",
    )
    manifest["grid"]["shards"] = [assignment.to_dict() for assignment in assignments]
    results_dir = tmp_path / "results"
    for assignment, fingerprint in zip(assignments, ("2" * 64, "9" * 64), strict=True):
        attestation = {
            "session_id": f"sess_cross_blade_{assignment.blade}",
            "scenario_set_sha256": "1" * 64,
            "session_fingerprint": fingerprint,
            "session_config_sha256": "3" * 64,
            "metric_schema_sha256": "4" * 64,
        }
        write_shard_result(
            results_dir / f"{assignment.shard_id}.json",
            run_id="cross_blade",
            shard_id=assignment.shard_id,
            shard_index=assignment.shard_index,
            ranges=assignment.ranges,
            rows=[
                {
                    "pool_index": assignment.shard_index,
                    "ordinal": assignment.shard_index,
                    "candidate_id": f"pool_{assignment.shard_index}",
                    "status": "ok",
                    "economic_fingerprint": "f" * 64,
                    "metrics": {"score": 1.0},
                }
            ],
            request_set_sha256=grid_request_set_sha256(manifest, attestation),
            session_attestation=attestation,
        )

    with pytest.raises(CollectionError, match="different opened session"):
        validate_shards(manifest, results_dir)


def test_dispatch_candidates_adaptive_optimizer(tmp_path: Path) -> None:
    run_dir = tmp_path / "test_opt_dispatch"
    run_dir.mkdir()
    manifest_file = run_dir / "manifest.json"
    _write_manifest(manifest_file, _manifest("test_opt_dispatch", 1, "opt_policy_v1"))
    created_clients: list[MockBatchEvaluatorClient] = []

    def mock_factory(blade: str, policy_id: str) -> MockBatchEvaluatorClient:
        client = MockBatchEvaluatorClient(blade, policy_id)
        created_clients.append(client)
        return client

    backend = ExecutionBackend(client_factory=mock_factory)
    results = backend.dispatch_candidates(manifest_file, [{"ordinal": i, "candidate_id": f"cand_{i}", "policy_params": [float(i)]} for i in range(3)])
    assert [result["metrics"]["score"] for result in results] == [100.0, 101.0, 102.0]


def test_execution_backend_failure_records_manifest_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "test_grid_fail"
    run_dir.mkdir()
    manifest_file = run_dir / "manifest.json"
    _write_manifest(manifest_file, _manifest("test_grid_fail", 5))

    def failing_factory(blade: str, policy_id: str) -> MockBatchEvaluatorClient:
        class FailingClient(MockBatchEvaluatorClient):
            def evaluate_batch(
                self,
                candidates: Sequence[CandidateSpec],
            ) -> BatchResultFrame:
                raise RuntimeError("harness crashed during batch")

        return FailingClient(blade, policy_id)

    backend = ExecutionBackend(client_factory=failing_factory)
    with pytest.raises(ExecutionBackendError, match="evaluation failed"):
        backend.run_grid(manifest_file)
    history = json.loads(manifest_file.read_text(encoding="utf-8"))["attempt_history"]
    assert len(history) == 1
    assert history[0]["status"] == "failed"
    assert history[0]["exit_code"] == 1


def _candidate_result(ordinal: int, candidate_id: str) -> CandidateResult:
    return CandidateResult(
        ordinal=ordinal,
        candidate_id=candidate_id,
        economic_fingerprint="f" * 64,
        metrics={"score": 1.0},
    )


@pytest.mark.parametrize(
    ("status", "results", "message"),
    [
        ("partial", [_candidate_result(0, "expected")], "status"),
        ("complete", [_candidate_result(0, "wrong")], "coverage/order"),
        (
            "complete",
            [_candidate_result(0, "expected"), _candidate_result(0, "expected")],
            "duplicate ordinals",
        ),
        (
            "complete",
            [_candidate_result(0, "expected"), _candidate_result(1, "extra")],
            "coverage/order",
        ),
    ],
)
def test_execution_backend_rejects_noncanonical_batch_receipts(
    status: str,
    results: list[CandidateResult],
    message: str,
) -> None:
    response = BatchResultFrame(
        request_id="batch-test",
        session_id="sess-test",
        status=status,
        results=results,
        elapsed_ms=0.0,
    )

    class FixedResponseClient(MockBatchEvaluatorClient):
        def evaluate_batch(
            self,
            candidates: Sequence[CandidateSpec],
        ) -> BatchResultFrame:
            return response

    client = FixedResponseClient("local", "test_policy_v1")
    client.current_session_id = "sess-test"
    backend = ExecutionBackend(
        client_factory=lambda blade, policy: MockBatchEvaluatorClient(blade, policy)
    )
    with pytest.raises(ExecutionBackendError, match=message):
        backend._evaluate_candidates_with_client(
            client,
            [
                {
                    "ordinal": 0,
                    "candidate_id": "expected",
                    "policy_params": [],
                    "pool_overrides": {},
                }
            ],
        )
