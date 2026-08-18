"""Deterministic tests for shared-NFS ownership and immutable staging."""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath

import pytest

from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.execution.adapter import MockProcessAdapter, ProcessResult
from curve_fx_sim.execution.shared_nfs import (
    SharedNFSError,
    SharedRunLease,
    fetch_authoritative_run,
    grid_identity_sha256,
    package_identity_sha256,
    stage_run_directory_atomic,
)
from curve_fx_sim.execution.site import ClusterConfig, SiteProfile


def _site() -> SiteProfile:
    return SiteProfile(
        name="nfs",
        site_type="ssh",
        cluster=ClusterConfig(
            coordinator="blade-c",
            transport="shared_nfs",
            remote_base=PurePosixPath("/srv/fx"),
            remote_run_root=PurePosixPath("/srv/fx/runs"),
            repository_root=PurePosixPath("/srv/fx/repo"),
            worker_command="/srv/fx/bin/worker",
            blades=("blade-c", "blade-w"),
        ),
    )


def test_shared_run_lease_rejects_contention_and_wrong_owner(monkeypatch) -> None:
    winner_adapter = MockProcessAdapter()
    winner = SharedRunLease(_site(), "run_a", adapter=winner_adapter)
    winner.acquire()
    assert "mkdir /srv/fx/.leases/run_a" in winner_adapter.calls[0]["argv"][-1]

    contender = SharedRunLease(
        _site(),
        "run_a",
        adapter=MockProcessAdapter(ProcessResult(1, "", "already exists")),
    )
    with pytest.raises(SharedNFSError, match="already owned"):
        contender.acquire()

    with monkeypatch.context() as env:
        env.setenv("FXSIM_RUN_LEASE_TOKEN", "wrong-owner")
        wrong_owner_adapter = MockProcessAdapter(
            ProcessResult(1, "", "owner mismatch")
        )
        wrong_owner = SharedRunLease(_site(), "run_a", adapter=wrong_owner_adapter)
        with pytest.raises(SharedNFSError, match="already owned"):
            wrong_owner.acquire()
        assert "test -f /srv/fx/.leases/run_a/owner" in wrong_owner_adapter.calls[0]["argv"][-1]
        assert "wrong-owner" in wrong_owner_adapter.calls[0]["argv"][-1]

    winner.release()


def test_new_run_stage_is_atomic_and_resume_fetches_authority_without_upload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    manifest = run_dir / "manifest.json"
    manifest.write_text('{"run_id":"run_a"}\n', encoding="utf-8")
    adapter = MockProcessAdapter()
    lease = SharedRunLease(_site(), "run_a", adapter=adapter)
    lease.token = "attempt-token"

    destination = stage_run_directory_atomic(lease, run_dir)
    assert destination == PurePosixPath("/srv/fx/runs/run_a")
    assert len(adapter.calls) == 3
    assert "sha256sum" in adapter.calls[-1]["argv"][-1]
    assert sha256_path(manifest) in adapter.calls[-1]["argv"][-1]
    assert "mv /srv/fx/.staging/attempt-token/run_a /srv/fx/runs/run_a" in adapter.calls[-1]["argv"][-1]

    resume_adapter = MockProcessAdapter()
    resumed = SharedRunLease(_site(), "run_a", adapter=resume_adapter)
    local_runs = tmp_path / "local-runs"
    assert fetch_authoritative_run(resumed, local_runs) == local_runs / "run_a"
    rsync_calls = [call["argv"] for call in resume_adapter.calls if call["argv"][0] == "rsync"]
    assert len(rsync_calls) == 1
    assert rsync_calls[0][-2].startswith("heswithme@blade-c:")
    assert "--protect-args" in rsync_calls[0]


def test_grid_identity_digest_detects_provenance_mismatch() -> None:
    manifest = {
        "run_id": "grid_a",
        "resolved_spec": {"policy": {"source_sha256": "a" * 64}},
        "core": {"policy_source_sha256": "a" * 64},
        "grid": {
            "grid_id": "grid",
            "pool_count": 1,
            "resolved_axes": [],
            "pools": [{"id": "pool_0", "policy_params": [1.0]}],
        },
    }
    authority = grid_identity_sha256(manifest)
    changed = copy.deepcopy(manifest)
    changed["grid"]["pools"][0]["policy_params"] = [2.0]
    assert grid_identity_sha256(changed) != authority


def test_package_digest_changes_for_policy_include_and_source_changes(tmp_path: Path) -> None:
    package = tmp_path / "src" / "curve_fx_sim"
    package.mkdir(parents=True)
    policy = tmp_path / "policies"
    policy.mkdir()
    source = package / "worker.py"
    include = policy / "policy.hpp"
    source.write_text("SOURCE = 1\n", encoding="utf-8")
    include.write_text("#define FEE 1\n", encoding="utf-8")

    baseline = package_identity_sha256(tmp_path)
    include.write_text("#define FEE 2\n", encoding="utf-8")
    policy_digest = package_identity_sha256(tmp_path)
    source.write_text("SOURCE = 2\n", encoding="utf-8")
    source_digest = package_identity_sha256(tmp_path)

    assert policy_digest != baseline
    assert source_digest != policy_digest
