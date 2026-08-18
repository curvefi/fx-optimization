"""Tests for site profiles and execution environment configurations."""

from pathlib import Path, PurePosixPath

import pytest

from curve_fx_sim.execution.site import (
    ClusterConfig,
    HarnessConfig,
    RunnerConfig,
    SSHConfig,
    SiteProfile,
    SiteProfileError,
    find_site_profile_path,
    load_site_profile,
    validate_remote_host,
    validate_remote_path,
)


def test_default_local_profile() -> None:
    profile = load_site_profile("local")
    assert profile.name == "local"
    assert profile.site_type == "local"
    assert profile.runner.max_workers >= 1


def test_blades_profile() -> None:
    profile = load_site_profile("blades")
    assert profile.name == "blades"
    assert profile.site_type == "ssh"
    assert len(profile.cluster.blades) > 0
    assert "blade-b1" in profile.cluster.blades
    assert profile.ssh.user == "heswithme"
    assert profile.cluster.coordinator == "blade-b6"
    assert profile.cluster.transport == "shared_nfs"
    assert profile.cluster.repository_root == PurePosixPath("/home/heswithme/arb/curve-fx-optimization")
    assert profile.cluster.worker_command == "/home/heswithme/arb/bin/fxsim-worker"
    assert profile.harness.remote_binary_path == PurePosixPath("/home/heswithme/arb/bin/arb_evaluator_ld")


def test_smoke_profile() -> None:
    profile = load_site_profile("smoke")
    assert profile.name == "smoke"
    assert profile.site_type == "local"
    assert profile.runner.max_workers == 2


def test_find_site_profile_error() -> None:
    with pytest.raises(SiteProfileError, match="not found"):
        find_site_profile_path("nonexistent_profile_12345")


def test_site_profile_from_dict() -> None:
    data = {
        "name": "custom",
        "site_type": "ssh",
        "ssh": {"user": "alice", "port": 2222},
        "cluster": {
            "coordinator": "node-1",
            "blades": ["node-1", "node-2"],
            "repository_root": "/srv/fxsim",
            "worker_command": "/srv/fxsim/.venv/bin/fxsim",
        },
        "harness": {
            "binary_name": "custom_eval",
            "remote_binary_path": "/opt/evaluators/custom_eval",
            "timeout_seconds": 1200,
        },
        "runner": {"max_workers": 8},
    }
    profile = SiteProfile.from_dict(data)
    assert profile.name == "custom"
    assert profile.site_type == "ssh"
    assert profile.ssh.user == "alice"
    assert profile.ssh.port == 2222
    assert profile.cluster.coordinator == "node-1"
    assert profile.cluster.blades == ("node-1", "node-2")
    assert profile.cluster.repository_root == PurePosixPath("/srv/fxsim")
    assert profile.cluster.worker_command == "/srv/fxsim/.venv/bin/fxsim"
    assert profile.harness.binary_name == "custom_eval"
    assert profile.harness.remote_binary_path == PurePosixPath("/opt/evaluators/custom_eval")
    assert profile.runner.max_workers == 8


def test_ssh_defaults_and_insecure_options() -> None:
    assert SSHConfig().options == (
        "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
    )
    for options in (
        ("-o", "StrictHostKeyChecking=no"),
        ("-o", "UserKnownHostsFile=/dev/null"),
    ):
        with pytest.raises(SiteProfileError, match="allow-listed"):
            SSHConfig.from_dict({"options": options})

    with pytest.raises(SiteProfileError, match="unsafe characters"):
        validate_remote_host("blade-1;rm")
    with pytest.raises(SiteProfileError, match="non-root absolute"):
        validate_remote_path("/srv/fx/../tmp")


def test_shared_nfs_site_validation_and_transport() -> None:
    data = {
        "name": "nfs", "site_type": "ssh",
        "cluster": {
            "coordinator": "node-1", "blades": ["node-1", "node-2"],
            "transport": "shared_nfs", "remote_base": "/srv/fx",
            "repository_root": "/srv/fx/repo", "worker_command": "/srv/fx/bin/worker",
        },
    }
    assert SiteProfile.from_dict(data).cluster.transport == "shared_nfs"
    with pytest.raises(SiteProfileError, match="coordinator"):
        SiteProfile.from_dict({**data, "cluster": {**data["cluster"], "coordinator": "node-3"}})
    with pytest.raises(SiteProfileError, match="unsupported cluster"):
        SiteProfile.from_dict({**data, "cluster": {**data["cluster"], "extra": True}})
    with pytest.raises(SiteProfileError, match="shared_nfs transport"):
        SiteProfile.from_dict({**data, "site_type": "local"})
