"""Tests for process and SSH execution adapters."""

from pathlib import Path

from curve_fx_sim.execution.adapter import (
    MockProcessAdapter,
    SSHProcessAdapter,
)
from curve_fx_sim.execution.site import SSHConfig


def _check_ssh_process_adapter_argv() -> None:
    config = SSHConfig(
        user="testuser",
        key=Path("/fake/key"),
        port=2222,
        options=("-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"),
    )
    ssh_adapter = SSHProcessAdapter(ssh_config=config, process_runner=MockProcessAdapter())
    argv = ssh_adapter.build_ssh_argv("blade-1", "ls -la /tmp")
    assert argv[0] == "ssh"
    assert "-i" in argv
    assert "/fake/key" in argv
    assert "-p" in argv
    assert "2222" in argv
    assert "testuser@blade-1" in argv
    assert "ls -la /tmp" in argv


def _check_rsync_uses_protected_arguments_for_non_nfs_transfers(tmp_path: Path) -> None:
    runner = MockProcessAdapter()
    adapter = SSHProcessAdapter(ssh_config=SSHConfig(), process_runner=runner)

    adapter.rsync_upload(tmp_path / "input dir", "blade-1", "/srv/fx/runs/run/input")
    adapter.rsync_download("blade-1", "/srv/fx/runs/run", tmp_path / "output dir")

    assert all("--protect-args" in call["argv"] for call in runner.calls)
    assert all("--" in call["argv"] for call in runner.calls)
    assert all("--delete" not in call["argv"] for call in runner.calls)


def test_execution_adapters_protect_ssh_and_rsync_argv(tmp_path: Path) -> None:
    _check_ssh_process_adapter_argv()
    _check_rsync_uses_protected_arguments_for_non_nfs_transfers(tmp_path)
