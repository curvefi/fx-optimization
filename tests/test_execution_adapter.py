"""Tests for process and SSH execution adapters."""

from pathlib import Path

from curve_fx_sim.execution.adapter import (
    LocalProcessAdapter,
    MockProcessAdapter,
    ProcessResult,
    SSHProcessAdapter,
)
from curve_fx_sim.execution.site import SSHConfig


def test_mock_process_adapter() -> None:
    mock = MockProcessAdapter(default_result=ProcessResult(returncode=0, stdout="default output", stderr=""))
    res = mock.run(["echo", "hello"])
    assert res.ok
    assert res.stdout == "default output"
    assert len(mock.calls) == 1
    assert mock.calls[0]["argv"] == ["echo", "hello"]


def test_mock_process_adapter_handler() -> None:
    mock = MockProcessAdapter()
    mock.register_handler(
        lambda argv: "special" in argv,
        ProcessResult(returncode=42, stdout="special hit", stderr=""),
    )

    res1 = mock.run(["run", "normal"])
    assert res1.returncode == 0

    res2 = mock.run(["run", "special", "mode"])
    assert res2.returncode == 42
    assert res2.stdout == "special hit"


def test_ssh_process_adapter_argv() -> None:
    config = SSHConfig(
        user="testuser",
        key=Path("/fake/key"),
        port=2222,
        options=("-o", "StrictHostKeyChecking=no"),
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


def test_local_process_adapter_echo() -> None:
    adapter = LocalProcessAdapter()
    res = adapter.run(["echo", "fxsim test"])
    assert res.ok
    assert "fxsim test" in res.stdout
