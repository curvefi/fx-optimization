from __future__ import annotations

from pathlib import Path
import json
import subprocess
import threading

from click.testing import CliRunner
import pytest

from fxopt.cli import _ProgressReporter, main
from fxopt.config import ConfigError
from fxopt import placement
from fxopt.results import ArtifactPaths, read_results
from fxopt.run import (
    REMOTE_JOB_FILENAME,
    RunConfig,
    follow_remote_run,
    open_session_request,
    remote_run_status,
    run_config,
    run_remote_config,
    stop_remote_run,
)


class FakeClient:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.batches: list[list[dict[str, object]]] = []

    def start(self) -> dict[str, bool]:
        self.events.append("start")
        return {"hello": True}

    def open_session(self, session_id: str, **request: object) -> None:
        self.events.append(("open", session_id, request))
        assert request["scenario_id"] == "scenario-1"
        assert str(request["market_path"]).endswith("market.json")

    def evaluate_batch(self, candidates: list[dict[str, object]], **request: object) -> dict[str, object]:
        self.events.append("evaluate")
        self.batches.append(candidates)
        return {
            "results": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": "ok",
                    "metrics": {"score": float(candidate["ordinal"])},
                }
                for candidate in reversed(candidates)
            ]
        }

    def close_session(self, session_id: str | None = None) -> None:
        self.events.append(("close", session_id))

    def shutdown(self) -> None:
        self.events.append("shutdown")


def test_run_uses_one_session_and_writes_only_two_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[run]
id = "test-run"
evaluator = "evaluator"
template = "template.json"
batch_size = 2
workers = 3
[session]
n_candles = 3
[scenario]
id = "scenario-1"
market = "market.json"
[candidate.defaults]
policy_params = [0.5, 0.0003]
pool = {}
[candidate.axes]
"pool.A" = [1, 2]
"pool.donation_apy" = [0.0, 0.1]
"""
    )
    fake = FakeClient()
    output = tmp_path / "output"
    progress: list[tuple[int, int]] = []

    run_config(
        config,
        output,
        client_factory=lambda: fake,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    assert RunConfig.from_toml(config).workers == 3

    assert fake.events.count("start") == 1
    assert [event[0] for event in fake.events if isinstance(event, tuple)] == [
        "open",
        "close",
    ]
    assert fake.events[-1] == "shutdown"
    assert [len(batch) for batch in fake.batches] == [2, 2]
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}
    assert progress == [(0, 4), (2, 4), (4, 4)]
    run_payload = json.loads((output / "run.json").read_text())
    assert run_payload["candidate_count"] == 4
    assert "candidates" not in run_payload
    assert len(read_results(output).results) == 4


def test_run_uses_two_placement_lanes_with_direct_open_session(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "curve-fx-sim"
    config = workspace / "curve-fx-optimization" / "configs" / "run.toml"
    config.parent.mkdir(parents=True)
    (workspace / "curve-fx-optimization" / "market.json").write_text("[]")
    config.write_text(
        """
[run]
id = "remote-run"
evaluator = "../../curve-fx-arb-harness/build/evaluator"
template = "../template.json"
batch_size = 256
workers = 2
metric_fields = ["score"]
[placement]
hosts = ["blade-b6", "blade-b7"]
[session]
n_candles = 1
[scenario]
id = "scenario-1"
market = "../market.json"
chainlink = "../chainlink.json"
[candidate.defaults]
policy_params = []
pool = {}
[candidate.axes]
"pool.A" = { start = 1, stop = 16, count = 16 }
"pool.donation_apy" = { start = 0, stop = 0.15, count = 16 }
"pool.mid_fee" = [0.0003]
"pool.out_fee" = [0.0003]
"""
    )
    clients: dict[str, FakeClient] = {}
    calls: list[tuple[str, str, bool]] = []
    ensured = []

    def fake_ssh(
        host: str,
        evaluator: str,
        *,
        workers: int,
        timeout: float,
        verify_local_inputs: bool,
    ):
        assert timeout == 600.0
        calls.append((host, str(evaluator), verify_local_inputs))
        clients[host] = FakeClient()
        return lambda: clients[host]

    monkeypatch.setattr("fxopt.run.ssh_client_factory", fake_ssh)
    monkeypatch.setattr("fxopt.run.prepare_remote", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "fxopt.run.ensure_remote_file",
        lambda *args, **kwargs: ensured.append((*args, kwargs.get("replace", False))),
    )
    output = tmp_path / "output"
    run_config(config, output)

    assert calls == [
        ("blade-b6", "/home/heswithme/arb/curve-fx-arb-harness/build/evaluator", False),
        ("blade-b7", "/home/heswithme/arb/curve-fx-arb-harness/build/evaluator", False),
    ]
    assert ensured == [
        ("blade-b6", str(config.parent / "../template.json"), "/home/heswithme/arb/curve-fx-optimization/template.json", True),
        ("blade-b6", str(config.parent / "../market.json"), "/home/heswithme/arb/curve-fx-optimization/market.json", False),
        ("blade-b6", str(config.parent / "../chainlink.json"), "/home/heswithme/arb/curve-fx-optimization/chainlink.json", False),
        ("blade-b6", config, "/home/heswithme/arb/curve-fx-optimization/configs/run.toml", True),
    ]
    for client in clients.values():
        opened = next(event for event in client.events if isinstance(event, tuple) and event[0] == "open")
        assert opened[2]["template_path"] == "/home/heswithme/arb/curve-fx-optimization/template.json"
        assert opened[2]["market_path"] == "/home/heswithme/arb/curve-fx-optimization/market.json"
        assert opened[2]["chainlink_path"] == "/home/heswithme/arb/curve-fx-optimization/chainlink.json"
    assert sorted(len(batch) for client in clients.values() for batch in client.batches) == [128, 128]
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}
    metadata = json.loads((output / "run.json").read_text())["metadata"]
    assert metadata["placement"] == "ssh"
    assert metadata["hosts"] == ["blade-b6", "blade-b7"]
    assert metadata["batch_size"] == 256
    assert metadata["effective_batch_size"] == 128
    assert metadata["evaluator"] == "/home/heswithme/arb/curve-fx-arb-harness/build/evaluator"
    assert metadata["template"] == "/home/heswithme/arb/curve-fx-optimization/template.json"
    assert metadata["market"] == "/home/heswithme/arb/curve-fx-optimization/market.json"
    config.write_text(config.read_text().replace(
        'evaluator = "../../curve-fx-arb-harness/build/evaluator"',
        'evaluator = "/outside/evaluator"',
    ))
    with pytest.raises(ConfigError, match="remote evaluator path must be inside"):
        RunConfig.from_toml(config)


def test_run_config_preserves_absolute_inputs_and_anchors_relative_market(
    tmp_path: Path,
) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[run]
id = "path-run"
evaluator = "/home/heswithme/evaluator"
template = "/home/heswithme/template.json"
batch_size = 1
[scenario]
id = "scenario-1"
market = "/home/heswithme/market.json"
chainlink = "/home/heswithme/chainlink.json"
[candidate.defaults]
policy_params = []
pool = {}
[candidate.axes]
"pool.A" = [1]
"pool.donation_apy" = [0.0]
"""
    )

    absolute = RunConfig.from_toml(config)
    assert str(absolute.evaluator) == "/home/heswithme/evaluator"
    assert str(absolute.template) == "/home/heswithme/template.json"
    assert absolute.scenario["market"] == "/home/heswithme/market.json"
    assert absolute.scenario["chainlink"] == "/home/heswithme/chainlink.json"
    request = open_session_request(absolute)
    assert request["market_path"] == "/home/heswithme/market.json"
    assert request["chainlink_path"] == "/home/heswithme/chainlink.json"

    config.write_text(config.read_text().replace(
        'market = "/home/heswithme/market.json"', 'market = "market.json"'
    ))
    relative = RunConfig.from_toml(config)
    assert relative.scenario["market"] == str(tmp_path / "market.json")


def test_remote_run_survives_disconnect_and_recovers(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "curve-fx-sim"
    config = workspace / "curve-fx-optimization" / "configs" / "run.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
[run]
id = "remote"
evaluator = "../../curve-fx-arb-harness/build/evaluator"
template = "template.json"
batch_size = 512
metric_fields = ["score"]
[placement]
hosts = ["blade-a5", "blade-a6"]
numa_nodes = [0, 1]
[scenario]
id = "scenario"
market = "market.json"
[candidate.defaults]
policy_params = []
pool = {}
[candidate.axes]
"pool.A" = [1, 2]
"""
    )
    monkeypatch.setattr("fxopt.run.prepare_remote", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "fxopt.run.stage_remote_run",
        lambda _config: "/home/heswithme/arb/curve-fx-optimization/configs/run.toml",
    )
    calls: list[list[str]] = []
    disconnected = True
    statuses = iter([
        "running\n\nblade-a5: 512/1024 (50%)\n",
        "running\n\nblade-a5: 512/1024 (50%)\n",
        "complete\n0\ncoordinator: wrote results\n",
    ])

    def fake_run(argv, **kwargs):
        nonlocal disconnected
        calls.append(list(argv))
        if "mktemp" in argv:
            return subprocess.CompletedProcess(argv, 0, "/tmp/fxopt-grid.test\n", "")
        command = argv[-1] if argv and isinstance(argv[-1], str) else ""
        if "tail --pid=" in command:
            if disconnected:
                disconnected = False
                raise subprocess.CalledProcessError(255, argv)
            return subprocess.CompletedProcess(argv, 0)
        if "state=stopped" in command:
            return subprocess.CompletedProcess(argv, 0, next(statuses), "")
        if argv[0] == "rsync":
            destination = Path(argv[-1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "run.json").write_text('{"run_id":"remote"}')
            (destination / "results.npz").write_bytes(b"npz")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("fxopt.run.subprocess.run", fake_run)
    output = tmp_path / "result"
    with pytest.raises(subprocess.CalledProcessError):
        run_remote_config(config, output, stream_blade="blade-a5")

    assert (output / REMOTE_JOB_FILENAME).is_file()
    status = remote_run_status(config, output)
    assert status.state == "running"
    assert status.detail == "blade-a5: 512/1024 (50%)"

    paths = follow_remote_run(config, output)

    assert paths == ArtifactPaths(output / "run.json", output / "results.npz")
    launch = next(call for call in calls if "nohup" in call[-1])
    assert launch[: len(["ssh", *placement.SSH_OPTIONS, "--", "blade-a5"])] == [
        "ssh", *placement.SSH_OPTIONS, "--", "blade-a5"
    ]
    assert "nohup setsid /bin/sh" in launch[-1]
    assert not (output / REMOTE_JOB_FILENAME).exists()
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}
    assert remote_run_status(config, output).state == "retrieved"
    assert sum(call[0] == "rsync" for call in calls) == 1
    assert calls[-1][-4:-1] == ["rm", "-rf", "--"]


def test_stop_remote_run_terminates_coordinator_and_retains_handle(
    tmp_path: Path, monkeypatch
) -> None:
    config = (
        tmp_path / "curve-fx-sim" / "curve-fx-optimization" /
        "configs" / "run.toml"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        """
[run]
id = "remote"
evaluator = "evaluator"
template = "template.json"
batch_size = 512
metric_fields = ["score"]
[placement]
hosts = ["blade-a5"]
[scenario]
id = "scenario"
market = "market.json"
[candidate.defaults]
policy_params = []
pool = {}
[candidate.axes]
"pool.A" = [1]
"""
    )
    output = tmp_path / "result"
    output.mkdir()
    handle = output / REMOTE_JOB_FILENAME
    handle.write_text(json.dumps({
        "run_id": "remote",
        "coordinator": "blade-a5",
        "remote_output": "/tmp/fxopt-grid.test",
    }))
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        command = argv[-1]
        if "state=stopped" in command:
            return subprocess.CompletedProcess(
                argv, 0, "running\n\nblade-a5: 512/1024 (50%)\n", ""
            )
        if "signal_tree()" in command:
            return subprocess.CompletedProcess(argv, 0, "stopped\n", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr("fxopt.run.subprocess.run", fake_run)
    status = stop_remote_run(config, output)

    assert status.state == "stopped"
    assert status.detail == "blade-a5: 512/1024 (50%)"
    assert handle.is_file()
    stop_command = calls[-1][-1]
    assert 'kill -TERM -- "-$pid"' in stop_command
    assert 'printf \'operator-stop\\n\' > "$work/stopped"' in stop_command


def test_fxopt_help_exposes_only_run_surface() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "generate" not in result.output
    assert "collect" not in result.output
    sample = RunConfig.from_toml("configs/experiments/eurusd-a-donation-16x16.toml")
    grid = sample.candidate.grid()
    assert grid.candidate_at(0).payload["pool"]["A"] == 10_000
    assert grid.candidate_at(240).payload["pool"]["A"] == 2_000_000
    assert grid.candidate_at(0).payload["pool"]["out_fee"] == 0.0003


def test_fxopt_run_cli_reports_progress(tmp_path, monkeypatch) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[run]
id = "test-run"
evaluator = "evaluator"
template = "template.json"
batch_size = 4
[scenario]
id = "scenario-1"
market = "market.json"
[candidate.defaults]
policy_params = []
pool = {}
[candidate.axes]
"pool.A" = { start = 1, stop = 200, count = 8, multiply = 10000 }
"pool.donation_apy" = { start = 0, stop = 0.10, count = 8 }
"pool.reserved_profit_fraction" = { start = 0, stop = 1, count = 8 }
"""
    )
    paths = ArtifactPaths(tmp_path / "run.json", tmp_path / "results.npz")
    heartbeat_seen = threading.Event()
    zero_reports = 0
    original_write = _ProgressReporter._write

    def observed_write(self, completed, total, **kwargs):
        nonlocal zero_reports
        original_write(self, completed, total, **kwargs)
        if completed == 0 and kwargs.get("working"):
            zero_reports += 1
            heartbeat_seen.set()

    monkeypatch.setattr(_ProgressReporter, "_write", observed_write)
    monkeypatch.setattr(
        "fxopt.cli._ProgressReporter",
        lambda label, **kwargs: _ProgressReporter(label, _interval=0.01, **kwargs),
    )

    def fake_run_config(*_args, progress_callback=None, **_kwargs):
        progress_callback(0, 4)
        assert heartbeat_seen.wait(1.0)
        progress_callback(2, 4)
        progress_callback(4, 4)
        return paths

    monkeypatch.setattr("fxopt.cli.run_config", fake_run_config)
    result = CliRunner().invoke(main, ["run", str(config), "--output", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert "run: saved 0/4 (0%)" in result.output
    stale = next(line for line in result.output.splitlines() if "run: waiting..." in line)
    assert "0/4 saved (0%)" in stale
    assert "pools/s" not in stale
    assert "ETA" not in stale
    assert "run: saved 4/4 (100%)" in result.output
    assert (
        "running 512 pools grid: A 1..200 (8 pts), donation 0..10% (8 pts), "
        "rpf 0..1 (8 pts)"
    ) in result.output
    assert "pools/s" in result.output
    assert "ETA" in result.output


def test_progress_reporter_streams_one_blade_without_numa_noise(capsys) -> None:
    reporter = _ProgressReporter(
        "run",
        stream_blade="blade-a5",
        blade_index=0,
        blade_count=2,
        lanes_per_blade=2,
    )
    reporter(0, 8)
    reporter.lane("blade-a5:numa0", 2, 1.0)
    reporter.lane("blade-b1:numa0", 2, 1.0)
    reporter(2, 8)
    reporter.lane("blade-a5:numa1", 2, 1.0)
    reporter.close()

    output = capsys.readouterr().err
    assert "blade-a5: 2/4 (50%) 2.0 pools/s ETA 1.0s" in output
    assert "blade-a5: 4/4 (100%) 4.0 pools/s ETA 0.0s" in output
    assert "working..." not in output
    assert output.count("waiting for first batch") == 1
    assert "numa" not in output
    assert "global" not in output
    assert "run:" not in output
