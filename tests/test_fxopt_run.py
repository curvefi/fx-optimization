from __future__ import annotations

from pathlib import Path
import json
import subprocess
import threading

from click.testing import CliRunner
import pytest

from fxopt.cli import _ProgressReporter, main
from fxopt import placement
from fxopt.results import ArtifactPaths, read_result_columns
from fxopt.run import (
    REMOTE_JOB_FILENAME,
    RunConfig,
    _shuffled_block_ranges,
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

    def register_grid(self, grid_id: str, grid: object, **request: object) -> object:
        self.events.append(("grid", grid_id))
        count = 1
        for length in grid["shape"]:
            count *= length
        return {"candidate_count": count}

    def evaluate_batch(self, candidates: list[dict[str, object]], **request: object) -> dict[str, object]:
        self.events.append("evaluate")
        if ranges := request.get("ranges"):
            ordinals = [
                ordinal
                for start, count in ranges
                for ordinal in range(int(start), int(start) + int(count))
            ]
            candidates = [
                {
                    "candidate_id": f"p{int(ordinal):08d}",
                    "ordinal": int(ordinal),
                }
                for ordinal in ordinals
            ]
        self.batches.append(candidates)
        if request.get("metrics_format") == "array":
            return {
                "metric_fields": request["metric_fields"],
                "results": [
                    {
                        "ordinal": candidate["ordinal"],
                        "candidate_id": candidate["candidate_id"],
                        "status": "ok",
                        "metrics": [float(candidate["ordinal"])],
                    }
                    for candidate in candidates
                ],
            }
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


def test_shuffled_blocks_are_balanced_reproducible_and_cover_the_grid() -> None:
    assignments = _shuffled_block_ranges(64, 4, 4)

    assert assignments == _shuffled_block_ranges(64, 4, 4)
    assert [[start for start, _stop in worker] for worker in assignments] == [
        [40, 36, 52, 0],
        [56, 8, 28, 24],
        [20, 12, 32, 60],
        [4, 44, 16, 48],
    ]
    assert sorted(
        ordinal
        for worker in assignments
        for start, stop in worker
        for ordinal in range(start, stop)
    ) == list(range(64))


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
metric_fields = ["score"]
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
        "grid",
        "close",
    ]
    assert fake.events[-1] == "shutdown"
    assert [len(batch) for batch in fake.batches] == [2, 2]
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}
    assert progress == [(0, 4), (2, 4), (4, 4)]
    run_payload = json.loads((output / "run.json").read_text())
    assert run_payload["candidate_count"] == 4
    assert "candidates" not in run_payload
    assert read_result_columns(output).row_count == 4


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
metric_fields = ["score"]
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
        run_remote_config(config, output)

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
metric_fields = ["score"]
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
    output = tmp_path / "out"
    output.mkdir()
    (output / "run.json").write_text("old")
    (output / "stale.txt").write_text("old")
    paths = ArtifactPaths(output / "run.json", output / "results.npz")
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
        assert not output.exists()
        output.mkdir()
        progress_callback(0, 4)
        assert heartbeat_seen.wait(1.0)
        progress_callback(2, 4)
        progress_callback(4, 4)
        paths.run_json.write_text("new")
        paths.results_npz.write_bytes(b"npz")
        return paths

    monkeypatch.setattr("fxopt.cli.run_config", fake_run_config)
    result = CliRunner().invoke(
        main, ["run", str(config), "--output", str(output), "--overwrite"]
    )

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
    assert not (output / "stale.txt").exists()


def test_overwrite_refuses_a_detached_remote_job(tmp_path) -> None:
    config = tmp_path / "curve-fx-optimization" / "configs" / "run.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
[run]
id = "test-run"
evaluator = "evaluator"
template = "template.json"
batch_size = 1
metric_fields = ["score"]
[placement]
hosts = ["blade-a5"]
[scenario]
id = "scenario"
market = "market.json"
[candidate.defaults]
policy_params = []
pool = {}
"""
    )
    output = tmp_path / "out"
    output.mkdir()
    handle = output / REMOTE_JOB_FILENAME
    handle.write_text("{}")

    result = CliRunner().invoke(
        main, ["run", str(config), "--output", str(output), "--overwrite"]
    )

    assert result.exit_code != 0
    assert "refuses a detached remote job" in result.output
    assert handle.is_file()
