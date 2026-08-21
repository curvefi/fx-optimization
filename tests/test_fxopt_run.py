from __future__ import annotations

from pathlib import Path
import json

from click.testing import CliRunner

from fxopt.cli import main
from fxopt.results import read_results
from fxopt.run import RunConfig, run_config


class FakeClient:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.batches: list[list[dict[str, object]]] = []

    def start(self) -> dict[str, bool]:
        self.events.append("start")
        return {"hello": True}

    def open_session(self, session_id: str, **request: object) -> None:
        self.events.append(("open", session_id, request))
        manifest = Path(str(request["manifest_path"]))
        payload = json.loads(manifest.read_text())
        assert payload["schema_version"] == "fxsim_manifest_v1"
        assert payload["resolved_spec"]["scenario"]["id"] == "scenario-1"
        assert payload["resolved_spec"]["scenario"]["market_files"][0]["path"].endswith("market.json")

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
market_sha256 = "abc"
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

    run_config(config, output, client_factory=lambda: fake)
    assert RunConfig.from_toml(config).workers == 3

    assert fake.events.count("start") == 1
    assert [event[0] for event in fake.events if isinstance(event, tuple)] == [
        "open",
        "close",
    ]
    assert fake.events[-1] == "shutdown"
    assert [len(batch) for batch in fake.batches] == [2, 2]
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}
    run_payload = json.loads((output / "run.json").read_text())
    assert run_payload["candidate_count"] == 4
    assert "candidates" not in run_payload
    assert len(read_results(output).results) == 4


def test_run_uses_two_placement_lanes_and_cleans_session_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[run]
id = "remote-run"
evaluator = "/shared/evaluator"
template = "/shared/template.json"
batch_size = 256
workers = 2
[placement]
hosts = ["blade-b6", "blade-b7"]
[session]
n_candles = 1
[scenario]
id = "scenario-1"
market = "market.json"
market_sha256 = "abc"
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
    calls: list[tuple[str, bool]] = []

    def fake_ssh(host: str, evaluator: str, *, workers: int, verify_local_inputs: bool):
        calls.append((host, verify_local_inputs))
        clients[host] = FakeClient()
        return lambda: clients[host]

    monkeypatch.setattr("fxopt.run.ssh_client_factory", fake_ssh)
    output = tmp_path / "output"
    run_config(config, output)

    assert calls == [("blade-b6", False), ("blade-b7", False)]
    assert sorted(len(batch) for client in clients.values() for batch in client.batches) == [128, 128]
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}
    assert not any(path.name.startswith(".fxopt-session-") for path in output.iterdir())
    metadata = json.loads((output / "run.json").read_text())["metadata"]
    assert metadata["placement"] == "ssh"
    assert metadata["hosts"] == ["blade-b6", "blade-b7"]
    assert metadata["batch_size"] == 256
    assert metadata["effective_batch_size"] == 128


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
