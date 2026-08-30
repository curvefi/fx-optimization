from pathlib import Path

from click.testing import CliRunner

from fxopt import Candidate
from fxopt.cli import main
from fxopt.engine import ProjectedBatch
from fxopt.explorer import open_fxopt_explorer
from fxopt.results import GridResultWriter
from fxopt.shiftclick import trace_candidate


class FakeClient:
    def __init__(self):
        self.open_requests = []

    def start(self):
        return {"hello": True}

    def open_session(self, session_id, **request):
        self.open_requests.append(request)

    def evaluate_batch(self, candidates, **request):
        return {"results": [{
            "candidate_id": item["candidate_id"],
            "status": "ok",
            "metrics": {"score": float(item["pool_overrides"]["A"])},
            "artifacts": ({"trace_path": "trace.json"}
                          if request.get("observation", {}).get("kind") == "full_trace"
                          else None),
        } for item in candidates]}

    def close_session(self, session_id=None):
        return None

    def shutdown(self):
        return None


def _config(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "run.toml"
    path.write_text("""
[run]
id = "workflow"
evaluator = "evaluator"
template = "template.json"
batch_size = 2
workers = 1
metric_fields = ["score"]
[scenario]
id = "scenario"
market = "market.json"
[candidate.defaults]
policy_params = [0.5]
pool = { A = 1 }
[candidate.axes]
"pool.A" = [1, 2, 3]
""")
    return path


def _remote_config(tmp_path):
    workspace = tmp_path / "curve-fx-sim"
    path = _config(workspace / "curve-fx-optimization" / "configs")
    source = path.read_text()
    source = source.replace(
        'evaluator = "evaluator"',
        'evaluator = "../../curve-fx-arb-harness/evaluator"',
    )
    source = source.replace('template = "template.json"', 'template = "../template.json"')
    source = source.replace('market = "market.json"', 'market = "../market.json"')
    path.write_text(source + '\n[placement]\nhosts = ["blade-b6", "blade-b7"]\n')
    return path


def _write_grid(path, *, run_id, metadata, values):
    writer = GridResultWriter(
        path,
        run_id=run_id,
        total=len(values),
        metadata=metadata,
        metric_names=("score",),
    )
    writer.append_projected(
        range(len(values)),
        ProjectedBatch(("score",), tuple({
            "candidate_id": f"p{ordinal:08d}",
            "status": "ok",
            "metrics": [float(value)],
        } for ordinal, value in enumerate(values))),
    )
    writer.finalize()


def test_heatmap_reads_the_two_file_result(tmp_path):
    _write_grid(
        tmp_path / "run",
        run_id="grid",
        metadata={
            "candidate_defaults": {"policy_params": [], "pool": {}},
            "axes": {"pool.A": [1, 2], "pool.donation_apy": [0, 1]},
            "shape": [2, 2],
            "config": str(tmp_path / "run.toml"),
        },
        values=range(4),
    )
    explorer = open_fxopt_explorer(
        tmp_path / "run",
        metrics=("score",),
        x_axis="pool.A",
        y_axis="pool.donation_apy",
    )
    output, state = explorer.save(tmp_path / "heatmap.png")
    explorer.close()
    assert output.is_file() and state.is_file()


def test_remote_grid_trace_replays_locally(tmp_path, monkeypatch):
    config = _remote_config(tmp_path)
    local = FakeClient()
    local_calls = []

    def fake_local(evaluator, *, work_dir, workers):
        local_calls.append((Path(evaluator), Path(work_dir), workers))
        return lambda: local

    monkeypatch.setattr("fxopt.shiftclick.local_client_factory", fake_local)
    candidate = Candidate("stored", [7.5], {"A": 99})
    trace_candidate(
        config,
        candidate=candidate,
        ordinal=3,
        output_dir=tmp_path / "trace",
    )

    workspace = tmp_path / "curve-fx-sim"
    assert local_calls[0][0].resolve() == workspace / "curve-fx-arb-harness" / "evaluator"
    assert local_calls[0][2] == 1
    assert Path(local.open_requests[0]["template_path"]).resolve() == (
        workspace / "curve-fx-optimization" / "template.json"
    )
    assert Path(local.open_requests[0]["market_path"]).resolve() == (
        workspace / "curve-fx-optimization" / "market.json"
    )
    assert local.open_requests[0]["event_cursor"] == "scalar"
    assert local.open_requests[0]["metric_profile"] == "full_summary"


def test_shiftclick_cli_replays_stored_grid_candidate(monkeypatch, tmp_path):
    config = _config(tmp_path)
    _write_grid(
        tmp_path / "grid",
        run_id="grid",
        metadata={
            "config": str(config),
            "candidate_defaults": {"policy_params": [7.5], "pool": {}},
            "axes": {"pool.A": [99]},
            "shape": [1],
        },
        values=(99,),
    )
    captured = {}

    def fake_trace(run_id, metadata, *, candidate, ordinal, output_dir, **_kwargs):
        captured.update(candidate=candidate, ordinal=ordinal, run_id=run_id)
        output_dir.mkdir(parents=True)
        summary = output_dir / "shiftclick.json"
        summary.write_text("{}")
        return summary

    def fake_plot(_summary, output, **_kwargs):
        output.write_bytes(b"png")
        return output

    monkeypatch.setattr("fxopt.shiftclick.trace_stored_candidate", fake_trace)
    monkeypatch.setattr("fxopt.shiftclick.save_shiftclick_plot", fake_plot)
    result = CliRunner().invoke(main, [
        "shiftclick", str(tmp_path / "grid"), "--ordinal", "0",
        "--output", str(tmp_path / "trace"),
    ])

    assert result.exit_code == 0, result.output
    assert captured["candidate"] == Candidate("p00000000", [7.5], {"A": 99})
    assert captured["ordinal"] == 0
