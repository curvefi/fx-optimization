import json
from pathlib import Path

from click.testing import CliRunner

from fxopt import Candidate, CandidateResult, ResultBundle, write_results
from fxopt.cli import main
from fxopt.explorer import open_fxopt_explorer
from fxopt.optimize import optimize_config
from fxopt.results import read_results
from fxopt.shiftclick import trace_candidate


class FakeClient:
    def __init__(self):
        self.open_requests = []

    def start(self):
        return {"hello": True}

    def open_session(self, session_id, **request):
        self.open_requests.append(request)
        return None

    def evaluate_batch(self, candidates, **request):
        return {"results": [{
            "candidate_id": item["candidate_id"],
            "status": "ok",
            "metrics": {"score": float(item["pool_overrides"]["A"])},
            "artifacts": ({"trace_path": "trace.json"}
                          if request.get("observation", {}).get("kind") == "full_trace" else None),
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
[scenario]
id = "scenario"
market = "market.json"
[candidate.defaults]
policy_params = [0.5]
pool = { A = 1 }
[candidate.axes]
"pool.A" = [1, 2, 3]
[optimization]
budget = 4
batch_size = 2
metric = "score"
maximize = true
seed = 7
""")
    return path


def _remote_config(tmp_path):
    workspace = tmp_path / "curve-fx-sim"
    path = _config(workspace / "curve-fx-optimization" / "configs")
    source = path.read_text()
    source = source.replace('evaluator = "evaluator"',
                            'evaluator = "../../curve-fx-arb-harness/evaluator"')
    source = source.replace('template = "template.json"', 'template = "../template.json"')
    source = source.replace('market = "market.json"', 'market = "../market.json"')
    path.write_text(source + """
[placement]
hosts = ["blade-b6", "blade-b7"]
""")
    return path


def test_heatmap_reads_the_two_file_result(tmp_path):
    candidates = tuple(Candidate(f"p{index}", (), {"A": a, "donation_apy": donation})
                       for index, (a, donation) in enumerate(((1, 0), (1, 1), (2, 0), (2, 1))))
    results = tuple(CandidateResult(item.candidate_id, metrics={"score": float(index)}, ordinal=index)
                    for index, item in enumerate(candidates))
    write_results(ResultBundle("grid", candidates, results, metadata={
        "axes": {"pool.A": [1, 2], "pool.donation_apy": [0, 1]},
        "shape": [2, 2], "config": str(tmp_path / "run.toml"),
    }), tmp_path / "run")
    explorer = open_fxopt_explorer(
        tmp_path / "run", metrics=("score",),
        x_axis="pool.A", y_axis="pool.donation_apy",
    )
    output, state = explorer.save(tmp_path / "heatmap.png")
    explorer.close()
    assert output.is_file() and state.is_file()


def test_nevergrad_and_shiftclick_share_the_fleet(tmp_path):
    config = _config(tmp_path)
    factory = FakeClient
    progress = []
    artifacts = optimize_config(
        config,
        tmp_path / "optimization",
        client_factory=factory,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    bundle = read_results(artifacts.run_json.parent)
    assert len(bundle.results) == 4
    assert bundle.metadata["best_metric_value"] == 3.0
    assert progress == [(0, 4), (2, 4), (4, 4)]

    candidate = bundle.candidates[1]
    summary = trace_candidate(config, candidate=candidate, ordinal=1, output_dir=tmp_path / "trace",
                              trace_actions=True, client_factory=factory)
    payload = json.loads(summary.read_text())
    assert payload["source_ordinal"] == 1
    assert payload["candidate"]["candidate_id"] == candidate.candidate_id
    assert payload["result"]["artifacts"]["trace_path"] == "trace.json"


def test_remote_optimize_maps_inputs_but_trace_replays_locally(
    tmp_path, monkeypatch
):
    config = _remote_config(tmp_path)
    remote = FakeClient()
    optimize_config(config, tmp_path / "optimization", client_factory=lambda: remote)
    assert remote.open_requests[0]["template_path"] == "/home/heswithme/arb/curve-fx-optimization/template.json"
    assert remote.open_requests[0]["market_path"] == "/home/heswithme/arb/curve-fx-optimization/market.json"

    local = FakeClient()
    local_calls = []

    def fake_local(evaluator, *, work_dir, workers):
        local_calls.append((Path(evaluator), Path(work_dir), workers))
        return lambda: local

    monkeypatch.setattr("fxopt.shiftclick.local_client_factory", fake_local)
    monkeypatch.setattr(
        "fxopt.run.ssh_client_factory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SSH replay")),
    )
    candidate = Candidate("stored", [7.5], {"A": 99})
    destination = tmp_path / "trace"
    trace_candidate(config, candidate=candidate, ordinal=3, output_dir=destination)

    workspace = tmp_path / "curve-fx-sim"
    assert local_calls[0][0].resolve() == workspace / "curve-fx-arb-harness" / "evaluator"
    assert local_calls[0][2] == 1
    assert Path(local.open_requests[0]["template_path"]).resolve() == workspace / "curve-fx-optimization" / "template.json"
    assert Path(local.open_requests[0]["market_path"]).resolve() == workspace / "curve-fx-optimization" / "market.json"


def test_shiftclick_cli_replays_stored_adaptive_candidate(monkeypatch, tmp_path):
    config = _config(tmp_path)
    stored = Candidate("adaptive-ordinal-1", [7.5], {"A": 99})
    write_results(
        ResultBundle(
            "adaptive",
            (stored,),
            (CandidateResult(stored.candidate_id, metrics={"score": 99.0}, ordinal=1),),
            metadata={"config": str(config)},
        ),
        tmp_path / "adaptive",
    )
    captured = {}

    def fake_trace(run_id, metadata, *, candidate, ordinal, output_dir, **_kwargs):
        captured.update(
            config=Path(metadata["config"]),
            candidate=candidate,
            ordinal=ordinal,
            run_id=run_id,
        )
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
        "shiftclick", str(tmp_path / "adaptive"), "--ordinal", "1",
        "--output", str(tmp_path / "trace"),
    ])

    assert result.exit_code == 0, result.output
    assert captured["candidate"] == stored
    assert captured["candidate"].pool_overrides["A"] == 99
    assert captured["ordinal"] == 1
