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
    def start(self):
        return {"hello": True}

    def open_session(self, session_id, **request):
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
    artifacts = optimize_config(config, tmp_path / "optimization", client_factory=factory)
    bundle = read_results(artifacts.run_json.parent)
    assert len(bundle.results) == 4
    assert bundle.metadata["best_metric_value"] == 3.0

    candidate = bundle.candidates[1]
    summary = trace_candidate(config, candidate=candidate, ordinal=1, output_dir=tmp_path / "trace",
                              trace_actions=True, client_factory=factory)
    payload = json.loads(summary.read_text())
    assert payload["source_ordinal"] == 1
    assert payload["candidate"]["candidate_id"] == candidate.candidate_id
    assert payload["result"]["artifacts"]["trace_path"] == "trace.json"


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

    def fake_trace(config_path, *, candidate, ordinal, output_dir, **_kwargs):
        captured.update(config=Path(config_path), candidate=candidate, ordinal=ordinal)
        output_dir.mkdir(parents=True)
        summary = output_dir / "shiftclick.json"
        summary.write_text("{}")
        return summary

    monkeypatch.setattr("fxopt.cli.trace_candidate", fake_trace)
    result = CliRunner().invoke(main, [
        "shiftclick", str(tmp_path / "adaptive"), "--ordinal", "1",
        "--output", str(tmp_path / "trace"),
    ])

    assert result.exit_code == 0, result.output
    assert captured["candidate"] == stored
    assert captured["candidate"].pool_overrides["A"] == 99
    assert captured["ordinal"] == 1
