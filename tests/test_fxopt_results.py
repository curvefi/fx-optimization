import json
import zipfile

import numpy as np
from fxopt import Candidate, CandidateResult, ResultBundle, read_results, write_results
from fxopt.engine import ProjectedBatch
from fxopt.results import GridResultWriter, ResultWriter, read_result_columns


def test_results_round_trip_is_exactly_two_canonical_files(tmp_path):
    bundle = ResultBundle(
        run_id="run-1",
        candidates=(Candidate("b", [2.0], {"fee": 1}), Candidate("a", [1.0])),
        results=(
            CandidateResult("b", metrics={"score": 2.5}),
            CandidateResult("a", status="failed", metrics={"loss": 4.0}, error="timeout", ordinal=1),
        ),
        metadata={"kind": "grid"},
    )
    paths = write_results(bundle, tmp_path)
    loaded = read_results(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {"run.json", "results.npz"}
    assert paths.run_json.name == "run.json" and paths.results_npz.name == "results.npz"
    assert loaded.run_id == bundle.run_id
    assert [result.to_dict() for result in loaded.results] == [result.to_dict() for result in bundle.results]
    assert loaded.candidates == bundle.candidates
    assert loaded.metadata == bundle.metadata


def test_result_writer_appends_batches_without_retaining_rows(tmp_path):
    writer = ResultWriter(tmp_path, run_id="stream-1", metadata={"kind": "adaptive"})
    for index in (2, 0, 1):
        candidate = Candidate(f"c{index}", [float(index)])
        writer.append([candidate], [CandidateResult(
            candidate.candidate_id, metrics={"score": float(index)}, ordinal=index
        )])
        assert writer.retained_rows == 0
    paths = writer.finalize()
    loaded = read_results(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {"run.json", "results.npz"}
    assert paths.results_npz.name == "results.npz"
    run = json.loads(paths.run_json.read_text())
    assert run["candidate_count"] == 3 and "candidates" not in run
    assert [candidate.candidate_id for candidate in loaded.candidates] == ["c0", "c1", "c2"]
    assert [result.metrics["score"] for result in loaded.results] == [0.0, 1.0, 2.0]


def test_result_writer_finalize_reads_spool_once(tmp_path):
    writer = ResultWriter(tmp_path, run_id="once")
    candidate = Candidate("c0", [1.0], {"fee": 1})
    writer.append([candidate], [CandidateResult("c0", error="failed")])
    original_rows = writer._rows
    calls = 0

    def counted_rows():
        nonlocal calls
        calls += 1
        yield from original_rows()

    writer._rows = counted_rows
    writer.finalize()

    assert calls == 1


def test_grid_writer_streams_typed_shards_and_reconstructs_candidates(tmp_path):
    metadata = {
        "candidate_defaults": {"policy_params": [], "pool": {}},
        "axes": {"pool.A": [1, 2, 3]},
        "shape": [3],
    }
    writer = GridResultWriter(
        tmp_path,
        run_id="grid",
        total=3,
        metadata=metadata,
        metric_names=("score",),
    )
    candidates = (Candidate("p00000001", pool_overrides={"A": 2}),
                  Candidate("p00000000", pool_overrides={"A": 1}))
    writer.append(
        candidates,
        (
            CandidateResult("p00000001", status="failed", error="collapsed", ordinal=1),
            CandidateResult("p00000000", metrics={"score": 3.0}, ordinal=0),
        ),
    )
    writer.append_projected(
        (2,),
        ProjectedBatch(
            ("score",),
            ({
                "candidate_id": "p00000002",
                "status": "failed",
                "error": "quarantined",
                "metrics": [-1.0],
            },),
        ),
    )
    writer.finalize()

    columns = read_result_columns(tmp_path, metrics=("score",))
    assert columns.candidate_at(1) == candidates[0]
    assert columns.status_at(1) == "failed"
    assert columns.error_at(1) == "collapsed"
    assert columns.metrics["score"][0] == 3.0
    assert np.isnan(columns.metrics["score"][1])
    assert columns.status_at(2) == "failed"
    assert columns.error_at(2) == "quarantined"
    assert np.isnan(columns.metrics["score"][2])
    assert {path.name for path in tmp_path.iterdir()} == {"run.json", "results.npz"}
    with zipfile.ZipFile(tmp_path / "results.npz") as archive:
        assert archive.namelist() == [
            "index_00000000.npy",
            "metric_0000_00000000.npy",
            "errors_00000000.npy",
        ]


def test_column_reader_selects_metrics_and_one_candidate(tmp_path):
    candidates = tuple(
        Candidate(f"c{index}", [float(index), float(index + 1)], {"A": index})
        for index in range(128)
    )
    results = tuple(
        CandidateResult(
            candidate.candidate_id,
            metrics={"wanted": float(index), "unused_a": 1.0, "unused_b": 2.0},
            ordinal=index,
        )
        for index, candidate in enumerate(candidates)
    )
    write_results(ResultBundle("columns", candidates, results), tmp_path)

    columns = read_result_columns(tmp_path, metrics=("wanted",))

    assert tuple(columns.metrics) == ("wanted",)
    assert columns.metrics["wanted"].shape == (128,)
    assert not hasattr(columns, "candidates") and not hasattr(columns, "results")
    assert columns.candidate_at(73) == candidates[73]
    with np.load(tmp_path / "results.npz", allow_pickle=False) as archive:
        assert len([name for name in archive.files if name.startswith("metric_")]) == 3


def test_result_writers_overwrite_complete_canonical_bundle(tmp_path):
    initial = ResultBundle("initial", (Candidate("old"),), (CandidateResult("old"),))
    write_results(initial, tmp_path)

    replacement = ResultBundle(
        "replacement", (Candidate("new", [3.0]),), (CandidateResult("new", metrics={"score": 7.0}),)
    )
    write_results(replacement, tmp_path)
    assert read_results(tmp_path) == replacement

    writer = ResultWriter(tmp_path, run_id="stream-replacement")
    candidate = Candidate("stream-new", [4.0])
    writer.append([candidate], [CandidateResult("stream-new", metrics={"score": 8.0})])
    writer.finalize()
    loaded = read_results(tmp_path)
    assert loaded.run_id == "stream-replacement"
    assert loaded.candidates == (candidate,)
    assert loaded.results[0].metrics == {"score": 8.0}
