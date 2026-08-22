import json

import numpy as np
import pytest

from fxopt import Candidate, CandidateResult, ResultBundle, read_results, write_results
from fxopt.results import ResultWriter, read_result_columns


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
    for index in range(3):
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


@pytest.mark.parametrize("existing", ["run.json", "results.npz"])
def test_result_writers_refuse_either_existing_canonical_artifact(tmp_path, existing):
    (tmp_path / existing).write_bytes(b"occupied")
    bundle = ResultBundle(
        "immutable",
        (Candidate("c0"),),
        (CandidateResult("c0"),),
    )

    with pytest.raises(FileExistsError):
        write_results(bundle, tmp_path)
    with pytest.raises(FileExistsError):
        ResultWriter(tmp_path, run_id="immutable")
    assert {path.name for path in tmp_path.iterdir()} == {existing}
