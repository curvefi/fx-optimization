import zipfile

import numpy as np
from fxopt import Candidate
from fxopt.engine import ProjectedBatch
from fxopt.results import (
    GridResultWriter,
    merge_grid_partitions,
    read_result_columns,
)


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
        metric_names=("score", "unused"),
    )
    candidates = (Candidate("p00000001", pool_overrides={"A": 2}),
                  Candidate("p00000000", pool_overrides={"A": 1}))
    writer.append_projected(
        (1, 0),
        ProjectedBatch(("score", "unused"), (
            {
                "candidate_id": "p00000001", "status": "failed",
                "error": "collapsed", "metrics": [-1.0, -1.0],
            },
            {
                "candidate_id": "p00000000", "status": "ok",
                "metrics": [3.0, 4.0],
            },
        )),
    )
    writer.append_projected(
        (2,),
        ProjectedBatch(
            ("score", "unused"),
            ({
                "candidate_id": "p00000002",
                "status": "failed",
                "error": "quarantined",
                "metrics": [-1.0, -2.0],
            },),
        ),
    )
    writer.finalize()

    columns = read_result_columns(tmp_path, metrics=("score",))
    assert tuple(columns.metrics) == ("score",)
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
            "metric_0001_00000000.npy",
            "errors_00000000.npy",
        ]


def test_grid_partitions_merge_by_global_ordinal(tmp_path):
    metadata = {
        "candidate_defaults": {"policy_params": [], "pool": {}},
        "axes": {"pool.A": [1, 2, 3, 4]},
        "shape": [4],
    }
    partitions = []
    for worker, ordinals in enumerate(((3, 0), (2, 1))):
        partition = tmp_path / f"worker-{worker}"
        writer = GridResultWriter(
            partition,
            run_id="grid",
            total=4,
            expected_count=2,
            metadata={"worker": worker},
            metric_names=("score",),
            shard_rows=1,
        )
        writer.append_projected(
            ordinals,
            ProjectedBatch(
                ("score",),
                tuple({
                    "candidate_id": f"p{ordinal:08d}",
                    "status": "ok",
                    "metrics": [float(ordinal)],
                } for ordinal in ordinals),
            ),
        )
        writer.finalize()
        partitions.append(partition)

    merge_grid_partitions(
        tmp_path / "merged",
        partitions,
        run_id="grid",
        total=4,
        metadata=metadata,
        metric_names=("score",),
    )

    columns = read_result_columns(tmp_path / "merged", metrics=("score",))
    assert columns.ordinals.tolist() == [0, 1, 2, 3]
    assert columns.metrics["score"].tolist() == [0.0, 1.0, 2.0, 3.0]
