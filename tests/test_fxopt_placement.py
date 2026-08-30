from __future__ import annotations

import math
import subprocess

import pytest

from fxopt import placement


class GridClient:
    def __init__(self, *, invalid=False, failure=None, requests=None):
        self.invalid = invalid
        self.failure = failure
        self.requests = [] if requests is None else requests

    def start(self):
        return None

    def open_session(self, session_id, **request):
        return None

    def register_grid(self, grid_id, grid, **request):
        count = 1
        for length in grid["shape"]:
            count *= length
        return {"candidate_count": count}

    def evaluate_batch(self, candidates, **request):
        if self.failure is not None:
            raise self.failure
        ranges = tuple(tuple(item) for item in request["ranges"])
        self.requests.append(ranges)
        ordinals = [
            ordinal
            for start, count in ranges
            for ordinal in range(start, start + count)
        ]
        return {
            "metric_fields": ["score"],
            "results": [{
                "ordinal": ordinal,
                "candidate_id": f"p{ordinal:08d}",
                "status": "ok",
                "metrics": [math.nan if self.invalid else float(ordinal)],
            } for ordinal in ordinals],
        }

    def close_session(self, session_id=None):
        return None

    def shutdown(self):
        return None


def _fleet(factories, *, size=4):
    return placement.EvaluatorFleet(
        [placement.PlacementLane(f"lane-{index}", factory)
         for index, factory in enumerate(factories)],
        session_id="session",
        metric_fields=("score",),
        grid={
            "candidate_defaults": {"policy_params": [], "pool": {}},
            "axes": {"pool.A": list(range(size))},
            "axis_order": ["pool.A"],
            "shape": [size],
        },
    )


def test_local_factory_and_copy_if_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(placement, "EvaluatorClient", lambda **kwargs: kwargs)
    local = placement.local_client_factory(
        "/opt/evaluator",
        work_dir="/tmp/run",
        workers=2,
        launch_prefix=("numactl", "--cpunodebind=0"),
    )()
    assert local["launch_argv"] == [
        "numactl", "--cpunodebind=0", "/opt/evaluator",
        "serve", "--workers", "2",
    ]
    with pytest.raises(ValueError, match="workers"):
        placement.local_client_factory(workers=0)

    source = tmp_path / "candles.json"
    source.write_text("[]")
    calls = []

    def fake_run(argv, *, check):
        calls.append((argv, check))
        return subprocess.CompletedProcess(argv, 0 if len(calls) > 1 else 1)

    monkeypatch.setattr(placement.subprocess, "run", fake_run)
    placement.ensure_remote_file(
        "blade-b6", source, "/home/heswithme/arb/data/candles.json"
    )
    assert [call[0][0] for call in calls] == ["ssh", "ssh", "rsync"]
    assert "--ignore-existing" in calls[-1][0]


def test_grid_fleet_refills_slots_from_one_range_queue():
    requests = []
    fleet = _fleet((
        lambda: GridClient(requests=requests),
        lambda: GridClient(requests=requests),
    ))
    blocks = iter(((0, 1), (1, 2), (2, 3), (3, 4)))

    batches = list(fleet.iter_grid_ranges(blocks))
    fleet.close()

    assert sorted(ordinal for batch in batches for ordinal in batch.ordinals) == list(range(4))
    assert sorted(requests) == [((0, 1),), ((1, 1),), ((2, 1),), ((3, 1),)]


def test_grid_fleet_retries_one_range_then_fails_cleanly():
    created = 0

    def recover():
        nonlocal created
        created += 1
        return GridClient(invalid=created == 1)

    fleet = _fleet((recover,), size=1)
    batches = list(fleet.iter_grid_ranges(((0, 1),)))
    fleet.close()
    assert created == 2
    assert batches[0].projected.rows[0]["metrics"] == [0.0]

    failures = 0

    def broken():
        nonlocal failures
        failures += 1
        return GridClient(failure=RuntimeError("broken"))

    fleet = _fleet((broken,), size=1)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        list(fleet.iter_grid_ranges(((0, 1),)))
    fleet.close()
    assert failures == 3
