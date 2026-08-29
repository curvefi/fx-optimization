from __future__ import annotations

from threading import Barrier, Lock
import subprocess

import pytest

from fxopt import Candidate
from fxopt import placement


class RecordingClient:
    def __init__(self, label: str, events: list[tuple], batches: list[tuple], barrier=None) -> None:
        self.label = label
        self.events = events
        self.batches = batches
        self.barrier = barrier
        self.barrier_lock = Lock()

    def start(self):
        self.events.append((self.label, "start"))
        return None

    def open_session(self, session_id, **request):
        self.events.append((self.label, "open", session_id))

    def evaluate_batch(self, candidates, **request):
        with self.barrier_lock:
            barrier = self.barrier
            self.barrier = None
        if barrier is not None:
            barrier.wait(timeout=1)
        self.batches.append((self.label, [item["candidate_id"] for item in candidates]))
        return {
            "results": [
                {"candidate_id": item["candidate_id"], "metrics": {"lane": 1.0}}
                for item in reversed(candidates)
            ]
        }

    def close_session(self, session_id=None):
        self.events.append((self.label, "close", session_id))

    def shutdown(self):
        self.events.append((self.label, "shutdown"))


def test_factories_use_argv_direct_serve_and_validate_tokens(monkeypatch):
    created = []

    def fake_client(**kwargs):
        created.append(kwargs)
        return kwargs

    monkeypatch.setattr(placement, "EvaluatorClient", fake_client)
    local = placement.local_client_factory("/opt/evaluator", work_dir="/tmp/run", workers=2)()
    remote = placement.ssh_client_factory(
        "blade-a1", "/home/heswithme/arb/evaluator", workers=3, verify_local_inputs=True
    )()

    assert local["launch_argv"] == ["/opt/evaluator", "serve", "--workers", "2"]
    assert remote["launch_argv"] == [
        "ssh", *placement.SSH_OPTIONS, "--",
        "blade-a1", "/home/heswithme/arb/evaluator", "serve", "--workers", "3"
    ]
    assert remote["verify_local_inputs"] is True
    assert all("optimizer" not in token and "python" not in token for token in remote["launch_argv"])
    with pytest.raises(ValueError, match="whitespace"):
        placement.ssh_client_factory("blade a1", "/home/heswithme/arb/evaluator")
    with pytest.raises(ValueError):
        placement.ssh_client_factory("--proxy-command=bad", "/home/heswithme/arb/evaluator")
    with pytest.raises(ValueError):
        placement.ssh_client_factory("blade-a1", "--help")
    with pytest.raises(ValueError, match="workers"):
        placement.local_client_factory(workers=0)
    assert len(created) == 2


def test_ssh_factory_uses_absolute_evaluator_without_remote_workdir(monkeypatch):
    monkeypatch.setattr(placement, "EvaluatorClient", lambda **kwargs: kwargs)
    remote = placement.ssh_client_factory(
        "blade-a1", "/home/heswithme/arb/run/evaluator", workers=2
    )()

    assert remote["work_dir"] is None
    assert remote["launch_argv"] == [
        "ssh", *placement.SSH_OPTIONS, "--",
        "blade-a1", "/home/heswithme/arb/run/evaluator", "serve", "--workers", "2",
    ]


@pytest.mark.parametrize("present", [True, False])
def test_ensure_remote_file_copies_only_when_missing(tmp_path, monkeypatch, present):
    source = tmp_path / "candles.json"
    source.write_text("[]")
    calls = []

    def fake_run(argv, *, check):
        calls.append((argv, check))
        return subprocess.CompletedProcess(argv, 0 if present or len(calls) > 1 else 1)

    monkeypatch.setattr(placement.subprocess, "run", fake_run)
    placement.ensure_remote_file(
        "blade-b6", source, "/home/heswithme/arb/optimizer/data/candles.json"
    )

    assert calls[0] == ([
        "ssh", *placement.SSH_OPTIONS, "--",
        "blade-b6", "test", "-f", "/home/heswithme/arb/optimizer/data/candles.json",
    ], False)
    assert len(calls) == (1 if present else 3)
    if not present:
        assert calls[1] == (["ssh", *placement.SSH_OPTIONS, "--",
                             "blade-b6", "mkdir", "-p",
                             "/home/heswithme/arb/optimizer/data"], True)
        assert calls[2] == (["rsync", "-a", "-e", placement.RSYNC_SSH,
                             "--ignore-existing", "--", str(source),
                             "blade-b6:/home/heswithme/arb/optimizer/data/candles.json"], True)


def test_fleet_balances_full_wave_across_all_lanes():
    batches: list[tuple] = []
    clients = {
        name: RecordingClient(name, [], batches) for name in ("a", "b")
    }
    fleet = placement.EvaluatorFleet(
        [lambda name=name: clients[name] for name in clients],
        session_id="session",
        batch_size=256,
    )

    fleet.evaluate(Candidate(f"c{index}") for index in range(256))
    fleet.close()

    assert sorted(len(batch) for _, batch in batches) == [128, 128]


def test_fleet_balances_partial_waves_and_preserves_global_order():
    events: list[tuple] = []
    batches: list[tuple] = []
    barrier = Barrier(2)
    clients = {
        name: RecordingClient(name, events, batches, barrier)
        for name in ("a", "b")
    }
    factories = {
        name: (lambda name=name: clients[name])
        for name in clients
    }
    lane_updates = []
    fleet = placement.EvaluatorFleet(
        [placement.PlacementLane(name, factory) for name, factory in factories.items()],
        session_id="session",
        batch_size=2,
        lane_callback=lambda name, count, elapsed: lane_updates.append((name, count, elapsed)),
    )

    candidates = [Candidate(f"c{index}") for index in range(5)]
    results = fleet.evaluate(candidates)
    follow_up = fleet.evaluate([Candidate("c5")])
    fleet.close()

    assert [result.candidate_id for result in results] == [candidate.candidate_id for candidate in candidates]
    assert [result.ordinal for result in results] == list(range(5))
    assert [result.ordinal for result in follow_up] == [5]
    assert sorted(batches[:3]) == sorted(
        [("a", ["c0", "c2"]), ("b", ["c1", "c3"]), ("a", ["c4"])]
    )
    assert batches[3] == ("b", ["c5"])
    assert sorted((name, count) for name, count, _elapsed in lane_updates) == [
        ("a", 1), ("a", 2), ("b", 1), ("b", 2),
    ]
    for name in clients:
        assert events.count((name, "start")) == 1
        assert events.count((name, "open", "session")) == 1
        assert events.count((name, "close", "session")) == 1
        assert events.count((name, "shutdown")) == 1


def test_fleet_consumes_generators_in_bounded_waves():
    events: list[tuple] = []
    pulled: list[int] = []

    class PullAwareClient(RecordingClient):
        def evaluate_batch(self, candidates, **request):
            events.append((self.label, "pulled", len(pulled)))
            return super().evaluate_batch(candidates, **request)

    clients = {
        name: PullAwareClient(name, events, []) for name in ("a", "b")
    }
    fleet = placement.EvaluatorFleet(
        [lambda name=name: clients[name] for name in clients],
        session_id="session",
        batch_size=2,
    )

    def candidates():
        for index in range(7):
            pulled.append(index)
            yield Candidate(f"c{index}")

    results = fleet.evaluate(candidates())
    fleet.close()

    assert [result.candidate_id for result in results] == [f"c{index}" for index in range(7)]
    assert sorted(event[2] for event in events if event[1] == "pulled") == [4, 4, 7, 7]


def test_fleet_streams_a_wave_before_consuming_later_candidates():
    pulled: list[int] = []
    clients = {
        name: RecordingClient(name, [], []) for name in ("a", "b")
    }
    fleet = placement.EvaluatorFleet(
        [lambda name=name: clients[name] for name in clients],
        session_id="session",
        batch_size=2,
    )

    def candidates():
        for index in range(7):
            pulled.append(index)
            yield Candidate(f"c{index}")

    stream = fleet.iter_evaluate(candidates())
    first = next(stream)
    assert pulled == [0, 1, 2, 3]
    second = next(stream)
    assert pulled == list(range(7))
    assert [item.ordinal for item in first + second] == list(range(7))
    assert list(stream) == []
    fleet.close()


def test_grid_fleet_records_failed_chunk_and_restarts_only_that_lane():
    events: list[tuple] = []
    batches: list[tuple] = []
    created = {"a": 0, "b": 0}

    class FailFirstClient(RecordingClient):
        def evaluate_batch(self, candidates, **request):
            raise RuntimeError("evaluator exited")

    def factory(name):
        def create():
            created[name] += 1
            cls = FailFirstClient if name == "a" and created[name] == 1 else RecordingClient
            return cls(name, events, batches)

        return create

    fleet = placement.EvaluatorFleet(
        [
            placement.PlacementLane("a", factory("a")),
            placement.PlacementLane("b", factory("b")),
        ],
        session_id="session",
        batch_size=2,
    )
    assignments = (
        tuple((index, Candidate(f"p{index:08d}")) for index in (0, 2, 4, 6)),
        tuple((index, Candidate(f"p{index:08d}")) for index in (1, 3, 5, 7)),
    )

    completed = list(fleet.iter_grid(assignments))
    fleet.close()

    results = sorted(
        (result for batch in completed for result in batch.results),
        key=lambda result: result.ordinal,
    )
    assert [result.ordinal for result in results] == list(range(8))
    assert [result.status for result in results] == [
        "failed", "ok", "failed", "ok", "ok", "ok", "ok", "ok",
    ]
    assert all(not result.metrics for result in results if result.status == "failed")
    assert sum(result.error is not None for result in results) == 1
    assert created == {"a": 2, "b": 1}


def test_grid_fleet_quarantines_a_lane_after_three_consecutive_failures():
    created = 0

    class AlwaysFailClient(RecordingClient):
        def evaluate_batch(self, candidates, **request):
            raise RuntimeError("still broken")

    def factory():
        nonlocal created
        created += 1
        return AlwaysFailClient("a", [], [])

    fleet = placement.EvaluatorFleet(
        [placement.PlacementLane("a", factory)],
        session_id="session",
        batch_size=1,
    )
    assignments = (tuple(
        (index, Candidate(f"p{index:08d}")) for index in range(10)
    ),)

    completed = list(fleet.iter_grid(assignments))
    fleet.close()

    assert created == 3
    assert all(batch.results[0].status == "failed" for batch in completed)
    assert "quarantined" not in completed[2].error
    assert "quarantined" in completed[3].error
