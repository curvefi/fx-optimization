from __future__ import annotations

from threading import Barrier, Lock

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
        "blade-a1", "/shared/evaluator", workers=3, verify_local_inputs=True
    )()

    assert local["launch_argv"] == ["/opt/evaluator", "serve", "--workers", "2"]
    assert remote["launch_argv"] == [
        "ssh", "--", "blade-a1", "/shared/evaluator", "serve", "--workers", "3"
    ]
    assert remote["verify_local_inputs"] is True
    assert all("optimizer" not in token and "python" not in token for token in remote["launch_argv"])
    with pytest.raises(ValueError, match="whitespace"):
        placement.ssh_client_factory("blade a1", "/shared/evaluator")
    with pytest.raises(ValueError):
        placement.ssh_client_factory("--proxy-command=bad", "/shared/evaluator")
    with pytest.raises(ValueError):
        placement.ssh_client_factory("blade-a1", "--help")
    with pytest.raises(ValueError, match="workers"):
        placement.local_client_factory(workers=0)
    assert len(created) == 2


def test_ssh_factory_uses_absolute_evaluator_without_remote_workdir(monkeypatch):
    monkeypatch.setattr(placement, "EvaluatorClient", lambda **kwargs: kwargs)
    remote = placement.ssh_client_factory("blade-a1", "/shared/run/evaluator", workers=2)()

    assert remote["work_dir"] is None
    assert remote["launch_argv"] == [
        "ssh", "--", "blade-a1", "/shared/run/evaluator", "serve", "--workers", "2",
    ]


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


def test_fleet_round_robins_whole_batches_and_preserves_global_order():
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
    fleet = placement.EvaluatorFleet(
        [placement.PlacementLane(name, factory) for name, factory in factories.items()],
        session_id="session",
        batch_size=2,
    )

    candidates = [Candidate(f"c{index}") for index in range(5)]
    results = fleet.evaluate(candidates)
    follow_up = fleet.evaluate([Candidate("c5")])
    fleet.close()

    assert [result.candidate_id for result in results] == [candidate.candidate_id for candidate in candidates]
    assert [result.ordinal for result in results] == list(range(5))
    assert [result.ordinal for result in follow_up] == [5]
    assert sorted(batches[:3]) == sorted(
        [("a", ["c0", "c1"]), ("b", ["c2", "c3"]), ("a", ["c4"])]
    )
    assert batches[3] == ("b", ["c5"])
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
