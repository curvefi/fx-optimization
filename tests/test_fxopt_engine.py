from fxopt import Candidate, OptimizerEngine


class FakeClient:
    def __init__(self):
        self.events = []
        self.batches = []

    def start(self):
        self.events.append("start")
        return {"hello": True}

    def open_session(self, session_id, **request):
        self.events.append(("open", session_id, request))

    def evaluate_batch(self, candidates, **request):
        self.events.append("evaluate")
        self.batches.append((candidates, request))
        return {"results": [
            {"candidate_id": item["candidate_id"], "status": "ok", "metrics": {"score": float(item["ordinal"])}}
            for item in reversed(candidates)
        ]}

    def close_session(self, session_id=None):
        self.events.append(("close", session_id))

    def shutdown(self):
        self.events.append("shutdown")


def test_engine_reuses_one_client_session_and_orders_repeated_batches():
    fake = FakeClient()
    engine = OptimizerEngine(
        lambda: fake,
        session_id="session-1",
        open_session={"n_candles": 3},
    )

    first = engine.evaluate([Candidate("b", [2]), Candidate("a", [1])])
    second = engine.evaluate([Candidate("c", [3])])
    engine.close()

    assert [result.candidate_id for result in first + second] == ["b", "a", "c"]
    assert [result.ordinal for result in first + second] == [0, 1, 2]
    assert [batch[1]["session_id"] for batch in fake.batches] == ["session-1", "session-1"]
    assert [event for event in fake.events if event == "start"] == ["start"]
    assert [event for event in fake.events if isinstance(event, tuple) and event[0] == "open"] == [
        ("open", "session-1", {"n_candles": 3})
    ]
    assert fake.events[-2:] == [("close", "session-1"), "shutdown"]
