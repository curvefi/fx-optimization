import pytest

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


def test_engine_normalizes_only_the_requested_metric_array_shape():
    class ArrayClient(FakeClient):
        def __init__(self, fields, metrics):
            super().__init__()
            self.fields = fields
            self.metrics = metrics

        def evaluate_batch(self, candidates, **request):
            self.batches.append((candidates, request))
            return {
                "metric_fields": self.fields,
                "results": [
                    {
                        "candidate_id": candidates[0]["candidate_id"],
                        "status": "ok",
                        "metrics": self.metrics,
                    }
                ],
            }

    good = ArrayClient(["score", "apy"], [2.0, 0.1])
    engine = OptimizerEngine(
        lambda: good,
        session_id="metric-arrays",
        metric_fields=["score", "apy"],
    )
    [result] = engine.evaluate([Candidate("a", [1])])
    engine.close()

    assert good.batches[0][1]["metric_fields"] == ["score", "apy"]
    assert good.batches[0][1]["metrics_format"] == "array"
    assert dict(result.metrics) == {"score": 2.0, "apy": 0.1}

    wrong_fields = ArrayClient(["apy", "score"], [0.1, 2.0])
    engine = OptimizerEngine(
        lambda: wrong_fields,
        session_id="wrong-fields",
        metric_fields=["score", "apy"],
    )
    with pytest.raises(ValueError, match="metric_fields"):
        engine.evaluate([Candidate("a", [1])])
    engine.close()

    wrong_length = ArrayClient(["score", "apy"], [2.0])
    engine = OptimizerEngine(
        lambda: wrong_length,
        session_id="wrong-length",
        metric_fields=["score", "apy"],
    )
    with pytest.raises(ValueError, match="wrong length"):
        engine.evaluate([Candidate("a", [1])])
    engine.close()

    non_finite = ArrayClient(["score", "apy"], [2.0, float("nan")])
    engine = OptimizerEngine(
        lambda: non_finite,
        session_id="non-finite",
        metric_fields=["score", "apy"],
    )
    with pytest.raises(ValueError, match="must be finite"):
        engine.evaluate([Candidate("a", [1])])
    engine.close()
