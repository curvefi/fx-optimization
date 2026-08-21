"""Persistent evaluator lifecycle for single-point, grid, and adaptive callers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from .contract import Candidate, CandidateResult


class EvaluatorClient(Protocol):
    """Minimal lifecycle implemented by the local or remote harness client."""

    def start(self) -> Any: ...

    def open_session(self, session_id: str, **request: Any) -> Any: ...

    def evaluate_batch(self, candidates: Sequence[Mapping[str, Any]], **request: Any) -> Any: ...

    def close_session(self, session_id: str | None = None) -> Any: ...

    def shutdown(self) -> Any: ...


ClientFactory = Callable[[], EvaluatorClient]


def _response_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_result(value: Any, candidate: Candidate, ordinal: int) -> CandidateResult:
    if isinstance(value, CandidateResult):
        if value.candidate_id != candidate.candidate_id:
            raise ValueError("evaluator returned a result for the wrong candidate")
        return CandidateResult(
            candidate_id=candidate.candidate_id,
            status=value.status,
            metrics=value.metrics,
            error=value.error,
            economic_fingerprint=value.economic_fingerprint,
            ordinal=ordinal,
        )
    candidate_id = _response_field(value, "candidate_id")
    if candidate_id != candidate.candidate_id:
        raise ValueError(
            f"evaluator returned candidate_id {candidate_id!r}; expected {candidate.candidate_id!r}"
        )
    metrics = _response_field(value, "metrics", {})
    if metrics is None:
        metrics = {}
    status = _response_field(value, "status", "ok")
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        status=status,
        metrics=metrics,
        error=_response_field(value, "error"),
        economic_fingerprint=_response_field(value, "economic_fingerprint"),
        ordinal=ordinal,
    )


class OptimizerEngine:
    """Keep one evaluator process and one open session across repeated batches."""

    def __init__(
        self,
        client_factory: ClientFactory,
        *,
        session_id: str,
        open_session: Mapping[str, Any] | None = None,
        metric_projection: str | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        self._factory = client_factory
        self.session_id = session_id
        self._open_request = dict(open_session or {})
        if "session_id" in self._open_request:
            raise ValueError("session_id belongs to the engine, not open_session")
        self._batch_request: dict[str, Any] = {}
        if metric_projection is not None:
            self._batch_request["metric_projection"] = metric_projection
        if observation is not None:
            self._batch_request["observation"] = dict(observation)
        self._client: EvaluatorClient | None = None
        self._started = False
        self._session_open = False
        self._closed = False
        self._next_ordinal = 0

    @property
    def client(self) -> EvaluatorClient | None:
        """Expose the injected client for read-only diagnostics and test fakes."""
        return self._client

    def start(self) -> Any:
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._started:
            return None
        self._client = self._factory()
        if self._client is None:
            raise RuntimeError("client_factory returned None")
        try:
            hello = self._client.start()
            self._client.open_session(self.session_id, **self._open_request)
        except Exception:
            self._client.shutdown()
            self._client = None
            raise
        self._started = True
        self._session_open = True
        return hello

    def evaluate(self, candidates: Iterable[Candidate]) -> list[CandidateResult]:
        """Evaluate candidates in input order while preserving one session."""
        items = list(candidates)
        if not items:
            return []
        if any(not isinstance(item, Candidate) for item in items):
            raise TypeError("candidates must contain Candidate values")
        candidate_ids = [item.candidate_id for item in items]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within a batch")
        self.start()
        assert self._client is not None
        start = self._next_ordinal
        payload = [item.to_dict(ordinal=start + index) for index, item in enumerate(items)]
        request = dict(self._batch_request)
        request["session_id"] = self.session_id
        response = self._client.evaluate_batch(payload, **request)
        values = _response_field(response, "results", response)
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise TypeError("evaluator batch response must contain iterable results")
        values = list(values)
        if len(values) != len(items):
            raise ValueError(
                f"evaluator returned {len(values)} results for {len(items)} candidates"
            )
        by_id: dict[str, Any] = {}
        for value in values:
            candidate_id = _response_field(value, "candidate_id")
            if candidate_id in by_id:
                raise ValueError(f"evaluator returned duplicate candidate_id {candidate_id!r}")
            by_id[candidate_id] = value
        missing = [item.candidate_id for item in items if item.candidate_id not in by_id]
        if missing:
            raise ValueError(f"evaluator omitted candidate IDs: {missing!r}")
        results = [
            _normalize_result(by_id[item.candidate_id], item, start + index)
            for index, item in enumerate(items)
        ]
        self._next_ordinal += len(items)
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        if client is None:
            return
        try:
            if self._session_open:
                client.close_session(self.session_id)
        finally:
            self._session_open = False
            if self._started:
                client.shutdown()
            self._started = False

    def __enter__(self) -> "OptimizerEngine":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = ["ClientFactory", "EvaluatorClient", "OptimizerEngine"]
