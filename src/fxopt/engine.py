"""Persistent evaluator lifecycle for registered grids and arbitrary points."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

from .candidates import candidate_id
from .contract import Candidate, CandidateResult


class EvaluatorClient(Protocol):
    """Minimal lifecycle implemented by the local or remote harness client."""

    def start(self) -> Any: ...

    def open_session(self, session_id: str, **request: Any) -> Any: ...

    def register_grid(
        self, grid_id: str, grid: Mapping[str, Any], **request: Any
    ) -> Any: ...

    def evaluate_batch(self, candidates: Sequence[Mapping[str, Any]], **request: Any) -> Any: ...

    def close_session(self, session_id: str | None = None) -> Any: ...

    def shutdown(self) -> Any: ...


ClientFactory = Callable[[], EvaluatorClient]


@dataclass(frozen=True, slots=True)
class ProjectedBatch:
    """Validated in-order projected rows returned by the evaluator."""

    metric_fields: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]


def _projected_row(
    value: Any,
    ordinal: int,
    metric_count: int,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("ordinal") != ordinal
        or value.get("candidate_id") != candidate_id(ordinal)
    ):
        raise ValueError("projected evaluator results must preserve input order")
    status = value.get("status", "ok")
    if status not in {"ok", "failed", "cancelled"}:
        raise ValueError("projected evaluator result has an invalid status")
    metrics = value.get("metrics")
    if (
        isinstance(metrics, (str, bytes))
        or not isinstance(metrics, Sequence)
        or len(metrics) != metric_count
        or any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(metric)
            for metric in metrics
        )
    ):
        raise ValueError("projected evaluator result has invalid metrics")
    error = value.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("projected evaluator result has an invalid error")
    artifacts = value.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, Mapping):
        raise ValueError("projected evaluator result has invalid artifacts")
    return value


def _response_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_result(
    value: Any,
    candidate: Candidate,
    ordinal: int,
    metric_fields: Sequence[str] | None = None,
) -> CandidateResult:
    if isinstance(value, CandidateResult):
        if value.candidate_id != candidate.candidate_id:
            raise ValueError("evaluator returned a result for the wrong candidate")
        return CandidateResult(
            candidate_id=candidate.candidate_id,
            status=value.status,
            metrics=value.metrics,
            error=value.error,
            artifacts=value.artifacts,
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
    if metric_fields is not None and not isinstance(metrics, Mapping):
        values = tuple(metrics)
        if len(values) != len(metric_fields):
            raise ValueError("evaluator metric array has the wrong length")
        return CandidateResult(
            candidate_id=candidate.candidate_id,
            status=_response_field(value, "status", "ok"),
            metrics=dict(zip(metric_fields, values, strict=True)),
            error=_response_field(value, "error"),
            artifacts=_response_field(value, "artifacts"),
            ordinal=ordinal,
        )
    status = _response_field(value, "status", "ok")
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        status=status,
        metrics=metrics,
        error=_response_field(value, "error"),
        artifacts=_response_field(value, "artifacts"),
        ordinal=ordinal,
    )


class EvaluatorSession:
    """Keep one evaluator process and one immutable session across requests.

    ``evaluate`` is the small point-batch seam for replay or a future adaptive
    driver. Registered-grid range evaluation is the specialized bulk path.
    """

    def __init__(
        self,
        client_factory: ClientFactory,
        *,
        session_id: str,
        open_session: Mapping[str, Any] | None = None,
        metric_fields: Sequence[str] | None = None,
        observation: Mapping[str, Any] | None = None,
        grid: Mapping[str, Any] | None = None,
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
        self._metric_fields: tuple[str, ...] | None = None
        if metric_fields is not None:
            fields = tuple(metric_fields)
            if (
                not fields
                or any(not isinstance(name, str) or not name for name in fields)
                or len(set(fields)) != len(fields)
            ):
                raise ValueError(
                    "metric_fields must contain unique non-empty strings"
                )
            self._metric_fields = fields
            self._batch_request["metric_fields"] = list(fields)
            self._batch_request["metrics_format"] = "array"
            self._batch_request["trusted_candidates"] = True
        if observation is not None:
            self._batch_request["observation"] = dict(observation)
        self._grid = None if grid is None else dict(grid)
        self._grid_id = "grid"
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
            if self._grid is not None:
                ready = self._client.register_grid(
                    self._grid_id,
                    self._grid,
                    session_id=self.session_id,
                )
                expected = 1
                for length in self._grid.get("shape", ()):
                    expected *= length
                if _response_field(ready, "candidate_count") != expected:
                    raise ValueError("registered grid size does not match its shape")
        except Exception:
            self._client.shutdown()
            self._client = None
            raise
        self._started = True
        self._session_open = True
        return hello

    def _request_batch(
        self,
        candidates: Iterable[Candidate],
    ) -> tuple[list[Candidate], int, list[Any], tuple[str, ...] | None]:
        items = list(candidates)
        if not items:
            return [], self._next_ordinal, [], self._metric_fields
        if any(not isinstance(item, Candidate) for item in items):
            raise TypeError("candidates must contain Candidate values")
        candidate_ids = [item.candidate_id for item in items]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within a batch")
        self.start()
        assert self._client is not None
        start = self._next_ordinal
        payload = [
            item.to_dict(ordinal=start + index)
            for index, item in enumerate(items)
        ]
        request = dict(self._batch_request)
        request["session_id"] = self.session_id
        response = self._client.evaluate_batch(payload, **request)
        values = _response_field(response, "results", response)
        response_metric_fields = _response_field(response, "metric_fields")
        if self._metric_fields is not None:
            if tuple(response_metric_fields or ()) != self._metric_fields:
                raise ValueError(
                    "evaluator metric_fields do not match the request"
                )
            response_metric_fields = self._metric_fields
        else:
            response_metric_fields = None
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise TypeError("evaluator batch response must contain iterable results")
        values = list(values)
        if len(values) != len(items):
            raise ValueError(
                f"evaluator returned {len(values)} results for {len(items)} candidates"
            )
        return items, start, values, response_metric_fields

    def evaluate(self, candidates: Iterable[Candidate]) -> list[CandidateResult]:
        """Evaluate candidates in input order while preserving one session."""
        items, start, values, response_metric_fields = self._request_batch(candidates)
        if not items:
            return []
        if all(
            _response_field(value, "candidate_id") == item.candidate_id
            for value, item in zip(values, items, strict=True)
        ):
            ordered = values
        else:
            by_id: dict[str, Any] = {}
            for value in values:
                candidate_id = _response_field(value, "candidate_id")
                if candidate_id in by_id:
                    raise ValueError(
                        f"evaluator returned duplicate candidate_id {candidate_id!r}"
                    )
                by_id[candidate_id] = value
            missing = [
                item.candidate_id
                for item in items
                if item.candidate_id not in by_id
            ]
            if missing:
                raise ValueError(f"evaluator omitted candidate IDs: {missing!r}")
            ordered = [by_id[item.candidate_id] for item in items]
        results = [
            _normalize_result(
                value, item, start + index,
                response_metric_fields,
            )
            for index, (item, value) in enumerate(zip(items, ordered, strict=True))
        ]
        self._next_ordinal += len(items)
        return results

    def evaluate_projected_ranges(
        self,
        ranges: Sequence[tuple[int, int]],
    ) -> ProjectedBatch:
        """Evaluate ordered, disjoint canonical ordinal ranges."""
        if self._grid is None or self._metric_fields is None:
            raise ValueError("projected grid evaluation requires grid metadata")
        ordinals = tuple(
            ordinal
            for start, count in ranges
            for ordinal in range(start, start + count)
        )
        if not ordinals:
            return ProjectedBatch(self._metric_fields, ())
        self.start()
        assert self._client is not None
        request = dict(self._batch_request)
        request["session_id"] = self.session_id
        request["grid_id"] = self._grid_id
        request["ranges"] = [list(item) for item in ranges]
        response = self._client.evaluate_batch([], **request)
        if tuple(_response_field(response, "metric_fields") or ()) != self._metric_fields:
            raise ValueError("evaluator metric_fields do not match the request")
        values = _response_field(response, "results", response)
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise TypeError("evaluator batch response must contain iterable results")
        rows = tuple(values)
        if len(rows) != len(ordinals):
            raise ValueError("evaluator returned the wrong number of grid results")
        validated = tuple(
            _projected_row(row, ordinal, len(self._metric_fields))
            for ordinal, row in zip(ordinals, rows, strict=True)
        )
        return ProjectedBatch(self._metric_fields, validated)

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

    def __enter__(self) -> "EvaluatorSession":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = ["ClientFactory", "EvaluatorClient", "EvaluatorSession", "ProjectedBatch"]
