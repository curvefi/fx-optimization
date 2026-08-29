"""Evaluator placement primitives built on top of :mod:`fxopt.engine`.

Placement deliberately knows about processes and lanes only.  Candidate creation,
search, and scoring remain callers' concerns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from os import PathLike, fspath
from pathlib import Path, PurePosixPath
import subprocess
from threading import Lock
from typing import Any

from curve_fx_harness_client import EvaluatorClient

from .contract import Candidate, CandidateResult
from .engine import ClientFactory, OptimizerEngine


def _token(value: str | PathLike[str], label: str) -> str:
    """Validate one argv token; subprocess argv supplies the shell boundary."""
    token = fspath(value)
    if not isinstance(token, str):
        raise TypeError(f"{label} must be a string or path")
    if not token or token.startswith("-") or any(
        character.isspace()
        or ord(character) < 32
        or ord(character) == 127
        or character in ";&|$`<>\\\"'(){}"
        for character in token
    ):
        raise ValueError(f"{label} must not contain whitespace or control characters")
    return token


def _argv_token(value: str | PathLike[str], label: str) -> str:
    """Validate a non-shell argv token while allowing ordinary option prefixes."""
    token = fspath(value)
    if not isinstance(token, str):
        raise TypeError(f"{label} must be a string or path")
    if not token or any(
        character.isspace()
        or ord(character) < 32
        or ord(character) == 127
        or character in ";&|$`<>\\\"'(){}"
        for character in token
    ):
        raise ValueError(f"{label} must not contain whitespace or control characters")
    return token


def _client_options(
    options: Mapping[str, Any] | None,
    updates: Mapping[str, Any],
    *,
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(options or {})
    merged.update(updates)
    for name in fixed:
        if name in merged:
            raise TypeError(f"{name} is fixed by the placement factory")
    merged.update(fixed)
    return merged


def local_client_factory(
    executable_path: str | PathLike[str] = "arb_evaluator_ld",
    *,
    work_dir: str | PathLike[str] | None = None,
    workers: int = 1,
    client_options: Mapping[str, Any] | None = None,
    **options: Any,
) -> ClientFactory:
    """Return a factory for a local evaluator in persistent ``serve`` mode."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    executable = _token(executable_path, "executable_path")
    directory = None if work_dir is None else _token(work_dir, "work_dir")
    fixed = {
        "executable_path": executable,
        "work_dir": directory,
        "launch_argv": [executable, "serve", "--workers", str(workers)],
        "verify_local_inputs": True,
    }

    def create() -> EvaluatorClient:
        return EvaluatorClient(
            **_client_options(client_options, options, fixed=fixed),
        )

    return create


def ssh_client_factory(
    host: str,
    executable_path: str | PathLike[str],
    *,
    workers: int = 1,
    remote_prefix: Sequence[str | PathLike[str]] = (),
    ssh_path: str | PathLike[str] = "ssh",
    verify_local_inputs: bool = False,
    client_options: Mapping[str, Any] | None = None,
    **options: Any,
) -> ClientFactory:
    """Return a factory launching the evaluator directly through ``ssh``.

    The remote process receives exactly ``<evaluator> serve``.  In particular,
    no optimizer or Python worker command is composed into the SSH argv.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    remote_host = _token(host, "host")
    executable = _token(executable_path, "executable_path")
    ssh = _token(ssh_path, "ssh_path")
    if isinstance(remote_prefix, (str, bytes)):
        raise TypeError("remote_prefix must be a sequence of argv tokens")
    prefix = [
        _argv_token(value, f"remote_prefix[{index}]")
        for index, value in enumerate(remote_prefix)
    ]
    if not isinstance(verify_local_inputs, bool):
        raise TypeError("verify_local_inputs must be a boolean")
    fixed = {
        "executable_path": executable,
        "work_dir": None,
        "launch_argv": [
            ssh,
            "--",
            remote_host,
            *prefix,
            executable,
            "serve",
            "--workers",
            str(workers),
        ],
        "verify_local_inputs": verify_local_inputs,
    }

    def create() -> EvaluatorClient:
        return EvaluatorClient(
            **_client_options(client_options, options, fixed=fixed),
        )

    return create


def ensure_remote_file(
    host: str,
    local_path: str | PathLike[str],
    remote_path: str | PathLike[str],
) -> None:
    """Publish a missing shared-NFS input through one placement host."""
    remote_host = _token(host, "host")
    source = Path(local_path)
    destination = _token(remote_path, "remote_path")
    if not source.is_file():
        raise FileNotFoundError(f"remote input source is not a file: {source}")
    present = subprocess.run(
        ["ssh", "--", remote_host, "test", "-f", destination], check=False
    )
    if present.returncode == 0:
        return
    if present.returncode != 1:
        present.check_returncode()
    parent = str(PurePosixPath(destination).parent)
    subprocess.run(["ssh", "--", remote_host, "mkdir", "-p", parent], check=True)
    subprocess.run(
        ["rsync", "--ignore-existing", "--", str(source), f"{remote_host}:{destination}"],
        check=True,
    )


@dataclass(frozen=True, slots=True)
class PlacementLane:
    """A named evaluator client factory assigned a sequence of batches."""

    name: str
    client_factory: ClientFactory

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("lane name must be a non-empty string")
        if not callable(self.client_factory):
            raise TypeError("lane client_factory must be callable")


class EvaluatorFleet:
    """Run bounded batches over persistent evaluator/session lanes.

    Batches are assigned round-robin; each lane processes its assigned sequence
    serially while participating lanes execute concurrently.  A fleet can be
    reused for multiple ``evaluate`` calls within one session.
    """

    def __init__(
        self,
        lanes: Sequence[PlacementLane | ClientFactory],
        *,
        session_id: str,
        batch_size: int,
        start_ordinal: int = 0,
        open_session: Mapping[str, Any] | None = None,
        metric_projection: str | None = None,
        metric_fields: Sequence[str] | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        if not lanes:
            raise ValueError("fleet requires at least one lane")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(start_ordinal, bool) or not isinstance(start_ordinal, int) or start_ordinal < 0:
            raise ValueError("start_ordinal must be a non-negative integer")
        normalized: list[PlacementLane] = []
        for index, lane in enumerate(lanes):
            if isinstance(lane, PlacementLane):
                normalized.append(lane)
            elif callable(lane):
                normalized.append(PlacementLane(f"lane-{index}", lane))
            else:
                raise TypeError("lanes must contain PlacementLane or client factories")
        self._engines = tuple(
            OptimizerEngine(
                lane.client_factory,
                session_id=session_id,
                open_session=open_session,
                metric_projection=metric_projection,
                metric_fields=metric_fields,
                observation=observation,
            )
            for lane in normalized
        )
        self.lanes = tuple(normalized)
        self.session_id = session_id
        self.batch_size = batch_size
        self._started = False
        self._closed = False
        self._next_ordinal = start_ordinal
        self._next_lane = 0
        self._evaluation_lock = Lock()

    @property
    def engines(self) -> tuple[OptimizerEngine, ...]:
        """Read-only access to lane engines for diagnostics."""
        return self._engines

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("fleet is closed")
        if self._started:
            return
        futures = []
        try:
            with ThreadPoolExecutor(max_workers=len(self._engines)) as executor:
                futures = [executor.submit(engine.start) for engine in self._engines]
                for future in futures:
                    future.result()
        except Exception:
            for engine in self._engines:
                engine.close()
            raise
        self._started = True

    def evaluate(self, candidates: Iterable[Candidate]) -> list[CandidateResult]:
        # A fleet may be reused by callers, but an engine must never receive two
        # requests concurrently; lane-level workers provide the useful parallelism.
        return [result for wave in self.iter_evaluate(candidates) for result in wave]

    def iter_evaluate(self, candidates: Iterable[Candidate]) -> Iterator[list[CandidateResult]]:
        """Yield globally ordered waves; cross-batch ID uniqueness is caller-owned."""
        with self._evaluation_lock:
            yield from self._iter_evaluate(candidates)

    def _iter_evaluate(self, candidates: Iterable[Candidate]) -> Iterator[list[CandidateResult]]:
        """Evaluate candidates in global input order with global ordinals."""
        if self._closed:
            raise RuntimeError("fleet is closed")
        base_ordinal = self._next_ordinal
        lane_count = len(self._engines)
        capacity = lane_count * self.batch_size
        iterator = iter(candidates)
        total = 0
        started = False

        def next_wave() -> list[Candidate]:
            wave: list[Candidate] = []
            while len(wave) < capacity:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    break
                if not isinstance(candidate, Candidate):
                    raise TypeError("candidates must contain Candidate values")
                wave.append(candidate)
            return wave

        executor: ThreadPoolExecutor | None = None
        try:
            while True:
                wave = next_wave()
                if not wave:
                    break
                if not started:
                    self.start()
                    started = True
                assignments: dict[int, list[tuple[int, Candidate]]] = {}
                active_lanes = min(lane_count, len(wave))
                lane_indices = tuple(
                    (self._next_lane + index) % lane_count
                    for index in range(active_lanes)
                )
                for lane_index in lane_indices:
                    assignments[lane_index] = []
                for offset, candidate in enumerate(wave):
                    assignments[lane_indices[offset % active_lanes]].append(
                        (offset, candidate)
                    )
                self._next_lane = (self._next_lane + active_lanes) % lane_count
                if executor is None:
                    executor = ThreadPoolExecutor(max_workers=lane_count)

                def run_lane(
                    lane_index: int, assignment: list[tuple[int, Candidate]]
                ) -> list[tuple[int, CandidateResult]]:
                    offsets, batch = zip(*assignment, strict=True)
                    engine = self._engines[lane_index]
                    try:
                        results = engine.evaluate(batch)
                    except Exception as exc:
                        first = base_ordinal + total + offsets[0]
                        last = base_ordinal + total + offsets[-1]
                        raise RuntimeError(
                            f"lane {self.lanes[lane_index].name} failed for "
                            f"{len(batch)} striped candidates spanning ordinals {first}..{last}"
                        ) from exc
                    if len(results) != len(batch):  # Defensive seam for injected engines.
                        raise ValueError("lane returned the wrong number of results")
                    return [
                        (offset, replace(result, ordinal=base_ordinal + total + offset))
                        for offset, result in zip(offsets, results, strict=True)
                    ]

                ordered: list[CandidateResult | None] = [None] * len(wave)
                futures = [executor.submit(run_lane, lane, assignment)
                           for lane, assignment in assignments.items()]
                for future in futures:
                    for offset, result in future.result():
                        ordered[offset] = result
                total += len(wave)
                self._next_ordinal += len(wave)
                yield [result for result in ordered if result is not None]
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def close(self) -> None:
        with self._evaluation_lock:
            if self._closed:
                return
            self._closed = True
            first_error: BaseException | None = None
            for engine in self._engines:
                try:
                    engine.close()
                except BaseException as error:  # Ensure every lane gets a close attempt.
                    if first_error is None:
                        first_error = error
            self._started = False
            if first_error is not None:
                raise first_error

    def __enter__(self) -> "EvaluatorFleet":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise


__all__ = [
    "EvaluatorFleet",
    "PlacementLane",
    "ensure_remote_file",
    "local_client_factory",
    "ssh_client_factory",
]
