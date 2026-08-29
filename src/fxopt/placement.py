"""Evaluator placement primitives built on top of :mod:`fxopt.engine`.

Placement deliberately knows about processes and lanes only.  Candidate creation,
search, and scoring remain callers' concerns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import islice
from os import PathLike, fspath
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from threading import Lock
import time
from typing import Any

from curve_fx_harness_client import EvaluatorClient

from .contract import Candidate, CandidateResult
from .engine import ClientFactory, OptimizerEngine


SSH_OPTIONS = (
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
)
RSYNC_SSH = (
    "ssh -o ControlMaster=no -o ControlPath=none "
    "-o ServerAliveInterval=30 -o ServerAliveCountMax=6"
)
REMOTE_BASE = PurePosixPath("/home/heswithme/arb")
_MAX_CONSECUTIVE_GRID_FAILURES = 3


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
            *SSH_OPTIONS,
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
    *,
    replace: bool = False,
) -> None:
    """Publish one shared-NFS input through one placement host."""
    remote_host = _token(host, "host")
    source = Path(local_path)
    destination = _token(remote_path, "remote_path")
    if not source.is_file():
        raise FileNotFoundError(f"remote input source is not a file: {source}")
    present = subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", remote_host, "test", "-f", destination],
        check=False,
    )
    if present.returncode == 0 and not replace:
        return
    if present.returncode != 1:
        present.check_returncode()
    parent = str(PurePosixPath(destination).parent)
    subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", remote_host, "mkdir", "-p", parent],
        check=True,
    )
    command = ["rsync", "-a", "-e", RSYNC_SSH]
    if not replace:
        command.append("--ignore-existing")
    subprocess.run([*command, "--", str(source), f"{remote_host}:{destination}"], check=True)


def require_reachable_hosts(hosts: Sequence[str]) -> tuple[str, ...]:
    """Fail once with the complete unreachable-host list."""
    if not hosts:
        return ()

    def reachable(host: str) -> tuple[str, bool]:
        result = subprocess.run(
            [
                "ssh", *SSH_OPTIONS,
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=8",
                "-o", "ConnectionAttempts=1",
                "--", host, "true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return host, result.returncode == 0

    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        results = dict(executor.map(reachable, hosts))
    missing = [host for host in hosts if not results[host]]
    if missing:
        raise RuntimeError("unreachable placement hosts: " + ", ".join(missing))
    return tuple(hosts)


def transfer_workspace(host: str, workspace: str | Path) -> None:
    """Rsync evaluator and coordinator sources once into shared cluster home."""
    remote_host = _token(host, "host")
    root = Path(workspace)
    excludes = (
        ".git/", ".venv/", ".venv-cluster/", "__pycache__/", "build/",
        "build-*/", "_install/",
    )
    for name in ("twocrypto-cpp", "curve-fx-arb-harness", "curve-fx-optimization"):
        source = root / name
        if not source.is_dir():
            raise FileNotFoundError(f"workspace source repository is missing: {source}")
        destination = str(REMOTE_BASE / name)
        subprocess.run(
            ["ssh", *SSH_OPTIONS, "--", remote_host, "mkdir", "-p", destination],
            check=True,
        )
        command = ["rsync", "-a", "-e", RSYNC_SSH]
        patterns = excludes + (
            ("configs/", "data/", "runs/")
            if name == "curve-fx-optimization"
            else ()
        )
        for pattern in patterns:
            command.extend(("--exclude", pattern))
        subprocess.run(
            [*command, "--", f"{source}/", f"{remote_host}:{destination}/"],
            check=True,
        )


def rebuild_shared_evaluator(host: str, evaluator: str) -> None:
    """Build the configured evaluator once in the shared cluster workspace."""
    remote_host = _token(host, "host")
    executable = PurePosixPath(_token(evaluator, "evaluator"))
    build_root = REMOTE_BASE / "curve-fx-arb-harness" / "build"
    if executable.parent == build_root or not executable.parent.is_relative_to(build_root):
        raise ValueError(f"evaluator build directory must be below {build_root}")
    build_dir = str(executable.parent)
    target = executable.name
    quoted_build = shlex.quote(build_dir)
    quoted_target = shlex.quote(target)
    build_script = " ".join((
        "set -euo pipefail;",
        f"root={shlex.quote(str(REMOTE_BASE))};",
        'cmake -S "$root/twocrypto-cpp" -B "$root/twocrypto-cpp/build/cluster-release"',
        '-DCMAKE_BUILD_TYPE=Release -DTWOCRYPTO_POOL_BUILD_TESTS=OFF',
        '-DTWOCRYPTO_POOL_BUILD_BENCHMARKS=OFF',
        '-DCMAKE_INSTALL_PREFIX="$root/twocrypto-cpp/_install";',
        'cmake --build "$root/twocrypto-cpp/build/cluster-release" --parallel --target install;',
        f"rm -rf -- {quoted_build};",
        f'cmake -S "$root/curve-fx-arb-harness" -B {quoted_build}',
        '-DCMAKE_BUILD_TYPE=Release -DCURVE_FX_NATIVE_TUNING=ON',
        '-DCURVE_FX_ENABLE_IPO=ON -DCMAKE_PREFIX_PATH="$root/twocrypto-cpp/_install";',
        f'cmake --build {quoted_build} --parallel --target {quoted_target}',
    ))
    command = (
        "nix-shell -p gcc cmake boost gnumake --run "
        + shlex.quote(build_script)
    )
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", remote_host, command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = "\n".join(
            line for line in (result.stdout + result.stderr).splitlines()[-20:] if line
        )
        raise RuntimeError("shared evaluator build failed" + (f":\n{detail}" if detail else ""))


def require_shared_evaluator(host: str, evaluator: str) -> None:
    """Check the shared evaluator once, not once per blade."""
    remote_host = _token(host, "host")
    executable = _token(evaluator, "evaluator")
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", remote_host, "test", "-x", executable],
        check=False,
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"shared evaluator is missing: {executable}; rerun with --rebuild"
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


@dataclass(frozen=True, slots=True)
class LaneBatchResult:
    """One bounded lane batch, returned as soon as that lane is free."""

    lane: str
    candidates: tuple[Candidate, ...]
    results: tuple[CandidateResult, ...]
    elapsed: float
    error: str | None = None


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
        lane_callback: Callable[[str, int, float], None] | None = None,
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
        self.lanes = tuple(normalized)
        self.session_id = session_id
        self.batch_size = batch_size
        self._engine_options = {
            "session_id": session_id,
            "open_session": open_session,
            "metric_projection": metric_projection,
            "metric_fields": metric_fields,
            "observation": observation,
        }
        self._engines = [self._new_engine(index) for index in range(len(self.lanes))]
        self._started = False
        self._closed = False
        self._next_ordinal = start_ordinal
        self._next_lane = 0
        self._evaluation_lock = Lock()
        self._lane_callback = lane_callback

    @property
    def engines(self) -> tuple[OptimizerEngine, ...]:
        """Read-only access to lane engines for diagnostics."""
        return tuple(self._engines)

    def _new_engine(self, lane_index: int) -> OptimizerEngine:
        return OptimizerEngine(
            self.lanes[lane_index].client_factory,
            **self._engine_options,
        )

    def _recycle_engine(self, lane_index: int) -> None:
        engine = self._engines[lane_index]
        try:
            if engine.client is None:
                engine.close()
            else:
                engine.client.shutdown()
        except Exception:
            pass
        self._engines[lane_index] = self._new_engine(lane_index)

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

    def iter_grid(
        self,
        assignments: Sequence[Iterable[tuple[int, Candidate]]],
    ) -> Iterator[LaneBatchResult]:
        """Run fixed lane stripes without barriers, recycling failed lanes."""
        if len(assignments) != len(self.lanes):
            raise ValueError("grid assignments must match the fleet lane count")
        with self._evaluation_lock:
            yield from self._iter_grid(assignments)

    def _iter_grid(
        self,
        assignments: Sequence[Iterable[tuple[int, Candidate]]],
    ) -> Iterator[LaneBatchResult]:
        if self._closed:
            raise RuntimeError("fleet is closed")
        iterators = tuple(iter(items) for items in assignments)
        consecutive_failures = [0] * len(self.lanes)
        quarantined = [False] * len(self.lanes)

        def next_batch(lane_index: int) -> tuple[tuple[int, Candidate], ...]:
            items = tuple(islice(iterators[lane_index], self.batch_size))
            for ordinal, candidate in items:
                if (
                    isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or ordinal < 0
                    or not isinstance(candidate, Candidate)
                ):
                    raise TypeError(
                        "grid assignments must contain non-negative ordinal/Candidate pairs"
                    )
            return items

        def run_lane(
            lane_index: int,
            items: tuple[tuple[int, Candidate], ...],
        ) -> LaneBatchResult:
            ordinals, candidates = zip(*items, strict=True)
            started_at = time.monotonic()
            error: str | None = None
            try:
                evaluated = self._engines[lane_index].evaluate(candidates)
                if len(evaluated) != len(candidates):
                    raise ValueError("lane returned the wrong number of results")
                results = tuple(
                    replace(result, ordinal=ordinal)
                    for ordinal, result in zip(ordinals, evaluated, strict=True)
                )
            except Exception as exc:
                detail = " | ".join(str(exc).splitlines()).strip()
                error = f"{type(exc).__name__}: {detail}"[:500]
                results = tuple(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status="failed",
                        error=error if index == 0 else None,
                        ordinal=ordinal,
                    )
                    for index, (ordinal, candidate) in enumerate(items)
                )
                self._recycle_engine(lane_index)
            return LaneBatchResult(
                lane=self.lanes[lane_index].name,
                candidates=tuple(candidates),
                results=results,
                elapsed=time.monotonic() - started_at,
                error=error,
            )

        def failed_lane(
            lane_index: int,
            items: tuple[tuple[int, Candidate], ...],
        ) -> LaneBatchResult:
            error = (
                f"lane quarantined after {_MAX_CONSECUTIVE_GRID_FAILURES} "
                "consecutive chunk failures"
            )
            return LaneBatchResult(
                lane=self.lanes[lane_index].name,
                candidates=tuple(candidate for _ordinal, candidate in items),
                results=tuple(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status="failed",
                        error=error if index == 0 else None,
                        ordinal=ordinal,
                    )
                    for index, (ordinal, candidate) in enumerate(items)
                ),
                elapsed=0.0,
                error=error,
            )

        def submit(executor, lane_index, items):
            task = failed_lane if quarantined[lane_index] else run_lane
            return executor.submit(task, lane_index, items)

        executor = ThreadPoolExecutor(max_workers=len(self.lanes))
        futures = {}
        try:
            for lane_index in range(len(self.lanes)):
                if items := next_batch(lane_index):
                    futures[submit(executor, lane_index, items)] = lane_index
            while futures:
                future = next(as_completed(futures))
                lane_index = futures.pop(future)
                completed = future.result()
                if completed.error is None:
                    consecutive_failures[lane_index] = 0
                elif not quarantined[lane_index]:
                    consecutive_failures[lane_index] += 1
                    quarantined[lane_index] = (
                        consecutive_failures[lane_index]
                        >= _MAX_CONSECUTIVE_GRID_FAILURES
                    )
                if self._lane_callback is not None:
                    self._lane_callback(
                        completed.lane,
                        len(completed.results),
                        completed.elapsed,
                    )
                yield completed
                if items := next_batch(lane_index):
                    futures[submit(executor, lane_index, items)] = lane_index
        finally:
            executor.shutdown(wait=True)

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
                ) -> tuple[list[tuple[int, CandidateResult]], float]:
                    offsets, batch = zip(*assignment, strict=True)
                    engine = self._engines[lane_index]
                    started_at = time.monotonic()
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
                    return (
                        [
                            (offset, replace(result, ordinal=base_ordinal + total + offset))
                            for offset, result in zip(offsets, results, strict=True)
                        ],
                        time.monotonic() - started_at,
                    )

                ordered: list[CandidateResult | None] = [None] * len(wave)
                futures = {
                    executor.submit(run_lane, lane, assignment): lane
                    for lane, assignment in assignments.items()
                }
                for future in as_completed(futures):
                    lane_index = futures[future]
                    lane_results, elapsed = future.result()
                    if self._lane_callback is not None:
                        self._lane_callback(
                            self.lanes[lane_index].name,
                            len(lane_results),
                            elapsed,
                        )
                    for offset, result in lane_results:
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
    "LaneBatchResult",
    "PlacementLane",
    "REMOTE_BASE",
    "RSYNC_SSH",
    "SSH_OPTIONS",
    "ensure_remote_file",
    "local_client_factory",
    "ssh_client_factory",
]
