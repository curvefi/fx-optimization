"""Process placement and shared range queues for Cartesian-grid workers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from os import PathLike, fspath
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from threading import Lock
from typing import Any

from curve_fx_harness_client import EvaluatorClient

from .engine import ClientFactory, EvaluatorSession, ProjectedBatch


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
    launch_prefix: Sequence[str | PathLike[str]] = (),
    client_options: Mapping[str, Any] | None = None,
    **options: Any,
) -> ClientFactory:
    """Return a factory for a local evaluator in persistent ``serve`` mode."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    executable = _token(executable_path, "executable_path")
    directory = None if work_dir is None else _token(work_dir, "work_dir")
    prefix = [
        _argv_token(value, f"launch_prefix[{index}]")
        for index, value in enumerate(launch_prefix)
    ]
    fixed = {
        "executable_path": executable,
        "work_dir": directory,
        "launch_argv": [*prefix, executable, "serve", "--workers", str(workers)],
        "verify_local_inputs": True,
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


def rebuild_shared_evaluator(
    host: str,
    evaluator: str,
    *,
    policy_header: str | None = None,
    policy_id: str | None = None,
) -> None:
    """Build the configured evaluator once in the shared cluster workspace."""
    remote_host = _token(host, "host")
    executable = PurePosixPath(_token(evaluator, "evaluator"))
    if (policy_header is None) != (policy_id is None):
        raise ValueError("compiled policy header and id must be configured together")
    policy_options = ""
    if policy_header is not None and policy_id is not None:
        policy_options = " " + " ".join((
            f"-DPOLICY_HEADER_PATH={shlex.quote(_token(policy_header, 'policy_header'))}",
            f"-DPOLICY_ID={shlex.quote(_token(policy_id, 'policy_id'))}",
        ))
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
        '-DCURVE_FX_ENABLE_IPO=ON -DCMAKE_PREFIX_PATH="$root/twocrypto-cpp/_install"'
        f'{policy_options};',
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
    """One completed registered-grid range batch."""

    ordinals: tuple[int, ...]
    projected: ProjectedBatch

    @property
    def count(self) -> int:
        return len(self.ordinals)


class EvaluatorFleet:
    """Feed one shared registered-grid range queue to persistent local slots."""

    def __init__(
        self,
        lanes: Sequence[PlacementLane | ClientFactory],
        *,
        session_id: str,
        open_session: Mapping[str, Any] | None = None,
        metric_fields: Sequence[str],
        grid: Mapping[str, Any],
    ) -> None:
        if not lanes:
            raise ValueError("fleet requires at least one lane")
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
        self._engine_options = {
            "session_id": session_id,
            "open_session": open_session,
            "metric_fields": metric_fields,
            "grid": grid,
        }
        self._engines = [self._new_engine(index) for index in range(len(self.lanes))]
        self._started = False
        self._closed = False
        self._evaluation_lock = Lock()

    def _new_engine(self, lane_index: int) -> EvaluatorSession:
        return EvaluatorSession(
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

    def iter_grid_ranges(
        self,
        blocks: Iterable[tuple[int, int]],
        *,
        blocks_per_batch: int = 1,
    ) -> Iterator[LaneBatchResult]:
        """Let every projected evaluator lane pull from one shared block queue."""
        if (
            isinstance(blocks_per_batch, bool)
            or not isinstance(blocks_per_batch, int)
            or blocks_per_batch < 1
        ):
            raise ValueError("blocks_per_batch must be a positive integer")
        with self._evaluation_lock:
            iterator = iter(blocks)

            def next_ranges() -> tuple[tuple[int, int], ...]:
                selected = tuple(islice(iterator, blocks_per_batch))
                if not selected:
                    return ()
                if any(
                    isinstance(start, bool)
                    or isinstance(stop, bool)
                    or not isinstance(start, int)
                    or not isinstance(stop, int)
                    or start < 0
                    or stop <= start
                    for start, stop in selected
                ):
                    raise ValueError("grid blocks must be non-empty ordinal ranges")
                ordered = sorted(selected)
                if any(start < previous_stop for (_previous_start, previous_stop), (start, _stop)
                       in zip(ordered, ordered[1:], strict=False)):
                    raise ValueError("grid blocks must not overlap")
                merged: list[tuple[int, int]] = []
                for start, stop in ordered:
                    if merged and start == merged[-1][0] + merged[-1][1]:
                        merged[-1] = (merged[-1][0], merged[-1][1] + stop - start)
                    else:
                        merged.append((start, stop - start))
                return tuple(merged)

            def run_lane(
                lane_index: int,
                ranges: tuple[tuple[int, int], ...],
            ) -> LaneBatchResult:
                ordinals = tuple(
                    ordinal
                    for start, count in ranges
                    for ordinal in range(start, start + count)
                )
                last_error: Exception | None = None
                for attempt in range(_MAX_CONSECUTIVE_GRID_FAILURES):
                    try:
                        engine = self._engines[lane_index]
                        engine.start()
                        projected = engine.evaluate_projected_ranges(ranges)
                        if len(projected.rows) != len(ordinals):
                            raise ValueError("lane returned the wrong number of results")
                        return LaneBatchResult(
                            ordinals=ordinals,
                            projected=projected,
                        )
                    except Exception as exc:
                        last_error = exc
                        self._recycle_engine(lane_index)
                        if attempt + 1 == _MAX_CONSECUTIVE_GRID_FAILURES:
                            break
                assert last_error is not None
                raise RuntimeError(
                    f"lane {self.lanes[lane_index].name} failed grid ranges "
                    f"after {_MAX_CONSECUTIVE_GRID_FAILURES} attempts"
                ) from last_error

            executor = ThreadPoolExecutor(max_workers=len(self.lanes))
            futures: dict[Any, int] = {}
            try:
                for lane_index in range(len(self.lanes)):
                    if ranges := next_ranges():
                        futures[executor.submit(run_lane, lane_index, ranges)] = lane_index
                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    ready: list[LaneBatchResult] = []
                    for future in sorted(done, key=futures.__getitem__):
                        lane_index = futures.pop(future)
                        completed = future.result()
                        if ranges := next_ranges():
                            futures[executor.submit(run_lane, lane_index, ranges)] = lane_index
                        ready.append(completed)
                    yield from ready
            finally:
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
]
