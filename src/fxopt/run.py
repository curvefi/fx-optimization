"""Execute a bounded local-or-SSH fxopt grid through one persistent harness session."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time

from typing import Any

import numpy as np

from .config import (
    ConfigError, RunConfig, EVALUATOR_POLICY_METADATA_KEY,
    _COMPILED_POLICY_ABI, _execution_inputs,
)
from .engine import ClientFactory
from .placement import (
    EvaluatorFleet,
    PlacementLane,
    REMOTE_BASE,
    RSYNC_SSH,
    SSH_OPTIONS,
    ensure_remote_file,
    local_client_factory,
    rebuild_shared_evaluator,
    require_reachable_hosts,
    require_shared_evaluator,
    transfer_workspace,
)
from .results import ArtifactPaths, GridResultWriter, merge_grid_partitions
from .robustness import robustness_metadata


ProgressCallback = Callable[[int, int], None]
WorkerProgressCallback = Callable[[int, int, float], None]
WorkerReadyCallback = Callable[[int | None, int, int, float], None]
_BLOCK_SHUFFLE_SEED = 0
_SCHEDULE_BLOCK_ROWS = 8

_AXIS_LABELS = {
    "pool.A": "A",
    "pool.donation_apy": "donation",
    "pool.reserved_profit_fraction": "rpf",
}


def _evaluator_policy_contract(config: RunConfig) -> dict[str, str | int]:
    if config.compiled_policy_header is None:
        return {
            "policy_id": "none",
            "policy_abi": "none",
            "policy_parameter_count": 0,
        }
    policy_id = config.compiled_policy_id
    assert policy_id is not None
    return {
        "policy_id": policy_id,
        "policy_abi": _COMPILED_POLICY_ABI,
        "policy_parameter_count": len(config.candidate.defaults["policy_params"]),
    }


def _display_value(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _evaluator_batch_size(config: RunConfig, total: int) -> int:
    lanes = max(1, len(config.hosts)) * max(1, len(config.numa_nodes))
    return min(config.batch_size, max(1, (total + lanes - 1) // lanes))


def _schedule_block_size(config: RunConfig, total: int) -> int:
    return min(_SCHEDULE_BLOCK_ROWS, _evaluator_batch_size(config, total))


@dataclass(frozen=True)
class GridLeases(Sequence):
    blocks: np.ndarray
    total: int
    block_size: int
    blocks_per_lease: int

    def __len__(self) -> int:
        return (len(self.blocks) + self.blocks_per_lease - 1) // self.blocks_per_lease

    def __getitem__(self, index: int) -> tuple[tuple[int, int], ...]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin = index * self.blocks_per_lease
        return tuple(
            (int(block) * self.block_size, min(self.total, (int(block) + 1) * self.block_size))
            for block in self.blocks[begin:begin + self.blocks_per_lease]
        )

    @property
    def max_rows(self) -> int:
        return min(self.total, self.blocks_per_lease * self.block_size)


def _shuffled_block_leases(
    total: int, block_size: int, batch_size: int, slots: int,
    *, seed: int = _BLOCK_SHUFFLE_SEED,
) -> GridLeases:
    if total < 1 or block_size < 1 or batch_size < 1 or slots < 1:
        raise ValueError("total, block size, batch size, and slots must be positive")
    count = (total + block_size - 1) // block_size
    blocks = np.arange(count, dtype=np.uint32 if count <= 2**32 else np.uint64)
    random.Random(seed).shuffle(blocks)
    return GridLeases(blocks, total, block_size, max(1, batch_size // block_size) * slots)


def _range_count(ranges: Iterable[tuple[int, int]]) -> int:
    return sum(stop - start for start, stop in ranges)


def grid_summary(config_path: str | Path | RunConfig) -> str:
    """Describe the configured Cartesian axes in their operator-facing units."""
    config = (
        config_path
        if isinstance(config_path, RunConfig)
        else RunConfig.from_toml(config_path)
    )
    total = len(config.candidate.grid())
    parts = []
    for name, values in config.candidate.axes.items():
        endpoints = [values[0], values[-1]]
        if isinstance(endpoints[0], Mapping):
            endpoints = [next(iter(value.values())) for value in endpoints]
        if name == "pool.donation_apy":
            endpoints = [float(value) * 100 for value in endpoints]
        span = _display_value(endpoints[0])
        if len(endpoints) > 1:
            span += f"..{_display_value(endpoints[-1])}"
        if name == "pool.donation_apy":
            span += "%"
        parts.append(f"{_AXIS_LABELS.get(name, name)} {span} ({len(values)} pts)")
    placement = ""
    if config.hosts:
        batch_size = _evaluator_batch_size(config, total)
        block_size = _schedule_block_size(config, total)
        lease_rows = max(1, batch_size // block_size) * block_size * max(1, len(config.numa_nodes))
        allocation = str(min(total, lease_rows))
        placement = f" on {len(config.hosts)} workers ({allocation} pools/lease)"
    suffix = f": {', '.join(parts)}" if parts else ""
    return f"running {total} pools grid{placement}{suffix}"


def stage_remote_run(config: RunConfig) -> str:
    """Publish the compact config and its portable inputs through shared NFS."""
    if not config.hosts:
        raise ConfigError("remote staging requires placement hosts")
    local_inputs = _execution_inputs(config, remote=False)
    remote_inputs = _execution_inputs(config, remote=True)
    first = config.hosts[0]
    for name in ("template", "market", "price_feed"):
        if name in remote_inputs:
            ensure_remote_file(
                first,
                local_inputs[name],
                remote_inputs[name],
                replace=name == "template",
            )
    workspace = next(
        parent.parent
        for parent in config.path.parents
        if parent.name == "curve-fx-optimization"
    )
    remote_config = str(
        REMOTE_BASE.joinpath(*config.path.relative_to(workspace).parts)
    )
    ensure_remote_file(first, config.path, remote_config, replace=True)
    return remote_config


def _local_worker_lanes(
    config: RunConfig,
    *,
    remote_paths: bool,
) -> tuple[PlacementLane, ...]:
    """Create the evaluator slots owned by one machine-local worker."""
    inputs = _execution_inputs(config, remote=remote_paths)
    nodes: tuple[int | None, ...] = config.numa_nodes or (None,)
    if config.workers < len(nodes):
        raise ConfigError("run.workers must cover every configured NUMA node")
    workers, extra = divmod(config.workers, len(nodes))
    return tuple(
        PlacementLane(
            "local" if node is None else f"numa{node}",
            local_client_factory(
                inputs["evaluator"],
                work_dir=None if remote_paths else config.path.parent,
                workers=workers + (index < extra),
                client_options={
                    f"expected_{name}": value
                    for name, value in _evaluator_policy_contract(config).items()
                },
                launch_prefix=(
                    (("env", "-u", "LD_LIBRARY_PATH") if remote_paths else ())
                    + (() if node is None else (
                        "numactl",
                        f"--cpunodebind={node}",
                        f"--membind={node}",
                    ))
                ),
                timeout=600.0,
            ),
        )
        for index, node in enumerate(nodes)
    )


def open_session_request(config: RunConfig, *, remote: bool | None = None) -> dict[str, Any]:
    inputs = _execution_inputs(config, remote=bool(config.hosts) if remote is None else remote)
    request = {
        "template_path": inputs["template"],
        "scenario_id": config.scenario["id"],
        "market_path": inputs["market"],
        **config.session,
    }
    if (price_feed := inputs.get("price_feed")) is not None:
        request["price_feed_path"] = price_feed
    if (yb_mode := config.scenario.get("yb_mode")) is not None:
        request.setdefault("yb_mode", yb_mode)
    return request


def run_metadata(
    config: RunConfig,
    *,
    effective_batch: int,
    origin_workspace: Path | None = None,
    origin_config: Path | None = None,
) -> dict[str, Any]:
    grid = config.candidate.grid()
    inputs = _execution_inputs(config, remote=bool(config.hosts))
    local_inputs = _execution_inputs(config, remote=False)
    replay_session = open_session_request(config, remote=False)

    def replay_path(value: str) -> str:
        path = Path(value).resolve()
        if origin_workspace is None:
            return str(path)
        try:
            relative = path.relative_to(REMOTE_BASE)
        except ValueError as exc:
            raise ConfigError(f"coordinator replay path must be below {REMOTE_BASE}") from exc
        return str(origin_workspace.joinpath(*relative.parts))

    for key in ("template_path", "market_path", "price_feed_path"):
        if key in replay_session:
            replay_session[key] = replay_path(replay_session[key])
    config_path = origin_config or config.path
    config_parent = config_path.parent
    origin = "external"
    for parent in config_path.parents:
        if (
            parent.parent.name == "configs"
            and parent.name in {"autoresearch", "experiments"}
        ):
            origin = parent.name
            break
    metadata = {
        "config": str(config_path),
        "config_origin": origin,
        "evaluator": inputs["evaluator"],
        "template": inputs["template"],
        "market": inputs["market"],
        "placement": "ssh" if config.hosts else "local",
        "hosts": list(config.hosts),
        "numa_nodes": list(config.numa_nodes),
        "batch_size": config.batch_size,
        "effective_batch_size": effective_batch,
        "workers": config.workers,
        "metric_fields": list(config.metric_fields),
        "axes": {name: list(grid.axes[name]) for name in sorted(grid.axes)},
        "shape": list(grid.shape),
        "candidate_defaults": config.candidate.defaults,
        EVALUATOR_POLICY_METADATA_KEY: _evaluator_policy_contract(config),
        "open_session": open_session_request(config),
        "replay": {
            "evaluator": replay_path(local_inputs["evaluator"]),
            "work_dir": str(config_parent),
            "open_session": replay_session,
        },
    }
    if config.robustness:
        metadata["robustness"] = robustness_metadata(config.robustness)
    if config.compiled_policy_header is not None:
        metadata["compiled_policy"] = {
            "id": config.compiled_policy_id,
            "header": inputs["policy_header"],
        }
    return metadata


def prepare_remote(config: RunConfig, *, transfer: bool, rebuild: bool) -> None:
    """Prepare one shared cluster workspace before evaluator lanes launch."""
    if not config.hosts:
        if transfer or rebuild:
            raise ConfigError("--transfer and --rebuild require remote placement hosts")
        return
    print(f"cluster: checking {len(config.hosts)} hosts...", file=sys.stderr)
    hosts = require_reachable_hosts(config.hosts)
    first = hosts[0]
    inputs = _execution_inputs(config, remote=True)
    try:
        optimizer_root = next(
            parent for parent in config.path.parents if parent.name == "curve-fx-optimization"
        )
    except StopIteration as exc:
        raise ConfigError("remote config must be inside curve-fx-optimization") from exc
    if rebuild:
        transfer = True
    if transfer:
        print("cluster: transferring sources...", file=sys.stderr)
        transfer_workspace(first, optimizer_root.parent)
    if rebuild:
        print(
            f"cluster: building {PurePosixPath(inputs['evaluator']).name} on {first}...",
            file=sys.stderr,
        )
        rebuild_shared_evaluator(
            first,
            inputs["evaluator"],
            policy_header=inputs.get("policy_header"),
            policy_id=config.compiled_policy_id,
        )
    require_shared_evaluator(first, inputs["evaluator"])
    print("cluster: ready", file=sys.stderr)


def _grid_request(config: RunConfig) -> tuple[Any, dict[str, Any]]:
    grid = config.candidate.grid()
    return grid, {
        "candidate_defaults": config.candidate.defaults,
        "axes": {
            name: list(grid.axes[name])
            for name in sorted(grid.axes)
        },
        "axis_order": list(sorted(grid.axes)),
        "shape": list(grid.shape),
    }


def _run(
    config: RunConfig,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None,
    progress_callback: ProgressCallback | None,
) -> ArtifactPaths:
    if config.hosts:
        raise ConfigError("remote grids require the detached cluster runner")
    grid, compact_grid = _grid_request(config)
    total = len(grid)
    batch_size = min(config.batch_size, total)
    metadata = run_metadata(config, effective_batch=batch_size)
    metadata["execution_order"] = {
        "kind": "contiguous_ranges_v1",
        "block_size": batch_size,
    }
    factory = client_factory or local_client_factory(
        config.evaluator,
        work_dir=config.path.parent,
        workers=config.workers,
        client_options={
            f"expected_{name}": value
            for name, value in _evaluator_policy_contract(config).items()
        },
    )
    if client_factory is not None:
        metadata["placement"] = "injected"
    writer = GridResultWriter(
        output_dir,
        run_id=config.run_id,
        total=total,
        metadata=metadata,
        metric_names=config.metric_fields,
    )
    fleet = EvaluatorFleet(
        (PlacementLane("injected" if client_factory else "local", factory),),
        session_id=config.run_id,
        open_session=metadata["open_session"],
        metric_fields=tuple(sorted(config.metric_fields)),
        grid=compact_grid,
    )
    with writer:
        try:
            fleet.start()
            completed = 0
            if progress_callback is not None:
                progress_callback(0, total)
            blocks = (
                (start, min(total, start + batch_size))
                for start in range(0, total, batch_size)
            )
            for batch in fleet.iter_grid_ranges(blocks):
                writer.append_projected(batch.ordinals, batch.projected)
                completed += batch.count
                if progress_callback is not None:
                    progress_callback(completed, total)
        finally:
            fleet.close()
        return writer.finalize()


def _run_leased_worker(
    config: RunConfig,
    output_dir: Path,
    *,
    worker_index: int,
    commands: Iterable[Mapping[str, Any]],
    progress_callback: WorkerProgressCallback | None = None,
    ready_callback: WorkerReadyCallback | None = None,
) -> tuple[ArtifactPaths, float, tuple[int, ...], str | None]:
    grid, compact_grid = _grid_request(config)
    total = len(grid)
    batch_size = _evaluator_batch_size(config, total)
    block_size = _schedule_block_size(config, total)
    slots = max(1, len(config.numa_nodes))
    leases = _shuffled_block_leases(total, block_size, batch_size, slots)
    metric_names = tuple(sorted(config.metric_fields))
    remote_paths = bool(config.hosts)
    lanes = _local_worker_lanes(config, remote_paths=remote_paths)
    writer = GridResultWriter(
        output_dir,
        run_id=config.run_id,
        total=total,
        metadata={
            "worker_index": worker_index,
            "block_size": block_size,
            "batch_size": batch_size,
            "seed": _BLOCK_SHUFFLE_SEED,
        },
        metric_names=metric_names,
    )
    with writer:
        fleet = EvaluatorFleet(
            lanes,
            session_id=config.run_id,
            open_session=open_session_request(config, remote=remote_paths),
            metric_fields=metric_names,
            grid=compact_grid,
        )
        error = None
        calculation_started = time.monotonic()
        lease_ids: list[int] = []
        try:
            fleet.start()
            completed = 0
            if progress_callback is not None:
                progress_callback(0, total, 0.0)
            if ready_callback is not None:
                ready_callback(None, 0, total, 0.0)
            finished = False
            for command in commands:
                if not isinstance(command, Mapping):
                    raise ValueError("worker command must be a mapping")
                command_type = command.get("type")
                if command_type == "finish":
                    finished = True
                    break
                if command_type != "lease":
                    raise ValueError("worker command must be lease or finish")
                lease_id = command.get("lease_id")
                if (
                    isinstance(lease_id, bool)
                    or not isinstance(lease_id, int)
                    or lease_id < 0
                    or lease_id >= len(leases)
                    or lease_id in lease_ids
                ):
                    raise ValueError("worker received an invalid lease ID")
                lease_ids.append(lease_id)
                for batch in fleet.iter_grid_ranges(
                    leases[lease_id],
                    blocks_per_batch=max(1, batch_size // block_size),
                ):
                    writer.append_projected(batch.ordinals, batch.projected)
                    completed += batch.count
                    if progress_callback is not None:
                        progress_callback(
                            completed,
                            total,
                            time.monotonic() - calculation_started,
                        )
                if ready_callback is not None:
                    ready_callback(
                        lease_id,
                        completed,
                        total,
                        time.monotonic() - calculation_started,
                    )
            if not finished:
                raise RuntimeError("coordinator closed before finishing worker")
        except Exception as exc:
            if writer.row_count == 0:
                raise
            error = str(exc)
        finally:
            fleet.close()
        calculation_s = time.monotonic() - calculation_started
        return writer.finalize_partition(), calculation_s, tuple(lease_ids), error


def _copy_partition(source: ArtifactPaths, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"worker partition already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        for source_path, name in (
            (source.results_npz, "results.npz"),
            (source.run_json, "run.json"),
        ):
            target = destination / name
            temporary = destination / f".{name}.tmp"
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def run_leased_worker(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    worker_index: int,
    commands: Iterable[Mapping[str, Any]],
    progress_callback: WorkerProgressCallback | None = None,
    ready_callback: WorkerReadyCallback | None = None,
) -> dict[str, Any]:
    """Evaluate coordinator leases and publish one machine-local partition."""
    config = RunConfig.from_toml(config_path)
    if worker_index < 0:
        raise ConfigError("worker index must be non-negative")
    destination = Path(output_dir).resolve()
    started = time.monotonic()
    paths, calculation_s, lease_ids, error = _run_leased_worker(
        config,
        destination,
        worker_index=worker_index,
        commands=commands,
        progress_callback=progress_callback,
        ready_callback=ready_callback,
    )
    receipt = json.loads(paths.run_json.read_text())
    return {
        "type": "complete",
        "worker_index": worker_index,
        "count": receipt["partition_count"],
        "status_counts": receipt["status_counts"],
        "elapsed_s": time.monotonic() - started,
        "calculation_s": calculation_s,
        "output": str(destination),
        "lease_ids": list(lease_ids),
        "error": error,
    }


class _ClusterProgress:
    """Aggregate low-rate worker heartbeats into one stable cluster ETA."""

    def __init__(self, total: int, workers: int, interval: float = 2.0) -> None:
        self.total = total
        self.workers = workers
        self.interval = interval
        self._started = time.monotonic()
        self._states: dict[int, tuple[int, int, float, bool]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def update(self, worker: int, message: Mapping[str, Any]) -> None:
        with self._lock:
            self._states[worker] = (
                int(message["completed"]),
                int(message["total"]),
                float(message["calculation_s"]),
                message.get("type") == "complete",
            )

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def write(self) -> None:
        with self._lock:
            states = tuple(self._states.values())
        if not states:
            return
        completed = sum(state[0] for state in states)
        producing_workers = sum(state[0] > 0 for state in states)
        complete_workers = sum(state[3] for state in states)
        percent = min(100, int(completed * 100 / self.total))
        elapsed = time.monotonic() - self._started
        if producing_workers < self.workers and completed < self.total:
            print(
                f"run: {completed}/{self.total} ({percent}%) warming "
                f"({producing_workers}/{self.workers} workers producing; "
                f"{elapsed:.1f}s elapsed)",
                file=sys.stderr,
                flush=True,
            )
            return
        rates = [
            done / elapsed
            for done, _total, elapsed, _complete in states
            if done > 0 and elapsed > 0.0 and (
                completed >= self.total or not _complete
            )
        ]
        rate = sum(rates)
        eta = (self.total - completed) / rate if rate > 0.0 else None
        eta_text = "--" if eta is None else f"{eta:.1f}s"
        print(
            f"run: {completed}/{self.total} ({percent}%) "
            f"{rate:.1f} pools/s "
            f"({elapsed:.1f}s elapsed, {eta_text} ETA; "
            f"{complete_workers}/{self.workers} workers complete)",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self.write()


def run_distributed_config(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    origin_workspace: str | Path | None = None,
    origin_config: str | Path | None = None,
) -> ArtifactPaths:
    """Launch one portable machine worker per placement and merge its partition."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts or not config.metric_fields:
        raise ConfigError("distributed grids require hosts and metric fields")
    total = len(config.candidate.grid())
    batch_size = _evaluator_batch_size(config, total)
    block_size = _schedule_block_size(config, total)
    leases = _shuffled_block_leases(
        total,
        block_size,
        batch_size,
        max(1, len(config.numa_nodes)),
    )
    active = list(enumerate(config.hosts[:min(len(config.hosts), len(leases))]))
    destination = Path(output_dir).resolve()
    partition_root = destination / ".partitions"
    if partition_root.exists():
        raise FileExistsError(f"partition staging already exists: {partition_root}")
    partition_root.mkdir(parents=True)

    remote_project = str(REMOTE_BASE / "curve-fx-optimization")
    remote_shell = f"{remote_project}/shell.nix"
    remote_python = f"{remote_project}/scripts/cluster-python"
    python_path = (
        f"{remote_project}/src:"
        f"{REMOTE_BASE}/curve-fx-arb-harness/python/src:"
        f"{remote_project}/.venv-cluster/lib/python3.12/site-packages"
    )
    coordinator = config.hosts[0]
    progress = _ClusterProgress(total, len(active))
    lease_lock = threading.Lock()
    reserved_leases = {index: index for index, _host in active}
    next_pending_lease = len(active)

    def acquire_lease(worker_index: int) -> tuple[int, tuple[tuple[int, int], ...]] | None:
        nonlocal next_pending_lease
        with lease_lock:
            lease_id = reserved_leases.pop(worker_index, None)
            if lease_id is None:
                if next_pending_lease >= len(leases):
                    return None
                lease_id = next_pending_lease
                next_pending_lease += 1
        return lease_id, leases[lease_id]

    def run_worker(index: int, host: str) -> tuple[str, dict[str, Any]]:
        partition = f"{destination}.worker-{index:03d}"
        worker = [
            remote_python, "-m", "fxopt.cli", "_worker",
            str(config.path), "--output", str(partition),
            "--worker-index", str(index),
        ]
        remote_script = " ".join((
            "set -euo pipefail;",
            f"export PYTHONPATH={shlex.quote(python_path)};",
            "export PYTHONNOUSERSITE=1;",
            "exec " + shlex.join(worker),
        ))
        remote_command = (
            f"nix-shell {shlex.quote(remote_shell)} "
            f"--run {shlex.quote(remote_script)}"
        )
        command = (
            ["/bin/sh", "-lc", remote_script]
            if host == coordinator
            else ["ssh", *SSH_OPTIONS, "--", host, remote_command]
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        receipt: dict[str, Any] | None = None
        tail: list[str] = []
        assigned_ids: list[int] = []
        assigned_count = 0
        awaiting_lease: int | None = None
        sent_finish = False
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                tail = [*tail[-4:], line]
                continue
            if not isinstance(message, Mapping) or message.get("worker_index") != index:
                tail = [*tail[-4:], line]
                continue
            if message.get("type") == "progress":
                progress.update(index, message)
            elif message.get("type") == "ready":
                progress.update(index, message)
                completed_lease = message.get("completed_lease_id")
                if awaiting_lease is None:
                    if completed_lease is not None:
                        raise RuntimeError(
                            f"{host} worker completed an unassigned lease"
                        )
                elif completed_lease != awaiting_lease:
                    raise RuntimeError(
                        f"{host} worker completed the wrong lease"
                    )
                awaiting_lease = None
                acquired = acquire_lease(index)
                if acquired is None:
                    command_message = {"type": "finish"}
                    sent_finish = True
                else:
                    lease_id, ranges = acquired
                    assigned_ids.append(lease_id)
                    assigned_count += _range_count(ranges)
                    awaiting_lease = lease_id
                    command_message = {"type": "lease", "lease_id": lease_id}
                process.stdin.write(json.dumps(
                    command_message,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n")
                process.stdin.flush()
            elif message.get("type") == "complete":
                receipt = dict(message)
        return_code = process.wait()
        if return_code != 0:
            detail = " | ".join(tail[-5:])
            raise RuntimeError(f"{host} worker failed: {detail}")
        if receipt is None:
            raise RuntimeError(f"{host} worker returned an invalid receipt")
        if (
            (not receipt.get("error") and (not sent_finish or awaiting_lease is not None))
            or not 0 < receipt.get("count", 0) <= assigned_count
            or receipt.get("output") != partition
            or receipt.get("lease_ids") != assigned_ids
        ):
            raise RuntimeError(f"{host} worker receipt does not match its assignment")
        progress.update(index, {
            "type": "complete",
            "completed": receipt["count"],
            "total": total,
            "calculation_s": receipt["calculation_s"],
        })
        return host, receipt

    def fetch_partition(index: int, host: str, receipt: Mapping[str, Any]) -> Path:
        source = Path(str(receipt["output"]))
        if (
            not str(source).startswith("/tmp/fxopt-grid.")
            or source.name != f"{destination.name}.worker-{index:03d}"
        ):
            raise RuntimeError(f"{host} returned an invalid partition path")
        target = partition_root / f"worker-{index:03d}"
        if host == coordinator:
            _copy_partition(
                ArtifactPaths(source / "run.json", source / "results.npz"),
                target,
            )
            shutil.rmtree(source)
            return target
        target.mkdir()
        fetched = subprocess.run(
            [
                "rsync", "-a", "--partial",
                "--include=/run.json", "--include=/results.npz", "--exclude=*",
                "-e", RSYNC_SSH, "--", f"{host}:{source}/", f"{target}/",
            ],
            check=False,
        )
        if fetched.returncode != 0:
            raise subprocess.CalledProcessError(fetched.returncode, fetched.args)
        if not (target / "run.json").is_file() or not (target / "results.npz").is_file():
            raise RuntimeError(f"{host} partition fetch is incomplete")
        cleanup = subprocess.run(
            ["ssh", *SSH_OPTIONS, "--", host, "rm", "-rf", "--", str(source)],
            check=False,
        )
        if cleanup.returncode != 0:
            print(f"cluster: warning: retained {host}:{source}", file=sys.stderr)
        return target

    started = time.monotonic()
    partitions: dict[int, Path] = {}
    receipts: dict[int, tuple[str, dict[str, Any]]] = {}
    progress_closed = False
    try:
        print(
            f"coordinator: queued {len(leases)} shuffled leases "
            f"({leases.max_rows} pools max) "
            f"for {len(active)} workers...",
            file=sys.stderr,
        )
        errors: list[str] = []
        fetches: dict[Any, int] = {}
        with ThreadPoolExecutor(max_workers=len(active)) as executor, \
                ThreadPoolExecutor(max_workers=min(4, len(active))) as fetcher:
            futures = {
                executor.submit(run_worker, index, host): (index, host)
                for index, host in active
            }
            for future in as_completed(futures):
                index, host = futures[future]
                try:
                    _host, receipt = future.result()
                except Exception as exc:
                    errors.append(f"{host}: {exc}")
                    continue
                receipts[index] = (host, receipt)
                if receipt.get("error"):
                    errors.append(f"{host}: {receipt['error']}")
                fetches[fetcher.submit(fetch_partition, index, host, receipt)] = index
            for future in as_completed(fetches):
                index = fetches[future]
                try:
                    partitions[index] = future.result()
                except Exception as exc:
                    errors.append(f"worker {index} fetch: {exc}")
        for error in errors:
            print(f"cluster: incomplete worker: {error}", file=sys.stderr)
        if not partitions:
            raise RuntimeError("cluster produced no usable results: " + " | ".join(errors))
        progress.close()
        progress_closed = True

        metadata = run_metadata(
            config,
            effective_batch=batch_size,
            origin_workspace=(
                None if origin_workspace is None else Path(origin_workspace).resolve()
            ),
            origin_config=(
                None if origin_config is None else Path(origin_config).resolve()
            ),
        )
        metadata["placement"] = "machine_workers"
        metadata["execution_order"] = {
            "kind": "dynamic_shuffled_leases_v1",
            "block_size": block_size,
            "batch_size": batch_size,
            "lease_size": leases.max_rows,
            "lease_count": len(leases),
            "seed": _BLOCK_SHUFFLE_SEED,
            "worker_count": len(active),
        }
        metadata["worker_stats"] = [
            {
                "worker_index": index,
                "host": host,
                "count": receipt["count"],
                "elapsed_s": receipt["elapsed_s"],
                "calculation_s": receipt["calculation_s"],
                "lease_ids": receipt["lease_ids"],
                "status_counts": receipt["status_counts"],
            }
            for index, (host, receipt) in sorted(receipts.items())
        ]
        metadata["transport"] = "heartbeat_and_partition"
        if errors:
            metadata["worker_errors"] = errors
        calculation_times = [
            float(receipt["calculation_s"])
            for _host, receipt in receipts.values()
        ]
        print(
            f"coordinator: worker calculation range "
            f"{min(calculation_times):.1f}-{max(calculation_times):.1f}s",
            file=sys.stderr,
        )
        print("coordinator: merging worker partitions...", file=sys.stderr)
        paths = merge_grid_partitions(
            output_dir,
            [partitions[index] for index in sorted(partitions)],
            run_id=config.run_id,
            total=total,
            metadata=metadata,
            metric_names=config.metric_fields,
        )
        elapsed = time.monotonic() - started
        calculated = sum(receipts[index][1]["count"] for index in partitions)
        print(
            f"coordinator: saved {calculated}/{total} pools in {elapsed:.1f}s "
            f"({calculated / elapsed:.1f} pools/s wall)",
            file=sys.stderr,
        )
        return paths
    finally:
        if not progress_closed:
            progress.close()
        shutil.rmtree(partition_root, ignore_errors=True)


def run_config(
    config_path: str | Path | RunConfig,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ArtifactPaths:
    """Run one local grid and atomically publish its two artifacts."""
    config = (
        config_path
        if isinstance(config_path, RunConfig)
        else RunConfig.from_toml(config_path)
    )
    return _run(
        config,
        Path(output_dir).expanduser().resolve(),
        client_factory=client_factory,
        progress_callback=progress_callback,
    )


__all__ = [
    "grid_summary", "open_session_request", "run_config", "run_metadata",
    "run_distributed_config", "run_leased_worker", "stage_remote_run",
]
