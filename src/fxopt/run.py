"""Execute a bounded local-or-SSH fxopt grid through one persistent harness session."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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
import tomllib
from typing import Any

from .config import CandidateConfig, ConfigError
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
from .robustness import (
    RobustnessAxis,
    parse_robustness_axes,
    robustness_metadata,
)


_RUN_KEYS = frozenset({"id", "evaluator", "template", "batch_size", "workers", "metric_fields"})
_PLACEMENT_KEYS = frozenset({"hosts", "numa_nodes"})
_CANDIDATE_KEYS = frozenset({"defaults", "axes"})
_SCENARIO_KEYS = frozenset({"id", "market", "chainlink", "yb_mode"})
_COMPILED_POLICY_KEYS = frozenset({"header", "id"})
EVALUATOR_POLICY_METADATA_KEY = "expected_evaluator_policy"
_COMPILED_POLICY_ABI = "twocrypto_policy_v1"

ProgressCallback = Callable[[int, int], None]
WorkerProgressCallback = Callable[[int, int, float], None]
WorkerReadyCallback = Callable[[int | None, int, int, float], None]
REMOTE_JOB_FILENAME = ".remote-job.json"
_BLOCK_SHUFFLE_SEED = 0
_SCHEDULE_BLOCK_ROWS = 8

_AXIS_LABELS = {
    "pool.A": "A",
    "pool.donation_apy": "donation",
    "pool.reserved_profit_fraction": "rpf",
}


def _required_string(section: Mapping[str, Any], key: str, label: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}.{key} must be a non-empty string")
    return value


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Resolved settings for the ordinary local-or-SSH ``fxopt run`` command."""

    path: Path
    run_id: str
    evaluator: Path
    template: Path
    batch_size: int
    workers: int
    metric_fields: tuple[str, ...]
    hosts: tuple[str, ...]
    numa_nodes: tuple[int, ...]
    candidate: CandidateConfig
    session: Mapping[str, Any]
    scenario: Mapping[str, Any]
    robustness: tuple[RobustnessAxis, ...]
    compiled_policy_header: Path | None
    compiled_policy_id: str | None

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            with config_path.open("rb") as stream:
                raw = tomllib.load(stream)
        except OSError as exc:
            raise ConfigError(f"cannot read config {config_path}: {exc}") from exc

        run = raw.get("run")
        if not isinstance(run, Mapping):
            raise ConfigError("config requires a [run] table")
        unknown_run = set(run) - _RUN_KEYS
        if unknown_run:
            raise ConfigError(f"unknown [run] keys: {sorted(unknown_run)}")
        batch_size = run.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigError("run.batch_size must be a positive integer")
        workers = run.get("workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ConfigError("run.workers must be a positive integer")
        raw_metric_fields = run.get("metric_fields", [])
        if (
            isinstance(raw_metric_fields, (str, bytes))
            or not isinstance(raw_metric_fields, list)
            or not raw_metric_fields
            or any(not isinstance(name, str) or not name for name in raw_metric_fields)
            or len(set(raw_metric_fields)) != len(raw_metric_fields)
        ):
            raise ConfigError("run.metric_fields must contain unique non-empty strings")

        placement = raw.get("placement", {})
        if not isinstance(placement, Mapping):
            raise ConfigError("[placement] must be a mapping")
        unknown_placement = set(placement) - _PLACEMENT_KEYS
        if unknown_placement:
            raise ConfigError(f"unknown [placement] keys: {sorted(unknown_placement)}")
        raw_hosts = placement.get("hosts", [])
        if isinstance(raw_hosts, (str, bytes)) or not isinstance(raw_hosts, list):
            raise ConfigError("placement.hosts must be an array")
        hosts: list[str] = []
        for host in raw_hosts:
            if not isinstance(host, str) or not host.strip():
                raise ConfigError("placement.hosts entries must be non-empty strings")
            if host.startswith("-"):
                raise ConfigError("placement.hosts entries must not start with '-'")
            if any(character.isspace() or ord(character) < 32 for character in host):
                raise ConfigError("placement.hosts entries must not contain whitespace")
            if host in hosts:
                raise ConfigError(f"duplicate placement host: {host!r}")
            hosts.append(host)
        raw_numa_nodes = placement.get("numa_nodes", [])
        if (
            isinstance(raw_numa_nodes, (str, bytes))
            or not isinstance(raw_numa_nodes, list)
        ):
            raise ConfigError("placement.numa_nodes must be an array")
        numa_nodes: list[int] = []
        for node in raw_numa_nodes:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ConfigError(
                    "placement.numa_nodes entries must be non-negative integers"
                )
            if node in numa_nodes:
                raise ConfigError(f"duplicate placement NUMA node: {node}")
            numa_nodes.append(node)
        if numa_nodes and not hosts:
            raise ConfigError("placement.numa_nodes requires placement.hosts")

        session = raw.get("session", {})
        if not isinstance(session, Mapping):
            raise ConfigError("[session] must be a mapping")
        forbidden_session = {
            "session_id", "template_path", "scenario_id", "market_path", "chainlink_path"
        } & set(session)
        if forbidden_session:
            raise ConfigError(
                "[session] cannot set " + ", ".join(sorted(forbidden_session))
            )
        resolved_session = dict(session)
        resolved_session.setdefault("event_cursor", "scalar")
        resolved_session.setdefault("metric_profile", "full_summary")
        resolved_session.setdefault("enable_slippage_probes", False)

        candidate = raw.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ConfigError("config requires a [candidate] table")
        unknown_candidate = set(candidate) - _CANDIDATE_KEYS
        if unknown_candidate:
            raise ConfigError(f"unknown [candidate] keys: {sorted(unknown_candidate)}")

        scenario = raw.get("scenario")
        if not isinstance(scenario, Mapping):
            raise ConfigError("config requires a [scenario] table")
        unknown_scenario = set(scenario) - _SCENARIO_KEYS
        if unknown_scenario:
            raise ConfigError(f"unknown [scenario] keys: {sorted(unknown_scenario)}")
        scenario_id = _required_string(scenario, "id", "scenario")
        market = _resolve_path(
            _required_string(scenario, "market", "scenario"), config_path.parent
        )
        resolved_scenario: dict[str, Any] = {
            "id": scenario_id,
            "market": str(market),
        }
        chainlink = scenario.get("chainlink")
        if chainlink is not None:
            if not isinstance(chainlink, str) or not chainlink.strip():
                raise ConfigError("scenario.chainlink must be a non-empty string")
            resolved_scenario["chainlink"] = str(
                _resolve_path(chainlink, config_path.parent)
            )
        scenario_yb_mode = scenario.get("yb_mode", "off")
        if not isinstance(scenario_yb_mode, str):
            raise ConfigError("scenario.yb_mode must be a string")
        resolved_scenario["yb_mode"] = scenario_yb_mode
        if (
            resolved_session["event_cursor"] == "exact_skip"
            and resolved_session["metric_profile"] != "grid_core"
        ):
            raise ConfigError("exact_skip requires metric_profile='grid_core'")
        if (
            resolved_session["metric_profile"] == "grid_core"
            and (
                scenario_yb_mode != "off"
                or bool(resolved_session["enable_slippage_probes"])
            )
        ):
            raise ConfigError(
                "grid_core requires yb_mode='off' and slippage disabled"
            )
        if (
            any(name.startswith("tw_real_slippage_") for name in raw_metric_fields)
            and resolved_session["enable_slippage_probes"] is not True
        ):
            raise ConfigError(
                "tw_real_slippage_* metrics require "
                "session.enable_slippage_probes=true"
            )

        compiled_policy = raw.get("compiled_policy")
        compiled_policy_header: Path | None = None
        compiled_policy_id: str | None = None
        if compiled_policy is not None:
            if not isinstance(compiled_policy, Mapping):
                raise ConfigError("[compiled_policy] must be a mapping")
            unknown_compiled_policy = set(compiled_policy) - _COMPILED_POLICY_KEYS
            if unknown_compiled_policy:
                raise ConfigError(
                    "unknown [compiled_policy] keys: "
                    f"{sorted(unknown_compiled_policy)}"
                )
            compiled_policy_header = _resolve_path(
                _required_string(compiled_policy, "header", "compiled_policy"),
                config_path.parent,
            )
            compiled_policy_id = _required_string(
                compiled_policy, "id", "compiled_policy"
            )
        candidate_config = CandidateConfig.from_mapping(candidate)
        if (
            compiled_policy_header is None
            and candidate_config.defaults["policy_params"]
        ):
            raise ConfigError(
                "candidate.defaults.policy_params must be empty without [compiled_policy]"
            )
        base = config_path.parent
        config = cls(
            path=config_path,
            run_id=_required_string(run, "id", "run"),
            evaluator=_resolve_path(_required_string(run, "evaluator", "run"), base),
            template=_resolve_path(_required_string(run, "template", "run"), base),
            batch_size=batch_size,
            workers=workers,
            metric_fields=tuple(raw_metric_fields),
            hosts=tuple(hosts),
            numa_nodes=tuple(numa_nodes),
            candidate=candidate_config,
            session=resolved_session,
            scenario=resolved_scenario,
            robustness=parse_robustness_axes(
                raw.get("robustness"), required=False
            ),
            compiled_policy_header=compiled_policy_header,
            compiled_policy_id=compiled_policy_id,
        )
        if hosts:
            _execution_inputs(config, remote=True)
        return config


@dataclass(frozen=True, slots=True)
class RemoteJob:
    """The minimal local handle for one detached coordinator process."""

    run_id: str
    coordinator: str
    remote_output: str


@dataclass(frozen=True, slots=True)
class RemoteRunStatus:
    """Current coordinator state plus its latest operator-facing log line."""

    state: str
    coordinator: str
    remote_output: str | None
    detail: str = ""
    exit_code: int | None = None


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


def _shuffled_block_leases(
    total: int,
    block_size: int,
    batch_size: int,
    slots: int,
    *,
    seed: int = _BLOCK_SHUFFLE_SEED,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Group reproducibly shuffled blocks into machine-sized leases."""
    if total < 1 or block_size < 1 or batch_size < 1 or slots < 1:
        raise ValueError("total, block size, batch size, and slots must be positive")
    blocks = [
        (start, min(total, start + block_size))
        for start in range(0, total, block_size)
    ]
    random.Random(seed).shuffle(blocks)
    blocks_per_lease = max(1, batch_size // block_size) * slots
    return tuple(
        tuple(blocks[index:index + blocks_per_lease])
        for index in range(0, len(blocks), blocks_per_lease)
    )


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
        leases = _shuffled_block_leases(
            total,
            _schedule_block_size(config, total),
            batch_size,
            max(1, len(config.numa_nodes)),
        )
        lease_counts = [_range_count(lease) for lease in leases]
        allocation = str(max(lease_counts))
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
    for name in ("template", "market", "chainlink"):
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


def _execution_inputs(config: RunConfig, *, remote: bool) -> dict[str, str]:
    inputs = {
        "evaluator": str(config.evaluator),
        "template": str(config.template),
        "market": config.scenario["market"],
    }
    if (chainlink := config.scenario.get("chainlink")) is not None:
        inputs["chainlink"] = chainlink
    if config.compiled_policy_header is not None:
        inputs["policy_header"] = str(config.compiled_policy_header)
    if remote:
        try:
            optimizer_root = next(
                parent
                for parent in config.path.parents
                if parent.name == "curve-fx-optimization"
            )
        except StopIteration as exc:
            raise ConfigError(
                "remote config must be inside a curve-fx-optimization repository"
            ) from exc
        workspace = optimizer_root.parent.resolve()
        mapped = {}
        for name, value in inputs.items():
            try:
                relative = Path(value).resolve().relative_to(workspace)
            except ValueError as exc:
                raise ConfigError(f"remote {name} path must be inside {workspace}") from exc
            mapped[name] = str(REMOTE_BASE.joinpath(*relative.parts))
        return mapped
    return inputs


def open_session_request(config: RunConfig, *, remote: bool | None = None) -> dict[str, Any]:
    inputs = _execution_inputs(config, remote=bool(config.hosts) if remote is None else remote)
    request = {
        "template_path": inputs["template"],
        "scenario_id": config.scenario["id"],
        "market_path": inputs["market"],
        **config.session,
    }
    if (chainlink := inputs.get("chainlink")) is not None:
        request["chainlink_path"] = chainlink
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

    for key in ("template_path", "market_path", "chainlink_path"):
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
) -> tuple[ArtifactPaths, float, tuple[int, ...]]:
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
        try:
            fleet.start()
            calculation_started = time.monotonic()
            completed = 0
            lease_ids: list[int] = []
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
        finally:
            fleet.close()
        calculation_s = time.monotonic() - calculation_started
        return writer.finalize_partition(), calculation_s, tuple(lease_ids)


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
    paths, calculation_s, lease_ids = _run_leased_worker(
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
            not sent_finish
            or awaiting_lease is not None
            or receipt.get("count") != assigned_count
            or receipt.get("output") != partition
            or receipt.get("lease_ids") != assigned_ids
        ):
            raise RuntimeError(f"{host} worker receipt does not match its assignment")
        progress.update(index, {
            "type": "complete",
            "completed": assigned_count,
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
            f"({max(_range_count(lease) for lease in leases)} pools max) "
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
                fetches[fetcher.submit(fetch_partition, index, host, receipt)] = index
            for future in as_completed(fetches):
                index = fetches[future]
                try:
                    partitions[index] = future.result()
                except Exception as exc:
                    errors.append(f"worker {index} fetch: {exc}")
        if errors:
            raise RuntimeError("cluster worker failure: " + " | ".join(errors))
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
            "lease_size": max(_range_count(lease) for lease in leases),
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
            [partitions[index] for index, _host in active],
            run_id=config.run_id,
            total=total,
            metadata=metadata,
            metric_names=config.metric_fields,
        )
        elapsed = time.monotonic() - started
        print(
            f"coordinator: complete {total} pools in {elapsed:.1f}s "
            f"({total / elapsed:.1f} pools/s wall)",
            file=sys.stderr,
        )
        return paths
    finally:
        if not progress_closed:
            progress.close()
        shutil.rmtree(partition_root, ignore_errors=True)


def _remote_paths(output_dir: str | Path) -> tuple[Path, ArtifactPaths, Path]:
    destination = Path(output_dir).expanduser().resolve()
    paths = ArtifactPaths(destination / "run.json", destination / "results.npz")
    return destination, paths, destination / REMOTE_JOB_FILENAME


def _local_run_complete(config: RunConfig, paths: ArtifactPaths) -> bool:
    if paths.run_json.is_file() and paths.results_npz.is_file():
        try:
            run_id = json.loads(paths.run_json.read_text())["run_id"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid local run metadata") from exc
        if run_id != config.run_id:
            raise RuntimeError("local results do not match the configured run")
        return True
    if paths.run_json.exists() or paths.results_npz.exists():
        raise RuntimeError("local result publication is incomplete")
    return False


def _validate_remote_output(remote_output: str) -> None:
    if (
        not remote_output.startswith("/tmp/fxopt-grid.")
        or any(character.isspace() for character in remote_output)
    ):
        raise RuntimeError("coordinator returned an invalid output directory")


def _write_remote_job(path: Path, job: RemoteJob) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({
        "run_id": job.run_id,
        "coordinator": job.coordinator,
        "remote_output": job.remote_output,
    }, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_remote_job(config: RunConfig, path: Path) -> RemoteJob:
    try:
        payload = json.loads(path.read_text())
        values = (
            payload["run_id"],
            payload["coordinator"],
            payload["remote_output"],
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise TypeError
        job = RemoteJob(
            run_id=values[0],
            coordinator=values[1],
            remote_output=values[2],
        )
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid remote job record: {path}") from exc
    if job.run_id != config.run_id or job.coordinator != config.hosts[0]:
        raise RuntimeError("remote job record does not match the configured run")
    _validate_remote_output(job.remote_output)
    return job


def _start_remote_job(
    config: RunConfig,
    output_dir: str | Path,
    *,
    transfer: bool,
    rebuild: bool,
) -> RemoteJob:
    destination, paths, job_path = _remote_paths(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if paths.run_json.exists() or paths.results_npz.exists():
        raise FileExistsError(f"completed result already exists in {destination}")
    if job_path.exists():
        raise FileExistsError(
            f"remote job already exists in {destination}; "
            "use --status, --follow, --retrieve, or --stop"
        )

    prepare_remote(config, transfer=transfer, rebuild=rebuild)
    remote_config = stage_remote_run(config)
    coordinator = config.hosts[0]
    created = subprocess.run(
        [
            "ssh", *SSH_OPTIONS, "--", coordinator,
            "mktemp", "-d", "/tmp/fxopt-grid.XXXXXXXX",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    remote_output = created.stdout.strip()
    _validate_remote_output(remote_output)
    job = RemoteJob(config.run_id, coordinator, remote_output)
    _write_remote_job(job_path, job)

    optimizer_root = next(
        parent
        for parent in config.path.parents
        if parent.name == "curve-fx-optimization"
    )
    workspace = optimizer_root.parent.resolve()
    remote_project = str(REMOTE_BASE / "curve-fx-optimization")
    worker = [
        f"{remote_project}/scripts/cluster-python",
        "-m", "fxopt.cli", "_cluster-worker", remote_config,
        "--output", remote_output,
        "--origin-workspace", str(workspace),
        "--origin-config", str(config.path),
    ]
    remote_script = " ".join((
        "set -euo pipefail;",
        "export PYTHONPATH=" + shlex.quote(
            f"{remote_project}/src:"
            f"{REMOTE_BASE}/curve-fx-arb-harness/python/src:"
            f"{remote_project}/.venv-cluster/lib/python3.12/site-packages"
        ) + ";",
        "export PYTHONNOUSERSITE=1;",
        "exec " + shlex.join(worker),
    ))
    remote_command = (
        f"nix-shell {shlex.quote(f'{remote_project}/shell.nix')} "
        f"--run {shlex.quote(remote_script)}"
    )
    log_path = f"{remote_output}/coordinator.log"
    exit_path = f"{remote_output}/exit_code"
    pid_path = f"{remote_output}/pid"
    wrapped = " ".join((
        f"{remote_command};",
        "status=$?;",
        f"printf '%s\\n' \"$status\" > {shlex.quote(exit_path)};",
        "exit \"$status\"",
    ))
    launch = " ".join((
        f": > {shlex.quote(log_path)};",
        f"nohup setsid /bin/sh -c {shlex.quote(wrapped)} </dev/null "
        f">>{shlex.quote(log_path)} 2>&1 &",
        f"printf '%s\\n' \"$!\" > {shlex.quote(pid_path)}",
    ))
    try:
        subprocess.run(
            ["ssh", *SSH_OPTIONS, "--", coordinator, launch],
            check=True,
        )
    except BaseException:
        print(
            f"cluster: launch state retained in {job_path}",
            file=sys.stderr,
        )
        raise
    print(
        f"cluster: started detached on {coordinator}:{remote_output}",
        file=sys.stderr,
    )
    return job


def remote_run_status(
    config_path: str | Path,
    output_dir: str | Path,
) -> RemoteRunStatus:
    """Read one detached coordinator's state without changing it."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote status requires placement hosts")
    _destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        return RemoteRunStatus("retrieved", config.hosts[0], None)
    if not job_path.is_file():
        raise FileNotFoundError(f"remote job record not found: {job_path}")
    job = _read_remote_job(config, job_path)
    script = " ".join((
        f"work={shlex.quote(job.remote_output)};",
        "state=stopped; code=;",
        "if ! test -d \"$work\"; then state=missing;",
        "elif test -s \"$work/run.json\" && test -s \"$work/results.npz\"; "
        "then state=complete; code=0;",
        "elif test -s \"$work/stopped\"; then state=stopped;",
        "elif test -s \"$work/exit_code\"; then "
        "code=$(cat \"$work/exit_code\"); state=failed;",
        "elif test -s \"$work/pid\" && "
        "kill -0 \"$(cat \"$work/pid\")\" 2>/dev/null; then state=running;",
        "fi;",
        "printf '%s\\n%s\\n' \"$state\" \"$code\";",
        "tail -n 1 \"$work/coordinator.log\" 2>/dev/null || true",
    ))
    checked = subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", job.coordinator, script],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = checked.stdout.splitlines()
    if len(lines) < 2 or lines[0] not in {
        "running", "complete", "failed", "stopped", "missing"
    }:
        raise RuntimeError("coordinator returned an invalid job status")
    try:
        exit_code = int(lines[1]) if lines[1] else None
    except ValueError as exc:
        raise RuntimeError("coordinator returned an invalid exit code") from exc
    return RemoteRunStatus(
        lines[0],
        job.coordinator,
        job.remote_output,
        detail=lines[2] if len(lines) > 2 else "",
        exit_code=exit_code,
    )


def stop_remote_run(
    config_path: str | Path,
    output_dir: str | Path,
) -> RemoteRunStatus:
    """Stop one detached coordinator while retaining its diagnostic state."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote stop requires placement hosts")
    destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        raise RuntimeError("remote job was already retrieved")
    status = remote_run_status(config.path, destination)
    if status.state == "complete":
        raise RuntimeError("remote job is complete; use --retrieve")
    if status.state != "running":
        return status

    job = _read_remote_job(config, job_path)
    script = " ".join((
        "set -eu;",
        f"work={shlex.quote(job.remote_output)};",
        "pid=$(cat \"$work/pid\");",
        "case \"$pid\" in ''|*[!0-9]*) exit 2;; esac;",
        "if test -s \"$work/run.json\" && test -s \"$work/results.npz\"; "
        "then printf 'complete\\n'; exit 0; fi;",
        "signal_tree() { signal=$1; current=$2; "
        "for child in $(cat \"/proc/$current/task/$current/children\" "
        "2>/dev/null || true); do signal_tree \"$signal\" \"$child\"; done; "
        "kill -\"$signal\" \"$current\" 2>/dev/null || true; };",
        "pgid=$(ps -o pgid= -p \"$pid\" 2>/dev/null | tr -d ' ' || true);",
        "if test \"$pgid\" = \"$pid\"; then "
        "kill -TERM -- \"-$pid\" 2>/dev/null || true; "
        "else signal_tree TERM \"$pid\"; fi;",
        "attempt=0; while kill -0 \"$pid\" 2>/dev/null && "
        "test \"$attempt\" -lt 20; do sleep 0.25; attempt=$((attempt + 1)); done;",
        "if kill -0 \"$pid\" 2>/dev/null; then "
        "if test \"$pgid\" = \"$pid\"; then "
        "kill -KILL -- \"-$pid\" 2>/dev/null || true; "
        "else signal_tree KILL \"$pid\"; fi; fi;",
        "printf 'operator-stop\\n' > \"$work/stopped\";",
        "printf 'stopped\\n'",
    ))
    stopped = subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", job.coordinator, script],
        capture_output=True,
        text=True,
        check=True,
    )
    state = stopped.stdout.strip()
    if state == "complete":
        raise RuntimeError(
            "remote job completed before it could be stopped; use --retrieve"
        )
    if state != "stopped":
        raise RuntimeError("coordinator returned an invalid stop result")
    return RemoteRunStatus(
        "stopped",
        job.coordinator,
        job.remote_output,
        detail=status.detail,
    )


def _retrieve_remote_job(
    destination: Path,
    paths: ArtifactPaths,
    job_path: Path,
    job: RemoteJob,
) -> ArtifactPaths:
    incoming = destination / ".fetch"
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir(parents=True)
    print("cluster: fetching final results...", file=sys.stderr)
    fetch = [
        "rsync", "-a", "--partial",
        "--include=/run.json", "--include=/results.npz", "--exclude=*",
        "-e", RSYNC_SSH, "--",
        f"{job.coordinator}:{job.remote_output}/", f"{incoming}/",
    ]
    for attempt in range(1, 4):
        fetched = subprocess.run(fetch, check=False)
        if fetched.returncode == 0:
            break
        if attempt < 3:
            print(f"cluster: fetch retry {attempt + 1}/3...", file=sys.stderr)
    else:
        raise subprocess.CalledProcessError(fetched.returncode, fetch)
    incoming_run = incoming / "run.json"
    incoming_results = incoming / "results.npz"
    if not incoming_run.is_file() or not incoming_results.is_file():
        raise RuntimeError("coordinator did not return both result artifacts")
    os.replace(incoming_results, paths.results_npz)
    try:
        os.replace(incoming_run, paths.run_json)
    except BaseException:
        paths.results_npz.unlink(missing_ok=True)
        raise
    incoming.rmdir()
    job_path.unlink()
    cleanup = subprocess.run(
        [
            "ssh", *SSH_OPTIONS, "--", job.coordinator,
            "rm", "-rf", "--", job.remote_output,
        ],
        check=False,
    )
    if cleanup.returncode != 0:
        print(
            f"cluster: warning: could not remove "
            f"{job.coordinator}:{job.remote_output}",
            file=sys.stderr,
        )
    return paths


def retrieve_remote_run(
    config_path: str | Path,
    output_dir: str | Path,
) -> ArtifactPaths:
    """Fetch and atomically publish one completed detached remote run."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote retrieval requires placement hosts")
    destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        return paths
    status = remote_run_status(config.path, destination)
    if status.state != "complete":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(f"remote job is {status.state}{detail}")
    return _retrieve_remote_job(
        destination,
        paths,
        job_path,
        _read_remote_job(config, job_path),
    )


def _follow_remote_job(job: RemoteJob, *, from_start: bool) -> None:
    lines = "+1" if from_start else "1"
    script = " ".join((
        f"work={shlex.quote(job.remote_output)};",
        "pid=$(cat \"$work/pid\");",
        f"exec tail --pid=\"$pid\" -n {lines} -f \"$work/coordinator.log\"",
    ))
    subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", job.coordinator, script],
        check=True,
    )


def _follow_and_retrieve(
    config: RunConfig,
    destination: Path,
    job: RemoteJob,
    *,
    from_start: bool,
) -> ArtifactPaths:
    try:
        _follow_remote_job(job, from_start=from_start)
    except BaseException:
        print(
            "cluster: connection ended; detached job retained "
            "(--status / --follow / --retrieve / --stop)",
            file=sys.stderr,
        )
        raise
    status = remote_run_status(config.path, destination)
    if status.state != "complete":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(f"remote job is {status.state}{detail}")
    _destination, paths, job_path = _remote_paths(destination)
    return _retrieve_remote_job(destination, paths, job_path, job)


def follow_remote_run(
    config_path: str | Path,
    output_dir: str | Path,
) -> ArtifactPaths:
    """Follow a detached coordinator and retrieve its completed artifacts."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote follow requires placement hosts")
    destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        return paths
    status = remote_run_status(config.path, destination)
    if status.state == "complete":
        return _retrieve_remote_job(
            destination,
            paths,
            job_path,
            _read_remote_job(config, job_path),
        )
    if status.state != "running":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(f"remote job is {status.state}{detail}")
    job = _read_remote_job(config, job_path)
    return _follow_and_retrieve(config, destination, job, from_start=False)


def run_remote_config(
    config_path: str | Path | RunConfig,
    output_dir: str | Path,
    *,
    transfer: bool = False,
    rebuild: bool = False,
) -> ArtifactPaths:
    """Launch, follow, and retrieve one disconnect-safe remote grid."""
    config = (
        config_path
        if isinstance(config_path, RunConfig)
        else RunConfig.from_toml(config_path)
    )
    if not config.hosts:
        raise ConfigError("remote coordinator requires placement hosts")
    job = _start_remote_job(
        config,
        output_dir,
        transfer=transfer,
        rebuild=rebuild,
    )
    destination = Path(output_dir).expanduser().resolve()
    return _follow_and_retrieve(config, destination, job, from_start=True)


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
    "follow_remote_run",
    "RunConfig",
    "RemoteRunStatus",
    "REMOTE_JOB_FILENAME",
    "grid_summary",
    "open_session_request",
    "ProgressCallback",
    "remote_run_status",
    "retrieve_remote_run",
    "run_config",
    "run_remote_config",
    "run_leased_worker",
    "run_metadata",
    "stage_remote_run",
]
