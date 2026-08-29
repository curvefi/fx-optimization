"""Execute a bounded local-or-SSH fxopt grid through one persistent harness session."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tomllib
from typing import Any

from .candidates import CandidateSpec
from .config import CandidateConfig, ConfigError
from .contract import Candidate
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
    ssh_client_factory,
    transfer_workspace,
)
from .results import ArtifactPaths, GridResultWriter
from .robustness import (
    RobustnessAxis,
    parse_robustness_axes,
    robustness_metadata,
)


_RUN_KEYS = frozenset({"id", "evaluator", "template", "batch_size", "workers", "metric_fields"})
_PLACEMENT_KEYS = frozenset({"hosts", "numa_nodes"})
_CANDIDATE_KEYS = frozenset({"defaults", "axes"})
_SCENARIO_KEYS = frozenset({"id", "market", "chainlink", "yb_mode"})

ProgressCallback = Callable[[int, int], None]
LaneCallback = Callable[[str, int, float], None]
REMOTE_JOB_FILENAME = ".remote-job.json"

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
            or any(not isinstance(name, str) or not name for name in raw_metric_fields)
            or len(set(raw_metric_fields)) != len(raw_metric_fields)
        ):
            raise ConfigError("run.metric_fields must be an array of unique non-empty strings")

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
        scenario_yb_mode = scenario.get("yb_mode")
        if scenario_yb_mode is not None and not isinstance(scenario_yb_mode, str):
            raise ConfigError("scenario.yb_mode must be a string")
        if scenario_yb_mode is not None:
            resolved_scenario["yb_mode"] = scenario_yb_mode
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
            candidate=CandidateConfig.from_mapping(candidate),
            session=dict(session),
            scenario=resolved_scenario,
            robustness=parse_robustness_axes(
                raw.get("robustness"), required=False
            ),
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


def _display_value(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def grid_summary(config_path: str | Path) -> str:
    """Describe the configured Cartesian axes in their operator-facing units."""
    config = RunConfig.from_toml(config_path)
    total = len(config.candidate.grid())
    with config.path.open("rb") as stream:
        raw_axes = tomllib.load(stream)["candidate"].get("axes", {})
    parts = []
    for name, values in config.candidate.axes.items():
        raw = raw_axes[name]
        displayed = raw.get("values") if isinstance(raw, Mapping) and "values" in raw else raw
        if isinstance(raw, Mapping) and "start" in raw:
            displayed = (raw["start"], raw["stop"])
        endpoints = displayed if isinstance(displayed, list) else list(displayed)
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
        lanes_per_blade = len(config.numa_nodes) or 1
        per_lane, extra = divmod(total, len(config.hosts) * lanes_per_blade)
        counts = [
            per_lane * lanes_per_blade
            + min(lanes_per_blade, max(0, extra - index * lanes_per_blade))
            for index in range(len(config.hosts))
        ]
        allocation = (
            str(counts[0])
            if min(counts) == max(counts)
            else f"{min(counts)}-{max(counts)}"
        )
        placement = f" on {len(config.hosts)} blades ({allocation} pools/blade)"
    suffix = f": {', '.join(parts)}" if parts else ""
    return f"running {total} pools grid{placement}{suffix}"


def candidate_from_spec(spec: CandidateSpec) -> Candidate:
    payload = spec.payload
    expected = {"policy_params", "pool"}
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ConfigError(
            "candidate payload must contain only policy_params and pool ("
            + "; ".join(details)
            + ")"
        )
    policy_params = payload["policy_params"]
    pool = payload["pool"]
    if not isinstance(policy_params, (list, tuple)):
        raise ConfigError("candidate.policy_params must be an array")
    if not isinstance(pool, Mapping):
        raise ConfigError("candidate.pool must be a mapping")
    return Candidate(
        candidate_id=spec.candidate_id,
        policy_params=tuple(policy_params),
        pool_overrides=pool,
    )


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


def placement_lanes(
    config: RunConfig,
    client_factory: ClientFactory | None = None,
    *,
    stage_inputs: bool = True,
) -> tuple[PlacementLane, ...]:
    """Resolve local, injected, or SSH evaluator lanes once for every workflow."""
    if client_factory is not None:
        return (PlacementLane("injected", client_factory),)
    if config.hosts:
        inputs = _execution_inputs(config, remote=True)
        if stage_inputs:
            stage_remote_run(config)
        nodes: tuple[int | None, ...] = config.numa_nodes or (None,)
        if config.workers < len(nodes):
            raise ConfigError("run.workers must cover every configured NUMA node")
        workers, extra = divmod(config.workers, len(nodes))
        return tuple(
            PlacementLane(
                host if node is None else f"{host}:numa{node}",
                ssh_client_factory(
                    host,
                    inputs["evaluator"],
                    workers=workers + (index < extra),
                    **(
                        {}
                        if node is None
                        else {
                            "remote_prefix": (
                                "numactl",
                                f"--cpunodebind={node}",
                                f"--membind={node}",
                            )
                        }
                    ),
                    timeout=600.0,
                    verify_local_inputs=False,
                ),
            )
            for host in config.hosts
            for index, node in enumerate(nodes)
        )
    return (
        PlacementLane(
            "local",
            local_client_factory(
                config.evaluator,
                work_dir=config.path.parent,
                workers=config.workers,
            ),
        ),
    )


def _execution_inputs(config: RunConfig, *, remote: bool) -> dict[str, str]:
    inputs = {
        "evaluator": str(config.evaluator),
        "template": str(config.template),
        "market": config.scenario["market"],
    }
    if (chainlink := config.scenario.get("chainlink")) is not None:
        inputs["chainlink"] = chainlink
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
    origin = (
        "autoresearch"
        if config_parent.name == "autoresearch"
        and config_parent.parent.name == "configs"
        else "human"
        if config_parent.name == "experiments"
        and config_parent.parent.name == "configs"
        else "external"
    )
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
        "open_session": open_session_request(config),
        "replay": {
            "evaluator": replay_path(local_inputs["evaluator"]),
            "work_dir": str(config_parent),
            "open_session": replay_session,
        },
    }
    if config.robustness:
        metadata["robustness"] = robustness_metadata(config.robustness)
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
        rebuild_shared_evaluator(first, inputs["evaluator"])
    require_shared_evaluator(first, inputs["evaluator"])
    print("cluster: ready", file=sys.stderr)


def _run(
    config: RunConfig,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None,
    progress_callback: ProgressCallback | None,
    lane_callback: LaneCallback | None,
    stage_inputs: bool,
    origin_workspace: Path | None,
    origin_config: Path | None,
) -> ArtifactPaths:
    if config.hosts and not config.metric_fields:
        raise ConfigError("remote grid runs require run.metric_fields")
    lanes = placement_lanes(config, client_factory, stage_inputs=stage_inputs)
    candidate_grid = config.candidate.grid()
    lane_count = len(lanes)
    effective_batch = min(config.batch_size, (len(candidate_grid) + lane_count - 1) // lane_count)
    metadata = run_metadata(
        config,
        effective_batch=effective_batch,
        origin_workspace=origin_workspace,
        origin_config=origin_config,
    )
    metadata["execution_order"] = {
        "kind": "rotating_blocks_striped_v1",
        "block_size": effective_batch,
        "rotations": lane_count,
    }
    if client_factory is not None:
        metadata["placement"] = "injected"
    open_session = metadata["open_session"]
    total = len(candidate_grid)
    writer = GridResultWriter(
        output_dir,
        run_id=config.run_id,
        total=total,
        metadata=metadata,
        metric_names=config.metric_fields or None,
    )
    with writer:
        completed = 0
        lane_failures: dict[str, int] = {}
        if progress_callback is not None:
            progress_callback(completed, total)
        fleet = EvaluatorFleet(
            lanes,
            session_id=config.run_id,
            batch_size=effective_batch,
            start_ordinal=completed,
            open_session=open_session,
            metric_fields=config.metric_fields or None,
            lane_callback=lane_callback,
        )

        def stripe(index: int):
            for ordinal, spec in candidate_grid.iter_stripe(
                effective_batch, lane_count, index
            ):
                yield ordinal, candidate_from_spec(spec)

        assignments = tuple(stripe(index) for index in range(lane_count))
        try:
            for batch in fleet.iter_grid(assignments):
                if batch.error is not None:
                    failures = lane_failures.get(batch.lane, 0) + 1
                    lane_failures[batch.lane] = failures
                    if failures <= 4 or failures % 100 == 0:
                        print(
                            f"{batch.lane}: failed chunk {failures} "
                            f"({len(batch.results)} pools): {batch.error}",
                            file=sys.stderr,
                        )
                writer.append(batch.candidates, batch.results)
                completed += len(batch.results)
                if progress_callback is not None:
                    progress_callback(completed, total)
        finally:
            fleet.close()
        return writer.finalize()


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
    stream_blade: str | None,
) -> RemoteJob:
    destination, paths, job_path = _remote_paths(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if paths.run_json.exists() or paths.results_npz.exists():
        raise FileExistsError(f"completed result already exists in {destination}")
    if job_path.exists():
        raise FileExistsError(
            f"remote job already exists in {destination}; use --status or --retrieve"
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
    remote_uv = "/home/heswithme/.nix-profile/bin/uv"
    remote_python = "/home/heswithme/.nix-profile/bin/python3"
    remote_venv = f"{remote_project}/.venv-cluster"
    worker = [
        f"{remote_venv}/bin/python",
        "-m", "fxopt.cli", "_cluster-worker", remote_config,
        "--output", remote_output,
        "--origin-workspace", str(workspace),
        "--origin-config", str(config.path),
    ]
    if stream_blade is not None:
        worker.extend(("--stream-blade", stream_blade))
    remote_script = " ".join((
        "set -euo pipefail;",
        f"export UV_PROJECT_ENVIRONMENT={shlex.quote(remote_venv)};",
        f"export UV_CACHE_DIR={shlex.quote('/tmp/uv-cache')};",
        f"{shlex.quote(remote_uv)} sync --quiet --project "
        f"{shlex.quote(remote_project)} --only-group cluster --frozen "
        f"--python {shlex.quote(remote_python)} --no-python-downloads "
        "--link-mode copy;",
        "export PYTHONPATH=" + shlex.quote(
            f"{remote_project}/src:"
            f"{REMOTE_BASE}/curve-fx-arb-harness/python/src"
        ) + ";",
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
        f"nohup /bin/sh -c {shlex.quote(wrapped)} </dev/null "
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
            "(--status / --follow / --retrieve)",
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
    config_path: str | Path,
    output_dir: str | Path,
    *,
    transfer: bool = False,
    rebuild: bool = False,
    stream_blade: str | None = None,
) -> ArtifactPaths:
    """Launch, follow, and retrieve one disconnect-safe remote grid."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote coordinator requires placement hosts")
    if stream_blade is not None and stream_blade not in config.hosts:
        raise ConfigError(f"--stream-blade is not a placement host: {stream_blade}")
    job = _start_remote_job(
        config,
        output_dir,
        transfer=transfer,
        rebuild=rebuild,
        stream_blade=stream_blade,
    )
    destination = Path(output_dir).expanduser().resolve()
    return _follow_and_retrieve(config, destination, job, from_start=True)


def run_config(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    lane_callback: LaneCallback | None = None,
    transfer: bool = False,
    rebuild: bool = False,
    prepared: bool = False,
    origin_workspace: str | Path | None = None,
    origin_config: str | Path | None = None,
) -> ArtifactPaths:
    """Run every grid point in bounded batches and publish the two artifacts."""
    config = RunConfig.from_toml(config_path)
    if client_factory is None and prepared and config.hosts:
        print(
            f"coordinator: checking {len(config.hosts)} direct blades...",
            file=sys.stderr,
        )
        require_reachable_hosts(config.hosts)
    if client_factory is None and not prepared:
        prepare_remote(config, transfer=transfer, rebuild=rebuild)
    output_path = Path(output_dir).expanduser().resolve()
    return _run(
        config,
        output_path,
        client_factory=client_factory,
        progress_callback=progress_callback,
        lane_callback=lane_callback,
        stage_inputs=not prepared,
        origin_workspace=(
            None if origin_workspace is None else Path(origin_workspace).expanduser().resolve()
        ),
        origin_config=(
            None if origin_config is None else Path(origin_config).expanduser().resolve()
        ),
    )


__all__ = [
    "follow_remote_run",
    "RunConfig",
    "RemoteRunStatus",
    "REMOTE_JOB_FILENAME",
    "candidate_from_spec",
    "grid_summary",
    "open_session_request",
    "placement_lanes",
    "ProgressCallback",
    "remote_run_status",
    "retrieve_remote_run",
    "run_config",
    "run_remote_config",
    "run_metadata",
    "stage_remote_run",
]
