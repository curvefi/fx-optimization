"""Execute a bounded local-or-SSH fxopt grid through one persistent harness session."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePosixPath
import tomllib
from typing import Any

from .candidates import CandidateSpec
from .config import CandidateConfig, ConfigError
from .contract import Candidate
from .engine import ClientFactory
from .placement import (
    EvaluatorFleet,
    PlacementLane,
    ensure_remote_file,
    local_client_factory,
    ssh_client_factory,
)
from .results import ArtifactPaths, ResultWriter
from .robustness import (
    RobustnessAxis,
    parse_robustness_axes,
    robustness_metadata,
)


_RUN_KEYS = frozenset({"id", "evaluator", "template", "batch_size", "workers"})
_PLACEMENT_KEYS = frozenset({"hosts", "numa_nodes"})
_CANDIDATE_KEYS = frozenset({"defaults", "axes"})
_SCENARIO_KEYS = frozenset({"id", "market", "chainlink", "yb_mode"})

ProgressCallback = Callable[[int, int], None]

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


def _display_value(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def grid_summary(config_path: str | Path) -> str:
    """Describe the configured Cartesian axes in their operator-facing units."""
    config = RunConfig.from_toml(config_path)
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
    suffix = f": {', '.join(parts)}" if parts else ""
    return f"running {len(config.candidate.grid())} pools grid{suffix}"


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


def placement_lanes(
    config: RunConfig,
    client_factory: ClientFactory | None = None,
) -> tuple[PlacementLane, ...]:
    """Resolve local, injected, or SSH evaluator lanes once for every workflow."""
    if client_factory is not None:
        return (PlacementLane("injected", client_factory),)
    if config.hosts:
        local_inputs = _execution_inputs(config, remote=False)
        inputs = _execution_inputs(config, remote=True)
        for name in ("template", "market", "chainlink"):
            if name in inputs:
                ensure_remote_file(config.hosts[0], local_inputs[name], inputs[name])
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
            mapped[name] = str(PurePosixPath("arb", *relative.parts))
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


def run_metadata(config: RunConfig, *, effective_batch: int) -> dict[str, Any]:
    grid = config.candidate.grid()
    inputs = _execution_inputs(config, remote=bool(config.hosts))
    local_inputs = _execution_inputs(config, remote=False)
    replay_session = open_session_request(config, remote=False)
    for key in ("template_path", "market_path", "chainlink_path"):
        if key in replay_session:
            replay_session[key] = str(Path(replay_session[key]).resolve())
    config_parent = config.path.parent
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
        "config": str(config.path),
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
        "axes": {name: list(grid.axes[name]) for name in sorted(grid.axes)},
        "shape": list(grid.shape),
        "candidate_defaults": config.candidate.defaults,
        "open_session": open_session_request(config),
        "replay": {
            "evaluator": str(Path(local_inputs["evaluator"]).resolve()),
            "work_dir": str(config_parent),
            "open_session": replay_session,
        },
    }
    if config.robustness:
        metadata["robustness"] = robustness_metadata(config.robustness)
    return metadata


def _run(
    config: RunConfig,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None,
    progress_callback: ProgressCallback | None,
) -> ArtifactPaths:
    lanes = placement_lanes(config, client_factory)
    candidate_grid = config.candidate.grid()
    lane_count = len(lanes)
    effective_batch = min(config.batch_size, (len(candidate_grid) + lane_count - 1) // lane_count)
    metadata = run_metadata(config, effective_batch=effective_batch)
    if client_factory is not None:
        metadata["placement"] = "injected"
    open_session = metadata["open_session"]
    writer = ResultWriter(
        output_dir,
        run_id=config.run_id,
        metadata=metadata,
        resumable=True,
    )
    with writer:
        total = len(candidate_grid)
        completed = writer.row_count
        if completed > total:
            raise ValueError("partial result contains more rows than this grid")
        if progress_callback is not None:
            progress_callback(completed, total)
        if completed == total:
            return writer.finalize()
        with EvaluatorFleet(
            lanes,
            session_id=config.run_id,
            batch_size=effective_batch,
            start_ordinal=completed,
            open_session=open_session,
        ) as fleet:
            pending = islice(candidate_grid, completed, None)
            batch_size = lane_count * effective_batch
            while specs := tuple(islice(pending, batch_size)):
                batch = tuple(candidate_from_spec(spec) for spec in specs)
                writer.append(batch, fleet.evaluate(batch))
                completed += len(batch)
                if progress_callback is not None:
                    progress_callback(completed, total)
        return writer.finalize()


def run_config(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ArtifactPaths:
    """Run every grid point in bounded batches and publish the two artifacts."""
    config = RunConfig.from_toml(config_path)
    output_path = Path(output_dir).expanduser().resolve()
    return _run(
        config,
        output_path,
        client_factory=client_factory,
        progress_callback=progress_callback,
    )


__all__ = [
    "RunConfig",
    "candidate_from_spec",
    "grid_summary",
    "open_session_request",
    "placement_lanes",
    "ProgressCallback",
    "run_config",
    "run_metadata",
]
