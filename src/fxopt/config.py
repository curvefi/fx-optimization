"""Cartesian candidates, resolved run settings, and input paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import tomllib
from typing import Any

from .candidates import (
    CandidateError,
    merge_payload,
    path_parts,
)
from .grid import CartesianGrid
from .placement import REMOTE_BASE
from .robustness import RobustnessAxis, parse_robustness_axes


class ConfigError(ValueError):
    """Raised when a candidate configuration is malformed."""


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise ConfigError(f"{label} must be numeric")
    try:
        value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ConfigError(f"{label} must be numeric") from exc
    if not value.is_finite():
        raise ConfigError(f"{label} must be finite")
    return value


def _range_values(spec: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    start = _decimal(spec["start"], f"{label}.start")
    stop = _decimal(spec["stop"], f"{label}.stop")
    scale = spec.get("scale", "linear")
    if scale not in {"linear", "log"}:
        raise ConfigError(f"{label}.scale must be 'linear' or 'log'")
    if "step" in spec:
        if scale != "linear":
            raise ConfigError(f"{label}.step is only supported for linear ranges")
        step = _decimal(spec["step"], f"{label}.step")
        if not step:
            raise ConfigError(f"{label}.step must be non-zero")
        if (stop - start) * step < 0:
            raise ConfigError(f"{label}.step does not reach stop")
        values: list[Decimal] = []
        current = start
        while (current <= stop if step > 0 else current >= stop):
            values.append(current)
            current += step
        if values[-1] != stop:
            raise ConfigError(f"{label} range does not land on stop")
        return tuple(_number_value(value) for value in values)
    count = spec.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ConfigError(f"{label} requires a positive integer count or non-zero step")
    if count == 1:
        return (_number_value(start),)
    if scale == "log":
        if start <= 0 or stop <= 0:
            raise ConfigError(f"{label} logarithmic endpoints must be positive")
        start_log = math.log(float(start))
        log_increment = (math.log(float(stop)) - start_log) / (count - 1)
        values = [math.exp(start_log + log_increment * index) for index in range(count)]
        values[0], values[-1] = _number_value(start), _number_value(stop)
        return tuple(values)
    increment = (stop - start) / (count - 1)
    return tuple(_number_value(start + increment * index) for index in range(count))


def _number_value(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _axis_values(spec: object, label: str) -> tuple[Any, ...]:
    multiplier: Decimal | None = None
    targets: tuple[str, ...] = ()
    if isinstance(spec, Mapping):
        if "multiply" in spec:
            multiplier = _decimal(spec["multiply"], f"{label}.multiply")
        if "targets" in spec:
            raw_targets = spec["targets"]
            if (
                isinstance(raw_targets, (str, bytes))
                or not isinstance(raw_targets, Sequence)
                or not raw_targets
                or any(not isinstance(target, str) or not target for target in raw_targets)
                or len(set(raw_targets)) != len(raw_targets)
            ):
                raise ConfigError(
                    f"{label}.targets must contain unique non-empty paths"
                )
            targets = tuple(raw_targets)
        if "values" in spec:
            values = spec["values"]
        elif "start" in spec and "stop" in spec:
            values = _range_values(spec, label)
        else:
            raise ConfigError(f"{label} must define values or start/stop")
    else:
        values = spec
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ConfigError(f"{label}.values must be an array")
    if not values:
        raise ConfigError(f"{label}.values must not be empty")
    if multiplier is not None:
        values = tuple(
            _number_value(_decimal(value, f"{label}.value") * multiplier) for value in values
        )
    if targets:
        if any(isinstance(value, Mapping) for value in values):
            raise ConfigError(f"{label}.targets requires scalar values")
        values = tuple({target: value for target in targets} for value in values)
    return tuple(values)


def _paths_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return left[: len(right)] == right or right[: len(left)] == left


def _validate_update_paths(paths: Sequence[tuple[str, ...]]) -> None:
    if any(
        _paths_overlap(path, other)
        for index, path in enumerate(paths)
        for other in paths[:index]
    ):
        raise ConfigError("axis update paths contain a collision")


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """A shared default payload plus optional named Cartesian dimensions."""

    defaults: Mapping[str, Any]
    axes: Mapping[str, tuple[Any, ...]]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CandidateConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError("candidate configuration must be a mapping")
        data = raw
        defaults: dict[str, Any] = {}
        explicit = data.get("defaults", {})
        if explicit is not None:
            if not isinstance(explicit, Mapping):
                raise ConfigError("defaults must be a mapping")
            defaults.update(explicit)
        if set(defaults) != {"policy_params", "pool"}:
            raise ConfigError("defaults must contain exactly policy_params and pool")
        if not isinstance(defaults["policy_params"], list):
            raise ConfigError("defaults.policy_params must be an array")
        if not isinstance(defaults["pool"], Mapping):
            raise ConfigError("defaults.pool must be a mapping")

        axes: dict[str, tuple[Any, ...]] = {}
        raw_axes = data.get("axes", {})
        if isinstance(raw_axes, Mapping):
            axes = {str(name): _axis_values(spec, f"axis {name}") for name, spec in raw_axes.items()}
        elif raw_axes:
            raise ConfigError("axes must be a mapping")

        axis_paths: dict[str, list[tuple[str, ...]]] = {}
        for name, values in axes.items():
            grouped = isinstance(values[0], Mapping)
            if any(isinstance(value, Mapping) != grouped for value in values):
                raise ConfigError(f"axis {name} values must all be mappings or scalars")
            declared_paths: frozenset[tuple[str, ...]] | None = None
            for value in values:
                updates = value if grouped else {name: value}
                if not updates:
                    raise ConfigError(f"axis {name} mapping values must not be empty")
                try:
                    value_paths = [path_parts(key) for key in updates]
                    if any(
                        len(path) < 2 or path[0] not in {"policy_params", "pool"}
                        for path in value_paths
                    ):
                        raise CandidateError(
                            "axes must target policy_params.<index> or pool.<field>"
                        )
                    _validate_update_paths(value_paths)
                    merge_payload(defaults, updates)
                except CandidateError as exc:
                    raise ConfigError(str(exc)) from exc
                current_paths = frozenset(value_paths)
                if declared_paths is None:
                    declared_paths = current_paths
                elif current_paths != declared_paths:
                    raise ConfigError(
                        f"axis {name} mapping values must update the same paths"
                    )
            axis_paths[name] = sorted(declared_paths or ())
        names = tuple(axis_paths)
        for index, name in enumerate(names):
            if any(
                _paths_overlap(path, other)
                for other_name in names[:index]
                for path in axis_paths[name]
                for other in axis_paths[other_name]
            ):
                raise ConfigError("axis update paths contain a collision")

        return cls(dict(defaults), axes)

    @classmethod
    def from_toml(cls, path: str | Path) -> "CandidateConfig":
        with Path(path).open("rb") as stream:
            return cls.from_mapping(tomllib.load(stream))

    def grid(self) -> CartesianGrid:
        return CartesianGrid(dict(self.defaults), self.axes)


_RUN_KEYS = frozenset({"id", "evaluator", "template", "batch_size", "workers", "metric_fields"})
_PLACEMENT_KEYS = frozenset({"hosts", "numa_nodes"})
_CANDIDATE_KEYS = frozenset({"defaults", "axes"})
_SCENARIO_KEYS = frozenset({"id", "market", "price_feed", "yb_mode"})
_COMPILED_POLICY_KEYS = frozenset({"header", "id"})
EVALUATOR_POLICY_METADATA_KEY = "expected_evaluator_policy"
_COMPILED_POLICY_ABI = "twocrypto_policy_v1"

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
            "session_id", "template_path", "scenario_id", "market_path", "price_feed_path"
        } & set(session)
        if forbidden_session:
            raise ConfigError(
                "[session] cannot set " + ", ".join(sorted(forbidden_session))
            )
        if "arbitrage_enabled" in session:
            raise ConfigError(
                "session.arbitrage_enabled was removed; model arbitrage friction "
                "with pool.costs.arb_fee_bps"
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
        price_feed = scenario.get("price_feed")
        if price_feed is not None:
            if not isinstance(price_feed, str) or not price_feed.strip():
                raise ConfigError("scenario.price_feed must be a non-empty string")
            resolved_scenario["price_feed"] = str(
                _resolve_path(price_feed, config_path.parent)
            )
        scenario_yb_mode = scenario.get("yb_mode", "off")
        if not isinstance(scenario_yb_mode, str):
            raise ConfigError("scenario.yb_mode must be a string")
        resolved_scenario["yb_mode"] = scenario_yb_mode
        if (
            resolved_session["event_cursor"] == "exact_skip"
            and resolved_session["metric_profile"] != "grid_core"
        ):
            raise ConfigError(
                "exact_skip requires metric_profile='grid_core'"
            )
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


def _execution_inputs(config: RunConfig, *, remote: bool) -> dict[str, str]:
    inputs = {
        "evaluator": str(config.evaluator),
        "template": str(config.template),
        "market": config.scenario["market"],
    }
    if (price_feed := config.scenario.get("price_feed")) is not None:
        inputs["price_feed"] = price_feed
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



__all__ = ["CandidateConfig", "ConfigError", "RunConfig"]
