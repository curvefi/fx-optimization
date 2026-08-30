"""TOML-friendly Cartesian-grid configuration."""

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


__all__ = ["CandidateConfig", "ConfigError"]
