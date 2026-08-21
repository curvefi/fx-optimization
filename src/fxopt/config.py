"""TOML-friendly configuration and compilation entry points."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import tomllib
from typing import Any

from .candidates import (
    CandidateError,
    CandidateSpec,
    compile_batch,
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
    if "step" in spec:
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
    increment = (stop - start) / (count - 1)
    return tuple(_number_value(start + increment * index) for index in range(count))


def _number_value(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _axis_values(spec: object, label: str) -> tuple[Any, ...]:
    multiplier: Decimal | None = None
    if isinstance(spec, Mapping):
        if "multiply" in spec:
            multiplier = _decimal(spec["multiply"], f"{label}.multiply")
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
    return tuple(values)


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

        axes: dict[str, tuple[Any, ...]] = {}
        raw_axes = data.get("axes", {})
        if isinstance(raw_axes, Mapping):
            axes = {str(name): _axis_values(spec, f"axis {name}") for name, spec in raw_axes.items()}
        elif raw_axes:
            raise ConfigError("axes must be a mapping")

        try:
            paths = [path_parts(name) for name in axes]
        except CandidateError as exc:
            raise ConfigError(str(exc)) from exc
        for index, path in enumerate(paths):
            if any(path[:size] == other for other in paths[:index] for size in range(1, len(path) + 1)):
                raise ConfigError("axis paths contain a collision")
            if any(other[:size] == path for other in paths[:index] for size in range(1, len(other) + 1)):
                raise ConfigError("axis paths contain a collision")
            if ".".join(path) in defaults:
                raise ConfigError("axis path collides with a flat default")
            cursor: object = defaults
            for part in path[:-1]:
                if not isinstance(cursor, Mapping) or part not in cursor:
                    break
                cursor = cursor[part]
                if not isinstance(cursor, Mapping):
                    raise ConfigError("axis path collides with a scalar default")

        return cls(dict(defaults), axes)

    @classmethod
    def from_toml(cls, path: str | Path) -> "CandidateConfig":
        with Path(path).open("rb") as stream:
            return cls.from_mapping(tomllib.load(stream))

    def point(self, overrides: Mapping[str, Any] | None = None) -> CandidateSpec:
        return CandidateSpec.from_payload(merge_payload(self.defaults, overrides or {}))

    def grid(self) -> CartesianGrid:
        return CartesianGrid(dict(self.defaults), self.axes)

    def batch(self, proposals: Iterable[Mapping[str, Any]]) -> tuple[CandidateSpec, ...]:
        return compile_batch(proposals, defaults=self.defaults)


def load_config(source: Mapping[str, Any] | str | Path) -> CandidateConfig:
    return (
        CandidateConfig.from_mapping(source)
        if isinstance(source, Mapping)
        else CandidateConfig.from_toml(source)
    )


def compile_candidates(
    source: CandidateConfig | Mapping[str, Any] | str | Path,
    mode: str = "point",
    *,
    overrides: Mapping[str, Any] | None = None,
    proposals: Iterable[Mapping[str, Any]] | None = None,
) -> CandidateSpec | CartesianGrid | tuple[CandidateSpec, ...]:
    config = source if isinstance(source, CandidateConfig) else load_config(source)
    if mode == "point":
        return config.point(overrides)
    if mode == "grid":
        return config.grid()
    if mode == "batch":
        if proposals is None:
            raise ConfigError("batch compilation requires proposals")
        return config.batch(proposals)
    raise ConfigError(f"unknown compilation mode: {mode}")


__all__ = ["CandidateConfig", "ConfigError", "compile_candidates", "load_config"]
