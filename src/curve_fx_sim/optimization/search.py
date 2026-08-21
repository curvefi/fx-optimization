"""Schema-driven named search geometry for numeric candidate proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import math
from typing import Any

from ..evaluation.plans import CandidateSchema, ParameterDescriptor
from ..specs.common import canonical_json_bytes
from .lattice import LatticeSpec, TickAxis

_FIELDS = frozenset({"min", "max", "step", "transform", "values"})
_MISSING = object()

class SearchLayoutError(ValueError):
    """A schema descriptor cannot form an exact numeric optimizer axis."""


def select_search_descriptors(
    schema: CandidateSchema, parameter_space: Mapping[str, Any] | None
) -> tuple[ParameterDescriptor, ...]:
    """Select the schema-ordered, numeric descriptors admitted to a search."""
    space = {} if parameter_space is None else parameter_space
    if not isinstance(space, Mapping) or any(not isinstance(name, str) for name in space):
        raise SearchLayoutError("parameter_space must have canonical string keys")
    known = {descriptor.name for descriptor in schema.descriptors}
    unknown = sorted(set(space) - known)
    if unknown:
        raise SearchLayoutError("unknown canonical names: " + ", ".join(unknown))
    active = set(space) or {
        descriptor.name for descriptor in schema.descriptors if descriptor.name.startswith("policy.")
    }
    selected = tuple(descriptor for descriptor in schema.descriptors if descriptor.name in active)
    if not selected:
        raise SearchLayoutError("parameter_space defines no numeric search dimensions")
    for descriptor in selected:
        if descriptor.classification == "observation":
            raise SearchLayoutError(f"observation parameter cannot be searched: {descriptor.name}")
        if descriptor.unit == "legacy_alias":
            raise SearchLayoutError(f"legacy alias cannot be searched: {descriptor.name}")
        if descriptor.unit in {"path", "sha256"} or descriptor.name == "run.session_id":
            raise SearchLayoutError(
                f"unsupported named optimization dimensions: {descriptor.name}"
            )
        if descriptor.value_type not in {"real", "integer"}:
            raise SearchLayoutError(
                f"unsupported optimizer type {descriptor.value_type}: {descriptor.name}"
            )
        if descriptor.classification != "candidate" and not (
            descriptor.classification == "session"
            and descriptor.lowering_path.startswith("open_session.")
        ):
            raise SearchLayoutError(f"invalid canonical search descriptor: {descriptor.name}")
    return selected


@dataclass(frozen=True, slots=True)
class SearchDimension:
    """One canonical, human-unit dimension in a dense optimizer vector."""

    name: str
    default: Decimal
    minimum: Decimal
    maximum: Decimal
    step: Decimal
    transform: str
    index: int


@dataclass(frozen=True, slots=True)
class SearchLayout:
    """Ordered optimizer geometry derived from a verified candidate schema."""

    schema_sha256: str
    dimensions: tuple[SearchDimension, ...]

    @classmethod
    def from_schema(
        cls,
        schema: CandidateSchema,
        parameter_space: Mapping[str, Any] | None,
        template_json: Mapping[str, Any] | None,
        open_session: Mapping[str, Any] | None,
    ) -> SearchLayout:
        space = {} if parameter_space is None else parameter_space
        selected = select_search_descriptors(schema, space)
        dimensions = tuple(
            _dimension(descriptor, space.get(descriptor.name, {}), index, template_json, open_session)
            for index, descriptor in enumerate(selected)
        )
        return cls(schema.sha256, dimensions)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_sha256": self.schema_sha256,
            "dimensions": [
                {
                    "name": item.name,
                    "default": str(item.default),
                    "minimum": str(item.minimum),
                    "maximum": str(item.maximum),
                    "step": str(item.step),
                    "transform": item.transform,
                    "index": item.index,
                }
                for item in self.dimensions
            ],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @property
    def default_vector(self) -> tuple[int | float, ...]:
        return tuple(_scalar(item.default) for item in self.dimensions)

    def to_proposal(self, vector: Sequence[int | float | Decimal]) -> dict[str, int | float]:
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise SearchLayoutError("optimizer vector must be a numeric sequence")
        if len(vector) != len(self.dimensions):
            raise SearchLayoutError(
                f"optimizer vector length {len(vector)} != layout dimension {len(self.dimensions)}"
            )
        proposal: dict[str, int | float] = {}
        for item, raw in zip(self.dimensions, vector, strict=True):
            value = _number(raw, f"vector[{item.index}]")
            if not item.minimum <= value <= item.maximum:
                raise SearchLayoutError(f"{item.name} value {value} is outside its bounds")
            if value % item.step:
                raise SearchLayoutError(f"{item.name} value {value} is off the exact step {item.step}")
            proposal[item.name] = _scalar(value)
        return proposal

    def create_lattice_spec(self) -> LatticeSpec:
        axes = tuple(
            TickAxis(
                index=item.index,
                name=item.name,
                quantum=item.step,
                min_tick=int(item.minimum / item.step),
                max_tick=int(item.maximum / item.step),
                is_log=item.transform == "log",
            )
            for item in self.dimensions
        )
        return LatticeSpec(profile_name=self.sha256, axes=axes, n_params=len(axes))


def build_search_layout(
    schema: CandidateSchema,
    parameter_space: Mapping[str, Any] | None,
    template_json: Mapping[str, Any] | None,
    open_session: Mapping[str, Any] | None,
) -> SearchLayout:
    return SearchLayout.from_schema(schema, parameter_space, template_json, open_session)


def _number(value: object, label: str, *, strings: bool = False) -> Decimal:
    allowed = (int, float, Decimal, str) if strings else (int, float, Decimal)
    if isinstance(value, bool) or not isinstance(value, allowed):
        raise SearchLayoutError(f"{label} must be a real number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise SearchLayoutError(f"{label} must be finite") from exc
    if not result.is_finite():
        raise SearchLayoutError(f"{label} must be finite")
    return result


def _scalar(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    result = float(value)
    if not math.isfinite(result):
        raise SearchLayoutError(f"value {value} is outside binary64")
    return result


def _values(name: str, raw: object) -> tuple[Decimal, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise SearchLayoutError(f"parameter_space.{name}.values must be a sequence")
    values = tuple(_number(value, f"parameter_space.{name}.values") for value in raw)
    if len(values) < 2:
        raise SearchLayoutError(f"parameter_space.{name}.values needs at least two entries")
    step = values[1] - values[0]
    if step <= 0:
        raise SearchLayoutError(f"parameter_space.{name}.values must increase")
    if any(b - a != step for a, b in zip(values, values[1:])):
        raise SearchLayoutError(f"parameter_space.{name}.values must be uniformly spaced")
    return values


def _geometry(
    name: str,
    raw: Any,
    fallback: tuple[Decimal, Decimal, Decimal] | None,
) -> tuple[Decimal, Decimal, Decimal, str, bool]:
    if not isinstance(raw, Mapping):
        values = _values(name, raw)
        return values[0], values[-1], values[1] - values[0], "linear", True
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise SearchLayoutError(f"parameter_space.{name} has unsupported fields: {', '.join(unknown)}")
    transform = raw.get("transform", "linear")
    if not isinstance(transform, str) or transform not in {"linear", "log"}:
        raise SearchLayoutError(f"parameter_space.{name}.transform must be 'linear' or 'log'")
    if "values" in raw:
        if any(field in raw for field in ("min", "max", "step")):
            raise SearchLayoutError(f"parameter_space.{name}.values conflicts with min/max/step")
        values = _values(name, raw["values"])
        return values[0], values[-1], values[1] - values[0], transform, True
    if fallback is None and any(field not in raw for field in ("min", "max", "step")):
        raise SearchLayoutError(f"parameter_space.{name} requires min, max, and step")
    base = fallback or (Decimal(0), Decimal(0), Decimal(0))
    return (
        _number(raw.get("min", base[0]), f"parameter_space.{name}.min"),
        _number(raw.get("max", base[1]), f"parameter_space.{name}.max"),
        _number(raw.get("step", base[2]), f"parameter_space.{name}.step"),
        transform,
        False,
    )


def _default(descriptor: ParameterDescriptor) -> Decimal | object:
    return (
        _number(descriptor.default, f"{descriptor.name} default")
        if descriptor.has_default
        else _MISSING
    )


def _pool_default(
    descriptor: ParameterDescriptor, template: Mapping[str, Any] | None
) -> Decimal | object:
    if template is None:
        return _default(descriptor)
    if not isinstance(template, Mapping):
        raise SearchLayoutError("template_json must be a mapping")
    if "pools" in template:
        pools = template["pools"]
        if not isinstance(pools, list) or len(pools) != 1 or not isinstance(pools[0], Mapping):
            raise SearchLayoutError("pool search requires exactly one pool")
        entry = pools[0]
    else:
        entry = template
    wrapped = entry.get("pool")
    pool = wrapped if isinstance(wrapped, Mapping) else entry
    costs = entry.get("costs", {})
    if not isinstance(pool, Mapping) or not isinstance(costs, Mapping):
        raise SearchLayoutError("template pool and costs must be objects")
    path = descriptor.lowering_path.split(".")
    if len(path) < 3 or path[:2] not in (["pool_overrides", "pool"], ["pool_overrides", "costs"]):
        raise SearchLayoutError(f"{descriptor.name} has an unsupported pool lowering path")
    value: object = pool if path[1] == "pool" else costs
    for component in path[2:]:
        if not isinstance(value, Mapping) or component not in value:
            return _default(descriptor)
        value = value[component]
    binary64 = float(_number(value, f"template default for {descriptor.name}", strings=True))
    if not math.isfinite(binary64):
        raise SearchLayoutError(f"template default for {descriptor.name} is outside binary64")
    materialized = Decimal(str(binary64))
    if descriptor.wire_representation == "binary64_fraction_or_1e10" and materialized > 1:
        return materialized / Decimal("1e10")
    if descriptor.wire_representation == "binary64_from_wad_1e18":
        return materialized / Decimal("1e18")
    return materialized


def _run_default(
    descriptor: ParameterDescriptor, open_session: Mapping[str, Any] | None
) -> Decimal | object:
    if open_session is not None and not isinstance(open_session, Mapping):
        raise SearchLayoutError("open_session must be a mapping")
    prefix = "open_session."
    if not descriptor.lowering_path.startswith(prefix):
        raise SearchLayoutError(f"{descriptor.name} has an invalid session path")
    field = descriptor.lowering_path.removeprefix(prefix)
    if open_session is not None and field in open_session:
        return _number(open_session[field], f"open_session.{field}")
    return _default(descriptor)


def _dimension(
    descriptor: ParameterDescriptor,
    raw: Any,
    index: int,
    template: Mapping[str, Any] | None,
    session: Mapping[str, Any] | None,
) -> SearchDimension:
    name = descriptor.name
    outer = None
    if name.startswith("policy."):
        outer = tuple(
            _number(value, f"{name} domain")
            for value in (descriptor.minimum, descriptor.maximum, descriptor.quantum)
        )
        default = _default(descriptor)
    elif name.startswith("pool.") and descriptor.classification == "candidate":
        default = _pool_default(descriptor, template)
    elif name.startswith("run.") and descriptor.classification == "session":
        default = _run_default(descriptor, session)
    else:
        raise SearchLayoutError(f"invalid canonical search descriptor: {name}")
    minimum, maximum, step, transform, explicit = _geometry(name, raw, outer)
    if default is _MISSING:
        if name.startswith("pool.") and explicit:
            default = minimum
        else:
            raise SearchLayoutError(f"parameter {name} has no base default")
    assert isinstance(default, Decimal)
    _validate(descriptor, default, minimum, maximum, step, outer)
    if transform == "log" and (minimum <= 0 or default <= 0):
        raise SearchLayoutError(f"log search dimension {name} requires positive bounds/default")
    return SearchDimension(name, default, minimum, maximum, step, transform, index)


def _validate(
    descriptor: ParameterDescriptor,
    default: Decimal,
    minimum: Decimal,
    maximum: Decimal,
    step: Decimal,
    outer: tuple[Decimal, Decimal, Decimal] | None,
) -> None:
    name = descriptor.name
    if minimum > maximum:
        raise SearchLayoutError(f"parameter_space.{name}.min exceeds max")
    if step <= 0:
        raise SearchLayoutError(f"parameter_space.{name}.step must be positive")
    if not minimum <= default <= maximum:
        raise SearchLayoutError(f"base default {default} for {name} lies outside search bounds")
    quantum, origin = None, Decimal(0)
    if outer is not None:
        origin, outer_maximum, quantum = outer
        if quantum <= 0:
            raise SearchLayoutError(f"schema quantum for {name} must be positive")
        if minimum < origin or maximum > outer_maximum:
            raise SearchLayoutError(f"parameter_space.{name} widens the schema domain")
        if step % quantum:
            raise SearchLayoutError(f"parameter_space.{name}.step is not an integer multiple")
    elif descriptor.value_type == "integer":
        quantum = Decimal(1)
    elif descriptor.wire_representation == "binary64_fraction_or_1e10":
        quantum = Decimal("1e-10")
    elif descriptor.wire_representation == "binary64_from_wad_1e18":
        quantum = Decimal("1e-18")
    if quantum is not None:
        if any((value - origin) % quantum for value in (minimum, maximum, default)):
            raise SearchLayoutError(f"parameter_space.{name} is off descriptor lattice {quantum}")
        if step % quantum:
            raise SearchLayoutError(f"parameter_space.{name}.step is off numeric lattice {quantum}")
    # TickAxis currently decodes tick * step and has no affine origin.
    if any(value % step for value in (minimum, maximum, default)):
        raise SearchLayoutError(f"parameter_space.{name} cannot use zero-origin TickAxis step {step}")


__all__ = [
    "SearchDimension",
    "SearchLayout",
    "SearchLayoutError",
    "build_search_layout",
    "select_search_descriptors",
]
