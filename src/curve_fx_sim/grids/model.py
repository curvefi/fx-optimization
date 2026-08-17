"""Finite grid expansion and canonical evaluator request construction."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence
from ..specs.grid import AxisSpec, GridSpec
from ..specs.policy import PolicySpec
from ..specs.parameters import ParameterDim



class GridValidationError(ValueError):
    """Raised when a grid does not define one exact finite Cartesian product."""


def _exact_value(value: Any) -> Any:
    """Return a stable, JSON-safe coordinate value without float coercion."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else format(value, "f")
    if isinstance(value, Mapping):
        return {str(key): _exact_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_exact_value(item) for item in value]
    return value


def coordinate_signature(coordinate: Mapping[str, Any]) -> str:
    """Canonical equality key used for exact coordinate lookup.

    Numeric spellings such as ``1``, ``1.0`` and ``"1.00"`` intentionally
    compare equal.  No tolerance or nearest-coordinate behavior is used.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return ["value", value]
        if isinstance(value, (int, float, Decimal, str)):
            try:
                number = Decimal(str(value))
                if number.is_finite():
                    normalized = "0" if number == 0 else str(number.normalize())
                    return ["decimal", normalized]
            except (InvalidOperation, ValueError):
                pass
        if isinstance(value, Mapping):
            return [
                "mapping",
                [[str(key), normalize(item)] for key, item in sorted(value.items())],
            ]
        if isinstance(value, (tuple, list)):
            return ["sequence", [normalize(item) for item in value]]
        return ["value", str(value)]

    return json.dumps(
        [[str(key), normalize(value)] for key, value in sorted(coordinate.items())],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _protocol_number(value: Any, *, label: str) -> int | float:
    if isinstance(value, bool):
        raise GridValidationError(f"{label} must be numeric, not boolean")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise GridValidationError(f"{label} must be numeric, got {value!r}") from exc
    if not number.is_finite():
        raise GridValidationError(f"{label} must be finite, got {value!r}")
    return int(number) if number == number.to_integral_value() else float(number)


def _policy_vector(
    raw: Any,
    *,
    parameter_names: Sequence[str] | None = None,
) -> tuple[int | float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        if not raw:
            return ()
        if parameter_names is not None:
            names = tuple(parameter_names)
            unknown = sorted(set(str(key) for key in raw) - set(names))
            missing = [name for name in names if name not in raw]
            if unknown or missing:
                raise GridValidationError(
                    "policy_params must exactly match PolicySpec names; "
                    f"unknown={unknown}, missing={missing}"
                )
            raw = [raw[name] for name in names]
        else:
            try:
                indexed = {int(key): value for key, value in raw.items()}
            except (TypeError, ValueError) as exc:
                raise GridValidationError(
                    "policy_params must be an array or a dense mapping keyed by integer indices"
                ) from exc
            if set(indexed) != set(range(len(indexed))):
                raise GridValidationError("policy_params indices must form the dense range [0, n)")
            raw = [indexed[index] for index in range(len(indexed))]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise GridValidationError("policy_params grid override must be an array")
    return tuple(
        _protocol_number(value, label=f"policy_params[{index}]")
        for index, value in enumerate(raw)
    )

def _detach(value: Any) -> Any:
    """Detach mutable containers without invoking recursive ``deepcopy``."""
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, set):
        return {_detach(item) for item in value}
    return value


def _split_request_overrides(
    overrides: Mapping[str, Any],
    *,
    parameter_names: Sequence[str] | None = None,
) -> tuple[tuple[int | float, ...], dict[str, Any]]:
    """Split protocol policy-vector parameters from pool overrides."""
    raw_policy = overrides.get("policy_params", ())
    raw_pool = overrides.get("pool_overrides", {})
    if raw_pool is None:
        raw_pool = {}
    if not isinstance(raw_pool, Mapping):
        raise GridValidationError("pool_overrides grid override must be a mapping")
    pool = dict(raw_pool)
    for key, value in overrides.items():
        if key in {"policy_params", "pool_overrides"}:
            continue
        if key in pool:
            raise GridValidationError(f"duplicate pool override namespace {key!r}")
        pool[key] = value
    return _policy_vector(raw_policy, parameter_names=parameter_names), _detach(pool)


@dataclass(frozen=True)
class GridPoint:
    """One canonical point in declaration-order Cartesian expansion."""

    ordinal: int
    candidate_id: str
    coordinate_indices: tuple[int, ...]
    coordinates: Mapping[str, Any]
    policy_params: tuple[int | float, ...]
    pool_overrides: Mapping[str, Any]
    coordinate_signature: str = ""


    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "candidate_id": self.candidate_id,
            "coordinate_indices": list(self.coordinate_indices),
            "coordinates": _exact_value(self.coordinates),
            "coordinate_signature": self.coordinate_signature,
            "policy_params": _exact_value(self.policy_params),
            "pool_overrides": _exact_value(self.pool_overrides),
        }


def _set_nested(target: dict[str, Any], path: Sequence[str], value: Any) -> None:
    if not path:
        return
    current = target
    for part in path[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise GridValidationError(
                f"grid target {'.'.join(path)!r} crosses scalar namespace {part!r}"
            )
        current = existing
    current[path[-1]] = value


def _axis_options(axis: AxisSpec) -> tuple[dict[str, Any], ...]:
    options: list[dict[str, Any]] = []
    if axis.is_coupled:
        if axis.targets and len(axis.targets) != len(axis.names):
            raise GridValidationError(
                f"coupled axis {axis.name!r} must declare one target per coordinate name"
            )
        for index, row in enumerate(axis.rows):
            display = {
                name: _exact_value(value)
                for name, value in zip(axis.names, row, strict=True)
            }
            overrides: dict[str, Any] = {}
            for column, value in enumerate(row):
                if column >= len(axis.targets):
                    continue
                target = axis.targets[column]
                transformed = target.transform_value(value)
                if target.kind in {"decimal", "integer", "bps"}:
                    transformed = _protocol_number(
                        transformed,
                        label=f"axis {axis.name} target {'.'.join(target.path)}",
                    )
                _set_nested(overrides, target.path, transformed)
            options.append({"index": index, "display": display, "overrides": overrides})
    else:
        for index, value in enumerate(axis.values):
            overrides = {}
            for target in axis.targets:
                transformed = target.transform_value(value)
                if target.kind in {"decimal", "integer", "bps"}:
                    transformed = _protocol_number(
                        transformed,
                        label=f"axis {axis.name} target {'.'.join(target.path)}",
                    )
                _set_nested(overrides, target.path, transformed)
            options.append(
                {
                    "index": index,
                    "display": {axis.name: _exact_value(value)},
                    "overrides": overrides,
                }
            )
    return tuple(options)


def _merge_axis_overrides(
    base: Mapping[str, Any],
    axis_updates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge axis writes with copy-on-write nested branches.

    The static base is treated as immutable.  Each point receives a fresh
    top-level mapping, while only mappings on paths written by an axis are
    copied.  This keeps unrelated static branches shared without allowing an
    axis write to mutate the base or a sibling point.
    """
    merged = dict(base)
    copied_paths: set[tuple[str, ...]] = set()
    written_paths: set[tuple[str, ...]] = set()

    def merge(target: dict[str, Any], updates: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for key, value in updates.items():
            path = (*prefix, str(key))
            if isinstance(value, Mapping):
                existing = target.get(key)
                if existing is None:
                    existing = {}
                elif not isinstance(existing, Mapping):
                    raise GridValidationError(
                        f"grid override {'.'.join(path)!r} crosses a scalar value"
                    )
                if path not in copied_paths:
                    existing = dict(existing)
                    copied_paths.add(path)
                    target[key] = existing
                merge(existing, value, path)
            else:
                if path in written_paths:
                    raise GridValidationError(
                        f"multiple grid axes write evaluator override {'.'.join(path)!r}"
                    )
                written_paths.add(path)
                target[key] = value

    for updates in axis_updates:
        merge(merged, updates, ())
    return merged


def expand_grid(
    grid_spec: GridSpec,
    *,
    policy_spec: PolicySpec | None = None,
    registry: Mapping[str, ParameterDim] | None = None,
) -> tuple[GridPoint, ...]:
    """Compile one exact declaration-order Cartesian product into protocol requests.

    When a compiled policy is supplied, its named defaults are the sole source
    of the dense request vector. Grid axes may overwrite declared policy
    names; the resulting request is emitted in PolicySpec declaration order.

    With a ``registry`` (``build_parameter_registry`` output), axis targets
    may additionally name registered pool parameters (their override paths are
    validated against the registry) alongside policy targets; unknown paths
    remain allowed.  Without a registry the compiled-policy grid contract is
    strictly policy_params-only, exactly as before.
    """
    parameter_names: tuple[str, ...] | None = None
    base_overrides = dict(grid_spec.static_overrides)
    if policy_spec is not None:
        parameter_names = tuple(parameter.name for parameter in policy_spec.parameters)
        if not parameter_names:
            raise GridValidationError(f"policy {policy_spec.id!r} declares no parameters")
        unexpected_static = sorted(set(base_overrides) - {"policy_params"})
        if unexpected_static:
            raise GridValidationError(
                "compiled-policy grids accept only policy_params; pool overrides are out of scope: "
                + ", ".join(unexpected_static)
            )
        configured = base_overrides.get("policy_params", {})
        if not isinstance(configured, Mapping):
            raise GridValidationError("static_overrides.policy_params must be a named mapping")
        unknown = sorted(set(str(key) for key in configured) - set(parameter_names))
        if unknown:
            raise GridValidationError(
                "static policy_params are not declared by PolicySpec: " + ", ".join(unknown)
            )
        defaults: dict[str, Any] = {}
        for parameter in policy_spec.parameters:
            if parameter.default is None:
                raise GridValidationError(
                    f"policy parameter {parameter.name!r} has no default for grid expansion"
                )
            defaults[parameter.name] = parameter.validate_value(parameter.default)
        defaults.update(dict(configured))
        base_overrides = {"policy_params": defaults}

    coordinate_names: list[str] = []
    target_paths: list[tuple[str, ...]] = []
    axis_options: list[tuple[dict[str, Any], ...]] = []
    for axis in grid_spec.axes:
        names = list(axis.names) if axis.is_coupled else [axis.name]
        for name in names:
            if name in coordinate_names:
                raise GridValidationError(f"grid coordinate name {name!r} is declared twice")
            coordinate_names.append(name)
        for target in axis.targets:
            if policy_spec is not None:
                if registry is not None:
                    if target.path and target.path[0] == "policy_params":
                        if (
                            len(target.path) != 2
                            or target.path[1] not in registry
                            or registry[target.path[1]].kind != "policy"
                        ):
                            raise GridValidationError(
                                f"compiled-policy grid target {'.'.join(target.path)!r} must be "
                                "policy_params.<registered policy parameter>"
                            )
                elif (
                    len(target.path) != 2
                    or target.path[0] != "policy_params"
                    or target.path[1] not in parameter_names
                ):
                    raise GridValidationError(
                        f"compiled-policy grid target {'.'.join(target.path)!r} must be "
                        "policy_params.<PolicySpec parameter name>"
                    )
            if target.path and target.path in target_paths:
                raise GridValidationError(
                    f"evaluator target {'.'.join(target.path)!r} is controlled by multiple axes"
                )
            target_paths.append(target.path)
        # Registered coordinate names must write exactly their canonical
        # registry path; names absent from the registry stay free-form.
        if registry is not None:
            for index, name in enumerate(names):
                spec = registry.get(name)
                if spec is None:
                    continue
                expected = (
                    ("policy_params", name)
                    if spec.kind == "policy"
                    else spec.target_path
                )
                target = (
                    axis.targets[index]
                    if axis.is_coupled and axis.targets and index < len(axis.targets)
                    else axis.targets[0] if axis.targets else None
                )
                if target is not None and target.path != expected:
                    raise GridValidationError(
                        f"grid coordinate {name!r} must target "
                        f"{'.'.join(expected)}, got {'.'.join(target.path)}"
                    )
        options = _axis_options(axis)
        if not options:
            raise GridValidationError(f"axis {axis.name!r} has no finite values")
        axis_options.append(options)

    points: list[GridPoint] = []
    coordinate_keys: set[str] = set()
    for ordinal, combination in enumerate(itertools.product(*axis_options)):
        coordinates: dict[str, Any] = {}
        indices: list[int] = []
        updates: list[Mapping[str, Any]] = []
        for option in combination:
            coordinates.update(option["display"])
            indices.append(int(option["index"]))
            updates.append(option["overrides"])
        signature = coordinate_signature(coordinates)
        if signature in coordinate_keys:
            raise GridValidationError(f"duplicate exact display coordinate at ordinal {ordinal}")
        coordinate_keys.add(signature)
        policy_params, pool_overrides = _split_request_overrides(
            _merge_axis_overrides(base_overrides, updates),
            parameter_names=parameter_names,
        )
        if grid_spec.fee_equalize and "mid_fee" in pool_overrides:
            # Legacy fee_equalize semantics: out_fee mirrors mid_fee for
            # every cell (generate_pools_nd.py: out_fee = mid if fee_equalize).
            if "out_fee" in pool_overrides and pool_overrides["out_fee"] != pool_overrides["mid_fee"]:
                raise GridValidationError(
                    "fee_equalize grid may not override out_fee independently of mid_fee"
                )
            pool_overrides = dict(pool_overrides)
            pool_overrides["out_fee"] = pool_overrides["mid_fee"]
        points.append(
            GridPoint(
                ordinal=ordinal,
                candidate_id=f"grid_{grid_spec.id}_p{ordinal:06d}",
                coordinate_indices=tuple(indices),
                coordinates=_detach(coordinates),
                policy_params=policy_params,
                pool_overrides=pool_overrides,
                coordinate_signature=signature,
            )
        )
    if len(points) != grid_spec.pool_count:
        raise GridValidationError(
            f"expanded {len(points)} points, expected {grid_spec.pool_count}"
        )
    return tuple(points)
