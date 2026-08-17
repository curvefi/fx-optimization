"""Registry-backed dense parameter geometry for optimization.

The checked-in :class:`PolicySpec` is the authority for policy parameter
order, defaults, bounds, and lattice quanta.  When a pair template is
supplied, the :func:`~curve_fx_sim.specs.parameters.build_parameter_registry`
additionally admits pool-economics dimensions the parameter space declares:
active pool dimensions occupy the tail of the dense optimizer vector (after
the policy ABI prefix) and are split into nested ``pool_overrides`` at
request construction (see :mod:`curve_fx_sim.optimization.requests`).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Mapping, Sequence

from ..specs.common import SpecError
from ..specs.parameters import (
    ParameterDim,
    build_parameter_registry,
)
from ..specs.policy import PolicySpec
from .lattice import LatticeSpec, TickAxis, decimal_value, quantize_lattice_float


@dataclass(frozen=True)
class PoolDim:
    """One active pool-economics dimension in the dense optimizer vector.

    ``index`` is its position in the dense vector (policy ABI count + its
    position among active pool dims, in registry order).  Values are human
    units; ``override_scale`` converts them to harness raw override units.
    """

    name: str
    target_path: tuple[str, ...]
    index: int
    default: float
    min_val: float
    max_val: float
    step: float
    transform: str = "linear"
    override_scale: Decimal = Decimal("1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_path": list(self.target_path),
            "index": self.index,
            "default": self.default,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "step": self.step,
            "transform": self.transform,
            "override_scale": format(self.override_scale, "f"),
        }


@dataclass(frozen=True)
class Profile:
    """Resolved dense search space over policy parameters and pool dims."""

    name: str
    header_file: str
    source_sha256: str
    policy_abi: str
    parameter_names: tuple[str, ...]
    initial_seed: tuple[float, ...]
    bounds: tuple[tuple[int, float, float, bool], ...]
    fixed_params: dict[int, float]
    quantize_steps: dict[int, float]
    pool_dims: tuple[PoolDim, ...] = ()

    def n_params(self) -> int:
        """Length of the dense policy ABI vector (evaluator contract)."""
        return len(self.parameter_names)

    def dense_dim(self) -> int:
        """Total dense vector length: policy ABI prefix plus active pool dims."""
        return self.n_params() + len(self.pool_dims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "header_file": self.header_file,
            "source_sha256": self.source_sha256,
            "policy_abi": self.policy_abi,
            "parameter_names": list(self.parameter_names),
            "initial_seed": list(self.initial_seed),
            "bounds": [list(bound) for bound in self.bounds],
            "fixed_params": {str(index): value for index, value in sorted(self.fixed_params.items())},
            "quantize_steps": {str(index): value for index, value in sorted(self.quantize_steps.items())},
            "pool_dims": [dim.to_dict() for dim in self.pool_dims],
        }


def _resolve_active_dim(
    *,
    name: str,
    raw: Any,
    spec: ParameterDim,
    default: float,
) -> tuple[float, float, float, bool]:
    """Resolve one active dimension's bounds, step, and log flag."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise SpecError(f"parameter_space.{name} must be a table")
    lower = float(raw.get("min", spec.min_val)) if raw.get("min", spec.min_val) is not None else None
    upper = float(raw.get("max", spec.max_val)) if raw.get("max", spec.max_val) is not None else None
    quantum = float(raw.get("step", spec.step)) if raw.get("step", spec.step) is not None else None
    if lower is None or upper is None or quantum is None:
        raise SpecError(
            f"parameter {name!r} requires min, max, and step in the registry or parameter_space"
        )
    if lower > upper:
        raise SpecError(f"parameter {name!r} has min {lower} > max {upper}")
    if quantum <= 0:
        raise SpecError(f"parameter {name!r} has non-positive step {quantum}")
    if spec.min_val is not None and lower < float(spec.min_val):
        raise SpecError(f"parameter_space.{name}.min widens the registry bound")
    if spec.max_val is not None and upper > float(spec.max_val):
        raise SpecError(f"parameter_space.{name}.max widens the registry bound")
    if not lower <= default <= upper:
        raise SpecError(
            f"default {default} for {name!r} lies outside optimization bounds [{lower}, {upper}]"
        )
    transform = str(raw.get("transform", spec.transform)).lower()
    if transform not in {"linear", "log"}:
        raise SpecError(f"parameter_space.{name} transform must be 'linear' or 'log'")
    return lower, upper, quantum, transform == "log"


def profile_from_policy_spec(
    policy: PolicySpec,
    parameter_space: Mapping[str, Any] | None = None,
    *,
    template_json: Mapping[str, Any] | None = None,
) -> Profile:
    """Resolve one exact dense lattice from a compiled policy spec.

    Without ``template_json`` (the pair template, e.g. the scenario's
    ``template_path`` JSON), the profile is strictly policy-only: every
    declared policy parameter is searched by default and omitted parameters
    are fixed at their policy defaults; pool names are rejected.  With a
    template, the parameter registry admits pool-economics dimensions too;
    active pool dimensions (only those explicitly named) are appended to the
    dense vector in registry order and split into pool overrides at request
    construction.
    """
    if policy.policy_kind != "compiled":
        raise SpecError(f"optimization requires a compiled policy, got {policy.policy_kind!r}")
    if not policy.source_sha256:
        raise SpecError(f"policy {policy.id!r} must attest source_sha256")
    if not policy.parameters:
        raise SpecError(f"policy {policy.id!r} declares no dense parameters")
    declared_names = tuple(parameter.name for parameter in policy.parameters)

    configured = dict(parameter_space or {})
    registry = build_parameter_registry(policy, template_json, configured)
    search_all = not configured

    seed: list[float] = []
    bounds: list[tuple[int, float, float, bool]] = []
    fixed: dict[int, float] = {}
    steps: dict[int, float] = {}
    for index, parameter in enumerate(policy.parameters):
        if parameter.type not in {"float", "int"}:
            raise SpecError(
                f"compiled policy parameter {parameter.name!r} must be numeric, got {parameter.type!r}"
            )
        if parameter.default is None:
            raise SpecError(f"compiled policy parameter {parameter.name!r} has no default")
        default = float(parameter.validate_value(parameter.default))
        seed.append(default)

        raw = configured.get(parameter.name)
        active = search_all or raw is not None
        if not active:
            fixed[index] = default
            continue
        spec = registry[parameter.name]
        lower, upper, quantum, is_log = _resolve_active_dim(
            name=parameter.name,
            raw=raw,
            spec=spec,
            default=default,
        )
        steps[index] = quantum
        bounds.append((index, lower, upper, is_log))

    pool_dims: list[PoolDim] = []
    for spec in (item for item in registry.values() if item.kind == "pool"):
        # Pool dims exist in the registry only where parameter_space declares
        # them, so every entry is an active search dimension.
        default = float(spec.default)
        seed.append(default)
        index = len(seed) - 1  # dense position after the policy ABI prefix
        lower, upper, quantum, is_log = _resolve_active_dim(
            name=spec.name,
            raw=None,
            spec=spec,
            default=default,
        )
        steps[index] = quantum
        bounds.append((index, lower, upper, is_log))
        pool_dims.append(
            PoolDim(
                name=spec.name,
                target_path=spec.target_path,
                index=index,
                default=default,
                min_val=lower,
                max_val=upper,
                step=quantum,
                transform=spec.transform,
                override_scale=spec.override_scale,
            )
        )

    if not bounds and not pool_dims:
        raise SpecError("optimization parameter_space freezes every policy parameter")
    return Profile(
        name=policy.id,
        header_file=str(policy.header_file),
        source_sha256=policy.source_sha256,
        policy_abi=policy.policy_abi,
        parameter_names=declared_names,
        initial_seed=tuple(seed),
        bounds=tuple(bounds),
        fixed_params=fixed,
        quantize_steps=steps,
        pool_dims=tuple(pool_dims),
    )


def quantized(profile: Profile, params: Sequence[float]) -> list[float]:
    """Snap one complete dense vector to the exact declared decimal lattice."""
    expected = profile.dense_dim()
    if len(params) != expected:
        if profile.pool_dims:
            raise ValueError(
                f"profile {profile.name!r} requires {expected} parameters "
                f"(policy {profile.n_params()} + pool {len(profile.pool_dims)}), got {len(params)}"
            )
        raise ValueError(
            f"policy {profile.name!r} requires {profile.n_params()} parameters, got {len(params)}"
        )
    out = [float(value) for value in params]
    for index, lower, upper, _is_log in profile.bounds:
        out[index] = quantize_lattice_float(
            out[index], profile.quantize_steps[index], lower=lower, upper=upper
        )
    for index, value in profile.fixed_params.items():
        out[index] = float(value)
    return out




def create_lattice_spec(profile: Profile) -> LatticeSpec:
    """Construct exact integer axes in dense declaration order (policy then pool)."""
    name_by_index: dict[int, str] = {
        index: name for index, name in enumerate(profile.parameter_names)
    }
    for dim in profile.pool_dims:
        name_by_index[dim.index] = dim.name
    axes = []
    for index, lower, upper, is_log in profile.bounds:
        quantum = decimal_value(profile.quantize_steps[index])
        min_tick = int(
            (decimal_value(lower) / quantum).to_integral_value(rounding=ROUND_CEILING)
        )
        max_tick = int(
            (decimal_value(upper) / quantum).to_integral_value(rounding=ROUND_FLOOR)
        )
        if min_tick > max_tick:
            raise SpecError(
                f"parameter {name_by_index.get(index, index)!r} bounds contain no exact lattice point"
            )
        axes.append(
            TickAxis(
                index=index,
                name=name_by_index.get(index, f"dim_{index}"),
                quantum=quantum,
                min_tick=min_tick,
                max_tick=max_tick,
                is_log=is_log,
            )
        )
    return LatticeSpec(
        profile_name=profile.name,
        axes=tuple(axes),
        fixed_params=dict(profile.fixed_params),
        n_params=profile.dense_dim(),
    )


__all__ = [
    "Profile",
    "PoolDim",
    "profile_from_policy_spec",
    "quantized",
    "create_lattice_spec",
]
