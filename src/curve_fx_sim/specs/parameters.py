"""Thin name-to-dimension resolver for optimization parameter spaces.

The registry does not duplicate spec data.  Policy dimensions come verbatim
from ``PolicySpec.parameters`` (name, ABI index, default, bounds, step);
PolicySpec remains the sole authority for policy dims.  Pool dimensions
exist only where the optimization spec's ``parameter_space`` declares them
with explicit bounds/step/transform (or an explicit values list), and their
defaults are read from the pair template JSON so the seed matches the
deployed pool.  Template raw units are converted to the human units the
optimizer searches by the per-field raw-unit scale lookup (how the harness
stores each field: plain units, 1e10 fee fields, or 1e18 wads); the same
scale is the override multiplier applied when requests are split (see
``optimization.requests``).  Parameter-space entries need no scale fields.

The tunable pool universe is enumerated from the harness pool-override
schema: ``curve-fx-arb-harness/cpp/include/pools/pool_config_parse.hpp``
(``parse_pool_entry`` / ``parse_pool_override`` field allowlists) plus the
``costs`` table.  Scenario-identity fields (``initial_price``,
``start_timestamp``, ``initial_liquidity``, ``balances``,
``block_timestamp``, ``historical_state``, the ``policy`` struct, and
boolean flags) are rejected as optimizer dims, as are unknown paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .common import SpecError, canonical_decimal
from .policy import PolicySpec

# Tunable pool-override fields the harness evaluator accepts (numeric
# pool-economics), keyed by parameter_space name with override nesting path.
# Source: pool_config_parse.hpp parse_pool_entry reject_unknown_fields
# allowlists, plus the entry "costs" table.
POOL_TUNABLE_PATHS: dict[str, tuple[str, ...]] = {
    "A": ("A",),
    "gamma": ("gamma",),
    "mid_fee": ("mid_fee",),
    "out_fee": ("out_fee",),
    "fee_gamma": ("fee_gamma",),
    "adjustment_step_min": ("adjustment_step_min",),
    "adjustment_step_max": ("adjustment_step_max",),
    "ma_time": ("ma_time",),
    "reserved_profit_fraction": ("reserved_profit_fraction",),
    "admin_fee": ("admin_fee",),
    "donation_apy": ("donation_apy",),
    "donation_frequency": ("donation_frequency",),
    "donation_duration": ("donation_duration",),
    "initial_donation_days": ("initial_donation_days",),
    "donation_coins_ratio": ("donation_coins_ratio",),
    "user_swap_size_frac": ("user_swap_size_frac",),
    "arb_fee_bps": ("costs", "arb_fee_bps"),
    "gas_coin0": ("costs", "gas_coin0"),
    "volume_cap_mult": ("costs", "volume_cap_mult"),
}

# Scenario-identity pool fields and boolean flags: they define the scenario,
# not a tunable economics dimension.  Any path rooted at one of these fields
# (e.g. policy.params, historical_state.balances) is identity too.
POOL_SCENARIO_IDENTITY_PATHS: dict[str, tuple[str, ...]] = {
    "initial_price": ("initial_price",),
    "start_timestamp": ("start_timestamp",),
    "initial_liquidity": ("initial_liquidity",),
    "balances": ("balances",),
    "block_timestamp": ("block_timestamp",),
    "historical_state": ("historical_state",),
    "policy": ("policy",),
    "use_volume_cap": ("costs", "use_volume_cap"),
    "volume_cap_is_coin_1": ("costs", "volume_cap_is_coin_1"),
}

_SCENARIO_IDENTITY_ROOTS = frozenset(
    path[0] for path in POOL_SCENARIO_IDENTITY_PATHS.values()
)
_TUNABLE_BY_PATH = {path: name for name, path in POOL_TUNABLE_PATHS.items()}
_IDENTITY_BY_PATH = {path: name for name, path in POOL_SCENARIO_IDENTITY_PATHS.items()}

# Raw-unit multiplier per tunable pool field: how the harness stores the
# field in pool_config_parse.hpp before the optimizer searches human units
# (parse_config_plain -> 1, parse_config_fee -> 1e10, parse_config_wad ->
# 1e18).  Source: curve-fx-arb-harness/cpp/include/pools/pool_config_parse.hpp
# (parse_pool_entry).  This per-field SCALE lookup is the only per-field
# table the registry keeps; bounds/step/transform live in the specs.
_POOL_FIELD_RAW_SCALE: dict[tuple[str, ...], Decimal] = {
    ("A",): Decimal("1"),
    ("gamma",): Decimal("1"),
    ("mid_fee",): Decimal("10000000000"),  # fee, 1e10
    ("out_fee",): Decimal("10000000000"),
    ("fee_gamma",): Decimal("1000000000000000000"),  # wad, 1e18
    ("adjustment_step_min",): Decimal("1000000000000000000"),
    ("adjustment_step_max",): Decimal("1000000000000000000"),
    ("ma_time",): Decimal("1"),
    ("reserved_profit_fraction",): Decimal("10000000000"),
    ("admin_fee",): Decimal("10000000000"),
    ("donation_apy",): Decimal("1"),
    ("donation_frequency",): Decimal("1"),
    ("donation_duration",): Decimal("1"),
    ("initial_donation_days",): Decimal("1"),
    ("donation_coins_ratio",): Decimal("1"),
    ("user_swap_size_frac",): Decimal("1"),
    ("costs", "arb_fee_bps"): Decimal("1"),
    ("costs", "gas_coin0"): Decimal("1"),
    ("costs", "volume_cap_mult"): Decimal("1"),
}


def classify_pool_path(path: Sequence[str]) -> str:
    """Classify one pool override path: ``'tunable'`` | ``'identity'`` | ``'unknown'``."""
    parts = tuple(str(part) for part in path)
    if parts in _TUNABLE_BY_PATH:
        return "tunable"
    if parts in _IDENTITY_BY_PATH or (parts and parts[0] in _SCENARIO_IDENTITY_ROOTS):
        return "identity"
    return "unknown"


@dataclass(frozen=True)
class ParameterDim:
    """One resolved optimization dimension.

    ``kind`` is ``'policy'`` (evaluator dense ABI parameter; ``abi_index`` is
    its position in ``PolicySpec.parameters``) or ``'pool'`` (harness
    pool-override field; ``target_path`` is its nesting path).  ``bounds``
    and ``step`` are exactly what the governing spec declares: PolicySpec for
    policy dims, the ``parameter_space`` entry for pool dims.  ``transform``
    is ``'linear'`` or ``'log'`` (advisory; the decimal lattice stays linear
    in value space).  ``override_scale`` is the per-field raw-unit multiplier
    converting human pool values to the harness's stored convention (1 for
    plain fields, 1e10 for fee fields, 1e18 for wad fields; see
    ``_POOL_FIELD_RAW_SCALE``); policy dims use 1.
    """

    name: str
    kind: str  # 'policy' | 'pool'
    target_path: tuple[str, ...] | None  # pool override nesting; None for policy
    abi_index: int | None  # policy only: order in PolicySpec.parameters
    default: Any
    bounds: tuple[Any, Any]  # (min, max) exactly as declared
    step: Any
    transform: str = "linear"  # linear or log (advisory)
    override_scale: Decimal = Decimal("1")

    @property
    def min_val(self) -> Any:
        return self.bounds[0]

    @property
    def max_val(self) -> Any:
        return self.bounds[1]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SpecError("parameter name must be non-empty")
        if self.kind not in {"policy", "pool"}:
            raise SpecError(f"parameter {self.name} has unsupported kind {self.kind!r}")
        if self.transform not in {"linear", "log"}:
            raise SpecError(
                f"parameter {self.name} has unsupported transform {self.transform!r}"
            )
        if self.kind == "policy":
            if (
                self.abi_index is None
                or isinstance(self.abi_index, bool)
                or not isinstance(self.abi_index, int)
            ):
                raise SpecError(
                    f"policy parameter {self.name} requires an integer abi_index"
                )
            if self.target_path is not None:
                raise SpecError(
                    f"policy parameter {self.name} must not declare a target_path"
                )
        else:
            if self.abi_index is not None:
                raise SpecError(
                    f"pool parameter {self.name} must not declare an abi_index"
                )
            if not self.target_path or any(not part for part in self.target_path):
                raise SpecError(
                    f"pool parameter {self.name} requires a non-empty target_path"
                )
        if self.default is None:
            raise SpecError(f"parameter {self.name} requires an explicit default")
        minimum = self._decimal(self.bounds[0], label="min")
        maximum = self._decimal(self.bounds[1], label="max")
        if minimum > maximum:
            raise SpecError(f"parameter {self.name} min exceeds max")
        if self._decimal(self.step, label="step") <= 0:
            raise SpecError(f"parameter {self.name} step must be positive")
        default = self._decimal(self.default, label="default")
        if default < minimum or default > maximum:
            raise SpecError(
                f"parameter {self.name} default {self.default!r} lies outside "
                f"bounds [{self.bounds[0]}, {self.bounds[1]}]"
            )

    def _decimal(self, value: Any, *, label: str = "value") -> Decimal:
        try:
            return canonical_decimal(value, label=label)
        except SpecError as exc:
            raise SpecError(
                f"parameter {self.name} {label} must be numeric, got {value!r}"
            ) from exc


_ENTRY_FIELDS = frozenset({"min", "max", "step", "transform", "values"})


def _entry_dimension(name: str, raw: Any) -> tuple[Decimal, Decimal, Decimal, str]:
    """Parse one pool parameter_space entry into (min, max, step, transform)."""
    if isinstance(raw, Mapping):
        unknown = sorted(set(raw) - _ENTRY_FIELDS)
        if unknown:
            raise SpecError(
                f"parameter_space.{name} has unsupported fields: "
                + ", ".join(unknown)
            )
        transform = str(raw.get("transform", "linear")).lower()
        if transform not in {"linear", "log"}:
            raise SpecError(
                f"parameter_space.{name} transform must be 'linear' or 'log'"
            )
        if "values" in raw:
            values = tuple(
                canonical_decimal(value, label=f"parameter_space.{name}.values")
                for value in raw["values"]
            )
        else:
            for key in ("min", "max", "step"):
                if key not in raw:
                    raise SpecError(
                        f"parameter_space.{name} requires min, max, and step"
                    )
            return (
                canonical_decimal(raw["min"], label=f"parameter_space.{name}.min"),
                canonical_decimal(raw["max"], label=f"parameter_space.{name}.max"),
                canonical_decimal(raw["step"], label=f"parameter_space.{name}.step"),
                transform,
            )
    elif isinstance(raw, (list, tuple)):
        values = tuple(
            canonical_decimal(value, label=f"parameter_space.{name} values")
            for value in raw
        )
        transform = "linear"
    else:
        raise SpecError(f"parameter_space.{name} must be a table or a values list")
    if len(values) < 2:
        raise SpecError(
            f"parameter_space.{name} values must contain at least two entries"
        )
    step = values[1] - values[0]
    if step <= 0:
        raise SpecError(f"parameter_space.{name} values must be strictly increasing")
    for lower, upper in zip(values, values[1:]):
        if upper - lower != step:
            raise SpecError(
                f"parameter_space.{name} values must be uniformly spaced; "
                "use min/max/step for custom spacing"
            )
    return values[0], values[-1], step, transform


def _template_tables(
    template_json: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Extract the (pool, costs) tables of a single-pool template.

    Optimization evaluates exactly one pool per scenario (the evaluator
    session uses pool index 0; the scenario spec has no pool selector).
    Multi-pool templates are rejected explicitly rather than silently
    defaulting ``pools[0]``.
    """
    if "pools" in template_json:
        pools = template_json["pools"]
        if not isinstance(pools, list) or not pools:
            raise SpecError("template declares no pools")
        if len(pools) != 1:
            raise SpecError(
                "optimization is single-pool (pool index 0); template must "
                f"declare exactly one pool, got {len(pools)}"
            )
        entry = pools[0]
        if not isinstance(entry, Mapping):
            raise SpecError("template pool entry must be an object")
        pool = entry.get("pool", entry)
        if not isinstance(pool, Mapping):
            raise SpecError("template pool entry 'pool' must be an object")
        costs = entry.get("costs", {})
        return pool, costs if isinstance(costs, Mapping) else {}
    pool = {key: value for key, value in template_json.items() if key != "costs"}
    costs = template_json.get("costs", {})
    return pool, costs if isinstance(costs, Mapping) else {}


def _template_field(
    pool: Mapping[str, Any],
    costs: Mapping[str, Any],
    path: tuple[str, ...],
) -> Decimal:
    """Read one raw template field; SpecError when the template lacks it."""
    if len(path) == 1:
        raw = pool.get(path[0])
    elif len(path) == 2 and path[0] == "costs":
        raw = costs.get(path[1])
    else:
        raw = None
    if raw is None:
        raise SpecError(
            f"pair template has no field {'.'.join(path)} for the pool dimension default"
        )
    return canonical_decimal(raw, label=f"template {'.'.join(path)}")


def _resolve_pool_path(name: str) -> tuple[str, ...]:
    """Resolve one parameter_space pool name to its harness override path."""
    if name in POOL_TUNABLE_PATHS:
        return POOL_TUNABLE_PATHS[name]
    if "." in name:
        parts = tuple(name.split("."))
        classification = classify_pool_path(parts)
        if classification == "tunable":
            return parts
        if classification == "identity":
            raise SpecError(
                f"parameter_space.{name} is a scenario-identity pool field, "
                "not a tunable dimension"
            )
        raise SpecError(f"parameter_space.{name} is not a tunable pool-override path")
    if name in POOL_SCENARIO_IDENTITY_PATHS:
        raise SpecError(
            f"parameter_space.{name} is a scenario-identity pool field, "
            "not a tunable dimension"
        )
    raise SpecError(f"parameter_space contains undeclared dimensions: {name}")


def _pool_field_raw_scale(name: str, path: tuple[str, ...]) -> Decimal:
    """Look up the raw-unit multiplier for one tunable pool override path.

    The per-field SCALE table is required: every tunable field must know how
    the harness stores it, or the template default and the override split
    would silently use the wrong units.
    """
    scale = _POOL_FIELD_RAW_SCALE.get(path)
    if scale is None:
        raise SpecError(
            f"parameter_space.{name} has no raw-unit scale for override path "
            f"{'.'.join(path)}"
        )
    return scale


def build_parameter_registry(
    policy_spec: PolicySpec,
    template_json: Mapping[str, Any] | None,
    parameter_space: Mapping[str, Any],
) -> dict[str, ParameterDim]:
    """Resolve the name -> ParameterDim registry for one policy/template/space.

    Policy dims come first in exact ``PolicySpec.parameters`` order (their
    evaluator ABI order), carrying PolicySpec's defaults/bounds/step.  Pool
    dims are resolved only from ``parameter_space`` entries that are not
    policy names: each entry must declare bounds/step or an explicit values
    list, and its default is read from ``template_json`` (raw template value
    divided by the per-field raw-unit scale).  A pair template is required
    for pool dims; scenario-identity and unknown names are SpecError.
    """
    registry: dict[str, ParameterDim] = {}
    for index, parameter in enumerate(policy_spec.parameters):
        registry[parameter.name] = ParameterDim(
            name=parameter.name,
            kind="policy",
            target_path=None,
            abi_index=index,
            default=parameter.default,
            bounds=(parameter.min_val, parameter.max_val),
            step=parameter.step,
        )
    if template_json is None:
        undeclared = sorted(set(parameter_space) - set(registry))
        if undeclared:
            raise SpecError(
                "parameter_space contains undeclared dimensions: "
                + ", ".join(undeclared)
            )
        return registry
    if not isinstance(template_json, Mapping):
        raise SpecError("template_json must be the parsed template mapping")
    pool, costs = _template_tables(template_json)
    for name, raw in parameter_space.items():
        if name in registry:
            continue  # policy names are always policy dims
        minimum, maximum, step, transform = _entry_dimension(name, raw)
        path = _resolve_pool_path(name)
        scale = _pool_field_raw_scale(name, path)
        default = _template_field(pool, costs, path) / scale
        registry[name] = ParameterDim(
            name=name,
            kind="pool",
            target_path=path,
            abi_index=None,
            default=(
                int(default) if default == default.to_integral_value() else float(default)
            ),
            bounds=(minimum, maximum),
            step=step,
            transform=transform,
            override_scale=scale,
        )
    return registry


def validate_parameter_space_names(
    parameter_space: Mapping[str, Any],
    registry: Mapping[str, ParameterDim],
) -> None:
    """Resolve every parameter_space name through the registry.

    With a registry built from the same parameter_space every entry is
    present by construction; this is the single name-resolution check kept
    for callers that reuse a registry across spaces.
    """
    unknown = sorted(set(parameter_space) - set(registry))
    if unknown:
        raise SpecError(
            "parameter_space contains undeclared dimensions: " + ", ".join(unknown)
        )


__all__ = [
    "ParameterDim",
    "POOL_TUNABLE_PATHS",
    "POOL_SCENARIO_IDENTITY_PATHS",
    "classify_pool_path",
    "build_parameter_registry",
    "validate_parameter_space_names",
]
