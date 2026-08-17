"""Exact decimal-lattice helpers and coordinate discretization for optimization.

Optimizer profiles describe decimal production grids. Performing the final
``tick * quantum`` operation in binary floating point can select an adjacent
binary64 value instead of the nearest representation of the intended decimal
coordinate. Keep tick arithmetic decimal and convert to ``float`` only once,
at the boundary consumed by the simulator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    localcontext,
)
from typing import Any, Sequence, TypeAlias

DecimalInput: TypeAlias = Decimal | float | int | str
TickPoint: TypeAlias = tuple[int, ...]

DECIMAL_LATTICE_VERSION = "decimal_ticks_v1"
DECIMAL_REQUEST_ARITHMETIC_VERSION = "decimal_arithmetic_v1"
POOL_INPUT_PRECISION_VERSION = "binary64_inputs_v1"


def decimal_value(value: DecimalInput) -> Decimal:
    """Return the human decimal spelling of a finite numeric value."""
    if isinstance(value, Decimal):
        result = value
    else:
        result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"decimal value must be finite, got {value!r}")
    return result


def lattice_tick(value: DecimalInput, quantum: DecimalInput) -> int:
    """Discretize a value to the nearest integer tick on the quantum lattice."""
    quantum_decimal = decimal_value(quantum)
    if quantum_decimal <= 0:
        raise ValueError(f"lattice quantum must be positive, got {quantum!r}")
    with localcontext() as context:
        context.prec = 80
        return int(
            (decimal_value(value) / quantum_decimal).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )


def lattice_float(tick: int, quantum: DecimalInput) -> float:
    """Decode a tick to the canonical binary64 for its decimal coordinate."""
    if not isinstance(tick, int) or isinstance(tick, bool):
        raise TypeError(f"lattice tick must be an integer, got {tick!r}")
    quantum_decimal = decimal_value(quantum)
    if quantum_decimal <= 0:
        raise ValueError(f"lattice quantum must be positive, got {quantum!r}")
    with localcontext() as context:
        context.prec = 80
        return float(Decimal(tick) * quantum_decimal)


def quantize_lattice_float(
    value: DecimalInput,
    quantum: DecimalInput,
    *,
    lower: DecimalInput | None = None,
    upper: DecimalInput | None = None,
) -> float:
    """Round to a decimal tick, optionally clamping to lattice-safe bounds."""
    quantum_decimal = decimal_value(quantum)
    tick = lattice_tick(value, quantum_decimal)
    with localcontext() as context:
        context.prec = 80
        if lower is not None:
            lower_tick = int(
                (decimal_value(lower) / quantum_decimal).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            tick = max(tick, lower_tick)
        if upper is not None:
            upper_tick = int(
                (decimal_value(upper) / quantum_decimal).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            tick = min(tick, upper_tick)
        if lower is not None and upper is not None and lower_tick > upper_tick:
            raise ValueError(
                f"bounds [{lower!r}, {upper!r}] contain no {quantum!r} tick"
            )
    return lattice_float(tick, quantum_decimal)


def decimal_multiply_float(
    value: DecimalInput,
    multiplier: DecimalInput,
) -> float:
    """Multiply decimal quantities and round once to binary64."""
    with localcontext() as context:
        context.prec = 80
        result = float(decimal_value(value) * decimal_value(multiplier))
    if not math.isfinite(result):
        raise ValueError("decimal product does not fit in a finite float")
    return result


def decimal_divide_float(value: DecimalInput, divisor: DecimalInput) -> float:
    """Divide decimal quantities and round once to binary64."""
    divisor_decimal = decimal_value(divisor)
    if divisor_decimal == 0:
        raise ZeroDivisionError("decimal divisor is zero")
    with localcontext() as context:
        context.prec = 80
        result = float(decimal_value(value) / divisor_decimal)
    if not math.isfinite(result):
        raise ValueError("decimal quotient does not fit in a finite float")
    return result


def decimal_add_float(*values: DecimalInput) -> float:
    """Add decimal quantities and round once to binary64."""
    with localcontext() as context:
        context.prec = 80
        total = sum((decimal_value(value) for value in values), Decimal(0))
        result = float(total)
    if not math.isfinite(result):
        raise ValueError("decimal sum does not fit in a finite float")
    return result


@dataclass(frozen=True)
class TickAxis:
    """A single discrete axis on the decimal lattice."""

    index: int
    name: str
    quantum: Decimal
    min_tick: int
    max_tick: int
    is_log: bool = False

    @property
    def cardinality(self) -> int:
        return self.max_tick - self.min_tick + 1

    def tick_to_float(self, tick: int) -> float:
        return lattice_float(tick, self.quantum)

    def value_to_tick(self, value: DecimalInput) -> int:
        tick = lattice_tick(value, self.quantum)
        return max(self.min_tick, min(self.max_tick, tick))


@dataclass
class LatticeSpec:
    """Exact integer coordinate lattice over an optimization profile."""

    profile_name: str
    axes: tuple[TickAxis, ...]
    fixed_params: dict[int, float] = field(default_factory=dict)
    n_params: int = 0

    def __post_init__(self) -> None:
        if not self.n_params:
            max_idx = max(
                (axis.index for axis in self.axes),
                default=-1,
            )
            if self.fixed_params:
                max_idx = max(max_idx, max(self.fixed_params.keys()))
            self.n_params = max_idx + 1

    @property
    def dim(self) -> int:
        return len(self.axes)

    def validate(self, point: TickPoint, *, require_feasible: bool = True) -> None:
        if len(point) != len(self.axes):
            raise ValueError(
                f"TickPoint length {len(point)} != lattice dim {len(self.axes)}"
            )
        if require_feasible:
            for val, axis in zip(point, self.axes, strict=True):
                if not (axis.min_tick <= val <= axis.max_tick):
                    raise ValueError(
                        f"axis {axis.index} ({axis.name}) tick {val} out of bounds "
                        f"[{axis.min_tick}, {axis.max_tick}]"
                    )

    def decode(self, point: TickPoint) -> list[float]:
        """Convert a discrete TickPoint into the complete dense params vector."""
        self.validate(point, require_feasible=False)
        out = [0.0] * self.n_params
        for idx, val in self.fixed_params.items():
            if 0 <= idx < self.n_params:
                out[idx] = float(val)
        for tick, axis in zip(point, self.axes, strict=True):
            out[axis.index] = axis.tick_to_float(tick)
        return out

    def encode(self, params: Sequence[DecimalInput]) -> TickPoint:
        """Convert a dense params vector into the nearest valid TickPoint."""
        ticks = []
        for axis in self.axes:
            if axis.index < len(params):
                ticks.append(axis.value_to_tick(params[axis.index]))
            else:
                ticks.append(axis.min_tick)
        return tuple(ticks)

    def key(self, point: TickPoint) -> str:
        """Deterministic JSON key for cache identity."""
        return json.dumps(list(point), separators=(",", ":"))


__all__ = [
    "DecimalInput",
    "TickPoint",
    "DECIMAL_LATTICE_VERSION",
    "DECIMAL_REQUEST_ARITHMETIC_VERSION",
    "POOL_INPUT_PRECISION_VERSION",
    "decimal_value",
    "lattice_tick",
    "lattice_float",
    "quantize_lattice_float",
    "decimal_multiply_float",
    "decimal_divide_float",
    "decimal_add_float",
    "TickAxis",
    "LatticeSpec",
]
