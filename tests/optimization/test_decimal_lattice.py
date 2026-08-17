"""Deterministic tests for exact decimal lattice discretization and arithmetic."""

import pytest
from decimal import Decimal

from curve_fx_sim.optimization.lattice import (
    decimal_value,
    lattice_tick,
    lattice_float,
    quantize_lattice_float,
    decimal_multiply_float,
    decimal_divide_float,
    decimal_add_float,
    TickAxis,
    LatticeSpec,
)


def test_decimal_value_parsing():
    assert decimal_value(10) == Decimal("10")
    assert decimal_value(10.5) == Decimal("10.5")
    assert decimal_value("0.0001") == Decimal("0.0001")
    assert decimal_value(Decimal("42.0")) == Decimal("42.0")

    with pytest.raises(ValueError):
        decimal_value(float("nan"))
    with pytest.raises(ValueError):
        decimal_value(float("inf"))


def test_lattice_tick_and_float_roundtrip():
    quantum = "0.05"
    assert lattice_tick(1.0, quantum) == 20
    assert lattice_tick(1.02, quantum) == 20
    assert lattice_tick(1.03, quantum) == 21
    assert lattice_float(20, quantum) == 1.0
    assert lattice_float(21, quantum) == 1.05


def test_quantize_lattice_float_bounds():
    step = 0.5
    # Value 1.2 quantizes to 1.0, clamped in [2.0, 5.0] becomes 2.0
    assert quantize_lattice_float(1.2, step, lower=2.0, upper=5.0) == 2.0
    # Value 7.2 quantizes to 7.0, clamped in [2.0, 5.0] becomes 5.0
    assert quantize_lattice_float(7.2, step, lower=2.0, upper=5.0) == 5.0
    # Value 3.4 quantizes to 3.5 in [2.0, 5.0]
    assert quantize_lattice_float(3.4, step, lower=2.0, upper=5.0) == 3.5


def test_exact_decimal_arithmetic():
    # 0.1 + 0.2 in float is 0.30000000000000004; decimal_add_float gives exact 0.3
    res_add = decimal_add_float("0.1", "0.2")
    assert res_add == 0.3

    res_mul = decimal_multiply_float("54.3", "10000")
    assert res_mul == 543000.0

    res_div = decimal_divide_float("4.9", "100")
    assert res_div == 0.049


def test_lattice_spec_encode_decode():
    axis0 = TickAxis(index=0, name="p0", quantum=Decimal("0.1"), min_tick=100, max_tick=1500)
    axis1 = TickAxis(index=1, name="p1", quantum=Decimal("0.01"), min_tick=0, max_tick=100)
    spec = LatticeSpec(
        profile_name="test_profile",
        axes=(axis0, axis1),
        fixed_params={2: 42.0},
        n_params=3,
    )

    point = (543, 50)  # 54.3, 0.50
    decoded = spec.decode(point)
    assert len(decoded) == 3
    assert decoded[0] == 54.3
    assert decoded[1] == 0.5
    assert decoded[2] == 42.0

    encoded = spec.encode([54.32, 0.499, 42.0])
    assert encoded == (543, 50)
    assert spec.key(point) == "[543,50]"
