"""Exact-lattice and deterministic-resume tests for Nevergrad TwoPointsDE."""

from decimal import Decimal

import pytest

from curve_fx_sim.optimization.lattice import LatticeSpec, TickAxis
from curve_fx_sim.optimization.nevergrad_adapter import NevergradTwoPointsDEOptimizer


def _lattice() -> LatticeSpec:
    return LatticeSpec(
        profile_name="test",
        axes=(
            TickAxis(index=0, name="x", quantum=Decimal("0.5"), min_tick=0, max_tick=10),
            TickAxis(index=1, name="y", quantum=Decimal("0.1"), min_tick=-5, max_tick=5),
        ),
        n_params=2,
    )


def _optimizer() -> NevergradTwoPointsDEOptimizer:
    return NevergradTwoPointsDEOptimizer(
        _lattice(),
        initial_params=[2.0, 0.0],
        budget=12,
        seed=123,
        num_workers=4,
    )


def _tell_batch(optimizer: NevergradTwoPointsDEOptimizer, points: list[list[float]]) -> None:
    losses = [sum(value * value for value in point) for point in points]
    optimizer.tell(
        points,
        losses,
        objectives=[-loss for loss in losses],
        scores=[{"loss": loss} for loss in losses],
    )


def test_nevergrad_uses_exact_bounded_lattice_and_resumes_deterministically() -> None:
    original = _optimizer()
    first = original.ask(4)
    assert all(_lattice().decode(_lattice().encode(point)) == point for point in first)
    _tell_batch(original, first)

    restored = _optimizer()
    restored.restore(original.snapshot())
    assert restored.step == 4
    restored_points = restored.ask(4)
    original_points = original.ask(4)
    assert restored_points == original_points

    _tell_batch(original, original_points)
    _tell_batch(restored, restored_points)
    assert restored.snapshot() == original.snapshot()


def test_nevergrad_enforces_complete_ask_tell_batches() -> None:
    optimizer = _optimizer()
    points = optimizer.ask(3)
    with pytest.raises(RuntimeError, match="previous batch"):
        optimizer.ask(1)
    with pytest.raises(ValueError, match="exactly match"):
        optimizer.tell(points[:2], [1.0, 2.0])
    _tell_batch(optimizer, points)
    assert optimizer.step == 3


def test_nevergrad_restore_rejects_tampered_history() -> None:
    optimizer = _optimizer()
    points = optimizer.ask(4)
    _tell_batch(optimizer, points)
    state = optimizer.snapshot()
    state["history"][0]["ticks"][0][0] += 1

    with pytest.raises(ValueError, match="does not reproduce"):
        _optimizer().restore(state)
