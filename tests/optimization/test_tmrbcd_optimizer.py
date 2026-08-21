"""Deterministic tests for discrete TMRBCD optimizer and resume equivalence."""

import copy
from decimal import Decimal

import pytest

from curve_fx_sim.optimization.lattice import LatticeSpec, TickAxis
from curve_fx_sim.optimization.tmrbcd import (
    TmrbcdOptimizer,
)


def _saturation_lattice(dim: int) -> LatticeSpec:
    """A small cubic lattice (ticks 0..4 per axis) with a mid-point seed."""
    return LatticeSpec(
        profile_name=f"saturation{dim}",
        axes=tuple(
            TickAxis(
                index=i,
                name=f"p{i}",
                quantum=Decimal("1"),
                min_tick=0,
                max_tick=4,
            )
            for i in range(dim)
        ),
        n_params=dim,
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda state: state.pop("rng_state"), "fields mismatch"),
        (lambda state: state.__setitem__("schema_version", 2), "schema_version"),
        (lambda state: state.__setitem__("step", state["step"] + 1), "evaluation cache"),
        (lambda state: state.__setitem__("best_loss", 999.0), "best point/loss"),
        (lambda state: state["rng_state"].clear(), "rng_state is invalid"),
    ],
)
def test_tmrbcd_restore_rejects_truncated_or_tampered_state(mutate, match):
    optimizer = TmrbcdOptimizer(
        lattice=_saturation_lattice(2),
        initial_params=(2.0, 2.0),
        budget=8,
        seed=42,
    )
    asks = optimizer.ask(batch_size=4)
    optimizer.tell(asks, [float(index + 1) for index in range(len(asks))])
    state = copy.deepcopy(optimizer.snapshot())
    mutate(state)

    restored = TmrbcdOptimizer(
        lattice=_saturation_lattice(2),
        initial_params=(2.0, 2.0),
        budget=8,
        seed=42,
    )
    with pytest.raises(ValueError, match=match):
        restored.restore(state)

def _flat_run(dim: int, budget: int, seed: int):
    """Drive one flat-objective run, recording per-tell saturation state."""
    opt = TmrbcdOptimizer(
        lattice=_saturation_lattice(dim),
        initial_params=(2.0,) * dim,
        budget=budget,
        seed=seed,
    )
    proposals = []
    states = []
    while True:
        batch = opt.ask(batch_size=8)
        if not batch:
            break
        opt.tell(batch, [1.0] * len(batch))
        proposals.extend(batch)
        states.append(
            (
                dict(opt.block_saturation),
                dict(opt.block_last_gain),
                opt.active_axis_idx,
                dict(opt.block_pass_start),
            )
        )
    return opt, proposals, states


def test_tmrbcd_flat_objective_rotates_saturated_blocks():
    """Flat objective: every full pass saturates, so the block order rotates.

    A saturated dim is deprioritized (never selected again until the round
    resets) and the next selection is deterministically randomized among the
    remaining unsaturated dims.
    """
    seed = 5
    opt1, proposals1, states1 = _flat_run(dim=3, budget=32, seed=seed)
    opt2, proposals2, states2 = _flat_run(dim=3, budget=32, seed=seed)
    assert proposals1 == proposals2
    assert states1 == states2

    # First full pass completes on block 0 (swept first): it saturates alone,
    # and the saturated dim is deprioritized for the next sweep.
    assert states1[0][0] == {0: 1, 1: 0, 2: 0}
    assert states1[0][1] == {0: 0.0, 1: 0.0, 2: 0.0}
    assert states1[0][2] in (1, 2)

    # Second pass completes on a block rotated out of the cyclic order
    # (block 2 before block 1 for this seed): the selection was randomized
    # among the remaining unsaturated dims.
    assert states1[1][0] == {0: 1, 1: 0, 2: 1}
    assert states1[1][2] == 1

    # After the third completion every dim is saturated: the round resets to a
    # fresh, randomized rotation.
    assert states1[2][0] == {0: 0, 1: 0, 2: 0}
    assert opt1.best_loss == 1.0
    assert opt1.best_point == (2, 2, 2)


def test_tmrbcd_sloped_objective_keeps_improving_block():
    """Sloped objective: the improving block never saturates and stays in the
    rotation, while the flat block is deprioritized after its zero-gain pass."""
    seed = 7
    lattice = _saturation_lattice(2)
    opt = TmrbcdOptimizer(
        lattice=lattice,
        initial_params=(2.0, 2.0),
        budget=32,
        seed=seed,
    )
    histories = []
    while True:
        batch = opt.ask(batch_size=8)
        if not batch:
            break
        opt.tell(batch, [params[0] for params in batch])
        histories.append(
            (
                dict(opt.block_saturation),
                dict(opt.block_last_gain),
                opt.active_axis_idx,
                dict(opt.block_pass_start),
            )
        )

    # Block 0 improved the incumbent on its first full pass: gain recorded,
    # never saturated, and it remains the selected block afterwards.
    assert opt.best_point == (0, 2)
    assert opt.best_loss == 0.0
    assert opt.block_saturation == {0: 0, 1: 1}
    assert opt.block_last_gain == {0: 2.0, 1: 0.0}
    assert opt.active_axis_idx == 0

    # The flat block saturated on its own pass and was dropped from rotation.
    assert histories[1][0] == {0: 0, 1: 1}
    assert histories[1][2] == 0
    for history in histories[2:]:
        assert history[2] == 0
        assert history[0][1] == 1


def test_tmrbcd_resume_with_saturation_state():
    """A checkpoint taken mid-run restores exact saturation state (including an
    in-flight pass) and reproduces the uninterrupted proposal sequence."""
    dim = 3
    budget = 32
    seed = 5

    # 1. Uninterrupted run.
    opt_full = TmrbcdOptimizer(
        lattice=_saturation_lattice(dim),
        initial_params=(2.0,) * dim,
        budget=budget,
        seed=seed,
    )
    full_proposals = []
    while True:
        batch = opt_full.ask(batch_size=8)
        if not batch:
            break
        full_proposals.extend(batch)
        opt_full.tell(batch, [1.0] * len(batch))

    # 2. Checkpoint after two batches (two saturated blocks and one in-flight
    #    pass) and resume.
    opt_check = TmrbcdOptimizer(
        lattice=_saturation_lattice(dim),
        initial_params=(2.0,) * dim,
        budget=budget,
        seed=seed,
    )
    resumed_proposals = []
    for _ in range(2):
        batch = opt_check.ask(batch_size=8)
        resumed_proposals.extend(batch)
        opt_check.tell(batch, [1.0] * len(batch))

    snapshot = opt_check.snapshot()
    assert snapshot["block_saturation"] == {"0": 1, "1": 0, "2": 1}
    assert snapshot["block_last_gain"] == {"0": 0.0, "1": 0.0, "2": 0.0}
    assert snapshot["block_pass_start"] == {"1": 1.0}

    opt_resumed = TmrbcdOptimizer(
        lattice=_saturation_lattice(dim),
        initial_params=(2.0,) * dim,
        budget=budget,
        seed=seed,
    )
    opt_resumed.restore(snapshot)
    while True:
        batch = opt_resumed.ask(batch_size=8)
        if not batch:
            break
        resumed_proposals.extend(batch)
        opt_resumed.tell(batch, [1.0] * len(batch))

    assert len(resumed_proposals) == len(full_proposals)
    assert resumed_proposals == full_proposals
    assert opt_resumed.block_saturation == opt_full.block_saturation
    assert opt_resumed.block_last_gain == opt_full.block_last_gain
    assert opt_resumed.block_pass_start == opt_full.block_pass_start
    assert opt_resumed.best_point == opt_full.best_point


def test_tmrbcd_restore_accepts_checkpoint_without_saturation_state():
    """Checkpoints written before saturation tracking load with defaults and
    keep proposing deterministically."""
    lattice = _saturation_lattice(3)
    seed = 42
    budget = 24

    opt = TmrbcdOptimizer(
        lattice=lattice,
        initial_params=(2.0, 2.0, 2.0),
        budget=budget,
        seed=seed,
    )
    batch = opt.ask(batch_size=8)
    opt.tell(batch, [1.0] * len(batch))
    old_state = copy.deepcopy(opt.snapshot())
    for field in ("block_last_gain", "block_saturation", "block_pass_start"):
        old_state.pop(field)

    def restore_and_run():
        restored = TmrbcdOptimizer(
            lattice=lattice,
            initial_params=(2.0, 2.0, 2.0),
            budget=budget,
            seed=seed,
        )
        restored.restore(copy.deepcopy(old_state))
        # Saturation state defaults to a fresh, unsaturated round.
        assert restored.block_saturation == {0: 0, 1: 0, 2: 0}
        assert restored.block_last_gain == {0: 0.0, 1: 0.0, 2: 0.0}
        assert restored.block_pass_start == {}
        assert restored.step == opt.step
        assert restored.active_axis_idx == opt.active_axis_idx
        proposals = []
        while True:
            batch = restored.ask(batch_size=8)
            if not batch:
                break
            proposals.extend(batch)
            restored.tell(batch, [1.0] * len(batch))
        return proposals, restored

    proposals_a, restored_a = restore_and_run()
    proposals_b, restored_b = restore_and_run()
    assert len(proposals_a) == budget - opt.step
    assert proposals_a == proposals_b
    assert restored_a.block_saturation == restored_b.block_saturation
