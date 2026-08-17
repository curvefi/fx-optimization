"""Discrete TMRBCD (Trust-Region-Free Block Coordinate Descent) optimizer implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .lattice import LatticeSpec, TickPoint


TMRBCD_STATE_SCHEMA_VERSION = 3
# Per-block saturation state is additive: old checkpoints (which lack these
# fields) restore with a fresh, unsaturated round, so the schema version is
# unchanged and the fields stay optional.
TMRBCD_STATE_OPTIONAL_FIELDS = frozenset(
    {
        "block_last_gain",
        "block_saturation",
        "block_pass_start",
    }
)
TMRBCD_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "step",
        "seed",
        "best_loss",
        "best_objective",
        "best_point",
        "best_params",
        "current_point",
        "active_axis_idx",
        "rng_state",
        "restart_rng_state",
        "restart_generated",
        "cache",
        "best_score",
        *TMRBCD_STATE_OPTIONAL_FIELDS,
    }
)


class RestartAllocator:
    """Deterministic random restart allocator for integer lattices."""

    def __init__(self, lattice: LatticeSpec, seed: int = 42) -> None:
        self.lattice = lattice
        self.rng = random.Random(seed)
        self.generated: set[TickPoint] = set()

    def sample(self, count: int = 1) -> list[TickPoint]:
        out: list[TickPoint] = []
        for _ in range(count):
            for _attempt in range(1000):
                point = tuple(
                    self.rng.randint(axis.min_tick, axis.max_tick)
                    for axis in self.lattice.axes
                )
                if point not in self.generated:
                    self.generated.add(point)
                    out.append(point)
                    break
        return out


def _rng_state_to_json(state: Any) -> list[Any]:
    # state is (version, tuple_of_ints, float_or_None)
    version, keys, pos = state
    return [version, list(keys), pos]


def _rng_state_from_json(payload: list[Any]) -> Any:
    version, keys, pos = payload
    return (version, tuple(keys), pos)


def _parse_block_map(
    payload: Any,
    dim: int,
    label: str,
    *,
    integer: bool,
) -> dict[int, int | float]:
    """Parse one optional per-block state map, fail closed on malformed input."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"TMRBCD optimizer_state {label} must be an object")
    out: dict[int, int | float] = {}
    for raw_index, raw_value in payload.items():
        if isinstance(raw_index, bool):
            raise ValueError(f"TMRBCD optimizer_state {label} has an invalid key")
        if isinstance(raw_index, int):
            index = raw_index
        elif isinstance(raw_index, str):
            try:
                index = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"TMRBCD optimizer_state {label} has an invalid key"
                ) from exc
        else:
            raise ValueError(f"TMRBCD optimizer_state {label} has an invalid key")
        if not 0 <= index < dim:
            raise ValueError(
                f"TMRBCD optimizer_state {label} index is outside the active lattice"
            )
        if integer:
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 0:
                raise ValueError(
                    f"TMRBCD optimizer_state {label} value must be a non-negative integer"
                )
        elif isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"TMRBCD optimizer_state {label} value must be numeric")
        out[index] = raw_value
    return out


@dataclass
class TmrbcdOptimizer:
    """Discrete Coordinate Descent optimizer operating directly on exact integer lattice ticks.

    Blocks (coordinate lines) with a completed full pass that produced zero
    incumbent improvement are marked saturated and deprioritized: the next
    block selection is deterministically randomized (seeded via the run's own
    RNG) among the remaining unsaturated dims, instead of the plain cyclic
    rotation.  Saturation state round-trips through checkpoints; checkpoints
    written before saturation tracking restore with a fresh unsaturated round.
    """

    lattice: LatticeSpec
    initial_params: Sequence[float] | None = None
    seed: int = 42
    budget: int = 100

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.restart = RestartAllocator(self.lattice, seed=self.seed)
        self.current_point: TickPoint = (
            self.lattice.encode(self.initial_params)
            if self.initial_params is not None
            else tuple((axis.min_tick + axis.max_tick) // 2 for axis in self.lattice.axes)
        )
        self.best_point: TickPoint = self.current_point
        self.best_loss: float = float("inf")
        self.best_objective: float = float("-inf")
        self.best_score: dict[str, Any] | None = None
        self.step: int = 0
        self.eval_cache: dict[TickPoint, float] = {}
        self.active_axis_idx: int = 0
        self.pending_asks: list[TickPoint] = []
        self.pending_block_tags: list[int | None] = []
        # Per-block saturation state, keyed by block (axis position).
        # last_gain: incumbent improvement of the block's most recent full pass.
        # saturation: consecutive full passes with zero gain since the last gain.
        # pass_start: incumbent when the block's in-flight pass began.
        self.block_last_gain: dict[int, float] = {
            index: 0.0 for index in range(len(self.lattice.axes))
        }
        self.block_saturation: dict[int, int] = {
            index: 0 for index in range(len(self.lattice.axes))
        }
        self.block_pass_start: dict[int, float] = {}

    def ask(self, batch_size: int = 1) -> list[list[float]]:
        """Generate the next batch of candidate parameter vectors.

        Returns empty list when budget is reached or exceeded.
        """
        if self.pending_asks:
            raise RuntimeError("TMRBCD ask requires the previous batch to be told first")
        if self.step >= self.budget:
            return []

        remaining = self.budget - self.step
        actual_batch = min(batch_size, remaining)
        if actual_batch <= 0:
            return []

        points: list[TickPoint] = []
        tags: list[int | None] = []

        # 1. First evaluation: initial point if not evaluated yet
        if self.current_point not in self.eval_cache:
            points.append(self.current_point)
            tags.append(None)

        while len(points) < actual_batch:
            # 2. Coordinate line sweep along the active block.
            block = self.active_axis_idx
            axis = self.lattice.axes[block]
            span = axis.max_tick - axis.min_tick
            if span > 0:
                steps = max(3, min(7, span + 1))
                step_size = max(1, span // (steps - 1))
                for t in range(axis.min_tick, axis.max_tick + 1, step_size):
                    pt_list = list(self.current_point)
                    pt_list[block] = t
                    pt = tuple(pt_list)
                    if pt not in self.eval_cache and pt not in points:
                        points.append(pt)
                        tags.append(block)
                        if len(points) >= actual_batch:
                            break
            # Rotate to the next block; tell() may randomize this selection
            # among the remaining unsaturated dims after a saturated pass.
            self.active_axis_idx = self._next_block_index(block)

            # 3. If the sweep is short, fill the batch with deterministic random restarts.
            if len(points) < actual_batch:
                extra = self.restart.sample(actual_batch - len(points))
                for pt in extra:
                    if pt not in self.eval_cache and pt not in points:
                        points.append(pt)
                        tags.append(None)
                if not extra:
                    break

        self.pending_asks = points
        self.pending_block_tags = tags
        return [self.lattice.decode(pt) for pt in points]

    def _next_block_index(self, finished: int) -> int:
        """Rotate to the next block, skipping saturated dims.

        When every block is saturated the round is over: reset the saturation
        counters and continue the rotation.  The pick never consumes RNG; only
        ``_randomize_saturated_selection`` draws from the run's RNG.
        """
        dim = len(self.lattice.axes)
        for offset in range(1, dim + 1):
            candidate = (finished + offset) % dim
            if self.block_saturation[candidate] == 0:
                return candidate
        self.block_saturation = {index: 0 for index in range(dim)}
        return (finished + 1) % dim

    def _randomize_saturated_selection(self) -> None:
        """Deterministically randomize the next block among unsaturated dims.

        Uses the run's own seeded RNG (not global state).  When no unsaturated
        dim remains, all blocks start a fresh round before the random pick.
        """
        dim = len(self.lattice.axes)
        unsaturated = [index for index in range(dim) if self.block_saturation[index] == 0]
        if not unsaturated:
            self.block_saturation = {index: 0 for index in range(dim)}
            unsaturated = list(range(dim))
        self.active_axis_idx = unsaturated[self.rng.randrange(len(unsaturated))]

    def _pass_line_losses(self, block: int) -> tuple[float, ...] | None:
        """Cached losses of every scheduled tick of block's current line.

        Returns ``None`` until all scheduled ticks of the line (around the
        current point) have been evaluated, i.e. the full pass has completed.
        """
        axis = self.lattice.axes[block]
        span = axis.max_tick - axis.min_tick
        if span <= 0:
            return ()
        steps = max(3, min(7, span + 1))
        step_size = max(1, span // (steps - 1))
        losses: list[float] = []
        for t in range(axis.min_tick, axis.max_tick + 1, step_size):
            pt_list = list(self.current_point)
            pt_list[block] = t
            loss = self.eval_cache.get(tuple(pt_list))
            if loss is None:
                return None
            losses.append(loss)
        return tuple(losses)

    def tell(
        self,
        candidates: Sequence[Sequence[float]],
        losses: Sequence[float],
        objectives: Sequence[float] | None = None,
        scores: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Inform the optimizer of exactly the pending evaluated batch."""
        expected = len(self.pending_asks)
        if len(candidates) != expected or len(losses) != expected:
            raise ValueError(
                "TMRBCD tell candidates and losses must exactly match the pending batch"
            )
        if objectives is not None and len(objectives) != expected:
            raise ValueError("TMRBCD tell objectives must exactly match the pending batch")
        if scores is not None and len(scores) != expected:
            raise ValueError("TMRBCD tell scores must exactly match the pending batch")

        points = [self.lattice.encode(params) for params in candidates]
        if points != self.pending_asks:
            raise ValueError("TMRBCD tell candidates do not match the pending ask order")
        if len(set(points)) != len(points):
            raise ValueError("TMRBCD tell candidates contain duplicate points")

        loss_values = [float(loss) for loss in losses]
        objective_values = (
            [float(objective) for objective in objectives]
            if objectives is not None
            else [-loss for loss in loss_values]
        )
        score_values = list(scores) if scores is not None else [None] * expected

        # Record the incumbent at the start of each block's in-flight pass, in
        # batch order, so a completed pass gain is measured against the
        # incumbent it started from.
        running_best = self.best_loss
        for pt, loss_val, tag in zip(
            points,
            loss_values,
            self.pending_block_tags,
            strict=True,
        ):
            if tag is not None and tag not in self.block_pass_start:
                self.block_pass_start[tag] = running_best
            running_best = min(running_best, loss_val)

        for pt, loss_val, obj_val, score_val in zip(
            points,
            loss_values,
            objective_values,
            score_values,
            strict=True,
        ):
            self.eval_cache[pt] = loss_val
            self.step += 1
            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.best_point = pt
                self.current_point = pt
                self.best_objective = obj_val
                self.best_score = score_val

        # A block saturates only after its full pass completes: every
        # scheduled tick of the line has been evaluated with zero incumbent
        # improvement.  Batch fragments never update saturation state.
        saturation_event = False
        for block in list(self.block_pass_start):
            line_losses = self._pass_line_losses(block)
            if line_losses is None:
                continue
            gain = max(0.0, self.block_pass_start[block] - min(line_losses))
            self.block_last_gain[block] = gain
            if gain > 0.0:
                self.block_saturation[block] = 0
            else:
                self.block_saturation[block] += 1
                saturation_event = True
            del self.block_pass_start[block]
        if saturation_event:
            self._randomize_saturated_selection()
        self.pending_asks = []
        self.pending_block_tags = []

    def snapshot(self) -> dict[str, Any]:
        """Save deterministic optimizer checkpoint including all RNG and allocator state."""
        return {
            "schema_version": TMRBCD_STATE_SCHEMA_VERSION,
            "step": self.step,
            "seed": self.seed,
            "best_loss": self.best_loss,
            "best_objective": self.best_objective,
            "best_point": list(self.best_point),
            "best_params": self.lattice.decode(self.best_point),
            "current_point": list(self.current_point),
            "active_axis_idx": self.active_axis_idx,
            "rng_state": _rng_state_to_json(self.rng.getstate()),
            "restart_rng_state": _rng_state_to_json(self.restart.rng.getstate()),
            "restart_generated": [list(pt) for pt in self.restart.generated],
            "cache": [
                {"point": list(pt), "loss": loss}
                for pt, loss in sorted(self.eval_cache.items())
            ],
            "best_score": self.best_score,
            "block_last_gain": {
                str(index): self.block_last_gain[index]
                for index in sorted(self.block_last_gain)
            },
            "block_saturation": {
                str(index): self.block_saturation[index]
                for index in sorted(self.block_saturation)
            },
            "block_pass_start": {
                str(index): self.block_pass_start[index]
                for index in sorted(self.block_pass_start)
            },
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        """Restore one exact, versioned optimizer state or fail closed."""
        if not isinstance(payload, Mapping):
            raise ValueError("TMRBCD optimizer_state must be an object")
        fields = set(payload)
        missing = sorted((TMRBCD_STATE_FIELDS - TMRBCD_STATE_OPTIONAL_FIELDS) - fields)
        unknown = sorted(fields - TMRBCD_STATE_FIELDS)
        if missing or unknown:
            raise ValueError(
                f"TMRBCD optimizer_state fields mismatch: missing={missing}, unknown={unknown}"
            )
        if payload["schema_version"] != TMRBCD_STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported TMRBCD optimizer_state schema_version "
                f"{payload['schema_version']!r}"
            )

        def strict_int(value: Any, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"TMRBCD optimizer_state {label} must be an integer")
            return value

        def point(value: Any, label: str) -> TickPoint:
            if not isinstance(value, list):
                raise ValueError(f"TMRBCD optimizer_state {label} must be an array")
            parsed = tuple(strict_int(item, f"{label} entry") for item in value)
            try:
                self.lattice.validate(parsed)
            except ValueError as exc:
                raise ValueError(
                    f"TMRBCD optimizer_state {label} is outside the active lattice"
                ) from exc
            return parsed

        step = strict_int(payload["step"], "step")
        if step < 0 or step > self.budget:
            raise ValueError(
                f"TMRBCD optimizer_state step {step} is outside budget {self.budget}"
            )
        seed = strict_int(payload["seed"], "seed")
        if seed != self.seed:
            raise ValueError(
                f"TMRBCD optimizer_state seed {seed} does not match configured seed {self.seed}"
            )
        active_axis_idx = strict_int(payload["active_axis_idx"], "active_axis_idx")
        if not 0 <= active_axis_idx < self.lattice.dim:
            raise ValueError("TMRBCD optimizer_state active_axis_idx is outside the active lattice")

        best_point = point(payload["best_point"], "best_point")
        current_point = point(payload["current_point"], "current_point")
        best_params = payload["best_params"]
        if not isinstance(best_params, list) or best_params != self.lattice.decode(best_point):
            raise ValueError("TMRBCD optimizer_state best_params does not match best_point")

        best_loss = payload["best_loss"]
        best_objective = payload["best_objective"]
        if (
            isinstance(best_loss, bool)
            or not isinstance(best_loss, (int, float))
            or isinstance(best_objective, bool)
            or not isinstance(best_objective, (int, float))
        ):
            raise ValueError("TMRBCD optimizer_state best loss/objective must be numeric")

        best_score = payload["best_score"]
        if best_score is not None and not isinstance(best_score, Mapping):
            raise ValueError("TMRBCD optimizer_state best_score must be an object or null")

        def rng_state(value: Any, label: str) -> Any:
            if not isinstance(value, list):
                raise ValueError(f"TMRBCD optimizer_state {label} must be an array")
            try:
                parsed = _rng_state_from_json(value)
                probe = random.Random()
                probe.setstate(parsed)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"TMRBCD optimizer_state {label} is invalid") from exc
            return parsed

        parsed_rng_state = rng_state(payload["rng_state"], "rng_state")
        parsed_restart_rng_state = rng_state(
            payload["restart_rng_state"], "restart_rng_state"
        )

        restart_generated_raw = payload["restart_generated"]
        if not isinstance(restart_generated_raw, list):
            raise ValueError("TMRBCD optimizer_state restart_generated must be an array")
        restart_generated = {
            point(item, "restart_generated point") for item in restart_generated_raw
        }
        if len(restart_generated) != len(restart_generated_raw):
            raise ValueError("TMRBCD optimizer_state restart_generated contains duplicates")

        cache_raw = payload["cache"]
        if not isinstance(cache_raw, list):
            raise ValueError("TMRBCD optimizer_state cache must be an array")
        eval_cache: dict[TickPoint, float] = {}
        for item in cache_raw:
            if not isinstance(item, Mapping) or set(item) != {"point", "loss"}:
                raise ValueError("TMRBCD optimizer_state cache entry has invalid fields")
            cache_point = point(item["point"], "cache point")
            loss = item["loss"]
            if isinstance(loss, bool) or not isinstance(loss, (int, float)):
                raise ValueError("TMRBCD optimizer_state cache loss must be numeric")
            if cache_point in eval_cache:
                raise ValueError("TMRBCD optimizer_state cache contains duplicate points")
            eval_cache[cache_point] = float(loss)
        if len(eval_cache) != step:
            raise ValueError(
                "TMRBCD optimizer_state step does not match the exact evaluation cache"
            )
        if step and (
            best_point not in eval_cache
            or float(best_loss) != min(eval_cache.values())
            or eval_cache[best_point] != float(best_loss)
        ):
            raise ValueError("TMRBCD optimizer_state best point/loss does not match cache")

        # Optional saturation state: old checkpoints restore a fresh,
        # unsaturated round with defaults.
        block_last_gain = {index: 0.0 for index in range(self.lattice.dim)}
        block_saturation = {index: 0 for index in range(self.lattice.dim)}
        block_pass_start: dict[int, float] = {}
        if "block_last_gain" in payload:
            block_last_gain = {
                index: float(value)
                for index, value in _parse_block_map(
                    payload["block_last_gain"],
                    self.lattice.dim,
                    "block_last_gain",
                    integer=False,
                ).items()
            }
        if "block_saturation" in payload:
            block_saturation = {
                index: int(value)
                for index, value in _parse_block_map(
                    payload["block_saturation"],
                    self.lattice.dim,
                    "block_saturation",
                    integer=True,
                ).items()
            }
        if "block_pass_start" in payload:
            block_pass_start = {
                index: float(value)
                for index, value in _parse_block_map(
                    payload["block_pass_start"],
                    self.lattice.dim,
                    "block_pass_start",
                    integer=False,
                ).items()
            }

        self.step = step
        self.best_loss = float(best_loss)
        self.best_objective = float(best_objective)
        self.active_axis_idx = active_axis_idx
        self.best_point = best_point
        self.current_point = current_point
        self.rng.setstate(parsed_rng_state)
        self.restart.rng.setstate(parsed_restart_rng_state)
        self.restart.generated = restart_generated
        self.eval_cache = eval_cache
        self.best_score = dict(best_score) if best_score is not None else None
        self.pending_asks = []
        self.pending_block_tags = []
        self.block_last_gain = block_last_gain
        self.block_saturation = block_saturation
        self.block_pass_start = block_pass_start


__all__ = [
    "RestartAllocator",
    "TMRBCD_STATE_FIELDS",
    "TMRBCD_STATE_OPTIONAL_FIELDS",
    "TMRBCD_STATE_SCHEMA_VERSION",
    "TmrbcdOptimizer",
]
