"""Deterministic Nevergrad TwoPointsDE adapter over the exact parameter lattice."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import nevergrad as ng
import numpy as np

from .lattice import LatticeSpec, TickPoint

NEVERGRAD_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _CompletedBatch:
    ticks: tuple[TickPoint, ...]
    losses: tuple[float, ...]


class NevergradTwoPointsDEOptimizer:
    """Expose TwoPointsDE through the optimizer runtime's batched ask/tell contract.

    Checkpoints contain deterministic ask/tell history rather than pickle. Restore
    rebuilds Nevergrad from the pinned version and verifies every regenerated tick.
    """

    def __init__(
        self,
        lattice: LatticeSpec,
        *,
        initial_params: Sequence[float],
        budget: int,
        seed: int,
        num_workers: int,
    ) -> None:
        if lattice.dim < 1:
            raise ValueError("Nevergrad optimization requires at least one lattice axis")
        if budget < 1:
            raise ValueError("Nevergrad optimization budget must be positive")
        if num_workers < 1:
            raise ValueError("Nevergrad num_workers must be positive")
        self.lattice = lattice
        self.initial_params = tuple(float(value) for value in initial_params)
        self.budget = int(budget)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.step = 0
        self.best_loss = float("inf")
        self.best_objective = float("-inf")
        self.best_params = list(self.initial_params)
        self.best_score: dict[str, Any] | None = None
        self._history: list[_CompletedBatch] = []
        self._pending: list[tuple[Any, TickPoint]] = []
        self._optimizer = self._new_optimizer()

    def _new_optimizer(self) -> Any:
        initial_ticks = np.asarray(self.lattice.encode(self.initial_params), dtype=float)
        lower = np.asarray([axis.min_tick for axis in self.lattice.axes], dtype=float)
        upper = np.asarray([axis.max_tick for axis in self.lattice.axes], dtype=float)
        parametrization = ng.p.Array(init=initial_ticks, lower=lower, upper=upper)
        parametrization.set_integer_casting()
        optimizer = ng.optimizers.TwoPointsDE(
            parametrization=parametrization,
            budget=self.budget,
            num_workers=self.num_workers,
        )
        optimizer.parametrization.random_state.seed(self.seed)
        return optimizer

    @staticmethod
    def _candidate_ticks(candidate: Any) -> TickPoint:
        values = np.asarray(candidate.value).reshape(-1)
        return tuple(int(value) for value in values)

    def ask(self, batch_size: int = 1) -> list[list[float]]:
        """Generate one bounded batch; a previous batch must be told first."""
        if self._pending:
            raise RuntimeError("Nevergrad ask requires the previous batch to be told first")
        if batch_size < 1:
            return []
        count = min(int(batch_size), self.budget - self.step)
        for _ in range(count):
            candidate = self._optimizer.ask()
            ticks = self._candidate_ticks(candidate)
            self.lattice.validate(ticks)
            self._pending.append((candidate, ticks))
        return [self.lattice.decode(ticks) for _, ticks in self._pending]

    def tell(
        self,
        points: Sequence[Sequence[float]],
        losses: Sequence[float],
        *,
        objectives: Sequence[float] | None = None,
        scores: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Tell exactly the outstanding batch while preserving candidate identity."""
        expected = len(self._pending)
        if len(points) != expected or len(losses) != expected:
            raise ValueError("Nevergrad tell must exactly match the pending batch")
        if objectives is not None and len(objectives) != expected:
            raise ValueError("Nevergrad tell objectives must exactly match the pending batch")
        if scores is not None and len(scores) != expected:
            raise ValueError("Nevergrad tell scores must exactly match the pending batch")

        expected_ticks = tuple(ticks for _, ticks in self._pending)
        observed_ticks = tuple(self.lattice.encode(point) for point in points)
        if observed_ticks != expected_ticks:
            raise ValueError("Nevergrad tell points do not match the pending candidates")

        normalized_losses = tuple(float(loss) for loss in losses)
        for index, ((candidate, ticks), loss) in enumerate(
            zip(self._pending, normalized_losses, strict=True)
        ):
            self._optimizer.tell(candidate, loss)
            objective = (
                float(objectives[index]) if objectives is not None else -loss
            )
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_objective = objective
                self.best_params = self.lattice.decode(ticks)
                self.best_score = (
                    deepcopy(dict(scores[index])) if scores is not None else None
                )

        self._history.append(_CompletedBatch(expected_ticks, normalized_losses))
        self.step += expected
        self._pending = []

    def snapshot(self) -> dict[str, Any]:
        """Return safe, portable replay state only at a completed batch boundary."""
        if self._pending:
            raise RuntimeError("cannot checkpoint Nevergrad with an outstanding ask batch")
        return {
            "schema_version": NEVERGRAD_STATE_SCHEMA_VERSION,
            "algorithm": "nevergrad_two_points_de",
            "seed": self.seed,
            "budget": self.budget,
            "num_workers": self.num_workers,
            "step": self.step,
            "history": [
                {
                    "ticks": [list(point) for point in batch.ticks],
                    "losses": list(batch.losses),
                }
                for batch in self._history
            ],
            "best_loss": self.best_loss,
            "best_objective": self.best_objective,
            "best_params": list(self.best_params),
            "best_score": deepcopy(self.best_score),
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        """Rebuild deterministic optimizer state from safe ask/tell history."""
        if payload.get("schema_version") != NEVERGRAD_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported Nevergrad optimizer_state schema_version")
        for field, expected in (
            ("algorithm", "nevergrad_two_points_de"),
            ("seed", self.seed),
            ("budget", self.budget),
            ("num_workers", self.num_workers),
        ):
            if payload.get(field) != expected:
                raise ValueError(f"Nevergrad optimizer_state {field} does not match run")
        history = payload.get("history")
        if not isinstance(history, list):
            raise ValueError("Nevergrad optimizer_state history must be an array")

        self._optimizer = self._new_optimizer()
        self._pending = []
        self._history = []
        self.step = 0
        for raw_batch in history:
            if not isinstance(raw_batch, Mapping):
                raise ValueError("Nevergrad optimizer_state history entry must be an object")
            raw_ticks = raw_batch.get("ticks")
            raw_losses = raw_batch.get("losses")
            if not isinstance(raw_ticks, list) or not isinstance(raw_losses, list):
                raise ValueError("Nevergrad optimizer_state history entry is malformed")
            generated = self.ask(len(raw_ticks))
            generated_ticks = tuple(self.lattice.encode(point) for point in generated)
            expected_ticks = tuple(tuple(int(value) for value in point) for point in raw_ticks)
            if generated_ticks != expected_ticks:
                raise ValueError("Nevergrad optimizer_state does not reproduce its ask history")
            if len(raw_losses) != len(raw_ticks):
                raise ValueError("Nevergrad optimizer_state history loss count is invalid")
            losses = tuple(float(loss) for loss in raw_losses)
            for (candidate, _), loss in zip(self._pending, losses, strict=True):
                self._optimizer.tell(candidate, loss)
            self._history.append(_CompletedBatch(expected_ticks, losses))
            self.step += len(expected_ticks)
            self._pending = []

        if payload.get("step") != self.step or self.step > self.budget:
            raise ValueError("Nevergrad optimizer_state step does not match history")
        self.best_loss = float(payload["best_loss"])
        self.best_objective = float(payload["best_objective"])
        best_params = payload.get("best_params")
        if not isinstance(best_params, list):
            raise ValueError("Nevergrad optimizer_state best_params must be an array")
        self.best_params = [float(value) for value in best_params]
        best_score = payload.get("best_score")
        if best_score is not None and not isinstance(best_score, Mapping):
            raise ValueError("Nevergrad optimizer_state best_score must be an object or null")
        self.best_score = deepcopy(dict(best_score)) if best_score is not None else None


__all__ = ["NEVERGRAD_STATE_SCHEMA_VERSION", "NevergradTwoPointsDEOptimizer"]
