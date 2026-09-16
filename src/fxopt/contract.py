"""Small dependency-free contracts shared by grids and replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Candidate:
    """One materialized pool candidate."""

    candidate_id: str
    policy_params: tuple[float, ...] = ()
    pool_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_params", tuple(self.policy_params))
        object.__setattr__(self, "pool_overrides", deepcopy(dict(self.pool_overrides)))

    def to_dict(self, *, ordinal: int) -> dict[str, Any]:
        """Return the evaluator-client shape, assigning a batch-local ordinal."""
        return {
            "ordinal": ordinal,
            "candidate_id": self.candidate_id,
            "policy_params": list(self.policy_params),
            "pool_overrides": dict(self.pool_overrides),
        }


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Compact normalized result returned by an evaluator session."""

    candidate_id: str
    status: str = "ok"
    metrics: Mapping[str, float] = field(default_factory=dict)
    error: str | None = None
    artifacts: Mapping[str, Any] | None = None
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))
        if self.artifacts is not None:
            object.__setattr__(self, "artifacts", dict(self.artifacts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "metrics": dict(self.metrics),
            "error": self.error,
            "artifacts": None if self.artifacts is None else dict(self.artifacts),
            "ordinal": self.ordinal,
        }


__all__ = ["Candidate", "CandidateResult"]
