"""Small, dependency-free contracts shared by optimizer execution surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


_STATUSES = frozenset({"ok", "failed", "cancelled"})


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _copy_json_value(value: Any, *, label: str) -> Any:
    """Copy and validate override values without coupling the contract to Pydantic."""
    if isinstance(value, Mapping):
        return {str(key): _copy_json_value(item, label=label) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item, label=label) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value, label=label)
    raise TypeError(f"{label} contains unsupported value {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One optimizer proposal, independent of grid or search strategy."""

    candidate_id: str
    policy_params: tuple[float, ...] = ()
    pool_overrides: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        params = tuple(
            _finite_number(value, label=f"policy_params[{index}]")
            for index, value in enumerate(self.policy_params)
        )
        overrides = _copy_json_value(self.pool_overrides, label="pool_overrides")
        if not isinstance(overrides, dict):
            raise TypeError("pool_overrides must be a mapping")
        object.__setattr__(self, "policy_params", params)
        object.__setattr__(self, "pool_overrides", MappingProxyType(overrides))

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
    """Compact normalized result returned by :class:`OptimizerEngine`."""

    candidate_id: str
    status: str = "ok"
    metrics: Mapping[str, float] = MappingProxyType({})
    error: str | None = None
    economic_fingerprint: str | None = None
    ordinal: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}")
        if not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        copied = {
            str(name): _finite_number(value, label=f"metrics[{name!r}]")
            for name, value in self.metrics.items()
        }
        object.__setattr__(self, "metrics", MappingProxyType(copied))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "metrics": dict(self.metrics),
            "error": self.error,
            "economic_fingerprint": self.economic_fingerprint,
            "ordinal": self.ordinal,
        }


__all__ = ["Candidate", "CandidateResult"]
