"""Portable candidate values shared by grids and adaptive search."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


class CandidateError(ValueError):
    """Raised when a candidate payload cannot be represented canonically."""


def path_parts(path: str) -> tuple[str, ...]:
    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        raise CandidateError(f"invalid dotted payload path: {path!r}")
    return tuple(path.split("."))


def merge_payload(defaults: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy defaults and apply flat or dotted semantic updates."""
    if not isinstance(defaults, Mapping) or not isinstance(updates, Mapping):
        raise CandidateError("candidate payloads must be mappings")
    result = deepcopy(dict(defaults))
    for key, value in updates.items():
        parts = path_parts(key)
        if len(parts) == 1:
            result[key] = deepcopy(value)
            continue
        cursor: Any = result
        for index, part in enumerate(parts[:-1]):
            next_part = parts[index + 1]
            if isinstance(cursor, dict):
                if part not in cursor:
                    cursor[part] = [] if next_part.isdigit() else {}
                cursor = cursor[part]
            elif isinstance(cursor, list) and part.isdigit():
                position = int(part)
                if position >= len(cursor):
                    raise CandidateError(f"list index out of range in {key!r}")
                cursor = cursor[position]
            else:
                raise CandidateError(f"dotted path collides at {part!r} in {key!r}")
        leaf = parts[-1]
        if isinstance(cursor, dict):
            cursor[leaf] = deepcopy(value)
        elif isinstance(cursor, list) and leaf.isdigit():
            position = int(leaf)
            if position >= len(cursor):
                raise CandidateError(f"list index out of range in {key!r}")
            cursor[position] = deepcopy(value)
        else:
            raise CandidateError(f"dotted path collides at {key!r}")
    return result


def _validate_number(value: int | float | Decimal) -> None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CandidateError(f"non-finite numeric value: {value!r}") from exc
    if not number.is_finite():
        raise CandidateError(f"non-finite numeric value: {value!r}")


def _validate_json_value(value: object) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        _validate_number(value)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CandidateError("candidate payload keys must be strings")
        for item in value.values():
            _validate_json_value(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)
        return
    raise CandidateError(f"unsupported candidate value: {type(value).__name__}")


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a payload while validating that it contains JSON-like values."""
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise CandidateError("candidate payload must be a mapping with string keys")
    result = deepcopy(dict(payload))
    _validate_json_value(result)
    return result


def candidate_id(ordinal: int) -> str:
    """Return one readable run-local ordinal ID."""
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise CandidateError("candidate ordinal must be a non-negative integer")
    return f"p{ordinal:08d}"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One evaluator-independent candidate and its deterministic identity."""

    candidate_id: str
    payload: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, ordinal: int = 0) -> "CandidateSpec":
        copied = canonical_payload(payload)
        return cls(candidate_id(ordinal), copied)

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "payload": dict(self.payload)}


def compile_batch(
    proposals: Iterable[Mapping[str, Any]],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> tuple[CandidateSpec, ...]:
    """Compile an adaptive optimizer's proposals into one bounded batch.

    Only the requested iterable is consumed; callers can pass a generator from an
    optimizer without constructing a full search space.
    """
    base = {} if defaults is None else canonical_payload(defaults)
    result: list[CandidateSpec] = []
    for ordinal, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise CandidateError("each proposal must be a mapping")
        result.append(CandidateSpec.from_payload(merge_payload(base, proposal), ordinal=ordinal))
    return tuple(result)


__all__ = [
    "CandidateError",
    "CandidateSpec",
    "candidate_id",
    "canonical_payload",
    "compile_batch",
    "merge_payload",
    "path_parts",
]
