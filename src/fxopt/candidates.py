"""Portable candidate values shared by grids and adaptive search."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
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
        cursor: dict[str, Any] = result
        for part in parts[:-1]:
            if part not in cursor:
                current = {}
                cursor[part] = current
            else:
                current = cursor[part]
            if not isinstance(current, Mapping):
                raise CandidateError(f"dotted path collides at {part!r} in {key!r}")
            if not isinstance(current, dict):
                current = dict(current)
                cursor[part] = current
            cursor = current
        leaf = parts[-1]
        if leaf in cursor and isinstance(cursor[leaf], Mapping) != isinstance(value, Mapping):
            raise CandidateError(f"dotted path collides at {key!r}")
        cursor[leaf] = deepcopy(value)
    return result


def _number_text(value: int | float | Decimal) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CandidateError(f"non-finite numeric value: {value!r}") from exc
    if not number.is_finite():
        raise CandidateError(f"non-finite numeric value: {value!r}")
    if not number:
        return "0"
    # Fixed notation makes 1, 1.0, and Decimal("1.00") one semantic value.
    text = format(number.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _canonical(value: object) -> object:
    """Encode JSON-like values with explicit primitive types for hashing."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return ["number", _number_text(value)]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CandidateError("candidate payload keys must be strings")
        return ["object", [[key, _canonical(value[key])] for key in sorted(value)]]
    if isinstance(value, (list, tuple)):
        return ["array", [_canonical(item) for item in value]]
    raise CandidateError(f"unsupported candidate value: {type(value).__name__}")


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a payload while validating that it contains JSON-like values."""
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise CandidateError("candidate payload must be a mapping with string keys")
    # The copy keeps the caller's useful numeric types while recursively checking values.
    result = deepcopy(dict(payload))
    _canonical(result)
    return result


def candidate_id(payload: Mapping[str, Any]) -> str:
    """Return the ID derived solely from canonical semantic payload content."""
    encoded = json.dumps(_canonical(payload), separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One evaluator-independent candidate and its deterministic identity."""

    candidate_id: str
    payload: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CandidateSpec":
        copied = canonical_payload(payload)
        return cls(candidate_id(copied), copied)

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
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            raise CandidateError("each proposal must be a mapping")
        result.append(CandidateSpec.from_payload(merge_payload(base, proposal)))
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
