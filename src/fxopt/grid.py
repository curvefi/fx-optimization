"""Lazy Cartesian candidate generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from .candidates import CandidateSpec, candidate_id, canonical_payload, merge_payload


@dataclass(frozen=True, slots=True)
class CartesianGrid:
    """A Cartesian product that materializes only candidates being consumed."""

    defaults: Mapping[str, Any]
    axes: Mapping[str, Sequence[Any]]
    _names: tuple[str, ...] = field(init=False, repr=False)
    _values: tuple[tuple[Any, ...], ...] = field(init=False, repr=False)
    _shape: tuple[int, ...] = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.axes):
            raise ValueError("grid axis names must be non-empty strings")
        if any(not values for values in self.axes.values()):
            raise ValueError("grid axes must contain at least one value")
        defaults = canonical_payload(self.defaults)
        axes = {
            name: tuple(
                canonical_payload({"value": value})["value"]
                for value in values
            )
            for name, values in self.axes.items()
        }
        names = tuple(sorted(axes))
        values = tuple(axes[name] for name in names)
        shape = tuple(len(axis) for axis in values)
        size = 1
        for length in shape:
            size *= length
        object.__setattr__(self, "defaults", defaults)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "_size", size)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[CandidateSpec]:
        for ordinal, point in enumerate(product(*self._values)):
            updates: dict[str, Any] = {}
            for name, value in zip(self._names, point, strict=True):
                if isinstance(value, Mapping):
                    updates.update(value)
                else:
                    updates[name] = value
            yield CandidateSpec(
                candidate_id(ordinal),
                merge_payload(self.defaults, updates),
            )

    def candidate_at(self, ordinal: int) -> CandidateSpec:
        """Resolve one product position without constructing preceding candidates."""
        if ordinal < 0 or ordinal >= len(self):
            raise IndexError(ordinal)
        coordinates: list[Any] = []
        remainder = ordinal
        for values in reversed(self._values):
            remainder, index = divmod(remainder, len(values))
            coordinates.append(values[index])
        updates: dict[str, Any] = {}
        for name, value in zip(self._names, reversed(coordinates), strict=True):
            if isinstance(value, Mapping):
                updates.update(value)
            else:
                updates[name] = value
        return CandidateSpec(
            candidate_id(ordinal),
            merge_payload(self.defaults, updates),
        )


__all__ = ["CartesianGrid"]
