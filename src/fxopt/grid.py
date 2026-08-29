"""Lazy Cartesian candidate generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice, product
from typing import Any

from .candidates import CandidateSpec, merge_payload


@dataclass(frozen=True, slots=True)
class CartesianGrid:
    """A Cartesian product that materializes only candidates being consumed."""

    defaults: Mapping[str, Any]
    axes: Mapping[str, Sequence[Any]]

    def __post_init__(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.axes):
            raise ValueError("grid axis names must be non-empty strings")
        if any(not values for values in self.axes.values()):
            raise ValueError("grid axes must contain at least one value")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(self.axes[name]) for name in sorted(self.axes))

    def __len__(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count

    def __iter__(self) -> Iterator[CandidateSpec]:
        names = tuple(sorted(self.axes))
        values = tuple(self.axes[name] for name in names)
        for ordinal, point in enumerate(product(*values)):
            updates: dict[str, Any] = {}
            for name, value in zip(names, point, strict=True):
                if isinstance(value, Mapping):
                    updates.update(value)
                else:
                    updates[name] = value
            yield CandidateSpec.from_payload(
                merge_payload(self.defaults, updates),
                ordinal=ordinal,
            )

    def iter_batches(self, batch_size: int) -> Iterator[tuple[CandidateSpec, ...]]:
        """Yield bounded immutable batches without materializing the grid."""
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        candidates = iter(self)
        while batch := tuple(islice(candidates, batch_size)):
            yield batch

    def iter_rotating_blocks(
        self, block_size: int, rotations: int
    ) -> Iterator[tuple[int, CandidateSpec]]:
        """Visit distant canonical blocks in a deterministic round-robin order."""
        if isinstance(block_size, bool) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if isinstance(rotations, bool) or rotations < 1:
            raise ValueError("rotations must be a positive integer")
        for block in self._rotating_block_indices(block_size, rotations):
            start = block * block_size
            stop = min(start + block_size, len(self))
            for ordinal in range(start, stop):
                yield ordinal, self.candidate_at(ordinal)

    def iter_stripe(
        self, block_size: int, stripes: int, stripe: int
    ) -> Iterator[tuple[int, CandidateSpec]]:
        """Yield one balanced stripe through the rotating block order."""
        if isinstance(stripe, bool) or not isinstance(stripe, int):
            raise ValueError("stripe must be an integer")
        if stripe < 0 or stripe >= stripes:
            raise ValueError("stripe must be in [0, stripes)")
        position = 0
        for block in self._rotating_block_indices(block_size, stripes):
            start = block * block_size
            stop = min(start + block_size, len(self))
            first = (stripe - position) % stripes
            for ordinal in range(start + first, stop, stripes):
                yield ordinal, self.candidate_at(ordinal)
            position += stop - start

    def _rotating_block_indices(
        self, block_size: int, rotations: int
    ) -> Iterator[int]:
        if isinstance(block_size, bool) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if isinstance(rotations, bool) or rotations < 1:
            raise ValueError("rotations must be a positive integer")
        block_count = (len(self) + block_size - 1) // block_size
        rounds = (block_count + rotations - 1) // rotations
        for round_index in range(rounds):
            for rotation in range(rotations):
                block = rotation * rounds + round_index
                if block < block_count:
                    yield block

    def candidate_at(self, ordinal: int) -> CandidateSpec:
        """Resolve one product position without constructing preceding candidates."""
        if ordinal < 0 or ordinal >= len(self):
            raise IndexError(ordinal)
        names = tuple(sorted(self.axes))
        coordinates: list[Any] = []
        remainder = ordinal
        for name in reversed(names):
            values = self.axes[name]
            remainder, index = divmod(remainder, len(values))
            coordinates.append(values[index])
        updates: dict[str, Any] = {}
        for name, value in zip(names, reversed(coordinates), strict=True):
            if isinstance(value, Mapping):
                updates.update(value)
            else:
                updates[name] = value
        return CandidateSpec.from_payload(
            merge_payload(self.defaults, updates),
            ordinal=ordinal,
        )


def compile_grid(
    defaults: Mapping[str, Any], axes: Mapping[str, Sequence[Any]]
) -> CartesianGrid:
    """Construct a lazy grid from ordinary mappings."""
    return CartesianGrid(dict(defaults), {name: tuple(values) for name, values in axes.items()})


__all__ = ["CartesianGrid", "compile_grid"]
