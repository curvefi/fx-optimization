"""Deterministic block-cyclic shard assignment for grid execution.

Finite parameter grid searches partition the total pool space into contiguous
chunks assigned in round-robin order across available blades. This ensures
balanced load distribution even when candidate pool runtime varies across parameter space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ShardAssignment:
    """One blade's assigned pool ranges for a partitioned run."""

    shard_id: str
    shard_index: int
    blade: str
    ranges: tuple[tuple[int, int], ...]
    chunk_size: int
    total_pools: int

    @property
    def total_assigned_pools(self) -> int:
        """Total number of pools covered by this assignment's ranges."""
        return sum(end - start for start, end in self.ranges)

    def to_dict(self) -> dict[str, Any]:
        """Serialize assignment for inclusion in run manifests."""
        return {
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "blade": self.blade,
            "ranges": [list(r) for r in self.ranges],
            "chunk_size": self.chunk_size,
            "total_pools": self.total_pools,
            "assigned_pools": self.total_assigned_pools,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShardAssignment:
        raw_ranges = data.get("ranges", ())
        ranges_tuple = tuple((int(r[0]), int(r[1])) for r in raw_ranges)
        return cls(
            shard_id=str(data.get("shard_id", "")),
            shard_index=int(data.get("shard_index", 0)),
            blade=str(data.get("blade", "")),
            ranges=ranges_tuple,
            chunk_size=int(data.get("chunk_size", 2048)),
            total_pools=int(data.get("total_pools", 0)),
        )


def make_assignments(
    n_pools: int,
    blades: Sequence[str],
    chunk_size: int = 2048,
    run_id: str | None = None,
) -> list[ShardAssignment]:
    """Generate one immutable assignment per chunk, round-robin across blades.

    A blade is an execution resource, not a shard. Keeping every chunk as its
    own assignment makes resume and collection precise even when all chunks run
    through one local evaluator.
    """
    if not blades:
        raise ValueError("blades sequence must not be empty")
    if n_pools < 0:
        raise ValueError(f"n_pools must be >= 0, got {n_pools}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    assignments: list[ShardAssignment] = []
    for shard_index, start in enumerate(range(0, n_pools, chunk_size)):
        end = min(start + chunk_size, n_pools)
        blade = str(blades[shard_index % len(blades)])
        assignments.append(
            ShardAssignment(
                shard_id=f"shard_{shard_index:03d}_{blade}",
                shard_index=shard_index,
                blade=blade,
                ranges=((start, end),),
                chunk_size=chunk_size,
                total_pools=n_pools,
            )
        )
    return assignments


def format_ranges(ranges: Sequence[tuple[int, int]]) -> str:
    """Format half-open ranges as plain text rows of ``start end\\n``."""
    return "".join(f"{start} {end}\n" for start, end in ranges)


def parse_ranges(text: str) -> list[tuple[int, int]]:
    """Parse plain text rows of ``start end`` into half-open intervals."""
    ranges: list[tuple[int, int]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"malformed range row {line!r}; expected 'start end'")
        start = int(parts[0])
        end = int(parts[1])
        if start >= end:
            raise ValueError(f"invalid range [{start}, {end}): start must be < end")
        ranges.append((start, end))
    return ranges


def write_ranges_file(path: Path, ranges: Sequence[tuple[int, int]]) -> None:
    """Write formatted ranges to a target text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_ranges(ranges), encoding="utf-8")


__all__ = [
    "ShardAssignment",
    "format_ranges",
    "make_assignments",
    "parse_ranges",
    "write_ranges_file",
]
