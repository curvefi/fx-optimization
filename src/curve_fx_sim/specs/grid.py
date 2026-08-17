"""Frozen grid specification contract, coordinate expansion, and loader."""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .common import (
    SpecError,
    assert_contained_path,
    canonical_decimal,
    canonical_primitive,
    format_exact_decimal,
    repository_relative,
    repository_root,
    serializable,
)


@dataclass(frozen=True)
class AxisTarget:
    """Mapping from display coordinate value to pool/policy override key path."""

    path: tuple[str, ...]
    scale: Decimal = Decimal("1")
    display_scale: Decimal = Decimal("1")
    kind: str = "decimal"  # decimal, integer, bps, string

    def transform_value(self, value: Decimal) -> Any:
        """Convert a display decimal value into pool override representation."""
        scaled = (value * self.scale) / self.display_scale
        if self.kind == "integer" or self.kind == "bps":
            return int(scaled)
        if self.kind == "decimal":
            if scaled == scaled.to_integral():
                return int(scaled)
            return format_exact_decimal(scaled)
        return str(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "scale": format_exact_decimal(self.scale),
            "display_scale": format_exact_decimal(self.display_scale),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class AxisSpec:
    """Specification of a single grid axis or coupled multi-parameter axis."""

    name: str
    values: tuple[Decimal, ...] = ()
    targets: tuple[AxisTarget, ...] = ()
    names: tuple[str, ...] = ()  # For coupled axes
    rows: tuple[tuple[Decimal, ...], ...] = ()  # For coupled axes
    generation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name and not self.names:
            raise SpecError("axis must have a name or names")
        if self.rows and self.values:
            raise SpecError("coupled rows and scalar values are mutually exclusive")
        if not self.rows and not self.values:
            raise SpecError(f"axis {self.name} has no values or rows")
        if self.rows:
            num_cols = len(self.names)
            if num_cols == 0:
                raise SpecError("coupled axis rows require non-empty names")
            if self.targets and len(self.targets) != num_cols:
                raise SpecError(
                    f"coupled axis {self.name} has {len(self.targets)} targets but {num_cols} names"
                )
            for r_idx, row in enumerate(self.rows):
                if len(row) != num_cols:
                    raise SpecError(
                        f"coupled axis {self.name} row {r_idx} has width {len(row)}, expected {num_cols}"
                    )

    @property
    def is_coupled(self) -> bool:
        return bool(self.rows)

    @property
    def size(self) -> int:
        return len(self.rows) if self.is_coupled else len(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": [format_exact_decimal(v) for v in self.values],
            "targets": [t.to_dict() for t in self.targets],
            "names": list(self.names),
            "rows": [[format_exact_decimal(v) for v in r] for r in self.rows],
            "generation": serializable(self.generation),
        }


@dataclass(frozen=True)
class GridSpec:
    """Immutable grid exploration specification."""

    id: str
    pair_id: str
    policy_id: str | None = None
    axes: tuple[AxisSpec, ...] = ()
    static_overrides: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    source_path: Path | None = None
    # Legacy grid builder semantics: when true, every cell's out_fee follows
    # its mid_fee (out_fee = mid_fee), mirroring generate_pools_nd.py
    # fee_equalize handling.
    fee_equalize: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise SpecError("grid id must be non-empty")
        if not self.pair_id:
            raise SpecError("grid pair_id must be non-empty")
        if not self.axes:
            raise SpecError("grid must specify at least one axis")

    @property
    def pool_count(self) -> int:
        """Total number of evaluation points in the Cartesian grid."""
        count = 1
        for axis in self.axes:
            count *= axis.size
        return count

    @property
    def coordinate_shape(self) -> tuple[int, ...]:
        """Dimensions of the grid hypercube."""
        return tuple(axis.size for axis in self.axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pair_id": self.pair_id,
            "policy_id": self.policy_id,
            "axes": [a.to_dict() for a in self.axes],
            "static_overrides": serializable(self.static_overrides),
            "tags": list(self.tags),
            "fee_equalize": self.fee_equalize,
            "source_path": self.source_path.as_posix() if self.source_path else None,
            "pool_count": self.pool_count,
            "coordinate_shape": list(self.coordinate_shape),
        }




def _parse_target(data: Any, default_name: str) -> AxisTarget:
    if isinstance(data, str):
        path = tuple(data.split("."))
        return AxisTarget(path=path)
    if isinstance(data, Mapping):
        path_raw = data.get("path") or data.get("field") or default_name
        path = tuple(path_raw.split(".")) if isinstance(path_raw, str) else tuple(str(p) for p in path_raw)
        scale = canonical_decimal(data.get("scale", "1"), label="scale")
        display_scale = canonical_decimal(data.get("display_scale", "1"), label="display_scale")
        kind = data.get("kind", data.get("type", "decimal"))
        return AxisTarget(path=path, scale=scale, display_scale=display_scale, kind=kind)
    return AxisTarget(path=tuple(default_name.split(".")))


def _generate_range_values(rng: Mapping[str, Any], axis_name: str) -> list[Decimal]:
    start = canonical_decimal(rng["start"], label=f"{axis_name} range.start")
    stop = canonical_decimal(rng["stop"], label=f"{axis_name} range.stop")
    num = int(rng.get("num", rng.get("steps", 10)))
    scale_type = rng.get("scale", rng.get("type", "linear")).lower()

    if num < 1:
        raise SpecError(f"axis {axis_name} range num must be >= 1, got {num}")
    if num == 1:
        return [start]

    if scale_type == "linear":
        step = (stop - start) / Decimal(num - 1)
        return [start + step * Decimal(i) for i in range(num)]
    elif scale_type in ("log", "logarithmic"):
        if start <= 0 or stop <= 0:
            raise SpecError(f"axis {axis_name} log range requires strictly positive start and stop")
        log_start = math.log10(float(start))
        log_stop = math.log10(float(stop))
        step = (log_stop - log_start) / (num - 1)
        values: list[Decimal] = []
        for i in range(num):
            if i == 0:
                values.append(start)
            elif i == num - 1:
                values.append(stop)
            else:
                raw_val = 10 ** (log_start + step * i)
                # Form canonical Decimal with up to 10 decimal digits
                values.append(Decimal(f"{raw_val:.10g}"))
        return values
    else:
        raise SpecError(f"unsupported range scale: {scale_type!r}")


def _parse_axis(data: Mapping[str, Any], index: int) -> AxisSpec:
    name = data.get("name", f"axis_{index}")
    targets_raw = data.get("targets", data.get("target"))
    targets: list[AxisTarget] = []

    if targets_raw is not None:
        if isinstance(targets_raw, list):
            targets = [_parse_target(t, name) for t in targets_raw]
        else:
            targets = [_parse_target(targets_raw, name)]
    elif "field" in data or "path" in data:
        targets = [_parse_target(data, name)]
    else:
        targets = [AxisTarget(path=tuple(name.split(".")))]

    # Values
    values: list[Decimal] = []
    rows: list[tuple[Decimal, ...]] = []
    names: list[str] = []

    if "rows" in data:
        names = list(data.get("names", []))
        for row in data["rows"]:
            rows.append(tuple(canonical_decimal(v, label=f"axis {name} row") for v in row))
    elif "values" in data:
        for v in data["values"]:
            values.append(canonical_decimal(v, label=f"axis {name} value"))
    elif "range" in data:
        values = _generate_range_values(data["range"], name)
    else:
        raise SpecError(f"axis {name} has no values, rows, or range specified")

    generation = dict(data.get("generation", {}))

    return AxisSpec(
        name=name,
        values=tuple(values),
        targets=tuple(targets),
        names=tuple(names),
        rows=tuple(rows),
        generation=generation,
    )


def load_grid_spec(
    path_or_id: str | os.PathLike[str],
    *,
    repository: Path | None = None,
) -> GridSpec:
    """Load and validate a grid exploration TOML specification."""
    root = repository.resolve() if repository is not None else repository_root()
    candidate = Path(path_or_id)

    if not candidate.is_file():
        search_paths = [
            root / "configs" / "grids" / f"{path_or_id}.toml",
            root / "configs" / f"{path_or_id}.toml",
            root / "grids" / f"{path_or_id}.toml",
        ]
        found = None
        for p in search_paths:
            if p.is_file():
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Grid specification not found: {path_or_id}")
        candidate = found

    assert_contained_path(candidate, root, allow_symlinks=True)

    with candidate.open("rb") as stream:
        raw_data = tomllib.load(stream)

    grid_data = raw_data.get("grid", raw_data)

    grid_id = grid_data.get("id") or candidate.stem
    pair_id = grid_data.get("pair_id") or grid_data.get("pair", "")
    policy_id = grid_data.get("policy_id") or grid_data.get("policy")

    axes_raw = grid_data.get("axes", [])
    axes: list[AxisSpec] = []
    for idx, axis_data in enumerate(axes_raw):
        if isinstance(axis_data, Mapping):
            axes.append(_parse_axis(axis_data, idx))

    static_overrides = dict(grid_data.get("static_overrides", {}))
    tags = tuple(grid_data.get("tags", []))
    fee_equalize = bool(grid_data.get("fee_equalize", False))
    source_path = repository_relative(candidate, root)

    return GridSpec(
        id=grid_id,
        pair_id=pair_id,
        policy_id=policy_id,
        axes=tuple(axes),
        static_overrides=static_overrides,
        tags=tags,
        fee_equalize=fee_equalize,
        source_path=source_path,
    )


__all__ = [
    "AxisTarget",
    "AxisSpec",
    "GridSpec",
    "load_grid_spec",
]
