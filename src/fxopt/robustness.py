"""Post-grid maximin scoring over exact physical parameter radii."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any

import numpy as np


class RobustnessError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RobustnessAxis:
    name: str
    radius: float
    field: str | None = None


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    robust_score: np.ndarray
    complete: np.ndarray
    member_count: np.ndarray
    worst_ordinal: np.ndarray


def parse_robustness_axes(
    raw: object,
    *,
    required: bool = True,
) -> tuple[RobustnessAxis, ...]:
    """Validate an axis-to-radius mapping from TOML, metadata, or CLI input."""
    if raw is None and not required:
        return ()
    if not isinstance(raw, Mapping) or not raw:
        label = "a non-empty robustness mapping" if required else "a robustness mapping"
        raise RobustnessError(f"expected {label}")
    axes = []
    for name, value in raw.items():
        field = None
        if isinstance(value, Mapping):
            if set(value) - {"field", "radius"}:
                raise RobustnessError(f"robustness axis {name!r} has unknown keys")
            field = value.get("field")
            value = value.get("radius")
        if not isinstance(name, str) or not name:
            raise RobustnessError("robustness axis names must be non-empty strings")
        if field is not None and (not isinstance(field, str) or not field):
            raise RobustnessError(f"robustness axis {name!r} field must be a string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RobustnessError(f"robustness axis {name!r} requires a numeric radius")
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise RobustnessError(f"robustness axis {name!r} radius must be positive")
        axes.append(RobustnessAxis(name, radius, field))
    return tuple(axes)


def load_robustness_axes(path: str | Path) -> tuple[RobustnessAxis, ...]:
    """Read axis = radius entries from a config's [robustness] table."""
    with Path(path).expanduser().open("rb") as stream:
        raw = tomllib.load(stream).get("robustness")
    try:
        return parse_robustness_axes(raw)
    except RobustnessError as exc:
        raise RobustnessError(
            "config requires a non-empty [robustness] table"
        ) from exc


def robustness_metadata(
    axes: Sequence[RobustnessAxis],
) -> dict[str, float | dict[str, str | float]]:
    """Serialize validated radii into the same compact shape accepted by TOML."""
    return {
        axis.name: (
            axis.radius
            if axis.field is None
            else {"field": axis.field, "radius": axis.radius}
        )
        for axis in axes
    }


def _numeric_axis(spec: RobustnessAxis, raw_values: object) -> np.ndarray:
    if not isinstance(raw_values, list) or len(raw_values) < 3:
        raise RobustnessError(f"grid axis {spec.name!r} needs at least three values")
    linked = isinstance(raw_values[0], Mapping)
    if any(isinstance(value, Mapping) != linked for value in raw_values):
        raise RobustnessError(f"grid axis {spec.name!r} mixes scalar and linked values")
    if linked:
        if spec.field is None:
            raise RobustnessError(f"linked robustness axis {spec.name!r} requires field")
        if any(spec.field not in value for value in raw_values):
            raise RobustnessError(
                f"linked robustness axis {spec.name!r} has no field {spec.field!r}"
            )
        raw_values = [value[spec.field] for value in raw_values]
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RobustnessError(f"robustness axis {spec.name!r} must be numeric") from exc
    differences = np.diff(values)
    if np.any(~np.isfinite(values)) or not (
        np.all(differences > 0.0) or np.all(differences < 0.0)
    ):
        raise RobustnessError(f"robustness axis {spec.name!r} must be finite and monotonic")
    return values


def _axis_arms(spec: RobustnessAxis, values: np.ndarray) -> tuple[tuple[int, ...], ...]:
    arms = []
    for center_index, center in enumerate(values):
        endpoints = []
        for target in (center - spec.radius, center + spec.radius):
            tolerance = max(
                1e-15,
                spec.radius * 1e-10,
                2 * math.ulp(float(center)),
                2 * math.ulp(float(target)),
            )
            matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=tolerance))
            if len(matches) != 1:
                endpoints = []
                break
            endpoints.append(int(matches[0]))
        if not endpoints:
            arms.append(())
            continue
        low, high = sorted((endpoints[0], endpoints[1]))
        arms.append(tuple(index for index in range(low, high + 1) if index != center_index))
    return tuple(arms)


def score_robustness(
    *,
    point_scores: Sequence[float],
    ordinals: Sequence[int],
    axes: Mapping[str, Any],
    shape: Sequence[int],
    robustness_axes: Sequence[RobustnessAxis],
) -> RobustnessResult:
    """Take the worst score in each exact two-sided axial cross."""
    if not robustness_axes:
        raise RobustnessError("at least one robustness axis is required")
    shape = tuple(int(size) for size in shape)
    axis_names = tuple(sorted(axes))
    if shape != tuple(len(axes[name]) for name in axis_names):
        raise RobustnessError("grid axes do not match shape")
    count = math.prod(shape)
    ordinal_rows = np.asarray(ordinals)
    score_rows = np.asarray(point_scores, dtype=float)
    if (
        score_rows.shape != (count,)
        or ordinal_rows.shape != (count,)
        or not np.issubdtype(ordinal_rows.dtype, np.integer)
        or not np.array_equal(np.sort(ordinal_rows), np.arange(count))
    ):
        raise RobustnessError("robustness requires one score for every grid ordinal")

    scores = np.empty(count)
    scores[ordinal_rows] = score_rows
    positions = {name: index for index, name in enumerate(axis_names)}
    configured = []
    unmeasurable = []
    for spec in robustness_axes:
        if spec.name not in positions:
            raise RobustnessError(f"robustness axis {spec.name!r} is not in this grid")
        arms = _axis_arms(spec, _numeric_axis(spec, axes[spec.name]))
        if any(arms):
            configured.append((positions[spec.name], arms))
        else:
            unmeasurable.append(f"{spec.name}=+/-{spec.radius:g}")
    if unmeasurable:
        raise RobustnessError(
            "grid cannot measure exact robustness for " + ", ".join(unmeasurable)
        )

    robust = np.full(count, np.nan)
    complete = np.zeros(count, dtype=bool)
    member_count = np.zeros(count, dtype=np.int64)
    worst_ordinal = np.full(count, -1, dtype=np.int64)
    for ordinal in range(count):
        coordinate = list(np.unravel_index(ordinal, shape))
        members = [ordinal]
        for position, arms in configured:
            arm = arms[coordinate[position]]
            if not arm:
                break
            for axis_index in arm:
                neighbor = coordinate.copy()
                neighbor[position] = axis_index
                members.append(int(np.ravel_multi_index(tuple(neighbor), shape)))
        else:
            values = scores[members]
            if np.all(np.isfinite(values)):
                worst = int(np.argmin(values))
                robust[ordinal] = float(values[worst])
                complete[ordinal] = True
                member_count[ordinal] = len(members)
                worst_ordinal[ordinal] = members[worst]

    rows = ordinal_rows.astype(np.int64, copy=False)
    return RobustnessResult(
        robust_score=robust[rows],
        complete=complete[rows],
        member_count=member_count[rows],
        worst_ordinal=worst_ordinal[rows],
    )


__all__ = [
    "RobustnessAxis",
    "RobustnessError",
    "RobustnessResult",
    "load_robustness_axes",
    "parse_robustness_axes",
    "robustness_metadata",
    "score_robustness",
]
