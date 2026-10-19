#!/usr/bin/env python3
"""Rank a completed Cartesian grid by one stored metric, per cell or over an exact axial star.

  uv run --no-sync python scripts/analyze_basins.py runs/RUN --metric yb_apy_gm30 \\
      --max-price-diff-bps 1000 --final-price-diff-bps 100 [--robust AXIS=RADIUS ...]

Higher metric values rank first. A row is eligible when it is ok, its value is at least --min, its
max_7d_rel_price_diff is at most --max-price-diff-bps and its |final_rel_price_diff| at most
--final-price-diff-bps. With robustness radii (run metadata, --robustness-config or --robust) the
ranked value is the minimum over the axial star and each price limit applies to the star maximum.
The basin is every eligible row within --gap of the best.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from fxopt.results import read_result_columns
from fxopt.robustness import (
    RobustnessAxis,
    load_robustness_axes,
    parse_robustness_axes,
    score_robustness,
)

PRICE_LIMITS = (
    ("max_price_diff_bps", "max_7d_rel_price_diff"),
    ("final_price_diff_bps", "final_rel_price_diff"),
)


def _leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaves(child, name)
    else:
        yield prefix, value


def _candidate_parameters(pool: dict[str, Any], policy: tuple[float, ...]) -> str:
    fields = [f"pool.{name}={value!r}" for name, value in _leaves(pool)]
    fields.extend(f"policy[{index}]={value:g}" for index, value in enumerate(policy))
    return " ".join(fields) or "(no parameters)"


def _coordinates(ordinal: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates: list[int] = []
    remainder = ordinal
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder //= size
    return tuple(reversed(coordinates))


def _axis_distributions(
    axes: dict[str, list[Any]],
    shape: tuple[int, ...],
    ordinals: np.ndarray,
    basin: np.ndarray,
    gap: float,
) -> None:
    print(f"\nBASIN DISTRIBUTIONS (rank gap <= {gap:g}; points={int(basin.sum())})")
    if (
        not axes
        or len(shape) != len(axes)
        or any(
            len(values) != size
            for values, size in zip(axes.values(), shape, strict=True)
        )
    ):
        print("  unavailable: run.json has no axes matching shape")
        return
    for dimension, (axis, values) in enumerate(axes.items()):
        counts = [0] * len(values)
        for ordinal in ordinals[basin].tolist():
            counts[_coordinates(int(ordinal), shape)[dimension]] += 1
        distribution = ", ".join(
            f"{value!r}:{count}"
            for value, count in zip(values, counts, strict=True)
            if count
        )
        print(f"  {axis}: {distribution}")


def _cli_robustness(values: list[str]) -> tuple[RobustnessAxis, ...]:
    raw: dict[str, float] = {}
    for value in values:
        name, separator, radius_text = value.rpartition("=")
        if not separator or not name:
            raise ValueError(f"invalid robustness radius {value!r}; use AXIS=RADIUS")
        try:
            radius = float(radius_text)
        except ValueError as exc:
            raise ValueError(f"invalid robustness radius {value!r}") from exc
        if name in raw:
            raise ValueError(f"duplicate robustness axis {name!r}")
        raw[name] = radius
    return parse_robustness_axes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metric", default="yb_apy_gm30", help="stored metric to rank, higher first")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--gap", type=float, default=0.001, help="basin width in metric units")
    parser.add_argument("--min", dest="minimum", type=float, default=-math.inf)
    parser.add_argument("--max-price-diff-bps", type=float, default=math.inf)
    parser.add_argument("--final-price-diff-bps", type=float, default=math.inf)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--robustness-config",
        type=Path,
        help="override run metadata with a TOML [robustness] table",
    )
    source.add_argument(
        "--robust",
        action="append",
        default=[],
        metavar="AXIS=RADIUS",
        help="override run metadata; repeat for each exact axial radius",
    )
    args = parser.parse_args()
    limits = {name: getattr(args, name) for name, _metric in PRICE_LIMITS}
    if (
        args.top < 1
        or not math.isfinite(args.gap)
        or args.gap < 0
        or math.isnan(args.minimum)
        or any(math.isnan(value) or value < 0 for value in limits.values())
    ):
        parser.error("limits must be non-negative; --top must be positive")
    masks = [(metric, limits[name] / 10_000.0) for name, metric in PRICE_LIMITS if math.isfinite(limits[name])]

    try:
        columns = read_result_columns(args.run_dir, metrics=[args.metric, *(m for m, _ in masks)])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(f"could not read result artifact: {exc}")
    count = columns.row_count
    metadata = dict(columns.metadata)
    ordinals = columns.ordinals
    raw_axes = metadata.get("axes", {})
    raw_shape = metadata.get("shape", [])
    axes = dict(raw_axes) if isinstance(raw_axes, dict) else {}
    shape = tuple(int(size) for size in raw_shape) if isinstance(raw_shape, list) else ()

    try:
        if args.robustness_config is not None:
            specs = load_robustness_axes(args.robustness_config)
        elif args.robust:
            specs = _cli_robustness(args.robust)
        else:
            specs = parse_robustness_axes(metadata.get("robustness"), required=False)
    except (OSError, ValueError) as exc:
        parser.error(f"could not load robustness radii: {exc}")

    def robust(values: np.ndarray, sign: float):
        points = sign * values
        points[~(columns.ok_mask & np.isfinite(values))] = np.nan
        return score_robustness(
            point_scores=points, ordinals=ordinals, axes=axes, shape=shape, robustness_axes=specs
        )

    point = np.asarray(columns.metrics[args.metric], dtype=float)
    ceilings = {metric: np.abs(np.asarray(columns.metrics[metric], dtype=float)) for metric, _ in masks}
    ranking = None
    try:
        if specs:
            ranking = robust(point.copy(), 1.0)
            rank_values = ranking.robust_score
            eligible = ranking.complete.copy()
            for metric, values in ceilings.items():
                worst = robust(values.copy(), -1.0)
                eligible &= worst.complete
                ceilings[metric] = -worst.robust_score
        else:
            rank_values = point
            eligible = columns.ok_mask & np.isfinite(point)
    except ValueError as exc:
        parser.error(f"could not score robustness: {exc}")
    eligible &= np.isfinite(rank_values) & (rank_values >= args.minimum)
    for metric, limit in masks:
        eligible &= ceilings[metric] <= limit
    if not np.any(eligible):
        suffix = " with complete robustness coverage" if specs else ""
        parser.error(f"no eligible rows{suffix} after filtering")

    eligible_rows = np.flatnonzero(eligible)
    ordered = eligible_rows[np.argsort(-rank_values[eligible_rows], kind="stable")]
    best = float(rank_values[ordered[0]])
    basin = eligible & (rank_values >= best - args.gap)
    print(f"RUN {columns.run_id}: {int(eligible.sum())}/{count} eligible; best {args.metric}={best:.8g}")
    if specs:
        radii = ", ".join(f"{axis.name}=+/-{axis.radius:g}" for axis in specs)
        print(f"ROBUSTNESS exact axial cross; reducer=min; {radii}")
    filters = [f"{args.metric} >= {args.minimum:g}"] if math.isfinite(args.minimum) else []
    filters.extend(f"|{metric}| <= {limit * 10_000.0:g} bps" for metric, limit in masks)
    if filters:
        print("FILTER " + "; ".join(filters))

    print("\nBEST ROWS (human review; no automatic optimization)")
    for row in ordered[: args.top]:
        candidate = columns.candidate_at(int(ordinals[row]))
        params = _candidate_parameters(dict(candidate.pool_overrides), candidate.policy_params)
        fields = f" {args.metric}={rank_values[row]:.8g}"
        if ranking is not None:
            fields += (
                f" point={point[row]:.8g} regret={point[row] - rank_values[row]:.8g}"
                f" n={int(ranking.member_count[row])} worst={int(ranking.worst_ordinal[row])}"
            )
        fields += "".join(f" {metric}={ceilings[metric][row]:.8g}" for metric, _ in masks)
        print(f"  ordinal={int(ordinals[row])}{fields} {params}")

    _axis_distributions(axes, shape, ordinals, basin, args.gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
