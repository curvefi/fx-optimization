#!/usr/bin/env python3
"""Rank a completed Cartesian grid by point or exact axial-star performance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from fxopt.robustness import (
    RobustnessAxis,
    load_robustness_axes,
    parse_robustness_axes,
    score_robustness,
)
from fxopt.scoring import combined_scores


METRICS = ("apy_net_gm", "yb_apy_gm", "detach_energy_ungated")
RANKINGS = ("score", "gm", "yb-gm")


def _leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaves(child, name)
    else:
        yield prefix, value


def _candidate_parameters(overrides: str, policy: np.ndarray) -> str:
    try:
        pool = json.loads(overrides)
    except (TypeError, json.JSONDecodeError):
        pool = {"<invalid>": overrides}
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


def _robust(
    values: np.ndarray,
    *,
    valid: np.ndarray,
    ordinals: np.ndarray,
    axes: dict[str, list[Any]],
    shape: tuple[int, ...],
    specs: tuple[RobustnessAxis, ...],
    sign: float = 1.0,
):
    points = sign * np.asarray(values, dtype=float).copy()
    points[~valid] = np.nan
    return score_robustness(
        point_scores=points,
        ordinals=ordinals,
        axes=axes,
        shape=shape,
        robustness_axes=specs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--rank",
        choices=RANKINGS,
        default="score",
        help="score, geometric mean of LP/YB GM, or YB GM (default: score)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        help="basin width in rank units (default: 0.1 for score, 0.001 otherwise)",
    )
    parser.add_argument("--min-lp-gm", type=float, default=0.0)
    parser.add_argument("--min-yb-gm", type=float, default=0.0)
    parser.add_argument("--max-detach", type=float, default=float("inf"))
    parser.add_argument("--flat-fee", action="store_true")
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
    gap = args.gap if args.gap is not None else (0.1 if args.rank == "score" else 0.001)
    finite_nonnegative = (gap, args.min_lp_gm, args.min_yb_gm)
    if (
        args.top < 1
        or any(not math.isfinite(value) or value < 0 for value in finite_nonnegative)
        or math.isnan(args.max_detach)
        or args.max_detach < 0
    ):
        parser.error("limits must be non-negative; --top must be positive")

    root = args.run_dir
    try:
        run = json.loads((root / "run.json").read_text())
        with np.load(root / "results.npz", allow_pickle=False) as archive:
            metric_names = list(run["metric_names"])
            columns = {name: metric_names.index(name) for name in METRICS}
            metrics = {
                name: archive[f"metric_{columns[name]:04d}"] for name in METRICS
            }
            statuses = archive["statuses"]
            ordinals = archive["ordinals"]
            overrides = archive["candidate_pool_overrides"]
            offsets = archive["candidate_policy_offsets"]
            policy_values = archive["candidate_policy_values"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"could not read result artifact: {exc}")

    count = len(ordinals)
    if any(
        array.shape != (count,)
        for array in (*metrics.values(), statuses, overrides, ordinals)
    ):
        parser.error("result arrays have inconsistent lengths")
    metadata = run.get("metadata", {})
    if not isinstance(metadata, dict):
        parser.error("run metadata must be an object")
    raw_axes = metadata.get("axes", {})
    raw_shape = metadata.get("shape", [])
    axes = dict(raw_axes) if isinstance(raw_axes, dict) else {}
    shape = tuple(int(size) for size in raw_shape) if isinstance(raw_shape, list) else ()

    lp = np.asarray(metrics["apy_net_gm"], dtype=float)
    yb = np.asarray(metrics["yb_apy_gm"], dtype=float)
    detach = np.asarray(metrics["detach_energy_ungated"], dtype=float)
    valid = (
        (statuses == "ok")
        & np.isfinite(lp)
        & np.isfinite(yb)
        & np.isfinite(detach)
        & (detach >= 0)
    )
    score = combined_scores(metrics)
    gm = np.sqrt(np.maximum(lp, 0.0) * np.maximum(yb, 0.0))
    score[~valid] = np.nan
    gm[~valid] = np.nan

    try:
        if args.robustness_config is not None:
            specs = load_robustness_axes(args.robustness_config)
        elif args.robust:
            specs = _cli_robustness(args.robust)
        else:
            specs = parse_robustness_axes(
                metadata.get("robustness"), required=False
            )
    except (OSError, ValueError) as exc:
        parser.error(f"could not load robustness radii: {exc}")

    robust_results = None
    if specs:
        try:
            robust_score = _robust(
                score, valid=valid, ordinals=ordinals, axes=axes, shape=shape, specs=specs
            )
            robust_gm = _robust(
                gm, valid=valid, ordinals=ordinals, axes=axes, shape=shape, specs=specs
            )
            robust_lp = _robust(
                lp, valid=valid, ordinals=ordinals, axes=axes, shape=shape, specs=specs
            )
            robust_yb = _robust(
                yb, valid=valid, ordinals=ordinals, axes=axes, shape=shape, specs=specs
            )
            robust_detach = _robust(
                detach,
                valid=valid,
                ordinals=ordinals,
                axes=axes,
                shape=shape,
                specs=specs,
                sign=-1.0,
            )
        except ValueError as exc:
            parser.error(f"could not score robustness: {exc}")
        complete = (
            robust_score.complete
            & robust_gm.complete
            & robust_lp.complete
            & robust_yb.complete
            & robust_detach.complete
        )
        score_floor = robust_score.robust_score
        gm_floor = robust_gm.robust_score
        lp_floor = robust_lp.robust_score
        yb_floor = robust_yb.robust_score
        detach_ceiling = -robust_detach.robust_score
        robust_results = {
            "score": robust_score,
            "gm": robust_gm,
            "yb-gm": robust_yb,
        }
    else:
        complete = valid
        score_floor = score
        gm_floor = gm
        lp_floor = lp
        yb_floor = yb
        detach_ceiling = detach

    rank_values = {
        "score": score_floor,
        "gm": gm_floor,
        "yb-gm": yb_floor,
    }[args.rank]
    point_rank = {"score": score, "gm": gm, "yb-gm": yb}[args.rank]
    eligible = (
        complete
        & np.isfinite(rank_values)
        & (lp_floor >= args.min_lp_gm)
        & (yb_floor >= args.min_yb_gm)
        & (detach_ceiling <= args.max_detach)
    )
    if args.flat_fee:
        fee_equal = np.zeros(count, dtype=bool)
        for row, raw in enumerate(overrides.tolist()):
            try:
                candidate = json.loads(str(raw))
                fee_equal[row] = candidate.get("mid_fee") == candidate.get("out_fee")
            except (TypeError, AttributeError, json.JSONDecodeError):
                pass
        eligible &= fee_equal
    if not np.any(eligible):
        suffix = " with complete robustness coverage" if specs else ""
        parser.error(f"no eligible rows{suffix} after filtering")

    eligible_rows = np.flatnonzero(eligible)
    ordered = eligible_rows[
        np.argsort(-rank_values[eligible_rows], kind="stable")
    ]
    best = float(rank_values[ordered[0]])
    basin = eligible & (rank_values >= best - gap)
    print(
        f"RUN {run.get('run_id', root.name)}: {int(eligible.sum())}/{count} eligible; "
        f"best {args.rank}={best:.8g}"
    )
    if specs:
        radii = ", ".join(f"{axis.name}=+/-{axis.radius:g}" for axis in specs)
        print(f"ROBUSTNESS exact axial cross; reducer=min; {radii}")
    filters = []
    if args.min_lp_gm:
        filters.append(f"LP GM >= {args.min_lp_gm:g}")
    if args.min_yb_gm:
        filters.append(f"YB GM >= {args.min_yb_gm:g}")
    if math.isfinite(args.max_detach):
        filters.append(f"detach <= {args.max_detach:g}")
    if args.flat_fee:
        filters.append("mid_fee == out_fee")
    if filters:
        print("FILTER " + "; ".join(filters))

    print("\nBEST ROWS (human review; no automatic optimization)")
    ranking_result = robust_results[args.rank] if robust_results is not None else None
    for row in ordered[: args.top]:
        start, stop = int(offsets[row]), int(offsets[row + 1])
        params = _candidate_parameters(str(overrides[row]), policy_values[start:stop])
        robust_fields = ""
        if ranking_result is not None:
            robust_fields = (
                f" point={point_rank[row]:.8g}"
                f" regret={point_rank[row] - rank_values[row]:.8g}"
                f" n={int(ranking_result.member_count[row])}"
                f" worst={int(ranking_result.worst_ordinal[row])}"
                f" lp_floor={lp_floor[row]:.8g}"
                f" yb_floor={yb_floor[row]:.8g}"
                f" detach_ceiling={detach_ceiling[row]:.8g}"
            )
        print(
            f"  ordinal={int(ordinals[row])} {args.rank}={rank_values[row]:.8g}"
            f"{robust_fields} lp={lp[row]:.8g} yb={yb[row]:.8g}"
            f" detach={detach[row]:.8g} {params}"
        )

    _axis_distributions(axes, shape, ordinals, basin, gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
