#!/usr/bin/env python3
"""Rank a completed Cartesian grid by point or exact axial-star performance."""

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
from fxopt.scoring import combined_scores, lp_detach_scores


RANKINGS = ("lp-score", "score", "gm", "yb-gm", "apy-net-robust")


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


def _axis_path_values(
    axis: str, values: list[Any], path: str
) -> np.ndarray | None:
    selected = []
    for value in values:
        updates = value if isinstance(value, dict) else {axis: value}
        if path not in updates:
            return None
        selected.append(float(updates[path]))
    return np.asarray(selected)


def _fee_grids(
    metadata: dict[str, Any], shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    axes = metadata.get("axes")
    defaults = metadata.get("candidate_defaults")
    if not isinstance(axes, dict) or not isinstance(defaults, dict):
        raise ValueError("run metadata cannot resolve flat fees")
    pool = defaults.get("pool", {})
    mid: Any = pool.get("mid_fee") if isinstance(pool, dict) else None
    out: Any = pool.get("out_fee") if isinstance(pool, dict) else None
    for dimension, (axis, values) in enumerate(axes.items()):
        if not isinstance(values, list):
            raise ValueError("run axis values must be arrays")
        view = [1] * len(shape)
        view[dimension] = len(values)
        mid_values = _axis_path_values(axis, values, "pool.mid_fee")
        out_values = _axis_path_values(axis, values, "pool.out_fee")
        if mid_values is not None:
            mid = mid_values.reshape(view)
        if out_values is not None:
            out = out_values.reshape(view)
    if mid is None or out is None:
        raise ValueError("run metadata does not define both fee coordinates")
    return (
        np.broadcast_to(np.asarray(mid, dtype=float), shape).reshape(-1),
        np.broadcast_to(np.asarray(out, dtype=float), shape).reshape(-1),
    )


def _flat_fee_mask(metadata: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    mid, out = _fee_grids(metadata, shape)
    return np.equal(mid, out)


def _max_fee_mask(
    metadata: dict[str, Any], shape: tuple[int, ...], limit_bps: float
) -> np.ndarray:
    mid, out = _fee_grids(metadata, shape)
    return np.maximum(mid, out) * 10_000.0 <= limit_bps


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
        help=(
            "LP-detachment score, joint score, geometric LP/YB GM, YB GM, "
            "or robust 90-day net APY"
        ),
    )
    parser.add_argument(
        "--gap",
        type=float,
        help="basin width in rank units (default: 0.1 for score, 0.001 otherwise)",
    )
    parser.add_argument("--min-lp-gm", type=float, default=0.0)
    parser.add_argument("--min-yb-gm", type=float, default=0.0)
    parser.add_argument("--max-detach", type=float, default=float("inf"))
    parser.add_argument(
        "--max-price-diff-bps",
        type=float,
        default=float("inf"),
        help="maximum absolute 7-day relative price difference in basis points",
    )
    parser.add_argument(
        "--max-fee-bps",
        type=float,
        default=float("inf"),
        help="inclusive upper bound for both mid and out fees in basis points",
    )
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
    gap = args.gap if args.gap is not None else (
        0.1 if args.rank in {"lp-score", "score"} else 0.001
    )
    finite_nonnegative = (gap, args.min_lp_gm, args.min_yb_gm)
    if (
        args.top < 1
        or any(not math.isfinite(value) or value < 0 for value in finite_nonnegative)
        or math.isnan(args.max_detach)
        or args.max_detach < 0
        or math.isnan(args.max_price_diff_bps)
        or args.max_price_diff_bps < 0
        or math.isnan(args.max_fee_bps)
        or args.max_fee_bps < 0
    ):
        parser.error("limits must be non-negative; --top must be positive")
    apy_robust_rank = args.rank == "apy-net-robust"
    if apy_robust_rank and (args.min_lp_gm or args.min_yb_gm):
        parser.error("GM limits do not apply to --rank apy-net-robust")

    root = args.run_dir
    try:
        required = ["detach_energy_ungated"]
        if apy_robust_rank:
            required.append("apy_net_robust_90d")
        else:
            required.extend(("apy_net_gm", "apy_net_robust_90d"))
            if args.rank != "lp-score":
                required.append("yb_apy_gm")
        if math.isfinite(args.max_price_diff_bps):
            required.append("max_7d_rel_price_diff")
        columns = read_result_columns(root, metrics=required)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(f"could not read result artifact: {exc}")

    count = columns.row_count
    metadata = dict(columns.metadata)
    if not isinstance(metadata, dict):
        parser.error("run metadata must be an object")
    ordinals = columns.ordinals
    raw_axes = metadata.get("axes", {})
    raw_shape = metadata.get("shape", [])
    axes = dict(raw_axes) if isinstance(raw_axes, dict) else {}
    shape = tuple(int(size) for size in raw_shape) if isinstance(raw_shape, list) else ()

    missing = np.full(count, np.nan)
    lp = np.asarray(columns.metrics.get("apy_net_gm", missing), dtype=float)
    apy_robust = np.asarray(
        columns.metrics.get("apy_net_robust_90d", missing), dtype=float
    )
    detach = np.asarray(columns.metrics["detach_energy_ungated"], dtype=float)
    yb = (
        np.asarray(columns.metrics["yb_apy_gm"], dtype=float)
        if "yb_apy_gm" in columns.metrics
        else np.full(count, np.nan)
    )
    price_diff = np.abs(
        np.asarray(
            columns.metrics.get("max_7d_rel_price_diff", missing), dtype=float
        )
    )
    base_valid = (
        columns.ok_mask
        & np.isfinite(detach)
        & (detach >= 0)
    )
    if apy_robust_rank:
        apy_robust_valid = base_valid & np.isfinite(apy_robust)
    else:
        metrics = {
            "apy_net_gm": lp,
            "apy_net_robust_90d": apy_robust,
            "yb_apy_gm": yb,
            "detach_energy_ungated": detach,
        }
        lp_valid = base_valid & np.isfinite(lp)
        valid = lp_valid & np.isfinite(yb)
        lp_score = lp_detach_scores(metrics)
        score = combined_scores(metrics)
        gm = np.sqrt(np.maximum(lp, 0.0) * np.maximum(yb, 0.0))
        score[~valid] = np.nan
        gm[~valid] = np.nan
        lp_score[~lp_valid] = np.nan

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

    ranking_result = None
    if specs and apy_robust_rank:
        try:
            ranking_result = _robust(
                apy_robust,
                valid=apy_robust_valid,
                ordinals=ordinals,
                axes=axes,
                shape=shape,
                specs=specs,
            )
            robust_detach = _robust(
                detach,
                valid=apy_robust_valid,
                ordinals=ordinals,
                axes=axes,
                shape=shape,
                specs=specs,
                sign=-1.0,
            )
        except ValueError as exc:
            parser.error(f"could not score robustness: {exc}")
        complete = ranking_result.complete & robust_detach.complete
        rank_values = ranking_result.robust_score
        point_rank = apy_robust
        detach_ceiling = -robust_detach.robust_score
    elif specs:
        try:
            robust_score = _robust(
                score, valid=valid, ordinals=ordinals, axes=axes, shape=shape, specs=specs
            )
            robust_lp_score = _robust(
                lp_score,
                valid=lp_valid,
                ordinals=ordinals,
                axes=axes,
                shape=shape,
                specs=specs,
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
        joint_complete = (
            robust_score.complete
            & robust_gm.complete
            & robust_lp.complete
            & robust_yb.complete
            & robust_detach.complete
        )
        lp_complete = (
            robust_lp_score.complete
            & robust_lp.complete
            & robust_detach.complete
        )
        complete = lp_complete if args.rank == "lp-score" else joint_complete
        lp_score_floor = robust_lp_score.robust_score
        score_floor = robust_score.robust_score
        gm_floor = robust_gm.robust_score
        lp_floor = robust_lp.robust_score
        yb_floor = robust_yb.robust_score
        detach_ceiling = -robust_detach.robust_score
        ranking_result = {
            "lp-score": robust_lp_score,
            "score": robust_score,
            "gm": robust_gm,
            "yb-gm": robust_yb,
        }[args.rank]
        rank_values = {
            "lp-score": lp_score_floor,
            "score": score_floor,
            "gm": gm_floor,
            "yb-gm": yb_floor,
        }[args.rank]
        point_rank = {
            "lp-score": lp_score,
            "score": score,
            "gm": gm,
            "yb-gm": yb,
        }[args.rank]
    elif apy_robust_rank:
        complete = apy_robust_valid
        rank_values = apy_robust
        point_rank = apy_robust
        detach_ceiling = detach
    else:
        complete = lp_valid if args.rank == "lp-score" else valid
        lp_score_floor = lp_score
        score_floor = score
        gm_floor = gm
        lp_floor = lp
        yb_floor = yb
        detach_ceiling = detach
        rank_values = {
            "lp-score": lp_score_floor,
            "score": score_floor,
            "gm": gm_floor,
            "yb-gm": yb_floor,
        }[args.rank]
        point_rank = {
            "lp-score": lp_score,
            "score": score,
            "gm": gm,
            "yb-gm": yb,
        }[args.rank]

    price_diff_ceiling = price_diff
    if specs and math.isfinite(args.max_price_diff_bps):
        try:
            robust_price_diff = _robust(
                price_diff,
                valid=columns.ok_mask & np.isfinite(price_diff),
                ordinals=ordinals,
                axes=axes,
                shape=shape,
                specs=specs,
                sign=-1.0,
            )
        except ValueError as exc:
            parser.error(f"could not score price-difference robustness: {exc}")
        complete &= robust_price_diff.complete
        price_diff_ceiling = -robust_price_diff.robust_score

    eligible = (
        complete
        & np.isfinite(rank_values)
        & (detach_ceiling <= args.max_detach)
    )
    if not apy_robust_rank:
        eligible &= lp_floor >= args.min_lp_gm
        if args.rank != "lp-score":
            eligible &= yb_floor >= args.min_yb_gm
    if math.isfinite(args.max_price_diff_bps):
        eligible &= price_diff_ceiling <= args.max_price_diff_bps / 10_000.0
    if math.isfinite(args.max_fee_bps):
        try:
            eligible &= _max_fee_mask(metadata, shape, args.max_fee_bps)
        except ValueError as exc:
            parser.error(str(exc))
    if args.flat_fee:
        try:
            eligible &= _flat_fee_mask(metadata, shape)
        except ValueError as exc:
            parser.error(str(exc))
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
        f"RUN {columns.run_id}: {int(eligible.sum())}/{count} eligible; "
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
    if math.isfinite(args.max_price_diff_bps):
        filters.append(f"7d price diff <= {args.max_price_diff_bps:g} bps")
    if math.isfinite(args.max_fee_bps):
        filters.append(f"max fee <= {args.max_fee_bps:g} bps")
    if args.flat_fee:
        filters.append("mid_fee == out_fee")
    if filters:
        print("FILTER " + "; ".join(filters))

    print("\nBEST ROWS (human review; no automatic optimization)")
    for row in ordered[: args.top]:
        candidate = columns.candidate_at(int(ordinals[row]))
        params = _candidate_parameters(
            dict(candidate.pool_overrides), candidate.policy_params
        )
        robust_fields = ""
        if ranking_result is not None:
            robust_fields = (
                f" point={point_rank[row]:.8g}"
                f" regret={point_rank[row] - rank_values[row]:.8g}"
                f" n={int(ranking_result.member_count[row])}"
                f" worst={int(ranking_result.worst_ordinal[row])}"
                f" detach_ceiling={detach_ceiling[row]:.8g}"
            )
            if not apy_robust_rank:
                robust_fields += (
                    f" lp_floor={lp_floor[row]:.8g}"
                    f" yb_floor={yb_floor[row]:.8g}"
                )
        metric_fields = (
            f" apy_net_robust_90d={apy_robust[row]:.8g}"
            if apy_robust_rank
            else f" lp={lp[row]:.8g} yb={yb[row]:.8g}"
        )
        if math.isfinite(args.max_price_diff_bps):
            metric_fields += f" max_7d_pdiff={price_diff_ceiling[row]:.8g}"
        print(
            f"  ordinal={int(ordinals[row])} {args.rank}={rank_values[row]:.8g}"
            f"{robust_fields}{metric_fields} detach={detach[row]:.8g} {params}"
        )

    _axis_distributions(axes, shape, ordinals, basin, gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
