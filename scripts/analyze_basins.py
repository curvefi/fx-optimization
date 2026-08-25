#!/usr/bin/env python3
"""Rank a completed Cartesian result artifact and summarize nearby basins.

This is an inspection aid for human review, not an optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np


METRICS = ("apy_net_gm", "yb_apy_gm", "detach_energy_ungated")
FLOOR = 1e-4


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


def _score(apy: np.ndarray, yb: np.ndarray, detach: np.ndarray) -> np.ndarray:
    return 2.0 * np.log(np.sqrt(np.maximum(apy, FLOOR) * np.maximum(yb, FLOOR))) - 2.5 * detach


def _coordinates(ordinal: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates: list[int] = []
    remainder = ordinal
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder //= size
    return tuple(reversed(coordinates))


def _axis_distributions(
    axes: dict[str, list[Any]], shape: tuple[int, ...], ordinals: np.ndarray, basin: np.ndarray, gap: float
) -> None:
    print(f"\nBASIN DISTRIBUTIONS (score gap <= {gap:g}; points={int(basin.sum())})")
    if not axes or len(shape) != len(axes) or any(len(values) != size for values, size in zip(axes.values(), shape)):
        print("  unavailable: run.json has no axes matching shape")
        return
    for axis, values in zip(axes, axes.values()):
        counts = [0] * len(values)
        dimension = list(axes).index(axis)
        for ordinal in ordinals[basin].tolist():
            coordinate = _coordinates(int(ordinal), shape)[dimension]
            counts[coordinate] += 1
        distribution = ", ".join(f"{value!r}:{count}" for value, count in zip(values, counts) if count)
        print(f"  {axis}: {distribution}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top", type=int, default=20, help="number of rows to print (default: 20)")
    parser.add_argument("--gap", type=float, default=0.10, help="score gap for basin distributions (default: 0.10)")
    parser.add_argument("--flat-fee", action="store_true", help="keep only candidates with mid_fee == out_fee")
    args = parser.parse_args()
    if args.top < 1 or not math.isfinite(args.gap) or args.gap < 0:
        parser.error("--top must be positive and --gap must be finite and non-negative")

    root = args.run_dir
    try:
        run = json.loads((root / "run.json").read_text())
        with np.load(root / "results.npz", allow_pickle=False) as archive:
            metric_names = list(run["metric_names"])
            metric_columns = {name: index for index, name in enumerate(metric_names)}
            metric_arrays = {
                name: archive[f"metric_{metric_columns[name]:04d}"] for name in METRICS
            }
            statuses = archive["statuses"]
            ordinals = archive["ordinals"]
            overrides = archive["candidate_pool_overrides"]
            offsets = archive["candidate_policy_offsets"]
            policy_values = archive["candidate_policy_values"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"could not read result artifact: {exc}")

    count = len(ordinals)
    if any(array.shape != (count,) for array in (*metric_arrays.values(), statuses, overrides, ordinals)):
        parser.error("result arrays have inconsistent lengths")
    finite = np.ones(count, dtype=bool)
    for values in metric_arrays.values():
        finite &= np.isfinite(values)
    eligible = (statuses == "ok") & finite
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
        parser.error("no eligible rows after status/finite/fee filtering")

    score = _score(*(metric_arrays[name] for name in METRICS))
    eligible_rows = np.flatnonzero(eligible)
    ordered = eligible_rows[np.argsort(-score[eligible_rows], kind="stable")]
    best = float(score[ordered[0]])
    basin = eligible & (score >= best - args.gap)
    print(f"RUN {run.get('run_id', root.name)}: {int(eligible.sum())}/{count} eligible; best score={best:.8g}")
    if args.flat_fee:
        print("FILTER flat-fee: mid_fee == out_fee")
    print("\nBEST ROWS (human review; no automatic optimization)")
    for row in ordered[: args.top]:
        start, stop = int(offsets[row]), int(offsets[row + 1])
        params = _candidate_parameters(str(overrides[row]), policy_values[start:stop])
        print(
            f"  ordinal={int(ordinals[row])} score={score[row]:.8g} "
            f"apy_net_gm={metric_arrays['apy_net_gm'][row]:.8g} "
            f"yb_apy_gm={metric_arrays['yb_apy_gm'][row]:.8g} "
            f"detach={metric_arrays['detach_energy_ungated'][row]:.8g} {params}"
        )

    metadata = run.get("metadata", {})
    axes = metadata.get("axes", {}) if isinstance(metadata, dict) else {}
    shape_value = metadata.get("shape", []) if isinstance(metadata, dict) else []
    shape = tuple(int(size) for size in shape_value) if isinstance(shape_value, list) else ()
    _axis_distributions(axes if isinstance(axes, dict) else {}, shape, ordinals, basin, args.gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
