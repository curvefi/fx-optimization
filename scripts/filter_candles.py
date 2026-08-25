#!/usr/bin/env python3
"""Apply the established centered-neighbor OHLC filter to a candle JSON file."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="neighboring rows on each side (default: 5, matching the legacy filter)",
    )
    return parser.parse_args()


def _load(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = np.asarray(payload, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 6 or rows.shape[0] == 0:
        raise ValueError("input must be a non-empty JSON array of six-field candle rows")
    if not np.all(np.isfinite(rows)):
        raise ValueError("input contains non-finite values")
    timestamps = rows[:, 0]
    if not np.all(timestamps == np.floor(timestamps)):
        raise ValueError("timestamps must be integer seconds")
    if len(rows) > 1:
        deltas = np.diff(timestamps)
        if np.any(deltas <= 0) or np.any(deltas % 60 != 0):
            raise ValueError("timestamps must be unique, increasing, and on the minute lattice")
    if np.any(rows[:, 1:5] <= 0) or np.any(rows[:, 5] < 0):
        raise ValueError("prices must be positive and volumes non-negative")
    open_, high, low, close = (rows[:, index] for index in range(1, 5))
    if np.any(high < np.maximum.reduce((open_, low, close))):
        raise ValueError("input contains a high below another OHLC field")
    if np.any(low > np.minimum.reduce((open_, high, close))):
        raise ValueError("input contains a low above another OHLC field")
    return rows


def _filter(rows: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    prices = rows[:, 1:5]
    if len(rows) == 1:
        return prices.copy(), np.zeros_like(prices, dtype=bool)
    row_minimum = np.min(prices, axis=1)
    row_maximum = np.max(prices, axis=1)
    neighbor_minimum = np.full(len(rows), math.inf)
    neighbor_maximum = np.full(len(rows), -math.inf)
    for offset in range(1, radius + 1):
        neighbor_minimum[offset:] = np.minimum(
            neighbor_minimum[offset:], row_minimum[:-offset]
        )
        neighbor_minimum[:-offset] = np.minimum(
            neighbor_minimum[:-offset], row_minimum[offset:]
        )
        neighbor_maximum[offset:] = np.maximum(
            neighbor_maximum[offset:], row_maximum[:-offset]
        )
        neighbor_maximum[:-offset] = np.maximum(
            neighbor_maximum[:-offset], row_maximum[offset:]
        )
    cooked = np.clip(prices, neighbor_minimum[:, None], neighbor_maximum[:, None])
    return cooked, cooked != prices


def _write(path: Path, rows: np.ndarray, *, chunk_size: int = 10_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    first_chunk = True
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("[")
            for start in range(0, len(rows), chunk_size):
                chunk = [
                    [int(row[0]), *(float(value) for value in row[1:])]
                    for row in rows[start: start + chunk_size]
                ]
                encoded = json.dumps(chunk, separators=(",", ":"))
                if not first_chunk:
                    handle.write(",")
                handle.write(encoded[1:-1])
                first_chunk = False
            handle.write("]")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = _arguments()
    if args.radius < 1:
        raise SystemExit("--radius must be positive")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise SystemExit("input and output must be different files")
    if output.exists():
        raise SystemExit(f"output already exists: {output}; choose a new dated output")

    rows = _load(source)
    cooked, changed = _filter(rows, args.radius)
    rows[:, 1:5] = cooked
    _write(output, rows)
    print(f"read {len(rows):,} rows from {source}")
    print(f"wrote {len(rows):,} rows to {output}")
    gaps = int((np.diff(rows[:, 0]) // 60 - 1).sum()) if len(rows) > 1 else 0
    print(f"preserved missing minutes: {gaps:,}")
    print(f"touched rows: {int(np.any(changed, axis=1).sum()):,}")
    print(f"changed OHLC fields: {[int(value) for value in changed.sum(axis=0)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
