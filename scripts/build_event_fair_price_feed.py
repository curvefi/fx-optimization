#!/usr/bin/env python3
"""Build an event-synchronous fair-price CSV from harness candle input."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import orjson


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="six-column OHLCV candle JSON")
    parser.add_argument("output", type=Path, help="timestamp,price CSV to create")
    return parser.parse_args()


def _load(path: Path) -> list[list[int | float]]:
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, list) or not payload:
        raise ValueError("input must be a non-empty JSON array")

    previous_ts = -1
    for index, row in enumerate(payload):
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError(f"row {index} must contain six OHLCV fields")
        timestamp, open_, high, low, close, volume = row[:6]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError(f"row {index} timestamp is not numeric")
        ts = int(timestamp)
        if timestamp != ts or ts <= previous_ts:
            raise ValueError("timestamps must be unique increasing integer seconds")
        values = (open_, high, low, close, volume)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"row {index} contains a non-finite numeric field")
        if min(open_, high, low, close) <= 0 or volume < 0:
            raise ValueError(f"row {index} contains an invalid price or volume")
        if high < max(open_, low, close) or low > min(open_, high, close):
            raise ValueError(f"row {index} violates OHLC bounds")
        previous_ts = ts
    return payload


def _write(path: Path, rows: list[list[int | float]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write("timestamp,price\n")
            for timestamp, open_, high, low, close, _volume, *_ in rows:
                ts = int(timestamp)
                path_low_first = abs(open_ - low) + abs(high - close)
                path_high_first = abs(open_ - high) + abs(low - close)
                first_low = path_low_first < path_high_first
                first = low if first_low else high
                second = high if first_low else low
                handle.write(f"{ts - 5},{float(first):.17g}\n")
                handle.write(f"{ts + 5},{float(second):.17g}\n")
                count += 2
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def main() -> int:
    args = _arguments()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise SystemExit("input and output must be different files")
    if output.exists():
        raise SystemExit(f"output already exists: {output}; choose a new dated output")

    rows = _load(source)
    count = _write(output, rows)
    print(f"read {len(rows):,} candles from {source}")
    print(f"wrote {count:,} event-synchronous fair prices to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
