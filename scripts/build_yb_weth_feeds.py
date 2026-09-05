"""Build causal two-point WETH marks from completed ETHUSDT closes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

START = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 8, 29, tzinfo=timezone.utc).timestamp())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    closes: dict[int, float] = {}
    previous = None
    for row in raw:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError("raw rows must contain six fields")
        timestamp = int(row[0])
        if timestamp % 60 or (previous is not None and timestamp <= previous):
            raise ValueError("raw timestamps must be unique and increasing on the minute lattice")
        closes[timestamp] = float(row[4])
        previous = timestamp

    rows = []
    for timestamp in range(START, END, 60):
        prior, recent = timestamp - 120, timestamp - 60
        if prior not in closes or recent not in closes:
            raise ValueError(f"missing completed raw close for candle {timestamp}")
        p0, p1 = closes[prior], closes[recent]
        high, low = max(p0, p1), min(p0, p1)
        rows.append([timestamp, p0, high, low, p1, 0.0])

    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({"rows": len(rows), "first": rows[0][0], "last": rows[-1][0], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
