"""Build causal two-point marks from completed BTCUSDC raw minute closes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

START = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 9, 4, tzinfo=timezone.utc).timestamp())
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/market/btcusd/candles-2026-07-03T2357-2026-09-03-btcusdc-raw.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/market/btcusd/candles-2026-07-04-2026-09-03-btcusdc-causal.json")
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
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
        prior = timestamp - 120
        recent = timestamp - 60
        if prior not in closes or recent not in closes:
            raise ValueError(f"missing completed raw close for candle {timestamp}")
        p0, p1 = closes[prior], closes[recent]
        high, low = max(p0, p1), min(p0, p1)
        row = [timestamp, p0, high, low, p1, 0.0]
        # Mirror gen_events(): its shortest path emits p0 at T-5 and p1 at T+5.
        path1 = abs(p0 - low) + abs(high - p1)
        path2 = abs(p0 - high) + abs(low - p1)
        first_low = path1 < path2
        event0 = low if first_low else high
        event1 = high if first_low else low
        if (event0, event1) != (p0, p1):
            raise RuntimeError(f"causal event ordering changed at candle {timestamp}")
        if not (prior + 60 <= timestamp - 5 and recent + 60 <= timestamp + 5):
            raise RuntimeError(f"causal event timestamp is not completed at candle {timestamp}")
        rows.append(row)

    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(json.dumps(rows, separators=(",", ":")))
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({"rows": len(rows), "first": rows[0][0], "last": rows[-1][0], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
