#!/usr/bin/env python3
"""Download a complete Binance minute-candle range with bounded-memory merging."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPResponse
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.binance.com/api/v3/klines"
MINUTE_MS = 60_000


@dataclass(frozen=True)
class Page:
    index: int
    start_ms: int
    end_ms: int

    @property
    def rows(self) -> int:
        return (self.end_ms - self.start_ms) // MINUTE_MS


def _timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    result = result.astimezone(timezone.utc)
    if result.second or result.microsecond:
        raise ValueError(f"timestamp is not minute-aligned: {value}")
    return result


def _latest_complete_minute() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def _pages(start_ms: int, end_ms: int, limit: int) -> list[Page]:
    span = limit * MINUTE_MS
    return [
        Page(index, page_start, min(page_start + span, end_ms))
        for index, page_start in enumerate(range(start_ms, end_ms, span))
    ]


def _response_json(response: HTTPResponse) -> Any:
    return json.loads(response.read().decode("utf-8"))


def _fetch_payload(
    page: Page,
    *,
    symbol: str,
    endpoint: str,
    timeout: float,
    retries: int,
) -> list[Any]:
    query = urlencode(
        {
            "symbol": symbol,
            "interval": "1m",
            "startTime": page.start_ms,
            "endTime": page.end_ms - 1,
            "limit": page.rows,
        }
    )
    request = Request(f"{endpoint}?{query}", headers={"User-Agent": "curve-fx-optimization/1"})
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoint by default
                payload = _response_json(response)
            if isinstance(payload, dict):
                raise RuntimeError(f"Binance error response: {payload}")
            if not isinstance(payload, list):
                raise RuntimeError("Binance response is not a JSON array")
            return payload
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def _validated_rows(payload: list[Any], page: Page) -> list[list[float | int]]:
    if len(payload) > page.rows:
        raise ValueError(
            f"page {page.index} permits at most {page.rows} rows, received {len(payload)} "
            f"for [{page.start_ms}, {page.end_ms})"
        )
    rows: list[list[float | int]] = []
    previous_open: int | None = None
    for offset, raw in enumerate(payload):
        if not isinstance(raw, list) or len(raw) < 7:
            raise ValueError(f"page {page.index} row {offset} has invalid kline shape")
        open_ms = int(raw[0])
        if (
            open_ms < page.start_ms
            or open_ms >= page.end_ms
            or open_ms % MINUTE_MS
            or (previous_open is not None and open_ms <= previous_open)
            or not open_ms <= int(raw[6]) < open_ms + MINUTE_MS
        ):
            raise ValueError(
                f"page {page.index} row {offset} is outside or out of order on the minute lattice"
            )
        open_, high, low, close, volume = (float(raw[index]) for index in range(1, 6))
        if not all(math.isfinite(value) for value in (open_, high, low, close, volume)):
            raise ValueError(f"page {page.index} row {offset} contains non-finite data")
        if min(open_, high, low, close) <= 0 or volume < 0:
            raise ValueError(f"page {page.index} row {offset} contains invalid price or volume")
        if high < max(open_, low, close) or low > min(open_, high, close):
            raise ValueError(f"page {page.index} row {offset} violates OHLC bounds")
        rows.append([open_ms // 1000, open_, high, low, close, volume])
        previous_open = open_ms
    return rows


def _download_page(
    page: Page,
    directory: Path,
    *,
    symbol: str,
    endpoint: str,
    timeout: float,
    retries: int,
) -> Path:
    payload = _fetch_payload(
        page,
        symbol=symbol,
        endpoint=endpoint,
        timeout=timeout,
        retries=retries,
    )
    rows = _validated_rows(payload, page)
    output = directory / f"part-{page.index:06d}.json"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, separators=(",", ":"))
    os.replace(temporary, output)
    return output


def _load_page(path: Path) -> list[list[float | int]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"invalid staged page: {path}")
    return rows


def _merge_pages(
    parts: list[Path], output: Path, *, start_ms: int, end_ms: int
) -> tuple[int, int, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    first_output = True
    first_timestamp: int | None = None
    previous_timestamp: int | None = None
    rows_written = 0
    missing_minutes = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("[")
            for part in parts:
                for row in _load_page(part):
                    timestamp = int(row[0])
                    if previous_timestamp is not None:
                        delta = timestamp - previous_timestamp
                        if delta <= 0 or delta % 60:
                            raise ValueError(
                                "staged pages overlap or are not ordered on the minute lattice"
                            )
                        missing_minutes += delta // 60 - 1
                    else:
                        first_timestamp = timestamp
                    if not first_output:
                        handle.write(",")
                    json.dump(row, handle, separators=(",", ":"))
                    first_output = False
                    previous_timestamp = timestamp
                    rows_written += 1
            handle.write("]")
            handle.flush()
            os.fsync(handle.fileno())
        if first_timestamp is None or previous_timestamp is None:
            raise ValueError("download returned no candles")
        missing_minutes += (first_timestamp * 1000 - start_ms) // MINUTE_MS
        missing_minutes += (end_ms - MINUTE_MS - previous_timestamp * 1000) // MINUTE_MS
        expected_slots = (end_ms - start_ms) // MINUTE_MS
        if rows_written + missing_minutes != expected_slots:
            raise ValueError("merged row and gap counts do not cover the requested range")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return rows_written, missing_minutes, first_timestamp, previous_timestamp


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--start", required=True, help="inclusive UTC minute")
    parser.add_argument("--end", help="exclusive UTC minute; defaults to the latest complete minute")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--endpoint", default=API_URL)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.workers < 1 or not 1 <= args.limit <= 1000:
        raise SystemExit("--workers must be positive and --limit must be in [1, 1000]")
    if args.timeout <= 0 or args.retries < 1:
        raise SystemExit("--timeout and --retries must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}; choose a new dated output")

    start = _timestamp(args.start)
    requested_end = _timestamp(args.end) if args.end else _latest_complete_minute()
    end = min(requested_end, _latest_complete_minute())
    if end <= start:
        raise SystemExit("end must be later than start")
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    pages = _pages(start_ms, end_ms, args.limit)
    expected_rows = (end_ms - start_ms) // MINUTE_MS
    print(
        f"downloading {expected_rows:,} {args.symbol} 1m candles in {len(pages):,} pages "
        f"with {args.workers} workers"
    )
    print(f"range: [{start.isoformat()}, {end.isoformat()})")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".binance-pages-", dir=output.parent) as directory_name:
        directory = Path(directory_name)
        completed = 0
        report_every = max(1, len(pages) // 20)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _download_page,
                    page,
                    directory,
                    symbol=args.symbol,
                    endpoint=args.endpoint,
                    timeout=args.timeout,
                    retries=args.retries,
                ): page
                for page in pages
            }
            try:
                for future in as_completed(futures):
                    page = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"page {page.index} failed for [{page.start_ms}, {page.end_ms})"
                        ) from exc
                    completed += 1
                    if completed == 1 or completed % report_every == 0 or completed == len(pages):
                        print(f"pages: {completed:,}/{len(pages):,}", flush=True)
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
        parts = [directory / f"part-{page.index:06d}.json" for page in pages]
        rows_written, missing_minutes, first_timestamp, last_timestamp = _merge_pages(
            parts, output, start_ms=start_ms, end_ms=end_ms
        )

    print(f"wrote {rows_written:,} rows to {output}")
    print(f"first candle: {datetime.fromtimestamp(first_timestamp, timezone.utc).isoformat()}")
    print(f"last candle: {datetime.fromtimestamp(last_timestamp, timezone.utc).isoformat()}")
    print(f"missing minutes: {missing_minutes:,}; duplicates: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
