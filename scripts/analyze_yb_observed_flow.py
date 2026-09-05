"""Describe native cbBTC swaps against an as-of raw BTCUSDC close."""

from __future__ import annotations

import hashlib
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "runs/yb-cbbtc-historical-20260904/activity"
OUT = ROOT / "runs/yb-cbbtc-historical-20260904/observed-flow"
RAW = ROOT / "data/market/btcusd/candles-2026-07-03T2357-2026-09-03-btcusdc-raw.json"
FILTERED = ROOT / "data/market/btcusd/candles-2026-07-03T2347-2026-09-03-btcusdc-filtered.json"
POOL = "0x862cb4e988fb66e72f128d1183829f8c05b6c6a0"
DEV_START = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp())
DEV_END = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp())
SCALE = (Decimal(10**18), Decimal(10**8))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dt(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def mark_value(index: int, amount: int, price: Decimal) -> Decimal:
    return Decimal(amount) / SCALE[index] * (Decimal(1) if index == 0 else price)


def surplus_bps(row: dict, shift_bps: int = 0) -> Decimal:
    price = Decimal(str(row["latest_completed_close"])) * (Decimal(10000 + shift_bps) / Decimal(10000))
    input_value = mark_value(row["sold_id"], row["tokens_sold_raw"], price)
    return (mark_value(row["bought_id"], row["tokens_bought_raw"], price) - input_value) / input_value * 10000


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("p01", "p05", "p25", "p50", "p75", "p95", "p99", "min", "max")}
    ordered = sorted(values)
    def q(frac: float) -> float:
        return ordered[min(len(ordered) - 1, int(frac * (len(ordered) - 1)))]
    return {"p01": q(.01), "p05": q(.05), "p25": q(.25), "p50": q(.50), "p75": q(.75), "p95": q(.95), "p99": q(.99), "min": ordered[0], "max": ordered[-1]}


def threshold(rows: list[dict], predicate) -> dict[str, float | int]:
    total = sum(Decimal(str(row["input_value_coin0"])) for row in rows)
    chosen = [row for row in rows if predicate(Decimal(str(row["surplus_bps"]))) ]
    amount = sum(Decimal(str(row["input_value_coin0"])) for row in chosen)
    return {"count": len(chosen), "input_notional_share": float(amount / total) if total else 0.0}


def group_summary(rows: list[dict]) -> dict:
    total_input = sum(Decimal(str(row["input_value_coin0"])) for row in rows)
    total_surplus = sum(Decimal(str(row["surplus_coin0"])) for row in rows)
    bps = [float(row["surplus_bps"]) for row in rows]
    result = {"count": len(rows), "input_notional_coin0": float(total_input), "surplus_coin0": float(total_surplus), "input_notional_weighted_surplus_bps": float(total_surplus / total_input * 10000) if total_input else None, "distribution_surplus_bps": quantiles(bps), "thresholds": {}}
    for level in (10, 50, 100):
        result["thresholds"][f"negative_lt_{level}bp"] = threshold(rows, lambda x, n=level: x < -n)
        result["thresholds"][f"positive_gt_{level}bp"] = threshold(rows, lambda x, n=level: x > n)
    result["sensitivity"] = {}
    for band in (50, 100):
        shifted = [[float(surplus_bps(row, shift)) for shift in (-band, 0, band)] for row in rows]
        result["sensitivity"][f"plus_minus_{band}bp"] = {"negative_by_mark": [sum(values[i] < 0 for values in shifted) for i in range(3)], "positive_by_mark": [sum(values[i] > 0 for values in shifted) for i in range(3)], "robust_negative": sum(max(values) < 0 for values in shifted), "ambiguous": sum(min(values) <= 0 <= max(values) and not max(values) < 0 for values in shifted)}
    return result


def model_stats(rows: list[dict]) -> dict:
    result = {}
    for label, subset in (("all", rows), ("material", [r for r in rows if r["input_value_coin0"] >= 100])):
        total = sum(Decimal(str(r["input_value_coin0"])) for r in subset)
        surplus = sum(Decimal(str(r["surplus_coin0"])) for r in subset)
        result[label] = {"count": len(subset), "input_notional_coin0": float(total), "fraction_negative_lt_10bp": sum(r["surplus_bps"] < -10 for r in subset) / len(subset) if subset else None, "weighted_surplus_bps": float(surplus / total * 10000) if total else None}
    return result


def model_summary(path: Path, closes: dict[int, float]) -> dict:
    actions = json.loads(path.read_text())
    exchanges = []
    for action in actions:
        if action.get("type") != "exchange":
            continue
        if not {"ts", "i", "j", "dx", "dy_after_fee"} <= action.keys():
            raise RuntimeError(f"exchange schema changed in {path.name}")
        ts = int(action["ts"]); candle = ((ts - 60) // 60) * 60
        if candle not in closes:
            raise RuntimeError(f"no raw close for model exchange at {ts}")
        price = Decimal(str(closes[candle])); inp = Decimal(str(action["dx"])) if action["i"] == 0 else Decimal(str(action["dx"])) * price
        out = Decimal(str(action["dy_after_fee"])) if action["j"] == 0 else Decimal(str(action["dy_after_fee"])) * price
        exchanges.append({"period": "dev" if DEV_START <= ts < DEV_END else "val", "direction": f"{action['i']}_to_{action['j']}", "input_value_coin0": float(inp), "surplus_coin0": float(out - inp), "surplus_bps": float((out - inp) / inp * 10000)})
    if len(exchanges) not in ({980, 1099}):
        raise RuntimeError(f"unexpected model exchange count {len(exchanges)} in {path.name}")
    metrics = json.loads((path.parent / "shiftclick.json").read_text())["result"]["metrics"]
    if int(metrics["trades"]) != len(exchanges):
        raise RuntimeError(f"action/metric trade count mismatch in {path.name}")
    return {"source": sha(path), "action_counts": dict(Counter(a.get("type") for a in actions)), "exchange_count": len(exchanges), "metric_trades": int(metrics["trades"]), "groups": {f"{p}/{d}": model_stats([r for r in exchanges if r["period"] == p and r["direction"] == d]) for p in ("dev", "val") for d in ("0_to_1", "1_to_0")}}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    events = [json.loads(line) for line in (ACT / "events_log_order.jsonl").read_text().splitlines()]
    receipts = {row["transaction"]: row for row in (json.loads(line) for line in (ACT / "receipts.jsonl").read_text().splitlines())}
    raw = {int(row[0]): float(row[4]) for row in json.loads(RAW.read_text())}
    swaps = []
    for event in events:
        if event["address"] != POOL or event["event"] != "TokenExchange" or event["decode_status"] != "known":
            continue
        receipt = receipts[event["transaction"]]
        timestamp = int(receipt["timestamp"])
        candle = ((timestamp - 60) // 60) * 60
        if candle not in raw:
            raise RuntimeError(f"no completed raw close for {event['transaction']} at {timestamp}")
        args = event["args"]
        input_value = mark_value(args["sold_id"], args["tokens_sold"], Decimal(str(raw[candle])))
        output_value = mark_value(args["bought_id"], args["tokens_bought"], Decimal(str(raw[candle])))
        row = {"tx_hash": event["transaction"], "block": event["block"], "timestamp": timestamp, "timestamp_utc": dt(timestamp), "asof_candle_open_utc": dt(candle), "latest_completed_close": raw[candle], "buyer": args["buyer"], "sold_id": args["sold_id"], "bought_id": args["bought_id"], "tokens_sold_raw": args["tokens_sold"], "tokens_bought_raw": args["tokens_bought"], "fee_raw_recorded_only": args["fee"], "input_value_coin0": float(input_value), "output_value_coin0": float(output_value), "surplus_coin0": float(output_value - input_value), "surplus_bps": float((output_value - input_value) / input_value * 10000), "period": "dev" if DEV_START <= timestamp < DEV_END else "val", "direction": f"{args['sold_id']}_to_{args['bought_id']}", "materiality": "material" if input_value >= 100 else "dust"}
        swaps.append(row)
    if len(swaps) != 1296:
        raise RuntimeError(f"expected 1296 native swaps, found {len(swaps)}")
    groups = {f"{period}/{direction}/{materiality}": group_summary([row for row in swaps if row["period"] == period and row["direction"] == direction and row["materiality"] == materiality]) for period in ("dev", "val") for direction in ("0_to_1", "1_to_0") for materiality in ("material", "dust")}
    dev = sorted((row for row in swaps if row["period"] == "dev"), key=lambda row: row["surplus_bps"])
    examples = []
    selected = [dev[0], dev[-1]]
    selected += [row for row in dev if row["direction"] != dev[0]["direction"]][:1]
    selected += dev
    for candidate in selected:
        if candidate["tx_hash"] not in {row["tx_hash"] for row in examples} and len(examples) < 3:
            examples.append({key: candidate[key] for key in ("tx_hash", "timestamp_utc", "asof_candle_open_utc", "latest_completed_close", "sold_id", "bought_id", "tokens_sold_raw", "tokens_bought_raw", "fee_raw_recorded_only", "input_value_coin0", "output_value_coin0", "surplus_coin0", "surplus_bps", "direction", "materiality")})
    buyers = {period: Counter(row["buyer"] for row in swaps if row["period"] == period).most_common(20) for period in ("dev", "val")}
    models = {"active_2l": model_summary(ROOT / "runs/yb-cbbtc-historical-20260904/btcusdc-causal/historical-state/active_2l-trace/candidate_0.yb-cbbtc-historical-20260904-active_2l.actions.json", raw), "reference_2l": model_summary(ROOT / "runs/yb-cbbtc-historical-20260904/btcusdc-causal/historical-state/reference_2l-trace/candidate_0.yb-cbbtc-historical-20260904-reference_2l.actions.json", raw)}
    robust = {period: {f"plus_minus_{band}bp": {"robust_negative": sum(max(float(surplus_bps(row, shift)) for shift in (-band, 0, band)) < 0 for row in swaps if row["period"] == period), "ambiguous": sum(min(float(surplus_bps(row, shift)) for shift in (-band, 0, band)) <= 0 <= max(float(surplus_bps(row, shift)) for shift in (-band, 0, band)) and max(float(surplus_bps(row, shift)) for shift in (-band, 0, band)) >= 0 for row in swaps if row["period"] == period)} for band in (50, 100)} for period in ("dev", "val")}
    result = {"schema_version": 1, "basis": "raw BTCUSDC close as of latest completed minute; coin0 valued at 1", "sources": {"events": sha(ACT / "events_log_order.jsonl"), "receipts": sha(ACT / "receipts.jsonl"), "raw_btcusdc": sha(RAW), "established_filtered": sha(FILTERED), "script": sha(Path(__file__))}, "counts": {"native_swaps": len(swaps), "dev": sum(row["period"] == "dev" for row in swaps), "val": sum(row["period"] == "val" for row in swaps), "material": sum(row["materiality"] == "material" for row in swaps), "dust": sum(row["materiality"] == "dust" for row in swaps)}, "groups": groups, "development_examples": examples, "buyer_histogram_top20": buyers, "observed_robust_negative": robust, "models": models, "limitations": ["Surplus is a mark-implied gross one-leg surplus, not realized profit or an arbitrage classification.", "Event tokens_bought is net output; fee_raw_recorded_only is retained and not subtracted again.", "The close is treated as available at the minute boundary with no latency; USDC/crvUSD/cbBTC basis is unknown.", "Buyer histograms are descriptive and do not identify actors.", "Model native exchange logs are compared under the same causal mark; model tick actions are counted separately and excluded from exchange metrics."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "observed_flow.json").write_text(json.dumps({"swaps": swaps, "summary": result}, indent=2) + "\n")
    report = ["# Observed native cbBTC flow", "", f"Native swaps: {len(swaps)}; development: {result['counts']['dev']}; validation: {result['counts']['val']}.", "", "Surplus is mark-implied gross one-leg surplus using the latest completed raw BTCUSDC minute close. It is not realized profit or an arbitrage label.", "", "| split | count | input coin0 | weighted surplus bp |", "|---|---:|---:|---:|"]
    for key, value in groups.items():
        report.append(f"| {key} | {value['count']} | {value['input_notional_coin0']:.2f} | {value['input_notional_weighted_surplus_bps'] if value['input_notional_weighted_surplus_bps'] is not None else 'n/a'} |")
    report += ["", "Observed robust-negative counts (assumed basis bands, not confidence intervals):"] + [f"- {period} +/-{band} bp: robust-negative {value[f'plus_minus_{band}bp']['robust_negative']}; ambiguous {value[f'plus_minus_{band}bp']['ambiguous']}." for period, value in robust.items() for band in (50, 100)]
    report += ["", "Observed robust-negative counts by materiality (material = input >=100 coin0; dust = input <100 coin0):", "", "| split | materiality | +/-50 robust-negative | +/-50 ambiguous | +/-100 robust-negative | +/-100 ambiguous |", "|---|---|---:|---:|---:|---:|"]
    for period in ("dev", "val"):
        for materiality, label in (("material", "material (>=100 coin0)"), ("dust", "dust (<100 coin0)")):
            values = {f"plus_minus_{band}bp": {key: sum(groups[f"{period}/{direction}/{materiality}"]["sensitivity"][f"plus_minus_{band}bp"][key] for direction in ("0_to_1", "1_to_0")) for key in ("robust_negative", "ambiguous")} for band in (50, 100)}
            report.append(f"| {period} | {label} | {values['plus_minus_50bp']['robust_negative']} | {values['plus_minus_50bp']['ambiguous']} | {values['plus_minus_100bp']['robust_negative']} | {values['plus_minus_100bp']['ambiguous']} |")
    report += ["", "Model native exchange comparison (same causal mark):", "", "| model/split | exchanges | material input coin0 | all negative < -10 bp | all weighted bp | material negative < -10 bp | material weighted bp |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model, info in models.items():
        report.append(f"- {model} action types: {info['action_counts']}; excluded ticks are counted here only.")
        for key, value in info["groups"].items():
            report.append(f"| {model}/{key} | {value['all']['count']} | {value['material']['input_notional_coin0']:.2f} | {value['all']['fraction_negative_lt_10bp']:.3f} | {value['all']['weighted_surplus_bps']:.3f} | {value['material']['fraction_negative_lt_10bp']:.3f} | {value['material']['weighted_surplus_bps']:.3f} |")
    report += ["", "Development examples:"] + [f"- `{row['tx_hash']}` {row['timestamp_utc']} {row['direction']} surplus {row['surplus_bps']:.2f} bp; raw quantities {row['tokens_sold_raw']} -> {row['tokens_bought_raw']}; as-of close {row['latest_completed_close']}." for row in examples]
    (OUT / "observed_flow.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"swaps": len(swaps), "dev": result["counts"]["dev"], "val": result["counts"]["val"], "output": str((OUT / "observed_flow.json").resolve())}))


if __name__ == "__main__":
    main()
