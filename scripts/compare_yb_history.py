#!/usr/bin/env python3
"""Compare observed cbBTC YB activity with the three fixed baseline traces."""
from __future__ import annotations
import bisect
import hashlib
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904"
ACTIVITY = STUDY / "activity"
BASELINE = STUDY / "baseline"
MARKET = ROOT / "data/market/btcusd/candles-2026-07-04-2026-09-03-raw.json"
PREHISTORY = STUDY / "market_prehistory.json"
ACTION_STATES = ACTIVITY / "action_block_native_states.jsonl"
START_BLOCK, END_BLOCK = 25_455_434, 25_860_758
MODES = ("off", "active_2l", "reference_2l")
SPLIT_TS = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp())
POOL = "0x862cb4e988fb66e72f128d1183829f8c05b6c6a0"
AMM = "0x49f51d7e279252f3c9a09678fdc65b4dbd5cb196"
VP = "0x04ca7a7e602335a261b63128e89d43b6fe1e2c87"
LT = "0x722fc3640ba007c3e9867ccdb0dca59f2e2f29f9"
USD_SCALE, BTC_SCALE = 10**18, 10**8
def read_json(path: Path):
    return json.loads(path.read_text())

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def market_rows(path: Path) -> list[list]:
    data = read_json(path)
    return data.get("raw_rows", data) if isinstance(data, dict) else data

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def repair_event_timestamps(events: list[dict], receipts: list[dict]) -> int:
    receipt_ts = {row["transaction"]: int(row["timestamp"]) for row in receipts}
    corrected = 0
    for event in events:
        raw_ts = int(event["raw"]["blockTimestamp"], 16)
        if raw_ts == 0 and event["transaction"] in receipt_ts:
            event["_timestamp"] = receipt_ts[event["transaction"]]
            corrected += 1
        else:
            event["_timestamp"] = raw_ts
    return corrected

def input_name(path: Path) -> str:
    try:
        return str(path.relative_to(STUDY))
    except ValueError:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

def timestamp(event: dict) -> int:
    return int(event.get("_timestamp", int(event["raw"]["blockTimestamp"], 16)))

def side(row: dict) -> str:
    args = row["args"]
    return f"coin{args['sold_id']}_to_coin{args['bought_id']}"

def causal_price(ts: int, market_times: list[int], market_closes: list[float]) -> float | None:
    minute = ts // 60 * 60
    index = bisect.bisect_left(market_times, minute) - 1
    return market_closes[index] if index >= 0 else None

def numeric_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(values), "min": ordered[0], "median": median(ordered), "max": ordered[-1], "sum": sum(values)}

def error_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(abs(value) for value in values)
    index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {"count": len(values), "mae": sum(ordered) / len(ordered), "p95_abs": ordered[index], "signed_mean": sum(values) / len(values)}

def raw_stats(rows: list[dict], coin1_scale: int = BTC_SCALE) -> dict:
    result = {}
    for name in ("coin0_to_coin1", "coin1_to_coin0"):
        selected = [row for row in rows if side(row) == name]
        sold = [int(row["args"]["tokens_sold"]) / (USD_SCALE if row["args"]["sold_id"] == 0 else coin1_scale) for row in selected]
        result[name] = {"count": len(selected), "sold_units": numeric_stats(sold), "sold_raw_sum": sum(int(row["args"]["tokens_sold"]) for row in selected), "bought_raw_sum": sum(int(row["args"]["tokens_bought"]) for row in selected)}
    return result

def gaps(rows: list[dict]) -> dict:
    times = sorted(timestamp(row) for row in rows)
    return numeric_stats([float(b - a) for a, b in zip(times, times[1:])])

def hourly(rows: list[dict]) -> dict[str, int]:
    counts = Counter(datetime.fromtimestamp(timestamp(row), timezone.utc).strftime("%Y-%m-%dT%H:00Z") for row in rows)
    return dict(sorted(counts.items()))

def observed_swaps(events: list[dict], prices: tuple[list[int], list[float]]) -> dict:
    native = [row for row in events if row["address"] == POOL and row["event"] == "TokenExchange"]
    vp = [row for row in events if row["address"] == VP and row["event"] == "TokenExchange"]
    amm = [row for row in events if row["address"] == AMM and row["event"] == "TokenExchange"]
    fee_usd = []
    fee_raw = Counter()
    notional, scale_notional = [], []
    for row in native:
        args, price = row["args"], causal_price(timestamp(row), *prices)
        sold = int(args["tokens_sold"]) / (USD_SCALE if args["sold_id"] == 0 else BTC_SCALE)
        notional.append(sold if args["sold_id"] == 0 else sold * (price or 0.0))
        scale_notional.append(sold if args["sold_id"] == 0 else sold * int(args["price_scale"]) / USD_SCALE)
        fee_raw[f"coin{args['bought_id']}"] += int(args["fee"])
        fee_units = int(args["fee"]) / (USD_SCALE if args["bought_id"] == 0 else BTC_SCALE)
        fee_usd.append(fee_units if args["bought_id"] == 0 else fee_units * (price or 0.0))
    return {"native": {"count": len(native), "directions": dict(Counter(side(row) for row in native)), "sizes": raw_stats(native), "notional_usd": numeric_stats(notional), "material_causal_proxy": {"at_least_1_usd": sum(value >= 1 for value in notional), "at_least_100_usd": sum(value >= 100 for value in notional)}, "scale_mark_quickcheck": {"below_1_usd": sum(value < 1 for value in scale_notional), "below_100_usd": sum(value < 100 for value in scale_notional), "note": "scale_mark only; material thresholds use causal USD proxy"}, "gaps_s": gaps(native), "hourly_counts_utc": hourly(native), "fee_tokens_raw_by_output": dict(fee_raw), "fee_usd_proxy": numeric_stats(fee_usd)}, "virtual_pool": {"count": len(vp), "directions": dict(Counter(side(row) for row in vp)), "sizes": raw_stats(vp), "direction_source": "observed VP TokenExchange"}, "amm_lp_collateral": {"count": len(amm), "directions": dict(Counter(side(row) for row in amm)), "sizes": raw_stats(amm, coin1_scale=USD_SCALE), "size_units": "AMM collateral LP and stablecoin are both 1e-18", "direction_source": "observed AMM TokenExchange LP collateral leg"}}

def model_swaps(mode: str, actions: list[dict], prices: tuple[list[int], list[float]]) -> dict:
    swaps = [row for row in actions if row.get("type") == "exchange"]
    ticks = [row for row in actions if row.get("type") == "tick"]
    rows = []
    for row in swaps:
        price = causal_price(int(row["ts"]), *prices)
        sold = float(row["dx"])
        if row["i"] == 1:
            sold *= price or 0.0
        rows.append((row, sold))
    return {"count": len(swaps), "directions": dict(Counter(f"coin{row['i']}_to_coin{row['j']}" for row in swaps)), "sizes": {name: numeric_stats([float(row["dx"]) for row in swaps if f"coin{row['i']}_to_coin{row['j']}" == name]) for name in ("coin0_to_coin1", "coin1_to_coin0")}, "notional_usd": numeric_stats([value for _, value in rows]), "gaps_s": numeric_stats([float(b["ts"] - a["ts"]) for a, b in zip(swaps, swaps[1:])]), "hourly_counts_utc": dict(sorted(Counter(datetime.fromtimestamp(int(row["ts"]), timezone.utc).strftime("%Y-%m-%dT%H:00Z") for row in swaps).items())), "ticks": {"count": len(ticks), "hourly_counts_utc": dict(sorted(Counter(datetime.fromtimestamp(int(row["ts"]), timezone.utc).strftime("%Y-%m-%dT%H:00Z") for row in ticks).items()))}, "mode": mode}

def model_yb(actions: list[dict]) -> dict:
    increments = []
    initial_count = int(actions[0].get("yb_releverage_trades", 0)) if actions else 0
    previous = float(initial_count)
    previous_collateral = float(actions[0].get("yb_collateral_lp", 0.0)) if actions else 0.0
    for row in actions:
        if "yb_releverage_trades" not in row:
            continue
        current = float(row["yb_releverage_trades"])
        if previous is not None and current > previous:
            delta = float(row["yb_collateral_lp"]) - previous_collateral
            increments.append({"ts": int(row["t"]), "trade_number": int(current), "collateral_lp_delta": delta, "direction": "lp_to_stable" if delta < 0 else "stable_to_lp" if delta > 0 else "flat"})
        previous, previous_collateral = current, float(row["yb_collateral_lp"])
    deltas = [row["collateral_lp_delta"] for row in increments]
    directions = Counter(row["direction"] for row in increments)
    if initial_count: directions["unknown_initial_state"] += initial_count
    return {"count": initial_count + len(increments), "directions": dict(directions), "collateral_lp_delta": numeric_stats(deltas), "first_increment": increments[0] if increments else None, "last_increment": increments[-1] if increments else None, "initial_count_increment": initial_count, "direction_source": "model-derived from yb_collateral_lp delta when yb_releverage_trades increments; initial trace count direction is unknown", "raw_vp_input_used": False}

def state_value(row: dict, key: str) -> int:
    return int(row["stored_native"][key]["value"])

def model_track(row: dict, price: float | None) -> dict:
    if row is None:
        return {"missing": True}
    marked_nav = None if price is None else (float(row["token0"]) + float(row["token1"]) * price) / float(row["total_supply"])
    return {"t": int(row["t"]), "price_scale": float(row["price_scale"]), "lp_xcp_profit": float(row["lp_xcp_profit"]), "vp": float(row["vp"]), "token0": float(row["token0"]), "token1": float(row["token1"]), "total_supply": float(row["total_supply"]), "marked_nav_per_lp": marked_nav, "missing": False}

def daily_tracks(snapshots: list[dict], traces: dict[str, list[dict]], prices: tuple[list[int], list[float]]) -> tuple[list[dict], dict[str, int]]:
    trace_times = {mode: [int(row["t"]) for row in rows] for mode, rows in traces.items()}
    first_snapshot = snapshots[0]
    first_ts = int(first_snapshot["block_header"]["timestamp"], 16)
    first_price = causal_price(first_ts, *prices)
    first_native_inventory = {"token0": state_value(first_snapshot, "balances_0") / USD_SCALE, "token1": state_value(first_snapshot, "balances_1") / BTC_SCALE, "supply": state_value(first_snapshot, "totalSupply") / USD_SCALE}
    first_native_nav = None if first_price is None else (first_native_inventory["token0"] + first_native_inventory["token1"] * first_price) / first_native_inventory["supply"]
    model_initial = {mode: {**first_native_inventory, "nav": first_native_nav} for mode in traces}
    rows, missing = [], Counter()
    for snapshot in snapshots:
        ts = int(snapshot["block_header"]["timestamp"], 16); price = causal_price(ts, *prices)
        token0 = state_value(snapshot, "balances_0") / USD_SCALE; token1 = state_value(snapshot, "balances_1") / BTC_SCALE; supply = state_value(snapshot, "totalSupply") / USD_SCALE
        native_lp = None if price is None else (token0 + token1 * price) / supply
        if first_native_nav is None and native_lp is not None: first_native_nav = native_lp
        native_hodl = None if price is None else (first_native_inventory["token0"] + first_native_inventory["token1"] * price) / first_native_inventory["supply"]
        item = {"block": snapshot["block"], "ts": ts, "price_proxy_usd": price, "native": {"price_scale": state_value(snapshot, "cached_price_scale") / USD_SCALE, "lp_xcp_profit": state_value(snapshot, "lp_xcp_profit") / USD_SCALE, "vp": state_value(snapshot, "virtual_price") / USD_SCALE, "token0": token0, "token1": token1, "total_supply": supply, "marked_nav_per_lp": native_lp, "marked_lp_unit_value_index": None if native_lp is None else native_lp / first_native_nav, "hodl_relative_index": None if native_lp is None else native_lp / native_hodl}}
        for mode, trace in traces.items():
            index = bisect.bisect_right(trace_times[mode], ts) - 1
            model = trace[index] if index >= 0 else None
            if model is None: missing[mode] += 1
            track = model_track(model, price)
            if not track.get("missing") and track["marked_nav_per_lp"] is not None:
                track["marked_lp_unit_value_index"] = track["marked_nav_per_lp"] / model_initial[mode]["nav"]
                model_hodl = (model_initial[mode]["token0"] + model_initial[mode]["token1"] * price) / model_initial[mode]["supply"]
                track["hodl_relative_index"] = track["marked_nav_per_lp"] / model_hodl
            item[mode] = track
        rows.append(item)
    return rows, dict(missing)

def path_errors(states: list[dict], traces: dict[str, list[dict]], events: list[dict], prices: tuple[list[int], list[float]], initial_state: dict) -> dict[str, dict]:
    times = {mode: [int(row["t"]) for row in trace] for mode, trace in traces.items()}
    values = {mode: defaultdict(lambda: {"scale": [], "inventory_coin0": [], "inventory_coin1": [], "inventory_l1": [], "per_lp_coin0": [], "per_lp_coin1": [], "per_lp_l1": [], "unit": []}) for mode in MODES}
    for state in states:
        ts = int(state["timestamp"]); price = causal_price(ts, *prices); actual = state["getters"]
        if price is None: continue
        a = {"price_scale": int(actual["price_scale"]["value"]) / USD_SCALE, "lp_xcp_profit": int(actual["lp_xcp_profit"]["value"]) / USD_SCALE, "vp": int(actual["virtual_price"]["value"]) / USD_SCALE, "token0": int(actual["balances_0"]["value"]) / USD_SCALE, "token1": int(actual["balances_1"]["value"]) / BTC_SCALE, "supply": int(actual["totalSupply"]["value"]) / USD_SCALE}
        actual_unit = (a["token0"] + a["token1"] * price) / a["supply"]
        for mode, trace in traces.items():
            index = bisect.bisect_right(times[mode], ts) - 1
            if index < 0: continue
            row = trace[index]; model_unit = (float(row["token0"]) + float(row["token1"]) * price) / float(row["total_supply"])
            bucket = "development" if ts < SPLIT_TS else "validation"
            target = values[mode][bucket]
            target["scale"].append((float(row["price_scale"]) / a["price_scale"] - 1) * 10_000)
            inventory_coin0 = (float(row["token0"]) / a["token0"] - 1) * 100
            inventory_coin1 = (float(row["token1"]) / a["token1"] - 1) * 100
            target["inventory_coin0"].append(inventory_coin0)
            target["inventory_coin1"].append(inventory_coin1)
            target["inventory_l1"].append((abs(inventory_coin0) + abs(inventory_coin1)) / 2)
            per_lp_coin0 = (float(row["token0"]) / float(row["total_supply"])) / (a["token0"] / a["supply"]) - 1
            per_lp_coin1 = (float(row["token1"]) / float(row["total_supply"])) / (a["token1"] / a["supply"]) - 1
            target["per_lp_coin0"].append(per_lp_coin0 * 100)
            target["per_lp_coin1"].append(per_lp_coin1 * 100)
            target["per_lp_l1"].append((abs(per_lp_coin0) + abs(per_lp_coin1)) * 50)
            target["unit"].append((model_unit / actual_unit - 1) * 10_000)
    result = {}
    for mode in MODES:
        result[mode] = {}
        for bucket, errors in values[mode].items():
            result[mode][bucket] = {"scale_bps": error_stats(errors["scale"]), "inventory_percent": {"coin0": error_stats(errors["inventory_coin0"]), "coin1": error_stats(errors["inventory_coin1"]), "aggregate_l1": error_stats(errors["inventory_l1"])}, "inventory_per_lp_percent": {"coin0": error_stats(errors["per_lp_coin0"]), "coin1": error_stats(errors["per_lp_coin1"]), "aggregate_l1": error_stats(errors["per_lp_l1"])}, "marked_lp_unit_value_bps": error_stats(errors["unit"])}
    return result

def donation_and_liquidity(events: list[dict], actions_by_mode: dict[str, list[dict]]) -> dict:
    native_donations = [row for row in events if row["address"] == POOL and row["event"] == "Donation"]
    by_tx = defaultdict(list)
    for row in events: by_tx[row["transaction"]].append(row)
    paired = 0
    for legs in by_tx.values():
        donations = [row for row in legs if row["address"] == POOL and row["event"] == "Donation"]
        adds = [row for row in legs if row["address"] == POOL and row["event"] == "AddLiquidity" and row["args"].get("receiver") == "0x0000000000000000000000000000000000000000"]
        if len(donations) == len(adds) == 1 and donations[0]["args"]["token_amounts"] == adds[0]["args"]["token_amounts"] and donations[0]["args"]["donor"] == adds[0]["args"].get("provider") and donations[0]["log_index"] < adds[0]["log_index"]: paired += 1
    result = {"observed": {"lt_deposits": sum(row["address"] == LT and row["event"] == "Deposit" for row in events), "lt_withdrawals": sum(row["address"] == LT and row["event"] == "Withdraw" for row in events), "native_donations": len(native_donations), "native_donations_paired_once": paired}, "model": {mode: {"donations": sum(row.get("type") == "donation" for row in actions)} for mode, actions in actions_by_mode.items()}, "model_lt_flow": "unsupported"}
    return result

def split_counts(rows: list[dict], time_key: str) -> dict[str, int]:
    return {"development_before_2026_08_01": sum(int(row[time_key]) < SPLIT_TS for row in rows), "validation_on_or_after_2026_08_01": sum(int(row[time_key]) >= SPLIT_TS for row in rows)}

def scale_change_counts(rows, value_fn, timestamp_fn, initial) -> dict[str, int]:
    result = {"development_before_2026_08_01": 0, "validation_on_or_after_2026_08_01": 0}; previous = initial
    for row in rows:
        current = value_fn(row); bucket = "development_before_2026_08_01" if timestamp_fn(row) < SPLIT_TS else "validation_on_or_after_2026_08_01"
        result[bucket] += previous is not None and current != previous; previous = current
    return result

def integrity(events: list[dict], receipts: list[dict], snapshots: list[dict], summary: dict, expected: dict) -> dict:
    checks = {"raw_logs": len(events) == expected["raw_logs"], "receipts": len(receipts) == expected["receipts"], "snapshots": len(snapshots) == expected["snapshots"], "receipt_status_all_one": all(row["status"] == 1 for row in receipts), "native_exchanges": sum(row["address"] == POOL and row["event"] == "TokenExchange" for row in events) == expected["native_exchanges"], "virtual_pool_exchanges": sum(row["address"] == VP and row["event"] == "TokenExchange" for row in events) == expected["virtual_pool_exchanges"], "lt_deposits": sum(row["address"] == LT and row["event"] == "Deposit" for row in events) == expected["lt_deposits"], "lt_withdrawals": sum(row["address"] == LT and row["event"] == "Withdraw" for row in events) == expected["lt_withdrawals"]}
    blocks = [row["block"] for row in events]; checks["event_blocks_in_window"] = min(blocks) >= START_BLOCK and max(blocks) <= END_BLOCK
    return {"passed": all(checks.values()), "checks": checks, "observed_summary_counts": {key: summary["counts"].get(key) for key in ("raw_logs", "receipts", "snapshot_blocks")}, "window": {"start_block": START_BLOCK, "end_block": END_BLOCK, "event_first_block": min(blocks), "event_last_block": max(blocks), "snapshot_first_block": snapshots[0]["block"], "snapshot_last_block": snapshots[-1]["block"]}}

def plot(tracks: list[dict], daily_native: dict, daily_models: dict[str, dict], out: Path) -> None:
    dates = [datetime.fromtimestamp(row["ts"], timezone.utc) for row in tracks]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(dates, [row["native"]["price_scale"] for row in tracks], label="native price_scale")
    axes[0].plot(dates, [row["price_proxy_usd"] for row in tracks], label="causal BTC close", alpha=.75)
    for mode in MODES: axes[0].plot(dates, [row[mode].get("price_scale") if not row[mode].get("missing") else float("nan") for row in tracks], label=f"{mode} price_scale", alpha=.7)
    axes[0].set_ylabel("USD / BTC"); axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(dates, [daily_native.get(row["ts"], 0) for row in tracks], label="observed native swaps")
    for mode in MODES:
        axes[1].plot(dates, [daily_models[mode]["swaps"].get(row["ts"], 0) for row in tracks], label=f"{mode} model swaps")
        axes[1].plot(dates, [daily_models[mode]["ticks"].get(row["ts"], 0) for row in tracks], ":", label=f"{mode} ticks", alpha=.6)
    axes[1].set_ylabel("swaps / UTC day"); axes[1].legend(ncol=3, fontsize=7)
    axes[2].plot(dates, [row["native"]["hodl_relative_index"] for row in tracks], label="native marked LP / HODL")
    for mode in MODES: axes[2].plot(dates, [row[mode].get("hodl_relative_index") if not row[mode].get("missing") else float("nan") for row in tracks], label=f"{mode} model marked LP / HODL")
    axes[2].set_ylabel("relative index (start = 1)"); axes[2].legend(ncol=3, fontsize=7)
    axes[2].set_xlabel("UTC date"); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)

def main() -> None:
    global STUDY, ACTIVITY, BASELINE, MARKET, PREHISTORY, ACTION_STATES, MODES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY, help="study run root")
    parser.add_argument("--run-root", type=Path, help="baseline run root, optionally containing historical-state")
    parser.add_argument("--market", type=Path, default=MARKET, help="completed-minute market JSON")
    parser.add_argument("--prehistory", type=Path, help="causal prestate market JSON")
    parser.add_argument("--output", type=Path, help="comparison artifact directory")
    args = parser.parse_args()
    STUDY = args.study
    ACTIVITY, BASELINE = STUDY / "activity", STUDY / "baseline"
    if args.run_root:
        BASELINE = args.run_root
    MARKET = args.market
    PREHISTORY = args.prehistory or STUDY / "market_prehistory.json"
    ACTION_STATES = ACTIVITY / "action_block_native_states.jsonl"
    out = args.output or STUDY / "comparison-baseline"; out.mkdir(parents=True, exist_ok=True)
    events, receipts, snapshots = read_jsonl(ACTIVITY / "events_log_order.jsonl"), read_jsonl(ACTIVITY / "receipts.jsonl"), read_jsonl(ACTIVITY / "state_snapshots.jsonl")
    timestamp_corrections = repair_event_timestamps(events, receipts)
    summary = read_json(ACTIVITY / "activity_summary.json")
    merged_market = {int(row[0]): row for row in market_rows(PREHISTORY)}
    merged_market.update({int(row[0]): row for row in market_rows(MARKET)})
    market = [merged_market[key] for key in sorted(merged_market)]
    prices = ([int(row[0]) for row in market], [float(row[4]) for row in market])
    action_states = read_jsonl(ACTION_STATES)
    run_specs = [(mode, BASELINE, mode) for mode in MODES if (BASELINE / f"{mode}-trace").exists()]
    historical_root = BASELINE / "historical-state"
    if historical_root.exists():
        run_specs.extend((f"historical_{mode}", historical_root, mode) for mode in ("active_2l", "reference_2l") if (historical_root / f"{mode}-trace").exists())
    MODES = tuple(label for label, _, _ in run_specs)
    def run_file(root: Path, file_mode: str, suffix: str) -> Path:
        matches = sorted((root / f"{file_mode}-trace").glob(f"*.{suffix}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one {suffix} file for {file_mode}, found {len(matches)}")
        return matches[0]
    traces = {label: read_json(run_file(root, file_mode, "trace")) for label, root, file_mode in run_specs}
    actions = {label: read_json(run_file(root, file_mode, "actions")) for label, root, file_mode in run_specs}
    observed = observed_swaps(events, prices); model = {mode: {"swaps": model_swaps(mode, actions[mode], prices), "yb": model_yb(traces[mode])} for mode in MODES}
    tracks, missing = daily_tracks(snapshots, traces, prices); liquidity = donation_and_liquidity(events, actions)
    expected = {"raw_logs": 15845, "receipts": 3090, "snapshots": 59, "native_exchanges": 1296, "virtual_pool_exchanges": 1047, "lt_deposits": 341, "lt_withdrawals": 172}
    gate = integrity(events, receipts, snapshots, summary, expected)
    gate["event_timestamp_receipt_fallbacks"] = timestamp_corrections
    gate["checks"]["action_block_states"] = len(action_states) == 2401 and all(row.get("required_getters_ok") for row in action_states)
    gate["action_block_count"] = len(action_states)
    scale_values = [state_value(snapshots[0], "cached_price_scale")] + [int(row["getters"]["price_scale"]["value"]) for row in action_states]
    gate["observed_true_price_scale_change_count"] = sum(a != b for a, b in zip(scale_values, scale_values[1:]))
    observed_scale_phases = scale_change_counts(action_states, lambda row: int(row["getters"]["price_scale"]["value"]), lambda row: int(row["timestamp"]), scale_values[0])
    gate["passed"] = gate["passed"] and gate["checks"]["action_block_states"]
    native_events = [row for row in events if row["address"] == POOL and row["event"] == "TokenExchange"]
    daily_native = Counter(datetime.fromtimestamp(timestamp(row), timezone.utc).strftime("%Y-%m-%d") for row in events if row["address"] == POOL and row["event"] == "TokenExchange")
    daily_models = {mode: {"swaps": Counter(datetime.fromtimestamp(int(row["ts"]), timezone.utc).strftime("%Y-%m-%d") for row in actions[mode] if row.get("type") == "exchange"), "ticks": Counter(datetime.fromtimestamp(int(row["ts"]), timezone.utc).strftime("%Y-%m-%d") for row in actions[mode] if row.get("type") == "tick")} for mode in MODES}
    daily_native_by_ts = {row["ts"]: daily_native.get(datetime.fromtimestamp(row["ts"], timezone.utc).strftime("%Y-%m-%d"), 0) for row in tracks}; daily_models_by_ts = {mode: {kind: {row["ts"]: daily_models[mode][kind].get(datetime.fromtimestamp(row["ts"], timezone.utc).strftime("%Y-%m-%d"), 0) for row in tracks} for kind in ("swaps", "ticks")} for mode in MODES}
    plot_path = out / "comparison_3panel.png"; plot(tracks, daily_native_by_ts, daily_models_by_ts, plot_path)
    inputs = [ACTIVITY / name for name in ("events_log_order.jsonl", "receipts.jsonl", "state_snapshots.jsonl", "action_block_native_states.jsonl", "activity_summary.json")] + [MARKET, PREHISTORY]
    inputs += [run_file(root, file_mode, suffix) for _, root, file_mode in run_specs for suffix in ("trace", "actions")]
    inputs += [root / file_mode / filename for _, root, file_mode in run_specs for filename in ("run.json", "results.npz")]
    binaries = {}
    for _, root, file_mode in run_specs:
        evaluator = Path(read_json(root / file_mode / "run.json")["metadata"]["evaluator"])
        if evaluator.exists(): binaries[str(evaluator)] = sha256(evaluator)
    model_scale_phases = {mode: scale_change_counts(traces[mode], lambda row: float(row["price_scale"]), lambda row: int(row["t"]), None) for mode in MODES}
    development_validation = {"split_timestamp": SPLIT_TS, "observed_native_exchanges": split_counts(native_events, "_timestamp"), "observed_vp_exchanges": split_counts([row for row in events if row["address"] == VP and row["event"] == "TokenExchange"], "_timestamp"), "observed_true_scale_changes": observed_scale_phases, "model_price_scale_changes": model_scale_phases, "model_exchanges": {mode: {"development_before_2026_08_01": sum(int(row["ts"]) < SPLIT_TS for row in actions[mode] if row.get("type") == "exchange"), "validation_on_or_after_2026_08_01": sum(int(row["ts"]) >= SPLIT_TS for row in actions[mode] if row.get("type") == "exchange")} for mode in MODES}, "path_errors": path_errors(action_states, traces, events, prices, snapshots[0])}
    feed_name = "BTCUSDC" if "btcusdc" in str(MARKET).lower() else "BTCUSDT proxy"
    feed_status = "requested BTCUSDC feed" if feed_name == "BTCUSDC" else "BTCUSDT proxy pending requested BTCUSDC runs"
    report = {"status": "pass" if gate["passed"] else "integrity_failure", "scope": {"start_block": START_BLOCK, "end_block": END_BLOCK, "split": "2026-08-01T00:00:00Z", "market_price": f"strictly causal previous completed minute close ({feed_name})", "market_status": feed_status}, "integrity": gate, "development_validation": development_validation, "observed": observed, "model": model, "liquidity": liquidity, "tracks": tracks, "track_matching": {"method": "backward_asof_model_trace_t_at_or_before_actual_snapshot_timestamp", "missing_rows_by_mode": missing, "first_native_checkpoint": tracks[0]}, "units": {"native_coin0_raw": "1e-18 USD stablecoin", "native_coin1_raw": "1e-8 BTC", "model_coin0": "USD", "model_coin1": "BTC", "D_and_xp": "USD normalized", "fee_usd": "output token converted with causal market close"}, "limitations": ["Model traces begin at t=1783123205; the native checkpoint at t=1783123199 has no backward-asof model row and remains explicitly missing.", "Observed LT liquidity flows are reported; model LT replay is unsupported.", "YB model directions are derived from yb_collateral_lp deltas at releverage increments and do not use VP raw input.", "Native fee accounting includes native TokenExchange fees only; receipt gas covers full outer transactions and is not allocated to pool legs.", "Marked LP unit value index and HODL-relative index are diagnostics with donation shares and ownership caveats; no consolidated YB wealth, APY, or compounded segmented PnL claim is made.", f"All USD comparisons use a {feed_name}; BTCUSDC runs remain pending." if feed_name == "BTCUSDT proxy" else "The selected BTCUSDC feed is used for USD comparisons.", "invalid_input_attempt.json is ignored."], "inputs": {input_name(path): sha256(path) for path in inputs}, "binaries": binaries, "artifacts": {"plot": str(plot_path.relative_to(out)), "markdown": "comparison.md"}}
    (out / "compact_comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    error_lines = []
    for mode in MODES:
        for phase in ("development", "validation"):
            errors = report["development_validation"]["path_errors"][mode].get(phase)
            if errors:
                inventory = errors["inventory_percent"]
                per_lp = errors["inventory_per_lp_percent"]
                error_lines.append(f"- {mode} {phase}: scale MAE/p95 {errors['scale_bps']['mae']:.2f}/{errors['scale_bps']['p95_abs']:.2f} bps; inventory L1 MAE/p95 {inventory['aggregate_l1']['mae']:.3f}/{inventory['aggregate_l1']['p95_abs']:.3f}% (coin0 {inventory['coin0']['mae']:.3f}/{inventory['coin0']['p95_abs']:.3f}%, coin1 {inventory['coin1']['mae']:.3f}/{inventory['coin1']['p95_abs']:.3f}%; signed {inventory['coin0']['signed_mean']:.3f}/{inventory['coin1']['signed_mean']:.3f}%); per-LP inventory L1 MAE/p95 {per_lp['aggregate_l1']['mae']:.3f}/{per_lp['aggregate_l1']['p95_abs']:.3f}%; LP unit MAE/p95 {errors['marked_lp_unit_value_bps']['mae']:.2f}/{errors['marked_lp_unit_value_bps']['p95_abs']:.2f} bps")
    yb_counts = " / ".join(f"{mode}={model[mode]['yb']['count']}" for mode in MODES)
    yb_directions = " / ".join(f"{mode}={model[mode]['yb']['directions']}" for mode in MODES)
    lines = ["# YB cbBTC historical comparison", "", f"Window: blocks {START_BLOCK}..{END_BLOCK}, split 2026-08-01 UTC. Integrity gate: **{report['status']}**.", "", "## Counts", ""]

    model_header = "| surface | observed | " + " | ".join(f"model {mode}" for mode in MODES) + " |"
    model_rule = "|---|---:|" + "---:|" * len(MODES)
    model_rows = [
        "| native TokenExchange | " + str(observed["native"]["count"]) + " | " + " | ".join(str(model[mode]["swaps"]["count"]) for mode in MODES) + " |",
        "| model ticks (separate) | - | " + " | ".join(str(model[mode]["swaps"]["ticks"]["count"]) for mode in MODES) + " |",
        "| model YB releverage count | - | " + " | ".join(str(model[mode]["yb"]["count"]) for mode in MODES) + " |",
        "| VP TokenExchange | " + str(observed["virtual_pool"]["count"]) + " | " + " | ".join("-" for _ in MODES) + " |",
        "| AMM LP collateral exchange | " + str(observed["amm_lp_collateral"]["count"]) + " | " + " | ".join("-" for _ in MODES) + " |",
        "| LT Deposit / Withdraw | " + f"{liquidity['observed']['lt_deposits']} / {liquidity['observed']['lt_withdrawals']}" + " | " + " | ".join("unsupported" for _ in MODES) + " |",
        "| native Donation paired once | " + f"{liquidity['observed']['native_donations_paired_once']} / {liquidity['observed']['native_donations']}" + " | " + " | ".join(str(liquidity['model'][mode]['donations']) for mode in MODES) + " |",
    ]
    lines[6:15] = [model_header, model_rule, *model_rows]
    lines.extend(["", "## Direction and accounting", "", f"Native directions: `{observed['native']['directions']}`. VP directions are observed raw VP events: `{observed['virtual_pool']['directions']}`. AMM LP collateral directions: `{observed['amm_lp_collateral']['directions']}`.", "", f"Observed true native `price_scale` changes: {gate['observed_true_price_scale_change_count']} across {gate['action_block_count']} action blocks. Per-phase observed/model counts: `{report['development_validation']['observed_true_scale_changes']}` / `{report['development_validation']['model_price_scale_changes']}`.", "", f"Model YB counts are {yb_counts}; directions are {yb_directions}. Directions use `yb_collateral_lp` deltas when `yb_releverage_trades` increments; the initial trace count is retained with unknown direction.", "", f"Native material proxy counts: `{observed['native']['material_causal_proxy']}`. Scale-mark quick checks are `{observed['native']['scale_mark_quickcheck']}` and carry no keeper attribution.", "", f"Native output fee raw totals: `{observed['native']['fee_tokens_raw_by_output']}`. USD conversion uses the strictly causal previous completed {feed_name} minute close. The aggregate is a fee diagnostic, not execution profit; receipt gas is full outer transaction gas.", "", "## Development and validation errors", "", *error_lines, "", "## Track matching", "", f"Backward as-of matching leaves missing model rows: `{missing}`. The first native checkpoint is retained in JSON; it precedes the first model trace row and is not forward filled.", "", f"Marked LP unit value and HODL-relative indices are diagnostics with donation shares and ownership caveats; HODL path error is algebraically redundant with LP unit error under the common prestate denominator. All USD comparisons use {feed_name}.", "", "![comparison](comparison_3panel.png)", "", "Inputs and SHA-256 hashes are recorded in `compact_comparison.json`."])
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["integrity"]["observed_summary_counts"], "native_exchanges": observed["native"]["count"], "vp_exchanges": observed["virtual_pool"]["count"], "lt_deposits": liquidity["observed"]["lt_deposits"], "lt_withdrawals": liquidity["observed"]["lt_withdrawals"], "missing_model_rows": missing, "output": str(out)}, indent=2))
if __name__ == "__main__":
    main()
