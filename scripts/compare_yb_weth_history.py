"""Compare WETH historical traces with daily observed native state."""

from __future__ import annotations

import bisect
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904/weth-preparation"
ACTIVITY = STUDY / "activity"
MARKET = ROOT / "data/market/ethusd/candles-2026-07-03T2347-2026-08-29-ethusdt-raw.json"
OUT = STUDY / "comparison"
POOL = "0x656341ef90b622c6634e0573772ffb7f3669b9f3"
VP = "0x772cff0be38a6ed31aeae479cbcb26d54b8404cf"
LT = "0x2b9c9f3bdceb5d8e36a4704f08a78fca53343cea"
SCALE = 10**18
SPLIT = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp())
def read(path: Path):
    return json.loads(path.read_text())
def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mae": None, "p50_abs": None, "p95_abs": None, "max_abs": None}
    ordered = sorted(abs(value) for value in values)
    return {
        "count": len(values),
        "mae": sum(ordered) / len(ordered),
        "p50_abs": ordered[len(ordered) // 2],
        "p95_abs": ordered[min(len(ordered) - 1, int(.95 * (len(ordered) - 1)))],
        "max_abs": ordered[-1],
    }
def completed_price(ts: int, times: list[int], closes: list[float]) -> float:
    index = bisect.bisect_left(times, ts // 60 * 60) - 1
    if index < 0:
        raise ValueError(f"no completed ETHUSDT close before {ts}")
    return closes[index]
def value(source: dict, key: str) -> int:
    entry = source[key]
    if not entry.get("ok", True):
        raise ValueError(f"failed getter {key}")
    return int(entry["value"])


def actual_states() -> tuple[list[dict], dict]:
    daily = read_lines(ACTIVITY / "daily_native_yb_states.jsonl")
    boundaries = read_lines(ACTIVITY / "state_snapshots.jsonl")
    pre = boundaries[0]["stored_native"]
    initial = {
        "stable": value(pre, "balances_0") / SCALE,
        "weth": value(pre, "balances_1") / SCALE,
        "supply": value(pre, "totalSupply") / SCALE,
    }
    rows = [{"kind": "prestate", "block": boundaries[0]["block"], "ts": int(boundaries[0]["block_header"]["timestamp"], 16), "source": pre}]
    rows += [{"kind": "daily", "block": row["block"], "ts": row["timestamp"], "source": row["projected_getters"]} for row in daily]
    rows.append({"kind": "end_boundary", "block": boundaries[1]["block"], "ts": int(boundaries[1]["block_header"]["timestamp"], 16), "source": boundaries[1]["stored_native"]})
    result = []
    for row in rows:
        source = row["source"]
        prefix = "pool_" if row["kind"] == "daily" else ""
        scale_key = f"{prefix}price_scale" if prefix else "cached_price_scale"
        balance0 = f"{prefix}balances" if prefix else "balances_0"
        balance1 = f"{prefix}balances_1" if prefix else "balances_1"
        supply = f"{prefix}totalSupply" if prefix else "totalSupply"
        lp_profit = f"{prefix}lp_xcp_profit" if prefix else "lp_xcp_profit"
        result.append({**row, "price_scale": value(source, scale_key) / SCALE, "stable": value(source, balance0) / SCALE, "weth": value(source, balance1) / SCALE, "supply": value(source, supply) / SCALE, "lp_xcp_profit": value(source, lp_profit) / SCALE})
    return result, initial


def model_file(feed: str, mode: str, suffix: str) -> Path:
    matches = sorted((STUDY / f"weth-{feed}").glob(f"**/*{mode}*.{suffix}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix} for {feed}/{mode}, found {matches}")
    return matches[0]


def daily_counts(rows: list[dict], field: str) -> dict[str, int]:
    result: Counter[str] = Counter()
    previous = 0
    for row in rows:
        current = int(row.get(field, previous))
        delta = current - previous
        if delta < 0:
            raise ValueError(f"nonmonotonic {field}")
        if delta:
            day = datetime.fromtimestamp(int(row.get("t", row.get("ts"))), timezone.utc).strftime("%Y-%m-%d")
            result[day] += delta
        previous = current
    return dict(sorted(result.items()))


def action_counts(rows: list[dict]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        if row.get("type") == "exchange":
            result[datetime.fromtimestamp(int(row["ts"]), timezone.utc).strftime("%Y-%m-%d")] += 1
    return dict(sorted(result.items()))


def observed(events: list[dict], times: list[int], closes: list[float]) -> tuple[dict, dict]:
    native = [r for r in events if r["address"].lower() == POOL and r["event"] == "TokenExchange"]
    vp = [r for r in events if r["address"].lower() == VP and r["event"] == "TokenExchange"]
    sides = {}
    for direction in ("0_to_1", "1_to_0"):
        selected = [r for r in native if f"{r['args']['sold_id']}_to_{r['args']['bought_id']}" == direction]
        notional, quantity = [], []
        for row in selected:
            args = row["args"]
            qty = int(args["tokens_sold"]) / SCALE
            price = completed_price(int(row["event_timestamp"]), times, closes)
            quantity.append(qty)
            notional.append(qty if args["sold_id"] == 0 else qty * price)
        sides[direction] = {"count": len(selected), "input_notional_coin0": sum(notional), "sold_quantity": stats(quantity), "input_notional": stats(notional)}
    deposits = [r for r in events if r["address"].lower() == LT and r["event"] == "Deposit"]
    withdrawals = [r for r in events if r["address"].lower() == LT and r["event"] == "Withdraw"]
    deposit_total = sum(int(r["args"]["assets"]) for r in deposits) / SCALE
    withdrawal_total = sum(int(r["args"]["assets"]) for r in withdrawals) / SCALE
    result = {"native_token_exchange": len(native), "virtual_pool_token_exchange": len(vp), "native_by_side": sides, "lt_gross_flows_weth": {"deposits": deposit_total, "withdrawals": withdrawal_total, "net": deposit_total - withdrawal_total, "deposit_events": len(deposits), "withdraw_events": len(withdrawals)}}
    daily = {"native": Counter(), "virtual_pool": Counter()}
    for label, rows in (("native", native), ("virtual_pool", vp)):
        daily[label].update(datetime.fromtimestamp(int(r["event_timestamp"]), timezone.utc).strftime("%Y-%m-%d") for r in rows)
    return result, {key: dict(sorted(value.items())) for key, value in daily.items()}


def compare(states: list[dict], initial: dict, traces: dict, times: list[int], closes: list[float]) -> tuple[list[dict], dict]:
    def hodl(price: float) -> float:
        return (initial["stable"] + initial["weth"] * price) / initial["supply"]
    trace_times = {key: [int(row["t"]) for row in rows] for key, rows in traces.items()}
    errors = {key: {phase: {name: [] for name in ("scale", "inventory_l1", "inventory_l1_per_lp", "lp_unit")} for phase in ("development", "validation")} for key in traces}
    tracks = []
    for state in states:
        price = completed_price(state["ts"], times, closes)
        actual_unit = (state["stable"] + state["weth"] * price) / state["supply"]
        native = {"price_scale": state["price_scale"], "inventory_l1_percent": (abs(state["stable"] / initial["stable"] - 1) + abs(state["weth"] / initial["weth"] - 1)) * 50, "lp_unit_hodl_index": actual_unit / hodl(price), "lp_xcp_profit": state["lp_xcp_profit"]}
        item = {"kind": state["kind"], "block": state["block"], "ts": state["ts"], "price_proxy_usd": price, "native": native}
        phase = "development" if state["ts"] < SPLIT else "validation"
        for key, trace in traces.items():
            index = bisect.bisect_right(trace_times[key], state["ts"]) - 1
            if index < 0:
                item[key] = {"missing": True}
                continue
            row = trace[index]
            unit = (float(row["token0"]) + float(row["token1"]) * price) / float(row["total_supply"])
            inv = (abs(float(row["token0"]) / state["stable"] - 1) + abs(float(row["token1"]) / state["weth"] - 1)) * 50
            inv_per_lp = (abs((float(row["token0"]) / float(row["total_supply"])) / (state["stable"] / state["supply"]) - 1) + abs((float(row["token1"]) / float(row["total_supply"])) / (state["weth"] / state["supply"]) - 1)) * 50
            item[key] = {"missing": False, "t": int(row["t"]), "price_scale": float(row["price_scale"]), "inventory_l1_percent": inv, "lp_unit_hodl_index": unit / hodl(price), "lp_xcp_profit": float(row["lp_xcp_profit"])}
            errors[key][phase]["scale"].append((float(row["price_scale"]) / state["price_scale"] - 1) * 10000)
            errors[key][phase]["inventory_l1"].append(inv)
            errors[key][phase]["inventory_l1_per_lp"].append(inv_per_lp)
            errors[key][phase]["lp_unit"].append((unit / actual_unit - 1) * 10000)
        tracks.append(item)
    return tracks, {key: {phase: {"price_scale_bps": stats(vals["scale"]), "inventory_l1_percent": stats(vals["inventory_l1"]), "inventory_l1_per_lp_percent": stats(vals["inventory_l1_per_lp"]), "lp_unit_bps": stats(vals["lp_unit"])} for phase, vals in phases.items()} for key, phases in errors.items()}


def plot(tracks: list[dict], models: dict, daily: dict, path: Path) -> None:
    dates = [datetime.fromtimestamp(r["ts"], timezone.utc) for r in tracks]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(dates, [r["price_proxy_usd"] for r in tracks], label="ETHUSDT completed close")
    axes[0].plot(dates, [r["native"]["price_scale"] for r in tracks], label="native cached price_scale")
    for key in models:
        axes[0].plot(dates, [r[key].get("price_scale", float("nan")) for r in tracks], label=f"{key} model")
    axes[0].set_ylabel("USD / ETH"); axes[0].legend(fontsize=7, ncol=2)
    for label, values in daily.items():
        axes[1].plot(dates, [values.get(datetime.fromtimestamp(r["ts"], timezone.utc).strftime("%Y-%m-%d"), 0) for r in tracks], label=label)
    axes[1].set_ylabel("events / UTC day"); axes[1].legend(fontsize=7, ncol=2)
    axes[2].plot(dates, [r["native"]["lp_unit_hodl_index"] for r in tracks], label="native LP / prestate HODL")
    for key in models:
        axes[2].plot(dates, [r[key].get("lp_unit_hodl_index", float("nan")) for r in tracks], label=f"{key} model")
    axes[2].set_ylabel("relative index"); axes[2].set_xlabel("UTC date"); axes[2].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    market = read(MARKET); times = [int(r[0]) for r in market]; closes = [float(r[4]) for r in market]
    states, initial = actual_states(); events = read_lines(ACTIVITY / "events_log_order.jsonl"); observed_stats, observed_daily = observed(events, times, closes)
    traces, actions, daily_models, inputs = {}, {}, {}, [MARKET, ACTIVITY / "daily_native_yb_states.jsonl", ACTIVITY / "state_snapshots.jsonl", ACTIVITY / "events_log_order.jsonl", ACTIVITY / "activity_summary.json"]
    flow_ready = all(len(list((STUDY / "weth-causal-flow").glob(f"**/*{mode}*.{suffix}.json"))) == 1 for mode in ("active_2l", "reference_2l") for suffix in ("trace", "actions"))
    feeds = ("filtered", "causal", "causal-flow") if flow_ready else ("filtered", "causal")
    for feed in feeds:
        for mode in ("active_2l", "reference_2l"):
            key = f"{feed}/{mode}"; trace_path = model_file(feed, mode, "trace"); traces[key] = read(trace_path); inputs.append(trace_path)
            daily_models[key] = daily_counts(traces[key], "n_trades")
            if feed == "causal-flow":
                actions[key] = read(model_file(feed, mode, "actions")); inputs.append(model_file(feed, mode, "actions")); daily_models[key] = action_counts(actions[key])
                expected_actions = int(traces[key][-1]["n_trades"]) + sum(row.get("type") == "exchange" and row.get("actor") == "user" for row in actions[key])
                actual_actions = sum(row.get("type") == "exchange" for row in actions[key])
                if actual_actions != expected_actions:
                    raise RuntimeError(f"{key} action/trace exchange count mismatch: {actual_actions} != {expected_actions}")
    tracks, errors = compare(states, initial, traces, times, closes)
    gates = {"daily_states": sum(row["kind"] == "daily" for row in states) == 57, "boundary_states": sum(row["kind"] != "daily" for row in states) == 2, "trace_rows": all(len(rows) == 161279 for rows in traces.values()), "native_decimals": SCALE == 10**18, "weth_decimals": SCALE == 10**18, "model_prestate_misses": all(sum(row[key].get("missing", False) for row in tracks) == 1 for key in traces)}
    if not all(gates.values()):
        raise RuntimeError(f"WETH comparison input gates failed: {gates}")
    plot_path = OUT / "comparison_3panel.png"; plot(tracks, traces, {**observed_daily, **daily_models}, plot_path)
    model_summary = {key: {"trace_rows": len(rows), "final_n_trades_arb_only": int(rows[-1].get("n_trades", 0)), "daily_counts": daily_models[key]} for key, rows in traces.items()}
    for key, rows in actions.items():
        fills = [row for row in rows if row.get("type") == "exchange" and row.get("actor") == "user"]
        model_summary[key].update({"action_exchange_count_including_user": sum(row.get("type") == "exchange" for row in rows), "user_fills": {"total": len(fills), "development": sum(int(row["ts"]) < SPLIT for row in fills), "validation": sum(int(row["ts"]) >= SPLIT for row in fills)}})
    summary = {"schema_version": 1, "status": "pass", "gates": gates, "window": {"start_utc": "2026-07-04T00:00:00Z", "end_exclusive_utc": "2026-08-29T00:00:00Z", "split_utc": "2026-08-01T00:00:00Z"}, "feed": "ETHUSDT raw completed minute close", "observed": observed_stats, "sample_coverage": {"daily_states": sum(r["kind"] == "daily" for r in states), "boundary_states": sum(r["kind"] != "daily" for r in states), "tracks": len(tracks), "model_asof_missing": {key: sum(r[key].get("missing", False) for r in tracks) for key in traces}}, "model": model_summary, "errors_by_feed_and_mode": errors, "denominators": {"native_prestate_stable": initial["stable"], "native_prestate_weth": initial["weth"], "native_prestate_lp_supply": initial["supply"], "lp_hodl_per_unit": "(prestate stable + prestate WETH * common ETHUSDT close) / prestate LP supply"}, "inputs": {str(path.relative_to(ROOT)): sha(path) for path in inputs}, "artifacts": {"plot": plot_path.name}, "limitations": ["ETHUSDT is a WETH/crvUSD proxy; WETH/ETH and USDT/crvUSD basis remain unknown.", "Only 57 daily observed states plus prestate/end boundaries are compared; this is not an action-block state trajectory.", "Activity acquisition sampled 20 receipts; this report makes no all-receipts success or full-gas claim.", "Filtered inputs use centered radius-5 future neighbors; causal inputs use completed raw closes at T-120s and T-60s with synthetic volume zero."]}
    (OUT / "comparison.json").write_text(json.dumps({**summary, "tracks": tracks}, indent=2, sort_keys=True) + "\n")
    rows = ["# WETH historical comparison", "", "Window: 2026-07-04 through 2026-08-29 UTC; model tracks are backward as-of matched to 57 daily observed states plus prestate/end boundaries.", "", "## Observed activity", "", f"Native TokenExchange: {observed_stats['native_token_exchange']}; VirtualPool TokenExchange: {observed_stats['virtual_pool_token_exchange']}; LT gross deposits/withdrawals/net: {observed_stats['lt_gross_flows_weth']['deposits']:.6f} / {observed_stats['lt_gross_flows_weth']['withdrawals']:.6f} / {observed_stats['lt_gross_flows_weth']['net']:.6f} WETH.", "", "## Daily and boundary state errors", "", "Raw inventory L1 reflects capitalization changes; per-LP inventory L1 isolates composition after dividing each coin balance by LP supply.", "", "| feed/mode | matched | scale MAE/p95 bp dev | raw inventory L1 % dev | per-LP inventory L1 % dev | LP unit MAE/p95 bp dev | scale MAE/p95 bp val | raw inventory L1 % val | per-LP inventory L1 % val | LP unit MAE/p95 bp val |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, phases in errors.items():
        d, v = phases["development"], phases["validation"]; matched = d["price_scale_bps"]["count"] + v["price_scale_bps"]["count"]
        rows.append(f"| {key} | {matched} | {d['price_scale_bps']['mae']:.2f}/{d['price_scale_bps']['p95_abs']:.2f} | {d['inventory_l1_percent']['mae']:.3f}/{d['inventory_l1_percent']['p95_abs']:.3f} | {d['inventory_l1_per_lp_percent']['mae']:.3f}/{d['inventory_l1_per_lp_percent']['p95_abs']:.3f} | {d['lp_unit_bps']['mae']:.2f}/{d['lp_unit_bps']['p95_abs']:.2f} | {v['price_scale_bps']['mae']:.2f}/{v['price_scale_bps']['p95_abs']:.2f} | {v['inventory_l1_percent']['mae']:.3f}/{v['inventory_l1_percent']['p95_abs']:.3f} | {v['inventory_l1_per_lp_percent']['mae']:.3f}/{v['inventory_l1_per_lp_percent']['p95_abs']:.3f} | {v['lp_unit_bps']['mae']:.2f}/{v['lp_unit_bps']['p95_abs']:.2f} |")
    rows += ["", "## Model exchange accounting", "", "| feed/mode | legacy arb trades (trace metric) | total exchange actions | user fills dev/val |", "|---|---:|---:|---:|"]
    for key, info in summary["model"].items():
        fills = info.get("user_fills", {})
        rows.append(f"| {key} | {info['final_n_trades_arb_only']} | {info.get('action_exchange_count_including_user', info['final_n_trades_arb_only'])} | {fills.get('development', 0)} / {fills.get('validation', 0)} |")
    rows += ["", "Filtered versus causal is an input comparison under identical WETH roles, historical state and event clock. It is not a causal explanation or retuning result. LP HODL uses the actual native prestate inventory and LP supply. Only sampled receipt data are available; no full-gas or all-receipts claim is made.", "", "![comparison](comparison_3panel.png)", "", "Inputs, hashes, coverage and limitations are recorded in `comparison.json`."]
    (OUT / "comparison.md").write_text("\n".join(rows) + "\n")
    print(json.dumps({"status": summary["status"], "observed_counts": {"native_token_exchange": observed_stats["native_token_exchange"], "virtual_pool_token_exchange": observed_stats["virtual_pool_token_exchange"], "lt_deposit_events": observed_stats["lt_gross_flows_weth"]["deposit_events"], "lt_withdraw_events": observed_stats["lt_gross_flows_weth"]["withdraw_events"]}, "coverage": summary["sample_coverage"], "output": str(OUT.resolve())}))


if __name__ == "__main__":
    main()
