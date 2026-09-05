#!/usr/bin/env python3
"""Freeze and run the observed-flow user-swap surrogate on the causal replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from decimal import Decimal as D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904"
CONFIG_BASE = ROOT / "configs/autoresearch/yb-cbbtc-historical-20260904"
SOURCE_CONFIG = CONFIG_BASE / "btcusdc-causal"
FLOW_CONFIG = CONFIG_BASE / "btcusdc-causal-flow"
FLOW_RUN = STUDY / "btcusdc-causal-flow"
OBSERVED = STUDY / "observed-flow/observed_flow.json"
SNAPSHOTS = STUDY / "activity/state_snapshots.jsonl"
RAW = ROOT / "data/market/btcusd/candles-2026-07-03T2357-2026-09-03-btcusdc-raw.json"
USER_BINARY = ROOT.parent / "curve-fx-arb-harness/build/yb-historical-userflow/arb_evaluator_f64"
START_TS, TRAIN_END_TS, END_TS = 1783123200, 1785542400, 1788004668
MODES = ("active_2l", "reference_2l")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive() -> dict:
    observed = json.loads(OBSERVED.read_text())
    selected = [row for row in observed["swaps"] if START_TS <= int(row["timestamp"]) < TRAIN_END_TS and row["materiality"] == "material" and D(str(row["input_value_coin0"])) >= 100 and D(str(row["surplus_bps"])) < -10]
    if len(selected) != 136:
        raise RuntimeError(f"expected 136 selected training swaps, found {len(selected)}")
    raw = json.loads(RAW.read_text())
    closes = {int(row[0]): D(str(row[4])) for row in raw}
    snapshot = json.loads(SNAPSHOTS.read_text().splitlines()[0])
    pre_ts = int(snapshot["block_header"]["timestamp"], 16)
    pre_close = closes[max(ts for ts in closes if ts < pre_ts // 60 * 60)]
    fair_tvl = D(snapshot["stored_native"]["balances_0"]["value"]) / D(10**18)
    fair_tvl += D(snapshot["stored_native"]["balances_1"]["value"]) / D(10**8) * pre_close
    inputs = sum(D(str(row["input_value_coin0"])) for row in selected)
    disadvantages = []
    buy, sell = [], []
    for row in selected:
        value = D(str(row["input_value_coin0"])) / D(str(row["output_value_coin0"])) - 1 if row["direction"] == "0_to_1" else 1 - D(str(row["output_value_coin0"])) / D(str(row["input_value_coin0"]))
        disadvantages.append(value); (buy if row["direction"] == "0_to_1" else sell).append(value)
    p95 = lambda values: sorted(values)[int(D("0.95") * (len(values) - 1))]
    threshold = max(p95(buy), p95(sell))
    selection_hash = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "selection": {"training_window": "2026-07-04T00:00:00Z..2026-08-01T00:00:00Z", "predicate": "surplus_bps < -10", "count": len(selected), "selection_sha256": selection_hash, "input_notional_coin0": str(inputs)}, "parameters": {"cadence_s": round((TRAIN_END_TS - START_TS) / len(selected)), "daily_fair_tvl_utilization": str(inputs / (D(28) * fair_tvl)), "user_swap_size_frac_semantics": "daily fair-TVL utilization; cadence scales per-swap notional", "user_swap_thresh": str(threshold), "p95_buy_ask_premium": str(p95(buy)), "p95_sell_bid_loss": str(p95(sell))}, "prestate": {"timestamp": pre_ts, "raw_btcusdc_close": str(pre_close), "fair_tvl_coin0": str(fair_tvl)}, "sources": {"observed_flow": sha(OBSERVED), "events": observed["summary"]["sources"]["events"], "raw_btcusdc": sha(RAW), "state_snapshots": sha(SNAPSHOTS), "causal_config_active": sha(SOURCE_CONFIG / "historical_active_2l.toml"), "causal_config_reference": sha(SOURCE_CONFIG / "historical_reference_2l.toml"), "causal_template": sha(SOURCE_CONFIG / "template.json")}, "observability": {"user_success_logging": False, "user_trade_metric_increment": False, "source": "curve-fx-arb-harness/cpp/include/harness/event_loop.hpp apply_user_swap and apply_user_swap call; ActionLogger logs arb exchanges only", "limitation": "scheduled user moments and successful user fills cannot be separated from logged actions until isolated observability fix"}}


def write_configs(spec: dict, evaluator: Path) -> None:
    FLOW_CONFIG.mkdir(parents=True, exist_ok=True); FLOW_RUN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_CONFIG / "template.json", FLOW_CONFIG / "template.json")
    params = spec["parameters"]
    for mode in MODES:
        text = (SOURCE_CONFIG / f"historical_{mode}.toml").read_text()
        evaluator_ref = os.path.relpath(evaluator.resolve(), FLOW_CONFIG.resolve())
        text = "\n".join(f'evaluator = "{evaluator_ref}"' if line.startswith("evaluator = ") else line for line in text.splitlines()) + "\n"
        text = text.replace("yb-historical-state/arb_evaluator_f64", "yb-historical-userflow/arb_evaluator_f64")
        text = text.replace(f"yb-cbbtc-historical-20260904-historical-{mode}-f64", f"yb-cbbtc-historical-20260904-causal-flow-{mode}-f64")
        text = text.replace(f"yb-cbbtc-historical-20260904-{mode}\"", f"yb-cbbtc-historical-20260904-causal-flow-{mode}\"")
        text = text.replace("user_swap_freq_s = 0", f"user_swap_freq_s = {params['cadence_s']}\nuser_swap_size_frac = {params['daily_fair_tvl_utilization']}\nuser_swap_thresh = {params['user_swap_thresh']}")
        (FLOW_CONFIG / f"{mode}.toml").write_text(text)


def run_mode(mode: str) -> None:
    config = FLOW_CONFIG / f"{mode}.toml"; run_dir = FLOW_RUN / mode; trace_dir = FLOW_RUN / f"{mode}-trace"
    env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}
    subprocess.run(["uv", "run", "fxopt", "run", str(config), "--output", str(run_dir), "--overwrite"], cwd=ROOT, env=env, check=True)
    subprocess.run(["uv", "run", "fxopt", "shiftclick", str(run_dir), "--ordinal", "0", "--output", str(trace_dir), "--trace-interval", "1", "--actions", "--yb-mode", mode, "--yb-cash-multiplier", "1.0"], cwd=ROOT, env=env, check=True)


def flow_stats(mode: str, baseline: dict, flow: dict, spec: dict) -> dict:
    trace = FLOW_RUN / f"{mode}-trace"; actions = json.loads(next(trace.glob("*.actions.json")).read_text())
    baseline_actions = json.loads(next((STUDY / f"btcusdc-causal/historical-state/{mode}-trace").glob("*.actions.json")).read_text())
    users = [a for a in actions if a.get("actor") == "user" and a.get("type") == "exchange"]
    def stats(rows):
        result = {}
        for period, lo, hi in (("development", START_TS, TRAIN_END_TS), ("validation", TRAIN_END_TS, END_TS)):
            part = [a for a in rows if lo <= int(a["ts"]) < hi]
            result[period] = {"count": len(part), "directions": {f"{i}_to_{i ^ 1}": sum(a["i"] == i for a in part) for i in (0, 1)}, "input_notional_coin0": sum(float(a["dx"]) if a["i"] == 0 else float(a["dx"]) * float(a["p_cex"]) for a in part)}
        return result
    def errors(report):
        return {p: {k: report["development_validation"]["path_errors"][mode][p][k] for k in ("scale_bps", "inventory_percent", "inventory_per_lp_percent", "marked_lp_unit_value_bps")} for p in ("development", "validation")}
    return {"baseline_metrics": baseline["result"]["metrics"], "flow_metrics": flow["result"]["metrics"], "baseline_metric_boundary": "trades/notional/LP fee/arb PnL are arb-only", "baseline_exchange_by_phase": stats([a for a in baseline_actions if a.get("type") == "exchange"]), "flow_exchange_by_phase": stats([a for a in actions if a.get("type") == "exchange"]), "flow_logged_exchange_actions": {"total": sum(a.get("type") == "exchange" for a in actions), "arb": sum(a.get("type") == "exchange" and a.get("actor") != "user" for a in actions), "user": len(users)}, "user_realized": stats(users), "user_attempts_scheduled": (END_TS - START_TS - 1) // int(spec["parameters"]["cadence_s"]), "path_errors": {"baseline": errors(json.loads((STUDY / "comparison-btcusdc-causal/compact_comparison.json").read_text())), "flow": errors(json.loads((STUDY / "comparison-btcusdc-causal-flow/compact_comparison.json").read_text()))}}

def write_report(spec: dict) -> None:
    baseline = {mode: json.loads((STUDY / f"btcusdc-causal/historical-state/{mode}-trace/shiftclick.json").read_text()) for mode in MODES}
    flow = {mode: json.loads((FLOW_RUN / f"{mode}-trace/shiftclick.json").read_text()) for mode in MODES}
    observed = json.loads(OBSERVED.read_text())["summary"]["counts"]
    report = {"calibration": spec, "observability_override": json.loads((FLOW_RUN / "run_provenance.json").read_text()), "observed_native_counts": {"total": observed["native_swaps"], "development": observed["dev"], "validation": observed["val"]}, "modes": {mode: flow_stats(mode, baseline[mode], flow[mode], spec) for mode in MODES}, "limitations": ["The selected flow is noisy feed-relative and not an identified non-arbitrage actor flow.", "User successful fills are actor-tagged by the isolated binary; failed scheduled moments emit no fill.", "Legacy trades, notional, LP fee and arb PnL metrics remain arb-only; user notional is separately reconstructed from tagged actions.", "Validation is chronologically exposed to development selection and prior August mechanics; no retuning was performed."]}
    (FLOW_RUN / "flow_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--report-only", action="store_true"); parser.add_argument("--binary", type=Path, default=USER_BINARY, help="user-flow evaluator executable"); args = parser.parse_args()
    evaluator = args.binary.resolve()
    if not evaluator.is_file():
        raise RuntimeError(f"user-flow evaluator is missing: {evaluator}")
    spec = derive(); spec_path = FLOW_RUN / "calibration_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise RuntimeError("frozen calibration spec differs from current inputs")
    FLOW_RUN.mkdir(parents=True, exist_ok=True); spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    (FLOW_RUN / "run_provenance.json").write_text(json.dumps({"binary": str(evaluator), "binary_sha256": sha(evaluator), "successful_user_action_tag": "actor=user", "legacy_metric_boundary": "trades/notional/LP fee/arb PnL remain arb-only"}, indent=2) + "\n")
    (FLOW_RUN / "calibration_context.json").write_text(json.dumps({"threshold_definition": "max(side-specific p95 selected ask premium for buys and bid loss for sells)", "flow_interpretation": "noisy feed-relative routed-flow surrogate; not an identified non-arbitrage actor classification", "direction_and_admission": "existing user_swap alternates directions and may reject; failed moments emit no fill", "validation_exposure": "selection uses development data and prior August mechanics plus aggregate paths, so chronological validation is not blind", "retuning": False, "user_observability": "approved isolated binary tags successful synthetic fills actor=user; legacy trades/notional/LP fee/arb PnL remain arb-only"}, indent=2) + "\n")
    write_configs(spec, evaluator)
    if args.prepare_only: print(json.dumps(spec, indent=2)); return
    if args.report_only: write_report(spec); return
    for mode in MODES: run_mode(mode)
    comparator = ROOT / "scripts/compare_yb_history.py"
    subprocess.run(["uv", "run", "python", str(comparator), "--study", str(STUDY), "--run-root", str(FLOW_RUN), "--market", str(RAW), "--prehistory", str(RAW), "--output", str(STUDY / "comparison-btcusdc-causal-flow")], cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}, check=True)
    write_report(spec)


if __name__ == "__main__": main()
