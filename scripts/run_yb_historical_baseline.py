"""Build fixed cbBTC checkpoint configs and run three bounded f64 baselines."""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "curve-fx-arb-harness"
TWOC = ROOT.parent / "twocrypto-cpp"
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904"
ACTIVITY = STUDY / "activity"
CONFIGURATION = STUDY / "configuration/configuration_window.json"
MARKET = ROOT / "data/market/btcusd/candles-2026-07-04-2026-09-03-raw.json"
PRESTART = 25_455_433
START_TS = 1_783_123_200
CHOSEN_INTERVAL = json.loads(CONFIGURATION.read_text())["chosen_candidate"]
END_TS = CHOSEN_INTERVAL["end_timestamp"] + 1
CHOSEN_END_BLOCK = CHOSEN_INTERVAL["end_block"]
RATE = 259_443_752
SECONDS_YEAR = 365 * 86_400
TARGET_APY = RATE * SECONDS_YEAR / 2 / 10**18
PRECISION_1_TO_18 = 10**10
MODES = ("off", "active_2l", "reference_2l")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def value(row: dict, name: str) -> int:
    return int(row[name]["value"])

def historical(row: dict) -> dict:
    native = row["stored_native"]
    pair = lambda a: [str(value(native, a[0])), str(value(native, a[1]) * PRECISION_1_TO_18)]
    return {
        "source_block": PRESTART, "source_timestamp": int(row["block_header"]["timestamp"], 16),
        "balances": pair(("balances_0", "balances_1")),
        "admin_balances": pair(("admin_balances_0", "admin_balances_1")),
        "last_admin_fee_claim_timestamp": 0,
        "D": str(value(native, "D")), "total_supply": str(value(native, "totalSupply")),
        "price_scale": str(value(native, "cached_price_scale")), "price_oracle": str(value(native, "cached_price_oracle")),
        "last_prices": str(value(native, "last_prices")), "last_timestamp": value(native, "last_timestamp"),
        "virtual_price": str(value(native, "virtual_price")), "xcp_profit": str(value(native, "xcp_profit")),
        "lp_xcp_profit": str(value(native, "lp_xcp_profit")), "donation_shares": str(value(native, "donation_shares")),
        "last_donation_release_ts": value(native, "last_donation_release_ts"),
        "donation_protection_expiry_ts": value(native, "donation_protection_expiry_ts"),
        "donation_protection_period": value(native, "donation_protection_period"),
        "donation_protection_lp_threshold": str(value(native, "donation_protection_lp_threshold")),
        "donation_protection_extension_remainder": str(value(native, "donation_protection_extension_remainder")),
        "donation_shares_max_ratio": "100000000000000000",
    }

def template(state: dict) -> dict:
    return {"pools": [{"tag": "yb_cbbtc_historical_native_checkpoint", "pool": {
        "A": "50000", "gamma": "11111111111",
        "mid_fee": "146000000", "out_fee": "170000000", "fee_gamma": "54202748000000000",
        "adjustment_step_min": "100000000", "adjustment_step_max": "5000000000000000", "ma_time": "865",
        "reserved_profit_fraction": "3010101009", "admin_fee": "0", "policy": {"kind": "none"},
        "initial_price": state["price_scale"], "start_timestamp": START_TS, "historical_state": state,
        "donation_apy": TARGET_APY, "donation_frequency": 86400, "donation_duration": 604800,
    }, "costs": {"arb_fee_bps": 0, "gas_coin0": 0.0, "use_volume_cap": False, "volume_cap_mult": 1.0}}]}

def write_config(path: Path, template_path: Path, mode: str, out: Path, market_path: Path) -> None:
    evaluator = os.path.relpath(HARNESS / "build/yb-historical-baseline/arb_evaluator_f64", path.parent)
    market = os.path.relpath(market_path, path.parent)
    text = f'''[run]\nid = "yb-cbbtc-historical-20260904-{mode}-f64"\nevaluator = "{evaluator}"\ntemplate = "template.json"\nbatch_size = 1\nworkers = 2\nmetric_fields = ["lp_xcp_profit", "apy", "apy_net", "avg_rel_price_diff", "max_rel_price_diff", "final_rel_price_diff", "trades", "n_rebalances", "donations", "donation_coin0_total", "tvl_growth", "elapsed_ms", "yb_apy", "yb_apy_gm", "yb_final_growth", "yb_fee", "yb_releverage_trades", "yb_gm_windows", "yb_gm_floored_windows", "yb_gm_floor_share"]\n\n[session]\nn_candles = 0\nstart_time = {START_TS}\nend_time = {END_TS}\ncandle_filter = 99.0\nmin_swap = 0.000001\nmax_swap = 1.0\ndustswap_freq_s = 3600\nuser_swap_freq_s = 0\nevent_cursor = "scalar"\nmetric_profile = "full_summary"\nyb_releverage_fee = 0.013\nyb_cash_multiplier = 1.0\n\n[scenario]\nid = "yb-cbbtc-historical-20260904-{mode}"\nmarket = "{market}"\nyb_mode = "{mode}"\n'''
    donation = "" if mode != "off" else "donation_apy = 0.0\ndonation_frequency = 0\n"
    text = text.replace("\n\n[session]", f"\n\n[candidate]\n[candidate.defaults]\npolicy_params = []\n[candidate.defaults.pool]\n{donation}\n[session]")
    path.write_text(text)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-run", action="store_true"); ap.add_argument("--market", type=Path, default=MARKET); ap.add_argument("--namespace", default=""); args = ap.parse_args()
    state_row = json.loads(ACTIVITY.joinpath("state_snapshots.jsonl").read_text().splitlines()[0])
    out = STUDY / (args.namespace or "baseline"); cfg = ROOT / "configs/autoresearch/yb-cbbtc-historical-20260904" / args.namespace; cfg.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    binary = HARNESS / "build/yb-historical-baseline/arb_evaluator_f64"
    if not binary.exists(): raise SystemExit("missing isolated baseline evaluator")
    core_headers = ["twocrypto.hpp", "stableswap_math.hpp", "helpers.hpp", "price_scale_actuator.hpp", "policy.hpp", "policy_types.hpp"]
    matches = {name: sha256(TWOC / "include/pools/twocrypto_fx" / name) == sha256(TWOC / "_install/include/pools/twocrypto_fx" / name) for name in core_headers}
    if not all(matches.values()): raise SystemExit("installed native headers do not match source")
    market = args.market.resolve(); state = historical(state_row); template_path = cfg / "template.json"; template_path.write_text(json.dumps(template(state), indent=2) + "\n")
    provenance = {"status": "valid_input_baseline", "namespace": args.namespace or "baseline", "binary": str(binary), "binary_sha256": sha256(binary), "binary_build_dir": str(binary.parent), "twocrypto_commit": subprocess.check_output(["git", "-C", str(TWOC), "rev-parse", "HEAD"], text=True).strip(), "harness_commit": subprocess.check_output(["git", "-C", str(HARNESS), "rev-parse", "HEAD"], text=True).strip(), "installed_native_header_matches": matches, "state_snapshot_sha256": sha256(ACTIVITY / "state_snapshots.jsonl"), "configuration_sha256": sha256(CONFIGURATION), "market": str(market), "market_sha256": sha256(market), "prestart_block": PRESTART, "start_timestamp": START_TS, "end_timestamp_exclusive": END_TS, "selected_interval_end_block": CHOSEN_END_BLOCK, "precision_coin1_to_wad": PRECISION_1_TO_18, "native_ma_time_raw": 865, "yb_rate_raw_wad_per_second": RATE, "target_donation_apy": TARGET_APY, "yb_releverage_fee": 0.013, "yb_cash_multiplier": 1.0, "donation_frequency_enabled_modes": 86400, "yb_donation_guard": "event_loop disables external donation scheduling when yb_on; enabled runs therefore apply calibrated dcfg.apy to synthetic fresh_2l without double counting"}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results = []
    for mode in MODES:
        config = cfg / f"{mode}.toml"; run_dir = out / mode; trace_dir = out / f"{mode}-trace"
        write_config(config, template_path, mode, run_dir, market)
        if not args.skip_run:
            subprocess.run(["uv", "run", "fxopt", "run", str(config), "--output", str(run_dir), "--overwrite"], cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}, check=True)
            subprocess.run(["uv", "run", "fxopt", "shiftclick", str(run_dir), "--ordinal", "0", "--output", str(trace_dir), "--trace-interval", "1", "--actions", "--yb-mode", mode, "--yb-cash-multiplier", "1.0"], cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}, check=True)
        shift = trace_dir / "shiftclick.json"
        if shift.exists():
            payload = json.loads(shift.read_text())["result"]
            (trace_dir / "effective_inputs.json").write_text(json.dumps(payload["artifacts"]["effective_inputs"], indent=2) + "\n")
        results.append({"mode": mode, "config": str(config), "run": str(run_dir), "trace": str(trace_dir), "run_json": (run_dir / "run.json").exists(), "results_npz": (run_dir / "results.npz").exists(), "trace_files": sorted(p.name for p in trace_dir.glob("*") if p.is_file()), "metrics": payload.get("metrics") if shift.exists() else None})
    (out / "baseline_manifest.json").write_text(json.dumps({"provenance": provenance, "modes": results, "notes": ["No exogenous user swaps.", "No external donation scheduler when YB is enabled.", "Enabled YB state is synthetic fresh_2l and is not a historical LT replay."]}, indent=2) + "\n")
    print(json.dumps({"status": "prepared" if args.skip_run else "completed", "modes": [{"mode": row["mode"], "run_json": row["run_json"], "results_npz": row["results_npz"]} for row in results], "output": str(out.resolve())}))

if __name__ == "__main__": main()
