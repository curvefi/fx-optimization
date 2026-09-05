"""Run matched WETH historical-state experiments on two ETH feeds."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904"
ACTIVITY = STUDY / "weth-preparation/activity"
CONFIGURATION = STUDY / "weth-preparation/configuration_window.json"
HARNESS = ROOT.parent / "curve-fx-arb-harness"
BINARY = HARNESS / "build/yb-historical-state/arb_evaluator_f64"
USER_BINARY = HARNESS / "build/yb-historical-userflow/arb_evaluator_f64"
START_TS, END_TS, PRESTART = 1_783_123_200, 1_787_961_600, 25_455_433
RATE = 3_170_979_198
TARGET_APY = RATE * 365 * 86_400 / 2 / 10**18
MODES = ("active_2l", "reference_2l")
FEEDS = {
    "weth-filtered": ROOT / "data/market/ethusd/candles-2026-07-03T2347-2026-08-29-ethusdt-filtered.json",
    "weth-causal": ROOT / "data/market/ethusd/candles-2026-07-04-2026-08-28-ethusdt-causal.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(row: dict, name: str) -> int:
    return int(row[name]["value"])


def human(value: int) -> float:
    return value / 10**18


def load_inputs() -> tuple[dict, dict]:
    snap = json.loads((ACTIVITY / "state_snapshots.jsonl").read_text().splitlines()[0])
    ext = json.loads((ACTIVITY / "yb_external_inputs.json").read_text())["blocks"][0]
    checks = {
        "prestate_boundary": snap.get("block") == PRESTART and int(snap["block_header"]["timestamp"], 16) == 1_783_123_199,
        "storage_layout_verified": snap.get("storage_layout_verification", {}).get("verified", False),
        "stablecoin_decimals": snap.get("projected_getters", {}).get("token_" + snap["role_balances"]["stablecoin_address"][-6:] + "_decimals", {}).get("value") == 18,
    }
    if not all(checks.values()):
        raise RuntimeError(f"WETH input validation failed: {checks}")
    return snap, ext


def state(snap: dict, ext: dict) -> dict:
    n, a = snap["stored_native"], snap["stored_amm"]
    req, role = ext["requests"], snap["role_balances"]
    return {
        "source_block": snap["block"], "source_timestamp": int(snap["block_header"]["timestamp"], 16), "block_hash": snap["block_header"]["hash"].lower(),
        "leverage": human(req["amm_leverage"]["value"]), "fee": human(a["fee"]["value"]), "collateral": human(a["collateral_amount"]["value"]), "debt": human(a["debt"]["value"]),
        "rate": human(a["rate"]["value"]), "rate_mul": human(a["rate_mul"]["value"]), "rate_time": a["rate_time"]["value"], "minted": human(a["minted"]["value"]), "redeemed": human(a["redeemed"]["value"]),
        "stable_balance": human(role["stable_balance_amm"]["value"]), "lt_stable_balance": human(role["lt_stable_balance"]["value"]), "flash_max_loan": human(req["factory_flash_max_loan_crvusd"]["value"]),
        "stable_aggregator": human(req["oracle_agg_price"]["value"]), "rounding_discount": human(req["vp_rounding_discount"]["value"]), "lt_donation_discount": 0.01, "killed": bool(a["is_killed"]["value"]),
    }


def native_state(snap: dict) -> dict:
    n = snap["stored_native"]
    pair = lambda x: [str(raw(n, x[0])), str(raw(n, x[1]))]
    return {
        "source_block": PRESTART, "source_timestamp": int(snap["block_header"]["timestamp"], 16), "balances": pair(("balances_0", "balances_1")), "admin_balances": pair(("admin_balances_0", "admin_balances_1")), "last_admin_fee_claim_timestamp": 0,
        "D": str(raw(n, "D")), "total_supply": str(raw(n, "totalSupply")), "price_scale": str(raw(n, "cached_price_scale")), "price_oracle": str(raw(n, "cached_price_oracle")), "last_prices": str(raw(n, "last_prices")), "last_timestamp": raw(n, "last_timestamp"),
        "virtual_price": str(raw(n, "virtual_price")), "xcp_profit": str(raw(n, "xcp_profit")), "lp_xcp_profit": str(raw(n, "lp_xcp_profit")), "donation_shares": str(raw(n, "donation_shares")), "last_donation_release_ts": raw(n, "last_donation_release_ts"),
        "donation_protection_expiry_ts": raw(n, "donation_protection_expiry_ts"), "donation_protection_period": raw(n, "donation_protection_period"), "donation_protection_lp_threshold": str(raw(n, "donation_protection_lp_threshold")), "donation_protection_extension_remainder": str(raw(n, "donation_protection_extension_remainder")), "donation_shares_max_ratio": "100000000000000000",
    }


def template(native: dict) -> dict:
    return {"pools": [{"tag": "yb_weth_historical_native_checkpoint", "pool": {
        "A": "50000", "gamma": "11111111111", "mid_fee": "136000000", "out_fee": "282000000", "fee_gamma": "4961947600000000", "adjustment_step_min": "100000000", "adjustment_step_max": "5000000000000000", "ma_time": "865", "reserved_profit_fraction": "4500000000", "admin_fee": "0", "policy": {"kind": "none"}, "initial_price": native["price_scale"], "start_timestamp": START_TS, "historical_state": native, "donation_apy": TARGET_APY, "donation_frequency": 86400, "donation_duration": 604800,
    }, "costs": {"arb_fee_bps": 0, "gas_coin0": 0.0, "use_volume_cap": False, "volume_cap_mult": 1.0}}]}


def config_text(mode: str, evaluator: str, template_path: Path, market: Path, ns: str) -> str:
    return f'''[run]\nid = "yb-weth-{ns}-{mode}-historical-f64"\nevaluator = "{os.path.relpath(evaluator, template_path.parent)}"\ntemplate = "template.json"\nbatch_size = 1\nworkers = 2\nmetric_fields = ["lp_xcp_profit", "apy", "apy_net", "avg_rel_price_diff", "max_rel_price_diff", "final_rel_price_diff", "trades", "n_rebalances", "donations", "donation_coin0_total", "tvl_growth", "elapsed_ms", "yb_apy", "yb_apy_gm", "yb_final_growth", "yb_fee", "yb_releverage_trades", "yb_gm_windows", "yb_gm_floored_windows", "yb_gm_floor_share"]\n\n[session]\nn_candles = 0\nstart_time = {START_TS}\nend_time = {END_TS}\ncandle_filter = 99.0\nmin_swap = 0.000001\nmax_swap = 1.0\ndustswap_freq_s = 3600\nuser_swap_freq_s = 0\nevent_cursor = "scalar"\nmetric_profile = "full_summary"\nyb_releverage_fee = 0.013\nyb_cash_multiplier = 1.0\n\n[scenario]\nid = "yb-weth-{ns}-{mode}"\nmarket = "{os.path.relpath(market, template_path.parent)}"\nyb_mode = "{mode}"\n\n[session.yb_initial_state]\n'''


def validate(snap: dict, ext: dict, native: dict, yb: dict, cfg: dict) -> dict:
    vals = cfg["current_config"]["values"]
    checks = {"coin0_crvUSD": vals["coins0"] == "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e", "coin1_WETH": vals["coins1"] == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "A": vals["A"] == 50000, "gamma": vals["gamma"] == 11111111111, "mid_fee": vals["mid_fee"] == 136000000, "out_fee": vals["out_fee"] == 282000000, "fee_gamma": vals["fee_gamma"] == 4961947600000000, "RPF": vals["reserved_profit_fraction"] == 4500000000, "amm_fee": vals["yb_amm_fee"] == 13000000000000000, "rate": vals["yb_amm_rate"] == RATE, "ma_time_raw": snap["projected_getters"]["pool_packed_rebalancing_params"]["ma_exp_time_raw"] == 865, "state_source": yb["source_block"] == PRESTART and yb["block_hash"] == snap["block_header"]["hash"].lower(), "all_native_balances_WAD": all(len(native[k]) == 2 for k in ("balances", "admin_balances"))}
    if not all(checks.values()):
        raise RuntimeError(f"WETH configuration validation failed: {checks}")
    return {"passed": True, "checks": checks, "template_units": {"coin0": "WAD", "coin1": "WAD", "ma_time": "packed raw 865; getter half-time 600", "rate": "WAD per second converted to float for YB state"}, "derived_target_donation_apy": TARGET_APY}


def main() -> None:
    args = argparse.ArgumentParser(); args.add_argument("--flow-spec", type=Path); args.add_argument("--binary", type=Path, default=BINARY, help="historical-state evaluator executable"); args.add_argument("--user-binary", type=Path, default=USER_BINARY, help="user-flow evaluator executable"); opts = args.parse_args()
    flow_spec = json.loads(opts.flow_spec.read_text()) if opts.flow_spec else None
    snap, ext = load_inputs(); yb = state(snap, ext); native = native_state(snap); cfg_window = json.loads(CONFIGURATION.read_text()); evaluator = (opts.user_binary if flow_spec else opts.binary).resolve()
    if not evaluator.is_file():
        raise RuntimeError(f"selected evaluator is missing: {evaluator}")
    if flow_spec:
        expected_parameters = {"cadence_s": 17788, "daily_fair_tvl_utilization": "0.005774191735713352152791607973", "user_swap_size_frac_semantics": "daily fair-TVL utilization; cadence scales per-swap notional", "user_swap_thresh": "0.004767551429321027700966511", "p95_buy_ask_premium": "0.004767551429321027700966511", "p95_sell_bid_loss": "0.0027206382658489524610752699"}
        if opts.flow_spec is None or flow_spec.get("parameters") != expected_parameters:
            raise RuntimeError("calibration flow spec parameters do not match the frozen study")
    feeds = {"weth-causal-flow": FEEDS["weth-causal"]} if flow_spec else FEEDS
    for ns, market in feeds.items():
        out, cfgdir = STUDY / "weth-preparation" / ns, ROOT / "configs/autoresearch/yb-cbbtc-historical-20260904" / ns
        out.mkdir(parents=True, exist_ok=True); cfgdir.mkdir(parents=True, exist_ok=True)
        (cfgdir / "template.json").write_text(json.dumps(template(native), indent=2) + "\n")
        validation = validate(snap, ext, native, yb, cfg_window)
        (out / "input_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        provenance = {"status": "valid_input", "namespace": ns, "market": str(market), "market_sha256": sha(market), "market_rows": len(json.loads(market.read_text())), "binary": str(evaluator), "binary_sha256": sha(evaluator), "activity_dir": str(ACTIVITY), "configuration_sha256": sha(CONFIGURATION), "source_provenance": cfg_window.get("source_provenance"), "start_timestamp": START_TS, "end_timestamp_exclusive": END_TS, "prestate_block": PRESTART, "historical_rate_raw_wad_per_second": RATE, "target_donation_apy": TARGET_APY, "coin_scaling": "both native coin balances are WAD; no cbBTC 1e10 multiplier", "donation_guard": "external donation scheduler disabled when YB is on"}
        if flow_spec:
            provenance["calibration_spec"] = {"path": str(opts.flow_spec.resolve()), "sha256": sha(opts.flow_spec), "parameters": flow_spec["parameters"]}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if flow_spec:
            (out / "calibration_spec.json").write_text(json.dumps(flow_spec, indent=2) + "\n")
        flow_reports = {}
        for mode in MODES:
            cfg_path, run_dir, trace_dir = cfgdir / f"historical_{mode}.toml", out / mode, out / f"{mode}-trace"
            fields = "\n".join(f"{k} = {json.dumps(v)}" for k, v in yb.items())
            text = config_text(mode, evaluator, cfgdir / "template.json", market, ns)
            text = text.replace("\n\n[session]", "\n\n[candidate]\n[candidate.defaults]\npolicy_params = []\n[candidate.defaults.pool]\n\n[session]")
            if flow_spec:
                p = flow_spec["parameters"]
                text = text.replace("user_swap_freq_s = 0", f"user_swap_freq_s = {p['cadence_s']}\nuser_swap_size_frac = {p['daily_fair_tvl_utilization']}\nuser_swap_thresh = {p['user_swap_thresh']}")
            cfg_path.write_text(text + fields + "\n")
            subprocess.run(["uv", "run", "fxopt", "run", str(cfg_path), "--output", str(run_dir), "--overwrite"], cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}, check=True)
            subprocess.run(["uv", "run", "fxopt", "shiftclick", str(run_dir), "--ordinal", "0", "--output", str(trace_dir), "--trace-interval", "1", "--actions", "--yb-mode", mode, "--yb-cash-multiplier", "1.0"], cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}, check=True)
            shift = json.loads((trace_dir / "shiftclick.json").read_text())["result"]
            (trace_dir / "effective_inputs.json").write_text(json.dumps(shift["artifacts"]["effective_inputs"], indent=2) + "\n")
            if flow_spec:
                actions = json.loads(next(trace_dir.glob("*.actions.json")).read_text()); exchanges = [a for a in actions if a.get("type") == "exchange"]; users = [a for a in exchanges if a.get("actor") == "user"]
                flow_reports[mode] = {"logged_exchange_actions": {"total": len(exchanges), "arb": len(exchanges) - len(users), "user": len(users)}, "user_directions": {f"{i}_to_{i ^ 1}": sum(a.get("i") == i for a in users) for i in (0, 1)}, "user_fee_tokens_by_direction": {f"{i}_to_{i ^ 1}": sum(float(a.get("fee_tokens", 0.0)) for a in users if a.get("i") == i) for i in (0, 1)}, "legacy_metric_boundary": "trades, notional, LP fee, and arb PnL remain arb-only; total native exchanges equal arb plus user actions"}
        if flow_spec:
            (out / "flow_observability.json").write_text(json.dumps({"calibration_spec_sha256": sha(opts.flow_spec), "binary_sha256": sha(evaluator), "modes": flow_reports, "no_tuning": True}, indent=2) + "\n")
        print(json.dumps({"namespace": ns, "market_sha256": sha(market), "modes": list(MODES)}))


if __name__ == "__main__":
    main()
