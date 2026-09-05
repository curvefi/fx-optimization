"""Run matched active/reference baselines with the observed YB checkpoint."""
from __future__ import annotations

import argparse, difflib, hashlib, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904"
ACTIVITY = STUDY / "activity"
CONFIG = STUDY / "configuration/configuration_window.json"
CFG = ROOT / "configs/autoresearch/yb-cbbtc-historical-20260904"
OUT = STUDY / "historical-state"
BINARY = ROOT.parent / "curve-fx-arb-harness/build/yb-historical-state/arb_evaluator_f64"
MODES = ("active_2l", "reference_2l")
WAD = 10**18

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def human(v: int) -> float:
    return v / WAD

def build_state() -> dict:
    snap = json.loads((ACTIVITY / "state_snapshots.jsonl").read_text().splitlines()[0])
    ext = json.loads((ACTIVITY / "yb_external_inputs.json").read_text())["blocks"][0]
    req = ext["requests"]
    amm = snap["stored_amm"]
    role = snap["role_balances"]
    state = {
        "source_block": snap["block"], "source_timestamp": int(snap["block_header"]["timestamp"], 16),
        "block_hash": snap["block_header"]["hash"], "leverage": human(req["amm_leverage"]["value"]),
        "fee": human(amm["fee"]["value"]), "collateral": human(amm["collateral_amount"]["value"]),
        "debt": human(amm["debt"]["value"]), "rate": human(amm["rate"]["value"]),
        "rate_mul": human(amm["rate_mul"]["value"]), "rate_time": amm["rate_time"]["value"],
        "minted": human(amm["minted"]["value"]), "redeemed": human(amm["redeemed"]["value"]),
        "stable_balance": human(role["stable_balance_amm"]["value"]),
        "lt_stable_balance": human(role["lt_stable_balance"]["value"]),
        "flash_max_loan": human(req["factory_flash_max_loan_crvusd"]["value"]),
        "stable_aggregator": human(req["oracle_agg_price"]["value"]),
        "rounding_discount": human(req["vp_rounding_discount"]["value"]),
        "lt_donation_discount": 0.01, "killed": bool(amm["is_killed"]["value"]),
    }
    checks = {
        "prestate_boundary": state["source_block"] == 25455433 and state["source_timestamp"] == 1783123199,
        "rate_before_source": state["rate_time"] <= state["source_timestamp"],
        "expected_leverage": state["leverage"] == 2.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"historical state input validation failed: {checks}")
    return state

def config_text(mode: str, state: dict, cfg_base: Path, market_path: Path | None, evaluator: Path) -> str:
    baseline = (cfg_base / f"{mode}.toml").read_text()
    evaluator_ref = os.path.relpath(evaluator.resolve(), cfg_base.resolve())
    baseline = "\n".join(
        f'evaluator = "{evaluator_ref}"' if line.startswith("evaluator = ") else line
        for line in baseline.splitlines()
    ) + "\n"
    if market_path is not None:
        market = os.path.relpath(market_path.resolve(), cfg_base)
        baseline = "\n".join(
            f'market = "{market}"' if line.startswith("market = ") else line
            for line in baseline.splitlines()
        ) + "\n"
    baseline = baseline.replace("yb-historical-baseline/arb_evaluator_f64", "yb-historical-state/arb_evaluator_f64")
    baseline = baseline.replace(f"yb-cbbtc-historical-20260904-{mode}-f64", f"yb-cbbtc-historical-20260904-historical-{mode}-f64")
    fields = [f"{k} = {json.dumps(v)}" for k, v in state.items()]
    return baseline.rstrip() + "\n\n[session.yb_initial_state]\n" + "\n".join(fields) + "\n"

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}, check=True)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-run", action="store_true"); ap.add_argument("--namespace", default=""); ap.add_argument("--market", type=Path, default=None); ap.add_argument("--binary", type=Path, default=BINARY, help="historical-state evaluator executable"); args = ap.parse_args()
    state = build_state()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"historical-state evaluator is missing: {binary}")
    cfg_base = CFG / args.namespace; out = STUDY / (f"{args.namespace}/historical-state" if args.namespace else "historical-state")
    out.mkdir(parents=True, exist_ok=True); diffs = {}; results = []; before = {}
    for mode in MODES:
        cfg = cfg_base / f"historical_{mode}.toml"; baseline_text = (cfg_base / f"{mode}.toml").read_text(); after = config_text(mode, state, cfg_base, args.market, binary); cfg.write_text(after)
        diffs[mode] = list(difflib.unified_diff(baseline_text.splitlines(), after.splitlines(), fromfile=f"{mode}.toml", tofile=cfg.name, lineterm=""))
        run_dir, trace_dir = out / mode, out / f"{mode}-trace"
        if not args.skip_run:
            run(["uv", "run", "fxopt", "run", str(cfg), "--output", str(run_dir), "--overwrite"])
            run(["uv", "run", "fxopt", "shiftclick", str(run_dir), "--ordinal", "0", "--output", str(trace_dir), "--trace-interval", "1", "--actions", "--yb-mode", mode, "--yb-cash-multiplier", "1.0"])
        shift = trace_dir / "shiftclick.json"
        baseline_shift = STUDY / f"{args.namespace or 'baseline'}/{mode}-trace/shiftclick.json"
        if baseline_shift.exists():
            before[mode] = json.loads(baseline_shift.read_text())["result"]["metrics"]
        results.append({"mode": mode, "config": str(cfg), "run": str(run_dir), "trace": str(trace_dir), "run_json": (run_dir / "run.json").exists(), "results_npz": (run_dir / "results.npz").exists(), "shiftclick": str(shift), "result": json.loads(shift.read_text())["result"] if shift.exists() else None})
    doc = {"status": "historical_state_matched_comparison", "state": state, "binary": str(binary), "binary_sha256": sha256(binary), "binary_used_for_execution": not args.skip_run, "baseline_binary_sha256": "7097a1f007bf8671ecff8b608f43fc49c4b99c185ad62bb51c1dfdfba78c715b", "configuration_sha256": sha256(CONFIG), "config_diffs": diffs, "before_baseline_metrics": before, "results": results, "limitations": ["Historical YB state is a checkpoint, not a chronological YB tape.", "Only active_2l and reference_2l are run; baseline off is unchanged and read-only.", "Metric changes are simulator comparisons and are not an onchain improvement claim."]}
    (out / "comparison.json").write_text(json.dumps(doc, indent=2) + "\n"); (out / "historical_state.json").write_text(json.dumps(state, indent=2) + "\n"); print(json.dumps({"status": doc["status"], "modes": [{"mode": x["mode"], "run_json": x["run_json"], "results_npz": x["results_npz"]} for x in results]}, indent=2))

if __name__ == "__main__": main()
