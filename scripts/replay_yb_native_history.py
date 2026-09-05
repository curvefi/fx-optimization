#!/usr/bin/env python3
"""Adapt the observed cbBTC tape to the current uint parity harness."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, tempfile
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[3]
RAW_DATA = WORKSPACE / "cpp-twocrypto-modular/ai_research/onchain_study/data/august_25787654_25802001"
DEFAULT_SDK_REPO = Path(__file__).resolve().parents[2] / "twocrypto-cpp"
CB_BTC_SCALE = 10_000_000_000
UINT_RE = re.compile(r"^[0-9]+$")
SUPPORTED_ACTIONS = {"exchange", "add_liquidity", "donation", "remove_liquidity_fixed_out"}
SUPPORTED_STATE_FIELDS = ("D_raw", "balance0_raw", "balance1_raw", "balance1_normalized_wad_raw", "admin_balance0_raw", "admin_balance1_raw", "cached_price_scale_raw", "cached_price_oracle_raw", "cached_virtual_price_raw", "last_prices_raw", "total_supply_raw", "xcp_profit_raw", "lp_xcp_profit_raw", "donation_shares_raw", "last_donation_release_ts", "donation_protection_expiry_ts", "block_timestamp")

def uint(value: Any, label: str) -> int:
    if isinstance(value, (bool, float)) or not UINT_RE.fullmatch(str(value)):
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)

def raw(value: Any, label: str) -> str:
    return str(uint(value, label))

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def adapt_pool(source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    entry, pool = source["pools"][0], dict(source["pools"][0]["pool"])
    pool["name"] = entry.get("tag", "yb_cbbtc_historical")
    dropped = []
    for key in ("donation_duration", "ma_half_time_view_s"):
        if key in pool:
            dropped.append(f"pool.{key} (harness uses its compiled constant/view path)")
            pool.pop(key)
    policy = pool.get("policy")
    if not isinstance(policy, dict) or policy.get("kind") != "none" or policy.get("address", "0x0") != "0x" + "0" * 40:
        raise ValueError("fixture policy is outside the native no-policy adapter")
    pool["policy"] = {"kind": "none"}
    pool["precisions"] = ["1", str(CB_BTC_SCALE)]
    historical = pool["historical_state"]
    for key in ("balances", "admin_balances"):
        normalized = uint(historical[key][1], f"historical_state.{key}[1]")
        if normalized % CB_BTC_SCALE:
            raise ValueError(f"historical_state.{key}[1] is not raw-representable")
        historical[key][1] = str(normalized // CB_BTC_SCALE)
    dropped.append("pool.policy.address (zero binding; current parser accepts kind only)")
    return {"pools": [pool]}, dropped

def adapt_actions(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[int, int], Counter[str], list[str]]:
    if not records:
        raise ValueError("empty action tape")
    actions, positions, counts, previous_ts = [], {}, Counter(), None
    for index, record in enumerate(records):
        if uint(record["action_index"], "action_index") != index:
            raise ValueError(f"action index discontinuity at tape row {index}")
        kind, timestamp = record.get("action_type"), uint(record["block_timestamp"], "block_timestamp")
        if kind not in SUPPORTED_ACTIONS:
            raise ValueError(f"unsupported observed action type: {kind!r}")
        if previous_ts is not None and timestamp < previous_ts:
            raise ValueError(f"action timestamps are not ordered at action {index}")
        if previous_ts is None:
            previous_ts = timestamp
        elif timestamp != previous_ts:
            actions.append({"type": "time_travel", "timestamp": timestamp}); previous_ts = timestamp
        request = record.get("call", {}).get("requested_args")
        if not isinstance(request, dict):
            raise ValueError(f"action {index} has no call.requested_args")
        if kind == "exchange":
            coin_i, coin_j = uint(request["i"], "exchange.i"), uint(request["j"], "exchange.j")
            action = {"type": kind, "i": coin_i, "j": coin_j, "dx": raw(request["dx"], "exchange.dx"), "min_dy": raw(request.get("min_dy", 0), "exchange.min_dy")}
        elif kind in ("add_liquidity", "donation"):
            amounts = request["amounts"]
            if not isinstance(amounts, list) or len(amounts) != 2:
                raise ValueError(f"action {index} amounts must have two entries")
            if kind == "donation" and "paired_add_event" not in record:
                raise ValueError(f"donation action {index} is missing paired_add_event")
            action = {"type": "add_liquidity", "amounts": [raw(amounts[0], "add.amount0"), raw(amounts[1], "add.amount1")], "min_mint_amount": str(uint(request.get("min_mint_amount", 0), "add.min_mint_amount")), "donation": kind == "donation"}
        else:
            coin = uint(request["i"], "remove.i")
            action = {"type": kind, "token_amount": str(uint(request["token_amount"], "remove.token_amount")), "i": coin, "amount_i": raw(request["amount_i"], "remove.amount_i"), "min_amount_j": raw(request.get("min_amount_j", 0), "remove.min_amount_j")}
        actions.append(action); positions[index] = len(actions); counts[kind] += 1
    limitations = ["caller/receiver identity is represented by the harness persistent caller", "fee, block hash, projected getters, donation extension remainder, and last_timestamp are not emitted by the harness snapshot"]
    return {"sequences": [{"name": "yb_cbbtc_august_historical", "start_timestamp": records[0]["block_timestamp"], "actions": actions}]}, positions, counts, limitations

def canonical(snapshot: dict[str, Any]) -> dict[str, str]:
    balances, admin = snapshot["balances"], snapshot["admin_balances"]; b1, a1 = int(balances[1]), int(admin[1])
    return {"D_raw": str(snapshot["D"]), "balance0_raw": str(balances[0]), "balance1_raw": str(balances[1]), "balance1_normalized_wad_raw": str(b1 * CB_BTC_SCALE), "admin_balance0_raw": str(admin[0]), "admin_balance1_raw": str(admin[1]), "cached_price_scale_raw": str(snapshot["price_scale"]), "cached_price_oracle_raw": str(snapshot["price_oracle"]), "cached_virtual_price_raw": str(snapshot["virtual_price"]), "last_prices_raw": str(snapshot["last_prices"]), "total_supply_raw": str(snapshot["totalSupply"]), "xcp_profit_raw": str(snapshot["xcp_profit"]), "lp_xcp_profit_raw": str(snapshot["lp_xcp_profit"]), "donation_shares_raw": str(snapshot["donation_shares"]), "last_donation_release_ts": str(snapshot["last_donation_release_ts"]), "donation_protection_expiry_ts": str(snapshot["donation_protection_expiry_ts"]), "block_timestamp": str(snapshot["timestamp"])}

def compare(expected: dict[str, Any], actual: dict[str, Any], timestamp: bool = True) -> list[dict[str, str]]:
    observed, mismatches = canonical(actual), []
    for field in SUPPORTED_STATE_FIELDS:
        if timestamp or field != "block_timestamp":
            want, got = str(expected[field]), observed[field]
            if want != got: mismatches.append({"field": field, "expected": want, "actual": got})
    return mismatches

def compare_action(record: dict[str, Any], actual: dict[str, Any]) -> list[dict[str, str]]:
    expected, result, kind = record.get("expected", {}), actual.get("action_result"), record["action_type"]
    if result is None: return []
    if kind == "exchange" and "dy" in expected:
        want, got, field = uint(expected["dy"], "expected.dy"), uint(result, "harness exchange result"), "dy_raw"
    elif kind == "remove_liquidity_fixed_out" and "amounts" in expected:
        coin = uint(record["inputs"]["i"], "remove.i"); want, got, field = uint(expected["amounts"][1 - coin], "expected.amount_j"), uint(result, "harness remove result"), "amount_j_raw"
    elif kind in ("add_liquidity", "donation") and "return_value" in record.get("call", {}):
        want, got, field = uint(record["call"]["return_value"], "expected return_value"), uint(result, "harness add result"), "minted_lp"
    else: return []
    return [] if want == got else [{"field": field, "expected": str(want), "actual": str(got), "delta": str(got - want)}]

def sdk_repo(path: Path) -> Path:
    repo = path.resolve()
    required = (repo / "src/benchmark_harness.cpp", repo / "include/parity/pool_config.hpp")
    if not all(item.is_file() for item in required):
        raise RuntimeError(f"SDK repo is missing required provenance files: {repo}")
    return repo

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, default=RAW_DATA); parser.add_argument("--binary", type=Path, required=True); parser.add_argument("--sdk-repo", type=Path, default=DEFAULT_SDK_REPO, help="twocrypto-cpp checkout used for provenance"); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    source = json.loads((args.data / "pool_config_historical.json").read_text()); records, expected_states = load_jsonl(args.data / "action_tape.jsonl"), load_jsonl(args.data / "action_block_expected_states.jsonl")
    pool, dropped = adapt_pool(source); sequence, positions, counts, limitations = adapt_actions(records)
    with tempfile.TemporaryDirectory(prefix="yb-native-replay-") as work:
        root = Path(work); paths = [root / name for name in ("pool.json", "sequence.json", "native.json")]
        paths[0].write_text(json.dumps(pool, separators=(",", ":"))); paths[1].write_text(json.dumps(sequence, separators=(",", ":")))
        run = subprocess.run([str(args.binary), *(str(path) for path in paths)], env={**os.environ, "CPP_THREADS": "1", "SNAPSHOT_EVERY": "1"}, capture_output=True, text=True)
        if run.returncode: raise RuntimeError(f"benchmark_harness_i failed: {run.stderr.strip() or run.stdout.strip()}")
        native = json.loads(paths[2].read_text())
    result, states, mismatches = native["results"][0]["result"], native["results"][0]["result"].get("states", []), []
    initial = next(x for x in expected_states if x["sample_kind"] == "pre_start_boundary")
    initial_fields = compare(initial, states[0], timestamp=False) if states else [{"field": "state", "expected": "present", "actual": "missing"}]
    if initial_fields:
        mismatches.append({"kind": "initial_state", "action_index": None, "fields": initial_fields})
    expected_by_block = {x["block_number"]: x for x in expected_states if x["sample_kind"] == "pool_action_block_end"}
    block_last = {record["block_number"]: record["action_index"] for record in records}
    for record in records:
        if mismatches:
            break
        index, position = record["action_index"], positions[record["action_index"]]
        if position >= len(states): mismatches.append({"kind": "missing_action_state", "action_index": index, "state_position": position}); break
        if not states[position].get("action_success", True): mismatches.append({"kind": "action_failure", "action_index": index, "error": states[position].get("error", "unknown action failure")}); break
        if fields := compare_action(record, states[position]): mismatches.append({"kind": "first_action_mismatch", "action_index": index, "fields": fields}); break
        block_number, action_index = record["block_number"], record["action_index"]
        if block_last[block_number] == action_index:
            expected = expected_by_block.get(block_number)
            if expected is None:
                mismatches.append({"kind": "missing_expected_block_state", "action_index": action_index, "block_number": block_number}); break
            fields = compare(expected, states[position])
            if fields:
                mismatches.append({"kind": "first_state_mismatch", "action_index": action_index, "block_number": block_number, "fields": fields}); break
    repo = sdk_repo(args.sdk_repo)
    source_info = {"data": str(args.data), "binary": str(args.binary), "binary_sha256": sha256(args.binary), "twocrypto_commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(), "benchmark_harness_cpp_sha256": sha256(repo / "src/benchmark_harness.cpp"), "pool_config_hpp_sha256": sha256(repo / "include/parity/pool_config.hpp"), "action_tape_sha256": sha256(args.data / "action_tape.jsonl"), "expected_states_sha256": sha256(args.data / "action_block_expected_states.jsonl")}
    output = {"status": "exact_supported_endpoint_equality" if not mismatches and result.get("success", False) else "mismatch", "source": source_info, "counts": {"tape_actions": len(records), "blocks": len(expected_by_block), "action_types": dict(counts), "harness_states": len(states)}, "adaptation": {"cbBTC_scale": CB_BTC_SCALE, "pool_fields_dropped": dropped, "limitations": dropped + limitations}, "initial_checkpoint_supported_equality": not initial_fields, "first_mismatch": mismatches[0] if mismatches else None, "all_supported_fields": list(SUPPORTED_STATE_FIELDS)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "counts": output["counts"], "first_mismatch": output["first_mismatch"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if output["status"] == "exact_supported_endpoint_equality" else 2

if __name__ == "__main__": raise SystemExit(main())
