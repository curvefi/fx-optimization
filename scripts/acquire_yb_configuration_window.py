"""Acquire a bounded, read-only cbBTC Twocrypto configuration window."""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from eth_abi import decode, encode
from eth_utils import keccak

POOL = "0x862CB4E988FB66E72f128d1183829f8c05B6c6A0"
AMM = "0x49F51d7e279252F3C9a09678fdC65B4dBd5CB196"
MULTICALL3 = "0xcA11bde05977b3631167028862bE2A173976CA11"
RPC_URL = "https://lb.drpc.live/ethereum/"
START_TS = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp())
END_TS = int(datetime(2026, 9, 4, tzinfo=timezone.utc).timestamp())
EVENT_SIGS = {
    "NewParameters": "NewParameters(uint256,uint256,uint256,uint256,uint256,uint256)",
    "RampAgamma": "RampAgamma(uint256,uint256,uint256,uint256,uint256,uint256)",
    "StopRampA": "StopRampA(uint256,uint256,uint256)",
    "SetDonationParameters": "SetDonationParameters(uint256,uint256,uint256,uint256)",
    "SetFeeParameters": "SetFeeParameters(uint256,uint256)",
    "SetPolicyContract": "SetPolicyContract(address)",
    "LPAllowlistChanged": "LPAllowlistChanged(address,bool)",
}
AMM_EVENT_SIGS = {"SetFee": "SetFee(uint256)", "SetKilled": "SetKilled(bool)", "SetRate": "SetRate(uint256,uint256,uint256)"}
GETTERS = {
    "coins0": ("coins(uint256)", [0], "address"), "coins1": ("coins(uint256)", [1], "address"),
    "POLICY": ("POLICY()", [], "address"), "A": ("A()", [], "uint256"), "gamma": ("gamma()", [], "uint256"),
    "mid_fee": ("mid_fee()", [], "uint256"), "out_fee": ("out_fee()", [], "uint256"),
    "fee_gamma": ("fee_gamma()", [], "uint256"), "adjustment_step": ("adjustment_step()", [], "uint256[2]"),
    "ma_time": ("ma_time()", [], "uint256"), "price_scale": ("price_scale()", [], "uint256"),
    "last_prices": ("last_prices()", [], "uint256"), "last_timestamp": ("last_timestamp()", [], "uint256"),
    "initial_A_gamma": ("initial_A_gamma()", [], "uint256"), "initial_A_gamma_time": ("initial_A_gamma_time()", [], "uint256"),
    "future_A_gamma": ("future_A_gamma()", [], "uint256"), "future_A_gamma_time": ("future_A_gamma_time()", [], "uint256"),
    "donation_shares": ("donation_shares()", [], "uint256"), "donation_shares_max_ratio": ("donation_shares_max_ratio()", [], "uint256"),
    "donation_duration": ("donation_duration()", [], "uint256"), "donation_protection_expiry_ts": ("donation_protection_expiry_ts()", [], "uint256"),
    "donation_protection_period": ("donation_protection_period()", [], "uint256"),
    "donation_protection_lp_threshold": ("donation_protection_lp_threshold()", [], "uint256"),
    "reserved_profit_fraction": ("reserved_profit_fraction()", [], "uint256"), "admin_fee": ("admin_fee()", [], "uint256"),
    "totalSupply": ("totalSupply()", [], "uint256"), "name": ("name()", [], "string"),
    "symbol": ("symbol()", [], "string"), "decimals": ("decimals()", [], "uint8"),
    "yb_amm_fee": ("fee()", [], "uint256"), "yb_amm_is_killed": ("is_killed()", [], "bool"),
    "yb_amm_rate": ("rate()", [], "uint256"), "yb_amm_rate_mul": ("rate_mul()", [], "uint256"),
    "yb_amm_lt": ("LT_CONTRACT()", [], "address"),
    "yb_amm_collateral": ("COLLATERAL()", [], "address"), "yb_amm_stablecoin": ("STABLECOIN()", [], "address"),
}
POLICY_GETTERS = {
    k: (f"{k}()", [], "uint256") for k in ("FAST_HALF_LIFE", "SLOW_HALF_LIFE", "KAPPA", "DEADBAND", "MIN_CAP", "MAX_CAP")
}
POLICY_GETTERS["POOL"] = ("POOL()", [], "address")

def selector(signature: str) -> bytes:
    return keccak(text=signature)[:4]

def arg_data(signature: str, args: list[int | str]) -> str:
    types = ["uint256" if isinstance(a, int) else "address" for a in args]
    return "0x" + (selector(signature) + encode(types, args)).hex()

class RPC:
    def __init__(self, out: Path):
        self.key = os.environ.get("DRPC_API_KEY")
        if not self.key:
            raise RuntimeError("DRPC_API_KEY is required")
        self.cache = out / "rpc_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, str] = {}

    def request(self, method: str, params: list) -> object:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, separators=(",", ":")).encode()
        reqid = hashlib.sha256(body).hexdigest()
        path = self.cache / f"{reqid}.json"
        if path.exists():
            env = json.loads(path.read_text())
        else:
            req = Request(RPC_URL, body, {"content-type": "application/json", "Drpc-Key": self.key})
            try:
                with urlopen(req, timeout=40) as response:
                    env = json.loads(response.read())
            except Exception as exc:
                raise RuntimeError(f"RPC {method} failed") from exc
            raw = json.dumps(env, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(raw)
        if "error" in env:
            raise RuntimeError(f"RPC {method} returned an error")
        envelope_hash = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.index[reqid] = envelope_hash
        return env["result"]

    def block(self, n: int) -> dict:
        return self.request("eth_getBlockByNumber", [hex(n), False])

    def at(self, to: str, data: bytes, block: int) -> bytes:
        result = self.request("eth_call", [{"to": to, "data": "0x" + data.hex()}, hex(block)])
        return bytes.fromhex(result[2:])

def first_block(rpc: RPC, target: int, high: int) -> int:
    low = 0
    while low < high:
        mid = (low + high) // 2
        if int(rpc.block(mid)["timestamp"], 16) >= target:
            high = mid
        else:
            low = mid + 1
    return low

def decode_one(raw: bytes, typ: str):
    return decode([typ], raw)[0]

def config(rpc: RPC, block: int, multicall: bool) -> tuple[dict, bool]:
    calls = [(name, AMM if name.startswith("yb_amm_") else POOL, selector(sig) + encode(["uint256"], args) if args else selector(sig), typ) for name, (sig, args, typ) in GETTERS.items()]
    values: dict[str, object] = {}
    used = multicall
    if multicall:
        payload = encode(["(address,bool,bytes)[]"], [[(target, True, data) for _, target, data, _ in calls]])
        try:
            result = rpc.at(MULTICALL3, selector("aggregate3((address,bool,bytes)[])") + payload, block)
            returned = decode(["(bool,bytes)[]"], result)[0]
            if len(returned) != len(calls):
                raise ValueError("bad multicall result")
            for (name, _, _, typ), (ok, raw) in zip(calls, returned):
                if not ok:
                    raise ValueError("getter reverted")
                values[name] = decode_one(raw, typ)
        except Exception:
            values, used = {}, False
    if not used:
        for name, target, data, typ in calls:
            values[name] = decode_one(rpc.at(target, data, block), typ)
    values = {k: (v.lower() if isinstance(v, str) and v.startswith("0x") else list(v) if isinstance(v, tuple) else v) for k, v in values.items()}
    policy = values["POLICY"]
    policy_info = {"address": policy, "code_hash": None, "runtime_bytes": None, "immutable_params": None, "internal_params_known": False}
    if int(policy, 16):
        code = rpc.request("eth_getCode", [policy, hex(block)])
        policy_info["code_hash"] = "0x" + keccak(bytes.fromhex(code[2:])).hex()
        pvals: dict[str, object] = {}
        for name, (sig, args, typ) in POLICY_GETTERS.items():
            try:
                pvals[name] = decode_one(rpc.at(policy, selector(sig), block), typ)
            except Exception:
                pvals[name] = None
        pvals = {k: (v.lower() if isinstance(v, str) and v.startswith("0x") else v) for k, v in pvals.items()}
        policy_info["immutable_params"] = pvals
        policy_info["internal_params_known"] = all(v is not None for v in pvals.values()) and pvals.get("POOL") == POOL.lower()
    return {"block": block, "timestamp": int(rpc.block(block)["timestamp"], 16), "values": values, "policy": policy_info}, used

def event_logs(rpc: RPC, start: int, end: int, address: str = POOL, definitions: dict = EVENT_SIGS) -> tuple[list[dict], dict[str, str]]:
    topics = {name: "0x" + keccak(text=sig).hex() for name, sig in definitions.items()}
    chunks = [(lo, min(lo + 200_000 - 1, end)) for lo in range(start, end + 1, 200_000)]
    def fetch(pair):
        lo, hi = pair
        return rpc.request("eth_getLogs", [{"address": address, "fromBlock": hex(lo), "toBlock": hex(hi), "topics": [[*topics.values()]]}])
    logs: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for part in pool.map(fetch, chunks):
            logs.extend(part)
    logs.sort(key=lambda x: (int(x["blockNumber"], 16), int(x["transactionIndex"], 16), int(x["logIndex"], 16)))
    parsed = []
    for item in logs:
        topic = item["topics"][0].lower()
        name = next(k for k, v in topics.items() if v.lower() == topic)
        types = {"NewParameters": ["uint256"] * 6, "RampAgamma": ["uint256"] * 6, "StopRampA": ["uint256"] * 3,
                 "SetDonationParameters": ["uint256"] * 4, "SetFeeParameters": ["uint256"] * 2,
                 "SetPolicyContract": ["address"], "LPAllowlistChanged": ["bool"], "SetFee": ["uint256"], "SetKilled": ["bool"], "SetRate": ["uint256"] * 3}[name]
        args = list(decode(types, bytes.fromhex(item["data"][2:]))) if name != "LPAllowlistChanged" else [bool(int(item["data"], 16))]
        if name == "LPAllowlistChanged": args.insert(0, "0x" + item["topics"][1][-40:])
        parsed.append({"contract": address.lower(), "event": name, "block": int(item["blockNumber"], 16), "transaction": item["transactionHash"], "log_index": int(item["logIndex"], 16), "args": args})
    return parsed, topics

def key(snapshot: dict) -> tuple:
    v = snapshot["values"]
    return tuple(v.get(k) for k in ("POLICY", "mid_fee", "out_fee", "fee_gamma", "adjustment_step", "ma_time", "initial_A_gamma", "initial_A_gamma_time", "future_A_gamma", "future_A_gamma_time", "donation_duration", "donation_protection_period", "donation_protection_lp_threshold", "donation_shares_max_ratio", "reserved_profit_fraction", "admin_fee", "yb_amm_fee", "yb_amm_is_killed"))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/yb-cbbtc-historical-20260904/configuration"))
    ap.add_argument("--offline-check", metavar="PATH")
    args = ap.parse_args()
    if args.offline_check:
        doc = json.loads(Path(args.offline_check).read_text())
        chosen = doc["chosen_candidate"]
        checks = {
            "positive_duration": chosen.get("duration_seconds", 0) > 0,
            "not_ramping": not chosen.get("ramping", True),
            "event_free": chosen.get("event_free", False),
            "getter_crosscheck": chosen.get("getter_crosscheck", False),
            "ordered_intervals": all(x["start_block"] < x["end_block"] for x in doc["intervals"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"offline invariant failed: {checks}")
        print("offline invariant: pass")
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rpc = RPC(out)
    if rpc.request("eth_chainId", []) != "0x1":
        raise RuntimeError("RPC is not Ethereum mainnet")
    latest = int(rpc.request("eth_blockNumber", []), 16)
    start = first_block(rpc, START_TS, latest)
    end = first_block(rpc, END_TS, latest) - 1
    prestart = max(0, start - 1)
    pool_events, pool_topics = event_logs(rpc, start, end)
    amm_events, amm_topics = event_logs(rpc, start, end, AMM, AMM_EVENT_SIGS)
    events = sorted(pool_events + amm_events, key=lambda x: (x["block"], x["log_index"]))
    event_blocks = {e["block"] for e in events}
    starts = {start}
    for block in event_blocks:
        if start <= block <= end:
            starts.add(block + 1)
    snapshots: dict[int, dict] = {}
    multicall = True
    def read(block: int) -> dict:
        nonlocal multicall
        if block not in snapshots:
            snapshots[block], multicall = config(rpc, block, multicall)
        return snapshots[block]
    read(prestart)
    def ranges() -> list[tuple[int, int]]:
        gaps = []
        cursor = start
        for block in sorted(b for b in event_blocks if start <= b <= end):
            if cursor <= block - 1:
                gaps.append((cursor, block - 1))
            cursor = max(cursor, block + 1)
        if cursor <= end:
            gaps.append((cursor, end))
        result = []
        for left, stop in gaps:
            cuts = sorted(b for b in starts if left < b <= stop)
            for split in [left] + cuts:
                next_split = next((b for b in cuts if b > split), stop + 1)
                result.append((split, next_split - 1))
        return result
    for left, stop in ranges():
        s = read(left)
        future = s["values"]["future_A_gamma_time"]
        if future > s["timestamp"] and future <= read(stop)["timestamp"]:
            ramp_block = first_block(rpc, future, end)
            if left < ramp_block <= stop:
                starts.add(ramp_block)
    intervals = []
    for left, stop in ranges():
        before = read(max(prestart, left - 1))
        s, e = read(left), read(stop)
        future_ts = s["values"]["future_A_gamma_time"]
        intervals.append({"start_block": left, "end_block": stop, "prestate_block": before["block"], "start_timestamp": s["timestamp"], "end_timestamp": e["timestamp"], "duration_seconds": max(0, e["timestamp"] - s["timestamp"]), "config_key": list(key(s)), "ramping": future_ts > s["timestamp"], "event_free": not any(left <= b <= stop for b in event_blocks), "getter_crosscheck": key(s) == key(e), "prestate_policy": before["values"]["POLICY"], "policy_address": s["values"]["POLICY"], "policy_internal_params_known": s["policy"]["internal_params_known"], "yb_amm_fee": s["values"]["yb_amm_fee"], "yb_amm_is_killed": s["values"]["yb_amm_is_killed"], "yb_amm_rate_start": s["values"]["yb_amm_rate"], "yb_amm_rate_end": e["values"]["yb_amm_rate"]})
    candidates = [x for x in intervals if not x["ramping"] and x["event_free"] and x["getter_crosscheck"] and (x["policy_address"] == "0x0000000000000000000000000000000000000000" or x["policy_internal_params_known"])]
    chosen = max(candidates, key=lambda x: x["duration_seconds"]) if candidates else {"status": "none", "reason": "no event-free non-ramping interval"}
    first, last = rpc.block(prestart), rpc.block(end)
    pool_code_start = rpc.request("eth_getCode", [POOL, hex(start)]); pool_code_end = rpc.request("eth_getCode", [POOL, hex(end)])
    workspace = Path(__file__).resolve().parents[2]
    source_candidates = [
        workspace.parent / "twocrypto-ng/contracts/main/Twocrypto.vy",
        workspace / "twocrypto-cpp/reference/twocrypto-ng/contracts/main/Twocrypto.vy",
    ]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_candidates if path.is_file()}
    missing_sources = [str(path) for path in source_candidates if not path.is_file()]
    if not source_hashes:
        raise RuntimeError("no local Twocrypto source available for provenance")
    current = snapshots[end]
    token = {k: current["values"][k] for k in ("coins0", "coins1", "name", "symbol", "decimals")}
    for i, coin in enumerate((token["coins0"], token["coins1"])):
        for field, typ in (("name", "string"), ("symbol", "string"), ("decimals", "uint8")):
            try: token[f"coin{i}_{field}"] = decode_one(rpc.at(coin, selector(f"{field}()"), end), typ)
            except Exception: token[f"coin{i}_{field}"] = None
    pool_code_pre = rpc.request("eth_getCode", [POOL, hex(prestart)]); amm_code_end = rpc.request("eth_getCode", [AMM, hex(end)])
    doc = {"chain_id": 1, "pool": POOL.lower(), "yb_amm": AMM.lower(), "window": {"start_timestamp": START_TS, "end_exclusive_timestamp": END_TS, "start_block": start, "end_block": end, "latest_block_at_acquisition": latest}, "boundary_blocks": [{"block": prestart, "hash": first["hash"], "timestamp": int(first["timestamp"], 16)}, {"block": end, "hash": last["hash"], "timestamp": int(last["timestamp"], 16)}], "pool_runtime": {"prestart_code_hash": "0x" + keccak(bytes.fromhex(pool_code_pre[2:])).hex(), "start_code_hash": "0x" + keccak(bytes.fromhex(pool_code_start[2:])).hex(), "end_code_hash": "0x" + keccak(bytes.fromhex(pool_code_end[2:])).hex(), "same": pool_code_pre == pool_code_end}, "yb_amm_runtime_end_code_hash": "0x" + keccak(bytes.fromhex(amm_code_end[2:])).hex(), "source_provenance": {"sha256": source_hashes, "missing_files": missing_sources, "same_source_bytes": not missing_sources and len(set(source_hashes.values())) == 1, "runtime_compiler_metadata": "not available from eth_getCode; code hashes are pinned"}, "token_identity": token, "multicall3": {"address": MULTICALL3.lower(), "used": multicall}, "topic_registry": {**pool_topics, **amm_topics}, "snapshots": [snapshots[b] for b in sorted(snapshots)], "change_events": events, "intervals": intervals, "chosen_candidate": chosen, "current_config": current, "rpc_cache": {"entries": len(rpc.index), "envelope_sha256": rpc.index}}
    (out / "configuration_window.json").write_text(json.dumps(doc, indent=2, default=lambda x: x.hex() if isinstance(x, bytes) else x) + "\n")
    (out / "change_events.json").write_text(json.dumps(events, indent=2) + "\n")
    (out / "intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")
    (out / "chosen_candidate.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print(json.dumps({"start_block": start, "end_block": end, "events": len(events), "intervals": len(intervals), "chosen_duration_seconds": chosen.get("duration_seconds", 0), "rpc_cache_entries": len(rpc.index)}))

if __name__ == "__main__":
    main()
