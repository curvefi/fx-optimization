#!/usr/bin/env python3
"""Replay two isolated July VirtualPool transactions in the f64 reference model."""
from __future__ import annotations

import argparse, hashlib, json, math, os, subprocess
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
HARNESS = WORKSPACE / "curve-fx-arb-harness"
SDK = WORKSPACE / "twocrypto-cpp"
ACTIVITY = ROOT / "runs/yb-cbbtc-historical-20260904/activity"
DEFAULT_OUT = ROOT / "runs/yb-cbbtc-historical-20260904/vp-samples"
POOL, AMM = "0x862cb4e988fb66e72f128d1183829f8c05b6c6a0", "0x49f51d7e279252f3c9a09678fdc65b4dbd5cb196"
LT, VP = "0x722fc3640ba007c3e9867ccdb0dca59f2e2f29f9", "0x04ca7a7e602335a261b63128e89d43b6fe1e2c87"
STABLE, AGG = "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e", "0x18672b1b0c623a30089a280ed9256379fb0e4e62"
SAMPLES = {25460091: "0xdf0007ada0322f8727c32a2a2b8e2231485fd315d45c3388ab5608d8e3a0c3b7",
           25520583: "0x498b72f4feba9c45f2d0b8e205714d369aab71f23274eaec00d4ad6f275caa88"}
POOL_SLOTS = {"price_scale": 1, "price_oracle": 2, "last_prices": 3, "last_timestamp": 4,
    "donation_shares": 9, "last_donation_release_ts": 12, "donation_protection_expiry_ts": 13,
    "donation_protection_extension_remainder": 16, "balance0": 17, "balance1": 18,
    "admin0": 19, "admin1": 20, "D": 21, "xcp_profit": 22, "lp_xcp_profit": 23,
    "virtual_price": 24, "total_supply": 32}
AMM_SLOTS = {"fee": 0, "collateral": 1, "debt": 2, "rate": 3, "rate_mul": 4,
    "rate_time": 5, "minted": 6, "redeemed": 7, "killed": 8}
SCALE = {**{k: 10**18 for k in POOL_SLOTS}, "balance1": 10**8, "admin1": 10**8,
         **{k: 10**18 for k in AMM_SLOTS}}
for name in ("last_timestamp", "last_donation_release_ts", "donation_protection_expiry_ts", "rate_time", "killed"): SCALE[name] = 1

SELECTORS = {"exchange(uint256,uint256,uint256,uint256)":"0x5b41b908",
    "exchange(uint256,uint256,uint256,uint256,address)":"0xa64833a0", "price()":"0xa035b1fe",
    "maxFlashLoan(address)":"0x613255ab", "balanceOf(address)":"0x70a08231", "price_w()":"0xceb7f759",
    "add_liquidity(uint256[2],uint256)":"0x0b4c7e4d",
    "add_liquidity(uint256[2],uint256,address,bool)":"0x86514738",
    "remove_liquidity(uint256,uint256[2])":"0x5b36389c"}
def sig(text: str) -> str: return SELECTORS[text]
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def word(raw: str) -> int: return int(raw or "0x0", 16)
def words(data: str) -> list[int]: return [int(data[i:i + 64], 16) for i in range(10, len(data), 64)]

class RPC:
    def __init__(self, out: Path):
        self.key = os.environ.get("DRPC_API_KEY")
        if not self.key: raise RuntimeError("DRPC_API_KEY is required")
        self.cache = out / "rpc_cache"; self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = 0; self.new_calls = 0; self.hashes: dict[str, str] = {}
    def request(self, method: str, params: list[Any]) -> Any:
        self.calls += 1
        if self.calls > 100: raise RuntimeError("100-RPC-call study cap exceeded")
        body = json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params}, separators=(",", ":")).encode()
        key = hashlib.sha256(body).hexdigest(); path = self.cache / f"{key}.json"
        if path.exists(): env = json.loads(path.read_text())
        else:
            req = Request("https://lb.drpc.live/ethereum/", body,
                {"content-type":"application/json", "Drpc-Key":self.key})
            try:
                with urlopen(req, timeout=90) as response: env = json.loads(response.read())
            except Exception as exc: raise RuntimeError(f"RPC {method} failed") from exc
            path.write_text(json.dumps(env, sort_keys=True, separators=(",", ":"))); self.new_calls += 1
        if "error" in env: raise RuntimeError(f"RPC {method} returned error {env['error'].get('code')}")
        self.hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return env["result"]

def walk(frame: dict[str, Any]):
    yield frame
    for child in frame.get("calls", []): yield from walk(child)

def storage(account: dict[str, Any], slot: int) -> int | None:
    for key, value in account.get("storage", {}).items():
        if int(key, 16) == slot: return word(value)
    return None

def traced_state(pre: dict[str, Any], diff: dict[str, Any], address: str, slots: dict[str, int],
                 rpc: RPC, block: int) -> tuple[dict[str, int], list[str]]:
    accounts = [pre.get(address, {}), diff.get("pre", {}).get(address, {})]; out={}; fallback=[]
    for name,slot in slots.items():
        value=next((v for account in accounts if (v:=storage(account,slot)) is not None),None)
        if value is None:
            value=word(rpc.request("eth_getStorageAt",[address,hex(slot),hex(block-1)])); fallback.append(name)
        out[name]=value
    return out,fallback

def post_state(before: dict[str, int], diff: dict[str, Any], address: str, slots: dict[str, int]) -> dict[str, int]:
    changed={int(k,16):word(v) for k,v in diff.get("post",{}).get(address,{}).get("storage",{}).items()}
    return {name:changed.get(slot,before[name]) for name,slot in slots.items()}

def validate_sample(rows: list[dict[str, Any]], block: int, tx_hash: str) -> dict[str, Any]:
    same_block = [r for r in rows if r["block"] == block]
    if len(same_block) != 1 or same_block[0]["transaction"].lower() != tx_hash:
        raise RuntimeError(f"block {block} is not isolated to the selected ecosystem transaction")
    row = same_block[0]; events = row["events"]
    vp = [e for e in events if e["address"] == VP and e["event"] == "TokenExchange"]
    forbidden = [e for e in events if (e["address"] == POOL and e["event"] == "TokenExchange") or
                 (e["address"] == LT and e["event"] in {"Deposit", "Withdraw"})]
    if row["outer_family"] != "virtual_pool" or len(vp) != 1 or forbidden:
        raise RuntimeError(f"block {block} fails VP route isolation criteria")
    return row

def event(row: dict[str, Any], address: str, name: str) -> dict[str, Any]:
    found = [x for x in row["events"] if x["address"] == address and x["event"] == name]
    if len(found) != 1: raise RuntimeError(f"expected one {address}:{name}, found {len(found)}")
    return found[0]["args"]

def trace_inputs(trace: dict[str, Any], row: dict[str, Any], rpc: RPC, block: int) -> dict[str, Any]:
    frames = list(walk(trace)); exchange_sigs = {sig("exchange(uint256,uint256,uint256,uint256)"), sig("exchange(uint256,uint256,uint256,uint256,address)")}
    vp_calls = [f for f in frames if f.get("to", "").lower() == VP and f.get("input", "")[:10] in exchange_sigs]
    if len(vp_calls) != 1: raise RuntimeError("could not isolate VirtualPool exchange calldata")
    calldata = words(vp_calls[0]["input"]); direction, output_coin, amount, min_out = calldata[:4]
    vp_event = event(row, VP, "TokenExchange")
    if (direction, output_coin, amount) != (vp_event["sold_id"], vp_event["bought_id"], vp_event["tokens_sold"]):
        raise RuntimeError("VirtualPool calldata disagrees with observed event")
    values = lambda target, selector: [word(f.get("output", "0x")) for f in frames
        if f.get("to", "").lower() == target and f.get("input", "")[:10] == selector]
    call_prices, flash_caps = values(AGG, sig("price()")), values("0x26de7861e213a5351f6ed767d00e0839930e9ee1", sig("maxFlashLoan(address)"))
    nested=rpc.request("trace_filter",[{"fromBlock":hex(block),"toBlock":hex(block),"toAddress":[AGG]}])
    agg_calls=[{"selector":x.get("action",{}).get("input","")[:10],"return":word(x.get("result",{}).get("output","0x"))}
        for x in nested if x.get("transactionHash","").lower()==row["transaction"] and
        x.get("action",{}).get("input","")[:10] in {sig("price()"),sig("price_w()")}]
    agg_prices=[x["return"] for x in agg_calls if x["selector"]==sig("price_w()")]
    if len(set(agg_prices+call_prices)) != 1 or not agg_prices or not flash_caps or len(set(flash_caps)) != 1:
        raise RuntimeError("trace does not provide a unique per-transaction aggregator price and flash capacity")
    balance_frames = [(words(f["input"])[0] & ((1 << 160) - 1), word(f.get("output", "0x"))) for f in frames
        if f.get("to", "").lower() == STABLE and f.get("input", "")[:10] == sig("balanceOf(address)")]
    balance = lambda addr: [v for a, v in balance_frames if a == int(addr, 16)]
    amm_event, collect = event(row, AMM, "TokenExchange"), event(row, AMM, "CollectFees")
    if not balance(AMM) or not balance(LT): raise RuntimeError("trace lacks AMM/LT stable balance observations")
    amm_seen, lt_seen = balance(AMM)[-1], balance(LT)[-1]
    cash_pre = amm_seen - amm_event["tokens_sold"] if direction == 0 else amm_seen + amm_event["tokens_bought"]
    lt_pre = lt_seen - collect["amount"]
    selector_labels = {
        (POOL, sig("add_liquidity(uint256[2],uint256)")): "pool_add",
        (POOL, sig("add_liquidity(uint256[2],uint256,address,bool)")): "pool_donation",
        (POOL, sig("remove_liquidity(uint256,uint256[2])")): "pool_remove",
        (AMM, sig("exchange(uint256,uint256,uint256,uint256)")): "amm_exchange",
        (AMM, sig("exchange(uint256,uint256,uint256,uint256,address)")): "amm_exchange",
    }
    selected = [(f, selector_labels[(f.get("to", "").lower(), f.get("input", "")[:10])]) for f in frames
                if (f.get("to", "").lower(), f.get("input", "")[:10]) in selector_labels]
    order = [label for _, label in selected]
    expected = ["amm_exchange", "pool_donation", "pool_remove"] if direction == 0 else ["pool_add", "amm_exchange", "pool_donation"]
    if order != expected: raise RuntimeError(f"unexpected modeled subleg order: {order}")
    effective = amount * (10**18 - 10**10) // 10**18
    route_add=[x["args"] for x in row["events"] if x["address"]==POOL and x["event"]=="AddLiquidity" and x["args"].get("receiver")==VP]
    flash_used = amm_event["tokens_sold"] - effective if direction == 0 else route_add[0]["token_amounts"][0]
    return {"direction":direction, "amount":amount, "min_out":min_out, "output":vp_event["tokens_bought"],
        "aggregator_price":agg_prices[0], "aggregator_calls":agg_calls,
        "aggregator_source":"trace_filter exact AGG.price_w return; callTracer view return cross-checked when present",
        "flash_capacity":flash_caps[0], "flash_used":flash_used,
        "amm_cash_pre":cash_pre, "amm_cash_post":amm_seen - collect["amount"], "lt_cash_pre":lt_pre,
        "lt_cash_post":lt_seen - event(row, POOL, "Donation")["token_amounts"][0],
        "lp_amount":amm_event["tokens_bought"] if direction == 0 else word(next(f["output"] for f,label in selected if label=="pool_add")),
        "donation":collect["amount"], "donation_min_mint":event(row,LT,"DistributeBorrowerFees")["min_amount"],
        "actual_order":order, "expected_order":expected,
        "oracle_price_w":amm_event["price_oracle"], "amm_input":amm_event["tokens_sold"], "amm_output":amm_event["tokens_bought"]}

def model_input(block: int, header: dict[str, Any], ps: dict[str, int], ys: dict[str, int], route: dict[str, Any]) -> dict[str, Any]:
    human = lambda state, name: state[name] / SCALE[name]
    return {"pool":{"A":50000.0, "gamma":11111111111/1e18, "mid_fee":146000000/1e10,
        "out_fee":170000000/1e10, "fee_gamma":54202748000000000/1e18,
        "adjustment_step_min":1e-10, "adjustment_step_max":0.005, "ma_time":865.0,
        "reserved_profit_fraction":3010101009/1e18, "admin_fee":0.0, "donation_duration":604800.0,
        "donation_shares_max_ratio":0.1, "donation_protection_period":600.0,
        "donation_protection_lp_threshold":0.2, "state":{k:human(ps,k) for k in ps}},
        "yb":{"source_block":block, "source_timestamp":int(header["timestamp"],16), "block_hash":header["hash"],
            "leverage":2.0, "fee":human(ys,"fee"), "collateral":human(ys,"collateral"), "debt":human(ys,"debt"),
            "rate":human(ys,"rate"), "rate_mul":human(ys,"rate_mul"), "rate_time":ys["rate_time"],
            "minted":human(ys,"minted"), "redeemed":human(ys,"redeemed"),
            "stable_balance":route["amm_cash_pre"]/1e18, "lt_stable_balance":route["lt_cash_pre"]/1e18,
            "flash_max_loan":route["flash_capacity"]/1e18, "stable_aggregator":route["aggregator_price"]/1e18,
            "rounding_discount":1e-8, "lt_donation_discount":0.01, "killed":bool(ys["killed"])},
        "route":{"direction":route["direction"], "input":route["amount"]/(1e18 if route["direction"]==0 else 1e8),
            "min_output":route["min_out"]/(1e8 if route["direction"]==0 else 1e18), "timestamp":int(header["timestamp"],16)}}

def compare(model: dict[str, Any], actual: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    rows=[]
    for name,(raw,scale) in actual.items():
        got=float(model[name]); rounded=round(got*scale); tol=max(1, math.ceil(64*math.ulp(max(abs(got), abs(raw/scale)))*scale))
        rows.append({"field":name,"actual_raw":str(raw),"model":got,"rounded_model_raw":str(rounded),
                     "raw_error":str(rounded-raw),"tolerance_raw":tol,"within_tolerance":abs(rounded-raw)<=tol})
    return rows

def compile_driver(source: Path, binary: Path) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["c++","-std=c++20","-O2",f"-I{HARNESS/'cpp/include'}",f"-I{SDK/'include'}","-I/opt/homebrew/include",
        str(source),"-L/opt/homebrew/lib","-lboost_json","-o",str(binary)],check=True)

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); args=ap.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True); rpc=RPC(out)
    if rpc.request("eth_chainId",[])!="0x1": raise RuntimeError("RPC is not Ethereum mainnet")
    activity=json.loads((ACTIVITY/"transactions.json").read_text()); driver=out/"vp_replay_driver.cpp"
    binary=HARNESS/"build/yb-historical-vp/vp_replay_driver"; compile_driver(driver,binary)
    reports=[]
    for block,tx_hash in SAMPLES.items():
        row=validate_sample(activity,block,tx_hash); tx=rpc.request("eth_getTransactionByHash",[tx_hash])
        receipt=rpc.request("eth_getTransactionReceipt",[tx_hash]); header=rpc.request("eth_getBlockByNumber",[hex(block),False])
        if word(receipt["status"])!=1 or word(tx["transactionIndex"])!=row["transaction_index"]: raise RuntimeError("transaction receipt mismatch")
        call=rpc.request("debug_traceTransaction",[tx_hash,{"tracer":"callTracer","tracerConfig":{"withLog":True}}])
        pre=rpc.request("debug_traceTransaction",[tx_hash,{"tracer":"prestateTracer"}])
        diff=rpc.request("debug_traceTransaction",[tx_hash,{"tracer":"prestateTracer","tracerConfig":{"diffMode":True}}])
        route=trace_inputs(call,row,rpc,block); sample=out/f"block-{block}"; sample.mkdir(exist_ok=True)
        pool_pre,pool_fallback=traced_state(pre,diff,POOL,POOL_SLOTS,rpc,block)
        yb_pre,yb_fallback=traced_state(pre,diff,AMM,AMM_SLOTS,rpc,block)
        inp=model_input(block,header,pool_pre,yb_pre,route); (sample/"model_input.json").write_text(json.dumps(inp,indent=2)+"\n")
        run=subprocess.run([str(binary),str(sample/"model_input.json")],capture_output=True,text=True)
        if run.returncode not in (0,2): raise RuntimeError(f"driver failed: {run.stderr.strip()}")
        model=json.loads(run.stdout); (sample/"model_output.json").write_text(json.dumps(model,indent=2)+"\n")
        ps=post_state(pool_pre,diff,POOL,POOL_SLOTS); ys=post_state(yb_pre,diff,AMM,AMM_SLOTS)
        actual={"output":(route["output"],1e8 if route["direction"]==0 else 1e18),
            "flash_amount":(route["flash_used"],1e18), "lp_amount":(route["lp_amount"],1e18),
            "donation":(route["donation"],1e18), "donation_min_mint":(route["donation_min_mint"],1e18)}
        route_cmp=compare(model,actual)
        pool_cmp=compare(model["pool"],{k:(v,SCALE[k]) for k,v in ps.items() if k in model["pool"]})
        yb_actual={k:(v,SCALE[k]) for k,v in ys.items() if k in model["yb"]}
        yb_actual.update({"stable_balance":(route["amm_cash_post"],1e18),"lt_stable_balance":(route["lt_cash_post"],1e18)})
        yb_cmp=compare(model["yb"],yb_actual); all_cmp=route_cmp+pool_cmp+yb_cmp
        report={"block":block,"transaction":tx_hash,"transaction_index":row["transaction_index"],
            "direction":"stable_to_cbBTC" if route["direction"]==0 else "cbBTC_to_stable",
            "isolation":{"ecosystem_transactions_in_block":1,"vp_exchanges":1,"forbidden_outer_events":0,
                "tx_before_state":"debug_traceTransaction prestateTracer","tx_post_state":"prestateTracer diffMode",
                "block_minus_one_fallback":{"pool_unaccessed_slots":pool_fallback,"amm_unaccessed_slots":yb_fallback,
                    "basis":"selected block contains no other decoded ecosystem transaction; these slots were neither accessed nor changed by the selected transaction"}},
            "trace_inputs":route,"model_committed":model["committed"],"comparisons":{"route":route_cmp,"pool_post":pool_cmp,"yb_post":yb_cmp},
            "first_mismatch":next((x for x in all_cmp if not x["within_tolerance"]),None)}
        (sample/"report.json").write_text(json.dumps(report,indent=2)+"\n"); reports.append(report)
    summary={"status":"matched" if all(r["first_mismatch"] is None for r in reports) else "mismatch",
        "represented_boundary":"transaction prestate through the selected VP call and its nested native/YB legs; unrelated block transactions are excluded by tracer prestate",
        "constant_parameters":{"A":50000,"gamma_raw":11111111111,"mid_fee_raw":146000000,"out_fee_raw":170000000,
            "fee_gamma_raw":54202748000000000,"adjustment_step_raw":[100000000,5000000000000000],"ma_time_internal_seconds":865,
            "ma_time_getter_half_life_seconds":600,
            "reserved_profit_fraction_raw":3010101009,"admin_fee_raw":0,"yb_fee_raw":13000000000000000},
        "tolerance":"raw comparison after binary64 result rounding; max(1 raw unit, 64 ulps at the field magnitude)",
        "rpc":{"total_requests":rpc.calls,"new_network_requests":rpc.new_calls,"cache_entries":len(rpc.hashes),"envelope_sha256":rpc.hashes},
        "provenance":{"activity_sha256":sha(ACTIVITY/"transactions.json"),"configuration_sha256":sha(ROOT/"runs/yb-cbbtc-historical-20260904/configuration/configuration_window.json"),
            "replay_script_sha256":sha(Path(__file__)),"driver_source_sha256":sha(driver),"driver_binary_sha256":sha(binary),
            "yb_reference_header_sha256":sha(HARNESS/"cpp/include/harness/yb_reference_2l.hpp")},
        "samples":reports,"limitations":["This isolates two observed routes; it does not establish full-window E0b parity.",
            "Binary64 errors are reported in token raw units after rounding and do not imply exact uint parity.",
            "No profit, caller strategy, or counterleg economics are inferred."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"status":summary["status"],"rpc_requests":rpc.calls,"new_network_requests":rpc.new_calls,
        "samples":[{"block":r["block"],"direction":r["direction"],"first_mismatch":r["first_mismatch"]} for r in reports]},indent=2))

if __name__=="__main__": main()
