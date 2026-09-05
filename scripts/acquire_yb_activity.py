"""Acquire read-only observed activity and state for the cbBTC YB window."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from eth_abi import decode, encode
from eth_utils import keccak

sys.path.insert(0, str(Path(__file__).parent))
from acquire_yb_configuration_window import MULTICALL3, RPC, first_block, selector  # noqa: E402

CHAIN_ID = 1
START_BLOCK = 25_455_434
END_BLOCK = 25_860_758
PRESTART_BLOCK = 25_455_433
POOL = "0x862cb4e988fb66e72f128d1183829f8c05b6c6a0"
AMM = "0x49f51d7e279252f3c9a09678fdc65b4dbd5cb196"
LT = "0x722fc3640ba007c3e9867ccdb0dca59f2e2f29f9"
VP = "0x04ca7a7e602335a261b63128e89d43b6fe1e2c87"
ZERO = "0x0000000000000000000000000000000000000000"
CHUNK = 20_000
RECEIPT_LIMIT = 10_000

WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE_FILES = {
    "twocrypto": str(WORKSPACE / "twocrypto-ng/contracts/main/Twocrypto.vy"),
    "amm": str(WORKSPACE / "yb-core/contracts/AMM.vy"),
    "lt": str(WORKSPACE / "yb-core/contracts/LT.vy"),
    "virtual_pool": str(WORKSPACE / "yb-core/contracts/VirtualPool.vy"),
}

# (signature, indexed names/types, data names/types), copied from source.
EVENTS: dict[str, dict[str, tuple[str, list[str], list[str], list[str], list[str]]]] = {
    POOL: {
        "Transfer": ("Transfer(address,address,uint256)", ["sender", "receiver"], ["address", "address"], ["value"], ["uint256"]),
        "Approval": ("Approval(address,address,uint256)", ["owner", "spender"], ["address", "address"], ["value"], ["uint256"]),
        "TokenExchange": ("TokenExchange(address,uint256,uint256,uint256,uint256,uint256,uint256)", ["buyer"], ["address"], ["sold_id", "tokens_sold", "bought_id", "tokens_bought", "fee", "price_scale"], ["uint256"] * 6),
        "AddLiquidity": ("AddLiquidity(address,address,uint256[2],uint256,uint256,uint256)", ["provider", "receiver"], ["address", "address"], ["token_amounts", "fee", "token_supply", "price_scale"], ["uint256[2]", "uint256", "uint256", "uint256"]),
        "Donation": ("Donation(address,uint256[2])", ["donor"], ["address"], ["token_amounts"], ["uint256[2]"]),
        "RemoveLiquidity": ("RemoveLiquidity(address,uint256[2],uint256)", ["provider"], ["address"], ["token_amounts", "token_supply"], ["uint256[2]", "uint256"]),
        "RemoveLiquidityOne": ("RemoveLiquidityOne(address,uint256,uint256,uint256,uint256,uint256)", ["provider"], ["address"], ["token_amount", "coin_index", "coin_amount", "approx_fee", "packed_price_scale"], ["uint256"] * 5),
        "RemoveLiquidityImbalance": ("RemoveLiquidityImbalance(address,uint256,uint256[2],uint256,uint256)", ["provider"], ["address"], ["lp_token_amount", "token_amounts", "approx_fee", "price_scale"], ["uint256", "uint256[2]", "uint256", "uint256"]),
        "NewParameters": ("NewParameters(uint256,uint256,uint256,uint256,uint256,uint256)", [], [], ["mid_fee", "out_fee", "fee_gamma", "adjustment_step_min", "adjustment_step_max", "ma_time"], ["uint256"] * 6),
        "RampAgamma": ("RampAgamma(uint256,uint256,uint256,uint256,uint256,uint256)", [], [], ["initial_A", "future_A", "initial_gamma", "future_gamma", "initial_time", "future_time"], ["uint256"] * 6),
        "StopRampA": ("StopRampA(uint256,uint256,uint256)", [], [], ["current_A", "current_gamma", "time"], ["uint256"] * 3),
        "ClaimAdminFee": ("ClaimAdminFee(address,uint256[2])", ["admin"], ["address"], ["tokens"], ["uint256[2]"]),
        "SetDonationParameters": ("SetDonationParameters(uint256,uint256,uint256,uint256)", [], [], ["duration", "donation_protection_period", "donation_protection_lp_threshold", "donation_shares_max_ratio"], ["uint256"] * 4),
        "SetFeeParameters": ("SetFeeParameters(uint256,uint256)", [], [], ["reserved_profit_fraction", "admin_fee"], ["uint256"] * 2),
        "SetPolicyContract": ("SetPolicyContract(address)", [], [], ["policy"], ["address"]),
        "LPAllowlistChanged": ("LPAllowlistChanged(address,bool)", ["user"], ["address"], ["allowed"], ["bool"]),
    },
    AMM: {
        "TokenExchange": ("TokenExchange(address,uint256,uint256,uint256,uint256,uint256,uint256)", ["buyer"], ["address"], ["sold_id", "tokens_sold", "bought_id", "tokens_bought", "fee", "price_oracle"], ["uint256"] * 6),
        "AddLiquidityRaw": ("AddLiquidityRaw(uint256[2],uint256,uint256)", [], [], ["token_amounts", "invariant", "price_oracle"], ["uint256[2]", "uint256", "uint256"]),
        "RemoveLiquidityRaw": ("RemoveLiquidityRaw(uint256,uint256)", [], [], ["collateral_change", "debt_change"], ["uint256"] * 2),
        "SetRate": ("SetRate(uint256,uint256,uint256)", [], [], ["rate", "rate_mul", "time"], ["uint256"] * 3),
        "CollectFees": ("CollectFees(uint256,uint256)", [], [], ["amount", "new_supply"], ["uint256"] * 2),
        "SetFee": ("SetFee(uint256)", [], [], ["fee"], ["uint256"]),
        "SetKilled": ("SetKilled(bool)", [], [], ["is_killed"], ["bool"]),
    },
    LT: {
        "Transfer": ("Transfer(address,address,uint256)", ["sender", "receiver"], ["address", "address"], ["value"], ["uint256"]),
        "Approval": ("Approval(address,address,uint256)", ["owner", "spender"], ["address", "address"], ["value"], ["uint256"]),
        "SetStaker": ("SetStaker(address)", ["staker"], ["address"], [], []),
        "WithdrawAdminFees": ("WithdrawAdminFees(address,uint256)", [], [], ["receiver", "amount"], ["address", "uint256"]),
        "AllocateStablecoins": ("AllocateStablecoins(address,uint256,uint256)", ["allocator"], ["address"], ["stablecoin_allocation", "stablecoin_allocated"], ["uint256"] * 2),
        "DistributeBorrowerFees": ("DistributeBorrowerFees(address,uint256,uint256,uint256)", ["sender"], ["address"], ["amount", "min_amount", "discount"], ["uint256"] * 3),
        "Deposit": ("Deposit(address,address,uint256,uint256)", ["sender", "owner"], ["address", "address"], ["assets", "shares"], ["uint256"] * 2),
        "Withdraw": ("Withdraw(address,address,address,uint256,uint256)", ["sender", "receiver", "owner"], ["address"] * 3, ["assets", "shares"], ["uint256"] * 2),
        "SetAdmin": ("SetAdmin(address)", [], [], ["admin"], ["address"]),
    },
    VP: {
        "TokenExchange": ("TokenExchange(address,uint256,uint256,uint256,uint256)", ["buyer"], ["address"], ["sold_id", "tokens_sold", "bought_id", "tokens_bought"], ["uint256"] * 4),
    },
}

POOL_SLOTS = {
    "cached_price_scale": 1, "cached_price_oracle": 2, "last_prices": 3,
    "last_timestamp": 4, "donation_shares": 9, "donation_duration": 11,
    "last_donation_release_ts": 12, "donation_protection_expiry_ts": 13,
    "donation_protection_period": 14, "donation_protection_lp_threshold": 15,
    "donation_protection_extension_remainder": 16, "balances_0": 17,
    "balances_1": 18, "admin_balances_0": 19, "admin_balances_1": 20,
    "D": 21, "xcp_profit": 22, "lp_xcp_profit": 23, "virtual_price": 24,
    "totalSupply": 32,
}

# AMM mutable layout follows the declarations in yb-core/contracts/AMM.vy;
# debt and rate_time are not public getters, so retain their raw words.
AMM_SLOTS = {
    "fee": 0, "collateral_amount": 1, "debt": 2, "rate": 3,
    "rate_mul": 4, "rate_time": 5, "minted": 6, "redeemed": 7,
    "is_killed": 8,
}


def jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, tuple):
        return [jsonable(x) for x in value]
    if isinstance(value, list):
        return [jsonable(x) for x in value]
    if isinstance(value, str):
        return value.lower() if value.startswith("0x") else value
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=jsonable) + "\n")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=jsonable) + "\n")


def chunks(start: int, end: int) -> list[tuple[int, int]]:
    return [(lo, min(end, lo + CHUNK - 1)) for lo in range(start, end + 1, CHUNK)]


def fetch_logs(rpc: RPC, addresses: list[str]) -> list[dict[str, Any]]:
    jobs = [(address, lo, hi) for address in addresses for lo, hi in chunks(START_BLOCK, END_BLOCK)]

    def fetch(job: tuple[str, int, int]) -> list[dict[str, Any]]:
        address, lo, hi = job
        return rpc.request("eth_getLogs", [{"address": address, "fromBlock": hex(lo), "toBlock": hex(hi)}])

    result: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for part in pool.map(fetch, jobs):
            result.extend(part)
    result.sort(key=lambda x: (int(x["blockNumber"], 16), int(x["transactionIndex"], 16), int(x["logIndex"], 16), x["address"].lower()))
    return result


def topic_registry() -> dict[tuple[str, str], tuple[str, str, list[str], list[str], list[str], list[str]]]:
    result = {}
    for address, defs in EVENTS.items():
        for name, (signature, inames, itypes, dnames, dtypes) in defs.items():
            result[(address, "0x" + keccak(text=signature).hex())] = (address, name, inames, itypes, dnames, dtypes)
    return result


def decode_log(raw: dict[str, Any], registry: dict[tuple[str, str], tuple]) -> dict[str, Any]:
    topics = [x.lower() for x in raw.get("topics", [])]
    base = {"address": raw["address"].lower(), "block": int(raw["blockNumber"], 16), "transaction": raw["transactionHash"].lower(), "transaction_index": int(raw["transactionIndex"], 16), "log_index": int(raw["logIndex"], 16), "raw": raw}
    spec = registry.get((raw["address"].lower(), topics[0])) if topics else None
    if spec is None:
        return {**base, "decode_status": "unknown_topic", "event": None, "args": None}
    _, name, inames, itypes, dnames, dtypes = spec
    try:
        if len(topics) != len(itypes) + 1:
            raise ValueError("indexed topic count mismatch")
        indexed = [jsonable(decode([typ], bytes.fromhex(topic[2:]))[0]) for typ, topic in zip(itypes, topics[1:])]
        data = [jsonable(x) for x in decode(dtypes, bytes.fromhex(raw["data"][2:]))] if dtypes else []
        args = dict(zip(inames, indexed)) | dict(zip(dnames, data))
        return {**base, "decode_status": "known", "event": name, "args": args}
    except Exception as exc:
        return {**base, "decode_status": "decode_error", "event": name, "args": None, "decode_error": type(exc).__name__}


def classify_transactions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[(event["transaction"], event["block"], event["transaction_index"])].append(event)
    rows = []
    for tx_order, (key, legs) in enumerate(sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))):
        families: set[str] = set()
        if any(x["address"] == VP and x["event"] == "TokenExchange" for x in legs):
            families.add("virtual_pool")
        if any(x["address"] == LT and x["event"] == "Deposit" for x in legs):
            families.add("lt_deposit")
        if any(x["address"] == LT and x["event"] == "Withdraw" for x in legs):
            families.add("lt_withdraw")
        if not families and any(x["address"] == AMM for x in legs):
            families.add("yb_amm")
        if not families and any(x["address"] == POOL for x in legs):
            families.add("native_pool")
        family = next(iter(families)) if len(families) == 1 else ("ambiguous" if len(families) > 1 else "unknown")
        donation = [x for x in legs if x["address"] == POOL and x["event"] == "Donation"]
        zero_add = [x for x in legs if x["address"] == POOL and x["event"] == "AddLiquidity" and x.get("args", {}).get("receiver") == ZERO]
        pairing = None
        pairing_status = "not_applicable"
        if donation or zero_add:
            pairing_status = "ambiguous_or_mismatch"
        if len(donation) == len(zero_add) == 1:
            dargs, aargs = donation[0].get("args", {}), zero_add[0].get("args", {})
            if (donation[0]["log_index"] < zero_add[0]["log_index"]
                    and dargs.get("donor") == aargs.get("provider")
                    and dargs.get("token_amounts") == aargs.get("token_amounts")):
                pairing = {"donation_log_index": donation[0]["log_index"], "add_liquidity_log_index": zero_add[0]["log_index"]}
                pairing_status = "paired"
        for leg in legs:
            leg["tx_order"] = tx_order
            leg["leg_type"] = "native_pool_subleg" if leg["address"] == POOL and leg["event"] in {"AddLiquidity", "Donation", "RemoveLiquidity", "RemoveLiquidityOne", "RemoveLiquidityImbalance"} else "outer_or_auxiliary"
        rows.append({"tx_order": tx_order, "transaction": key[0], "block": key[1], "transaction_index": key[2], "outer_family": family, "family_candidates": sorted(families), "donation_add_pair": pairing, "donation_add_pair_status": pairing_status, "event_log_indices": [x["log_index"] for x in legs], "known_event_count": sum(x["decode_status"] == "known" for x in legs), "quarantined_event_count": sum(x["decode_status"] != "known" for x in legs), "events": legs})
    return rows


def call_data(signature: str, arg_types: list[str] = [], args: list[Any] = []) -> bytes:
    return selector(signature) + (encode(arg_types, args) if arg_types else b"")


def multicall(rpc: RPC, calls: list[tuple[str, str, bytes, list[str]]], block: int) -> dict[str, Any]:
    payload = encode(["(address,bool,bytes)[]"], [[(target, True, data) for _, target, data, _ in calls]])
    result = rpc.at(MULTICALL3, selector("aggregate3((address,bool,bytes)[])") + payload, block)
    returned = decode(["(bool,bytes)[]"], result)[0]
    values = {}
    for (name, _, _, output_types), (ok, raw) in zip(calls, returned):
        values[name] = {"ok": bool(ok), "value": jsonable(decode(output_types, raw)[0] if ok and len(output_types) == 1 else [jsonable(x) for x in decode(output_types, raw)] if ok else None), "raw": "0x" + raw.hex()}
    return values


def specs(token_addresses: list[str]) -> list[tuple[str, str, bytes, list[str]]]:
    c: list[tuple[str, str, bytes, list[str]]] = []
    def add(name: str, target: str, sig: str, args: list[Any], outputs: list[str], arg_types: list[str] = []):
        c.append((name, target, call_data(sig, arg_types, args), outputs))
    pool_calls = [
        ("pool_coins", "coins(uint256)", [0], ["address"], ["uint256"]), ("pool_coins_1", "coins(uint256)", [1], ["address"], ["uint256"]),
        ("pool_A", "A()", [], ["uint256"], []), ("pool_gamma", "gamma()", [], ["uint256"], []), ("pool_price_scale", "price_scale()", [], ["uint256"], []), ("pool_price_oracle_projected", "price_oracle()", [], ["uint256"], []), ("pool_last_prices", "last_prices()", [], ["uint256"], []), ("pool_last_timestamp", "last_timestamp()", [], ["uint256"], []), ("pool_packed_rebalancing_params", "packed_rebalancing_params()", [], ["uint256"], []), ("pool_packed_fee_params", "packed_fee_params()", [], ["uint256"], []), ("pool_reserved_profit_fraction", "reserved_profit_fraction()", [], ["uint256"], []), ("pool_admin_fee", "admin_fee()", [], ["uint256"], []), ("pool_donation_shares_max_ratio", "donation_shares_max_ratio()", [], ["uint256"], []), ("pool_donation_protection_lp_threshold", "donation_protection_lp_threshold()", [], ["uint256"], []), ("pool_ma_time_projected", "ma_time()", [], ["uint256"], []),
        ("pool_balances", "balances(uint256)", [0], ["uint256"], ["uint256"]), ("pool_balances_1", "balances(uint256)", [1], ["uint256"], ["uint256"]), ("pool_admin_balances", "admin_balances(uint256)", [0], ["uint256"], ["uint256"]), ("pool_admin_balances_1", "admin_balances(uint256)", [1], ["uint256"], ["uint256"]), ("pool_D", "D()", [], ["uint256"], []), ("pool_totalSupply", "totalSupply()", [], ["uint256"], []), ("pool_virtual_price", "virtual_price()", [], ["uint256"], []), ("pool_xcp_profit", "xcp_profit()", [], ["uint256"], []), ("pool_lp_xcp_profit", "lp_xcp_profit()", [], ["uint256"], []), ("pool_donation_shares", "donation_shares()", [], ["uint256"], []), ("pool_donation_duration", "donation_duration()", [], ["uint256"], []), ("pool_last_donation_release_ts", "last_donation_release_ts()", [], ["uint256"], []), ("pool_donation_protection_expiry_ts", "donation_protection_expiry_ts()", [], ["uint256"], []), ("pool_donation_protection_period", "donation_protection_period()", [], ["uint256"], []),
    ]
    for name, sig, args, outputs, arg_types in pool_calls:
        add(name, POOL, sig, args, outputs, arg_types)
    for name, sig, outputs in [("amm_LT_CONTRACT", "LT_CONTRACT()", ["address"]), ("amm_COLLATERAL", "COLLATERAL()", ["address"]), ("amm_STABLECOIN", "STABLECOIN()", ["address"]), ("amm_PRICE_ORACLE_CONTRACT", "PRICE_ORACLE_CONTRACT()", ["address"]), ("amm_fee", "fee()", ["uint256"]), ("amm_rate", "rate()", ["uint256"]), ("amm_rate_mul", "rate_mul()", ["uint256"]), ("amm_collateral_amount", "collateral_amount()", ["uint256"]), ("amm_minted", "minted()", ["uint256"]), ("amm_redeemed", "redeemed()", ["uint256"]), ("amm_get_debt_projected", "get_debt()", ["uint256"]), ("amm_get_rate_mul_projected", "get_rate_mul()", ["uint256"]), ("amm_get_p_projected", "get_p()", ["uint256"]), ("amm_get_state", "get_state()", ["uint256", "uint256", "uint256"]), ("amm_is_killed", "is_killed()", ["bool"])]:
        add(name, AMM, sig, [], outputs)
    for name, sig, outputs in [("lt_CRYPTOPOOL", "CRYPTOPOOL()", ["address"]), ("lt_amm", "amm()", ["address"]), ("lt_STABLECOIN", "STABLECOIN()", ["address"]), ("lt_ASSET_TOKEN", "ASSET_TOKEN()", ["address"]), ("lt_totalSupply", "totalSupply()", ["uint256"]), ("lt_stablecoin_allocation", "stablecoin_allocation()", ["uint256"]), ("lt_stablecoin_allocated", "stablecoin_allocated()", ["uint256"]), ("lt_updated_balances", "updated_balances()", ["uint256", "uint256"]), ("lt_pricePerShare", "pricePerShare()", ["uint256"]), ("lt_decimals", "decimals()", ["uint8"])]:
        add(name, LT, sig, [], outputs)
    for name, sig in [("vp_POOL", "POOL()"), ("vp_AMM", "AMM()"), ("vp_ASSET_TOKEN", "ASSET_TOKEN()"), ("vp_STABLECOIN", "STABLECOIN()")]:
        add(name, VP, sig, [], ["address"])
    for token in token_addresses:
        label = "token_" + token[-6:]
        for suffix, sig, outputs in [("name", "name()", ["string"]), ("symbol", "symbol()", ["string"]), ("decimals", "decimals()", ["uint8"])]:
            add(f"{label}_{suffix}", token, sig, [], outputs)
        for holder, holder_address in [("pool", POOL), ("amm", AMM), ("lt", LT), ("vp", VP)]:
            add(f"{label}_balance_{holder}", token, "balanceOf(address)", [holder_address], ["uint256"], ["address"])
    return c


def snapshot(rpc: RPC, block: int, token_addresses: list[str]) -> dict[str, Any]:
    projected = multicall(rpc, specs(token_addresses), block)
    stored = {}
    for name, slot in POOL_SLOTS.items():
        raw = rpc.request("eth_getStorageAt", [POOL, hex(slot), hex(block)])
        stored[name] = {"slot": slot, "raw": raw.lower(), "value": int(raw, 16)}
    amm_stored = {}
    for name, slot in AMM_SLOTS.items():
        raw = rpc.request("eth_getStorageAt", [AMM, hex(slot), hex(block)])
        amm_stored[name] = {"slot": slot, "raw": raw.lower(), "value": int(raw, 16)}
    packed = projected.get("pool_packed_rebalancing_params", {}).get("value")
    if isinstance(packed, int):
        projected["pool_packed_rebalancing_params"] = {"raw": str(packed), "adjustment_step_min": packed >> 128, "adjustment_step_max": (packed >> 64) & ((1 << 64) - 1), "ma_exp_time_raw": packed & ((1 << 64) - 1), "projected_ma_time_half_time": projected.get("pool_ma_time_projected", {}).get("value")}
    checks = {}
    for stored_name, getter_name in {"balances_0": "pool_balances", "balances_1": "pool_balances_1", "admin_balances_0": "pool_admin_balances", "admin_balances_1": "pool_admin_balances_1", "D": "pool_D", "totalSupply": "pool_totalSupply", "virtual_price": "pool_virtual_price", "xcp_profit": "pool_xcp_profit", "lp_xcp_profit": "pool_lp_xcp_profit", "donation_shares": "pool_donation_shares", "donation_duration": "pool_donation_duration", "last_donation_release_ts": "pool_last_donation_release_ts", "donation_protection_expiry_ts": "pool_donation_protection_expiry_ts", "donation_protection_period": "pool_donation_protection_period"}.items():
        checks[stored_name] = stored[stored_name]["value"] == projected.get(getter_name, {}).get("value")
    for stored_name, getter_name in {"fee": "amm_fee", "collateral_amount": "amm_collateral_amount", "rate": "amm_rate", "rate_mul": "amm_rate_mul", "minted": "amm_minted", "redeemed": "amm_redeemed", "is_killed": "amm_is_killed"}.items():
        checks["amm_" + stored_name] = stored_name in amm_stored and amm_stored[stored_name]["value"] == projected.get(getter_name, {}).get("value")
    verification = {"checks": checks, "verified": all(checks.values()), "incomplete_if_false": "storage layout must not seed a simulator checkpoint when any check is false"}
    stable_label = "token_" + token_addresses[0][-6:]
    return {"block": block, "block_header": rpc.block(block), "stored_native": stored, "stored_amm": amm_stored, "projected_getters": projected, "role_balances": {"stablecoin_address": token_addresses[0], "stable_balance_amm": projected.get(stable_label + "_balance_amm"), "lt_stable_balance": projected.get(stable_label + "_balance_lt")}, "storage_layout_verification": verification, "classification": {"cached_price_oracle": "eth_getStorageAt slot 2 is authoritative", "price_oracle_projected": "eth_call getter retained as auxiliary; it is not substituted for cached storage", "packed_rebalancing_ma_time": "low 64 bits are raw exponential time; pool ma_time getter is half-time", "lt_stablecoin_allocation": "separate LT accounting field; never mapped from AMM stablecoin balance"}}


def acquire_external_inputs(rpc: RPC, out: Path) -> None:
    aug1_target = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp())
    aug1_block = first_block(rpc, aug1_target, END_BLOCK)
    blocks = [PRESTART_BLOCK, aug1_block]

    def read(block: int, target: str, signature: str, output_type: str, arg_types: list[str] = [], args: list[Any] = []) -> dict[str, Any]:
        data = call_data(signature, arg_types, args)
        raw = rpc.request("eth_call", [{"to": target, "data": "0x" + data.hex()}, hex(block)])
        value = jsonable(decode([output_type], bytes.fromhex(raw[2:]))[0])
        return {"target": target, "signature": signature, "block": block, "calldata": "0x" + data.hex(), "raw_return": raw.lower(), "value": value}

    rows = []
    for block in blocks:
        header = rpc.block(block)
        coin0 = read(block, POOL, "coins(uint256)", "address", ["uint256"], [0])
        amm_leverage = read(block, AMM, "LEVERAGE()", "uint256")
        oracle_proxy = read(block, AMM, "PRICE_ORACLE_CONTRACT()", "address")
        agg = read(block, oracle_proxy["value"], "AGG()", "address")
        agg_price = read(block, agg["value"], "price()", "uint256")
        vp_factory = read(block, VP, "FACTORY()", "address")
        flash = read(block, vp_factory["value"], "flash()", "address")
        max_flash = read(block, flash["value"], "maxFlashLoan(address)", "uint256", ["address"], [coin0["value"]])
        rounding = read(block, VP, "ROUNDING_DISCOUNT()", "uint256")
        rows.append({"block": block, "block_hash": header["hash"].lower(), "timestamp": int(header["timestamp"], 16), "requests": {"pool_coin0": coin0, "amm_leverage": amm_leverage, "amm_price_oracle_contract": oracle_proxy, "oracle_agg": agg, "oracle_agg_price": agg_price, "vp_factory": vp_factory, "factory_flash": flash, "factory_flash_max_loan_crvusd": max_flash, "vp_rounding_discount": rounding}, "semantics": {"leverage_raw": amm_leverage["value"], "oracle_price_raw": agg_price["value"], "rounding_discount_raw": rounding["value"], "rounding_discount_fraction_denominator": 10**18, "rounding_discount_note": "ROUNDING_DISCOUNT is the fraction removed from 1e18, equal to 1e10; it is not one-minus", "max_flash_loan_token": coin0["value"]}})
    source_hashes = {name: sha256_file(Path(path)) for name, path in SOURCE_FILES.items()}
    write_json(out / "yb_external_inputs.json", {"schema_version": 1, "chain_id": CHAIN_ID, "blocks": rows, "source_provenance": {"files": SOURCE_FILES, "sha256": source_hashes}, "rpc_cache": {"directory": "rpc_cache", "entries": len(rpc.index), "envelope_sha256": rpc.index}, "limitation": "Exact external inputs are captured at prestate and the 2026-08-01 UTC checkpoint only; this is not a per-transaction external-state tape."})
    print(json.dumps({"external_blocks": blocks, "cache_entries": len(rpc.index)}))


def acquire_action_states(rpc: RPC, out: Path) -> None:
    events = [json.loads(line) for line in (out / "events_log_order.jsonl").read_text().splitlines()]
    receipts = [json.loads(line) for line in (out / "receipts.jsonl").read_text().splitlines()]
    action_blocks = sorted({event["block"] for event in events})
    headers = {row["block"]: row["header"] for row in receipts}
    missing_headers = [block for block in action_blocks if block not in headers]
    if missing_headers:
        raise RuntimeError(f"missing cached receipt headers for {len(missing_headers)} action blocks")
    calls = [("price_scale", POOL, call_data("price_scale()"), ["uint256"]), ("virtual_price", POOL, call_data("virtual_price()"), ["uint256"]), ("lp_xcp_profit", POOL, call_data("lp_xcp_profit()"), ["uint256"]), ("xcp_profit", POOL, call_data("xcp_profit()"), ["uint256"]), ("balances_0", POOL, call_data("balances(uint256)", ["uint256"], [0]), ["uint256"]), ("balances_1", POOL, call_data("balances(uint256)", ["uint256"], [1]), ["uint256"]), ("totalSupply", POOL, call_data("totalSupply()"), ["uint256"]), ("donation_shares", POOL, call_data("donation_shares()"), ["uint256"]), ("last_timestamp", POOL, call_data("last_timestamp()"), ["uint256"]), ("last_prices", POOL, call_data("last_prices()"), ["uint256"]), ("D", POOL, call_data("D()"), ["uint256"])]
    cache_before = len(rpc.index)
    def read(block: int) -> dict[str, Any]:
        values = multicall(rpc, calls, block)
        if not all(item["ok"] for item in values.values()):
            failed = [name for name, item in values.items() if not item["ok"]]
            raise RuntimeError(f"required native getter failed at block {block}: {failed}")
        return {"block": block, "block_hash": headers[block]["hash"].lower(), "timestamp": int(headers[block]["timestamp"], 16), "getters": values, "required_getters_ok": True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(read, action_blocks))
    daily = {row["block"]: row for row in (json.loads(line) for line in (out / "state_snapshots.jsonl").read_text().splitlines())}
    fields = {"price_scale": "pool_price_scale", "virtual_price": "pool_virtual_price", "lp_xcp_profit": "pool_lp_xcp_profit", "xcp_profit": "pool_xcp_profit", "balances_0": "pool_balances", "balances_1": "pool_balances_1", "totalSupply": "pool_totalSupply", "donation_shares": "pool_donation_shares", "last_timestamp": "pool_last_timestamp", "last_prices": "pool_last_prices", "D": "pool_D"}
    overlap = []
    for row in rows:
        reference = daily.get(row["block"])
        if reference:
            mismatches = [name for name, getter in fields.items() if row["getters"][name]["value"] != reference["projected_getters"].get(getter, {}).get("value")]
            overlap.append({"block": row["block"], "match": not mismatches, "mismatches": mismatches})
    if any(not row["match"] for row in overlap):
        raise RuntimeError("action state disagrees with an overlapping daily snapshot")
    write_jsonl(out / "action_block_native_states.jsonl", rows)
    artifact = out / "action_block_native_states.jsonl"
    summary = {"schema_version": 1, "window": {"start_block": START_BLOCK, "end_block": END_BLOCK, "prestate_block": PRESTART_BLOCK}, "coverage": "all observed action blocks from decoded event tape; block headers reused from cached receipt artifacts", "counts": {"action_blocks": len(action_blocks), "rows": len(rows), "daily_overlap_blocks": len(overlap), "daily_overlap_mismatches": sum(not row["match"] for row in overlap)}, "rpc": {"multicall_calls": len(action_blocks), "new_cache_entries": len(rpc.index) - cache_before, "worker_count": 4}, "artifact": {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}, "limitations": ["Each row is block-end state; intratransaction state and action ordering are not reconstructed here.", "Only the required native pool getters are sampled; this is not a replacement for the full activity tape."]}
    write_json(out / "action_block_native_states_summary.json", summary)
    print(json.dumps({"action_blocks": len(rows), "daily_overlaps": len(overlap), "new_cache_entries": summary["rpc"]["new_cache_entries"], "sha256": summary["artifact"]["sha256"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/yb-cbbtc-historical-20260904/activity"))
    ap.add_argument("--external-only", action="store_true", help="capture prestate and Aug 1 external inputs without rereading activity tapes")
    ap.add_argument("--action-state-only", action="store_true", help="capture required native state at every observed action block")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rpc = RPC(out)
    if rpc.request("eth_chainId", []) != "0x1":
        raise RuntimeError("RPC is not Ethereum mainnet")
    if args.external_only:
        acquire_external_inputs(rpc, out)
        return
    if args.action_state_only:
        acquire_action_states(rpc, out)
        return
    boundary_headers = {str(n): rpc.block(n) for n in (PRESTART_BLOCK, START_BLOCK, END_BLOCK)}
    all_raw = fetch_logs(rpc, [POOL, AMM, LT, VP])
    registry = topic_registry()
    decoded = [decode_log(raw, registry) for raw in all_raw]
    for index, event in enumerate(decoded):
        event["log_order"] = index
    tx_rows = classify_transactions(decoded)
    # Publish the raw and decoded tapes before slower receipt/state harvesting.
    write_jsonl(out / "raw_logs.jsonl", all_raw)
    write_jsonl(out / "events_log_order.jsonl", decoded)
    write_jsonl(out / "events_transaction_order.jsonl", [event for tx in tx_rows for event in tx["events"]])
    write_json(out / "transactions.json", tx_rows)
    token_probe = multicall(rpc, [("pool_coins", POOL, call_data("coins(uint256)", ["uint256"], [0]), ["address"]), ("pool_coins_1", POOL, call_data("coins(uint256)", ["uint256"], [1]), ["address"])], END_BLOCK)
    token_addresses = [token_probe["pool_coins"]["value"], token_probe["pool_coins_1"]["value"]]
    # Make the simulator's historical starting point available before receipt
    # and daily harvesting, which can involve many more RPC calls.
    prestate_snapshot = snapshot(rpc, PRESTART_BLOCK, token_addresses)
    write_jsonl(out / "state_snapshots.jsonl", [prestate_snapshot])
    tx_keys = [(row["transaction"], row["block"], row["transaction_index"]) for row in tx_rows]
    if len(tx_keys) > RECEIPT_LIMIT:
        raise RuntimeError(f"{len(tx_keys)} transactions exceed receipt call cap {RECEIPT_LIMIT}")

    def receipt(key: tuple[str, int, int]) -> dict[str, Any]:
        tx, block, tx_index = key
        return {"transaction": tx, "block": block, "transaction_index": tx_index, "receipt": rpc.request("eth_getTransactionReceipt", [tx]), "header": rpc.block(block)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(receipt, tx_keys))
    for row in receipts:
        receipt_raw = row["receipt"] or {}
        gas = int(receipt_raw.get("gasUsed", "0x0"), 16) if receipt_raw else 0
        price = int(receipt_raw.get("effectiveGasPrice", "0x0"), 16) if receipt_raw else 0
        row["gas_used"] = gas
        row["effective_gas_price_wei"] = price
        row["gas_cost_wei"] = gas * price
        row["status"] = int(receipt_raw.get("status", "0x0"), 16) if receipt_raw else None
        row["timestamp"] = int(row["header"]["timestamp"], 16)

    action_blocks = sorted({event["block"] for event in decoded if event["decode_status"] == "known"})
    start_ts = int(boundary_headers[str(START_BLOCK)]["timestamp"], 16)
    end_ts = int(boundary_headers[str(END_BLOCK)]["timestamp"], 16)
    daily = []
    cursor = datetime.fromtimestamp(start_ts, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor.timestamp() <= end_ts:
        daily.append(first_block(rpc, max(start_ts, int(cursor.timestamp())), END_BLOCK))
        cursor += timedelta(days=1)
    daily = sorted(set(daily))
    # A snapshot includes one Multicall3 plus cached native and AMM storage
    # reads. Keep the initial harvest within the total RPC budget; action
    # blocks remain identified for a later targeted expansion.
    snapshot_blocks = sorted(set([PRESTART_BLOCK, START_BLOCK, END_BLOCK, *daily]))
    coverage = "daily_and_boundaries_only_action_blocks_retained_for_targeted_expansion"
    snapshots = [prestate_snapshot] + [snapshot(rpc, block, token_addresses) for block in snapshot_blocks if block != PRESTART_BLOCK]
    write_jsonl(out / "receipts.jsonl", receipts)
    write_jsonl(out / "state_snapshots.jsonl", snapshots)
    source_hashes = {name: sha256_file(Path(path)) for name, path in SOURCE_FILES.items()}
    end_values = snapshots[-1]["projected_getters"]
    immutable_expected = {"amm_COLLATERAL": POOL, "amm_LT_CONTRACT": LT, "lt_CRYPTOPOOL": POOL, "lt_amm": AMM, "vp_POOL": POOL, "vp_AMM": AMM}
    immutable_checks = {name: end_values.get(name, {}).get("value") == expected for name, expected in immutable_expected.items()}
    token_decimals = {name: end_values.get("token_" + address[-6:] + "_decimals", {}).get("value") for name, address in zip(("stablecoin", "asset"), token_addresses)}
    if not all(immutable_checks.values()) or token_decimals != {"stablecoin": 18, "asset": 8}:
        raise RuntimeError(f"immutable/token verification failed: {immutable_checks} {token_decimals}")
    artifact_paths = [out / name for name in ("raw_logs.jsonl", "events_log_order.jsonl", "events_transaction_order.jsonl", "transactions.json", "receipts.jsonl", "state_snapshots.jsonl")]
    summary = {"schema_version": 1, "chain_id": CHAIN_ID, "acquisition_command": "UV_CACHE_DIR=/tmp/uv-cache uv run --group research python scripts/acquire_yb_activity.py", "addresses": {"pool": POOL, "amm": AMM, "lt": LT, "virtual_pool": VP}, "window": {"start_block": START_BLOCK, "end_block": END_BLOCK, "prestate_block": PRESTART_BLOCK, "start_timestamp": start_ts, "end_timestamp": end_ts, "boundary_headers": boundary_headers}, "counts": {"raw_logs": len(all_raw), "decoded_events": len(decoded), "known_events": sum(x["decode_status"] == "known" for x in decoded), "quarantined_logs": sum(x["decode_status"] != "known" for x in decoded), "transactions": len(tx_rows), "receipts": len(receipts), "action_blocks": len(action_blocks), "snapshot_blocks": len(snapshots)}, "outer_family_counts": {family: sum(row["outer_family"] == family for row in tx_rows) for family in sorted({row["outer_family"] for row in tx_rows})}, "receipt_status_counts": {str(status): sum(row["status"] == status for row in receipts) for status in sorted({row["status"] for row in receipts})}, "gas": {"total_gas_used": sum(row["gas_used"] for row in receipts), "total_cost_wei": sum(row["gas_cost_wei"] for row in receipts)}, "snapshot_coverage": coverage, "snapshot_storage_layout_verified": all(row["storage_layout_verification"]["verified"] for row in snapshots), "token_identity": {"pool_coins": token_addresses, "token_decimals": token_decimals, "immutable_getter_checks": immutable_checks, "immutable_getters_verified_at_end": {k: v for k, v in snapshots[-1]["projected_getters"].items() if k.startswith(("amm_", "lt_", "vp_", "token_"))}}, "source_provenance": {"files": SOURCE_FILES, "sha256": source_hashes, "pool_storage_layout": "cook_august.py native slots 1-4,9,11-24,32; cached fields authoritative"}, "rpc_cache": {"directory": "rpc_cache", "entries": len(rpc.index), "envelope_sha256": rpc.index}, "limitations": ["Observed contract logs and receipts are recorded; no profit labels are inferred.", "The event tape has no order-book or executable-depth observation.", "Intratransaction state is not reconstructed; snapshots are block-end views.", "Projected price_oracle() is retained for reference and is never used as the cached native price oracle.", "Unknown event topics and decode failures remain in raw_logs.jsonl and are quarantined in decoded counts."]}
    summary["artifacts"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifact_paths}
    write_json(out / "activity_summary.json", summary)
    print(json.dumps({"logs": len(all_raw), "known": summary["counts"]["known_events"], "quarantined": summary["counts"]["quarantined_logs"], "transactions": len(tx_rows), "snapshots": len(snapshots), "rpc_cache_entries": len(rpc.index)}))


if __name__ == "__main__":
    main()
