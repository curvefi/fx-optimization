"""Acquire bounded WETH market activity using the verified market-10 graph."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import acquire_yb_activity as h  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904"
OUT = STUDY / "weth-preparation/activity"
WINDOW = STUDY / "weth-preparation/configuration_window.json"
CB_DAILY = STUDY / "activity/state_snapshots.jsonl"
START_BLOCK, END_BLOCK, PRESTART_BLOCK = 25_455_434, 25_857_181, 25_455_433
START_TS, END_TS = 1_783_123_200, 1_787_961_600
CRVUSD = "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


def patch_roles() -> dict:
    graph = json.loads(WINDOW.read_text())["factory_market_graph"]["current_graph"]
    if graph.get("cryptopool") != "0x656341ef90b622c6634e0573772ffb7f3669b9f3":
        raise RuntimeError("WETH preparation graph does not select the market-10 cryptopool")
    roles = {"pool": graph["cryptopool"], "amm": graph["amm"], "lt": graph["lt"], "vp": graph["virtual_pool"]}
    old = h.EVENTS
    h.EVENTS = {roles["pool"]: old[h.POOL], roles["amm"]: old[h.AMM], roles["lt"]: old[h.LT], roles["vp"]: old[h.VP]}
    h.POOL, h.AMM, h.LT, h.VP = roles["pool"], roles["amm"], roles["lt"], roles["vp"]
    h.START_BLOCK, h.END_BLOCK, h.PRESTART_BLOCK = START_BLOCK, END_BLOCK, PRESTART_BLOCK
    return roles


def read(rpc, target: str, signature: str, output: str, block: int, arg_types=None, args=None) -> dict:
    data = h.call_data(signature, arg_types or [], args or [])
    raw = rpc.request("eth_call", [{"to": target, "data": "0x" + data.hex()}, hex(block)])
    return {"target": target, "signature": signature, "block": block, "calldata": "0x" + data.hex(), "raw_return": raw.lower(), "value": h.jsonable(h.decode([output], bytes.fromhex(raw[2:]))[0])}


def external_inputs(rpc, aug1_block: int) -> None:
    rows = []
    for block in (PRESTART_BLOCK, aug1_block):
        header = rpc.block(block)
        coin0 = read(rpc, h.POOL, "coins(uint256)", "address", block, ["uint256"], [0])
        leverage = read(rpc, h.AMM, "LEVERAGE()", "uint256", block)
        oracle = read(rpc, h.AMM, "PRICE_ORACLE_CONTRACT()", "address", block)
        agg = read(rpc, oracle["value"], "AGG()", "address", block)
        price = read(rpc, agg["value"], "price()", "uint256", block)
        factory = read(rpc, h.VP, "FACTORY()", "address", block)
        flash = read(rpc, factory["value"], "flash()", "address", block)
        max_loan = read(rpc, flash["value"], "maxFlashLoan(address)", "uint256", block, ["address"], [coin0["value"]])
        rounding = read(rpc, h.VP, "ROUNDING_DISCOUNT()", "uint256", block)
        rows.append({"block": block, "block_hash": header["hash"].lower(), "timestamp": int(header["timestamp"], 16), "requests": {"pool_coin0": coin0, "amm_leverage": leverage, "amm_price_oracle_contract": oracle, "oracle_agg": agg, "oracle_agg_price": price, "vp_factory": factory, "factory_flash": flash, "factory_flash_max_loan_crvusd": max_loan, "vp_rounding_discount": rounding}})
    h.write_json(OUT / "yb_external_inputs.json", {"schema_version": 1, "chain_id": 1, "blocks": rows, "source_provenance": {"files": h.SOURCE_FILES}, "limitation": "Exact external inputs are captured at prestate and the 2026-08-01 UTC checkpoint only; no per-transaction external tape."})


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    roles = patch_roles()
    rpc = h.RPC(OUT)
    if rpc.request("eth_chainId", []) != "0x1":
        raise RuntimeError("RPC is not Ethereum mainnet")
    all_raw = h.fetch_logs(rpc, list(roles.values()))
    registry = h.topic_registry()
    decoded = [h.decode_log(raw, registry) for raw in all_raw]
    headers = {}
    for raw, event in zip(all_raw, decoded):
        block = event["block"]
        stamp = raw.get("blockTimestamp")
        if not stamp or int(stamp, 16) == 0:
            headers.setdefault(block, rpc.block(block))
            event["event_timestamp"] = int(headers[block]["timestamp"], 16)
            event["timestamp_provenance"] = "eth_getBlockByNumber"
        else:
            event["event_timestamp"] = int(stamp, 16)
            event["timestamp_provenance"] = "eth_getLogs.blockTimestamp"
    for index, event in enumerate(decoded):
        event["log_order"] = index
    tx_rows = h.classify_transactions(decoded)
    h.write_jsonl(OUT / "raw_logs.jsonl", all_raw)
    h.write_jsonl(OUT / "events_log_order.jsonl", decoded)
    h.write_jsonl(OUT / "events_transaction_order.jsonl", [event for tx in tx_rows for event in tx["events"]])
    h.write_json(OUT / "transactions.json", tx_rows)
    token_probe = h.multicall(rpc, [("coin0", h.POOL, h.call_data("coins(uint256)", ["uint256"], [0]), ["address"]), ("coin1", h.POOL, h.call_data("coins(uint256)", ["uint256"], [1]), ["address"])], END_BLOCK)
    tokens = [token_probe["coin0"]["value"].lower(), token_probe["coin1"]["value"].lower()]
    if tokens != [CRVUSD, WETH]:
        raise RuntimeError(f"unexpected WETH pool coins: {tokens}")
    prestate, endstate = h.snapshot(rpc, PRESTART_BLOCK, tokens), h.snapshot(rpc, END_BLOCK, tokens)
    labels = ["token_" + token[-6:] + "_decimals" for token in tokens]
    decimals = [endstate["projected_getters"][label]["value"] for label in labels]
    if decimals != [18, 18]:
        raise RuntimeError(f"unexpected WETH pool coin decimals: {decimals}")
    h.write_jsonl(OUT / "state_snapshots.jsonl", [prestate, endstate])
    cb_rows = [json.loads(line) for line in CB_DAILY.read_text().splitlines()]
    daily_headers = [{"block": row["block"], "header": row["block_header"], "provenance": "reused_cbBTC_daily_header"} for row in cb_rows if START_BLOCK <= row["block"] <= END_BLOCK]
    if not any(row["block"] == END_BLOCK for row in daily_headers):
        daily_headers.append({"block": END_BLOCK, "header": rpc.block(END_BLOCK), "provenance": "fetched_WETH_end_boundary"})
    daily_headers.sort(key=lambda row: row["block"])
    h.write_jsonl(OUT / "daily_block_headers.jsonl", daily_headers)
    daily_states = [{"block": row["block"], "block_hash": row["header"]["hash"].lower(), "timestamp": int(row["header"]["timestamp"], 16), "header_provenance": row["provenance"], "projected_getters": h.multicall(rpc, h.specs(tokens), row["block"])} for row in daily_headers]
    h.write_jsonl(OUT / "daily_native_yb_states.jsonl", daily_states)
    sampled = []
    for tx in tx_rows[:20]:
        sampled.append({"transaction": tx["transaction"], "block": tx["block"], "transaction_index": tx["transaction_index"], "receipt": rpc.request("eth_getTransactionReceipt", [tx["transaction"]]), "header": rpc.block(tx["block"]), "sampled_nonexhaustive": True})
    h.write_jsonl(OUT / "receipts.jsonl", sampled)
    aug1 = next(row["block"] for row in daily_headers if int(row["header"]["timestamp"], 16) >= int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp()))
    external_inputs(rpc, aug1)
    artifacts = ["raw_logs.jsonl", "events_log_order.jsonl", "events_transaction_order.jsonl", "transactions.json", "receipts.jsonl", "state_snapshots.jsonl", "daily_block_headers.jsonl", "daily_native_yb_states.jsonl", "yb_external_inputs.json"]
    summary = {"schema_version": 1, "status": "prepared", "chain_id": 1, "acquisition_command": "UV_CACHE_DIR=/tmp/uv-cache uv run --group research python scripts/acquire_yb_weth_activity.py", "addresses": roles, "window": {"start_block": START_BLOCK, "end_block": END_BLOCK, "prestate_block": PRESTART_BLOCK, "start_timestamp": START_TS, "end_exclusive_timestamp": END_TS}, "counts": {"raw_logs": len(all_raw), "decoded_events": len(decoded), "known_events": sum(x["decode_status"] == "known" for x in decoded), "unknown_or_decode_error": sum(x["decode_status"] != "known" for x in decoded), "transactions": len(tx_rows), "sampled_receipts": len(sampled), "daily_states": len(daily_states)}, "event_timestamp_provenance": {name: sum(x["timestamp_provenance"] == name for x in decoded) for name in {x["timestamp_provenance"] for x in decoded}}, "outer_family_counts": {name: sum(x["outer_family"] == name for x in tx_rows) for name in {x["outer_family"] for x in tx_rows}}, "coverage": {"logs": "all ordered logs for pool/AMM/LT/VirtualPool in window", "receipts": "first 20 transaction groups only; nonexhaustive gas sample", "daily_headers": {"reused_cbBTC": sum(x["provenance"] == "reused_cbBTC_daily_header" for x in daily_headers), "fetched_WETH_end": sum(x["provenance"] == "fetched_WETH_end_boundary" for x in daily_headers), "outside_window_excluded": len(cb_rows) - len(daily_headers) + 1}, "daily_state": "WETH projected getter snapshots; full storage snapshots at prestate/end"}, "token_identity": {"pool_coins": tokens, "decimals": {"crvUSD": 18, "WETH": 18}}, "source_provenance": {"files": h.SOURCE_FILES, "sha256": {name: h.sha256_file(Path(path)) for name, path in h.SOURCE_FILES.items()}, "configuration_window_sha256": h.sha256_file(WINDOW)}, "rpc_cache": {"directory": "rpc_cache", "entries": len(rpc.index), "envelope_sha256": rpc.index}, "limitations": ["No actor or profit labels are inferred.", "Unknown topics and decode errors remain in raw_logs.jsonl and decoded event outputs.", "Receipt/gas coverage is explicitly nonexhaustive and capped at 20."]}
    summary["artifacts"] = {name: {"bytes": (OUT / name).stat().st_size, "sha256": h.sha256_file(OUT / name)} for name in artifacts}
    h.write_json(OUT / "activity_summary.json", summary)
    print(json.dumps({"logs": len(all_raw), "known": summary["counts"]["known_events"], "unknown_or_decode_error": summary["counts"]["unknown_or_decode_error"], "transactions": len(tx_rows), "sampled_receipts": len(sampled), "daily_states": len(daily_states), "rpc_cache_entries": len(rpc.index)}))


if __name__ == "__main__":
    main()
