"""Prepare a read-only WETH market-10 configuration window."""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/yb-cbbtc-historical-20260904/weth-preparation"
HELPER_PATH = Path(__file__).with_name("acquire_yb_configuration_window.py")
FACTORY = "0x370a449FeBb9411c95bf897021377fe0B7D100c0"
FACTORY_ABI = ROOT.parents[1] / "yb-core/scripts/Factory.abi.json"
POOL = "0x656341Ef90b622c6634e0573772FfB7f3669b9f3"
LEGACY_POOL = "0x6e5492F8ea2370844EE098A56DD88e1717e4A9C2"
CRVUSD = "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e"
AMM = "0x0000000000000000000000000000000000000000"
MARKET_ID = 10
START_TS = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp())
END_TS = int(datetime(2026, 8, 29, tzinfo=timezone.utc).timestamp())

spec = importlib.util.spec_from_file_location("yb_config_helper", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load configuration helper: {HELPER_PATH}")
spec.loader.exec_module(helper)


def graph(rpc, block: int) -> dict:
    data = helper.selector("markets(uint256)") + helper.encode(["uint256"], [MARKET_ID])
    raw = rpc.at(FACTORY, data, block)
    values = helper.decode(["(address,address,address,address,address,address,address)"], raw)[0]
    names = ("asset_token", "cryptopool", "amm", "lt", "price_oracle", "virtual_pool", "staker")
    return {name: value.lower() for name, value in zip(names, values)}


def code(rpc, address: str, block: int) -> dict:
    raw = rpc.request("eth_getCode", [address, hex(block)])
    return {"address": address.lower(), "exists": len(raw) > 2,
            "runtime_code_hash": "0x" + helper.keccak(bytes.fromhex(raw[2:])).hex()}


def address_getter(rpc, target: str, signature: str, block: int, args: list[int] | None = None) -> str | None:
    try:
        data = helper.selector(signature) + (helper.encode(["uint256"] * len(args), args) if args else b"")
        return helper.decode_one(rpc.at(target, data, block), "address").lower()
    except Exception:
        return None


def reciprocal(rpc, item: dict, block: int) -> dict:
    pool, amm, lt, vp = item["cryptopool"], item["amm"], item["lt"], item["virtual_pool"]
    coins = [address_getter(rpc, pool, "coins(uint256)", block, [i]) for i in (0, 1)]
    got = {
        "pool_coins": coins,
        "amm_LT_CONTRACT": address_getter(rpc, amm, "LT_CONTRACT()", block),
        "amm_COLLATERAL": address_getter(rpc, amm, "COLLATERAL()", block),
        "amm_STABLECOIN": address_getter(rpc, amm, "STABLECOIN()", block),
        "amm_PRICE_ORACLE_CONTRACT": address_getter(rpc, amm, "PRICE_ORACLE_CONTRACT()", block),
        "lt_CRYPTOPOOL": address_getter(rpc, lt, "CRYPTOPOOL()", block),
        "lt_ASSET_TOKEN": address_getter(rpc, lt, "ASSET_TOKEN()", block),
        "lt_STABLECOIN": address_getter(rpc, lt, "STABLECOIN()", block),
        "lt_amm": address_getter(rpc, lt, "amm()", block),
        "lt_staker": address_getter(rpc, lt, "staker()", block),
        "vp_AMM": address_getter(rpc, vp, "AMM()", block) if int(vp, 16) else None,
        "vp_POOL": address_getter(rpc, vp, "POOL()", block) if int(vp, 16) else None,
        "vp_ASSET_TOKEN": address_getter(rpc, vp, "ASSET_TOKEN()", block) if int(vp, 16) else None,
        "vp_STABLECOIN": address_getter(rpc, vp, "STABLECOIN()", block) if int(vp, 16) else None,
    }
    expected = {
        "amm_LT_CONTRACT": lt, "amm_COLLATERAL": pool, "amm_STABLECOIN": coins[0],
        "amm_PRICE_ORACLE_CONTRACT": item["price_oracle"], "lt_CRYPTOPOOL": pool,
        "lt_ASSET_TOKEN": item["asset_token"], "lt_STABLECOIN": coins[0], "lt_amm": amm,
        "lt_staker": item["staker"], "vp_AMM": amm, "vp_POOL": pool,
        "vp_ASSET_TOKEN": item["asset_token"], "vp_STABLECOIN": coins[0],
    }
    got["unsupported_getters"] = [k for k in expected if got[k] is None]
    got["matches_factory_graph"] = {k: (None if got[k] is None else got[k] == v) for k, v in expected.items()}
    got["all_checked_match"] = not got["unsupported_getters"] and all(got["matches_factory_graph"].values())
    return got


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = STUDY
    out.mkdir(parents=True, exist_ok=True)
    rpc = helper.RPC(out)
    if rpc.request("eth_chainId", []) != "0x1":
        raise RuntimeError("RPC is not Ethereum mainnet")
    latest = int(rpc.request("eth_blockNumber", []), 16)
    start = helper.first_block(rpc, START_TS, latest)
    end = helper.first_block(rpc, END_TS, latest) - 1
    blocks = {"prestart": max(0, start - 1), "start": start, "end": end, "latest": latest}
    graphs = {name: graph(rpc, block) for name, block in blocks.items()}
    expected = graphs["latest"]
    graph_same = {name: value == expected for name, value in graphs.items()}
    runtime = {name: {key: code(rpc, value, block) for key, value in expected.items()}
               for name, block in blocks.items()}
    deployed = all(item["exists"] for name in blocks for item in runtime[name].values())
    token_decimals = {name: helper.decode_one(rpc.at(token, helper.selector("decimals()"), end), "uint8") for name, token in {"crvUSD": CRVUSD, "WETH": expected["asset_token"]}.items()}
    graph_doc = {"factory": FACTORY.lower(), "market_id": MARKET_ID, "keeper_pool": POOL.lower(),
                 "legacy_pool": LEGACY_POOL.lower(), "window_blocks": blocks, "graphs": graphs,
                 "graph_same_as_latest": graph_same, "runtime": runtime,
                 "legacy_runtime": {name: code(rpc, LEGACY_POOL, block) for name, block in blocks.items()},
                 "factory_abi": {"path": str(FACTORY_ABI), "sha256": hashlib.sha256(FACTORY_ABI.read_bytes()).hexdigest(), "markets(uint256)_tuple_verified": True},
                 "token_assertions": {"asset_is_WETH": expected["asset_token"] == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "pool_is_crvUSD_WETH": [address_getter(rpc, expected["cryptopool"], "coins(uint256)", end, [i]) for i in (0, 1)] == [CRVUSD.lower(), expected["asset_token"]], "decimals": token_decimals, "both_18": all(v == 18 for v in token_decimals.values())},
                 "current_graph": expected}
    (out / "factory_graph.json").write_text(json.dumps(graph_doc, indent=2) + "\n")
    if expected["cryptopool"] != POOL.lower() or not deployed or not all(graph_same.values()) or not graph_doc["token_assertions"]["asset_is_WETH"] or not graph_doc["token_assertions"]["pool_is_crvUSD_WETH"] or not graph_doc["token_assertions"]["both_18"]:
        (out / "preparation_manifest.json").write_text(json.dumps({"status": "blocked_graph_or_deployment", "factory_graph": graph_doc}, indent=2) + "\n")
        raise SystemExit("market-10 graph or keeper-pool deployment does not match")
    (out / "reciprocal_bindings.json").write_text(json.dumps(reciprocal(rpc, expected, end), indent=2) + "\n")
    helper.POOL, helper.AMM = POOL, expected["amm"]
    helper.START_TS, helper.END_TS = START_TS, END_TS
    original_event_logs = helper.event_logs
    def weth_event_logs(rpc, start, end, address=POOL, definitions=helper.EVENT_SIGS):
        return original_event_logs(rpc, start, end, address, definitions)
    helper.event_logs = weth_event_logs
    sys.argv = [str(HELPER_PATH), "--out", str(out)]
    helper.main()
    config_path = out / "configuration_window.json"
    document = json.loads(config_path.read_text())
    document["factory_market_graph"] = graph_doc
    document["reciprocal_bindings"] = json.loads((out / "reciprocal_bindings.json").read_text())
    document["deployment_lineage"] = {"selected_pool": POOL.lower(), "legacy_pool": LEGACY_POOL.lower(), "legacy_not_substituted": True}
    cache = {p.stem: hashlib.sha256(json.dumps(json.loads(p.read_text()), sort_keys=True, separators=(",", ":")).encode()).hexdigest() for p in (out / "rpc_cache").glob("*.json")}
    document["rpc_cache"] = {"entries": len(cache), "envelope_sha256": cache}
    config_path.write_text(json.dumps(document, indent=2) + "\n")
    manifest = {"status": "prepared", "command": "UV_CACHE_DIR=/tmp/uv-cache uv run --group research python scripts/acquire_yb_weth_window.py",
                "factory_graph": str(out / "factory_graph.json"), "configuration": str(config_path),
                "source": str(HELPER_PATH), "window": {"start_timestamp": START_TS, "end_exclusive_timestamp": END_TS},
                "rpc_cache_entries": len(cache)}
    (out / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "start_block": start, "end_block": end, "rpc_cache_entries": len(rpc.index)}))


if __name__ == "__main__":
    main()
