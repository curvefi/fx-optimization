#!/usr/bin/env python3
"""report_lean slice heatmaps: write, run, view and rank the grids.

Every stage is a Don 0.1..3.5% (128) x RPF 0.20..0.50 (128) plane on the Half-Gap report_lean.hpp policy and the live
WETH pool, with real Pyth reports, the picky [B - 12 s, B + 3 s] report window (arb_report_offset 2003), a 1 s settle
delay, the active_2l YieldBasis actor and the report keeper; the remaining axes are the slices:
  top100_128x128          the 100 top slices of the 16 x 16 discovery grids (configs/autoresearch/top100_slices_20261018.json)
  ng_top13_128x128        seed 637315 and the 12 Nevergrad 12x12 basin winners (configs/autoresearch/nevergrad_top13_20261018.json)
  clean_live_2x2_128x128  round parameters: 2 EMA regimes x 2 relax times
The 2026-10-18 runs of these grids (8 policy parameters, the idle keeper, pool settings ma_time 600, step max 3% and
donation share cap 55%, evaluator report-lean-20261016) are archived with their configs.

  uv run --no-sync python scripts/yb_lean_study.py write [STAGE...]    write and parse configs/autoresearch/<rid>.toml
  uv run --no-sync python scripts/yb_lean_study.py run [STAGE...]      fxopt run into runs/<rid>, then the 4x4 heatmap PNG
  uv run --no-sync python scripts/yb_lean_study.py analyze [STAGE...]  rank the slices by earning area
  uv run --no-sync python scripts/yb_lean_study.py local STAGE ORDINAL [NAME=VALUE...]
      write configs/autoresearch/local-<stage>-<ordinal>.toml: that cell of the stage's run and its 3 x 3 Don x RPF
      neighbours on this machine, with any policy parameter (base_fee ... report_lifetime_s), donation_apy or
      reserved_profit_fraction set to VALUE; run it with `uv run fxopt run CONFIG --output runs/<same name>`
Without STAGE arguments every stage is used.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from fxopt.results import read_result_columns

OPT = Path(__file__).resolve().parents[1]
CONFIGS = OPT / "configs" / "autoresearch"
PYTH_FEED = "../../data/market/pyth/ethusd/reports-20250324-20261001.npz"
BLOCK_TAPE = "../../data/market/binance/spot/blocks/ETHUSDT-blocks-20250324-20261001.npz"
# The first host builds the evaluator and coordinates the run. blade-a4 since 2026-10-12: root's nix channel on
# blade-a3 was moved to 26.05 that day, and what is built there links a glibc the other blades lack; blade-a4 is still
# on the 23.05 channel that every blade can run.
HOSTS = ["blade-a4", "blade-a3", "blade-a5", "blade-a6", "blade-a7", "blade-a8", "blade-a10",
         "blade-b4", "blade-b5", "blade-b6", "blade-b7", "blade-b8", "blade-b9", "blade-b10"]
METRICS = [
    "pool_nav_vs_hold", "arb_pnl_coin0", "lp_fee_coin0", "lp_xcp_profit",
    "vp", "apy", "apy_net", "apy_net_gm60",
    "apy_net_gm60_worst", "trades", "n_rebalances", "avg_rel_price_diff",
    "max_7d_rel_price_diff", "max_rel_price_diff", "final_rel_price_diff", "total_notional_coin0",
    "donation_coin0_total", "supply_ratio", "tw_avg_pool_fee", "avg_imbalance",
    "early_stop_ts", "elapsed_ms", "vp_growth_apy", "donated_apy",
    "donation_burn_apy", "xcp_growth_apy", "vp_growth_less_burn_apy", "step_cost_apy",
    "step_gain_apy", "net_step_cost_apy", "fee_xcp_apy", "arb_take_apy",
    "fee_coin0_apy", "trades_per_year", "steps_per_year", "xcp_apy_gm",
    "xcp_apy_gm30", "xcp_apy_gm60", "xcp_apy_gm90", "xcp_apy_gm30_60_90",
    "xcp_apy_gm30_floor_share", "xcp_apy_gm60_floor_share", "xcp_apy_gm90_floor_share", "xcp_apy_gm30_worst",
    "xcp_apy_gm60_worst", "xcp_apy_gm90_worst", "xcp_apy_gm60_worst_end", "yb_apy",
    "yb_apy_gm", "yb_apy_gm30", "yb_apy_gm60", "yb_apy_gm90",
    "yb_apy_gm30_floor_share", "yb_apy_gm60_floor_share", "yb_apy_gm90_floor_share", "yb_apy_gm30_worst",
    "yb_apy_gm60_worst", "yb_apy_gm90_worst", "yb_apy_gm60_worst_end", "yb_market_gm60",
    "yb_external_growth_eth", "yb_releverage_trades", "apy_net_robust_90d", "arb_offered_report_trades",
    "arb_withheld_report_trades",
]
DON_128 = [round(0.001 + (0.035 - 0.001) * i / 127, 8) for i in range(128)]
RPF_128 = [round(0.20 + (0.50 - 0.20) * i / 127, 8) for i in range(128)]
LEAN_NAMES = [
    "base_fee", "capture", "fallback_fee", "relax_time_s",
    "fast_half_life_s", "slow_half_life_s", "trend_gate", "step_cap", "report_lifetime_s",
]
LEAN_BASE = {
    "base_fee": 0.003, "capture": 0.80, "fallback_fee": 0.015, "relax_time_s": 300.0,
    "fast_half_life_s": 3600.0, "slow_half_life_s": 10800.0, "trend_gate": 0.008,
    "step_cap": 0.00125, "report_lifetime_s": 5400.0,
}
# The evaluator of the 9-parameter report_lean with the report keeper (report-lean-20261016 replays the archived runs).
EVALUATOR = "report-lean-live-20261019"
CLEAN_2X2_SET = {"base_fee": 0.005, "capture": 0.675, "fallback_fee": 0.0175, "relax_time_s": 150.0, "fast_half_life_s": 4800.0, "slow_half_life_s": 9600.0, "trend_gate": 0.0070, "step_cap": 0.0050}
CLEAN_2X2_AXES = {
    "ma_regime": [
        {"policy_params.4": 4500.0, "policy_params.5": 9000.0, "policy_params.6": 0.0067},
        {"policy_params.4": 4800.0, "policy_params.5": 9600.0, "policy_params.6": 0.0070},
    ],
    "policy_params.3": [150.0, 300.0],
}


def slices(name: str) -> list[dict]:
    """The ordered slice records of a selection file: each record's update_dict is one linked-axis value."""
    return [s["update_dict"] for s in json.loads((CONFIGS / name).read_text())]


STAGES = {
    "top100_128x128": {
        "rid": "top100-128x128-don01to35-rpf20to50-1638400-20261019",
        "note": "Top 100 Slices High-Resolution 128x128 Don x RPF Grid on the live pool and keeper (100 slices = 1,638,400 pools on Half-Gap report_lean.hpp, relax=300s fixed; ma_time 865, step max 0.5%, donation share cap 10%, report lifetime 90 min, keeper hourly or on a fallback-fee report move): Don 0.1..3.5% (128) x RPF 0.20..0.50 (128) x 100 Top Slices across prod1b_1to2, prod1a_1to2, prod1a_30m, prod1a_20m, hm7, and hm6 (1..54 pure 1:2 EMAs, 55..100 pre-1:2 EMAs)",
        "set": {"base_fee": 0.005, "capture": 0.75, "fallback_fee": 0.020, "relax_time_s": 300.0, "fast_half_life_s": 4800.0, "slow_half_life_s": 9600.0, "trend_gate": 0.0070, "step_cap": 0.0045},
        "axes": lambda: {"top_slice": slices("top100_slices_20261018.json")},
    },
    "ng_top13_128x128": {
        "rid": "nevergrad-top13-128x128-don01to35-rpf20to50-212992-20261019",
        "note": "Nevergrad 1h Polished Top 12 Slices + Baseline Seed 637315 on 128x128 Don x RPF Grid on the live pool and keeper (13 slices = 212,992 pools on Half-Gap report_lean.hpp; ma_time 865, step max 0.5%, donation share cap 10%, report lifetime 90 min, keeper hourly or on a fallback-fee report move): Don 0.1..3.5% (128) x RPF 0.20..0.50 (128) x 13 Polished Slices (1=Seed 637315, 2..13=Top 12 Nevergrad 12x12-Block Polished Winners)",
        "set": {"base_fee": 0.005, "capture": 0.675, "fallback_fee": 0.0175, "relax_time_s": 300.0, "fast_half_life_s": 4800.0, "slow_half_life_s": 9600.0, "trend_gate": 0.0070, "step_cap": 0.0050},
        "axes": lambda: {"polished_slice": slices("nevergrad_top13_20261018.json")},
    },
    "clean_live_2x2_128x128": {
        "rid": "clean-live-2x2-ema-relax-128x128-65536-20261019",
        "note": "Clean Round-Parameter 2x2 Grid on the live pool and keeper (65,536 pools; ma_time 865, step max 0.5%, donation share cap 10%, report lifetime 90 min, keeper hourly or on a fallback-fee report move; bf=50bp, cap=0.675, fb=175bp, sc=50bp, yb=150bp): Don 0.1..3.5% (128) x RPF 0.20..0.50 (128) x 2 EMAs(75/150m@67bp, 80/160m@70bp) x 2 RelaxTime(150s, 300s)",
        "set": CLEAN_2X2_SET,
        "axes": lambda: CLEAN_2X2_AXES,
    },
}


def stage_axes(s: str) -> dict:
    return {"pool.donation_apy": DON_128, "pool.reserved_profit_fraction": RPF_128, **STAGES[s]["axes"]()}


def toml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, dict):
        return "{ " + ", ".join(f'"{k}" = {toml(x)}' for k, x in v.items()) + " }"
    return "[" + ", ".join(toml(x) for x in v) + "]"


def inline(d: dict) -> str:
    return "{ " + ", ".join(f"{k} = {inline(v) if isinstance(v, dict) else toml(v)}" for k, v in d.items()) + " }"


def cfg_of(s: str) -> Path:
    return CONFIGS / f"{STAGES[s]['rid']}.toml"


def run_of(s: str) -> Path:
    return OPT / "runs" / STAGES[s]["rid"]


def write_stage_cfg(s: str) -> Path:
    S = STAGES[s]
    p = [LEAN_BASE[k] for k in LEAN_NAMES]
    for k, v in S["set"].items():
        p[LEAN_NAMES.index(k)] = v
    pool = {  # the live WETH pool: raw ma_time 865 (a 600 s half-life), a 0.5% largest step, a 10% donation share cap
        "donation_apy": 0.025,
        "adjustment_step_max": 0.005,
        "ma_time": 865,
        "initial_donation_days": 0,
        "donation_frequency": 86400,
        "donation_coins_ratio": 0.5,
        "reserved_profit_fraction": 0.5,
        "donation_shares_max_ratio": 0.10,
        "costs": {"entry_edge_bps": 1.5, "arb_fee_bps": 0.0, "gas_coin0": 0.0, "report_coin0": 0.0},
        "run": {
            "arb_report_offset": 2003,  # [B - 12 s, B + 3 s]: the arbitrageur picks the most profitable report
            "keeper_report": 3,         # the latest report, hourly (dustswap_freq_s) or once it moved the fallback fee
            "arb_report_rate": 1.0,
            "arb_settle_offset_s": 1,
            "yb_releverage_fee": 0.015,
        },
    }
    ses = {
        "event_mode": "block",
        "event_cursor": "fast_skip",
        "arb_settle_offset_s": 1,
        "start_time": 1742774400,
        "end_time": 1790812800,
        "dustswap_freq_s": 3600,
        "yb_releverage_fee": 0.013,
        "yb_cash_multiplier": 3.0,
        "yb_execution_bps": 5.0,
        "yb_min_net_profit_coin0": 1.0,
        "early_stop_max_7d_rel_price_diff": 0.30,
    }
    L = [
        f"# {S['note']}",
        "[run]",
        f'id = "{S["rid"]}"',
        f'evaluator = "../../../curve-fx-arb-harness/build/{EVALUATOR}/arb_evaluator_f64"',
        'template = "../templates/ethusd-yb-weth-live-20250324.json"',
        "batch_size = 1024",
        "workers = 128",
        f"metric_fields = {toml(METRICS)}",
        "",
        "[placement]",
        f"hosts = {toml(HOSTS)}",
        "numa_nodes = [0, 1]",
        "",
        "[session]",
        *[f"{k} = {toml(v)}" for k, v in ses.items()],
        "",
        "[scenario]",
        'id = "ethusd-blocks"',
        f'block_tape = "{BLOCK_TAPE}"',
        f'price_feed = "{PYTH_FEED}"',
        'yb_mode = "active_2l"',
        "",
        "[compiled_policy]",
        'id = "report_lean"',
        'header = "../../../twocrypto-cpp/include/pools/twocrypto_fx/policies/report_lean.hpp"',
        f"parameter_names = {toml(LEAN_NAMES)}",
        "",
        "[candidate.defaults]",
        f"policy_params = {toml(p)}",
        f"pool = {inline(pool)}",
        "",
        "[candidate.axes]",
    ]
    for k, v in stage_axes(s).items():
        if v and isinstance(v[0], dict):  # linked axis: each value sets several fields together
            items = ", ".join("{ " + ", ".join(f'"{dk}" = {toml(dv)}' for dk, dv in d.items()) + " }" for d in v)
            L.append(f'"{k}" = [ {items} ]')
        else:
            L.append(f'"{k}" = {toml(v)}')
    path = cfg_of(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return path


def write_local_cfg(s: str, ordinal: int, assignments: list[str]) -> Path:
    """A config for this machine (no placement): cell `ordinal` of stage s's run, with the assignments applied, at the
    centre of a 3 x 3 Don x RPF neighbourhood one grid step apart."""
    cell = read_result_columns(run_of(s), metrics=["yb_apy_gm60"]).candidate_at(ordinal)
    params = list(cell.policy_params)
    pool = json.loads(json.dumps(dict(cell.pool_overrides)))
    for assignment in assignments:
        name, _, value = assignment.partition("=")
        if name in LEAN_NAMES:
            params[LEAN_NAMES.index(name)] = float(value)
        elif name in ("donation_apy", "reserved_profit_fraction"):
            pool[name] = float(value)
        else:
            raise SystemExit(f"unknown NAME {name!r}: one of {', '.join(LEAN_NAMES)}, donation_apy, reserved_profit_fraction")
    don, rpf = pool["donation_apy"], pool["reserved_profit_fraction"]
    axes = {"pool.donation_apy": [round(don + k * (DON_128[1] - DON_128[0]), 8) for k in (-1, 0, 1)],
            "pool.reserved_profit_fraction": [round(rpf + k * (RPF_128[1] - RPF_128[0]), 8) for k in (-1, 0, 1)]}
    rid = f"local-{s}-{ordinal}"
    lines, section = [], ""
    for line in cfg_of(s).read_text().splitlines():
        if line.startswith("["):
            section = line
        if section in ("[placement]", "[candidate.axes]"):
            continue
        if line.startswith("# "):
            line = f"# {rid}: {s} ordinal {ordinal}, its 3 x 3 Don x RPF neighbours, this machine" + "".join(f"; {a}" for a in assignments)
        elif section == "[run]" and line.startswith("id = "):
            line = f'id = "{rid}"'
        elif line.startswith("batch_size = "):
            line = "batch_size = 9"
        elif line.startswith("workers = "):
            line = f"workers = {min(9, os.cpu_count() or 1)}"
        elif line.startswith("policy_params = "):
            line = f"policy_params = {toml(params)}"
        elif line.startswith("pool = "):
            line = f"pool = {inline(pool)}"
        lines.append(line)
    path = CONFIGS / f"{rid}.toml"
    path.write_text("\n".join(lines).rstrip("\n") + "\n\n[candidate.axes]\n" + "".join(f'"{k}" = {toml(v)}\n' for k, v in axes.items()))
    return path


def validate_stage(s: str) -> tuple[Path, int]:
    from fxopt.run import RunConfig
    path = write_stage_cfg(s)
    c = RunConfig.from_toml(path)
    m = len(c.candidate.grid())
    print(f"{c.run_id}: {m} pools validated, yb_mode={c.scenario['yb_mode']}, metrics={len(c.metric_fields)}")
    return path, m


def run_stage(s: str) -> int:
    path, _ = validate_stage(s)
    out_dir = run_of(s)
    if not (out_dir / "run.json").exists() or not (out_dir / "results.npz").exists():
        cmd = [
            "env", "-u", "VIRTUAL_ENV", "UV_CACHE_DIR=/tmp/uv-cache",
            "uv", "run", "--no-sync", "fxopt", "run",
            str(path.relative_to(OPT)),
            "--output", str(out_dir.relative_to(OPT)),
        ]
        print("Running:", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=OPT)
        if res.returncode != 0:
            return res.returncode
    hm_cmd = [
        "bash", "scripts/heatmap-pool-yb.sh",
        str(path.relative_to(OPT)),
        str(out_dir.relative_to(OPT)),
        "--output", str((out_dir / "heatmap.png").relative_to(OPT)),
        "--no-show",
    ]
    subprocess.run(hm_cmd, cwd=OPT, check=False)
    return 0


def analyze_stage(stage_key: str, top_k: int = 12) -> dict:
    S = STAGES[stage_key]
    out_dir = run_of(stage_key)
    axes = stage_axes(stage_key)
    keys = list(axes.keys())
    nx = len(axes["pool.donation_apy"])
    ny = len(axes["pool.reserved_profit_fraction"])
    slider_keys = [k for k in keys if k not in ("pool.donation_apy", "pool.reserved_profit_fraction")]
    slider_shapes = [len(axes[k]) for k in slider_keys]
    n_sliders = int(np.prod(slider_shapes))

    cols = (
        "yb_apy", "yb_apy_gm30", "yb_apy_gm60", "yb_apy_gm90",
        "max_7d_rel_price_diff", "final_rel_price_diff", "early_stop_ts",
        "arb_offered_report_trades", "arb_withheld_report_trades", "trades",
    )
    r = read_result_columns(out_dir, metrics=cols)
    order = sorted(r.metadata["axes"])  # stored rows follow the canonical (sorted-axis) ordinal order
    raw_sh = tuple(r.metadata["shape"])
    perm = [order.index(k) for k in keys]

    M = {}
    for k, v in r.metrics.items():
        M[k] = np.transpose(np.asarray(v, dtype=float).reshape(raw_sh), perm).reshape(nx, ny, n_sliders)

    valid = (M["early_stop_ts"] == 0) & np.isfinite(M["yb_apy_gm90"]) & (M["max_7d_rel_price_diff"] < 0.15) & (M["final_rel_price_diff"] < 0.01)
    g30 = np.where(valid, M["yb_apy_gm30"] * 100.0, -10.0)
    g60 = np.where(valid, M["yb_apy_gm60"] * 100.0, -10.0)
    g90 = np.where(valid, M["yb_apy_gm90"] * 100.0, -10.0)

    # Earning masks on each Don x RPF slice plane
    m90_1 = valid & (g90 >= 1.0)
    m60_05 = valid & (g60 >= 0.5) & (g90 >= 1.0)
    m30_0 = valid & (g30 >= 0.0) & (g60 >= 0.5) & (g90 >= 1.0)
    m30_1 = valid & (g30 >= 1.0) & (g60 >= 1.0) & (g90 >= 1.0)

    print(f"\n==================== {S['rid']} ({nx}x{ny} = {nx*ny} cells/slice, {n_sliders} slices) ====================")
    print(f"Total valid cells (<15% 7d diff): {int(valid.sum())}/{r.row_count} ({100*valid.mean():.1f}%)")
    print(f"Total cells with gm90>=1%: {int(m90_1.sum())} | gm60>=0.5%&gm90>=1%: {int(m60_05.sum())} | gm30>=0%&gm60>=0.5%&gm90>=1%: {int(m30_0.sum())} | gm30&60&90>=1%: {int(m30_1.sum())}")

    for sk_i, sk in enumerate(slider_keys):
        vals = axes[sk]
        c_valid = valid.sum(axis=(0, 1)).reshape(slider_shapes)
        c_90 = m90_1.sum(axis=(0, 1)).reshape(slider_shapes)
        c_60 = m60_05.sum(axis=(0, 1)).reshape(slider_shapes)
        c_30_0 = m30_0.sum(axis=(0, 1)).reshape(slider_shapes)
        c_30_1 = m30_1.sum(axis=(0, 1)).reshape(slider_shapes)
        print(f"\n--- Marginal Effect of {sk} (out of {nx*ny} cells/slice) ---")
        for vi, v in enumerate(vals):
            sv = np.take(c_valid, vi, axis=sk_i)
            s90 = np.take(c_90, vi, axis=sk_i)
            s60 = np.take(c_60, vi, axis=sk_i)
            s30_0 = np.take(c_30_0, vi, axis=sk_i)
            s30_1 = np.take(c_30_1, vi, axis=sk_i)
            print(
                f"  {sk} = {v!s:>8}: valid={sv.mean():5.1f} | "
                f"avg (gm30>=1 / gm30>=0 / gm60>=0.5 / gm90>=1) = ({s30_1.mean():5.1f} / {s30_0.mean():5.1f} / {s60.mean():5.1f} / {s90.mean():5.1f}) | "
                f"max = ({int(s30_1.max()):3d} / {int(s30_0.max()):3d} / {int(s60.max()):3d} / {int(s90.max()):3d})"
            )

    slice_rows = []
    for s_idx in range(n_sliders):
        v_s = valid[:, :, s_idx]
        n_v = int(v_s.sum())
        if n_v == 0:
            continue
        g30_v = M["yb_apy_gm30"][:, :, s_idx][v_s] * 100.0
        g60_v = M["yb_apy_gm60"][:, :, s_idx][v_s] * 100.0
        g90_v = M["yb_apy_gm90"][:, :, s_idx][v_s] * 100.0
        w_mean = float(0.2 * np.mean(g30_v) + 0.3 * np.mean(g60_v) + 0.5 * np.mean(g90_v))
        w_med = float(0.2 * np.median(g30_v) + 0.3 * np.median(g60_v) + 0.5 * np.median(g90_v))
        c90 = int(m90_1[:, :, s_idx].sum())
        c60 = int(m60_05[:, :, s_idx].sum())
        c30_0 = int(m30_0[:, :, s_idx].sum())
        c30_1 = int(m30_1[:, :, s_idx].sum())
        nbr_60 = 0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if m60_05[i - 1:i + 2, j - 1:j + 2, s_idx].all():
                    nbr_60 += 1
        coords = np.unravel_index(s_idx, slider_shapes)
        s_dict = {k: axes[k][ci] for k, ci in zip(slider_keys, coords)}
        slice_rows.append((c60, c30_0, c30_1, c90, nbr_60, w_med, w_mean, n_v, s_idx, s_dict))

    slice_rows.sort(reverse=True, key=lambda row: row[:9])
    print(f"\n--- Top {top_k} Slices by Earning Area (gm60>=0.5%&gm90>=1%, gm30>=0%, gm30>=1%) & Weighted Horizon Median ---")
    for rank, (c60, c30_0, c30_1, c90, nbr_60, w_med, w_mean, n_v, s_idx, s_dict) in enumerate(slice_rows[:top_k], 1):
        v_s = valid[:, :, s_idx]
        g30_v = M["yb_apy_gm30"][:, :, s_idx][v_s] * 100.0
        g60_v = M["yb_apy_gm60"][:, :, s_idx][v_s] * 100.0
        g90_v = M["yb_apy_gm90"][:, :, s_idx][v_s] * 100.0
        wh = M["arb_withheld_report_trades"][:, :, s_idx][v_s] / np.maximum(M["trades"][:, :, s_idx][v_s], 1.0) * 100.0
        print(
            f"#{rank:2d} s={s_idx:4d}: gm60>=.5&90>=1={c60:3d}/{nx*ny} (3x3={nbr_60:2d}), gm30>=0={c30_0:3d}, gm30>=1={c30_1:3d}, gm90>=1={c90:3d} (valid={n_v:3d}) | "
            f"w_med={w_med:+5.2f}%, w_mean={w_mean:+5.2f}% (med gm30/60/90={np.median(g30_v):+5.2f}/{np.median(g60_v):+5.2f}/{np.median(g90_v):+5.2f}%, wh={np.mean(wh):.1f}%) | {s_dict}"
        )
    return {"rid": S["rid"], "top_slices": slice_rows[:top_k]}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "local":
        if len(sys.argv) < 4 or sys.argv[2] not in STAGES or not sys.argv[3].isdigit():
            print(__doc__, file=sys.stderr)
            return 2
        path = write_local_cfg(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
        from fxopt.run import RunConfig
        RunConfig.from_toml(path)
        rel = path.relative_to(OPT)
        print(f"wrote {rel}; run it with:\n  uv run fxopt run {rel} --output runs/{path.stem}\n"
              f"  bash scripts/hm-yb.sh {rel} runs/{path.stem}")
        return 0
    stages = sys.argv[2:] or list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if cmd not in ("write", "run", "analyze") or unknown:
        print(__doc__, file=sys.stderr)
        return 2
    for st in stages:
        if cmd == "write":
            print("Validated", validate_stage(st)[0])
        elif cmd == "run":
            rc = run_stage(st)
            if rc != 0:
                return rc
        else:
            analyze_stage(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
