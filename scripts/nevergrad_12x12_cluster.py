#!/usr/bin/env python3
"""Full-cluster Nevergrad 12x12 (+-0.25% Don, +-0.025 RPF) block gm30 optimizer.

  uv sync --group nevergrad   # once: nevergrad is an optional dependency group
  uv run --no-sync python scripts/nevergrad_12x12_cluster.py --output-dir runs/<new-run> [--budget-seconds S] [--max-hosts N]

The output directory receives evaluations.jsonl and best_summary.json; a directory that already holds them is refused.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import nevergrad as ng
import numpy as np

from curve_fx_harness_client import EvaluatorClient
from fxopt.config import RunConfig
from fxopt.contract import Candidate
from fxopt.engine import EvaluatorSession
from fxopt.run import _evaluator_policy_contract, _execution_inputs, open_session_request


METRICS = (
    "yb_apy_gm30",
    "yb_apy_gm60",
    "yb_apy_gm90",
    "yb_apy_gm30_worst",
    "yb_apy_gm60_worst",
    "yb_apy_gm90_worst",
    "yb_apy",
    "apy_net",
    "apy_net_gm60",
    "apy_net_gm60_worst",
    "max_7d_rel_price_diff",
    "avg_rel_price_diff",
    "final_rel_price_diff",
    "early_stop_ts",
    "trades_per_year",
    "steps_per_year",
    "fee_xcp_apy",
    "net_step_cost_apy",
    "arb_take_apy",
)

# Seed = ordinal 637315 (slider 16/100: 80/160m@70bp, bf=50bp, cap=0.675, fb=175bp, relax=300s, sc=50bp, yb=150bp, Don=1.411811%, RPF=0.43858268)
SEED_637315 = {
    "base_fee": 0.0050,
    "capture": 0.6750,
    "fallback_fee": 0.0175,
    "relax_time_s": 300.0,
    "fast_half_life_s": 4800.0,
    "slow_ratio": 2.00,
    "trend_gate": 0.0070,
    "step_cap": 0.0050,
    "yb_releverage_fee": 0.0150,
    "don_center": 0.01411811,
    "rpf_center": 0.43858268,
}

# Warm seeds: all variations of SEED_637315 along its ridge so we polish 637315's basin directly
WARM_SEEDS = [
    SEED_637315,
    {**SEED_637315, "don_center": 0.0145, "rpf_center": 0.4340},
    {**SEED_637315, "don_center": 0.0130, "rpf_center": 0.4220},
    {**SEED_637315, "don_center": 0.0115, "rpf_center": 0.4100},
    {**SEED_637315, "don_center": 0.0100, "rpf_center": 0.3980},
    {**SEED_637315, "don_center": 0.0085, "rpf_center": 0.3860},
]

PARAM_BOUNDS_LOCAL_072 = {
    "base_fee": (0.0038, 0.0060),
    "capture": (0.5800, 0.7200),
    "fallback_fee": (0.0125, 0.0225),
    "relax_time_s": (120.0, 600.0),
    "fast_half_life_s": (3000.0, 7200.0),
    "slow_ratio": (1.40, 2.50),
    "trend_gate": (0.0035, 0.0105),
    "step_cap": (0.0030, 0.0075),
    "yb_releverage_fee": (0.0100, 0.0200),
    "don_center": (0.0050, 0.0200),
    "rpf_center": (0.3400, 0.4650),
}

PARAM_BOUNDS_FULL_075 = {
    "base_fee": (0.0030, 0.0065),
    "capture": (0.5500, 0.7500),
    "fallback_fee": (0.0110, 0.0240),
    "relax_time_s": (90.0, 900.0),
    "fast_half_life_s": (2400.0, 9000.0),
    "slow_ratio": (1.25, 2.80),
    "trend_gate": (0.0020, 0.0125),
    "step_cap": (0.0025, 0.0085),
    "yb_releverage_fee": (0.0090, 0.0210),
    "don_center": (0.0035, 0.0220),
    "rpf_center": (0.3100, 0.4700),
}

PARAM_BOUNDS_FULL_082 = {
    **PARAM_BOUNDS_FULL_075,
    "capture": (0.5500, 0.8200),
}


def make_parametrization(
    bounds_map: dict[str, tuple[float, float]],
    center: dict[str, float] | None = None,
    trust_scale: float = 1.0,
    seed: int = 20261018,
) -> ng.p.Dict:
    init_map = center or SEED_637315
    scalars = {}
    for k, (lo, hi) in bounds_map.items():
        c = float(init_map.get(k, SEED_637315[k]))
        c = min(hi, max(lo, c))
        if trust_scale < 1.0:
            half = 0.5 * (hi - lo) * trust_scale
            sub_lo = max(lo, c - half)
            sub_hi = min(hi, c + half)
        else:
            sub_lo, sub_hi = lo, hi
        scalars[k] = ng.p.Scalar(init=c, lower=sub_lo, upper=sub_hi)
    p = ng.p.Dict(**scalars)
    p.random_state.seed(seed)
    return p


def build_12x12_candidates(
    eval_id: int,
    params: dict[str, float],
    base_pool: dict[str, Any],
    *,
    grid_n: int = 12,
    don_radius: float = 0.0025,
    rpf_radius: float = 0.025,
) -> tuple[list[Candidate], dict[str, float], np.ndarray, np.ndarray]:
    bf = round(float(params["base_fee"]), 6)
    cap = round(float(params["capture"]), 6)
    fb = round(float(params["fallback_fee"]), 6)
    relax = round(float(params["relax_time_s"]), 3)
    fast_s = round(float(params["fast_half_life_s"]), 2)
    slow_ratio = round(float(params["slow_ratio"]), 5)
    slow_s = round(fast_s * slow_ratio, 2)
    tg = round(float(params["trend_gate"]), 6)
    sc = round(float(params["step_cap"]), 6)
    yb_fee = round(float(params["yb_releverage_fee"]), 6)
    don0 = round(float(params["don_center"]), 8)
    rpf0 = round(float(params["rpf_center"]), 8)

    pp = (bf, cap, fb, relax, fast_s, slow_s, tg, sc)
    dons = np.linspace(max(0.0005, don0 - don_radius), don0 + don_radius, grid_n)
    rpfs = np.linspace(max(0.15, rpf0 - rpf_radius), min(0.55, rpf0 + rpf_radius), grid_n)

    cands: list[Candidate] = []
    idx = 0
    for d in dons:
        for r_val in rpfs:
            po = {
                **base_pool,
                "donation_apy": round(float(d), 8),
                "reserved_profit_fraction": round(float(r_val), 8),
                "run": {**base_pool["run"], "yb_releverage_fee": yb_fee},
            }
            cands.append(Candidate(f"e{eval_id:06d}_{idx:03d}", pp, po))
            idx += 1

    clean_params = {
        "base_fee": bf,
        "capture": cap,
        "fallback_fee": fb,
        "relax_time_s": relax,
        "fast_half_life_s": fast_s,
        "slow_half_life_s": slow_s,
        "slow_ratio": slow_ratio,
        "trend_gate": tg,
        "step_cap": sc,
        "yb_releverage_fee": yb_fee,
        "don_center": don0,
        "rpf_center": rpf0,
    }
    return cands, clean_params, dons, rpfs


def score_12x12_block(
    results: list[Any],
    clean_params: dict[str, float],
    dons: np.ndarray,
    rpfs: np.ndarray,
    grid_n: int = 12,
) -> dict[str, Any]:
    sh = (grid_n, grid_n)
    g30 = np.array([float(r.metrics["yb_apy_gm30"]) * 100.0 for r in results], dtype=np.float64).reshape(sh)
    g60 = np.array([float(r.metrics["yb_apy_gm60"]) * 100.0 for r in results], dtype=np.float64).reshape(sh)
    g90 = np.array([float(r.metrics["yb_apy_gm90"]) * 100.0 for r in results], dtype=np.float64).reshape(sh)
    w60 = np.array([float(r.metrics["yb_apy_gm60_worst"]) * 100.0 for r in results], dtype=np.float64).reshape(sh)
    g7d = np.array([float(r.metrics["max_7d_rel_price_diff"]) * 100.0 for r in results], dtype=np.float64).reshape(sh)
    fdiff = np.array([float(r.metrics["final_rel_price_diff"]) * 100.0 for r in results], dtype=np.float64).reshape(sh)
    stop = np.array([float(r.metrics["early_stop_ts"]) for r in results], dtype=np.float64).reshape(sh)
    tpy = np.array([float(r.metrics["trades_per_year"]) for r in results], dtype=np.float64).reshape(sh)

    base_ok = (stop == 0) & (fdiff <= 1.0) & np.isfinite(g7d) & np.isfinite(g30)
    v10 = base_ok & (g7d <= 10.0)
    v13 = base_ok & (g7d <= 13.0)
    v75 = base_ok & (g7d <= 7.5)

    pos_g30 = np.where(np.isfinite(g30) & (g30 > 0.0), g30, 0.0)
    pos_g60 = np.where(np.isfinite(g60) & (g60 > 0.0), g60, 0.0)

    g30_strict10 = np.where(v10, pos_g30, 0.0)
    g60_strict10 = np.where(v10, pos_g60, 0.0)
    g30_strict75 = np.where(v75, pos_g30, 0.0)

    # Smooth hinge between 10% and 13.5% max_7d to guide optimizer into <10%
    hinge = np.clip(1.0 - np.maximum(0.0, g7d - 10.0) / 3.5, 0.0, 1.0)
    g30_soft = np.where(base_ok, pos_g30 * hinge, 0.0)

    block_mean_gm30_10 = float(g30_strict10.mean())
    block_min_gm30_10 = float(g30_strict10.min())
    block_max_gm30_10 = float(g30_strict10.max())
    block_mean_gm60_10 = float(g60_strict10.mean())
    block_mean_gm30_75 = float(g30_strict75.mean())
    block_mean_gm30_soft = float(g30_soft.mean())

    # Inner 6x6 core (indices 3..8 in 0..11) around (don_center, rpf_center)
    c_lo = grid_n // 4
    c_hi = grid_n - c_lo
    core_g30_10 = g30_strict10[c_lo:c_hi, c_lo:c_hi]
    core_w60 = np.where(v10[c_lo:c_hi, c_lo:c_hi], w60[c_lo:c_hi, c_lo:c_hi], -10.0)
    core_mean_gm30_10 = float(core_g30_10.mean())
    core_min_gm30_10 = float(core_g30_10.min())
    core_min_w60_10 = float(core_w60.min())

    # Center 2x2 around exact (don_center, rpf_center)
    m0 = grid_n // 2 - 1
    m1 = grid_n // 2 + 1
    center_2x2_gm30_10 = float(g30_strict10[m0:m1, m0:m1].mean())
    center_2x2_min_gm30_10 = float(g30_strict10[m0:m1, m0:m1].min())
    center_2x2_gm60_10 = float(g60_strict10[m0:m1, m0:m1].mean())
    center_2x2_w60 = float(w60[m0:m1, m0:m1].mean())
    center_2x2_pdif = float(g7d[m0:m1, m0:m1].max())
    center_2x2_tpy = float(tpy[m0:m1, m0:m1].mean())

    # Primary objective: maximize 12x12 (+-0.25% Don, +-0.025 RPF) block mean gm30 (<10% pdif),
    # plus inner 6x6 core mean & min floor so the center is rock-solid, plus soft gradient
    objective = (
        1.00 * block_mean_gm30_10
        + 0.20 * core_mean_gm30_10
        + 0.15 * core_min_gm30_10
        + 0.10 * center_2x2_min_gm30_10
        + 0.08 * block_mean_gm30_soft
    )

    n_cells = grid_n * grid_n
    return {
        "objective": round(float(objective), 6),
        "block_mean_gm30_10": round(block_mean_gm30_10, 4),
        "block_min_gm30_10": round(block_min_gm30_10, 4),
        "block_max_gm30_10": round(block_max_gm30_10, 4),
        "block_mean_gm60_10": round(block_mean_gm60_10, 4),
        "block_mean_gm30_75": round(block_mean_gm30_75, 4),
        "block_mean_gm30_soft": round(block_mean_gm30_soft, 4),
        "core_6x6_mean_gm30_10": round(core_mean_gm30_10, 4),
        "core_6x6_min_gm30_10": round(core_min_gm30_10, 4),
        "core_6x6_min_w60_10": round(core_min_w60_10, 4),
        "center_2x2_gm30_10": round(center_2x2_gm30_10, 4),
        "center_2x2_min_gm30_10": round(center_2x2_min_gm30_10, 4),
        "center_2x2_gm60_10": round(center_2x2_gm60_10, 4),
        "center_2x2_w60": round(center_2x2_w60, 4),
        "center_2x2_pdif": round(center_2x2_pdif, 4),
        "center_2x2_trades_yr": round(center_2x2_tpy, 1),
        "valid_10_count": int(v10.sum()),
        "valid_13_count": int(v13.sum()),
        "valid_75_count": int(v75.sum()),
        "c30_ge_025_10": int((v10 & (pos_g30 >= 0.25)).sum()),
        "c30_ge_100_10": int((v10 & (pos_g30 >= 1.00)).sum()),
        "c30_ge_300_10": int((v10 & (pos_g30 >= 3.00)).sum()),
        "c30_ge_500_10": int((v10 & (pos_g30 >= 5.00)).sum()),
        "w60_pos_10": int((v10 & (pos_g30 >= 1.00) & (w60 > 0.0)).sum()),
        "n_cells": n_cells,
        "params": clean_params,
        "don_range": [round(float(dons[0]) * 100.0, 3), round(float(dons[-1]) * 100.0, 3)],
        "rpf_range": [round(float(rpfs[0]), 4), round(float(rpfs[-1]), 4)],
        "grid_gm30": np.round(pos_g30, 2).tolist(),
        "grid_gm60": np.round(pos_g60, 2).tolist(),
        "grid_w60": np.round(w60, 2).tolist(),
        "grid_pdif": np.round(g7d, 2).tolist(),
    }


def make_numa_factory(
    host: str,
    node: int | None,
    workers: int,
    evaluator_path: str,
    contract: dict[str, Any],
):
    argv = [
        "ssh",
        "-o", "ControlMaster=no",
        "-o", "ControlPath=none",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
        "--",
        host,
        "env", "-u", "LD_LIBRARY_PATH",
    ]
    if node is not None:
        argv.extend(["numactl", f"--cpunodebind={node}", f"--membind={node}"])
    argv.extend([evaluator_path, "serve", "--workers", str(workers)])

    def create() -> EvaluatorClient:
        return EvaluatorClient(
            executable_path=evaluator_path,
            launch_argv=argv,
            timeout=600.0,
            **{f"expected_{k}": v for k, v in contract.items()},
        )

    return create


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/autoresearch/top100-128x128-don01to35-rpf20to50-1638400-20261018.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=3600.0)
    parser.add_argument("--grid-n", type=int, default=12)
    parser.add_argument("--don-radius", type=float, default=0.0025)
    parser.add_argument("--rpf-radius", type=float, default=0.025)
    parser.add_argument("--max-hosts", type=int, default=14)
    args = parser.parse_args()

    cfg = RunConfig.from_toml(args.config)
    inputs = _execution_inputs(cfg, remote=True)
    open_req = open_session_request(cfg, remote=True)
    contract = _evaluator_policy_contract(cfg)
    base_pool = dict(cfg.candidate.defaults["pool"])

    hosts = list(cfg.hosts[: args.max_hosts])
    nodes = cfg.numa_nodes or (None,)
    workers_per_node = max(1, cfg.workers // len(nodes))

    out_dir = args.output_dir.resolve()
    jsonl_path = out_dir / "evaluations.jsonl"
    summary_path = out_dir / "best_summary.json"
    if jsonl_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds a Nevergrad run; choose a new --output-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    lane_specs = [
        (f"{host}:numa{node}" if node is not None else host, host, node)
        for host in hosts
        for node in nodes
    ]
    print(
        f"[ng-12x12] Opening {len(lane_specs)} persistent NUMA evaluator lanes across {len(hosts)} blades "
        f"({workers_per_node} threads/lane = {len(lane_specs)*workers_per_node} total threads)...",
        flush=True,
    )

    t_start = time.monotonic()

    def open_lane(spec: tuple[str, str, int | None]) -> tuple[str, EvaluatorSession]:
        name, host, node = spec
        sess = EvaluatorSession(
            make_numa_factory(host, node, workers_per_node, inputs["evaluator"], contract),
            session_id=f"ng12x12-{name.replace(':', '-')}",
            open_session=open_req,
            metric_fields=METRICS,
        )
        sess.start()
        return name, sess

    active_lanes: list[tuple[str, EvaluatorSession, tuple[str, str, int | None]]] = []
    with ThreadPoolExecutor(max_workers=len(lane_specs)) as opener:
        futs = {opener.submit(open_lane, spec): spec for spec in lane_specs}
        for fut in futs:
            spec = futs[fut]
            try:
                name, sess = fut.result()
                active_lanes.append((name, sess, spec))
            except Exception as exc:
                print(f"[ng-12x12] WARNING: lane {spec[0]} failed to open: {exc}", flush=True)

    if not active_lanes:
        raise RuntimeError("No cluster evaluator lanes could be opened")

    n_workers = len(active_lanes)
    print(
        f"[ng-12x12] {n_workers}/{len(lane_specs)} lanes ready in {time.monotonic() - t_start:.1f}s. "
        f"Starting {args.budget_seconds:.0f}s optimization of {args.grid_n}x{args.grid_n} "
        f"(+-{args.don_radius*100:.2f}% Don, +-{args.rpf_radius:.3f} RPF) block gm30...",
        flush=True,
    )

    # 4 complementary Nevergrad demes all initialized at SEED_637315:
    # 0: local_072_de   (35%): TwoPointsDE around 637315 with capture <= 0.72
    # 1: local_072_cma  (25%): DiagonalCMA trust-region around best low-capture candidate (re-centered every 1200 evals)
    # 2: full_075_de    (25%): Full-range TwoPointsDE with capture <= 0.75
    # 3: full_082_de    (15%): Full-range TwoPointsDE with capture <= 0.82
    optimizers: dict[str, Any] = {
        "local_072_de": ng.optimizers.TwoPointsDE(
            parametrization=make_parametrization(PARAM_BOUNDS_LOCAL_072, SEED_637315, 1.0, 20261018),
            budget=60000,
            num_workers=n_workers,
        ),
        "local_072_cma": ng.optimizers.DiagonalCMA(
            parametrization=make_parametrization(PARAM_BOUNDS_LOCAL_072, SEED_637315, 0.6, 20261019),
            budget=60000,
            num_workers=n_workers,
        ),
        "full_075_de": ng.optimizers.TwoPointsDE(
            parametrization=make_parametrization(PARAM_BOUNDS_FULL_075, SEED_637315, 1.0, 20261020),
            budget=60000,
            num_workers=n_workers,
        ),
        "full_082_de": ng.optimizers.TwoPointsDE(
            parametrization=make_parametrization(PARAM_BOUNDS_FULL_082, SEED_637315, 1.0, 20261021),
            budget=60000,
            num_workers=n_workers,
        ),
    }
    cma_generation = 0

    lock = threading.Lock()
    eval_counter = 0
    completed_evals = 0
    total_pools = 0
    stop_event = threading.Event()
    opt_started = time.monotonic()
    deadline = opt_started + args.budget_seconds

    best_overall: dict[str, Any] | None = None
    best_lowcap_072: dict[str, Any] | None = None
    best_lowcap_075: dict[str, Any] | None = None
    seed_record: dict[str, Any] | None = None
    top_records: list[dict[str, Any]] = []
    top_lowcap_records: list[dict[str, Any]] = []

    def next_candidate_task() -> tuple[int, str, Any, dict[str, float]]:
        nonlocal eval_counter, cma_generation
        with lock:
            eval_id = eval_counter
            eval_counter += 1
            if eval_id < len(WARM_SEEDS):
                seed_dict = dict(WARM_SEEDS[eval_id])
                cand_obj = optimizers["local_072_de"].parametrization.spawn_child(new_value=seed_dict)
                return eval_id, f"warm_seed_{eval_id}", cand_obj, seed_dict

            # Periodically re-center local_072_cma at the best low-capture candidate with tightened radius
            if eval_id > 0 and eval_id % 1200 == 0 and best_lowcap_072 is not None:
                cma_generation += 1
                center_p = dict(best_lowcap_072["params"])
                center_p.pop("slow_half_life_s", None)
                scale = max(0.20, 0.55 * (0.85 ** cma_generation))
                optimizers["local_072_cma"] = ng.optimizers.DiagonalCMA(
                    parametrization=make_parametrization(
                        PARAM_BOUNDS_LOCAL_072, center_p, scale, 20261100 + cma_generation
                    ),
                    budget=60000,
                    num_workers=n_workers,
                )
                print(
                    f"[ng-12x12] Re-centered local_072_cma (gen #{cma_generation}, scale={scale:.2f}) "
                    f"at best_lowcap_072 (obj={best_lowcap_072['objective']:.4f}, "
                    f"mean_gm30={best_lowcap_072['block_mean_gm30_10']:.3f}%)",
                    flush=True,
                )

            r_mod = eval_id % 20
            if r_mod < 7:
                opt_name = "local_072_de"
            elif r_mod < 12:
                opt_name = "local_072_cma"
            elif r_mod < 17:
                opt_name = "full_075_de"
            else:
                opt_name = "full_082_de"
            opt = optimizers[opt_name]
            cand_obj = opt.ask()
            return eval_id, opt_name, cand_obj, dict(cand_obj.value)

    def record_result(
        eval_id: int,
        opt_name: str,
        cand_obj: Any,
        scored: dict[str, Any],
        elapsed_eval: float,
        lane_name: str,
    ) -> None:
        nonlocal completed_evals, total_pools, best_overall, best_lowcap_072, best_lowcap_075, seed_record, top_records, top_lowcap_records
        loss = -float(scored["objective"])
        with lock:
            try:
                if opt_name.startswith("warm_seed_"):
                    for oname in ("local_072_de", "full_075_de", "full_082_de"):
                        c_child = optimizers[oname].parametrization.spawn_child(
                            new_value={k: v for k, v in scored["params"].items() if k != "slow_half_life_s"}
                        )
                        optimizers[oname].tell(c_child, loss)
                elif opt_name in optimizers:
                    optimizers[opt_name].tell(cand_obj, loss)
            except Exception:
                pass

            completed_evals += 1
            total_pools += scored["n_cells"]
            wall_s = time.monotonic() - opt_started

            entry = {
                "eval_id": eval_id,
                "optimizer": opt_name,
                "lane": lane_name,
                "eval_s": round(elapsed_eval, 3),
                "wall_s": round(wall_s, 2),
                **scored,
            }
            compact_entry = {k: v for k, v in entry.items() if not k.startswith("grid_")}
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(compact_entry) + "\n")

            if eval_id == 0:
                seed_record = entry
                print(
                    f"[ng-12x12] BASELINE SEED (ordinal=637315, eval #0): "
                    f"obj={entry['objective']:.4f} | 12x12 mean_gm30(<10%)={entry['block_mean_gm30_10']:.3f}% "
                    f"(min={entry['block_min_gm30_10']:.2f}%, max={entry['block_max_gm30_10']:.2f}%) | "
                    f"core_6x6 mean/min=({entry['core_6x6_mean_gm30_10']:.2f}%/{entry['core_6x6_min_gm30_10']:.2f}%) | "
                    f"valid(<10%)= {entry['valid_10_count']}/144, c30>=1%={entry['c30_ge_100_10']}/144",
                    flush=True,
                )

            cap_val = float(entry["params"]["capture"])
            is_new_best = False
            tag = ""
            if best_overall is None or entry["objective"] > best_overall["objective"]:
                best_overall = entry
                is_new_best = True
                tag = "OVERALL"
            if cap_val <= 0.7501 and (best_lowcap_075 is None or entry["objective"] > best_lowcap_075["objective"]):
                best_lowcap_075 = entry
                is_new_best = True
                tag = tag or "CAP<=0.75"
            if cap_val <= 0.7201 and (best_lowcap_072 is None or entry["objective"] > best_lowcap_072["objective"]):
                best_lowcap_072 = entry
                is_new_best = True
                tag = tag or "CAP<=0.72"

            if is_new_best:
                p = entry["params"]
                print(
                    f"[ng-12x12] ★ NEW BEST [{tag}] (eval #{eval_id}, {opt_name}, {wall_s:.0f}s): "
                    f"obj={entry['objective']:.4f} | 12x12 mean_gm30(<10%)={entry['block_mean_gm30_10']:.3f}% "
                    f"(min={entry['block_min_gm30_10']:.2f}%, max={entry['block_max_gm30_10']:.2f}%, gm60={entry['block_mean_gm60_10']:.2f}%) | "
                    f"core_6x6 mean/min=({entry['core_6x6_mean_gm30_10']:.2f}%/{entry['core_6x6_min_gm30_10']:.2f}%, w60_min={entry['core_6x6_min_w60_10']:+.2f}%) | "
                    f"val10={entry['valid_10_count']}/144 c30>=1%={entry['c30_ge_100_10']}/144 c30>=3%={entry['c30_ge_300_10']}/144 | "
                    f"EMA={p['fast_half_life_s']/60:.1f}/{p['slow_half_life_s']/60:.1f}m@{p['trend_gate']*1e4:.1f}bp "
                    f"bf={p['base_fee']*1e4:.1f}bp cap={p['capture']:.4f} fb={p['fallback_fee']*1e4:.1f}bp "
                    f"relax={p['relax_time_s']:.0f}s sc={p['step_cap']*1e4:.1f}bp yb={p['yb_releverage_fee']*1e4:.1f}bp "
                    f"Don0={p['don_center']*100:.3f}% RPF0={p['rpf_center']:.4f}",
                    flush=True,
                )

            top_records.append(entry)
            top_records.sort(key=lambda r: r["objective"], reverse=True)
            top_records = top_records[:50]

            if cap_val <= 0.725:
                top_lowcap_records.append(entry)
                top_lowcap_records.sort(key=lambda r: r["objective"], reverse=True)
                top_lowcap_records = top_lowcap_records[:30]

            if is_new_best or completed_evals % 25 == 0:
                rate_blocks = completed_evals / max(1e-6, wall_s)
                rate_pools = total_pools / max(1e-6, wall_s)
                summary_payload = {
                    "updated_wall_s": round(wall_s, 2),
                    "budget_seconds": args.budget_seconds,
                    "completed_blocks": completed_evals,
                    "total_pools_evaluated": total_pools,
                    "blocks_per_s": round(rate_blocks, 2),
                    "pools_per_s": round(rate_pools, 1),
                    "seed_637315": seed_record,
                    "best_lowcap_072": best_lowcap_072,
                    "best_lowcap_075": best_lowcap_075,
                    "best_overall": best_overall,
                    "top_25_overall": [{k: v for k, v in rec.items() if not k.startswith("grid_")} for rec in top_records[:25]],
                    "top_20_lowcap_072": [{k: v for k, v in rec.items() if not k.startswith("grid_")} for rec in top_lowcap_records[:20]],
                    "top_10_with_grids": top_records[:10],
                }
                tmp_path = summary_path.with_suffix(".json.tmp")
                tmp_path.write_text(json.dumps(summary_payload, indent=2))
                os.replace(tmp_path, summary_path)

            if completed_evals % 100 == 0:
                rate_blocks = completed_evals / max(1e-6, wall_s)
                rate_pools = total_pools / max(1e-6, wall_s)
                rem_s = max(0.0, deadline - time.monotonic())
                b_gm30 = best_overall["block_mean_gm30_10"] if best_overall else 0.0
                b_lc = best_lowcap_072["block_mean_gm30_10"] if best_lowcap_072 else 0.0
                print(
                    f"[ng-12x12] progress: {completed_evals} blocks ({total_pools} pools) | "
                    f"{rate_blocks:.2f} blocks/s ({rate_pools:.1f} pools/s) | "
                    f"elapsed={wall_s:.0f}s, ETA={rem_s:.0f}s | "
                    f"best_12x12_gm30: overall={b_gm30:.3f}%, cap<=0.72={b_lc:.3f}%",
                    flush=True,
                )

    def lane_worker(lane_tuple: tuple[str, EvaluatorSession, tuple[str, str, int | None]]) -> None:
        name, sess, spec = lane_tuple
        while time.monotonic() < deadline and not stop_event.is_set():
            eval_id, opt_name, cand_obj, raw_params = next_candidate_task()
            cands, clean_params, dons, rpfs = build_12x12_candidates(
                eval_id,
                raw_params,
                base_pool,
                grid_n=args.grid_n,
                don_radius=args.don_radius,
                rpf_radius=args.rpf_radius,
            )
            t0 = time.monotonic()
            try:
                res = sess.evaluate(cands)
            except Exception as exc:
                print(f"[ng-12x12] lane {name} error on eval #{eval_id}: {exc}; reconnecting...", flush=True)
                try:
                    sess.close()
                except Exception:
                    pass
                try:
                    _, sess = open_lane(spec)
                    res = sess.evaluate(cands)
                except Exception as exc2:
                    print(f"[ng-12x12] lane {name} reconnect failed: {exc2}", flush=True)
                    break
            dt = time.monotonic() - t0
            scored = score_12x12_block(res, clean_params, dons, rpfs, grid_n=args.grid_n)
            record_result(eval_id, opt_name, cand_obj, scored, dt, name)

    try:
        with ThreadPoolExecutor(max_workers=len(active_lanes)) as pool:
            futures = [pool.submit(lane_worker, lt) for lt in active_lanes]
            for fut in futures:
                fut.result()
    finally:
        stop_event.set()
        for _name, sess, _spec in active_lanes:
            try:
                sess.close()
            except Exception:
                pass

    wall_total = time.monotonic() - opt_started
    print(
        f"[ng-12x12] Finished {completed_evals} 12x12 blocks ({total_pools} pools) in {wall_total:.1f}s "
        f"({total_pools/max(1e-6, wall_total):.1f} pools/s). Summary written to {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
