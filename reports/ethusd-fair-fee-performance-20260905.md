# ETHUSD fair-fee throughput and precision — 2026-09-05

The accepted changes preserve f64 results exactly: enable the existing conservative
skip cursor for non-flat native fee settings, and cache the fair policy's committed
marginal price instead of recomputing it for every sizing probe. Use 128 threads per
blade, two NUMA-bound evaluator processes, and batches of 2,048 candidates.

The 161,280-pool comparison improved from 64.5 to 52.1 seconds wall time. Every
reported metric except elapsed time matched baseline f64 bit-for-bit. The full
2,949,120-pool confirmation completed in 539.3 seconds, with zero failed pools.
Calculation throughput reached 5,553 pools/s versus the old artifact's 4,146:
33.9% faster overall, or 25.0% faster per blade after accounting for 15 versus
14 blades. Including startup and merge, the new run delivered 5,468 pools/s.

The full result is `runs/ethusd-fair-fee-dense-f64-20260905` and its six-panel
heatmap rendered successfully. All shared non-timing columns except endpoint
`apy` and `apy_net` match the old dense artifact exactly, including robust APY,
LP controller growth, price errors, trades and rebalances. The endpoint APYs
include the start-value normalization from the preceding cleanup commit:
maximum differences are 1.77e-8 and 8.31e-9 respectively in stored APY fractions
(less than 0.00018 bp). These are separate from the performance changes, whose
controlled comparison against the current scalar f64 baseline is bit-exact.

## Workload and builds

The numerical reference is the completed
`runs/ethusd-fair-fee-perf-161k-ld-20260904/{run.json,results.npz}`:
161,280 successful candidates, zero failures. It uses scalar execution with the
`grid_core` metric profile and genuine x86 long double. All real inputs cross the
existing binary64 protocol before widening. LD is a numerical reference, not an
onchain integer-equivalence claim.

All benchmark variants use the full ETH history from 2024-01-01 through
2026-08-25, the event-aligned fair-price feed, 2 bp external arb cost, hourly idle
ticks, no synthetic user swaps, no slippage probes, and YB off. The axes are sampled
from the original dense grid, including its low-fee region and endpoints:

| Axis | Values |
|---|---|
| Base fee | 32 original values spanning 0.001–0.03 |
| Capture | 28 original values spanning 0–1 |
| A | 30000, 50000, 70000 |
| Donation APY | 1%, 2%, 3%, 4%, 5% |
| Oracle half-life | 600, 3600, 14400 seconds |
| Reserved-profit fraction | 0.1, 0.23333333333333334, 0.36666666666666664, 0.5 |

The original full grid has 64 × 64 × 3 × 5 × 3 × 16 = 2,949,120 candidates.
The original stored ordinal 85920's fee/capture coordinates are included in the
reference at ordinal 9240. Ordinals are local to each grid; compare by ordinal only
when axes and their values match.

Fifteen matching Intel Xeon Platinum 8352Y blades were used: a5, a3, a4, a6, a7,
a8, a10, b1, b4, b5, b6, b7, b8, b9, b10. Each has two 32-core sockets and 128
hardware threads. Builds use GCC 12.2, `-O3`, native tuning and LTO. These compiler
settings were already enabled in the old run. Baseline executables are retained
separately under `build/fair-perf-baseline`; accepted executables are under
`build/fair-perf-cache`. No fast-math, event coarsening, reduced sizing budget, or
new convergence tolerance is enabled.

## Measured variants

All rows below calculated the same 161,280 pools. Wall time is the coordinator's
reported calculation-through-merge time, excluding build, source transfer and final
Mac retrieval. Calculation rate is the sum of each worker's count/calculation time
from `run.json`; it removes startup and final collection effects but is not a claim
that every blade remained busy through the final tail. Each variant was measured
once, so small timing differences should not be treated as precise estimates.

| Variant | Threads/blade | Batch | Calculation pools/s | Wall seconds | f64 results |
|---|---:|---:|---:|---:|---|
| Scalar LD reference | 128 | 512 | 1170 | 147.0 | Numeric reference |
| Scalar f64 baseline | 128 | 512 | 2711 | 64.5 | Baseline |
| Exact skip | 128 | 512 | 2819 | 64.2 | Exact |
| Exact skip, physical-core count | 64 | 512 | 2361 | 74.1 | Exact |
| Exact skip + marginal-price cache | 128 | 512 | 3180 | 55.5 | Exact |
| Same + wider quote-root arithmetic | 128 | 512 | 2952 | 59.8 | Changed; rejected |
| Exact skip + cache, larger batches | 128 | 2048 | 3737 | 52.1 | Exact |

The cache and skip combination alone gained 17.3% in calculation throughput at
fixed placement/batch size. The final settings reduced total benchmark time by
19.2%, or raised end-to-end throughput by 23.8%. Increasing batch size improves
utilization inside evaluator batches, while making the final scheduling tail
coarser. The dense run has 720 leases even at batch 2048, so it has ample work to
balance across fifteen machines.

Run directories use prefix `runs/ethusd-fair-fee-perf-161k-`, followed by
`ld`, `f64`, `skip`, `skip-64t`, `cache`, `wide-y`, or `cache-2048`, and suffix
`-20260904`. The earlier 46,080-pool LD pilot is not used for timing comparisons.

## Fidelity to LD

Error units below are basis points of APY: one basis point is 0.0001 in the stored
fraction. The accepted optimized f64 has exactly the same errors as baseline f64.

| Robust 90-day APY comparison | Baseline/optimized f64 | Mixed quote root |
|---|---:|---:|
| Median absolute error, bp | 0.00000632 | 0.00000574 |
| 90th-percentile absolute error, bp | 0.00980 | 0.00616 |
| 99th-percentile absolute error, bp | 7.628 | 7.238 |
| Maximum absolute error, bp | 556.57 | 581.72 |
| Spearman rank correlation | 0.999530 | 0.999500 |
| Top 1% overlap | 99.07% | 98.88% |

Widening only `get_y` intermediate arithmetic modestly improved typical errors but
cost ~7% throughput relative to optimized f64, worsened the worst robust-APY error,
and reduced top-region overlap. That experiment was reverted; its artifact remains
for comparison. It did not solve the divergent trajectory tail.

The attached region is much more stable. Using the **LD** maximum seven-day
price-error cutoff:

| Cutoff | Candidates | Robust APY p99 error, bp | Maximum error, bp | Top 1% overlap |
|---|---:|---:|---:|---:|
| 1% | 7826 | 0.00254 | 0.07639 | 100% |
| 2% | 32061 | 0.00188 | 0.51103 | 100% |

Both modes choose reference-grid ordinal 4273 as the best robust-APY candidate
inside either cutoff. Classification at 0.5%, 1%, and 2% maximum seven-day price
error agrees for every candidate. This does not assert agreement at arbitrary
slider thresholds or on every trade count.

The largest robust-APY divergence is reference ordinal 29514: LD 8.7801% versus
f64 3.2144%, with LD maximum seven-day price error 40.26%. Across the entire
reference, trade counts differ in 57,109 cells; the median count difference is
zero, p99 is 181, and maximum is 14,388. Small arithmetic changes can lead to
different later sizing and price-scale decisions. The inference is trajectory
forking, not a proven localization of each divergent cell's first cause.

Use optimized f64 for broad scans. Use blade LD for finalists, close ranking ties,
and neighborhoods around any constraints that matter to a decision. Tighter
arithmetic in one quote calculation is not a substitute for that final check.

## Hotspots and remaining options

Ten-second `perf` user-cycle samples on one active evaluator found f64 spending
roughly 80% of samples in sizing and quote routines: ~20% in the pool fee function,
~26% in the main sizing function, ~22% in refinement, with further quote/envelope
helpers. Around 15% was in event-loop work. LD likewise concentrated in sizing.
These are sampled symbol shares, with inlined code attributed to its enclosing
symbol; they are not an exact source-line cost breakdown.

| Candidate | Decision |
|---|---|
| Reuse committed fair marginal price | Implemented; exact and measurably faster |
| Skip events rejected by existing conservative arb floor | Implemented; preserves mandatory tick/donation/metric events |
| 128 vs 64 threads | Keep 128; 64 was slower on this CPU/workload |
| Larger batches | Use 2048; faster in the measured grid |
| Widen only quote-root arithmetic | Rejected on measured speed/accuracy tradeoff |
| Fewer sizing probes or earlier search termination | Could reduce the dominant cost, but changes selected trades; not result-preserving |
| Coarser candles, fewer causal events, less frequent arb | Changes the simulated market/actor sequence; not enabled |
| Blanket fast-math or f32 | Unmeasured here; cannot be claimed to retain LD ranking or decision guards |
| More generic validation/caches | Not justified by the observed hotspot profile |

## Validation and use

The accepted f64 variants were compared by canonical ordinal against baseline
across every stored metric except `elapsed_ms`, including exact status coverage.
All 161,280 rows match. The updated C++ fair-fee test passes its uint lattice,
floating admission, and post-update quote cases. A separate optimized x86 LD
check matched the scalar LD reference exactly on 70 sampled and divergent-tail
candidates across all 17 non-timing metrics. The maintained native evaluator's
six public contract tests pass (0.58 seconds). The fair-policy build passes five
of those cases; the generic user-scheduling case expects a slippage sample despite
its deliberately stale feed. It also fails on the untouched pre-change fair-policy
binary, so it is not a regression from these optimizations.

The accepted evaluator was also built locally for f64 and LD; local Apple ARM LD
has f64 precision and is only for local workflow/replay, not the x86 reference.
Local trace replay and plot rendering passed for reference ordinal 9240 with YB
off (2.52 seconds) and active 2L at 3x cash (2.76 seconds). Temporary traces were
deleted after rendering.

Tracked configurations:

- `configs/experiments/ethusd-fair-fee-reference-161k-ld.toml`
- `configs/experiments/ethusd-fair-fee-dense-f64.toml`
- `configs/experiments/ethusd-fair-fee-arb2bps.json`

Run the dense grid with `uv run fxopt run
configs/experiments/ethusd-fair-fee-dense-f64.toml --output runs/NEW_RUN`.
Add `--rebuild` when refreshing the cluster evaluator from changed sources.
Do not overwrite an active run or the retained baseline build while comparing it.

Open the accepted dense result with the same six panels:

```sh
uv run fxopt heatmap runs/ethusd-fair-fee-dense-f64-20260905 \
  --x policy_params.0 --y policy_params.1 \
  --metric apy_masked --metric apy_net_robust_90d \
  --metric lp_xcp_profit --metric max_7d_rel_price_diff \
  --metric avg_rel_price_diff --metric trades --columns 3
```
