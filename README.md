# curve-fx-optimization (`fxopt`)

`curve-fx-optimization` is the small, user-facing workflow layer for Curve FX experiments. It owns candidate grids, cluster execution, result files, interactive heatmaps, and trace replay. It calls the evaluator supplied by `curve-fx-arb-harness`; pool mechanics remain in `twocrypto-cpp`.

## Repository split

- [`twocrypto-cpp`](https://github.com/curvefi/twocrypto-cpp) — C++ Twocrypto pool implementation and Vyper parity; no market simulation or experiment orchestration.
- [`fx-arb-harness`](https://github.com/curvefi/fx-arb-harness) — C++ arbitrage simulation and evaluator protocol; owns market-event execution and raw metrics.
- [`fx-optimization`](https://github.com/curvefi/fx-optimization) — cluster orchestration, parameter grids, scoring, result storage, robustness analysis, heatmaps, and replay.

The iterative stable-plateau research process is documented in
[`workflow.md`](workflow.md).

## Setup

Use Python 3.12 and `uv`. Build the pool and harness first, then install this checkout:

```sh
cd /path/to/twocrypto-cpp
uv sync --frozen --extra test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix "$PWD/_install"

cd /path/to/curve-fx-arb-harness
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/twocrypto-cpp/_install
cmake --build build/native --target arb_evaluator_f64 --parallel

cmake -S . -B build/grid-dual-current -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/twocrypto-cpp/_install \
  -DPOLICY_ID=yieldbasis_twocrypto_policy \
  -DPOLICY_HEADER_PATH=/path/to/twocrypto-cpp/include/pools/twocrypto_fx/policies/yieldbasis.hpp
cmake --build build/grid-dual-current \
  --target arb_evaluator_f64 arb_evaluator_ld --parallel

cd /path/to/curve-fx-optimization
uv sync --frozen --group dev
```

## Market data

Download raw candles and filter them as separate steps. Use dated filenames;
both commands refuse to replace an existing output.

```sh
uv run python scripts/fetch_binance_candles.py \
  --start 2023-01-01T00:00:00Z --end 2026-08-26T00:00:00Z \
  --workers 16 \
  --output data/market/ethusd/candles-2023-2026-08-25-raw.json

uv run python scripts/filter_candles.py \
  data/market/ethusd/candles-2023-2026-08-25-raw.json \
  data/market/ethusd/candles-2023-2026-08-25-filtered.json
```

The downloader partitions the requested minute range into independent Binance
pages, fetches them concurrently, preserves legitimate exchange gaps, rejects
duplicates or disorder, and publishes the merged JSON atomically. The filter
owns only the established centered-neighbor OHLC clipping step.

Configs live under `configs/`. Human-curated manifests live in
`configs/experiments/`; LLM-generated iterative manifests belong in the ignored
`configs/autoresearch/` workbench. The maintained BTC starters are
`btcusd-native-discovery-f64.toml`, `btcusd-dual-ema-robust-f64.toml`, and
`btcusd-p3-finalist-comparison.toml`. The last is one compact finalist star:
change only its evaluator target and `scenario.yb_mode` when repeating it for
f64/LD or off/active_2l/reference_2l comparisons. Give each variant a distinct
output directory (for example, `runs/btcusd-p3-off-f64`) because the manifest
ID is intentionally neutral. A completed run embeds resolved candidate
defaults, axes, robustness radii, execution inputs, and local replay inputs in
`run.json`, so autoresearch TOMLs can be reused or discarded without losing the
result's meaning. Pool templates remain under `configs/templates`; compiled
policy headers remain harness build inputs.

Pool templates keep contract encodings (for example, WAD integers). Candidate
defaults and grid axes use human simulation units: `fee_gamma` and both
`adjustment_step` controls are fractions such as `0.03`, not `3e16`. The
evaluator rejects sub-WAD ratios instead of silently running a collapsed value.
Every Shift-click replay records the resolved numeric controls under
`result.artifacts.effective_inputs` for a quick initialization check.

Non-empty `[placement].hosts` implies SSH; otherwise execution is local.
Optional `[placement].numa_nodes` creates one persistent evaluator lane per NUMA
node. Remote runs map config-relative sibling-workspace paths under the shared
`/home/heswithme/arb/...` workspace. Missing portable session inputs are copied once
through the first shared-NFS host, while existing files are kept; the evaluator
is never copied.

## Workflow

`run` writes a compact result bundle containing exactly `run.json` and
`results.npz`; heatmap and Shift-click add their own image/state or trace
artifacts. Grid results are buffered as typed NPZ shards. If a worker fails, completed
results are retained and unevaluated cells remain blank. Missing cells are
marked `uncalculated`; interrupted grids are not automatically resumed.
The output directory is the durable hand-off between running, plotting, and
replaying.

Run a Cartesian candidate grid in bounded batches:

```sh
uv run fxopt run configs/experiments/eurusd-a-donation-rpf-8x8x8.toml \
  --output runs/eurusd-a-donation-rpf-8x8x8
```

Remote grids deterministically shuffle small contiguous ordinal tiles into
machine-sized leases. One worker process per configured machine reconstructs
the lease table locally and asks the coordinator only for its next lease ID;
there are no fixed blade shards. Each worker owns its machine-local evaluator
slots, and each lease contains one full evaluator batch per slot. Every
evaluator registers the typed grid once and later receives only ordinal ranges.
Workers write one local `/tmp` partition. The first placement host is the
coordinator: it collects the available partitions once, verifies disjoint
coverage, merges their typed NPZ members, and sends only the final `run.json`
and `results.npz` to the Mac. The launcher is SSH-specific; scheduling and
worker execution are not.

The evaluator session also retains a separate ordinary candidate-batch API.
That is the extension seam for a future adaptive ask/evaluate/tell controller:
the controller may keep one machine worker alive per host and submit arbitrary
point batches, while grid execution continues to use its cheaper registered
ordinal-range path. Optimizer libraries and optimizer state do not belong in
the evaluator or the grid runner.

The shared managed Python runtime is entered through `scripts/cluster-python`
and `shell.nix`; it does not synchronize an environment for each run. Source
transfer and compilation happen once through shared home:

```sh
uv run fxopt run configs/experiments/eurusd-a-donation-rpf-16x16x16-two-blades.toml \
  --output runs/eurusd-a-donation-rpf-16x16x16-two-blades \
  --transfer --rebuild
```

The coordinator is detached before the command follows its log, so a Mac or
Wi-Fi disconnect does not stop the grid. A normal run follows and retrieves the
two final artifacts automatically. The same config and output directory expose
the remaining state transitions:

```sh
uv run fxopt run <config> --output <run-dir> --status
uv run fxopt run <config> --output <run-dir> --follow
uv run fxopt run <config> --output <run-dir> --retrieve
uv run fxopt run <config> --output <run-dir> --stop
```

`--status` checks once. `--follow` resumes the concise coordinator log and
retrieves on completion. `--retrieve` fetches only an already-complete job.
`--stop` terminates the coordinator and its evaluator connections while retaining
the remote directory and local job handle for diagnosis; partial grids are not
resumable. A small hidden job handle exists locally until successful retrieval,
after which the run directory again contains exactly `run.json` and `results.npz`.
`--overwrite` removes an existing completed or empty fxopt run directory before
starting the supplied config. It refuses directories containing a detached-job
handle and cannot be combined with the four remote job-control flags.

`--transfer` rsyncs the pool, harness, and workflow sources once to the first
configured blade. `--rebuild` implies transfer and builds the configured
evaluator target—f64 or long double—once there. Dated market inputs remain
copy-if-missing; the small pool template is refreshed through ordinary rsync.
Workers send compact progress snapshots every two seconds and the coordinator
prints one aggregate heartbeat every two seconds. Rate and ETA remain hidden
until every worker has produced a batch; afterward ETA uses the remaining
global queue and currently active worker rate. Deterministic candidate failures
remain result rows; an evaluator transport failure is retried locally three
times; if the worker still fails, healthy workers continue and completed rows
are published as a partial grid. Final
`run.json` records the schedule, configured evaluator path, validated policy
contract, per-worker timing, and aggregate status counts.
Remote manifests must declare `run.metric_fields`, which fixes the typed result
schema even when the first chunk fails. Subsequent runs can omit preparation
flags when the requested evaluator target and sources are already present.

Evaluator builds select either native/no-policy behavior or one compiled
policy, such as the `yieldbasis_twocrypto_policy` dual-EMA controller. Native manifests omit
`[compiled_policy]` and use `policy_params = []`. Compiled-policy grids declare
the build input explicitly so `--rebuild` selects the dual-EMA header:

```toml
[compiled_policy]
id = "yieldbasis_twocrypto_policy"
header = "../../../twocrypto-cpp/include/pools/twocrypto_fx/policies/yieldbasis.hpp"
```

The same build input supports `fair_price_fee` (base fee and capture),
`implied_fair_fee` (plus assumed arb cost), `revealed_fair_fee` (plus revelation
weight; floating-only), and `price_feed_passthrough` (no parameters). Select the
matching header under `twocrypto-cpp/include/pools/twocrypto_fx/policies/` and
use its descriptor's parameter order. Experimental headers are repository-local
build inputs, not part of the installed pool library.

The dual-EMA policy returns zero for its policy fee, so the pool's native fee
surface remains active. Its six candidate parameters are fast and slow EMA
half-lives, kappa, deadband, and minimum/maximum caps.

Use blade f64 for broad discovery and x86-64 blade long double for production
finalist ranking. Apple ARM `long double` has binary64 width, so local Mac replay
checks workflow and behavioral stability rather than x86 extended precision.

Open the mature interactive heatmap explorer (or save a PNG and its state):

```sh
uv run fxopt heatmap runs/eurusd-a-donation-rpf-8x8x8
uv run fxopt heatmap runs/eurusd-a-donation-rpf-8x8x8 \
  --metric apy_net_robust_90d_masked --metric detach_energy_ungated --columns 2 \
  --max-price-diff-bps 1000 \
  --output runs/eurusd-a-donation-rpf-8x8x8/heatmap.png \
  --no-show
```

The explorer supports metric filters, slice-local color limits, interactive
adaptive limits for price difference and detachment, fixed CLI filters for
slippage and final price difference, axis selection,
and multi-metric views. `--columns` controls the panel layout. Clicking a cell
selects its exact candidate. Right-click replays that candidate with YieldBasis
disabled. Shift-left-click replays it with the configured YB mode. Plain left-click
prints the selection without opening a plot. Replay prints the selected parameter coordinates to the console. Interactive
replay traces are temporary and removed after plotting; titles and summaries
use the local replay metrics.

Raw panels never hide observations. Append `_masked` to any stored metric name
to filter that panel by the interactive 7-day price-difference and detachment
controls—for example `apy_masked`, `apy_net_masked`, or
`apy_net_robust_90d_masked`. `--max-price-diff-bps` and
`--max-detach-energy` set their initial thresholds. The fixed
`--final-price-diff-bps` and `--slippage-bps` filters apply when explicitly
provided; unsuffixed diagnostic panels remain unmasked.

The legacy-compatible `apy_1_masked` and `apy_5_masked` views use `apy_net`
and mask it against the matching 1%- or 5%-of-TVL slippage probe. Selecting
either view adds an interactive slippage-cap slider (0-100 bp, 20 bp default).
The raw `tw_real_slippage_*_masked` panels remain price-difference-masked only.

No-YB discovery ranks `apy_net_robust_90d` with detachment. The earnings
metric gives equal weight to the mean and worst-5% mean of daily-sampled
trailing-90-day net log returns, then converts that blended rate to APY.
Negative weak regimes remain finite and rankable. `lp_detach_score` subtracts
`2.5 * detach_energy_ungated` from the metric's log-growth form. This avoids
the positive floor and hourly power/log work of legacy `apy_net_gm`, which
remains available in full reference and YB runs.

Replay one ordinal with a full trace:

```sh
uv run fxopt shiftclick runs/eurusd-a-donation-rpf-8x8x8 \
  --ordinal 12 --output runs/eurusd-a-donation-rpf-8x8x8/inspections/ordinal-12
```

`--trace-interval` and `--actions` enable denser traces and action recording.
Replay takes the exact stored candidate from `results.npz` and the local evaluator
and session inputs embedded in `run.json`; SSH placement is not reused. It writes
`shiftclick.json`, the trace artifacts, and `shiftclick.png` under the selected
output directory. The plot title labels the local platform so
double-versus-production-long-double stability checks stay explicit.

Local Shift-click replay defaults to YB `active_2l` with a 3x cash multiplier,
including when the source grid ran without YB. Right-click continues to force
YB off. Override these defaults with `--shiftclick-yb-mode` and
`--shiftclick-yb-cash-multiplier` on `fxopt heatmap`, or with `--yb-mode` and
`--yb-cash-multiplier` on `fxopt shiftclick`.

## Results and configuration

`run.json` contains the resolved run metadata, config origin, axes, robustness
radii, configured evaluator path, validated policy contract, session settings,
and local replay inputs. `results.npz` contains candidate results and metrics.
Heatmaps and Shift-click use this bundle directly.

Rank point values or exact axial stars without creating another manifest:

```sh
uv run python scripts/analyze_basins.py runs/RUN --rank score
uv run python scripts/analyze_basins.py runs/RUN --rank yb-gm \
  --min-lp-gm 0.05 --max-detach 5
uv run python scripts/analyze_basins.py runs/RUN --rank apy-net-robust \
  --max-price-diff-bps 2000 --max-fee-bps 200
```

Override embedded robustness radii for a one-off analysis with
`--robust AXIS=RADIUS`, for example
`--robust pool.mid_fee=0.0002 --robust pool.out_fee=0.0002`.
Do not copy evaluator or pool code into this repository. Run outputs are
ordinary local artifacts and can be inspected or plotted again without
rebuilding the grid.

For a production run, fetch authorized Git-LFS market data first:

```sh
git lfs pull
```

Use `fxopt` for runs, heatmaps, replay, and remote lifecycle. The small
`scripts/` surface is limited to data preparation and post-run basin analysis;
pool and evaluator implementation details stay in their sibling repositories.
