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

## Prepared inputs and data workbench

Selected candles, block tapes, and oracle feeds live in the local, Git-ignored
[`data/`](data/README.md) beside small provenance manifests; only the legacy
`candles.xz` archives are tracked, under Git LFS. Experiment
TOMLs use relative paths into this collection or the sibling data workbench.
Each selected input is checked against its adjacent `dataset.json`, and the
dataset revision and checksums are recorded in `run.json`.

Acquisition, packing and historical onchain fitting and plotting stay in the
sibling [`../data/`](../data/README.md) workbench, with their own dependencies.
The optimizer consumes prepared files without importing or running those
generators. External input paths remain supported.

The harness arbitrages by the fixed rules in
[`arbitrage_rules.md`](../curve-fx-arb-harness/docs/arbitrage_rules.md): one
native arbitrageur per pool and, with `scenario.yb_mode = "active_2l"`, the
YieldBasis LEVAMM actor. Their settings are `pool.costs.entry_edge_bps`,
`pool.costs.arb_fee_bps`, `pool.costs.gas_coin0`, `session.yb_execution_bps` and
`session.yb_min_net_profit_coin0`. Pick one market mode per config:

- **Candles:** `scenario.market` (OHLC JSON) with the default
  `session.event_mode = "candles"`. Use this to ballpark assets without trade data.
- **Blocks:** `scenario.block_tape` (NPZ from `data/market/build_block_tape.py`)
  with `session.event_mode = "block"`. The event price of each block is the
  external price `session.arb_settle_offset_s` seconds after its timestamp.

`session.excluded_time_ranges = [[start, end)]` removes UTC intervals (for
example 10 October 2025, `[[1760054400, 1760140800]]`). Their events and
price-feed samples are omitted, calendar time advances across the gap, and
pool/YB state resumes at the next retained event. The
[harness protocol](../curve-fx-arb-harness/protocol/protocol_spec.md) defines
these modes, the clock and trace behavior. Build a new evaluator remotely with
`fxopt run ... --rebuild` before its first cluster run.

`session.early_stop_max_7d_rel_price_diff` ends a pool's run once its maximum
7-day price divergence passes the value (default 0.3; 0 never stops). Shift-click
replays always run the full history.

`pool_nav_vs_hold` is final total pool assets divided by the final marked value
of the starting token basket, minus one, with both portfolios at the final
external price. It includes donations and net liquidity changes, is not
annualized, and does not normalize LP supply.

Configs live under `configs/`. Pool templates are in `configs/templates`;
LLM-generated iterative manifests belong in the ignored `configs/autoresearch/`
workbench. A completed run embeds resolved candidate defaults, axes, robustness
radii, execution inputs, and local replay inputs in `run.json`, so manifests can
be reused or discarded without losing the result's meaning. Compiled policy
headers remain harness build inputs.

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
uv run fxopt run configs/autoresearch/GRID.toml --output runs/GRID
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
uv run fxopt run configs/autoresearch/GRID.toml --output runs/GRID --transfer --rebuild
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
remain result rows; an evaluator transport failure ends that worker and its
unfinished lease is not reassigned, while healthy workers continue and completed
rows are published as a partial grid. Final
`run.json` records the schedule, configured evaluator path, validated policy
contract, per-worker timing, and aggregate status counts.
Every manifest must declare `run.metric_fields`, which fixes the typed result
schema. Subsequent runs can omit preparation
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

Select the header under `twocrypto-cpp/include/pools/twocrypto_fx/policies/` and
use its descriptor's parameter order. Policy headers are repository-local
build inputs, not part of the installed pool library.

The current research policy is `report_dual_ema.hpp` (id `report_dual_ema`): a
report fee over the dual-EMA price-scale driver. Its 13 parameters, in order:
`base_fee`, `away_capture`, `toward_ratio`, `fallback_fee`, `fast_half_life_s`,
`slow_half_life_s`, `kappa`, `deadband`, `min_cap`, `max_cap`, `age_offset_s`,
`future_window_s`, `aging_s`. Report probabilities grid independently of the
policy parameters, for example
`"pool.run.arb_report_rate" = { start = 0.1, stop = 1.0, count = 10 }`.

Use blade f64 for broad discovery and x86-64 blade long double for production
finalist ranking. Apple ARM `long double` has binary64 width, so local Mac replay
checks workflow and behavioral stability rather than x86 extended precision.

For YB grids, use the standard launcher (paths relative to this repository):

```sh
bash scripts/heatmap-yb.sh configs/autoresearch/GRID.toml runs/GRID
```

It follows and retrieves unfinished remote runs, or opens completed local
artifacts directly. The default view is donation × RPF, with three columns:

| Pool APY | YB APY | YB GM APY |
| --- | --- | --- |
| `apy_net` | `yb_apy` | `yb_apy_gm` |
| `apy_net_masked` | `yb_apy_masked` | `yb_apy_gm_masked` |
| `max_7d_rel_price_diff` | `apy_net_robust_90d` | `avg_imbalance` |

Masked panels exclude price divergence above 1500 bp (15%); they do not require
positive yields. Shift-click defaults to `active_2l` and cash multiplier 3.
Append heatmap options to override axes or thresholds while retaining the layout.

Open the mature interactive heatmap explorer directly (or save a PNG and its state):

```sh
uv run fxopt heatmap runs/GRID
uv run fxopt heatmap runs/GRID \
  --metric apy_net_robust_90d_masked --metric detach_energy_ungated --columns 2 \
  --max-price-diff-bps 1000 \
  --output runs/GRID/heatmap.png \
  --no-show
```

The explorer supports metric filters, slice-local color limits, an interactive
adaptive limit for price difference, fixed CLI filters for detachment,
slippage and final price difference, axis selection,
and multi-metric views. `--columns` controls the panel layout. Clicking a cell
selects its exact candidate. Right-click replays that candidate with YieldBasis
disabled. Shift-left-click replays it with the configured YB mode. Plain left-click
prints the selection without opening a plot. Replay prints the selected parameter coordinates to the console. Interactive
replay traces are temporary and removed after plotting; titles and summaries
use the local replay metrics.

Raw panels never hide observations. Append `_masked` to any stored metric name
to filter that panel by the interactive 7-day price-difference control and the
fixed detachment limit—for example `apy_net_masked` or
`apy_net_robust_90d_masked` (`apy_masked` is an alias that shows `apy_net`).
`--max-price-diff-bps` sets the initial threshold. The fixed
`--max-detach-energy`, `--final-price-diff-bps` and `--slippage-bps` filters apply when explicitly
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
uv run fxopt shiftclick runs/GRID \
  --ordinal 12 --output runs/GRID/inspections/ordinal-12
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

`run.json` contains the resolved run metadata, config path, axes, robustness
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

For a production run, fetch the published Git-LFS inputs required by the config:

```sh
git lfs pull --include="data/market/btcusd/**"
```

Use `fxopt` for runs, heatmaps, replay, and remote lifecycle. The small
`scripts/` surface contains event fair-price feed generation, basin analysis,
and the cluster Python launcher;
pool and evaluator implementation details stay in their sibling repositories.

Compiled policies can provide optional heatmap labels in parameter order:

```toml
[compiled_policy]
id = "example"
header = "example.hpp"
parameter_names = ["fast_ema (s)", "slow_ema (s)", "kappa"]
```

When supplied, names must be unique and match the number of default policy
parameters. They are stored in `run.json` and used as display labels; axis keys,
Cartesian ordering, and replay payloads remain `policy_params.<index>`.

Find connected regions above a metric floor, optionally below the 7-day
price-difference limit, with `scripts/find_blobs.py`:

```sh
uv run python scripts/find_blobs.py runs/RUN \
  --metric yb_gm --min-value 0.001 --max-pdif 0.15 \
  --axes fast_ema kappa pool.reserved_profit_fraction pool.donation_apy \
  --top 10 --output runs/RUN/inspections/blobs.json
```

Thresholds are exclusive and use raw metric units: `0.001` means 0.1% APY;
`0.15` means 15% for `max_7d_rel_price_diff`. Omit `--max-pdif` to remove that
constraint. `yb_gm` aliases `yb_apy_gm`; other stored metric names work directly.
Axes accept canonical keys (such as `policy_params.3`) or stored policy labels
(with an optional trailing unit omitted, such as `fast_ema`). Other axes stay
fixed. Only adjacent sampled indices along one selected axis connect: there
are no diagonal links, interpolation, or links across failed/nonfinite points.
No other yield filters are applied.

Blobs rank by point count. Reports include exact member ordinals, fixed axes,
parameter extents, metric minimum/median/maximum, bounding-box fill, and the
number of centers whose two immediate neighbors on every selected axis pass.
Missing neighbors at grid boundaries do not pass. Extents are not guaranteed
passing rectangles. The representative has the most passing immediate
neighbors, with ties resolved by smallest ordinal; it is not an optimized
parameter recommendation.
