# curve-fx-optimization (`fxopt`)

`curve-fx-optimization` is the small, user-facing workflow layer for Curve FX experiments. It owns candidate grids, optimization, result files, interactive heatmaps, and trace replay. It calls the evaluator supplied by `curve-fx-arb-harness`; pool mechanics remain in `twocrypto-cpp`.

```
twocrypto-cpp  ->  curve-fx-arb-harness  ->  curve-fx-optimization (fxopt)
 pool mechanics     evaluator session        runs, plots, replay
```

## Setup

Use Python 3.12 and `uv`. Build the pool and harness first, then install this checkout:

```sh
cd /path/to/twocrypto-cpp
uv sync --frozen --extra test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix "$PWD/_install"

cd /path/to/curve-fx-arb-harness
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/twocrypto-cpp/_install
cmake --build build --parallel

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
`configs/autoresearch/` workbench. A completed run embeds resolved candidate
defaults, axes, robustness radii, execution inputs, and local replay inputs in
`run.json`, so autoresearch TOMLs can be reused or discarded without losing the
result's meaning. The pool template remains under `configs/templates`, while
compiled policies are harness build inputs.

Non-empty `[placement].hosts` implies SSH; otherwise execution is local.
Optional `[placement].numa_nodes` creates one persistent evaluator lane per NUMA
node. Remote runs map config-relative sibling-workspace paths to
remote-home-relative `arb/...`. Missing portable session inputs are copied once
through the first shared-NFS host, while existing files are kept; the evaluator
is never copied.

## The four commands

`run` and `optimize` write a compact result bundle containing exactly `run.json`
and `results.npz`; heatmap and Shift-click add their own image/state or trace
artifacts. An interrupted grid keeps one `.fxopt-partial` spool and resumes when
the same config and output directory are run again. The output directory is the
durable hand-off between running, plotting, and replaying.

Run a Cartesian candidate grid in bounded batches:

```sh
uv run fxopt run configs/experiments/eurusd-a-donation-rpf-8x8x8.toml \
  --output runs/eurusd-a-donation-rpf-8x8x8
```

Run adaptive Nevergrad optimization through the same evaluator fleet:

```sh
uv run fxopt optimize configs/experiments/eurusd-a-donation-rpf-8x8x8.toml \
  --output runs/eurusd-a-donation-rpf-8x8x8-opt
```

Open the mature interactive heatmap explorer (or save a PNG and its state):

```sh
uv run fxopt heatmap runs/eurusd-a-donation-rpf-8x8x8
uv run fxopt heatmap runs/eurusd-a-donation-rpf-8x8x8 \
  --metric apy_net --output runs/eurusd-a-donation-rpf-8x8x8/heatmap.png \
  --no-show
```

The explorer supports metric filters, adaptive limits for price difference, skew, slippage, and final price difference, axis selection, and multi-metric views. Clicking a cell selects its exact candidate. Right-click replays that candidate with YieldBasis disabled. Shift-click replays it with the configured YB mode, preserving the run's session setting. These interactions are part of the heatmap workflow; no separate metrics window is required.

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

## Results and configuration

`run.json` contains the resolved run metadata, config origin, axes, robustness
radii, evaluator/session settings, and local replay inputs. `results.npz`
contains candidate results and metrics. Heatmaps and Shift-click use this bundle
directly; the mutable source TOML is only a fallback for older runs.

Rank point values or exact axial stars without creating another manifest:

```sh
uv run python scripts/analyze_basins.py runs/RUN --rank score
uv run python scripts/analyze_basins.py runs/RUN --rank yb-gm \
  --min-lp-gm 0.05 --max-detach 5
```

For an older run with no embedded radii, repeat `--robust AXIS=RADIUS`, for
example `--robust pool.mid_fee=0.0002 --robust pool.out_fee=0.0002`.
Do not copy evaluator or pool code into this repository. Run outputs are
ordinary local artifacts and can be inspected or plotted again without
rebuilding the grid.

For a production run, fetch authorized Git-LFS market data first:

```sh
git lfs pull
```

The supported user-facing surface is `fxopt`; pool and evaluator implementation details stay in their sibling repositories.
