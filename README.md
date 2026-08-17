# curve-fx-optimization (`fxsim`)

The orchestrator is the one user-facing Python distribution and CLI for Curve FX simulation. It owns data manifests and run artifacts, pair/scenario/policy registries, grid generation/search, adaptive optimization and scoring, local/SSH execution, selection normalization, shiftclick replay, ranking, heatmaps, and trajectories. It consumes `arb_evaluator_ld` from `curve-fx-arb-harness`; it does not embed pool or evaluator code.

## Setup the three repositories

All repositories are independent Python/CMake projects. Use Python 3.12 and uv; use C++17/CMake for the native components. Build/install the pool first, then build the harness against that install, then install this CLI:

```sh
cd /path/to/twocrypto-cpp
uv sync --frozen --extra test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix "$PWD/_install"

cd /path/to/curve-fx-arb-harness
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/twocrypto-cpp/_install
cmake --build build --target arb_evaluator_ld --parallel

cd /path/to/curve-fx-optimization
uv sync --frozen --extra dev
```

For production policy runs, configure the harness with a concrete header from this repository and `POLICY_EXPECTED_SHA256`; see [`docs/workflows.md`](docs/workflows.md) and the harness build guide. The evaluator binary must be inspected with `fxsim harness identity` before use. A sibling path is a build/install prerequisite only; normal commands and all outputs stay under this repository.

## Repository-owned inputs

- `configs/pairs/` contains pair/feed identity specs (for example `chfusd`, `btcusd`); it does not duplicate pool-template economics.
- `configs/scenarios/` contains attested market inputs and the complete typed harness session settings, including the sole optional YieldBasis 2L releverage mode (`yb_releverage`, `yb_releverage_fee`, and `yb_cash_multiplier`).
- `configs/grids/` contains finite grid specs.
- `configs/optimization/` contains the source-backed TMRBCD spec `smoke-chfusd.toml`.
- `configs/policies/` is authoritative for compiled-policy parameter order, defaults, bounds, lattice steps, ABI, header, and source hash.
- `configs/sites/local.toml`, `smoke.toml`, and `blades.toml` inject execution topology without credentials in source.
- `data/manifest.toml` attests market inputs and deterministic fixtures. Production market files may be Git-LFS/private data.

Fetch authorized LFS data before a production run and verify it from the orchestrator root:

```sh
git lfs pull
uv run --frozen --no-sync fxsim data verify
```

A missing/private dataset is an access or provenance failure. This private repository does not imply redistribution or license rights for market data, fixtures, policies, or generated artifacts.

## Inspect evaluator identity

```sh
uv run --frozen --no-sync fxsim harness identity \
  /path/to/curve-fx-arb-harness/build/arb_evaluator_ld
```

The JSON identity includes binary/pool/harness/policy revisions or hashes, compiler and numeric mode, capabilities, metric fields, and protocol limits. Keep the output with the run manifest.

## Finite grids: local and blades

Generate an immutable request set. `--pair`, `--grid`, and `--scenario` accept IDs or TOML paths; `--harness` is required:

```sh
uv run --frozen --no-sync fxsim grid generate \
  --pair chfusd --grid chfusd-policy-smoke --scenario chfusd-smoke \
  --harness /path/to/curve-fx-arb-harness/build/arb_evaluator_ld \
  --run-id grid_chfusd_policy_smoke
```

The command creates the immutable `runs/grid_chfusd_policy_smoke/manifest.json`; its `grid.pools` section is the sole compiled request representation. Every point carries the complete dense policy vector; the pool template fixes pool economics and the scenario fixes typed session economics.

Run all points locally, or dispatch deterministic block-cyclic shards to the injected SSH blade profile:

```sh
uv run --frozen --no-sync fxsim grid run \
  runs/grid_chfusd_policy_smoke/manifest.json --site local \
  --harness /path/to/curve-fx-arb-harness/build/arb_evaluator_ld

uv run --frozen --no-sync fxsim grid run \
  runs/grid_chfusd_policy_smoke/manifest.json --site blades \
  --harness /path/to/curve-fx-arb-harness/build/arb_evaluator_ld
```

Use `--blades blade-b1 --blades blade-b2` to restrict SSH targets, `--chunk-size 2048` to set block size, and `--resume` to continue incomplete execution. Collect/validate a completed grid with:

```sh
uv run --frozen --no-sync fxsim grid collect \
  runs/grid_chfusd_policy_smoke/manifest.json
```

Collection rejects missing, overlapping, unknown, or hash-mismatched shards and writes `runs/grid_chfusd_policy_smoke/evaluation_table.json`.

## Adaptive optimization: local and distributed

Preflight a typed spec, then run TMRBCD locally:

```sh
uv run --frozen --no-sync fxsim optimize preflight \
  configs/optimization/smoke-chfusd.toml
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/smoke-chfusd.toml --run-id opt_chfusd_local
```

The same spec contract supports Nevergrad TwoPointsDE by setting
`algorithm = "nevergrad_two_points_de"`. The pinned adapter asks in batches,
quantizes every proposal onto the exact decimal lattice, checkpoints the
optimizer and pending batch synchronously, and restores deterministically.
TMRBCD remains the default for existing specs.

After the blade deployment in [`docs/workflows.md`](docs/workflows.md), run the same exact-lattice TMRBCD search through the SSH worker transport:

```sh
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/smoke-chfusd.toml --site blades \
  --blades blade-b6 --run-id opt_chfusd_blades
```

Omit `--blades` to use the complete site profile. Local and distributed runs share the same optimizer, scoring, checkpoint, and artifact contracts; the distributed manifest additionally records the remote evaluator SHA-256 identity.

Resume an incomplete run with `--resume`. Inspect and collect by run directory or manifest path:

```sh
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/smoke-chfusd.toml --run-id opt_chfusd_local --resume
uv run --frozen --no-sync fxsim optimize status runs/opt_chfusd_local
uv run --frozen --no-sync fxsim optimize collect runs/opt_chfusd_local
```

The run directory contains `manifest.json`, incremental `checkpoint.json`, `evaluation_table.json`, `winner.json`, and `topk.json`. The winner and top-k records are attested `SelectionRef` values; local and blade execution use the same scoring, dense policy request, and candidate identity.

## Ranking, heatmaps, and shiftclick

Rank one attested evaluation table:

```sh
uv run --frozen --no-sync fxsim analyze rank runs/grid_chfusd_policy_smoke \
  --metric apy --top 10
```

Render a parameter heatmap (default output is the run directory; explicit output is preferable for reports):

```sh
uv run --frozen --no-sync fxsim plot heatmap \
  runs/grid_chfusd_policy_smoke --metric apy --out runs/grid_chfusd_policy_smoke/heatmap.png
```

Create a TOML file under `configs/shiftclick/` (or `shiftclick/specs/`) with explicit `source_run_id`, `source_kind`, and selection. An optimizer winner uses `source_kind = "optimization"`, `selection_kind = "best"`; a grid point uses `source_kind = "grid"`, `selection_kind = "coordinates"`, and a non-empty coordinate table. Never use ambiguous grid `best`.

```toml
[shiftclick]
id = "chfusd-optimizer-winner"
source_kind = "optimization"
source_run_id = "opt_chfusd_local"
selection_kind = "best"
pair_id = "chfusd"
scenario_id = "chfusd-smoke"
policy_id = "native_policy_dual_ema_stale_cap_v1"
trace_interval = 1
trace_actions = true
```

```sh
uv run --frozen --no-sync fxsim replay shiftclick \
  configs/shiftclick/chfusd-optimizer-winner.toml \
  --harness /path/to/curve-fx-arb-harness/build/arb_evaluator_ld
```

For a heatmap point, replace the source/selection fields and give exact coordinates, for example:

```toml
[shiftclick]
id = "chfusd-grid-point-50-10"
source_kind = "grid"
source_run_id = "grid_chfusd_policy_smoke"
selection_kind = "coordinates"
selection_value = { fast_half_life_s = 1800, kappa = 0.5 }
pair_id = "chfusd"
scenario_id = "chfusd-smoke"
policy_id = "native_policy_dual_ema_stale_cap_v1"
```

Each replay creates `runs/shiftclick_{id}/manifest.json`, `trace/` sidecars, `replay_result.json`, and `economic_comparison.json`. It verifies the source row, full-trace economic fingerprint, sidecar hashes, and summary/full metric projection. Render its trajectory through the manifest-attested path:

For cluster replay, replace `--harness ...` with `--site blades --blades blade-b6`. Exactly one blade is required. The source run is staged in an isolated remote workspace, verified with the same shiftclick code, and downloaded into the same local run-artifact contract.

```sh
uv run --frozen --no-sync fxsim plot trajectory \
  runs/shiftclick_chfusd-optimizer-winner
```

## Ownership and dependency direction

`twocrypto-cpp` owns pool mechanics and parity; `curve-fx-arb-harness` owns one evaluator loop and `curve_fx_eval_v1`; `curve-fx-optimization` owns all user workflows, data provenance, scoring, execution, replay, and plots. Dependency direction is pool -> installed harness target -> orchestrator client. Run manifests record both build identities and economic inputs; observation level is not economic identity. Historical checkout names, copied binaries, generated runs, and `..` runtime paths are not supported.
