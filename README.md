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
uv sync --frozen --group dev
```

Build or select one evaluator artifact for the workflow. The artifact directory is the authority for the evaluator, policy, numeric mode, and parameter schema; verify it before generating a run:

```sh
artifact=/private/tmp/curve-fx-packet-b-f64.PyhpyD/artifact
uv run --frozen --no-sync fxsim evaluator build \
  --pool-root ../twocrypto-cpp \
  --harness-root ../curve-fx-arb-harness \
  --artifact-dir "$artifact" \
  --policy native_policy_dual_ema_stale_cap_v1 \
  --numeric-mode f64
uv run --frozen --no-sync fxsim evaluator verify "$artifact"
```

The configured Linux coordinator must build/select and launch the same grouped artifact mode for blade execution; this checkout does not claim live blade proof. A sibling repository is a build prerequisite only; normal commands and outputs stay under this repository.

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
uv run --frozen --no-sync fxsim evaluator verify "$artifact"
```

Keep the verification output with the run manifest. The selected artifact records the binary, source closures, policy, numeric mode, and canonical parameter schema.

## Finite grids: local and blades

Generate an immutable request set from the selected artifact. `--pair`, `--grid`, and `--scenario` accept IDs or TOML paths:

```sh
uv run --frozen --no-sync fxsim grid generate \
  --pair yb-weth --grid yb-weth-a-fee-donation-8 \
  --scenario yb-weth-ethusd-2024-latest \
  --artifact-dir "$artifact" \
  --run-id grid_yb_weth_a_fee_donation_8
```

The command creates an immutable manifest whose grouped candidate plans are compiled from the selected schema. The pool template supplies base economics and the scenario supplies base typed session inputs.

Run the artifact-selected manifest locally without a harness override:

```sh
uv run --frozen --no-sync fxsim grid run \
  runs/grid_yb_weth_a_fee_donation_8/manifest.json --site local
```

Use the configured coordinator/blade grouped transport only after the artifact has been build-selected there. Collect/validate a completed grid with:

```sh
uv run --frozen --no-sync fxsim grid collect \
  runs/grid_yb_weth_a_fee_donation_8/manifest.json
```

Collection rejects missing, overlapping, unknown, or hash-mismatched shards and writes the evaluation table beside the manifest.

## Adaptive optimization: local and distributed

Preflight and run the representative Nevergrad TwoPointsDE spec against the same selected artifact:

```sh
uv run --frozen --no-sync fxsim optimize preflight \
  configs/optimization/example-pool-dims.toml --artifact-dir "$artifact"
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/example-pool-dims.toml --artifact-dir "$artifact" \
  --run-id opt_example_pool_dims
```

The adapter asks in batches and quantizes proposals onto the exact schema lattice. TMRBCD remains available for existing specs.

Grouped blade execution must be launched from the configured Linux coordinator after selecting/building the artifact there; no live blade result is implied by these local commands. The coordinator owns grouped artifact selection and dispatch; local and distributed runs share candidate, scoring, and artifact contracts.

Resume an incomplete run with `--resume`. Inspect and collect by run directory or manifest path:

```sh
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/example-pool-dims.toml --artifact-dir "$artifact" \
  --run-id opt_example_pool_dims --resume
uv run --frozen --no-sync fxsim optimize status runs/opt_example_pool_dims
uv run --frozen --no-sync fxsim optimize collect runs/opt_example_pool_dims
```

The run directory contains `manifest.json`, incremental `checkpoint.json`, `evaluation_table.json`, `winner.json`, and `topk.json`. The winner and top-k records are attested `SelectionRef` values; local and blade execution use the same scoring, dense policy request, and candidate identity.

## Ranking, heatmaps, and shiftclick

Ranking and plots consume attested run artifacts. The shiftclick replay examples below retain the compatibility binary path; they do not replace the selected artifact workflow.

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

## Compatibility-only legacy mode

Direct `--harness` execution and a site `remote_binary_path` are compatibility paths for older binary-selected runs. They are not equivalent to the selected artifact authority and are not the canonical grouped workflow above.

## Ownership and dependency direction

`twocrypto-cpp` owns pool mechanics and parity; `curve-fx-arb-harness` owns one evaluator loop and `curve_fx_eval_v1`; `curve-fx-optimization` owns all user workflows, data provenance, scoring, execution, replay, and plots. Dependency direction is pool -> installed harness target -> orchestrator client. Run manifests record both build identities and economic inputs; observation level is not economic identity. Historical checkout names, copied binaries, generated runs, and `..` runtime paths are not supported.
