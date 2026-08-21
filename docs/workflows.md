# `fxsim` workflow and artifact guide

Run every command from `curve-fx-optimization/`. The orchestrator is the only workflow CLI; pool and harness repositories provide build inputs for one selected evaluator artifact.

## Data and artifact identity

```sh
git lfs pull
uv run --frozen --no-sync fxsim data verify
artifact=/private/tmp/curve-fx-packet-b-f64.PyhpyD/artifact
uv run --frozen --no-sync fxsim evaluator build \
  --pool-root ../twocrypto-cpp \
  --harness-root ../curve-fx-arb-harness \
  --artifact-dir "$artifact" \
  --policy native_policy_dual_ema_stale_cap_v1 \
  --numeric-mode f64
uv run --frozen --no-sync fxsim evaluator verify "$artifact"
```

`data/manifest.toml` lists expected paths and SHA-256 values. Keep artifact verification and data-verification output with each run. Production market data may be private Git-LFS content; access, licensing, and redistribution are maintainer-controlled and are not inferred from this repository.

## Grouped blade deployment

Grouped blade artifact mode must be build-selected and launched on the configured Linux coordinator. The coordinator owns artifact selection and grouped dispatch; do not infer live blade proof from local artifact runs. The existing site/bootstrap files describe deployment inputs, but remain separate from artifact authority.

## Grid lifecycle

```sh
uv run --frozen --no-sync fxsim grid generate \
  --pair yb-weth --grid yb-weth-a-fee-donation-8 \
  --scenario yb-weth-ethusd-2024-latest \
  --artifact-dir "$artifact" \
  --run-id grid_yb_weth_a_fee_donation_8
uv run --frozen --no-sync fxsim grid run \
  runs/grid_yb_weth_a_fee_donation_8/manifest.json --site local
uv run --frozen --no-sync fxsim grid collect \
  runs/grid_yb_weth_a_fee_donation_8/manifest.json
```

The generated manifest records grouped candidate plans compiled from the selected schema. Run it locally with the selected artifact; grouped blade execution requires the configured Linux coordinator and its selected artifact.

## Optimization lifecycle

```sh
uv run --frozen --no-sync fxsim optimize preflight \
  configs/optimization/example-pool-dims.toml --artifact-dir "$artifact"
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/example-pool-dims.toml --artifact-dir "$artifact" \
  --run-id opt_example_pool_dims
uv run --frozen --no-sync fxsim optimize status runs/opt_example_pool_dims
uv run --frozen --no-sync fxsim optimize collect runs/opt_example_pool_dims
```

The example uses Nevergrad TwoPointsDE. Pass `--resume` only for an identity-matching checkpoint. The run directory contains the manifest, checkpoint, evaluation table, winner, and top-k selections. The same CandidateCompiler and SessionGroup execution contract is shared by grid and optimization.

## Ranking and plots

```sh
uv run --frozen --no-sync fxsim analyze rank runs/grid_chfusd_policy_smoke --metric apy --top 10
uv run --frozen --no-sync fxsim plot heatmap \
  runs/grid_chfusd_policy_smoke --metric apy \
  --out runs/grid_chfusd_policy_smoke/heatmap.png
uv run --frozen --no-sync fxsim plot trajectory \
  runs/shiftclick_chfusd_optimizer_winner
uv run --frozen --no-sync fxsim view \
  runs/grid_chfusd_policy_smoke --metrics apy \
  --out runs/grid_chfusd_policy_smoke/view.png
```

`fxsim view RUN_DIR` opens separate Heatmaps, Controls, and Metrics windows. Use
`--out` for a PNG export and adjacent state sidecar; export is immutable and
does not open a direct-binary path. Plain clicks update the
metrics window. Shift-click replays the exact table cell with its attested
source YieldBasis mode and strict economic comparison. Right-click selects the
same `SelectionRef`, disables YieldBasis, and records a sparse counterfactual
trace targeting roughly 10,000 observations. Both use the selected artifact
locally or a configured `--site`/`--blade` remotely.
Heatmaps read the attested `evaluation_table.npz`; trajectories read an attested
trace selected from a shiftclick `manifest.json`.

## Shiftclick selections

The replay examples below use the selected evaluator artifact.

Create the directory and a TOML under `configs/shiftclick/` with a required `source_run_id`:

```toml
[shiftclick]
id = "chfusd_optimizer_winner"
source_kind = "optimization"
source_run_id = "opt_chfusd_local"
selection_kind = "best"
pair_id = "chfusd"
scenario_id = "chfusd-smoke"
policy_id = "native_policy_dual_ema_stale_cap_v1"
trace_actions = true
```

Run it with:

```sh
uv run --frozen --no-sync fxsim replay shiftclick \
  configs/shiftclick/chfusd_optimizer_winner.toml
```

For one-blade replay, use `--site blades --blades blade-b6`. The command stages the source run in an isolated remote workspace, runs the same exact selection against the deployed compiled evaluator, verifies fingerprint/metrics and the packed NPZ/companion there, and downloads the attested shiftclick run.

For a grid point, use `source_kind = "grid"`, `selection_kind = "coordinates"`, and a non-empty `selection_value` mapping (or select an integer `index`/`candidate_id`); grid `best` is intentionally ambiguous and rejected. A shiftclick run is written to `runs/shiftclick_{id}/` with `replay_trace.npz`, `replay_trace.json`, `replay_result.json`, `economic_comparison.json`, and `manifest.json`. Private evaluator JSON staging is removed after packing; packed hashes and the summary/full economic fingerprint are checked before publication.
