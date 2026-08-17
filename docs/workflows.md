# `fxsim` workflow and artifact guide

Run every command from `curve-fx-optimization/` after building an installed pool and an `arb_evaluator_ld` harness binary. The orchestrator is the only workflow CLI; pool and harness repositories provide build products only.

## Data and identity

```sh
git lfs pull
uv run --frozen --no-sync fxsim data verify
uv run --frozen --no-sync fxsim harness identity \
  /path/to/curve-fx-arb-harness/build/arb_evaluator_ld
```

`data/manifest.toml` lists expected paths and SHA-256 values. Keep the identity JSON and data-verification output with each run. Production market data may be private Git-LFS content; access, licensing, and redistribution are maintainer-controlled and are not inferred from this repository.

## Blade deployment

The blade profile expects the three repositories at `$HOME/arb/{twocrypto-cpp,curve-fx-arb-harness,curve-fx-optimization}` on shared NFS. Deploy or update those checkouts, then run the checked-in bootstrap once from a blade:

```sh
ssh blade-b6 '$HOME/arb/curve-fx-optimization/scripts/bootstrap_blade.sh $HOME/arb'
```

`scripts/bootstrap_blade.sh` pins nixpkgs, uv, and Python; builds and installs `twocrypto-cpp`; builds `$HOME/arb/bin/arb_evaluator_ld`; creates the frozen optimization environment; and emits `$HOME/arb/bin/fxsim-worker`. `configs/sites/blades.toml` names those deployed paths. Re-run the bootstrap after changing any repository or after replacing the shared workspace; cluster commands fail closed when either executable is absent.

## Grid lifecycle

```sh
uv run --frozen --no-sync fxsim grid generate \
  --pair chfusd --grid chfusd-policy-smoke --scenario chfusd-smoke \
  --harness /path/to/curve-fx-arb-harness/build/arb_evaluator_ld \
  --run-id grid_chfusd_policy_smoke
uv run --frozen --no-sync fxsim grid run \
  runs/grid_chfusd_policy_smoke/manifest.json --site local \
  --harness /path/to/curve-fx-arb-harness/build/arb_evaluator_ld
uv run --frozen --no-sync fxsim grid collect \
  runs/grid_chfusd_policy_smoke/manifest.json
```

Use `--site blades`, repeated `--blades NAME`, `--chunk-size N`, and `--resume` for SSH execution/resume. The generated run contains `manifest.json` with canonical `grid.pools`; execution adds `grid_results.json` and the single JSON `evaluation_table.json`. Collection validates exact candidate coverage and the shared `MetricProjection`.

## Optimization lifecycle

```sh
uv run --frozen --no-sync fxsim optimize preflight \
  configs/optimization/smoke-chfusd.toml
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/smoke-chfusd.toml --run-id opt_chfusd_local
uv run --frozen --no-sync fxsim optimize status runs/opt_chfusd_local
uv run --frozen --no-sync fxsim optimize collect runs/opt_chfusd_local
```

Pass `--resume` to continue only an identity-matching checkpoint. Use `--site blades --blades blade-b6` to run the same TMRBCD work bundles on one blade. The run directory contains `manifest.json`, `checkpoint.json`, `evaluation_table.json`, `winner.json`, and `topk.json`. `winner.json` is an optimizer-winner `SelectionRef`, not an executable request until normalized into a replay plan.

## Ranking and plots

```sh
uv run --frozen --no-sync fxsim analyze rank runs/grid_chfusd_policy_smoke --metric apy --top 10
uv run --frozen --no-sync fxsim plot heatmap \
  runs/grid_chfusd_policy_smoke --metric apy \
  --out runs/grid_chfusd_policy_smoke/heatmap.png
uv run --frozen --no-sync fxsim plot trajectory \
  runs/shiftclick_chfusd_optimizer_winner
```

Heatmaps read `evaluation_table.json`; trajectories read an attested trace selected from a shiftclick `manifest.json`.

## Shiftclick selections

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
  configs/shiftclick/chfusd_optimizer_winner.toml \
  --harness /path/to/curve-fx-arb-harness/build/arb_evaluator_ld
```

For one-blade replay, use `--site blades --blades blade-b6` instead of the local harness option. The command stages the source run in an isolated remote workspace, runs the same exact selection against the deployed compiled evaluator, verifies fingerprint/metrics/sidecars there, and downloads the attested shiftclick run.

For a grid point, use `source_kind = "grid"`, `selection_kind = "coordinates"`, and a non-empty `selection_value` mapping (or select an integer `index`/`candidate_id`); grid `best` is intentionally ambiguous and rejected. A shiftclick run is written to `runs/shiftclick_{id}/` with `trace/` sidecars, `replay_result.json`, `economic_comparison.json`, and `manifest.json`. Sidecar hashes and the summary/full economic fingerprint are checked before publication.
