# Adaptive optimization workflow

`curve_fx_sim.optimization` owns schema-driven candidate compilation, Nevergrad TwoPointsDE, raw-metric scoring, checkpoint/resume, and winner publication. The evaluator returns raw metrics; scoring and eligibility gates stay in Python. Artifact-selected local and grouped execution use the same CandidateCompiler and SessionGroup contract.

## Configuration

An optimization TOML names a pair, policy, scenarios, budget/batch size, scoring key, and an optional narrowing parameter table. With `--artifact-dir`, the selected evaluator schema is the authority for canonical names, defaults, bounds, and quanta; unknown or widened names fail preflight.

Quantization occurs in integer lattice ticks, then converts to binary64 only at the evaluator request boundary. The scenario template supplies base pool economics, while canonical candidate pool overrides may narrow selected fields. A candidate's raw metrics are scored with the configured Python score key; failures use bounded sentinels rather than poisoning the optimizer.

Numeric policy parameters, candidate pool overrides, and numeric `open_session`/run parameters can be searched. Observation, transport, path, and alias fields cannot. `ScenarioSpec` owns the base session and scenario inputs; `parameter_space` narrows only searchable schema dimensions.

`ScenarioSpec` owns the complete typed session configuration sent to `open_session`, including candle bounds/filtering, swap and keeper cadence, and the sole optional state-mutating YieldBasis 2L releverage mode. Set `yb_releverage=true`; configure only `yb_releverage_fee` and `yb_cash_multiplier`. `require_yb` remains a scoring gate and requires that mode in every scenario. Generic economic-default bags, obsolete YB modes, and scoring-level YB overrides are rejected.

## Run and resume

From `curve-fx-optimization/`, select and verify one artifact, then preflight and run the representative spec:

```sh
artifact=/private/tmp/curve-fx-packet-b-f64.PyhpyD/artifact
uv run --frozen --no-sync fxsim evaluator verify "$artifact"
uv run --frozen --no-sync fxsim optimize preflight \
  configs/optimization/example-pool-dims.toml --artifact-dir "$artifact"
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/example-pool-dims.toml --artifact-dir "$artifact" \
  --run-id opt_example_pool_dims
```

`example-pool-dims.toml` uses Nevergrad TwoPointsDE with three canonical dimensions. The optimize command accepts the spec, `--run-id`, `--resume`, and `--output-root`; transport options do not change candidate translation. Grouped blade artifact mode must be launched/build-selected on the configured Linux coordinator; no live blade proof is claimed here.

`--resume` reopens `checkpoint.json` only when immutable identity (pair, policy, score key, lattice/spec closure, evaluator identity, and scenario inputs) matches. Do not edit a run in place:

```sh
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/smoke-chfusd.toml --run-id opt_chfusd_local --resume
uv run --frozen --no-sync fxsim optimize status runs/opt_chfusd_local
uv run --frozen --no-sync fxsim optimize collect runs/opt_chfusd_local
```

## Artifact contract

A run is immutable below `runs/opt_chfusd_local/` (or the explicit run ID supplied to the command):

- `manifest.json`: `fxsim_manifest_v1` identity, resolved spec, execution history, and artifact hashes;
- `checkpoint.json`: incremental optimizer state, deterministic restart state, evaluated rows, and complete resume identity;
- `evaluation_table.json`: canonical raw results and the exact `MetricProjection`;
- `winner.json`: one `SelectionRef` with `kind = "optimizer_winner"` and policy/scenario lineage;
- `topk.json`: ranked `SelectionRef` records with the same lineage.

Both `winner.json` and `topk.json` are normalized through `SelectionRef -> ReplayPlan`; shiftclick never infers a candidate from row order. The subsequent full-trace replay lives in a separate `runs/shiftclick_chfusd_optimizer_winner/` directory and must echo the source economic fingerprint.

Each optimization row keeps the primary scenario's raw evaluator metrics plus the Python score and loss fields. Row status records evaluator success; the independent scoring `gate` stays in the metrics. This separation lets an optimizer winner be replayed exactly even when a diagnostic eligibility gate is false.

## Algorithms and audit identity

Nevergrad TwoPointsDE searches the exact schema lattice in batches. A run's manifest records algorithm state, schema/lattice, evaluator identity, scenario/template/input hashes, candidate requests, score key, gate settings, and protocol version. Resume rejects any change in that identity. Blade names are execution metadata, not economic inputs.

TMRBCD remains available for existing specifications, but is not the representative artifact-selected path.

## Compatibility-only legacy mode

Direct `--harness` execution and site `remote_binary_path` are compatibility-only binary-selected paths. They do not provide the selected artifact schema authority and are not equivalent to grouped execution.
