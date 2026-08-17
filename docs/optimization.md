# Adaptive optimization workflow

`curve_fx_sim.optimization` owns one compiled policy's exact decimal lattice, raw-metric scoring, TMRBCD, checkpoint/resume, and winner publication. The evaluator returns raw metrics; scoring and eligibility gates stay in Python. Both local and blade execution send the same complete dense `policy_params` vector.

## Configuration

An optimization TOML names a pair, a checked-in `PolicySpec`, scenarios, budget/batch size, scoring key, and an optional narrowing parameter table. The `PolicySpec` alone defines dense order, defaults, outer bounds, quanta, ABI, and header digest. Omitted names are fixed at their defaults; unknown or widened names fail preflight.

Quantization occurs in integer lattice ticks, then converts to binary64 only at the evaluator request boundary. Pool economics are not optimizer dimensions; the attested pool template owns them. A candidate's raw metrics are scored with the configured Python score key; failures use bounded sentinels rather than poisoning the optimizer.

`ScenarioSpec` owns the complete typed session configuration sent to `open_session`, including candle bounds/filtering, swap and keeper cadence, and the sole optional state-mutating YieldBasis 2L releverage mode. Set `yb_releverage=true`; configure only `yb_releverage_fee` and `yb_cash_multiplier`. `require_yb` remains a scoring gate and requires that mode in every scenario. Generic economic-default bags, obsolete YB modes, and scoring-level YB overrides are rejected.

## Run and resume

From `curve-fx-optimization/`, after building an evaluator in the location selected by the harness setup:

```sh
uv run --frozen --no-sync fxsim optimize preflight \
  configs/optimization/smoke-chfusd.toml
uv run --frozen --no-sync fxsim optimize run \
  configs/optimization/smoke-chfusd.toml --run-id opt_chfusd_local
```

The optimize command accepts the spec, `--run-id`, `--resume`, and `--output-root`; `--site blades` and repeated `--blades` select transport only, never a different optimizer or request translation.

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

TMRBCD searches discrete coordinate lines with deterministic lattice ticks and deterministic random restarts. A run's manifest records algorithm state, PolicySpec/lattice, evaluator identity, scenario/template/input hashes, candidate requests, score key, gate settings, and protocol version. Resume rejects any change in that identity. Blade names are execution metadata, not economic inputs.
