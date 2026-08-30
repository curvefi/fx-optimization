# Exhaustive stable-pool grid-search workflow

This is the research playbook for finding a parameter **plateau** on one pinned
candlestick dataset. It is deliberately grid-first: broad Cartesian scans find
mechanism families, dense lower-dimensional scans localize them, and exact
physical perturbations reject fragile point winners.

The normal order is:

```text
campaign contract
-> broad blade-f64 / YB-off discovery
-> controller and economics localization
-> fine fee surface
-> exact axial stars
-> interaction boxes
-> blade-LD, Mac-f64, arb-cost, and time-fold gates
-> active_2l YB fee/donation study
-> ShiftClick diagnostics and final report
```

Do not start with YB, long double, a 1-bps fee lattice, or a full Cartesian
robustness box. Earn those expensive stages by first finding a connected native
region.

## Agent quick start

For a new pair or policy, an agent should do exactly this:

1. Read this file, the repository `README.md`, the current `fxopt ... --help`,
   and `../curve-fx-arb-harness/protocol/protocol_spec.md`.
2. Inspect all three repository worktrees and preserve unrelated changes.
3. Create `configs/autoresearch/<pair>-optimization/<policy>/report.md` and a
   numbered `00-...toml` source grid.
4. Freeze the objective, hard constraints, units, perturbation radii, dataset,
   template, policy, arb cost, and wall-time budget in the report **before** the
   first result.
5. Parse the exact TOML and print its cardinality before giving or running a
   cluster command.
6. After every run, validate the two-file artifact, run the heatmap and basin
   analysis, append the conclusion, and design only the next uncertainty-
   reducing grid.
7. Never promote a center until its complete physical neighborhood passes every
   hard constraint on the required numeric modes and cost assumptions.

The user normally runs cluster commands. An agent may launch, stop, overwrite,
or retrieve a cluster job only when the user explicitly authorizes that action.

## 1. Bind the current authority

The active stack is:

```text
twocrypto-cpp          pool and policy mechanics
curve-fx-arb-harness   event loop, actors, sizing, metrics, evaluator
curve-fx-optimization  grids, cluster execution, artifacts, analysis, replay
```

Do not borrow manifests, commands, YB modes, or result schemas from
`cpp-twocrypto-modular`. A completed `run.json` plus `results.npz` is the durable
authority for a run; the source TOML under ignored `configs/autoresearch/` is
only a convenient recipe.

There is currently no supported `fxopt optimize` or Nevergrad surface. Use the
registered-grid runner for exhaustive work. Do not add optimizer state or an
ask/tell loop merely to complete an ordinary grid campaign.

Use a campaign layout like:

```text
configs/autoresearch/btc-optimization/dual-ema/
    00-source-highd-f64.toml
    01-region-a-controller-f64.toml
    02-region-b-economics-f64.toml
    ...
    report.md
    plots/
```

Mirror the config purpose in the run name. Number configs in causal order and
never recycle a number for a different experiment.

## 2. Freeze the campaign contract

Write these fields at the top of `report.md`:

```text
dataset path and covered timestamps
pool template and compiled-policy identity
fixed parameters and searchable parameters
numeric mode, YB mode, metric profile, event cursor, and slippage setting
arb cost and any volume/gas assumptions
hard fee, donation, attachment, and detachment constraints
ranking metric and promotion reducer
physical perturbation radii
per-run wall-time budget
```

Separate three concepts:

- **Hard constraints** decide feasibility, for example fee <= 150 bps,
  donation < 3%, or max 7-day price difference <= 15%.
- **Point ranking** orders individual feasible rows.
- **Promotion ranking** orders complete neighborhoods by their worst member.

Do not silently change any of them. If the user changes the attachment gate,
adds a minimum terminal APY, or replaces consistency-first ranking with a
balanced APY/GM score, create a named campaign branch in the report. Retain the
old winner under its original objective.

Useful objective lanes are:

```text
consistency-first: apy_net_robust_90d, then lp_detach_score
raw-yield gate:    require minimum apy_net, then maximize neighborhood floor
balanced earnings: sqrt(max(apy_net,0) * max(apy_net_gm,0))
YB consistency:    yb_apy_gm, with LP and attachment constraints independent
```

The balanced formula is a campaign reducer, not a license to hide either raw
metric. If an objective is not supported by `analyze_basins.py`, derive it from
the stored result columns in a campaign-local analysis and document the formula.

### Units checklist

Record exact wire values as well as human units. Common conversions are:

```text
100 bps pool fee       -> 0.0100
1.5% donation APY      -> 0.015
2.4% YB AMM fee        -> 0.024
A = 5                  -> 50000
8 bps cap/deadband     -> 0.0008
EMA times              -> integer seconds
RPF and kappa          -> ordinary ratios
```

Never compare a display label to a wire value by eye.

## 3. Budget and validate every grid

Use measured throughput from the latest run with the same policy, numeric mode,
YB mode, metric profile, event cursor, and fleet:

```text
candidate budget = measured pools/s * usable calculation seconds
```

Allow for startup, market loading, and result merging. A no-YB native-policy
rate says nothing about a compiled-policy or YB-enabled run.

Before launch, state:

```text
axis names, ranges, quanta, and point counts
Cartesian cardinality
evaluator target and compiled policy
YB/profile/cursor/slippage modes
host and NUMA count
estimated calculation and wall time
output directory
```

Parse the exact config:

```sh
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync python - <<'PY'
from pathlib import Path
from fxopt.run import RunConfig, grid_summary

path = Path("configs/autoresearch/CAMPAIGN/00-source.toml")
config = RunConfig.from_toml(path)
grid = config.candidate.grid()
print(grid_summary(path))
print(config.run_id, config.evaluator)
print(config.scenario, config.session)
print(grid.shape, len(grid), config.metric_fields)
PY
```

Candidate defaults must contain exactly `policy_params` and `pool`. Use a linked
axis for a flat fee so mid and out fees move together:

```toml
[candidate.axes]
flat_fee = {
  start = 0.0100,
  stop = 0.0200,
  step = 0.0001,
  targets = ["pool.mid_fee", "pool.out_fee"],
}
```

An established config-only research iteration needs this parse gate, not new
unit tests. Use a deliberately small production smoke grid only when introducing
a new evaluator/profile/protocol surface.

## 4. Search stages and go/no-go gates

| Stage | Grid purpose | Advance only when |
|---|---|---|
| A | Broad high-dimensional discovery | At least one feasible mechanism family exists |
| B | Separate controller/economics regions | Promising intervals are connected and interpretable |
| C | Resolve controller boundaries | Important controller axes have interior values |
| D | Fine fee/economics surface | Fee/economic plateau has exact physical neighbors |
| E | Axial-star screening | Every required arm passes hard constraints |
| F | Interaction box | Simultaneous perturbation corners pass |
| G | Numeric/cost/time validation | Required modes, costs, and folds preserve feasibility |
| H | YB promotion | LP and YB metrics independently remain acceptable |

### Stage A: broad blade-f64, YB-off discovery

Use:

```text
arb_evaluator_f64
yb_mode = "off"
event_cursor = "exact_skip"
metric_profile = "grid_core"
slippage disabled
```

Give genuinely uncertain axes roughly 6–10 values. Prefer wide meaningful
ranges over premature density. Retain at least:

```text
apy_net
apy_net_robust_90d
max_7d_rel_price_diff
detach_energy_ungated
trades
n_rebalances
elapsed_ms
```

The source grid answers which controller families and economic regimes can
work. It does not produce a deployable point.

### Stage B: split distant regions and mechanism blocks

Advance two to four distinct regions, including a broad lower-scoring region
when the raw winner is isolated. Use a separate config/artifact for each distant
family instead of one huge rectangle spanning empty space.

Decompose dimensions by mechanism:

```text
controller block: fast EMA, slow EMA, kappa, deadband, min cap, max cap
economics block:  flat fee, donation APY, RPF, optionally A
nuisance block:   pool MA time, adjustment limits, other pair-specific inputs
```

Donation and RPF are coupled and normally belong in the same grid. It is often
efficient to fix recurring coarse-grid economics while localizing the
controller, then reopen a small economics neighborhood around the surviving
controller families.

Use 16–32 points only on the three to five axes currently reducing uncertainty.
Do not repeatedly scan every dimension at high density.

### Stage C: resolve boundaries before densifying fee

If a promising set hits a searchable controller boundary, expand only that
boundary and rerun a compact grid. Do not spend 101 one-bps fee cells on a
controller whose kappa, EMA, deadband, or cap is still unresolved.

Special cases:

- `kappa = 1` removes the slow-EMA contribution. Treat that family as
  single-EMA and stop wasting a slow-EMA axis.
- A winner at a hard fee ceiling is evidence of pressure against the constraint,
  not permission to widen it.
- A boundary may remain fixed when it is an external deployment requirement;
  say so explicitly rather than calling it an interior optimum.

### Stage D: fine fee and economics surface

Once controller families are interior, scan fee at its deployment quantum and
cross the local donation/RPF/A neighborhoods. Add guard values beyond the
nominal reporting ceiling when they are needed to measure a two-sided physical
radius; exclude guard-only centers from promotion.

For example, a nominal fee ceiling of 200 bps with +/-2 bps robustness needs
axis values through 202 bps so the 200-bps center has a complete star.

## 5. Analyze every completed artifact

First verify complete typed coverage:

```sh
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync python - <<'PY'
from pathlib import Path
from fxopt.results import read_result_columns

root = Path("runs/RUN")
result = read_result_columns(root, metrics=())
print("rows", result.row_count)
print("ok", int(result.ok_mask.sum()))
print("failed", len(result.failures))
print("metrics", result.available_metrics)
PY
```

Then inspect masked earnings beside the raw mask sources:

```sh
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync fxopt heatmap runs/RUN \
  --metric apy_net_masked \
  --metric apy_net_robust_90d_masked \
  --metric detach_energy_ungated \
  --metric max_7d_rel_price_diff \
  --max-price-diff-bps PRICE_LIMIT_BPS \
  --columns 2
```

Add repeatable `--log-axis AXIS` for positive numeric axes such as EMA time. Do
not use it for a linked categorical axis.

Run hard filters before ranking:

```sh
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync python scripts/analyze_basins.py runs/RUN \
  --rank apy-net-robust \
  --max-price-diff-bps PRICE_LIMIT_BPS \
  --max-fee-bps FEE_CEILING_BPS \
  --top 100
```

Available rank modes are `lp-score`, `score`, `gm`, `yb-gm`, and
`apy-net-robust`. `--max-fee-bps` is inclusive.

For each result report:

```text
run identity, modes, cardinality, statuses, and throughput
exact heatmap and analyzer commands
hard-filter survivor count and fraction
best point and best complete robust center
point-to-neighborhood regret
promising-set ranges for every axis
boundary hits and irrelevant dimensions
trade/rebalance regime changes
observed conclusion, separate from mechanism hypotheses
next grid cardinality, time estimate, and uncertainty it resolves
```

A bright cell on one heatmap slice is not a basin.

## 6. Physical robustness: axial first, interactions second

Robustness is defined in physical units, not “one current grid cell.” Store the
radii in the config so `run.json` preserves them:

```toml
[robustness]
flat_fee = { field = "pool.mid_fee", radius = 0.0002 } # +/-2 bps
"policy_params.2" = 0.02                              # +/-0.02 kappa
```

For a linked axis, `field` identifies the numeric component. The axis must be
finite and monotonic and must contain exact `center - radius` and
`center + radius` values. The analyzer includes **every lattice value between
those endpoints** in the arm. Thus a 1-bps fee grid with a +/-2 bps radius adds
four fee members, not two.

### Axial-star screening

An axial star is:

```text
center
+ every value on the negative and positive interval of each varied axis
```

It is cache-efficient and suitable for screening many centers. A center is
incomplete if an arm is missing or any required member fails a hard filter.
Never interpolate a missing arm and never rank an incomplete center.

The BTC campaign used the following operator-approved example radii:

```text
flat fee       +/-2 bps
A              +/-0.1
donation APY   +/-0.05 percentage points
RPF            +/-0.01
fast EMA       +/-10 seconds
slow EMA       +/-60 seconds
kappa          +/-0.02
deadband       +/-1 bps
min/max cap    +/-1 bps
```

These are campaign inputs, not universal defaults. Use the policy quantum and
the deployment uncertainty that the user actually wants to tolerate.

### Full interaction boxes

Axial safety does not prove simultaneous-perturbation safety. In the BTC
campaign, high-scoring axial winners reached 25–32% max 7-day detachment in
combined corners and were correctly rejected.

For each finalist, evaluate a three-level Cartesian box over axes likely to
interact:

```text
center - radius, center, center + radius
```

`3^7 = 2,187` rows is cheap in a no-YB finalist stage. If a full box over all
axes is unaffordable, split it into declared interacting blocks and call the
result partially validated; do not relabel an axial-only candidate fully
robust.

### Recovery when a box fails

Do not immediately loosen the constraint. Instead:

1. Record the exact worst ordinal, corner coordinates, and failed metric.
2. Identify which simultaneous changes crossed the action/rebalancing regime.
3. Screen the source surface for centers with more attachment margin.
4. Search inward along the failing axes and retain alternate controller
   families.
5. Re-run the same box unchanged around the replacement center.

A near miss remains useful evidence but is not promoted.

## 7. Efficient finalist transport

After selecting centers, stop re-running their source grids. Encode a small
explicit linked axis such as `star_point = [{...}, {...}]` containing the center
and every required arm. Use the identical ordered list for:

```text
blade f64
blade long double
Mac f64
arb-cost sensitivity
YB finalist checks
```

Keep the center first and document the arm order. Join runs by canonical
ordinal or reconstructed payload, never NPZ row order. `analyze_basins.py`
cannot infer physical-star semantics from a custom linked `star_point` axis, so
compute the known group floors and ceilings explicitly.

This small-star form is also useful when one axis is an external scenario input,
such as arb cost, rather than a parameter being optimized.

## 8. Native promotion gates

### Numeric/platform gate

Use blade f64 for discovery and x86-64 blade long double for finalist ranking.
Replay the same explicit star on Mac f64 as an independent trajectory-stability
check. Apple ARM `long double` is binary64 and is not production extended-
precision evidence.

Compare:

```text
hard-feasibility flips
point and star floors
worst max 7-day price difference and detach energy
trades and rebalances
worst mismatching arm
```

Ignore `elapsed_ms`. Sparse platform forks near decision thresholds can be
economically material even when global rank correlation is high.

### Arb-cost gate

Apply every required arb-cost assumption to the **entire physical star**, not
only the center. Cost sensitivity is path-dependent and need not be monotonic.
A candidate that passes at 2 and 5 bps may fail at 4 bps.

For every cost report:

```text
center score
star floor
center max-7d difference
star max-7d difference
failing arm, if any
trade and rebalance counts
```

If one cost exposes a controller boundary, either retune jointly across costs
or narrow the stated operating assumption. Do not interpolate a pass.

### Candlestick folds

Re-evaluate the unchanged shortlist on subperiods chosen before viewing their
results: low volatility, high volatility, trends, and drawdowns. Hard
constraints must pass in every required fold. Use a declared worst-fold or
lower-quantile reducer; do not optimize a different policy for each fold and
call the collection stable.

## 9. YieldBasis promotion and donation regime

YB is a secondary gate after native convergence. Use:

```text
metric_profile = "full_summary"
event_cursor = "scalar"
yb_mode = "active_2l"
slippage disabled unless explicitly needed
yb_cash_multiplier stated explicitly
```

Keep the native candidate fixed and scan the YB AMM fee through the candidate
override `pool.run.yb_releverage_fee`. A useful first pass is a coarse bounded
fee curve; densify only the best interior region.

Retain and report at least:

```text
yb_apy
yb_apy_gm
yb_final_growth
yb_gm_floored_windows and yb_gm_floor_share
yb_releverage_trades
apy_net and apy_net_gm
max_7d_rel_price_diff and detach_energy_ungated
donations and donation_coin0_total
trades and n_rebalances
```

Terminal YB APY and YB APY GM answer different questions. If terminal APY wins
at the fee boundary while GM is near zero, report both and do not call the
terminal rate consistent. Prefer an interior fee with a complete physical fee
neighborhood unless the boundary is an explicit allowed constraint optimum.

In enabled YB modes, the ordinary periodic native-donation scheduler is
disabled. `pool.donation_apy` initializes the YB interest/donation target, and
donations occur as causal sublegs of YB routes. Check whether
`donations == yb_releverage_trades`; do not assume every attempted or committed
route necessarily donated. Never add LP APY and YB APY without a defined
consolidated accounting boundary.

Use `reference_2l` only for a small final model-sensitivity sample. Neither
enabled YB mode is proven historical or onchain parity.

## 10. ShiftClick the exact stored candidate

Replay by source ordinal from the completed artifact:

```sh
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync fxopt shiftclick runs/RUN \
  --ordinal ORDINAL \
  --output runs/RUN/inspections/ordinal-ORDINAL
```

ShiftClick preserves the run's YB mode, fee, cash multiplier, session, and exact
candidate payload. Right-click in the explorer intentionally replays with YB
off; Shift-click uses the stored mode.

The replay is local. Its plot is a path diagnostic, not blade-LD ranking
evidence. Put canonical blade metrics and local replay metrics side by side when
they differ.

If the local client rejects the evaluator hello or its compiled-policy identity
is stale, rebuild the existing local Twocrypto install and the exact evaluator
target named by `run.json`; do not edit the artifact or reconstruct the
candidate manually.

## 11. Cluster lifecycle

First prepared run for an evaluator target:

```sh
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync fxopt run CONFIG \
  --output runs/RUN --transfer --rebuild
```

Warm subsequent run with unchanged staged sources/target:

```sh
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync fxopt run CONFIG --output runs/RUN
```

Detached job controls use the same config and output:

```sh
uv run fxopt run CONFIG --output runs/RUN --status
uv run fxopt run CONFIG --output runs/RUN --follow
uv run fxopt run CONFIG --output runs/RUN --retrieve
uv run fxopt run CONFIG --output runs/RUN --stop
```

Use `--overwrite` only for an existing completed or empty output directory. It
refuses a live `.remote-job.json`. Interrupted grids restart from zero; they are
not resumable.

The first placement host is the coordinator. Blade `/tmp` is node-local even
though `/home/heswithme` is shared. The Mac receives progress/control traffic
and the final merged `run.json`/`results.npz`, not per-candidate result streams.

## 12. Convergence and reporting contract

A candidate is ready for promotion only when:

1. Its objective branch and hard constraints are explicit.
2. Important searchable axes are interior or externally fixed.
3. Its complete axial star passes every hard constraint.
4. Its declared interaction box passes simultaneous perturbations.
5. Its neighborhood score is competitive, not merely its center.
6. Required blade-LD, Mac-f64, arb-cost, and time-fold gates pass.
7. Nearby points preserve a comparable action regime.
8. YB metrics pass independently when YB is in scope.
9. The report names stable intervals, worst members, and remaining fidelity
   limitations.

Otherwise call it a point winner, promising region, partial star, near miss, or
candidate plateau—whichever is actually proven.

Append this compact block after every run:

```text
Config and artifact
Objective branch and hard gates
Modes, cardinality, statuses, calculation/wall time
Exact heatmap and analyzer commands
Eligible rows and complete neighborhoods
Best point, best robust center, floor, ceiling, worst ordinal
Promising intervals and boundary hits
Trade/rebalance regime observations
Observed conclusion
Next grid and the uncertainty it removes
```

## Agent rules

- Lead with artifact-backed observations and label mechanism explanations as
  inference.
- Keep hard constraints, point score, and promotion reducer separate.
- Prefer a connected lower plateau over an isolated higher needle.
- Never accept incomplete robustness or infer safety between tested costs.
- Preserve raw earnings, attachment, detachment, donation, trade, and rebalance
  metrics alongside scores.
- Join artifacts by ordinal or exact payload, never NPZ row order.
- Never infer throughput from another policy, numeric mode, profile, or YB
  mode.
- Never propose a grid without parsed cardinality and measured time estimate.
- Keep LLM configs under `configs/autoresearch/` and run artifacts uncommitted.
- Do not modify simulator mechanics to explain a research result; route changes
  to the repository that owns them.
- At each iteration, update the report before designing the next grid.
