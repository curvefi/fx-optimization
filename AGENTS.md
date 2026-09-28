# Standard YB heatmaps

- Use `bash scripts/heatmap-yb.sh CONFIG RUN_DIR [heatmap options...]` for YB
  grids. It follows/retrieves unfinished remote runs and opens completed local
  artifacts directly. Paths are relative to this repository unless absolute.
- Preserve the established three-column layout and metric order:
  1. `apy_net`, `yb_apy`, `yb_apy_gm`
  2. `apy_net_masked`, `yb_apy_masked`, `yb_apy_gm_masked`
  3. `max_7d_rel_price_diff`, `apy_net_robust_90d`, `avg_imbalance`
- Default axes are donation x RPF, price-divergence mask is 1500 bp (15%), and
  Shift-click uses `active_2l` with cash multiplier 3. The mask does not impose
  positive yields. Preserve this layout unless the user requests a change;
  adapt axes only when the grid lacks donation/RPF.
- When providing a run-local `heatmap-yb.sh`, make it a thin wrapper around the
  shared launcher with the exact config/run paths; do not invent another layout.

# Test policy

- For a bug fix, add one regression test. For a feature, add at most 3 test
  functions and 6 collected cases, using at most 150 test LOC. Get explicit
  user approval before authoring anything beyond those limits.
- Test one public invariant at the highest useful seam. Use one representative
  tamper table where a table adds coverage.
- Do not test private call counts, helper order, constructor signatures,
  `hasattr`, imports, help text, or implementation details. Assert exact errors
  only at CLI or protocol boundaries.
- Mock only external boundaries. For numerical behavior, state units, lattice,
  and reference values explicitly.
- Every added test must fail before the fix. Delete superseded tests.
- Obtain explicit approval before adding render, subprocess, real-environment,
  or cluster tests; before adding serial coverage taking over 1 second, record
  its timing and obtain approval. Keep focused runs under 2 seconds. Target a full serial time of <=16 seconds on the reference Mac.
  A single milestone run is evidence, not a regression verdict; require approval only for an attributable increase over 1 second in a like-for-like measurement.
  Do not rerun solely to adjudicate timing noise. Run full-suite validation only at milestones; do not enable xdist by default; keep real, native, and cluster tests outside the default run.
- Every subagent packet must state its test budget and prohibit routine
  validator chains or full-suite reruns after narrow edits.
