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
