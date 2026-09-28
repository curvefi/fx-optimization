# Sequential price-scale policy search — 2026-09-17

## Objective and constraints

Find about ten diverse price-scale policies with round-number fast/slow EMA half-lives, kappa to 0.01, and tolerable sensitivity to neighboring parameters, donation/RPF, and reporting. Do not present isolated maxima or boundary points as robust centers. Keep fewer than ten if evidence does not support ten.

- Initial campaign ceiling: **2,000,000 candidate evaluations**. Record actual counts below; no implicit large Cartesian expansion.
- **Current bounds: base fee >=30 bp (50 bp reference), capture strictly >80%, donation strictly <3%, RPF strictly <0.5.** Reporting probability is distinct from capture. Latest user instruction authorizes joint base/capture/fallback optimization; this supersedes the earlier fixed-base restriction.
- Discovery uses fresh reports, f64, `active_2l`, cash multiplier 3, YB fee 1.2%, mixed 30-second depth. Same 2024-02-15 through 2026-08-20 window, excluding 2025-10-10 and 2025-10-11, as the preceding grids. Pyth staleness is not included.
- Default fallback 300 bp. No simulation code changes or source transfers. Remote evaluator SHA256 `1e15bd2cbcbd01d201b3bc57a9d9a43ddffdd239822fd8e98d1a7f6f18ff1fb5` verified before launch.
- Retain distinct families through a capture sensitivity screen before narrowing. Refine EMA/kappa, then steps/deadband, then recheck joint neighborhoods and economic/report surfaces.
- Preliminary passing criterion: YB GM >0.1%, maximum 7-day relative price difference <15%, positive terminal YB and net pool APY. Rank scenario coverage and neighboring-parameter performance; keep a few scenario-specific alternatives because sparse anchors can miss good regions.
- Existing policy limits maximum step to 60 bp and minimum fast half-life to 600 seconds. Later dense step sweep uses 60 bp instead of the initially suggested 75 bp.

## Evaluation ledger

| Stage | Points | Status |
|---|---:|---|
| S1: round EMA/kappa discovery | 619,632 | Complete; all successful |
| Early capture sensitivity | 6,480 | Complete; all successful |
| S2: local EMA/kappa refinement | 71,064 | Complete; all successful |
| S3: dense steps/deadband | 45,000 | Complete; all successful |
| S4: joint fee/economic surfaces | 680,400 | Complete; all successful |
| S4b: prior-controller comparison | 20,250 | Complete; all successful |
| S5: local fee/economic stability | 17,982 | Complete; all successful |
| S6: joint driver/economic refinement | 150,903 | Complete; all successful |
| S7: fee/economic recheck | 6,804 | Complete; all successful |
| S8: retuned f64 + report stress | 60 | Complete; all successful |
| S7b: matched parent-driver recheck | 6,804 | Complete; all successful |
| S8: parent f64 / paired LD / paired reference | 300 | Complete; all successful |
| S9: local regions across f64 / LD / reference | 2,025 | Complete; all successful |
| S10: final local sensitivity + fee-box validation | 2,319 | Complete; all successful |

Completed campaign: **1,630,023 evaluations**. Unused budget: **369,977**. All campaign runs are complete; no further grid is queued.

## S1 — discovery

Config: `configs/experiments/ethusd-pscale-sequential-s1-discovery-619632-f64-20260917.toml`.
Output: `runs/ethusd-pscale-sequential-s1-discovery-619632-f64-20260917`.
Coordinator: `blade-a5:/tmp/fxopt-grid.zuhmL22I`, 15 hosts, 606 shuffled leases.

- Fast: 10–120 minutes every 5 minutes.
- Slow: 1/2/3/4 hours, then 8–48 hours every 4 hours.
- Only fast < slow: **331 EMA pairs**.
- Kappa 1.00–1.50 every 0.02: **26**.
- Coupled min/max steps: **5/25, 5/50, 10/25, 10/50 bp**.
- Deadband: **0, 5, 15 bp**.
- Coupled donation/RPF anchors: **1.6%/0.28, 2.0%/0.30, 2.8%/0.30**.
- Reporting: **50%, 100%**. Capture **95%**.
- Cardinality checked: 331 × 26 × 4 × 3 × 3 × 2 = **619,632**. EMA durations explicitly integer seconds.

## Next decision gate

Validate complete ordinal coverage and all result statuses. Select up to 30 diverse EMA/kappa modes using scenario coverage plus fixed-step/deadband neighbor performance. Immediately screen those modes at capture **85/90/95%**, retaining the 12 step/deadband combinations and six economic/report scenarios: **6,480 points** if 30 modes.

Only after observing this screen, choose local refinements. Report 0% is a later shortlist stress, not another dimension of the full discovery grid. Favor exact round values during evaluation, never post-hoc rounding of a measured winner. No final policies are selected yet.

## S1 result and S1b decision

Completed in 761.5 seconds (813.7 points/s wall), zero failures; independent canonical ordinal and status checks passed. Of 103,272 fixed controller tuples: **102,273 pass zero of six scenarios; 971 pass one; 27 pass two; one passes three; none passes four or more.** This is not evidence of a generally robust policy.

The 3/6 tuple is fast 50min, slow 48h, kappa 1.12, deadband 0, steps 10–50bp. Its immediate sampled EMA/kappa neighbors mostly lose coverage. Selection retains 30 physically separated families, including scenario-specific candidates, ranked by coverage and fixed-step neighbor performance. Duplicate fast/kappa=1 modes with different slow half-lives are omitted because slow cancels from the target.

Analysis: `runs/ethusd-pscale-sequential-s1-discovery-619632-f64-20260917/inspections/modes.json`.
S1b config: `configs/experiments/ethusd-pscale-sequential-s1b-capture-6480-f64-20260917.toml`.
S1b crosses those 30 EMA/kappa modes with captures 85/90/95%, the original four step pairs and three deadbands, and all six economic/report scenarios. The existing remote evaluator is reused without transfer or rebuild.

## S1b result and S2 decision

S1b completed in 21.8 seconds, 6,480/6,480 successful; canonical ordinal/status checks passed. Fee/pscale coupling is material even within captures 85–95%: the 50min/48h/kappa1.12 discovery leader passes only 3/18 cross-capture scenarios at its best common step policy. The 20min/2h/kappa1.26 family passes 6/18, with two passing scenarios at each capture. No claim of general robustness follows from either result.

S2 retains the top 20 physically separated parent modes. Each mode keeps the union of its best common step/deadband policy and its per-capture best policies (not all 12 possibilities). Local fast values use 5-minute increments within +/-10min; slow uses 1h increments up to 12h and 2h increments above, within +/-2 increments; kappa uses 0.01 within +/-0.02. Valid half-life order is preserved, with no fast values below 600s. Deduplication yields **3,948 controller trials × 18 scenarios = 71,064**.

Config: `configs/experiments/ethusd-pscale-sequential-s2-local-71064-f64-20260917.toml`.
Results will be in `runs/ethusd-pscale-sequential-s2-local-71064-f64-20260917`.
S1b analysis: `runs/ethusd-pscale-sequential-s1b-capture-6480-f64-20260917/inspections/capture-modes.json`.

Planned dense step sweep will retain all three permitted captures, so its upper budget is 45,000 (20 modes × 125 step/deadband settings × 18 scenarios), rather than the earlier 20,000 placeholder. The overall 2-million ceiling is unchanged.

## S2 result and S3 decision

S2 completed in 89.5 seconds, 71,064/71,064 successful; independent ordinal/status checks passed. Best fixed controller still passes 6/18 scenarios: 20min fast, 2h slow, kappa 1.26, deadband 0, steps 10–50bp. All six passing scenarios are the 2.8%/0.30 economic anchor across the three captures and two reporting rates. This suggests a potentially useful economic region, not robustness to arbitrary donation/RPF.

Twenty refined, physically separated EMA/kappa modes retained. Selection prefers complete six-direction EMA/kappa neighborhoods when coverage ties; missing/boundary neighbors are explicitly penalized. Kappa=1 duplicate slow settings are removed. Examples include 15min/42h/kappa1.06 and 10min/22h/kappa1.05, each passing 5/18 at its selected step policy. Fast 10min sits at the supported lower boundary and is not an interior optimum.

S3 fixes these 20 EMA/kappa modes and crosses min steps {2,5,8,10,12}bp, max steps {20,25,35,50,60}bp, deadbands {0,2,5,10,15}bp, and all 18 permitted capture/economic/report scenarios: **45,000 points**.
Config: `configs/experiments/ethusd-pscale-sequential-s3-steps-45000-f64-20260917.toml`.
S2 analysis: `runs/ethusd-pscale-sequential-s2-local-71064-f64-20260917/inspections/local-modes.json`.

## S3 result and S4 joint search

S3 completed in 65.9 seconds, 45,000/45,000 successful; independent ordinal/status checks passed. No fixed controller exceeds 6/18 passing sparse scenarios. Twenty distinct round EMA/kappa modes now have selected step/deadband settings, retaining the most scenario coverage and sampled step-neighbor support. The 20min/2h/kappa1.26 leader retains zero deadband and 10–50bp steps; other modes shift, e.g. 10min/2h/kappa0.99 selects 10bp deadband and 8–35bp steps.

User clarified the end goal: a joint economic/fee/pscale optimization producing about ten diagnostic, robust driver candidates for later fee-design experiments. Explicitly include base/capture/fallback in the current search; base may not fall below 30bp, capture stays >80%, donation <3%, RPF <0.5. Staleness/asymmetric fee designs remain future experiments. A broader search over the full Cartesian fee lattice is more informative about stability than selecting unrelated best fee tuples per driver.

S4 config: `configs/experiments/ethusd-pscale-sequential-s4-joint-surfaces-680400-f64-20260917.toml`.
Output: `runs/ethusd-pscale-sequential-s4-joint-surfaces-680400-f64-20260917`.
Axes: 20 selected pscale policies × base {30,50,75}bp × capture {85,90,95}% × fallback {150,225,300}bp × reporting {50,75,100}% × donation 1.0–2.9% by 0.1 percentage points (20) × RPF .20–.40 by .01 (21) = **680,400**.

The unlaunched 714,420-point draft was replaced before launch to enforce the latest bounds. No simulations were spent on that draft. All current S4 fees meet the new base floor; default/reference base remains 50bp. Max-step and EMA choices are exact sampled round values, not rounded after evaluation.

Next: find connected stable fee/economic regions per driver, then perturb the pscale and fee parameters jointly at selected interior economic points. Do not describe a boundary member or highest-degree member as a stable center. Numerical/reference/temporal checks use only a bounded finalist set from the remaining 577,424-point budget.

## Numerical validation preparation

Built `arb_evaluator_ld` as a separate target from the existing remote source tree, without source transfer or rebuilding the f64 target. Its SHA256 is `6712a908bd04168419c3e51dacac3a49a401ec4f6f652a37287604d10bc45785`. Rechecked the running f64 SHA256: unchanged at `1e15bd2cbcbd01d201b3bc57a9d9a43ddffdd239822fd8e98d1a7f6f18ff1fb5`. This prepares a future check; it is not yet parity evidence.

Selection-bias check planned: compare the earlier ten-controller shortlist as controls, rounding only the two non-round fast EMAs to 90min and 60min and explicitly reevaluating them. The sparse discovery anchors and even-hundredth kappa lattice can miss the previously observed 30min/24h/kappa1.15 and 50min/15h/kappa1.15 families. A small independent fee/economic panel will keep these known alternatives from being silently lost.

## Validation interpretation

The full 2024–2026 history is already used for selection. Later splits of that same history are retrospective subperiod sensitivity checks, not genuinely held-out validation. Final reporting must distinguish these from out-of-sample evidence. Blade long-double and reference_2l comparisons assess different numerical/model assumptions and may invalidate attractive active_2l/f64 candidates.

## S4 result and control comparison

S4 completed in 711.0 seconds (957.0 points/s wall), 680,400/680,400 successful; independent canonical ordinal/status checks passed. Connected components are now measured in base/capture/fallback/donation/RPF, requiring all reporting rates 50/75/100% to pass at each node.

Largest all-report region: **36 nodes**, driver 20min/6h/kappa1.26, deadband0, steps10–25bp. There are 59 total all-report passing fee/economic points for that driver, across 15 components. The anchor is chosen inside its largest component (not the highest-yield isolated component): base50bp, capture90%, fallback300bp, donation2.2%, RPF.26; YB GM across report rates is 0.534/0.597/0.602%, detachment 14.501/13.653/13.608%. Its sampled boundary distance is one: do not call it a proven interior center.

Next families by largest all-report component: 25min/2h/kappa1.47, zero deadband, steps10–60bp (27 nodes); 10min/5h/kappa0.98, deadband15bp, steps12–25bp (17 nodes). These remain model/discovery results.

S4 analysis: `runs/ethusd-pscale-sequential-s4-joint-surfaces-680400-f64-20260917/inspections/joint-regions.json`.
S4b is running: ten prior controller families × 27 fee tuples × 3 report rates × 5 donation × 5 RPF = 20,250. Config `configs/experiments/ethusd-pscale-sequential-s4b-prior-controls-20250-f64-20260917.toml`. Its two non-round prior fast EMAs are explicitly re-evaluated at 90min and 60min; other values are retained. This is a guard against sparse-discovery selection bias, not a claim that rounded prior results transfer.

Next local fee/economic grid uses identical small increments around each new/prior family's largest connected region: base +/-5bp (floor30), capture +/-1 percentage point (>80%), fallback +/-25bp, donation +/-0.05 percentage points (<3%), RPF +/-.005 (<.5), across all three report rates. Subsequent driver refinement depends on those observations.

## S4b result and S5 local stability grid

S4b completed in 46.0 seconds, 20,250/20,250 successful; independent ordinal/status checks passed. The strongest prior control by all-report connected-region size is 15min/12h/kappa1.05, steps5–48bp, zero deadband (7-node component on its sparse panel). The prior 50min/15h/kappa1.15 family has a 2-node component. Raw component sizes are not compared directly to S4 because economic lattice spacing differs.

S5 evaluates the same local fee/economic perturbation pattern for all **30** families (20 new, 10 prior) rather than discarding old controls based on incomparable lattice counts. Bounds truncate some cubes, yielding **5,994 unique family/fee/economic trials × 3 report rates = 17,982**. There are no captures <=80%, base fees <30bp, donation >=3%, or RPF >=.5. Fallback local increments can reach 325bp, to test just beyond the previous 300bp boundary; this is a local sensitivity experiment, not a chosen production fee.

Config: `configs/experiments/ethusd-pscale-sequential-s5-local-fees-17982-f64-20260917.toml`.
Selection will measure fraction of the local cube passing all three reports, connected size, and complete 2x2x2x2x2 fee/economic boxes (32 points). Pscale local refinement follows for up to 12 diverse families using the fee/economic settings supported by this stage.

## S5 result and S6 joint refinement

S5 completed in 44.9 seconds, 17,982/17,982 successful; independent ordinal/status checks passed. Carrying prior controls mattered: **15min/12h/kappa1.05, zero deadband, steps5–48bp** has **125/162** local fee/economic tuples passing all three report rates, all in one component, including **one complete 32-point Cartesian box**. Its local fee domain uses base30/35bp, capture89/90/91%, fallback275/300/325bp, donation2.55/2.60/2.65%, RPF.285/.290/.295. The full domain is NOT uniformly passing; only the reported 125 members and the complete box are verified. Anchor base30/capture89/fallback300/donation2.55/RPF.295 has GM 3.75/3.34/3.46% across the three reports.

Other local families include 10min/6h/kappa1.05 (71/162 stable), 10min/22h/kappa1.05 (71/162), and a base50bp alternative: 25min/14h/kappa1.14, deadband15bp, steps10–50bp (61/243 stable). The 30min/24h/kappa1.15 prior family has 38/162 stable.

S6 keeps twelve physically distinct families and holds their locally supported fee tuple while varying ALL six driver parameters jointly: fast +/-5min; slow +/-1h if <=12h or +/-2h otherwise; kappa +/-.01; deadband +/-2bp; min step +/-2bp; max step +/-5bp. Each driver is crossed with a 3x3 local donation/RPF patch (increments .05 percentage points and .005 respectively) and reporting50/75/100%. Invalid policy durations/steps and user-bound violations are excluded; duplicate tuples within a family are not introduced. **50,301 trials × 3 report rates = 150,903**.

Config: `configs/experiments/ethusd-pscale-sequential-s6-joint-driver-150903-f64-20260917.toml`.
S5 evidence: `runs/ethusd-pscale-sequential-s5-local-fees-17982-f64-20260917/inspections/local-fee-regions.json`.
The goal now is to select driver values that preserve the surrounding economic region, then recheck fee sensitivity after any driver change. Full early-history selection remains in-sample.

## S6 result and S7 recheck

S6 completed in 188.6 seconds, 150,903/150,903 successful; independent ordinal/status checks passed. Five candidate families have a driver passing **all 9 economic points × all 3 reports = 27 cases**. These include 10min/22h/kappa1.04 (steps7–55bp, db0), 15min/2h/kappa1.07 (7–50bp, db0), 25min/7h/kappa1.25 (8–30bp, db0), 25min/2h/kappa1.48 (12–50bp, db0), and 25min/22h/kappa1.14 (12–24bp, db0). The latter two have materially weaker neighboring-driver support than the first two.

No candidate is called an interior center merely because it ranks well: several lie at artificial local search edges and some at the supported fast-half-life/deadband bounds. Missing driver directions are recorded explicitly as incomplete neighborhoods. The best first family's observed neighboring-driver economic survival averages 92%; this is not a guarantee beyond the sampled neighborhood.

S7 reruns small fee/economic cubes after the selected driver changes: **2,268 trials × 3 report rates = 6,804**. Config `configs/experiments/ethusd-pscale-sequential-s7-fee-recheck-6804-f64-20260917.toml`.
S6 evidence: `runs/ethusd-pscale-sequential-s6-joint-driver-150903-f64-20260917/inspections/driver-finalists.json`.

Prepared numerical/model validation (not yet launched): the identical twelve finalist parameter sets at report rates {0,25,50,75,100}% in three lanes: active_2l/f64, active_2l/blade-LD, reference_2l/f64. Sixty points per lane. Rates 0/25% are stress cases; the core region criterion remains 50/75/100%. The prepared LD config names the LD executable explicitly.

## S7 and early validation findings

S7 completed in 24.4 seconds, 6,804/6,804 successful. Fee tolerance does not automatically survive driver retuning: all-report passing local tuples include new-13 113/243, new-16 85/243, new-9 68/243, prior-0 61/162; the retuned prior-5 version has 56/162 and no complete 32-point fee/economic box. Its earlier version had a full box in its earlier economic neighborhood. Because neighborhoods shifted, this is a warning, not yet a controlled old/new effect.

Therefore S7b compares the parent and retuned drivers at IDENTICAL S7 fee/economic points (6,804 additional evaluations). Do not discard a parent merely because a retune improved a fixed-fee economic score.

The 60-point retuned f64 baseline completed in 16.2 seconds, all successful. All twelve anchors pass the three core report rates (50/75/100%). **None passes the full GM/pdif/positive-yield criterion at 0% or 25% reporting.** These are explicitly failing report-availability stress cases, not robust-to-outage policies.

Queued paired checks: 60 parent-driver f64 cases, 120 LD cases (12 retuned + 12 parents × five report rates), and 120 reference_2l cases. Parent and retuned versions use identical fee/economic anchor values for the paired comparison. If a final selected anchor differs, it needs its own final check.

## Paired fee and model checks

Matched S7b confirms genuine retuning trade-offs on identical fee/economic points: new-13 improves all-report passing tuples from 61 to 113; new-9 falls from 102 to 68; new-4 falls from 65 to 36. Retained parent versions remain candidates. All 6,804 S7b evaluations succeeded.

All 300 further S8 evaluations succeeded. Ten of twelve retuned anchors pass all core reports in blade LD; prior-5 and new-3 each fail one. GM can vary by percentage points despite pass/fail survival; numerical equality is not established.

Reference_2l is more selective. At the tested anchors, only **new-13 retuned** (25min/7h/kappa1.25, db0, steps8–30bp) and **new-16** (both parent 25min/14h/kappa1.14/db15/10–50bp and retuned 20min/12h/kappa1.13/db13/8–55bp) pass all three core reports across f64, LD and reference. Thus there are currently only TWO distinct driver families with this cross-lane anchor support, not ten validated robust policies. Many other reference failures are detachment breaches while GM remains positive.

S9 checks economic-region drift rather than rejecting an entire family from one anchor: 24 parent/retuned versions plus the prior-5 complete-box seed (35bp base,90% capture,300bp fallback,donation2.65%,RPF.29) each crossed with a 3x3 local donation/RPF patch and three core report rates. **675 points per lane × f64/LD/reference = 2,025**. The exact same candidates are used across lanes. No staleness/asymmetric fee change is included.

Paired validation evidence: `runs/ethusd-pscale-sequential-20260917/paired-validation.json`.

## S9 model-common economic regions

All 675 cases in each lane succeeded. Wall times: f64 17.4s, LD 19.1s, reference 149.4s. The full-region test is materially stronger than validation at one anchor.

- Prior-5 complete-box seed, 15min/12h/kappa1.05/db0/steps5–48bp, base35bp/capture90%/fallback300bp: **all 9 economic points pass in all 3 lanes and all 3 core reports**. Economic patch donation2.60/2.65/2.70%, RPF.285/.290/.295. At the middle point, minimum GM across lanes/reports is 1.0295%.
- New-13 retuned: 7/9 common points, one connected region; new-16 retuned: 4/9; new-9 parent: 4/9 in two components, largest3; new-16 parent, new-0, new-4 retuned, prior-0 retuned each have 3/9; new-7 retuned 2 adjacent points; new-15 retuned 2 isolated points. Several other families have no common point and are excluded.

Ten provisional configurations are retained from nine distinct families; new-16's parent/retuned pair is deliberately retained as a sensitivity comparison. They are not ten equally robust policies: support ranges from a full common economic patch to isolated cross-model passing points. Provisional artifact `runs/ethusd-pscale-sequential-20260917/candidates10-provisional.json` preserves exact evidence and parameters.

S10: 773 cases per lane (f64/LD/reference), total2,319. At each candidate's validated model-common economic point, change one driver parameter, one fee parameter, or one economic parameter at a time; evaluate core reporting rates. Baselines additionally test reporting0/25%. Include exact slow24h and max50bp alternatives where applicable rather than rounding results silently. Also check all 32 corners of prior-5's earlier complete fee/economic box across all three reports and all three lanes.

This final sensitivity map is intended to say what can move safely and what requires retuning when fee design changes. It does not validate Pyth staleness, asymmetric fees, or live deployment.

## Final outcome

S10 completed successfully in all lanes (773 each; reference wall158.1s). Final baseline metrics reproduced the corresponding S9 core metrics exactly in each lane. Aggregate run-manifest counts independently sum to **1,630,023**; the reusable driver TOML parses to ten tuples. No new simulator tests were needed: this campaign changed experiment configurations/analysis artifacts, not simulation code.

**P01 is the primary reference**, not one of ten equally robust policies: 15min/12h/kappa1.05, zero deadband, steps5–48bp, anchor base35bp/capture90%/fallback300bp/donation2.65%/RPF.29. Ten of eleven valid driver perturbations and four of six fee perturbations pass all three lanes and all core report rates. Its full nine-point economic patch passes across all lanes. The larger prior32-point fee/economic box passes22/32 across all lanes, but contains a complete8-point joint box: base30/35bp, capture90%, fallback300bp, donation2.60/2.65%, RPF.285/.29. Worst GM across that box/lane/report set is0.2125%; maximum detachment14.8243%.

P02 is a useful driver-stable but fee-sensitive alternative (7/11 driver moves,1/6 fee moves). Other configurations have narrower support. P03/P05 have0/12 passing driver moves at their anchors; these are deliberately labeled fragile diagnostic alternatives, not robust defaults. All final candidates fail the full criterion at0% and25% reporting in every lane. Exact round max50bp forP01 and slow24h forP07/P08 pass anchor checks only; the surrounding reported regions remain attached to the exact original tuples.

Deliverables:
- `runs/ethusd-pscale-sequential-20260917/RESULTS.md`: self-contained report.
- `runs/ethusd-pscale-sequential-20260917/drivers10.json`: exact drivers, anchors, common economic members, sensitivity directions, box/stress evidence.
- `runs/ethusd-pscale-sequential-20260917/drivers10.toml`: reusable coupled driver axis, not an automatically launched grid.
- `runs/ethusd-pscale-sequential-20260917/reproduce/`: retained analysis scripts and stage mappings.

No commit, push, deployment, simulator math change, Pyth-staleness campaign, or asymmetric-fee implementation was performed. This is an in-sample research reference set for subsequent fee-design experiments, with one clear primary candidate and graded alternatives.

## Fixed-ten economic/fee grid — requested 2026-09-17

Exact P01–P10 driver tuples from `drivers10.toml`, held fixed. Fallback is **300 bp for all ten**, explicitly superseding the per-policy 300/325 bp anchors. Capture includes **80%** per latest user instruction. Same historical window/exclusions, mixed-depth30s, active_2l/f64, cash3, YB fee1.2%, full_summary on 15 blades.

Donation:24 points,1.80–2.95% in0.05pp; RPF:24 points,.20–.43 in.01; capture:{.8,.85,.9,.925,.95,.975}; base:{30,35,40,45,50,55,60,65,70,75}bp; reporting:{0,.5,1}. **10×24×24×6×10×3=1,036,800 evaluations.** This is a new authorized grid, separate from the completed2m-ceiling search. Remote evaluator SHA256 matches the previous campaign. Estimated compute18–22min based on prior15-blade runs, before load/collection variation.

Config: `configs/experiments/ethusd-fixed10-donrpf24-capture6-base10-report3-1036800-f64-20260917.toml`. Output: `runs/ethusd-fixed10-donrpf24-capture6-base10-report3-1036800-f64-20260917`. Status: prepared and cardinality-checked; launch follows.

Launch confirmed: detached coordinator `blade-a5:/tmp/fxopt-grid.2R2PAMXX`, 1,013 shuffled leases across 15 workers. First successful progress:2,048/1,036,800 after26s (warm-up). Local follower is active and will retrieve the final two-file artifact. No source transfer/rebuild was performed. Results are pending; not yet evaluated economically.

## Local report participation split — 2026-09-18

Replayed ordinals168115 (report1,RPF.31) and168045 (report.5,RPF.29) locally with full native exchange actions. Reconstructed2,638,088 unique event timestamps using the existing evaluator static library event loaders and production arb_brings_report hash; joined every executed exchange exactly. No simulator code changed. At50% reporting,83,566/83,684 native swaps carry reports (99.859%);118 no-report swaps all pay3%, totaling$7.141m/0.5913% of input volume. Another111 report-bearing swaps hit the3% cap. At100%,all113,384 swaps carry reports;196 hit3%. Input notionals valued at each action CEX price in coin0. Different RPF means this pair is not a controlled causal reporting comparison. Local trade counts differ slightly from cluster113,320/83,705. Exact results:`runs/ethusd-fixed10-donrpf24-capture6-base10-report3-1036800-f64-20260917/inspections/arb-report-split.json`.

## Public report memory with time inflation — 2026-09-18

User authorized a new standalone fee implementation and rescan. `aged_fair_fee.hpp` does not import the old fee policy. q=min(F,base+capture*max(edge-base,0)); f=q+(F-q)*min(age/T,1). Cache stores original report price/time; only a committed report-assisted swap renews it with a strictly newer publication. Ordinary swaps use cached reference and cannot reset its clock. The native actor compares report submission against withholding it by net sized profit. No new staleness window in this scan; inputs remain fresh event-market reports (not Pyth). Positive adversarial admission windows are rejected for this policy until age-aware historical report search is implemented. Non-swap fallback behavior and dual-EMA math unchanged.

Grid:10 exact prior drivers × donation24(1.80–2.95%) × RPF24(.20–.43) × base2(35/50bp) × capture3(.9/.925/.95) × fallback3(100/200/300bp) × inflation4(60/120/180/300s) × reporting3(0/.5/1) = **1,244,160**. Report0 has no initial report: T is deliberately inert, providing repeated controls. Same mixed-depth30s history/exclusions, active_2l, cash3, fee1.2%, f64. Fee fields0–3, driver fields4–9. New isolated evaluator build `aged-fair-fee-20260918`; previous run binaries preserved.

Config:`configs/experiments/ethusd-aged-fair-fee-fixed10-donrpf24-base2-capture3-fallback3-ttl4-report3-1244160-f64-20260918.toml`. Output:`runs/ethusd-aged-fair-fee-fixed10-donrpf24-base2-capture3-fallback3-ttl4-report3-1244160-f64-20260918`. Status:prepared; unit/integration and24-case local gate before remote launch. Standard heatmap launcher retained.

Validation: `test_aged_fair_fee` passed (float/uint fee, cache/late-arrival equivalence, floor, expiry, duplicate/older reports, rollback, fractional age); `test_aged_reporting` passed (executed refresh, withholding, reportless reuse, no-trade non-refresh). Combined focused test wall <1s; 132 new test LOC. The 24-case full-history local gate completed in3.7s, all successful; every report0 metric except elapsed time was identical across T60/300. Diff whitespace checks passed.

User explicitly approved cluster source transfer after automatic approval review initially rejected the unspecified destination. Rebuild and launch then succeeded on the existing blade-a5 shared workspace. Detached job:`blade-a5:/tmp/fxopt-grid.MfpC1Y45`,1,215 leases on15 workers. Initial progress15,360/1,244,160; results pending. Standard `heatmap-yb.sh` is ready in the output directory.
