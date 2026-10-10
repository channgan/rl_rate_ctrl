# V49 training sufficiency audit and continuation admission draft

Status: PREPARATION ONLY / NOT ADMITTED. No new native calls authorized by this document. Current fixed V49 54-job development run must finish unchanged. No checkpoint selection using its results. Existing holdouts 12701/12702 remain sealed. This proposal requires a later explicit launch decision.

## Evidence and limits

V49 source commit: 48c02ec93cd805e4d16f399568093134194b5b3c. Final40 model ID: 4142227144. Binary SHA256: c0183c13bcf588c940a88f2a10aff59bc9256e8eb4b91066331ac9e4efe01855. Recovery checkpoint SHA256: 492dea55be406cbb9aa36c1ae7dde7878854936c0d3cb9a95832cdcc1e37e4c8.

Training ledger reports 1,233,041 charged native calls, 28 audited trajectories and 40 accepted updates, with no outstanding reservation. Audits report 573,440 scoring transitions: 28 x 20,480 native 1 ms transitions, or 28 x 20.48 = 573.44 simulation seconds. The 659,601-call difference is non-scoring overhead including startup/warmup; it is not independently useful new learning experience. The 2,048 episode steps refer to the 100 Hz horizon, not the 1 kHz transition count.

Each batch has 143,360 unique scoring transitions. Five scenario groups each receive equal mass, split equally between initial 3 seconds and the remainder; group 0 has three trajectories. Each optimizer update draws 2,000 rows using per-stratum shuffled permutations without replacement until exhaustion. Ten updates consume at most 2,000 rows in any single-trajectory stratum, below the smallest stratum size 3,000. Thus code plus successful update records imply 80,000 distinct direct policy-gradient row selections across the four fresh batches (13.95% of available transitions), not repeated full-dataset epochs. Checkpoint permutation tensors were not independently deserialized in this read-only audit. All scoring rows are nevertheless reused for returns/baselines, nine constraint gradients, adaptive gradient balance and acceptance diagnostics. Those repeated computations do not create new transitions.

Rows are temporally correlated: adjacent 1 ms states, shared history/returns and AR-conditioned actions. The discount time constant is 30 seconds. Reported importance-weight ESS is not an autocorrelation-adjusted count of independent samples. No measured integrated autocorrelation time or independent-sample estimate was produced. Statistical replication must be at trajectory/seed level, not at native-tick level.

| Updates | End batch KL | Weighted ESS | alpha first -> last | End maximum same-input action difference from A |
|---|---:|---:|---:|---:|
| 1-10 | 1.837905e-4 | .9996346 | .760069 -> .376797 | 1.788829e-4 |
| 11-20 | 2.490017e-6 | .9999950 | .376797 -> .376797 | 1.744442e-4 |
| 21-30 | 1.650538e-5 | .9999670 | same | 1.702448e-4 |
| 31-40 | 2.191130e-5 | .9999561 | same | 2.072407e-4 |

All 40 steps have accepted=true, frozen_ok=true, passed projection, nonzero parameter/action deltas and scale=1: no backtracking scale reduction. Per-step maximum parameter movement is 1.30878e-5 to 3.00407e-5; same-input action movement is 2.54344e-6 to 4.41214e-5. All nine end-of-batch fixed-data cost proxies decrease relative to their batch behavior policy. These importance-weighted proxies do not demonstrate closed-loop improvement.

Active projection constraints by batch: (0) bias_roll_lower, cost_0, cost_7; (1) cost_0/2/4/7/8; (2) cost_0/1/2/4/7; (3) cost_1/5. Nonzero balance tracking-gradient ranges before batch-3 updates 1 and 10 were [.02599,2.19805] and [.02703,2.69199]. These are per-stratum balance diagnostics, not full objective gradient norms. No trend to zero is established. Complete objective-loss and unclipped total-gradient histories were not recorded. Batch native reward means 29.66026,29.63816,29.71867,29.56906 use different seeds and behavior actors; they are not a controlled learning curve. Source update_batch.py explicitly stops after ten updates per batch. Conclusion: fixed-cap completion, not demonstrated convergence.

## Proposed fixed continuation (not launched)

Start from exact final40 actor, Adam moments/step counters, alpha and frozen buffers; retain original global A anchor and all reward/solver/acceptance rules, LR 3e-5, 154 RP parameters and ten updates per fresh seven-capture batch. Preserve the original checkpoint and RNG states as immutable provenance. Intentional new-data sampler stream begins at seed 510901; process RNG begins at 510902. These are explicit new-stream overrides, not a claim of bitwise continuation of the old random stream. No other state reset: especially no fresh Adam, alpha reset or anchor reset. New batch local counters/permutations start empty; global update count starts at 40. Sampler state then persists across batches as in V49. Training capture seeds for new batches b=0..11, profile p=0..6: 510101 + 1000*b + p. Verify no collision with actual existing manifests before admission.

At most 12 new batches / 84 fresh trajectories / 120 updates, stage endpoints global80,120,160. Stage boundaries are every four batches. No additional seeds, retries or budget expansion. Emit each checkpoint for traceability, but only the terminal stage checkpoint is eligible for final acceptance. No retrospective best-checkpoint selection.

### Fixed monitoring set and comparable baseline

Exactly three cases: noise200_hover_hold, low_x_050_goal100, low_y_025_goal100. Each uses all three new seeds 520101,520102,520103. Identical deterministic evaluation protocol, exploration_std=0, 2048 scoring steps, initial-state settings and sensor-noise recipes across baseline and endpoints. Run these nine jobs once on final40 BEFORE any continuation optimizer update, then nine after each completed stage. These are training monitors, never called holdouts. They must not enter gradient datasets or alter rewards/guards. Nine cases x seeds are paired across models; three seeds are too few for a strong convergence claim.

Primary per-trajectory metric J: sqrt((integral(e_roll^2)+integral(e_pitch^2))/(2*T)), rad/s, over the full 20.48-second native scoring window, with native-consumed angular-rate reference as primary. Compute the same metric separately against the ordered requested-reference reconstruction; never blend the two. Use physics angular-rate truth, not noisy observations. Equal weight for each of three cases. First3s, cruise and tail metrics also remain visible. A materially improved stage must reduce equal-case mean J by at least max(0.01 rad/s, 5% of comparator J), both versus final40 and versus the preceding stage.

For uncertainty, resample paired SEED BLOCKS (all three cases together), 10,000 bootstrap replicates with fixed analysis seed 520999; report percentile 95% intervals for paired J difference and relative improvement, alongside all nine individual paired outcomes. Do not bootstrap ticks. Require the primary difference interval upper endpoint <0 for a stage to count as improvement. The secondary-reference mean must not regress by more than max(0.005 rad/s,2%) against baseline or previous stage; do not use the easier reference to override primary failure. State explicitly the low power of only three seed blocks.

No-improvement counter increments at a stage that fails any material-improvement criterion; reset only on full improvement. Two consecutive no-improvement stages end training at that stage's terminal checkpoint. At stage3 always stop. Continue from stage1 to stage2 if hard protection passes even if stage1 is inconclusive; never use dev54 results to choose another checkpoint. An isolated statistically inconclusive stage is not proof of plateau.

Protections: all scoring episodes must survive 2048 with no physical failure, ownership/sync/history/finite/bounds failure or unintended PID takeover. Existing velocity coverage gates remain hard; memory quantiles remain diagnostic as in V49 while history/finite/bounds checks remain hard. All original yaw/ESC/other final acceptance gates remain unchanged. Additionally stop extension if any paired monitor's yaw RMSE exceeds baseline by max(0.01 rad/s,5%), or normalized ESC slew RMS exceeds baseline by max(0.01 normalized-units/s,5%). Report saturation and peak/p99 actions explicitly. Freeze exact ESC slew definition as sqrt(mean over four motors of integral((delta ESC/dt)^2)/T)), using the same audited native motor timing and units for all models. These are extension safety screens, not substitutes for final acceptance. A genuine audit/optimizer/protection failure preserves first cause/checkpoint and stops; no automatic retry or rollback-and-relaunch.

### Finite budget including baseline and failures

| Allocation | Maximum native calls |
|---|---:|
| 84 training captures x 47,000 | 3,948,000 |
| 9 final40 baseline monitor captures x 47,000 | 423,000 |
| 27 stage monitor captures x 47,000 | 1,269,000 |
| One final54: 18 PID x56,000 +36 NN x47,000 | 2,700,000 |
| TOTAL maximum | 8,340,000 |

This corrects the earlier 7,917,000 proposal, which omitted the comparable final40 monitor baseline. The complete per-attempt cap is reserved before launch, including warmup/probes and a failing attempt's consumed calls. No separate retry pool: failure allowance is already inside each reservation, and failure stops instead of creating another attempt. No unbudgeted probes. Conservative per-attempt caps are not permission to exceed the global cap. Baseline plus up to three monitor stages are included even if all stages run. Early stop releases unstarted-job allocations. A hard safety failure does not automatically trigger final54.

### Final acceptance and reduced experiment cost

If eligible, run final54 once: existing nine-case recipe x seeds 530101/530102 x PID/A/terminal candidate. These seeds are predeclared and must remain unseen until that final acceptance. Do not access sealed 12701/12702. Compare both reference definitions per case with all original V49 gates and original PID/A baselines; additionally preserve the monitor comparison against final40. No model promotion solely on survival or audits. Small implementation edits first receive offline regression/replay; a frozen training campaign uses cheap monitors at stages and one full54 at the end, rather than full54 per small edit.

Convergence evidence would require a reproducible closed-loop plateau with stated uncertainty, adequate independent seeds, stable objectives and gradient/projected-step trends across multiple stages. Constraint-limited stagnation is separately reported when raw gradients/proposals remain nonzero but projection removes most movement or repeatedly binds a boundary. Low KL, high importance ESS, tiny steps, saturated alpha or hitting a fixed cap alone prove neither convergence nor adequacy.

## Mandatory zero-native admission tests (pending, launch blocked)

1. Deserialize final40 recovery and compare actor exactly with NPZ/binary; verify Adam moments and step counters, alpha=.3767965169092372, original frozen tensors and global A anchor. Save/load a disposable copy and compare every retained state tensor, then test explicit new RNG stream determinism separately. Preserve old RNG provenance; do not accidentally restore old data permutations into fresh batches.
2. Instrument a COPY of the runtime only. Log unclipped/clipped weighted surrogate objective, full fixed-data surrogate before/after, unnormalized tracking/smooth costs, raw total gradient norm and clipping factor, Adam proposal, projected displacement, active constraints/KKT certificate, accepted displacement, sample-index digest, KL/ESS and alpha. Observational logging must detach existing tensors and not consume RNG or alter operation order; gradients from separate diagnostic objectives must be computed only in an isolated copy with restored state.
3. Run matched disposable checkpoint replays with logging off/on using identical inputs/RNG. Require bitwise identical actor/Adam/RNG/sampler/alpha, accepted proposals and final exported bytes. Require all logged losses/norms finite; cross-check raw gradient and clipping norm against an independent offline calculation. No actual training ledger writes and no native tools in this test.
4. Test baseline-required admission, seed collisions, exact counts/caps, failure charging/no retry, endpoint-only selection, two-stage stop logic and both-reference/yaw/ESC screens using synthetic fixtures. Verify read-only preservation of V49/V48 and sealed holdout paths.

Admission status on 2026-10-10: source/JSON audit completed; numerical checkpoint restore and logging-equivalence tests NOT RUN. Ordinary Windows Python execution returned access denied in this environment. No alternate interpreter/WSL or elevated execution was used to bypass that restriction. No instrumentation implementation has been admitted; this document specifies required tests, not passing results. Do not launch until these tests pass in an authorized execution context, final V49 dev54 is complete, and the detailed contract receives a launch decision.
