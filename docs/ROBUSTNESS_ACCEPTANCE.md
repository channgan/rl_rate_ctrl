# Robust controller and transfer-readiness acceptance

Updated 2026-10-02 at the user's request. This supersedes the requirement in `LEARNING_EFFICACY_PROTOCOL.txt` that improvements must originate from PPO. The goal is a robust, transferable controller. Analytical initialization, measured calibration, network changes and training are permitted routes; every result must identify its source. Existing four-way comparisons remain useful diagnostics and will finish unchanged.

## Acceptance matrix

| Area | Reproducible measurement | Current status / next gate |
|---|---|---|
| Nominal mission | 2048-step physical survival; each-axis full-episode RMSE strictly below5deg/s; MAE/P95/peak, waypoint count | Retain original model; compare candidates on paired fresh nominal seeds before promotion |
| Dynamics | Independent mass/inertia, thrust, motor-time-constant conditions; failures and worst-axis margins | Baseline14/14survival,10/14precision; mass/inertia pilot plus unseen1.075/1.125 factors underway |
| Timing / sensors | Accepted-action delay trace, jitter, sensor-side noise/bias; not merely metadata settings | Fixed10msdelay and2xwhite/walk tested; jitter andfixedbias notyettested |
| External disturbances | Defined force/torque amplitudes, direction, duration, seed and onset; post-disturbance recovery | Not implemented; requires validated injection and single-factor screening before combinations |
| Altitude / waypoint | Height=-NED down; issuedtarget vs actual; switch3swindow; separate commandeddescent from sag; ±.2m recovery held.5s; censor unfinishedwindows | Detailed baseline recorded; compute identical candidate metrics, exclude native takeoff for policy episodes |
| Control cost | Accepted normalized torque RMS/slew/saturation; per-motor command saturation and variation | Available; these are not motor energy or achieved torque |
| Cross-condition reliability | Preserve every failure, paired error deltas and tails, then additional untouched seeds for selected candidates | Two seeds per pilot condition only screen candidates; insufficient to claim robust population reliability |
| Interface consistency | 9float32 inputs:gyro/5,error/5,previousacceptedtorque; FRD rad/s;3clipped normalizedtorques;100Hz; PX4collective/allocator retained | Simulation code audited; realfirmware/allocator/driver/inference deadline not yet verified |
| Outside tested envelope / failure | Detect nonfinite orstaledata, misseddeadlines, excessrates/altitude/attitude/saturation; validate fallback ownership andtransition | Current sim validates data freshness/action sequencing; hardwarewatchdog, fallbackandbumplesstransfer not implemented or validated |
| Hardware transfer | Actual modelidentification, trace replay, on-target timing, HIL/failsafe evaluation | Blocked on actualvehicle/flightcontroller/compute/interfaces/parameters/logs; no realflight approval orclaim |

Promotion requires repeatable heldout robustness benefit with nominal survival and precision retained, and explicitly reviewed altitude/control-cost/overshoot tradeoffs. No arbitrary aggregate reward may hide a worse axis or a new failure. If multiple candidates trade precision against actuator activity, preserve the default and present named alternatives rather than silently overwriting it. A small pilot or a nonzero update is not sufficient evidence of robustness.

## Current experiment

Source retained PPO4096 model; equal8192newsteps DR andnominal continuation, identical sourceweights/Adam andseed. DR uses discrete shuffled pairs [1,1.05] then[1,1.1] of coupled uniform mass/inertia. Shape andCoM remain unchanged, inertia stays physically valid under uniformscale. LR1e-5, std.003fixed, originalreward/termination. Unseenfactors1.075/1.125 andseeds3101/3102 compare analyticinitializer, source, nominalcontinuation, DR. No model chosen on trainingreward alone. Eachversion remains below1milliontrainingsteps.

All ranges are **provisional sensitivity assumptions**, not measured realairframe distributions. PhysicalSDF, resolvedworld, launchreferences anddelay traces are recorded. Actual hardware may differ in mass, inertia, motor mapping, filtering, timing or aerodynamic effects. Simulation sensitivity robustness is not sim-to-real acceptance.

## Iteration rules

1. Finish current controlled comparisons and quantify the weakconditions and any nominal regression.
2. Select the smallest evidence-supported intervention: controllercalibration, reward/advantage conditioning, physicalrandomization schedule, or observation/history change. Do not change all simultaneously or run an endless sweep.
3. Record provenance and compare against untouched source and equalbudget controls where learning claims are made. Keep newparameter sets and checkpoints in independent directories.
4. Validate first on isolatedperturbations, then physically plausible coupled conditions; retain separate unseen seeds andboundarytests. Do not combine every independent extreme without justification.
5. Only extend training after learning/robustness evidence supports it; preserve atomiccheckpointing,40ktrajectory cadence andepisode-levelphysics manifests.

## Hardware information and staged transfer plan

Reviewed current project README, trainingparameters, architecture, runtimeandSDF: these identify the **2.0643kg Gazebo x500-based simulation**, not a confirmed targetrealairframe. No confirmed targetflightcontroller oronboardcompute specification was found in those projectdocuments. Legacy siblingproject material, iffound, must not be assumed todescribe this target.

Needed: targetframe, all-upmass/CoM/inertia estimate; motors/props/ESC/battery; autopilot/firmware/allocatorgeometry; intended normalizedtorque injectionpoint; companioncomputer/runtime; IMUcalibration/filters; command/sensorlatencyandjitter; availableflightlogs andhardwareparameterpaths. Parentthread alreadyasked the user; simulationworkdoesnotwaitforthese answers.

Stages: identifytarget andfitmeasuredparameters → offlineunit/frame/action/timestamp andlogreplaytests → calibratedsimulation/heldoutfaultcases → separatelyauthorizedon-targetdryrun/HIL andfailsafe/fallbackverification → separatelyplannedandauthorizedcontrolledflight. No actualdevice deployment, motoroperation, arming, flight orsystemsecuritychanges are authorized in the current stage. A fallback is a future tested mechanism, not an assumed safety guarantee or a currentfeature.
2026-10-02 hardware clarification: user candidates ZEROONE X6 Air or 微空743V2 (not finalized); onboard NX exact model unknown. Same lower rate controller must support nativePX4 and userNMPC as selectable complete upper command sources, not concurrent writers. See docs/DUAL_UPPER_INTERFACE.md. Actualtarget airframe/firmware/NMPCoutputs/latency remain unknown; reviewedcurrentdocs andlegacyPython show no verifiedNMPCcontract. Acceptance includes both uppers; nominalSITL alone cannotcloseNMPCorrealhardwaregates. Existing24heldouttrialscontinue; no hardwareoperations.
