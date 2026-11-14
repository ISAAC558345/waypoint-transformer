# Round-29 Working Rules

Round-29 is a diagnostic-only Cross-Scale Deferred-Service Pressure
Normalization and Independent Confirmation Audit.  It may examine the locked
Round-28 train/validation feature table only for exploratory scale analysis
and train-only pre-selection.  It may run exactly one pre-registered,
independent 40-scenario confirmation diagnostic set after its manifest is
frozen.

## Immutable inputs and confirmation protocol

- Verify the original Round-28 lock SHA-256
  `9E3F10B893891B6E0E0415D886E00AEC234CD1376B3146E5704093A210384C91`
  and every registered Round-28 artifact before reading evidence.
- Preserve the Round-27/28 same-snapshot utility: H5 first action versus H1
  first action, frozen causal H1 continuation, actual fixed environment
  transition, and payload-difference utility.
- The confirmation set is exactly 40 new scenarios: 20 K=3 and 20 K=5,
  balanced as ten scenarios in each K x eta={.25,.5} cell, with fixed seeds
  33000--33039 and timesteps {0,2,4,6}.  Its outcomes may be read once only.
- Its environment remains M=2 shared-spectrum with static buoys/static
  temporal channel, unchanged planner, reward, action interface, matching,
  and interference semantics.

## Scope

Only the frozen original DSP and exactly DSP-V1 queue-share, DSP-V2
per-valid-buoy, and DSP-V3 current-capacity-tension representations may be
audited.  All weights stay .5/.5; all formal variant thresholds are train P75
and no K-specific threshold may enter the Cross-Scale Gate.  Selection of one
confirmation variant is train-only and is frozen before confirmation utility
is evaluated.

No optimizer, backward pass, neural/classifier/logistic/tree/forest/SVM,
policy/predictor/Task-Waypoint training, checkpoint write, seed extension,
planner/environment/reward/action/dataset change, HGNN, PPO/MARL, Round-20
development-test access, Round-22/Round-24 fresh access, untouched-evaluation
access, or Round-30 creation is allowed.

## Completion

Record the original-scale audit, variants, train-only selection, frozen
confirmation manifest, same-snapshot evidence, cross-K/eta/mechanism audits,
CS gate, figures, report, tests, integrity audit, and immutable lock under
`round_29`; then stop.  Any training is a later-round possibility only after
CS-A and is never performed here.
