# Round-25 Working Rules

Round-25 is an inference-and-audit-only study of ambiguity in the frozen
Round-20 H=5 Task-Waypoint supervision and its alignment with the frozen
Round-19 causal joint planner.  It produces diagnostic evidence only.

## Immutable inputs

- The original Round-24 lock, outputs, figures, report, predictor, and policy
  checkpoints are byte-stable evidence.  A narrowly scoped lifecycle amendment
  may register this explicitly authorized successor, but cannot change any
  Round-24 scientific artifact.
- Round-20 data, labels, split, quality labels, behavior identities, and the
  frozen bridge/conflict definitions are read-only.  Decode only `train` and
  `validation` rows; the development-test split is prohibited.
- The Round-19 centralized joint planner and its physical protocol are used
  without changes.  Planner simulations receive only the observed current
  state and the protocol's constant arrival expectation, never a future trace.

## Permitted work

- Read existing Round-20 train/validation raw samples and transitions for
  diagnostic grouping and recomputation.
- Run the frozen predictor and frozen policy checkpoints in `eval()` with
  gradients disabled.
- Run causal H=5 (primary) and H=10 (secondary) planner diagnostics from
  registered current states; these simulations are not a new dataset, policy
  rollout, fresh-development evaluation, or untouched evaluation.

## Prohibited work

- Any optimizer, backward pass, neural training, checkpoint write, new seed,
  Task-Waypoint head/horizon/architecture/loss/sampler change, or data/quality/
  split/environment/reward/matching/interference modification.
- Any use of behavior, quality, episode, scenario, or future labels as model
  inputs.  They are audit-only grouping metadata.
- HGNN, PPO/MAPPO/MARL, fresh-development rollout, untouched evaluation, and
  Round-26 creation.

## Completion

Record the ambiguity and planner-alignment definitions, all required audit
tables, gates, access counters, figures, report, and immutable lock beneath
`round_25`.  Stop after the Round-25 lock.
