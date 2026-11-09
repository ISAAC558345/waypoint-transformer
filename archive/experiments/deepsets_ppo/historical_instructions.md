# Round-30 Working Rules

Round-30 is the **Direct Multi-UAV State-Only DeepSets-PPO Learnability
Baseline**.  It is a new, explicitly authorized on-policy RL baseline branch;
it does not reopen the completed Task-Waypoint, planner-trigger, or
conditional-distillation performance route.

## Frozen physical task

- Use only the Round-19 shared-spectrum primary environment with `M=2`,
  `K in {3,5}`, static buoys, static temporal channel, fixed power/bandwidth,
  shared queues, exclusive deterministic matching, and 15 m UAV separation.
- The sole learned action is the existing joint movement action `[M, 2]`.
  Do not add power, beamforming, service, association, matching, or other
  action channels and do not alter environment physics or reward.
- PPO may use only the raw reward produced by this environment.  Advantage
  normalization is permitted; reward shaping is not.

## Model and information contract

- The only learned policy is a parameter-shared, state-only DeepSets actor
  with a centralized permutation-invariant DeepSets critic.
- Inputs are current observable UAV features, buoy features, masks, and
  current physical global configuration only.  Raw UAV/buoy IDs and assigned
  indices are prohibited; current binary/mask representations are permitted.
- No future state/label, Task-Waypoint, planner action/result/utility, quality,
  behavior-generator metadata, bridge label, HGNN, graph message passing,
  Transformer, RNN/GRU/LSTM, offline BC pretraining, warm start, or
  behavior-quality weighting may be used.
- Store the sampled, tanh-squashed policy action separately from the actual
  safety/boundary-projected displacement.  PPO log probabilities always refer
  to the raw policy sample, never to its executed projection.

## Data isolation and evaluation

- Generate new, pre-registered training, validation, and fresh scenario sets.
  They must be seed-disjoint from every historical Round-20/22/24/29 set and
  from each other.
- Validation is deterministic, gradient-free, and may be used only for the
  frozen checkpoint schedule and selection metric (cumulative payload).
- The fresh set may be evaluated exactly once, after all five PPO checkpoints,
  baselines, metrics, statistics, subgroups, and gates are frozen.
- Round-20 development-test, Round-22 fresh, Round-24 fresh, and historical
  untouched evaluation remain inaccessible.  Untouched access stays zero.

## Formal protocol

- Run only one short engineering smoke before formal configuration freeze.
- Freeze one PPO configuration, formal environment-step budget, scenario
  manifests, validation schedule, and seeds `(0,1,2,3,4)` before formal PPO
  training.  Do not sweep, retune, extend the budget, or add seeds afterwards.
- Compare Random Valid, Independent Myopic, Joint H1/H5/H10, the frozen
  Round-22 state-only BC Transformer, and the five DeepSets-PPO policies on
  the same new validation/fresh scenarios.
- Generate all artifacts only under `round_30`; do not modify earlier
  scientific artifacts.  Do not create Round-31.
