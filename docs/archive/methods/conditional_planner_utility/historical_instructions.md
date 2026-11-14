# Round-27 Working Rules

Round-27 is a diagnostic-only audit of whether the frozen Round-19 H=5
planner's executed first action has direct, conditional communication utility
relative to the frozen causal H=1 action, and whether a current-state-only
trigger can identify those states. It creates no learned model or dataset.

## Immutable inputs and state pool

- Verify the original Round-26 lock SHA-256
  `606558C28019A0B4C49DB0CE1B09B10CCC49750CAEF8648B1648D3BC66666FEE`
  and the narrow Round-26 legal-successor amendment before decoding data.
- Decode only Round-20 `train` and `validation` rows. Never decode
  development-test, fresh, or untouched rows.
- The diagnostic state pool is pre-registered as exactly one `t=0` snapshot
  for every allowed Round-20 scenario (48 train, 16 validation). The
  `joint_h5` row is a frozen action-identity control only, never a feature or
  selection outcome; both counterfactual branches reset the registered
  scenario seed and therefore start from an identical physical snapshot.
- The only branch difference is the requested current first joint action:
  frozen causal H=5 versus frozen causal H=1. Both then apply the same frozen
  deterministic causal H=1 continuation rule for the registered horizon.

## Causal trigger restrictions

- Trigger inputs must be derived only from the current physical snapshot:
  queues, rates, ISR/matching calculated from the current snapshot, geometry,
  current assignment, K, eta, safety, and current served-buoy information.
- Future queues/rates/assignments/payload, planner actions, Task-Waypoint
  targets, behavior generator, quality tier, and bridge labels are forbidden
  trigger inputs. Bridge is diagnostic-only.
- The sole numerical trigger threshold is the training-split P75 of the
  pre-registered queue-competition score. It is frozen before validation;
  no threshold sweep or validation refit is allowed.

## Prohibited work

No optimizer, backward pass, neural/classifier/predictor/policy/distillation
training, seed, checkpoint write, model/environment/reward/planner/action/
Task-Waypoint/dataset/split/quality modification, HGNN, PPO/MAPPO/MARL,
fresh rollout, development-test reuse, untouched-evaluation access, or
Round-28 creation is permitted.

## Completion

Record direct same-snapshot utility, useful/neutral/harmful labels,
pre-registered subgroup and trigger diagnostics, oracle-only routing upper
bound, design risks, gates, figures, tests, and an immutable lock under
`round_27`; then stop. Any future training remains only a recommendation.
