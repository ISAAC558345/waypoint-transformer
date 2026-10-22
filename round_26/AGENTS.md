# Round-26 Working Rules

Round-26 is an inference-and-audit-only study of mixed-behavior executed
joint-action labels, Task-Waypoint action disambiguation, and the frozen
Round-19 H=5 planner's first action. It creates diagnostic evidence only.

## Immutable inputs and access

- The original Round-25 lock and every earlier completed-round scientific
  artifact are byte-stable. The narrowly scoped Round-25 amendment only permits
  this authorized lifecycle successor.
- Decode only Round-20 `train` and `validation` rows. Development-test rows,
  fresh outcomes, and untouched evaluation are prohibited.
- Analyze `joint_action_applied_t` as the physically valid executed action;
  requested/raw action is semantics-only context.
- The Round-19 planner may receive only registered current graph state plus its
  frozen constant-arrival protocol. It cannot receive future trace, queue,
  randomness, or protected scenario data.

## Prohibited work

No optimizer, backward pass, neural/predictor/policy training, seed, checkpoint
write, model/environment/reward/planner/data/label/split modification, HGNN,
PPO/MAPPO/MARL, fresh rollout, development-test reuse, untouched evaluation, or
Round-27 creation is permitted. Behavior, quality, episode, scenario, and
future-payload fields are diagnostic metadata, never model inputs.

## Completion

Record action semantics, exact/near-state ambiguity, multimodality and
mean-action risk, frozen Task-Waypoint and policy inference audits, causal
planner-action diagnostics, pre-registered gates, figures, report, tests, and
an immutable lock under `round_26`; then stop.
