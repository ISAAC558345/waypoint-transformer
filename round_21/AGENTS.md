# Round-21 Working Rules

Round-21 is limited to supervised learning of a single-time-step
heterogeneous graph spatial encoder and Multi-UAV Task-Waypoint predictor,
plus its registered graph-value audits.  It begins only after the immutable
Round-20 lock has been rehashed exactly.

## Permitted scope

- Train the fixed two-layer, hidden-128 HGNN and its DeepSets comparison on
  Round-20's read-only H=5/H=10 Task-Waypoint samples.
- Train only the two specified selected-horizon HGNN ablations: no-UU and
  no-interference-feature.
- Use current physical graph state as input and the three frozen
  Task-Waypoint targets as supervision.

## Non-negotiable guards

- Do not modify or regenerate any Round-20 data, schema, split, label,
  environment, physical equation, matching, reward, action, or planner.
- The raw assignment identity arrays are adapter-excluded.  Only the binary
  relation fields `current_assignment_relation`, `has_current_assignment`,
  and `both_currently_assigned_distinct` are model inputs.
- Do not use IDs, split, behavior, quality, difficulty, string tier metadata,
  future values, or raw identity arrays as model input.
- Before selection is frozen, no development-test sample is decoded or passed
  to a model.  The guarded single evaluation after freeze is recorded as
  `development_test_access_count=1`.
- Never open the Round-20 untouched-evaluation manifest/data, run a model or
  policy on it, or run a policy rollout.  Its model-access count is always 0.
- No Transformer, Decision Transformer, temporal module, action/behavior
  cloning policy, PPO, MAPPO, MARL, power/association policy, or neural
  closed-loop evaluation is permitted.

## Evidence and stopping

All new artifacts belong under `round_21`.  Verify the Round-20 registered
hashes before and after training.  Train exactly 12 Phase-A and six Phase-B
runs with the registered configuration; make no unregistered architecture or
hyperparameter search.  Stop after Round-21 is locked.  Do not create
Round-22.
