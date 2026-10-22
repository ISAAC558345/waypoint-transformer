# Round-22 Working Rules

Round-22 tests one question only: whether a frozen H=5 Round-21 DeepSets
Task-Waypoint predictor improves a causal multi-UAV movement policy over an
otherwise matched State-only DeepSets policy.  It begins after exact Round-21
lock verification and uses the frozen Round-20 development dataset.

## Permitted models

- Predicted Task-Waypoint DeepSets-Transformer.
- Matched State-only DeepSets-Transformer.
- Predicted Task-Waypoint DeepSets-MLP without temporal context.
- Oracle Task-Waypoint DeepSets-Transformer for offline teacher-forced action
  prediction only.  It is permanently non-deployable and cannot enter a
  causal rollout.

The DeepSets state encoder has shared UAV and buoy encoders with masked
pooling only.  No UB/UU edges, graph messages, agent-specific parameters,
raw identities, metadata, future labels, or historical actions may be policy
inputs.  The predictor is loaded in evaluation mode with all parameters frozen
and is never passed to an optimizer.

## Dataset and access guards

- Round-20 train is the only policy-training split; validation is the only
  checkpoint-selection split.  Round-20 development test is not decoded or
  used in Round-22.
- The Round-20 untouched evaluation remains completely inaccessible to models
  and policies.  Its model-access and rollout counts remain zero.
- A manifest for `round22_fresh_multi_uav_policy_dev_v1` is registered before
  training.  No environment is constructed and no outcome is read until the
  frozen validation gate is V-A or V-B.  It may then be accessed exactly once,
  without retraining.
- All Round-20 data, labels, schema, physics, matching, rewards and safety
  projection remain immutable.

## Fixed protocol

Use `H_task=5`, `H_c=10`, `d_model=128`, two Transformer layers, four heads,
FFN 512, dropout 0.1, the three seeds 0/1/2, and exactly twelve formal runs.
Select every checkpoint solely by validation raw joint movement-action RMSE.
Use unweighted valid-UAV/time masked behavioral-cloning MSE and record the
balanced sampler composition without admitting metadata as model input.

After formal training, validation closed-loop evaluation may run only the
three deployable families.  Fresh development access is conditional on V-A or
V-B; Round-20 untouched evaluation is forbidden regardless of result.  Do not
create Round-23.
