# Round-31 Working Rules

Round-31 is the **State-Only Multi-UAV PPO Failure Diagnosis: Optimization,
Credit Assignment, Actor Observability and Relational-State Sufficiency
Audit**.

## Scope

- This is an inference-only, frozen-artifact audit of the locked Round-30
  DeepSets-PPO baseline.  It creates no model, checkpoint, optimizer update,
  backward pass, training run, new seed, tuning sweep, or budget extension.
- It may read locked Round-30 training histories/checkpoints and run only
  gradient-free current-state diagnostics on the registered Round-30 training
  and validation populations.  A diagnostic environment transition is allowed
  only for those two populations and must use the unchanged Round-19 task.
- The Round-30 fresh raw outputs and manifests may be read as frozen evidence.
  They must never be regenerated.  Round-20 development data, Round-22/24
  fresh data, and untouched evaluation remain inaccessible.

## Frozen technical contract

- Do not alter the Round-19 environment, reward, matching, safety projection,
  action interface, PPO configuration, checkpoint, or planner.
- Do not train/retrain PPO; do not run an optimizer or call a backward pass.
- Do not add HGNNs, graph message passing, a Transformer actor, Task-Waypoints,
  planner triggers, BC pretraining, resource actions, or future-state inputs.
- Pairwise relational features are diagnostic-only current-state arrays.  They
  cannot be passed to a policy/model and cannot include labels, future traces,
  future returns, planner utilities, or behavior metadata.

## Completion

- All outputs, figures, reports, and logs belong under `round_31/`.
- Preserve the Round-30 lock byte-for-byte.  Record hashes before and after
  the audit, plus access counts of zero for new fresh and untouched rollouts.
- Stop after the Round-31 lock.  Do not create Round-32.
