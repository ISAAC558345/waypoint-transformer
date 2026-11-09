# Round-14 Working Rules

- Round-13, its completion amendment, Round-12, and all earlier completed
  rounds are immutable evidence.  Do not edit their locks, source, data,
  checkpoints, figures, reports, manifests, or generated outputs.
- Round-14 is an inference-and-audit-only task-to-motion mapping study.  It
  may read the hash-pinned Round-11 H=10 Task-Waypoint predictor strictly in
  evaluation/no-gradient mode.  It must not train, update, save, or select a
  neural model, optimizer, checkpoint, MLP, Transformer, PPO, HGNN, or policy.
- The public maritime action remains exactly the two-dimensional movement
  displacement `[delta_x, delta_y]`.  Do not modify the environment, reward,
  queue dynamics, resource allocation, service action, power, beamforming,
  association, UAV count, buoy drift, or Spatial Waypoint interface.
- Use only Round-10 registered `train` and `validation` Task-Waypoint rows
  with `H_task=10`, and the pre-registered Round-13 validation scenarios for
  counterfactual rollouts.  Old-development-test, fresh-development, and
  untouched-holdout content must never be opened.
- A deployable mapping may consume only current predicted
  `[service_distribution, gross_service_progress, H_step_return]`, current
  valid-node mask, current UAV/buoy geometry, and current rates.  Future
  outcomes may appear only in clearly labelled offline diagnostic columns.
- Every counterfactual starts from an identical reconstructed snapshot, uses
  a radially legal first movement, and then the deterministic
  `queue_rate_energy` continuation.  All generated files belong under
  `round_14/` or `report/`; never overwrite earlier artifacts.
- Stop at the Round-14 readiness gate.  Do not create Round-15.
