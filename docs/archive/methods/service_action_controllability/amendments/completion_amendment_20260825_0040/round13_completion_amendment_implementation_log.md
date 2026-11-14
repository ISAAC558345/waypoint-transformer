# Round-13 completion-amendment implementation log

## Scope and preservation audit

- Rehashed the original Round-13 lock and every file in its immutable formal
  output before starting; the amendment records these hashes and rechecks them
  before its separate lock is written.
- Rechecked the locked Round-12 predictor and all 24 frozen policy checkpoint
  hashes.  No Round-12 source, checkpoint, manifest, report, figure, or
  output was modified.
- Split the immutable Round-13 raw controllability table into service,
  progress, and return CSVs; those tables are derived data, not reruns.

## Scenarios and counterfactual execution

- Used exactly the ten pre-registered Round-10 `H_task=10` validation
  scenarios that were used by the original Round-13 headroom diagnostic.
- Every current/discrete/continuous comparison begins from the same reset
  seed, buoy positions, initial queue, arrival trace, environment config, UAV
  motion, and 128-step horizon.  The sole experimental variable is the
  diagnostic-sidecar resource mechanism.
- At every slot, the movement action and continuation are the deterministic
  `queue_rate_energy` baseline.  The sidecar reads only the current sidecar
  queue and current post-movement full-bandwidth link rate; it reads no future
  arrival, queue, rate, payload, reward, position, action, or Task-Waypoint.
- For Return-Task-Waypoint alignment, the same 40 registered snapshots, 14
  legal first actions, and ten-step `queue_rate_energy` continuation were
  reconstructed (560 rollouts).  Each return and payload is checked against
  the immutable formal raw row before writing the additive table.

## What was and was not executed

- Implemented a pure diagnostic masked-simplex priority decoder.  It is not
  connected to `MaritimeEnv.action_space`, its resource decoder, any training
  pipeline, or a neural checkpoint.
- Reconstructed the exact v1 reward components from `MaritimeEnv._step_v1`
  for fixed-movement sidecars; the current reconstruction must have zero
  error.
- Did not construct or run an optimizer, backward pass, neural-network
  training, PPO, HGNN, multi-UAV, power-control, beamforming, or a new
  predictor/policy checkpoint.
- Did not access old development test, Round-7 untouched holdout, or Round-12
  fresh-development scenarios.  Both protected access counts remain zero.

## Known limits

This is a diagnostic-only single-UAV, fixed-trajectory comparison of resource
mechanisms on ten validation scenarios.  It does not establish a new
environment action interface, policy value, closed-loop ranking, or a
next-round authorization.
