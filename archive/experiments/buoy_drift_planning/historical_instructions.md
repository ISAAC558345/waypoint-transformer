# Round-16 Working Rules

## Scope

Round-16 is exclusively an observable, predictable two-dimensional buoy-drift
extension and a deterministic planning-necessity revalidation. It may extend
the maritime environment only with buoy position and drift-velocity dynamics.
The default `static` configuration must reproduce the prior physical behavior.

## Prohibitions

- No neural-policy, Waypoint, transformer, MLP, HGNN, PPO, optimizer, backward
  pass, reinforcement-learning, checkpoint, or training-dataset work.
- No multi-UAV, time-varying small-scale fading, power control, beamforming,
  service action, urgency, AoI, reward change, action change, or resource
  semantics change.
- Do not read any old-development, fresh-development, holdout, or untouched
  holdout artifact, including its manifest, metadata, hash, or scenario ID.
  The only allowed historical statement is the Round-15 lock's already-recorded
  metadata-only access fact. `untouched_holdout_new_access_count` is exactly 0.
- Do not change anything under `round_15`, `round_14`, or an earlier locked
  round. Do not create Round-17.

## Required controls

- Reuse the Round-15 candidate lattice, continuation, ranking, safe projection,
  planning objective, and H={1,5,10,20} unchanged. Only the causal transition
  model may advance buoy position by currently observable drift velocity.
- Ordinary causal state may expose current buoy position and velocity only;
  never future buoy positions. Oracle future-drift knowledge is diagnostic-only
  and must be marked `oracle_future_drift=true`, `deployable=false`.
- Boundary behavior is deterministic reflection at the configured buoy-margin
  rectangle. Do not clip with outward velocity, teleport, or randomly reset.
- New data is limited to `round16_dynamic_planning_diagnostic` trajectories and
  is never a training dataset. All generated outputs, figures, reports, and
  logs stay under `round_16`.
- Before dynamic runs, static drift must pass a same-seed, same-action physical
  regression against the ordinary static environment.
