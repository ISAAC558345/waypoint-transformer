# Round-15 Working Rules

- Round-14 and every earlier completed round are immutable evidence.  Do not
  modify their locks, source, environment, data, checkpoints, figures,
  reports, manifests, or generated outputs.
- This is a deterministic planning-headroom audit only.  Do not construct,
  train, update, save, or select any neural model, optimizer, backward pass,
  Transformer, MLP, PPO, HGNN, or checkpoint.
- The public environment remains single-UAV, static-buoy, and movement-only
  with action `[delta_x, delta_y]`.  Do not modify service allocation, power,
  beamforming, association, reward, queue semantics, channel semantics,
  buoy motion, or Task-Waypoint/mapping variants.
- The one historical pre-initialization inspection of the first rows of
  `round_07/.../trajectory_quality_audit.csv` exposed holdout metadata.  It
  is recorded as one metadata-only access and is prohibited from every
  analysis, selection, output, and conclusion.  Do not open any further old
  development test, fresh development, or untouched-holdout content.
- Use only newly materialized `round15_planning_diagnostic` scenarios.  The
  causal planner may use current state, static environment parameters, and a
  fixed current-arrival persistence assumption; it must never read a future
  realized arrival trace.  The oracle planner may use the copied branch's
  materialized trace, must be marked non-deployable, and is an upper-bound
  diagnostic only.
- The action lattice and horizons `{1,5,10,20}` are pre-registered.  Every
  action uses the common radial projection; every branch begins from an
  identical snapshot and uses the fixed `queue_rate_energy` continuation.
- Generated artifacts must remain under `round_15/`.  Stop when the Round-15
  gate and lock are complete.  Do not create Round-16.
