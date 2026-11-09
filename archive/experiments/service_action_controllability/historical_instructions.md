# Round-13 Working Rules

- Round-12 is locked evidence. Do not edit its source, config, predictor,
  policy checkpoints, manifests, figures, reports, or lock files.
- This is an audit and design-only round. Do not construct/train/update a
  neural model, optimizer, backward pass, PPO, HGNN, Transformer, MLP, or
  predictor. `training_performed` and `policy_training_performed` stay false.
- Read and simulate only Round-7/Round-10 registered train or validation
  development scenarios. Old development test, Round-7 untouched holdout, and
  Round-12 fresh-development scenarios are prohibited for selection, training,
  or formal evaluation.
- The implemented maritime environment remains movement-only: its public
  action stays `[delta_x, delta_y]`. A service intent is a prospective design
  and diagnostic-sidecar calculation only; do not modify `MaritimeEnv`'s
  action interface or resource decoder in this round.
- All perturbations must be finite, radially legal, boundary-safe through the
  existing environment projection, restored from identical snapshots, and
  followed only by a deterministic `queue_rate_energy` continuation.
- Headroom uses current-state queue/rate information only. It must never read
  future arrivals, future queues, future actions, or future Task-Waypoints.
- Do not introduce multi-UAV, drift, beamforming, power control, continuous
  power optimization, reward changes, HGNN, PPO, or Round-14.
