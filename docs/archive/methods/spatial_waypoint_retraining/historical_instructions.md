# Round-8 Working Rules

- This round is limited to supervised spatial Waypoint and causal Transformer
  retraining on the locked Round-7 v2 **development** data and its fixed
  train/validation/development-test split.  It is not an online-RL round.
- Round-3 through Round-7 artifacts, reports, datasets, figures, checkpoints,
  manifests, and locks are immutable provenance inputs.
- The Round-7 untouched-holdout directory is content-isolated.  Round-8 may
  read only its manifest, scenario identifiers, counts, and hashes.  It must
  never read trajectory arrays, states, actions, rewards, traces, metrics, or
  plots from that directory.  All data readers must use `holdout_access_guard`.
- The only deployable goal is the three-field `goal_schema_v2` task threshold
  vector.  Hindsight goals, terminal positions, future actions/rewards/payloads
  or queues, and policy/quality/difficulty labels are forbidden model inputs.
- Train only H_w={10,20}, H_c=10, seeds={0,1,2}; all selection is validation
  only.  Development test is accessed exactly once after selection is frozen.
- Keep the existing single-UAV/static-buoy environment.  Do not add UAVs,
  drift, channel/SINR, power/association/beamforming actions, PPO, HGNN,
  mixture-density heads, multi-candidate waypoints, new reward/queue heads, or
  online training.
- Generated Round-8 artifacts belong under `round_08/`, `report/`, and
  `docs/` only.  Stop at the Round-8 lock; do not create Round-9.
