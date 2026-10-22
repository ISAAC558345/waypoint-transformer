# Round-4 Working Rules

These rules apply to every file and artifact under `round_04/` and are stricter
than the repository-level rules.

- This round may train only a simple spatial Waypoint Network whose output is
  the two-dimensional relative UAV displacement after a fixed horizon.
- Transformer, Decision Transformer, attention, behavior-cloning action
  policies, reinforcement-learning policies, HGNNs, value heads, return heads,
  queue heads, service heads, and all other policy networks are prohibited.
- Do not add multiple UAVs, moving buoys, or communication-resource actions.
- The Waypoint output must not include queues, service allocation, reward, or
  return. It is exactly `[delta_x, delta_y]`.
- Treat
  `round_03/outputs/round03_pilot_canonical_20260730_230000` as read-only.
  Never regenerate, overwrite, edit, or re-split it.
- Preserve the authoritative Round-3 episode-level
  train/validation/test split. Never split individual transitions.
- Validation data alone selects checkpoints, hyperparameters, and the
  recommended horizon. Test metrics may be computed only after selection and
  must never trigger retraining.
- `achieved_terminal_goal` is an explicitly declared hindsight goal channel,
  never an ordinary state feature. Metadata and target-only arrays may not be
  predictor inputs.
- Candidate horizons use independent configurations, dataloaders, checkpoints,
  metric files, and figure directories.
- Generated artifacts must remain under `round_04/`, `report/`, or
  `logs/codex/` and must not overwrite earlier-round artifacts.
- After the Round-4 acceptance criteria pass, stop. Do not create or begin a
  later round.
