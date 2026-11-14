# Round-12 Working Rules

- Freeze the selected Round-11 H10 task-waypoint predictor.  It must remain in
  `eval()` with `requires_grad=False`; no optimizer, backward pass, update, or
  new predictor checkpoint is permitted.
- Train only the four authorized H10 policy families: predicted Task-Waypoint
  Transformer, matched state-only Transformer, non-temporal Task-Waypoint MLP,
  and non-deployable Oracle offline diagnostic.  H20 policy training is banned.
- Train/evaluate only designated train/validation data before the one-time,
  pre-registered fresh policy-development access.  Never read old development
  test or untouched holdout content.  Do not create Round-13.
- No spatial waypoint, constant goal, PPO, HGNN, multi-UAV, drift, reward
  modification, or communication-resource action may be introduced.
