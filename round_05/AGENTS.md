# Round-5 Working Rules

These constraints are stricter than the repository-level rules.

- Treat the locked Round-3 pilot, its episode split, and all prior-round locks
  as read-only inputs.
- Load exactly the validation-selected Round-4 H=30 goal-conditioned seed-1
  `best.pt`; verify SHA-256
  `F92133AACA1FF6F0839EE7C817F40B696F7FD8FB15973A6118D222F83F01368D`
  before use. Never update its parameters or optimizer state.
- The policy action is the two-dimensional executed physical displacement
  `next_uav_positions_m - uav_positions_m`, normalized by the configured
  maximum movement distance. The model must end in `tanh` and may not receive
  current or historical action labels as input.
- Derive state, goal, mask, condition, and action dimensions from schemas and
  configuration. Do not hard-code the current 62/13/5/2 dimensions inside
  models.
- The deployable Waypoint Transformer may consume state, node mask, predicted
  two-dimensional spatial waypoint, remaining-time feature, position, and
  attention mask. It must not consume the global goal or oracle waypoint.
- Direct-goal and state-only controls must exclude waypoint features. Oracle
  H=30 is a clearly labelled non-deployable diagnostic and is not eligible for
  selection.
- Reuse the official causal Transformer block and padding-attention logic. Do
  not enable the legacy action-token path.
- Use deterministic episode histories with no split crossing. Padding tokens
  must neither receive attention nor contribute to loss.
- Formal sampling is balanced across the four collection policies. Record
  sampler counts and seeds.
- Choose best epochs, context length, and recommended seed only with validation
  data. Test data is evaluated once after selection and cannot change a model
  or hyperparameter.
- Formal scope is exactly 27 runs: 18 Transformer deployable controls, three
  oracle Hc=10 diagnostics, and six one-step MLP controls.
- Interface validation is bounded to 1–5 steps and must be labelled
  `interface_smoke=true`, `formal_closed_loop_evaluation=false`.
- Do not add PPO, online updates, HGNN, multiple UAVs, buoy drift, new reward or
  queue semantics, reward waypoints, or power/association/slot/beam actions.
- Generated outputs and figures must remain under `round_05/`. Stop after the
  Round-5 acceptance criteria are audited and locked; do not create Round-6.
