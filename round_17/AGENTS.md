# Round-17 Working Rules

Round-17 is exclusively a deterministic, observable and predictable
time-varying maritime-channel extension and planning-necessity audit. Its
primary experiments set `buoy_drift_mode="static"`; the Round-16 drift code is
retained but is not a primary experimental factor.

## Prohibitions

- No neural model, waypoint, transformer, MLP, HGNN, PPO, optimizer,
  backward-pass, reinforcement-learning, checkpoint, or training-dataset work.
- No multi-UAV, service-action, power-control, beamforming, association action,
  action lattice, beam width, continuation policy, planning objective, safe
  projection, resource mechanism, queue/arrival semantics, or reward change.
- Do not alter the base geometry/path-loss, bandwidth, transmit-power, noise,
  or rate formula. The only channel extension is a bounded, multiplicative
  slow predictable state factor.
- Do not read old-development, fresh-development, or any holdout artifact,
  including its metadata, hash, or scenario IDs. All new access counts are 0.
- Do not modify Round-16 or earlier locked artifacts and do not create Round-18.

## Required controls

- The primary channel state is a deterministic oscillator exposed only through
  present per-buoy `[sin(theta), cos(theta), frequency_normalized]`; no future
  multiplier, rate, SINR, window, or trace is exposed to an ordinary agent.
- The causal planner may forward-simulate the registered state dynamics. Only
  the oracle diagnostic may use a copied environment trace, and it must mark
  `oracle_future_channel=true` and `deployable=false`.
- Static channel mode must exactly reproduce the prior static-channel physics
  before any dynamic-channel audit runs.
- Diagnostic scenarios use only the new `round17_channel_planning_diagnostic`
  namespace, fresh fixed seeds, and pre-registered phases. Generated artifacts
  remain under `round_17`.
