# Round-36 Working Rules

Round-36 is an audit-only, non-learning planning diagnostic:
**Level-2 Relational Benchmark Non-Learning Planning Necessity,
Association-Action Headroom, and Task-Waypoint Action-Relevance Validation**.

## Scope

- The frozen benchmark is the locked Round-35 Level-2 configuration: four
  UAVs, eight buoys, twenty slots, static geometry, deterministic channels,
  fixed bandwidth/power/noise/queue equations, and the external-preference
  association decoder.
- No learning model, optimizer, gradient/backward pass, model fitting,
  training, reward redesign, environment redesign, decoder redesign, dynamic
  channel, drift, planner-trigger change, or protected-data read is allowed.
- `src/wt/round36/` may contain audit-only code.  In particular, a direct
  legal-assignment executor and an equivalence adapter may be used only by the
  Round-36 audit harness.  They must never be imported by a production
  environment API or alter Round-35 behavior.
- The primary continuation controller is the frozen current-state legacy H1
  movement rule and frozen deterministic association.  It is not a future
  joint planner.

## Data and access discipline

- `round36_design_v1` and `round36_confirmation_v1` are newly generated,
  disjoint clean scenario namespaces.  They are the only formal data sources.
- Protected, historical, train, validation, development, fresh, untouched,
  and prior engineering scenario contents must not be decoded or read.
- All preflight checks fail closed.  A protected decode, unknown source,
  confirmation contamination, outcome-dependent pruning, historical-lock
  mutation, or direct-executor equivalence failure stops the round.
- The confirmation namespace is accessed once, only after the prefinal freeze
  records the protocol, manifests, source hashes, and design completion.

## Lifecycle

- Preserve all completed-round artifacts.  Write generated artifacts only
  under `round_36/`.
- Round-36 must stop after its acceptance criteria pass.  Do not initialize
  or create Round-37.
