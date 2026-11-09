# Round-20 Working Rules

Round-20 constructs a deterministic, non-learning M=2 shared-spectrum
multi-UAV offline dataset and future-supervision labels.  It begins only after
the immutable Round-19 lock and all its registered hashes have been verified.

## Scope

- Primary environment: M=2, K in {3, 5}, static buoys, static channel,
  fixed power/bandwidth, shared queues, exclusive matching, and the existing
  15 m hard UAV-separation projection.
- The four pre-registered spectrum tiers are exactly orthogonal/0.00,
  low/0.25, medium/0.50, and full/1.00.  Every tier has the same development
  scenario count; no post-result resampling is permitted.
- Development trajectories use only the frozen Round-19 independent,
  joint-H1, joint-H5, joint-H10, controlled-noise, pre-registered policy
  switch, and legal random-joint movement behaviors.
- The round saves graph-ready current-observable state and H={5,10}
  Task-Waypoint supervision.  Quality is outcome-ranked within a physical
  scenario class, never assigned by behavior name.

## Prohibitions

- No neural model, optimizer, backward pass, checkpoint, HGNN, PPO, MARL,
  reinforcement learning, or policy training.
- Do not alter Round-19 source, physical parameters, shared-spectrum formula,
  matching, reward, action semantics, or safety projection.
- No power action, beamforming, association/service action, backhaul, buoy
  drift, dynamic channel, primary M>2 experiment, historical protected-set
  access, or rollout from the new untouched-evaluation seed set.
- Metadata such as behavior, quality tier, and named interference tier must
  not be a default graph-model input.  Only explicitly observable physical
  quantities may be in the graph state.
- Do not create Round-21.

## Evidence and stopping rule

All generated artifacts stay within `round_20`.  Scenario split assignment,
protocol, development seed manifest, and untouched-evaluation manifest are
frozen before formal data generation.  A Round-20 lock is written only after
all focused tests, full regression, compileall, serialization, permutation,
quality, graph, Task-Waypoint, and protection audits pass.  Stop immediately
after the Round-20 acceptance decision is locked.
