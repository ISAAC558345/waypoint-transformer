# Round-10 environment task-semantics audit

The existing v1 environment exposes (implemented_and_observable): UAV position,
velocity, remaining energy, movement-energy accumulation, elapsed/remaining
horizon; per-buoy queue, initial queue, arrivals, actual transmitted payload,
rate, link capacity, completion, position and valid-node mask; per-step reward
and reward components; and cumulative payload/return.  Service is fixed
parallel orthogonal allocation, so no unique selected buoy exists
(implemented_but_ambiguous); actual per-buoy transmitted payload is the valid
supervision signal.  Desired goals are observable but are constant across the
development data and not a variable episode task goal.

Queue update is audited as current queue + arrivals - actual payload - dropped
arrivals, including clipping; the maximum H-window conservation residual is
`9.78e-09` bits.  No new environment variable was invented.
