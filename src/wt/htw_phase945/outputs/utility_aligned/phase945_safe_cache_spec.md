# Phase-9.4.5 safe cache spec

Only values that do not depend on a candidate assignment may be reused across
the 24 (frozen) candidate slots of one `(scenario, t)` state.

## Safe to cache (assignment-independent)

- Scenario seed and reset geometry (`uav_positions`, `buoy_positions` at `t`)
- Buoy cluster ids, area / power / noise constants
- Arrival schedule already realized up to `t`
- Feature masks and static node coordinates at `t`
- Candidate assignment hashes (identities), not their utilities

## Forbidden to cache across candidates

- SINR / interference / `rate_bps` after the assignment is applied
- Queue evolution, service, drops
- UAV trajectories under `fly-toward`
- H=15 payload increment
- Any quantity that `env.step(..., assignment)` writes

H=15 rollout always uses an independent `copy.deepcopy` / restored env per
assignment. Parallel workers never share a mutable `L3BEnv`.
