# Round-18 Working Rules

Round-18 is exclusively a deterministic, non-learning `M >= 1` multi-UAV
maritime task-coupling extension and coordination-necessity audit.  Formal
primary experiments use `num_uavs=2`, `buoy_drift_mode="static"`, and
`channel_variation_mode="static"`.

## Permitted scope

- Add configurable UAV count, per-UAV planar motion, per-UAV energy, a shared
  per-buoy queue, exclusive deterministic UAV--buoy matching, and hard,
  deterministic UAV--UAV separation.
- Use only the existing geometry, path-loss, fixed per-buoy bandwidth share,
  queue arrival, payload, energy, and reward semantics.  M=1 must delegate to
  and exactly reproduce the completed static single-UAV environment.
- Implement deterministic independent, centralized Joint-H1, and causal
  Joint-H5/H10 finite search diagnostics with the frozen action set.

## Prohibitions

- No neural model, HGNN, Transformer, Task-Waypoint, optimizer, backward pass,
  checkpoint, PPO/MAPPO/DDPG, multi-agent RL, training dataset, or training.
- No primary buoy drift or time-varying channel; no power, beamforming,
  association action, inter-UAV interference, backhaul, hierarchy, reward
  modification, coverage/diversity/role bonus, or resource redesign.
- Do not access old-development, fresh-development, or untouched-holdout
  artifacts, including metadata, hashes, or scenario IDs.  New access counts
  remain zero.
- Do not alter Round-17 or earlier locked artifacts and do not create Round-19.

## Required controls

- The Multi-UAV environment must remain dynamically sized in `num_uavs` and
  `num_buoys`; `M=2` is primary but no API may hard-code that count.
- One real queue exists per buoy.  Each slot's total transmitted payload for a
  buoy is bounded by that queue and its assigned link capacity.
- Matching is present-state deterministic exhaustive assignment with a frozen,
  reproducible tie rule.  Each UAV and buoy has at most one primary match.
- Separation is a permutation-safe deterministic projection, not a reward
  penalty.  All planners use exactly the same matching, safety projection,
  candidate set, arrivals, and reward.
- Generated artifacts belong only under `round_18`; all scenarios use the new
  `round18_multi_uav_planning_diagnostic` namespace and frozen fresh seeds.
