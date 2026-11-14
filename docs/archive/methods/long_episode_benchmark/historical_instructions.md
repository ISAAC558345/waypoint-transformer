# Round-33 Working Rules

Round-33 is **Long-Episode Multi-UAV Benchmark Design and Non-Learning
Planning-Horizon Separation Validation**.

## Allowed scope

- The sole environment degree of freedom is the `episode_horizon` configuration
  value. Historical-compatible regression is exactly T=12; the new benchmark
  is exactly T=20. No other physical, geometry, dynamics, queue, rate,
  interference, matching, safety, reward, action, power, channel, or
  initialization rule may change.
- A human-authorized, narrow planner extension admits H=20 to the pre-existing
  joint finite-beam search. The candidate set, scoring, shadow transition,
  matching, beam width, tie break, and all H1/H5/H10 behavior must remain
  byte-audited or T=12-exact. H20 is descriptive only; H10-vs-H5 is primary.
- New T=20 physical scenarios must be manifest-frozen and disjoint from all
  protected/historical populations before design or confirmation outcomes are
  accessed. The confirmation population is accessed once only after the
  prefinal freeze.

## Prohibitions

- No neural training, model/optimizer/backward pass, PPO, BC, HGNN, DeepSets,
  Transformer, Task-Waypoint use or training, planner trigger, reward/action
  redesign, protected development-test access, or untouched evaluation access.
- Do not create Round-34. Stop after the Round-33 lock.

## Required interpretation

- Task-Waypoint reentry can be true only for HS-A. H5-over-H1 alone is not a
  reentry result. H20 may not select the primary protocol or support an H10
  specific claim.
