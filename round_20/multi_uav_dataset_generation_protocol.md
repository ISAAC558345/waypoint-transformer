# Round-20 Multi-UAV Dataset Generation Protocol

The formal dataset is frozen before rollout at 80 scenario configurations:
two buoy counts, four equal interference tiers, and ten pre-registered
geometry/queue/stress templates per K/tier block.  Scenario seeds, split,
physical initialization, tier, and behavior schedule are stored in the
scenario manifest before any development trajectory is generated.

Every development scenario generates ten deterministic behavior episodes:
independent, Joint-H1/H5/H10, two legal controlled-noise variants of Joint-H1,
two of Joint-H10, a time-only Independent/H1/H10 switch, and a legal random
joint movement source.  Noise is applied to a normalized movement command and
then processed by the unchanged environment projection.  No behavior reads a
future state or trace.

Quality is ranked from actual episode payload, completion, residual queue, and
environment return within the same physical scenario; it is never forced from
the behavior name.  The exact rank allocation is 4 high, 4 medium, and 2 low
episodes per scenario.  Graph input contains only current physical data.  H=5
and H=10 service responsibility, gross service progress, and discounted team
return are separate offline supervision labels.

The new untouched set is manifest/snapshot only and has zero policy rollouts.
No neural training or protected historical data access is part of this round.
