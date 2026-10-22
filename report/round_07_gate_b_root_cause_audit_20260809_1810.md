# Round-7 Gate-B root-cause audit

Round-7 created no model, optimizer, backward pass, checkpoint, or closed-loop
model evaluation.  It audits Round-6 Gate B and creates a separate, versioned
offline data candidate using only the existing single-UAV/static-buoy task.

## Frozen provenance

- Round-3 data, Round-4 checkpoint, Round-5 checkpoints, and Round-6 output
  are read-only inputs.  Round-6 lock SHA-256 is
  `E77A83EE903739AB6EB327238E632BAEB8FE93E7298258429BA471295AA71772`.
- The new data version is `maritime_offline_dataset_v2`; it has independent
  scenario seeds `2000..2029`, separate manifests, and an untouched holdout.

## A. Task difficulty

Round-3's own pilot shows a substantial random baseline (166.130 Mbit) and
successful simple policies.  This is consistent with Round-6's finding that a
Direct-Goal control is already highly competitive: the original v1 collection
does not reliably force a long-horizon routing choice.  That is not an
environment bug; it is a coverage limitation for testing an intermediate
target.

The new pool varies only existing initial UAV position, buoy geometry,
initial queues, horizon, and arrival traces.  Its difficulty score is derived
from normalized best--worst payload gap, random-to-best loss, and achieved
completion, then quantiled: 20 easy, 20 medium, 20 hard physical scenarios.
All hard candidates have a non-learning feasible trajectory; none are kept
because of invalid actions, energy failure, or an environment exception.

## B. Offline trajectory quality

The Round-3 pilot had high/mid/low = 55/25/0 episodes.  V2 adds only legal
behaviours: noisy nearest, noisy queue-rate-energy, causal policy switching,
and causal delayed commitment, while retaining the original heuristics and
random-valid.  Noise is added to a two-dimensional normalized action, then
the shared radial projection is applied before one environment step.

Outcome-derived tiers are assigned inside a difficulty class from return
quantiles, not behaviour names.  V2 has 720 episodes with high/medium/low =
288/288/144 (40%/40%/20%).  This repairs the missing low-quality tail without
fabricating a reward label or violating a physical constraint.

## C. Stitching coverage

V1 demonstrated coarse occupancy overlap but had no low-quality tail and only
80 episodes, so it was not sufficient evidence of cross-episode bridge
structure.  V2 uses normalized current observable state only, excludes all
same-episode neighbours, and derives thresholds from within-trajectory local
state distances.  On development data: P10/P25/P50 thresholds give
835/1641/2272 bridge candidates over 3308 cross-episode nearest-neighbour
queries; cross-policy neighbour overlap is 0.7461.  The P25 graph has 576
nodes, 1065 edges, 146 components, and 0.7296 cross-policy edges.

`stitching_stress_constructible=true` means documented existing fragment
pairs exist; it does not create a transition or claim that a later model will
necessarily stitch them.

## D. Goal informativeness

The 13-D `achieved_terminal_goal` is hindsight-only.  In particular, final
UAV x and y have correlations essentially 1.0 with their corresponding
terminal coordinates.  This makes a Direct-Goal control unusually strong and
is a leading explanation for Gate B.  The v2 audit marks all terminal fields
as not known before rollout and retains them only as declared hindsight data.

The existing three-field desired task goal is available before rollout and is
recorded as `goal_schema_v2_candidate`; it has no supplied spatial destination.
Thus it is deployable as a task-goal candidate, but it is not evidence that a
fully specified deployable spatial goal exists.

## E. Waypoint horizon and multimodality

H=30 has the best prior prediction selection result, not a proven control
horizon.  V2 finds high local target variance at every audited horizon:
H=5/10/20/30 = 146.7/331.0/486.9/482.0, with opposite-direction rates
0.225/0.244/0.229/0.211.  Therefore MSE averaging risk is high, particularly
for H=20/30.  The H=30 future-delta correlation with immediate action is low
(x/y 0.046/0.092), while its relation to a next-five-step mean action is
0.190/0.274: it is not merely the next action, but it is also a multimodal,
noisy long-horizon target.

Round-7 does not choose a winner.  The recommended retraining comparison
candidates are H=10 and H=20, to be selected later by validation plus
development closed-loop evidence, not future-position RMSE alone.

## Root-cause conclusion

Gate B most plausibly reflects the combination of an over-informative
hindsight terminal goal, insufficient v1 low-quality/bridge coverage, and a
long-horizon target with measurable multimodality--not a catastrophic rollout
OOD failure.  The v2 collection is suitable for a later bounded retraining
study, but it does not justify HGNN, multiple UAVs, drift, PPO, or resource
optimization.
