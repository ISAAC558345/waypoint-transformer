# round_03 Instructions

## Goal

Complete the maritime task semantics needed for meaningful non-learning
baseline comparison, collect schema-v1 offline trajectories, and materialize
audited supervised Waypoint targets for `K in {1, 5, 10, 20, 30}`.

## Allowed modifications

- Minimal extensions to the single-UAV/static-buoy environment for real
  per-buoy queues, arrivals, service allocation, delivered payload, task
  completion, goals, and truthful reward components.
- Non-learning policies, shared-trace evaluation, schema-v1 trajectory
  persistence, Waypoint sample construction, metrics, audits, plots, tests,
  reports, and round-control files.
- A versioned v0 compatibility path required to keep all round-01/02 tests and
  stored data readable.

## Forbidden

- Neural networks, trainable parameters, optimizers, backward passes, behavior
  cloning, Waypoint Network training, Transformer training, Decision
  Transformer training, or any reinforcement-learning training.
- Multi-UAV dynamics, moving buoys, channel-model replacement, large-scale
  maritime migration, or unrelated repository restructuring.
- Mixing trajectory/dataset/goal/waypoint schema v0 and v1 without an explicit
  validation error.
- Overwriting or deleting any completed round-01/02 artifact or lock.
- Starting or creating round_04.

## Semantics and fairness

- Task semantics v1 uses service-before-arrival slot order:
  `D_t <= min(Q_t, R_t * delta_t * allocation_t)` and
  `Q_(t+1) = clip(Q_t - D_t + A_t, 0, Q_max)`.
- Every comparison group shares its buoy initialization and complete external
  arrival trace across policies. Private policy RNGs remain isolated.
- Report actual delivered payload separately from potential link capacity.
- No policy may use future arrivals, future states, or held-out split data.

## Commands

```powershell
$env:PYTHONPATH="src"
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m unittest discover -s tests -v
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m compileall -q src tests
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m wt.maritime.runner --round3-smoke
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m wt.maritime.runner --round3-pilot
```

## Acceptance

All previous 100 tests plus at least 28 round-03 tests pass; the bounded smoke
and the 80-episode pilot complete without errors; all four required v1
non-learning policies are behaviorally distinct in the constructed audit
scenario; schema-v1 trajectories save/reload exactly; split boundaries and
K-step boundaries do not leak; positive `K=10,20,30` samples exist; all 12
required figure categories are generated; and every output records
`training_performed=false`.

Stop after round-03 acceptance and wait for human review.
