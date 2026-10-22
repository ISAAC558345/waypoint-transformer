# round_02 Instructions

## Goal

Validate non-learning baselines, collect a small semantically structured offline
trajectory dataset, and audit data quality and K-step sample availability.

## Allowed modifications

- New non-learning maritime policy, runner, trajectory, storage, audit, metrics,
  plotting, and CLI modules under `src/wt/maritime/`.
- Round-02 tests, documentation, logs, reports, manifests, datasets, CSVs, and
  figures.
- Minimal environment fixes only when a concrete defect is demonstrated and
  documented.

## Forbidden

- Any neural network, trainable parameter, optimizer, backward pass, learning
  algorithm, behavior cloning, Waypoint Network, Transformer, Decision
  Transformer, or reinforcement-learning training.
- Invented queues, urgency, service completion, goals, or actual transmitted
  payload not exposed by the environment.
- Multi-UAV dynamics, buoy motion, resource optimization, or changes to the
  existing action/state physics.
- Starting or creating the next round.

## Environment interface

Use `MaritimeEnv.reset(seed)` and
`MaritimeEnv.step(action) -> observation, reward, terminated, truncated, info`.
Actions are finite normalized vectors of dynamic `action_dim`; the current
environment uses a two-dimensional unit disk. Observations have dynamic
`state_dim = 6 + 4*num_buoys`.

## Variable-scale and mask constraints

Never hard-code four buoys, state dimension 22, or a permanent buoy index.
Obtain dimensions from configuration/environment. Preserve per-episode node
count and valid-node masks. Never split an episode across dataset partitions or
let K-step indices cross a terminal/truncation boundary.

## Commands

```powershell
$env:PYTHONPATH="src"
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m unittest discover -s tests -v
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m compileall -q src tests
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m wt.maritime.runner --smoke
```

## Acceptance

Four policies run over multiple seeds and at least two buoy counts; trajectories
save/reload exactly; episode-level splits are disjoint; K in `{1,5,10,20,30}`
is audited; required CSV/JSON outputs and six figures exist; all values and
actions are valid; all previous tests still pass; no model is trained.

## Outputs

- `round_02/outputs/<run_id>/`
- `round_02/figures/<run_id>/`
- `logs/codex/`
- `report/`

Stop after round-02 acceptance and wait for human review.
