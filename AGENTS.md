# Repository Working Rules

These rules apply to the repository unless a more specific `round_XX/AGENTS.md`
adds stricter constraints.

- Round state is managed with `CURRENT_ROUND.txt`, `ROUND_VERSIONING.md`,
  `round_manifest.json`, immutable pre-round snapshots, and per-round reports.
- Do not use Git as the round-versioning mechanism.
- Preserve completed-round behavior. Per the user's 2026-11-04 instruction,
  use only necessary targeted regression and fair method comparisons for the
  current question. Do not automatically run the full historical suite or
  unrelated safety, stress, or fault-injection suites. Broaden checks only
  when a concrete changed dependency or observed failure justifies them.
- Keep the maritime environment dynamically sized from its configuration and
  public `state_dim`, `action_dim`, and `num_buoys` attributes.
- Do not invent physical variables that the environment does not expose.
- Learning models, optimizers, backward passes, and neural-network training
  were outside the completed Round-1 through Round-3 scope. Round-4 trained
  and locked the two-dimensional spatial Waypoint Network. Round-5 is
  explicitly authorized to freeze that H=30 network and train only the
  offline two-dimensional trajectory policies defined in
  `round_05/AGENTS.md`. Reinforcement learning, formal closed-loop ranking,
  and all later-round extensions remain prohibited.
- Generated data, figures, logs, and reports belong under the active round's
  output directories and must not overwrite earlier-round artifacts.
- A round must stop after its own acceptance criteria pass; do not begin the
  next round without a new instruction.

## Current user scope and testing policy

The user's latest 2026-11-04 instruction restricts current work to improving
the existing system. Preserve the new ocean-observation proposal in its
isolated branch and do not continue its development or merge it into the
existing system unless the user later explicitly requests that merge.
The active maintenance branch is feature/legacy-system-maintenance.
See [Legacy-only workflow](docs/research/Legacy_Only_Workflow_CN.md).
Use only necessary targeted regression or fair method comparisons.
This current user instruction supersedes older blanket full-suite and
unrelated safety-testing requirements, including historical round rules.
Keep actual errors and original evidence; do not weaken constraints to
make a result pass.
