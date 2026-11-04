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

## Current user priority and testing policy

The user's 2026-11-04 instruction prioritizes completing and improving the
existing system while advancing the new ocean-observation direction when
resources and dependencies do not conflict. Resolve conflicts in favor of
the existing system. See
[Research priority and necessary testing](docs/research/Research_Priority_and_Necessary_Testing_CN.md).
This current user instruction supersedes older blanket full-suite and
unrelated safety-testing requirements, including those in historical round
instructions. Keep actual errors and constraint events in normal outputs;
do not weaken model constraints or rewrite old evidence to obtain a pass.
