# Repository Working Rules

These rules apply to the repository unless a more specific `round_XX/AGENTS.md`
adds stricter constraints.

- Round state is managed with `CURRENT_ROUND.txt`, `ROUND_VERSIONING.md`,
  `round_manifest.json`, immutable pre-round snapshots, and per-round reports.
- Do not use Git as the round-versioning mechanism.
- Preserve completed-round behavior and run the full existing test suite after
  changes.
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
