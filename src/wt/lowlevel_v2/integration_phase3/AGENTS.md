# Integration Phase-3: Executor-Conditioned Utility Supervision

Authorized unique change: replace artificial / prefix / tau-softmax labels with
linear-normalized frozen Low-Level executor utilities, and train High-Level
with `L_exec + 0.25 L_rank + 0.10 L_ref`.

Do not modify Low-Level MAPPO, SafetyProjector, collision/interference, reward,
actor/critic, decoder, DeepSets, ranking/reference losses, hold, or horizon.
Do not overwrite Phase-2 / 2A / 2B / 2C / 2D artifacts.
Do not start the next phase after this one stops.
