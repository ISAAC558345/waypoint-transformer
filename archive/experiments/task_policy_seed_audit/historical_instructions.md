# Round-23 Working Rules

Round-23 performs exactly one inference-only mechanism audit: seed stability,
Task-Waypoint head causal utilization, and offline-to-closed-loop alignment of
the frozen Round-22 multi-UAV policies.

## Immutable dependencies

- `round_22/ROUND_LOCK.json` and its original outputs/checkpoints are
  byte-stable scientific evidence.
- The Round-21 H=5 DeepSets Task-Waypoint predictor remains frozen in eval
  mode. Its SHA must remain
  `645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221`.
- Use only Round-20 train/validation development data. The Round-20
  development test and untouched multi-UAV evaluation are prohibited.
- Fresh-development results are read-only Round-22 raw outputs. No new fresh
  rollout, environment construction, or access is allowed.

## Permitted work

- Load only frozen Round-22 checkpoints for inference and representation
  extraction.
- Run paired validation-only diagnostic rollouts under the same registered
  scenario configurations/snapshots, with normal, zero-head, and
  validation-local shuffled-Phi conditions.
- Recompute uncertainty, subgroup, tail, and seed agreement summaries from
  frozen Round-22 fresh-development output rows.

## Prohibited work

- Any optimizer, backward pass, training, checkpoint write, new seed, model
  architecture, loss, sampler, or predictor/environment/reward/matching/safety
  modification.
- HGNN, PPO/MAPPO, MARL, power, beamforming, or association actions.
- Fresh-development reruns, untouched-evaluation access, and Round-24
  creation.

## Completion

All Round-23 artifacts belong under `round_23`. The round must retain
`fresh_dev_new_rollout_count=0`, untouched-evaluation access count zero, and
the Round-22 MU-WP-B conclusion unchanged. Stop after the Round-23 lock.
