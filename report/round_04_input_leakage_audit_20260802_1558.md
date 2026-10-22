# Round-4 input-leakage audit

Date: 2026-08-02 15:58 +08:00  
Canonical source:
`round_03/outputs/round03_pilot_canonical_20260730_230000`

## Decision

The Round-3 `state_t` tensor is causal at sample time `t` and may be used after
applying its stored mask. The primary `achieved_terminal_goal` is deliberately
non-causal task-outcome information and is allowed only as an explicitly
separated hindsight goal-conditioning channel. It must never be merged into or
described as the observable state.

No Round-4 predictor input may contain target arrays, future masks, terminal
metadata, returns, future rewards, future arrivals, future actions, episode
identifiers, split identifiers, policy identifiers, or validation/test
statistics.

## Audited predictor channels

| Channel | Shape | Time semantics | Decision |
|---|---:|---|---|
| `state_t` | `[B,62]` | current waypoint projection, current observation, current node mask | allow after `state_mask_t` |
| `state_mask_t` | `[B,62]` | structural validity at `t` | use only to clear padding |
| `goal` | `[B,13]` | achieved terminal task outcome | allow only as declared hindsight goal |
| `goal_mask` | `[B,13]` | valid goal coordinates | use only to clear padding |
| `valid_node_mask_t` | `[B,5]` | current valid nodes | allow |

The first 13 entries of `state_t` duplicate a current-state waypoint projection.
The duplication is redundant but contains no future value. Invalid three-buoy
slots are exactly zero and masked.

The final model tensors are derived from schema widths rather than constants:

- goal-conditioned input:
  `[masked state_t, masked goal, valid_node_mask_t]`;
- state-only input:
  `[masked state_t, valid_node_mask_t]`;
- output: normalized two-dimensional relative displacement.

The observed canonical widths are 80, 67, and 2, respectively.

## Explicit denylist

The adapter rejects these values as predictor features:

- `waypoint_target_t_plus_k`, `waypoint_mask_t_plus_k`;
- `valid_node_mask_t_plus_k`;
- `terminated_between_t_and_t_plus_k`,
  `truncated_between_t_and_t_plus_k`;
- `endpoint_terminated`, `endpoint_truncated`, `episode_length`;
- `episode_id`, `sample_id`, `arrival_trace_id`, `policy_id`, `split`,
  `scenario_id`, schema/version identifiers, and seed;
- future actions, velocities, arrivals, rewards, cumulative payload, target
  delta, and validation/test aggregate statistics.

These fields remain available only for target reconstruction, provenance, or
post-prediction grouped evaluation.

## Hindsight-goal caveat

`achieved_terminal_goal` contains terminal UAV position, terminal queues,
completion ratios, and remaining energy. It is therefore future information
under a strictly causal deployment interpretation. Round-4 uses it because the
authorized experiment explicitly defines hindsight goal conditioning.

For 400 samples whose `t+H` endpoint is the episode endpoint, the terminal goal
position equals the spatial target endpoint. This creates a measurable
shortcut. Results must be described as offline hindsight-conditioned
prediction, not online causal forecasting. Endpoint and non-endpoint samples
will be retained and may be audited separately; none are silently deleted.

`desired_goal` is not used as a primary experiment because it supplies only
two effective patterns (three- versus five-buoy padding) and has no specified
position target. Fabricated goal variation is prohibited.

## Target and normalization

For every retained sample:

```text
q_current = episode.uav_positions_m[t]
q_future  = episode.next_uav_positions_m[target_transition_index]
raw_delta_xy_m = q_future - q_current
maximum_step_distance_m = run_config.max_speed * run_config.delta_t
normalized_delta_xy = raw_delta_xy_m / (H * maximum_step_distance_m)
```

The scale is read from the locked run configuration. It is not estimated from
train, validation, or test statistics. Current values are
`max_speed=10 m/s`, `delta_t=1 s`, and
`maximum_step_distance=10 m`.

Full-data audit: maximum normalized displacement norm is
`1.0000000000000018`; reachability violations are zero. Raw-to-normalized
reconstruction maximum error is below `9e-6 m`.

## Dataset and lock verification

- repository regression: 146/146 passed before Round-4 changes;
- Round-3 locked hashes: 57/57 matched;
- episodes/transitions: 80/6,604;
- split episodes: 48 train, 16 validation, 16 test, no overlap;
- K counts for 1/5/10/20/30:
  6,604 / 6,284 / 5,884 / 5,084 / 4,284;
- Round-3 arrays remain read-only and are not rewritten by the adapter.

## Residual risks

- Hindsight-goal shortcut at terminal endpoints.
- Valid K samples condition on survival to `t+H`, creating a
  survivorship/censoring bias.
- The canonical set contains high- and mid-quality episodes but no low-quality
  class.
- `policy_id` is retained only for evaluation stratification; it is not a
  model feature.

These risks do not block the authorized offline supervised experiment, but
they limit any claim to learnability under this fixed hindsight-conditioned
dataset.
