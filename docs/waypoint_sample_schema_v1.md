# Waypoint supervised-sample schema v1

## Scope

Round 03 defines and materializes supervised labels only. It does **not**
train a Waypoint Network, Transformer, Decision Transformer, behavior-cloning
policy, or reinforcement-learning policy. Every generated manifest records
`training_performed=false`.

The source must be a schema-v1 maritime trajectory. Schema-v0 Round 02 data
does not contain real queue, delivered-payload, or task-completion semantics
and is rejected rather than silently mixed.

## Temporal definition

For one episode, an anchor timestep `t`, and
`K in {1, 5, 10, 20, 30}`, the supervised relation is

```text
(state_t, goal) -> waypoint_(t+K)
```

The exact target is read from
`next_observations[t + K - 1]`, because transition index `j` stores
`s_j -> s_(j+1)`. Therefore:

- `target_t = t + K`;
- `target_transition_index = t + K - 1`;
- `t` and `target_t` always belong to the same episode and split;
- every transition in `[t, t+K-1]` must have a true
  `valid_transition_mask`;
- no `terminated` or `truncated` flag may occur in the internal transition
  range `[t, t+K-2]`;
- a done flag at `t+K-1` is allowed and is saved separately as
  `endpoint_terminated` or `endpoint_truncated`;
- temporal padding is never a valid label.

Hindsight goals may read the terminal state of their **own source episode**.
They never read a validation/test episode while constructing a train sample.
The supplied source-dataset episode-to-split assignments are authoritative;
the sample builder does not resplit transitions.

## Source observation layout

Task-semantics v1 uses a flat observation of width

```text
D_obs(N) = 9 + 7*N
```

where `N` is the episode's real buoy count. The fields consumed by this
builder are:

| Quantity | Flat location |
|---|---:|
| normalized UAV `(x, y)` | base indices `0:2` |
| remaining-energy ratio | base index `4` |
| normalized queue of buoy `i` | node `i`, offset `4` |
| service-completion ratio of buoy `i` | node `i`, offset `6` |

The remaining observation fields are preserved as state inputs but are not
copied into the waypoint target.

## Waypoint vector

For dataset-wide maximum buoy count `N_max`, the waypoint width is

```text
W = 3 + 2*N_max
```

and its deterministic order is:

```text
[
  normalized_uav_x,
  normalized_uav_y,
  normalized_queue[0:N_max],
  service_completion_ratio[0:N_max],
  remaining_energy_ratio
]
```

An episode with `N < N_max` pads queue and completion slots `[N:N_max]` with
exact zeros. The matching `waypoint_mask_t_plus_k` is false for those padded
features, so future loss reduction must assign them zero weight.
`valid_node_mask_t_plus_k` is also stored explicitly.

Instantaneous random channel noise, independent future arrivals, cumulative
logging counters, episode IDs, seeds, and other metadata are intentionally
not prediction targets.

## Model-input state

The stored `state_t` has fixed width

```text
W + (9 + 7*N_max) + N_max
```

and concatenates:

1. the current-state waypoint projection (`W`);
2. the complete source observation padded to `9 + 7*N_max`;
3. `valid_node_mask_t` converted to float.

`state_mask_t` has exactly the same shape. Base features, real-node features,
and the explicit mask-vector slots are valid. Padded node features are false
and numerically zero. The explicit node-mask slots themselves remain valid
inputs, including their zero values for padded nodes.

## Goal sources

The configurable sources are:

### `achieved_terminal_goal` (default)

This hindsight goal maps the actual terminal result to waypoint order:

- actual final normalized UAV position;
- actual final per-buoy normalized queue;
- actual final per-buoy service-completion ratio;
- actual final remaining-energy ratio.

The builder cross-checks these goal fields against the final
`next_observation`; inconsistent trajectories are rejected.

### `desired_goal`

The desired goal has source order

```text
[
  target_task_completion_ratio,
  maximum_allowed_residual_queue_ratio,
  minimum_terminal_energy_ratio
]
```

It maps into waypoint order as follows:

- terminal position is unspecified: numeric zero with mask false;
- the residual-queue threshold is broadcast to every real buoy queue slot;
- the task-completion target is broadcast to every real buoy completion slot;
- the minimum terminal-energy ratio is valid;
- padded buoy slots remain zero with mask false.

Desired and achieved goals remain separate semantics and are never stored as
one ambiguous field.

## Persisted arrays

Each split has an independent compressed NPZ file:

- `waypoint_samples_train.npz`;
- `waypoint_samples_validation.npz`;
- `waypoint_samples_test.npz`.

Every sample stores at least:

- `split`, `sample_id`, `episode_id`, `scenario_id`, `seed`, `policy_id`,
  `arrival_trace_id`, `num_buoys`, and `episode_length`;
- `t`, `k`, `target_t`, and `target_transition_index`;
- `state_t`, `state_mask_t`, and `valid_node_mask_t`;
- `goal`, `goal_mask`, and `goal_source`;
- `waypoint_target_t_plus_k`, `waypoint_mask_t_plus_k`, and
  `valid_node_mask_t_plus_k`;
- internal termination/truncation flags and endpoint done flags;
- trajectory, goal, and waypoint schema versions.

All numeric arrays have explicit NumPy dtypes. Text arrays use fixed-width
Unicode, never Python-object arrays. Loading always uses
`allow_pickle=False`. The manifest records SHA-256 checksums for every NPZ,
the machine-readable schema, and the K-statistics CSV.

## Reproducibility and ordering

Samples are sorted by:

1. split order `train`, `validation`, `test`;
2. `episode_id`;
3. `K`;
4. `t`.

For each split and K, `waypoint_k_statistics.csv` records candidate anchors,
valid samples, short horizons, padding rejection, internal done rejection,
and valid endpoint done counts.

## Future official-model integration boundary

The official Waypoint Transformer training code currently assumes its own
fixed feature layout and loss interface. Round 03 deliberately does not alter
that model. Before any later training round, two explicit adapters are still
required:

1. an input/output adapter that maps this variable-scale, schema-v1 state,
   goal, and waypoint order into the official model's configured dimensions;
2. a mask-aware waypoint loss that applies
   `waypoint_mask_t_plus_k` **before** feature reduction, so padded buoy
   targets and the unspecified desired-goal position contribute exactly zero.

Until those adapters and their tests exist, successful sample construction is
not evidence that official-model training is ready or valid.
