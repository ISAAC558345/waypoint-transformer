# Round-5 causal policy schema v1

## Scope and provenance

Round-5 consumes the immutable Round-3 pilot and its fixed episode-level
train/validation/test split. It loads, but never updates, the validation-picked
Round-4 goal-conditioned spatial Waypoint Network with horizon 30. Every cache,
sequence manifest, checkpoint, and metric file records both source hashes.

The experiment is offline hindsight action prediction. The materialized goal
is `achieved_terminal_goal`; it is future outcome information available for
this offline comparison, not an established online goal interface. No result
from this round is a closed-loop return claim.

## Per-transition fields

Dimensions are read from the locked schemas rather than model constants. The
current canonical data happens to have:

| Field | Current shape | Model use |
| --- | ---: | --- |
| `masked_state_t` | `[62]` | all policies |
| `valid_node_mask_t` | `[5]` | state encoder masking/interface |
| `masked_goal_t` | `[13]` | frozen waypoint input; Direct-Goal only |
| `predicted_spatial_delta_h30` | `[2]` | WT and waypoint MLP only |
| `oracle_spatial_delta_h30` | `[2]` | non-deployable oracle diagnostic only |
| `remaining_task_fraction` | `[1]` | Transformer policies |
| `executed_action_normalized` | `[2]` | target only |

The physical action target is computed independently of the stored policy
command:

```text
executed_action_m = next_uav_positions_m[t] - uav_positions_m[t]
executed_action_normalized = executed_action_m / maximum_step_distance_m
```

`maximum_step_distance_m = max_speed * delta_t` comes from the source
environment configuration. Model outputs pass through `tanh`. Raw predicted
actions are de-normalized with the same factor and may be radially projected to
the configured disk for evaluation. Projected predictions never become labels.

## Conditions and forbidden inputs

- WT: state + node mask + predicted H=30 spatial delta + remaining fraction.
  It does not receive the global goal or oracle delta.
- Direct-Goal: state + node mask + materialized global goal + remaining
  fraction. It does not receive a waypoint.
- State-only: state + node mask + remaining fraction. It receives neither goal
  nor waypoint.
- Oracle-Waypoint: same policy interface as WT but substitutes the true H=30
  displacement. It is `non_deployable=true` and ineligible for selection.
- MLP controls are one-step versions of state-only and predicted-waypoint
  conditioning.

No model input contains action labels, historical actions, future states,
rewards, return/RTG, policy ID, quality tier, seed, or terminal reason. Policy
and quality metadata exist only for sampling audits and grouped evaluation.

## Sequence layout and masks

One deterministic example ends at every valid transition. Its history contains
at most `H_c` states from the same episode and split, left padded when needed.

- feature mask: schema-derived state/node validity;
- attention mask: boolean `[batch, time]`, `true` for a real timestep;
- loss mask: boolean `[batch, time]`; only the current final real timestep is a
  supervised target, so every source transition contributes exactly once.

The official causal self-attention implementation combines a lower-triangular
causal mask with the key padding mask. Padding activations and losses are
explicitly zeroed. Windows never cross an episode, terminal, truncation, or
split boundary.

## Frozen H=30 cache

Each split cache records episode ID, timestep, split, source-state hash, goal
hash, checkpoint hash, normalized/raw predicted delta, normalization version,
and effective remaining horizon. The frozen network is loaded strictly,
switched to evaluation mode, marked `requires_grad_(False)`, omitted from all
optimizers, and called under `torch.no_grad()`.

For oracle diagnostics near an episode end, the target index is clamped to the
valid terminal endpoint and `effective_horizon < 30` is recorded. Oracle values
never enter a deployable model.

## Selection boundary

Best epochs and the WT context `H_c in {1,10}` are selected solely by
validation physical action RMSE. The test split is evaluated only after an
immutable selection record has been written. Test metrics cannot trigger
training, checkpoint, seed, context, or hyperparameter changes.

## Reproducibility and action-space tolerance

Each run resets Python, NumPy, CPU-Torch, and CUDA-Torch RNG state before
**model construction**, as well as before optimizer construction. This makes
parameter initialization, policy-balanced sampling, dropout, checkpoints, and
resumed optimization run-specific and reproducible from the declared seed.

Action-space validity metrics use `max(1e-9 m, action_near_zero_epsilon_m)` as
their numerical tolerance. The canonical value is `1e-6 m`; it prevents a
float32 reconstruction of a nominal 10 m executed boundary action from being
misreported as invalid. Radial projection remains explicit and is not a label
transformation.
