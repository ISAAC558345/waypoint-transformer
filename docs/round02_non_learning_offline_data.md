# Round 02: Non-learning baselines and offline-data audit

## Scope

Round 02 validates parameter-free baselines, collects complete short episodes,
and audits the resulting offline data. It does not create, optimize, or train a
Waypoint Network, Transformer, Decision Transformer, behavior-cloning model,
or reinforcement-learning policy.

The Stage-1A `MaritimeEnv` remains unchanged. In particular, it provides
positions, link rates, energy, reward components, and action/boundary
processing, but it does not model:

- buoy queues or urgency;
- service completion or task success;
- actual delivered/collected payload;
- an episode-level desired goal;
- an action mask (the current action is a continuous unit-disk command).

Unsupported values are represented by explicit availability flags and
zero-width arrays or blank CSV cells. They are never synthesized from link
rates.

## Non-learning policies

All policies implement:

```python
action = policy.act(observation, info)
```

They use a private explicit RNG where randomness is needed, consume only the
current observation/current `info`, return a finite vector with the dynamic
`action_dim`, and enforce `||action||_2 <= 1`.

| Policy | Current rule and fallback |
|---|---|
| `random_valid` | Uniform sample in the normalized action ball, using a private seeded NumPy generator. It is a coverage policy, not a performance reference. |
| `nearest_buoy` | Move toward the nearest valid physical buoy. If a queue field is supplied by a future compatible environment, empty queues are excluded. Stage 1A has no queue, so every valid physical buoy is eligible. |
| `max_queue` | Stable `argmax` over masked current queues, with the lowest index breaking ties. Stage 1A has no queue, so it returns the zero/hover action. |
| `urgency_rate_energy` | `normalized(U) * normalized(Q) * normalized(current_rate) / (M + epsilon)`. Missing `U` or `Q` factors are omitted, never invented. In Stage 1A both are absent, so the effective score is `normalized(current_rate) / (normalized one-step movement proxy + epsilon)`. `M` is a documented current movement proxy, not a duplicated environment energy formula. |

The policies support dynamic node masks and dynamic buoy counts. The current
environment has no padded node slots, so collected `valid_node_mask` values are
all true; the mask is nevertheless stored and tested.

## Runner and comparison schedule

`wt.maritime.runner` gives every policy the same ordered set of:

- environment configuration;
- `(scenario_id, seed, num_buoys)` cases;
- episode maximum length;
- termination and truncation rules.

The smoke command is:

```powershell
$env:PYTHONPATH = "src"
& "..\decision-transformer-master\.venv-dt-gym\python.exe" `
  -m wt.maritime.runner --smoke
```

The accepted smoke run used four policies, seeds `101,202,303`, buoy counts
`3,5`, and eight transitions per episode. It created 24 episodes and 192
transitions. A collector limit is stored as `truncated=true` with
`terminal_reason="collector_limit"`, so every persisted episode has an explicit
terminal boundary.

## Episode data model

Let:

- `T` be an episode's real transition count;
- `K_b` be its physical buoy count;
- `S` be the environment-provided state dimension (currently `6 + 4*K_b`);
- `A` be the environment-provided action dimension (currently 2).

Each episode is saved independently as one compressed NPZ file and one JSON
metadata file. No object arrays or pickle are used.

| Field group | Shape | Time meaning |
|---|---:|---|
| `observations`, `next_observations` | `[T,S]` | normalized `s_t` and `s_(t+1)` |
| `actions` | `[T,A]` | normalized `action_t` |
| `rewards`, `terminated`, `truncated`, `time_steps` | `[T]` | transition `t -> t+1` |
| `valid_transition_mask` | `[T]` | all true for unpadded complete episodes |
| `valid_node_mask` | `[T,K_b]` | real node slots at `s_t` |
| `action_mask` | `[T,0]` | explicitly unavailable in the current environment |
| UAV position/velocity and `next_*` | `[T,2]` | raw state before/after the action |
| buoy positions and `next_*` | `[T,K_b,2]` | raw state before/after the action |
| energy and `next_remaining_energy` | `[T]` | raw energy before/after the action |
| distance, gain, SNR, rate fields and `next_*` | `[T,K_b]` | raw channel state before/after the action |
| total rate and `next_total_rate_bps` | `[T]` | summed raw rate before/after the action |
| action processing, energy cost, reward components, violations | `[T,...]` | physical effects of `action_t` |
| `queue_values`, `transmitted_payload_bits` | `[T,0]` | unavailable, not fabricated |
| `desired_goal` | `[0]` | unavailable under `goal_spec_version="v0"` |
| `achieved_goal` | `[3]` | final physical `[x_m, y_m, remaining_energy]` |

`rates_bps` and `total_rate_bps` are the raw counterparts of `observation_t`.
The post-action `next_rates_bps` and `next_total_rate_bps` are used for the
environment's step communication utility and for the reported link-capacity
proxy. That proxy is not actual delivered payload.

The episode JSON stores `episode_id`, `policy_id`, `seed`, `scenario_id`, and
`num_buoys`; these identity values apply to every transition in that NPZ.
`dataset_schema.json` records the normalized observation segments, raw
counterparts, normalization formulas, identity broadcast semantics, and
pre/post time alignment.

## Split and leakage prevention

Splits are assigned to complete `(scenario_id, seed)` comparison groups, not to
individual transitions. Therefore:

- one episode cannot appear in more than one split;
- adjacent transitions cannot leak across splits;
- every policy and buoy scale initialized from the same scenario seed remains
  in one split.

The accepted smoke split is balanced:

| Split | Episodes | Transitions | Policies | Buoy counts |
|---|---:|---:|---:|---|
| train | 8 | 64 | all 4 | 3 and 5 |
| validation | 8 | 64 | all 4 | 3 and 5 |
| test | 8 | 64 | all 4 | 3 and 5 |

## K-step audit

A valid pair starts at `observation[t]` and targets
`next_observations[t+K-1] = s_(t+K)`. The target terminal state is legal, but a
done flag before that target is not. Real-transition masks must be true
throughout the window, and no pair may cross an episode boundary.

Accepted smoke results:

| K | Valid samples | Invalid because future state is beyond episode | Short episodes |
|---:|---:|---:|---:|
| 1 | 192 | 0 | 0 |
| 5 | 96 | 96 | 0 |
| 10 | 0 | 192 | 24 |
| 20 | 0 | 192 | 24 |
| 30 | 0 | 192 | 24 |

Terminal/padding boundary violations are zero for every policy, scale, split,
and K.

## Metrics and plots

Real metrics include return, length, potential cumulative link capacity, Jain
fairness over per-buoy link capacity, consumed and remaining energy, travelled
distance, average total rate, boundary/constraint/invalid-action counts, and
non-finite-value counts.

Task success, task completion, collected payload, remaining queue, per-buoy
service completion, and service fairness are marked unavailable. The two
required queue plots are availability notices rather than fake curves.

The six accepted figures are:

1. episode-return comparison;
2. final-queue availability notice;
3. cumulative-energy comparison;
4. queue-time-series availability notice;
5. raw episode and mean cumulative link-capacity proxy curves;
6. representative 2-D UAV trajectories and buoy locations.

All use a white background and the final two real time/trajectory figures
include policy, seed/scenario, and buoy-scale context. No 3-D plots are used.

## Accepted smoke result

Canonical run:

- data: `round_02/outputs/round02_smoke_20260730_212701_283649/`
- figures: `round_02/figures/round02_smoke_20260730_212701_283649/`
- 24 episodes, 192 transitions, 54 data/metadata files, and 6 figures;
- exact save/reload and checksum validation passed;
- no NaN/Inf, illegal action, boundary violation, or K-step boundary crossing;
- `training_performed=false`.

Mean values over six episodes per policy:

| Policy | Return | Link-capacity proxy [bit] | Energy | Distance [m] |
|---|---:|---:|---:|---:|
| `max_queue` | 3.6291 | 15,496,151 | 0.8000 | 0.000 |
| `nearest_buoy` | 3.5817 | 17,001,582 | 1.6000 | 80.000 |
| `random_valid` | 3.4222 | 15,820,201 | 1.3656 | 56.556 |
| `urgency_rate_energy` | 3.5817 | 17,001,582 | 1.6000 | 80.000 |

These numbers validate the pipeline, not policy optimality. In particular,
`max_queue` is a hover fallback because no queue exists, while the current
effective urgency heuristic selects the same targets as `nearest_buoy` in this
small configuration.

## Data sufficiency and stop condition

The smoke data is sufficient to verify collection, schema, splitting, exact
reload, metrics, plotting, and K-step construction for `K=1` and `K=5`.

It is **not sufficient for substantive Waypoint sample construction or model
work**:

- the eight-step horizon yields no samples for `K=10,20,30`;
- `desired_goal` is unavailable under goal spec v0;
- queue, actual payload, task completion, and urgency are absent;
- only three seeds, two buoy counts, one scenario family, and short episodes
  are covered;
- two heuristics collapse to the same behavior and `max_queue` must hover.

Round 02 therefore stops after pipeline validation. No next round, Waypoint
Network, Transformer, or training code is created.
