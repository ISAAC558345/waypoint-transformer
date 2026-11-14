# Reward semantics and alignment audit (amendment)

## Exact active v1 reward

Source of truth: `MaritimeEnv._step_v1` (lines 666--719).  For each slot,

`R = r_payload + r_energy + r_boundary + r_invalid + r_depletion + r_unfinished + r_success`.

| component | exact formula | coefficient / normalization | unit and physical meaning | sign and range |
|---|---|---|---|---|
| payload_reward | `w_p * clip(sum(payload_k)/(R_max_total*delta_t), 0, 1)` | `w_p=1`; denominator is theoretical full-bandwidth aggregate bit capacity | dimensionless reward for actual served bits | `[0, 1]` |
| energy_penalty | `-w_e * clip(actual_energy_consumed/(hover_energy_per_step + move_energy_per_meter*maximum_step_distance), 0, 1)` | `w_e=0.05` | dimensionless movement/hover energy cost | `[-0.05, 0]` |
| boundary_penalty | `-boundary_penalty` iff the candidate movement crosses the area boundary | `boundary_penalty=0.5` | projected-motion constraint violation | `{0, -0.5}` |
| invalid_action_penalty | `-invalid_action_penalty` iff clipping or radial projection adjusts the action | `invalid_action_penalty=0.2` | action-interface constraint violation | `{0, -0.2}` |
| energy_depletion_penalty | `-energy_depletion_penalty` iff energy is depleted | `energy_depletion_penalty=1` | terminal energy constraint | `{0, -1}` |
| unfinished_queue_penalty | `-w_u * residual_queue/(K*queue_capacity)` only at episode end and only without task success | `w_u=0.1` | terminal residual communication backlog | `[-0.1, 0]` |
| task_success_bonus | `task_success_bonus` only for task success when terminate-on-success applies | `task_success_bonus=1` | terminal completion/reliability bonus | `{0, 1}` |

`payload_k=min(queue_k, allocated_capacity_k)`, so the payload term respects the
queue and physical link-capacity constraints.  The resource sidecars preserve
the original movement, movement costs, arrivals, and horizon; only the valid
resource fraction changes.

## Resolution of the apparent discrete-return anomaly

The immutable original discrete sidecar reported `Delta_return=20.35574786907741`
with 95% CI [`13.746273722783798`, `27.281686678540655`].  Its calculation was not
an exact reconstruction of `MaritimeEnv._step_v1`: it normalized replacement
payload by `after[total_rate_bps]*delta_t` instead of the environment's
theoretical full-bandwidth denominator, and used offered-data normalization
for the terminal residual term.  This amendment first verifies zero
current-mechanism reconstruction error, then applies the exact formula above
to every sidecar.  Therefore the earlier positive discrete return is not
evidence that the active reward prefers worse communication.

The exact component distributions and paired deltas are in
`reward_component_metrics.csv`; correlations are in
`reward_alignment_audit.csv`; the independent H=10 return-head audit is in
`return_head_alignment_audit.csv`.
