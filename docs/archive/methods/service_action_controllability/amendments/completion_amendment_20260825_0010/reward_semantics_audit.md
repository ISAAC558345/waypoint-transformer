# Reward semantics and alignment audit (amendment)

## Exact active v1 reward

`reward = payload_reward + energy_penalty + boundary_penalty + invalid_action_penalty + energy_depletion_penalty + unfinished_queue_penalty + task_success_bonus`.

- `payload_reward = payload_reward_weight * clip(sum(payload)/(theoretical_max_total_rate_bps*delta_t), 0, 1)`; weight is 1.0.
- `energy_penalty = -energy_penalty_weight * clip(actual_energy_consumed/(hover_energy_per_step + move_energy_per_meter*maximum_step_distance),0,1)`; weight is 0.05.
- Boundary penalty is -0.5 only on a boundary violation; invalid-action penalty is -0.2 only after action adjustment; depletion penalty is -1.0 only on depletion.
- At terminal only, unfinished penalty is `-0.1 * residual_queue/(K*queue_capacity)` unless task-success conditions hold; then the success bonus is +1.0 when termination-on-success applies.

The original discrete sidecar return was not an exact environment reconstruction: it normalized replacement payload by `after[total_rate_bps]*delta_t` rather than theoretical maximum total rate, and used offered-data normalization for the final residual penalty.  The amendment's current-mechanism reconstruction error is required to be zero, then applies the exact same formula to the fixed-movement sidecars.  Therefore the original +20.356 discrete return must not be treated as proof that the real environment reward prefers worse communication.

Component correlations, H=10 return correlations, and grouped/partial correlations are in `reward_alignment_audit.csv`; per-scenario component contributions are in `reward_component_metrics.csv`.
