# Service-resource decoder specification (amendment)

The diagnostic continuous action is `[delta_x, delta_y, pi_1,...,pi_K]`.  For valid buoys only, `pi_k >= 0`, `sum(pi)=1`, and padding entries are exactly zero.  The current-observable diagnostic priority is `normalized_queue_k * normalized_full_bandwidth_rate_k`; it is normalized to the masked simplex.  If all scores are zero, the deterministic fallback is the existing equal `1/K` allocation.

The sidecar assigns `fraction_k=pi_k`, `allocated_capacity_k=fraction_k*full_bandwidth_rate_k*delta_t`, and `payload_k=min(queue_k, allocated_capacity_k)`.  It rejects negative/nonfinite fractions, fraction sum above one, padding mass, and payload above queue/capacity.  It is not connected to `MaritimeEnv.action_space` or a training pipeline.
