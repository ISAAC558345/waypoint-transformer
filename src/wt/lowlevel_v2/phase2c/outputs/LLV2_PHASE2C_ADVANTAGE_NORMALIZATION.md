# Advantage normalization

Implemented as **A**: normalize GAE over all env-steps in the rollout, then apply UAV `policy_active_mask` only inside the policy/entropy terms.

Not **B** (active-only advantage normalize).

See `LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md` and minibatch replay stats.
