# HTW Phase-10.1 Working Rules

L3-B six-UAV goal-conditioned low-level PPO pilot.

- Train only the new low-level actor-critic. Freeze DeepSets-util high-level seed 0.
- Do not load Phase-6.2 M=5 weights. Do not silently fall back to fly-toward.
- Do not edit L3-B physics, SINR, association, or the environment payload reward.
- Do not change hold=15, the order-free decoder, HGNN, or Transformer.
- Do not regenerate utility data or open confirmation / a new Round.
- Smoke (2 updates) then a fresh 100-update single-seed pilot. Stop after 100.
