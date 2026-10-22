# Round-23 Completion Report

This is a frozen-checkpoint, inference-only audit.  It does not add a model seed, retrain a policy or predictor, run fresh-development rollouts, or access untouched evaluation.

## Seed identity and uncertainty

- Positive fresh-development seed: **2**; negative seeds: **[0, 1]**.
- Positive-seed win rate: 0.533; top-five positive contribution share: 0.505.
- Hierarchical seed-then-scenario bootstrap: 44001.50 bits, 95% CI [-134638.38, 260591.77].
- Removing the positive seed: -69996.54 bits.

## Causal heads and gates

- Service: HV-C; Progress: HV-B; Return: HV-A.
- Stability gate: **STAB-B**; temporal gate: **TV-B**; Round-23 readiness: **R23-A**.
- More training seeds recommended: true (recommendation only; no seed was added).

## Mechanism evidence

- predictor_bottleneck: **weak** — Oracle-versus-predicted offline gap is small in the locked Round-22 record and is reassessed by frozen subgroup rows.
- policy_initialization_instability: **strong** — Task-Waypoint policy payload seed variance is compared with matched State-only Transformer variance.
- return_head_over_dominance: **weak** — Return sensitivity and zero-Return causal payload result are reported separately.
- insufficient_service_utilization: **moderate** — Per-seed Service-to-total sensitivity ratios.
- insufficient_progress_utilization: **moderate** — Per-seed Progress-to-total sensitivity ratios.
- temporal_transformer_instability: **strong** — Task-Waypoint Transformer and MLP seed payload variances.
- conditional_conflict_bridge_waypoint_value: **moderate** — Pre-registered validation conflict/bridge delta payload rows by seed.

Round-22’s locked MU-WP-B gate and P-MU1 + P-MU4 classification are historical facts and remain unchanged. Final regression and source/output-lock verification are recorded in `ROUND_LOCK.json`.

## Final verification

Full regression passed **857/857** and `compileall` passed before this lock. The original Round-22 lock SHA remains `1D464A696ED8DE11E7AEDB14479F55ABB4F9F3E2621837E89CD854CD850190E4`; fresh-development new-rollout count and untouched-evaluation access count remain zero.
