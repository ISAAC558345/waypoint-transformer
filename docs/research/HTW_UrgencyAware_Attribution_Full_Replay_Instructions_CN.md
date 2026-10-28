# Stage E9：在已验证运行时上下文下完成紧急度奖励归因重放

- Repository: ISAAC558345/waypoint-transformer
- Formal baseline branch: feature/maritime-wt-stage1a
- Source experiment branch: experiment/priority-tag-urgency-reward
- Attribution source branch: experiment/priority-tag-reward-attribution
- Runtime-context source branch: experiment/priority-tag-eval-context
- Current branch: experiment/priority-tag-attribution-full-replay
- E8 source commit: b3ff3842c2d6611f575a43002a6cc9a4482573d1
- Prior gate: EVAL_CONTEXT_REPRO_PASS
- This-stage gates: PRIORITY_TAG_ATTRIBUTION_COMPLETE / PRIORITY_TAG_ATTRIBUTION_INCONCLUSIVE / PRIORITY_TAG_ATTRIBUTION_BLOCKED
- Scope: deterministic instrumentation replay and metric attribution only. No training, no reward changes, no merge.

## 1. E8 result

E8 verified two independent processes replaying L0, model seed 0, DEV scenario 321101 for 48 steps. Both matched the saved episode within the unchanged 1-bit accounting/payload tolerance and 1e-9 m path tolerance. The manifest records:
- torch threads = 1;
- deterministic algorithms = True;
- historical inherited global RNG seed = 2, explicitly marked as inherited context, not a canonical evaluation seed;
- model seed = 0, environment reset seed = 321101, assignment RNG seed = 7321101;
- exact checkpoint and code hashes.
The current environment context reproduced the historical saved result. Do not claim seed 2 was a documented canonical evaluation seed; it was the inherited post-training process state. Preserve this distinction.

## 2. Objective

Now complete the E6 attribution for the same 224 saved evaluation episodes using the validated runtime context, to determine whether the observed rise in combined urgency-weighted service comes from:
1. age-only service;
2. incremental synthetic-tag boost;
3. actual tagged bits served;
and whether clipping suppresses tag boost for older tagged buckets.

This is a diagnostic replay of existing frozen policies, not a new benchmark or new policy comparison.

## 3. Frozen scope

Use the exact 224 evaluation rows and same variants/seeds/scenarios already present in:
- results/phase6i_priority_tag_reward/scenario_rows.csv
- results/phase6i_priority_tag_reward/summary.json
The rows are L0 seed 0; L1 seeds 0/1/2; L2 seeds 0/1/2; each evaluated on DEV_POOL 321096–321127 (32 scenarios), totaling 7 × 32 = 224 episodes.

Before running, verify the repository registry and split guard for every scenario. Do not use TRAIN_POOL for this evaluation, and never access reserved/test splits.

Reuse the exact frozen checkpoints from source commit f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb and verify every SHA-256 against the existing manifest. Do not retrain, alter checkpoint, alter action selection, alter model config, or tune any parameter.

For every episode, explicitly apply the validated runtime flags: torch threads=1 and deterministic_algorithms=True; restore the historical inherited global RNG context only as documented in runtime_context.json (seed=2), while preserving per-episode environment reset and assignment RNG seeds from the original protocol. Record this as a historical-context reproduction, not a canonical evaluation protocol. Do not mix or relabel model seed, reset seed, assignment seed, and process RNG seed.

## 4. Preflight and time budget

- Read E8 runtime_context.json, summary.json, step logs, E6 source artifacts, source code and checkpoint manifest.
- Run one episode first (L0 seed 0, scenario 321101) and require payload/served/drop fields within 1 bit, path within 1e-9 m, and exact age+tag decomposition within numerical tolerance.
- Estimate the full 224-episode runtime from the measured episode. If estimated runtime exceeds 10 minutes, stop and report; do not extend the budget.
- If the preflight episode fails, stop immediately with BLOCKED. Do not run the remaining episodes.
- Do not alter tolerances to force a pass.

## 5. Required per-bucket attribution

At each step, use the actual pre-service FIFO bucket state and exact served bits per bucket:
- u_age = clip(age / 48, 0, 1)
- u_combined = clip(u_age + 0.25 * synthetic_priority_tag, 0, 1)
- age_service_bits = sum(u_age * served_bits_by_bucket)
- tag_increment_service_bits = sum((u_combined - u_age) * served_bits_by_bucket)
- tagged_served_bits = sum(synthetic_priority_tag * served_bits_by_bucket)
- combined_urgency_served_bits = age_service_bits + tag_increment_service_bits

Do not infer tag service from buoy-level aggregates. Verify the decomposition against the existing combined urgency metric. Record tagged-service bits whose incremental boost is below 0.25 and the portion whose tag boost is fully saturated because u_age is already 1.

For each variant/model seed/scenario, report:
- total payload;
- ordinary/tagged served and dropped bits;
- age-only weighted service, tag-increment service, combined weighted service;
- tag rate, tagged service ratio, mean/max age at service by tag;
- final ordinary/tagged queue;
- SINR/interference, path length, projection rate, safety violations, constraint failures;
- ledger/accounting gaps and match-to-saved flags.

## 6. Reconciliation and validity

- Recompute episode and seed summaries directly from scenario rows; check all counts and confidence intervals against the original report.
- Require all 224 rows to match saved payload/served/drop/queue/path values within the established tolerances.
- Require age_service_bits + tag_increment_service_bits to match combined_urgency_served_bits within 1e-6 bit numerical tolerance.
- Keep the original 1-bit accounting tolerance and 1e-9 m path tolerance.
- Verify formal checkpoint SHA-256 and formal code hashes unchanged.
- If any episode fails reproducibility, stop further attribution and mark BLOCKED; do not interpret partial attribution as a full result.

## 7. Deliverables

- docs/research/HTW_UrgencyAware_Reward_Attribution_Report_CN.md (update only on this branch, preserving the E6 blocked report on its original branch)
- results/phase6i_priority_tag_attribution_full_replay/scenario_attribution.csv
- results/phase6i_priority_tag_attribution_full_replay/step_attribution.csv
- results/phase6i_priority_tag_attribution_full_replay/summary.json
- results/phase6i_priority_tag_attribution_full_replay/runtime_manifest.json
- results/phase6i_priority_tag_attribution_full_replay/schema.json

The report must explain which share of the combined urgency signal comes from age vs tag increment; whether tag clipping saturates; whether L1/L2 changes actual tagged service; and whether the original metric mismatch is confirmed. Explicitly state tags are synthetic.

## 8. Stop and merge rules

- Do not retrain L1/L2, change lambda/rho/beta/tag_rate, change queue order or modify formal system.
- Do not merge any branch during this stage.
- If complete and reproducible, mark PRIORITY_TAG_ATTRIBUTION_COMPLETE and recommend the next model-design step based on evidence.
- If results are stable but do not support a clear attribution, mark INCONCLUSIVE.
- If replay, hashes, split guard or accounting fails, mark BLOCKED and stop.
- Complete the report and artifacts on this branch, then stop.
