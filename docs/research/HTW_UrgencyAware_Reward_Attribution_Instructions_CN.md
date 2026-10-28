# Stage E6：紧急度奖励归因审计与标签目标对齐

- Repository: ISAAC558345/waypoint-transformer
- Base branch: feature/maritime-wt-stage1a
- Source experiment branch: experiment/priority-tag-urgency-reward
- Diagnostic branch: experiment/priority-tag-reward-attribution
- Source result commit: f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb
- Gate: PRIORITY_TAG_EXPERIMENT_NOT_MERGED
- Purpose: explain why age-weighted urgency service rose in some seeds while actual tagged service did not improve. This is a diagnostic stage, not another reward sweep.

## 1. Current evidence and decision

The E1–E3 experimental implementation trained L1 (new age/tag/validity observations, base reward unchanged) and L2 (same observations plus bounded urgency reward lambda_u=0.0005, rho=0.001) for 300 updates on seeds 0/1/2. The latest paired DEV_POOL report found:
- L1 and L2 each had zero seeds meeting the predeclared merge criteria.
- L1's age-weighted urgency service was positive in all seeds, but tagged-served-bit confidence intervals crossed zero in all seeds.
- L2's age-weighted urgency service was positive in all seeds, but tagged-served-bit mean was negative in all three seeds and all tagged-service confidence intervals crossed zero.
- L2 seed 2 had +2 safety violations versus L0.
- Payload mean differences were positive for the seeds, but confidence intervals crossed zero for all L2 seeds and L1 seeds 0/1; scenario-level payload regressions occurred frequently.
- Result is PRIORITY_TAG_EXPERIMENT_NOT_MERGED; do not merge this branch into the base branch.

These results suggest a target/metric mismatch: urgency_served_bits combines waiting-age contribution and tag contribution, so it can rise without increasing the actual number of tagged bits served. This hypothesis must be measured rather than assumed.

## 2. Required diagnostic questions

1. How much of urgency_served_bits comes from age-only urgency versus the incremental priority-tag boost?
2. Are tagged bits typically older or younger than ordinary bits when served? Does clipping clip(age/48 + 0.25*tag, 0, 1) saturate the tag boost for old buckets?
3. Is L2's positive weighted-urgency result mainly due to serving older ordinary information rather than tagged information?
4. Are tagged service changes caused by target selection across buoys, FIFO within each buoy, tag-generation rate, or accounting/metric mismatch?
5. Does the reward actually incentivize the user-stated objective (serve tagged urgent information) or mostly incentivize old data service?
6. Are any evaluation/report discrepancies present between per-scenario CSV rows, summary counts, confidence intervals, and the report? Reconcile all counts from source CSVs before interpretation.

## 3. Data and replay protocol

First inspect the existing implementation and artifacts:
- src/wt/benchmark_v2/phase6i_priority_tag_reward/
- results/phase6i_priority_tag_reward/scenario_rows.csv
- results/phase6i_priority_tag_reward/summary.json
- results/phase6i_priority_tag_reward/logs/
- docs/research/HTW_UrgencyAware_Experiment_Report_CN.md

Record source hashes and commit IDs. If existing artifacts contain exact per-step/per-bucket age and tag service, analyze them directly and do not rerun evaluation.

If exact age/tag decomposition is not logged, run only an instrumented deterministic evaluation replay using the existing frozen L0 and already-trained L1/L2 checkpoints. Do not retrain or change any checkpoint/config/action. Use the same registered DEV_POOL scenarios and paired reset seeds from the original evaluation, same deterministic actions, same synthetic tag seed protocol, and split guard. First run one scenario and estimate time; if the full replay is estimated to exceed 10 minutes, stop after the first reproducibility check and report the missing instrumentation/estimate rather than expanding.

Replay must not use reserved/test splits or tune any parameter. It is an instrumentation replay, not a new model comparison and must reproduce the existing payload/served-bit results within the recorded accounting tolerance. If it does not, stop and report BLOCKED.

## 4. Exact attribution definitions

For each FIFO bucket b before service:
- u_age_b = clip(age_b/48, 0, 1)
- u_combined_b = clip(u_age_b + 0.25*synthetic_tag_b, 0, 1)
- age_service_bits = sum_b(u_age_b * served_bits_b)
- tag_increment_service_bits = sum_b((u_combined_b-u_age_b) * served_bits_b)
- tagged_served_bits = sum_b(synthetic_tag_b * served_bits_b)
- combined_urgency_served_bits = age_service_bits + tag_increment_service_bits

Use the actual pre-service bucket state and actual per-bucket FIFO service allocation. Do not infer tagged service from buoy-level totals. Confirm the decomposition equals the existing combined urgency metric within numeric tolerance. Report clipping saturation: tagged bits served where u_combined_b-u_age_b < 0.25, including the fraction of tagged served bits whose incremental tag boost was fully saturated.

Report per scenario/seed/model:
- payload bits;
- ordinary/tagged served bits and dropped bits;
- age-only weighted service;
- incremental tag-boost service;
- combined weighted service;
- tag rate and tagged service ratio;
- mean/max age at service by tag;
- final tagged/ordinary queue;
- queue accounting, safety violations, projection rate, SINR/interference;
- paired deltas versus L0 and 95% confidence intervals at the scenario level.

## 5. No model changes in this stage

- Do not change lambda_u, rho, beta_tag, tag_rate, queue order, action space, observation schema, task distribution or training budget.
- Do not retrain L1/L2 or run a new reward sweep.
- Do not change the official reward, formal Planner, formal observation, SafetyProjector or L0 checkpoint.
- Do not merge the source experiment branch.
- Do not use synthetic tag results to claim real maritime semantic priority performance.

## 6. Decision output

Create:
- docs/research/HTW_UrgencyAware_Reward_Attribution_Report_CN.md
- results/phase6i_priority_tag_reward_attribution/summary.json
- results/phase6i_priority_tag_reward_attribution/scenario_attribution.csv
- results/phase6i_priority_tag_reward_attribution/schema.json

The report must state whether the current reward primarily responds to age or to tag, whether tag clipping materially suppresses the tag contribution, whether metric accounting is valid, and which next design is justified:
A. stop priority-tag reward work because it does not improve tagged service;
B. design a new capped reward with separate age and direct tagged-service components, then evaluate it on a fresh experiment branch;
C. fix instrumentation/accounting if the current metrics cannot be reconciled.

Do not choose a new formula or start training in this stage. Any next design must remain isolated on a new branch and be evaluated against L0 under predeclared criteria.

## 7. Stop and merge rule

No merge. The current experiment failed its predeclared criteria. This diagnostic branch is also not merged unless it contains only diagnostics that are useful independent of model performance; preferably retain the report on the experiment branch and keep the formal branch unchanged.

Complete the audit, commit the report/results on experiment/priority-tag-reward-attribution, and stop. Do not start another training run.
