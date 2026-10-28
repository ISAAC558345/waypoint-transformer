# Stage E10：分离年龄奖励与直接标签服务奖励的受控实验

- Repository: ISAAC558345/waypoint-transformer
- Formal baseline branch: feature/maritime-wt-stage1a
- Source experiment: experiment/priority-tag-urgency-reward
- Attribution source commit: 88c0e0296d6f64998a856af47f9c55dc6667c04b
- Current experiment branch: experiment/priority-tag-direct-service-reward
- Prior gate: PRIORITY_TAG_ATTRIBUTION_COMPLETE
- This-stage gates: DIRECT_TAG_REWARD_BENCHMARK_PASS / DIRECT_TAG_REWARD_BENCHMARK_FAIL / DIRECT_TAG_REWARD_BENCHMARK_INCONCLUSIVE / DIRECT_TAG_REWARD_BLOCKED
- Workflow: no extra owner-approval gate. Implement and test in this isolated branch; merge only if predeclared criteria pass.

## 1. E9 evidence and design conclusion

E9 successfully replayed all 224 frozen evaluation episodes with validated runtime context:
- 224/224 rows matched saved results; max payload/bit gap = 0; max path gap = 0;
- age + tag-increment decomposition matched combined urgency and ledger accounting;
- formal checkpoint hash remained unchanged.
Gate: PRIORITY_TAG_ATTRIBUTION_COMPLETE.

Main findings:
- On L0, age-weighted service sum = 436,047,051.662 bit-equivalent; tag-increment service = 86,899,997.030; combined urgency service = 522,947,048.692.
- L0 mean generated synthetic tag rate was about 0.251.
- Full saturation of the tag boost was 0%; tag increment was reduced below its nominal 0.25 for about 2.94% of tagged served bits; approximately 99.23% of the nominal tag boost was retained. Therefore clip saturation is not the main explanation for the failure.
- For L1 and L2, the combined urgency-weighted service improvement was mostly driven by age service, especially ordinary information, while the direct tag-increment delta was small or negative and the tagged-served-bit confidence intervals crossed zero.
- For L2, tagged-served-bit mean deltas versus L0 were negative for all three seeds; the 95% intervals crossed zero. The old reward is therefore not demonstrably improving actual tagged service.

Interpretation: the next hypothesis to test is reward-target misalignment. Keep the age component but give actual tagged bits a separate, explicit and bounded service bonus. This is a new experiment hypothesis, not a claimed improvement.

## 2. Isolated variant definitions

Keep L0, L1, L2 as existing frozen references:
- L0: original MAPPO baseline checkpoint.
- L1: urgency/tag/validity observation features, original reward.
- L2: same features plus prior combined-urgency capped bonus, lambda_u=0.0005, rho=0.001.

Train only a new L3 on this branch:
- Observation: identical schema and implementation to L1/L2. Do not add features.
- Base reward: unchanged.
- Age term: based only on pre-service age urgency, age_u = clip(age/48, 0, 1).
- Direct tag term: based on actual successfully served bits with synthetic_priority_tag=1, independent of bucket age.
- Keep FIFO service order, environment dynamics, SafetyProjector, action space, task distribution and MAPPO architecture unchanged.

Use this predeclared reward:
- payload_term = served_bits / 1e6
- age_term = age_service_bits / 1e6
- tag_term = served_tagged_bits / 1e6
- bonus_age = min(0.0005 * age_term, 0.00025 * payload_term)
- bonus_tag = min(0.0010 * tag_term, 0.00075 * payload_term)
- r_L3 = r_base + bonus_age + bonus_tag
- r_base = payload_term - 0.05*c_project - 0.01*c_residual

Thus the combined bonus is bounded by 0.1% of the current successful payload term; 25% of that maximum is allocated to age and 75% to direct tagged service. If served_bits=0, both bonuses must be zero. Generated, admitted, dropped or queued bits do not receive service bonus. Do not change these coefficients after seeing evaluation results.

The tag is synthetic and only validates algorithmic behavior under the registered synthetic-tag protocol. Do not claim real-world maritime semantic priority improvement.

## 3. Required implementation tests

Implement only under the current branch's isolated phase directory. Reuse the E1 priority-tag ledger and E8 runtime context helper. Verify:
1. Per-bucket ordinary/tagged service and age decomposition is exact within 1e-6 bit.
2. bonus_age >= 0, bonus_tag >= 0, bonus_age <= 0.00025 * payload_term, bonus_tag <= 0.00075 * payload_term, and total bonus <= 0.001 * payload_term.
3. served_bits=0 implies both bonuses are zero.
4. The L3 reward is computed from pre-service bucket ages and actual per-bucket FIFO served bits; no future-state leakage.
5. The bonus-disabled path exactly reproduces L1's base reward on the same frozen transitions.
6. Ledger accounting, capture/restore, tag determinism, FIFO order and environment transition invariance pass.
7. No formal files/checkpoints are modified; default path remains L0-compatible.
8. Runtime context is explicitly set/recorded for evaluation. Do not rely on leaked training-process state.

Run at most 2 registered TRAIN_POOL scenarios × 16 steps for smoke first. If tests fail, fix only isolated code and rerun relevant tests. Do not train until all tests pass.

## 4. Controlled training protocol

After smoke passes, train L3 only:
- training seeds: 0, 1, 2;
- same existing L1/L2 MAPPO protocol and 300-update budget;
- same optimizer, rollout length, architecture, task distribution and synthetic-tag generation protocol;
- no additional hyperparameter search or budget expansion;
- save independent checkpoints, full configs, source commit, schema, runtime manifest and SHA-256 under the current experiment branch.

Reuse already trained L0/L1/L2 checkpoints and do not retrain them. If training produces non-finite losses/gradients or safety/ledger accounting errors, stop and report; do not silently alter hyperparameters.

## 5. Paired evaluation

Evaluate L0/L1/L2/L3 on the same 32 registered DEV_POOL scenarios (321096–321127) with the same paired environment/reset/assignment seeds and deterministic action protocol. Apply the validated evaluation runtime context explicitly. Never access reserved/test splits.

Report per seed and scenario:
- total payload bits (primary objective);
- actual tagged served bits (direct target);
- ordinary served bits;
- age-only weighted service and direct tag-service bonus;
- tagged/ordinary dropped bits and final queue;
- queue age, SINR, interference, projection rate;
- safety violations, constraint failures;
- reward, training stability, runtime;
- paired deltas and 95% confidence intervals versus L0 and L2.

Recompute summary metrics directly from scenario CSV. Distinguish actual tagged service from age-weighted service. Do not use training reward alone as evidence.

## 6. Predeclared merge criteria

Merge this branch into feature/maritime-wt-stage1a only if all conditions pass:
1. L3 actual tagged-served-bit delta versus L0 has a paired 95% CI lower bound > 0 in at least 2 of 3 model seeds.
2. L3 mean total payload delta versus L0 is >= 0 in at least 2 of 3 seeds, and the 95% CI lower bound is no worse than -0.5% of that seed's L0 mean payload. Report every scenario regression; do not hide them.
3. No increase in safety violations or constraint failures versus L0 in aggregate, and no material SINR/interference degradation.
4. The direct tagged-service gain is not driven by one or two scenarios; report per-scenario distribution and leave-one-scenario sensitivity.
5. All tests pass; L0/formal checkpoint and formal code hashes remain unchanged; diff contains only intended isolated experiment files.

If L3 fails any criterion or confidence intervals remain inconclusive, do not merge. Keep the branch and report the failure. Do not relax criteria after results. If L3 passes, create a PR and merge only this isolated L3 implementation/config/report, not training artifacts unless repository policy explicitly keeps them versioned. Preserve L0 and all historical results.

## 7. Deliverables

- docs/research/HTW_UrgencyAware_Direct_Tag_Reward_Experiment_Report_CN.md
- isolated L3 reward builder and tests;
- training/evaluation configs and runtime manifest;
- results/phase6i_priority_tag_direct_service_reward/scenario_rows.csv
- results/phase6i_priority_tag_direct_service_reward/summary.json
- results/phase6i_priority_tag_direct_service_reward/schema.json
- checkpoint manifest with SHA-256 and seed metadata.

## 8. Stop rules

- Do not modify formal Planner, formal reward, default observation, SafetyProjector, action space, L0 checkpoint or historical E0–E9 artifacts.
- Do not merge before complete paired evaluation.
- If results are worse or inconclusive, do not merge; summarize why and recommend the next step.
- Finish this branch with a clear Gate and stop.