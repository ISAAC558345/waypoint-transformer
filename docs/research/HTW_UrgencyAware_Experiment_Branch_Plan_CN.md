# Branch Experiment Plan: Priority-Tag Urgency Reward

- Repository: ISAAC558345/waypoint-transformer
- Base branch: feature/maritime-wt-stage1a
- Experiment branch: experiment/priority-tag-urgency-reward
- Goal: add a small, bounded reward for actually serving urgent-tagged information while preserving total payload as the primary objective.
- Workflow policy: no separate owner approval gate. Implement and test on this branch; open a PR and merge only if predeclared comparative criteria are met. If results are worse or inconclusive, do not merge.

## 1. Evidence entering this experiment

E1 priority-tag ledger prototype passed its accounting and environment-invariance checks.
E2 reward-scale replication completed 8 frozen-policy episodes but was inconclusive for choosing a weight.
E3 offline reward-form comparison found no combination passing all conservative fixed-trajectory thresholds. Its result is not a trained-policy result.

Use the E3 conditional candidate only as the first experimental candidate:
- lambda_u = 0.0005
- rho = 0.001
- reward = r_base + min(lambda_u * urgency_served_bits/1e6, rho * served_bits/1e6)

This is a test candidate, not an asserted optimum. Do not expand the parameter grid in this branch.

Synthetic tags are not real maritime application labels. Every report and figure must clearly label them as synthetic.

## 2. Branch isolation and baseline protection

- All implementation, training artifacts, logs and checkpoints stay on this branch under a new phase directory.
- Never modify or overwrite the formal Planner, formal reward, formal observation defaults, SafetyProjector, action space, original checkpoint phase6g_seed0.pt, or E0–E4 results.
- Preserve the original L0 checkpoint and exact SHA-256.
- Keep the original default path unchanged. New behavior must require an explicit experiment config/feature flag.
- Do not force-push or rewrite branch history.
- Do not read reserved/test splits to tune reward parameters.

## 3. Implementation and smoke

Implement the independent priority-aware observation/reward builder using the existing E1 ledger:
- urgency per FIFO bucket: clip(clip(age/48,0,1) + 0.25 * synthetic_tag, 0, 1);
- validity mask must distinguish empty queues from valid zero-urgency buckets;
- service bonus is calculated only from actual bits served from buckets, using urgency before service;
- served_bits == 0 implies zero bonus;
- tags do not change FIFO order or physical queue transitions;
- no future arrivals or future served bits may enter observations.

Required tests:
1. ledger and formal q_bits conservation;
2. ordinary/tagged served, admitted, dropped and remaining-bit conservation;
3. deterministic tags and reproducible reset/capture/restore;
4. lambda=0 reward exactly matches the formal reward;
5. capped bonus always satisfies 0 <= bonus <= 0.001 * served_bits/1e6;
6. ledger on/off leaves formal environment state, actions and payload unchanged;
7. old checkpoint loads unchanged through the default path; experimental checkpoint records observation schema and reward config.

Run at most two registered training scenarios for 16 steps each for smoke. If smoke fails, fix only the isolated experiment code and rerun the failing test; do not start training until all smoke tests pass.

## 4. Controlled training

After smoke passes, train independent variants:
- L0: existing frozen baseline phase6g_seed0.pt (no retraining needed for reference);
- L1: new observation features (age urgency, synthetic priority tag, validity), original reward;
- L2: same new observation features plus bounded reward with lambda_u=0.0005 and rho=0.001.

Use the existing MAPPO training protocol, optimizer settings, rollout length and training budget for L1/L2. Use at least 3 fixed training seeds. Do not change architecture, action space, task distribution, SafetyProjector or any other reward term in this experiment. Save each seed checkpoint under the experiment branch with full config, source commit, schema version, synthetic-tag seed protocol and SHA-256.

If training fails or produces non-finite loss/gradients, stop and report; do not silently change hyperparameters or extend the budget.

## 5. Evaluation protocol

Evaluate L0, L1 and L2 on the same registered development scenarios and paired environment seeds, using deterministic evaluation actions. Use the repository registry and split guard. Training scenarios may be used for training only; use the designated DEV_POOL for selection/paired evaluation, but never use reserved/test splits. Do not tune lambda/rho on the evaluation results.

Report per-seed and per-scenario paired metrics:
- total delivered payload bits (primary metric);
- tagged information served bits and urgency-weighted served bits;
- tagged/ordinary dropped bits and final queue;
- average and maximum queue age;
- SINR/interference, path length, projection rate, safety violations and constraint failures;
- reward/return, training stability and runtime.

The comparison must distinguish L1's input-only effect from L2's reward effect. Do not claim real-world semantic urgency from synthetic tags.

## 6. Merge decision: predeclared criteria

Merge the experiment branch into feature/maritime-wt-stage1a only if all conditions hold:
1. L2 improves tagged/urgency-weighted service versus L0 with a paired 95% confidence interval above zero on the designated development evaluation.
2. Total payload is not worse than L0: paired mean payload difference is non-negative and the paired 95% confidence interval does not indicate a meaningful negative regression. Report raw differences and confidence intervals; do not hide any scenario regressions.
3. No increase in safety violations or constraint failures, and no material degradation in SINR/interference or dropped bits.
4. L2's gain is reproducible across at least 2 of 3 training seeds and is not driven by a single scenario.
5. All tests pass, the formal baseline hashes remain unchanged, and the PR diff contains only intended isolated experiment changes.

If L1 improves urgency service but L2 does not, do not merge L2; report L1 separately and open a separate PR only if it independently meets the same payload/safety criteria. If either model reduces payload materially, causes safety regressions, or results are statistically inconclusive, do not merge. Keep the experiment branch and report the result.

## 7. Deliverables

- docs/research/HTW_UrgencyAware_Experiment_Report_CN.md
- independent experiment implementation and tests under src/wt/benchmark_v2/phase6i_priority_tag_reward/ (or an equivalent isolated directory consistent with repository layout);
- training/evaluation configs and logs;
- results/phase6i_priority_tag_reward/summary.json;
- per-seed and per-scenario CSVs;
- checkpoint manifest with hashes;
- a PR from experiment/priority-tag-urgency-reward to feature/maritime-wt-stage1a only after evaluation is complete.

## 8. Stop rule

Do not merge automatically merely because training reward rises. Use the predeclared payload, urgency-service, safety and reproducibility criteria. If the result is worse or inconclusive, do not merge and report why. Keep the production/base branch unchanged.
