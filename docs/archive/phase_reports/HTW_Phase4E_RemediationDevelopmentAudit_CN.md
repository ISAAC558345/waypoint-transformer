# HTW Phase-4E Remediation Development-Set Audit

本阶段只做数据谱系和 development-set 审计。没有修改 B2、τ、Gate-1、Candidate-C、planner 或 `decide()`，没有重跑 deployment verification，也没有把 230000–230127 重新放进训练或阈值选择。

**status = DEPLOYMENT_VERIFICATION_FAILED**

**deployment_verified = False**

**planner_ready = False**

**remediation_status = B2_REMEDIATION_BLOCKED_BY_NO_ELIGIBLE_DEVELOPMENT_DATA**

**eligible development data = NO_ELIGIBLE_DEVELOPMENT_DATA**

## 结论

当前 B2 的 fit、特征对照、GroupKFold 和 τ 都落在 `LLV2_design` 221000–221015。220000–220063 已经是 Phase-4C/4D validation。222000–223063 和 104000–106063 是历史测试池。230000–230127 已永久标为 `DEPLOYMENT_HOLDOUT_CONSUMED`。`LLV2_confirmation` 和 `LLV2_final_untouched` 在 registry 里仍是 null。

因此不存在可以合法用于下一轮 B2 development 的 pool，也不能把上述 validation / test / holdout 改称为 development pool。Development pool、validation pool、consumed holdout 三者不能同时分开，因为前两者没有一个未被占用的新池。

已完成的 Phase-4E 归因仍支持 candidate ranking error。design 集上 margin 与真实 ΔR 的 Spearman 只有 0.185，说明 margin 即使在拟合数据上也不是紧密的 trajectory-aware 指标。这不足以建立应该修改 B2、τ 或 Candidate-C。

## 1–6. 每个池的实际用途

用途来自 registry、manifest 和已有审计，不是只按 ID 区间猜测。矩阵共 528 行，`eligible_for_future_development` 全为 0。

| pool | IDs | historical_role | B2 fit | feature ablation | hyperparameter search | tau / Gate-1 | criterion | Phase-4C | Phase-4E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLV2_train | 220000–220063 | VALIDATION | 否 | 否 | 否 | 否 | 否 | 是 | 否 |
| LLV2_design | 221000–221015 | DESIGN | 是 | 是 | 否 | 是 | 否 | 否 | 否 |
| phase2_val | 222000–222063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase3_val | 223000–223063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase104 | 104000–104063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase105 | 105000–105063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase106 | 106000–106063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| deployment holdout | 230000–230127 | DEPLOYMENT_HOLDOUT_CONSUMED | 否 | 否 | 否 | 否 | 否 | 否 | 是 |

1. 当前 B2 fit：221000–221015 的 EXEC 标签，每场 24 个 candidate。目标是 `C_abs = U_traj - U_single`。特征是锁定的 `U_single + Phase-3E candidate geometry`。ExtraTrees 为 `n_estimators=300`、`max_depth=8`、`min_samples_leaf=5`、`random_state=0`、`n_jobs=1`。没有单独保存的模型文件，冻结模型按这 16 个 state 重建。
2. feature selection：同一 16 个 state。Phase-3E 和 Phase-4B 在这些 state 上做了特征消融和 GroupKFold。最终 B2 使用 U+candidate geometry。
3. hyperparameter selection：没有 scenario 参与搜索。Phase-4B 写明无独立 validation fold、不做超参搜索。ET 配置是预先固定的。
4. tau / Gate-1：τ = 1534933.5944128654，来自这 16 个 design state 里 changed-state 的 mean `dU_hat`。Gate-1 的数值阈值因此也来自 design。220000–220063 只审计过这个已锁定的 gate，没有用来选择 τ。
5. deployment holdout 已消耗：230000–230127，Batch-1 与 Batch-2 共 128 个。
6. 这 128 个已经隔离。登记在 `deployment_holdout_consumed_registry.json`，关闭训练、重训、特征/超参/τ/Gate-1/criterion 选择、candidate design、model comparison、threshold tuning 和 planner development。seed 和原始标签都没有改。

Candidate-C 的 C1–C8 来自人工规格，没有任何 scenario pool 被用来选择这些数值。

另外扫到 1353 个已登记的历史 seed 不在上表中。它们是 blocked historical seeds，不是已注册的 L3-B B2 development pool。`LLV2_confirmation` 与 `LLV2_final_untouched` 不存在。

## 7–9. 合法 development data

不存在。

不能当下一轮 development pool 的原因：

- 221000–221015 已经用于 fit、特征消融、GroupKFold 和 τ。再把它当成未使用的 development/validation split 会产生 contamination。
- 220000–220063 已经是 Phase-4C/4D validation。
- 222000–223063、104000–106063 是历史测试，协议拒绝，且没有 B2 first-hold 标签。
- 230000–230127 是已消耗的 deployment holdout。
- confirmation / final untouched 没有被创建。

所以没有一组互相分离的 Development pool 和 Validation pool 可以配上已经消耗的 holdout。状态是 `B2_REMEDIATION_BLOCKED_BY_NO_ELIGIBLE_DEVELOPMENT_DATA`。

## 10–12. 失败方向

已有 evidence：

- A. candidate ranking error。Phase-4E 归因里，230070、230077、230078、230091 的 B2 top-1 都不是 oracle；230078 的 margin 大于 2τ 仍有 ΔR = -0.182。
- D 的诊断，不是新阈值。在 design 的已有 shadow 表上，`corr(margin, ΔR)` Pearson = 0.253，Spearman = 0.185。margin ≥ τ 的 5 场中有 1 场 ΔR < 0，比例 0.20。在已消耗的 Phase-4C validation 表上，Pearson = 0.550，Spearman = 0.530；26 场中有 3 场 ΔR < 0，比例 0.115。这些数字没有用来拟合新 τ。

Phase-4E 只作为外部记录：50 个 margin ≥ τ 的 scenario 里有 4 个已实现 ΔR < 0，比例 0.08。该结果表在 margin < τ 时把 ΔR 记为 0，因为 Gate-1 选择了 LOCAL，所以整表相关系数不能解释成 margin 对真实 B2 ΔR 的预测力，也不进入选择。

仍然 **NOT ESTABLISHED**：

- B. U_hat calibration error
- C. first-hold geometry 与长期 U_traj mismatch 是否就是该改的原因
- E. training distribution 覆盖不足是否解释了这次失败。fit 只有 16 个 state 是事实，Batch-2 的环境几何并没有显示出足以单独解释失败的整体偏移。
- F. candidate-conditioned feature 不足
- 是否应该修改 B2、τ 或 Candidate-C

最值得在下一轮调查的方向是 A，并同时检查 margin 与真实 ΔR 的弱对应。本阶段不设计新模型、新特征、新 τ、新 Gate-1 或新 criterion。

## 十二个问题

1. B2 fit：221000–221015。
2. feature selection：221000–221015 上的消融；没有其他池。
3. hyperparameter selection：没有。ET 是预先固定的。
4. tau / Gate-1：τ 来自 221000–221015 的 shadow mean `dU_hat`。
5. 已消耗的 deployment holdout：230000–230127。
6. 这 128 个已经彻底隔离：是。
7. 是否存在合法 future development pool：否。
8. 具体 pool：没有。
9. 阻塞原因：所有已注册的 L3-B 池要么是 design/fit，要么是 validation，要么是历史测试，要么是已消耗 holdout；confirmation 池不存在。
10. 当前失败最有证据的类型：candidate ranking error。margin 与 ΔR 在 design 上只有弱相关。
11. 已有 evidence：A，以及 design / Phase-4C 上的 margin–ΔR 诊断。
12. 仍是 NOT ESTABLISHED：B、C、E、F，以及是否修改 B2、τ、Candidate-C。

本阶段只完成 data lineage 和 development-data audit。不进入重新训练。
