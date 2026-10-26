# Stage D2 执行指令：高层 payload–urgency 权衡与 lambda_H 选择规则

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 依据报告：docs/research/HTW_UrgencyAware_HighLevel_T30_Pilot_Report_CN.md
- 前置 Gate：T30_URGENCY_SIGNAL_SUPPORTED
- 本阶段 Gate：HIGHLEVEL_UTILITY_RULE_READY / HIGHLEVEL_UTILITY_RULE_INCONCLUSIVE / HIGHLEVEL_UTILITY_RULE_BLOCKED
- 本阶段只做决策规则和已有训练样本的敏感性分析，不生成完整高层数据集、不训练 HGNN、不修改正式 Planner/低层/奖励/checkpoint，不读取 validation 或 reserved test 来选择权重。

## 1. D1 结论及边界

D1 在训练场景 322004–322015 的 t=30、每状态 128 个候选上完成 1,536 个 rollout。状态、候选、payload 标签和年龄账本检查通过。探索性信号达到预注册条件：
- 平均 Spearman = 0.9447；
- 平均候选对逆序比例 = 0.0673；
- 10/12 状态的逆序比例大于 0.05。

但这不证明完整 HGNN 会提高性能。紧急服务第一名的平均 top-1 payload regret 为 -133,418 bit，几乎全部由场景 322015 的 -1,601,020 bit 导致；其余 11 个状态的差在 1e-8 bit 以内。紧急服务第一名与 payload 第一名的候选 ID 不同，并不总意味着 payload 有实质损失，许多情况只是候选并列或 payload 相同。

因此本阶段不把 D1 结果用来直接设定 lambda_H，也不开始完整高层数据集或 HGNN 训练。先把“如何定义可接受的 payload 损失”与“如何从 Pareto 候选中选择权衡”分开。

## 2. 核心原则：不要先猜一个 lambda_H

当前 utility 候选为：
- payload_term = payload_bits / 1e6
- urgent_service_term = sum_t,k(urgency_k_before_service[t] * served_bits[k,t]) / 1e6
- utility(lambda_H) = payload_term + lambda_H * urgent_service_term

两个项的尺度随决策时刻改变，且目前有证据的排序信号主要来自 t=30。因此不能直接把 D0 的 t=0/15/30 平均尺度或 D1 的 t=30 比值当成全局 lambda_H。

本阶段必须先生成 Pareto/敏感性分析，再提出可解释的选择规则。不得用 validation/reserved 的结果选择 lambda_H。

## 3. 数据范围与冻结边界

- 只读取已有 D0/D1 pilot 的训练场景结果：D0 322000–322003 的 t=0/15/30；D1 322004–322015 的 t=30。
- 不重新运行 rollout，不增加候选，不重新抽样，不读 validation 322048–322063 或 reserved test 322064–322079。
- 不修改 D0/D1 结果、旧 J15 标签、原始 dataset shard 或任何 checkpoint。
- 如果需要复核字段，只能读取已提交的 labels.csv、selected_candidates.csv、summary/schema 和报告。
- 新分析输出到 results/phase6i_urgency_utility_rule/。

## 4. 必须完成的分析

### A. 逐状态 Pareto 前沿

对每个状态已抽样的 128 个候选，使用现有 payload_term 与 urgent_service_term：
1. 找出 Pareto 非支配候选；候选 a 支配 b 的条件为 payload_a >= payload_b 且 urgent_a >= urgent_b，并至少一项严格大于 1 bit 对应的归一化容差。
2. 报告 Pareto 候选数、payload 最优点、urgent-service 最优点、二者的 payload regret/urgent regret。
3. 对 payload 相同（1 bit 容差内）的候选分组，优先识别“没有实际 payload 损失但可提高 urgent service”的候选；不要把候选 ID 不同等同于效用改善。
4. 报告每个状态的非支配前沿和 top-1 取舍，不把候选子集结果冒充完整 13,327 候选池的结论。

### B. lambda_H 临界点与选择敏感性

对每个状态，计算候选 utility 交叉点：
`lambda_cross = (payload_i - payload_j) / (urgent_j - urgent_i)`
仅保留非负且有限的交叉点，并在计算中保持 payload_term 和 urgent_service_term 使用同一归一化单位。

对 lambda_H=0 以及由训练数据交叉点形成的有序区间，报告：
- 被选中的 top-1 candidate ID；
- 相对 payload-only top-1 的 payload regret；
- urgent-service 增益；
- top-1 assignment 改变比例；
- 跨状态的选择稳定性。

不得预先指定一个任意固定 lambda_H 并把它包装成最优值。若临界点过密、候选并列使得选择不稳定，报告该事实。

### C. 先定义业务/研究约束，再考虑标量权重

生成一张 trade-off 表，至少包含以下可解释选项：
1. Payload-first：只选 payload 最大候选，紧急度只作为并列时的 tie-break。
2. Payload-constrained urgency：先设最大可接受 payload 损失预算 `delta_payload_max`，仅在 payload regret 不超过该预算的候选中最大化 urgent-service。
3. Weighted-sum：utility = payload + lambda_H * urgency；只在前两项分析后，展示各临界区间的结果。

不能替用户决定系统可以牺牲多少 payload。报告必须将 `delta_payload_max` 标记为“需要研究负责人明确的设计约束”，不得从同一批结果中反向选择一个最有利阈值。如果没有预先批准的损失预算，则不能宣布 lambda_H 已选定。

### D. 结论稳健性

- D1 的 322015 是唯一有明显 top-1 payload regret 的状态，必须做 leave-one-state-out 描述性敏感性分析，报告去掉该状态前后的平均 regret 与 Pareto 结论；这只用于揭示单场景支配，不可据此删除该场景或修改正式 Gate。
- 分开报告 t=0、t=15、t=30，不用单一平均值掩盖时刻差异。
- 将 payload 相差不超过 1 bit 的候选视为 payload 并列，报告并列比例。
- 报告所有字段缺失、候选覆盖范围和采样局限。

## 5. Gate 判据

HIGHLEVEL_UTILITY_RULE_READY：
- 逐状态 Pareto 前沿和 lambda 临界点可复现；
- payload 并列与实际 payload regret 被正确区分；
- 形成 Payload-first、Payload-constrained urgency、Weighted-sum 三类规则的比较；
- 明确指出最终 `delta_payload_max` 仍需负责人确认，若未确认则 lambda_H 保持未定。

HIGHLEVEL_UTILITY_RULE_INCONCLUSIVE：
- 采样候选不足以形成稳定的前沿/敏感性结论，或候选并列导致权重选择高度不稳定。

HIGHLEVEL_UTILITY_RULE_BLOCKED：
- pilot CSV 与报告不一致、数据来源/哈希无法确认、单位不一致或必须读取 validation/reserved 才能继续。

## 6. 交付物

- docs/research/HTW_UrgencyAware_HighLevel_Utility_Rule_Report_CN.md
- results/phase6i_urgency_utility_rule/pareto_rows.csv
- results/phase6i_urgency_utility_rule/lambda_sensitivity.csv
- results/phase6i_urgency_utility_rule/summary.json
- results/phase6i_urgency_utility_rule/schema.json
- 如有助于解释，生成 Pareto 散点图和 lambda 临界点图。

报告必须说明本阶段没有训练、没有产生正式标签、没有选择最终 lambda_H。

## 7. 完成后停止

- 若 Pareto 结果显示大部分收益可在 payload 1-bit 并列候选中获得，建议先采用 payload-first + urgency tie-break 作为后续待验证假设，而不是立即训练 weighted-sum 模型。
- 若需要牺牲实质 payload 才能提高 urgent-service，则停止，等待研究负责人明确可接受的 payload 损失预算。
- 若权衡在不同决策时刻不稳定，不建立一个全局 lambda_H；后续应先考虑分时刻/分任务的明确目标设计，但不能未经确认实现。
- 本阶段结束后不训练 HGNN、不构建完整数据集、不修改正式系统。提交报告并等待下一步确认。
