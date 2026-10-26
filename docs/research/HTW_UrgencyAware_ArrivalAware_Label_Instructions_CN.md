# Stage D5 执行指令：修正候选标签与真实环境转移的语义差异

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_Tiebreak_Single_Intervention_Report_CN.md
- 前置 Gate：URGENCY_TIEBREAK_SINGLE_INTERVENTION_INCONCLUSIVE
- 本阶段 Gate：ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE / ARRIVAL_AWARE_URGENCY_LABEL_INCONCLUSIVE / ARRIVAL_AWARE_URGENCY_LABEL_BLOCKED
- 本阶段只做训练池内的标签语义诊断，不训练 HGNN/MAPPO，不修改正式 Planner、奖励、动作、checkpoint 或既有数据集。

## 1. D4 的决定性发现

D4 报告显示，单点 tie-break 在真实闭环中的方向不一致，Gate 为 `URGENCY_TIEBREAK_SINGLE_INTERVENTION_INCONCLUSIVE`。其中：
- t=0 干预的前 15 步真实 payload 差回到 1 bit 以内，但后续 Scheme C 重新匹配后，完整 episode 差异达到 Mbit 量级。
- t=30 干预的候选标签 rollout 将 `arrivals/arrivals_at` 置零，标签 payload 差仍在 1 bit 内；但保留真实到达的 15 步配对诊断，在 4 个场景中分别出现约 -3.90、-3.57、-1.67、+1.09 Mbit 的 payload 差。
- t=30 的真实紧急服务差也有正有负，与零到达标签中的正增益不一致。
- 这表明现有零到达候选标签不能保证预测真实到达条件下的 payload/urgency 差。现有证据不能把闭环变化归因于某一个单独机制。

因此，停止继续扩展 tie-break 规则。下一步先检查并量化标签生成约定和真实环境转移之间的差异，不训练模型。

## 2. 本阶段唯一问题

在相同的当前状态、相同候选 assignment、相同冻结低层和相同外生到达序列下，比较两种候选标签：
- Label-ZeroArrival：沿用现有 J15 约定，在 scratch 副本中将 `arrivals/arrivals_at` 置零；
- Label-RealArrival：保持环境真实的 `arrivals_at(t)`，按实际服务、到达、丢弃与队列上限规则执行。

两种标签都计算：
- `payload_bits`；
- `urgent_service_bits = sum_t,k(urgency_before_service[k,t] * served_bits[k,t])`；
- 每浮标 served bits、终态队列、dropped bits；
- ledger 与环境计账误差。

目标是判断零到达标签是否会改变候选排序，尤其是否会把紧急服务增益方向判断错。此阶段不选择新的高层策略，也不把真实到达标签自动定义为最终训练目标。

## 3. 场景、候选与预算

- 仅使用 `highlevel_v2_train` registry 中的四个训练场景 322000–322003，并在运行前调用 `assert_allowed(split="train")`。
- 只检查 t=30，使用相同的 128 候选子集生成规则：32 个稳定 J15 top + 96 个按既有 `utility_rank` 分层抽样；种子固定为 604，`SeedSequence([604, scenario_id, 30])`。
- 候选列表必须在任何新 rollout 之前写入磁盘；两个标签臂使用完全相同的候选 ID 和 assignment。
- 每个状态最多 128 个候选、每候选两个 15-step rollout，总计最多 1,024 次 rollout。
- 先执行一个状态并估算时间；若单状态超过 900 秒或总预计时间超过 20 分钟，停止并报告，不扩充预算。
- 不读取 validation 322048–322063 或 reserved test 322064–322079。
- 冻结 MAPPO-V2 seed 0、环境、SafetyProjector、Stage A ledger 版本和所有 checkpoint/config。输出写入 `results/phase6i_urgency_arrival_aware_labels/`。

## 4. 状态与到达配对

对每个场景：
1. 按既有高层数据生成路径和冻结低层重放到 t=30，记录完整 live state、q_bits、ledger capture、state hash、候选池 hash 和所有模型/config SHA-256。
2. 生成候选子集并保存后，从同一 t=30 状态和 ledger 快照分叉。
3. 对每个候选分别运行 Label-ZeroArrival 与 Label-RealArrival。
4. RealArrival 必须使用环境正式的确定性 `arrivals_at(t)` 语义。若到达由时钟函数决定，两个标签臂必须使用相同的到达序列；不得因候选顺序或随机采样顺序改变到达。
5. ZeroArrival 只在 scratch 副本中禁用到达；不得修改源状态或正式环境。
6. 两臂使用相同低层 checkpoint、动作确定性、TaskOption hold 和 15-step horizon。服务、到达、丢弃的更新顺序必须遵循环境真实转移。
7. 每步使用服务前 ledger view 计算 urgency，使用实际逐浮标 served bits 更新账本；不允许用未来服务量计算当前 urgency。
8. 若任一候选的零到达 payload 标签不能复现原有 `j15_bits`（在环境计账容差内），标记 BLOCKED，不要把两套标签混为一套。

## 5. 必须分析的结果

### A. 标签一致性
- 每候选的 ZeroArrival payload 与既有 `j15_bits` 的误差；
- ZeroArrival 与 RealArrival 的 payload 差；
- 两臂 urgent-service 差、队列终态差和 dropped bits 差；
- ledger/q_bits 与 payload/served 守恒误差。

### B. 排序稳定性
分别对每个场景 128 个候选计算：
- payload 标签的 Spearman、Kendall 或 pairwise inversion；
- urgent-service 标签的 Spearman、Kendall 或 pairwise inversion；
- 两标签臂的 payload top-1 和 urgency top-1 是否改变；
- 候选 pair 中，ZeroArrival 判断 urgent-service A>B、RealArrival 判断 A<B 的比例；
- 按 payload 分位数、idle UAV 数量和 urgency 起始分布分层报告，不把这些分层误认为因果。

### C. 重点检查
- D4 中出现反向紧急服务增益的场景，检查真实到达后是否出现队列竞争、服务饱和或高紧急度队列变化。
- 区分标签差异与完整闭环差异；本阶段只有固定 15 步候选标签，不执行完整 48 步策略对照。
- 不使用 RealArrival 的未来 rollout 结果构造当前 observation；RealArrival 结果只作为候选监督标签/离线分析数据。
- 不从结果中选取“更好看的”候选，不调整候选池、随机种子或指标阈值。

## 6. 交付物

- `docs/research/HTW_UrgencyAware_ArrivalAware_Label_Report_CN.md`
- `results/phase6i_urgency_arrival_aware_labels/selected_candidates.csv`
- `results/phase6i_urgency_arrival_aware_labels/labels.csv`
- `results/phase6i_urgency_arrival_aware_labels/summary.json`
- `results/phase6i_urgency_arrival_aware_labels/schema.json`

报告记录场景 ID、候选/rollout 数、运行时间、状态与候选 hash、checkpoint/config SHA-256、两种标签的差异、排序反转比例、payload top-1 regret、urgent top-1 变化和 Gate。

## 7. Gate

ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE：
- 4 个训练场景均完成 128 候选的两种标签计算；
- 原 J15 payload 标签复现、候选合法性、ledger 与环境计账检查通过；
- 两种标签的排序稳定性与方向反转统计完整报告。

ARRIVAL_AWARE_URGENCY_LABEL_INCONCLUSIVE：
- 到达对标签排序影响较小或跨场景方向不稳定，不能据此决定后续标签方案。

ARRIVAL_AWARE_URGENCY_LABEL_BLOCKED：
- split、状态/候选 hash、payload 标签复现、外生到达配对、ledger/accounting 或 checkpoint/config 检查失败，或耗时超过预算。

## 8. 完成后停止

本阶段不训练、不修改正式系统、不构建完整高层紧急度数据集。
- 若真实到达标签显著改变 urgent 排序且可复现，下一步只提出“arrival-aware label”小规模扩展方案，再决定是否构建训练集。
- 若排序仍不稳定，停止高层 urgency 标签路线，保留 payload-first 正式高层，重新考虑紧急度是否应只作为报告指标或并列规则。
- 无论结果如何，都不得据此宣称完整 HGNN 或正式系统已经提升。
