# Stage D7 执行指令：基于 D6 结果构建高层双目标标签决策规则

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_ArrivalAware_Expansion_Report_CN.md
- 前置 Gate：ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE
- 本阶段 Gate：HIGHLEVEL_LABEL_RULE_DESIGN_READY / HIGHLEVEL_LABEL_RULE_DESIGN_INCONCLUSIVE / HIGHLEVEL_LABEL_RULE_DESIGN_BLOCKED
- 性质：仅对已生成 D0/D1/D5/D6 标签做离线决策规则分析；不运行 rollout、不训练 HGNN/MAPPO、不修改正式 Planner、奖励、动作、checkpoint 或旧数据集。

## 1. D6 的决定性结果

D6 在训练场景 322016–322023、t=0/15/30/45 完成 8,192 次研究 rollout 与 256 次复核。Gate 为 `ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE`，原因不是完整性失败，而是不同决策时刻的信号不同：
- t=0：8/8 场景的零到达与真实到达标签一致。该 15 步窗口内新到达未成为交付量，36 Mbit 到达主要进入队列或被丢弃。
- t=15：8/8 场景的真实到达标签都达到预注册的 material 标准；payload top-1 8/8 改变，urgent top-1 7/8 改变。
- t=30：8/8 场景都达到 material 标准；payload 和 urgent top-1 均 8/8 改变。
- t=45：剩余 3 步的正式 arrivals 为 0，两标签臂一致。
- t=15 的 payload 与 urgent-service 排序相关均值约 0.40；t=30 约 0.53。Pareto 前沿显示部分候选必须牺牲实际 payload 才能提高紧急服务。
- `lambda_H` 和 `delta_payload_max` 仍未确定。D6 是候选子集的离线标签，不是 learned policy 或闭环性能改进证据。

因此不应将四个决策时刻合并为一个全局目标，也不应立即训练 HGNN。下一步只用现有标签设计可审计、时刻分开的监督目标方案，并明确哪些决策需要研究负责人选择 payload 损失预算。

## 2. 本阶段唯一目标

利用现有 D0/D1/D5/D6 结果，为高层 HGNN 提出可执行但尚不训练的候选监督规则：
1. t=0 和 t=45：在当前已测标签协议下，urgent 目标没有显示 arrival-aware 排序增益；采用 payload-first 作为基线，不把紧急度奖励强行加入这些时刻。
2. t=15 和 t=30：使用真实到达标签，建立 payload–urgent-service Pareto 前沿，并比较两种不依赖任意 `lambda_H` 的候选规则：
   - Rule P：payload-first，urgent-service 仅用于 payload 在 1 bit 容差内并列时打破平局；
   - Rule C(`delta_payload_max`)：先保留真实 payload 相对该状态最高值的损失不超过符号预算 `delta_payload_max` 的候选，再最大化 urgent-service。
3. 输出 Rule C 随不同 `delta_payload_max` 的敏感性曲线/表，但不得自行选一个最终预算，不得把离线标签结果当作已验证闭环策略。
4. 评估训练标签覆盖率、Pareto 候选比例、top-1 assignment 改变比例和潜在 payload regret，决定是否值得开始构建正式 arrival-aware HGNN 数据集。

## 3. 数据范围与冻结边界

- 仅读取已有 D0、D1、D5、D6 的已提交结果文件和报告。
- 不运行任何新的环境 rollout，不新增候选，不改变 seed，不访问 validation 322048–322063 或 reserved test 322064–322079。
- 不修改旧 J15 shard、D0–D6 结果、正式 Planner、HGNN/MAPPO checkpoint、奖励、动作或环境。
- 新输出写入 `results/phase6i_urgency_label_rule_design/`。
- 对所有源文件记录路径、Git commit、SHA-256；如果某项所需字段在现有 CSV/summary 中不存在，报告字段缺口，不重算标签、不猜测缺失值。

## 4. 必须执行的分析

### A. 分时刻建立规则
- 按决策时刻分别分析：D0 的 t=0/15/30、D1 的 t=30、D5 的 t=30、D6 的 t=0/15/30/45。
- 15-step 和 3-step 标签必须分开；t=45 不与其他时刻合并。
- 明确 D0/D1/D5 与 D6 使用的候选状态、状态路径、候选子集和 arrival semantics 不同。不得直接将不同状态的 bit 值逐项相减。
- 对 t=15/t=30 以 D6 的 RealArrival 标签作为本次规则分析的主数据；ZeroArrival/J15 只作为 legacy 对照。

### B. Pareto 与 payload 约束
对每个 D6 状态的已抽样候选：
- 计算真实到达标签下的 Pareto 非支配候选；
- 标记 payload 与最大值相差不超过 1 bit 的候选；
- 计算 Rule P 选择和各个符号/网格化 `delta_payload_max` 下 Rule C 选择的候选 ID、urgent-service gain、payload regret 和 assignment 改变率。
- 可展示一组清晰的敏感性网格，例如从 0 到每状态观测 payload regret 范围的若干比例点；所有点只用于显示权衡，不得据此选最终预算。
- 规则必须在每个状态独立应用，不允许跨状态比较绝对 payload 值后排序。

### C. 目标构造建议
比较以下监督形式的利弊：
1. 仅 payload 排序，urgent 只作 1-bit tie-break；
2. 两个 head 分别预测真实 payload 和 urgent-service，再在推理时执行约束规则；
3. 单一标量 utility `payload + lambda_H * urgent`。

重点说明：形式 2 可将“预测两种量”与“研究负责人决定可接受损失”分离，避免现在就猜 `lambda_H`。但多头预测仍需独立验证两个标签的预测误差，不能仅凭 oracle 标签 Pareto 前沿宣称 HGNN 能学会它。

### D. 数据质量和覆盖
- 报告每个时刻的状态数、候选数、horizon、arrival 语义、Pareto 数、payload 并列比例、Rule P/Rule C 的候选改变比例。
- 明确 D6 每状态只采样 128/13,327 候选；不得称其为全候选池最优或完整 Pareto 前沿。
- 场景数 8，只作开发期标签设计分析，不宣称统计显著或泛化。
- 不把候选 pair 当作独立 episode 样本。

## 5. Gate

`HIGHLEVEL_LABEL_RULE_DESIGN_READY`：
- t=0/15/30/45 的标签语义分别说明；
- t=15/t=30 的 Rule P 与 Rule C 敏感性完整；
- `delta_payload_max` 明确保持为待负责人决定的设计参数；
- 给出是否值得构建完整 arrival-aware HGNN 数据集的建议和所需额外验证。

`HIGHLEVEL_LABEL_RULE_DESIGN_INCONCLUSIVE`：
- 现有数据缺少必要字段，或候选子集/标签差异不足以支持可执行规则。

`HIGHLEVEL_LABEL_RULE_DESIGN_BLOCKED`：
- 源文件哈希、标签来源或数据分割无法核实，或分析需要读取 validation/reserved 数据。

## 6. 交付物

- `docs/research/HTW_UrgencyAware_HighLevel_Label_Rule_Design_Report_CN.md`
- `results/phase6i_urgency_label_rule_design/state_rule_rows.csv`
- `results/phase6i_urgency_label_rule_design/delta_sensitivity.csv`
- `results/phase6i_urgency_label_rule_design/summary.json`
- `results/phase6i_urgency_label_rule_design/schema.json`

报告需记录源文件哈希、按时刻的标签语义、候选子集覆盖、Pareto/敏感性结果、字段缺口、Gate 和下一步建议。

## 7. 完成后停止

本阶段不训练、不重跑 rollout、不修改正式系统。若 Rule P 在 t=15/t=30 仍能保留绝大部分 payload 并获得可重复的 urgent-service 改善，可建议下一阶段建立独立的双头 HGNN 训练数据/模型试验；若必须牺牲实质 payload，则等待研究负责人明确 `delta_payload_max`。不能擅自设置该值，也不能宣称正式系统已经改善。
