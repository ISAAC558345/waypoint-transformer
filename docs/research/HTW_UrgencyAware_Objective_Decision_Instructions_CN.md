# Stage D8 执行指令：高层紧急度目标的损失预算决策门

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 依据报告：docs/research/HTW_UrgencyAware_HighLevel_Label_Rule_Design_Report_CN.md
- 前置 Gate：HIGHLEVEL_LABEL_RULE_DESIGN_READY
- 本阶段 Gate：HIGHLEVEL_URGENCY_OBJECTIVE_DECISION_READY / HIGHLEVEL_URGENCY_OBJECTIVE_WAITING_FOR_OWNER / HIGHLEVEL_URGENCY_OBJECTIVE_BLOCKED
- 本阶段不训练 HGNN/MAPPO、不运行 rollout、不修改正式 Planner、奖励、动作、checkpoint 或旧数据集。

## 1. D7 的结论

D7 已完成只读分析，`rollouts_run=0`，没有训练或修改正式系统。核心结果：
- D6 的 t=15 真实到达标签中，8 个状态有 7 个状态的 urgent-first 不在 1-bit payload 并列组内；其平均 payload regret 为 1,352,814 bit。
- D6 的 t=30 真实到达标签中，8/8 状态的 urgent-first 不在 1-bit payload 并列组内；平均 payload regret 为 2,320,860 bit。
- Payload-first + urgency tie-break 在 D6 的 t=15、t=30 对原 payload-ID tie-break 的平均紧急服务增益均为 0 bit，因为这些时刻 payload 最优候选基本是唯一的。
- t=0 存在大量 payload 并列，tie-break 可在抽样标签中提高 urgent-service 而不牺牲 payload；t=45 是 3-step horizon，必须独立看待。
- `lambda_H` 和 `delta_payload_max` 均未确定。D7 的比例网格只是敏感性分析，不是研究负责人已批准的损失预算。
- 每状态仅抽样 128/13,327 个合法候选，结果不能视为全池最优或正式闭环收益。

## 2. 当前需要作出的研究决策

高层目标的核心选择不是技术参数调优，而是明确业务/研究优先级：是否允许为了更早服务紧急数据而牺牲部分总 payload。

Cursor 不得替研究负责人决定这个预算。当前必须保持两个候选设计分离：

### 选项 A：严格 payload-first
- 目标是最大化 payload。
- 只在 payload 1-bit 容差内的并列候选中，用 urgent-service 打破平局。
- 不引入会选择低 payload 候选的紧急度奖励。
- 适用于总传输量优先、不能接受实质吞吐量下降的目标。
- 根据 D7，t=15/t=30 在现有抽样数据上的额外紧急服务收益为 0；t=0 有并列候选收益，但不能据此推断完整闭环收益。

### 选项 B：有约束的紧急度优先
- 先定义每个决策状态允许的最大 payload 损失 `delta_payload_max`，再在允许集合内最大化 urgent-service。
- 该预算必须以绝对 bit 数或明确百分比表达，写入配置和实验协议。
- 不允许从 D7 敏感性曲线中挑选一个看起来最好的点并当作预先批准的设计目标。
- 只有负责人明确预算后，才可制定双输出 HGNN（预测 payload 与 urgent-service）和独立训练/闭环验证计划。

## 3. 本阶段工作

1. 复核 D7 报告及 `summary.json`，确保上述结论与已提交文件一致。
2. 新建决策记录 `docs/research/HTW_UrgencyAware_Objective_Decision_CN.md`，完整说明选项 A/B、适用条件、D7 证据、局限及待决参数。
3. 在决策记录中将 `delta_payload_max` 标记为 `OWNER_DECISION_REQUIRED`，将 `lambda_H` 标记为 `NOT_SELECTED`。
4. 给出明确停止条件：在负责人选择前，不构建完整 urgency-aware 高层训练集，不训练双输出 HGNN，不改正式 Planner。
5. 不要启动额外 rollout，不要读取 validation/reserved test，不要更改 D0–D7 结果。
6. 将决策记录提交到当前分支。不要在本阶段自动选择 A 或 B，也不要自动进入 D9。

## 4. 交付物

- `docs/research/HTW_UrgencyAware_Objective_Decision_CN.md`

记录应包含源报告 commit/hash、关键数值、两个可选目标、对低层冻结策略的约束、负责人需要决定的单一问题，以及后续每个选项对应的执行路线。

## 5. Gate

- `HIGHLEVEL_URGENCY_OBJECTIVE_DECISION_READY`：决策记录完整、数值与 D7 源文件一致，且明确保留负责人决策。
- `HIGHLEVEL_URGENCY_OBJECTIVE_WAITING_FOR_OWNER`：决策材料已准备好，但 `delta_payload_max` 尚未获批准；这是正常的等待状态。
- `HIGHLEVEL_URGENCY_OBJECTIVE_BLOCKED`：D7 源报告与 summary 不一致或来源哈希无法验证。

完成后停止并报告，不启动训练或新的实验。
