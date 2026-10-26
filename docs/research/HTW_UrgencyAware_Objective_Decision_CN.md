# 高层紧急度目标决策记录

- 仓库：`ISAAC558345/waypoint-transformer`
- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`3fcd38cd5072f3e2f989ce582a93d7f5fb7c2cc3`
- 依据报告：`docs/research/HTW_UrgencyAware_HighLevel_Label_Rule_Design_Report_CN.md`
- 依据结果：`results/phase6i_urgency_label_rule_design/summary.json`
- D7 结果提交：`cec2ffd1400782733a3e2f897f9a69caa7e21974`
- 前置 Gate：`HIGHLEVEL_LABEL_RULE_DESIGN_READY`
- 本阶段 Gate：`HIGHLEVEL_URGENCY_OBJECTIVE_WAITING_FOR_OWNER`
- `delta_payload_max`：`OWNER_DECISION_REQUIRED`
- `lambda_H`：`NOT_SELECTED`
- 性质：决策记录。`rollouts_run = 0`。没有训练 HGNN 或 MAPPO，没有修改正式 Planner、奖励、动作、checkpoint 或旧数据集。

## 1. 复核结论

D7 报告与 `summary.json` 一致，来源哈希可以复核。指令里的整数是对 summary 均值的普通四舍五入，逐位对照如下。

| 陈述 | 指令中的写法 | summary.json |
|---|---|---|
| D6 t=15，urgent-first 离开 1 bit payload 并列组 | 7/8 | `material_regret_states = 7`，状态数 8 |
| D6 t=15，payload regret 均值 | 1,352,814 bit | 1,352,814.324 bit |
| D6 t=30，urgent-first 离开并列组 | 8/8 | `material_regret_states = 8`，状态数 8 |
| D6 t=30，payload regret 均值 | 2,320,860 bit | 2,320,860.037 bit |
| t=15 与 t=30 的 Rule P 相对 payload-ID tie-break 的 urgent 增益 | 0 bit | 两个时刻的均值都是 0.0 bit，超过 1 bit 的状态都是 0/8 |
| `lambda_H`、`delta_payload_max` | 未确定 | 二者均为 `null`，`budget_selected = false` |
| 抽样 | 128/13,327 | `candidates_per_state = 128`，`d6_pool_count = 13327` |

文件哈希（SHA-256）：

- 报告 `2d8d5b6472fa63da8b9f52d9bec2d448157642f1edb0f95b8ee2a4a2879bf774`
- summary `b82743763d3b75181a3b382d10f924ce544cda5b1f29ef7f4d8fe599beb4eb37`
- schema `8ae50b576a1dbc8234f9a8bd6cd4ff22aa68ee6d6ff244fada45614e49f56c0f`

两份文件都在提交 `cec2ffd1400782733a3e2f897f9a69caa7e21974`。D7 Gate 仍是 `HIGHLEVEL_LABEL_RULE_DESIGN_READY`。`worth_dual_head_trial = false`，`worth_full_dataset = false`，`wait_for_delta_payload_max = true`。`rollouts_run = 0`，`trained_hgnn = false`，`trained_mappo = false`，`formal_system_modified = false`，`held_out_splits_read = false`。

## 2. 保留的 D7 证据与局限

主数据是 D6 场景 322016–322023 的真实到达标签。状态来自 `MAX_RATE_INDEPENDENT` 数据集重放和冻结 MAPPO-V2 seed 0，不是 Scheme C 轨迹。每个状态 128 个候选，合法池 13,327。Pareto 数是这 128 个候选上的非支配点数，不是全池前沿。

- t=15，horizon 15：Pareto 大小 1–7。8 个状态里只有 322022 的 Rule P 与抽样 urgent-first 是同一个候选。其余 7 个状态要换成 urgent-first，payload regret 范围是 0–2,616,257.151 bit，均值 1,352,814.324 bit；对应 urgent 增益均值 1,712,151.946 bit。描述性交换比 0.306220–2.171875，共 7 个状态。
- t=30，horizon 15：Pareto 大小 2–5。8/8 个状态的 urgent-first 都在并列组外。payload regret 范围 196,666.770–4,138,211.027 bit，均值 2,320,860.037 bit；urgent 增益均值 811,754.254 bit。描述性交换比 0.723035–33.885982。这些比值只描述抽样标签上的交换，`lambda_H` 保持 `NOT_SELECTED`。
- t=0，horizon 15：8/8 个状态的抽样 urgent-first 就在 1 bit payload 并列组内，payload regret 均值为 0。并列候选约占抽样的 28.3%。Rule P 相对更小候选 ID 的 urgent 增益均值 320,769.223 bit，8/8 个状态超过 1 bit。这是抽样标签上的并列收益，不是闭环收益。
- t=45，horizon 3：单独记录，不与 15 步标签平均，也不称作旧 J15。4/8 个状态的 urgent-first 需要超过 1 bit 的 payload 损失，regret 均值 79,947.171 bit，最大 410,149.031 bit。Rule P 的 tie-break 增益均值为 0。

D5 的 t=30 是另一组场景 322000–322003，4/4 个状态的 urgent-first 也在并列组外，payload regret 均值 2,172,220.624 bit。它与 D6 的方向相同，两组 bit 值没有相减，也没有并进上面的均值。D0/D1 没有 `real_payload_bits`，只提供零到达 legacy 对照。

局限保持不变：8 个 D6 场景只覆盖训练池的一部分；128/13,327 不是全池最优；标签是离线的，没有闭环 payload 或紧急服务测量；不宣称统计显著或泛化；没有读取 validation 或 reserved test。D7 的比例网格是 0、0.05、0.10、0.25、0.50、1.00 乘以各状态自己的抽样 payload 极差。比例 0.50 时这 8 个状态的抽样 urgent-first 都已进入集合。该现象只说明抽样极差比前沿 regret 更宽，不是批准的预算。

## 3. 需要负责人回答的一个问题

是否允许为了提高紧急服务，而在每个决策状态牺牲有限的总 payload？

- 若不允许，后续高层目标采用选项 A，不设置 `delta_payload_max`。
- 若允许，请给出每个决策状态的 `delta_payload_max`。它必须是绝对 bit 数，或相对该状态最高真实 payload 的明确百分比，并写入之后的配置和实验协议。

本记录不代替这个回答。D7 网格上的任何一个比例都还不是该预算。

## 4. 选项 A：严格 payload-first

目标是最大化 payload。urgent-service 只在 payload 相差不超过 1 bit 的候选之间打破平局，再以更小的候选 ID 结束剩余并列。这个规则不会为了紧急服务去选一个 payload 更低的候选。

适用条件：总传输量优先，研究目标不能接受实质吞吐量下降。

和 D7 的关系：

- t=15 与 t=30 的抽样最高 payload 在 1 bit 内是唯一的，所以这条规则相对“只按候选 ID 打破平局”的紧急服务增益为 0。D6 里看到的到达相关 urgent 冠军留在并列组外面。
- t=0 的抽样里有大量 payload 并列，tie-break 可以提高 urgent-service，同时 payload regret 为 0。这不能外推成完整闭环改进。
- t=45 继续用同一条 payload-first 规则，horizon 保持 3，不并入 15 步目标。

若负责人以后确认选项 A，后续路线是：高层监督保持 payload-first 加 1 bit urgent tie-break；低层执行器继续用下面的冻结 MAPPO-V2；不把会降低 payload 的紧急度项写进高层目标。那一步仍要另有指令。本记录不授权构建完整 urgency-aware 训练集，也不授权训练双输出 HGNN。

## 5. 选项 B：有约束的紧急度优先

先给定每个决策状态允许的最大 payload 损失 `delta_payload_max`，再在这个集合里最大化 urgent-service。1 bit 以内的 payload 差距仍记为 0 损失。集合内的 urgent 并列用更小候选 ID 打破。

适用条件：研究目标明确接受用有限的总 payload 换取更高的紧急服务，并且这个上限已经由负责人写成绝对 bit 或明确百分比。

当前状态：`delta_payload_max = OWNER_DECISION_REQUIRED`。在该值写入协议之前，没有可执行的选项 B 目标。

若负责人以后给出预算，后续路线才可以分开做这几件事：

1. 用真实到达标签建立双输出高层目标，一个头预测 payload，另一个头预测 urgent-service。推理时用已写入的 `delta_payload_max` 做约束。预测误差要分头报告。
2. 训练与闭环验证单独设计。离线 Pareto 不能代替闭环 payload 和紧急服务。
3. t=0、t=15、t=30、t=45 继续分开。t=45 保持 3 步，不与 15 步标签合成一个损失预算。
4. 低层执行器在其自身指令出现之前保持冻结。

这条路线现在不开始。不允许从 D7 曲线里挑一个比例充当预先批准的 `delta_payload_max`。`lambda_H` 也保持 `NOT_SELECTED`；单一标量效用不是本记录选定的目标。

## 6. 低层冻结约束

两个选项都不更换低层执行器。正式低层仍是冻结的 MAPPO-V2 seed 0，checkpoint SHA-256 为 `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。正式 Planner、奖励、动作、SafetyProjector 和该 checkpoint 保持不动。高层标签是离线监督，不写回当前 observation。

## 7. 停止条件

负责人作出第 3 节的选择之前：

- 不构建完整 urgency-aware 高层训练集；
- 不训练双输出 HGNN；
- 不训练 MAPPO；
- 不修改正式 Planner、奖励、动作、checkpoint 或旧数据集；
- 不运行新的 rollout；
- 不读取 validation 或 reserved test；
- 不进入 D9。

## 8. Gate

`HIGHLEVEL_URGENCY_OBJECTIVE_WAITING_FOR_OWNER`。

D7 报告与 summary 的关键数值一致，来源哈希已复核，选项 A 和选项 B 都保留在记录中。`delta_payload_max` 尚未获批准，所以本阶段停在等待。本记录没有选择 A 或 B。
