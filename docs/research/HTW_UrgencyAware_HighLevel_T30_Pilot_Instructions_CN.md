# Stage D1 执行指令：独立扩展 t=30 紧急度标签信号试验

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_HighLevel_Label_Pilot_Report_CN.md
- 前置 Gate：HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE
- 本阶段 Gate：T30_URGENCY_SIGNAL_SUPPORTED / T30_URGENCY_SIGNAL_INCONCLUSIVE / T30_URGENCY_SIGNAL_BLOCKED
- 本阶段是预注册的探索性 pilot，不是最终确认性测试；不训练 HGNN，不改正式 Planner、低层、奖励或 checkpoint。

## 1. 为什么只检查 t=30

D0 完成 12 个状态、每状态 128 个候选，共 1,536 个 15-step rollout，状态重放、候选合法性、payload 标签复现、年龄账本守恒和 checkpoint/config 哈希全部通过。

D0 总体 Gate 是 HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE：
- 总体平均 Spearman 为 0.9507，平均成对逆序比例为 0.0431，未达到预注册的整体“有用信号”阈值。
- 分时刻结果中，t=0 的平均逆序比例为 0.0052，t=15 为 0.0289，t=30 为 0.0953；t=30 平均 Spearman 为 0.9277。
- payload 与 urgency 项的平均尺度比在 t=0 约 0.0825，t=15 约 0.3336，t=30 约 0.3222。因此不能从 D0 结果直接指定一个适用于所有决策时刻的 lambda_H。
- D0 的紧急服务第一名在 10/12 个状态与 payload 第一名有相同 payload；另外两个状态的 payload 分别少 67,008.16 bit 和 99,002.36 bit。

本阶段只检验一个有限假设：**在中段决策 t=30，等待年龄加权的服务标签是否能在更多独立训练状态上稳定提供额外的候选排序信息。** 这是探索性检查；不能把它当作 D0 的确认性显著结果，也不能据此宣称整体高层策略已改进。

## 2. 冻结与数据范围

- 仅使用 phase6ib_highlevel_dataset 的 TRAIN_POOL。
- 使用 registry 中按 scenario ID 排序后，排除 D0 pilot 场景 322000–322003 后的前 12 个训练场景；预期为 322004–322015。执行前必须通过现有 registry/assert_allowed 校验，不得硬编码绕过 guard。
- 只检查 t=30，不读取 validation 322048–322063 或 reserved test 322064–322079。
- 使用冻结 MAPPO-V2 seed 0、现有 G1_RATE k=6 合法候选池、原始 J15 标签和 Stage A ledger。
- Scheme C checkpoint、MAPPO checkpoint、epoch 6 checkpoint、adapter config、旧数据集及 D0 输出全部只读。
- 不训练、不调参、不修改正式模型/Planner/奖励/动作。
- 新结果写入 results/phase6i_urgency_highlevel_t30_pilot/。

若 registry 的实际排序、split 或数据完整性与预期不一致，按 registry 实际结果记录并停止请求确认；不得改用验证或保留测试场景补齐样本。

## 3. 候选采样

每个状态最多 128 个候选：
- 32 个从现有 J15 标签中稳定排序的 top-32；
- 96 个从其余合法候选按既有 utility_rank 分层等频抽样。
- 采样种子在运行前固定为 602，使用 SeedSequence([602, scenario_id, 30])。
- 候选列表必须在任何新 urgency rollout 之前生成并保存。
- 不得根据新标签重新选择或补入候选。
- 12 个状态总计最多 1,536 个 15-step rollout。先跑一个状态估算耗时；若单状态超过 900 秒或预计总耗时超过 30 分钟，停止并报告，不扩大预算。

## 4. 标签重放

每个 t=30 状态：
1. 从 reset 开始按 D0 同一条 MAX_RATE_INDEPENDENT 状态生成路径、冻结 MAPPO-V2 seed 0 和 hold=15 进行重放。
2. 重建 Stage A episode_admission_age_v1 ledger，并验证 live state、q_bits、state hash、candidate pool 顺序与冻结 shard 一致。
3. 从同一状态和 ledger 快照分叉，每个候选固定执行 15 步。
4. 为与现有 J15 标签可比，候选标签 rollout 的 scratch environment 中 arrivals/arrivals_at 继续置零；不得修改源状态、正式环境或未来状态特征。
5. 每步在服务前读取 urgency，使用实际 served_bits 更新 ledger。

定义：
- payload_term = payload_increment_bits / 1e6
- urgent_service_term = sum_t,k(urgency_k_before_service[t] * served_bits[k,t]) / 1e6
- utility(lambda_H) = payload_term + lambda_H * urgent_service_term

本阶段不指定最终 lambda_H。报告 urgent_service_term 的分布、与 payload_term 的相关性、候选对的排序逆序比例、Spearman、top-1 改变率及 top-1 payload regret。不要把 rollout 的未来 served bits 写入当前 observation。

## 5. 预注册判据

这些阈值只用于本次 t=30 探索性 pilot；不据此声称跨时间点或完整系统的普遍结论。

信号支持候选（T30_URGENCY_SIGNAL_SUPPORTED）须同时满足：
- 12 个状态的平均 Spearman 小于 0.95，或平均成对逆序比例大于 0.05；
- 至少 8/12 状态的逆序比例大于 0.05；
- 计算结果来自完整的 128 候选抽样，标签复现和账本守恒全部通过。

如果平均 Spearman >= 0.95 且平均逆序比例 <= 0.05，或只有少数状态出现变化，则为 T30_URGENCY_SIGNAL_INCONCLUSIVE。

若 state hash、候选合法性、payload 标签复现、ledger 守恒、checkpoint/config 哈希或 split 检查失败，则为 T30_URGENCY_SIGNAL_BLOCKED。

还必须报告 t=30 urgency 第一名相对 payload 第一名的 payload regret。即使排序信号达到阈值，若 top-1 regret 明显为负，也只能说明存在 trade-off，不得宣称高层效用更优。

## 6. 必须检查的完整性

- 每个被评估候选的 payload_term 与现有 j15_bits 差不超过环境容差。
- 每步 ledger bucket sum 与 q_bits 差不超过 env.tol_bits。
- capture/restore 不改变环境快照。
- 重复第一状态的同一批候选，payload/urgent-service 标签必须完全复现。
- 不读取 D0 之外的验证或 reserved 场景。
- 不修改旧 dataset shard，不在旧目录写回新标签。

## 7. 交付物

- docs/research/HTW_UrgencyAware_HighLevel_T30_Pilot_Report_CN.md
- results/phase6i_urgency_highlevel_t30_pilot/selected_candidates.csv
- results/phase6i_urgency_highlevel_t30_pilot/labels.csv
- results/phase6i_urgency_highlevel_t30_pilot/summary.json
- results/phase6i_urgency_highlevel_t30_pilot/schema.json

报告必须记录实际场景 ID、候选数、rollout 数、耗时、各状态 Spearman/逆序比例/top-1 regret、标签复现误差、ledger mismatch、所有哈希、Gate 和限制。

## 8. 完成后停止

- 若信号仍不明确：停止高层 urgency 标签扩展，不构建完整 urgency-aware 数据集；保留当前高层和低层正式模型。
- 若信号存在但 top-1 payload regret 明显：下一步只设计 utility 权衡与 lambda_H 选择规则，仍不训练 HGNN。
- 只有排序信号稳定、标签复现可靠且 utility 权衡规则事先写定后，才提出完整高层数据集与 HGNN 训练计划。

本阶段只运行 t=30 pilot，不训练，不修改正式系统。提交结果后停止并等待确认。
