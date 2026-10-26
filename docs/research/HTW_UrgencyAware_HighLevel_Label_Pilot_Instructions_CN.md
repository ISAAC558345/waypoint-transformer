# Stage D0 执行指令：高层紧急度标签可行性试验

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 依据报告：docs/research/HTW_UrgencyAware_LowLevel_Training_Report_CN.md
- 前置 Gate：URGENCY_LOWLEVEL_EVAL_COMPLETE
- 本阶段 Gate：HIGHLEVEL_URGENCY_LABEL_PILOT_PASS / HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE / HIGHLEVEL_URGENCY_LABEL_PILOT_BLOCKED
- 本阶段只做小规模标签 pilot，不启动完整 HGNN 训练，不修改正式 Planner、模型或 checkpoint，不读取 reserved test。

## 1. 最新低层结果与决策

低层报告 Gate 为 URGENCY_LOWLEVEL_INCONCLUSIVE：
- L1（增加 urgency/validity 输入、原奖励不变）完整闭环 payload 平均高于 L0 471,504.750 bit，但 95% CI [-757,724.445, 1,739,034.741] 跨 0；不替换正式低层。
- L2（增加 urgency 加权奖励）相对 L0 的 payload 平均低 1,336,567.578 bit，95% CI [-2,363,407.001, -316,417.315]；紧急服务提高 349,828.948 bit，95% CI [166,538.614, 534,941.364]。这是明确的 payload–urgency trade-off，不进入默认低层。
- 三个 L2 seed 的 payload 极差约 5.74 Mbit，结果不够稳定。
- 正式低层继续冻结为 phase6g_seed0.pt。高层 pilot 必须使用 L0，不得用 L1/L2 替换执行器。

因此下一步单独验证高层是否能利用等待年龄改善 assignment 排序，不同时训练高层和低层。

## 2. 本阶段目标与边界

验证：
1. 能否从原始状态重放中恢复 t=0、15、30 的 episode admission-age ledger，并与冻结 state shard 对齐；
2. 能否在不修改原 J15 标签的情况下，为小型候选子集计算 age-weighted urgent-service 标签；
3. 新 utility 与 payload-only J15 标签的尺度和排序差异是否足以支持下一阶段数据集构建。

本阶段不训练 HGNN、不修改现有 hgnn_tensors、SchemeCAdapter、FormalPlanner 或旧数据集，不覆盖 checkpoint/旧 shard。只新增独立 pilot 代码、结果和报告。

## 3. 固定数据与执行器

- 只使用 phase6ib_highlevel_dataset 的 TRAIN_POOL 322000–322047；pilot 固定选 registry 中前 4 个场景，必须经 registry 读取和验证。
- 时间点仅为 t=0、15、30。
- 不用 validation 322048–322063 选择权重；不读取 reserved test 322064–322079。
- 使用现有 frozen state/candidate shards、冻结 MAPPO-V2 seed 0 和现有 J15 标签。
- 使用 Stage A AdmissionAgeLedger 的 episode_admission_age_v1，age_scale_steps=48；reset 存量标记 reset_convention。
- candidate generator 保持 G1_RATE k=6，合法候选规则保持不变。

## 4. 候选子集与计算预算

每个状态最多评估 128 个候选，避免可行性阶段重跑全部约 13,327 个候选：
- 32 个按现有 J15 payload 标签稳定排序选出的 top-32；
- 96 个从其余合法候选中按固定 seed、按现有 J15 payload-rank 分位分层抽样，避免只评估高 payload 候选。
- 抽样仅依赖训练状态已有标签、候选 ID 和预注册随机种子，不得查看新 urgent-service rollout 结果后再挑候选。
- 若候选重复或样本不足，按稳定候选 ID 顺序补齐并记录。
- 4 个场景 × 3 个状态 × 最多 128 个候选，共最多 1,536 个 15-step rollout。先测单状态耗时并估算总耗时；若超出预算，停止并报告，不得静默扩容或启动全量任务。

## 5. 重建状态年龄与状态一致性

对每个 pilot 场景：
1. 从 reset 开始，用原数据集状态生成路径中的同一确定性 MAX_RATE_INDEPENDENT assignment、冻结 MAPPO-V2 seed-0 和 hold=15 重放。
2. 同时维护 Stage A ledger；在 t=0、15、30 保存 ledger capture/restore 快照。
3. 核对 live state arrays 与冻结 state shard、state hash、q_bits 和候选池哈希。
4. 核对 ledger bucket sum 与 q_bits 的差不超过 env.tol_bits；年龄桶和环境状态都必须能独立 restore。
5. 若重放状态、候选顺序或 checkpoint/config hash 不匹配，标记 BLOCKED，不得强行继续。

现有 J15 标签 rollout 会将 arrivals/arrivals_at 置零。新 pilot 必须沿用同一标签约定，保证 payload 标签和 urgency 标签可比；不得把未来 arrivals 加入当前 observation。

## 6. 计算 urgency-aware 标签

对每个抽样候选，从同一个 t 状态和对应 ledger 快照分叉：
- 低层固定为冻结 MAPPO-V2 seed 0；
- assignment 固定 15 步；
- 为对齐现有 J15 目标，candidate label rollout 中 arrivals/arrivals_at 仍置零；
- 每步服务前读取当前 ledger urgency；step 后用真实 served_bits、admitted/dropped 和 snap 更新 ledger；
- 记录 payload_bits、urgent_service_bits、每浮标 served bits、队列年龄和计账误差。

定义：
- payload_term = payload_increment_bits / 1e6
- urgent_service_term = sum_t,k(urgency_k_before_service[t] * served_bits[k,t]) / 1e6
- utility(lambda_H) = payload_term + lambda_H * urgent_service_term

本阶段先报告 payload_term 与 urgent_service_term 的分布、相关性、候选排序变化和尺度；不要用 validation/reserved 数据选择 lambda_H。若训练池尺度分析不足以支持一个有依据的固定权重，本阶段不决定最终 lambda_H、不生成正式训练标签。

## 7. 必须通过的检查

- 抽样候选的 payload_term 与已有 j15_bits 差不超过环境计账容差；不能复现则停止。
- ledger bucket sum 与 q_bits 每步守恒。
- arrivals/arrivals_at 置零只作用于 candidate label rollout 的环境副本，不得改变保存的状态快照或正式环境。
- 候选合法性、state hash、checkpoint/config hash 均一致。
- 不读取未来 served bits 来构造当前 urgency；urgent_service 只能用于候选未来 rollout 的训练标签。
- 重复同一 pilot seed 得到相同的候选样本、payload 标签和 urgent-service 标签。
- 原始 J15 shard、旧 Scheme C checkpoint、phase6g_seed0.pt 均保持不变。

## 8. 交付物

- docs/research/HTW_UrgencyAware_HighLevel_Label_Pilot_Report_CN.md
- 独立代码目录：src/wt/benchmark_v2/phase6i_urgency_highlevel_pilot/
- results/phase6i_urgency_highlevel_pilot/selected_candidates.csv
- results/phase6i_urgency_highlevel_pilot/labels.csv
- results/phase6i_urgency_highlevel_pilot/summary.json
- 新 schema.json，记录字段、公式、数据 split、seed 和所有 checkpoint/config SHA-256。

报告需列出实际 rollout 数、单状态和总耗时估计、payload 复现误差、age-ledger 守恒误差、utility 尺度、排序改变比例、缺失字段及 Gate。

## 9. Gate 与停止条件

HIGHLEVEL_URGENCY_LABEL_PILOT_PASS：
- 状态与候选来源可复现；
- payload 标签复现、年龄账本守恒和 checkpoint/config 校验全部通过；
- utility 两项的尺度、候选排序差异与计算预算有完整报告。

HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE：
- urgent-service 与 payload 排序关系无法形成有用信号，或样本覆盖不足。

HIGHLEVEL_URGENCY_LABEL_PILOT_BLOCKED：
- 状态/ledger 恢复不一致、payload 标签无法复现、候选非法、split 或哈希校验失败，或计算预算不可接受。

本阶段只做 pilot，不训练高层、不改低层、不改正式 Planner。完成后提交报告和结果，给出是否值得构建完整 urgency-aware HGNN 数据集的建议，然后停止等待确认。
