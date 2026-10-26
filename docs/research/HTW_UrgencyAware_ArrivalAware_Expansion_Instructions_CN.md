# Stage D6 执行指令：扩展 Arrival-aware 高层候选标签验证

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_ArrivalAware_Label_Report_CN.md
- 前置 Gate：ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE
- 本阶段 Gate：ARRIVAL_AWARE_LABEL_EXPANSION_COMPLETE / ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE / ARRIVAL_AWARE_LABEL_EXPANSION_BLOCKED
- 性质：扩展训练池中的离线标签验证；不训练 HGNN/MAPPO，不修改正式 Planner、奖励、动作、checkpoint 或旧数据集。

## 1. D5 结论与决策

D5 在训练场景 322000–322003、t=30 对每个状态的 128 个候选分别运行零到达与真实到达标签，完成 1,024 个 rollout。Gate 为 `ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE`。

主要发现：
- 零到达 payload 标签与原始 J15 标签完全复现，最大误差为 0。
- 真实到达使四个场景的 payload top-1 全部改变，真实 payload top-1 相对零到达 top-1 的增益约为 6.84–8.00 Mbit。
- 紧急服务 top-1 也全部改变，真实到达下的 urgent-service 标签相对零到达冠军改善约 0.64–1.03 Mbit。
- 每个候选在真实到达下接纳 24 Mbit 新到达，平均其中约 7.33–8.33 Mbit 被服务、9.67–10.85 Mbit 被丢弃、5.50–6.35 Mbit 留在终态队列。
- 年龄账本与队列、payload 计账检查通过；没有训练或修改正式系统。

结论：当前零到达 J15 标签不适合直接作为真实到达条件下的 payload/urgency 排序监督。下一步应扩大 arrival-aware 标签的训练池覆盖并检查不同决策时刻的稳定性，但仍不能直接训练 HGNN，因为 `lambda_H` 和 `delta_payload_max` 尚未确定。

## 2. 本阶段唯一目标

验证 D5 发现能否在更多独立训练场景及 t=0、15、30、45 时刻复现，并为后续高层训练集准备可信的双目标标签：
- `payload_real_arrival_bits`：保留环境真实 arrivals 的候选执行实际交付量；
- `urgent_service_real_arrival_bits`：使用每步服务前的 Stage A ledger urgency 对实际 served bits 加权；
- 同时保留原始零到达 `j15_bits` 作为 legacy reference，不将其误称为真实到达标签。

本阶段不选择最终标量效用、不设定 `lambda_H` 或 `delta_payload_max`，不把 payload 与 urgency 合并为正式 HGNN 排序标签。

## 3. 场景、时刻和计算预算

- 从 `phase6ib_highlevel_dataset` 的 TRAIN_POOL registry 中读取按 ID 排序、未用于 D0/D1/D5 pilot 的前 8 个场景；预期为 322016–322023。执行前必须通过 registry 和 `assert_allowed(split="train")`。若实际 registry 不一致，记录实际结果并停止请求确认，不得手工绕过 guard。
- 每个场景评估 t=0、15、30、45。状态路径使用既有高层数据生成协议、冻结 MAPPO-V2 seed 0 和相同 hold=15。t=0/15/30 标签 horizon 为 15；t=45 horizon 为真实剩余 3 步，并在所有汇总中与 15 步 horizon 分开。
- 每个状态最多 128 个候选，总状态数 32；两个 arrival arms 最多 8,192 个 rollout。
- 先运行一个状态，记录真实耗时并估算总时间；若预计总耗时超过 20 分钟或任一状态超过 120 秒，停止并报告，不得扩容。
- 不读取 validation 322048–322063 或 reserved test 322064–322079。
- 冻结 MAPPO-V2 seed 0、Scheme C/decoder、adapter、环境、SafetyProjector、Stage A ledger 版本和所有 checkpoint/config。
- 新结果写入 `results/phase6i_urgency_arrival_aware_expansion/`，不覆盖 D0–D5 结果、原始 J15 shard 或任何 checkpoint。

## 4. 状态重放与候选抽样

每个场景从 reset 开始，沿既有 `MAX_RATE_INDEPENDENT` 状态生成路径和冻结 MAPPO-V2 重放，记录 t=0、15、30、45 的 live state、ledger snapshot、q_bits、state hash、candidate pool hash 和 checkpoint/config SHA-256。

候选列表必须在任何新 arrival-aware rollout 前保存。每个状态最多 128 个候选：
- t=0/15/30：若对应的既有训练 shard 有原始 J15 标签，选取稳定排序的 32 个高 payload 候选，再从其余合法候选按既有 `utility_rank` 分层抽样 96 个。
- t=45：不存在 15-step J15 标签时，使用当前状态合法 G1_RATE k=6 池的确定性加性速率排名构建 32+96 分层子集；该分数只用于抽样，不用于新标签选择。
- 使用新固定 seed 605，`SeedSequence([605, scenario_id, time_step])`。记录每个候选的 sampling source、rank stratum、idle count、候选池和子集 hash。
- 两个 arrival arms 必须使用完全相同的候选 ID/assignment。禁止根据新 rollout 结果补选或替换候选。
- 若某个时间点状态无法与其协议状态路径重建一致，标记 BLOCKED，不得强行套用冻结 shard。

## 5. 双标签 rollout 规则

对每个选定候选，从同一 live state 和对应 Stage A ledger snapshot 分叉：

### Arm Z：Legacy Zero-arrival
- 只在 scratch 环境副本中关闭 `arrivals/arrivals_at`，用于复现旧标签语义。
- t=0/15/30 时，与现有 `j15_bits` 对齐；若不能复现，在该状态停止并报告 BLOCKED。
- t=45 使用 3 步零到达诊断标签，不声称它是旧 J15 标签。

### Arm R：Real-arrival
- 保留正式环境 `arrivals_at(t)`，使用同一当前状态、同一候选、同一冻结低层确定性动作和同一 horizon。
- 到达量必须由环境真实函数产生；对同一状态不同候选，到达序列应一致，不得随候选顺序改变。
- 两臂均使用相同的服务先于新到达、容量丢弃、float snap 和 queue accounting 规则。
- 每步在服务前读取 Stage A ledger urgency；step 后用真实 per-buoy served bits、admitted/dropped 和 snap 更新 ledger。
- 真实到达标签仅作为离线监督/分析数据，绝不写回当前 observation 或用未来服务量构造当前 urgency。

记录每个候选的 payload、urgent-service bits、逐浮标 served bits、generated/admitted/dropped bits、终态队列、age-weighted backlog、ledger mismatch 和 accounting error。

## 6. 必须分析的指标

按场景、时刻、horizon 报告：
- Legacy J15 与 Arm Z 的 payload 标签复现误差；
- Arm Z 与 Arm R 的 payload Spearman、Kendall/pairwise inversion、payload top-1 改变率与真实 payload regret；
- 两臂 urgent-service 的排序相关性、反转比例、urgent top-1 改变率与真实 urgent regret；
- payload 和 urgent-service 的尺度、相关性、Pareto 前沿摘要；
- 每候选的 24-Mbit arrivals 分配到 served/dropped/final-queue 的守恒分解（只按实际环境返回值计算）；
- idle count、原始 queue urgency 分位数、决策时刻和 horizon 分层。
- t=45 的 3-step 结果必须独立呈现，不能与 15-step 标签混合排序或平均。

统计单位是场景/状态，候选 pair 仅用于描述排序，不当作独立 episode 样本。8 个场景只作扩展验证，不宣称最终泛化或统计显著性。

## 7. 完整性与 Gate

必须通过：
- 所有场景来自 TRAIN_POOL 且通过 registry guard；
- 状态/候选哈希可复现，两个 arms 使用相同候选和状态；
- t=0/15/30 的 Arm Z payload 复现旧 J15 标签；
- ledger 与 q_bits、payload 与 served、队列守恒误差均在环境容差内；
- 同状态候选的真实外生 arrivals 一致；
- 相同 seed 重跑一个预注册状态，标签和候选顺序可复现；
- D0–D5 结果、正式 Planner、checkpoint、旧 dataset hashes 不变。

`ARRIVAL_AWARE_LABEL_EXPANSION_COMPLETE`：8 个场景 × 4 个时刻全部完成，完整性检查通过，分时刻结果已报告。

`ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE`：到达影响在时刻/场景间不稳定，或标签排序信号不足以提出可靠的后续训练目标。

`ARRIVAL_AWARE_LABEL_EXPANSION_BLOCKED`：split、状态恢复、旧 J15 复现、候选合法性、到达配对、ledger/accounting 或时间预算检查失败。

## 8. 交付物与停止条件

生成：
- `docs/research/HTW_UrgencyAware_ArrivalAware_Expansion_Report_CN.md`
- `results/phase6i_urgency_arrival_aware_expansion/selected_candidates.csv`
- `results/phase6i_urgency_arrival_aware_expansion/labels.csv`
- `results/phase6i_urgency_arrival_aware_expansion/summary.json`
- `results/phase6i_urgency_arrival_aware_expansion/schema.json`

报告必须写明实际场景、时刻、候选数、rollout 数、耗时、各时刻排序变化、payload/urgent trade-off、队列丢弃/残留、所有 hash、Gate 和限制。

完成后停止：不训练 HGNN、不构建正式全量数据集、不修改正式系统。若 arrival-aware 信号在更多训练状态中可复现，下一步只提出“如何将双目标标签转为高层训练目标”的决策文件；仍需研究负责人明确 payload 损失预算，不能擅自选 `lambda_H`。
