# Stage D4 执行指令：单决策点 urgency tie-break 因果归因试验

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_Tiebreak_ClosedLoop_Pilot_Report_CN.md
- 前置 Gate：URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE
- 本阶段 Gate：URGENCY_TIEBREAK_SINGLE_INTERVENTION_COMPLETE / URGENCY_TIEBREAK_SINGLE_INTERVENTION_INCONCLUSIVE / URGENCY_TIEBREAK_SINGLE_INTERVENTION_BLOCKED
- 性质：有限的单决策干预诊断；不训练、不修改正式策略/模型/奖励/checkpoint，不读取 validation/reserved test。

## 1. D3 结果与问题

D3 的四个训练场景两臂都完成 48 步，所有选择的标签 payload 差都在 1 bit 内，账本和计账通过。但完整 episode 的 tie-break 臂相对 payload-first 臂：
- 平均 payload 差：-2,252,512 bit；
- 4 个场景中 3 个 payload 下降，1 个上升；
- 平均 urgent-service 差：-18,282 bit，中位数 +135,467 bit；
- 平均 SINR 在四个场景中均下降；
- Gate 为 URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE。

这说明“候选标签在 1 bit 内并列”并不能保证真实 48 步闭环收益相同。D3 的候选标签 rollout 将 arrivals/arrivals_at 置零，而真实闭环保留 arrivals；加上高层决策后续重复应用 tie-break，可能使两臂轨迹和后续 assignment 分叉。当前数据不能区分这两种机制的贡献。

因此不扩大 tie-break 规则、不训练 HGNN。下一步仅做单决策点干预，定位闭环负收益是由哪个决策时刻的 tie-break 引发，以及是否主要来自后续状态/assignment 分叉。

## 2. 固定范围与预算

- 只使用已登记 highlevel_v2_train 中的四个训练场景 322000–322003，运行前从 registry 读取并调用 assert_allowed(split="train")。
- 不读取 validation 322048–322063 或 reserved test 322064–322079。
- 使用冻结 MAPPO-V2 seed 0、原始正式 Scheme C checkpoint、matching decoder、adapter config、SafetyProjector 和环境。
- 不修改正式 Planner；通过独立诊断 runner 实现干预。不得绕过任何 split/候选合法性 guard。
- 新结果写入 results/phase6i_urgency_tiebreak_single_intervention/，不覆盖 D0–D3 的文件。
- 先复用 D3 的已有逐步日志和决策记录，只有缺少所需反事实轨迹时才重跑。
- 最多执行 4 场景 × 2 个干预时刻 × 2 臂 = 16 个完整 episode。先运行一个场景估算时间；若预计总时间超过 20 分钟，停止并报告，不得扩大预算。

## 3. 三类策略臂

每个场景、每个干预时刻都从相同 reset seed 开始，配对运行：

A. Scheme-C control：
- t=0、15、30、45 全部使用正式 Scheme C assignment 选择逻辑。

B. Single tie-break intervention：
- 仅在指定干预时刻使用 D3 的 payload-first + urgency tie-break 规则；
- 其他所有高层决策时刻都恢复使用正式 Scheme C；
- 低层、环境、SafetyProjector 完全相同。

干预时刻分两组：
- Intervention-0：仅 t=0 使用 tie-break；t=15、30、45 恢复正式 Scheme C。
- Intervention-30：仅 t=30 使用 tie-break；t=0、15、45 使用正式 Scheme C。

注意：D3 的 tie-break 规则使用当前状态的候选标签；本阶段仍是 oracle-label 诊断，不是 learned policy。标签候选生成、1-bit tie 规则与 D3 保持一致。对 t=0 可复用 D3 现有标签/候选；t=30 若当前状态已因早期干预而改变，必须在该臂自己的 live state 重新生成合法候选子集与标签，不可复用控制臂未来状态的标签。

## 4. 额外的分解对照

为区分“标签 rollout 无 arrivals”与“真实闭环 arrivals”的影响，在干预决策时刻额外保存同一状态快照，计算两类 3/15 步诊断：
1. D3-compatible label rollout：按现有标签约定将 arrivals/arrivals_at 置零，只用于确认所选候选的标签 payload 差确实不超过 1 bit。
2. Real-transition paired rollout：在两个独立状态副本上保留真实 arrivals 和原始队列转移，执行相同的后续 horizon；此结果只用于诊断，不用于重新选择候选。

不允许用真实 rollout 的未来 payload 反向选择 tie-break 候选。若环境随机到达过程不能做到同一状态、同一外生随机序列配对，应记录该限制，不得声称完全因果隔离。

## 5. 必须记录的量

每个场景/干预时刻/策略臂记录：
- intervention_time、是否实际改变 assignment、payload-first assignment、tie-break assignment、候选子集哈希、状态哈希；
- 干预时刻的 payload 标签差、urgent-service 标签增益；
- t=0–48 每步 payload 增量、每浮标 served bits、arrivals/admitted/dropped bits、q_bits、urgency/age；
- 每个高层决策时刻的候选池哈希、候选数量、选中 assignment、Scheme C score；
- 每 UAV 位置、目标距离、首次进入服务半径、有效服务步数、SINR、干扰、路径长度、SafetyProjector 触发/范数；
- 完整 episode payload、urgent-service、终态队列、dropped bits、账本/环境计账误差；
- 与 control 的每步累计 payload 差曲线，以及 t=15、30、45 后的差值。

## 6. 解释规则

- 如果 Intervention-0 负收益而 Intervention-30 无明显变化：初始任务分配的微小变化可能通过后续状态演化放大；需检查后续 Scheme C assignment 和 SINR/队列的变化。
- 如果 Intervention-30 负收益：中段 tie-break 可能影响剩余 18 步的任务/链路状态。
- 如果两类干预均无一致方向，或变化主要由随机 arrivals 解释：停止继续研究 tie-break，不扩大数据集。
- 若单次干预的实际 payload 差仍大，即使标签 payload 差在 1 bit 内，也必须如实报告。不能以标签并列推断真实闭环 payload 不变。
- 只有四个场景，统计仅作逐场景配对描述，不宣称显著性或泛化。

## 7. 完整性与交付物

完整性检查：
- 两臂干预前共享相同状态 hash；
- 候选合法、payload 标签差不超过 1 bit；
- 所有 episode 完成 48 步；
- ledger/q_bits、payload/served bits、queue accounting 检查通过；
- 原 checkpoint、D0–D3 结果和正式 Planner 文件哈希不变。

交付：
- docs/research/HTW_UrgencyAware_Tiebreak_Single_Intervention_Report_CN.md
- results/phase6i_urgency_tiebreak_single_intervention/episode_rows.csv
- results/phase6i_urgency_tiebreak_single_intervention/step_rows.csv
- results/phase6i_urgency_tiebreak_single_intervention/decision_rows.csv
- results/phase6i_urgency_tiebreak_single_intervention/summary.json
- results/phase6i_urgency_tiebreak_single_intervention/schema.json

## 8. Gate 与停止

URGENCY_TIEBREAK_SINGLE_INTERVENTION_COMPLETE：两类单决策干预均完成配对，数据完整性检查通过，并报告逐场景差值与分叉时间。

URGENCY_TIEBREAK_SINGLE_INTERVENTION_INCONCLUSIVE：干预影响方向不一致、样本不足或随机到达无法配对，无法识别稳定机制。

URGENCY_TIEBREAK_SINGLE_INTERVENTION_BLOCKED：split、候选、状态哈希、账本/计账、checkpoint/config 或时间预算检查失败。

完成后停止，不训练 HGNN、不构建完整紧急度数据集、不改正式系统。只给出一个有证据支持的后续方向。
