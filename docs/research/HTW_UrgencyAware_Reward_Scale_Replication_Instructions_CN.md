# Stage E2 执行指令：冻结低层策略下的多场景紧急奖励尺度验证

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_PriorityTag_Prototype_Report_CN.md
- 前置 Gate：PRIORITY_LEDGER_PROTOTYPE_PASS、URGENCY_REWARD_SCALE_READY
- 本阶段 Gate：URGENCY_REWARD_SCALE_REPLICATED / URGENCY_REWARD_SCALE_INCONCLUSIVE / URGENCY_REWARD_SCALE_BLOCKED
- 目标：确认 E1 的 16 步脚本化 smoke 是否能代表冻结低层策略下的奖励尺度；为负责人提供权重选择证据。
- 本阶段不训练、不更新策略、不修改正式 Planner/奖励/动作/checkpoint/旧数据集，不选择最终 lambda_u。

## 1. E1 结果与限制

E1 已通过 priority ledger prototype：
- 最大 ledger-q_bits gap = 9.313e-10 bit；
- ledger on/off 的正式环境状态指纹差为 0；
- synthetic tag 可复现，FIFO 顺序未受 tag 影响；
- 16 步 scripted smoke 记录普通/tagged 服务量和奖励尺度；
- Gate 为 PRIORITY_LEDGER_PROTOTYPE_PASS、URGENCY_REWARD_SCALE_READY。

但 E1 的尺度样本只有 scenario 321000 的 16 步 scripted smoke，且不是 MAPPO actor 产生的动作；记录的运动惩罚来自另一条一次更新 L1 probe，投影和残差惩罚未拆分。因此当前还不能据此确认某个 lambda_u 在真实低层训练分布中足够小。lambda_u 仍未选择。

## 2. 本阶段目标

在不训练的情况下，使用冻结的原始低层 checkpoint 与现有合法任务协议，采集多场景、完整 episode 的 reward-scale 统计：
- payload term；
- tag-aware urgency term；
- projection penalty 与 residual penalty（若环境/worker 可以分别准确记录）；
- 各候选 lambda_u 对每步 reward、总 episode reward 和排序的影响；
- ordinary/tagged served bits、tag rate、队列、丢弃和安全指标。

重点是确定 lambda_u 候选在真实冻结策略轨迹上相对 payload 和运动惩罚的分布，而不是通过这些离线统计选择表现最好的策略。

## 3. 场景与预算

- 只从 phase6g low-level TRAIN_POOL registry 读取前 8 个合法训练场景，预期 321000–321007；执行前必须通过 registry 和 assert_allowed(split="train")。
- 不使用 DEV_POOL 321096–321127、BENCHMARK_DEV、RESERVED_POOL、HISTORICAL，亦不使用高层 validation 322048–322063 或 reserved 322064–322079。
- 每个场景最多运行一个 48 步 episode；使用冻结 phase6g_seed0.pt 和既有合法任务协议，不能更新权重。
- 若现有正式 runner 不能在不改变正式行为的情况下暴露动作/惩罚项，则使用独立 diagnostic wrapper；不得修改正式 worker 或绕过 split guard。
- 先运行一个场景计时。若预计 8 个 episode 总耗时超过 10 分钟，停止并报告实际吞吐，不得扩容。
- synthetic tags 使用已冻结协议：tag_rate=0.25、beta_tag=0.25、SeedSequence([606, scenario_id, admission_step, buoy])。不读取未来信息。
- 所有输出写入 results/phase6i_priority_tag_reward_scale_e2/，不覆盖 E1 输出。

## 4. 实验组与权重范围

不需要分别运行不同 lambda 的环境 episode，因为 lambda 不改变环境转移，只改变计算出的奖励。对每条冻结轨迹离线重算：
- lambda_u ∈ {0, 0.0005, 0.001, 0.002}；
- 0.05 仅作为高风险参考，不作为训练候选。

对于每个 lambda，计算：
- step reward 的均值、P50/P95/P99、最大值；
- urgency bonus / payload term 的均值和分位数；
- urgency bonus / (projection + residual penalties) 的分布；若惩罚不能拆分，明确说明；
- 每个 episode 的累计 payload、urgent-service bits、累计 bonus、累计惩罚；
- step-level reward 排序变化及其是否只发生在 payload 差距不超过 1 bit 的 tie；
- 在相同轨迹下 reward sign changes（若存在），但不要将重算 reward 当成新策略表现。

不得用 reward scale 推断训练后 payload 必然不下降。不同 lambda 训练后的闭环收益仍需要后续独立训练与评估。

## 5. 完整性与测试

- 原始 checkpoint SHA-256 与 E1 记录一致；
- 每个场景通过 split registry；
- 同一场景/seed 重跑一个预注册场景时，tag 序列、动作、服务、payload 与账本轨迹可复现；
- ledger bucket sums 与 q_bits 在每步误差不超过 1 bit；
- served_ordinary + served_tagged = served_bits；
- generated = admitted + dropped；
- 日志开启/关闭不改变正式环境转移；
- tag 仅在到达时生成，当前 observation 不含未来 tag/arrivals；
- 不修改正式 checkpoint、Planner、reward、action、SafetyProjector、E1 或 D0–D7 文件。

## 6. 交付物

- docs/research/HTW_UrgencyAware_Reward_Scale_Replication_Report_CN.md
- results/phase6i_priority_tag_reward_scale_e2/step_rows.csv
- results/phase6i_priority_tag_reward_scale_e2/episode_rows.csv
- results/phase6i_priority_tag_reward_scale_e2/summary.json
- results/phase6i_priority_tag_reward_scale_e2/schema.json

报告必须包含实际场景 IDs、耗时、checkpoint/config hashes、lambda 候选分布、payload/bonus/penalty 比例、队列守恒和 Gate，并给出候选权重的证据化建议；明确最终权重仍需负责人确认。

## 7. Gate 与完成后停止

URGENCY_REWARD_SCALE_REPLICATED：8 个 episode 完成、账本和环境不变性检查通过、尺度统计齐全。

URGENCY_REWARD_SCALE_INCONCLUSIVE：奖励尺度跨场景变化大，或样本不足以推荐一个保守候选。

URGENCY_REWARD_SCALE_BLOCKED：split、checkpoint、账本、计账或日志一致性失败，或预算超限。

完成后提交报告与结果并停止。不训练 L1/L2，不改正式奖励，不启动完整 benchmark。由报告提出建议后，等待负责人明确 lambda_u；只有确认后才进入独立 observation/reward builder 和短程 smoke。
