# P6：信息紧急度与跨任务低层评估的联合推进方案

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 综合依据：P5 收益归因报告与 HTW_UrgencyAware_Hierarchical_Training_Plan_CN.md。
- 本方案是执行路线，不表示算法代码已修改或训练已完成。

## 1. 最新证据与原计划的调整

P5 Gate 为 T30_LOSS_ATTRIBUTION_COMPLETE。验证场景 322048–322063 的受约束距离重分配相对 Scheme C：
- t=30–45 平均 payload 差为 -409,993.669 bit；
- 完整 48 步 episode 平均 payload 差为 -466,677.647 bit；
- 87.85% 的有符号 episode 差出现在 t=30–45，12.15% 出现在 t=45–48；
- 10 个 changed 场景中，重分配 UAV 的窗口交付差为 -4,000,289.483 bit，固定关联 UAV 的窗口交付差为 -2,559,609.225 bit；
- 现有 UAV 表只有窗口汇总，没有逐步 served bits、SINR、干扰、位置、动作和逐步投影日志，因此不能把低收益的原因归结为轨迹、干扰或 t=45 重分配中的任一项。

结论：停止继续设计手工距离规则。下一轮必须先建立可信的紧急度定义和逐步诊断日志，再评估跨任务低层策略。不能把现有队列长度直接改名为信息紧急程度。

## 2. P0：合并审计——紧急度语义与诊断日志

先检查以下代码：
- src/wt/benchmark_v2/bit_queue_v2.py
- src/wt/l3b/env.py
- src/wt/benchmark_v2/phase6e/generators.py
- src/wt/benchmark_v2/phase6ib_highlevel_dataset/schema.py
- src/wt/benchmark_v2/phase6ib_highlevel_dataset/hgnn_revision.py
- src/wt/lowlevel_v2/observation.py
- src/wt/lowlevel_v2/critic_state.py
- src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py
- 高层 J15 标签、Scheme C 特征/排序和完整 episode evaluator。

审计必须明确：
1. L3BEnv.arrivals_at(t) 的到达规则、初始队列含义和服务顺序；
2. 当前没有逐包时间戳或真实 deadline 时，是否能在不改变服务容量和总队列语义的前提下追踪 bit-age；
3. 初始队列的历史年龄未知时，必须明确规定 reset 时已有数据的初始 age，不能假装知道其真实历史等待时间；
4. 低层奖励目前如何组合 served bits、投影惩罚和残差惩罚；高层 J15 标签是否只反映 payload；
5. 当前结果缺少哪些逐步字段，以及日志应在哪个调用位置采集才不会改变环境转移。

交付：docs/research/HTW_UrgencyAware_Training_Audit_CN.md。若无法证明 FIFO/年龄追踪语义安全，先提出明确标记为 queue_priority_proxy 的代理方案，不直接宣称是真实数据时延。

P0 不训练、不修改正式模型、不读取 reserved test。

## 3. Stage A：实现可审计的 urgency 状态与逐步日志

P0 审计通过后，在新版本实验目录实现，不覆盖正式环境行为；优先采用独立 wrapper/subclass 或显式实验配置。

### Urgency 状态
- 若可以实现真实 bit-age，为每个浮标维护按到达时刻分桶的 queued bits。每步按已审计的顺序服务旧队列，再处理本步到达。
- 年龄桶的 bit 总量必须与 q_bits 在环境计账容差内一致，并对服务、到达、队列上限丢弃、reset 和 episode 终止做守恒测试。
- 维护 oldest_wait_age_k、mean_wait_age_k 或 age_weighted_backlog_k，以及 urgency_valid_k，用于区分空队列与 urgency 数值为零。
- 固定并记录 age_scale_steps。不能根据 reserved test 调整。
- 如果无法可靠追踪年龄，只输出明确命名的 queue_priority_proxy，不报告为真实等待时延或 deadline miss。

### 每步诊断日志
每个完整 episode 的每个 time step 记录：
- scenario ID、time step、策略/checkpoint/config SHA-256；
- UAV 绝对位置、assignment、目标浮标、last movement；
- a_des、SafetyProjector 输出动作、最终执行动作、投影状态/范数；
- 每架 UAV 的 served bits、当前 SINR、rate、干扰；
- 每浮标 queue bits、generated/admitted/dropped bits、urgency/age 状态；
- 高层决策时刻、TaskOption hold、候选池哈希、候选索引和 Scheme C 分数；
- payload 与队列守恒误差。

日志只记录当前或已发生状态，不把未来 arrivals 或未来 served bits 写入当前 observation。必须测试启用/关闭日志时环境状态转移和最终 payload 一致。禁止覆盖 P1–P5 的历史结果。

## 4. Stage B：跨任务低层基准，暂不改奖励

先用冻结 MAPPO-V2 做跨任务分布评估，隔离任务分布效应与 urgency 奖励效应。

- assignment 来源包括正式 Scheme C、G1_RATE k=6，以及按预注册规则从合法候选池采样的 assignment。
- 在相同开发/验证场景和 reset seed 下配对执行；记录每种任务分布的生成器与概率。
- 用逐步日志测量首次进入服务半径时间、served bits、SINR/干扰、路径长度、投影率和完整 48 步 payload。
- 固定 assignment 的低层诊断与高层每 15 步重选的完整闭环评估分开报告。
- P1–P5 已用过的 16 个 validation 场景只能用于开发诊断，不可当作独立最终测试或反复选择最终模型。split 必须从 registry 读取并登记。

交付：docs/research/HTW_CrossTask_LowLevel_Baseline_CN.md、逐步日志和结果 summary。若冻结 MAPPO 在不同任务分布下系统性退化，再提出一个具体低层改进假设；不因距离启发式失败就直接改奖励。

## 5. Stage C：Urgency-aware 低层候选

Stage A/B 通过后才训练新低层：
1. 将目标浮标 urgency 与 validity mask 加入 Actor task observation；
2. 将每架 UAV 当前目标 urgency 和必要的全局摘要加入 centralized Critic；
3. 第一组先保持原奖励不变，只训练“增加 urgency 输入”的候选，隔离输入特征贡献；
4. 第二组再单独做 urgency-weighted reward 消融。候选形式为：
   r_t = served_bits_t / PAYLOAD_SCALE + lambda_urgency * sum_k(urgency_k[t] * served_bits_k[t]) / PAYLOAD_SCALE - lambda_project*c_project - lambda_residual*c_residual
   urgency 必须在本步服务前根据当前状态计算，并与本步 served_bits 的浮标索引对应。权重需先核对奖励尺度并在训练/开发验证池预先固定。
5. 保持 fly_toward + residual -> clip -> SafetyProjector 动作结构不变。旧 checkpoint 冻结，新 checkpoint 独立保存。
6. 训练任务分布覆盖 Scheme C、G1_RATE 和合法候选池采样；记录各来源占比和随机种子。

## 6. Stage D：Urgency-aware 高层 HGNN

低层候选通过基础接口和稳定性测试后，再开始高层：
1. 将 urgency/validity 加入 buoy node feature；edge feature 若加入 urgency，必须明示其语义。
2. 版本化数据集、manifest、state hash、模型输入维度和 checkpoint metadata。
3. 保留 payload-only 的 J15 标签作为对照，新增 urgency-aware 标签/utility。
4. 可用 utility = normalized_payload + lambda_high * normalized_urgent_service，其中 urgent_service = sum_t sum_k urgency_k[t] * served_bits_k[t]。尺度和权重必须在训练/开发验证池预先确定。
5. 如果原排序标签只按 payload 生成，不能只增加 urgency 输入就声称高层已优化紧急服务。
6. 固定低层执行器生成/评估高层标签，避免两层同时变化造成归因不清。

## 7. Stage E：高低层完整闭环消融

所有组使用相同场景、reset seeds、环境参数和评估协议：

| 组 | 高层 | 低层 | 目的 |
|---|---|---|---|
| A | 原 Scheme C | 原 MAPPO | 正式冻结基线 |
| B | 原 Scheme C | urgency-input MAPPO（原 reward） | 隔离低层输入作用 |
| C | 原 Scheme C | urgency-input + urgency-reward MAPPO | 隔离低层奖励作用 |
| D | urgency-aware HGNN | 原 MAPPO | 隔离高层紧急度作用 |
| E | urgency-aware HGNN | urgency-aware MAPPO | 测量组合效应 |
| F | G1_RATE k=6 | 原 MAPPO | 原有高层基线 |
| G | G1_RATE k=6 | urgency-aware MAPPO | 检查低层跨高层泛化 |

主指标是完整 48 步真实累计 payload。紧急度指标包括最老/平均等待年龄、age-weighted backlog、按 urgency 分位数的服务 bits、高紧急度队列首次服务时间、剩余队列和 dropped bits。只有存在真实 deadline 时才报告 deadline miss。

辅助指标包括逐步 SINR/干扰、有效服务步数、路径长度、投影触发率、安全违规率和不同训练 seed 的稳定性。以场景为配对单位，报告均值差、95% bootstrap CI、正负收益场景数。不能把时间步当作独立样本。若 urgency-aware 提高紧急队列服务但降低总 payload，必须报告 trade-off。

## 8. 数据隔离与保护

- 原始 Scheme C、phase6g_seed0.pt、旧 dataset 和 P1–P5 结果永不覆盖。
- 新数据集、模型、配置、日志和图表放在独立版本目录。
- train/dev/reserved split 从仓库 registry 读取，不得手工猜测场景 ID。
- 已用于 P1–P5 的 validation 场景只作开发/诊断；最终模型选择不能反复依赖它们。reserved test 只在设计冻结后用于一次最终确认。
- 每阶段记录 Git commit、config/checkpoint SHA-256、数据 schema 版本、随机种子和逐场景原始结果。

## 9. 本轮执行边界与 Gate

当前下一步只执行 P0 审计并创建 docs/research/HTW_UrgencyAware_Training_Audit_CN.md。不要直接实现年龄桶、修改模型或启动训练。

P0 Gate：
- URGENCY_STATE_AUDIT_COMPLETE
- URGENCY_PROXY_REQUIRED
- URGENCY_AUDIT_BLOCKED

P0 完成后停止，报告字段语义、数据限制、建议的 urgency 定义和 Stage A 的具体实现边界，等待确认。后续按 Stage A → B → C → D → E 顺序推进。
