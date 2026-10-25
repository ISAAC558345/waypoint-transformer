# Stage A 执行指令：独立 urgency 年龄账本与逐步诊断日志

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置审计：docs/research/HTW_UrgencyAware_Training_Audit_CN.md
- 前置 Gate：URGENCY_STATE_AUDIT_COMPLETE
- 本阶段 Gate：URGENCY_LEDGER_TESTS_PASS / URGENCY_LOGGING_INVARIANT_PASS / URGENCY_STAGE_A_BLOCKED
- 本阶段不训练、不修改正式 HGNN/MAPPO/奖励/动作/SafetyProjector，不覆盖正式 checkpoint 和 P1–P5 历史结果。

## 1. 目标

只实现两项基础能力：
1. 在正式队列转移之外维护可复核的 episode admission-age bit ledger，计算浮标当前等待年龄与 urgency。
2. 在独立实验 runner 中记录完整 episode 的逐步诊断数据，以便后续判断低层收益变化来自运动、链路还是高层任务切换。

本阶段不把 urgency 接入模型输入，也不将它加入奖励；不启动训练。Stage A 的通过条件是账本与日志正确且不改变环境转移。

## 2. 当前已确认的队列语义

依据审计报告：
- 环境：L3BBitQueueEnvV2，M=6、K=12、episode_horizon=48、block_payload_bits=1e6、queue_capacity_blocks=4。
- 正式转移先服务本步开始时已存在的队列，再接纳本步 arrivals。
- 服务量按浮点 bit 计算；新到达无法放进剩余队列容量时记为 dropped。
- reset 时已有的队列数据没有历史时间戳，必须标记 age_origin=reset_convention，不能声称其 episode 之前的真实等待时间已知。
- 当前浮标特征里名为 urgency 的旧列是 queue/max(queue)，它代表相对队列大小，不是等待年龄。Stage A 不修改或覆盖此字段。
- 现有 P5 结果缺少逐步位置、动作、SINR、干扰和 served bits。新日志须在独立 runner 中记录，不修改 L3BBitQueueEnvV2.step。

## 3. 实现边界

新建独立模块/实验目录；实际命名可依仓库约定，但必须避免覆盖：
- src/wt/benchmark_v2/bit_queue_v2.py 的正式 step 语义；
- 正式 Scheme C / FormalPlanner；
- src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py 的正式训练行为；
- phase6g_seed0.pt、Scheme C checkpoint、旧 dataset 和 P1–P5 结果。

建议新增：
- urgency ledger wrapper/helper；
- 独立逐步日志 runner；
- schema/version 与配置文件；
- 单元测试和短程 smoke test；
- results/phase6i_urgency_stage_a/ 独立输出目录；
- docs/research/HTW_UrgencyAware_StageA_Report_CN.md。

先检查仓库目录和测试习惯，遵循既有结构；不要为满足上述示例路径而覆盖现有文件。

## 4. 年龄账本规则

对每个浮标维护按入院时间分桶的 queued bits。每个桶至少记录：
- admission_step；
- age_origin（reset_convention 或 environment_arrival）；
- bits_remaining。

### Reset
- 将 reset 时的 q_bits 全部放入 admission_step=0 的初始桶，age_origin=reset_convention。
- 在 t=0 服务前年龄为 0。
- 报告必须说明：这是可复核的 episode 内年龄约定，不是 episode 开始前的真实年龄。

### 每步
在 env.step 前，根据当前 env.t 计算各桶 age = max(0, env.t - admission_step)，并导出本步服务前的：
- oldest_wait_age_k；
- mean_wait_age_k（按当前排队 bits 加权）；
- age_weighted_backlog_k = sum(age * bits_remaining)；
- urgency_valid_k（q_bits>0 时为 1，否则为 0）；
- urgency_k = clip(oldest_wait_age_k / 48, 0, 1)，空队列的 urgency 数值置 0，并用 validity mask 区分。

在 env.step 返回后，按环境实际返回的 served_bits、admitted_arrival_bits、dropped_bits_step 和 float_snap_bits 更新账本。具体字段名必须先检查 info 的实际 schema，不得猜测字段：
1. 先从最老入院桶开始扣除实际 served bits，扣减总量必须与本步真实 served 总量一致。
2. 本步 arrivals 只能在服务完成后入桶，不能被同一步 served。
3. 若本步新到达发生容量丢弃，只从本步新入院桶扣掉实际 dropped bits。
4. 若环境执行 float snap，则按环境实际 snap 量修正账本，不能让年龄桶总量漂离 q_bits。
5. 每步检查对每个浮标：ledger_bits_sum 与 env.q_bits 的绝对差不超过 env.tol_bits（审计基线为 1 bit）。
6. episode 结束时仍要检查桶总量与 q_bits 一致。

若现有 info 未暴露某个必要计账字段，优先通过 env.step 返回值已有数据或在 wrapper 中复用同一已计算的 accounting 结果；不要第二次调用 arrivals()，不要重算或改变环境的队列转移。若不能在不改正式转移的前提下获取字段，标记 blocked 并报告缺口，不得以估算值冒充真实字段。

## 5. Capture/restore

如果 runner 支持同一状态分支，wrapper 必须单独实现 ledger capture/restore：
- 保存每个浮标的全部年龄桶、age_origin、admission_step、bits_remaining 和 ledger schema version；
- restore 后 ledger 总量与当前 env.q_bits 一致；
- 分支后的服务历史不同，不能只从一个 q_bits 标量重建各年龄桶；
- 不修改正式 shadow.capture_state；只在新 wrapper 的 state object 中保存 ledger。

## 6. 逐步诊断日志

在独立 runner 中包住既有 step，日志函数只复制数据、不修改 env 状态。每个时间步至少记录：

### Step 前
- scenario_id、time_step、policy/checkpoint/config SHA-256；
- UAV 绝对位置、assignment、目标浮标、TaskOption remaining_hold；
- a_des、SafetyProjector 输出动作、最终执行动作、constraint_status、投影范数；
- 每浮标 q_bits、服务前 urgency、oldest/mean age、validity；
- 本次高层决策的 candidate_pool_hash、candidate_index、Scheme C score（仅在 t=0,15,30,45 决策时记录）。

### Step 后
- 每浮标 served_bits、generated/admitted/dropped bits、q_bits_after；
- 每 UAV 的 SINR、rate_bps；
- 干扰矩阵/每链路干扰（在同一步位置与关联下计算）；
- 本步累计 payload 增量、conservation_error_bits、ledger mismatch；
- SafetyProjector 状态、投影前后动作差；
- 最终 env.t。

重要：served_bits 必须按浮标存储。合法的一对一 assignment 下，可另导出 UAV-to-buoy 映射，但不能把全队 served bits 平均后伪装成单 UAV 服务量。每步记录必须保留浮标和 UAV 的明确索引。

日志不属于 observation，不可让 policy 读取未来值；禁止记录未来 arrivals 或未来 served bits 到当前状态特征中。

## 7. 必须通过的测试

1. Reset 初始化：年龄桶总和等于 q_bits；reset 初始队列 age=0 且 age_origin=reset_convention。
2. 服务先于到达：本步新到达不能计入本步 served。
3. FIFO/oldest-first 记账：served 只扣最老桶，且扣减总量与 env 返回 served 一致。
4. 容量丢弃：dropped 只减少本步新到达桶，不误删旧桶。
5. Float snap：账本按同一次 snap 对齐 q_bits，最大差不超过 tol_bits。
6. 空队列：urgency_valid=0；非空 age=0 的队列 urgency_valid=1。
7. 年龄单调：同一批未服务数据的 age 随 env.t 递增，不因观测调用改变。
8. Capture/restore：分支前恢复 ledger 和 q_bits 完全一致；分支后两臂 ledger 可独立演化。
9. Logging invariance：同一场景、seed、策略、动作序列分别开启/关闭日志，最终 payload、q_bits、assignment、位置和 episode 终止时刻一致。
10. Determinism：相同 seed 重复运行得到相同的 urgency/ledger 轨迹。
11. No future leakage：urgency 特征只依赖当前已存在的年龄桶与 q_bits；不调用 remaining_arrivals、不读取未来 served bits。
12. 正式模型兼容：当前 Scheme C 和 phase6g_seed0.pt 不做任何输入维度修改，旧模型加载/基线 smoke test 仍通过。

## 8. Smoke test 与输出

先运行上述单元测试，再对 2 个非 reserved 的 registry-approved smoke 场景执行完整或短程 episode。场景 ID 必须从 split registry 读取，不得手工猜测；不得使用 reserved 322064–322079。

输出：
- results/phase6i_urgency_stage_a/step_rows.csv 或 parquet；
- results/phase6i_urgency_stage_a/summary.json；
- results/phase6i_urgency_stage_a/schema.json；
- docs/research/HTW_UrgencyAware_StageA_Report_CN.md。

summary 记录 split、scenario IDs、git commit、policy/config SHA-256、schema version、每步/终止 ledger mismatch、payload/accounting error、日志开启关闭一致性、测试结果和 Gate。

## 9. Gate 与停止条件

URGENCY_LEDGER_TESTS_PASS：年龄桶在 reset、服务、到达、丢弃、snap、终止和 restore 下都与 q_bits 守恒。

URGENCY_LOGGING_INVARIANT_PASS：启用日志不改变动作、环境状态转移或最终 payload。

URGENCY_STAGE_A_BLOCKED：必要计账字段无法可靠获得、年龄账本无法守恒、restore 不完整或日志改变环境行为。

通过前不接入 HGNN/MAPPO、不改奖励、不训练。通过后先提交报告并等待确认，再进入 Stage B 跨任务基线评估。
