# Stage B 执行指令：冻结 MAPPO-V2 的跨任务分布基准

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_StageA_Report_CN.md
- 前置 Gate：URGENCY_LEDGER_TESTS_PASS、URGENCY_LOGGING_INVARIANT_PASS
- 本阶段 Gate：CROSSTASK_BASELINE_COMPLETE / CROSSTASK_BASELINE_INCONCLUSIVE / CROSSTASK_BASELINE_BLOCKED
- 本阶段只评估冻结策略，不训练，不将 urgency 接入 Actor/Critic/HGNN，不修改奖励、动作、SafetyProjector 或正式 Planner。

## 1. 为什么进入 Stage B

Stage A 已通过：
- 14 项单元测试全部通过；
- 年龄账本与 q_bits 的最大差约 1.86e-9 bit，远小于 1 bit 环境容差；
- 两个 smoke 场景 322000、322001 均运行到 t=48；
- 日志开启/关闭时 payload、队列、assignment、位置和终止时刻一致；
- 重复运行的 urgency 轨迹一致；
- 没有训练，没有修改正式 checkpoint。

本阶段目标是先建立“没有 urgency 输入/奖励时，冻结 MAPPO-V2 在不同合法任务分布下表现如何”的基线。这样后续 urgency-aware 低层的变化才能归因于紧急度机制，而不是任务分布或日志实现变化。

## 2. 冻结对象和数据边界

冻结并在 summary 中记录 SHA-256：
- Scheme C：使用正式当前 checkpoint 与 matching decoder；
- 低层：phase6g_seed0.pt，确定性 actor mean；
- candidate adapter/config、环境参数、SafetyProjector；
- 原始低层奖励和原始 Actor/Critic observation schema。

场景：
- 主开发评估池使用 phase6g_lowlevel_mappo/protocol.py 的 DEV_POOL，即 321096–321127（32 个场景）。启动前必须调用该模块的 assert_allowed(scenario, split="dev")，不能手工绕过 split registry。
- 不使用 phase6g RESERVED_POOL、BENCHMARK_DEV、HISTORICAL，也不使用高层 reserved test 322064–322079。
- 高层验证场景 322048–322063 已用于 P1–P5，只可作为历史诊断，不作为本阶段模型选择依据或最终独立测试。
- 不训练、不调参、不根据结果改变 checkpoint 或 reward。

若官方 FormalPlanner 的 episode runner 不接受 low-level DEV_POOL，可创建本阶段独立 runner：它必须先用 phase6g 的 dev split registry 验证 scenario ID，再复用 FormalPlanner.select_assignment、官方 graph builder、candidate adapter、冻结 actor 和 SafetyProjector；不得修改或关闭现有 guard，不得改正式 Planner。若无法在不破坏接口契约的情况下完成，报告 CROSSTASK_BASELINE_BLOCKED，不绕过检查。

## 3. 四种任务分布（同场景配对）

每种条件均执行完整 48 步 episode，高层任务在 t=0、15、30、45 更新，hold=15。每个场景在所有条件下使用相同 reset seed。

### A. Scheme C 动态分配
- 每个决策时刻使用当前正式 FormalPlanner 的 Scheme C scorer，在正式 G1_RATE k=6 合法候选池中选最高分 assignment。
- 这是将冻结低层放到 Scheme C 任务分布下的诊断性推理评估。因场景来自低层 DEV_POOL，结果标为 cross-task development diagnostic，不宣称为正式高层 benchmark。

### B. G1_RATE k=6 动态基线
- 每个决策时刻使用正式 G1_RATE k=6 候选池和现有 additive_assignment / tie-break 规则。
- 不更改候选生成器、k、合法性规则或 hold。

### C. G1_RATE k=5 随机候选（复现低层原训练任务分布）
- 每个决策时刻使用 build_pools(env, (("G1_RATE", 5),)) 生成合法池，并从池中均匀采样一个 assignment。
- 这是当前 phase6g TrainWorker 使用的任务分布；随机数由 scenario ID、条件 ID 和 replication seed 明确派生。
- 不能把它与 G1_RATE k=6 的确定性选择混为一组。

### D. G1_RATE k=6 合法候选池均匀采样
- 每个决策时刻从当前正式 frozen_candidate_keys(env) 返回的合法候选池中均匀抽取一个 assignment；预期候选数按实际池记录，不能硬编码为 13,327。
- 此条件用于测试低层在更广的合法任务分布下的执行表现，不作为推荐策略。

随机条件 C、D 各使用 3 个预注册 replication seeds；A、B 为确定性策略。若运行资源不足，不得静默减少 seed，先报告并等待决定。

## 4. 每步记录和 urgency 指标

复用 Stage A 的 AdmissionAgeLedger 与日志 schema；将新输出放到 results/phase6i_urgency_stage_b/，不覆盖 Stage A 结果。逐步记录每个场景/条件/replication：
- step 前后 UAV 位置、assignment、目标、hold、a_des、projected/executed action、SafetyProjector 状态/范数；
- 每个浮标的 served bits、q_bits、generated/admitted/dropped bits；
- 服务前 oldest_wait_age、mean_wait_age、age_weighted_backlog、urgency、urgency_valid；
- 每架 UAV 的 SINR/rate、服务发生时对应的干扰；
- 每个高层决策的候选池 hash、候选数量、选中索引、Scheme C score（非 Scheme C 条件可为空）；
- 每步 payload increment、queue conservation error、ledger mismatch。

Urgency 在本阶段只用于记录与评价，不进入任何策略 observation 或 reward。禁止未来信息泄漏。

## 5. 指标

主要指标：
- 48 步完整 episode 的真实累计 payload_bits。

低层执行指标：
- 首次进入服务半径的时间、有效服务步数、路径长度、服务半径内的步数；
- SafetyProjector 触发率、投影范数、安全违规；
- SINR、rate、干扰与 served bits 的逐步关系。

紧急度相关诊断指标（不是本阶段的优化目标）：
- urgent-service bits = sum_t sum_k urgency_k[t] * served_bits_k[t]；
- 高 urgency 分位组获得的 served bits；
- oldest_wait_age 和 mean_wait_age 的 episode 最大值/均值；
- age_weighted_backlog、剩余队列和 dropped bits。

本阶段的 urgent-service bits 是诊断指标，不是策略奖励。不要仅凭它变大就宣称系统整体变好；必须同时报告 payload 和等待年龄指标。

## 6. 统计与复现

- 主分析以场景为配对单位。条件 C、D 先在每场景内对 3 个 replication 求均值，再与 A、B 做配对比较；另报告随机重复间方差。
- 报告均值、median、95% paired bootstrap CI、正/负/持平场景数、最差场景及所有异常。
- 每组必须包含相同的 32 个 scenario IDs；任何失败 episode 不得静默删除。
- 记录 Git commit、checkpoint/config SHA-256、候选池哈希、随机种子、环境参数、输出 schema version。
- 验证每个 episode 的 payload 与 served bits 总和一致、年龄账本与 q_bits 守恒，日志开启/关闭时环境结果不变。
- 由于 321096–321127 是开发池，结果用于理解任务分布与形成低层假设，不作为独立最终泛化证明。保留测试不得用于选择策略或超参数。

## 7. 交付物

- docs/research/HTW_CrossTask_LowLevel_Baseline_CN.md
- results/phase6i_urgency_stage_b/step_rows.csv
- results/phase6i_urgency_stage_b/episode_rows.csv
- results/phase6i_urgency_stage_b/summary.json
- results/phase6i_urgency_stage_b/schema.json
- 数据足够时，输出各任务分布的 payload、等待年龄、有效服务步数和投影率比较图。

summary 必须明确：
- 实际执行的场景/条件/replication；
- 每个条件的 assignment 生成规则与候选池大小；
- 模型/配置哈希；
- 每组失败数、计账异常、账本 mismatch；
- 是否调用 Scheme C，以及它在 low-level DEV_POOL 上属于诊断性推理而非正式高层测试；
- Gate 与后续唯一建议。

## 8. Gate

CROSSTASK_BASELINE_COMPLETE：
- 32 个场景在 A、B、C、D 四种条件下全部完成 48 步，随机条件完成 3 个 replication；
- 候选合法性、payload 计账、年龄账本守恒和日志一致性检查通过；
- 完整报告主指标与紧急度诊断指标。

CROSSTASK_BASELINE_INCONCLUSIVE：
- 任务分布差异没有稳定模式，或统计区间过宽。

CROSSTASK_BASELINE_BLOCKED：
- split/checkpoint/候选合法性/计账不一致，或不能安全复用正式 scorer 而需要绕过正式 guard。

## 9. 完成后停止

本阶段只评估冻结 MAPPO-V2，不训练、不修改 reward、不将 urgency 加入任何模型输入。完成后提交报告与结果到当前分支，给出 Gate、各任务分布下的主要结果、数据限制，并提出一个最有证据支持的低层优化假设。等待确认后才进入 Stage C urgency-aware 低层训练。
