# Stage E1 执行指令：优先级桶账本原型与紧急奖励尺度校准

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置审计：docs/research/HTW_UrgencyAware_PriorityTag_Audit_CN.md
- 前置 Gate：URGENCY_PRIORITY_SCHEMA_BLOCKED
- 本阶段 Gate：PRIORITY_LEDGER_PROTOTYPE_PASS / URGENCY_REWARD_SCALE_READY / PRIORITY_LEDGER_BLOCKED
- 目标：解决当前服务量无法按 priority tag 分解，以及 lambda_u=0.05 相对现有运动惩罚可能过大的问题。
- 本阶段不训练、不运行完整 benchmark、不修改正式环境、Planner、奖励、动作、checkpoint 或旧数据集。

## 1. E0 审计后的决定

E0 已确认：
- arrivals API 只返回浮标级到达块数；正式 served_bits、dropped_bits 也是浮标级聚合，不能反推出被服务数据的 priority tag。
- 当前不存在 priority_tag。真实海事业务标签不可用，因此第一版只能在独立实验中使用明确标注的 synthetic priority tag。
- lambda_u=0.05 的 bonus 虽不超过 payload 项的 5%，但一次 L1 探针中的惩罚约为 0.0001697，而 5% payload bonus 上界约为 0.07354，约为惩罚的 433 倍。按旧 buoy-level urgency 比例 0.268 作示例时，估计 bonus 约为惩罚的 116 倍。旧比例不是 tag-aware 数据，不能当成最终校准值。
- 因此不能直接用 0.05 训练。先构造可审计的 tag ledger，并对更小的权重候选进行尺度核算。

## 2. 本阶段只实现独立原型

新建独立目录，建议为 src/wt/benchmark_v2/phase6i_priority_tag/。遵循仓库结构，但不得修改或覆盖：
- src/wt/benchmark_v2/bit_queue_v2.py
- Stage A AdmissionAgeLedger
- 正式 lowlevel_v2/observation.py 与 critic_state.py
- 正式 reward、SafetyProjector、FormalPlanner
- phase6g_seed0.pt、Scheme C checkpoint、D0–D7 历史结果

实现只读跟随正式环境转移的 priority-tag ledger prototype。它记录 tag 桶，不改变正式队列出队规则，也不把 tag 用作 FIFO 排序键。

## 3. 合成 tag 协议（仅实验原型）

第一版仅使用二元 tag：
- synthetic_priority_tag=0：普通信息；
- synthetic_priority_tag=1：合成的紧急信息。

预注册候选配置：
- tag_rate=0.25：每个新到达块独立抽样一次，整块 1e6 bit 共用一个 tag；
- beta_tag=0.25；
- tag_seed 使用 SeedSequence([606, scenario_id, admission_step, buoy])；
- reset 时已有队列全部 tag=0、age_origin=reset_convention、admission_step=0。这只是 episode 内约定，不代表真实历史优先级。

桶级紧急度：
- age_b(t)=max(0, t-admission_step_b)；
- u_age_b=clip(age_b/48,0,1)；
- u_b=clip(u_age_b+beta_tag*tag_b,0,1)。

所有配置、日志和报告都必须写明 synthetic，不能宣称 tag 来自真实业务。第一版 tag 不改变 FIFO 服务顺序，只影响离线服务奖励统计。若 arrivals 接口无法将每个到达块与确定性 tag 映射起来，报告 BLOCKED，不得伪造 served-tag 分解。

## 4. 桶级服务、丢弃与守恒

每个桶记录 admission_step、age_origin、priority_tag、bits_remaining 和稳定插入顺序。

每步只使用正式环境已返回的 per-buoy served_bits、admitted_arrival_bits、dropped_bits、float_snap_bits 更新账本：
1. 服务按最老桶 FIFO 扣减，并累计 served_ordinary_bits 与 served_tagged_bits。
2. 本步溢出丢弃只从本步新到达桶扣除，不能从旧队列扣除。
3. 新接纳的数据在服务后入桶，不能在同一步被服务。
4. 正 float snap 从最老桶扣，负 snap 加到最新桶；snap 不计入服务奖励。
5. 每浮标 ledger_bits_sum 与 q_bits 差不超过 tol_bits=1。
6. served_ordinary_bits + served_tagged_bits = served_bits；tag0_bits + tag1_bits = q_bits；generated = admitted + dropped。误差必须在既有容差内。
7. capture/restore 保存每个桶的 tag、age_origin、admission_step、bits 和插入顺序。

必须先核验 info 的实际字段名，不得猜测，也不得重复调用 arrivals_at 来重算环境转移。

## 5. 本阶段只做奖励尺度分析，不选择最终权重

原奖励：
r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual

候选紧急奖励：
r = r_base + lambda_u * urgency_served_bits/1e6

其中 urgency_served_bits 是每个实际服务桶在服务前的 u_b 乘以该桶本步实际 served bits 后求和。

本阶段只核算以下候选：
- lambda_u ∈ {0, 0.0005, 0.001, 0.002}。
- 0.05 只作为 E0 已审计的高风险参照，不是默认训练候选，也不得自动训练。
- 不训练、不跑长 episode。使用已保存训练池 reward-scale 记录与 tag-ledger 单元测试/短程 smoke 数据，报告各候选 bonus 的均值、P50/P95/P99、最大值、相对 payload 项比例、相对投影/残差惩罚比例，以及是否改变奖励排序或符号。
- 若现有数据不足以准确计算 tag-aware bonus，只报告解析上界和字段缺口；不得用旧 buoy-level urgency 冒充 tag-aware 统计。
- 不使用 validation/reserved 数据选择权重，本阶段不选择最终 lambda。

即使 lambda 很小，若 penalty 数值极小，bonus 仍可能显著影响策略。必须分别报告相对 payload 和相对 penalty 的尺度。

## 6. 必须测试

- reset 初始 q_bits 全部映射到 tag=0 桶，账本总量守恒；
- 相同场景、seed、tag config 重复运行得到相同 synthetic tag；
- 每个新到达块只有一个 tag，tagged 与 ordinary bits 之和等于到达 bits；
- FIFO 顺序不受 tag 影响；
- 本步新到达不能进入本步服务量；
- drop 只减少本步新到达；
- snap 不计入服务奖励；
- 服务分类、队列分类、总 q_bits 全部守恒；
- capture/restore 后 ledger 与 q_bits 及每个桶 tag 一致；
- urgency 只基于当前已有桶，不读取未来 arrivals 或 served bits；
- 账本/日志开启与关闭时，正式环境的真实状态转移、served_bits 和 payload 完全一致；
- 正式 checkpoint、Planner、reward、D0–D7 文件哈希保持不变。

## 7. 交付物

- docs/research/HTW_UrgencyAware_PriorityTag_Prototype_Report_CN.md
- 独立 priority-tag ledger/helper 与 schema
- 单元测试和短程 smoke 测试结果
- results/phase6i_priority_tag_prototype/summary.json
- results/phase6i_priority_tag_prototype/reward_scale.json
- results/phase6i_priority_tag_prototype/schema.json

报告必须列出实际修改文件、tag 配置与 seed、守恒误差、测试结果、lambda 候选尺度、已知限制和 Gate。

## 8. Gate 与停止

PRIORITY_LEDGER_PROTOTYPE_PASS：tag 桶账本与正式 q_bits、服务、丢弃、snap 守恒；FIFO 不变；tag 可复现；日志不改变环境。

URGENCY_REWARD_SCALE_READY：存在可信 tag-aware 服务数据或明确解析界限，且候选权重相对 payload 与 penalty 的尺度已报告。最终 lambda 仍需负责人确认。

PRIORITY_LEDGER_BLOCKED：无法把实际 served bits 精确分配到 tag 桶、无法保持队列守恒，或环境转移一致性测试失败。

完成后提交原型、测试与报告并停止。不要训练 L1/L2，不修改正式 reward，不启动完整 benchmark。下一阶段必须先复核本报告，再由负责人确认最终 lambda_u 后才进入 observation/reward builder 与训练。
