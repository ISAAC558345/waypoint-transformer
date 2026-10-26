# Stage E0 执行指令：小权重紧急度奖励与信息优先级标签设计

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 依据：docs/research/HTW_UrgencyAware_Objective_Decision_CN.md、docs/research/HTW_UrgencyAware_LowLevel_Training_Report_CN.md、docs/research/HTW_UrgencyAware_HighLevel_Label_Rule_Design_Report_CN.md
- 研究负责人已选择“方案 A 的修订版”：payload 仍是主目标；信息紧急度获得小幅额外奖励；特定信息可通过 priority tag 在年龄较小时也获得较高紧急度。
- 本阶段 Gate：URGENCY_PRIORITY_SCHEMA_READY / URGENCY_PRIORITY_SCHEMA_BLOCKED
- 本阶段只做数据语义审计、奖励尺度核算和独立实验协议，不启动长训练、不修改正式 Planner/奖励/动作/checkpoint/旧数据集。

## 1. 目标解释与边界

不要再把方案 A 解释为“紧急度完全不进入奖励”。负责人已明确要求：传输紧急信息可获得额外奖励，但权重必须较小；紧急程度既由等待年龄决定，也可由信息标签提高。

将其实现为**payload 主目标 + 有界 urgency bonus**，而不是任意加权和：
- 高层：payload 仍为首要排序目标。第一版只允许 urgency 在 payload 1-bit 并列候选中作为 tie-break，不允许高层用 urgency 选择 payload 明显较低的 assignment。
- 低层：在原 served-bits reward 上加入小权重 urgency-weighted served-bits bonus；保持原有 projection/residual penalties、动作结构和 SafetyProjector 不变。
- 新训练模型/实验必须独立保存。正式 Scheme C、正式 MAPPO-V2 与历史结果均冻结。
- 本阶段先设计和审计 urgency 标签，不立即训练模型。先确认当前队列能否表示不同信息的优先级，以及如何在服务/丢弃时准确统计每类 bits。

## 2. 信息紧急度不能只有年龄

当前 Stage A ledger 能跟踪 episode admission age，但当前环境 q_bits 是每浮标聚合队列。不能把一个 buoy-level priority 标量直接赋给整条队列，因为同一浮标可能同时有普通数据和高优先级数据。

必须先审计当前 arrivals/queue 的数据结构和服务规则，并设计可追踪的 priority-class ledger。最小可行分类：
- priority_tag=0：普通信息；
- priority_tag=1：标记为紧急的信息。

若现有环境不能提供真实应用标签，本阶段可以在独立实验配置中生成**明确标注为合成的 priority tag**，但必须固定生成规则和 seed，不得声称来自真实海事数据集。优先级标签必须随数据块进入队列并随服务/丢弃扣减；不能仅给某个 buoy 永久贴标签。

若现有环境按 FIFO 服务，新增 ledger 必须遵守 FIFO，且不改变正式队列服务顺序。tag 影响奖励优先级，不自动改变环境的队列出队顺序。若研究意图是优先级队列调度，需另立实验，不能与奖励改动混在一起。

## 3. 候选紧急度定义

每个浮标队列按 admission step 和 priority tag 维护独立 bit buckets。对桶 b：
- age_b(t)=max(0, t-admission_step_b)；
- age urgency：u_age_b(t)=clip(age_b(t)/48, 0, 1)；
- tag boost：tag_b ∈ {0,1}；
- 合成 urgency：u_b(t)=clip(u_age_b(t)+beta_tag*tag_b, 0, 1)。

第一版建议 beta_tag=0.25 作为**预注册候选值**，含义是紧急标签在刚入队时就增加 0.25 urgency；它不是已验证的最优值。对普通信息，紧急度仍随等待年龄增加；对 tagged information，紧急度从更高的基线开始增长。空队列通过 validity mask 区分，不能仅靠 urgency=0 表示空队列。

如果 priority tag 有多级语义，先不要自行扩成多级。第一版只用二元 tag，并将 tag 生成率、每浮标 tag 分布和 seed 写入配置。若当前 arrivals API 无法稳定区分不同数据块的 tag，报告 BLOCKED 并提出最小接口改动，不要伪造逐信息标签。

## 4. 低层小权重奖励

低层候选先使用 urgency-input only，随后再单独增加 reward bonus。第一版 reward 形式：
`r_t = served_bits_t / 1e6 + lambda_u * urgency_served_bits_t / 1e6 - 0.05*c_project - 0.01*c_residual`
其中：
- `served_bits_t` 是全队实际交付 bits；
- `urgency_served_bits_t = sum_b u_b(t-before-service) * served_bits_b,t`；
- served bits 必须按真实 priority bucket/tag 归属统计；
- urgency 必须在本步服务前计算，不使用未来状态；
- 只在实际成功服务的数据上加 bonus；generated/admitted/dropped bits 不计为服务奖励。

权重约束：
- 第一版预注册 `lambda_u=0.05`，即 urgency=1 的服务最多增加相当于普通 payload 项 5% 的奖励。此值仅为小权重起点，不得宣称最优。
- 保留 `lambda_u=0` 的原奖励对照。
- 只有在训练池奖励尺度审计后才开始训练；若其他 reward terms 的量纲/尺度导致 0.05 并非小影响，应报告并停止，请负责人确认，不要擅自放大。
- 可在开发集上做预先限定的敏感性检查 `lambda_u ∈ {0, 0.02, 0.05}`。0.1 不作为默认；若要扩展须另行说明理由。
- 不以训练 reward 上升作为成功标准。必须同时报告 payload、urgent-service bits、age、tagged-service rate、queue/drop、SINR/干扰和安全指标。

## 5. 高层规则与低层奖励分工

- 高层第一版仍按 payload-first 选候选；只有 payload 在 1 bit 容差内并列时，才按候选真实到达标签的 urgency utility tie-break。不要把 D7 中比例网格改成预算，也不设置 `delta_payload_max`。
- 低层可使用小权重 urgency bonus 学习在已分配目标下更有效地服务高紧急信息。
- 不能声称这个组合严格保证完整 episode payload 不下降；低层轨迹变化仍可能改变实际 payload。需要在冻结 Scheme C 下进行配对闭环 Benchmark，并报告 payload–urgency trade-off。
- 高层训练标签只使用真实到达语义。t=0/15/30/45 分开处理；t=45 的 horizon=3 不与 15-step 标签混合。
- 现有数据表对 priority tag 的监督标签可能缺失。若没有 tag-aware served bits，必须先扩展独立 schema 和离线标签，不得从聚合的 buoy-level served bits 猜测哪类信息被服务。

## 6. 本阶段审计任务

检查并报告：
1. arrivals API 是否可为不同新到达块提供 tag；是否需要可复现的 synthetic tag generator；
2. queue capacity、FIFO、float snap、drop accounting 如何在 priority buckets 上守恒；
3. 如何从实际 served_bits 准确分解 ordinary/tagged served bits；
4. Actor/Critic 观测如何加入当前目标的 age urgency、tag 和 validity；
5. reward 各项量纲、基线 payload 项和 penalty 的实际分布，确认 lambda_u=0.05 是否确实小；
6. 高层 payload-first tie-break 能否在现有标签 schema 中使用真实到达的 urgency-service 标签；
7. 所需的新 schema、独立代码目录、checkpoint metadata 和测试；
8. 对照组 L0 原始低层、L1 urgency input only、L2 urgency input + lambda_u=0.05 bonus 的公平比较设计。

交付 `docs/research/HTW_UrgencyAware_PriorityTag_Audit_CN.md`，明确哪些字段现有、哪些需要新增、tag 如何生成、账本如何守恒、奖励尺度与具体实现边界。

## 7. 本阶段禁止事项

- 不运行长时间训练或完整 benchmark；
- 不修改正式 Planner、正式 HGNN/MAPPO、正式奖励、动作、SafetyProjector；
- 不覆盖 phase6g_seed0.pt、Scheme C checkpoint、D0–D7 数据或旧标签；
- 不读取 validation/reserved test 来决定 tag 比例或 lambda；
- 不把 synthetic tag 说成真实业务标签；
- 不把 tag bonus 解释为改变物理传输能力，它只改变策略优化优先级。

## 8. Gate 与后续顺序

`URGENCY_PRIORITY_SCHEMA_READY`：tag 来源/生成规则、年龄公式、桶级服务/丢弃守恒、奖励尺度、模型接口和消融协议都明确，且无未来泄漏。

`URGENCY_PRIORITY_SCHEMA_BLOCKED`：无法从环境准确识别被服务数据的 tag、无法保持 queue accounting，或奖励尺度无法证明小权重是小影响。

完成后提交审计报告并停止。通过后下一阶段才实现独立 priority-aware ledger 与观测/reward builder，做单元测试和 smoke test；随后再训练低层 L1/L2。高层 HGNN 仍保持冻结，直至低层实验和高层 tie-break 标签接口分别验证。
