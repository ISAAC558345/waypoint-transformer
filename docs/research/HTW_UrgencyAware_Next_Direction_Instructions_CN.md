# Stage E17：冻结低层的任务分配可行性诊断执行指令

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线：feature/maritime-wt-stage1a；正式低层继续为 L0
- 固定来源提交：d54137679a3854d1f70b2cad091ef312df8a96bc
- E16 结案实现提交：eabfd03f6fe2b71b39df313f77c57ae4c6e286fa
- 本阶段分支：experiment/priority-tag-next-direction
- E16 Gate：CLIP_GROUP_LINE_CLOSED_UNADOPTED
- E15 Gate：CLIP_GROUP_BENCHMARK_INCONCLUSIVE，保持原样
- 方向状态：CHOSEN_A；用户于 2026-10-29（Asia/Shanghai）明确选择 A
- 已选 A：冻结低层的任务分配可行性诊断；零模型训练
- 未选 B：基于既有轨迹规范安全测量定义；本阶段不启动该替代研究
- 执行范围：完成第 3 节前置核查后，直接执行第 4–8 节 A 诊断，不执行第 9 节 B
- 方向性选择已经完成；常规实现、已登记检查及预算内诊断直接执行，不重复请求确认

## 1. 上一步结论及本阶段的区别

E16 已独立对账 E15，score_max_abs_gap=0，passing_seeds=0，正式 L0 哈希未变，没有采用 FULL_J/FULL_S。裁剪分组的本轮采用尝试已关闭，不能换名字、扩预算或变更指标继续追逐 PASS。

E14 的机制效应依然成立，E15 的性能证据依然不足。这不结束“payload 为主、小权重紧急服务”的研究目标。

结案报告给出两个真正不同的未回答问题：
1. 在低层冻结时，任务分配本身是否会改变实际 tagged 服务，并存在守住 payload/安全的局部机会？
2. 是否先将安全计数与投影接受的不同容差整理为明确、可复核的测量定义？

推荐 A，因为它直接检验任务层是否存在下一步可利用的控制空间，而不是再次微调已经结案的低层裁剪。A 不等于修改或训练正式高层，不承诺性能改善。

## 2. 先明确实验身份，避免把不同高层协议混为一谈

已核对源码：
- E15 FULL_J/FULL_S 评估沿用 QueueLookaheadWorker / ExperimentWorker；
- ExperimentWorker 在 pending<=0 时调用 sample_assignment；
- sample_assignment 从 build_pools(env, (("G1_RATE", 5),)) 得到的池中抽样；
- HOLD=15，常规决策时刻为 0、15、30、45；
- 正式 FormalPlanner 的 Scheme C 使用另一个 G1_RATE k=6 高层契约、不同入口和场景许可。

因此，E15 是既定抽样分配协议下的低层对照，不能称为正式 Scheme C 端到端系统已经失败。这是实验适用范围说明，不是对原注册低层对照的否定。

A 仅在原低层 benchmark 的 G1_RATE k=5 合法候选池中做有界任务分配反事实。它不是正式 Scheme C 的候选池实验，也不会把 TRAIN 场景强行送入不许可它们的正式 Planner 入口。

## 3. A 的前置身份与来源核查

- 读取并核对 E16 结案、E15 原始结果、正式基线 manifest 与当前源码。
- 形成 interface_identity.json，列明候选生成、任务抽样/评分、hold、低层动作、FIFO、账本、split guards 和正式 Scheme C 的区别。
- 不修改 active_policy.json、正式候选生成器、Scheme C/HGNN checkpoint、正式奖励或 SafetyProjector。
- 前置身份核查不运行新仿真。完成真实实现提交和必要测试后，按 A 的固定预算运行诊断；不训练、不读取保留集、不以新候选结果反向修改预登记。
- 用户已回复“A：诊断任务分配能否改善标签服务（推荐；8 个训练场景）”，选择记录位于 results/phase6i_priority_tag_assignment_feasibility/direction_decision.json。
- 本次只激活 A，不能同时启动 B；无需再次询问是否开始这项已登记诊断。

## 4. A 的研究问题与固定条件

A 检验：
> 使用相同冻结 L0、同一状态和同一合法候选池，仅改变本次 15 步任务关联，是否观察到既有 tagged 数据的服务改善机会，同时保持总 payload、安全及通信指标？

不训练任何模型，不创建新高层策略，不更新梯度或执行 optimizer.step。正式 L0 必须保持 SHA-256：
92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e。

固定环境、到达/合成标签协议、FIFO、动作空间、SafetyProjector 和低层观测/动作逻辑。不在低层添加新标签特征，不改变高层 hold，不给未服务数据奖励。

使用原 ResidualActor 的确定性动作；actor.eval()，不依赖 critic。不将该诊断命名为正式性能 benchmark 或部署策略。

## 5. A 的场景、快照与明确预算

仅用注册 TRAIN_POOL 的前 8 个场景：321000–321007；不得扩池挑有利场景，不访问 DEV、保留集或测试集。

参考轨迹：
- 每场景一条完整冻结 L0 轨迹，共 8 条；
- 环境 reset seed = scenario_id；
- 任务分配 RNG seed = 7000000 + scenario_id，复用已经核对的抽样协议；
- torch 线程 1、确定性算法开启；进程 seed 2 仅标记为历史复现上下文，不是新增模型种子；
- 记录模型、环境、任务分配和进程 RNG 各自角色。

每条参考轨迹仅在 t=0、15、30 的真实分配决策之前取快照，共 24 个。统一比较完整 15 步 hold，t=45 的尾部不足 15 步，不纳入快照比较。

候选仅来自该时刻原 G1_RATE k=5 池：
- 原抽样选择是基准 C0，记录其原池索引；
- 穷举同池其余唯一合法关联，含基准在内最多 5 个；
- 保留原生成顺序和去重映射；若池少于 5 个，不补新生成器或复制重复候选充数；
- 候选生成不使用未来到达、未来标签或未来服务结果。

每个快照：
- 每个唯一候选运行一次 15 步；
- 额外重复基准 C0 一次，用于 capture/restore 核对。

环境 step 上界：
- 8 条参考轨迹，默认 horizon=48 时共 384 步；
- 24 × 5 × 15，最多 1800 候选步；
- 24 × 15，共 360 个基准重复步；
- 合计最多 2544 步，无模型训练和影子参数更新。

若实际 horizon、HOLD、决策边界或池契约与上述不符，先报告 BLOCKED，不静默更换预算或场景。构造 fixture 测试与自然仿真分别计数，不以测试名义增加自然场景。

## 6. 同源任务分配干预与无泄漏要求

快照必须完整保存：环境状态及 RNG、队列、priority-tag ledger 及桶 ID/插入计数、option 的有无与全部执行字段、pending/remaining_hold、last_movement、分配 RNG 以及 Python/NumPy/Torch 进程状态。

每个候选从同一快照开始：
1. 按原抽样协议推进一次分配 RNG，保留同一个抽样后 RNG 状态；
2. 基准使用自然抽到的关联；其他分支只替换这一 hold 的关联，用原 make_option 建立合法任务；
3. 不额外抽样、不重置物理环境、不变更后续 RNG 规则；
4. 同一 hold 内关联固定 15 步，保持 TaskOption 和 remaining_hold 契约；
5. 各分支采用共同外生随机数，记录生成的到达与 tag 一致，允许实际接纳/丢弃因队列状态不同而正常分叉。

基准第一次反事实回放必须匹配参考轨迹对应的 15 步；重复 C0 再核对动作、位置、payload、队列、账本和 RNG。比特采用原 1 bit 核对容差、路径 1e-9 m；桶级分解沿用 1e-6 bit。关键不匹配时停止该快照的解释并报告。

未来到达仅能由仿真环境按时间自然产生，不可进入决策时刻的候选生成、可见特征或选择规则。最终回放结果可以做事后诊断，但必须标记 hindsight_diagnostic，不能把事后最佳候选称为可部署的无未来信息高层。

## 7. 区分当前已知标签与未来生成标签

在每个快照建立已接纳桶的稳定 ID 清单，记录当前 Q、tagged bits、队首信息和 FIFO 阻挡；只包括 admission_step<=t 的有效桶。

旁路跟踪该 hold 实际服务的：
- 快照时已存在的 tagged/ordinary bits；
- 快照之后新接纳的 tagged/ordinary bits；
- 总 payload、丢弃、最终队列与年龄指标。

使用实际 FIFO 服务分配，不能把队列减少、丢弃或 ledger snapping 算作成功传输。跟踪包装不得改变原环境转移或桶顺序。

初始 reset 桶按既有协议为 ordinary；t=0 的“快照已知 tagged 服务”应为零，是负对照。若这一核对失败，先查归因与信息泄漏，不发布机会结论。

这一步特别重要：未来才生成的标签服务增量不等于当前标签信息可以支持的选取收益。两类分别报告，不以最终 tagged 总量替代“当时已知”的证据。

## 8. A 的统计、判定和交付

每个候选相对同快照基准，报告：
- 总 payload、全部 tagged、快照已知 tagged 的服务增量；
- ordinary 服务、年龄加权服务、按标签丢弃/最终队列；
- SINR、干扰、投影和路径；
- 原始安全违规与约束失败、最小距离；
- 候选合法性、参考匹配、共同随机数与当前可见信息。

分层展示机会，不改原指标：
- S0 服务机会：快照已知 tagged 增量 >1 bit，payload 增量 >=-1 bit；
- S1 加安全条件：S0 且原始安全/约束计数均不增加；
- S2 加通信条件：S1 且平均 SINR 不下降、平均干扰不增加。

另列“全部 tagged 增量”的同类描述，但不混入快照已知标签机会。以不同场景和不同快照为层次报告，不把同池候选或 UAV 步当成独立场景。

posthoc_best 和 Pareto 集只代表注册有限候选集合内的诊断结果；不是全局上界，不是政策收益或长时域可控性证明。不计算新的性能 PASS，也不训练排序器。

Gate：
- ASSIGNMENT_FEASIBILITY_COMPLETE：登记仿真、来源、合法性、基准匹配和归因检查完成；
- ASSIGNMENT_FEASIBILITY_BLOCKED：必要核对、契约或数据完整性失败。

opportunity_evidence：
- REPLICATED_LOCAL_OPPORTUNITY：至少两个不同 TRAIN 场景观察到 S2 候选；
- LOCAL_OR_CONSTRAINED_OPPORTUNITY：有服务机会，但场景覆盖不足或安全/通信条件筛除；
- NOT_OBSERVED_IN_REGISTERED_POOL：有效登记样本内没有观察到 S0；
- INSUFFICIENT_COVERAGE：有效参考/候选不足，不能作前述描述。

这些标签是预登记的描述性分类，不是统计显著性，不证明正式 Scheme C 有收益。没有观察到机会也不能证明任务分配在所有场景/时域都无效。

交付：
- docs/research/HTW_UrgencyAware_Assignment_Feasibility_Report_CN.md；
- 独立 phase6i_priority_tag_assignment_feasibility 代码/协议及必要测试；
- results/phase6i_priority_tag_assignment_feasibility/ 下的 summary/schema、interface_identity、baseline_rows、candidate_rows、snapshot_manifest、归因明细、完整 seed/source/runtime/哈希与检查日志；
- 明确 release_eligible=false、merged=false、training_updates=0。

先提交真实实现再运行。相关测试必须通过；新增代码按仓库规则执行完整测试、与同一来源干净基线核对，既有失败保留为发布阻塞。无需再请求常规实现审批。

## 9. B 的替代方案：仅规范安全测量定义

若用户选择 B，不执行第 4–8 节的新仿真。

仅使用已保存的 E15/E16 轨迹/事件和固定源码，形成 safety_measurement_spec.json 与中文说明，分别定义：
- 距离阈值、单位、dtype；
- 原执行计数器：distance+1e-9<1 m；
- 原投影接受：全局最小预测距离+1e-6>=1 m；
- 位于两套规则之间的数值边界分类；
- 执行动作、被拒绝的投影候选、全局最小值与指定 UAV 对的量的区别；
- 无数据时的 unknown，不补造 E14 位置或 E15 场景/时间。

对 41 条既有 E15 事件独立核对分类与原始 40/1 计数，不改变阈值、不重标 E15 Gate、不修正式 SafetyProjector。新增内容只是测量契约及验证，不接入正式产品流程。

交付独立文档和 results/phase6i_priority_tag_safety_measurement/ 产物，Gate 为 SAFETY_MEASUREMENT_SPEC_COMPLETE 或 SAFETY_MEASUREMENT_SPEC_BLOCKED。明确 B 不提供新的 tagged 服务改善证据。

## 10. 共同停止规则与方向记录

- direction_status=CHOSEN_A，依据为本聊天中用户于 2026-10-29 的明确答复；不是诊断已经完成的结果记录。
- 已记录用户原话、日期、A 的范围和固定预算；按 A 执行。普通技术参数不再逐项请求确认。
- 正式 Planner/Scheme C、奖励、SafetyProjector、L0–L4/FULL/pilot 模型及 E0–E16 历史保持冻结。
- 不访问新的 DEV/保留/测试数据，不修改 split guards，不执行训练或模型参数更新。
- 本阶段无论选择 A 或 B，都不采用新策略、不合并、不追加预算。完成后最多给一个有证据的新阶段建议，然后停止。

本方案是新的研究层级/目标选择，不重启已经结案的裁剪采用尝试。
