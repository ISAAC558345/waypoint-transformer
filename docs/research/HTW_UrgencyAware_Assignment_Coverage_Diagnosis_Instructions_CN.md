# Stage E18：任务分配候选覆盖审计与等参与集合诊断

- 仓库：ISAAC558345/waypoint-transformer
- 固定来源提交：87509424b6b2119b422146ff0a64ec98c0d2656b
- E17 结果提交：3465a926512fd0bdb485455ee73240ddd010cd1f
- E17 实现提交：d940484ca55411430e727b8ac33b891f6f60fc3d
- 本阶段分支：experiment/priority-tag-assignment-coverage
- 已有用户方向：A，冻结低层的任务分配可行性诊断，继续有效
- 本阶段：改进候选可比性，不换研究方向；常规实现和本预算内诊断直接完成，无需再次请求选择 A
- 新增训练/参数更新：0
- 独立环境 step 上限：2544，不能记作 E17 的剩余预算
- 正式 Planner、Scheme C、L0、奖励、FIFO、SafetyProjector 和全部历史结果保持冻结

## 1. E17 结果已经收到，不能重复旧续办指令

E17 已完成 2544 个环境 steps、24 个快照、120 行候选评估：
- scientific_gate=ASSIGNMENT_FEASIBILITY_COMPLETE；
- opportunity_evidence=NOT_OBSERVED_IN_REGISTERED_POOL；
- S0/S1/S2 均为零，无模型训练、无合并。

8750942 已保留完成结果并说明此前执行交接清单早于结果可见。不要再凭旧 handoff 重做 E17。E18 是一项独立预登记的方法修正，不能覆盖或回改 E17 的完成状态、预算和零机会结果。

## 2. 已经核实的覆盖问题

从 E17 原始 candidate_rows.csv 独立核对：
- 24 个自然基准分配，活跃 UAV 数为 2–5；
- 96 个非基准分配中，24 个全空闲、72 个只有一架 UAV；
- 所有非基准候选都来自联合池索引 0、1、2、3；
- 自然基准的池索引范围为 260–5444。

select_registered 按原生成顺序取前四个非基准并保留自然基准，确实执行了旧截断规定。问题是该规定没有控制参与服务的 UAV 集合；不能把这一点归咎为执行端没有遵守协议。

build_pools 的 k=5 是每个 UAV 的 top-k 目标数，之后 enumerate_legal 形成联合关联池，不是只有五个联合分配。生成顺序前缀包含全空闲和单 UAV 关联，因此“前四条加自然基准”不能代表同规模联合分配的服务机会。

E17 的正确范围仍是“在其已登记的前缀截断集合内未观察到机会”。不能据此断言整个 G1_RATE k=5 联合池、高层分配或正式 Scheme C 无效。本阶段不更改 E17 Gate，也不将零机会结果改写成执行失败。

## 3. 唯一方法改动与冻结条件

继续使用同一个冻结 L0、相同 8 个 TRAIN 场景、相同 24 个决策前状态、相同 G1_RATE k=5 联合池、相同 HOLD=15、相同奖励与安全计数。

唯一方法改动：
> 在自然基准之外，只从活跃 UAV 掩码与基准完全相同的合法联合分配中取最多四个替代候选，并按池索引分层抽样，替代旧前缀截断。

active_mask[u] = assignment[u] >= 0。要求逐 UAV 相同，不只是活跃总数相同。保持谁参与服务不变，研究这些 UAV 的目标选择。

这会限制可解释范围，可能遗漏通过改变参与 UAV 获得的收益；本轮接受该限制，不再额外加入其他候选族或放宽掩码条件。

不扩 k，不增候选生成器，不用 tag 奖励排序选候选，不引入未来服务/到达信息，不训练排序器、不修改正式高层。

## 4. 阶段 A：覆盖审计与参考重建

先读取 E17 的基准表、候选表、snapshot_manifest、源码和来源清单，核对上述参与数分布和池索引。

E17 的公开 snapshot_manifest 保存的是摘要、池哈希及去重映射，不是 session.capture 的完整环境/账本/option/RNG 状态。不得直接把它称为可恢复快照。

因此本阶段固定重建参考轨迹：
- 场景为 TRAIN 321000–321007；
- 冻结 L0 确定性动作，8 条完整参考轨迹，共最多 384 steps；
- 决策前时刻 t=0、15、30，共 24 个完整快照；
- 环境 seed、assignment seed=7000000+scenario_id、历史进程上下文 seed 2、单线程与确定性算法均与 E17 一致；
- 核对参考 episode 汇总、快照时刻、基准分配、池哈希、队列与已知标签摘要，必须与 E17 保存记录对齐；
- 无法对齐时报告 BLOCKED，不能放宽容差、另挑场景或先跑新候选。

每个快照从原源码重建完整联合池，生成只读覆盖清单：
- per-UAV top-k 的真实含义、完整池大小、唯一池大小与合法性；
- 活跃 UAV 数分布及活跃掩码分布；
- 基准掩码、旧截断候选掩码；
- 与基准完全同掩码的唯一候选数；
- 只依据当前已入队数据可见的 tagged 目标覆盖，不读取新候选回放结果。

保存这次完整 capture 数据及序列化/恢复说明，不能再次只保存摘要。旧 E17 文件不修改。

## 5. 固定的新候选选取规则

对每个已核对快照：

1. 按原生成顺序对完整池按关联数组去重，保留原池索引和映射。
2. 保留自然抽样的基准 C0。筛出与 C0 具有完全相同 active_mask 的其他唯一关联，按其原池索引递增排序。
3. 设合格替代数为 N：
   - N=0：只有基准，标记没有同掩码替代，不伪造候选；
   - 1<=N<=4：全部保留；
   - N>4：按 numpy.array_split 将索引序列分成四个连续、非空且大小尽量相等的层。
4. 为该快照单独建立 numpy.default_rng：
   candidate_seed = 9300000 + 100*scenario_id + decision_time。
   每层均匀抽取一条，最多四个替代。
5. 将 C0 与已选替代按原池索引排序输出；baseline 标记保持正确。
6. 候选 RNG 与环境、tag、分配和进程 RNG 独立，不消耗原分配流，不根据 tagged 服务结果换种子或重抽。

先提交真实实现和注册配置，完成 24 个状态的选择清单及哈希后，再进行任何新替代候选回放。若使用两个执行入口，应把候选清单作为固定输入，禁止根据先看到的结果重选。

需要验证：
- 全部候选属于同一个旧联合池；
- 每个替代的 active_mask 与 C0 逐项相同；
- 唯一且不超过总数 5；
- 相同快照与 candidate_seed 重复得到相同清单；
- 在合格替代不足时不放宽掩码、不增 k、不补重复项；
- 选择函数不访问未来到达、未来 tag 或仿真收益。

## 6. 阶段 B：预算与同源回放

固定预算：
- 参考重建最多 384 steps；
- 24 快照 × 最多 5 候选 × 15 步，最多 1800 steps；
- 每快照额外重复基准一次，共最多 360 steps；
- 总计最多 2544 steps。合格替代不足时实际用量减少，不能把节省预算转成新场景或新候选。
- 模型训练、梯度计算、参数更新及影子 Adam steps 全为零。

恢复完整环境、队列账本、option、pending、last_movement、所有相关 RNG 后，每个分支仍按原协议推进一次分配抽样，使抽样后的 RNG 状态一致，再安装相应候选。

同一 hold 固定关联 15 步；共同外生到达/tag、FIFO、实际服务奖励和 SafetyProjector 均保持不变。基准回放与参考窗口一致，并通过额外重复回放。

沿用 E17 核对容差：比特 1 bit、路径 1e-9 m、桶级分解 1e-6 bit，不因匹配失败提高容差。

捕获到约束失败或其他异常时保留原动作回退和原始计数，不把失败候选包装成安全可行机会。源码/账本/恢复契约失效则 BLOCKED；不得通过重跑和换 seed 消除异常。

## 7. 归因与判定不变

继续区分快照时已经接纳的 tagged/ordinary 服务和此后才接纳的 tagged/ordinary 服务。以稳定桶 ID 跟踪实际 FIFO 服务，丢弃和 snapping 不算成功传输。

t=0 重置桶按原约定为 ordinary，其已知 tagged 服务必须为零。未来才生成的标签不能进入候选选择，也不能混进已知 tagged 机会指标。

仍采用 E17 的三层诊断：
- S0：快照已知 tagged 服务增量 >1 bit，payload 增量 >=-1 bit；
- S1：S0 且原始安全/约束计数均不增加；
- S2：S1 且平均 SINR 不下降、平均干扰不增加。

全部 tagged 的结果另作描述，不替换已知 tagged 目标。报告层次为场景、快照、候选，不将同快照的多个候选当独立场景，不计算新的性能 PASS。

本轮 Gate：
- ASSIGNMENT_COVERAGE_DIAG_COMPLETE；
- ASSIGNMENT_COVERAGE_DIAG_BLOCKED。

opportunity_evidence：
- REPLICATED_LOCAL_OPPORTUNITY：至少两个不同场景观察到 S2；
- LOCAL_OR_CONSTRAINED_OPPORTUNITY：有 S0，但场景覆盖不足或安全/通信条件筛除；
- NOT_OBSERVED_UNDER_MATCHED_MASK：24 个参考快照均核对有效、每个至少有一个有效同掩码替代，且所有已登记替代中没有观察到 S0；
- INSUFFICIENT_MATCHED_ALTERNATIVES：未达到上述覆盖，且没有足够证据归入前两个机会类别；不能以没有候选冒充没有机会。

机会分类按上述顺序判定；参考/恢复等必要完整性失败仍先给 BLOCKED，不以机会分类掩盖。

登记诊断完成不是策略更好。同掩码的有限分层样本不是整个联合池的穷举，不给全局上界，也不外推为正式 Scheme C 性能结论。

## 8. 交付物与方法限制

- docs/research/HTW_UrgencyAware_Assignment_Coverage_Diagnosis_Report_CN.md；
- 独立 src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/ 实现、配置及必要测试；
- results/phase6i_priority_tag_assignment_coverage/ 下：
  - summary.json、schema.json、runtime/source/seed manifest；
  - prefix_coverage_audit.csv、pool_coverage.csv；
  - candidate_selection_manifest.json（回放前固定）；
  - baseline_reconciliation.json、baseline_rows.csv、candidate_rows.csv；
  - 已知/未来桶级服务明细；
  - 24 个完整 snapshot 的数据、哈希和恢复核对清单；
  - 原始安全事件、预算实际消耗、测试日志、基线对照和 evidence_gaps.json。

对旧前缀与新方法，只比较其选择覆盖和独立登记的诊断结果，不重跑旧 96 个前缀替代，也不改旧 summary。避免将方法修正后的机会直接归因成“新高层已提升性能”。

相关测试必须通过；新增代码按仓库规则执行完整测试，与相同来源的干净基线逐项核对。既有失败继续是发布阻塞，不能称为全量通过。来源哈希对比需区分 Git 内容与 Windows checkout 换行差异，不能把换行规范化误报为正式代码漂移。

E18 报告开头明确：
- E17 实际完成且原 Gate 保持；
- 旧前缀的参与规模偏差及本轮唯一改动；
- 新方法的有效覆盖、实际 step 数、机会分类及限制；
- 零训练、零参数更新、release_eligible=false、merged=false。

完成后停止，最多建议一个有证据的下一阶段。不得扩大 k、增加场景/候选/预算、转入模型训练、修改正式系统或重启裁剪路线。候选方法修正沿用用户已选 A，无需再次要求同一研究方向确认。
