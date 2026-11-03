# 固定标签规则与屏蔽对照诊断

本阶段检验一条预先固定的小权重标签规则，训练更新为 0，参数更新为 0。屏蔽对照使用同一候选集合、同一数值运算和同一稳定排序，只把分数里的标签项去掉；回放环境仍使用真实合成标签。场景是训练集 321008–321023，共 16 个，它们属于训练池，不是项目级的未见保留集。每个场景只在自然参考的 t=15 保存一次可恢复快照，主比较窗口只有 [15, 30)。只看 t=15 是明确的开发选择，因为 E18 的全部 S0 行都出现在 t=15；本阶段不检验 t=0、t=30、t=45，也不评估一条新规则的完整闭环回合。步数上限是 1728，旧阶段剩余的步数不能挪用。这不是正式 Scheme C，不是新高层策略的完整闭环，也不是 E17 或 E18 的改写。E17 的 scientific_gate 仍是 `ASSIGNMENT_FEASIBILITY_COMPLETE`，机会结论仍是 `NOT_OBSERVED_IN_REGISTERED_POOL`。E18 的 scientific_gate 仍是 `ASSIGNMENT_COVERAGE_DIAG_COMPLETE`，机会结论仍是 `REPLICATED_LOCAL_OPPORTUNITY`。正式回放的步数是 1488，其中参考 768、候选 480、重复 240。第一次回放没有打开确定性算法，动作差约 1e-8，在进入机制分类前停止。那次额外使用了 15 个候选步和 15 个重复步，没有被当成第二个样本。正式回放仍读取同一份已提交的选择清单。把这 30 个作废步与正式回放相加，总量仍低于 1728。标签规则与屏蔽对照选择不同的场景数是 0。机制证据是 `NO_TAG_DECISION_EFFECT`。release_eligible 为 false，merged 为 false，adopted 为 false，performance_pass 为 false。

- E15 scientific_gate: `CLIP_GROUP_BENCHMARK_INCONCLUSIVE`
- E16 scientific_gate: `CLIP_GROUP_LINE_CLOSED_UNADOPTED`
- E17 scientific_gate: `ASSIGNMENT_FEASIBILITY_COMPLETE`
- E17 opportunity_evidence: `NOT_OBSERVED_IN_REGISTERED_POOL`
- E18 scientific_gate: `ASSIGNMENT_COVERAGE_DIAG_COMPLETE`
- E18 opportunity_evidence: `REPLICATED_LOCAL_OPPORTUNITY`
- 本阶段 scientific_gate: `FIXED_TAG_RULE_DIAG_COMPLETE`
- mechanism_evidence: `NO_TAG_DECISION_EFFECT`
- mechanism_published: true
- performance_pass: false
- release_eligible: false
- merged: false
- adopted: false
- 训练更新: 0
- 参数更新: 0
- 选择不同的场景数: 0
- 复用了屏蔽对照回放的场景数: 0
- 复用了标签规则回放的场景数: 16
- T−M 的 S0/S1/S2: 0/0/0
- T−N 的 S0/S1/S2: 2/2/1
- M−N 的 S0/S1/S2: 2/2/1
- T−M 已知标签服务差: 均值 0，中位数 0，95% 区间 [0, 0]，区间包含 0
- T−M payload 差: 均值 0，中位数 0，95% 区间 [0, 0]，区间包含 0
- T−N 已知标签服务差: 均值 -36603.9，中位数 0，95% 区间 [-562500, 489292]，区间包含 0
- T−N payload 差: 均值 40057.6，中位数 372319，95% 区间 [-1.08593e+06, 1.06614e+06]，区间包含 0
- M−N 已知标签服务差: 均值 -36603.9，中位数 0，95% 区间 [-562500, 489292]，区间包含 0
- M−N payload 差: 均值 40057.6，中位数 372319，95% 区间 [-1.08593e+06, 1.06614e+06]，区间包含 0
- 总体增益仍不确定: true
- 正式低层检查点 SHA-256: `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`
- 选择清单 SHA-256: `ae857bef729b351771c23ffda198f3ddaed306dd159a7f9f2250c9dd2f6d0674`
- 实现提交: `c5acf6bf3867bdc394fbad2df5af498b0ccb465e`
- 选择清单提交: `50a3b143db8d925cc3b743cbd4378f9c5ffab153`
- 回放代码提交: `ab54f268bc9a51dc3b915f6a8d02402e6ee3ef86`

## 规则

- N 是快照上的自然抽样 C0。
- M 在与 C0 逐 UAV 活跃掩码相同的去重候选上最大化 payload 代理 P。
- T 在同一集合上最大化 P + 0.001 K。K 是当前 FIFO 队列里、容量前缀落在 priority_tag=1 的部分。
- 普通桶排在前面时会先占用容量。空队列的 P 和 K 都是 0。空闲 UAV 两项都贡献 0。
- 分数完全相同时取原池最小索引。不用观测到的服务打破平局，也不加随机抖动。
- P 是当前速率扣掉几何 ETA 后的有限窗代理，并被当前队列截断。它不是实际传输，也不是联合干扰预测。
- M 只在打分时不使用标签。M 的回放环境不会把标签清零。
- 选择器不读取 scenario_id、未来到达、E18 事后最优索引或已经实现的服务。

## 未通过或保留的核对

- 没有额外的机制核对失败项。
- 没有完整性问题。

## 下一步

这 16 个状态上，0.001 的标签项没有改变任何选择。这只说明这条固定代理和这个权重在本样本里没有决策差，不说明标签没有用。不要在本阶段加大权重、增大 k 或改去 DEV/测试场景。本阶段到此停止。

## 停止

本阶段到此停止。不搜索其他权重，不增大 k，不接触 DEV 或测试场景，不启动 HGNN 或强化学习，不修改正式装载器，也不合并。

## 测试

- 干净基线 Ran 1813，失败签名 206。
- 叠加本阶段后 Ran 1829，失败签名 206。
- 新增失败 0。历史失败仍然是发布阻断，全量套件不是绿灯。
- 本阶段相关测试通过: true。
