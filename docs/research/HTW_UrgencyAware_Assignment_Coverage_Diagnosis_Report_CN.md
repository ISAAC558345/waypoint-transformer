# 活动掩码匹配的任务分配覆盖诊断

E17 已经完成，scientific_gate 仍是 `ASSIGNMENT_FEASIBILITY_COMPLETE`，机会结论仍是 `NOT_OBSERVED_IN_REGISTERED_POOL`。该结论只说明当时登记的前缀截断集合里没有观察到已知标签服务机会，不改写成执行失败，也不表示整个 G1_RATE k=5 池、高层任务分配或正式 Scheme C 无效。旧规则的问题是参与规模：非基准候选落在联合池生成顺序的最前面，表现为空闲或单 UAV。本阶段唯一的方法改变，是在自然基准之外，按原池索引把活跃掩码逐 UAV 相同的合法关联分成最多四层，每层抽一条，作为替代候选。有效覆盖为 24 / 24 个状态至少有一条同掩码替代，实际抽到的替代条数是 96，同掩码唯一替代总数是 2696。实际步数是 2544。机会类别是 `REPLICATED_LOCAL_OPPORTUNITY`。这个分层样本不是联合池的穷尽搜索，不是全局上界，也不是正式 Scheme C 的性能结论。训练更新为 0，参数更新为 0，release_eligible 为 false，merged 为 false。

- E15 scientific_gate: `CLIP_GROUP_BENCHMARK_INCONCLUSIVE`
- E16 scientific_gate: `CLIP_GROUP_LINE_CLOSED_UNADOPTED`
- E17 scientific_gate: `ASSIGNMENT_FEASIBILITY_COMPLETE`
- E17 opportunity_evidence: `NOT_OBSERVED_IN_REGISTERED_POOL`
- 本阶段 scientific_gate: `ASSIGNMENT_COVERAGE_DIAG_COMPLETE`
- opportunity_evidence: `REPLICATED_LOCAL_OPPORTUNITY`
- opportunity_published: true
- release_eligible: false
- merged: false
- training_updates: 0
- parameter_updates: 0

旧前缀和新样本只比较选择覆盖和本次登记的诊断结论。后出现的服务差不能解读成新的高层已经提升了性能。

## 预算与核对

- 参考轨迹步数: 384
- 候选回放步数: 1800
- 基准重复步数: 360
- 总步数: 2544
- 选择清单哈希: `34f8b1069b35d0812f3dcb280b883406f87136d49f74b58420718f0fc7650f4d`
- 正式低层 SHA-256: `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`
- t=0 负对照通过: true

## 机会计数

- S0: 11
- S1: 11
- S2: 5
- S0 场景: [321001, 321002, 321004, 321005, 321006]
- S2 场景: [321002, 321006]

## 测量到的服务差

- 达到 S0 的替代有 11 条，已知标签服务差最小 1000000.0 bit，最大 2000000.0 bit。
- 这些行的决策时刻是 [15]。全部 S0 行的安全差和约束差都没有上升，所以 S1 与 S0 计数相同。
- S2 只保留其中均值 SINR 不下降且均值干扰不上升的行。
- 回放安全或约束事件行数: 0。约束失败仍保留原始计数，没有被记成安全机会。

## 限制

同掩码抽样会漏掉更换参与 UAV 才能出现的收益。k 仍是 5，场景仍是 321000–321007，决策时刻仍是 0、15、30，t=45 尾部没有进入比较。
事后最优只描述已经回放的样本，标记为 hindsight_diagnostic，不是可部署高层。

## 下一步

至少两个训练场景在同掩码分层样本里出现了同时守住 payload、原始安全和通信条件的已知标签服务差。若要检验一条固定规则，必须另开预登记，并且仍然冻结低层、不使用事后最优候选。本阶段不启动该对照。

本阶段到此停止，不扩大 k、场景、候选或预算，不训练，不修改正式系统，不重开裁剪线。

## 测试

- 干净基线测试数: 1805
- 干净基线加上本阶段代码后的测试数: 1813
- 新增失败: 0
- 相关测试通过: true
- 历史失败仍是发布阻塞项。完整历史套件不记为全绿。
