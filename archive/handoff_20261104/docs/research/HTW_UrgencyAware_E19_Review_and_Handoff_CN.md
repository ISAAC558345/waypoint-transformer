# E19 固定标签规则审阅与研究交接

审阅日期：2026-11-04。结果来源为 `experiment/priority-tag-assignment-rule` 的 `f9e9a9e007d7d84046232181fdab6f9a732a4a05`。当前固定规则诊断完成，不采纳该配置；下一项工作沿用用户已确定的海洋观测方向，执行已登记 M0 核算。

## 1 结果与解释

E19 在 TRAIN 321008–321023 的 16 个自然参考 t=15 状态比较三种选择：N 为自然分配，M 最大化 payload 代理 P，T 最大化 P+0.001K。K 是当前已知 FIFO 前缀中的标签代理。窗口仅为 [15,30)，不是整回合新策略比较。

| 比较与指标 | 场景平均差 Mbit | 配对 bootstrap 95% 区间 Mbit |
| --- | --- | --- |
| T−M 已知标签服务 | 0 | [0, 0] |
| T−M payload | 0 | [0, 0] |
| T−N 已知标签服务 | -0.036604 | [-0.562500, 0.489292] |
| T−N payload | 0.040058 | [-1.085928, 1.066138] |

16/16 状态 T 与 M 选择相同，T 的逻辑结果复用 M，不能将两者作为独立样本。T−N 的 S0/S1/S2 为 2/2/1，M−N 完全相同，因此两例局部机会不能归因于标签项。payload 相对 N 在 1 bit 阈值下为 9 正、1 平、6 负，最大已测负差为场景 321017 的 -6 Mbit。全部已发表策略行的原始安全及约束计数为零；这只描述本样本。

保留原结论：
- scientific_gate=FIXED_TAG_RULE_DIAG_COMPLETE。
- mechanism_evidence=NO_TAG_DECISION_EFFECT。
- performance_pass=false、adopted=false、release_eligible=false、merged=false。
- E18 的局部机会结论仍有效；E19 不否定标签的潜在作用，也不证明整个任务分配方向无效。

## 2 只读复核

本次从已发表 CSV 重算 16 个场景的均值、方向和全部 1636 条静态候选的 M/T 选中索引，与选择清单和报告一致。没有新环境 step、策略模型推理或训练。

6 个状态的全部候选 K=0；1 个状态的 payload 最优选择同时已经达到候选最大 K；其余 9 个状态存在更大 K 的候选，但本次 0.001 加分没有翻转 P 的排序。此处的 K 是代理值，不是已实现的标签服务，也不等同于没有实际标签。该结构解释支持固定规则在本样本未激活的判断，不授权尝试其他权重。

报告与原始对账表记录 16/16 参考、重复和共同外生事件核对通过，正式模型 SHA-256 为 `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。相关测试通过；干净基线 1813 项与新增后 1829 项均有 206 个失败签名，新增失败为零。历史失败仍是发布阻断，不能称全量套件通过。

## 3 回放中止和预算

第一次回放未开启确定性算法，动作误差约 1e-8，超过既有动作核对标准；执行端在机制分类前中止。修复只恢复确定性运行条件，随后读取相同冻结选择清单，未换规则、场景或权重。

| 已披露运行 | steps |
| --- | --- |
| 正式自然参考 | 768 |
| 正式候选回放 | 480 |
| 正式 N 重复 | 240 |
| 正式合计 | 1488 |
| 作废候选与重复 | 15+15=30 |
| 已披露实际合计 | 1518 |
| 原登记上限 | 1728 |

原 summary.budget.total_steps=1488 描述正式回放，作废 30 步由原报告文字披露。本交接同时列出两者，不改写历史结果，不把节省步数转成新运行额度。

## 4 来源字段辨认

- 实现提交：`c5acf6bf3867bdc394fbad2df5af498b0ccb465e`。
- 回放前选择清单提交：`50a3b143db8d925cc3b743cbd4378f9c5ffab153`。
- 确定性运行条件修复提交：`ab54f268bc9a51dc3b915f6a8d02402e6ee3ef86`。
- 选择清单 SHA-256：`ae857bef729b351771c23ffda198f3ddaed306dd159a7f9f2250c9dd2f6d0674`。

summary 中旧字段 selection_commit 指向 ab54；明确的 selection_manifest_commit 指向 50a，replay_context_commit 指向 ab54。审阅和后续引用使用明确字段，不把运行条件修复误认为选择清单首次冻结时间。此项是溯源字段说明，不据此将有效机制结果判为失败。

## 5 下一步

当前停止的是 E19 这条固定 P+0.001K 规则的继续投入。不增加权重、不扫描候选、不进入整回合扩展或训练；未来重开标签研究仍需新的科学问题和登记。

下一项执行 [海洋观测 M0 交接指令](Ocean_Observation_M0_Next_Instructions_CN.md)。依据此前用户选择，先核算观测产品、通信、航程、能量与可用信息。E19 没有提供真实海况观测或信息价值收益的证据，M0 要建立的是新的可解释任务基础。

截至本次读取，仓库分支列表未发现海洋 M0 结果分支；这不排除执行端存在本地未推送成果。执行前先检查已有工作，完成的成果优先核验与提交，避免重复核算。

## 6 固定来源

- [E19 原报告](https://github.com/ISAAC558345/waypoint-transformer/blob/f9e9a9e007d7d84046232181fdab6f9a732a4a05/docs/research/HTW_UrgencyAware_Fixed_Tag_Rule_Diagnosis_Report_CN.md)
- [E19 原 summary](https://github.com/ISAAC558345/waypoint-transformer/blob/f9e9a9e007d7d84046232181fdab6f9a732a4a05/results/phase6i_priority_tag_assignment_rule/summary.json)
- [E19 配对原始表](https://github.com/ISAAC558345/waypoint-transformer/blob/f9e9a9e007d7d84046232181fdab6f9a732a4a05/results/phase6i_priority_tag_assignment_rule/paired_differences.csv)
- [E19 静态候选表](https://github.com/ISAAC558345/waypoint-transformer/blob/f9e9a9e007d7d84046232181fdab6f9a732a4a05/results/phase6i_priority_tag_assignment_rule/static_scores.csv)
- [E19 对账](https://github.com/ISAAC558345/waypoint-transformer/blob/f9e9a9e007d7d84046232181fdab6f9a732a4a05/results/phase6i_priority_tag_assignment_rule/baseline_reconciliation.json)
- [E19 测试对比](https://github.com/ISAAC558345/waypoint-transformer/blob/f9e9a9e007d7d84046232181fdab6f9a732a4a05/results/phase6i_priority_tag_assignment_rule/test_comparison.json)
- [M0 已登记范围](Ocean_Observation_M0_Contract.json)
