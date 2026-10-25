# P3：保持在服务中的 UAV 关联、仅重分配未进入服务半径的 UAV

- 前置报告：`docs/research/HTW_NextStep_T45Counterfactual_CN.md`
- 前置 Gate：`T45_COUNTERFACTUAL_INCONCLUSIVE`
- 本阶段 Gate：`T45_ACTIVE_ASSIGNMENT_PRESERVED_COMPLETE` / `T45_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE` / `T45_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED`
- 性质：同状态、末 3 步的受控反事实评估；不训练、不更改正式策略或模型，不读取独立保留测试。

## 1. 前一阶段结果与动机

P2 在验证场景 322048–322063 的 t=45 同状态上比较 Scheme C 与距离导向合法 assignment。16 个配对场景的 payload 差值（距离规则减 Scheme C）均值为 -60,776.989 bit，95% bootstrap CI 为 [-204,861.511, 22,530.546] bit，Gate 为 `T45_COUNTERFACTUAL_INCONCLUSIVE`。

14 个场景的 assignment 没有改变；只有 2 个场景实际改变：
- 322048：payload 增加 120,162.910 bit。
- 322061：payload 减少 1,092,594.726 bit。该距离规则把一架原本仍在服务的 UAV 改成 idle，说明仅优化几何可达数和总距离可能损失正在进行的有效服务。

因此，P3 只测试一个更保守的假设：**在最后 3 步，只允许调整尚未进入服务半径且当前未交付有效 payload 的 UAV；保持已在服务半径内并正在交付的 UAV 目标不变。** 不把它视为已经证明有效的策略。

## 2. 冻结与数据边界

- 使用验证场景 322048–322063，仅用于此项预先定义的诊断实验。
- 不读取 reserved/independent holdout 322064–322079。
- 高层 Scheme C checkpoint、matching decoder、候选配置、低层 `phase6g_seed0.pt`、SafetyProjector、奖励和动作都冻结。
- 不训练，不改正式 Planner，不覆盖 P1/P2 结果。
- 输出到新目录：`results/phase6i_t45_active_preserve/`。

## 3. 预注册的 assignment 规则

对每个场景捕获 t=45 的完整同一状态，并重建相同的 13,327 个合法候选池。

### 固定集合
- 固定所有当前目标在服务半径内、且当前/上一窗口有真实交付证据的 UAV 的 assignment。
- 固定当前已经处于服务半径内的 UAV assignment，即使其在最后 3 步的交付暂时为 0，也不轻易重新分配。
- 固定 idle UAV 的 idle 状态，除非代码审计证明其原本没有被分配目标且本规则明确允许其参与候选。为避免额外混杂，本阶段默认不把 idle UAV 变成 active。
- 只有当前被分配到非 idle 目标、但 t=45 时在服务半径外的 UAV 可以重新选择目标。

注：如果当前目标在半径外的 UAV 在剩余 3 步内几何上不可能进入半径，才将其列为“待重新分配 UAV”。判断式沿用环境单位：
`max(distance_m - service_radius_m, 0) <= remaining_steps * step_length_m`。
该式只是几何可达代理，不保证在投影/其他运动约束下必然可达。

### 候选选择
1. 只从同一状态正式合法候选池中选择，不能构造候选池外的 assignment。
2. 候选必须保留固定集合中的所有 UAV 目标，并满足浮标不重复等合法性约束。
3. 在剩余可重分配 UAV 中，优先最大化几何上可能进入服务半径的数量。
4. 若并列，最小化这些待重分配 UAV 到新目标的总距离。
5. 若仍并列，用候选池原始稳定顺序打破平局。
6. 选择过程中不能查看 3 步后的 payload、served bits 或反事实结果；不得用结果挑选 assignment。
7. 若不存在满足固定集合的合法候选，标记 `no_legal_preserving_candidate`，该场景不进行替换，仍记录原始 Scheme C 结果。

此规则不改变正式 Scheme C，只生成一条诊断对照轨迹。

## 4. 同状态配对执行

每个场景运行两臂：
- Arm S：原始 Scheme C assignment。
- Arm P：保留 active/in-range UAV 的受约束 assignment。

两臂从同一状态快照、同一 last movement、相同随机/到达状态出发，使用冻结 MAPPO-V2 和相同 SafetyProjector，各执行 3 步。

必须记录：
- 分支前环境状态哈希一致；
- 原始与对照 assignment、候选索引、候选池哈希；
- 哪些 UAV 固定、哪些 UAV 被允许重分配以及原因；
- 每步 served bits、3 步 payload 增量；
- 每 UAV 的目标距离、首次进入服务半径、有效服务步数；
- 路径长度、SINR、干扰、投影率、安全违规；
- payload 计账误差与队列守恒误差。

主要指标为场景级配对 payload 差：
`delta_payload = payload_preserve - payload_scheme`。

## 5. 统计与解释边界

- 统计单位为 16 个场景的配对结果，不将 UAV 或时间步当作独立样本。
- 报告均值、中位数、95% 配对 bootstrap CI、正/负/持平场景数。
- 另报告实际改变 assignment 的场景数量；若绝大多数场景没有合法可变更方案，应如实报告证据覆盖不足。
- 报告所有场景，不得只展示 payload 增益场景。
- 若平均增益为正但置信区间跨 0，Gate 仍为 inconclusive。
- 即使结果改善，也只能支持“t=45 在保留正在服务关联的条件下，受约束重分配可能改善末窗收益”；不能据此宣称完整 48 步高层策略已经改进或低层需要重训。

## 6. 交付物

- `docs/research/HTW_NextStep_T45ActiveAssignmentPreserved_CN.md`
- `results/phase6i_t45_active_preserve/scenario_pairs.csv`
- `results/phase6i_t45_active_preserve/uav_pairs.csv`
- `results/phase6i_t45_active_preserve/summary.json`
- 可选图：场景配对 payload 差、assignment 改变前后 active UAV 服务量对比。

## 7. Gate

`T45_ACTIVE_ASSIGNMENT_PRESERVED_COMPLETE`：
- 16 个场景均完成同状态配对，或明确记录无可用合法对照；
- 固定集合约束、候选合法性、状态哈希和 payload/队列计账检查通过；
- 全部逐场景差值与区间已报告。

`T45_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE`：
- 结果区间跨 0，或真正改变 assignment 的场景太少，无法形成稳定结论。

`T45_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED`：
- 状态恢复、候选合法性、计账或 split 检查失败。

完成后停止。不要自动进入训练，不修改正式高层/低层。汇报 Gate、结果、局限与唯一推荐下一步。
