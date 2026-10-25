# P4：t=30 保留在服务中的关联，评估 15 步剩余窗口的受约束重分配

- 前置报告：`docs/research/HTW_NextStep_T45ActiveAssignmentPreserved_CN.md`
- 前置 Gate：`T45_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE`
- 本阶段 Gate：`T30_ACTIVE_ASSIGNMENT_PRESERVED_COMPLETE` / `T30_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE` / `T30_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED`
- 性质：同状态反事实诊断；不训练、不修改正式策略/奖励/动作/权重，不读取独立保留测试。

## 1. 前一阶段结论

P3 在验证场景 322048–322063 的 t=45 状态上，比较 Scheme C 与“保持当前关联、只重分配几何上无法在 3 步内到达目标的 UAV”的规则。

结果为 `T45_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE`：
- 16 个场景均完成配对，计账误差和安全检查通过。
- 14 个场景 assignment 不变；只有 2 个场景改变，低于预先规定的 3 个场景最低覆盖数。
- 16 场景平均差为 -1.659 bit，95% CI 为 [-4.263, 0] bit。
- 两个改变场景里，被交换的目标在剩余 3 步内仍然不可达；没有新的 UAV 进入服务半径。
- 因此 t=45 只有 3 步时，保留已服务关联的规则没有可检验的稳定增益；这不能证明 15 步窗口内也没有可改善的空间。

## 2. 本阶段问题

在 t=30 的高层决策状态，后续有完整 15 步窗口。检验以下有限假设：

**若保持 t=30 已经位于服务半径内的 UAV 的关联，只对仍在半径外的 UAV 做合法的、距离/几何可达性约束下的重分配，能否提高接下来 15 步的真实交付量，并对最终 48 步 episode payload 产生正向影响？**

这是诊断实验，不预设重分配一定有益，也不等于修改正式 Scheme C。

## 3. 数据与冻结边界

- 仅使用已登记验证池 322048–322063。
- 不读取 reserved/independent holdout 322064–322079。
- 使用当前 Scheme C `epoch_001.pt`、matching decoder、正式候选配置及冻结 `phase6g_seed0.pt`。
- SafetyProjector、环境、奖励、动作和 TaskOption 接口全部冻结。
- 不训练、不更新正式 Planner、不覆盖 P1/P2/P3 结果。
- 新结果写入 `results/phase6i_t30_active_preserve/`。

运行前重新核实 split registry、active policy、所有 checkpoint/config SHA-256 与候选池哈希。

## 4. 实验设计

### A. 捕获 t=30 同一状态

每个场景从 reset 开始，运行正式 Scheme C + 冻结 MAPPO-V2 到 t=30。保存完整环境状态、队列、payload 计账、到达时钟、UAV 位置、last movement、assignment、TaskOption、候选池与哈希。

不得从独立测试场景恢复状态。两臂必须从相同的 t=30 快照分叉，并验证状态哈希一致。

### B. 构造保留服务关联的合法候选

- 固定 t=30 已处于服务半径内的 UAV 的当前 assignment。
- 对 t=30 位于服务半径外且当前有目标的 UAV，允许在合法候选池内重新选择目标。
- 当前 idle UAV 默认保持 idle，避免将“启用额外 UAV”与“重新分配未到达 UAV”混为同一个因素。
- 候选必须来自该状态正式生成的合法 13,327 assignment 池，并保留固定 UAV 的目标，不得重复分配浮标。
- 对可重分配 UAV，计算几何可达性代理：
  `max(distance_m - service_radius_m, 0) <= 15 * step_length_m`。
- 依次最大化几何上可能进入服务半径的可重分配 UAV 数量、最小化这些 UAV 到候选目标的总距离，最后按候选池稳定原始顺序打破平局。
- 选择过程中不得读取未来 payload、served bits 或反事实执行结果。
- 若没有满足固定关联约束的合法候选，记录 `no_legal_preserving_candidate`，保留 Scheme C assignment。

记录固定 UAV、可重分配 UAV、候选索引、池哈希、每架 UAV 距离与规则版本。该几何条件只是代理，不保证在干扰/安全约束下必然可达。

### C. 两臂配对执行

每个场景运行：
- Arm S：正式 Scheme C assignment。
- Arm P：保留服务关联的受约束 assignment。

从同一 t=30 快照出发，使用相同冻结低层和 SafetyProjector。先执行 t=30 至 t=45 的 15 步窗口。

为了评价系统级后果，在 t=45 两臂分别恢复正式 Scheme C 决策，并继续执行到完整 episode 结束（t=48）。这一步是两臂各自根据其真实分支状态重新评分，不强制两臂在 t=45 使用相同 assignment。明确区分：
1. t=30 至 t=45 的局部窗口增量 payload；
2. t=30 至 t=48 的分支后累计增量 payload；
3. 完整 48 步 episode 的最终 payload（包含共同的 t=0 至 t=30 前缀，配对差只来自分支后轨迹）。

两臂在 t=45 的后续执行必须使用相同的正式策略、模型哈希、候选生成器和配置。任何异常、候选失败或安全投影失败都需记录，不得静默丢弃场景。

## 5. 指标与统计

统计单位为 16 个场景的配对结果，不把 UAV 或时间步当独立样本。

主要指标：
- t=30 至 t=45 的 15 步真实 `served_bits` 总和；
- t=30 至 t=48 的分支后实际交付比特；
- 48 步完整 episode 最终 payload；
- 进入服务半径的 UAV 数、首次进入步数、有效服务步数。

辅助指标：
- 路径长度、SINR、干扰、队列变化；
- SafetyProjector 触发率、安全违规率；
- 改变 assignment 的场景数、被改变的 UAV 数、无合法候选的场景数；
- payload 计账与队列守恒误差。

报告均值、中位数、95% 配对 bootstrap CI、正/负/持平场景数、每场景差值。对无 assignment 变化的场景仍须保留在总体分析中，不得只汇报有收益场景。

若只有极少数场景实际改变 assignment，或者置信区间跨 0，必须标记 inconclusive。即使局部 15 步窗口提升，也不能在没有完整 episode 收益证据时宣称系统级提升。

## 6. 交付物

- `docs/research/HTW_NextStep_T30ActiveAssignmentPreserved_CN.md`
- `results/phase6i_t30_active_preserve/scenario_pairs.csv`
- `results/phase6i_t30_active_preserve/uav_window_rows.csv`
- `results/phase6i_t30_active_preserve/summary.json`
- 可选图：场景级局部窗口 payload 差、完整 episode payload 差、进入服务半径 UAV 数变化。

报告需写明数据 split、checkpoint/config SHA-256、候选池哈希、两臂状态哈希、计账检查和所有失败场景。

## 7. Gate

`T30_ACTIVE_ASSIGNMENT_PRESERVED_COMPLETE`：
- 16 个场景均完成配对评估，或明确记录无合法对照；
- 状态哈希、候选合法性、payload 计账与安全检查通过；
- 局部 15 步和最终 48 步结果均完整报告。

`T30_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE`：
- 置信区间跨 0、改变分配场景数不足，或局部与完整 episode 结果不一致。

`T30_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED`：
- 状态恢复、split、候选合法性、计账或策略哈希检查失败。

## 8. 完成后停止

本阶段结束后不自动训练、不修改正式高层/低层。报告结果并推荐唯一下一步：
- 若 t=30 局部及完整 episode 收益均有稳定正向证据，再考虑开发“剩余时间感知的高层候选排序”实验；
- 若结果不确定，停止继续手工设计末窗距离规则，转向完整 episode 多场景归因与训练分布评估；
- 若候选大多不变，说明该启发式覆盖不足，不把它纳入正式系统。
