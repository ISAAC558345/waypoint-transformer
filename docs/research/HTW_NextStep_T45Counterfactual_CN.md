# P2：t=45 最后 3 步的同状态合法分配反事实评估

- 前置报告：`docs/research/HTW_NextStep_ExecutionAttribution_CN.md`
- 前置 Gate：`EXECUTION_ATTRIBUTION_COMPLETE`
- 本阶段 Gate：`T45_COUNTERFACTUAL_COMPLETE` / `T45_COUNTERFACTUAL_INCONCLUSIVE` / `T45_COUNTERFACTUAL_BLOCKED`
- 本阶段性质：仅做受控评估；不训练、不改正式模型、不更新 Planner，不读取独立保留测试集。

## 1. P1 结论与本阶段问题

P1 在验证场景 322048–322063 上重新回放正式 Scheme C + 冻结 MAPPO-V2，计账与队列守恒检查通过。没有确认普遍性的低层瓶颈。唯一成组出现的未进入服务半径事件集中在 t=45 的最后 3 步：11 条 UAV-window 的起始距离约 87.94–99.61 m，结束时仍在服务半径外，未发生 SafetyProjector 投影。P1 也明确指出，高层分配错误尚未证实，因为没有在同一状态比较替代合法 assignment。

因此，本阶段只检验一个具体假设：

**在 t=45 只剩 3 步时，Scheme C 是否选择了一个相对于更近的合法 assignment 会导致更低 payload 的分配？**

这不是低层策略训练，也不预设 Scheme C 有错。

## 2. 场景与数据边界

- 仅使用已登记验证池：322048–322063（16 个场景）。
- 不读取 reserved/independent holdout 322064–322079。
- 不使用这次评估结果训练或选择低层 checkpoint。
- 运行前核实 split registry、active policy、Scheme C/MAPPO/decoder/config SHA-256。
- 输出写入全新目录：`results/phase6i_t45_counterfactual/`，不得覆盖 P1 原始归因结果。

## 3. 反事实实验设计

### A. 捕获同一状态

对每个验证场景运行正式 Scheme C + 冻结 `phase6g_seed0.pt`，到达 t=45 前保存完整环境快照与当前状态，包括 UAV/浮标位置、队列、到达时钟、SINR/速率状态、服务余量、payload 计账状态、last movement、当前正式 assignment、候选池及其哈希。

在同一状态记录：
- Scheme C 原始 assignment；
- 13,327 个合法候选的绑定关系与评分；
- 每架 UAV 到原目标的距离；
- t=45 仅剩 3 步这一时间约束。

### B. 选择唯一的距离导向合法对照

只在当前同状态的既有合法候选池内选择一个对照 assignment，不修改候选生成器或正式策略。优先使用以下确定性规则：

1. 对每个候选计算 t=45 当前 UAV 到被分配目标的距离；idle 不算作浮标目标。
2. 计算 3 步可达性代理：`max(distance - service_radius, 0) <= 3 * step_length` 视为几何上可能进入服务半径；这只是必要的几何代理，不保证在动态约束下可达。
3. 从合法候选中优先选择“可在剩余窗口内几何到达的非 idle UAV 数量最多”的 assignment。
4. 若并列，选择总目标距离最小的 assignment；再并列时用候选池的稳定原始顺序打破平局。
5. 记录该候选的原始索引、pool hash、距离分数和规则版本。不得使用任何反事实 rollout 的未来 payload 来挑选该候选，否则会构成结果泄漏。

若没有候选比 Scheme C 的几何可达性更好，则记录“无更好几何候选”，仍保留原始分配，不伪造对照优势。

### C. 配对执行

对每个场景，从同一个 t=45 快照创建两个独立副本：

- Arm S：原始 Scheme C assignment；
- Arm N：上一步按预注册规则选择的距离导向合法 assignment。

两臂均使用相同的冻结 MAPPO-V2 确定性 Actor、同一 SafetyProjector、相同到达时钟和相同 3 步 horizon。每臂只执行剩余 3 步。检查两个副本在分支前的状态哈希一致，且执行期间没有改变模型权重。

主要结果：3 步内实际增量 `served_bits` / payload。

辅助结果：
- 进入服务半径的 UAV 数量；
- 每架 UAV 首次进入服务半径时间；
- 有效服务步数；
- 路径长度、投影触发率、安全违规；
- SINR 与干扰（仅作描述，不据窗口平均 SINR 单独推断因果）。

必须验证每臂的 payload 增量等于 3 步 `served_bits` 之和，误差不超过环境计账容差。

## 4. 统计方法与解释边界

- 统计单位是场景，共 16 个配对场景；不要把 UAV 或时间步当独立样本。
- 报告每场景差值 `delta_payload = payload_N - payload_S`、均值/中位数、正负场景数和 95% 配对 bootstrap CI。
- 报告有多少场景中距离导向候选确实增加几何可达 UAV 数。
- 若候选只改变少数 UAV 任务，报告 assignment 差异明细。
- 本实验只评估“t=45 的距离导向合法 assignment”与 Scheme C assignment 的 3 步局部反事实差异。它不证明整个 48 步最优分配，不证明 Scheme C 整体有误，也不直接证明低层需要改进。
- 由于距离规则本身不是通信收益 oracle，即使距离导向方案胜出，也只能说明该启发式在这个窗口可能有用；若要声称高层分配有因果问题，需要进一步预注册更全面的同状态候选评估。

## 5. 交付物

生成：
- `docs/research/HTW_NextStep_T45Counterfactual_CN.md`
- `results/phase6i_t45_counterfactual/scenario_pairs.csv`
- `results/phase6i_t45_counterfactual/uav_pairs.csv`
- `results/phase6i_t45_counterfactual/summary.json`
- 若可读性有帮助：`results/phase6i_t45_counterfactual/figures/` 中的场景配对差值图和几何可达数量对比图。

报告必须包括每场景的原始 assignment、距离导向 assignment、候选索引与哈希、两臂 payload、差值、到达/服务指标、投影/安全指标、计账误差和 Gate。

## 6. Gate

`T45_COUNTERFACTUAL_COMPLETE`：
- 16 个验证场景均成功捕获同状态并完成两臂 3 步评估；
- 候选合法性、状态哈希和 payload 计账检查通过；
- 每场景结果及配对区间已报告。

`T45_COUNTERFACTUAL_INCONCLUSIVE`：
- 对照方案平均收益区间跨 0，或差异不足以支持明确结论。

`T45_COUNTERFACTUAL_BLOCKED`：
- 无法可靠恢复同一状态、候选绑定不一致、计账/安全检查失败，或场景划分无法确认。

完成后停止，不自动训练、不修改正式策略。只汇报证据，并推荐一个下一步方向。
