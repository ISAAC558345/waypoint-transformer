# 后续优化方案：低层改进与高低层闭环联合评估

- 仓库：`ISAAC558345/waypoint-transformer`
- 分支：`feature/maritime-wt-stage1a`
- 文档性质：研究执行计划；本计划不自动启动训练、不修改正式模型。
- 核心原则：每轮只改变一个主要因素；每个低层候选都必须在完整高低层闭环中评估。

## 1. 当前系统基线

以下配置以仓库中的正式入口、配置和 checkpoint 为准；执行前应再次核对实际值。

- 环境：`L3BBitQueueEnvV2`，6 架 UAV、12 个浮标、48 步 episode。
- 高层：Scheme C（HGNN 候选评分）；候选生成配置为 `G1_RATE k=6`。
- 当前配置下，每个决策状态的候选池记录为 13,327 个合法联合 assignment。
- 高层决策时刻：0、15、30、45；TaskOption hold 为 15 步。
- 低层：冻结 Residual MAPPO-V2，正式基线权重 `phase6g_seed0.pt`，残差系数 `alpha=0.25`。
- 动作链路：`fly_toward + residual -> clip -> SafetyProjector -> environment.step`。
- 主指标：48 步完整 episode 的实际累计 payload（以环境的真实计账字段为准）。
- 辅助指标：有效服务步数、首次进入服务半径时间、路径长度、SINR、干扰、SafetyProjector 触发率、安全违规率及剩余队列。
- 正式 checkpoint、原始日志和历史实验结果必须保留，不得覆盖。

## 2. 要回答的研究问题

1. 低层训练是否正确区分 15 步高层任务窗口、PPO rollout 截断和 48 步环境终止？
2. 当前 Scheme C 的 assignment 分布是否限制低层对其他合法任务分配的泛化？
3. 新低层是否提高完整闭环 payload，而不只是改善局部奖励或轨迹指标？
4. 低层更新后，高层 Scheme C 的排序收益是否仍然成立？

## 3. P0：时间尺度、GAE 与回报语义审计（先审计，不训练）

审查以下入口及其实际调用链：

- `src/wt/benchmark_v2/phase6ir_scheme_c_align/worker.py`
- `src/wt/benchmark_v2/phase6ir_scheme_c_align/train.py`
- `src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py`
- MAPPO 更新实现中的 GAE、terminal mask、rollout bootstrap 与 critic target。
- TaskOption 更新、环境快照恢复、队列计账及完整 episode 执行入口。

必须核实：

- 窗口结束时设置 `done=True` 并加载下一个记录窗口、窗口耗尽时 bootstrap value 归零，是否是有意的窗口级训练目标，还是意外截断跨窗口长期回报。
- 区分环境真正终止、rollout 截断与高层决策边界（15、30、45 步）；高层决策边界不能自动当作环境终止。
- GAE/bootstrap 对每一种边界的处理是否与训练目标一致。
- 环境状态、队列、到达过程、运动历史、服务进度和 TaskOption 的恢复是否一致。
- 奖励是否使用当前步真实 `served_bits`，是否存在 payload 重复累计或把未来窗口收益误当即时奖励。

交付文件：`docs/research/HTW_NextStep_TemporalCreditAudit_CN.md`。报告须列出实际代码位置、终止语义表、GAE/bootstrap 处理、复现测试及结论。

Gate：
- `TEMPORAL_CREDIT_AUDIT_COMPLETE`
- `TEMPORAL_CREDIT_BUG_CONFIRMED`
- `TEMPORAL_CREDIT_IMPLEMENTATION_VALID`
- `TEMPORAL_CREDIT_AUDIT_BLOCKED`

在 P0 结论明确前，不启动长时间训练。

## 4. P1：完整 episode 执行归因（不训练）

利用已有开发/验证 episode 轨迹，按每个 15 步高层窗口建立归因表。每个 UAV/窗口至少记录：

- assignment 与初始目标距离；
- 首次进入服务半径的步数；
- 有效服务步数与窗口实际交付 bits；
- 目标切换、队列变化和剩余 hold；
- SINR、干扰、路径长度；
- SafetyProjector 触发情况与安全违规；
- 完整 episode 的实际累计 payload。

根据日志证据将低收益窗口归类为候选原因，例如：未及时到达、到达后链路质量差或干扰高、安全投影频繁、服务窗口利用不足。分类必须由数据支持，不能预先假定根因。

交付文件：`docs/research/HTW_NextStep_ExecutionAttribution_CN.md`。本阶段只分析，不改变模型。

## 5. P2：单因素低层改进

仅在 P0 和 P1 完成后，根据证据选择一个主要改进因素：

1. 若终止、GAE 或 bootstrap 语义有问题，先修复并增加回归测试。
2. 若任务分布过窄，构建覆盖 Scheme C、G1_RATE 及经过合法性检查的其他 assignment 的多任务训练分布。
3. 若任务分布足够但存在明确动作控制失败模式，再单独调整观测或动作参数化。
4. 暂时保持奖励函数不变，除非归因结果证明奖励存在具体缺陷。不要同时修改观测、奖励、动作和任务分布。

训练要求：

- 将 `phase6g_seed0.pt` 作为不可覆盖的正式基线。
- 新实验使用独立输出目录，保存配置、日志、checkpoint SHA-256 和随机种子。
- 优先 warm-start；若结构变化导致无法严格加载，必须记录权重迁移规则或明确采用 scratch，禁止静默忽略权重不匹配。
- 使用真实环境交互、逐步 served bits 和真实状态转移。
- 先做短程 smoke test，再开展多 seed 训练。
- 严格分离训练、验证和最终独立测试场景。

Gate：
- `LOW_LEVEL_TRAINING_SMOKE_PASS`
- `LOW_LEVEL_TRAINING_COMPLETE`
- `LOW_LEVEL_GAIN_SUPPORTED`
- `LOW_LEVEL_GAIN_INCONCLUSIVE`
- `LOW_LEVEL_TRAINING_BLOCKED`

## 6. P3：必须执行的 2×2 高低层闭环 Benchmark

每个低层候选都必须运行完整 48 步闭环。固定 assignment 的窗口级评估可用于诊断，但不能替代完整 episode 结果。

| 组别 | 高层 | 低层 | 用途 |
|---|---|---|---|
| A | Scheme C | 冻结 MAPPO-V2 | 正式系统基线 |
| B | Scheme C | 新低层候选 | 测量新低层在正式高层下的实际增益 |
| C | G1_RATE k=6 | 冻结 MAPPO-V2 | 高层分配基线 |
| D | G1_RATE k=6 | 新低层候选 | 检查新低层是否依赖 Scheme C 的任务分布 |

实验控制：

- 使用相同场景 ID、reset seed 和环境参数；如存在随机策略，记录其 seed。
- 主结果报告完整 episode payload，不能只报告 15 步窗口奖励。
- 报告场景级配对差、95% bootstrap 置信区间、正/负收益场景数、最差场景，以及服务步数、路径长度、SINR、干扰、投影率和安全违规率。
- 将固定 assignment 的低层诊断实验与高层动态重新选择的完整闭环实验分开报告。
- 低层改变后若高层 assignment 排序收益也改变，须单独报告交互效应，不能全部归因于低层。

主要验收标准：

- B 相对 A 的完整 episode payload 配对收益为正，且 95% 置信区间下界大于 0。
- 安全指标未超过实验前预先规定的退化容限。
- 不存在计账、接口或非法 assignment 错误。
- 若声称跨高层泛化，D 相对 C 也必须独立满足预先规定的标准；不能只凭 B 的提升宣称泛化。

## 7. P4：高层适配与联合优化（后续阶段）

只有新低层通过 P3 后才开展：

1. 冻结新低层，重新生成高层候选在新执行器下的训练标签。
2. 在训练/验证开发池重新评估 Scheme C 排序器；不得使用保留测试集选择 epoch、损失或超参数。
3. 比较旧/新执行器下的排序相关性、Top-1 regret 和完整 episode payload。
4. 只有分开优化已稳定后，才开展交替优化或联合微调，并保留冻结模块对照。

## 8. 数据划分与复现要求

- 先读取仓库中各阶段登记的实际 train/dev/reserved split，不得凭本计划猜测场景 ID。
- 已使用过的独立保留测试结果只能作为历史结果引用，不能再用于调参或选择 checkpoint。
- 若保留测试池已被读取或参与选择，登记其污染状态，并建立新的独立保留池。
- 统计以场景为配对单位，不把同一 episode 内的时间步视为独立样本。
- 记录 Git commit、配置、权重 SHA-256、软件版本、训练 seed、评估场景及逐场景原始结果。

## 9. 执行边界

- 本计划只定义研究步骤，不直接修改模型权重、不启动训练、不改变正式 Planner。
- 不覆盖或删除正式 checkpoint、历史报告或原始结果。
- 每阶段结束后先报告证据与 Gate，再决定是否进入下一阶段。
- 研究目标是提高完整闭环 payload 与跨高层泛化能力，不以生产部署审计为目标。
