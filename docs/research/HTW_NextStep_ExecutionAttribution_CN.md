# P1：完整 Episode 的高层目标—低层执行归因

- 前置审计：`docs/research/HTW_NextStep_TemporalCreditAudit_CN.md`
- 前置 Gate：`TEMPORAL_CREDIT_AUDIT_COMPLETE`、`TEMPORAL_CREDIT_IMPLEMENTATION_VALID`
- 本阶段 Gate：`EXECUTION_ATTRIBUTION_COMPLETE` / `EXECUTION_ATTRIBUTION_INCONCLUSIVE` / `EXECUTION_ATTRIBUTION_BLOCKED`
- 目标：只分析现有完整 episode 证据，判断低收益来自目标选择、目标可达性、低层运动执行、通信/干扰，还是服务窗口利用率。
- 本阶段不训练、不修改模型/奖励/动作、不改正式 Planner、不覆盖 checkpoint。

## 1. 先核对前置审计结论

P0 报告认为：
- 正式 `phase6g` 训练在 t=15/30/45 的高层切换处不终止 episode，GAE 可跨过这些边界。
- `phase6ir` 是从独立记录快照训练固定窗口；窗口结束后切换到另一条记录，因此该路径使用 `done=True` 和零 bootstrap 符合窗口级训练目标。
- 四项回归测试在报告中记录为通过；本阶段不重复训练，也不把窗口级 `phase6ir` 的目标误认为完整 episode 目标。

## 2. 数据范围与防泄漏

1. 先检查当前仓库中已有的完整 episode 逐场景结果、轨迹日志和 summary 文件，确认其来源、Git 状态、场景 ID、策略 checkpoint SHA-256、配置哈希及是否曾用于模型选择。
2. 优先使用已有开发/验证场景日志。可引用已完成的独立保留测试作为描述性历史结果，但不得根据它调参、定义新阈值或选择模型。
3. 如果日志没有保存逐步 UAV 位置、assignment、served bits、SINR、干扰、投影状态等字段，明确列出缺失字段；不得把 summary 均值推断成不存在的逐窗口数据。
4. 不可为补齐数据而静默覆盖旧结果。确需重新评估时，只能使用预先登记的开发/验证池，写入全新输出目录，并先确认不会触及 reserved test。
5. 不把 episode 内时间步当作独立统计样本。统计单位为场景；场景级差值采用配对分析。

## 3. 每个 15 步窗口必须核查的量

对决策时刻 t=0、15、30、45 分别建立记录；最后一个窗口长度按环境 horizon 计算，不要假设每个窗口都完整 15 步。

### 高层选择
- scenario ID、决策时刻、前一 assignment、当前 assignment、候选池大小/哈希、选择状态与耗时。
- 每架 UAV 的目标浮标 ID；idle 必须保留为 idle，不可当成真实浮标。
- 决策时各 UAV 到所分配目标的距离、是否已在服务半径内。
- 如果日志没有“同一状态下替代候选”的反事实评估，不得声称当前 assignment 是局部最优或高层选择错误。

### 低层运动与可达性
- 每架已分配 UAV 的首次进入服务半径步数；窗口内始终未进入则明确记为未到达。
- 实际路径长度、净位移、目标距离随时间变化、方向变化。
- 低层动作意图与最终执行动作（若日志保存）、SafetyProjector 是否介入、介入范数或失败状态。
- hold 剩余量、窗口结束时是否仍未到达、服务半径内的有效服务步数。

### 通信与队列
- 每步真实 `served_bits`，窗口总和；完整 episode 的环境 `payload_bits`。
- 平均/分位 SINR、干扰功率、队列初末值、到达量和剩余队列。
- 检查每步计账：`payload_bits[t+1] - payload_bits[t] = sum(served_bits[t])`（容差遵循环境配置）。
- 明确区分“已进入服务半径”与“本步实际产生服务”；不能把进入半径直接当成成功交付。

## 4. 分析输出

生成下列文件：

1. `docs/research/HTW_NextStep_ExecutionAttribution_CN.md`：结论、数据来源、局限、失败模式及下一步建议。
2. `results/phase6i_execution_attribution/window_rows.csv`：每个场景 × 每个决策窗口的原始归因表。
3. `results/phase6i_execution_attribution/uav_window_rows.csv`：每个场景 × 窗口 × UAV 的归因表。
4. `results/phase6i_execution_attribution/summary.json`：样本数、缺失字段、计账检查、指标汇总、数据来源哈希及 Gate。
5. 若数据足够，保存图到 `results/phase6i_execution_attribution/figures/`：目标距离/轨迹图、服务进入时间分布、窗口 payload 分布、投影率与 payload 的场景级关系、SINR/干扰与服务量关系。每张图标注场景数和统计单位。

建议表格字段：

### window_rows.csv
`scenario_id, decision_t, window_steps, assignment, candidate_count, pool_hash, selection_status, payload_before_bits, payload_after_bits, served_bits_sum, accounting_error_bits, mean_sinr, mean_interference, projection_rate, safety_violation_rate, mean_queue_before_bits, mean_queue_after_bits, episode_complete, source_log, config_hash, checkpoint_hash`

### uav_window_rows.csv
`scenario_id, decision_t, uav_id, target_buoy_id, idle, initial_distance_m, first_entry_step, entered_service_radius, effective_service_steps, served_bits, path_length_m, net_displacement_m, end_distance_m, projection_count, projection_norm_mean, mean_sinr, mean_interference, hold_remaining`

若某字段在原始日志中不存在，留空并在 `summary.json` 的 `missing_fields` 说明原因，不允许填入估算值冒充观测值。

## 5. 归因规则

- **可达性瓶颈候选**：被分配 UAV 在窗口结束前未进入服务半径，且日志能确认该事实。
- **窗口利用率瓶颈候选**：进入服务半径较晚，导致窗口内有效服务步数较少。
- **通信/干扰瓶颈候选**：已经进入服务半径，但真实 served bits 仍低，同时有 SINR/干扰日志支持。
- **动作投影瓶颈候选**：投影频繁或投影范数较大，并与轨迹偏差/服务损失共同出现。
- **高层分配瓶颈候选**：只有在同一状态的替代 assignment 被可靠评估、且有配对反事实证据时才可下此结论；否则标为“未证实”。
- **不可归因**：关键字段缺失、样本不足或相关性无法区分因果方向时，必须明确报告，不得过度解释。

相关性不是因果证据。此阶段不根据观察到的相关性直接修改奖励或策略。

## 6. Gate 判据

`EXECUTION_ATTRIBUTION_COMPLETE`：
- 已核对数据来源、场景分割和 checkpoint/config 哈希；
- 所有可用完整 episode 都生成逐窗口归因记录；
- payload 计账检查通过，异常被单独列出；
- 对缺失字段明确报告，没有以估算值填补；
- 至少将可观察现象与尚未证实的因果解释分开。

`EXECUTION_ATTRIBUTION_INCONCLUSIVE`：
- 关键轨迹或逐步服务字段缺失，无法区分目标可达性、运动和链路因素。

`EXECUTION_ATTRIBUTION_BLOCKED`：
- 数据来源或 split 无法核实、计账失败，或运行需要触及保留测试/修改正式模型。

## 7. 完成后唯一下一步

完成 P1 后停止，不自动开始训练。汇报：
1. Gate；
2. 可由现有证据支持的首要瓶颈（若有）；
3. 不可识别的问题与缺失数据；
4. 最值得优先验证的一个低层改进假设；
5. 下一阶段实验设计及其训练/验证场景来源。

在收到下一步授权前，不改高层 HGNN、候选生成器、低层 MAPPO、奖励函数、SafetyProjector 或正式 Planner。
