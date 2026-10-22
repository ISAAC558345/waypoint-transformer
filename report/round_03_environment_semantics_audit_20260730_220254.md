# Round-3 海上通信环境语义审计

时间：`2026-07-30T22:02:54+08:00`  
审计基线：Round-2 锁定实现，100/100 回归测试通过  
训练：`training_performed=false`

## 审计结论

| 变量 | 修改前状态分类 | 证据与处理 |
|---|---|---|
| 每浮标缓存 `Q_i` | 尚未实现但项目设计明确需要 | 环境内部原来只有时间、位置、能量、浮标位置和信道；Round-2 trajectory 只能写 `[T,0]`。Round-3 v1 实现真实 bit 队列。 |
| 新到达 `A_i` | 尚未实现但项目设计明确需要 | 原仓库无 arrival 实现。Round-3 v1 增加 constant、Poisson packet 和 trace 三种可复现模式。 |
| 实际传输 `D_i` | 尚未实现但项目设计明确需要 | 原指标仅为容量代理。Round-3 v1 使用 `min(available data, allocated capacity)`。 |
| 瞬时速率 `R_i` | 已实现并正确暴露 | `channel.py` 计算并由 observation/info 暴露。继续复用，不改信道公式。 |
| 单时隙链路容量 | 已实现但未写入 observation/info | 原 metrics 由 rate 推导潜在容量。Round-3 v1 在每步 info/dataset 中显式保存逐浮标 bit 容量。 |
| 固定资源分配 | 已实现但未显式暴露 | `channel.py` 已使用 `B_total/K` 固定正交均分。Round-3 显式暴露 `bandwidth_allocation_fraction=1/K`。 |
| 可控服务对象/调度权重 | 当前设计未定义，不得虚构 | 现有 action 只有二维移动；没有关联、调度或功率分配动作。Round-3 不把导航目标伪装成通信调度。所有有效浮标在固定正交子带并行服务。 |
| UAV 剩余能量 | 已实现并正确暴露 | observation 有比例，info 有 raw 值；单位是抽象能量单位。 |
| 悬停与移动能耗 | 已实现并正确暴露 | `hover + move_per_meter × commanded_distance`；属于占位物理模型。 |
| 通信能耗 | 当前设计未定义，不得虚构 | 浮标发射功率只用于 SNR；没有 UAV 通信能耗模型。v1 明确标记 unavailable，不把 W·s 混入抽象 UAV 能量。 |
| 服务/任务完成率 | 尚未实现但项目设计明确需要 | Round-3 v1 由实际 payload 与截至当前已到达工作量计算。 |
| urgency/deadline/AoI | 当前设计未定义，不得虚构 | 不新增这些变量；旧 `urgency_rate_energy` 仅保留 v0 读取兼容，新 v1 策略名为 `queue_rate_energy`。 |
| 最大 episode horizon | 已实现并正确暴露 | 配置、剩余时间 observation 和 truncation 已有；pilot 同步设置为 128。 |
| terminal reason | 已实现但未写入环境 observation/info | 原因由采集器推断。Round-3 v1 由环境直接给出 `ongoing/task_success/energy_depleted/task_success_energy_depleted/horizon`。 |

## 锁定的时隙顺序

Round-3 采用 **service-before-arrival**，因此
`available_data_i[t] = Q_i[t]`：

1. 策略只读取 `s_t`、`Q[t]` 和已经实现的历史量；
2. 处理二维移动动作与边界投影；
3. 按现有占位模型扣除悬停/移动能量；
4. 在移动后位置计算当前链路速率；
5. 计算容量和真实 payload；
6. 读取预生成 trace 中当前时隙的 `A[t]`；
7. 更新队列、overflow/drop 和累计量；
8. 由真实 payload、能耗和终端状态计算 reward；
9. 判定任务成功、能量耗尽和 horizon。

公式为：

```text
C_i[t] = rates_bps_i[t] * delta_t
D_i[t] = min(Q_i[t], C_i[t])
preclip_i = Q_i[t] - D_i[t] + A_i[t]
drop_i[t] = max(preclip_i - Q_i,max, 0)
Q_i[t+1] = clip(preclip_i, 0, Q_i,max)
```

并逐步审计守恒：

```text
Q_i[t+1] = Q_i[t] + A_i[t] - D_i[t] - drop_i[t]
0 <= D_i[t] <= Q_i[t]
0 <= D_i[t] <= C_i[t]
0 <= Q_i[t+1] <= Q_i,max
```

缓存、arrival、capacity、drop 和 payload 均使用 `float64 bit`；rate
使用 `bit/s`；`delta_t` 使用 `s`。

## 固定并行服务说明

信道模块内部已经计算：

```text
full_rate_i = B_total * log2(1 + SNR_i)
bandwidth_allocation_fraction_i = 1 / K
rates_bps_i = full_rate_i * bandwidth_allocation_fraction_i
```

因此 v1 使用 `C_i = rates_bps_i * delta_t`，不会再次乘 `1/K`。再次相乘
会重复分配并错误缩小容量。四个非学习基线选择的是 UAV 的当前导航目标；
通信仍使用原项目已有的固定正交并行服务，不虚构新的调度动作。

## Arrival 可复现性

- constant：每浮标每步固定 bit；
- Poisson：私有 RNG 生成 packet count，再乘配置的 packet size；
- trace：读取完整 `[horizon, num_buoys]` bit 数组。

正式比较先按 `(scenario_id, seed, num_buoys, horizon)` 物化一次 trace，
四个策略使用完全相同的副本。`arrival_trace_id` 为 schema/header 和
little-endian `float64` trace 内容的 SHA-256。策略只看到上一已实现 arrival，
看不到完整 trace 或未来 arrival。

## Observation v1

扁平状态为 `state_dim = 9 + 7*num_buoys`：

```text
global[9]:
  normalized position (2)
  normalized velocity (2)
  remaining energy ratio
  remaining horizon fraction
  normalized cumulative distance
  normalized cumulative energy consumption
  task completion ratio

per buoy[7]:
  relative x
  relative y
  normalized 3-D distance
  normalized allocated current rate
  normalized queue
  normalized last realized arrival
  service completion ratio
```

`normalized_queue = raw_queue_bits / queue_capacity_bits`，可直接反归一化。
Poisson arrival 使用可逆有界映射
`A/(A+queue_capacity_bits)`，不进行无说明截断。动作后量
`link_capacity_bits`、`transmitted_payload_bits` 和本时隙
`raw_arrival_bits` 存入 `info_after` 和 dataset，不泄漏到 `s_t`。

## Reward v1

Reward 与物理指标分别保存，不用 reward 代替任务完成率：

```text
payload_utility = sum(D_i) / (theoretical_max_total_rate * delta_t)
energy_utility = actual_energy_consumed / maximum_one_step_energy
residual_queue_ratio = sum(Q_i) / sum(Q_i,max)

r =
  + payload_reward_weight * payload_utility
  - energy_penalty_weight * energy_utility
  - boundary_penalty * I(boundary)
  - invalid_action_penalty * I(action adjusted)
  - energy_depletion_penalty * I(energy depleted)
  - unfinished_queue_penalty_weight * residual_queue_ratio * I(episode end and not success)
  + task_success_bonus * I(task success)
```

原始 payload、原始/归一化能耗、原始/归一化队列、完成率、总 reward 和七个
带符号 reward component 均独立暴露。v1 不再同时加入 movement reward
penalty，避免对同一能量代价重复处罚；movement cost 仅作诊断。

## 兼容与未解决语义

- v0 是 Round-1/2 的严格兼容路径；v1 trajectory/dataset/goal schema 单独
  版本化，旧 smoke 不得静默混入。
- 浮标保持静止；漂移、风浪和随机信道扰动仍为 unavailable。
- communication energy、urgency、deadline、AoI、可控通信调度和终端区域
  仍未定义，不会为满足接口而虚构。
