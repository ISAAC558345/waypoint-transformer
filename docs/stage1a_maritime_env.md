# 阶段 1A：单无人机—静止浮标最小海上通信环境

## 1. 阶段目标

阶段 1A 建立一个仅依赖 Python 标准库与 NumPy 的最小环境，用来验证单架固定高度无人机在二维矩形海域内移动、多个静止浮标通过固定正交资源上传数据，以及位置、信道、能量、奖励和 episode 生命周期能否形成稳定闭环。

环境采用与 Gymnasium 相同的函数返回形式，但不继承或导入 Gym/Gymnasium：

```text
reset(seed=None) -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)
```

环境只输出 `state, action, reward, next_state, terminated, truncated, info` 所需的信息。本阶段不定义 `goal`、`goal_dim` 或 waypoint 语义。

## 2. 未实现范围

本阶段有意不实现以下功能：

- 多无人机、无人机间协作或避碰；
- 浮标漂移、海流、海浪和随机小尺度衰落；
- 浮标数据队列、到达过程和丢包；
- 功率优化、带宽调度、用户关联、波束赋形和回传链路；
- GNN、PPO、D4RL、MuJoCo 或其他训练/仿真框架；
- 离线训练数据生成与存储；
- Waypoint Network 或 Transformer 的训练、推理和结构修改；
- 最终任务 goal 的定义；
- 阶段 1B 的策略或轨迹采集代码。

因此，当前浮标静止、没有队列、没有海流、只有单架无人机；通信资源采用固定正交分配，信道只有确定性大尺度路径损耗，环境尚未接入 Waypoint Transformer 训练。

## 3. 目录结构

```text
src/wt/maritime/
├── __init__.py   # 阶段 1A 公共接口
├── config.py     # 冻结配置及参数验证
├── channel.py    # 独立、无状态的信道与理论速率计算
├── env.py        # MaritimeEnv 状态机
└── smoke.py      # 固定 seed 的单 episode 冒烟运行

tests/test_stage1a_maritime_env.py
docs/stage1a_maritime_env.md
```

上述模块与论文原有 `dataset.py`、`goal_net.py`、`transformer.py`、`policies.py`、`train.py` 和 `step.py` 相互独立。

### 3.1 版本保护记录

阶段 1A 开始前的版本检查结果：

```text
branch: feature/maritime-wt-stage0-5
HEAD:   716f514b942dfe413dd23fd51cf1c523d8211c0f
git diff --check: PASS
```

当时工作树包含阶段 0.5/0.6 尚未提交的 README、启动脚本、配置、依赖、
七个 WT 核心文件、测试和审计文档修改。它们均原样保留，没有执行
`reset`、`checkout --` 或 `clean`。确认目标分支此前不存在后，创建并切换到：

```text
feature/maritime-wt-stage1a
```

阶段 1A 只新增本节列出的 `src/wt/maritime/`、测试和文档文件。实现结束后，
对 `dataset.py`、`goal_net.py`、`transformer.py`、`policies.py`、`train.py`、
`step.py`、`analyze_d4rl.py` 的阶段 0.6 基线 SHA-256 进行复核，结果为
`7/7 unchanged`。

## 4. 配置参数

`MaritimeEnvConfig` 是 `frozen=True` 的 dataclass。非法输入直接抛出 `ValueError`，不会被静默修正。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `area_width` | `200.0` | 海域 x 方向长度（m） |
| `area_height` | `200.0` | 海域 y 方向长度（m） |
| `num_buoys` | `4` | 静止浮标数 |
| `episode_horizon` | `20` | 最大时间步数 |
| `delta_t` | `1.0` | 单步时长（s） |
| `uav_altitude` | `50.0` | 固定飞行高度（m） |
| `max_speed` | `10.0` | 最大水平速度（m/s） |
| `initial_uav_x` | `100.0` | 初始 x 坐标（m） |
| `initial_uav_y` | `100.0` | 初始 y 坐标（m） |
| `initial_energy` | `100.0` | 初始抽象能量 |
| `hover_energy_per_step` | `0.1` | 单步悬停能耗 |
| `move_energy_per_meter` | `0.01` | 单位移动距离能耗 |
| `total_bandwidth_hz` | `1.0e6` | 总带宽（Hz） |
| `buoy_tx_power_w` | `0.1` | 每个浮标发射功率（W） |
| `reference_channel_gain` | `1.0e-3` | 参考信道增益 |
| `pathloss_exponent` | `2.2` | 路径损耗指数 |
| `noise_power_w` | `1.0e-9` | 噪声功率（W） |
| `rate_reward_weight` | `1.0` | 速率收益权重 |
| `movement_penalty_weight` | `0.05` | 移动惩罚权重 |
| `boundary_penalty` | `0.5` | 单次越界惩罚 |
| `energy_depletion_penalty` | `1.0` | 能量耗尽惩罚 |
| `randomize_buoys` | `False` | 是否在 reset 时随机布置浮标 |
| `buoy_margin` | `20.0` | 浮标到边界的最小距离（m） |
| `seed` | `0` | 环境内部随机生成器初始 seed |

验证范围包括正区域尺寸、至少一个浮标、正 horizon/步长/速度/高度、非负能量参数、正通信参数和路径损耗指数、可用的浮标 margin，以及位于区域内的无人机初始位置。

默认四浮标坐标依次是 `(50,50)`、`(50,150)`、`(150,50)`、`(150,150)`。非四浮标的确定性模式使用覆盖可用区域的规则网格；随机模式只使用环境自己的 `np.random.Generator`，不会调用全局 `np.random.seed`。所有位置均至少离边界 `buoy_margin`。

## 5. 动作定义

输入动作 shape 严格为 `[2]`：

```text
u_t = [u_x, u_y]
```

环境先验证 shape 和有限性，再依次执行：

1. 将每个分量裁剪到 `[-1,1]`；
2. 若裁剪后向量的二范数大于 1，则做径向单位球投影；
3. 映射为物理位移

```text
Δq_t = u_t * max_speed * delta_t
```

因此 `||Δq_t||₂ <= max_speed * delta_t`，对角动作不会突破速度上限。原动作、分量裁剪结果、最终归一化动作、物理位移和两类投影标志都进入 `info`，不存在静默裁剪。

## 6. 状态向量顺序

状态是一维、固定长度的 `np.float32` 数组，顺序如下：

```text
0:2   UAV normalized position       [x_norm, y_norm]
2:4   UAV normalized velocity       [vx_norm, vy_norm]
4     remaining energy ratio        E_norm
5     remaining time ratio          time_remaining
6:    four values per buoy:
      [relative_x, relative_y, normalized_3d_distance, normalized_rate]
```

浮标特征按照 `buoy_positions_m` 中的稳定顺序依次展开。

## 7. `state_dim` 计算

无人机公共特征为 6 维，每个浮标增加 4 维：

```text
state_dim = 6 + 4 * num_buoys
action_dim = 2
```

默认四浮标环境的 `state_dim=22`。环境公开 `state_dim`、`action_dim` 和 `num_buoys` 属性。

## 8. 信道公式

无人机水平位置为 `q(t)`，高度为 `H`，第 k 个浮标位置为 `w_k`。三维距离与确定性大尺度信道增益为：

```text
d_k(t)    = sqrt(H² + ||q(t)-w_k||₂²)
beta_k(t) = beta_0 / d_k(t)^alpha
```

`compute_channel_metrics` 是无状态函数；它只根据输入返回水平距离、三维距离、信道增益、线性 SNR、逐浮标速率和总速率，不读取或修改环境状态。输入浮标数组 shape 为 `[num_buoys,2]`，各逐浮标输出 shape 为 `[num_buoys]`。

## 9. 速率公式

浮标使用固定正交资源，每个浮标带宽为：

```text
B_k = B_total / num_buoys
```

对应的 SNR 和 Shannon 速率为：

```text
SNR_k(t) = P_b * beta_k(t) / sigma²
R_k(t)   = B_k * log2(1 + SNR_k(t))
```

理论单浮标最大速率和理论总速率使用水平距离为 0、三维距离为 `H` 的情形计算。它们只用于速率归一化和测试，并不表示多个真实浮标可以占据同一位置。

## 10. 奖励函数

归一化速率效用和移动代价为：

```text
U_rate(t) = sum_k R_k(t) / R_sum,max
C_move(t) = ||Δq_t||₂ / (max_speed * delta_t)
```

奖励为：

```text
r_t =
    rate_reward_weight * U_rate(t)
  - movement_penalty_weight * C_move(t)
  - boundary_penalty * I_boundary
  - energy_depletion_penalty * I_energy_depleted
```

返回值是 Python `float`。`reward_components` 使用
`rate_reward`、`movement_penalty`、`boundary_penalty` 和
`energy_depletion_penalty` 四个带符号键，其数值之和严格等于返回
reward。`rate_utility` 和 `movement_cost` 是归一化原始量；
`boundary_cost` 与 `energy_depletion_cost` 是两个非负惩罚量；
`energy_cost` 是本步占位模型请求能耗。它们均保留在 `info` 中，便于后续审计。

## 11. 能量占位模型

单步能耗和更新为：

```text
E_cost(t) = hover_energy_per_step
          + move_energy_per_meter * ||Δq_t||₂

E(t+1) = max(0, E(t) - E_cost(t))
```

能量变为 0 时 `terminated=True` 并加入 `energy_depletion_penalty`。

**该模型仅用于验证软件接口、终止条件和奖励分解，不代表最终论文中的真实旋翼无人机推进能耗模型。** 后续如果采用空气动力学模型，必须作为独立阶段设计，并重新核对量纲、参数来源和归一化。

## 12. `reset` / `step` 接口

`reset(seed=None)`：

- 非空 seed 会重建环境内部 `np.random.Generator`；
- 时间步归零；
- 无人机恢复初始位置、零速度和初始能量；
- 根据配置重建静止浮标；
- 重新计算信道；
- 返回 `(observation, info)`。

`step(action)` 按以下顺序执行：

1. 拒绝已结束 episode 上的调用；
2. 验证、裁剪并径向投影动作；
3. 构造候选位置；
4. 将越界候选位置投影到最近边界点；
5. 根据实际移动更新速度；
6. 计算并扣除能量；
7. 时间步加一；
8. 重新计算信道和速率；
9. 计算奖励；
10. 令 `terminated = energy depleted`；
11. 令 `truncated = time_step >= episode_horizon`；
12. 构造并返回下一状态。

越界本身不会终止 episode。episode 结束后继续调用 `step` 会抛出 `RuntimeError`，调用者必须先 `reset`。

## 13. `info` 字段

reset 和 step 均保留未归一化物理状态。公共字段至少包括：

| 字段 | 类型/shape |
| --- | --- |
| `time_step` | `int` |
| `uav_position_m` | `[2]` |
| `uav_velocity_mps` | `[2]` |
| `remaining_energy` | `float` |
| `buoy_positions_m` | `[num_buoys,2]` |
| `rates_bps` | `[num_buoys]` |
| `total_rate_bps` | `float` |
| `channel_gains` | `[num_buoys]` |
| `snr_linear` | `[num_buoys]` |
| `state_dim` | `int` |
| `action_dim` | `int` |

step 额外包含：

- `raw_action`、`component_clipped_action`、`normalized_applied_action`：`[2]`；
- `physical_displacement_m`：动作映射得到的请求位移 `[2]`；
- `applied_displacement_m`：边界投影后的真实位移 `[2]`；
- `action_component_clipped`、`action_norm_projected`：`bool`；
- `boundary_violation`：`bool`；
- `candidate_position_m`、`applied_position_m`：`[2]`；
- `boundary_projection_distance_m`：`float`；
- `rate_utility`、`movement_cost`、`boundary_cost`、`energy_cost`；
- `actual_energy_consumed`、`energy_depleted`、`energy_depletion_cost`；
- `reward_components`：四个带符号奖励分量的 mapping。

所有 `info` 数组均为副本，调用者修改它们不会改变环境内部状态。

## 14. 归一化方式

```text
x_norm = 2*x/area_width - 1
y_norm = 2*y/area_height - 1
v_norm = v/max_speed

dx_norm = (w_x-x)/area_width
dy_norm = (w_y-y)/area_height

d_3d,max = sqrt(H² + area_width² + area_height²)
d_norm   = d_3d/d_3d,max
R_norm   = R_k/R_k,max

E_norm          = E/initial_energy
time_remaining  = (episode_horizon-time_step)/episode_horizon
```

实现逐项检查 shape、有限性和理论范围，不使用末尾统一 `nan_to_num` 掩盖计算错误。若初始能量为 0，能量比例采用明确的零值分支，而不是执行除零。

## 15. 测试和 smoke 命令

在仓库根目录使用已经验证的同一解释器：

```powershell
$env:PYTHONPATH="src"
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m wt.maritime.smoke
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m unittest discover -s tests -v
& "..\decision-transformer-master\.venv-dt-gym\python.exe" -m compileall -q src
git diff --check
```

Linux/WSL 对应的 smoke 命令是：

```bash
PYTHONPATH=src python -m wt.maritime.smoke
```

smoke 使用固定 seed `2023` 和脚本私有随机生成器产生动作并运行完整 episode，
逐步验证 observation 与 reward 有限，最后只打印八项统计：episode 步数、累计
奖励、累计通信量、总移动距离、最终位置、剩余能量、越界次数和动作分量裁剪
次数。它不绘图、不保存模型，也不调用论文训练入口。相同 rollout 另外发生
14 次径向投影；该标志逐步保留在 `info["action_norm_projected"]` 中。

本轮阶段 1A 验收结果：

```text
原阶段 0.6 测试：33
阶段 1A 新测试：45
完整测试总数：78
Ran 78 tests
OK

compileall: PASS
git diff --check: PASS
```

固定 seed smoke 输出：

```text
episode_steps: 20
cumulative_reward: 11.481533
cumulative_communication_bits: 52819467.295168
total_movement_distance_m: 177.704502
final_position_m: [107.041512, 149.187125]
remaining_energy: 96.222955
boundary_violation_count: 0
action_clipping_count: 12
```

## 16. 已知限制

- 浮标完全静止，不含系泊或漂移模型；
- 没有数据队列、数据到达、缓存上限、时延或丢包；
- 没有海流、海浪、风场和传感器噪声；
- 只有单架无人机；
- 无人机高度固定，动作只控制二维水平移动；
- 通信资源固定正交均分，不做带宽、功率或用户关联优化；
- 信道仅有确定性大尺度距离路径损耗，没有 LoS/NLoS 概率、遮挡、多径、阴影或小尺度衰落；
- 能量为接口占位模型，不是经物理标定的推进模型；
- reward 权重没有经过任务标定；
- 尚未确定 Waypoint Transformer 的 goal、return target 或轨迹切片语义；
- 尚未生成任何离线数据，也没有训练任何神经网络。

## 17. 阶段 1B 接口计划（仅文件级设计）

阶段 1B 应先确定离线轨迹采集协议，再决定 goal/waypoint 语义。建议的文件级边界如下；本阶段不创建这些文件：

```text
src/wt/maritime/heuristic_policy.py
    DeterministicHeuristicPolicy
    只依赖 observation/info 选择动作，作为可解释基线。

src/wt/maritime/random_policy.py
    RandomPolicy
    拥有独立 Generator，可复现实验 seed。

src/wt/maritime/rate_greedy_policy.py
    RateGreedyPolicy
    面向即时通信速率的确定性贪心动作，不执行资源优化。

src/wt/maritime/trajectory.py
    Transition、EpisodeTrajectory、TrajectoryCollector
    采集 state/action/reward/next_state/terminated/truncated/info。

src/wt/maritime/trajectory_io.py
    定义版本化离线存储 schema、dtype、shape、metadata 和校验。
    明确完整 episode 与定长片段的生成及 episode 边界规则。

docs/stage1b_offline_trajectory_design.md
    记录策略组合、seed 划分、轨迹数量、质量分层、完整轨迹与
    片段关系，以及与 Waypoint Transformer dataset 的映射方案。
```

进入阶段 1B 前需要先决定：

1. 启发式、随机和通信贪心策略的混合比例；
2. 数据集 train/validation/test 的 episode 级 seed 划分；
3. 是否保存完整 `info`，以及大数组的持久化策略；
4. 完整 episode 是唯一事实来源，定长片段只在加载时生成，还是二者分别存储；
5. goal 是空间终点、累计通信回报、返航约束还是其他任务目标。

在这些语义确定之前，不应把浮标中心或累计速率临时硬编码为 Waypoint Transformer goal。
