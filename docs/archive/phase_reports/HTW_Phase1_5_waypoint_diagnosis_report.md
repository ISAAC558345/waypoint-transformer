# HTW Phase-1.5 Waypoint Validity Diagnosis Report

阶段：诊断，不修复。未改 HGNN、PPO 核心、环境、reward；未开 Round；未访问 fresh / untouched。

预算与 Phase-1 验证相同：`rollout_length=16`，`num_envs=2`，8 次 PPO update（256 environment steps），seed 0，训练 `training_scenarios()[:16]`，评估其后 4 个 training 场景。Transformer 仅 warmup，不作为正式训练。

数值来源：`src/wt/htwc_phase1/outputs/diagnosis_phase15/summary.json`。

---

## 1. Waypoint 来源分析（Label Audit）

读取 `src/wt/htwc_phase1/labels.py` 与 `src/wt/round19/planner.py`。

Phase-1 默认 `collect_planner_window(..., policy="independent_queue_rate")`：

1. 逐步调用 `independent_queue_rate_action`；
2. 在 **真实 env.step**（标签收集）或 **shared_shadow_step**（oracle，不改 live env）上滚动 `H` 步；
3. `goal_distribution` = 窗口内各 UAV 对 buoy 的累计 payload 份额；
4. `waypoint_xy` = **第 H 步结束时的 UAV 位置**（归一化到 `[-1,1]`），不是整条轨迹。

### 1.1 waypoint 是否来自 planner？

**是。** `W_t` 是 planner 滚动 H 步后的末端位置，不是人工几何点，也不是 Round-20 JSONL。

Oracle 路径与标签一致：`||W - normalize(p_H)|| ≈ 0`（H=1/3/5/10 均为 `~1e-9`）。

### 1.2 planner 是否优化 reward？

**否。** 默认策略不搜索 reward、不优化折扣回报。

`independent_queue_rate_action` 对每个 UAV：在当前可服务队列上取

`argmax_k (queue_k / capacity) * rate_{uav,k}`，

再输出指向该 buoy 的单位向量。这是 **当前时刻的 queue-rate 启发式**。

`joint_plan` 才对预测 payload 做有限 beam search，且 Phase-1 **默认标签未使用它**。`nearest_unsatisfied_buoy_direction` 存在于候选集合，但独立策略只取 `queue_rate_direction`。

### 1.3 生成类型

| 假设 | 结论 |
| --- | --- |
| 最近目标 | **否**（最近未满足 buoy 是另一候选） |
| queue 策略 | **是**（queue 加权瞬时速率） |
| shortest path | **否** |
| joint optimization | **否**（beam 可用但未用于 Phase-1 标签） |

标签质量含义：`W_t` 是 **myopic 运动积分终点**，不是最优控制子目标，也不是对 PPO reward 的最优可达点。这给判定 B 留下空间，但不能单独解释「连 oracle 也无 payload 增益」。

---

## 2. Oracle waypoint 结果

构造：`oracle_planner_window` 用当前 `(positions, queues, energy)` 做 H=5 的 independent queue-rate shadow rollout，把真实 `(G, W, H)` 拼进 UAV 特征，**不经过 Transformer**。再训练 mini PPO。

| 量 | State-only | Oracle TW PPO |
| --- | ---: | ---: |
| payload | **3.431091e7** | **3.431091e7**（相同） |
| return | 3.675 | 3.668 |
| residual queue | 2.925e6 | 2.925e6 |
| `\|\|W_t - p_{t+H}\|\|` | n/a（W=0） | 0.335 |
| ADE | n/a | 0.333 |
| `cos(W-p_t, p_{t+H}-p_t)` | n/a | **0.032** |
| action ρ | n/a | **0.023** |
| TW 进入 actor | 否（UAV dim=8） | **是**（dim=16，goal 与 waypoint 均非零） |

解释：oracle `W` 是 **planner 轨迹** 的 `p_{t+H}`，指标里的 `p_{t+H}` 是 **PPO 实际轨迹**。一致性 ≈0、位移误差 ≈0.33（归一化坐标）说明 **PPO 没有跟踪 oracle waypoint**。payload 与 state-only 到浮点完全相同，说明在该 mini 预算下 TW 没有控制收益。

---

## 3. Predicted waypoint 结果

预测器：node-token Transformer warmup（最后 loss ≈1.33），再拼 `(Ĝ, Ŵ, H)` 训练 PPO。

| 量 | Predicted G+W | vs State-only |
| --- | ---: | --- |
| payload | 3.428131e7 | **更差** |
| `\|\|W_t - p_{t+H}\|\|` | 0.447 | 大于 oracle 的 0.335 |
| ADE | 0.433 | 大于 oracle |
| consistency | **-0.274** | 劣于 oracle 的 +0.032 |
| action ρ | **-0.266** | 劣于 oracle 的 +0.023 |

预测 TW 比 oracle 更不像未来位移，payload 也略低于 state-only。这支持「预测器偏弱」（判定 A 的次要因素），但 **即使换成 oracle 仍无 payload 增益**，因此主因不是预测器。

Phase-1 报告中的 ρ=-0.453 与本次 predicted ρ=-0.266 同号：动作与 waypoint 方向整体相反或接近正交。该旧 ρ 不能单独定罪；新指标同样显示 **跟踪失败**。

---

## 4. PPO 是否利用 TW

### 4.1 接口确认

- State-only：`DeepSetsActorCritic`，`uav_dim=8`。
- TW 条件：`GoalConditionedActorCritic`，`uav_dim=16` = 8 + goal(5) + waypoint(2) + horizon(1)。
- Actor 的 `uav_encoder` 吃完整 `uav` 向量，**TW 通道确实进入 actor**（不是只进 critic、也不是旁路日志）。
- `tw_enters_actor`：oracle / predicted G+W / predicted waypoint 的 `any_waypoint_nonzero=True`；predicted goal 故意把 waypoint 置零，`any_waypoint_nonzero=False`。

未改 `round30/ppo.py`（GAE / clip / 系数不变）。

### 4.2 消融 payload

| 条件 | 输入 | payload |
| --- | --- | ---: |
| A | state | **3.431091e7** |
| B | state + predicted waypoint | 3.430036e7 |
| C | state + predicted goal | 3.430332e7 |
| D predicted | state + predicted goal + waypoint | 3.428131e7 |
| D oracle | state + **真实** goal + waypoint | **3.431091e7**（=A） |

Goal 与 waypoint **都没有带来 payload 增益**。预测通道略伤性能。因此问题不是「只缺 goal」或「只缺 waypoint」，而是 **扁平 PPO 在该预算下不把 TW 当控制目标**。

注意：B/C 的 `\|\|W-p_{t+H}\|\|` 在 C 中因 waypoint 被置零而不具跟踪含义。

---

## 5. Action relevance 与新指标

旧指标：`ρ = cos(action, W_t - p_t)`。对环境已执行的位移，H=1 时 ρ 会接近 `cos(W-p_t, p_{t+1}-p_t)`，两者高度相关，**不能单独证明 waypoint 有规划价值**。

补充（均在 PPO 实际轨迹上）：

1. 未来位移误差 `\|\|W_t - p_{t+H}\|\|`
2. ADE：对 `s=1..H` 的 `mean_s \|\|W_t - p_{t+s}\|\|`
3. 目标一致性 `cos(W_t-p_t, p_{t+H}-p_t)`

| 来源 | `\|\|W-p_H\|\|` | ADE | consistency | action ρ |
| --- | ---: | ---: | ---: | ---: |
| Old TW（φ 方向 vs 实际位移） | — | — | **-0.230** | **-0.236** |
| New TW predicted | 0.447 | 0.433 | -0.274 | -0.266 |
| New TW oracle H=5 | 0.335 | 0.333 | 0.032 | 0.023 |

Old TW 与 New TW 预测一样：**策略动作不指向 TW 所暗示的方向**。Oracle 把 ρ 从负值抬到约 0，仍远低于「跟踪」应有的大正余弦。

---

## 6. Horizon 分析

Planner 几何（独立于 PPO；标签 W 对 planner 自身轨迹）：

| H | 标签一致性 `\|\|W-p_H\|\|` | planner 位移 (m) | 第一步动作 vs H 步位移 cos |
| ---: | ---: | ---: | ---: |
| 1 | ~0 | 10.00 | **1.000** |
| 3 | ~0 | 20.21 | 0.740 |
| 5 | ~0 | 26.69 | 0.732 |
| 10 | ~0 | 32.65 | 0.641 |

H 越大，myopic 第一步与 H 步终点越分叉，**H=5/10 的子目标对当前动作更不可执行**。这解释了「把 H 步位置当即时目标」的设计风险。

Oracle PPO（每个 H 各训 8 updates）：

| H | payload | `\|\|W-p_H\|\|` | ADE | consistency | ρ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.431091e7 | 0.102 | 0.102 | -0.078 | -0.078 |
| 3 | 3.431091e7 | 0.230 | 0.229 | -0.256 | -0.263 |
| 5 | 3.431091e7 | 0.331 | 0.329 | -0.259 | -0.273 |
| 10 | 3.431091e7 | 0.449 | 0.442 | -0.403 | -0.415 |

**缩短到 H=1 仍无 payload 增益。** 跟踪误差随 H 增大而变差，但「不可执行窗口过长」**不是** Phase-1 无收益的充分原因：即使窗口只有一步，扁平 PPO 也不跟踪、也不提高 payload。

（主表 oracle H=5 与 `horizon_ppo.oracle_h5` 因另一次独立训练，ρ 分别为 0.023 与 -0.273，payload 相同。结论对随机种子敏感的是角度，不是 payload。）

---

## 7. 最终判断

四选一：

| 代号 | 含义 | 本次证据 |
| --- | --- | --- |
| A | Waypoint 预测失败，需要改预测器 | 次要：预测劣于 oracle，但 oracle 仍无 payload 收益 |
| B | Waypoint 标签无规划价值，需要重新设计 label | 部分成立：标签是 queue-rate 启发式，不是 reward 最优；但 H=1 oracle 仍无收益，单靠换 label 不能解释 PPO 不跟踪 |
| **C** | **PPO 无法利用 TW，需要 hierarchical controller** | **主因** |
| D | TW 定义错误，需要重新设计 | 未作为主因：接口维度与 oracle 注入均正确；定义「H 步位置」在 H>1 时确实偏弱，但 H=1 同样失败 |

**判定：C**

依据：

1. TW（含 oracle）已进入 actor（UAV 宽 16，waypoint 非零）。
2. Oracle 与 State-only **payload 相同**（3.431091e7）；goal / waypoint 消融均无增益。
3. Oracle 轨迹一致性 ≈0，ρ≈0 或为负：策略不跟踪 `W`。
4. 把 H 降到 1 不能打破 payload 平台。

次要并存：标签非最优规划（B）；warmup 预测器更差（A）；H>1 子目标与瞬时动作分叉（定义偏弱，接近 D 的局部问题）。

本次 **不进入 HTW-C 实现、不改预测器、不改 label、不加 HGNN**。下一步若继续，应先决定是否用 **显式低层跟踪** 验证「强制执行 W 是否改变 payload」；只有在强制跟踪仍无 payload 时，才把主因从 C 改判为 B。

JSON：`src/wt/htwc_phase1/outputs/diagnosis_phase15/summary.json`
