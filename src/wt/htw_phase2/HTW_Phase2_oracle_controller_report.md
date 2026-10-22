# HTW Phase-2 Oracle Hierarchical Controller Report

阶段：最小层次控制器原型。不改 TW 表示、HGNN、Transformer、PPO GAE/clip/optimizer/loss、环境 reward；不开 Round；不访问 fresh / untouched。

预算与 Phase-1 / 1.5 相同：`rollout_length=16`，`num_envs=2`，8 次 update（256 environment steps），seed 0，训练 `training_scenarios()[:16]`，评估其后 4 个 training 场景。

High-level 只用 **Oracle TW**（planner shadow window），不经过预测器。

数值：`src/wt/htw_phase2/outputs/oracle_controller/summary.json`。

---

## 结构

```
High-level Task Planner (oracle independent_queue_rate, H=5)
        ↓
TW_new = (goal_distribution, waypoint_xy, horizon)
        ↓
Low-level PPO  actor([state, TW_new]) → movement
        ↓
env.step  →  r_comm  （环境公式未改）
        ↓
PPO 看到的奖励： r_total = r_comm − λ · mean_i ||p_i − W_i||
```

- Actor：Phase-1 的 `GoalConditionedActorCritic`，UAV dim **16**。
- 距离：与 TW 相同的归一化 XY，`p` 取 **step 之后** 的位置，`W` 取决策时刻的 oracle waypoint。
- `λ = 1.0`（controller 超参，不是环境超参）。
- 评估指标里的 **return 是环境 `r_comm` 回合和**，不是 `r_total`。

---

## 结果

| 方法 | payload | env return | residual queue | tracking error | action–W cosine ρ | TW 进入 actor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| State-only PPO | **3.431091e7** | 3.679 | 2.925e6 | 0.344 | **-0.310** | 否（dim=8） |
| HTW-Oracle-PPO | 3.422361e7 | 3.577 | 3.012e6 | **0.308** | **+0.015** | 是（dim=16，W 非零） |

训练末步 PPO 内部均值奖励：state-only `+0.095`（即 `r_comm`）；HTW-Oracle `-0.167`（`r_comm − λ d`，说明 tracking 项在起作用）。

相对 Phase-1.5（flat PPO + oracle、无 tracking 项）：当时 ρ≈0 且 payload 与 state-only 相同。本次 ρ 从 **-0.31 提到 +0.02**，跟踪误差从 0.344 降到 0.308。

---

## 四个问题

### 1. Oracle TW + low-level controller 是否提升收益？

**否。** payload 从 3.431091e7 降到 3.422361e7（Δ≈−8.73e4）；env return 与剩余队列也略差。

在该 mini 预算下，跟踪辅助项还没有把通信收益带上去，甚至出现 **跟踪 vs 通信** 的轻微折中。

### 2. waypoint tracking error 是否下降？

**是，但幅度小。** 均值 `||p − W||`：0.344 → 0.308（降约 10%）。闭环开始把位置往 oracle W 拉，尚未形成紧跟踪。

### 3. action relevance 是否提高？

**是。** `cos(action, W − p)`：−0.310 → +0.015（Δρ≈+0.325）。动作从「整体背向 waypoint」变为接近正交略偏同向。这是相对 Phase-1.5 最清楚的变化：**低层开始使用 TW**。

ρ 仍远小于 1，不能称为可靠跟踪控制器。

### 4. HTW 结构是否值得继续？

**值得继续层次控制器，但不值得据此宣称 payload 已打通。**

| 证据 | 含义 |
| --- | --- |
| TW 进入 actor，且 ρ 明显上升、tracking error 下降 | 相对 flat PPO，**层次 + 内部 tracking 目标能让 TW 进入控制** |
| payload / env return 未升 | **控制价值（通信）尚未验证** |
| λ=1、仅 256 steps | 可能过强惩罚或训练不足 |

下一步（仍不要 HGNN / 不要改环境 reward）：调小 λ、加长训练，或把 tracking 做成显式到达约束而不是与 `r_comm` 硬减。只有 payload 稳定高于 state-only 时，才进入预测 TW。

---

## 约束核对

- HGNN：未用
- Transformer：未改、未跑预测
- 新 Round：未开
- benchmark：未改
- `round30/ppo.py`：未改；aux 只出现在 `HierarchicalTrackingVector.step` 交给 PPO 的 reward 数组

JSON：`src/wt/htw_phase2/outputs/oracle_controller/summary.json`
