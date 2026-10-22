# HTW Phase-3 Goal-conditioned Low-level Controller Report

阶段：去掉强制 waypoint tracking，TW 只作为条件输入。未改 HGNN、Transformer、TW 定义、环境、benchmark、PPO GAE/clip/optimizer/loss；未训练 predictor；未开 Round。

预算与 Phase-2 相同：8 updates，256 environment steps，seed 0。High-level 仍是 **Oracle TW**。

JSON：`src/wt/htw_phase3/outputs/goal_conditioned/summary.json`

---

## 1. 网络结构变化

| | State-only | Goal-conditioned |
| --- | --- | --- |
| Actor 输入 | `uav` dim=8 | `[state, goal, waypoint, H]` dim=**16** |
| Critic | `V(s)`，encoder width=8 | **`V(s, TW)`**，encoder width=**16** |
| 输出 | movement `(M,2)` | 同左 |
| PPO | Round-30 GAE/clip/Adam | 同左，未改 `round30/ppo.py` |

TW 经 Phase-1 adapter 拼到每个 UAV token 上，actor 与 critic **共用**这条通道，不是只进 actor。

High-level：`independent_queue_rate` oracle shadow window → `(goal_distribution, waypoint_xy, horizon)`。

---

## 2. Reward 变化

- Phase-2：controller 内部 `r_total = r_comm − λ‖p−W‖`，λ=1。
- Phase-3：**删除该辅助项**。训练向量是 `ConditionedVector`，PPO 收到的就是 `env.step` 的 `r_comm`。
- 环境 reward 公式未改。训练末步内部均值奖励：State-only 0.095，Goal-conditioned 0.096（均为通信奖励量级，不再出现 Phase-2 的负值 −0.167）。

---

## 3. 实验结果

| 方法 | payload | env return | residual queue | waypoint distance | action–W cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| State-only PPO | **3.431091e7** | 3.676 | 2.925e6 | 0.343 | −0.234 |
| HTW tracking PPO（Phase-2，λ=1） | 3.422361e7 | 3.577 | 3.012e6 | **0.308** | **+0.015** |
| HTW goal-conditioned PPO | **3.431091e7**（持平） | **3.680** | 2.925e6 | 0.330 | ≈0 |

Goal-conditioned：TW 进入 actor（dim=16，waypoint 非零），critic UAV dim=16。

---

## 4. 与 Phase-2 比较

| | Tracking（Phase-2） | Goal-conditioned（Phase-3） |
| --- | --- | --- |
| 相对 State-only payload | 下降 | **持平，未超过** |
| ρ | 从负到略正（强制跟 W） | 从 −0.23 到 ≈0（不再强制跟） |
| waypoint distance | 最低 | 介于 State-only 与 tracking 之间 |
| env return | 更差 | 略高于 State-only（3.680 vs 3.676） |

去掉 tracking 后，通信收益不再被惩罚项压低，回到 State-only 的 payload 平台；TW 条件化本身 **没有额外 payload**。这与 Phase-2.5「λ=0 也不超过 State-only」一致。

---

## 5. HTW-C 是否成立

判定规则：Goal-conditioned payload **严格大于** State-only 才算闭环成立。

**不成立。** payload 与 State-only 完全相同（3.431091e7）。

含义：问题已不是「强制 tracking 限制了通信」那么简单——取消 tracking 只是消除了伤害，**没有产生收益**。下一步应按 Phase-3 预案，**重新设计 high-level planner / 条件如何进入低层**（例如可执行的短期子目标、门控），而不是恢复 λd，也不是加 HGNN。
