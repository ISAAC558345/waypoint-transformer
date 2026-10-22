# HTW Phase-2.5 Waypoint–Communication Alignment Report

诊断：Phase-2 中 tracking 变好但 payload 下降，是 **λ 过大**，还是 **waypoint 没有通信价值**。

未改 HGNN、Transformer、TW 定义、环境 reward、PPO GAE/clip/optimizer；未开 Round。Oracle TW。预算与 Phase-2 相同：8 updates，256 environment steps，seed 0。

CSV：

- `lambda_sensitivity.csv`（仓库根目录与 `src/wt/htw_phase25/outputs/alignment/`）
- `waypoint_communication_alignment.csv`（同上）
- `src/wt/htw_phase25/outputs/alignment/tracking_loss.csv`

---

## 实验1：λ 敏感性（controller 内部 `r_comm − λ d`）

State-only payload = **3.431091e7**。

| λ | payload | env return | residual queue | tracking error | action–W ρ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| State-only | **3.431091e7** | 3.667 | 2.925e6 | 0.305 | 0.349 |
| 0 | 3.428384e7 | 3.651 | 2.952e6 | 0.327 | 0.108 |
| 0.01 | 3.428355e7 | 3.651 | 2.952e6 | 0.327 | 0.107 |
| 0.05 | 3.428793e7 | 3.656 | 2.948e6 | 0.323 | 0.151 |
| 0.1 | 3.429470e7 | 3.658 | 2.941e6 | 0.320 | 0.186 |
| 0.5 | **3.431091e7**（持平） | 3.679 | 2.925e6 | 0.321 | 0.213 |
| 1.0 | **3.431091e7**（持平） | 3.677 | 2.925e6 | 0.307 | **0.478** |

趋势：λ 增大时 ρ 上升、tracking error 略降；小 λ 的 payload **低于** State-only；λ∈{0.5, 1.0} 只回到 State-only 的 payload 平台，**没有超过**。

因此：Phase-2 的通信下降 **不能** 用「只要把 λ 从 1 降到很小」来修好——更小的 λ 更差。λ=1 在本预算下已经是 payload 最好的 penalty 之一（与 State-only 持平），同时 ρ 最高。

---

## 实验2：soft tracking loss

环境 reward 仍是 `r_comm`。Controller 内：

`L = L_PPO + β · mean ||p̂ − W||`

`p̂` 是对当前策略均值动作的一步位移（可反传；环境转移本身不进图）。未改 `round30/ppo.py`，仅 `TrackingLossAgent` 包装。

| β | payload | env return | tracking error | ρ |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 3.428438e7 | 3.651 | 0.326 | 0.110 |
| 0.05 | 3.428654e7 | 3.651 | 0.326 | 0.115 |
| 0.1 | 3.428924e7 | 3.651 | 0.325 | 0.122 |

β 增大时 payload / ρ 略升，但全部低于 State-only，也低于 λ=0.5/1.0 的 reward penalty。本预算下 **reward 内 penalty 比 loss 附加项更有效**，两者都没有打出通信增益。

---

## 实验3：waypoint 是否倾向低通信位置？

128 条样本（16 个 training 场景 × 最多 4 步 × 2 UAV）。W 是 `independent_queue_rate` 滚动 H=5 后的位置；目标 buoy 是当前 `(queue/capacity)×rate` 的 argmax。

| 统计 | 值 |
| --- | ---: |
| W 更靠近目标 buoy | **80.5%** |
| W 对目标 channel gain 更高 | **80.5%** |
| W 对目标 rate 更高 | **80.5%** |
| 目标即最大队列 | 77.3% |
| 平均距离变化 | **−11.5 m**（更近） |
| 平均 rate 变化 | **+8.09e4 bps** |
| 当前对目标 rate | 9.33e5 bps |
| W 处对目标 rate | 1.01e6 bps |

结论：oracle waypoint **不是** 低通信位置；它系统性靠近高 queue×rate 目标，并提高对该目标的增益与速率。标签仍是启发式而非全局最优，但 **缺少通信价值不是 Phase-2 掉 payload 的主因**。

---

## 三个问题

1. **是否存在合适 λ 使 HTW payload > State-only？否。** 扫描的 λ 中没有一个严格更高；最好是持平。
2. **tracking 应作为 reward 还是 loss？** 不要写进环境 reward。在 controller 内部，本预算下 **reward penalty 优于 loss**（ρ、payload 都更好）。但换形式仍超不过 State-only，所以 **不是「改成 loss 就能赢」**。
3. **当前 waypoint 是否具有通信价值？有。** 约 80% 样本在目标链路上增益/速率/距离三项同时改善。

---

## 最终判断

| 代号 | 含义 | 本次 |
| --- | --- | --- |
| A | 调整 λ 即可继续 HTW | 否：无 λ 使 payload 超过 State-only |
| B | 需要 communication-aware waypoint | 否（主因）：W 已偏向更好的目标信道 |
| **C** | **需要重新设计 hierarchical controller** | **是** |

**判定：C**

依据：TW 能进控制、W 有通信指向、调 λ 只能把 payload 拉回 State-only 平台。扁平低层用「减距离」或「加 L_track」都会和 `r_comm` 抢梯度，且 mini 预算下 payload 对运动不敏感（出现完全相同的 3.431091e7 平台）。下一步应改控制结构（例如只在接近 W 时跟踪、或用到达约束/门控，而不是全程 λd），而不是加 HGNN 或重定义 TW。
