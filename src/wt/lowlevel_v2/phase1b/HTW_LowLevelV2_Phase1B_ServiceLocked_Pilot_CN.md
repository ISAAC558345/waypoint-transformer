# HTW Low-Level V2 Phase-1B Service-Locked Residual MAPPO

单 seed 0，100 updates，仅 `LLV2_design`。非 fresh / confirmation。未访问 Phase-10.6 untouched。未加载 Phase-1 Actor。高层冻结。

Phase-1A：primary=SERVICE，secondary=ACTION，MECH-B。本阶段只修 SERVICE。

## 方法

ServiceResidualLock（动作执行前状态）：

`service_lock_m = target_valid AND not_idle AND queue>0 AND in_service_region`

服务区判定复用 `envelope_active` / `env._in_range`，无第二套半径。

锁定时 `delta_used=(0,0)`，因而 `a_des=a_base=(0,0)`（float 容差内）。圈外仍为 `a_des=clip(a_base+0.25*tanh(delta_RL))`。SafetyProjector 仍开启。

Actor 输入不变。locked UAV 的 policy/entropy 用 `policy_active_mask=1-service_lock` 从 PPO surrogate 中剔除；Critic / GAE / value loss 不 mask。Reward 公式不变，`C_residual` 用 `delta_used`。

## 曲线（primary=update100，禁止按中间点选模）

| update | payload | blocks | residual_norm | projector | envelope_rate | pairwise_rate | policy_active |
|---|---|---|---|---|---|---|---|
| 0 | 10.375e6 | 10.375 | 0 | 0.260% | 0 | 0.260% | 0.637 |
| 20 | 11.438e6 | 11.438 | 0.00306 | 12.5% | 0 | 12.5% | 0.635 |
| 40 | 11.375e6 | 11.375 | 0.00607 | 15.4% | 0 | 15.4% | 0.638 |
| 60 | 11.000e6 | 11.000 | 0.00668 | 3.65% | 0 | 3.65% | 0.633 |
| 80 | 11.250e6 | 11.250 | 0.00707 | 16.8% | 0 | 16.8% | 0.650 |
| **100** | **11.063e6** | **11.063** | 0.0133 | **7.68%** | 0.130% | 7.55% | 0.641 |

L0=10.375e6。旧 P1 u100=9.25e6。

u100 projector among intervened：PAIRWISE 98.3%，ENVELOPE 1.69%，STEP/MULTI/OTHER=0。旧 P1 u100 ENVELOPE 占介入 75.4% → **已基本消除**。

锁定态 `delta_used_nonzero_fraction=0`，violation=0。collision=0，min_sep_worst≥1.0（float 容差）。neighbor shuffle disagreement≈1.4e-9（本阶段不修 REL）。

Gate **P1B-A**。不自动进入 multi-seed。
