# HTW Phase-5.1 Oracle Task Option Report

最小层次验证：规则 Oracle 给出 buoy option `z`，低层 PPO 看 `[state, z]` 并带完成度辅助。未改 HGNN、Transformer、环境、原始 reward；未开 Round；**不使用 Task-Waypoint**。

预算与 Phase-2/3 相同：8 updates，256 environment steps，seed 0。

JSON：`src/wt/htw_phase51/outputs/oracle_option/summary.json`

---

## 1. 网络结构

| | State-only | Oracle Task Option |
| --- | --- | --- |
| Actor / Critic UAV 宽 | 8 | **14** = 8 + 5 buoy one-hot + 1 idle |
| 策略 | `π(a\|s)`，`V(s)` | `π(a\|s,z)`，`V(s,z)` |
| PPO | Round-30 GAE / clip / Adam / loss | 同左，未改 `round30/ppo.py` |

High-level **不是网络**：每 `hold=5` 步用当前状态的 `joint_aware_matching` 生成每机目标 buoy 或 idle。这只写进特征，**不改** `env.step` 的 matching。

---

## 2. 输入输出

- High-level：`z ∈ {0,1}^{M×(K_max+1)}`，最后一维 idle。
- Low-level 输入：`[state, z]`。
- Low-level 输出：movement `(M,2)`，与 Round-30 相同。

---

## 3. Reward 设计

- 环境仍只算 `r_comm`。表中 **return 是环境回合和**。
- Controller 内部：`r_ppo = r_comm + 0.25 × mean_m 1[assignment_m = z_m]`。
- 不用 waypoint 距离。

---

## 4. 实验结果

| 方法 | payload | env return | residual queue | goal completion | action efficiency (bits/m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| State-only PPO | **3.431091e7** | **3.667** | **2.925e6** | **0.948** | **1.168e6** |
| Oracle Task Option PPO | 3.430674e7 | 3.608 | 2.929e6 | 0.604 | 4.509e5 |

Option 策略运动更多（76 m vs 29 m），完成率反而更低，payload 未超过 State-only。

State-only 对「当前 joint-aware 意图」的完成率已经约 95%：冻结 matching 在现位置上几乎已经给出 oracle 任务。把 `z` 再喂给低层并加完成奖励，会驱动多余位移，干扰通信。

---

## 5. 是否进入 High-level policy 训练

**不进入。**

成功标准是 Oracle Task Option payload **严格高于** State-only。本次未达到。

含义：问题不只是 TW 表示。在当前环境里，**离散 buoy option + 完成型低层** 也没有带来通信收益——任务分解与已由 matching 决定的服务对象高度重叠。在未证明 option 有价值之前，不应训练可学习 π_H，也不应加 HGNN。
