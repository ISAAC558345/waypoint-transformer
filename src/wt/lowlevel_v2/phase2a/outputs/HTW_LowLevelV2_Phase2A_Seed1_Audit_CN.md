# HTW Low-Level V2 Phase-2A：Seed-1 Failure Causal Audit（LL-V2.2A）

本阶段只对 **Phase-2 已冻结的五个 update100 primary checkpoint** 做诊断。未重新训练、未增加 updates、未改 seed1、未换中间 checkpoint、未改 reward / `alpha_residual` / `d_safe` / ServiceResidualLock / SafetyProjector / Actor / Critic / Neighbor Encoder / 高层，未加 relation 模块，未访问 confirmation / Phase-10.6 untouched。

Phase-2 consumed validation（`LLV2_PHASE2_VALIDATION_V1`，222000–222063）仅用于 **post-hoc 解释**，禁止据此调参或挑 checkpoint。机制统计在允许的 `LLV2_design`（221000–221015）上完成。

正式结论保持不变：

- `LLV2-MS-B`
- `stable_residual_advantage=false`
- `REL-1`

---

## 1. 恢复的 Phase-2 正式结果（磁盘）

来源：`src/wt/lowlevel_v2/phase2/LLV2_PHASE2_PER_SEED_METRICS.csv`、`LLV2_PHASE2_BOOTSTRAP.json`、`LLV2_PHASE2_GATE.json`。

| 项 | 值 |
|---|---|
| L0 mean payload | **10,234,375**（≈10.234e6） |
| seed0 | 11,328,125（Δ +1,093,750） |
| seed1 | 8,578,125（Δ **−1,656,250**） |
| seed2 | 13,359,375（Δ +3,125,000） |
| seed3 | 10,468,750（Δ +234,375） |
| seed4 | 10,328,125（Δ +93,750） |
| 正向 seed | **4/5** |
| overall mean Δ | **+578,125** |
| hierarchical 95% CI | [−800,000, +1,997,031] **跨 0** |
| seed-cluster 95% CI | [−756,250, +1,940,625] **跨 0** |
| 注册碰撞（1e-6） | 全部 seed 为 0 |
| projector_failure | 0 |
| service_lock_violation | 0 |

primary checkpoint 全部是 **update100**。中间 update 只用于训练稳定性，不替换 seed1。

---

## 2. 学习曲线（`LLV2_PHASE2A_LEARNING_CURVES.csv`）

seed1 训练设计曲线 payload（`llv2_train_curve16`）：

| update | payload | residual | projector | policy_loss | value_loss | entropy | KL | clip | grad |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.000e6 | 0 | 0.13% | — | — | — | — | — | — |
| 20 | **11.625e6** | 0.0024 | 10.0% | −0.065 | 2.19 | −1.54 | 0.0016 | 0.011 | 2.82 |
| 40 | 10.125e6 | 0.0132 | 43.9% | −0.016 | 2.23 | −1.52 | −0.0004 | 0.016 | 2.97 |
| 60 | 10.188e6 | 0.0178 | 25.5% | −0.079 | 2.10 | −1.49 | −0.0011 | 0.061 | 3.61 |
| 80 | 9.375e6 | 0.0201 | 37.2% | +0.006 | 2.73 | −1.47 | 0.0087 | 0.074 | 3.36 |
| 100 | **8.562e6** | **0.0283** | **3.4%** | +0.008 | 1.60 | −1.48 | 0.0033 | 0.027 | 3.02 |

seed2 对照：u20 11.750e6 → u100 **13.312e6**，projector 升到 66%；residual 维持在 0.011–0.020。

**曲线类型 = B：早期学好（u20 高于 L0），后期退化。** 不是 A（从头没学）、不是纯 C（全程无结构振荡）、不是 D（value_loss 未先崩再带坏 actor；u100 value_loss 反而低于中期）、不能算 E（后期 residual 膨胀 + projector 塌缩是可复现结构）。

entropy 从 −1.54 到 −1.47，**无 entropy collapse**。u20–u100 的 KL/clip 仍在正常 PPO 量级。

---

## 3. Residual 统计（design，final u100）

`LLV2_PHASE2A_RESIDUAL_STATS.csv`：

| seed | mean | median | p95 | max | tanh sat | actor mean | log_std | z vs others | rank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0133 | 0.0064 | 0.0498 | 0.0638 | 0 | 0.062 | −2.188 | −0.92 | 2 |
| **1** | **0.0283** | **0.0376** | 0.0432 | 0.0440 | 0 | **0.147** | −2.157 | **+2.28** | **5（最大）** |
| 2 | 0.0157 | 0.0159 | 0.0482 | 0.0902 | 0 | 0.094 | −2.175 | −0.47 | 3 |
| 3 | 0.0105 | 0.0090 | 0.0307 | 0.0380 | 0 | 0.058 | −2.194 | −1.61 | 1 |
| 4 | 0.0248 | 0.0329 | 0.0494 | 0.0977 | 0 | 0.130 | −2.174 | +1.16 | 4 |

**seed1 不是 Residual 过小，而是典型幅度过大**（median 最高）。tanh 饱和全 0，不是饱和崩坏。max 反而低于 seed2/4，说明 seed1 把残差“摊满”在中等幅度，而不是偶发极大值。

方向（`LLV2_PHASE2A_RESIDUAL_DIRECTION.csv`）：

| seed | mean forward | mean lateral | reverse fraction | base≈0 residual |
|---|---|---|---|---|
| 0 | −0.0049 | 0.0120 | 0.596 | 0.0123 |
| **1** | **+0.0052** | 0.0127 | **0.401** | **0.0291** |
| 2 | −0.0069 | **0.0268** | 0.514 | 0.0128 |
| 3 | +0.0046 | 0.0132 | 0.378 | 0.0099 |
| 4 | +0.0400 | 0.0213 | 0.219 | 0.0207 |

相对最佳 seed2：**seed1 反向更少、横向更小、沿 base 略为正**。异常点是 **base≈0 时 residual 仍 0.029（约 2.3× seed2）**。  
因此：**不是过度减速/频繁反向，而是空闲/无飞向基线上仍在乱推，整体幅度过大。**

---

## 4. ServiceResidualLock

`LLV2_PHASE2A_SERVICE_LOCK.csv`：五 seed `service_lock_violation=0`，`locked_residual_nonzero_fraction=0`。seed1 envelope 介入为 0。

**SERVICE 排除为 seed1 主因。** 锁在五 seed 上一致生效。

---

## 5. Projector 不是“介入少更好”

Phase-2 validation：seed1 介入 ≈4.8%，seed2 ≈59.1%，seed2 最好。

design 分解（`LLV2_PHASE2A_PROJECTOR_CONTEXT.csv`）：

| seed | n_intervened | PAIRWISE | ENVELOPE | MULTI | STEP | near-neighbor | high-interf | target approach |
|---|---|---|---|---|---|---|---|---|
| 0 | 59 | 98.3% | 1.7% | 0 | 0 | 88% | 81% | 36% |
| **1** | **25** | 100% | 0 | 0 | 0 | 96% | **4%** | **96%** |
| 2 | **439** | 100% | 0 | 0 | 0 | 99% | **74%** | 35% |
| 3 | 345 | 99.4% | 0 | 0.6% | 0 | 99% | 93% | 50% |
| 4 | 449 | 98.7% | 0 | 0.2% | 0 | 93% | 91% | 53% |

seed1 的少量介入几乎全在 **near-neighbor + 低干扰 + 接近目标**。seed2 大量介入发生在 **高干扰近邻**，属于驻站避碰修正。

**低介入率不代表更好**：seed1 的 desired 动作更少把 UAV 送进需要 pairwise 校正的服务几何，而是把部分 UAV 带离簇。

---

## 6. 逐场景失败结构（post-hoc validation，禁止调参）

64 场全部保留（`LLV2_PHASE2A_SEED1_SCENARIO_FAILURE.csv`）。

| bucket | n | mean Δ | mean L0 payload | flight | near_events | projector | blocks | SINR |
|---|---|---|---|---|---|---|---|---|
| large-negative | 31 | −3.26e6 | **11.55e6** | 744 | 20.0 | 5.4% | 8.29 | 47.8 |
| moderate-negative | 17 | −1.00e6 | 9.41e6 | 791 | 17.3 | 5.3% | 8.41 | 54.1 |
| near-zero | 13 | +0.31e6 | 8.92e6 | 727 | 14.7 | 2.7% | 9.23 | 60.0 |
| positive | 3 | +2.67e6 | **7.00e6** | 640 | 12.3 | 5.6% | 9.67 | 60.3 |

**seed1 最差集中在 L0 原本就高 payload 的场景**：fly-toward 已经能完成较多块，seed1 额外飞行距离更长、完成块更少、SINR 更低。正向只有 3/64，且 L0 基线最弱。queue/drop 各桶接近，不是排队爆炸独有。

这些场景 **不得** 用于后续方法选择。

---

## 7. seed1 vs seed2 轨迹（design 最小 8 个 ID：221000–221007）

图：`src/wt/lowlevel_v2/phase2a/figures/seed1_vs_seed2_traj_*.png`。

相同高层目标下最明显差异：

- L0 / seed2：多数 UAV 留在左侧浮标簇的服务圆内，做局部机动（seed2 更密、更碎）。
- **seed1：从服务簇拉出长射线，UAV 飞向空域而不是停在圈内。** 221000 / 221003 尤其清楚。

seed2 做了驻站 + 近邻修正；seed1 没有把 residual 用在圈内微调，而是在 base≈0 或巡航时把飞机推离服务区。这与 validation 飞行距离 748 vs seed2 522、design `time_in_range` 67 vs 96 一致。

---

## 8. 到达与服务

`LLV2_PHASE2A_NAVIGATION.csv` / `LLV2_PHASE2A_SERVICE.csv`（design）：

| | mean arrival | median | p90 | never_enter 计数 | time-in-range | blocks | active SINR | active rate |
|---|---|---|---|---|---|---|---|---|
| L0 | 6.54 | 4 | — | 38 | 104.7 | 10.38 | 144 | 3.82e6 |
| seed1 | **7.79** | 4 | 14 | **20** | **67.1** | **8.44** | **182** | 4.46e6 |
| seed2 | 6.43 | 4 | 14 | 23 | 96.1 | 12.63 | 139 | 4.91e6 |

首次进入：seed1 均值慢约 1.3 step，中位数相同。never_enter 计数并不更高。

**进圈后 SINR/rate 不差，甚至更高；真正差的是圈内停留时间（约 L0 的 64%）和完成块。** 问题主要在圈外/离站轨迹，而不是锁失效后的服务破坏。

---

## 9. Neighbor / 通信特征

design shuffle（`LLV2_PHASE2A_FEATURE_SENSITIVITY.csv`）：五 seed neighbor L2 均 ~1e-9。interference 同样 ~1e-8。SINR shuffle seed1=0.002，seed2=0。

Phase-2 validation 的 neighbor_shuffle 同样全体 ~1e-9。

**关系欠利用是全体 seed 的共同性质，不是 seed1 独有，不能解释 seed1 vs 正向 seed 的差距。REL 不能升为 seed1 主因。**

---

## 10. Actor 表示 / Critic / PPO / 初始化 / 高层反馈

Actor（`LLV2_PHASE2A_ACTOR_REPRESENTATION.csv`）：seed1 hidden var **最高** 0.104，有效秩 10（中等），pairwise cosine 0.36（不比别人更塌），head norm 0.084 略低于 seed2 的 0.137。**不是 representation collapse。** log_std −2.16 vs 其他 −2.17～−2.19，尺度正常。

Critic（`LLV2_PHASE2A_CRITIC_CALIBRATION.csv`）：seed1 MSE 7.58 **优于** seed2 的 11.42；Pearson 0.425 vs 0.388。**排除 CRITIC 主因。**

PPO（`LLV2_PHASE2A_OPTIMIZATION.csv`）：seed1 窗口 41–60 的 preclip grad **277**（seed2 同期 63）是中期尖峰；此后 KL/clip/entropy 恢复。**OPT 是相关机制，不是单独可解释全部差距的主因**（u20 已经学好，崩在 residual 幅度与离站，而不在 entropy/KL 饱和）。

初始化（`LLV2_PHASE2A_INIT.csv`）：actor 范数 9.48–9.66，seed1 9.66 略高；log_std 全是 −2.2。**不是异常初值 bug。INIT 排除。**

高层闭环（`LLV2_PHASE2A_HIGHLEVEL_FEEDBACK.csv`）：决策次数均为 4。seed1 assignment-change **2.25 < L0 2.75 < seed2 2.81**；idle **0.675 vs L0 0.552**。低层把 UAV 推离/闲置后，高层后续更少换任务、更多 idle。**HIGHLEVEL_FEEDBACK 是次级相关机制，不是独立高层 bug。**

---

## 11. 机制标签与 Gate

允许标签中的判定：

| 标签 | 判定 |
|---|---|
| ACTION | **primary**：residual 过大；base≈0 仍输出 0.029；后期 projector 塌缩 |
| NAV | **secondary**：到达略慢；time-in-range 大降；轨迹离站长射线 |
| HIGHLEVEL_FEEDBACK | 相关：idle 升高、assignment 变化减少 |
| OPT | 相关：u41–60 grad spike |
| REL / CRITIC / SHIELD / INIT | 排除为 seed1 主因 |
| NONE | 否：存在可复现结构，不是纯无机制噪声 |

- `primary_mechanism=ACTION`
- `secondary_mechanism=NAV`
- **S1 Gate = B**（多个相关机制，优先顺序明确：先限制空闲/过大 residual，再观察驻站与高层 idle；不是单一 forward-trust 就能概括，因为 seed1 反向并不高于 seed2）

禁止下一步“修 seed1”：不得多种子、换 u20 checkpoint、seed 专用超参。

**唯一** `minimal_method_change`：

> 对 **全部 seed 统一**：限制 residual 范数，并在 `||a_base||≈0`（idle / 无飞向）时冻结 residual。不加 relation 模块。本阶段不执行。

若后续在训练/设计集上该改动不能稳住 5-seed，再考虑把 OPT 尖峰当作第二优先级的最小 PPO 稳定化；**不要**因为 seed1 方差去加关系网络。

---

## 12. 二十问（最终）

1. **曲线类型 B**：u20 学好，u40–u100 退化。  
2. Residual **过大**（z=+2.28，rank 5），不是过小。  
3. 方向 **不是过度反向**；异常是 base≈0 仍大幅修正。  
4. 首次到达 **略慢**（7.79 vs L0 6.54），中位数相同。  
5. 进圈后 SINR/rate **正常偏高**；差在停留时间。  
6. Projector 低介入 **不代表更好**。  
7. 最差在 **高 L0 payload 场景**（额外飞行、完成块不足）。  
8. 与 seed2：相同目标下 seed1 **把 UAV 拉出服务簇**，seed2 留在簇内做局部修正。  
9. neighbor/interference under-use **不是 seed1 独有**。  
10. Actor 表示 **无 collapse**。  
11. Critic **不明显更差**（MSE 更好）。  
12. PPO：**中期 grad spike**，无终期 KL/entropy 崩坏；OPT 非主标签。  
13. 初始化 **不异常**。  
14. **有**低层状态反馈诱发更高 idle、更少 assignment change。  
15. **primary=ACTION**  
16. **secondary=NAV**  
17. **S1=B**  
18. **minimal_method_change=全 seed 统一 residual 范数上限 + base≈0 冻结 residual**（本阶段不改代码）  
19. **本阶段是否训练？否。**  
20. **是否立刻增加 relation 模块？否。** 只有 REL 成为主要机制才可以；当前证据不支持。

完成后停止。
