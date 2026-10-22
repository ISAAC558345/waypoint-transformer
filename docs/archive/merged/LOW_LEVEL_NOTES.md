# LOW_LEVEL_NOTES

以下内容由 `docs/archive/reports/` 中的过程文档按原文件全文合并。合并后的正文是这些过程记录的归档副本。

## 源文件 `HTW_LowLevelV2_Phase1A_Instability_Audit_CN.md`

# LL-V2.1A Frozen Checkpoint Instability and Feature-Utilization Audit

本阶段只对 Phase-1 已冻结 checkpoint 做确定性诊断。未重新训练、未增加 update、未改 lr / reward / alpha_residual / d_safe / 网络 / SafetyProjector，未访问 confirmation / Phase-10.6 untouched。场景仅使用已消费的 `LLV2_design`（机制诊断，不是 fresh）。

`LARGE_CHANGE` 阈值在跑数前固定为 **0.20**（归一化 residual 范围的 20%）。

## 1. Checkpoint inventory

全部存在，无 missing，因此全部进入后续分析。

| checkpoint | update | exists | bytes | SHA256 |
|---|---|---|---|---|
| u0 | 0 | true | 199843 | `68b863e5af30267b2c30ac81ff1899c12a400fceecb697ecc3589833b2721c0c` |
| u20 | 20 | true | 599635 | `44000b79b49ead18bcbd2f95e6f2694bff419c8dfd953355af738fa9f07918c6` |
| u40 | 40 | true | 599635 | `eed10d9bc73800b63e678d17fa8986126c39de1076d6411f2d7e1fb11688b0d0` |
| u60 | 60 | true | 599635 | `1c8c98c862736bd0a32665333c2a321e375114667837d9c12c547f6815acf190` |
| u80 | 80 | true | 599635 | `b01e18ef310f2467a6f5fc29efe2853daa81d6eae4061a903e37f2b89aa0b927` |
| u100 | 100 | true | 599709 | `6a9de2f9bf819c259e930f07f8627c9794a7fb9be7cefe581408d9bc27b28d63` |

u0 体积更小是因为 Adam optimizer 状态尚未积累，不是缺文件。

## 2. Residual 幅度（`LLV2_PHASE1A_RESIDUAL_STATS.csv`）

| ckpt | mean_residual_norm | p95 | max | mean(des−base) | mean(exec−base) | tanh_sat | actor_mean | log_std | base=0 residual |
|---|---|---|---|---|---|---|---|---|---|
| u0 | 0 | 0 | 0 | 0 | 6.8e-6 | 0 | 0 | −2.200 | 0 |
| u20 | 0.0243 | 0.0359 | 0.0444 | 0.00585 | 0.00575 | 0 | 0.0243 | −2.189 | 0.0241 |
| u40 | 0.0147 | 0.0288 | 0.0728 | 0.00363 | 0.00361 | 0 | 0.0148 | −2.176 | 0.0138 |
| u60 | 0.0154 | 0.0233 | 0.0750 | 0.00385 | 0.00384 | 0 | 0.0154 | −2.162 | 0.0147 |
| u80 | 0.0193 | 0.0247 | 0.127 | 0.00482 | 0.00477 | 0 | 0.0193 | −2.157 | 0.0180 |
| u100 | **0.0762** | **0.107** | **0.187** | **0.0190** | **0.0181** | 0 | 0.0764 | −2.153 | 0.0749 |

相对 u100：u20/u40/u80 的 mean residual 分别约为 32% / 19% / 25%。**u100 相对 u20 膨胀约 3.1×，相对 u80 约 3.9×。tanh 饱和分数全程为 0，不是饱和崩坏，是幅度膨胀。**

## 3. Projector 分解（`LLV2_PHASE1A_PROJECTOR_BREAKDOWN.csv`）

介入判定：`||a_exec−a_des||` 超过 1e-6。原因按 desired 下一步是否违反步长 / 成对距离 / 服务包络分类（未改 SafetyProjector）。

| ckpt | rate | among intervened: STEP / PAIRWISE / ENVELOPE / MULTI / OTHER |
|---|---|---|
| u0 | 0.260% | 0 / **100%** / 0 / 0 / 0 |
| u20 | 35.3% | 0 / **100%** / 0 / 0 / 0 |
| u40 | 20.2% | 0 / 99.4% / 0 / 0.65% / 0 |
| u60 | 12.4% | 0 / **100%** / 0 / 0 / 0 |
| u80 | 15.2% | 0 / 86.3% / 13.7% / 0 / 0 |
| u100 | **32.8%** | 0 / 22.6% / **75.4%** / 1.6% / 0.4% |

**u100 的 32.8% 主要是 SERVICE_SAFE_ENVELOPE（介入样本的 75.4%），不是步长，也不是组合约束。** 成对碰撞只占介入的 22.6%。u20 的高介入几乎全是微小平移后的 pairwise 校正（intervention_norm 仅 0.00125），envelope 为 0。

## 4. 服务区内 Residual（`LLV2_PHASE1A_SERVICE_BEHAVIOR.csv`）

筛选：target valid、queue>0、UAV 已在 service region、base action≈0。

| ckpt | residual_nonzero | mean_res | envelope_int | completed_blocks | unfinished | switch |
|---|---|---|---|---|---|---|
| u0 | 0 | 0 | 0 | 10.375 | 0.231 | 3.625 |
| u20 | 1.0 | 0.0277 | 0 | **11.875** | 0.145 | 4.125 |
| u40 | 0.999 | 0.0243 | 0.0012 | 11.75 | 0.177 | 4.313 |
| u60 | 1.0 | 0.0228 | 0 | 10.125 | 0.204 | 4.563 |
| u80 | 1.0 | 0.0240 | 0.022 | 11.50 | 0.198 | 4.063 |
| u100 | 1.0 | **0.0905** | **0.274** | **9.25** | 0.136 | 3.625 |

中期策略在服务区几乎**每步都输出非零 residual**，但幅度小、envelope 几乎不介入。u100 同条件下 residual 放大到 0.090，envelope 介入 27.4%。**是的：u100 频繁尝试离开已经稳定的服务区。**

## 5. Neighbor 利用（`LLV2_PHASE1A_NEIGHBOR_UTILIZATION.csv`）

只改 actor 的 neighbor embedding：zero / 同 batch 跨 UAV shuffle。按 nearest-UAV 距离等频 near/middle/far。

near 组 shuffle 的 mean actor L2：u20=0.00373，u40=0.00234，u80=0.00634，u100=0.00857。相对各自 mean residual（0.024 / 0.015 / 0.019 / 0.076）约为 15% / 16% / 33% / 11%。**near 组不是 ≈0，不能判定 neighbor 完全未用；利用弱，且 u100 相对幅度更弱。** far 组 L2 更小。

## 6. 通信特征利用（`LLV2_PHASE1A_COMM_UTILIZATION.csv`）

高干扰 tertile：

- `interf_zero` / `interf_shuffle`：全程 L2 ~1e-7 到 1e-5，**interference/INR 基本未进入动作。**
- `sinr_zero` high L2：u20=0.052，u40=0.046，u80=0.030，u100=0.068。
- `sinr_shuffle` high L2：u20=0.027，u40=0.027，u80=0.017，u100=0.036。

**不能说 u20/u40/u80 在高干扰下比 u100 更依赖通信特征。** interference 通道全体不用；SINR/rate 有作用，u100 的 SINR 消融 L2 反而最大。interference 分布大量钉在噪声底 0.001，middle tertile 经常为空。

## 7. 关系学习信号（`LLV2_PHASE1A_STATE_SIGNAL_AUDIT.json`）

来源：L0 在 `LLV2_design` 全轨迹 + `LLV2_train` 初始位置（无学习）。

- nearest UAV distance：min=1.00，p10=1.22，p25=1.40，median=1.82，p75=3.04，p90=106.3
- d_safe=1.0 的 margin：median=0.82，p10=0.22
- near-neighbor（d<1.5 d_safe）比例：**0.371**
- interference：min=median=0.001（噪声底），p90=0.041，max=1.001
- SINR：median=0，p90=17.2，max=1000
- rate：median=0，p90=4.18e6
- 高干扰（≥p75）比例：0.253（由分位数定义）
- near AND high-interf：**0.088**
- `feature_signal_weak=false`

近邻几何信号足够。干扰多数时间等于噪声底，**不能把 interference 通道不用完全归咎于网络，但 near-neighbor 冲突状态并不稀缺。**

## 8. Reward 尺度（`LLV2_PHASE1A_REWARD_SCALE.csv`）

训练 CSV **没有** `r_payload` / `C_project` / `C_residual` 列（`train_csv_has_reward_terms=false`）。尺度来自冻结 design rollout 逐步项，按 checkpoint 对应区间。

u100：mean |r_payload|=0.193，mean |0.05 C_project|=1.15e-6，mean |0.01 C_residual|=4.12e-6。  
project_penalty_ratio=6.0e-6，residual_penalty_ratio=2.1e-5。全程 **TOO_WEAK**。只诊断，未改 lambda。

## 9. PPO 稳定性（`LLV2_PHASE1A_OPTIMIZATION.csv`）

`return_mean/std`：**missing**。其余列齐全。窗口均值：

| window | KL | clip | value | grad | entropy | EV | KL_prev |
|---|---|---|---|---|---|---|---|
| 20–40 | 0.0181 | 0.207 | 2.75 | 24.4 | −9.17 | 0.053 | 0.0139 |
| 40–60 | 0.0089 | 0.154 | 2.56 | 15.9 | −8.98 | 0.090 | 0.0190 |
| 60–80 | 0.0117 | 0.135 | 2.45 | 32.9 | −8.87 | 0.037 | 0.0094 |
| 80–100 | 0.0199 | 0.191 | 2.77 | 42.9 | −8.82 | 0.138 | 0.0122 |

80→100 **没有 2× KL spike、没有 entropy collapse、没有 EV collapse、value 也无结构性跳变。** 但存在单步 gradient 尖峰：update 20≈320，60≈162，78≈361，90≈578。clip 在 update 86 到 0.29。优化有噪声，不是 u100 掉点的主因。

## 10. Policy drift（阈值 0.20 预先固定）

| pair | mean L2 | median | p95 | cosine | large_change_frac |
|---|---|---|---|---|---|
| u20→u40 | 0.0324 | 0.0322 | 0.0389 | 0.257 | 0 |
| u40→u60 | 0.0208 | 0.0194 | 0.0324 | −0.201 | 0 |
| u60→u80 | 0.0196 | 0.0196 | 0.0233 | 0.161 | 0 |
| u80→u100 | **0.0594** | **0.0592** | **0.0712** | **0.012** | 0 |

**u80→u100 的 mean L2 约为前两跳的 3×，cosine 接近 0（方向翻转），但相对 0.20 阈值 large_change_fraction=0，不是合同意义上的大幅跳变。**

## 11. Critic（`LLV2_PHASE1A_CRITIC_CALIBRATION.csv`）

`critic_calibration_resolved=true`。固定 design 轨迹上 V_pred vs 实际折扣回报：

| ckpt | MSE | bias | Pearson |
|---|---|---|---|
| u0 | 27.85 | −4.41 | −0.122 |
| u20 | 9.44 | 0.78 | 0.262 |
| u40 | 10.20 | 0.35 | 0.308 |
| u60 | 8.59 | 1.29 | 0.393 |
| u80 | 8.27 | 0.48 | **0.508** |
| u100 | 6.99 | 0.42 | 0.340 |

u100 Pearson 从 u80 的 0.51 降到 0.34，**失准加重但未崩到不可用；MSE 反而最低。** 不是主机制。

## 12. 轨迹对照（最小 4 个 design ID：221000–221003）

图：`src/wt/lowlevel_v2/phase1a/outputs/figures/traj_seed22100{0-3}.png`。含 UAV 轨迹、buoy/service radius=16 m、近邻事件、projector 点。

全 design 均值 completed_blocks：L0=10.375，u20=11.875，u80=11.50，u100=9.25（与 Pilot payload 1.0375e7 / 更高中期 / 9.25e6 一致）。最小 ID 四场景上，u20 缩短飞行并提高 221000 的 payload（9e6→14e6）；u100 在 221002 掉到 7e6，mean SINR 全面低于 L0/u80。

## 13. 避碰方向（`LLV2_PHASE1A_SEP_RESPONSE.csv`）

near 组 `dot(r_m, v_sep)`：u20=−8.7e-4，u40=−4.6e-4，u80=−1.8e-4，u100=+3.1e-4。**near 响应不显著为正。没有主动远离邻居的证据。** u20 高 pairwise 介入来自投影器，不是 actor 的分离方向。

## 14. 高干扰 SINR/rate（现有 `sinr_rates_for_links`）

高干扰步上，desired residual 相对 base 的预测 ΔSINR / Δrate 全程为负：u20 (−2.20, −1.24e4)，u80 (−1.73, −8.89e3)，u100 (−0.25, −1.88e4)。**Residual 没有改善高干扰通信。**

## 15–18. 机制与下一步

- **primary_mechanism = SERVICE**
- **secondary_mechanism = ACTION**
- **MECH-B**
- **minimal_change（本轮不执行）**：只在已处于 in-range 且 queue>0 的 UAV 上把 residual 置零，恢复 L0 门控停留；不改 lr / reward / alpha / 网络 / attention。

证据链：u100 envelope 占介入 75%；服务区 zero-base 状态下 residual 全非零且幅度 0.090、envelope 介入 27%；同时 residual 相对 u80 膨胀 3.9× 但未 tanh 饱和。OPT/CRITIC/REL 都不够单独解释掉点。

## 19. 本阶段是否训练？

否。

## 20. 是否进入 multi-seed？

否。

停止。

## 源文件 `HTW_LowLevelV2_Phase2A_Seed1_Audit_CN.md`

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

## 源文件 `HTW_LowLevelV2_Phase2C_Mask_and_BaseZero_Audit_CN.md`

# HTW Low-Level V2 Phase-2C：Base-Zero Freeze 与 Active-Policy-Mask 审计（LL-V2.2C）

本阶段只做冻结代码/数据/checkpoint 审计。没有 `optimizer.step`，没有重新训练 seed0/1，没有训练 seed2/3/4，没有改 beta / epsilon / reward / lr / PPO clip / entropy / Actor / Critic，没有 relation 模块，没有访问 Phase-2 validation 222000–222063。

恢复对象：LL-V2.2B，门控 **P2B-B**。磁盘协议：`beta_residual=0.25`，`epsilon_base_fraction=0.05`，`a_max=1.0` 故 `epsilon_base=0.05`。design u100：seed0 freeze `0.9247`，seed1 freeze `0.9052`；全部 design 评估点 `cap_activation_rate=0`。

---

## 1. 训练期间 cap 是否真的触发？

**不能从训练 log 证明逐 update 触发次数。**  
`training_cap_activation_resolved=false`：Phase-2B training history 没有 `cap_activation_count/fraction` 字段，也没有保存训练 rollout。

冻结证据（design 全部 `u0/u20/u40/u60/u80/u100`，以及 train16 的 u100）：**cap_activation_count=0，fraction=0**。  
Phase-2B design eval 曲线上每个记录点的 `cap_activation_rate` 也全是 0。

归因选择：**A。cap 在已观测的开发/冻结评估中基本从未生效。**  
不能声称“训练时曾经生效、只是 final evaluation 没生效”，因为训练 rollout 未保存。

## 2. 最终 cap_activation=0 说明什么？

在 `policy_active_mask=1` 的状态下，`||r_raw|| / (||a_base|| + eps)` 的 p99 为：

- seed0 design：0.123 < 0.25
- seed1 design：0.091 < 0.25
- frac_gt_beta = 0

因此 **beta=0.25 的 Euclidean norm cap 在可学习子集上是冗余的**。最终 `cap_activation=0` 不是统计漏记；实现把 cap 标在 `r_norm > beta * ||a_base||` 且非 freeze 上，这些状态根本没越界。

## 3. 90% 以上 base-zero 主要由哪些状态组成？

对 **pre-action** 几何（恢复位置与 queue）的互斥分解（design u100，占全部 base-zero）：

| 原因 | seed0 | seed1 |
|---|---|---|
| IDLE_OPTION | 59.8% | 61.1% |
| SERVICE_LOCK | 40.2% | 38.9% |
| EMPTY_QUEUE / INVALID / ALREADY_REACHED / OTHER | 0 | 0 |
| MULTI_REASON | 0 | 0 |

train16 同样是 **idle ≈ 59–60% + service ≈ 40–41%**，没有第三类。

结论：约 90% freeze = **高层 idle（约 60%）+ 在服务区内的 service lock（约 40%）**。不是 empty queue，也不是神秘的 BASE_EXECUTOR_ZERO_OTHER。

## 4. Service lock 与 base-zero 重叠多少？

design u100：

**seed0**：P(bz)=0.925，P(sl)=0.371，P(bz ∧ sl)=0.371，P(bz ∧ ¬sl)=0.553，P(sl ∧ ¬bz)=0。  
P(bz|sl)=1，P(sl|bz)=0.402。

**seed1**：P(bz)=0.905，P(sl)=0.352，P(bz ∧ sl)=0.352，P(bz ∧ ¬sl)=0.553，P(sl ∧ ¬bz)=0。  
P(sl|bz)=0.389。

**L0 design**（无 trust-region）：P(sl)=0.363，P(bz)=0。

解释：service lock **被 base-zero 完全覆盖**（在服务时 `a_base` 也 ≈0，所以 freeze 与 lock 重复）。但 **base-zero 不是 service-lock 的重复**：一半以上 freeze 来自 idle。新机制的增量主要是把 idle（以及其余 `||a_base||≤0.05`）从 Actor 更新中拿掉；服务区内本来已被 ServiceResidualLock 置零。

## 5. Actor 真正 active fraction 是多少？

`policy_active_mask = NOT(service_lock OR base_zero_lock)`。

训练 log（100 update，真实训练 mask 均值）：

- seed0：mean 7.21%，min 2.86%，max 13.11%，p10 4.34%，p50 6.99%，p90 10.43%
- seed1：mean 7.38%，min 2.60%，max 12.15%，p10 4.75%，p50 7.20%，p90 10.34%

冻结 design u100：seed0 7.53%（347/4608），seed1 9.48%（437/4608）。  
冻结 train16 u100：6.23% / 7.36%。

**Actor 大约只在 7–10% 的 UAV-step 上学习。**

## 6. policy loss 当前除数是 total 还是 active samples？

**ACTIVE_SAMPLES（形式 B）。**

`policy_loss = sum(mask * per_uav) / max(sum(mask), 1)`  
`actor_mask_denominator = ACTIVE_SAMPLES`  
不是 `mean(mask * loss_i)`。

## 7. entropy 除数是什么？

**同样是 ACTIVE_SAMPLES。**  
`entropy_term = sum(mask * entropy) / max(sum(mask), 1)`。

## 8. 是否存在 zero-active minibatch？

冻结 replay（与训练相同的 `rollout_length=48, num_envs=4, num_minibatches=4`，UAV-step 每批 288）：**zero-active minibatch count = 0**。

但 active 很少且不均：

- 合并 20 个诊断 minibatch：`<5%` 共 4 个（全部 seed1）；`<10%` 共 17 个。
- seed1 最小 active samples = 7（2.4%）。
- 没有空批，但大量 batch 只有几十个有效 UAV-step。

原训练 minibatch 未保存；以上是冻结 forward/data pipeline 重放，**不是训练**。

## 9. Advantage normalization 是否包含 locked samples？

**是，形式 A。**  
GAE 在全部 env-step 上 `(A - mean) / std`，再 broadcast 到 6 个 UAV；mask 只进 policy/entropy。

冻结 replay 上，在“至少一架 UAV active”的 env-step 子集中：all-sample 归一化后的 advantage 均值 seed0 为 −0.605（相对 active-only 的 0），seed1 为 +0.088。locked 多数 env-step **会平移真正可控样本的 advantage 尺度**，尤其 seed0。

## 10. 当前 Actor 梯度被 mask 缩小多少？

相对 **正确的 active 归一化**：`G_current / G_active_normalized = 1`（10+10 个冻结 minibatch）。**没有**被 total-sample 除数缩小。

若错误地用 `mean(mask * loss)`（形式 A），梯度大约会再乘 `active_fraction`，诊断比 `G_current / G_totalmean` 约 **9–41×**（随该批 active 比例变化）。那不是当前代码。

## 11. seed0/seed1 active fraction 是否明显不同？

**训练均值几乎相同（7.21% vs 7.38%）。** 不能把 seed 差异归因于长期 active 比例不同。

有效更新代理 `active_fraction * grad_norm` 均值：seed0 1.47 vs seed1 0.64。seed0 均值被个别极大 `gradient_norm_preclip`（例如 update 55 ≈ 487，update 92 ≈ 384）拉高。mean KL 0.019 vs 0.015，clip fraction 0.170 vs 0.157，接近。seed 性能差更可能来自轨迹/优势噪声，而不是稳定的 active 比例差。

## 12. base-zero 状态里 Actor raw residual 多大？

冻结 design，`r_raw = α tanh(pre)`，entropy 由全局 `log_std` 决定（同类状态几乎常数）：

- seed0：SERVICE 与 IDLE 的 mean `||r_raw||` 都约 **0.059**。
- seed1：SERVICE 约 **0.0048**，IDLE 约 **0.077**。

Actor **会在不可控状态输出非零 residual**；seed0 在 lock/idle 上几乎一样大。这些状态已 mask Actor loss，长期随机/共享骨干泄漏属于预期，不能当策略错误。seed1 在 service 上 raw 更小，说明骨干仍能把服务几何和 idle 分开，尽管 service 不进 policy loss。

## 13. base-zero freeze 是否误杀有价值动作？

目标集合：`base_zero AND NOT service_lock AND NOT idle`。  
design u100 上 **种群为空（0 步）**。所有 `¬service_lock` 的 freeze 都是 **IDLE_OPTION**。

一步 C0/C1 反事实因此没有非 idle 样本可评。局部诊断结论：freeze 没有额外锁住“有任务、不在服务、但 base 很小”的飞向目标步；那些步要么 `||a_base||>ε`（active），要么 idle。**未观察到误杀有价值动作的样本。** 这是局部几何事实，不是正式性能结论。

## 14. 为什么 beta=0.25 从未触发？

Active 子集上 `||r_raw|| / ||a_base||` 的 p99 < 0.13 < 0.25，max < 0.13。raw residual 相对仍在运动的 base 太小。cap 条件 `||r|| > 0.25 ||a_base||` 从未成立。不是实现把 cap 算错：若大量 ratio>0.25 而 cap 仍为 0 才会报警；这里 **frac_gt_beta=0**。

## 15. Phase-2B 真正起作用的机制是什么？

不要再统称 “Trust Region 成功”。

- **动作约束：A. BaseZeroResidualFreeze。** Norm cap 未绑定（非 B、非 C）。
- **优化侧：policy/entropy 已按 active 归一化（非 MASK-A），但只有 ~7% UAV-step 进 Actor，构成 MASK-B 噪声。**
- 不能把 payload 改善归因于 `beta_residual=0.25`。
- seed1 vs L0 的 +1.8125e6 与旧 seed1 的 +3.75e6，更合理的机制叙述是：冻结 idle/近零 base 上的 residual，并把 Actor 更新限制在真正飞向目标的少数步。

## 16. MASK Gate？

**MASK-B。**

不是 MASK-A（除数已经是 active）。  
不是 MASK-C：active 过少（~7%）、大量 minibatch <10%、seed1 诊断批可低至 2.4%，足以造成 seed-dependent update noise。

## 17. BASEZERO Gate？

**BASEZERO-A。**

freeze 主体是 idle（无 option）与已在服务区（与 ServiceResidualLock 重复）的状态；非 idle、非 service 的 freeze 种群为空，没有证据显示误杀可获益飞行动作。

## 18. 下一步唯一 minimal change？

**按 MASK-B：只允许 active-sample-aware minibatching（或等价的单一优化机制）。**  
不改 beta，不改 epsilon，不改 reward，不重新设计 base-zero 条件（那是 BASEZERO-B 才允许的）。本阶段不执行该修改。

## 19. 本阶段是否训练？

**否。**

## 20. 是否进入 relation 增强？

**否。REL-1 保持。** 禁止 HGNN / attention / relation loss / collision reward / interference reward。问题仍是 Residual 训练稳定性与有效样本构造。

---

## 产物

目录：`src/wt/lowlevel_v2/phase2c/` 与 `outputs/`、仓库根目录副本。

- `LLV2_PHASE2C_PROTOCOL.json`
- `LLV2_PHASE2C_CAP_ACTIVATION.csv`
- `LLV2_PHASE2C_BASE_ZERO_DECOMPOSITION.csv`
- `LLV2_PHASE2C_LOCK_OVERLAP.json`
- `LLV2_PHASE2C_ACTIVE_SAMPLE_STATS.csv`
- `LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md`
- `LLV2_PHASE2C_MINIBATCH_ACTIVE_STATS.csv`
- `LLV2_PHASE2C_ADVANTAGE_NORMALIZATION.md`
- `LLV2_PHASE2C_GRADIENT_SCALING.csv`
- `LLV2_PHASE2C_BASEZERO_COUNTERFACTUAL.csv`
- `LLV2_PHASE2C_RESIDUAL_BASE_RATIO.csv`
- `LLV2_PHASE2C_MECHANISM_GATE.json`

辅助：`LLV2_PHASE2C_UPDATE_INTENSITY.csv`，`LLV2_PHASE2C_BASEZERO_RAW_RESIDUAL.csv`，`LLV2_PHASE2C_ADVANTAGE_REPLAY.json`。

Phase-2C 停止。

## 源文件 `LLV2_PHASE2C_ADVANTAGE_NORMALIZATION.md`

# Advantage normalization

Implemented as **A**: normalize GAE over all env-steps in the rollout, then apply UAV `policy_active_mask` only inside the policy/entropy terms.

Not **B** (active-only advantage normalize).

See `LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md` and minibatch replay stats.

## 源文件 `LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md`

# LL-V2.2C MAPPO mask reduction audit

Source: `src/wt/lowlevel_v2/phase1b/mappo.py` `ICARMAPPOLocked.update` (no code change in this phase).

## Policy surrogate

Actual implementation:

```
per_uav = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
denom = mask.sum().clamp_min(1)
policy_loss = (per_uav * mask).sum() / denom
```

This is **form B**:

`loss = sum(mask * loss_i) / max(sum(mask), 1)`

`actor_mask_denominator = ACTIVE_SAMPLES`

(UAV-steps with `policy_active_mask = 1 = NOT(service_lock OR base_zero_lock)`).

Not form A `mean(mask * loss_i)`, which would divide by total UAV-steps and shrink the actor gradient by `active_fraction`.

Lines in `mappo.py`:

- `denom = mask.sum().clamp_min(1.0)`
- `policy_loss = (per_uav * mask).sum() / denom`

## Entropy

```
entropy_term = (entropy * mask).sum() / denom
```

Same **ACTIVE_SAMPLES** denominator.

`entropy_mask_denominator = ACTIVE_SAMPLES`

KL and clip fraction in the logger also use the same `denom`.

## Value / GAE

`value_loss = 0.5 * (target - value).mean()` over **all env-steps**, unmasked.

GAE advantages are env-level `(T, n_env)`, then:

```
adv = (advantages.reshape(-1) - advantages.mean()) / advantages.std()
```

Advantage **normalization includes all env-steps**, including locked UAV majorities. Then `advantage` is broadcast to 6 UAVs.

So:

- Actor loss denom: ACTIVE UAV-samples
- Entropy denom: ACTIVE UAV-samples
- Advantage scale: ALL env-samples
- Value loss denom: ALL env-samples

## Frozen gradient check

On frozen minibatches, `G_current / G_active_normalized = 1.0` for all 10+10 batches, confirming the live code is already active-normalized.

## 源文件 `LLV2_PROTOCOL.md`

# LL-V2.0 Protocol

Independent branch. FINAL-HIER is read-only. No RL training.

- frozen checkpoint SHA: `dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`
- hold=15
- low-level: movement only; high-level: association / Task Option
- L0: fly-toward + InRangeZeroMotionGate
- future method: ICAR-MAPPO
- collision d_safe is a **design placeholder**; `requires_pretraining_freeze=true`
- Phase-10.6 seeds 106000–106063 are permanently blocked
- confirmation / new final untouched pools are not created in this phase

## 源文件 `LLV2_REWARD_DRAFT.md`

# LL-V2 reward draft

Coefficients are **not** frozen. Auto-search is forbidden. One design freeze is required before any training.

{
  "primary": "actual payload increment / normalization scale",
  "r_progress": "service progress increment",
  "r_near": "soft near-collision cost",
  "r_project": "SafetyProjector intervention penalty",
  "r_residual": "residual action magnitude penalty",
  "formula": "r = r_payload + lp*r_progress - ln*r_near - lproj*r_project - lres*r_residual",
  "lambda_progress": null,
  "lambda_near": null,
  "lambda_project": null,
  "lambda_residual": null,
  "auto_search_forbidden": true,
  "design_freeze_required_before_training": true,
  "evaluation_uses_train_reward": false,
  "evaluation_metrics": [
    "payload",
    "completed_blocks",
    "collision_count",
    "minimum_separation",
    "flight_distance",
    "queue",
    "drop"
  ],
  "extra_interference_penalty": false,
  "reason_no_interference_penalty": "SINR->rate->payload already carries interference; do not add a second hand-designed penalty in v1"
}

Evaluation never uses the training reward. Flight distance is not energy.

## 源文件 `LLV2_SAFETY_PROJECTOR_REPORT.md`

# SafetyProjector report (interface only)

Deterministic iterative projection on predicted next positions.
Constraints: unit-step, pairwise predicted distance >= d_safe (placeholder), ServiceSafeEnvelope inner disk.
If infeasible: `constraint_status=failure`. Not silently labeled safe.
No QP dependency in the current environment; iterative projection is the v1 implementation.

- design seed 8 status: `ok`
- intervention_norm: 0.0
- min predicted distance: 1.2433337643669597
- envelope_ok: True
- finite: True
- inner service radius: 15.5 (service_radius - placeholder margin 0.5)
- envelope active count on seed 8 after reset: 5

Not a scientific benchmark result.
