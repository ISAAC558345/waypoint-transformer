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
