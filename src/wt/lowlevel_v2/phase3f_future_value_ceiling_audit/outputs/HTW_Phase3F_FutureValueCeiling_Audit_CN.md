# HTW Phase-3F：Future-Value Predictability Ceiling Audit

本阶段只诊断 Phase-3E 的 0.553 离可观测信息上限有多远。未训练 Low-Level，未改正式方法，未使用未来变量，未做 XGBoost / 深网 / 调参。
协议：ExtraTrees + **scenario GroupKFold**。同一 state 的 24 个 candidate 不拆开。Oracle-F 只作诊断上界，不进入正式方法。

Sanity：NATURAL=5.812e+06，LOCAL=9.000e+06，TRAJ=1.188e+07。

**经验判断（不是理论上下界）：information-limited。**

## 1. Feature-group OOF（EXEC，ExtraTrees）

| group | features | OOF Spearman | train Spearman | OOF top-1 vs F |
|---|---|---:|---:|---:|
| A U_single | 1 | 0.295 | 0.354 | 0.062 |
| B candidate geometry | 17 | 0.546 | 0.700 | 0.062 |
| C state geometry | 6 | 0.071 | 0.234 | 0.188 |
| D candidate+state | 23 | 0.526 | 0.726 | 0.000 |
| E U_single+candidate | 18 | 0.570 | 0.724 | 0.125 |
| F U_single+state | 7 | 0.328 | 0.493 | 0.062 |
| G U_single+candidate+state | 25 | 0.553 | 0.744 | 0.062 |

B=0.546，C=0.071，G=0.553。state-only 几乎无信号；candidate geometry 已经接近 full 组合。

## 2. Within-state F 噪声与 U_single

16 个 EXEC state 的 F：range 中位=9.000e+06，IQR 中位=4.550e+06。
within-state Spearman(U_single,F) median=0.259。F 主要不是 local utility 的再编码。
within-state Spearman(F_hat,F) median=0.577。易预测 state（≥0.4）12 个，难预测（<0.1）1 个。未删除任何 state。

## 3. Ranking / regret / permutation

OOF G 的 within-state top-1(F_hat,F) mean=0.062，NDCG@3=0.690，R_rank median=0.434。
Permutation（每 state 打乱 F，100 次）：observed within Spearman mean=0.511，null p95 mean=0.326，median p=0.000。
0.55 量级的全局 OOF 显著高于 candidate 内随机匹配；但 top-1 仍然低，排序不够稳。

## 4. Recovery 复核（不搜新 β）

| beta | PREDICTED | ORACLE-F | R_pred | R_oracle |
|---:|---:|---:|---:|---:|
| 0.0 | 9.000e+06 | 9.000e+06 | 0.000 | 0.000 |
| 0.25 | 9.375e+06 | 1.081e+07 | 0.135 | 0.488 |
| 0.5 | 9.312e+06 | 1.144e+07 | 0.135 | 0.758 |
| 0.75 | 9.188e+06 | 1.175e+07 | 0.096 | 0.942 |
| 1.0 | 8.625e+06 | 1.188e+07 | -0.058 | 1.000 |

LOCAL=9.000e+06，TRAJ=1.188e+07。预测恢复仍约 0.13；oracle F 在 β=0.5 已接近 TRAJ。prediction gap 仍在。

## 5. 四方法 G_full OOF

| method | G OOF Spearman | A OOF Spearman | B OOF Spearman |
|---|---:|---:|---:|
| OLD | 0.676 | 0.304 | 0.688 |
| ABS | 0.476 | 0.019 | 0.487 |
| PREFIX | 0.590 | 0.205 | 0.590 |
| EXEC | 0.553 | 0.295 | 0.546 |

## 6. 六个问题

**Q1.** 是。candidate-conditioned geometry 能预测 F（B OOF=0.546），且高于 permutation null。

**Q2.** 是。G=0.553 vs A=0.295。

**Q3.** 经验判断：**information-limited**。state-only 无信号、B≈G → 主要是 **feature information 不足**（t=0 queue/progress 几乎不变，F 含 later-greedy）。train−OOF≈0.191，有轻度模型/异质性成分，但不是“换更强模型就能到 0.95”。

**Q4.** within-state U vs F median=0.259，F 的组内排序 **不稳定地依附 U_single**。F 有 candidate 结构（B 有效），也有 state 间异质性（难预测 state=1）。

**Q5.** 不足以支撑可靠的 candidate-level future-value predictor。OOF 0.55 + R_pred≈0.13 不能当上线 critic。

**Q6.** 应该停止堆模型复杂度。下一阶段应转向 **trajectory-aware credit 的可观测近似 / 结构化估计**，而不是 ExtraTrees / MLP / Transformer 调参。

empirical evidence suggests：当前 first-hold 可观测特征的 OOF Spearman 天花板大约就在 0.5–0.6，而不是 0.9。这不是数学上界。

本阶段停止。不训练。不进入 Phase-4。不进入 End-to-End Independent Validation。
