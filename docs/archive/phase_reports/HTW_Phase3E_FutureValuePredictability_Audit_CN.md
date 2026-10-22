# HTW Phase-3E：Candidate-Level Future-Value Predictability Audit

本阶段只验证：第一 hold 时刻可观测的 candidate/state 特征，能否预测 Phase-3C/3D 的 `F`。未训练 Low-Level，未改 Actor/Critic/reward/decoder/pool/checkpoint/正式 evaluation。

禁止特征：`F`、`U_traj`、后续 hold、后续 decoder、`remaining_arrivals`、未来位置/动作。`remaining_arrivals` 虽是环境日程，但正式 observation 视为未来信息，未使用。

验证协议：按 **scenario GroupKFold**（同一 state 的 24 个 candidate 不拆开）。EXEC 16 个 group，5 fold。报告的是 **out-of-fold** 预测，不是挑最好的 split。

Sanity：LOCAL-ORACLE=9.000e+06，TRAJ-ORACLE=1.188e+07，NATURAL=5.812e+06，与 Phase-3C 一致。

## 1. Feature inventory

见 `outputs/feature_inventory.csv`。实际入模：`U_single`、池内 local rank、assignment 几何 / 竞争 / 是否在服务半径、把该 assignment 放到 t=0 上算出的 SINR/rate（不 step）、以及 t=0 UAV 几何。
t=0 的 queue / payload / progress 在设计集上几乎不变；跨 scenario 的 state 差异主要是 UAV y jitter。

## 2. EXEC 预测性能（OOF）

| model | ablation | Pearson | Spearman | Kendall | MAE | nMAE |
|---|---|---:|---:|---:|---:|---:|
| constant | constant | -0.116 | -0.100 | -0.070 | 2.486e+06 | 0.558 |
| ridge_u | A U_single only | 0.079 | 0.193 | 0.117 | 2.477e+06 | 0.556 |
| ridge_sc | B state+candidate | 0.475 | 0.496 | 0.337 | 2.156e+06 | 0.484 |
| ridge_full | C U_single+state+candidate | 0.529 | 0.539 | 0.374 | 2.056e+06 | 0.462 |
| et_sc | B ExtraTrees | 0.497 | 0.527 | 0.358 | 2.145e+06 | 0.482 |
| et_full | C ExtraTrees | 0.525 | 0.553 | 0.378 | 2.094e+06 | 0.470 |
| mlp_full | C tiny MLP | -0.127 | -0.137 | -0.093 | 4.453e+06 | 1.000 |

ExtraTrees-C 训练折 Spearman=0.744，OOF=0.553。有 gap，但不是 train 很高、test 接近 0 的崩溃。
Model-B（不含 U_single）OOF Spearman=0.527，与 Model-C 接近；真正有用的是 **candidate-conditioned geometry**，不是把 U_single 再编码一遍。
tiny MLP 在 16×24 上失败（nMAE≈1），本阶段不以神经网络为准。

## 3. EXEC first-hold selection（OOF F_hat，后续仍正式 greedy）

| condition | U_traj | R vs LOCAL |
|---|---:|---:|
| NATURAL | 5.812e+06 | NA |
| LOCAL-ORACLE | 9.000e+06 | 0 |
| TRAJ-ORACLE | 1.188e+07 | 1 |
| PREDICTED ExtraTrees β=0.0 | 9.000e+06 | 0.000 |
| ORACLE-F β=0.0 | 9.000e+06 | 0.000 |
| PREDICTED ExtraTrees β=0.25 | 9.375e+06 | 0.135 |
| PREDICTED ExtraTrees β=0.5 | 9.312e+06 | 0.135 |
| ORACLE-F β=0.5 | 1.144e+07 | 0.758 |
| PREDICTED ExtraTrees β=0.75 | 9.188e+06 | 0.096 |
| PREDICTED ExtraTrees β=1.0 | 8.625e+06 | -0.058 |
| ORACLE-F β=1.0 | 1.188e+07 | 1.000 |

最佳预测选择：ExtraTrees-C β=0.25 → U_traj=9.375e+06，R_pred=0.130。
β=0 的 PREDICTED 必须等于 LOCAL-ORACLE（只用 U_single）。β=1 的 F_hat 会低于 LOCAL：未校准的满权重预测有害。
Oracle-F β=0.5 仍是 1.144e+07；预测曲线与之差约 2e6。这是 prediction gap，不是 Phase-3D 假设被推翻。

## 4. 四方法 OOF Spearman(F_hat,F) 与 β=0.5 选择

| method | ExtraTrees-C Spearman | Ridge U_single Spearman | PRED β=0.5 U_traj | LOCAL | TRAJ | R_pred(β=0.5) |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 0.676 | 0.334 | 1.250e+07 | 1.169e+07 | 1.381e+07 | 0.273 |
| ABS | 0.476 | -0.061 | 1.188e+07 | 1.112e+07 | 1.312e+07 | 0.333 |
| PREFIX | 0.590 | -0.007 | 1.081e+07 | 1.031e+07 | 1.438e+07 | 0.085 |
| EXEC | 0.553 | 0.193 | 9.312e+06 | 9.000e+06 | 1.188e+07 | 0.135 |

## 5. 四个问题

**Q1. first-hold 可观测信息是否足够预测 F？**  
有部分 signal。EXEC ExtraTrees-C OOF Spearman=0.553，constant=-0.100，训练折=0.744。
train/test 差距见上表，未把高训练相关当成可上线依据。

**Q2. F_hat 是否显著优于只使用 U_single？**  
是。Ridge-A Spearman=0.193 vs ExtraTrees-C 0.553。

**Q3. U_single+β F_hat 能否恢复一部分 trajectory-aware 收益？**  
能恢复一部分。最佳预测 U_traj=9.375e+06，LOCAL=9.000e+06，TRAJ=1.188e+07，R_pred=0.130。
Oracle F 的 β 曲线仍是上界；prediction gap = oracle 曲线 − predicted 曲线。

**Q4.**  
**predictor** — 下一步应做 candidate-level future-value predictor / critic，而不是重训整个 Low-Level。
R_pred 只有约 0.13，不能写成已经接近 TRAJ-ORACLE。剩余 gap 来自 t=0 特征对 later-greedy 诱导的 F 近似不足（queue/progress 几乎不变），不是“F 完全随机”（OOF Spearman=0.55）。因此下一阶段仍是把 predictor/critic 做准，而不是重训整个 Low-Level executor。

本阶段停止。不训练 Low-Level。不进入 Phase-4。
