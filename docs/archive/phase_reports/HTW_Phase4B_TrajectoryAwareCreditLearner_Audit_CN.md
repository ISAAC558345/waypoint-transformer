# HTW Phase-4B Trajectory-Aware Residual Credit Learner Audit

本阶段只做 offline residual-credit learner。未改 Low-Level / Actor / Critic / reward / decoder / pool / checkpoint / High-Level planner，未改 Phase-3 / Phase-4A 数据，未引入 timed subgoal / joint training / HIRO / 正式部署。

**门控：PARTIAL** — C 可预测，且 candidate-geometry residual 能改善选点；但指定的 Baseline-3（加上 state geometry）不能稳定改变 planner decision。

## 1. 数据来源

- 只读 `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\htw_phase4a_hindsight_credit\outputs\phase4a_hindsight_credit.csv`
- 主判断 family=**EXEC**，384 行，16 个 state × 24 candidate
- 未重跑 trajectory audit，未改 C / U_single / U_traj

## 2. Target C 定义

`C_abs = U_traj - U_single`，与 Phase-4A / Phase-3C 的 `F` 同一数字。本阶段 **不** 训练 `C_ratio`。

C_abs 与 F 是同一标签。本实验比较的是用法：`U_hat = U_single + C_hat`（beta=1）对照 Phase-3E 的 `U_single + 0.25 F_hat`，以及 LOCAL / TRAJ-ORACLE。禁止写成“C 一定比 F 更容易学”。

## 3. Observable feature

只使用 first-hold 可观测量：`U_single`、Phase-3E `candidate_vector`、t=0 state geometry、Phase-3G leftover/access（当前未分配 queue，**不用** remaining_arrivals）。

禁止：U_traj、C_abs、F、hold2/3/4、未来位置/动作、future decoder、trajectory_length。

## 4. GroupKFold protocol

`GroupKFold(n_splits=5)`，`group=scenario_id`。同一 state 的 24 个 candidate 不会同时出现在 train 与 test。无独立 validation fold（不做超参搜索）。

- fold 0: train=[221001, 221002, 221003, 221004, 221006, 221007, 221008, 221009, 221011, 221012, 221013, 221014]；val=[]；test=[221000, 221005, 221010, 221015]
- fold 1: train=[221000, 221001, 221002, 221003, 221005, 221006, 221007, 221008, 221010, 221011, 221012, 221013, 221015]；val=[]；test=[221004, 221009, 221014]
- fold 2: train=[221000, 221001, 221002, 221004, 221005, 221006, 221007, 221009, 221010, 221011, 221012, 221014, 221015]；val=[]；test=[221003, 221008, 221013]
- fold 3: train=[221000, 221001, 221003, 221004, 221005, 221006, 221008, 221009, 221010, 221011, 221013, 221014, 221015]；val=[]；test=[221002, 221007, 221012]
- fold 4: train=[221000, 221002, 221003, 221004, 221005, 221007, 221008, 221009, 221010, 221012, 221013, 221014, 221015]；val=[]；test=[221001, 221006, 221011]

ExtraTrees: n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=0。

## 5. 四个 baseline / ablation

- B0: `C_hat = mean(C_train)`
- B1 / A: ExtraTrees(U_single)
- B2 / B: ExtraTrees(candidate geometry + leftover/access)
- C: ExtraTrees(U_single + candidate geometry)
- B3 / D: ExtraTrees(U_single + candidate + state geometry)

## 6. C prediction OOF（EXEC，pooled）

| model | Pearson | Spearman | Kendall | MAE | nMAE | within Spearman |
|---|---:|---:|---:|---:|---:|---:|
| B0_constant | -0.116 | -0.106 | -0.070 | 2.486e+06 | 0.558 | -0.180 |
| B1_U_single | 0.362 | 0.298 | 0.203 | 2.273e+06 | 0.511 | 0.338 |
| B2_cand_geom | 0.528 | 0.551 | 0.379 | 2.069e+06 | 0.465 | 0.570 |
| C_U_cand | 0.563 | 0.576 | 0.401 | 1.997e+06 | 0.448 | 0.596 |
| B3_full | 0.537 | 0.562 | 0.387 | 2.047e+06 | 0.460 | 0.576 |

Phase-3E/F 锁定的 future-value OOF Spearman（同一 F/C 标签）约为 0.553。本表 B3 pooled Spearman=0.562。

## 7. U_hat_traj = U_single + C_hat

| model | pooled Spearman | pooled Kendall | within Spearman |
|---|---:|---:|---:|
| B0_constant | 0.503 | 0.350 | 0.483 |
| B1_U_single | 0.486 | 0.339 | 0.486 |
| B2_cand_geom | 0.657 | 0.474 | 0.647 |
| C_U_cand | 0.675 | 0.487 | 0.667 |
| B3_full | 0.658 | 0.475 | 0.661 |

## 8-9. First-hold selection 与 recovery

`R_pred = U_traj(argmax U_hat) / max U_traj`，按 state 再平均。真实 U_traj 只用于评价。

| method | mean U_traj | R_pred mean | R_pred median | top-1 | top-3 | regret | NDCG@3 | R_gain(Phase-3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LOCAL | 9.000e+06 | 0.758 | 0.721 | 0.125 | 0.250 | 0.242 | 0.768 | 0.000 |
| FV_beta025 | 9.438e+06 | 0.803 | 0.806 | 0.125 | 0.312 | 0.197 | 0.804 | 0.212 |
| B0_constant | 9.000e+06 | 0.758 | 0.721 | 0.125 | 0.250 | 0.242 | 0.768 | 0.000 |
| B1_U_single | 1.012e+07 | 0.850 | 0.917 | 0.188 | 0.312 | 0.150 | 0.792 | 0.358 |
| B2_cand_geom | 1.062e+07 | 0.891 | 0.923 | 0.250 | 0.438 | 0.109 | 0.828 | 0.550 |
| C_U_cand | 1.062e+07 | 0.889 | 0.964 | 0.312 | 0.500 | 0.111 | 0.847 | 0.473 |
| B3_full | 9.312e+06 | 0.785 | 0.868 | 0.125 | 0.500 | 0.215 | 0.820 | -0.012 |
| TRAJ_ORACLE | 1.188e+07 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |

LOCAL 应对齐 Phase-4A：U=9.000e+06，R≈0.758。本表 LOCAL U=9.000e+06，R=0.758。
TRAJ-ORACLE 应对齐 1.188e+07。本表 1.188e+07。

## 10. Ablation（within-state Spearman vs C）

- A U_single-only: 0.338
- B candidate geometry: 0.570
- C U+cand: 0.596
- D full: 0.576

主判断看组内 Spearman，不看 pooled。

## 11. Permutation sanity（打乱每个 state 内的 C，不重训）

- OOF Spearman(C_hat,C)=0.562
- perm Spearman mean=0.039 std=0.050
- perm top-1 mean=0.058；OOF top-1 (C space 不直接比选择 top-1)
- perm NDCG@3 mean=0.488
- perm R_pred (on U_single+C_perm) mean=0.680

## 12. LOCAL / FUTURE-VALUE / RESIDUAL / ORACLE

| rung | source | EXEC U_traj | notes |
|---|---|---:|---|
| LOCAL | argmax U_single | 9.000e+06 | Phase-4A R_local≈0.758 |
| FUTURE-VALUE (locked 3E) | U_single+0.25 F_hat | 9.375e+06 | OOF Spearman≈0.553；Phase-3 R_gain≈0.130 |
| FUTURE-VALUE (this OOF, beta=0.25) | U_single+0.25 C_hat | 9.438e+06 | 同折 C_hat，旧配方 |
| RESIDUAL B3 (specified) | U_single+C_hat_full | 9.312e+06 | beta=1；state 项伤害选点 |
| RESIDUAL C (U+cand) | U_single+C_hat_C | 1.062e+07 | ablation 里选点最好的之一 |
| RESIDUAL B2 (geom) | U_single+C_hat_geom | 1.062e+07 | |
| TRAJ-ORACLE | argmax U_traj | 1.188e+07 | |

## 13. 是否改善 candidate ranking

- 指定 B3 vs LOCAL: R_pred 0.785 vs 0.758；top-1 0.125 vs 0.125；mean U 9.312e+06 vs 9.000e+06。Phase-3 风格 R_gain=-0.012（部分 state 变差，均值可正可负）。
- B2/C residual 选点明显更好: B2 U=1.062e+07 R=0.891 top-1=0.250；C U=1.062e+07 R=0.889 top-1=0.312。
- within Spearman(U_hat,U_traj) B3=0.661，C=0.667，相对 U_single 排序 0.483。

## 14. Leakage

特征均在 t=0 可计算。GroupKFold 按 scenario。Permutation 后 Spearman 应接近 0。state geometry 在组内为常数，不能单独解释组内 C；若 D 相对 C 没有组内增益，说明 state 项主要吃组间偏移。

- geom_gain=True；perm_drop=True；rec_gain=True

## 15. PASS / FAIL

**PARTIAL**。C 可预测，且 candidate-geometry residual 能改善选点；但指定的 Baseline-3（加上 state geometry）不能稳定改变 planner decision。

指定 Baseline-3 不能当作已经超过 Phase-3E future-value 闭环。B2/C 的选点增益是 ablation 发现，不是预先指定的成功。不要据此部署 planner，不要继续堆更复杂模型，不要进入 Phase-4C。

这是 offline evidence，不是 High-Level/Low-Level mismatch 已解决，也不是 C 比 F 更好学。

本阶段停止。不要训练 Low-Level。不要改 planner。

