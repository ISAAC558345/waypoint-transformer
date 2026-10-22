# HTW Phase-4C Independent Validation Audit

本阶段只验证冻结 B2 candidate-conditioned residual credit 是否能在真正未参与 Phase-4B / Phase-4B shadow 的独立 scenario 上成立。
不是方法开发，不是调参，不是 planner integration。未训练 Low-Level，未改 reward / utility / decoder / pool / checkpoint，未根据 independent test 反向调整 B2。

**门控：PASS** — B2 在 independent scenario 上相对 LOCAL 有稳定改善，permutation sanity 通过，且不是少数 scenario 驱动。

**planner_ready = False**。本阶段结束后停止，不进入 planner integration / Phase-4D。

## 二、Independent data audit

- 真正 unseen scenario：**是**。n=64，IDs=[220000, 220001, 220002, 220003, 220004, 220005, 220006, 220007, 220008, 220009, 220010, 220011, 220012, 220013, 220014, 220015, 220016, 220017, 220018, 220019, 220020, 220021, 220022, 220023, 220024, 220025, 220026, 220027, 220028, 220029, 220030, 220031, 220032, 220033, 220034, 220035, 220036, 220037, 220038, 220039, 220040, 220041, 220042, 220043, 220044, 220045, 220046, 220047, 220048, 220049, 220050, 220051, 220052, 220053, 220054, 220055, 220056, 220057, 220058, 220059, 220060, 220061, 220062, 220063]
- 来源：`LLV2_train`。这些 ID **从未** 进入 Phase-4B GroupKFold。
- Low-Level 训练是否用过这些 scenario：是。B2 ExtraTrees 是否用过：否。
- Phase-3 validation 223000-223063 / Phase-2 validation 222000-222063 / Phase-10.6 106000-106063：拒绝使用。
- 不存在则应写“当前工程没有真正的 unseen scenario，因此只能进行 seed-held-out validation。” 本阶段**不**走这条，因为存在 unseen scenario。
- 不用 LSO 冒充 independent validation。

## 八、防泄漏

全部 64 个 independent scenario 的泄漏项均为 0：不在 Phase-4B design / train fold / validation fold / test fold，也未用于 model / feature / threshold / hyperparameter selection。明细见 `independent_summary.json` 的 leakage 字段。

B2 只在原 Phase-4A/4B 的 16 个 design EXEC state 上用冻结 ExtraTrees 拟合一次；independent labels 不进入 fit。

## 固定 B2

`C_hat = ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)`，`U_hat = U_single + C_hat`。
ExtraTrees 固定：{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5, 'random_state': 0, 'n_jobs': 1}。未改 feature / beta / C 定义。
U_single：5-seed official shadow_h15 均值。U_traj：FirstHoldOverrideWorker 覆盖 first-hold，hold 2/3/4 走正式 EXEC greedy，LL_RUN=0，deterministic=True。
`C_abs = U_traj - U_single`。pool=`generate_pool(env, seed_key=scenario*1009+0)`。

## 四 / 五、Independent test（primary = R_B2 vs R_LOCAL）

- mean/median/std R_B2 = 0.892 / 1.000 / 0.154
- mean/median/std R_LOCAL = 0.763 / 0.800 / 0.194
- mean ΔR = 0.129；worst-case ΔR = -0.500 (scenario 220012)
- top-1 / top-3 = 0.391 / 0.766
- mean/median U_traj(B2) = 1.103e+07 / 1.100e+07
- B2-ORACLE overlap = 25/64
- B2 改变 first-hold = 46/64
- improved / degraded / unchanged = 31 / 11 / 22
- payload gain vs LOCAL = 1.562e+06；residual gap vs ORACLE = 1.359e+06
- dominated_by_few = False
- Spearman(C_hat, C) = 0.595
- within-state Spearman(C_hat, C) mean = 0.583
- Spearman(U_hat, U_traj) = 0.704

恶化 scenario（全部保留，不删除）：

| scenario | R_LOCAL | R_B2 | ΔR |
|---|---:|---:|---:|
| 220000 | 1.000 | 0.786 | -0.214 |
| 220007 | 0.818 | 0.727 | -0.091 |
| 220012 | 1.000 | 0.500 | -0.500 |
| 220023 | 0.800 | 0.600 | -0.200 |
| 220028 | 0.900 | 0.700 | -0.200 |
| 220032 | 1.000 | 0.929 | -0.071 |
| 220036 | 1.000 | 0.929 | -0.071 |
| 220040 | 0.800 | 0.700 | -0.100 |
| 220041 | 1.000 | 0.929 | -0.071 |
| 220046 | 1.000 | 0.929 | -0.071 |
| 220059 | 1.000 | 0.786 | -0.214 |

逐 scenario 全表见 `scenario_metrics.csv` / `independent_decisions.csv`。

## 六、Permutation sanity（不重训，只打乱 geometry↔candidate）

- R_B2 = 0.892
- R_perm = 0.672 ± 0.031
- drop = 0.220

## 七、Residual-driven mechanism

| class | n | fraction |
|---|---:|---:|
| residual-driven | 46 | 1.000 |
| local+residual | 0 | 0.000 |
| local-driven | 0 | 0.000 |
| conflicting | 0 | 0.000 |

改变 state 上 mean dU_single=-1.261e+06，mean dC=3.339e+06，mean dU_hat=2.078e+06。

## PASS / PARTIAL / FAIL

**PASS**。B2 在 independent scenario 上相对 LOCAL 有稳定改善，permutation sanity 通过，且不是少数 scenario 驱动。

本阶段停止。不修改 B2，不修改 planner，不进入 Phase-4D。

