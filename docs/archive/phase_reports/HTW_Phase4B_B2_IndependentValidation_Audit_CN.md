# HTW Phase-4B B2 Leave-Scenario-Out Verification

本阶段只验证固定 B2 residual，不是方法开发。未训练 Low-Level，未改 planner / reward / decoder / pool / checkpoint，未重跑 Phase-3 rollout，未把结果写入正式 checkpoint。

**门控：PARTIAL** — 无法真正独立验证。leave-scenario-out 上 B2 相对 LOCAL 有可重复增益，但不能当作 held-out deployment 证据。

## A. 是否存在真正独立验证数据

**不存在。无法进行真正独立验证。** 不伪造 independent validation。

- 已标注 first-hold `U_traj`/`C` 的 scenario 只有 `LLV2_design`：[221000, 221001, 221002, 221003, 221004, 221005, 221006, 221007, 221008, 221009, 221010, 221011, 221012, 221013, 221014, 221015]
- 这 16 个 state 已全部进入 Phase-4B GroupKFold，没有留下未拟合过的 EXEC 标签
- registry 中另有 64 个 `LLV2_train` seed，但没有 Phase-3 counterfactual，因此没有 `C_abs`
- PREFIX/OLD/ABS 不是独立 EXEC scenario：后续 hold greedy 不同，`C` 不可直接当 EXEC 测试
- Phase-3 validation seed 被 `refuse_compat` 拒绝，且无本协议标签

因此本报告只做 **leave-scenario-out stability audit**。

## 固定方法

本审计称 **B2** = 用户指定：`C_hat = ExtraTrees(U_single + Phase-3E candidate geometry)`，`U_hat = U_single + C_hat`。
这对应 Phase-4B 的 ablation **C_U_cand**（U+cand），不是 Phase-4B 表里的 geometry-only `B2_cand_geom`。

为回答“0.891 是否少数 scenario 驱动”，另报固定的 Phase-4B `B2_cand_geom`（只含 candidate geometry / leftover，不含 U_single）的 LSO 结果。不调参。

ExtraTrees 固定：{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5, 'random_state': 0, 'n_jobs': 1}。`GroupKFold(5)`，group=`scenario_id`。

## C. Fold 稳定性（指定 B2 = U+cand）

| fold | test scenarios | Spearman(C) | within C | R_B2 | R_LOCAL | top-1 | top-3 | mean U_B2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | [221000, 221005, 221010, 221015] | 0.602 | 0.604 | 0.944 | 0.829 | 0.250 | 0.750 | 1.200e+07 |
| 1 | [221004, 221009, 221014] | 0.525 | 0.445 | 0.909 | 0.633 | 0.667 | 0.667 | 1.067e+07 |
| 2 | [221003, 221008, 221013] | 0.607 | 0.653 | 0.902 | 0.737 | 0.000 | 0.333 | 9.333e+06 |
| 3 | [221002, 221007, 221012] | 0.629 | 0.628 | 1.000 | 0.762 | 0.667 | 0.667 | 1.300e+07 |
| 4 | [221001, 221006, 221011] | 0.551 | 0.393 | 0.818 | 0.807 | 0.000 | 0.000 | 9.000e+06 |

- fold R_B2 mean±std = 0.915 ± 0.066
- fold R_LOCAL mean = 0.754

## F. 每 scenario 的 recovery（必须保留恶化）

| scenario | R_LOCAL | R_B2 | dR | R_B2P4B(0.891源) | R_B3 | B2 changed | vs ORACLE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 221000 | 0.700 | 1.000 | 0.300 | 1.000 | 1.000 | 1 | 0.000 |
| 221001 | 0.917 | 0.917 | 0.000 | 0.917 | 0.917 | 0 | 0.083 |
| 221002 | 0.714 | 1.000 | 0.286 | 1.000 | 0.714 | 1 | 0.000 |
| 221003 | 0.600 | 0.900 | 0.300 | 0.900 | 0.900 | 1 | 0.100 |
| 221004 | 0.600 | 1.000 | 0.400 | 0.600 | 0.400 | 1 | 0.000 |
| 221005 | 0.615 | 0.846 | 0.231 | 0.846 | 0.846 | 1 | 0.154 |
| 221006 | 0.889 | 1.000 | 0.111 | 1.000 | 0.556 | 1 | 0.000 |
| 221007 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1 | 0.000 |
| 221008 | 0.833 | 0.917 | 0.083 | 0.917 | 0.917 | 1 | 0.083 |
| 221009 | 0.571 | 1.000 | 0.429 | 1.000 | 0.571 | 1 | 0.000 |
| 221010 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0 | 0.000 |
| 221011 | 0.615 | 0.538 | -0.077 | 0.538 | 0.615 | 1 | 0.462 |
| 221012 | 0.571 | 1.000 | 0.429 | 1.000 | 0.571 | 1 | 0.000 |
| 221013 | 0.778 | 0.889 | 0.111 | 0.889 | 0.889 | 1 | 0.111 |
| 221014 | 0.727 | 0.727 | 0.000 | 0.727 | 0.727 | 0 | 0.273 |
| 221015 | 1.000 | 0.929 | -0.071 | 0.929 | 0.929 | 1 | 0.071 |

指定 B2：mean R=0.916 median=0.964；LOCAL mean=0.758。改善 10 / 恶化 2 / 平 4。
Phase-4B geometry-only B2P4B：mean R=0.891（对应此前 0.891）。B3 full：mean R=0.785。
最大增益 scenario=221009 dR=0.429；最大恶化 scenario=221011 dR=-0.077。dominated_by_few=False。

## D. Permutation（打乱 geometry 与 candidate 对应，U_single 不动，不重训）

- 真实折上重复均值 R=0.916
- 打乱 geometry 后 R=0.680 ± 0.048
- drop=0.237

## E. Ranking consistency

- B2 改变 first-hold：13/16
- B2 与 TRAJ-ORACLE overlap：5/16
- LOCAL 与 ORACLE overlap：2/16
- 相对 LOCAL 的平均 payload 差：1.875e+06
- 相对 ORACLE 的剩余 gap：1.000e+06

## 对照

| method | mean R_pred | mean U |
|---|---:|---:|
| LOCAL | 0.758 | 9.000e+06 |
| Phase-3E locked FV | (cite) | 9.375e+06 |
| B2 specified (U+cand) | 0.916 | 1.088e+07 |
| B2 Phase-4B geom-only | 0.891 | 1.062e+07 |
| B3 +state | 0.785 | 9.312e+06 |
| TRAJ-ORACLE | 1.000 | 1.188e+07 |

## 五个问题

1. 跨 scenario 稳定性？**有限。** fold std=0.066，10 个改善、2 个恶化。这是 LSO，不是独立验证。
2. 0.891 是否少数 scenario 驱动？Phase-4B geom-only 的 LSO 均值=0.891。见每行 dR，不能只看 0.891。dominated_by_few=False。
3. 相对 LOCAL 是否稳定改善？指定 B2 mean dR=0.158，恶化数=2。
4. 是否明显优于 B3？**是**。B3 mean R=0.785。
5. 是否足够进入正式 planner integration？**否。** 没有独立标签集，未改 planner，也不能把 LSO 当部署证据。

## PASS / PARTIAL / FAIL

**PARTIAL**。无法真正独立验证。leave-scenario-out 上 B2 相对 LOCAL 有可重复增益，但不能当作 held-out deployment 证据。

本阶段停止。不修改 B2，不修改 planner，不进入 Phase-4C。

