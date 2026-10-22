# HTW Phase-4B B2 Shadow Integration Audit

本阶段只验证：把冻结 B2 的 `U_hat = U_single + C_hat` 当作 first-hold shadow score 时，能否复现已锁定的 leave-scenario-out 结果。
未训练 Low-Level，未改 reward / utility / decoder / pool / checkpoint，未改正式 planner 默认行为，未重跑 Phase-3 counterfactual，未调 ExtraTrees，未进入 Phase-4C。

**门控：PARTIAL** — shadow scorer 复现了 LSO 的 U_hat first-hold 选择，Gate-1/2 通过。仍缺少真正 held-out independent scenario，Gate-3 未定义 deployment threshold，因此不能正式替换 planner。

**planner_ready = False**。即使 Gate-1 / Gate-2 通过，也不能正式替换 planner：当前仍缺少真正 held-out independent scenario。

## 固定方法

`C_hat = ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)`，`U_hat = U_single + C_hat`。
ExtraTrees 固定：{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5, 'random_state': 0, 'n_jobs': 1}。`GroupKFold(5)`，group=`scenario_id`，同一 state 的 24 candidates 不拆分。
LOCAL = argmax U_single；B2-SHADOW = argmax U_hat；ORACLE = argmax U_traj（仅评价）。
U_single 复用冻结 Phase-4A，不重新调用 Low-Level / shadow_h15。U_traj / F / remaining_arrivals 不进入输入。

## 正式 planner 未被替换

- official decide module: `wt.htw_phase100.high_level.decide`
- 默认仍走 `decode_assignment`: True
- high_level.py 中的 B2 token: none
- sha256: {'src\\wt\\htw_phase100\\high_level.py': 'a995aa8a87c08ecf05abf28210decbdec9d6606ae057b8def80805c0a919336f', 'src\\wt\\htw_phase943\\evaluation.py': '4e7c8b082a52e4cfafcb84dd9e512cbb1f5f8a09abec42181d39074cf16e31e0', 'src\\wt\\htw_phase945\\candidate_pool.py': '93bf0a4917e6968e6b4416cb4a1daa1816e9b672b39c6703faf42be0b157fe84'}

## A. Shadow scorer consistency

- 特征键与冻结 LSO B2_KEYS 一致: True
- 无 normalization: none
- candidate_id 连续 0-23: True
- live Phase-3E geometry 与 attach_observables 一致: True (max abs err=0.0)
- 与冻结 ranking CSV 决策一致: 16/16
- shadow mean R_B2=0.916 vs locked 0.916 (0.916)
- shadow mean R_LOCAL(state)=0.758 vs locked 0.758 (0.758)
- shadow fold-mean R_LOCAL=0.754 vs locked 0.754 (0.754)
- shadow fold-mean R_B2=0.915 ± 0.066
- consistency_ok=True

上一阶段报告中的 LOCAL≈0.754 是 **fold mean**；state-mean LOCAL 是 0.758。两者都复现则不诊断为 join 错误。

| fold | test scenarios | R_B2 | R_LOCAL | top-1 | top-3 |
|---|---|---:|---:|---:|---:|
| 0 | [221000, 221005, 221010, 221015] | 0.944 | 0.829 | 0.250 | 0.750 |
| 1 | [221004, 221009, 221014] | 0.909 | 0.633 | 0.667 | 0.667 |
| 2 | [221003, 221008, 221013] | 0.902 | 0.737 | 0.000 | 0.333 |
| 3 | [221002, 221007, 221012] | 1.000 | 0.762 | 0.667 | 0.667 |
| 4 | [221001, 221006, 221011] | 0.818 | 0.807 | 0.000 | 0.000 |

## B. Shadow first-hold decision audit

| scenario | LOCAL | B2 | ORACLE | U_single_L | U_single_B2 | U_traj_L | U_traj_B2 | U_traj_O | R_L | R_B2 | dR | changed | vsO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 221000 | 19 | 20 | 7 | 3.400e+06 | 3.000e+06 | 7.000e+06 | 1.000e+07 | 1.000e+07 | 0.700 | 1.000 | 0.300 | 1 | 0 |
| 221001 | 6 | 6 | 0 | 4.000e+06 | 4.000e+06 | 1.100e+07 | 1.100e+07 | 1.200e+07 | 0.917 | 0.917 | 0.000 | 0 | 0 |
| 221002 | 8 | 6 | 6 | 5.000e+06 | 4.000e+06 | 1.000e+07 | 1.400e+07 | 1.400e+07 | 0.714 | 1.000 | 0.286 | 1 | 1 |
| 221003 | 14 | 8 | 0 | 4.200e+06 | 4.000e+06 | 6.000e+06 | 9.000e+06 | 1.000e+07 | 0.600 | 0.900 | 0.300 | 1 | 0 |
| 221004 | 21 | 19 | 19 | 4.400e+06 | 1.000e+06 | 6.000e+06 | 1.000e+07 | 1.000e+07 | 0.600 | 1.000 | 0.400 | 1 | 1 |
| 221005 | 7 | 5 | 3 | 5.000e+06 | 4.000e+06 | 8.000e+06 | 1.100e+07 | 1.300e+07 | 0.615 | 0.846 | 0.231 | 1 | 0 |
| 221006 | 15 | 8 | 2 | 4.400e+06 | 4.000e+06 | 8.000e+06 | 9.000e+06 | 9.000e+06 | 0.889 | 1.000 | 0.111 | 1 | 0 |
| 221007 | 7 | 5 | 5 | 5.000e+06 | 4.000e+06 | 1.100e+07 | 1.100e+07 | 1.100e+07 | 1.000 | 1.000 | 0.000 | 1 | 1 |
| 221008 | 8 | 6 | 7 | 5.000e+06 | 4.000e+06 | 1.000e+07 | 1.100e+07 | 1.200e+07 | 0.833 | 0.917 | 0.083 | 1 | 0 |
| 221009 | 8 | 5 | 5 | 5.000e+06 | 4.000e+06 | 8.000e+06 | 1.400e+07 | 1.400e+07 | 0.571 | 1.000 | 0.429 | 1 | 1 |
| 221010 | 5 | 5 | 5 | 4.000e+06 | 4.000e+06 | 1.400e+07 | 1.400e+07 | 1.400e+07 | 1.000 | 1.000 | 0.000 | 0 | 1 |
| 221011 | 8 | 5 | 7 | 5.000e+06 | 4.000e+06 | 8.000e+06 | 7.000e+06 | 1.300e+07 | 0.615 | 0.538 | -0.077 | 1 | 0 |
| 221012 | 7 | 4 | 0 | 5.000e+06 | 4.000e+06 | 8.000e+06 | 1.400e+07 | 1.400e+07 | 0.571 | 1.000 | 0.429 | 1 | 0 |
| 221013 | 21 | 20 | 3 | 4.400e+06 | 2.000e+06 | 7.000e+06 | 8.000e+06 | 9.000e+06 | 0.778 | 0.889 | 0.111 | 1 | 0 |
| 221014 | 19 | 19 | 2 | 5.000e+06 | 5.000e+06 | 8.000e+06 | 8.000e+06 | 1.100e+07 | 0.727 | 0.727 | 0.000 | 0 | 0 |
| 221015 | 7 | 5 | 7 | 5.000e+06 | 4.000e+06 | 1.400e+07 | 1.300e+07 | 1.400e+07 | 1.000 | 0.929 | -0.071 | 1 | 0 |

- B2 改变 first-hold: 13/16
- 改变后改善 / 恶化 / 持平: 10 / 2 / 4
- B2-ORACLE overlap: 5/16（LOCAL-ORACLE=2/16）
- 恶化 scenario: [221011, 221015]
- 恶化侧 B2 hash: ['569b38c6496e8e18d27f9ee0a962279949f822b182644d9372ccd934cf0dad24', 'b0c517dfbd28b205c8c0abb5e2e29b66113e7d75c10e6fb26cf78dc88c2eb347']
- 恶化侧是否共享同一 assignment hash: None

恶化 scenario 的 B2-LOCAL geometry 差（Phase-3E 已有特征，不新增）：

`{'idle_count': 0.0, 'idle_frac': 0.0, 'n_unique_buoys': 0.0, 'competition_count': 0.0, 'mean_uav_target_dist': 0.8333594197910799, 'max_uav_target_dist': 0.0, 'n_in_range': 0.0, 'assigned_queue_mean': 0.0, 'cand_mean_sinr': -143.63650903823486, 'cand_mean_rate': -476955.2875259066, 'cand_mean_interference': 0.0, 'assign_0': 0.0, 'assign_1': 0.0, 'assign_2': -1.0, 'assign_3': 1.5, 'assign_4': 1.0, 'assign_5': 0.0}`

改善 scenario 的对应差：

`{'idle_count': 0.3, 'idle_frac': 0.049999999999999996, 'n_unique_buoys': -0.3, 'competition_count': 0.0, 'mean_uav_target_dist': -12.760734810141182, 'max_uav_target_dist': -9.812576789809269, 'n_in_range': 0.2, 'assigned_queue_mean': 0.5666666666666667, 'cand_mean_sinr': -83.28607348546753, 'cand_mean_rate': -407117.8770176505, 'cand_mean_interference': 0.02738915246292613, 'assign_0': 0.0, 'assign_1': -0.3, 'assign_2': -0.1, 'assign_3': 0.4, 'assign_4': -4.4, 'assign_5': 0.6}`

## Decision mechanism（只统计 B2 改变的 state）

| class | n | fraction |
|---|---:|---:|
| residual-driven | 13 | 1.000 |
| local+residual | 0 | 0.000 |
| local-driven | 0 | 0.000 |
| conflicting | 0 | 0.000 |

- 改变 state 上 mean dU_single=-1.138e+06，mean dC=2.673e+06，mean dU_hat=1.535e+06
- sign(dU_single)*sign(dC)<0 的改变数: 13/13
- residual-driven 表示选了更低或持平的 U_single，但 C_hat 更高：这才是 residual correction，而不是重发现 LOCAL。

## C. Conservative deployment gate

- Gate-1 B2 mean R > LOCAL mean R: **True** (0.916 > 0.758)
- Gate-2 permutation 明显低于真实 B2: **True** (real=0.916, perm=0.680 ± 0.048, drop=0.237；沿用既有审计 drop>0.03)
- Gate-3 恶化数阈值: **未定义 deployment threshold**。工程中没有预先定义 deployment threshold，不自行发明。当前恶化数=2。

- mean/median/std R_B2 = 0.916 / 0.964 / 0.127
- mean/median/std R_LOCAL = 0.758 / 0.721 / 0.161
- improved / degraded / unchanged = 10 / 2 / 4
- top-1 / top-3 = 0.312 / 0.500
- B2-ORACLE overlap = 5/16
- average payload gain vs LOCAL = 1.875e+06
- residual gap vs ORACLE = 1.000e+06

## 对照

| method | mean R | mean U_traj |
|---|---:|---:|
| LOCAL | 0.758 | 9.000e+06 |
| Phase-3E locked FV | (cite) | 9.375e+06 |
| B2-SHADOW | 0.916 | 1.088e+07 |
| TRAJ-ORACLE | 1.000 | 1.188e+07 |

## PASS / PARTIAL / FAIL

**PARTIAL**。shadow scorer 复现了 LSO 的 U_hat first-hold 选择，Gate-1/2 通过。仍缺少真正 held-out independent scenario，Gate-3 未定义 deployment threshold，因此不能正式替换 planner。

本阶段停止。不修改 B2，不修改 planner，不进入 Phase-4C。

