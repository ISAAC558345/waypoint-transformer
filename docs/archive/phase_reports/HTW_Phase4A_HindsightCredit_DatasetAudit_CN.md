# HTW Phase-4A Hindsight Credit Dataset Audit

本阶段只构建并审计 offline candidate-level hindsight credit。未训练，未改 Low-Level / Actor / Critic / reward / decoder / pool / checkpoint，未改 `U_single` / `U_traj`，未引入 predictor / timed subgoal / joint training / HIRO，未重跑 Phase-3C-3H，未调用 `traj_audit.run`。

**门控：PASS**

## 1. 数据来源

优先复用已冻结 Phase-3 rollout，不重复闭环。

- pool: `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\traj_audit\outputs\first_hold_pool.csv`
- counterfactual: `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\traj_audit\outputs\counterfactual.json`
- Phase-3C candidates: `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\phase3c_credit_counterfactual\outputs\first_hold_all_candidates.csv`
- Phase-3C interventions: `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\phase3c_credit_counterfactual\outputs\first_hold_counterfactual.csv`

`C_abs` 直接复制 Phase-3C 的 `F`。`U_single` 来自 `first_hold_U_single`（= `U_LL` / `U_LL_first`）。`U_traj` 来自 `final_U_traj`（= counterfactual `U_traj`）。

## 2. Rollout protocol（冻结，未重跑）

与 Phase-3 trajectory audit 同一条链：

- first-hold: `generate_pool(env, seed_key=scenario*1009+0)` 的 24 个合法 Gamma
- `U_single`: `Auditor.u_ll` 对 5 个 Low-Level seed 的 `shadow_h15` 取均值
- 闭环: `FirstHoldOverrideWorker` 只覆盖 first-hold；hold 2/3/4 走正式 `decide()` / `decode_assignment()` joint greedy
- Low-Level: `ckpt_seed0_update100.pt`，`deterministic=True`
- `U_traj = payload_bits(end) - payload_bits(reset)`，horizon=48，hold=15，4 个 decision hold

正式 NATURAL first-hold 是 High-Level joint greedy，不是 `argmax U_single`。EXEC 的 16 条 NATURAL 全部不在 24-pool 内。本数据集的 24 点是 pool 干预，不是 NATURAL 解码。

## 3-5. 规模

- family: 4（OLD, ABS, PREFIX, EXEC）；主判断 **EXEC**
- state / scenario: 16（221000-221015）；`state_id = scenario_id`（t=0）
- 每 state candidate: [24]
- 总行数: 1536；EXEC 行数: 384
- 闭环 seed: `seed=0`（`LL_RUN`）
- U_single seeds: 0,1,2,3,4（均值，不另定义）
- checkpoint_id: `ckpt_seed0_update100`

## 6. U_single / U_traj 对齐

- max |U_single - pool U_LL| = 0.000e+00
- max |U_single - cf U_LL_first| = 0.000e+00
- max |U_traj - cf U_traj| = 0.000e+00
- max |C_abs - F| = 0.000e+00
- max |U_traj - (U_single+C_abs)| = 0.000e+00

| condition | locked EXEC | this table |
|---|---:|---:|
| NATURAL | 5.812e+06 | 5.812e+06 |
| LOCAL-ORACLE | 9.000e+06 | 9.000e+06 |
| TRAJ-ORACLE | 1.188e+07 | 1.188e+07 |

从表内 argmax 重建: LOCAL=9.000e+06，TRAJ=1.188e+07。

## 7. C_abs 分布（EXEC pool 384 点）

- mean=4.453e+06 median=3.700e+06 std=2.901e+06
- p25=2.000e+06 p75=7.000e+06 min=0.000e+00 max=1.200e+07

## 8. C_ratio 分布（EXEC）

`C_ratio=(U_traj-U_single)/(|U_single|+1.0)`，epsilon=Phase-3 `UTILITY_EPS`。

- mean=549482.155 median=3.500 std=1487158.816
- p25=1.500 p75=9.000 min=0.000 max=10000000.000

C_ratio 的均值被 `U_single=0` 的行拉爆：此时分母=epsilon=1，C_ratio=C_abs。看分布请优先用 median / p25 / p75，不要用 mean 当主指标。

## 9. Within-state 统计（EXEC）

- pooled Spearman(U_single, U_traj)=0.515
- pooled Spearman(C_abs, U_traj)=0.904
- pooled Spearman(C_abs, U_traj-U_single)=1.000（应为 1）
- within-state Spearman(U_single, U_traj) median=0.483 mean=0.518
- within-state Spearman(C_abs, U_traj) median=0.889 mean=0.884
- within-state Spearman(C_abs, U_single) median=0.240

这些是描述统计，不是模型精度。

## 10. argmax U_single 到 U_traj recovery（EXEC）

`R_local = U_traj(argmax U_single) / max_i U_traj(i)`，按 state 再平均。

- argmax overlap mean=0.125
- R_local mean=0.758 median=0.721
- mean U_traj(argmax U_single)=9.000e+06
- mean max U_traj=1.188e+07

这不是 predictor accuracy。它只回答：若仍按 U_single 选 first-hold，相对 pool 内最优 U_traj 丢掉多少。

## 11. Between / within-state 方差（EXEC C_abs）

- total Var(C)=8.414e+12
- mean within-state Var(C)=8.152e+12（share=0.969）
- between-state Var(state-mean C)=6.185e+11（share=0.074）

C 不是纯 state offset：组内仍有 candidate 变异。同时 C 也不是与 U_single 无关的独立量。

## 7 补充：C 与 first-hold geometry（描述，非训练）

- idle_count: pooled Spearman(C)=0.059，within median=0.022
- mean_uav_target_dist: pooled Spearman(C)=0.086，within median=0.115
- n_in_range: pooled Spearman(C)=-0.007，within median=-0.061
- n_unique_buoys: pooled Spearman(C)=-0.053，within median=-0.038
- assigned_queue_mean: pooled Spearman(C)=0.122，within median=0.122

特征来自 t=0 `reset(seed=scenario)` 上的 first-hold 几何，不含未来 hold / U_traj / F。未拟合任何模型。

## 12. 与 Phase-3 的一致性

EXEC NATURAL / LOCAL / TRAJ 均值与 Phase-3 锁定值一致（允许显示精度）。`C_abs=F`。未改 utility。未覆写任何 Phase-3 文件。

## 13. Rollout 随机性

- Low-Level policy: `deterministic=True`，用 actor 均值，不采样。
- High-Level decoder: `pick_candidate` 默认无 rng，平局走 `canonical_keys`。
- 环境到达: `arrivals_at(t)` 由 t 决定，无逐步 RNG。reset 的 UAV y jitter 由 scenario seed 决定。
- 本阶段没有重新 step。同一 candidate 的 U_traj 以冻结 JSON 为准。
- 若将来重放: 必须固定 scenario seed、`LL_RUN=0`、同一 High-Level family；禁止与 shadow worker 共用同一个 env。

## 14. 是否可直接用于下一阶段 credit learner

可以作为 offline 监督标签: 键 `(family, scenario_id, candidate_id)` 唯一，`C_abs` 与 Phase-3 `F` 逐行相同，EXEC 三条件对齐。

不能当作已经证明“学 C 一定比学 F / 学 U_traj 更容易”。本阶段没有训练。组内 Spearman(C, U_traj) 高，部分是因为 U_traj=U_single+C 的代数结构，不是新信息。下一阶段若做 learner，必须另开指令，且不能把 U_traj / 后续 hold 当 first-hold 特征。

## PASS / FAIL

**PASS**。判据: candidate → 冻结 Low-Level → 完整 4-hold trajectory → U_traj → `C_abs=U_traj-U_single` 的数据链由 Phase-3 冻结表稳定复现；EXEC NATURAL/LOCAL/TRAJ 与锁定值一致；identity 逐行成立。

本阶段停止。不要训练。不要改 planner。不要进入 Phase-4B。

