# HTW Phase-3C：First-Hold Long-Horizon Credit Counterfactual

本阶段只做第一 hold 干预。未训练、未改模型 / checkpoint / q / L_exec / decoder / pool / Low-Level / reward / 正式 evaluation。

数据全部追溯到 Phase-3 Trajectory Utility Audit 的已冻结 rollout：

- NATURAL：`E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\traj_audit\outputs\natural_trajectories.json`
- 24-pool first-hold + 后续正式 greedy：`E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\traj_audit\outputs\counterfactual.json`
- 官方 first-hold pool：`E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\lowlevel_v2\traj_audit\outputs\first_hold_pool.csv`

四种条件里，**只有第一 hold 被替换**。hold 2/3/4 仍是当时正式 greedy decoder 的真实闭环，不是 4-hold oracle。未新增随机 rollout；`POOL-UNIFORM` 只是已有 24 条合法 first-hold 的均值参考。

**EXEC：NATURAL=5.812e+06，LOCAL-ORACLE=9.000e+06，TRAJ-ORACLE=1.188e+07。** 与 Phase-3B 的 5.81e6 / 9.00e6 / 11.88e6 一致。

## 1. 四方法对照

| method | NATURAL | LOCAL-ORACLE | TRAJ-ORACLE | Δ_local | Δ_traj | R=Δ_local/Δ_traj |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 1.088e+07 | 1.169e+07 | 1.381e+07 | 8.125e+05 | 2.938e+06 | 0.278 |
| ABS | 9.812e+06 | 1.112e+07 | 1.312e+07 | 1.312e+06 | 3.312e+06 | 0.446 |
| PREFIX | 1.156e+07 | 1.031e+07 | 1.438e+07 | -1.250e+06 | 2.812e+06 | -0.637 |
| EXEC | 5.812e+06 | 9.000e+06 | 1.188e+07 | 3.188e+06 | 6.062e+06 | 0.479 |

`R` 只在该 scenario 的 `Δ_traj>0` 上计算。PREFIX 的 LOCAL-ORACLE 低于 NATURAL（OOD greedy 已经好于 pool `argmax U_single`）。

POOL-UNIFORM（已有 24 条均值，不是新实验）：

| method | POOL-UNIFORM U_traj |
|---|---:|
| OLD | 8.208e+06 |
| ABS | 7.990e+06 |
| PREFIX | 9.419e+06 |
| EXEC | 5.935e+06 |

## 2. EXEC 专项

| condition | mean | median | p25 | p75 | p90 | std |
|---|---:|---:|---:|---:|---:|---:|
| NATURAL | 5.812e+06 | 5.000e+06 | 2.750e+06 | 9.250e+06 | 1.000e+07 | 3.487e+06 |
| LOCAL-ORACLE | 9.000e+06 | 8.000e+06 | 7.750e+06 | 1.025e+07 | 1.250e+07 | 2.477e+06 |
| TRAJ-ORACLE | 1.188e+07 | 1.200e+07 | 1.000e+07 | 1.400e+07 | 1.400e+07 | 1.893e+06 |

- LOCAL-ORACLE 相对 NATURAL：Δ_local mean=3.188e+06，median=3.500e+06。
- TRAJ-ORACLE 相对 NATURAL：Δ_traj mean=6.062e+06，median=6.000e+06。
- 恢复比例 R mean=0.479，median=0.500（n=15 个 Δ_traj>0 的 scenario）。
- LOCAL / TRAJ 水平比=0.758。
- PREFIX NATURAL=1.156e+07。EXEC TRAJ-ORACLE=1.188e+07。

只修正第一 hold：

1. 当前 policy 的 first-hold 确实差：NATURAL 5.812e+06。
2. 只改成 `argmax U_single` 能恢复一部分：5.812e+06 → 9.000e+06。
3. 仍有一块来自 long-horizon 选点：LOCAL 9.000e+06 vs TRAJ 1.188e+07，缺口 2.875e+06。

## 3. 第一 hold 后续状态效应

对每个合法 first-hold Γ：`F(Γ)=U_traj(Γ)-U_single(Γ)`。图见 `outputs/figures/u_single_vs_F.png`。

| method | Pearson(U,F) | Spearman(U,F) | Kendall(U,F) | NATURAL F | LOCAL F | TRAJ F |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 0.292 | 0.321 | 0.223 | 7.200e+06 | 7.075e+06 | 1.016e+07 |
| ABS | 0.078 | 0.054 | 0.059 | 6.038e+06 | 6.512e+06 | 9.938e+06 |
| PREFIX | -0.101 | -0.042 | -0.022 | 7.288e+06 | 5.700e+06 | 1.236e+07 |
| EXEC | 0.129 | 0.211 | 0.136 | 5.125e+06 | 4.388e+06 | 9.625e+06 |

EXEC：U_single 与 F 的 Spearman=0.211。TRAJ-ORACLE 的 F mean=9.625e+06，高于 LOCAL 4.388e+06 与 NATURAL 5.125e+06。
第一 hold 的轨迹价值大量通过后续状态体现：traj-oracle 选的往往不是最高 U_single，而是更高的 F。

## 4. 四个问题

**Q1. EXEC 有多少可以只靠修正 first-hold 恢复？**  
相对 NATURAL，LOCAL-ORACLE 抬升 3.188e+06（到 9.000e+06），TRAJ-ORACLE 抬升 6.062e+06（到 1.188e+07）。
只改第一 hold、后续仍正式 greedy，可以把 seed0 从 5.812e+06 拉到最多 1.188e+07，已经贴近 PREFIX NATURAL 1.156e+07。
其中 local ranking 只贡献 R=0.479 的可恢复增益。

**Q2. argmax U_single 是否已经足以接近 argmax U_traj？**  
否。LOCAL=9.000e+06，TRAJ=1.188e+07，水平比=0.758，剩余 2.875e+06。

**Q3. 剩余 gap 能否归因于 long-horizon credit / state-transition？**  
可以。控制变量只有第一 hold；后续 greedy、pool、decoder 都冻结。LOCAL 与 TRAJ 的差只能来自“第一 hold 按 U_single 还是按 U_traj 选”，而 TRAJ 的优势主要来自更高的 F（后续状态）。不是 4-hold 全轨迹 oracle。

**Q4. 下一阶段最小假设？**  
**B** — 只改第一 hold 就够，但必须是 trajectory-aware credit，而不是 argmax U_single。
排除：C（U_single 沿路径可加，不是错误标签）；D（traj-oracle 第一 hold 已贴近 PREFIX，不必先重训整个 Low-Level）。A 单独不够，因为 R 只有一半左右。

本阶段停止。不要训练。不要改方法或 checkpoint。不要进入 End-to-End Independent Validation。
