# Phase-5ZE：一步执行响应蒸馏

本阶段只训练 `OneStepExecutorResponsePredictor`。输入是当前 state、候选 assignment，以及冻结 MAPPO 的 ACT0。输出是 Phase-5ZD 已经冻结的 14 维 H=1 response。这个预测再送入冻结的 `ExecutorConditionedJointScorer`。

价值网络、HGNN、REP、TEMP、候选池和 MAPPO 都没有更新。预测器的损失只有归一化连续通道上的 Huber，以及三个二值通道上的 BCE，权重 1:1。J15 不进入这个损失。

预测器不是新的低层策略，不是中间决策器，也不是另一个 assignment selector。最终 assignment 仍由高层价值函数一次选出。option hold 仍是 15。H_dyn = 1 只表示执行信息第一次出现的观察尺度，高层不因此改成每步重选。

结构是 local MLP（宽度 128）加上 UAV 均值，再接 response MLP。没有新的 GNN、attention 或 Transformer。5 个 seed 都保存 epoch-10 之后，才读取 310110–310117。predictor seed k 只配 Phase-5ZD value seed k。交叉 seed 只作审计，没有用来挑组合。

Gate = `DISTILL-C`。`FROZEN_FOR_FRESH_CONFIRMATION = False`。

`planner_ready = False`

`deployment_verified = False`

没有打开 310118–310125，没有调用 deployment 决策接口，没有进入 deployment verification。

真实 H1 路径可以复现 Phase-5ZD。R1 − S 的 mean selected J15 仍是 +4.5000e5 bit，scenario-cluster 95% CI 是 [1.6667e5, 7.4167e5]，5/5 个 seed 都为正。调用方环境在 shadow 之后的最大绝对变化是 0。UAV 置换的最大绝对误差是 1.91e-6。浮标重编号没有现成的 encoder API，记为 not_applicable。

## 下游 selected J15

主指标是 argmax 之后读到的真实 J15。下表是确认集、5 个 seed、全部决策时刻。

| 路径 | mean J15 | median J15 | normalized regret | Top-1 | Top-3 | Top-5 | Spearman | 与 R1 的 assignment 一致率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S 冻结静态 | 2.3417e6 | 3.0000e6 | 0.1958 | 0.6917 | 0.8417 | 0.9083 | 0.5766 |  |
| P1 预测 H1 | 2.3083e6 | 3.0000e6 | 0.2458 | 0.5833 | 0.8250 | 0.9583 | 0.5727 | 0.3500 |
| R1 真实 H1 | 2.7917e6 | 3.0000e6 | 0.0625 | 0.8750 | 0.9583 | 0.9583 | 0.8008 |  |
| REF8 H=8 Score-B | 2.8750e6 | 3.0000e6 | 0.0625 | 0.9167 | 1.0000 | 1.0000 | 0.8193 |  |
| O 真实 J15 | 2.9583e6 | 3.0000e6 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |

## Q1. 各字段能否从 state、assignment 和 ACT0 预测出来？

能预测的是几何和二值执行状态。预测不好的是 SINR 的数值。确认集上 5 个 seed 合在一起：

| 字段 | MAE | R² | Spearman |
| --- | --- | --- | --- |
| sinr_1 | 1.083 | 0.279 | 0.724 |
| delta_sinr | 1.238 | 0.313 | 0.815 |
| interference_1 | 0.133 | 0.344 | 0.765 |
| delta_interference | 0.232 | 0.421 | 0.817 |
| service_progress_1 | 0.0179 | 0.653 | 0.754 |
| delta_service_progress | 0.0579 | 0.737 | 0.895 |
| rate_1 | 3.220e5 bit/s | 0.623 | 0.788 |
| distance_to_target_1 | 0.435 m | 0.816 | 0.845 |
| delta_distance_to_target | 0.595 m | 0.983 | 0.341 |
| radial_motion_1 | 0.584 m | 0.981 | 0.397 |
| heading_alignment_1 | 0.0847 | 0.774 | 0.471 |
| assigned_in_range_1 | 0.00037 | 1.000 | 0.792 |
| service_active_1 | 0.0027 | 0.991 | 0.732 |
| same_target_after_step_1 | 0.0022 | 0.999 | 0.468 |

距离变化和径向位移的 R² 很高，Spearman 却只有 0.34 和 0.40，因为这两个量在一步之内很集中，排序相关会被并列值压低。二值通道的 MAE 接近 0。干扰的排序相关约 0.77–0.82，但 R² 只有 0.34–0.42。

## Q2. SINR

SINR_1：MAE = 1.083，R² = 0.279，Spearman = 0.724，Pearson = 0.565，bias = −0.490。

排序比数值好，但绝对值明显偏低。Phase-5ZD 用到的正是这个数值差，而不是一个粗糙的名次。

## Q3. service progress

service_progress_1：MAE = 0.0179，R² = 0.653，Spearman = 0.754，Pearson = 0.834，bias = +0.0026。

delta_service_progress 的 R² 是 0.737，Spearman 是 0.895。进度比 SINR 好预测。它没有单独保住下游的 assignment。

## Q4. P1 相对 S 的 selected J15

没有提高。P1 − S 的均值是 **−3.3333e4 bit**。scenario-cluster 95% CI 是 **[−2.3333e5, +1.6667e5]**，区间跨过 0。

P1 的 Top-3 是 0.8250，normalized regret 是 0.2458。两条都比冻结静态对照差，也没有达到 0.85 和 0.12。

## Q5. 保留了多少真实 H1 增益？

整体 retention = Gain_P1 / Gain_R1 = **−0.0741**。预测响应没有保留 Phase-5ZD 的 +4.5000e5 bit，方向是反的。Gain_R1 为 0 的 310112、310116、310117 不进入比值。其余 scenario 的 retention 分别是 310110 −1.75、310111 −0.133、310113 −0.600、310114 +0.333、310115 +0.600。

## Q6. 相对真实 H1 损失多少？

P1 − R1 的均值是 **−4.8333e5 bit**，95% CI 是 **[−8.0000e5, −1.8333e5]**，整个区间都在 0 以下。5/5 个 seed 都是 P1 低于 R1。assignment 一致率只有 0.3500。

## Q7. 5 个 seed 是否方向一致？

不一致。P1 − S 的五个方向是正、负、正、负、负。只有 seed 0 和 seed 2 的平均 selected J15 高于 S。seed 2 是 2.5000e6 对 S 的 2.2917e6；seed 1 和 seed 4 都是 2.2083e6，低于对应的 S。协议使用全部 5 个配对 seed，没有在看到确认集之后改选 seed 2。

R1 − S 仍然是 5/5 为正，而且五个 seed 的 R1 都是 2.7917e6。

## Q8. 8 个 scenario 里有多少提高、持平、下降？

P1 相对 S：提高 2，持平 3，下降 3。

提高的是 310114（+3.3333e5）和 310115（+4.0000e5）。下降的是 310110（−4.6667e5）、310111（−1.3333e5）、310113（−4.0000e5）。持平的是 310112、310116、310117。

P1 相对 R1：提高 0，持平 3，下降 5。

## Q9. 310114 t=15 上 A_opt 排第几？

A_pred 的 J15 是 4e6，A_opt 的 J15 是 5e6。

| 路径 | 5 个 seed 的 A_opt rank |
| --- | --- |
| S | 53，54，53，54，52 |
| P1 | 74，72，76，85，74 |
| R1 | 2，3，3，2，2 |

预测响应没有保住 Top-3。它把这个反例排得比静态对照更靠后。五个 seed 里，P1 的 V(A_pred) 都高于 V(A_opt)。seed 0 是 2.2049 对 0.6549；R1 是 3.1529 对 3.2622。

原因在 SINR 顺序被预测反转。只看已分配 UAV 的平均，seed 0：

| assignment | 真实 SINR | 预测 SINR | 真实 progress | 预测 progress |
| --- | --- | --- | --- | --- |
| A_pred | 12.17 | 5.58 | 0.167 | 0.157 |
| A_opt | 14.82 | 2.05 | 0.133 | 0.045 |

真实一步之后 A_opt 的 SINR 更高。预测器把两边都压低，而且把 A_opt 压得更低。进度也把 A_opt 从 0.133 估到 0.045，A_pred 基本保持。

这个状态上，P1 选中的 J15 是 4e6，S 是 2e6，R1 是 5e6。所以 scenario 310114 的平均 J15 可以上升，同时 A_opt 的名次变差：P1 更稳定地选中 A_pred，没有回到 5e6 的最优分配。

## Q10. 10 个 residual cases

P1 相对 S：修复 5，不变 3，恶化 2。

R1 已经高于 S 的修复里，P1 仍高于 S 的有 5 条，丢掉的有 1 条。丢掉的是 310110 t=0：S 2.2e6，P1 1.0e6，R1 3.0e6。另一条恶化是 310114 t=0：S 和 R1 都是 3e6，P1 降到 2e6。

310114 t=15 这一条，P1 从 S 的 2e6 升到 4e6，R1 是 5e6。它算部分修复，不是 A_opt 回到 Top-3。

## Q11. 哪类预测误差更接近选错 assignment？

状态级别上，和 P1/R1 assignment disagreement 相关最强的是 SINR 误差，Spearman 0.534。motion 是 0.228。progress 是 −0.172，interference 是 −0.468。progress 误差不是选错 assignment 的主要伴生量。

assignment 级别上，整体 response MAE 和 |V_P1 − V_R1| 的 Spearman 是 0.722。预测偏差确实会推动价值，但推回去的位置不对。

按状态的 SINR MAE 分成四档之后，P1 的 normalized regret 不是“误差越大、regret 越高”：

| SINR 误差档 | 平均 SINR MAE | disagreement | P1 normalized regret |
| --- | --- | --- | --- |
| Q1 | 0.437 | 0.300 | 0.383 |
| Q2 | 0.659 | 0.767 | 0.256 |
| Q3 | 1.064 | 0.533 | 0.200 |
| Q4 | 2.483 | 1.000 | 0.144 |

误差最大的一档 disagreement 是 1，但 regret 反而最低。SINR 误差和 selected-J15 gap 的 Spearman 只有 0.149。因此不能把平均 regret 直接归到“SINR MAE 太大”。已经能定位的失败是跨 assignment 的 SINR 顺序，310114 t=15 就是例子。

推理期把预测通道换成训练集均值，不重新训练：SINR disagreement 0.058，progress 0.050，interference 0.017，motion 0.042。selected J15 的变化都远小于一个 block。换成均值之后决策几乎不动，说明当前预测响应没有形成稳定、可分离的赋值信号。

## Q12. 120 个候选要多久？

同一个已经到达决策时刻的 state，warm-up 之后重复 100 次：

| 路径 | encoder | MAPPO ACT0 | clone | env step | extract | predictor | value | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 真实 H1 | 2.1 ms | 138.5 ms | 17.4 ms | 294.8 ms | 58.9 ms |  | 3.5 ms | 515.3 ms |
| 预测 H1 | 2.3 ms | 109.1 ms | 0 | 0 | 0 | 3.6 ms | 3.4 ms | 118.4 ms |
| 静态 | 2.3 ms | 0 | 0 | 0 | 0 | 0 | 3.0 ms | 5.3 ms |

预测路径没有 environment clone，也没有 environment step。调用方最大绝对变化是 0。相对真实 H1 的 speedup 是 **4.35 倍**。速度门槛本身达到了。时间现在主要在 120 次冻结 actor forward，价值网络和预测器各约 3.5 ms。

这里的 encoder 只计决策时刻上的一次前向。它不是把 episode 重放到该时刻的时间。

## Q13. 是否达到 DISTILL-A？

没有。Gate 是 **DISTILL-C**。

真实 H1 仍然稳定优于静态对照，所以不是 DISTILL-D。预测 H1 的平均 selected J15 不高于静态对照，所以也不是“有提升但不稳定”的 DISTILL-B。

同时没过的还有：CI 下界小于 0，只有 2/8 个 scenario 上升、3 个下降，只有 2/5 个 seed 上升，Top-3 = 0.825，normalized regret = 0.246，retention = −0.074，相对 R1 下降 4.8333e5 bit，310114 的 A_opt 没有进入 Top-3。

速度和置换审计是通过的。它们不能把一个没有价值增益的预测器标成 DISTILL-A。

## Q14. 现在能不能冻结结构并打开新的确认集？

不能。预测器没有留住已经确认的 executor-conditioned value 增益，没有证据进入 fresh confirmation，也不打开 310118–310125。

下一阶段只允许定位误差来源：SINR、progress、interference、motion 里，哪一项预测误差导致决策失败。本阶段已经看到的优先对象是跨 assignment 的 SINR 顺序，而不是进度回归，也不是一步几何。不允许为此加大网络、加 GNN、加 Transformer，也不允许改 HGNN 或 MAPPO。

`planner_ready` 仍然是 false。`deployment_verified` 仍然是 false。本阶段没有宣布 deployment PASS。
