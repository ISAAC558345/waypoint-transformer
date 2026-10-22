# HTW Phase-5W Residual Temporal / Joint-Execution Mechanism Audit

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练模型，没有新特征，没有打开 310118–310125。反事实执行只解释已经消耗的 development state。

## 1. 服务进度的真实语义

progress 存在每个 UAV 上。served_k 记录这份累加器属于哪一个浮标。它不是浮标资产，也不会转给另一架 UAV。

切换目标时，step 先把 progress 清零，再按新目标累加。离开服务半径、队列为空或与其他 UAV 抢同一个浮标时，这一步不累加，也不清零。

完成条件不是连续 D 步。每个有效服务步增加 (rate / reference_rate) / service_slots，service_slots = 3。累加器到达 1 后归零，并交付 1 个 block。

same_target 为真时，已有 progress 会保留。Phase-5V 的 continue_progress = same_target * progress[uav] 与这条清零规则一致。它没有表示半径外暂停，也没有把 buoy_progress 转移给新 UAV。Phase-5U 的 z_rel 还会把 progress[uav] 抄到已经切换的边上，这一项和环境语义不一致。

## 2. Continuation 与真实收益

evaluation 内部，continuation 与 J_MAPPO 的平均 Spearman 是 0.4764，与 score 的平均 Spearman 是 0.6994。
preserved progress 与 J 的 Spearman 是 0.5568，与 score 是 0.7637。
匹配后 continuation 更高但 J 更低、同时 score 更高的比例是 0.2007。

## 3. 310114 t=15

预测 `2;5;-1;4;-1;-1` 的 payload 是 4.0000e6。最优 `-1;4;2;5;-1;-1` 的 payload 是 5.0000e6。

| assignment | UAV | current | candidate | same | progress | distance | radial | heading | rate | SINR | interference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred | 0 | 2 | 2 | 1 | 0.0000 | 20.15 | -1.0000 | -1.0000 | 1.2330e+06 | 0.3198 | 3.2227e-03 |
| pred | 1 | 0 | 5 | 0 | 0.0555 | 21.60 | -0.9898 | -0.9898 | 1.1105e+06 | 0.3915 | 1.9605e-03 |
| pred | 2 | 5 | -1 | 0 | 0.0000 | NA | NA | 0.0000 | NA | NA | NA |
| pred | 3 | 4 | 4 | 1 | 0.1568 | 21.80 | -1.0000 | -1.0000 | 1.0950e+06 | 0.2985 | 2.8063e-03 |
| pred | 4 | 1 | -1 | 0 | 0.0000 | NA | NA | 0.0000 | NA | NA | NA |
| pred | 5 | 10 | -1 | 0 | 0.0000 | NA | NA | 0.0000 | NA | NA | NA |
| opt | 0 | 2 | -1 | 0 | 0.0000 | NA | NA | 0.0000 | NA | NA | NA |
| opt | 1 | 0 | 4 | 0 | 0.0555 | 21.25 | -0.9887 | -0.9887 | 1.1387e+06 | 0.4066 | 1.9559e-03 |
| opt | 2 | 5 | 2 | 0 | 0.0000 | 22.57 | -0.9831 | -0.9831 | 1.0372e+06 | 0.2492 | 3.2227e-03 |
| opt | 3 | 4 | 5 | 0 | 0.1568 | 22.14 | -0.9594 | -0.9594 | 1.0693e+06 | 0.3020 | 2.6376e-03 |
| opt | 4 | 1 | -1 | 0 | 0.0000 | NA | NA | 0.0000 | NA | NA | NA |
| opt | 5 | 10 | -1 | 0 | 0.0000 | NA | NA | 0.0000 | NA | NA | NA |

预测保留 UAV0→2 和 UAV3→4。UAV3 的 progress 是 0.1568，UAV0 是 0。最优不保留这些目标。环境没有剩余步数计数；切换会把对应 UAV 的累加器清零。

feature swap 后的 score margin（预测减最优）：

| group | margin | 反转 seed 数 |
| --- | --- | --- |
| continuation | 1.1340 | 0/5 |
| service_continuity | -0.4094 | 5/5 |
| motion_trend | 1.2052 | 0/5 |
| continuation_plus_service | -0.3934 | 5/5 |

单 UAV 切换的最大增益是 -2.0000e+06。两架同时切换的最大增益是 1.0000e+06。全部切换增益是 1.0000e+06。
SYNERGY = 3.0000e+06。可执行单切换之和是 -2.0000e+06。

| n_switch | best J | mean J | scorer 选中的 J |
| --- | --- | --- | --- |
| 0 | 4.0000e+06 | 4.0000e+06 | 4.0000e+06 |
| 1 | 2.0000e+06 | 2.0000e+06 | 2.0000e+06 |
| 2 | 5.0000e+06 | 4.5000e+06 | 5.0000e+06 |
| 3 | 2.0000e+06 | 2.0000e+06 | 2.0000e+06 |
| 4 | 5.0000e+06 | 5.0000e+06 | 5.0000e+06 |
| 5 | NA | NA | NA |
| 6 | NA | NA | NA |

预先登记的 |ΔSINR|>1 在 step 1 成立，记录为 sinr。预测与最优这时的平均 SINR 大约是 12.2 和 14.8，两边都已经在服务。payload 第一次差出一个 block 在 step 6，当时预测领先；最优在 step 8 完成 3 个 block 并反超，15 步结束时是 5 个 block 对 4 个。

## 4. 其余 residual case

310110–310117 上 regret 至少 1 个 block 的记录有 43 条，去重后 10 个 assignment 对。
SYNERGY 中位数 1.0000e+06，均值 1.3000e+06，正值比例 0.9000。
synergy ratio 中位数 0.5000。motion 近乎相同的比例 0.9000。
分叉步分布：{'1': 9, '10': 1}。

## 5. 结论

Gate 是 **W-C**。二进制 same_target 不是越高越好，所以预先定义的 CONTINUITY_BIAS 是 False。continue_progress 从 0 到 1 会单调抬高分数。TEMP_SHORTCUT_SHIFT 是 False，训练集和 evaluation 里高延续对应高收益的比例没有分开。

continuation shortcut 和联合执行增益同时存在。下一阶段优先做 executor-conditioned value，由它同时学习这两类收益。

Q1. 服务进度属于谁，切换会不会丢失？ progress 属于 UAV，并通过 served_k 绑定当前浮标。切换目标会把该 UAV 的 progress 清零。离开半径只暂停累加。

Q2. scorer 是否有 continuation shortcut？ 有一部分。continue_progress 从 0 抬到 1 会稳定抬高分数，service continuity 交换让 5 个 seed 的 310114 排名反转。same_target 本身不是越高越好。匹配反转比例 0.2007，score 与 continuation 的 Spearman 0.6994，J 的 Spearman 0.4764。

Q3. 为什么部分延续的分数更高？ 六个合法 hybrid 的分数随延续链路数下降。预测保留 2 条当前目标，分数最高，payload 是 4e6。payload 为 5e6 的两架联合切换只延续 1 条，分数排在它后面。

Q4. 单独切换一架 UAV 有没有收益？ 不能。唯一合法的单 UAV 切换增益是 -2.0000e+06。另外三架 UAV 单独改到最优目标会和仍占用该浮标的 UAV 冲突，环境会同时停止这两条链路的服务。

Q5. 是否存在联合切换增益？ 存在。ΔJ_all = 1.0000e+06。可执行单切换之和 = -2.0000e+06。SYNERGY = 3.0000e+06。放开 UAV0 并让 UAV2 接手浮标 2 的两架切换增益是 1.0000e+06，已经等于全部切换。

Q6. 轨迹从第几步分叉？ 预先登记的 |ΔSINR|>1 在 step 1 成立，来源是 sinr。两边这时都已经进入服务，SINR 大约是 12 和 15。payload 第一次差出一个 block 在 step 6；最优在 step 8 反超，最终多 1 个 block。

Q7. 剩余误差更符合哪种机制？ 二者同时存在。scorer 按延续程度给合法 hybrid 排序；多出来的 1 个 block 只有两架 UAV 一起改派才出现，一阶运动特征分不开。

Q8. 下一步做什么？ continuation shortcut 和联合执行增益同时存在。下一阶段优先做 executor-conditioned value，由它同时学习这两类收益。

`planner_ready = False`

`deployment_verified = False`
