# HTW Phase-5X Frozen Executor Short-Horizon Lookahead

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练网络，没有修改 planner，没有调用 `decide()`，没有打开 310118–310125。每个候选分配只在环境副本上交给冻结 MAPPO 执行。H=15 是完整执行的上界，不是本阶段的方法。

门控分数是预先登记的 Score-B。Score-A 保留为对照。

## 1. Shadow integrity

检查了 6491 次滚动。主环境最大绝对变化是 0.0。检查项包括 UAV 位置、队列、浮标、`served_k`、service progress、assignment、`last_sinr`、`last_rate`、累计 payload、dropped、episode step 和 RNG 状态。速度不是 `L3BEnv` 的字段。调用方保存的上一拍动作也没有被改写。H=15 的 payload 与已经保存的 J15 标签一致，偏差超过 1 bit 时运行会中止。

## 2. 两个预先登记的分数

Score-A 是前 H 步已经交付的 payload。Score-B 是这段 payload，加上已经发生的 progress 之和乘以一个 block（1e6 bit）。环境在 progress 达到 1 时正好交付一个 block 并清零，所以 Score-B 只使用已经发生的进度，没有用未来 SINR 估计未来 progress。

confirmation 的 24 个状态里，H=1 有 22 个状态的全部候选 payload 都是 0，H=5 仍有 15 个。Score-A 的 optimum rank 中位数因此等于 1：它只表示没有候选拿到更高的 payload，不表示选中了最优分配。

| model | H | Spearman | Top-1 | Top-3 | Top-5 | normalized regret | optimum rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase5u | 0 | 0.5408 | 0.5750 | 0.7083 | 0.7083 | 0.2736 | 1.0 |
| phase5v | 0 | 0.6181 | 0.6417 | 0.8583 | 0.9167 | 0.1903 | 1.0 |
| payload | 1 | 0.5661 | 0.1250 | 0.4583 | 0.5417 | 0.7056 | 1.0 |
| exec | 1 | 0.7816 | 0.8333 | 1.0000 | 1.0000 | 0.1042 | 1.0 |
| payload | 3 | 0.7491 | 0.2500 | 0.5000 | 0.5833 | 0.6083 | 1.0 |
| exec | 3 | 0.8001 | 0.8333 | 1.0000 | 1.0000 | 0.1042 | 1.0 |
| payload | 5 | 0.3807 | 0.2500 | 0.5417 | 0.5833 | 0.5528 | 1.0 |
| exec | 5 | 0.8081 | 0.8333 | 1.0000 | 1.0000 | 0.1042 | 1.0 |
| payload | 8 | 0.5788 | 0.6667 | 0.8333 | 0.8750 | 0.2361 | 1.0 |
| exec | 8 | 0.8193 | 0.9167 | 1.0000 | 1.0000 | 0.0625 | 1.0 |
| payload | 10 | 0.7385 | 0.7917 | 0.8333 | 0.9167 | 0.0986 | 1.0 |
| exec | 10 | 0.8197 | 0.9167 | 0.9583 | 1.0000 | 0.0625 | 1.0 |
| payload | 15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0 |
| exec | 15 | 0.7941 | 0.9167 | 0.9583 | 1.0000 | 0.0625 | 1.0 |

H=15 的 Score-A 与 J15 相同，所以它的排序指标是 1。Score-B 在 H=15 仍把残余 progress 加回去，normalized regret 留在 0.0625。310117 t=0 就是这个例子：Score-B 到 H=15 仍有 1e6 bit regret，Score-A 在 H=15 把它降到 0。

## 3. 相对 Phase-5V

ΔJ15 是所选分配的 15 步 payload 减去 Phase-5V 所选分配。bootstrap 以 scenario 为 cluster，10000 次，种子 0。正比例是 state×seed 上严格变好的比例。很多状态与 Phase-5V 持平，所以正比例低于 scenario 上升比例。没有 scenario 在 Score-B 上变差。

Score-B：

| H | ΔJ15 mean | 95% CI | ΔTop-3 | Δnormalized regret | scenario 上升 | scenario 下降 | 正比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.5833e+05 | [1.5833e+05, 5.8333e+05] | 0.1417 | -0.0861 | 5/8 | 0/8 | 0.2333 |
| 3 | 3.5833e+05 | [1.5833e+05, 5.8333e+05] | 0.1417 | -0.0861 | 5/8 | 0/8 | 0.2333 |
| 5 | 3.5833e+05 | [1.5833e+05, 5.8333e+05] | 0.1417 | -0.0861 | 5/8 | 0/8 | 0.2333 |
| 8 | 4.8333e+05 | [2.5000e+05, 6.8333e+05] | 0.1417 | -0.1278 | 6/8 | 0/8 | 0.3167 |
| 10 | 4.8333e+05 | [2.5000e+05, 6.8333e+05] | 0.1000 | -0.1278 | 6/8 | 0/8 | 0.3167 |

H=1、3、5 的 Score-B 选出同一批分配，所以这三行相同。5 个 seed 的平均 ΔJ15 都为正。H=8 再修正 310111 t=30 和 310116 t=15，mean ΔJ15 从 3.5833e5 升到 4.8333e5，normalized regret 从 0.1042 降到 0.0625。

Score-A 对照：

| H | ΔJ15 mean | 95% CI | ΔTop-3 | Δnormalized regret | scenario 上升 | scenario 下降 | 正比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -1.4750e+06 | [-1.9167e+06, -9.9167e+05] | -0.4000 | 0.5153 | 0/8 | 8/8 | 0.0250 |
| 3 | -1.1833e+06 | [-1.7917e+06, -5.7500e+05] | -0.3583 | 0.4181 | 0/8 | 7/8 | 0.0250 |
| 5 | -1.0167e+06 | [-1.5000e+06, -5.0833e+05] | -0.3167 | 0.3625 | 0/8 | 7/8 | 0.0250 |
| 8 | -1.6667e+04 | [-3.7500e+05, 3.0000e+05] | -0.0250 | 0.0458 | 4/8 | 3/8 | 0.1917 |
| 10 | 2.7500e+05 | [8.3333e+04, 4.8333e+05] | -0.0250 | -0.0917 | 4/8 | 0/8 | 0.1917 |

只按已交付 payload 排序时，H≤5 稳定差于 Phase-5V。H=10 的所选 J15 才整体变好，Top-3 仍略降。

## 4. 310114 t=15

Phase-5V predicted 是 `2;5;-1;4;-1;-1`，J15 = 4e6。命名 optimal 是 `-1;4;2;5;-1;-1`，J15 = 5e6。rank 是该字符串在 120 个候选中的名次。两边 payload 都是 0 时，payload rank 为 1 只是平局。

| H | pred payload | opt payload | pred SINR | opt SINR | pred blocks | opt blocks | payload rank | exec rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 12.1743 | 14.8150 | 0 | 0 | 1 | 11 |
| 3 | 0 | 0 | 12.1743 | 14.8150 | 0 | 0 | 1 | 4 |
| 5 | 0 | 0 | 12.1743 | 14.8150 | 0 | 0 | 1 | 3 |
| 8 | 2e6 | 3e6 | 11.3462 | 14.8150 | 2 | 3 | 1 | 1 |
| 10 | 3e6 | 3e6 | 26.1264 | 78.5187 | 3 | 3 | 1 | 1 |
| 15 | 4e6 | 5e6 | 45.8332 | 163.4569 | 4 | 5 | 1 | 1 |

Score-B 在 H=1 的 argmax 已经是另一个 J15 = 5e6 的联合分配 `-1;5;2;4;-1;-1`，所选 regret 为 0。命名 optimal 这时的 Score-B 名次是 11，因为 predicted 的 progress 和是 0.4996，optimal 是 0.3990。从 H=3 起，命名 optimal 的 Score-B 高于 predicted，并且一直保持到 H=15。它在 H=5 进入 Top-3，在 H=8 成为 argmax。

Score-A 上，两边 payload 在 H=5 之前都是 0。H=8 时 optimal 是 3e6、predicted 是 2e6。H=10 两边都是 3e6。H=15 才是 5e6 对 4e6。严格的稳定领先因此从 H=15 才开始。

## 5. Synergy、residual 与分叉

协同只统计合法单切换。Score-B 的 synergy 中位数在 H=1 已经是 +1.240e4 bit，但 310114 这个案例本身是 −1.447e5。案例 synergy 在 H=5 转为 +3.466e4，在 H=8 达到 +1.000e6，与 Phase-5W 的整段协同同量级。预先登记的“中位数为正且 310114 为正”从 H=5 成立，块量级的协同从 H=8 才清楚。

Score-A 的 synergy 中位数在 H=10 及以前不大于 0，到 H=15 才是 +1.000e6。

Score-B 的 correction horizon：310110 t=0 为 1，310111 t=15 的两例为 1，310111 t=30 为 8，310113 t=15 为 1，310114 t=0 为 1，310114 t=15 为 1，310115 t=15 为 1，310116 t=15 为 8，310117 t=0 到 H=15 仍未修正。不超过 5 的比例是 0.70，不超过 8 的比例是 0.90。

10 个 residual pair 的分叉 horizon：SINR 中位数 1.0、IQR 10.5；progress 中位数 1.0、IQR 2.0；payload 中位数 10.0、IQR 3.75。7/10 的 SINR 在 H=1 已经分开，payload 多数要到 H=8–15。

## 6. 运行时间

下表是同一次 15 步副本滚动里，到达该 H 的累计墙钟时间，不是六次独立运行。

| H | 每个 assignment ms | 120 个 assignment 的决策 ms |
| --- | --- | --- |
| 1 | 5.37 | 645.0 |
| 3 | 15.84 | 1901.2 |
| 5 | 26.46 | 3175.0 |
| 8 | 42.22 | 5066.7 |
| 10 | 52.27 | 6272.5 |
| 15 | 75.38 | 9045.3 |

本阶段不设 deployment 延迟阈值。以这次测量为准，120 个候选做到 H=3 约 1.9 s，做到 H=5 约 3.2 s，做到 H=8 约 5.1 s。

## 7. 结论

Gate 是 **X-C**。H* = 8。

H=1 到 H=5 的 Score-B 已经稳定提高所选 J15：scenario-cluster 95% CI 下界是 +1.5833e5，5/8 scenario 上升，0/8 下降，5/5 seed 上升，Top-3 从 0.8583 升到 1，7/10 residual case 被修正。它没有成为最小充分 horizon，因为 H=8 仍把 normalized regret 从 0.1042 降到 0.0625，降幅 0.0417，大于预先登记的饱和阈值 0.02，并再修正两个 residual case。H* 因此是 8，大于 5。

这不是“短执行没有信息”。短执行里的 progress 和 SINR 已经解释了主要排序。已完成的 payload block 要到大约第 10 步才稳定分开，所以只看 Score-A 会得到相反的结论。

下一步训练 executor-conditioned value，直接预测 J15。不再增加手工 REP、TEMP、continuity 或 motion 特征。本阶段不开始这次训练。

`planner_ready = False`

`deployment_verified = False`

## 8. 九个问题

Q1. Shadow rollout 能否恢复原状态？能。6491 次副本执行后，主状态最大变化是 0。

Q2. H=1 / 3 / 5 / 8 / 10 对 J15 排序的预测如何？Score-B 在 H=1 的 Spearman 是 0.7816，Top-1 是 0.8333，Top-3 是 1，normalized regret 是 0.1042。H=3 和 H=5 与 H=1 选出相同的分配。H=8 和 H=10 的 Spearman 约 0.819，Top-1 是 0.9167，normalized regret 是 0.0625。Score-A 在 H=1 的 Top-1 只有 0.1250，normalized regret 是 0.7056，差于 Phase-5V 的 0.1903。

Q3. 最小充分 horizon 是多少？H* = 8。H=5 已经稳定优于 Phase-5V，但相对 H=8 还没有饱和。

Q4. 310114 t=15 从第几步开始，optimal 稳定排在 predicted 前面？两个命名字符串的 Score-B 从 H=3 起 optimal 更高，并保持到 H=15。Score-A 的严格稳定领先从 H=15 才开始。H=1 的 Score-B 已经选中另一个 J15 = 5e6 的分配，所选 regret 为 0。

Q5. 联合 switching synergy 最早在哪个 H 能被观察到？Score-B 的中位数从 H=1 就为正，但只有 +1.240e4 bit，而且 310114 在 H=1 和 H=3 仍为负。310114 从 H=5 转为正，H=8 达到 +1e6 bit。Score-A 的块量级协同要到 H=15。

Q6. 高层剩余错误能否被短期 frozen MAPPO 发现？能发现大部分。Score-B 在 H≤5 修正 7/10，在 H=8 修正 9/10。310117 t=0 的 1e6 bit 不能被 Score-B 修正，完整 payload 要到 H=15 才分开。

Q7. 最有信息量的早期执行信号是什么？H=1 的 state 内平均 Spearman：mean SINR 0.7929，progress 之和 0.7587，已完成 block 与 payload 都是 0.5661，mean interference −0.5098。最早的稳定信号是执行后的 SINR，其次是 service progress。payload 本身在 H=5 的 Spearman 降到 0.3807，因为较差的分配可能先完成一个 block。

Q8. H=3 和 H=5 的计算量是多少？H=3 每个分配 15.84 ms，120 个分配共 1901.2 ms。H=5 每个分配 26.46 ms，120 个分配共 3175.0 ms。H=8 是 42.22 ms 和 5066.7 ms。

Q9. 下一步做什么？训练 executor-conditioned value，直接预测 J15。H* = 8，不把 H=5 蒸馏当成已经充分，也不再加手工状态特征。本阶段不训练这个模型，不进入 deployment verification。
