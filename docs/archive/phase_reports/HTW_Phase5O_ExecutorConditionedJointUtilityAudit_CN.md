# HTW Phase-5O Executor-Conditioned Joint Utility Label Audit

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练 HGNN、Joint Assignment Head 或 Residual MAPPO，没有修改 `mappo_original.pt`、reward、低层观测、B2、tau、Gate-1、Candidate-C 或 planner，没有调用 deployment `decide()`，没有使用 230000–230127，没有生成新 scenario。

## 1. 两个标签

候选池与 Phase-5N 相同：每个决策时刻 120 个 top-3 极大互斥分配，共 720 条。`J_FLY` 直接读取 Phase-5N 标签缓存，并用 Phase-5J 的定义抽查了每个状态的第 0 号分配。

`J_MAPPO(A)` 是把同一个状态快照上的完整分配 A 锁住，交给冻结的 `mappo_original.pt` 执行 15 步后，累计通信载荷的增量。它只作为离线标签。数值区分以一个 payload block = 1.0000e+06 bit 为准；小于 1 bit 的差视为浮点噪声，小于一个 block 的差不记为有效区分。

六个状态里，`J_FLY` 可区分的有 2 个，`J_MAPPO` 可区分的有 6 个。

## 2. 对齐

720 条合并后的 Spearman 是 0.2402，Kendall 是 0.2296。合并后的 Top-1 是 1.0000，regret 是 0.0000e+00。这个合并 Top-1 只说明全局最大的 `J_FLY` 碰巧也达到了全局最大 `J_MAPPO`，不能代表单个状态里的排序。

按状态，在 `J_MAPPO` 可区分的六个状态上，Top-1 overlap 的平均是 0.1667，归一化 regret 的平均是 0.7222。两个标签都有差异的状态上，Spearman 平均是 0.4845。

| scenario | t | J_FLY unique | J_MAPPO unique | Spearman | Kendall | Top-1 | regret | normalized regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 310100 | 0 | 1 | 2 | NA | NA | 0.0000 | 1.0000e+06 | 1.0000 |
| 310100 | 15 | 1 | 4 | NA | NA | 0.0000 | 2.0000e+06 | 0.6667 |
| 310100 | 30 | 2 | 4 | 0.3131 | 0.3038 | 0.0000 | 2.0000e+06 | 0.6667 |
| 310101 | 0 | 1 | 2 | NA | NA | 0.0000 | 1.0000e+06 | 1.0000 |
| 310101 | 15 | 1 | 4 | NA | NA | 0.0000 | 3.0000e+06 | 1.0000 |
| 310101 | 30 | 3 | 4 | 0.6558 | 0.6386 | 1.0000 | 0.0000e+00 | 0.0000 |

Top-1 使用 `J_FLY` 的稳定 argmax，也就是并列时的第一个分配，与 Phase-5N 穷尽搜索的取法一致。`J_FLY` 全部相同时，这个第一个分配没有排序依据。

## 3. 310101 t=15

310101 t=15 的 J_FLY 范围是 0.0000e+00 到 0.0000e+00，独立取值 1 个。J_MAPPO 范围是 0.0000e+00 到 3.0000e+06，独立取值 4 个，标准差 1.1542e+06。稳定 J_FLY argmax 的执行 regret 是 3.0000e+06，归一化 regret 是 1.0000。

## 4. 机制

310101 t=15 pair 0：高收益 `4;5;3;-1;-1;-1` 的 J_MAPPO 是 3.0000e+06，低收益 `-1;-1;-1;4;3;5` 是 0.0000e+00，J_FLY 都是 0.0000e+00。两组指派的都是浮标 3、4、5，这些浮标的 queue 都是 1.0，urgency 都是 0.2500。第 1 步两组已指派 UAV 都进入服务半径，距离分别是 1.12 m 和 2.77 m。第 5 步动作已经是悬停。高收益组 SINR 均值 4.229，干扰和 0.534，progress 均值 0.403；低收益组 SINR 均值 0.891，干扰和 1.234，progress 均值 0.143。第 10 步 progress 分别是 0.800 和 0.285。第 15 步累计载荷是 3.0000e+06 和 0.0000e+00，低收益组的 progress 仍是 0.428，没有完成一个 block。

同一组浮标交给不同 UAV 时，fly-toward 的 15 步载荷可以同为 0，冻结 MAPPO 却会在第 1 步进入服务半径并悬停。之后的差距是同时服务下的 SINR 和 service progress：SINR 较高的一组在 15 步内完成 block，SINR 被干扰压低的一组 progress 到第 15 步仍低于 1。决策时刻的 queue 和 urgency 在这些浮标上相同，不能解释这组差异。

池外的 Original HGNN 和 Phase-5L 分配没有在四个及以上状态上超过池内最大 `J_MAPPO`，所以这次结果不支持先改 candidate generator。

比较使用的是决策时刻已经存在的距离、`rate_if_assigned`、queue、urgency、UAV-UAV interference，以及冻结 MAPPO 在第 1、3、5、10、15 步的动作、SINR、逐步载荷和服务半径。没有把未来到达写进任何网络输入。

## 5. 结论

判定是 **LABEL-A**。

下一阶段可以把高层联合监督从 J_FLY 改成离线的 executor-conditioned J_MAPPO，再训练 Joint Planner。J_MAPPO 仍然只作为标签，不能作为网络输入。

Q1. `J_FLY` 适合作为当前冻结执行器下的高层监督吗？ 不适合。在四个状态上它给 120 个分配同一个值，而冻结 MAPPO 的 15 步载荷可以差到一个以上 payload block。

Q2. `J_MAPPO` 是否比 `J_FLY` 更能区分联合分配？ 可以。J_MAPPO 在 6 个状态上超过一个 payload block 的差距，J_FLY 只有 2 个。

Q3. Spearman、Kendall、Top-1、regret 见第 2 节和 `phase5o_fly_vs_mappo_alignment.csv`。

Q4. 310101 t=15 的差异见第 3 节和第 4 节。

Q5. 当前失败归类： label mismatch。J_MAPPO 能区分分配，J_FLY 与它不对齐。

Q6. 下一阶段： 用离线 J_MAPPO 重新监督 Joint Planner。先不改 candidate，也不把 J_MAPPO 送进网络。

`planner_ready = False`

`deployment_verified = False`
