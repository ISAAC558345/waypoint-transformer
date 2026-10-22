# Phase-5ZC：执行器响应的动态可观测性

本阶段没有训练任何模型，没有拟合 probe，没有打开 310118–310125，没有调用 deployment `decide()`，没有修改正式 planner。

Gate = `DYN-C`。`H_dyn = 1`。`EXECUTOR_FEEDBACK_NEEDED = True`。

`planner_ready = False`

`deployment_verified = False`

人口是已经消费过的 310100–310117，共 18 个 scenario、54 个决策状态、每个状态 120 个候选。H=1/3/5/8/15 的执行量直接复用 Phase-5X。新计算只有每个候选一次 frozen MAPPO 前向，环境不推进。Phase-5X 的 H=15 payload 与已存 J15 的最大绝对差是 0（6480 条）。调用方环境指纹的最大绝对差是 0。310114 t=15 的第一步动作与 Phase-5W 轨迹第 1 步动作的最大绝对差是 0。

排序用的标量都是事先登记的，没有事后挑选：L0 是已选边的边际 rate 之和，PHY_NOW 是当前联合 SINR rate 之和，ACT0 是指向已分配浮标的平均径向动作，H=1/3/5/8 是 Phase-5X 的 Score-B（`q_exec_h`）。J15 只作为评价标签。

## 信息阶梯

EXECUTION-INFORMATIVE 的锁定标准是：J15 Spearman ≥ 0.70，Top-3 ≥ 0.85，normalized regret ≤ 0.12，并且至少一个非累计 payload 的动态量与 future residual `J15 - payload_up_to_H` 的有向 Spearman ≥ 0.40。干扰取负号后再比较。区间是 scenario 聚类 bootstrap，10000 次。

| 层级 | Spearman | Top-1 | Top-3 | normalized regret | NN contradiction ≥1 block | residual Spearman | 每候选耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.5156 [0.453, 0.574] | 0.5370 | 0.6667 | 0.3383 | 0.2330 | 0.5156 | 0 |
| PHY_NOW | 0.5312 [0.470, 0.592] | 0.6296 | 0.7407 | 0.2765 | 0.2711 | 0.5312 | 0 |
| ACT0 | 0.1251 [0.080, 0.171] | 0.0741 | 0.4074 | 0.7753 | 0.4789 | 0.1251 | 1.29 ms |
| RESP1 | 0.7736 [0.749, 0.797] | 0.8704 | 0.9815 [0.944, 1.000] | 0.0679 [0.019, 0.139] | 0.2037 | 0.7963 | 5.38 ms |
| H3 | 0.7961 [0.773, 0.817] | 0.8704 | 0.9815 | 0.0679 [0.019, 0.139] | 0.1981 | 0.7893 | 15.85 ms |
| H5 | 0.8040 [0.780, 0.826] | 0.8704 | 0.9815 | 0.0679 [0.019, 0.139] | 0.1866 | 0.7899 | 26.47 ms |
| H8 | 0.8139 [0.789, 0.837] | 0.9074 | 0.9815 | 0.0494 [0.012, 0.096] | 0.1617 | 0.7064 | 42.25 ms |
| J15 oracle | 1 | 1 | 1 | 0 | 0 | 1 | — |

H=1、H=3、H=5 的 normalized regret 完全相同。Score-B 选出的 J15 从第一步起就不再变，Spearman 仍从 0.774 升到 0.804，名次的细节还在变。H=8 才把 regret 再降到 0.0494。

H=1 的 Spearman 区间和 Top-3 区间都落在门槛内。regret 的均值 0.0679 低于 0.12，但区间上端 0.139 越过了 0.12。锁定规则比较的是均值，所以 `H_dyn` 仍是 1。H=8 是 regret 区间也整个落到 0.12 以下的第一层。这个差别记在这里，不改门槛。

## Q1. 为什么 Phase-5ZB 的二阶 correspondence 仍然失败？

Phase-5ZB 先把每架 UAV 和目标的向量单位化，再对集合取外积平均。两个目标对调之后，这组方向的平均几乎不变，collision 仍是 1，J15 Spearman 约等于 0。那不是“配对发生在平均之后”，旧实现已经先绑定 `(H_UAV, H_Buoy, H_edge)` 再做 mean。失败的名字是一阶集合平均把对应关系抹掉，而二阶矩没有把对应关系找回来。

本阶段说明，继续换静态标量也过不了同一道门槛。边际 rate 的 Spearman 是 0.516，当前联合 SINR rate 是 0.531，Top-3 分别是 0.667 和 0.741，normalized regret 是 0.338 和 0.277。缺的不是另一个静态 pooling。

## Q2. 当前静态 state 是否已经足够恢复 J15 排序？

不够。L0 和 PHY_NOW 都没有同时达到 Spearman、Top-3 和 regret 三道门槛，因此不是 DYN-A。

REP 与 TEMP 各自单位化后拼接，最近邻 cosine ≥ 0.99 且 |ΔJ15| ≥ 1 block 的比例是 0.233。PHY_NOW 用标量归一化距离，同样的 contradiction 是 0.271。静态表示里仍然有一批 J15 差一个 block 以上、几何上却挨在一起的分配。Phase-5Y / Phase-5V 原先也过不了这三道门槛，所以问题不是“信息已经在当前特征里、只差一个 scorer”。

## Q3. frozen MAPPO 的第一步动作有没有额外信息？

没有达到锁定标准。事先登记的 ACT0 标量是各有效 UAV 指向自己目标的平均径向分量。它的 Spearman 只有 0.125，Top-3 是 0.407，normalized regret 是 0.775，比 L0 和 PHY_NOW 更差，增益是 -0.406。

动作几何本身也不跟踪 J15 差距。同一状态里，|ΔJ15| 与 joint action L2 的 Spearman 是 0.021，与每架 UAV 动作差是 0.028，与径向差是 0.021，与航向差是 0.016，与 joint cosine distance 是 0.026。动作向量上的最近邻 contradiction 是 0.479，高于静态表示。

径向分量接近饱和：低层策略看到目标后，第一步大体就是朝目标走，所以“径向更大”区分不了高 J15 和低 J15。这不是 DYN-B。

## Q4. 最小 informative horizon 是多少？

`H_dyn = 1`。RESP1 的 Score-B 同时满足四条锁定标准：Spearman 0.774，Top-3 0.981，normalized regret 0.068，future residual 与服务进度的 Spearman 0.770。H=3、H=5、H=8 也满足。ACT0 不满足，所以最小层是执行一步之后，不是动作前向本身。

它不是 deployment horizon，也不是 Phase-5X 那个饱和意义下的 H*=8。H*=8 回答的是“再往后看，regret 还会不会明显下降”。这里回答的是“执行差异从哪一层开始能够排序 J15，并且不是只在记录已经交付的 payload”。

## Q5. 哪一个信号最早出现？

按锁定方向，H=1 同时出现三个信号：

| 信号 | 与 J15 的 Spearman | Kendall | pairwise sign accuracy |
| --- | --- | --- | --- |
| mean SINR | 0.790 | 0.656 | 0.950 |
| service progress | 0.763 | 0.637 | 0.936 |
| mean interference（原始，越低越好） | -0.484 | -0.390 | 0.239 |

有向之后，干扰是 0.484，也过了 0.40。最早、最强的是 mean SINR。服务进度紧随其后。

距离缩短和运动效率在 H=1 只有 0.139 和 0.160，到 H=8 变成负的，不能当作早期信号。H=1 的 in-range 计数在状态内没有变化，Spearman 为空，缓存里也没有积分时长，这里不另造一个 duration。同一目标连续性的 Spearman 在四个 horizon 上都是 0.468，它来自分配和决策时刻原目标的比较，不随执行展开。

累计 payload 在 H=1 与 J15 的 Spearman 只有 0.566。单靠已经交付的 block 排不出这条阶梯。

## Q6. 最早的信号是在预测 residual，还是只在记录已交付 payload？

是在预测还没交付的部分。H=1 时大量候选的 payload 仍是 0，residual 几乎就是 J15。

| H=1 信号 | 与 future residual | 与已交付 payload |
| --- | --- | --- |
| mean SINR | 0.782 | 0.568 |
| service progress | 0.770 | -0.059 |
| mean interference（取负号） | 0.478 | 0.519 |
| distance reduction | 0.141 | 0.016 |

服务进度几乎不解释已经交付的 payload，却能排序剩余的 J15。SINR 两边都相关，和 residual 的关系更强。所以 H=1 通过标准，不是因为“第一步已经把 block 做完了”。

到 H=8，累计 payload 与 J15 的 Spearman 仍只有 0.529。服务进度与 residual 的 Spearman 是 0.696，与已交付 payload 是 -0.185。SINR 与 residual 降到 0.574，与 payload 升到 0.594。干扰与 residual 降到 0.373，掉出门槛。H=8 的预测成分主要还在服务进度，不只是已经入账的 payload。H=15 的 residual 为 0，相关不再有定义。

## Q7. 310114 t=15：A_opt 最早在哪一步稳定优于 A_pred？

A_pred 是 `2;5;-1;4;-1;-1`，J15 = 4e6。A_opt 是 `-1;4;2;5;-1;-1`，J15 = 5e6。

决策时刻的静态量偏向 A_pred。边际 rate 是 3.438e6 对 3.245e6，当前联合 rate 是 1.254e6 对 1.194e6，平均距离是 21.18 m 对 21.99 m，当前 SINR 估计是 0.337 对 0.319。A_pred 保住的服务进度是 0.157，A_opt 是 0。REP cosine 是 0.900，TEMP cosine 是 0.891。

ACT0 的平均径向分量几乎一样：A_pred 0.99998，A_opt 0.9985。joint L2 是 1.732 对 1.730。动作向量本身不同，因为空闲的 UAV 换了：A_pred 的 UAV0 是 `(0.924, -0.382)`，A_opt 的 UAV0 是 `(-0.004, -0.020)`。这是“谁被分配了目标”的直接反应，不是对最终 J15 的预判。第一步径向动作不偏向 A_opt。

执行之后：

| 层 | pred SINR | opt SINR | pred interference | opt interference | pred payload | opt payload | pred Score-B | opt Score-B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H=1 | 12.17 | 14.82 | 0.0689 | 0.0671 | 0 | 0 | 5.00e5 | 3.99e5 |
| H=3 | 12.17 | 14.82 | 0.0689 | 0.0671 | 0 | 0 | 1.185e6 | 1.197e6 |
| H=5 | 12.17 | 14.82 | 0.0689 | 0.0671 | 0 | 0 | 1.871e6 | 1.995e6 |
| H=8 | 11.35 | 14.82 | 0.0500 | 0.0587 | 2e6 | 3e6 | 2.635e6 | 3.000e6 |
| H=15 | 45.83 | 163.46 | 0.0267 | 0.0313 | 4e6 | 5e6 | 4.268e6 | 5.333e6 |

A_opt 的 mean SINR 从 H=1 起就更高，并且一直保持到 H=15。Score-B 在 H=1 仍偏向 A_pred，因为 A_pred 的进度是 0.500，A_opt 是 0.399；从 H=3 起 Score-B 转向 A_opt 并不再反转。已交付 payload 到 H=8 才分开，A_opt 是 3 个 block，A_pred 是 2 个。

因此：信道质量的执行优势在第一步闭环之后已经出现；把进度折进 Score-B 之后，稳定优势从 H=3 开始。只看第一步动作看不到这个差距。

## Q8. 改变其他 UAV 的目标，会不会改变当前 UAV 的第一步动作？

不明显。120 个候选里没有“只改 UAV i、其余目标完全不变”的配对，所以 own-only 一格是空的。只要 UAV i 自己的目标变了，动作 L2 的均值是 0.265、中位数是 0.0069。只改其他 UAV 的目标时，UAV i 的动作 L2 均值是 0.00085、中位数是 0.00044。均值之比约 0.003，中位数之比约 0.064，都低于锁定的 0.20。

frozen MAPPO 的第一步动作主要跟着自己的目标走，不是一套明显的跨机联合响应。高层不能靠“看一眼别人的目标就读出我的第一步”来补上 J15。

## Q9. multi-UAV synergy 从哪一层开始出现？

动作差和价值差不是同一层。

9 个 Phase-5W synergy 案例里，ACT0 的 `joint action L2 - sum of single L2` 中位数只有 0.0052。其中 4 个案例大于 0.4，310114 t=15 是 0.475；另外 5 个接近 0。动作空间的交互不是稳定的群体现象。机械的“中位数大于 0”会落在 ACT0，但 0.005 的动作差不能称为 synergy 已经出现。

Score-B synergy 的中位数从 H=1 起为正，幅度是 1.24e4 bit，9 个案例里 5 个为正。这是未完成 block 的进度差，不是已经入账的 block。payload synergy 的中位数在 H=1/3/5 是 0，H=8 是 -1e6，H=15 才是 +1e6。

310114 t=15 的 Score-B synergy 在 H=1 是 -1.45e5，H=3 是 -1.20e5，H=5 转为 +3.47e4，H=8 是 +1e6。这个案例的价值交互出现在 H=5，和 Phase-5X 一致。它支持短程闭环，不支持把 ACT0 当成 synergy 的载体。

## Q10. 下一步应该做什么？

进入 short-horizon、executor-conditioned value，而不是继续改静态 scorer，也不是把 ACT0 拼进价值头。

建议蒸馏的是 H=1 已经能看到的执行响应，尤其是 mean SINR 和服务进度，标签仍是 J15 的剩余部分，而不是把 H=15 rollout 放到在线决策里。MAPPO 继续冻结。低层响应是高层 assignment value 的输入：候选分配，冻结执行器的短程响应估计，执行器条件下的联合价值，再给出最终分配。不是两层各算一个目标再投票。

H=1 已经满足锁定的均值标准。H=8 仍把 regret 从 0.068 降到 0.049，并且是 regret 区间也落在 0.12 以下的第一层。蒸馏时可以先用 H=1 的动态量，不要把 Phase-5X 的饱和 horizon 8 误当成这里的 `H_dyn`。

## Q11. 是否需要执行器反馈？

`EXECUTOR_FEEDBACK_NEEDED = True`。依据是 Gate = DYN-C：静态 state 和第一步动作都不够，一步闭环之后的 SINR 和进度能够排序 J15，并且进度预测的是尚未交付的 residual。

这不表示现在就加一个独立的中间决策层。

本阶段没有训练网络，没有打开新鲜确认集，没有进入 deployment verification，也没有宣布 deployment PASS。
