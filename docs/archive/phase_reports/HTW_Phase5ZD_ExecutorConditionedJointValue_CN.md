# Phase-5ZD：一步执行响应条件下的联合价值

本阶段只回答一个问题：把冻结 MAPPO 执行恰好一步之后的环境响应，接到高层 assignment value 上，能不能稳定提高最终选中的 J15。

结构是：

高层候选 assignment → 冻结 MAPPO 的一步环境响应 → Executor-conditioned joint value → 高层最终 assignment。

响应只作为价值函数的输入。低层不重新选目标，也没有中间投票层。

训练只用 310100–310109，t = 0、15、30，已有的 120 个候选和已有的 J15。5 个 seed 都把 epoch-10 checkpoint 写盘之后，才一次性读取 310110–310117。没有 early stopping，没有按确认集选 checkpoint，没有搜索超参数。

Model-S 和 Model-R1 使用同一个 `ExecutorConditionedJointScorer`。参数量都是 164993。唯一差别是每个 UAV token 后面拼接的 14 维块：S 填零，R1 填 H=1 响应。HGNN、匹配网络和 MAPPO 的 `requires_grad` 为 false，探针反向传播后梯度绝对值和都是 0。H=1 已交付 payload 不进入模型。

Gate = `RESP-A`。`FROZEN_FOR_RESPONSE_DISTILLATION = True`。

`planner_ready = False`

`deployment_verified = False`

没有打开 310118–310125，没有调用 deployment 决策接口，没有进入 deployment verification。

## 完整性

训练集随机抽查 100 个 state × assignment。clone、执行一步、回到原环境之后，调用方位置、队列、进度、served、assignment、SINR 相关状态、episode step、RNG，以及 batch 中的速度切片，最大绝对变化都是 0。训练集 3600 个候选、确认集 2880 个候选，重放的 H=1 mean SINR、progress sum、payload 与 Phase-5X 的最大绝对差都是 0。确认集上的 H=1 Score-B 仍然达到预注册复现门槛：Spearman 0.7816，Top-3 1.0000，normalized regret 0.1042。训练没有因为 `RESP-D` 或 `RESP-STATE-FAIL` 中止。

训练曲线没有 collapse。10 个 epoch 结束时，两种模型在全部 seed 上的 state 内 score std 都大于 0.41，collapse fraction 都是 0。

## 最终选中的 J15

主指标是 argmax 之后读到的真实 J15。下表是 310110–310117、5 个 seed、全部决策时刻的平均。回归指标是确认集上的 block 单位，1 block = 1e6 bit。

| 模型 | mean J15 | median J15 | normalized regret | Top-1 | Top-3 | Top-5 | Spearman | Kendall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 Phase-5V | 2.3917e6 | 3.0000e6 | 0.1903 | 0.6417 | 0.8583 | 0.9167 | 0.6181 | 0.5077 |
| B1 Phase-5Y | 2.3417e6 | 3.0000e6 | 0.2003 | 0.6833 | 0.8417 | 0.9083 | 0.5785 | 0.4769 |
| S 匹配静态对照 | 2.3417e6 | 3.0000e6 | 0.1958 | 0.6917 | 0.8417 | 0.9083 | 0.5766 | 0.4753 |
| R1 H=1 响应价值 | 2.7917e6 | 3.0000e6 | 0.0625 | 0.8750 | 0.9583 | 0.9583 | 0.8008 | 0.6723 |
| REF1 H=1 Score-B | 2.7500e6 | 3.0000e6 | 0.1042 | 0.8333 | 1.0000 | 1.0000 | 0.7816 | 0.6555 |
| REF8 H=8 Score-B | 2.8750e6 | 3.0000e6 | 0.0625 | 0.9167 | 1.0000 | 1.0000 | 0.8193 | 0.6974 |
| O 真实 J15 | 2.9583e6 | 3.0000e6 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

确认集回归：S 的 MAE 0.5730 block，RMSE 0.7137，R² 0.2640，bias +0.0033，Huber 0.2740。R1 的 MAE 0.3716 block，RMSE 0.4950，R² 0.6367，bias −0.1098，Huber 0.1333。

## Q1. 匹配参数量之后，H1 response 是否优于静态对照？

是。R1 和 S 的结构、参数量、优化器、学习率 1e-3、weight decay 1e-4、梯度裁剪 1.0、seed、训练状态和 J15 标签都相同。R1 的 mean selected J15 高于 S，scenario-cluster 95% CI 的下界大于 0，8 个确认 scenario 里没有一个变差，5 个 seed 方向全部为正。

## Q2. selected J15 提高多少？置信区间是多少？

R1 − S 的 mean selected J15 是 +4.5000e5 bit。以 scenario 为 cluster、10000 次、seed 0 的 paired bootstrap，95% CI 是 [1.6667e5, 7.4167e5]。

相对 Phase-5V（B0），均值是 +4.0000e5 bit，CI [1.6667e5, 5.8333e5]。相对 Phase-5Y（B1），均值是 +4.5000e5 bit，CI [1.8333e5, 7.0833e5]。

R1 的 Top-3 是 0.9583，normalized regret 是 0.0625。两条都过了 RESP-A 的 0.85 和 0.12。

## Q3. 5 个 seed 是否方向一致？

一致，而且幅度也一致。5 个 seed 的 R1 mean selected J15 都是 2.7917e6，normalized regret 都是 0.0625，Top-1 都是 0.8750，Top-3 都是 0.9583，collapse rate 都是 0。对应的 S 在 2.2917e6 到 2.3750e6 之间。5/5 seed 都是 R1 高于 S。

8 个确认 scenario 上，R1 − S 为正向 5、持平 3、负向 0。持平的是 310112、310116、310117。正向的是 310110（+2.6667e5）、310111（+1.0000e6）、310113（+6.6667e5）、310114（+1.0000e6）、310115（+6.6667e5）。相对 Phase-5V 同样是 5 正、3 平、0 负。相对 Phase-5Y 是 5 正、2 平、1 负；负的是 310116，差 −6.6667e4 bit。

## Q4. H1 SINR 是否真的被 value 使用？

使用了，但用法不是“去掉 SINR 之后 selected J15 就下降”。

训练结束后只做推理消融，不重新训练。把 SINR 与 delta SINR 置零再按训练集统计量归一化，120 个 state × seed 里有 26.67% 的 argmax 改变。这超过预注册的 0.10 disagreement 门槛，所以记为 noticeable。

平均 selected J15 没有下降，而是上升 3.3333e4 bit。消融后的 normalized regret 是 0.0569，略低于完整 R1 的 0.0625。因此 SINR 通道确实改变排序，但当前回归头对它的使用并不单调地提高平均 J15。

## Q5. H1 progress 是否真的被 value 使用？

使用了。置零 service progress 与 delta progress 之后，argmax disagreement 是 0.1083，刚好超过 0.10。平均 selected J15 下降 1.6667e4 bit，normalized regret 从 0.0625 升到 0.0681。下降幅度远小于一个 payload block，预注册的“明显”是由 disagreement 触发的，不是由 J15 跌出一个 block 触发的。

对照：置零 interference 的 disagreement 只有 0.0250，selected J15 不变。置零 motion / distance 的 disagreement 是 0.0417，selected J15 下降 4.1667e4 bit，两项都没有过预注册门槛。没有根据消融重选特征。

## Q6. 310114 t=15 上，A_opt 排第几？

A_pred 是 `2;5;-1;4;-1;-1`，J15 = 4e6。A_opt 是 `-1;4;2;5;-1;-1`，J15 = 5e6。

Model-S 仍然把 A_pred 打得更高，A_opt 进不了前排。5 个 seed 的 named rank 是 53、54、53、54、52。seed 0 的 V(A_pred) = 2.0802 block，V(A_opt) = 1.0964 block。

Model-R1 把这个反例拉回 Top-3。5 个 seed 的 named rank 是 2、3、3、2、2，中位数是 2。seed 0 的 V(A_pred) = 3.1529 block，V(A_opt) = 3.2622 block。五个 seed 里 V(A_opt) 都高于 V(A_pred)。该状态上 R1 选中的真实 J15 是 5e6，与 oracle 相同。H=1 Score-B 在这个状态上仍然把 A_opt 排在第 11，数值是 3.9904e5 对 A_pred 的 4.9965e5。学到的价值在这个反例上比 H=1 Score-B 更接近最终 J15。

## Q7. 10 个 residual cases 修复多少？

沿用 Phase-5W 登记的 10 条。比较的是 5 个 seed 的平均 selected J15。R1 相对 Phase-5V：修复 7，不变 3，恶化 0。

修复的是 310110 t=0（1e6 → 3e6）、310113 t=15（1e6 → 3e6）、310114 t=0（2e6 → 3e6）、310114 t=15（4e6 → 5e6）、310115 t=15（1e6 → 3e6），以及 310111 t=15 的两条不同预测 assignment。后两条的 seed 平均从 1.4e6 到 3e6；其中 seed 0 和 seed 4 与 Phase-5V 持平，seed 1/2/3 被修好。不变的是 310111 t=30、310116 t=15、310117 t=0。310111 t=30 和 310116 t=15 上，H=8 Score-B 仍能到 3e6，R1 停在 1e6 和 2e6。

## Q8. 与 H1 Score-B、H8 Score-B 还有多大差距？

按 mean selected J15：R1 比 REF1 高 4.1667e4 bit，比 REF8 低 8.3333e4 bit，比 oracle 低 1.6667e5 bit。

Spearman：R1 0.8008，REF1 0.7816，REF8 0.8193。Top-3：R1 0.9583，REF1 和 REF8 都是 1.0000。normalized regret：R1 与 REF8 都是 0.0625，REF1 是 0.1042。

R1 已经用真实 H=1 响应，在 selected J15 和 regret 上超过手工 H=1 Score-B，并在 regret 上追上 H=8 Score-B。剩下的差距主要在 Top-1 / Top-3，以及少数 residual 状态上 H=8 仍能多拿到一个 block。

## Q9. 主要问题是否已经转到 executor-conditioned value？

是。匹配静态对照 S 的 selected J15、Spearman 和 regret 都停在 Phase-5V / Phase-5Y 附近。加上同一次冻结执行的 H=1 响应之后，selected J15、排序和 310114 反例一起改善。这一步没有加深 HGNN，没有改 pooling，也没有改 MAPPO。当前还没解决的是：价值头对 SINR 的使用不稳定，以及 120 次真实环境 shadow 的代价。问题已经不在“静态表示里还有没有没挖到的 J15”。

## Q10. 120 个候选的完整推理时间是多少？

确认集上，一个 state、120 个 assignment 的平均耗时：

| 部分 | 毫秒 |
| --- | --- |
| T_encoder | 253.5 |
| T_MAPPO_action | 207.3 |
| T_env_H1 | 418.2 |
| T_response_extract | 16.7 |
| T_value_forward | 3.8 |
| T_total | 899.5 |

Phase-5X 记录的单个候选累计 shadow 时间是 H=1 5.30 ms、H=5 25.99 ms、H=8 41.43 ms。按 120 个候选粗算，H=1 约 635 ms，H=8 约 4972 ms。本阶段 120 次真实 H=1 shadow 再加 encoder 和 value，大约 0.90 秒一个决策时刻。价值网络本身只有 3.8 ms。时间几乎全在环境和 actor。

## Q11. 是否达到 RESP-A？

达到。九条同时成立：

1. R1 − S 的 mean selected J15 为 +4.5000e5 bit。
2. scenario-cluster 95% CI 下界是 +1.6667e5 bit。
3. R1 − Phase-5V 的均值是 +4.0000e5 bit。
4. 5/8 个确认 scenario 上 R1 高于 S，其余 3 个持平，没有负向。
5. 5/5 个 seed 上 R1 高于 S。
6. Top-3 = 0.9583。
7. normalized regret = 0.0625。
8. 310114 t=15 的 A_opt 中位 rank 是 2，五个 seed 都在 Top-3。
9. 置零 SINR 或 progress 至少一项造成明显 assignment disagreement。两项的 disagreement 都超过 0.10。

因此冻结：HGNN encoder、REP、TEMP、candidate pool、H=1 response 定义、`ExecutorConditionedJointScorer`、Huber loss 和 10 epoch 训练预算。标记 `FROZEN_FOR_RESPONSE_DISTILLATION`。

## Q12. 下一阶段应该蒸馏 H1 response，还是继续改 HGNN / MAPPO？

应该蒸馏 H1 executor response。真实一步响应已经能稳定修好高层价值，所以下一阶段要学的是：由 state、assignment 和冻结 actor 的动作，预测 H=1 的 SINR、progress、interference 和 motion response，再交给已经冻结的 ExecutorConditionedJointValue。

不要把 120 次在线环境复制放进正式 planner。不要增加 HGNN 深度或宽度，不要改 MAPPO，不要联合训练，不要打开 310118–310125。

`planner_ready` 仍然是 false。`deployment_verified` 仍然是 false。本阶段没有宣布 deployment PASS。
