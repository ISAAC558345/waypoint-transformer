# Phase-5ZG：响应空间价值敏感性审计

本阶段只读取冻结的 Model-R1，计算它对 H1 response 的价值梯度。没有 optimizer step，没有新的 checkpoint，没有重训 predictor、HGNN 或 value。混合 response 只作为诊断性反事实，不把它当成环境里必然出现的物理状态。

数据只使用已经消费的 310110–310117。Jacobian 在标准化 response 上用自动微分计算，再按 Phase-5ZD 的 `std` 换回原始物理尺度。16 点路径用梯形积分，端点是真实 response 和预测 response。

Gate = `SENS-C`。`FROZEN_FOR_FRESH_CONFIRMATION = False`。

`planner_ready = False`

`deployment_verified = False`

没有打开 310118–310125，没有调用 deployment 决策接口，没有进入 deployment verification。

## Q1. R1 和 P1 能否严格复现？

能。从磁盘重新加载 Model-R1 checkpoint、Phase-5ZE 的 response 表和冻结候选池之后：

| 检查 | 结果 |
| --- | --- |
| 真实 response 对已保存 R1 value 的最大绝对误差 | 0 |
| 预测 response 对已保存 P1 value 的最大绝对误差 | 0 |
| argmax assignment 一致率 | 1 |
| 特征顺序与 `phase5zd_response_feature_spec.json` | 一致 |
| 五个 seed 的真实 response 彼此差 | 0 |
| 分组线性贡献之和与全通道点积的最大差 | 8.88e-16 |

复算的 selected J15 仍是 P1 = 2.3083e6 bit，R1 = 2.7917e6 bit。因此不是 `SENS-D`。

120 个 state-seed 里，P1 与 R1 的 assignment 关系是：

| 类别 | 数量 |
| --- | --- |
| 同一个 assignment | 42 |
| assignment 不同，J15 相同 | 43 |
| P1 的 J15 更低 | 35 |
| P1 的 J15 更高 | 0 |

P1 没有在任何一个 state-seed 上选出比 R1 更高的 J15。35 个损失点的平均 J15 缺口是 1.6571e6 bit。下面的通道归因集中在这 35 个点。

## 价值梯度的尺度

Model-R1 吃的是标准化 response。原始尺度的梯度等于标准化梯度除以该通道的 `std`。Rate 的物理单位是 bit/s，`std` 很大，所以“每 bit 的梯度”会小到 1e-7，不能拿来和 SINR 的原始梯度比大小。可比的是标准化 L2，以及梯度与误差的乘积。后者在两种尺度上数值相同。

在真实 response 上，五个 seed、全部候选的平均：

| 通道 | 标准化 L2 | 原始 L2 | 选中 UAV 上梯度为正的比例 | 误差造成的平均绝对价值贡献 |
| --- | --- | --- | --- | --- |
| rate | 0.1321 | 1.334e-7 | 1.000 | 0.1015 |
| SINR | 0.1227 | 0.0361 | 1.000 | 0.1507 |
| progress | 0.0863 | 0.9577 | 0.999 | 0.0420 |
| interference | 0.0519 | 0.1704 | 0.100 | 0.0395 |
| motion | 0.0480 | 0.0257 | 0.231 | 0.0209 |
| binary | 0.0122 | 0.0284 | 0.535 | 0.00023 |

## Q2. 哪些 response channel 的 value sensitivity 最大？

标准化尺度上，rate 的平均 L2 最大，SINR 次之。原始尺度上，progress 和 interference 因为单位不同而显得更大，rate 反而最小。这个对照说明不能用原始量纲给通道排名。

另外，`sinr_1` 与 `rate_1` 在 UAV 维上的梯度余弦是 1。Model-R1 先对选中 UAV 做均值池化，再进入 MLP，所以这两条通道对每架 UAV 的梯度都正比于同一张选中掩码。它们在 UAV 之间的方向完全重合，差别只在一个标量权重，而且两个权重的符号都是正的。Value 没有把 SINR 和 rate 分成两个可独立排序的方向。

## Q3. 哪些预测误差对 value 的实际影响最大？

是 SINR，然后是 rate。平均绝对一阶贡献是 SINR 0.1507、rate 0.1015、progress 0.0420、interference 0.0395、motion 0.0209、binary 0.00023。

在 35 个 J15 损失点上，真实偏好 margin 的平均变化是 −1.2650。各通道的平均一阶贡献：

| 通道 | 一阶 | 单通道注入 | 16 点积分 |
| --- | --- | --- | --- |
| SINR | −0.4809 | −0.5607 | −0.5926 |
| rate | −0.3439 | −0.3655 | −0.3983 |
| progress | −0.1316 | −0.1342 | −0.1472 |
| interference | −0.0758 | −0.0765 | −0.0801 |
| motion | −0.0407 | −0.0414 | −0.0469 |
| binary | +0.00032 | +0.00032 | +0.00036 |

负号表示这个误差缩小了 `V(A_R) − V(A_P)`，也就是把 R1 喜欢的 assignment 往下压。Binary 几乎不参与。Interference 有梯度，但预测误差带进 value 的量远小于 SINR 和 rate。

有唯一主导通道的 scenario 投票：310110、310111、310113、310115 都是 SINR。310114 的 SINR 和 rate 太接近，投不出唯一通道。310112、310116、310117 没有 J15 损失，不投票。五个 seed 的平均贡献都唯一指向 SINR。预注册的稳定条件是至少 6 个 scenario 和 4 个 seed。Seed 满足，scenario 只有 4 票，所以主导通道不算稳定，`SENS-A` 不成立。

## Q4. 一阶 Jacobian 能解释多少 value error？

在每个 state 的 120 个 assignment 上，真实 response 处的 `Gᵀe` 与 `V(r_pred) − V(r_true)`：

| 指标 | 数值 |
| --- | --- |
| 状态内符号一致率的平均 | 0.9996 |
| 状态内 Spearman 的平均 | 0.9946 |
| 状态内 Pearson 的平均 | 0.9906 |
| 线性残差 MAE | 0.0333 |
| 实际 value 误差 MAE | 0.3283 |
| J15 损失点上 margin 变化的符号一致率 | 1.000 |
| 这些点上线性 margin / 实际 margin | −1.0726 / −1.2650 = 0.848 |

16 点积分和实际 value 差的平均绝对值是 4.21e-5，最大绝对值是 0.00349。积分残差是梯形公式的数值误差。价值变化本身是一阶的：线性项已经说明方向，并覆盖损失点上约 85% 的 margin 幅度。剩下的幅度来自沿路径的梯度变化，而不是一个很大的交叉项。

## Q5. 是否存在明显的跨通道非线性交互？

有一个小的、方向稳定的 SINR × rate 交互，但它不是 margin 的主体。

在 J15 损失点上，交互项 `V(e_sinr + e_rate) − V(e_sinr) − V(e_rate) + V(0)` 的 margin 平均是 −0.0576。它占实际 margin 变化的 0.0455。Scenario-cluster 95% CI 是 [−0.0825, −0.0271]，不含 0。同号的 scenario 有 5 个，不到预注册的 6 个。其余 14 对通道的平均 margin 交互都更小，绝对值最大的下一对是 SINR × progress，−0.0185。

预注册的 `SENS-B` 要求这个交互至少占 margin 的 0.25，并且至少 6 个 scenario 同号。两项都不满足。

## SINR 与 rate 的联合反事实

同一 state 内，真实 mean-SINR 和 mean-rate 的 Spearman 平均是 0.9781，预测的是 0.9900。选中 UAV 上两条误差的 Pearson 是 0.8255。状态内排序的 pairwise accuracy 是 SINR 0.7563、rate 0.7737。

| 路径 | mean J15 | normalized regret | 与 R1 的 assignment 一致率 | J15 Recovery | 损失点上的 margin Recovery |
| --- | --- | --- | --- | --- | --- |
| P1 | 2.3083e6 | 0.2458 | 0.3500 | 0 | 0 |
| P1+TRUE_SINR | 2.6167e6 | 0.1264 | 0.6917 | 0.6379 | 0.4864 |
| P1+TRUE_RATE | 2.6250e6 | 0.1403 | 0.5000 | 0.6552 | 0.3348 |
| P1+TRUE_SINR+TRUE_RATE | 2.6583e6 | 0.1058 | 0.8000 | 0.7241 | 0.7913 |
| R1 | 2.7917e6 | 0.0625 | 1 | 1 | 1 |

相对 P1 的 scenario-cluster 95% CI：SINR +3.0833e5 bit，[4.1667e4, 6.2521e5]；rate +3.1667e5，[7.5000e4, 6.0000e5]；两者一起 +3.5000e5，[2.5000e4, 7.0021e5]。两者一起减去只换 SINR 的 CI 是 [−1.0833e5, +2.1667e5]，减去只换 rate 的 CI 是 [−1.6667e5, +2.3333e5]，都跨过 0。第二条真实通道没有带来稳定的额外 J15。310112 上，两条一起换成真实值后，mean J15 从 3.0000e6 降到 2.6667e6。

## Q6. 为什么单独换成真实 SINR 或真实 rate 都不能恢复全部 R1 增益？

三条事实叠在一起。

第一，两条通道的预测误差高度相关，而且 value 对它们的 UAV 梯度平行。换掉其中一条，另一条误差仍沿同一个池化方向推动 value。损失点上，只换 SINR 收回 48.6% 的 value margin，只换 rate 收回 33.5%，两条一起收回 79.1%。J15 因为以 1e6 bit 量化，两条一起的 Recovery 只有 0.724，而且相对任一单通道的增量不稳定。

第二，progress 在损失点上仍有 −0.132 的一阶 margin 贡献。SINR 和 rate 都换成真实值以后，这条贡献还在。

第三，状态内排序本身没有对齐。SINR pairwise accuracy 0.7563，rate 0.7737。全局相关高，不代表 120 个候选之间的顺序可用。

## Q7. 310114 t=15 为什么发生严重排序反转？

五个 seed 上，`A_opt` 的名次是：

| seed | S | P1 | 真实 SINR | 真实 rate | 真实 SINR+rate | R1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 53 | 74 | 7 | 18 | 4 | 2 |
| 1 | 54 | 72 | 8 | 24 | 4 | 3 |
| 2 | 53 | 76 | 9 | 22 | 4 | 3 |
| 3 | 54 | 85 | 8 | 19 | 4 | 2 |
| 4 | 52 | 74 | 7 | 19 | 4 | 2 |

真实 response 下，`V(A_opt) − V(A_pred)` 只有 +0.00085 到 +0.109，但足够让 R1 把 A_opt 放进前三。预测 response 下，这个 margin 变成 −1.475 到 −1.577。P1 因此把 A_opt 从静态模型的大约第 53 名再压到第 72–85 名。

Seed 0 上，把 A_opt 相对 A_pred 的 margin 拆开：

| 通道 | 一阶 | 积分 |
| --- | --- | --- |
| SINR | −0.332 | −0.732 |
| rate | −0.332 | −0.593 |
| progress | −0.119 | −0.189 |
| motion | −0.078 | −0.116 |
| interference | −0.016 | −0.028 |
| binary | +0.001 | +0.001 |

SINR × rate 的交互 margin 在五个 seed 上是 −0.256、−0.217、−0.200、−0.221、−0.211。交互存在，但一阶的 SINR 和 rate 各自已经和它同一量级。这个 state 上两条通道谁也不是唯一原因，所以 scenario 投票是空的。两条都换成真实值以后，名次从 74 左右回到第 4，仍差 R1 的第 2–3 名。剩下的下压来自 progress 和 motion。

## Q8. 哪些 response error 最容易造成实际 J15 损失？

SINR 和 rate。35 个损失点上，它们的平均一阶 margin 贡献是 −0.481 和 −0.344。Progress 是 −0.132。Interference、motion、binary 都更小。P1 相对 R1 没有 J15 更高的 state。

Phase-5ZD 的 10 个 residual case、50 个 seed 行里，R1 的 selected J15 高于 S 的有 27 行，其中 P1 又丢掉这笔修复的有 20 行。丢掉的共同位置是：

| case | S | P1 | R1 | 丢掉的 seed 数 | 命名 pair 的 margin 反转 |
| --- | --- | --- | --- | --- | --- |
| 310110 t=0 | 2.2000e6 | 1.0000e6 | 3.0000e6 | 2 | 5/5 |
| 310111 t=15，两条命名 assignment | 0 | 6.0000e5 | 3.0000e6 | 各 4 | 0/5 与 5/5 |
| 310113 t=15 | 1.0000e6 | 1.8000e6 | 3.0000e6 | 3 | 0/5 |
| 310114 t=15 | 2.0000e6 | 4.0000e6 | 5.0000e6 | 5 | 5/5 |
| 310115 t=15 | 1.0000e6 | 2.2000e6 | 3.0000e6 | 2 | 2/5 |

310111 t=30、310116 t=15、310117 t=0 上，R1 并没有高于 S，P1 也没有再丢。310114 t=0 上 R1 与 S 都是 3.0000e6，P1 降到 2.0000e6；这是 P1 在已经持平的状态上新增的损失，五个 seed 的命名 margin 都反转。能投出唯一通道的丢失案例主要指向 SINR。310114 t=15 的五个 seed 里，主导通道是 rate 或空，和上面的双通道并列一致。

## Q9. 当前主要问题是 value sensitivity、joint interaction，还是 ordering error？

是 state 内的 ordering error。

Value 对 response 误差几乎是线性的：符号一致率 0.9996，Spearman 0.9946，损失点 margin 的符号一致率是 1。跨通道交互只占 margin 的 4.55%，低于 0.25 的门槛。SINR 在平均贡献和全部 seed 上都领先，但只有 4 个 scenario 能投出唯一的 SINR，310114 上 SINR 与 rate 分不开，所以也不能把机制收成单一通道的 sensitivity loss。

与此同时，SINR 和 rate 的状态内 pairwise accuracy 是 0.7563 和 0.7737，都低于 0.80。预测误差沿着 Model-R1 已经很敏感的方向，把候选顺序推反了。

因此 Gate 是 `SENS-C`：`PREDICTION_ORDERING_DOMINANT`。

## Q10. 下一阶段应该修复哪一个最小机制？

按这个 Gate，下一阶段只研究 state-conditional、ranking-aware 的 response loss。监督对象是同一个 state 里候选之间的 response 顺序，优先覆盖 SINR 和 rate 这条已经被 value 合成在一起的方向。

不扩大 predictor，不加 attention、Transformer 或 GNN，不改 HGNN，不改 Model-R1，不做 MAPPO 更新，不打开 310118–310125，不调用 deployment 决策接口。本阶段到此停止。
