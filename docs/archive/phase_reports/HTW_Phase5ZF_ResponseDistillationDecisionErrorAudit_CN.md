# Phase-5ZF：响应蒸馏决策误差因果审计

本阶段只做 inference-only 诊断。没有 optimizer step，没有新的 checkpoint，没有 checkpoint selection，没有重训 response predictor、HGNN 或 value。J15 没有重新生成。候选池、归一化和 Phase-5ZD 的 Model-R1 都保持冻结。

数据只使用已经消费的 310110–310117。预测 response 和真实 response 来自 Phase-5ZE 已经写好的表。本阶段只为了让冻结的 Model-R1 给混合 response 打分，才重放这 24 个确认 state 的 encoder embedding。没有新的 environment step 标签。

Gate = `ERR-E`。`FROZEN_FOR_FRESH_CONFIRMATION = False`。

`planner_ready = False`

`deployment_verified = False`

没有打开 310118–310125，没有调用 deployment 决策接口，没有进入 deployment verification。

对齐检查先于科学结论。把 CSV 里的真实 response 和预测 response 按 Phase-5ZD 的 `(raw - mean) / std` 送回冻结 Model-R1 后，相对已保存的 `v_p1`、`v_r1` 最大绝对误差都是 0。argmax 一致率是 1。五个 seed 的真实 response 彼此最大绝对差是 0。复算的 R1 mean selected J15 是 2.7917e6 bit，P1 是 2.3083e6 bit，与 Phase-5ZE 锁定结果一致。因此不是 `ERR-F`。

R1 − P1 的分母是 +4.8333e5 bit。下面的 Recovery 都是这个分母上的均值之比。scenario 是 bootstrap 的 cluster，10000 次、seed 0。平局阈值是 1 bit。

## 替换之后的 selected J15

| 路径 | mean J15 | normalized regret | Top-1 | Top-3 | Top-5 | 与 R1 一致 | Recovery |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 2.3083e6 | 0.2458 | 0.5833 | 0.8250 | 0.9583 | 0.3500 | 0 |
| P1+TRUE_SINR | 2.6167e6 | 0.1264 | 0.7417 | 0.9167 | 0.9583 | 0.6917 | 0.6379 |
| P1+TRUE_INTERFERENCE | 2.3083e6 | 0.2458 | 0.5750 | 0.8750 | 0.9583 | 0.3500 | 0.0000 |
| P1+TRUE_PROGRESS | 2.4917e6 | 0.1847 | 0.6667 | 0.8750 | 0.9667 | 0.3917 | 0.3793 |
| P1+TRUE_RATE | 2.6250e6 | 0.1403 | 0.7500 | 0.9250 | 1.0000 | 0.5000 | 0.6552 |
| P1+TRUE_MOTION | 2.4000e6 | 0.2153 | 0.6250 | 0.8500 | 0.9583 | 0.3500 | 0.1897 |
| P1+TRUE_BINARY | 2.3083e6 | 0.2458 | 0.5833 | 0.8250 | 0.9583 | 0.3500 | 0.0000 |
| C1 SINR+interference | 2.6500e6 | 0.1042 | 0.7750 | 0.9333 | 0.9583 | 0.7333 | 0.7069 |
| C2 SINR+progress | 2.6667e6 | 0.0986 | 0.7917 | 0.9417 | 0.9583 | 0.7083 | 0.7414 |
| C3 SINR+interference+progress | 2.6667e6 | 0.0986 | 0.7917 | 0.9583 | 0.9583 | 0.7500 | 0.7414 |
| R1 | 2.7917e6 | 0.0625 | 0.8750 | 0.9583 | 0.9583 | 1.0000 | 1 |

相对 P1 的 scenario-cluster 95% CI：

| 对比 | mean | 95% CI | scenario | seed |
| --- | --- | --- | --- | --- |
| P1+TRUE_SINR − P1 | +3.0833e5 | [4.1667e4, 6.2521e5] | 5 正 / 3 平 / 0 负 | 5 正 / 0 平 / 0 负 |
| P1+TRUE_PROGRESS − P1 | +1.8333e5 | [1.6667e4, 3.8354e5] | 4 正 / 3 平 / 1 负 | 4 正 / 1 平 / 0 负 |
| P1+TRUE_INTERFERENCE − P1 | 0 | [−1.4167e5, 1.2500e5] | 3 正 / 3 平 / 2 负 | 1 正 / 0 平 / 4 负 |
| P1+TRUE_RATE − P1 | +3.1667e5 | [7.5000e4, 6.0000e5] | 5 正 / 3 平 / 0 负 | 5 正 / 0 平 / 0 负 |
| C3 − P1 | +3.5833e5 | [8.3333e4, 6.8333e5] | 5 正 / 3 平 / 0 负 | 5 正 / 0 平 / 0 负 |

三个分母为 0 的 scenario 是 310112、310116、310117：那里 P1 与 R1 的 mean J15 已经相同，Recovery 按预注册规则不计算。Progress 的唯一负 scenario 是 310110，−6.6667e4 bit。Interference 的两个负 scenario 是 310115（−4.0000e5）和 310116（−1.3333e5）。

## 把预测通道注入真实 R1

Damage = J_R1 − J_R1-pred_group。正值表示这条预测通道拉低了已经有效的 R1。

| 注入 | mean J15 | Damage | 与 R1 的 assignment 一致率 |
| --- | --- | --- | --- |
| R1-PRED_RATE | 2.6667e6 | +1.2500e5 | 0.7500 |
| R1-PRED_SINR | 2.6833e6 | +1.0833e5 | 0.6083 |
| R1-PRED_MOTION | 2.7500e6 | +4.1667e4 | 0.9500 |
| R1-PRED_INTERFERENCE | 2.7583e6 | +3.3333e4 | 0.9417 |
| R1-PRED_PROGRESS | 2.7917e6 | 0 | 0.9667 |
| R1-PRED_BINARY | 2.7917e6 | 0 | 1.0000 |

Rate 对 R1 最必要，SINR 次之。Progress 和 binary 的预测值替换进 R1 后，selected J15 不变。

## Q1. 为什么 SINR 的 overall Spearman 约 0.724，P1 的 selected J15 仍然低于静态控制？

Phase-5ZE 的 0.724 是把全部 state、全部 assignment、全部 UAV 摊平之后的 Spearman。它主要吃到了“在服务”和“空闲”之间的大差距。本阶段把同一个 state 里的 120 个 assignment 先排好序再平均：mean-SINR 的 Spearman 是 0.6507，Kendall 是 0.5126，pairwise accuracy 是 0.7563，Top-1 一致率只有 0.0833，Top-3 overlap 是 0.2889。若再把全部 state 的 mean-SINR 摊平，Spearman 会升到 0.8056。摊平相关因此高于状态内排序，不能当作选 assignment 的能力。

在真实 J15 的 Top-10 里面，mean-SINR pairwise accuracy 是 0.6815，Spearman 是 0.4514。高价值候选之间的顺序比全局相关更差。R1 最终 Top-3 assignment 上，预测 SINR 名次相对真实 SINR 名次的平均绝对偏移是 14.33 名。

310114 t=15 把这件事变成了具体的反序。seed 0 上，被选中 UAV 的 mean SINR：

| assignment | 真实 mean SINR | 预测 mean SINR | J15 |
| --- | --- | --- | --- |
| A_pred `2;5;-1;4;-1;-1` | 12.17 | 5.58 | 4e6 |
| A_opt `-1;4;2;5;-1;-1` | 14.82 | 2.05 | 5e6 |

真实顺序是 A_opt 更高。预测顺序反过来。P1 把 A_opt 排到 72–85 名，中位数 74，比静态 Model-S 的 52–54 名更差。P1 的 mean selected J15 是 2.3083e6 bit，低于 Phase-5ZE 锁定的静态控制 2.3417e6 bit。选错 assignment 的是状态内、高价值区域上的顺序，不是那一个摊平 Spearman。

## Q2. 同一个 state 的 120 个候选里，SINR 的平均 Spearman / Kendall 是多少？

mean-SINR：Spearman 0.6507，Kendall 0.5126。

min-SINR：Spearman 0.4542。max-SINR Spearman 0.6326，std-SINR Spearman 0.5456。决策用的平均 SINR 排序，弱于摊平之后的 0.724。

其余主标量的状态内排序：progress Spearman 0.7053、Kendall 0.5666；rate 0.6774 / 0.5473；interference 0.4295 / 0.3113；motion 0.4694 / 0.3825；binary service-active −0.1685 / −0.1382。Binary 的状态内相关差，是因为多数候选的服务状态几乎没有可比差异；它的替换实验也不改变 J15。

## Q3. 真实 J15 Top-10 内部，SINR pairwise accuracy 是多少？

0.6815。Top-5 是 0.6858，Top-20 是 0.6902。高价值区域没有比全池的 0.7563 更好，Spearman 还从全池的 0.6507 降到 Top-10 的 0.4514。全局相关被大量明显较差的候选托高了。

## Q4. P1+TRUE_SINR 能恢复多少 R1−P1 的收益？

0.6379，也就是 +3.0833e5 / +4.8333e5 bit。CI 下界 4.1667e4 > 0，8 个 scenario 里 5 正 3 平 0 负，5 个 seed 全部为正。它是稳定的正改善，但没有达到预注册的 0.70，也没有把 310114 的 A_opt 送回 Top-3。中位名次是 8。

## Q5. TRUE progress 和 TRUE interference 分别恢复多少？

TRUE progress 的 Recovery 是 0.3793（+1.8333e5 bit）。CI 下界 1.6667e4 > 0，但 scenario 有 1 个负例（310110）。TRUE interference 的 Recovery 是 0，CI [−1.4167e5, 1.2500e5] 跨过 0，5 个 seed 里 4 个为负。

同一张表里，TRUE rate 的 Recovery 是 0.6552，是六个单组里最高的，仍然低于 0.70。TRUE motion 是 0.1897。TRUE binary 是 0。

## Q6. 哪一个 response group 对 R1 最必要？

Rate。把预测 rate 注入真实 response 后，Damage 是 +1.2500e5 bit，五个 seed 的 Damage 都为正。SINR 的 Damage 是 +1.0833e5 bit，居第二。Motion +4.1667e4，interference +3.3333e4。Progress 和 binary 的 Damage 是 0。

这里有一个和“替换进 P1”不对称的事实：真实 progress 能帮 P1 收回 37.9% 的缺口，但预测 progress 注入 R1 后 selected J15 不变。Progress 误差不是摧毁 R1 的那一条通道。

## Q7. 哪一个预测 group 注入 R1 后 damage 最大？

仍然是 rate，+1.2500e5 bit。其次是 SINR，+1.0833e5 bit。

## Q8. 310114 t=15：哪项 TRUE response 能把 A_opt 拉回 Top-3？

没有任何预注册的单组或 C1/C2/C3 做到。五个 seed 的 A_opt 名次中位数：

| 评分 | A_opt 名次 |
| --- | --- |
| S_saved | 53, 54, 53, 54, 52，中位 53 |
| P1 | 74, 72, 76, 85, 74，中位 74 |
| P1+TRUE_SINR | 7, 8, 9, 8, 7，中位 8 |
| P1+TRUE_INTERFERENCE | 中位 64 |
| P1+TRUE_PROGRESS | 中位 47 |
| P1+TRUE_RATE | 中位 19 |
| P1+TRUE_MOTION | 中位 59 |
| P1+TRUE_BINARY | 中位 74 |
| C1 | 中位 7 |
| C2 | 中位 6 |
| C3 | 6, 6, 6, 6, 6，中位 6 |
| R1 | 2, 3, 3, 2, 2，中位 2 |

最近的是 C2 和 C3，停在第 6 名。只有把 14 维真实 response 全部放回去的 R1，才回到 Top-3。

seed 0 上，空闲 UAV 的真实值是 0，相对误差没有解释意义。在服务的 UAV 如下。service_active 的绝对误差都小于 0.003，二值服务状态不是名次掉到 74 的原因。

A_pred 的在服务 UAV：

| UAV | 真实 SINR | 预测 SINR | SINR 绝对误差 | 真实 progress | 预测 progress | 真实 rate | 预测 rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.065 | 6.216 | 3.151 | 0.0677 | 0.1456 | 2.023e6 | 2.744e6 |
| 1 | 12.900 | 3.517 | 9.383 | 0.1270 | 0.0609 | 3.797e6 | 1.851e6 |
| 3 | 20.558 | 6.997 | 13.561 | 0.3050 | 0.2657 | 4.430e6 | 2.930e6 |

A_opt 的在服务 UAV：

| UAV | 真实 SINR | 预测 SINR | SINR 绝对误差 | 真实 progress | 预测 progress | 真实 rate | 预测 rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.129 | 3.096 | 11.033 | 0.1311 | 0.0619 | 3.919e6 | 1.672e6 |
| 2 | 13.487 | 0.532 | 12.955 | 0.1290 | 0.0238 | 3.857e6 | 5.131e5 |
| 3 | 16.829 | 2.520 | 14.310 | 0.1390 | 0.0490 | 4.156e6 | 1.365e6 |

A_opt 上三架正在服务的 UAV，真实 SINR 都在 13–17，预测被压到 0.53–3.10。A_pred 的 UAV 0 真实 SINR 只有 3.07，预测却升到 6.22。Progress 的顺序没有反转：真实和预测都是 A_pred 的 progress 更高。反转发生在 SINR，也发生在 rate。

## Q9. 主要问题是 calibration error，还是 ordering error？

决策层面是 ordering error。正斜率的仿射变换不改变候选之间的排序；120 个 state-seed 里，mean-SINR 诊断斜率大于 0 的比例是 0.95。因此数值重标定留不住已经反了的 pair。本阶段没有把任何拟合结果送进 Gate。

按预注册的特征诊断：

| 通道 | 状态内 Spearman | pairwise | scale ratio | 诊断 |
| --- | --- | --- | --- | --- |
| sinr_1 | 0.6507 | 0.7563 | 0.583 | MIXED |
| delta_sinr | 0.6494 | 0.7559 | 0.582 | MIXED |
| interference_1 | 0.4295 | 0.6557 | 0.160 | ORDERING_ERROR |
| delta_interference | 0.2741 | 0.5968 | 0.213 | ORDERING_ERROR |
| service_progress_1 | 0.7148 | 0.7896 | 0.995 | ORDERING_OK |
| delta_service_progress | 0.7500 | 0.8071 | 0.934 | ORDERING_OK |
| rate_1 | 0.6774 | 0.7737 | 0.761 | ORDERING_OK |
| distance_to_target_1 | 0.7254 | 0.8039 | 0.754 | ORDERING_OK |
| delta_distance_to_target | 0.4694 | 0.6913 | 4.583 | ORDERING_ERROR |
| radial_motion_1 | 0.1186 | 0.5403 | 10.527 | ORDERING_ERROR |
| heading_alignment_1 | 0.0936 | 0.5323 | 528.8 | ORDERING_ERROR |

SINR 同时有幅度问题：bias −0.963，scale ratio 0.583。它的状态内 Spearman 0.65、pairwise 0.76，没有差到纯 ORDERING_ERROR 的阈值，也没有好到“只调尺度就能恢复排序”。Top-10 里 31.9% 的 SINR pair 方向是错的，标定消不掉这些错。

Rate 的平均排序达到了 ORDERING_OK 阈值，但它仍是 Damage 最大、单组 Recovery 最高的通道。平均 Spearman 掩盖了少数 value-sensitive pair。

## Q10. 需要单通道修复，还是多通道联合修复？

预注册规则两边都不成立。

六个单组的 Recovery 都低于 0.70。最高的 rate 是 0.6552，SINR 是 0.6379。因此不是单通道修复，也不是 `ERR-A` / `ERR-B` / `ERR-C`。

C3 的 Recovery 是 0.7414，CI 下界大于 0，但低于联合门槛 0.75，而且 310114 的 A_opt 仍在第 6 名。因此也不是 `ERR-D`。C2 与 C3 的 mean J15 相同，都是 2.6667e6；在 SINR 和 progress 已经换成真实值之后，再换成真实 interference，ensemble selected J15 没有再上升。

全部真实 response 仍能复现 R1。缺口留在单组和 C3 都没有覆盖的 response 联合方向上。

## Q11. Gate 是哪一个？

`ERR-E`。

`ERR-F` 不成立：对齐误差是 0，R1 的 selected J15 与 Phase-5ZD 一致。

`ERR-A` 不成立：SINR Recovery 0.6379 < 0.70，而且 310114 的中位名次是 8，不是 Top-3。CI 为正只说明 SINR 有稳定贡献，不够打开 SINR-only 的下一阶段。

`ERR-B` 不成立：progress Recovery 0.3793，不是最大的单组。

`ERR-C` 不成立：interference Recovery 是 0，CI 跨过 0。

`ERR-D` 不成立：C3 Recovery 0.7414 < 0.75。

高 interference 四分位也没有更集中的 P1 ≠ R1。按真实 assigned interference 把 24 个 state 分成四档后，Q1（最低）的 P1 不一致率是 0.933，normalized regret 是 0.350；Q4（最高）的不一致率是 0.500，regret 是 0.333。Q3 的 regret 最低，是 0.0556。Interference 的 response MAE 从 Q1 的 0.106 升到 Q4 的 0.230，但这没有变成更多的选错。R1 所选 assignment 与 P1 所选 assignment 之间，interference 绝对差和 J15 regret 的 Spearman 是 0.525，高于 SINR 的 0.251；可是把真实 interference 换进去以后 selected J15 不变。这是伴随相关，不是可操作的因果通道。

Progress 方面，没有任何 state 的候选池最大 progress 为 0。3 个 state、15 个 state-seed 的最大 progress ≥ 0.8，这些点上 P1 与 R1 的 assignment 一致，regret 为 0。其余 21 个 state、105 个 state-seed 落在部分 progress，不一致率 0.743，mean normalized regret 0.281。P1 ≠ R1 集中在未完成的部分进度，不集中在临近 block completion。

Decision flip：R1 严格分出高低的 pair 共 856800 个，其中 P1 反号 165412 个，比例 0.193。在这些 flip 上，SINR 符号反转率是 0.868，而 R1 顺序被 P1 保持时只有 0.094。Rate 是 0.849 对 0.077。Progress 是 0.716 对 0.097。Interference 是 0.625 对 0.277，保持顺序时仍然经常反号，所以它和 flip 的对照更弱。Motion 在 flip 和 keep 上都约是 0.20。各通道误差对 `|V_P1 − V_R1|` 的状态内 Spearman 均值：SINR 0.625，rate 0.574，progress 0.397，motion 0.231，binary 0.214，interference −0.064。

## Q12. 下一阶段应该修哪一个最小机制？

按 `ERR-E`，下一阶段只审计 response 空间相对冻结 Model-R1 的 value Jacobian，以及 sensitivity-weighted loss。目标是保住真正改变 argmax 的联合方向。Rate 和 SINR 是这条审计里已经看到的两条主要方向：它们的单组 Recovery 最高，注入后的 Damage 也最大，但都没有单独跨过 0.70。

不扩大 predictor，不加 attention、Transformer 或 GNN，不改 HGNN，不改 Model-R1，不做 MAPPO update，不打开 310118–310125，不调用 deployment 决策接口。本阶段到此停止。
