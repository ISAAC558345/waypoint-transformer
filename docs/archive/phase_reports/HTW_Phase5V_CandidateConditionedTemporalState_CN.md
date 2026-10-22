# HTW Phase-5V Candidate-Conditioned Temporal State

**planner_ready = False**

**deployment_verified = False**

本阶段只加入决策时刻已经存在、并且会随候选链路变化的执行历史。Phase-5U 的静态关系特征、冻结 encoder、ranking loss、候选池和训练场景都没有改。hold remaining 没有输入。未来 SINR、未来轨迹和 J_MAPPO 只作为标签。

## 1. 哪些时间量真实存在

当前任务、UAV 服务进度、上一拍位置、由上一拍位置得到的当前速度，以及上一拍 SINR，都可以从当前环境和 replay 恢复。连续服务步数、option age、AoI、deadline 和 hold remaining 没有记录，没有补造。episode fraction、remaining fraction 和 decision index 对同一个 state 里的全部 assignment 相同，只作为 context。置零后它们不改变任何预测。

## 2. 表示

| 表示 | collision | cosine separation | Fisher | effective rank |
| --- | --- | --- | --- | --- |
| V0 z_static，全部时刻 | 0.2521 | 0.0036 | 9.5987 | 2.7722 |
| V1 z_final，全部时刻 | 0.2428 | 0.0327 | 6.4389 | 2.1767 |
| V0，只看 t=15 | 0.5237 | 0.0004 | 2.0464 | 3.5886 |
| V1，只看 t=15 | 0.5029 | 0.0042 | 0.9216 | 2.3706 |

nearest-neighbor payload contradiction 与 collision 使用同一条规则：近邻 cosine 大于 0.99，且 payload 差至少 1 个 block。t=15 的近邻碰撞几乎没有下降。TEMP 改善的是排序，不是表示碰撞。

## 3. 310114 t=15

五个 seed 都预测 `2;5;-1;4;-1;-1`。池内最优是 `-1;4;2;5;-1;-1`。

| seed | predicted | optimum rank | J_pred | J_opt | regret |
| --- | --- | --- | --- | --- | --- |
| 0–4 | 2;5;-1;4;-1;-1 | 6 | 4.0000e6 | 5.0000e6 | 1.0000e6 |

Phase-5U 的中位排名是 47，预测 payload 是 0，regret 是 5.0000e6。Phase-5V 的中位排名是 6，预测 payload 是 4.0000e6，还差 1 个 block。

预测 assignment 的 same-target 均值是 0.6667，延续进度是 0.0523。最优 assignment 的 same-target 是 0，延续进度是 0。两边的接近速度都约为 -0.98，距离增量都为正，说明两边都在远离各自目标。运动趋势没有把它们分开。剩下的 1 个 block 是“延续当前目标”和“完全切换”之间的差别。

## 4. 按决策时刻

| t | V0 regret | V1 regret | V0 J | V1 J | ΔJ | V1 Top-1 | V1 Top-3 | V1 Spearman | V1 optimum rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5.0000e5 | 5.0000e5 | 2.1250e6 | 2.1250e6 | 0 | 0.625 | 0.875 | 0.739 | 1 |
| 15 | 1.9000e6 | 9.5000e5 | 1.3500e6 | 2.3000e6 | +9.5000e5 | 0.425 | 0.700 | 0.317 | 2 |
| 30 | 2.5000e5 | 2.5000e5 | 2.7500e6 | 2.7500e6 | 0 | 0.875 | 1.000 | 0.798 | 1 |

t=0 和 t=30 与 Phase-5U 相同。全部平均收益来自 t=15。

## 5. V1 相对 V0

scenario-cluster bootstrap，10000 次，scenario 为 cluster。

| 量 | mean | median | std | 正向比例 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| ΔJ_MAPPO | +3.1667e5 | 0 | 9.4854e5 | 0.1167 | [+5.8333e4, +6.7500e5] |
| Δregret | -3.1667e5 | 0 | 9.4854e5 | 0 | [-6.7500e5, -5.8333e4] |
| Δnormalized regret | -0.0833 | 0 | 0.2382 | 0 | |
| ΔTop-1 | +0.0667 | 0 | 0.2494 | 0.0667 | |
| Δoptimum rank | -3.375 | 0 | 8.6613 | 0 | |

scenario 均值：310111 +4.6667e5，310112 +6.6667e5，310114 +1.3333e6，310117 +6.6667e4。310110、310113、310115、310116 为 0。4 个上升，4 个持平，0 个下降。5 个 seed 的平均 ΔJ 都大于 0，范围是 +2.5000e5 到 +4.1667e5。

预先写好的多数规则是上升 scenario 数的两倍大于 8。4 个不满足。因此即使 CI 下界大于 0，也不能记为 V-A。

## 6. TEMP ablation

只做推理置零，没有重训。

| group | disagreement | Δregret | ΔJ | 越过阈值 |
| --- | --- | --- | --- | --- |
| continuation / switch | 0.0500 | +8.3333e3 | -8.3333e3 | 否 |
| service continuity | 0.2000 | +1.9167e5 | -1.9167e5 | 是 |
| motion trend | 0.2250 | +1.7500e5 | -1.7500e5 | 是 |
| global episode context | 0.0000 | 0 | 0 | 否 |

服务延续和运动趋势被模型使用。单纯的延续/切换指示和全局时间没有越过阈值。

## 7. 结论

Gate 是 **V-B**。

t=15 regret 从 1.9000e6 降到 9.5000e5。310114 t=15 的最优排名从 47 移到 6，regret 从 5.0000e6 降到 1.0000e6。总体 CI 下界是 +5.8333e4，没有跨过 0，也没有 scenario 变差。它没有达到 V-A，因为只有 4/8 个 scenario 发生变化，state×seed 的正向比例只有 0.1167，中位数仍是 0。

Q1. 哪些时间量真实存在？ 当前任务、服务进度、上一拍位置、当前速度和上一拍 SINR。连续服务步数、option age、hold remaining 和 AoI 不存在。

Q2. 哪些会随 candidate 改变？ same target、switch、延续进度、浮标进度、接近速度、航向、距离变化，以及只在延续当前目标时保留的上一拍 SINR。全局时间对所有候选相同，ablation 显示它没有改变预测。

Q3. t=15 regret 降到多少？ 从 1.9000e6 降到 9.5000e5 bit。

Q4. 310114 的最优排名是否继续前移？ 是。中位数从 47 移到 6。五个 seed 的预测 payload 从 0 升到 4.0000e6，距离最优还差 1.0000e6。

Q5. J_MAPPO 是否稳定提高？ 均值提高 3.1667e5，CI 全部为正，5/5 seed 为正，没有 scenario 变差。提高集中在 t=15，另外两个时刻不变，只有 4/8 个 scenario 移动，所以还不能称为全面稳定。

Q6. 哪一组有效？ service continuity 和 motion trend。continuation/switch 与 global time 没有越过预先写好的阈值。

Q7. 能否解释“当前更近但 15 步收益更差”？ 只能解释一部分。310114 t=15 不再选择 payload 为 0 的匹配。预测和最优使用同一组浮标，运动趋势几乎一样；差别是预测保留了当前目标，最优则完全切换。最后 1 个 block 仍没有被 TEMP 分开。

Q8. 下一步做什么？ 不做 310118–310125 的 fresh confirmation，也不增加网络。下一阶段只做 TEMP mechanism refinement audit，重点看服务延续和运动趋势为什么把“部分延续”排在“完全切换”前面。

`planner_ready = False`

`deployment_verified = False`
