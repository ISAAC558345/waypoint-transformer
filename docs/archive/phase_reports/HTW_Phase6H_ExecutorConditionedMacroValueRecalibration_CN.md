# HTW Phase-6H：按冻结低层执行器重算宏动作价值

## 结论

Gate 为 **ALIGN-A EXECUTOR_CONDITIONED_MACRO_PLANNING_CONFIRMED**。主低层检查点固定为 `phase6g_seed0.pt`，规则是最小 seed ID，没有按 payload、SINR 或开发表现挑选。seed 1–4 只做了固定 assignment 集合上的稳健性审计。

没有训练，没有改 MAPPO-V2，没有改 `G1_RATE k=5` 或 `hold=15`。`MAPPO_V2_FROZEN = True`。高层 `NOT_TRAINED_YET`。`planner_ready = False`，`deployment_verified = False`。没有 deployment PASS。

320016–320047 上，四个系统的平均 cumulative payload 是：

| 系统 | 高层 | 低层 | 平均 payload (bit) |
| --- | --- | --- | --- |
| C3，历史对照 | MAX_RATE | fly-toward | 74808486.31 |
| C2，执行器匹配 | argmax J15_MAPPO_CURRENT | MAPPO-V2 seed0 | 72698129.23 |
| C1，执行器错配 | argmax J15_FLY_CURRENT | MAPPO-V2 seed0 | 69733468.45 |
| C0 | MAX_RATE | MAPPO-V2 seed0 | 67624412.18 |

C2 稳定高于同一个 MAPPO 执行器下的 C0 和 C1。C2 仍低于历史的 MAX_RATE + fly-toward。这个历史对照不进入 ALIGN 判定。

Shadow contamination = 0（576 次检查）。记账最大绝对误差 9.3e-9 bit。非法分配 = 0。Actor 无循环隐状态，`ACTOR_STATELESS = True`。检查点在评测前后哈希一致。

## Q1. 为什么同任务 H15 上 MAPPO 更好，MAX_RATE 全 episode 却没有提高？

同一初始状态上，MAPPO 没有更差。320000–320015 的 t=0 有 16/16 个窗口：MAX_RATE 分配相同，UAV 位置相同。这些窗口里 MAPPO-V2 减 fly-toward 的服务差均值是 **+699601 bit**，负窗口数是 0。这和 Phase-6G 的同任务增益方向一致。

全 episode 下降发生在轨迹已经分开之后。64 个高层决策里，40 个（比例 0.625）后来选了不同 assignment。另外 8 个“分配字符串相同”的后续窗口，UAV 位置已经不同，服务差均值是 **-9896846 bit**，其中 7 个为负。把预注册规则要求的全部同一分配窗口放在一起，均值被这 8 个分叉状态拉到 **-2832548 bit**，中位数却接近 0。

因此全 episode 下降不是“同一状态、同一 TaskOption 上低层执行失败”。低层改变了位置，后续 MAX_RATE 经常换分配，分叉后的状态上也少了很多服务比特。

## Q2. 两套 J15 的 assignment ranking 差多少？

320000–320015、每个决策时刻的同一 `G1_RATE k=5` 候选池上，scenario 等权均值是：

- Spearman 0.787554
- Kendall 0.716476
- Top-3 overlap 0.119792
- Top-5 overlap 0.125000

整体次序还有相关，但排在最前面的少数 assignment 经常换掉。

## Q3. Top-1 disagreement 是多少？

**0.375000**（64 个状态里 24 个 top-1 不同）。MAPPO top-1 在 fly ranking 里的平均名次是 **1326**。预注册的“明显不同”门槛是 0.20，这项达到了。

## Q4. matched planner 是否好于 mismatched planner？

是。C2 减 C1 的平均 payload 差是 **2964660.78 bit**，95% CI **[1107583.76, 4750679.60]**，下界大于 0。

## Q5. matched planner 是否稳定超过同一个 MAPPO 下的 MAX_RATE？

是。

## Q6. C2 vs C0 的 mean、CI、positive fraction？

mean **5073717.05 bit**，median **6350261.15 bit**，scenario-cluster bootstrap 95% CI **[2989979.76, 6945103.51]**，positive fraction **0.843750**（27/32）。三项都过了预注册门槛：均值大于 0，CI 下界大于 0，正比例不低于 0.60。

## Q7. C2 vs C1 的 mean 和 CI？

mean **2964660.78 bit**，95% CI **[1107583.76, 4750679.60]**。

## Q8. 预测的宏价值是否更接近真实 H15 return？

没有达到预注册的“明显提高”。完整 15 步窗口共 96 个（32 个场景 × t=0,15,30）。t=45 只有 3 步，已标成 `truncated_window`，没有放进相关。

| 控制器 | Spearman | Kendall | MAE (bit) | bias (bit) |
| --- | ---: | ---: | ---: | ---: |
| C1，J15_FLY 预测对 MAPPO 真实回报 | 0.615574 | 0.429471 | 2841184.04 | -2715579.50 |
| C2，J15_MAPPO 预测对 MAPPO 真实回报 | 0.609682 | 0.451314 | 3502154.08 | -3502154.08 |

Spearman 差是 -0.0059，没有达到 +0.05，所以 `EXECUTOR_VALUE_ALIGNMENT_IMPROVED = False`。C2 的 bias 等于 -MAE，说明这些完整窗口里因果预测都没有高于真实服务。标签按 Phase-6D 把未来到达置零，真实环境仍有到达，所以预测系统性偏低。闭环 payload 的提高来自换了被选中的 assignment，不是来自这条 Spearman 明显变高。

## Q9. MAX_RATE + MAPPO-V2 下降的真正原因是什么？

预注册分类结果是 **MIXED**。

同一状态的 t=0 不是低层执行失败。下降同时包含两件事：轨迹分开以后，MAX_RATE 有 62.5% 的决策换了 assignment；已经分叉的状态上，服务比特也下降。明细在 `phase6h_maxrate_executor_shift.csv`。

## Q10. k=5 optimum 是否经常压在 rank-5 boundary？

是，经常包含边界边，但边本身并没有堆在 rank 5 上。closed-loop C2 的 128 个 J15_MAPPO 最优分配里，**0.695312** 至少含一条 instantaneous-rate rank-5 边。这些非空闲边的名次比例是：rank 1 为 0.202，rank 2 为 0.230，rank 3 为 0.196，rank 4 为 0.199，rank 5 为 0.173。

本阶段没有把 k 改成 6。候选覆盖是否该扩大，留给后续单独审计，这里没有开始。

## Q11. 不同 MAPPO seed 对同一个 assignment 的 H15 return 是否稳定？

服务比特稳定，五个近邻 assignment 的名次不稳定。320048–320063 上，同一 assignment 跨 seed 0–4 的平均变异系数是 **0.041195**，低于 0.50 的 ALIGN-D 门槛。

seed0 选出的 assignment 在其他 seed 的五个固定候选里，名次次数是 rank1 69、rank2 55、rank3 56、rank4 44、rank5 32（共 256）。进入前两名的比例是 **0.484375**。没有为 seed 1–4 重新穷举，也没有改 primary。

## Q12. 高层 macro label 是否必须由实际冻结的低层执行器重新生成？

**YES。** Top-1 disagreement 0.375，而且在同一个 MAPPO-V2 seed0 上，J15_MAPPO 规划比 J15_FLY 规划多 2.96e6 bit，CI 下界大于 0。继续用 fly-toward 生成的旧标签做最终高层监督，和当前执行器不匹配。

## Q13. 最终 Gate？

**ALIGN-A** `EXECUTOR_CONDITIONED_MACRO_PLANNING_CONFIRMED`。

## Q14. 如果 ALIGN-A，下一阶段是否可以开始构建 Benchmark-V2 learned high-level dataset？

**YES。** 监督标签应使用 `J15_MAPPO_CURRENT`。本阶段没有构建该数据集，也没有把旧 Phase-5 HGNN 权重拿来用。

## Q15. 现在是否直接训练 HGNN？

**NO。**

## Q16. 现在是否重新训练 MAPPO？

**NO。** 若以后改 MAPPO，必须重生成全部高层 macro label。

## Q17. 320100–320131 是否仍完全 untouched？

**YES。**

## 运行时间

运行时间不参与 Gate。label-shift 上，J15_FLY 全池平均 7.49 s（单候选 0.00153 s），J15_MAPPO 全池平均 74.81 s（单候选 0.0151 s，首个候选 0.0101 s）。closed-loop 一次高层决策平均：C1 11.60 s，C2 114.10 s。完整 episode 平均：C0 0.16 s，C1 46.72 s，C2 456.78 s，C3 0.015 s。没有做 pruning、batching 或 surrogate。
