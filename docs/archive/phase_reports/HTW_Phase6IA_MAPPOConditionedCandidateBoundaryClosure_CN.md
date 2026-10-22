# HTW Phase-6I-A：MAPPO 条件下的候选边界闭合审计

## 结论

Gate 为 **BOUND-B K6_REQUIRED**。冻结候选规则是 G1_RATE，k = 6。没有训练。MAPPO-V2 仍冻结在 seed 0。`planner_ready = False`，`deployment_verified = False`。

状态来源在第一次 k=6/7 J15 之前写死：MAX_RATE_INDEPENDENT held for 15 steps and executed by frozen MAPPO-V2 seed0。主分析只用 t=0、15、30。t=45 只作次要记录。k=7 只在已完成的 design set 上计算过：96 个主状态的 mean NGain_7_6 = 0，P95 = 0，因此 confirmation 和闭环不再计算 k=7。已保存的 design k=7 结果保留。干扰分层沿用 Phase-6C 的三分位规则，没有另设阈值。

## Q1. 为什么 69.5% 的 k=5 optimum 触及 rank-5 boundary？

Phase-6H 看到的是 k=5 最优分配里经常包含瞬时速率第 5 名的边。本阶段把同一排序扩到 k=6 和 k=7 后，design 的 mean NGain_6_5 是 0.051308，confirmation 是 0.051505。k=6 在池外给出了更高的 J15_MAPPO_CURRENT，所以 rank-5 接触确实带有截断。

## Q2. k=6 相对 k=5 提高多少 J15_MAPPO_CURRENT？

design 平均 **1217652.92 bit**，confirmation 平均 **1224061.85 bit**。归一化增益分别是 0.051308 和 0.051505。

## Q3. 95% CI？

比特增益的 scenario-cluster 95% CI：design [1148600.31, 1278906.47]，confirmation [1062431.95, 1323054.59]。归一化增益 CI：design [0.048727, 0.053584]，confirmation [0.045400, 0.055195]。

## Q4. confirmation subset 是否复现？

design mean NGain_6_5 = 0.051308，P95 = 0.166667。confirmation mean = 0.051505，P95 = 0.166667。两边是否都不超过 0.02 / 0.05：否。

## Q5. k=7 相对 k=6 是否仍有实质增益？

design mean NGain_7_6 = 0.000000，P95 = 0.000000，95% CI [0.000000, 0.000000]。confirmation 没有重新计算 k=7。稳定实质增益不成立，k7_stable = False。

## Q6. A7* 中有多少 assignment 不能由 k=5 表示？

design 上 A7* 被 k=5 完整覆盖的比例是 0.104167，不能表示的比例是 0.895833。confirmation 覆盖比例是 0.166667，不能表示的比例是 0.833333。k=6 覆盖比例 design 1.000000，confirmation 1.000000。

## Q7. 最终应该冻结 k=5、6、7 中的哪一个？

6。

## Q8. 候选池扩大多少？

主状态上，k=5 平均池大小 5369.56，k=6 是它的 2.481955 倍。k=7 的池大小只统计已完成的 design set，是 k=5 的 6.954907 倍，最大池 37633.00。

## Q9. J15 exhaustive runtime 增长多少？

主状态上，k=5 全池平均 176.886045 秒，k=6 为 2.630536 倍。k=7 耗时只来自 design set，为当时 k=5 的 7.051966 倍。单候选平均 k=5 0.021673 秒，design set 上的 k=7 为 0.015352 秒。k=7 峰值内存 320626688.00 字节。

## Q10. 候选边界问题是否主要发生在高 interference state？

全体主状态的高干扰层 mean NGain_6_5 = 0.000000，k=5 覆盖 0.200000，状态数 40。低干扰层 mean NGain_6_5 = 0.154043，k=5 覆盖 0.000000，状态数 40。分层阈值是本次 development 主状态上的三分位，分位水平没有在看到增益后改动。

## Q11. rank-6/7 edge 为什么会在 H15 下有价值？

A6* 不同于 A5* 的主状态里，新增 rank>5 边有 139 条，只在 k=5 内部重新组合的状态有 0 个。这些新边的平均距离 37.184556 m，队列 2852111.53 bit，瞬时速率 2054238.51，干扰项 0.340786，因果首次服务步 1.064748，该浮标 H15 服务 2835838.84 bit。这是机制记录，没有据此修改 G1_RATE。

## Q12. 不同 MAPPO seed 下，candidate expansion gain 是否基本稳定？

confirmation 主状态上，同一 assignment 的平均变异系数是 0.036776。seed0 的 A6*-A5* 增益大于 1 bit 的状态比例是 0.416667，其他 seed 的平均增益大于 1 bit 的比例是 0.375000。只在 seed0 上为正的状态比例是 0.083333。primary 没有更换。

## Q13. 最终 Gate？

BOUND-B。

## Q14. 是否已经可以开始构建 high-level training dataset？

YES。本阶段没有构建数据集。

## Q15. 现在是否训练 HGNN？

NO。

## Q16. 现在是否重新训练 MAPPO？

NO。

## Q17. 320100–320131 是否仍完全 untouched？

YES。

闭环 sanity 只在 320032–320039。C6 减 C5 的平均 payload 差是 305827.39 bit，95% CI [-1952025.22, 2388145.98]。C7 是否运行：False。
