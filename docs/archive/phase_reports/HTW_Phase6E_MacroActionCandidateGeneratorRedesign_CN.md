# HTW Phase-6E：宏动作候选生成器重设计审计

Gate 是 CAND-D（TOP5_REQUIRED）。

Phase-6B 仍是 BITQ-A，Phase-6C 仍是 DYN-A，Phase-6D 仍是 TEMP-F。环境是 `L3BBitQueueEnvV2`。hold 仍是 15，低层仍是 deterministic fly-toward。320100–320131 没有打开。没有训练。`J15_CURRENT` 只用于候选覆盖评价，未来到达保持为 0，没有进入候选分数。`planner_ready = False`，`deployment_verified = False`。

Design 集是 320000–320047，confirmation 集是 320048–320063。这个划分在第一次 J15 之前写入协议，之后没有改动。Reference 是 instantaneous-rate top-5 池里的 J15_CURRENT optimum，名称是 top5 candidate-pool optimum，不是 global Oracle。G1 k=5 与这个 reference 是同一个池，它的 coverage 为 1、regret 为 0 是定义，不是新的压缩结果。

## Q1. instantaneous-rate top-3 为什么漏掉 J15 高价值边？

rate top-3 覆盖到的 reference 边有 635 条，漏掉 252 条。覆盖边的距离/eta/rate/队列/几何可服务比例是 24.5656 m、0.745658 步、3.71198e+06 bit/s、2.98722e+06 bit、0.949995。漏掉的边是 41.9206 m、1.3778 步、1.55247e+06 bit/s、3.08599e+06 bit、0.907979。漏掉的边瞬时 rate 大约只有覆盖边的一半，距离大约远 17 m，eta 大约多 0.63 步，队列大小接近。它们的几何可服务比例仍有 0.91，所以这些边不是飞不到，而是 instantaneous-rate top-3 没有给它们留位置。

## Q2. 扩大到 k=4 是否足够？

G1 k=3 full coverage 0.276042，mean nregret 0.120426，通过 False。G1 k=4 full coverage 0.453125，mean nregret 0.0535033，P95 0.0907115，通过 False。G1 的 K4_SUFFICIENT = False。

## Q3. 是否必须 k=5？

G1 k=5 是 reference，full coverage 1，mean nregret 0。design 上通过 Gate 的规则只有 G1、G4、G5、G6 的 k=5。没有任何 k=3 或 k=4 通过。因此候选池必须用 k=5，并记下它的组合规模。

## Q4. distance / ETA proposal 是否优于 instantaneous-rate？

G2_DISTANCE k=3 full coverage 0.135417，mean nregret 0.307527，mean pool 861.781，通过 False。G2_DISTANCE k=4 full coverage 0.28125，mean nregret 0.268197，mean pool 2509.59，通过 False。G2_DISTANCE k=5 full coverage 0.520833，mean nregret 0.24651，mean pool 6604.78，通过 False。G3_ETA k=3 full coverage 0.0520833，mean nregret 0.423508，mean pool 542.224，通过 False。G3_ETA k=4 full coverage 0.0729167，mean nregret 0.362066，mean pool 1815.41，通过 False。G3_ETA k=5 full coverage 0.203125，mean nregret 0.298699，mean pool 5525.61，通过 False。G5_REACHABLE_RATE k=3 full coverage 0.276042，mean nregret 0.120426，mean pool 504.036，通过 False。G5_REACHABLE_RATE k=4 full coverage 0.453125，mean nregret 0.0535033，mean pool 1626.55，通过 False。G5_REACHABLE_RATE k=5 full coverage 1，mean nregret 0，mean pool 4692.62，通过 True。

## Q5. queue-rate 是否改善 macro coverage？

G4_QUEUE_RATE k=3 full coverage 0.203125，mean nregret 0.166729，mean pool 302.057，通过 False。G4_QUEUE_RATE k=4 full coverage 0.333333，mean nregret 0.0554241，mean pool 1375.22，通过 False。G4_QUEUE_RATE k=5 full coverage 0.947917，mean nregret -0.00808517，mean pool 4643.25，通过 True。G6_REACHABLE_QUEUE_RATE k=3 full coverage 0.203125，mean nregret 0.170809，mean pool 300.651，通过 False。G6_REACHABLE_QUEUE_RATE k=4 full coverage 0.328125，mean nregret 0.0629372，mean pool 1369.93，通过 False。G6_REACHABLE_QUEUE_RATE k=5 full coverage 0.958333，mean nregret -0.00109475，mean pool 4629.1，通过 True。

## Q6. 哪个规则的 coverage、regret、complexity 综合最好？

| 规则 | k | full coverage | edge coverage | mean nregret | P95 nregret | mean pool | 通过 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1_RATE | 3 | 0.276042 | 0.746788 | 0.120426 | 0.173387 | 500.484 | False |
| G1_RATE | 4 | 0.453125 | 0.878819 | 0.0535033 | 0.0907115 | 1626.39 | False |
| G1_RATE | 5 | 1 | 1 | 0 | 0 | 4692.62 | True |
| G2_DISTANCE | 3 | 0.135417 | 0.467882 | 0.307527 | 0.488083 | 861.781 | False |
| G2_DISTANCE | 4 | 0.28125 | 0.577257 | 0.268197 | 0.389625 | 2509.59 | False |
| G2_DISTANCE | 5 | 0.520833 | 0.692014 | 0.24651 | 0.26331 | 6604.78 | False |
| G3_ETA | 3 | 0.0520833 | 0.375868 | 0.423508 | 0.575722 | 542.224 | False |
| G3_ETA | 4 | 0.0729167 | 0.484288 | 0.362066 | 0.533333 | 1815.41 | False |
| G3_ETA | 5 | 0.203125 | 0.610764 | 0.298699 | 0.499167 | 5525.61 | False |
| G4_QUEUE_RATE | 3 | 0.203125 | 0.660069 | 0.166729 | 0.212183 | 302.057 | False |
| G4_QUEUE_RATE | 4 | 0.333333 | 0.858594 | 0.0554241 | 0.0895041 | 1375.22 | False |
| G4_QUEUE_RATE | 5 | 0.947917 | 0.989931 | -0.00808517 | 0.0126273 | 4643.25 | True |
| G5_REACHABLE_RATE | 3 | 0.276042 | 0.746788 | 0.120426 | 0.173387 | 504.036 | False |
| G5_REACHABLE_RATE | 4 | 0.453125 | 0.878819 | 0.0535033 | 0.0907115 | 1626.55 | False |
| G5_REACHABLE_RATE | 5 | 1 | 1 | 0 | 0 | 4692.62 | True |
| G6_REACHABLE_QUEUE_RATE | 3 | 0.203125 | 0.661979 | 0.170809 | 0.226266 | 300.651 | False |
| G6_REACHABLE_QUEUE_RATE | 4 | 0.328125 | 0.858594 | 0.0629372 | 0.0895041 | 1369.93 | False |
| G6_REACHABLE_QUEUE_RATE | 5 | 0.958333 | 0.993056 | -0.00109475 | 0.0126273 | 4629.1 | True |

按预注册优先级冻结的规则是 G1_RATE k=5。G4 k=5 的 mean nregret 是 -0.00808517，它的池有时比 rate top-5 reference 的 J15 更高，但 full coverage 是 0.947917，比 G1 的 1 低超过预注册的 0.02，所以没有进入最终等价带。G5 k=5 与 G1 k=5 的 coverage、regret 和池大小相同，简单性规则留下 G1。

## Q7. 冻结规则在 320048–320063 上是否通过？

confirmation full coverage 1，mean nregret 0，P95 nregret 0，state pass fraction 1，非法分配 0。通过：True。

## Q8. full-assignment coverage 是多少？

design 是 1。confirmation 是 1。

## Q9. mean / P95 normalized pool regret 是多少？

design mean 0，P95 0。confirmation mean 0，P95 0。

## Q10. candidate pool size 增长多少？

G1 k=3 的 scenario 均值池大小是 500.484，最大状态池是 1063。冻结规则的均值池大小是 4692.62，P95 5004.61，最大状态池是 6492。相对 G1 k=3 的均值倍数是 9.37617。

## Q11. J15 exhaustive evaluation 的计算代价是多少？

按该决策上实测的每次分配耗时乘以该池分配数，冻结规则每个决策的均值是 10.0306 秒，中位数 8.18498 秒，P95 17.015 秒。这个时间是单决策的 exhaustive J15，不含跨规则缓存。

## Q12. high-interference state 是否更容易漏掉宏动作边？

预注册标志 INTERFERENCE_CANDIDATE_SENSITIVITY = False。它看的是冻结规则 G1 k=5，而这个池对 reference 的 coverage 在每个状态都是 1，所以高干扰和低干扰都是 1，这个标志不能发现漏边。改看能够漏边的 G1 k=3：干扰不一致状态的 full coverage 是 0.284722，一致状态是 0.222222；高三分位是 0.229167，低三分位是 0.173611。G1 k=4 的高三分位是 0.635417，低三分位是 0.190972。高干扰状态没有更差；漏掉完整 reference assignment 的情况更多出现在低干扰状态。

## Q13. idle 是否占掉有效 buoy candidate slot？

没有。每个 UAV 的 top-k 只放浮标，idle 用 -1 单独加入合法联合分配，不占用 buoy slot。本阶段没有改 idle 的最终策略。

## Q14. 是否已经关闭 Phase-6D 的候选覆盖失败？

top-3 的覆盖失败没有关闭，k=3 和 k=4 仍然通不过 Gate。关闭的是“没有可用候选池”这一步：k=5 的 rate 池在 confirmation 上满足覆盖标准，Phase-6F 可以使用它。Phase-6D 对 instantaneous-rate top-3 的失败结论仍然成立。

## Q15. 下一阶段是否可以重新比较 hold-aware joint planning 与 independent max-rate？

YES

## Q16. 现在是否可以训练 HGNN？

NO。

## Q17. 320100–320131 是否完全 untouched？

YES。

## max-rate 闭环与 reachability

max-rate 与 top5-reference 边的重合比例均值是 0.152257。reference 边的距离/eta/队列/rate 均值是 28.3807 m、0.865152 步、3.01971e+06 bit、2.86983e+06 bit/s。max-rate 边是 29.7392 m、0.921325 步、3.01934e+06 bit、2.87689e+06 bit/s。两者平均距离只差约 1.36 m，平均 eta 只差约 0.056 步。边集合重合只有 0.152，但平均距离和 eta 接近。max-rate 的闭环优势不是把 UAV 派向明显更远或明显更近的另一类目标。

附加标志：SELECTED_GENERATOR = G1_RATE；SELECTED_K = 5；K4_SUFFICIENT = False；INTERFERENCE_CANDIDATE_SENSITIVITY = False；IDLE_SLOT_INTERACTION = False；MEAN_POOL_SIZE = 4692.62；P95_POOL_SIZE = 5004.61；MEAN_J15_EVAL_RUNTIME = 10.0306 秒/决策。
