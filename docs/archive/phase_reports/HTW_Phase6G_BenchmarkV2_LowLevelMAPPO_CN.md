# HTW Phase-6G：Benchmark-V2 低层 Residual MAPPO

## 结论

Gate 为 **LL-A V2_LOW_LEVEL_MAPPO_CONFIRMED**。本阶段只训练低层运动残差。高层候选规则、hold、队列记账和 Benchmark-V2 指标没有改。`planner_ready = False`，`deployment_verified = False`。

## Q1. 旧 mappo_original.pt 能否作为初始化？

能。Actor 的权重与现有 ResidualActor 严格对齐，动作维是 2。检查点里没有 critic。

## Q2. MAPPO-V2 最终是 warm-start 还是 scratch？

LEGACY_WARM_START。Actor 从 mappo_original.pt 继续训练，critic 按同一 CentralizedCritic 结构重新初始化。

## Q3. 新 reward 是否完全基于 bit-level served_bits？

是。主项是 sum(served_bits) / B_block，再减去 Phase-1 原有的 projection 和 residual 惩罚。部分传输立即计入。

## Q4. 训练过程中是否修改了高层？

NO。

## Q5. MAPPO 是否仍然只能控制运动，不能修改 assignment？

YES。

## Q6. 同 state 与同 TaskOption 下，MAPPO-V2 相对 fly-toward 平均多传多少 bit？

五个 seed 先在 scenario 内平均，再对 scenario 等权，paired ΔBits = 621455.223968。

## Q7. 95% CI？

[502582.599264, 752570.564350]。

## Q8. 多少 scenario 为正？

1.000000。正的定义是 scenario 均值大于 1 bit。

## Q9. 5 个 seed 是否稳定？

平均为正的 seed 数是 5/5。各种子见 phase6g_seed_stability.csv。

## Q10. MAPPO-V2 是否优于 legacy MAPPO？

V2 减 legacy 的 paired ΔBits = -7377.943032，95% CI [-36716.573697, 22736.927240]。

## Q11. 性能变化主要来自哪里？

相对 fly-toward，SINR 差 -2.061293，interference 差 -0.019590，路径长度差 2.581070，首次服务步差 -0.325195，距离缩短差 0.974155。

## Q12. Full episode + MAX_RATE 下 MAPPO-V2 是否提高最终 payload？

Fly-toward 75465838.671874，legacy 67827319.215939，MAPPO-V2 67904041.543567。这组数没有用来改训练。

## Q13. 是否存在 safety / projection regression？

MAPPO-V2 safety violation rate 0.000247，fly-toward 0.050651，投影事件率 0.276823。没有超过预注册的安全余量。

## Q14. 最终 Gate？

LL-A。

## Q15. 如果 LL-A，是否已经冻结低层？

YES

## Q16. 如果 LL-A，下一阶段是否必须重新生成 J15_CURRENT？

YES

## Q17. 现在是否训练 HGNN？

NO。

## Q18. 现在是否修改 candidate generator？

NO。

## Q19. 320100–320131 是否仍完全 untouched？

YES。
