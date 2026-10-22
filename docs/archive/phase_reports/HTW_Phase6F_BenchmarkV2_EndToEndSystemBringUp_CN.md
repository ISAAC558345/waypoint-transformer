# HTW Phase-6F：Benchmark-V2 端到端系统接通

## 结论

Gate 为 **RUN-A END_TO_END_SYSTEM_OPERATIONAL**。本阶段只把已冻结的 `L3BBitQueueEnvV2`、`G1_RATE k=5`、`J15_CURRENT` 和 deterministic fly-toward 串成一条 episode。`planner_ready = False`，`deployment_verified = False`。

## Q1. 完整系统能否从 reset 跑到 episode 结束？

Smoke 完成 4/4，integration 完成 8/8。完成的 episode 都从 reset 走到 T=48。

## Q2. candidate → J15_CURRENT → assignment 是否完整工作？

每个高层决策调用 Phase-6E 的 `build_pools` 和 `batch_j15`，再用已有的字典序、1 bit 容差选出 `A_star`。平均候选池大小为 4902.521。链路完整：是。

## Q3. TaskOption hold=15 是否正确？

高层决策时刻是 t=0、15、30、45。`remaining_hold` 在决策时设为 15，之后每步减 1，持有期内不会被重置。t=45 只剩下 3 个环境步，计数仍从 15 递减，episode 在 T=48 停止。Hold 语义：正确。

## Q4. fly-toward 是否正确执行 assignment？

低层调用 `wt.l3b.policies.fly_toward`。目标不是 idle 时，动作是朝当前浮标的单位步；idle 时位移为 0。Phase-6C/6D 的这个执行器没有单独的服务半径内停留规则，本阶段没有新增停留或最小运动。方向检查：通过。

## Q5. shadow evaluator 是否完全不污染主环境？

每次 `batch_j15` 前后都比对主环境的队列、位置、关联、SINR、速率、服务与丢弃、累计 payload、步数、到达、RNG 和已有 TaskOption。该环境没有速度状态，检查记录为速度字段不存在，没有另造速度。污染计数为 0。

## Q6. bit queue accounting 是否继续严格守恒？

逐步和 episode 级都使用 Phase-6B 的 `conservation_error_bits`，容差是环境的 `tol_bits`。失败次数为 0。

## Q7. 所有 assignment 是否合法？

非法计数为 0。

## Q8. 4 个 smoke scenario 是否全部通过？

是。

## Q9. 8 个 integration regression 是否全部通过？

是。payload 大小不作为失败条件。

## Q10. 完整一个 episode 平均耗时是多少？

J15 系统的平均 episode 时间为 56.431 秒。

## Q11. J15 exhaustive 当前耗时多少？

平均每个高层决策 14.033 秒。这个耗时不作为失败，本阶段没有做 batching 或 pruning。

## Q12. MAX_RATE baseline 是否同样能完整跑通？

是。累计 payload：320000=75667549.175096, 320001=74310033.924698, 320002=80382051.019232, 320003=67640470.427508。

## Q13. 是否在本阶段为了性能修改了任何参数？

NO。

## Q14. 最终 Gate 是否为 RUN-A？

是。当前 Gate 是 RUN-A。

## Q15. 如果 RUN-A，下一阶段是否只训练 low-level MAPPO？

YES

## Q16. 现在是否训练 HGNN？

NO。

## Q17. 现在是否优化 k=5 / J15 runtime？

NO。

## Q18. 320100–320131 是否仍完全 untouched？

YES。
