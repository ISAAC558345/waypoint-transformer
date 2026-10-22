# HTW Phase-5ZP：一次性 Fresh Confirmation

Gate 是 CONF-C（NO_INDEPENDENT_GAIN）。

这不是 deployment verification，也没有进入正式 planner。`planner_ready = False`，`deployment_verified = False`。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。

## Q1. 执行前 310118–310125 是否仍未使用？

是。访问前的冻结检查只读取 Phase-5ZO 已经写好的协议和哈希清单，没有重置这些 scenario。Phase-5ZO 的盲审计结论是这些编号没有进入训练、验证、checkpoint 选择、结构设计、损失设计、阈值设计、候选设计、聚合设计、True-H1 优化、时延优化或 fallback 设计。

## Q2. 第一次访问时间和 consumed 标记

第一次访问的 UTC 时间是 2026-10-09T03:08:28.762919+00:00。访问记录在第一次重置之前写入，`fresh_confirmation_pool_consumed = True`，角色是 ONE_SHOT_FRESH_CONFIRMATION_CONSUMED。

## Q3. 候选系统和 Model-S 是否完全 paired？

是。每个 scenario 的主系统和五个冻结 Model-S seed 都从同一个 scenario seed 的初始状态各自复制环境。初始状态不一致的次数是 0。Model-S 的 payload 是五个冻结 static seed 整局 payload 的平均值，不是 payload 最高的 seed。选择规则在打开预留池之前已经固定为每个 seed 在同一 120 候选上对零响应做 stable argmax。

## Q4. 八个 scenario 的 paired payload

- 310118：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。
- 310119：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。
- 310120：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。
- 310121：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。
- 310122：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。
- 310123：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。
- 310124：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。
- 310125：主系统 3e+06 bit，Model-S 3e+06 bit，Δ 0 bit，tie。

## Q5. mean Δpayload

0 bit。

## Q6. median Δpayload

0 bit。

## Q7. scenario-cluster 95% CI

[0, 0] bit。统计单位是 scenario。bootstrap 抽 10000 次，种子是 0。决策时刻、UAV、seed 和时间步没有被当成独立样本。

## Q8. 正、平、负 scenario

正 0，平 8，负 0。

## Q9. 是否达到 CONF-A 的两个性能条件？

平均差大于 0：否。95% 区间下界大于 0：否。两个条件同时成立才算性能确认。只凭平均值为正不够。

## Q10. illegal assignment

0。

## Q11. state contamination

0。

## Q12. NaN、Inf 和未处理运行异常

NaN 0，Inf 0，未处理运行异常 0。已记录的运行异常次数是 0。

## Q13. fallback 和 timeout

fallback 0 次，timeout 0 次。timeout 只统计新决策没有产出、因而保留当前合法 TaskOption 且不把 hold 重置为 15 的次数。没有使用 500 ms 作为超时门槛。SafetyProjector 失败沿用原有零动作恢复，次数是 0，高层没有因此改目标。

## Q14. 五个 R1 head 的一致性

主系统高层决策 32 次。五 seed Top-1 完全相同的比例是 0.625。平均 consensus strength 是 0.86875。平均 pairwise Spearman 是 0.91868。被选候选的平均 mean rank 是 1.4125。Top-1 与 Top-2 的平均 rank margin 是 1.1375。出现不一致时仍执行冻结的 Rank-Mean V1，没有改聚合。

## Q15. secondary metrics 相对 Model-S

平均剩余队列：主系统 36，Model-S 36。
平均完成块数：主系统 3，Model-S 3。
平均 SINR：主系统 5.37491，Model-S 1.86964。
平均冲突率：主系统 0，Model-S 0。
平均服务进度：主系统 0.0557054，Model-S 0.0740492。
normalized regret 沿用已登记的候选池 J15 oracle 定义。本阶段不运行 15 步 oracle rollout，因此这项没有新的数值。丢包块数作为环境已有诊断：主系统平均 78，Model-S 平均 78。环境没有 fairness 记录。

## Q16. 高层时延

主系统每次高层决策的 total：mean 275.326 ms，median 274.788 ms，P95 298.804 ms，P99 372.664 ms，max 405.697 ms。
分项平均：encoder 5.65244 ms，candidate 91.6761 ms，True-H1 161.09 ms，五个 R1 16.3848 ms，Rank-Mean 0.494088 ms，TaskOption 0.0293594 ms。
这些数字只是当前研究计算机上的诊断。`latency_gate_active = false`。它们不参与 CONF-A、CONF-B 或 CONF-C，也不是 deployment latency verification。

## Q17. 最终 Gate

CONF-C。`independent_algorithm_confirmation = False`。

## Q18. 这 8 个 scenario 还能不能再当 fresh confirmation？

YES。只要本阶段已经打开它们，它们就永久是 consumed confirmation data。以后同一套系统的重复运行只能称为 REPRODUCTION_ON_CONSUMED_CONFIRMATION_DATA。

## 事后描述

下面的分组是 Gate 计算完成之后的解释，不构成新门槛，也不改变主总体。
干扰中位数分组：高干扰组平均 Δ 0 bit（4 个），低干扰组 0 bit（4 个）。
冲突率在 8 个 scenario 上都是 0，所以没有低于中位数的一组。队列中位数分组的平均 Δ 也是 0。

八个 scenario 的累计 payload 都是 3000000 bit，也就是 3 个服务块。主系统和 Model-S 的剩余队列都是 36，丢弃块数都是 78。按环境记账，初始队列 15 块加上 102 块到达，等于完成 3 块、丢弃 78 块、结束时剩下 36 块。两条轨迹的平均 SINR 和服务进度并不相同，说明控制器输出了不同的运动，但服务块完成数没有分开。因此配对差全部为 0。这是确认结果，不是重新选 seed、改聚合或删 scenario 的理由。

True-H1 仍是一步 executor-conditioned response，不是 15 步 rollout。低层 MAPPO 只执行已经给出的 TaskOption，不重新选择 buoy。
