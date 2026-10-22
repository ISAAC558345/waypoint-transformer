# HTW Phase-6B：比特队列状态转移

Gate 是 BITQ-A（BIT_LEVEL_ACCOUNTING_VALID）。

Phase-5ZP 仍然是 CONF-C。Phase-5ZQ 仍然是 ATTR-C。Phase-6A 仍然是 BV2-E。310100–310125 没有打开。320100–320131 没有打开。没有训练，没有定义 lambda，没有加入 continuity penalty。`planner_ready = False`，`deployment_verified = False`。

## Q1. 旧环境每个 step 的事件顺序是什么？

SafetyProjector 和 reward 不在 `L3BEnv.step` 里面。冻结 MAPPO 的调用顺序是：先投影动作，投影失败就把执行动作清成 0，然后调用 `step`，最后用 payload 增量除以 `PAYLOAD_SCALE=1e6` 计算 reward。

`step` 内部顺序是：裁剪动作、积分位置、写入 assignment、目标变化时把 UAV progress 清零、按距离和队列和冲突确定链路、计算 SINR 和 rate、按当前时刻读取到达、用 progress 完成整块并增加 payload、用完成块扣队列、再把到达加进去并按容量丢弃、时间加 1、判断终止、生成 info。

V2 保留这个顺序。队列扣减仍然发生在本步到达之前。改变的只是扣减量：从“完成一整块才减 1”换成“本步 `rate × Δt` 能传多少就减多少”。

## Q2. rate 的真实单位是什么？

bit/s。公式是 `bandwidth_hz * log2(1 + SINR)`，`bandwidth_hz = 1e+06`。代码没有把 rate 乘上 Δt。`delta_t = 1` 秒，所以本步可传比特是 `R × Δt`。reference rate 用同一公式，在 1 m 处的值是 9.96723e+06 bit/s。

## Q3. Benchmark-V2 的 Q_bits 如何定义？

每个浮标一个 float64 状态 `Q_bits`，单位是 bit。块大小从 `L3BConfig.block_payload_bits` 读取，是 1e+06 bit。初始队列和容量都乘这个块大小。`queue_blocks_equivalent = Q_bits / B_block` 只是派生量。容量对所有浮标都是 4 block，旧代码没有按远近簇分开容量。

## Q4. 每一步 served_bits 如何计算？

先由当前位置和 assignment 得到合法链路和 rate。`raw = Σ R × Δt`，再 `S = min(Q_before, raw)`。本步到达不参与这个 min。S 不小于 0，服务之后的队列也不小于 0。

## Q5. queue / arrival / service / drop 是否严格守恒？

守恒式是 `Q_before + generated_arrival = Q_after + served + dropped + float_snap`。11 项确定性测试全部通过。四个 scripted development scenario 的最大绝对误差是 4.65661e-09 bit，容差是 1 bit。

## Q6. payload 是否与真实 queue decrement 完全一致？

是。`cumulative_payload_bits` 只累加本步 `S_bits`。它不由 progress 推算，也不用完成块乘块大小代替。

## Q7. partial transmission 是否立即计入 payload？

是。队列里有 1 block、本步只能传 0.4 block 时，payload 立刻增加 0.4 block，队列剩下 0.6 block。

## Q8. target switch 后已发送 bit 是否保留？

保留。先传 0.4 block，切走再回来传 0.6 block，浮标上的累计传输是 1 block。切换不会把已经离开队列的比特加回去，也不会从 payload 里删掉。

## Q9. 换 UAV 后是否能够继续服务同一 buoy？

能够。余数 `service_remainder_bits` 记在浮标上。UAV1 传 0.4 block、UAV2 再传 0.6 block，累计交付 1 block，队列减少 1 block。

## Q10. completed block 是否已经降为 secondary metric？

是。它只对浮标余数做 `floor((remainder + S) / B)`。0.3B、0.2B、0.5B 三步的 payload 分别增加这三笔，完成块计数只在最后一步加 1。

## Q11. 是否存在 double decrement？

不存在。一次传 1.5 block 时，队列只减少 1.5 block，完成块计数加 1，不会再减 1 block。

## Q12. 旧 progress 现在还控制哪些量？

它只作为 `legacy_task_progress` 诊断量，目标变化时仍会清零。它不再改变 `Q_bits`，也不再改变 payload。

## Q13. frozen MAPPO 能否接口兼容？

结论是 MAPPO_RETRAINING_LIKELY_REQUIRED。观测向量里的 queue 原来是块数，不是归一化比例；V2 可以用 `Q_bits / B_block` 恢复同一数值尺度。progress 通道仍被观测读取。但队列会随部分传输连续下降，reward 用的 payload 增量也不再只是 0 或整块，冻结权重看到的状态分布已经变了。本阶段没有为了兼容去改比特记账。

## Q14. 是否需要现在修改 MAPPO？

现在不修改。接口尺度可以用块当量对齐，但冻结策略不能因此被当成 V2 上的有效策略。重训不在本阶段。

## Q15. 是否有任何证据支持现在加入 continuity penalty？

NO。Phase-6A 没有证实切换会损失比特吞吐。lambda_switch 和 lambda_progress 都是 NOT_DEFINED。

## Q16. 320100–320131 是否仍完全 untouched？

YES。本阶段只跑了确定性测试和 320000–320003 的 scripted sanity check。

## Q17. 最终 Gate 是什么？

BITQ-A（BIT_LEVEL_ACCOUNTING_VALID）。

状态链是：到达进入比特队列；association 决定链路；SINR 得到 rate；`served_bits = min(Q_bits, rate × Δt)`；队列扣掉这部分；payload 加上同一笔。completed blocks 是旁路计数，不反向控制队列。

旧 `src/wt/l3b/env.py` 没有修改。Phase-5 结果仍走 legacy environment。
