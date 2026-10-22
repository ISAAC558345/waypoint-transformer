# HTW Phase-6A：Benchmark-V2 通信效用与服务连续性审计

Gate 是 **BV2-E（ACCOUNTING_MODEL_INVALID）**。

Phase-5ZP 仍然是 CONF-C。Phase-5ZQ 仍然是 ATTR-C。310100–310125 没有打开。320100–320131 没有打开。本阶段没有训练，没有选择 lambda。`planner_ready = False`，`deployment_verified = False`。

320000–320063 是新的 development id。环境生成器对它们只做初始 UAV y 的 ±2 m 抖动，浮标位置、初始队列和到达过程与历史场景相同。因此这批数据能检查指标，不能制造新的业务量动态范围。

## Q1. 旧 payload 是 throughput 还是 task completion？

是 task completion。队列单位是 block。瞬时 rate 只推动当前 UAV 的 progress：

`progress += min((R / reference_rate) / service_slots, 1)`

progress 达到 `1 - 1e-9` 时，payload 增加 `1e6` bit，被服务浮标的队列减 1 block。没到 1 的 progress 不进入 payload。

## Q2. 未完成的已传输数据为什么被记为 0？

环境没有比特累加器。rate 的时间积分不写入队列，也不写入 payload。progress 存在 UAV 上。UAV 更换目标时，这段 progress 先被清零，再从 0 重新计时。已经按 rate 走过的时间不留下可累计的比特。

## Q3. continuous delivered bits 能否从现有环境严格计算？

不能。不存在 bit-level queue。`min(R Δt, remaining_queue_bits)` 里的 remaining bits 必须把 block 乘上 1e6，这是额外解释，不是状态变量。本阶段只记录诊断量 `Σ rate_bps · Δt`，并标明它不是环境交付量。

## Q4. continuous delivered bits 和 queue reduction 是否守恒？

块守恒成立。全部参考回合里，E2 = 初始块 + 到达 − 丢弃 − 剩余，与完成块的最大绝对误差是 0。因此 E2 乘上 1e6 就是 E0，它不是另一套吞吐。

假设 rate 积分相对 E0 的平均差是 1.56442e8 bit。两者不守恒。

## Q5. 旧 completed-block payload 的 saturation ratio 是多少？

冻结 Phase-5 控制器和 Model-S 的 E0 saturation ratio 都是 **1**。64 个 scenario 全部是 3 个 block、3e6 bit。fly-toward 是 0.671875，queue-rate heuristic 是 0.3125，random legal 是 0.21875，瞬时 rate 诊断是 0.1875。

## Q6. 新 endpoint 是否能区分不同策略？

连续比特 endpoint 不能采用。旧 E0 上，冻结控制器和 Model-S 在 64 个 scenario 中的可区分比例是 0。random legal 相对冻结控制器的可区分比例是 0.8125，fly-toward 相对瞬时 rate 诊断是 0.96875。预先登记的可区分门槛是 0.75，但这只说明某些非学习参考策略能改变完成块数，不能把 rate 积分升级成主指标。

## Q7. 新 development data 上是否重新出现 higher SINR/rate 但长期收益不变？

在完成块上，是。冻结控制器平均 SINR 4.82171，Model-S 2.03508。平均 rate 371495 bps 对 335160 bps。两边完成块都是 3，丢弃都是 78，剩余队列都是 36。

在假设 rate 积分上，不是“没有增益”：冻结控制器 1.06990e8 bit，Model-S 9.65260e7 bit。这个积分不是环境交付量。

## Q8. target switching 是否和长期通信效用负相关？

冻结控制器平均切换 7.85938 次，Model-S 3.06875 次。中位服务段 16.4531 对 33.9375。被放弃的 progress 是 0.338102 对 0.014999。服务工作量是 3.33810 对 3.01500。多出来的工作被清掉，完成块仍然是 3。

E0 在冻结控制器内部没有方差，切换对 E0 的 Spearman 无法计算。切换对假设 rate 积分的 Spearman 是 -0.0881312，服务段对假设积分是 0.201481。预先登记的负相关门槛是 Spearman ≤ -0.20。没有达到。

六个控制器均值上，切换对 E0 的 Spearman 是 -0.20292。样本量是 6，而且 random 同时有很高的切换和很高的 rate 积分，不能把这个系数当成连续性损失已经证实。

## Q9. service progress 应该属于 UAV、buoy，还是 bit queue？

现在属于 UAV。UAV 服务浮标 A 一半后切到 B，A 的队列块没有减少，A 上的计时被丢掉。另一架 UAV 后来服务 A，从 progress = 0 开始，不继承前一架的计时。

这不是分组传输复位，也不是握手复位。它是“同一架 UAV 连续驻留，累计够 service_slots 个等效时隙才算完成一块”的人工计时。完成块是任务指标，不能直接当成通信吞吐。

P0：当前 UAV-owned reset。合理于“必须连续驻留”，不合理于“已经发出的信息”。
P1：buoy-owned progress。计时可以换 UAV 接着走，服务时长约束还在，但仍然不是比特。
P2：比特队列由 rate 扣减，不再使用人工 block progress。这才和“最大化信息传输”同单位。它会取消现在的连续驻留完成条件。

本阶段不改这三条规则。

## Q10. 当前 hold=15 是否仍然是问题？

离线诊断把同一次高层 assignment 保持下去，不作为在线策略。窗口内平均完成块是：H=1 为 0.00390625，H=3 为 0.046875，H=5 为 0.171875，H=10 为 0.682292，H=15 为 1。对应的假设 rate 积分从 2.85706e6 增到 3.56635e7。窗口内守恒误差是 0。

这表示更长的不间断服务会累计更多完成块。它不是把 hold 改成别的数的依据。在线 hold 仍是 15。

## Q11. 真正问题是 metric saturation、continuity loss，还是两者都有？

正式 Gate 是 **BV2-E**。可信的 bit-level throughput 还不存在，所以不能把 BV2-B 或 BV2-C 写成结论。

在任务指标上，旧现象再次出现：更高的 SINR 和 rate，更高的切换，被放弃的 progress 更多，完成块不变。假设 rate 积分并没有因为更多切换而下降。先修环境记账，再判断连续性是不是比特收益上的瓶颈。

## Q12. 是否有充分依据进入 continuity-aware high-level objective？

NO。公式已经写成设计稿：`U(A_t) = V_exec(A_t) - lambda_switch * C_switch - lambda_progress * C_abandon`。`C_switch` 和 `C_abandon` 都能在当前时刻从 assignment 和 progress 算出，不用未来信息，也不是 J15 标签。`V_exec` 已经看过一步执行后的 progress 特征，`C_abandon` 可能把 progress 计两次。归一化是除以 UAV 数。lambda 仍是 HUMAN_OR_NEXT_PHASE_SELECTION_REQUIRED。没有把距离、轨迹或 waypoint 加回高层。

## Q13. 是否需要修改低层 MAPPO？

NO。低层执行给定 TaskOption。完成门槛和 progress 清零发生在环境与高层换目标。

## Q14. 是否需要增加中间评价层？

NO。

## Q15. 是否需要继续增加 HGNN 复杂度？

NO。HGNN 仍然只产生 Stage-1 候选。增加深度不能把 block 计数变成比特吞吐。

## 设计边界

下一阶段如果继续，应先定义环境里的比特队列，使 E1 和队列扣减成为同一次状态转移。在那之前不训练、不改 hold、不改 TaskOption、不改 MAPPO、不加中间层。320100–320131 保持未使用。
