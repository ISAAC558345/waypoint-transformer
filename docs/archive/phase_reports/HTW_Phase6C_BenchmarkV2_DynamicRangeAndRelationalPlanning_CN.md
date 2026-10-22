# HTW Phase-6C：Benchmark-V2 动态范围与关系规划价值

Gate 是 DYN-A（BIT_LEVEL_BENCHMARK_DISCRIMINATIVE）。

Phase-5ZP 仍是 CONF-C，Phase-5ZQ 仍是 ATTR-C，Phase-6A 仍是 BV2-E，Phase-6B 仍是 BITQ-A。环境是 `L3BBitQueueEnvV2`。320100–320131 没有打开。没有训练，没有把 legacy MAPPO 放进正式比较。`planner_ready = False`，`deployment_verified = False`。

## Q1. cumulative_payload_bits 是否有足够动态范围？

random、nearest、max-queue、max-rate、queue-rate 的 saturation ratio 都低于 0.50，pairwise tie fraction 是 0.0104167，可区分 scenario 比例是 1。joint true-bit 自己的 saturation ratio 是 1，因为每个 scenario 的闭环 payload 都相同；这是该策略对 ±2 m 初始抖动不敏感，不是不同控制器被打成同一个值。

## Q2. 旧的 3-block saturation 是否消失？

消失了。max-rate 的 payload 从 4.93215e+07 到 9.05837e+07 bit，nearest、queue-rate、random 也同样随 scenario 变化，不再锁在 3e6。joint true-bit 的闭环结果恒为 1.5e+07 bit，等于初始队列总量：它传完初始数据后不再服务后续到达。

## Q3. 各参考控制器的收益是多少？

random_legal 均值 7.20365e+07 bit；nearest 均值 7.37225e+07 bit；max_queue 均值 7.14709e+07 bit；max_rate_independent 均值 7.61185e+07 bit；queue_rate 均值 6.60851e+07 bit；joint_rate 均值 1.37188e+07 bit；joint_true_bit 均值 1.5e+07 bit

R7 是 R6 的闭环本身：所有参考控制器的低层都是 fly-toward，没有使用旧 MAPPO。因此 R7 与 joint_true_bit 是同一条轨迹，不重复计入动态范围。

## Q4. independent max-rate 与 joint true-bit 的差距是多少？

闭环 payload 差（joint − max-rate）的均值是 -6.11185e+07 bit，95% CI 是 [-6.24912e+07, -5.96466e+07]，正收益 scenario 比例是 0。

相对 queue-rate 的均值差是 -5.10851e+07 bit，95% CI 是 [-5.2719e+07, -4.94678e+07]，正收益比例是 0。

在相同的初始状态下，一拍 served bits：joint 7.78159e+06，max-rate 3.86398e+06，queue-rate 1.75238e+06。这是分配诊断，不是闭环结论。

## Q5. 多 UAV interference 是否改变最优 assignment？

把 UAV-UAV interference 关掉后，one-step true-bit assignment 的不一致比例是 0.546875。按真实干扰强度分层后的不一致比例是 low 0.171171，medium 0.783333，high 0.870588。

## Q6. queue 是否必须进入高层规划？

服务事件里 raw capacity 超过当前队列的比例，joint true-bit 是 0.676056，joint rate 是 0.978593。只最大化 rate、不看队列时，剪裁更多，说明最高 rate 不等于最高实际比特。queue 需要进入这一拍的目标。

## Q7. 距离、channel、interference、queue 对最优 assignment 的作用是什么？

distance 改变 assignment 的比例是 0.488281，真实 served bits 平均下降 1.18605e+06；queue 改变 assignment 的比例是 0.566406，真实 served bits 平均下降 641879；rate_if_assigned 改变 assignment 的比例是 1，真实 served bits 平均下降 1.51683e+06；uav_uav_interference 改变 assignment 的比例是 0.546875，真实 served bits 平均下降 2.10536e+06；current_association 改变 assignment 的比例是 0，真实 served bits 平均下降 0；uav_positions 改变 assignment 的比例是 0.65625，真实 served bits 平均下降 1.80555e+06

current association 不进入 one-step 比特目标。单测确认改写当前 assignment 不会改变这一拍的最优解。

## Q8. Stage-1 top-3 的 coverage@3 是多少？

instantaneous_rate coverage@1 = 0.760417；instantaneous_rate coverage@3 = 0.989583；instantaneous_rate coverage@5 = 1；queue_rate coverage@1 = 0.755208；queue_rate coverage@3 = 0.976562；queue_rate coverage@5 = 1；individual_served_bits coverage@1 = 0.348958；individual_served_bits coverage@3 = 0.963542；individual_served_bits coverage@5 = 1

## Q9. 是否存在 candidate coverage failure？

否。判定用三种排序里最好的 coverage@3，门槛是 0.90。这里的覆盖率是边覆盖：联合最优里每条 UAV-buoy 边是否落在该 UAV 的 top-k。这是 candidate-pool 诊断，不是 global oracle。

## Q10. 联合规划是否有稳定的 payload 增益？

闭环上没有。joint 比 max-rate 平均少 6.11185e+07 bit，64 个 scenario 全部为负，95% CI 整个位于 0 以下。冻结位置的一拍上，joint 的 served bits 更高，7.78159e+06 对 max-rate 的 3.86398e+06。fly-toward 闭环里，joint true-bit 约 92.2% 的步没有服务：当前队列很快传完，UAV 停在原地，没有去接后续到达。

## Q11. 是否已经有证据支持 HGNN？

没有。一拍上，干扰、队列、距离和位置都会改变最优分配；闭环上，这个一拍联合最优明显差于独立 max-rate。这不能支持现在训练 HGNN，也不等于 HGNN 会优于 DeepSets 或 MLP。

## Q12. 是否需要现在增加 HGNN 深度？

NO。

## Q13. 是否需要现在加入 continuity penalty？

NO。这只看本阶段的比特 payload，不引用 Phase-5 的 block progress。joint true-bit 的切换次数对 payload 的 Spearman 是 NA。没有达到“切换稳定损害比特收益”的证据，lambda 仍是 NOT_DEFINED。

## Q14. legacy MAPPO 是否可以用于正式 Benchmark-V2 comparison？

NO。低层状态记为 LOW_LEVEL_RETRAIN_PENDING。正式动态范围只来自 scripted reference。

## Q15. 320100–320131 是否仍然完全 untouched？

YES。fresh_confirmation_untouched = True。

## Q16. 最终 Gate 是什么？

DYN-A（BIT_LEVEL_BENCHMARK_DISCRIMINATIVE）。守恒最大绝对误差是 1.01281e-08 bit。没有构造短时域上界，所以不给 DYN-C。hold 仍是 15，没有调参。

