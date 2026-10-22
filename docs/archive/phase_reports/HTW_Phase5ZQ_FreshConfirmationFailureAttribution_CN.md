# HTW Phase-5ZQ：Fresh Confirmation 失败归因

归因 Gate 是 ATTR-C（ENDPOINT_SATURATION_CONFIRMED）。

Phase-5ZP 仍然是 CONF-C。`independent_algorithm_confirmation = False`。310118–310125 仍然是 FRESH_CONFIRMATION_CONSUMED。这次轨迹是同一冻结控制器的复现，不是新的 fresh confirmation。`planner_ready = False`，`deployment_verified = False`。

## Q1. payload 在代码里如何计算？

队列单位是服务块。每步先按目标切换把 progress 清零，再只对在服务半径内、队列为正、且没有冲突的链路计算 SINR 和 rate。progress 增加 `(rate / reference_rate) / service_slots`，并且单步增幅不超过 1。只有 progress 达到 1 时，payload 增加一个 block 的比特数，progress 再回到 0。丢块发生在 `queue - service + arrivals` 超过队列容量 4 的时候。残余队列是裁剪后的队列块数之和。

## Q2. 3.0e6 是不是 3 个完整 block？

是。`block_payload_bits = 1000000`。复现得到的完成块数取值是 [3.0]。

## Q3. partial service 是否进入 payload？

不进入。环境里没有按速率累计的连续比特字段。progress 不到 1，payload 就不变。

## Q4. 为什么 8 个 scenario、两个系统都恰好完成 3 个 block？

代码没有把完成数固定成 3。每个 UAV 每步最多完成 1 个 block，48 步、6 架 UAV 的上界是 288。48 条复现轨迹的初始块都是 15，到达都是 102，完成都是 3，丢弃都是 78，剩余都是 36，所以 `15 + 102 = 3 + 78 + 36` 对每一条轨迹成立。Primary 的 24 次完成全部落在 buoy 3、4、5，各 8 次；五个 Model-S seed 的 120 次完成也全部落在这三只 buoy。它们是近簇里初始队列为 1 的三只 buoy，完成一次就被清空。初始队列为 4 的 buoy 0、1、2 一次都没有完成。远簇 buoy 6–11 从第 4 步起每两步到达 1 块，容量是 4，每只最多留下 4 块，六只一共留下 24 块，其余 `6 × 13 = 78` 块被丢掉。残余 36 正好是这 24 块加上三只热 buoy 上没被服务的 12 块。两个控制器都没有把第二块服务送过完成门槛。

## Q5. Primary 的 SINR 为什么更高？

按 Phase-5ZP 同样的全 UAV 平均，Primary SINR 是 5.37491，Model-S 是 1.86964。只看正在服务的 UAV，分别是 36.177 和 9.93556。正在服务的步数分别是 48 和 50.675。

## Q6. 高 SINR 是否变成更高 rate？

正在服务的样本上，SINR 对 rate 的 Spearman 是 Primary 1，Model-S 1。平均服务 rate 是 Primary 2.44435e+06，Model-S 2.02109e+06。

## Q7. 更高 rate 是否变成更多 progress？

rate 对单步 progress 增量的 Spearman 是 Primary 0.961706，Model-S 0.912712。累计 service work 是 Primary 3.67361，Model-S 3.09654。被切换丢掉的 progress 是 Primary 0.673611，Model-S 0.0965433。

## Q8. progress 为什么没有变成更多 completed blocks？

断点在 COMPLETION。SINR 已经变成更高的 rate，rate 也变成了更多 progress：Primary 的累计 service work 是 3.67361，Model-S 是 3.09654。多出来的大约 0.67 被目标切换清掉了，Primary 丢掉的 progress 是 0.673611，Model-S 是 0.0965433。清掉之后两边都只剩下刚好 3 个完整 block。payload 不记录这 0.67。

## Q9. Primary 是否切换更多、连续服务更短？

平均目标切换次数：Primary 7.625，Model-S 3.825。服务段长度中位数：Primary 15.9375，Model-S 31.4625。完成前被打断的次数：Primary 1.25，Model-S 0.1。

## Q10. hold=15 是否经常在 block 完成前结束？

Primary 的 option 窗口里，结束时仍有未完成 progress 的比例是 0。Model-S 是 0.03125。一次完成前的服务步数中位数是 Primary 7，Model-S 7.5。这里只描述轨迹，不修改 hold。

## Q11. drop 是不是主要瓶颈？

8 个 scenario 的到达都是 102 块，Primary 与 Model-S 的丢块都是 78，残余队列都是 36。更高的 SINR 没有减少任何丢块。

## Q12. 结束时 Primary 是否留下更多接近完成的 progress？

没有。结束时 Primary 的 48 个 UAV progress 全部落在 [0, 0.25)，Model-S 的 240 个也是。接近 1 的比例两边都是 0。多出来的 progress 在中途切换时已经被清零，不是留到 episode 结束才差一点点。

## Q13. 主指标是否被 block 量化压平？

最终完成块数的取值集合是 [3.0]。`COMPLETION_METRIC_SATURATION = True`。SINR、rate、service work、完成时刻和结束 progress 并不都相同，所以最终 payload 把这些差别压成了同一个 3 block。这不能把 CONF-C 改成通过。

## Q14. 失败属于哪一层？

ATTR-C。因果链在 PHY 和 RATE 上是通的，多出来的 progress 没有跨过第 4 个 block，断点记在 COMPLETION。最终 block 计数全部是 3，所以主 endpoint 没有分辨率。这仍然是 CONF-C，不是算法通过。

## Q15. 是否发现实现 bug？

守恒误差最大值是 0。与 Phase-5ZP 记录的 payload 和 SINR 不一致的 scenario 数是 0。结论是 没有发现 payload 记账错误。

## Q16. Phase-5ZP 的 CONF-C 是否仍然有效？

YES。没有发现能够推翻原确认记账的实现错误。归因结果不改写 CONF-C，也不宣布 deployment PASS。

## 事后机制

逐 scenario 的主导标签计数是 {'MIXED': 8}
