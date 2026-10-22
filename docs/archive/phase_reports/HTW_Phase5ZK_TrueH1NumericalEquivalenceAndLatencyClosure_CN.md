# Phase-5ZK：真实 H1 数值等价与耗时复核

Gate = ZK-A。

本阶段没有改 Phase-5ZJ 的 Gate 文件，也没有改它的报告。下面是对那次 FAST-A 的复核。

planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False

## 数值例外

Phase-5ZJ 提到的相对误差 9.35e-5 发生在未交付的批量 GEMM 实验，不在已交付路径上。

- 路径：batch_gemm_not_shipped
- 状态：310110 t=30
- assignment：-1;3;4;-1;-1;5
- UAV：5
- 字段：delta_distance_to_target（m）
- 原始值：-0.00010984696217830603
- 批量前向值：-0.00010985722903633643
- 绝对误差：1.0266858030405501e-08
- 相对误差：9.346510660659062e-05
- 组合容差上限：1.001098469621783e-06
- 是否超过组合容差：False
- 参考值是否近零：True

已交付路径的 response 失败元素数是 0。批量 GEMM 超过组合容差的元素只有 1 个，而且不在上面这个最大相对误差上。

同一个候选、同一架 UAV 的 `heading_alignment_1` 才超过组合容差：原始值 0.015074，批量前向值 0.015075，绝对误差 `1.272e-6`，组合上限 `1.151e-6`。它不是近零分母。绝对误差只比上限大出约 `1.2e-7`，来自同一处 `2.67e-9` 的动作扰动。交付路径没有这个元素。

根因是 float32 批量矩阵乘的归约顺序，不是反事实速率表的删除。用原始动作走删除该表后的环境一步，response 与串行结果逐位相等。反事实速率表只在 `info()` 里计算，不写环境状态，也不消耗随机数。

没有修改 adapter。交付路径本来就使用原始逐候选 `executor.act`。

## 耗时口径

端到端包含冻结 encoder、候选准备、H1、一个 Model-R1 前向和 argmax。每个状态 1 次 warmup、5 次重复。中位数是 24 个状态均值的中位数。95% CI 是这些状态配对绝对改善（串行减优化）的 scenario 聚类均值区间，不是中位数区间，也不是相对百分比。

- 串行中位数 956.11 ms
- 交付路径中位数 357.55 ms
- 中位数相对下降 0.6260
- 配对绝对改善的均值 897.37 ms
- scenario-cluster 95% CI [847.80, 932.79] ms

Phase-5ZI 的约 515 ms 是单个状态、把反事实速率表算进去、只含一个 value 头的分量之和。Phase-5ZJ 的 711.43 ms 和 216.47 ms 没有把 encoder 算进端到端，而且 value 头测了五个。本次 CI 若与 `[814.07, 862.10]` 不同，是因为计时边界不同，不是把均值和中位数当成同一个数。

## 问题

Q1. 9.35e-5 量级的最大相对误差落在 `delta_distance_to_target`，assignment `-1;3;4;-1;-1;5`，UAV 5。

Q2. 这个 9.35e-5 没有超过组合容差。绝对误差只有 `1.027e-8` m，上限是 `1.001e-6` m。参考距离变化约 `1.1e-4` m，相对误差被近零分母放大。真正超过组合容差的是同一 UAV 的 `heading_alignment_1`，而且只出现在未交付的批量 GEMM 上。

Q3. 根因已经定位。它不在交付路径里，所以没有再改 adapter。交付路径不需要这个修复。

Q4. 交付路径的环境状态、SINR、progress 和 payload：逐位一致。组合容差失败数 0。

Q5. assignment 和 selected J15：全部一致。

Q6. 按本次统一边界，中位数相对下降 0.6260。加速可以复现。

Q7. 515 ms、711.43 ms 和 216.47 ms 的边界不同。515 ms 含反事实速率表且只覆盖一个状态；711.43 / 216.47 ms 是 24 个状态的 H1 中位数，不含 encoder，含五个 value 头。本次统一边界的两个中位数写在上面。

Q8. 交付路径没有 neural batching。环境复用和删除未使用的反事实速率表是已实现部分。Actor observation batching、MAPPO batching、安全投影 batching、环境一步向量化都是 NOT_IMPLEMENTED。

Q9. 没有正式在线时延要求。deployment_latency_requirement = NOT_SPECIFIED。仿真步长是 1 秒，高层保持 15 步。这些不是墙上时钟预算。

Q10. 可以把它作为后续独立集成审计的候选实现，但不能接进正式 planner，也不是 deployment PASS。

planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False
