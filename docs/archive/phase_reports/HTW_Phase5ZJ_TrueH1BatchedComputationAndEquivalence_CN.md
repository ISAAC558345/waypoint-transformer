# Phase-5ZJ：真实 H1 的批量计算与等价性

Gate = FAST-A。

交付的是计算路径，不是新的策略。候选 assignment、冻结 MAPPO、安全投影、环境动力学和 Model-R1 都没有改。正式 planner 没有接入。

```text
planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False
```

## 交付了什么

串行真实 H1 对 120 个候选各做一次深拷贝，并各调用两次完整 `info()`。其中 `rate_if_assigned` 是 6×12 的反事实速率表，H1 response 的 14 个字段都不读它。

交付路径改为：

- 仍调用原始 `LockedExecutor.act`。
- 只深拷贝一次，每个候选执行前把位置、队列、进度、服务目标和 SINR 写回去。
- 实验副本的 `info()` 仍计算 `interference_matrix`，不再重建 `rate_if_assigned`。
- 原始 `step` 里的联合 SINR、服务进度和 payload 逻辑没有改。

批量 GEMM 做过，没有交付。冻结 actor 只输出 `mu`，不抽样，调用也不消耗环境随机数，所以按候选维做一次前向在随机数意义上是安全的。安全投影仍在每个 assignment 内部串行。批量矩阵乘改变了浮点归约顺序：动作绝对误差最大约 `7.2e-9`，但 `310110 t=30` 的 response 最大相对误差是 `9.35e-5`，超过预先锁定的 `rtol=1e-5`。该状态的离散字段和最终 assignment 没有变，可是它不满足锁定容差，所以这条路径停用。

## 等价性

对照是 `310110–310117` 在 `t=0、15、30` 上已经消费的 120 个候选。saved Model-R1 的最大绝对误差是 0。交付路径相对串行路径：

- 动作、期望动作、位置、队列、进度、SINR、速率、干扰和全部 response 字段逐位相等。
- payload、served target、assignment、完成相关的二值 response 字段完全一致。
- 五个种子的 Model-R1 value 逐位相等，Top-1、Top-3、Top-5 和 argmax 全部一致。
- selected J15 完全保留。

`310114 t=15` 上，`A_opt = -1;4;2;5;-1;-1` 与 `A_pred = 2;5;-1;4;-1;-1` 的 response、value 和排名都与串行实现一致。五个种子上这两条 assignment 的 value 绝对误差都是 0。

环境没有单独的速度状态。记录并对照的是执行后的位置差，它与串行结果逐位相等。

## 耗时

计时在同一台机器上完成。每个状态先做 1 次 warmup，再做 3 次重复。先对重复取平均，再把 scenario 当作 cluster。重复次数没有当成新的 scenario。计时窗口里没有打开内存跟踪。

状态平均后的端到端耗时：

| 路径 | 中位数 | 最小 | 最大 |
| --- | ---: | ---: | ---: |
| 串行 H1 | 711.43 ms | 508.16 ms | 2178.89 ms |
| 交付路径 | 216.47 ms | 161.55 ms | 441.39 ms |

中位数下降 69.6%。配对改善的 scenario-cluster 95% CI 是 `[814.07, 862.10]` ms，下界大于 0。

重复测量的平均分量：

| 分量 | 串行 | 交付路径 |
| --- | ---: | ---: |
| 状态拷贝 | 23.50 ms | 0.24 ms |
| 原始 `executor.act` | 194.95 ms | 153.80 ms |
| 环境一步 | 397.46 ms | 37.93 ms |
| response 组装 | 426.63 ms | 16.43 ms |
| Model-R1 | 17.46 ms | 17.35 ms |

下降最多的计时桶是 response 组装。串行版本在这里为每个候选调用一次完整 `info()`，主要时间花在反事实速率表。环境一步是第二大来源，因为 `step` 结束时又调用了一次同样的 `info()`。这两处被去掉之后，联合 SINR 仍在 `step` 内部按原公式计算。

Phase-5ZI 报告的 515.26 ms 是单个状态、单个 value 头的测量。本阶段的 711.43 ms 是 24 个决策时刻的中位数，并且包含五个 Model-R1 前向。`t=30` 的串行耗时更高，因为当时的活动链路更多，反事实速率表更贵。

一次内存跟踪里，串行峰值约 337 KB，交付路径约 471 KB。这是 Python 分配跟踪，不是进程驻留集。

## 问题

Q1. 120 个候选的网络前向可以按候选维一次计算，而且不会消耗随机数。它没有交付，因为 `310110 t=30` 的 response 相对误差超过 `1e-5`。

Q2. Safety projector 不阻止网络批量化。它耦合的是同一个 assignment 里的 UAV，所以保持逐候选执行。

Q3. 环境一步里可以省掉的是每个候选重复建造的反事实速率表。联合 SINR、干扰和服务进度仍按候选单独计算。

Q4. assignment 内部的 UAV–UAV 干扰和联合 SINR 被完整保留，没有拆成独立 UAV SINR。

Q5. 交付路径没有改变低层动作。动作与串行 `executor.act` 逐位相等。

Q6. 交付路径没有改变 H1 response 和 Model-R1 排序。value 绝对误差是 0。

Q7. 优化后的完整推理耗时中位数是 216.47 ms。串行中位数是 711.43 ms。

Q8. 最大的加速来自不再重复建造反事实速率表。按计时桶，response 组装从 426.63 ms 降到 16.43 ms，环境一步从 397.46 ms 降到 37.93 ms。

Q9. R1 的 selected J15 完全保留。

Q10. 515 ms 没有违反项目里已经写明的在线时延上限。`L3BConfig` 给出仿真步长 `delta_t = 1 s`、高层保持 15 步、回合 48 步，没有推理延迟上限。`deployment_latency_requirement = NOT_SPECIFIED`。本机也没有被指定为部署硬件。

Q11. 不可以。本阶段没有把优化后的 H1 接进正式 planner，也没有做 deployment verification。下一阶段如果要讨论实时部署，需要单独的指令；当前 `planner_ready` 仍是 False。

## 边界

没有训练，没有新 checkpoint，没有新 scenario，没有新 J15。`310118–310125` 没有打开。历史审计文件和已保存 checkpoint 没有改。正式 `decide()` 没有调用。
