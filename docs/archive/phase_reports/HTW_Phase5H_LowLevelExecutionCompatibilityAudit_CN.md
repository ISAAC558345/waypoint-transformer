# HTW Phase-5H Low-Level Execution Compatibility Audit

**planner_ready = False**

**deployment_verified = False**

本阶段只比较冻结目标和冻结执行器。没有重训 HGNN，没有重训 Residual MAPPO，没有修改 B2、τ、Gate-1、Candidate-C、planner 或 deployment `decide()`，没有使用 230000–230127。

## 1. 问题定位

Phase-5G 的单目标 MAPPO payload 是 1.57e6 bit，联合分配后再执行降到 3.33e5 bit。这里把同一状态拆成三类：

- A：联合目标本身的 fly-toward payload 就低，问题在高层组合。
- B：同一条单目标分配上，fly-toward 明显高于 MAPPO，问题在低层跟踪。
- C：单目标能执行，联合分配的服务冲突、干扰或同时服务把收益打掉。

主判定是 `A_joint_target_selection`。单目标 execution loss = 111111.1111 bit。utility 联合分配的 execution loss = 0.0000 bit。utility 联合 MAPPO payload = 333333.3333 bit，单目标 MAPPO payload = 1277777.7778 bit。

## 2. High-level ranking

场景仍是 310102 和 310103。oracle assignment 是同一状态下按 fly-toward payload 做排他贪心，不读取未来到达，也不是 Phase-5B 的簇 oracle。

单目标替换沿用 Phase-5G：其他无人机保持当前排他分配，只换一名无人机的 utility HGNN 目标。

这条单目标的 fly-toward payload = 694444.4444，MAPPO payload = 1277777.7778。这是全部 36 个无人机决策，包含 payload 没有拉开的时刻。Phase-5G 的 1.57e6 只统计已经分开的组。两者的 Spearman = 0.4978，Kendall = 0.2587。

全部候选上，fly-toward utility 和 MAPPO payload 的 Spearman = 0.5762，Kendall = 0.3667，n = 252。U_exec 用的 execution_loss 就是这次冻结 MAPPO 相对 fly-toward 的落差，不是新训练出来的量。

## 3. Low-level execution gap

同一条 utility HGNN 单目标分配，oracle trajectory 是 fly-toward，对照是冻结 `mappo_original.pt`。

execution_loss = max(0, fly payload - MAPPO payload)，平均 111111.1111 bit。U_exec = expected_payload - execution_loss，平均 583333.3333 bit。

联合分配上的同一对执行器：

| assignment | fly payload | MAPPO payload | execution loss | conflict rate | interference | min UAV distance |
| --- | --- | --- | --- | --- | --- | --- |
| utility HGNN | 0.0000 | 333333.3333 | 0.0000 | 0.0000 | 0.3919 | 1.0728 |
| DeepSets | 666666.6667 | 1500000.0000 | 0.0000 | 0.0000 | 0.6679 | 1.1138 |
| fly-toward utility oracle | 333333.3333 | 1000000.0000 | 0.0000 | 0.0000 | 1.9472 | 1.3932 |

## 4. Multi-UAV conflict

环境没有单独的碰撞事件。这里记录它已经给出的服务冲突 `conflict_rate`、`n_conflict_pairs`，以及 UAV–UAV 干扰矩阵的和。最近无人机距离只作为几何量。

排他贪心的重复目标数是 0.0000。utility 联合分配在 MAPPO 执行中的平均冲突率是 0.0000，平均干扰是 0.3919。

单架服务是只派出 utility HGNN 选中的那一架、其余悬停。它的 MAPPO payload 平均 694444.4444 bit，干扰 0.0000。联合六架的 MAPPO payload 是 333333.3333 bit，干扰 0.3919。

单目标替换已经是多机环境，只是其他无人机仍用排他匹配。它的 MAPPO payload 1277777.7778 高于六架都改成 utility 目标之后的 333333.3333。

## 5. Reward

同一条单目标分配上，MAPPO payload 不低于 fly-toward。联合分配上 MAPPO 也不低于 fly-toward。收益下降不是当前 reward 没让低层跟踪目标。本阶段不修改 reward。

本阶段没有修改 reward。

## 6. Joint optimization

utility 联合分配的 fly-toward payload 低于 DeepSets，服务冲突率是 0。六架目标同时替换之后，目标组合本身的通信收益就低。本阶段不开始高低层联合训练。

本阶段不进入 joint optimization，也不进入 deployment verification。

`planner_ready = False`

`deployment_verified = False`
