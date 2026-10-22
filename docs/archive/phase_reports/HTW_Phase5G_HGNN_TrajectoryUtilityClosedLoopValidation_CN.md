# HTW Phase-5G HGNN Trajectory Utility Closed-loop Validation

**planner_ready = False**

**deployment_verified = False**

本阶段检查冻结的 utility HGNN 在新场景上，交给冻结 Residual MAPPO 执行 15 步之后还有没有 payload 收益。没有训练，没有调用 deployment `decide()`，没有修改 B2、τ、Gate-1、Candidate-C、低层结构或 planner，没有使用 230000–230127。

## 1. 验证数据

场景是已经存在的隔离重置 310102 和 310103。它们没有进入 Phase-5F 的训练。310100 是训练场景，本阶段不使用。没有新的 scenario 生成器。

每个决策时刻是 t=0、15、30。时刻之间用当前排他匹配和 fly-toward 走到决策点。决策之后的 15 步才换成冻结 MAPPO。offline utility 仍是同一分配下 fly-toward 的 15 步 payload，和 Phase-5E 的 baseline 一致。

## 2. Scenario isolation

训练场景 310100 与本阶段场景 [310102, 310103] 不相交。检查点只读取，不更新。

## 3. HGNN 排序

下表只统计 fly-toward payload 已经分开的组。Spearman 和 Kendall 比较预测分数与 utility 名次。Top-1、Top-3 看预测最高分是否覆盖 offline utility 最优候选。

| model | scenario | n | candidates | Spearman | Kendall | Top-1 | Top-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| assignment HGNN | 310102 | 12 | 7.0000 | -0.5030 | -0.2698 | 0.0000 | 0.9167 |
| assignment HGNN | 310103 | 11 | 7.0000 | -0.3344 | -0.1039 | 0.0000 | 0.9091 |
| assignment HGNN | both | 23 | 7.0000 | -0.4224 | -0.1905 | 0.0000 | 0.9130 |
| DeepSets | 310102 | 12 | 7.0000 | 0.2887 | 0.1032 | 0.6667 | 0.9167 |
| DeepSets | 310103 | 11 | 7.0000 | 0.1883 | -0.0260 | 0.3636 | 0.5455 |
| DeepSets | both | 23 | 7.0000 | 0.2407 | 0.0414 | 0.5217 | 0.7391 |
| utility HGNN | 310102 | 12 | 7.0000 | 0.3839 | 0.2778 | 0.9167 | 0.9167 |
| utility HGNN | 310103 | 11 | 7.0000 | 0.2500 | 0.0952 | 0.5455 | 0.5455 |
| utility HGNN | both | 23 | 7.0000 | 0.3199 | 0.1905 | 0.7391 | 0.7391 |

## 4. Payload

候选级 payload 是只替换一名无人机的目标、其他无人机保持当前排他分配后，冻结 MAPPO 走出的 15 步通信增量。联合 payload 是按分数从高到低贪心占用浮标后，再交给同一个 MAPPO 执行的 15 步增量。候选级收益没有带到联合分配上：utility HGNN 的联合 payload 与 assignment HGNN 相同，并且低于 DeepSets。

| model | MAPPO payload of chosen target | MAPPO best in the group | joint MAPPO payload |
| --- | --- | --- | --- |
| assignment HGNN | 478260.8696 | 2260869.5652 | 333333.3333 |
| DeepSets | 1260869.5652 | 2260869.5652 | 1500000.0000 |
| utility HGNN | 1565217.3913 | 2260869.5652 | 333333.3333 |

## 5. Baseline

assignment HGNN 是 Phase-5 的 `hgnn_orderfree.pt`。DeepSets 是冻结的 utility-pool 检查点，不是 assignment NLL。utility HGNN 是 Phase-5F 的 `hgnn_utility_rank.pt`。三个模型的低层都是同一个 `mappo_original.pt`，urgency 没有送进低层。

## 6. Failure cases

文件：`phase5g_closed_loop/closed_loop_failures.csv`。只记录 utility HGNN。

Case A，预测选中的目标不是 offline utility 最优，而且 MAPPO payload 至少少一个服务块：6。

Case B，offline utility 最优目标的 MAPPO payload 至少比组内 MAPPO 最优少一个服务块：0。这是 fly-toward 标签和冻结 MAPPO 执行不一致。

Case C，最高的两个 offline utility 相差不到一个服务块：24。这些组需要并列处理，单点 argmax 不稳定。

## 7. 高低层一致性

这里的 Spearman、Kendall、Top-1、Top-3 比较的是预测分数和 MAPPO 实际 payload，只统计 MAPPO payload 已经分开的组。

| model | Spearman | Kendall | Top-1 | Top-3 |
| --- | --- | --- | --- | --- |
| assignment HGNN | -0.1282 | -0.0560 | 0.0588 | 0.8235 |
| DeepSets | 0.0630 | -0.0070 | 0.3235 | 0.6765 |
| utility HGNN | 0.2405 | 0.1218 | 0.3824 | 0.6471 |

## 8. Phase-5H

单目标替换上，utility HGNN 的 MAPPO payload 和 utility Top-1 高于 assignment HGNN。联合分配再交给 MAPPO 时，两者 payload 相同，并且低于 DeepSets。因此完整闭环没有稳定收益。

本阶段不进入 Phase-5H，也不做 deployment verification。planner 没有修改。系统优化没有完成。

`planner_ready = False`

`deployment_verified = False`
