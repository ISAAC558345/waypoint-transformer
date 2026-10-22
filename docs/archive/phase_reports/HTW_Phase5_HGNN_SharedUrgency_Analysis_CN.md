# HTW Phase-5 HGNN + Shared Urgency

**phase_status = INCONCLUSIVE_BOUNDED_BUDGET**

**deployment_verified = False**

**planner_ready = False**

本阶段按隔离种子执行了四组消融。没有修改 reward，没有修改 `high_level.decide()`，没有使用 230000–230127，没有中间评价层，没有 HAC，也没有高低层联合更新。

## 实现

- 高层：HGNN 只替换 encoder。UAV–UAV 与 UAV–Buoy 做消息传递。解码器仍是 `OrderFreeSetDecoder` 的 joint greedy。
- 共享紧急程度：`u_k(t) = q_k(t) / (max_j q_j(t) + 1e-8)`，范围 [0, 1]。只读当前队列。一个 option 的 15 步内不重算。
- 低层：urgency-aware Residual MAPPO 在原观测后追加该无人机所分配浮标的 urgency。动作仍是二维位移残差。
- 环境没有能耗状态。表中的路径长度是位移积分，不是新的能耗模型。

训练种子 [310000, 310001, 310002, 310003, 310004, 310005, 310006, 310007]，评估种子 [310100, 310101]。高层 6 个 epoch，低层 8 次 MAPPO 更新。这是有界探针，不是收敛后的部署模型。

## 消融

| 实验 | payload | residual queue | completion | path length | trajectory reward | assignment match | conflict | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline DeepSets+MAPPO | 11000000.0000 | 30.0000 | 48.0000 | 474.7861 | 11.0000 | 0.1875 | 0.0000 | 2.4089 |
| HGNN+MAPPO | 7000000.0000 | 32.0000 | 48.0000 | 835.5855 | 7.0000 | 0.7708 | 0.0000 | 1.4090 |
| DeepSets+UrgencyMAPPO | 9500000.0000 | 31.5000 | 48.0000 | 563.2526 | 9.5000 | 0.1042 | 0.0000 | 2.1473 |
| HGNN+UrgencyMAPPO | 6500000.0000 | 32.5000 | 48.0000 | 835.4375 | 6.5000 | 0.7500 | 0.0000 | 1.8718 |

## 高低层一致性

Q_H 是 order-free decoder 在其他无人机已分配条件下对候选浮标的 logit。U_LL 是把该候选锁定 15 步后，当前低层策略的累计 reward。

| 实验 | Spearman | Kendall | Top-1 | normalized regret | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline DeepSets+MAPPO | 0.3235 | 0.2597 | 0.2500 | 0.3194 | 12 |
| HGNN+MAPPO | -0.0286 | -0.0849 | 0.1667 | 0.7500 | 12 |
| DeepSets+UrgencyMAPPO | 0.3003 | 0.1901 | 0.1667 | 0.3561 | 12 |
| HGNN+UrgencyMAPPO | -0.0125 | -0.1151 | 0.2500 | 0.6667 | 12 |

## Urgency 是否被使用

- Baseline DeepSets+MAPPO: low-level |Δμ|=NA (NOT_APPLICABLE); high-level zero-urgency changes assignment=False (NOT_IN_ENCODER); assigned-minus-rest urgency=NA.
- HGNN+MAPPO: low-level |Δμ|=NA (NOT_APPLICABLE); high-level zero-urgency changes assignment=False (NOT_IN_ENCODER); assigned-minus-rest urgency=NA.
- DeepSets+UrgencyMAPPO: low-level |Δμ|=0.0006 (USED); high-level zero-urgency changes assignment=False (NOT_IN_ENCODER); assigned-minus-rest urgency=0.0424.
- HGNN+UrgencyMAPPO: low-level |Δμ|=0.0008 (USED); high-level zero-urgency changes assignment=False (URGENCY_IGNORED_BY_HIGH_LEVEL); assigned-minus-rest urgency=-0.2292.

## 判断

完整方案的 payload 是 6.5e6，低于基线的 1.1e7。完整方案的 Spearman 约为 -0.01，低于基线的 0.32；Top-1 都是 0.25。把 HGNN 的 urgency 通道置零后，分配没有变化，低层把 urgency 从 0 改到 1 时动作均值只变化约 8e-4。

因此这组有界探针没有出现“HGNN + urgency 明显改善系统，于是不必做联合优化”，也没有出现“这两项已经有效、但排序仍然不一致”。不能据此宣布目标不一致已经被解决，也不能据此进入 HAC 或联合训练。

下一阶段没有启动。`planner_ready` 保持 false。
