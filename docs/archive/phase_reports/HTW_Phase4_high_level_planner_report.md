# HTW Phase-4 High-level Planner Report

同一低层 Goal-conditioned PPO（原始 `r_comm`，无 tracking）。只替换 Oracle TW 来源。未改 HGNN / PPO 核心 / 环境 / reward；未训练 Transformer；未开 Round。

## Planner 设计

- TW 槽位不变：`(goal, waypoint, horizon)`。
- A：independent queue-rate（当前）。
- B：`joint_plan` H=1，一步最大化干扰下 payload（queue pressure × 共享速率）。
- C：每步 `joint_plan` H=5，窗口累计预测 payload。

Waypoint 差异（归一化 L2）：{"mean_waypoint_l2_A_vs_B": 0.5053357407450676, "mean_waypoint_l2_A_vs_C": 0.5061453059315681, "mean_waypoint_l2_B_vs_C": 0.06244190502911806}

## 结果

| 方法 | payload | return | residual queue | waypoint dist | action–W cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| State-only PPO | 34297561.122 | 3.633889 | 2938352.643 | nan | nan |
| TW_A conditioned | 34310913.765 | 3.659546 | 2925000.000 | 0.340215 | -0.044833898458493504 |
| TW_B conditioned | 34279312.413 | 3.646518 | 2956601.351 | 0.436654 | -0.4749695433932843 |
| TW_C conditioned | 34310913.765 | 3.658072 | 2925000.000 | 0.459143 | -0.8819475626658132 |

## 四个问题

1. 当前 planner 为什么无收益？TW_A 是当前 queue/rate 的确定性滚动，信息已在 state 中，且不优化累计通信。详见 `planner_audit_report.md`。
2. 什么类型 TW 可以提供额外信息？相对 A 的 waypoint L2 更大、且 payload 更高的来源才算有新信息。本次最佳非 state 来源：**tw_a**（payload=34310913.765）。
3. 是否值得继续 HTW-C？ **值得**：已有 TW 源超过 State-only，层次结构有通信价值。
4. HGNN 未来应该放在哪个位置？若继续 HTW-C：**只放在 high-level planner**（对多 UAV/多 buoy 关系建模以生成更好的 `(G,W,H)`），不要放进低层 PPO，也不要替代环境 matching。当前 **尚未证明** 该位置的 HGNN 有必要。

**相对 State-only 是否有 TW 增益：是**

JSON：`E:/srdp/NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference/waypoint_transformer_official/src/wt/htw_phase4/outputs/planner_compare/summary.json`
