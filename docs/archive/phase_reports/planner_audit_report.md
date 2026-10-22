# Planner Audit Report

范围：Round-19 `src/wt/round19/planner.py`，以及 Phase-1/3 用来生成 TW 的 `independent_queue_rate`。未改环境、reward、PPO、HGNN。

## 1. 当前 planner 优化目标

`independent_queue_rate_action` **不优化**环境 reward，也 **不优化**折扣累计通信收益。

对每个 UAV 独立取

`k* = argmax_k (queue_k / capacity) * rate_{uav,k}`

其中 `rate` 来自 `rate_matrix`（单 UAV 静态大尺度信道，**不含**共享频谱干扰与联合 matching）。动作是指向 `k*` 的单位向量。

`joint_plan` 才对 **预测 payload 之和** 做有限 beam search，并在 shadow 里用 `joint_aware` matching。Phase-1/3 默认 TW **没有**用它。

## 2. 输入状态

仅当前可观测量：

- UAV 位置
- buoy 位置
- 当前队列 `raw_queue_bits`
- 由位置算出的当前速率/距离

不输入：其他 UAV 的未来意图、未来到达、累计 return、干扰后的 SINR（独立策略路径上）。

## 3. 输出 TW 的方式

滚动该启发式 H=5 步（shadow，不改 live env）：

- `goal` = 窗口内各 UAV 对 buoy 的累计 payload 份额
- `waypoint` = 第 H 步位置（归一化）
- `horizon` = 5

这是 **myopic 方向的积分终点**，不是最优控制子目标。

## 4. 为什么不能提供长期价值

1. 目标是瞬时 queue×正交速率，不是 `sum_t r_comm`。
2. 忽略干扰：两机可能冲向同一高队列 buoy，共享频谱下互相压速率。
3. H 步滚动仍每步 myopic，终点不必是累计 payload 最优位置。
4. 这些量（队列、位置、近似速率）**已经在低层 state 里**。TW_A 近似是 state 的确定性函数，所以 Phase-3 条件化没有新信息。

Phase-4 用同一 TW 槽位接入：

- TW_A：当前 queue-rate
- TW_B：`joint_plan(H=1)`，最大化一步干扰感知 payload
- TW_C：一次 `joint_plan` 长度 H=5 的 beam 轨迹终点（累计预测 payload），不是对 H 次独立 receding 搜索。
