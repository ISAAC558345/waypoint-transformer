# HTW Phase-8.2 SINR Relational Validation Report

独立 **Level-3-B**：M=6，K=12，hold=15。在锁定的 L3-A 竞争机制上 **只加 SINR 耦合与动态关系图**。
未训练、未改 HGNN、未改低层 PPO、未加 cooperative service、未开 Round、未改 `src/wt/l3a/`。

运动：飞向当前 option。低层 PPO 未加载、未更新。

## 1. 环境定义

代码：`src/wt/l3b/`。

- 规模：M=6，K=12，T=48，hold=15。
- 近簇 / 远簇各 6 个 buoy，簇内间距约 2–6 m，簇间约 100 m。
- 仍保留 UAV–UAV 竞争：`sum_m x_mk > 1` 则冲突机本步不发射、不服务。
- **未加入** 协同服务、能量。

速率：

`R_mk = B log2(1 + SINR_mk)`，
`SINR_mk = G_mk P / (σ² + Σ_{j≠m} G_jk P κ_{ℓk})`。

B=1.0e+06 Hz，P=1.0，σ²=0.001，α=2.2，
同簇 κ=1.0，跨簇 κ=0.08。

服务进度 `Δρ = min(1, (R / R_ref) / D)`，`R_ref` 为 `min_link` 处的孤立链路速率。同簇密铺会显著压低 SINR。
Oracle 允许 **idle**（不发射），用来减干扰。

## 2. 动态关系图

每步重算，写入 `info()`：

| 边 / 特征 | 含义 |
| --- | --- |
| `distance_uav_buoy` | UAV–Buoy 距离 |
| `distance_uav_uav` | UAV–UAV 距离 |
| `rate_bps` | 当前并发 SINR 下的速率 |
| `rate_if_assigned` | 固定其他链路时，自己改选 k 的假设速率 |
| `interference_uav_uav` | 当前链路上 j 对 i 接收 buoy 的干扰功率 |
| `queue_pressure` | Q_k / Q_max |
| `competition_pairs` | 同 buoy 冲突边 |

观测不含未来到达；`remaining_arrivals()` 只给 Oracle 影子。

## 3. 为何 Phase-8.1 不够

L3-A：Myopic 2.625e6（conflict 0.913），Exclusive myopic 4.338e7，Oracle 4.763e7。
排他规则已接近 Oracle，问题仍可被排序 matching 解掉。
L3-B 让「合法排他」仍可能很差：同簇塞满会互相压速率。

## 4. 非学习结果

1. Myopic：独立 Q/距离，允许碰撞。
2. Exclusive matching：顺序排他 Q/距离，**忽略 SINR 与 idle**。
3. Oracle：t=0 影子搜索 `(n_near, n_far)`，其余 idle；簇内再排他。

| 方法 | mean payload | conflict | mean rate (bps) | dropped | active frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| Myopic | 1500000 | 0.935 | — | 78.0 | — |
| Exclusive matching | 2250000 | 0.000 | 5.357e+05 | 77.0 | 1.000 |
| Oracle option | 5375000 | 0.000 | 1.127e+06 | 77.0 | 0.542 |
| Oracle − Exclusive | 3125000 | — | — | — | — |

**Oracle 明显高于 Exclusive matching：成立**
（相对差 138.89%；判据：绝对差 > 5e6 或相对差 > 20%）

- Oracle 平均选用 n_near=2.25，n_far=1.00（其余 idle）。
- Myopic 与 Oracle option 一致率 = 0.262。

种子 0–7。未训练任何神经网络。

## 5. 成功标准与下一步

- 关系推理空间（相对简单规则）：已增强。
- 本阶段不训练 MLP / DeepSets / HGNN。
- 若成立，下一阶段才在 L3-B 上比较三个高层编码器；策略应输出可冲突 z，环境用冲突全灭 + SINR，不要再做「更近者留下」。
- 仍未加入 cooperative service。

数值：`src/wt/l3b/outputs/causal/summary.json`
