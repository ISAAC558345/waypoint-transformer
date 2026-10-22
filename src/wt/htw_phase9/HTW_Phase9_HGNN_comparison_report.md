# HTW Phase-9 HGNN Comparison Report

在 **未修改的 Level-3-B** 上比较高层编码器。hold=15，reward=`delta`（窗口 Δpayload），SINR 与竞争物理不变。
未改 `src/wt/l3b/env.py`、未改低层 PPO 核心、未加载 L2-A M=5 的冻结 PPO（维度不兼容），低层执行器为与 8.2 相同的 **fly-toward**。
seed=0，80 updates，rollout=16，num_envs=4。策略输出可冲突的 z，环境冲突全灭。

能量：L3-B 不暴露 energy，UAV 能量通道恒为 0，未发明物理量。

## 1. 结构

- MLP：固定拼接 UAV/buoy。
- DeepSets：置换不变集合均值。
- HGNN：UAV–Buoy 边（distance / rate_if_assigned / ETA）+ UAV–UAV 边（distance / interference），两层 UB 消息 + 一层 UU 消息。

## 2. 主结果（默认 L3-B，eval seeds 8–15）

| 方法 | payload | completion | service | option accuracy | conflict | interference utility | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Myopic (8.2) | 1500000 | — | — | — | 0.935 | — | — |
| Exclusive matching (8.2) | 2250000 | — | — | — | 0.000 | — | — |
| Oracle (8.2) | 5375000 | — | — | — | 0.000 | — | — |
| MLP | 8250000 | 0.685 | 0.035 | 0.206 | 0.000 | 1.203e+06 | 8.707 |
| DeepSets | 0 | 1.000 | 0.000 | 0.417 | 0.000 | 0.000e+00 | 10.291 |
| HGNN | 0 | 1.000 | 0.000 | 0.417 | 0.000 | 0.000e+00 | 13.731 |

HGNN − DeepSets payload = 0。
HGNN − MLP payload = -8250000。

## 3. 泛化（同一权重，不改 L3-B 源码）

M/K 不能改 `L3BEnv` 的锁定规模，因此用 mask：`m5_mask` 强制第 6 机 idle；`k8_mask` 禁止选择 buoy≥8。
干扰强度通过 `L3BConfig.kappa_same_cluster` 传入，不改 env.py。

| 变体 | MLP payload | DeepSets payload | HGNN payload |
| --- | ---: | ---: | ---: |
| m5_mask | 7500000 | 0 | 0 |
| k8_mask | 6500000 | 0 | 0 |
| kappa_0.5 | 9500000 | 0 | 0 |
| kappa_1.5 | 7625000 | 0 | 0 |

## 4. 三个问题

1. **HGNN 是否超过 DeepSets？否。** 两者 deterministic eval 的 payload 都是 0：argmax 塌缩到全体 idle（completion=1 因 idle 记为到达目标，service=0，accuracy≈0.417 只是碰巧对齐 Oracle 的 idle 比例）。MLP payload **8.25e6**，conflict=0，interference utility 1.20e6。
2. **优势是否来自关系建模？否。** HGNN 训练末期 entropy 仍约 14，UU 干扰边没有变成可执行的 option。MLP 用固定拼接顺序携带 UAV 身份，更能学会「少发射、避免同簇互扰」。这与 Phase-7 一致：关系编码器更难在 80 updates 内收敛。
3. **是否形成最终论文模型？不能把 HGNN 定为最终高层。** L3-B 上当前可写的是 **MLP option + hold=15 + fly-toward**。DeepSets/HGNN 需要更长训练或更强 idle 正则之后再比。Oracle 脚本数字来自 seeds 0–7；本表学习策略评估为 seeds 8–15。

JSON：`src/wt/htw_phase9/outputs/hgnn_comparison/summary.json`
