# HTW Phase-9.1 HGNN Training Stabilization Report

修正 Phase-9 的全体 idle 崩溃。未改 L3-B、SINR、低层 PPO、hold=15。未改 Phase-9 代码与结果。
低层仍为 fly-toward。高层动作 **去掉 idle**：`z ∈ {0,…,11}`，每机必须选一个 buoy。
seed=0，PPO updates=300，reward=`delta`。

## 1. 失败诊断（Phase-9）

DeepSets/HGNN greedy eval payload=0，argmax 逃到 idle。不能据此判 HGNN 无效。

## 2. 稳定化设置

1. 动作空间：无 idle。
2. 训练预算：300 updates（不是 80）。
3. Warm start：冻结 MLP 当教师，交叉熵监督 HGNN option，再 PPO fine-tune。

## 3. 结果（eval seeds 8–15）

| 方法 | payload | option accuracy | entropy | conflict | interference utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP baseline | 9500000 | 0.136 | 0.247 | 0.156 | 1.312e+06 |
| HGNN random init | 875000 | 0.000 | 13.936 | 0.954 | 1.010e+05 |
| HGNN imitation warm start | 3625000 | 0.161 | 8.215 | 0.590 | 7.530e+05 |

模仿阶段准确率 = 0.978。
Warm / MLP payload 比 = 38.16%。

## 4. 成功标准

**HGNN warm start 接近或超过 MLP（≥80% payload）：不成立**

- 若成立：HGNN 具备关系建模潜力，失败主要是优化/idle 而不是图无用。
- 若仍失败：需要重新设计 HGNN 输出（例如排他 matching 头、匈牙利层、或显式冲突惩罚），而不是继续加 updates。

补充：去掉 idle 后 MLP 可学（payload 9.50e6，eval entropy 0.25）。随机 HGNN 仍高熵、高冲突（0.954），几乎未形成 assignment。模仿准确率 0.978 说明 HGNN **能拟合** MLP 的 option；随后 PPO 把 entropy 从约 2.2 冲回 13，warm 最终只有 MLP 的 38%。因此下一刀应改 **HGNN 输出/优化接口**（保留模仿、约束 PPO 别冲掉 teacher，或输出置换/排他 matching），而不是再加 updates。

JSON：`src/wt/htw_phase91/outputs/hgnn_stabilization/summary.json`
