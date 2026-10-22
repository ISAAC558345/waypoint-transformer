# HTW Phase-6.3 Learned High-level Task Option Policy

Level-2-A 上训练 π_H(z|s)，z 为 buoy+idle。低层为 Phase-6.2 最佳冻结控制器（λ_c=0.5, λ_d=0.05）。
未加 HGNN、未改环境、未改 Round-30 PPO 核心文件、未用 Task-Waypoint、未开 Round。
Stage-1：冻结低层，高层奖励为逐步 payload 增量（回合累计 payload 的分解）。预算 80 updates，seed 0。
Stage-2 联合微调：未执行（仅当 Stage-1 高于 State-only 至少 1000000 bits）。

## 1. 结构

- π_H：DeepSets + 每机 Categorical(13)，采样后排他投影。
- π_L：冻结（Stage-1）的 `[state,z]→movement`，PPO 为 Round-30 连续动作实现。
- 高层 Discrete PPO 复用 `compute_gae` 与相同 clip/Adam/value/entropy 系数，不修改 `round30/ppo.py`。

## 2. 结果

| 方法 | payload | env return | completion | service | option accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| State-only PPO | 2875000 | 2.88 | 0.108 | 0.012 | 0.462 |
| Oracle Option + Low-level PPO | 10250000 | 10.25 | 0.586 | 0.043 | 1.000 |
| Learned Option + Low-level PPO | 2625000 | 2.62 | 0.803 | 0.055 | 0.020 |

Learned / Oracle payload = **0.256**（接近阈值 0.80）。绝对差 7625000 bits。

诊断：Learned 的 in-range completion 高达 0.80，但 option accuracy 仅 0.02、payload 低于 State-only。说明 π_H 倾向选择**已经靠近、易完成的 buoy**，没有学到 Oracle 的远簇/延迟到达分配。HRL 结构成立（Oracle+低层 1.025e7），**可学习的 π_H 尚未成立**。

## 3. 三个问题

1. **高层是否能学习任务选择？当前预算下很弱。** option accuracy=0.020（Oracle 对照为 1.0）。
2. **距离 Oracle 差距：** payload 比 Oracle+低层低 7625000 bits（74.4% 相对缺口）。明显低于 Oracle，option policy 仍难：合法匹配组合大、排他耦合、与冻结低层的联合非平稳。
3. **是否进入 HGNN 阶段？建议作为后续关系建模，本阶段不加入。** 本阶段 **未加入 HGNN**。若 accuracy 低且 payload 明显落后，图结构关联是候选，而不是现在实现。

JSON：`src/wt/htw_phase63/outputs/learned_option/summary.json`
