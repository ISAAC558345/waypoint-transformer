# HTW Phase-6.1 Oracle Option + Low-level PPO Report

Level-2-A（M=5, K=12）上验证层次控制器：高层仍为 **Oracle buoy option**，低层 PPO 只输出 movement。
未加 HGNN、未训练 high-level policy、未改 Transformer、未改 L2-A 环境、未改 Round-30 PPO（GAE/clip/Adam/loss）、未开 Round。
预算：80 updates，rollout=48，num_envs=4，seed 0。
训练种子 [0, 1, 2, 3, 4, 5, 6, 7]，评估种子 [8, 9, 10, 11, 12, 13, 14, 15]。

## 1. 结构

- High-level：因果验证中的 `oracle_option_assignment`（簇 option 影子搜索 + 簇内排他）。输出每机 buoy index 或 idle，**one-hot**（12+idle）。不学习 π_H。
- Low-level：输入 `[state, z]`，UAV dim=21。输出 movement `(5,2)`。
- State-only PPO：输入 state，UAV dim=8；关联为当前贪心 `myopic_assignment`。
- 环境关联由高层/贪心脚本写入 `env.step(movement, association)`；低层不改 matching 规则。

## 2. Reward

- PPO 使用环境 payload 增量换算的 **r_comm（完成块数）**。不改环境内部公式。
- 不把 waypoint 距离写入 reward。goal completion / service success 只作记录。

## 3. 实验结果

| 方法 | payload bits | env return (blocks) | residual queue | dropped | goal completion (in-range) | service success | movement m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Myopic + fly-toward（非学习） | 13000000 | — | 29.62 | 77.38 | — | — | — |
| Oracle option + fly-toward（影子） | 27750000 | — | 26.62 | 65.62 | — | — | — |
| Level-2 State-only PPO | 2875000 | 2.88 | 39.12 | 78.00 | 0.108 | 0.012 | 779.0 |
| Oracle Option + Low-level PPO | 500000 | 0.50 | 41.62 | 77.88 | 0.110 | 0.002 | 1348.1 |

Option PPO / Oracle shadow = **0.018**（接近阈值 0.80）。

诊断：影子控制器用单位方向飞向 z，payload 高；PPO 只拿稀疏 r_comm（D=3 且须停在 16 m 内）。Option PPO 位移更大但 in-range/service 接近 0.11/0.00，说明 **z 没有被执行成服务**，不是 option 标签无价值。

## 4. 三个问题

1. **Level-2 是否需要 HRL？需要。** 评估种子上 Oracle 影子 payload 27750000 vs 贪心脚本 13000000。长期任务 option 仍有间隙；State-only PPO 不能靠运动补上高层分配。
2. **低层是否可以执行 task option？当前预算下不足。** 低层应对准 z 指定 buoy 并进入服务半径。Option PPO payload 500000，影子 27750000；in-range completion 0.110。
3. **是否进入 learned high-level policy？不进入。** 条件：低层接近影子 **且** Option PPO 严格高于 State-only PPO。当前：接近影子=False，Option>State-only=False。

未训练 π_H。HGNN 仍不加入。

JSON：`src/wt/htw_phase61/outputs/oracle_option_ppo/summary.json`
