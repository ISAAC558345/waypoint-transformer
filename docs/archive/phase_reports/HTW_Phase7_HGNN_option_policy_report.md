# HTW Phase-7 HGNN High-level Option Policy

HGNN 只编码高层 option。低层仍是冻结的 `[state,z]→movement` PPO。hold=15，reward=`delta`，与 Phase-6.4 最佳设置相同。未改环境、未改低层、未用 Task-Waypoint。
相同 seed=0，80 updates，rollout=16，num_envs=4。

## 1. 结构

- MLP：把 padded UAV/buoy 拼成向量再出每机 13 维 logits。
- DeepSets：与 6.4 相同的置换不变集合编码器。
- HGNN：UAV–Buoy 二部图两层消息传递。边特征为 distance / ETA / rate 代理 / 邻机干扰。

## 2. 结果

| 方法 | payload | completion | service | option accuracy | eval entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| State-only (6.3) | 2875000 | 0.108 | 0.012 | 0.462 | — |
| Oracle + frozen LL | 10250000 | 0.586 | 0.043 | 1.000 | — |
| DeepSets hold=15 (6.4) | 11000000 | 0.692 | 0.087 | 0.105 | — |
| MLP High-level | 15750000 | 0.476 | 0.070 | 0.210 | 2.731 |
| DeepSets High-level | 12500000 | 0.582 | 0.076 | 0.131 | 5.904 |
| HGNN High-level | 4500000 | 0.604 | 0.031 | 0.086 | 7.894 |

HGNN − DeepSets payload = -8000000；accuracy 差 = -0.044。

## 3. 三个问题

1. **HGNN 是否提升高层任务选择？否。** 同预算下 payload 与 accuracy 都低于 DeepSets（4.5e6 / 0.086 vs 1.25e7 / 0.131），更低于 MLP（1.575e7 / 0.210）。训练末期 HGNN entropy 仍约 7.7，MLP 已降到约 3.2，图编码器更难在 80 updates 内收敛。
2. **图结构优势来自哪里？本实验没有观察到。** 边特征（distance/ETA/rate/interference）在理论上应对齐「谁该去哪个 buoy」，但优化失败。MLP 因固定拼接顺序隐式带了 UAV 身份，更利于排他分配；DeepSets 置换不变，介于两者之间。
3. **是否形成最终论文模型？不能把 HGNN 定成最终高层。** 论文可写的 HRL 骨架仍是：**hold=15 option + 冻结低层 PPO**；高层目前最强的是 MLP，DeepSets 次之。HGNN 只用于高层、未进低层。更长训练或更好的图正则之后才能再判。

JSON：`src/wt/htw_phase7/outputs/hgnn_option/summary.json`
