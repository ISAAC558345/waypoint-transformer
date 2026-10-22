# HTW Phase-9.3 Relational HGNN Assignment Ranking Report

改进 9.2：HGNN **编码器不变**，边分改为 relation-aware，损失加入 pairwise ranking。
未改 L3-B、SINR、低层 PPO、hold=15。未直接 PPO。Hungarian matching 仍用于解码。

Relation-aware score 输入：UAV embedding、Buoy embedding、UAV–UAV interference embedding、global graph embedding、原始 global。
Ranking：`-log sigmoid(score_pos - score_neg)`，含 UAV 内正负边、idle、以及同 buoy 上 UAV 间竞争。CE 权重 0.25。

## 结果（eval seeds 8–15）

| 方法 | payload | assignment accuracy | conflict | SINR utility | ranking accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase-9.2 HGNN edge score | 2250000 | 0.414 | 0.000 | 3.855e+05 | 0.979 |
| Phase-9.2 DeepSets + Hungarian | 4000000 | 0.304 | 0.000 | 5.449e+05 | — |
| Phase-9.3 relational HGNN | 2625000 | 0.408 | 0.000 | 5.594e+05 | 0.992 |

9.3 − 9.2 HGNN payload = 375000。
9.3 − DeepSets payload = -1375000。

**HGNN payload > DeepSets：不成立**

关系边分 + ranking 相对 9.2 HGNN 有小幅收益（payload +3.75e5，SINR utility 升至 5.59e5，ranking acc 0.992），但仍低于 DeepSets。训练集 pairwise 已经几乎排对，说明瓶颈不只是 CE，而是 **加性边分 + Hungarian 仍无法表达 SINR 组合值**（联合选谁 idle）。

JSON：`src/wt/htw_phase93/outputs/relational_hgnn/summary.json`
