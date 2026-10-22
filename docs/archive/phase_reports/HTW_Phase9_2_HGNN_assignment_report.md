# HTW Phase-9.2 HGNN Assignment Head Report

HGNN 不再做独立 buoy 分类。边分 `S(i,j)` 经 **Hungarian matching** 得到排他 assignment。
未改 L3-B、SINR、低层 PPO、hold=15。未直接 PPO；标签为 Oracle assignment。低层 fly-toward。
监督：collect seeds [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]，80 epochs。Eval seeds 8–15。
Buoy 容量=1（L3-B 竞争）。允许 idle（每机一条 dummy 列）。

## 1. 结构

- MLP：独立 categorical（buoy+idle），argmax，可冲突。
- DeepSets：UAV/Buoy/边编码 → `S(i,j)` + idle → Hungarian。
- HGNN：UB 消息 + UU 干扰边 → `S(i,j)` + idle → Hungarian。

## 2. 结果

| 方法 | payload | assignment accuracy | conflict | SINR utility (mean rate) |
| --- | ---: | ---: | ---: | ---: |
| MLP option classification | 4375000 | 0.418 | 0.099 | 9.433e+05 |
| DeepSets assignment head | 4000000 | 0.304 | 0.000 | 5.449e+05 |
| HGNN edge scoring + matching | 2250000 | 0.414 | 0.000 | 3.855e+05 |

HGNN − DeepSets payload = -1750000。

## 3. 成功标准

**HGNN assignment > DeepSets assignment：不成立**

若成立：HGNN 适合关系匹配，问题出在独立 option 分类而不是图编码器。
若不成：即使有 matching 层，边分仍不够；需要改 score 设计或标签，而不是再对独立 categorical 做 PPO。

未做 PPO fine-tune（Hungarian 不可微；本阶段只验证匹配头）。

JSON：`src/wt/htw_phase92/outputs/assignment_head/summary.json`
