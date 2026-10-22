# HTW Phase-5T Non-Additive Scorer Failure Mechanism Audit

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练。Phase-5S 的五个 checkpoint 只做推理。310100–310117 都是已经消耗的 development 数据，没有新开 confirmation。

## 1. 时间位置

| 集合 | t | mean regret | Top-1 | Top-3 | Spearman | J_MAPPO |
| --- | --- | --- | --- | --- | --- | --- |
| confirmation | 0 | 1.0000e+06 | 0.2500 | 0.6250 | 0.6399 | 1.6250e+06 |
| confirmation | 15 | 2.4250e+06 | 0.1250 | 0.1250 | 0.0189 | 8.2500e+05 |
| confirmation | 30 | 0.0000e+00 | 1.0000 | 1.0000 | 0.7687 | 3.0000e+06 |
| train | 0 | 4.0000e+05 | 0.7000 | 0.9000 | 0.6816 | 2.0000e+06 |
| train | 15 | 2.2800e+06 | 0.1000 | 0.3000 | 0.2575 | 9.2000e+05 |
| train | 30 | 0.0000e+00 | 1.0000 | 1.0000 | 0.7431 | 2.9000e+06 |

confirmation 上 t=15 的平均 regret 是 2.4250e+06，t=0 与 t=30 的平均是 5.0000e+05。失败明显集中在 t=15。不能只归因于训练场景数量。

## 2. OOD 与 regret

OOD 分数是标准化最近邻距离。绝对坐标不进入这个分数，因为每个 scenario 的地图不同。
Pearson 0.1408，Spearman 0.2217。
最高 regret 25% 的平均 OOD 0.6112，最低 25% 0.2829。
高 regret 组超出训练支持的比例 0.0769，低 regret 组 0.0000。

## 3. Pairwise loss 和 score 排名

训练 pair 117438。Top-1 pair fraction 0.2976，Top-3 0.4083，低收益 pair 0.5917。
all-pair loss 与 regret 的 Spearman 0.6813。top1-margin loss 与 regret 的 Spearman 0.8250。
真实最优的 score 排名中位数 1.0。Top-5 以内的 state 比例 0.7037。

## 4. Representation

Fisher 中位数 4.0892。训练集 Fisher 中位数 4.7635。
collision rate 均值 0.3261。五个 seed 的 score Spearman 0.9991。

## 5. 310114 t=15

4/5 个 seed 选择 4;2;-1;-1;-1;5。池内最优是 -1;4;2;5;-1;-1。第 1 步累计差 0.0000e+00，第 15 步累计差 5.0000e+06。决策时刻的假设速率更高，但第 5 步实现的平均 SINR 是预测 0.1712、最优 7.4075。z_A cosine 超过 0.99，冻结表示没有把这组 15 步联合信道差分开。

| 来源 | assignment | J_MAPPO | score | score rank | mean distance | mean rate | mean urgency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| J | -1;4;2;5;-1;-1 | 5.0000e+06 | -13.4254 | 81 | 21.9859 | 1.0818e+06 | 0.5000 |
| J | -1;4;2;-1;5;-1 | 5.0000e+06 | -13.5032 | 92 | 22.3889 | 1.0531e+06 | 0.5000 |
| J | -1;5;2;4;-1;-1 | 5.0000e+06 | -13.4319 | 82 | 21.9914 | 1.0809e+06 | 0.5000 |
| J | -1;5;2;-1;-1;4 | 5.0000e+06 | -13.5789 | 104 | 22.3408 | 1.0551e+06 | 0.5000 |
| J | -1;5;-1;4;2;-1 | 5.0000e+06 | -13.4006 | 78 | 22.1210 | 1.0717e+06 | 0.5000 |
| score | 4;2;-1;-1;-1;5 | 0.0000e+00 | -12.1606 | 1 | 18.3903 | 1.4070e+06 | 0.5000 |
| score | 4;-1;-1;2;-1;5 | 0.0000e+00 | -12.1647 | 2 | 18.5737 | 1.3877e+06 | 0.5000 |
| score | 4;2;5;-1;-1;-1 | 0.0000e+00 | -12.2795 | 3 | 18.7678 | 1.3679e+06 | 0.5000 |
| score | 4;-1;5;2;-1;-1 | 0.0000e+00 | -12.2911 | 4 | 18.9513 | 1.3486e+06 | 0.5000 |
| score | -1;2;-1;-1;4;5 | 0.0000e+00 | -12.3056 | 5 | 18.6581 | 1.3800e+06 | 0.5000 |

| role | step | cumulative | mean SINR | interference | progress | mean distance | assigned in range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predicted | 1 | 0.0000e+00 | 0.1712 | 0.6433 | 0.0070 | 3.5239 | 3 |
| predicted | 3 | 0.0000e+00 | 0.1712 | 0.6433 | 0.0210 | 3.5239 | 3 |
| predicted | 5 | 0.0000e+00 | 0.1712 | 0.6433 | 0.0349 | 3.5239 | 3 |
| predicted | 10 | 0.0000e+00 | 0.1712 | 0.6433 | 0.0699 | 3.5239 | 3 |
| predicted | 15 | 0.0000e+00 | 0.1712 | 0.6433 | 0.1048 | 3.5239 | 3 |
| optimal | 1 | 0.0000e+00 | 7.4075 | 0.2013 | 0.0665 | 0.5579 | 3 |
| optimal | 3 | 0.0000e+00 | 7.4075 | 0.2013 | 0.1995 | 0.5579 | 3 |
| optimal | 5 | 0.0000e+00 | 7.4075 | 0.2013 | 0.3325 | 0.5579 | 3 |
| optimal | 10 | 3.0000e+06 | 166.6667 | 0.0000 | 0.1111 | 0.5691 | 3 |
| optimal | 15 | 5.0000e+06 | 166.6667 | 0.0000 | 0.0556 | 14.4666 | 1 |

## 6. Regret 最大的 20 个 state × seed

| scenario | t | seed | regret | mechanisms |
| --- | --- | --- | --- | --- |
| 310114 | 15 | 0 | 5.0000e+06 | REP|TEMP |
| 310114 | 15 | 1 | 5.0000e+06 | REP|TEMP |
| 310114 | 15 | 2 | 5.0000e+06 | REP|TEMP |
| 310114 | 15 | 3 | 5.0000e+06 | REP|TEMP |
| 310114 | 15 | 4 | 5.0000e+06 | REP|TEMP |
| 310100 | 15 | 0 | 3.0000e+06 | REP|TEMP |
| 310100 | 15 | 1 | 3.0000e+06 | REP|TEMP |
| 310100 | 15 | 2 | 3.0000e+06 | REP|TEMP |
| 310100 | 15 | 3 | 3.0000e+06 | REP|TEMP |
| 310100 | 15 | 4 | 3.0000e+06 | REP|TEMP |
| 310102 | 15 | 0 | 3.0000e+06 | REP|TEMP |
| 310102 | 15 | 1 | 3.0000e+06 | REP|TEMP |
| 310102 | 15 | 2 | 3.0000e+06 | REP|TEMP |
| 310102 | 15 | 4 | 3.0000e+06 | REP|TEMP |
| 310104 | 15 | 0 | 3.0000e+06 | REP|TEMP |
| 310104 | 15 | 1 | 3.0000e+06 | REP|TEMP |
| 310104 | 15 | 2 | 3.0000e+06 | REP|TEMP |
| 310104 | 15 | 3 | 3.0000e+06 | REP|TEMP |
| 310104 | 15 | 4 | 3.0000e+06 | REP|TEMP |
| 310105 | 15 | 0 | 3.0000e+06 | REP|TEMP |

## 7. 结论

Gate 是 **T-E**。优先级：REP, TEMP。

先处理 REP。下一阶段只修改必要的 state representation，不增加 HGNN 深度，不同时改 ranking loss 或扩大数据。

Q1. 为什么 310114 t=15 五个 seed 选同一个错 assignment？ 4/5 个 seed 选择 4;2;-1;-1;-1;5，其余 seed 仍选择 J_MAPPO 为 0 的近距离集合。决策时刻预测集合平均距离 18.3903 m、瞬时 rate_if_assigned 1.4070e+06；池内最优平均距离 21.9859 m、瞬时 rate_if_assigned 1.0818e+06。执行后预测 J_MAPPO 0.0000e+00，最优 5.0000e+06。最优 assignment 的 score 排名中位数是 78.0。两组 z_A 的 cosine 是 0.9980。决策时刻的假设速率更高，但第 5 步实现的平均 SINR 是预测 0.1712、最优 7.4075。z_A cosine 超过 0.99，冻结表示没有把这组 15 步联合信道差分开。

Q2. 失败主要来自哪一类？ REP、TEMP

Q3. 真实最优通常排第几？ 全部 state×seed 的 rank 中位数是 1.0，均值 8.6。排名不超过 3 的比例 0.6667，不超过 5 的比例 0.7037。

Q4. 高低 J_MAPPO 的表示是否可分？ 类均值沿均值差方向的 Fisher ratio 中位数是 4.0892，但 cosine separation 只有 0.0011，最近邻 collision rate 是 0.3261。高低 J_MAPPO 的 z_A 几乎平行，最近邻里有大量 payload 差至少 1 个 block 的碰撞。

Q5. pairwise loss 是否主要在排低价值 assignment？ 训练集有效 pair 共 117438。涉及真实 Top-1 的比例 0.2976，涉及 Top-3 的比例 0.4083，两边都在 Top-3 之外的比例 0.5917。低价值 pair 占多数，但 all-pair loss 与 regret 的 Spearman 仍有 0.6813，并不是完全浪费。

Q6. 是否缺少执行历史？ option age、service duration、target lock duration、hold remaining 没有单独输入。在 t=0、15、30 这三个决策点，新 option 的 hold remaining 都是 15，单靠这个常数不能区分 assignment。service progress 和当前 assignment 已经在冻结 encoder 里。encoder 的 global mix 权重绝对值最大值是 0，episode time 没有进入节点表示。310114 的 payload 差来自 15 步内实现的 SINR，不是缺少一个 option-age 计数器。

Q7. 下一阶段做什么？ 先处理 REP。下一阶段只修改必要的 state representation，不增加 HGNN 深度，不同时改 ranking loss 或扩大数据。

`planner_ready = False`

`deployment_verified = False`
