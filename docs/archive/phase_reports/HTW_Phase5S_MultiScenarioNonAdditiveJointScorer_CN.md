# HTW Phase-5S Multi-Scenario Non-Additive Joint Scorer

**planner_ready = False**

**deployment_verified = False**

本阶段只把训练场景从 310100 扩到 310100–310109。AssignmentConditionedSetScorer、冻结 encoder、loss、学习率、10 个 epoch 和 seed 0–4 都不变。310110–310117 在五个 checkpoint 保存之后才第一次生成标签，并且只评价一次。

## 1. 训练

训练状态 30 个，标签全部来自已有的 Phase-5O / Phase-5R，没有为训练集重新 rollout。
第 10 个 epoch 的平均损失是 0.3462。平均 collapse ratio 是 0.0000。平均训练 Top-1 是 0.6000。

## 2. Fresh confirmation（310110–310117）

比较的是 Phase-5S 减去只用 310100 训练的 Phase-5Q Model-C。统计以 scenario 为 cluster。

ΔJ_MAPPO mean 1.0833e+05，median 0.0000e+00，std 6.1897e+05，positive fraction 0.0833。
95% CI [-8.3333e+04, 3.2500e+05]。
Δnormalized regret mean -0.0361，95% CI [-0.1083, 0.0278]。
8 个 scenario：更高 2，持平 5，更低 1。5 个 seed 里 5 个平均差大于 0。

| scenario | ΔJ_MAPPO | Δregret | Phase-5S Top-1 | Phase-5Q Top-1 | scorer regret | coverage regret |
| --- | --- | --- | --- | --- | --- | --- |
| 310110 | 0.0000e+00 | 0.0000 | 0.6667 | 0.6667 | 6.6667e+05 | 0.0000e+00 |
| 310111 | 6.6667e+05 | -0.2222 | 0.3333 | 0.0000 | 1.3333e+06 | 0.0000e+00 |
| 310112 | 0.0000e+00 | 0.0000 | 0.6667 | 0.6667 | 1.0000e+06 | 1.0000e+06 |
| 310113 | 5.3333e+05 | -0.1778 | 0.3333 | 0.0000 | 1.4667e+06 | 0.0000e+00 |
| 310114 | 0.0000e+00 | 0.0000 | 0.3333 | 0.3333 | 2.0000e+06 | 0.0000e+00 |
| 310115 | 0.0000e+00 | 0.0000 | 0.6667 | 0.6667 | 6.6667e+05 | 3.3333e+05 |
| 310116 | 0.0000e+00 | 0.0000 | 0.3333 | 0.3333 | 6.6667e+05 | 6.6667e+05 |
| 310117 | -3.3333e+05 | 0.1111 | 0.3333 | 0.3333 | 1.3333e+06 | 6.6667e+05 |

Phase-5S 平均 scorer regret 1.1417e+06。平均 coverage regret 3.3333e+05。

## 3. Regret 最大的 10 个 state × seed

| scenario | t | seed | regret | class |
| --- | --- | --- | --- | --- |
| 310114 | 15 | 0 | 5.0000e+06 | A_score_ordering_error |
| 310114 | 15 | 1 | 5.0000e+06 | A_score_ordering_error |
| 310114 | 15 | 2 | 5.0000e+06 | A_score_ordering_error |
| 310114 | 15 | 3 | 5.0000e+06 | A_score_ordering_error |
| 310114 | 15 | 4 | 5.0000e+06 | A_score_ordering_error |
| 310111 | 15 | 0 | 3.0000e+06 | A_score_ordering_error |
| 310111 | 15 | 1 | 3.0000e+06 | A_score_ordering_error |
| 310111 | 15 | 2 | 3.0000e+06 | A_score_ordering_error |
| 310111 | 15 | 3 | 3.0000e+06 | A_score_ordering_error |
| 310111 | 15 | 4 | 3.0000e+06 | A_score_ordering_error |

## 4. 结论

Gate 是 **S-C**。

平均收益为正，但置信区间或 scenario / seed 方向还不稳。继续扩大 development training scenarios，不增加网络复杂度。

Q1. 训练场景从 1 个扩到 10 个是否提高了 scorer？ 没有达到预先写好的稳定提高标准。 mean ΔJ_MAPPO 1.0833e+05，95% CI [-8.3333e+04, 3.2500e+05]。

Q2. 新 confirmation 上是否稳定优于 Phase-5Q？ 没有达到预先写好的稳定提高标准。 mean ΔJ_MAPPO 1.0833e+05，95% CI [-8.3333e+04, 3.2500e+05]。

Q3. seed 和 scenario 是否同向？ scenario：更高 2，持平 5，更低 1。seed：5/5 个平均差大于 0。

Q4. 哪一类 regret 更大？ SCORER_REGRET 更大，1.1417e+06，coverage regret 3.3333e+05。

Q5. 下一步做什么？ 继续扩大 development training scenarios，不增加网络复杂度。

Q6. 能否把 encoder 加 non-additive scorer 固定为当前高层结构？ 架构可以继续沿用。Phase-5Q 和 Phase-5R 已经说明，冻结的 Phase-5L encoder 加上完整 non-additive scorer 优于加性打分。Phase-5S 没有给出替换这套结构的理由，也没有给出把本次多场景权重当成更优 checkpoint 的稳定证据。现在仍不能部署。

`planner_ready = False`

`deployment_verified = False`
