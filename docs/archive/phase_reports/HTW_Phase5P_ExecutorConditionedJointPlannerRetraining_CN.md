# HTW Phase-5P Executor-Conditioned Joint Planner Retraining

**planner_ready = False**

**deployment_verified = False**

本阶段是 label-only 对照。Joint Assignment Head、Sinkhorn、Hungarian、top-3 候选、学习率、优化器和 10 个 epoch 与 Phase-5N 相同。唯一变化是把 `J_FLY` 换成 Phase-5O 已经冻结的 `J_MAPPO`。没有重新 rollout，没有使用 310101 做梯度、early stopping 或选模。310101 只作为 development evaluation，不是 untouched deployment test。

## 1. 训练信号

310100 上，`J_FLY` 的有效有序对共 684，`J_MAPPO` 共 10301。监督信号确实增加了。

第 10 个 epoch 的平均损失：J_FLY 头 0.2310，J_MAPPO 头 0.6931。J_MAPPO 头停在 0.6931，也就是 `-log sigmoid(0)`，说明 S(A)-S(B) 的平均差距接近 0。

310100 t=15 上有 3/5 个 seed 的解码载荷从 0 升到 1.0e6 bit，但仍低于该状态 3.0e6 bit 的池内 oracle，Top-1 仍是 0。310100 t=0 的新头分数标准差低于 1e-6，其中 3 个 seed 的旧头 Spearman 为负、新头变为 NA。因此不能把丢掉 NA 之后的平均当成训练集排序提升。

## 2. Development evaluation（310101）

Top-1 和 regret 先在每个 state 内计算，再对三个时刻做 macro average。五个 seed 都保留。Spearman 差也先在 state 内配对。只要某个 state 的分配分数标准差低于 1e-6，该 seed 的 Spearman 差就记为 NA，不能把这个 state 从平均里丢掉后再声称排序提高。

`J_MAPPO` 相对 `J_FLY` 的 macro Spearman 差是 NA（median NA，std NA）。
Top-1 差是 0.0000（median 0.0000，std 0.0000）。
解码载荷差是 -6.6667e+04（median 0.0000e+00，std 1.4907e+05）。
归一化 regret 差是 0.0222（median 0.0000，std 0.0497）。正的 regret 差表示 regret 变大。
五个 seed 里，载荷上升且 regret 下降的有 0 个。

| model | t | Spearman | Top-1 | J_MAPPO | normalized regret |
| --- | --- | --- | --- | --- | --- |
| original_hgnn | 0 | -0.1627 | 0.0000 | 0.0000e+00 | 1.0000 |
| original_hgnn | 15 | 0.1115 | 0.0000 | 1.0000e+06 | 0.6667 |
| original_hgnn | 30 | -0.1048 | 0.0000 | 0.0000e+00 | 1.0000 |
| phase5l_hgnn | 0 | 0.3118 | 0.0000 | 0.0000e+00 | 1.0000 |
| phase5l_hgnn | 15 | 0.0917 | 0.0000 | 1.0000e+06 | 0.6667 |
| phase5l_hgnn | 30 | -0.1144 | 0.0000 | 0.0000e+00 | 1.0000 |
| phase5n_jfly | 0 | -0.5951 | 0.0000 | 0.0000e+00 | 1.0000 |
| phase5n_jfly | 15 | 0.5428 | 1.0000 | 3.0000e+06 | 0.0000 |
| phase5n_jfly | 30 | -0.0436 | 0.0000 | 2.0000e+05 | 0.9333 |
| phase5p_jmappo | 0 | NA | 0.0000 | 0.0000e+00 | 1.0000 |
| phase5p_jmappo | 15 | 0.5373 | 1.0000 | 3.0000e+06 | 0.0000 |
| phase5p_jmappo | 30 | -0.0348 | 0.0000 | 0.0000e+00 | 1.0000 |
| pool_oracle_mappo | 0 | 1.0000 | 1.0000 | 1.0000e+06 | 0.0000 |
| pool_oracle_mappo | 15 | 1.0000 | 1.0000 | 3.0000e+06 | 0.0000 |
| pool_oracle_mappo | 30 | 1.0000 | 1.0000 | 3.0000e+06 | 0.0000 |

310101 t=15 的池内 oracle J_MAPPO 是 3.0000e+06。 seed 0：J_FLY 头的解码载荷是 3.0000e+06，J_MAPPO 头是 3.0000e+06。 seed 1：J_FLY 头的解码载荷是 3.0000e+06，J_MAPPO 头是 3.0000e+06。 seed 2：J_FLY 头的解码载荷是 3.0000e+06，J_MAPPO 头是 3.0000e+06。 seed 3：J_FLY 头的解码载荷是 3.0000e+06，J_MAPPO 头是 3.0000e+06。 seed 4：J_FLY 头的解码载荷是 3.0000e+06，J_MAPPO 头是 3.0000e+06。

## 3. 结论

Gate 是 **LABELTRAIN-C**。

换成 J_MAPPO 后，排序和真实载荷都没有稳定提高。冻结的 Phase-5L 表示不足以支撑 executor-conditioned joint utility。下一阶段才允许在标签保持 J_MAPPO 的前提下重新训练 encoder。

Q1. 只替换标签是否提高了 planner？ 没有稳定提高。

Q2. 提升出现在哪些层面？ 排序没有提高；解码后的真实 J_MAPPO 没有提高。

Q3. 310101 t=15 是否修好？ 五个 seed 的两种头都解码到 3.0e6 bit。J_FLY 头已经到达池内最优，换成 J_MAPPO 没有改变这个时刻。

Q4. 五个 seed 是否同向？ 0/5 个 seed 同时提高了解码载荷并降低了 regret。

Q5. 下一步瓶颈： 冻结 encoder representation。

`planner_ready = False`

`deployment_verified = False`
