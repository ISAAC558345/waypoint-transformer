# HTW Phase-5Q Encoder vs Assignment Scorer Audit

**planner_ready = False**

**deployment_verified = False**

本阶段只分开两个因素。Model-A 直接使用 Phase-5P 已冻结的 J_MAPPO 头结果。Model-B 只把同一个 Phase-5L encoder 从冻结改为可训练，Joint Head、Sinkhorn、Hungarian 和候选池不变。Model-C 保持 encoder 冻结，用 AssignmentConditionedSetScorer 直接给 120 个合法 assignment 打分，不再把分数拆回 pair score 后做 Hungarian。两组都使用相同的 J_MAPPO ranking pairs、10 个 epoch 和 seed 0 到 4。310101 不进入梯度、early stopping 或选模，只作为 development evaluation。

## 1. 训练信号与 score collapse

Model-A 的记录损失是 0.6931。它在训练集上的 collapse 比例是 0.3333，在 310101 上是 0.3333。
Model-B 第 1 个 epoch 的平均损失是 0.6931，第 10 个 epoch 是 0.6931。训练 collapse 比例 1.0000，310101 collapse 比例 1.0000。训练拟合是否成立：False。
Model-C 第 1 个 epoch 的平均损失是 0.6931，第 10 个 epoch 是 0.4361。训练 collapse 比例 0.0000，310101 collapse 比例 0.0000。训练拟合是否成立：True。

score collapse 的定义是 state 内 score 标准差或极差低于 1e-6。Model-A 没有保存每个 epoch 的分数，表里 1 到 9 epoch 只有当时记录的损失；第 10 epoch 的排序指标来自 Phase-5P。

## 2. Development evaluation（310101）

Top-1 和 regret 先在每个 state 内计算，再对 seed 和时刻做平均。稳定改善的预先标准是：相对 Model-A，五个 seed 的 macro J_MAPPO 平均至少增加 1 个 payload block（1.0e6 bit），macro normalized regret 下降，并且至少 3 个 seed 同时满足载荷上升和 regret 下降。

Model-B 相对 Model-A：mean ΔJ_MAPPO -3.3333e+05，median -3.3333e+05，std 2.3570e+05。
mean Δnormalized regret 0.1111，median 0.1111，std 0.0786。
Model-C 相对 Model-A：mean ΔJ_MAPPO 1.3333e+06，median 1.3333e+06，std 0.0000e+00。
mean Δnormalized regret -0.6667，median -0.6667，std 0.0000。

model_b seed 0：ΔJ_MAPPO -3.3333e+05，Δnormalized regret 0.1111，ΔSpearman NA。
model_b seed 1：ΔJ_MAPPO -6.6667e+05，Δnormalized regret 0.2222，ΔSpearman NA。
model_b seed 2：ΔJ_MAPPO -3.3333e+05，Δnormalized regret 0.1111，ΔSpearman NA。
model_b seed 3：ΔJ_MAPPO 0.0000e+00，Δnormalized regret 0.0000，ΔSpearman NA。
model_b seed 4：ΔJ_MAPPO -3.3333e+05，Δnormalized regret 0.1111，ΔSpearman NA。
model_c seed 0：ΔJ_MAPPO 1.3333e+06，Δnormalized regret -0.6667，ΔSpearman NA。
model_c seed 1：ΔJ_MAPPO 1.3333e+06，Δnormalized regret -0.6667，ΔSpearman NA。
model_c seed 2：ΔJ_MAPPO 1.3333e+06，Δnormalized regret -0.6667，ΔSpearman NA。
model_c seed 3：ΔJ_MAPPO 1.3333e+06，Δnormalized regret -0.6667，ΔSpearman NA。
model_c seed 4：ΔJ_MAPPO 1.3333e+06，Δnormalized regret -0.6667，ΔSpearman NA。

ΔSpearman 为 NA，是因为 Model-A 在 310101 t=0 的 Spearman 没有定义，配对后的 seed macro 不能丢掉这个 state。Model-C 自己在三个时刻的 Spearman 分别是 0.6416、0.4871、0.5439。

| model | t | Spearman | Top-1 | J_MAPPO | normalized regret | collapsed |
| --- | --- | --- | --- | --- | --- | --- |
| model_a | 0 | NA | 0.0000 | 0.0000e+00 | 1.0000 | 1.0000 |
| model_a | 15 | 0.5373 | 1.0000 | 3.0000e+06 | 0.0000 | 0.0000 |
| model_a | 30 | -0.0348 | 0.0000 | 0.0000e+00 | 1.0000 | 0.0000 |
| model_b | 0 | NA | 0.0000 | 0.0000e+00 | 1.0000 | 1.0000 |
| model_b | 15 | NA | 0.2000 | 1.4000e+06 | 0.5333 | 1.0000 |
| model_b | 30 | NA | 0.0000 | 6.0000e+05 | 0.8000 | 1.0000 |
| model_c | 0 | 0.6416 | 1.0000 | 1.0000e+06 | 0.0000 | 0.0000 |
| model_c | 15 | 0.4871 | 1.0000 | 3.0000e+06 | 0.0000 | 0.0000 |
| model_c | 30 | 0.5439 | 1.0000 | 3.0000e+06 | 0.0000 | 0.0000 |

310101 t=15 的平均解码 J_MAPPO：Model-A 3.0000e+06，Model-B 1.4000e+06，Model-C 3.0000e+06。这个时刻不能单独作为成功标准。310101 t=30 的平均解码 J_MAPPO：Model-A 0.0000e+00，Model-B 6.0000e+05，Model-C 3.0000e+06，pool oracle 3.0000e+06。平均载荷最高的是 model_c。

## 3. Additivity audit

additive 模型和 pairwise 模型只在 310100 上做无惩罚最小二乘，再在 310101 上评价。它们是离线诊断，不是可部署模型。

310101 macro：additive R2 -0.9723，Spearman 0.1075，Top-1 0.3333，normalized regret 0.6667，MAE 7.9526e+05。
pairwise R2 -1.0701，Spearman 0.1093，Top-1 0.3333，normalized regret 0.6667，MAE 7.9856e+05。
pairwise 是否按预先标准显著优于 additive：False。

## 4. 结论

Gate 是 **Q-E**。

五个预先条件没有一条被完整满足。Model-C 在 310101 上达到了载荷改善线，Model-B 没有，所以不是 Q-A、Q-C 或 Q-D。状态无关的 pairwise 回归没有优于 additive，所以也不满足 Q-B 的 audit 条款。Q-E 的原始定义是两边都无法在 310100 上学习；这个定义与 Model-C 的训练曲线不符。文件中的 Q-E 只是这个未覆盖组合的回退标签。因果对照本身指向 assignment scorer：下一步若继续，应保留冻结 encoder，检查非加性 scorer，而不是先把 HGNN 加深。

Q1. score collapse 在哪里解除？ 重新训练 encoder 后 collapse 比例没有下降：训练集 0.3333 -> 1.0000，310101 0.3333 -> 1.0000。换成 non-additive scorer 后 collapse 比例下降：训练集 0.3333 -> 0.0000，310101 0.3333 -> 0.0000。

Q2. J_MAPPO 能否被单边效用的和充分表示？ 不能。310101 上 additive macro R2 是 -0.9723，Spearman 是 0.1075，Top-1 是 0.3333，normalized regret 是 0.6667。

Q3. UAV-UAV / assignment pair interaction 是否必要？ 现有 audit 没有证明它是必要项。pairwise 诊断没有按预先标准优于 additive。

Q4. 310101 t=30 哪个模型更接近 pool oracle？ pool oracle 是 3.0000e+06。Model-A 0.0000e+00，Model-B 6.0000e+05，Model-C 3.0000e+06。平均载荷最高的是 model_c。

Q5. 下一步应该优化什么？ assignment scorer。Model-B 没有改善，Model-C 解除了 score collapse，并在五个 seed 上提高了 310101 的真实载荷。状态无关的 pairwise 回归没有优于 additive，因此这还不是 Q-B 的完整授权，也不能据此把 encoder 加深。

`planner_ready = False`

`deployment_verified = False`
