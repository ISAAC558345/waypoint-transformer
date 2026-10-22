# HTW Phase-5U Assignment Representation Repair

**planner_ready = False**

**deployment_verified = False**

本阶段只把当前状态能算出来的 assignment 关系特征拼到原来的 z_A 后面。Phase-5L encoder、ranking loss、训练场景和候选池都没有改。310110–310117 没有进入梯度。

## 1. Collision

REP-BASE confirmation collision 0.3310。
z_rel 本身 0.2420。
REP-AUG 初始化 0.2805，epoch 10 0.2521。

## 2. 310114 t=15

| seed | predicted | J_pred | optimum rank | J_opt | regret |
| --- | --- | --- | --- | --- | --- |
| 0 | 4;-1;5;2;-1;-1 | 0.0000e+00 | 49 | 5.0000e+06 | 5.0000e+06 |
| 1 | 4;-1;5;2;-1;-1 | 0.0000e+00 | 46 | 5.0000e+06 | 5.0000e+06 |
| 2 | 4;-1;5;2;-1;-1 | 0.0000e+00 | 47 | 5.0000e+06 | 5.0000e+06 |
| 3 | 4;-1;5;2;-1;-1 | 0.0000e+00 | 48 | 5.0000e+06 | 5.0000e+06 |
| 4 | 4;-1;5;2;-1;-1 | 0.0000e+00 | 46 | 5.0000e+06 | 5.0000e+06 |

五个 seed 的 optimum rank 中位数是 47.0。Phase-5T 大约在 78–104。

## 3. 按决策时刻

| t | BASE regret | AUG regret | BASE J | AUG J | ΔJ |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.0000e+06 | 5.0000e+05 | 1.6250e+06 | 2.1250e+06 | 5.0000e+05 |
| 15 | 2.4250e+06 | 1.9000e+06 | 8.2500e+05 | 1.3500e+06 | 5.2500e+05 |
| 30 | 0.0000e+00 | 2.5000e+05 | 3.0000e+06 | 2.7500e+06 | -2.5000e+05 |

整体 ΔJ mean 2.5833e+05，95% CI [0.0000e+00, 5.4167e+05]。
Δregret mean -2.5833e+05。scenario：更高 4，更低 1。seed 正向 5/5。

## 4. Inference ablation

| group | disagreement | Δregret | used |
| --- | --- | --- | --- |
| selected_edge_raw | 0.4417 | 1.1667e+05 | True |
| joint_geometry | 0.0750 | -5.0000e+04 | False |
| cross_geometry | 0.0000 | 0.0000e+00 | False |
| interference_summary | 0.2083 | 1.4167e+05 | True |

## 5. 结论

Gate 是 **U-B**。

表示有改善，但 t=15 的 regret 还高，或者 J_MAPPO 不稳定。下一阶段只做 TEMP augmentation。

Q1. collision 是否下降？ epoch-10 collision 0.2521，BASE 0.3310，下降 0.0790。

Q2. 310114 t=15 的最优排名前移到多少？ 中位数 47.0。

Q3. t=15 regret 是否明显下降？ t=15 AUG regret 1.9000e+06。仍然不低于 1 个 payload block。

Q4. J_MAPPO 是否稳定提高？ mean ΔJ 2.5833e+05，95% CI [0.0000e+00, 5.4167e+05]。scenario 正向 4/8，seed 正向 5/5。

Q5. 哪一组特征在起作用？ selected_edge_raw、interference_summary

Q6. 是否还需要 TEMP？ 需要。t=15 的执行差还在，下一阶段再加入 TEMP。

Q7. 下一阶段做什么？ 表示有改善，但 t=15 的 regret 还高，或者 J_MAPPO 不稳定。下一阶段只做 TEMP augmentation。

`planner_ready = False`

`deployment_verified = False`
