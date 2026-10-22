# HTW Phase-5R Frozen Non-Additive Joint Scorer Generalization

**planner_ready = False**

**deployment_verified = False**

310102–310109 的角色是 `PHASE5_DEVELOPMENT_GENERALIZATION`。它们不是 deployment holdout，也不是最终 untouched test。Phase-5Q 没有保存权重。本阶段用原来的 seed、310100 数据和 10 个 epoch 把 Model-A、Model-C 恢复出来，并与 Phase-5Q 在 310100/310101 上的记录对齐。对齐之后，这 8 个 scenario 只做无梯度推理。

310102 和 310103 曾出现在更早的闭环审计里，但不是 Phase-5Q 的训练场景。310104–310109 没有进入先前的 Phase-5 protocol 名单。

## 1. 候选池与覆盖

模型池平均大小 120.0，扩展参考池平均大小 720.0。
Top-5 极大匹配不一定包含 top-3 极大匹配，因为 top-3 里空闲的 UAV 在 top-5 中可能仍有可分配浮标。覆盖 oracle 取两个池的并集。
平均 candidate coverage ratio 0.9083。平均 coverage regret 4.1667e+05 bit。平均 scorer regret 6.6667e+05 bit。24 个 state 里有 7 个的 coverage regret 至少为一个 payload block。
扩展池只用于覆盖诊断，没有提供给 Model-A 或 Model-C。边际指标是冻结 Phase-5L matching logit 的 top-5，和 top-3 候选来自同一次前向。

## 2. Paired comparison

每个 scenario、时刻和 model seed 先配对，再按 scenario 做 cluster bootstrap。同一 scenario 的三个时刻不会被当成独立场景。

ΔJ_MAPPO mean 9.6667e+05，median 1.0000e+06，std 1.6497e+06，positive fraction 0.6083。
cluster bootstrap 95% CI [5.9979e+05, 1.3167e+06]。
Δnormalized regret mean -0.3247，95% CI [-0.4486, -0.1953]。
8 个 scenario 中，Model-C 更高 7 个，持平 1 个，更低 0 个。
5 个 seed 里，平均 ΔJ_MAPPO 大于 0 的有 5 个。

| scenario | ΔJ_MAPPO | Δregret | Model-C Top-1 | coverage ratio | coverage regret | scorer regret |
| --- | --- | --- | --- | --- | --- | --- |
| 310102 | 8.6667e+05 | -0.2978 | 0.0000 | 1.0000 | 0.0000e+00 | 1.6667e+06 |
| 310103 | 1.5333e+06 | -0.5556 | 1.0000 | 0.8667 | 6.6667e+05 | 0.0000e+00 |
| 310104 | 1.0000e+06 | -0.3333 | 0.6667 | 0.9167 | 3.3333e+05 | 6.6667e+05 |
| 310105 | 5.3333e+05 | -0.1444 | 0.3333 | 1.0000 | 0.0000e+00 | 6.6667e+05 |
| 310106 | 0.0000e+00 | 0.0000 | 0.3333 | 0.8667 | 6.6667e+05 | 1.3333e+06 |
| 310107 | 8.0000e+05 | -0.2667 | 0.3333 | 0.8667 | 6.6667e+05 | 1.0000e+06 |
| 310108 | 1.2667e+06 | -0.4222 | 1.0000 | 0.8333 | 6.6667e+05 | 0.0000e+00 |
| 310109 | 1.7333e+06 | -0.5778 | 1.0000 | 0.9167 | 3.3333e+05 | 0.0000e+00 |

## 3. Inference ablation

消融只发生在推理时，没有重新训练。它测量已经训练好的 Model-C 用了哪些输入，不是公平的性能比赛。

| ablation | disagreement | mean |score| change | ΔJ_MAPPO | Δregret |
| --- | --- | --- | --- | --- |
| no_global | 0.0000 | 9.5350e-01 | 0.0000e+00 | 0.0000e+00 |
| no_uu | 0.0000 | 7.4046e-01 | 0.0000e+00 | 0.0000e+00 |
| no_interaction | 0.3750 | 6.3246e+00 | -2.0833e+05 | 2.0833e+05 |

Phase-5Q 的 additive / pairwise 回归是状态无关的系数表。它们在 310101 上 R2 为负，不能用来说明状态条件下的 non-additive set interaction 没有用。

## 4. 结论

Gate 是 **R-A**。

Model-C 在这 8 个 development-generalization scenario 上稳定优于 Model-A，候选覆盖也尚可。下一阶段可以固定这个 non-additive scorer，再在更多 development data 上训练。这仍然不是 deployment。

Q1. Model-C 是否仍稳定优于 Model-A？ 是。mean ΔJ_MAPPO 9.6667e+05，95% CI 下界 5.9979e+05。

Q2. 收益是否跨 scenario 稳定？ 8 个 scenario 里更高 7 个、持平 1 个、更低 0 个。

Q3. 五个 seed 是否同向？ 5/5 个 seed 的平均 ΔJ_MAPPO 大于 0。

Q4. 剩余误差主要来自哪里？ scorer。平均 scorer regret 6.6667e+05，平均 coverage regret 4.1667e+05。7/24 个 state 的候选池比并集 oracle 至少少一个 payload block。

Q5. Model-C 实际依赖哪些部分？ no_interaction

Q6. 是否可以把 non-additive scorer 正式固定为高层架构？ 可以进入下一阶段把它固定下来并继续用 development data 训练，但现在仍不能部署。

`planner_ready = False`

`deployment_verified = False`
