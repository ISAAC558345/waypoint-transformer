# Phase-5ZH：状态条件价值排序响应蒸馏

门控：**RANK-C（RANKING_OBJECTIVE_FAILED）**。

本阶段只改了 Response Predictor 的训练损失，新增项为冻结的 \(\lambda=0.2\) 与 \(T_v=1.0\)。网络、数据、优化器和归一化与 Phase-5ZE 相同。确认集 `310110–310117` 只做了一次配对开发评估，没有用来选择 checkpoint。`310118–310125` 没有打开。

```text
planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False
```

当前损失路线停止。没有进入独立 fresh confirmation，也没有进入 deployment verification。

## 直接回答

Q1. 状态内 assignment value 排序没有改善。相对冻结 Model-R1 的 teacher value，ZH-Rank 的 statewise Spearman 从 P1 的 0.7343 降到 0.7290，Kendall 从 0.6139 降到 0.6035，pairwise accuracy 从 0.8076 降到 0.8023。

Q2. SINR 与 rate 的状态内排序都略降。SINR Spearman 0.6507 → 0.6466，pairwise 0.7563 → 0.7523。rate Spearman 0.6774 → 0.6737，pairwise 0.7737 → 0.7697。降幅都在 0.004 左右。

Q3. Response MAE 有小幅恶化，没有触及 3 倍崩溃线。14 个通道的平均 MAE 从 2.300e4 升到 2.629e4。`sinr_1` MAE 从 1.083 升到 1.192，比值 1.101。`rate_1` MAE 从 3.220e5 升到 3.681e5，比值 1.143。`rate_1` 的 \(R^2\) 从 0.623 降到 0.541。

Q4. 没有看到用极端 response 钻冻结价值网络空子的迹象。确认集上 \(|z|>8\) 的连续通道比例从 P1 的 2.64e-4 升到 4.04e-4，仍低于 0.01。最小通道标准差从 0.090 升到 0.102。预测 SINR 与预测 rate 的 statewise Spearman 为 0.9869，P1 为 0.9900，真实 response 为 0.9781。score collapse 比例为 0。没有新增 clipping，也没有改 normalization。

Q5. ZH-Rank 的 mean selected J15 是 2.2667e6 bit，P1 是 2.3083e6 bit，差值 −4.1667e4 bit。两者 median 都是 3.0000e6 bit。scenario-cluster 95% CI 为 [−2.2500e5, +1.5000e5]，跨过 0。normalized regret 从 0.2458 升到 0.2597。Top-1 从 0.5833 降到 0.5250，Top-3 从 0.8250 升到 0.8333，Top-5 保持 0.9583。与 R1 assignment 的一致率从 0.3500 降到 0.2250。

Q6. Retention = \((J_{ZH}-J_S)/(J_{R1}-J_S) = -0.1667\)。S 为 2.3417e6 bit，R1 为 2.7917e6 bit，R1−S 仍是 +4.5000e5 bit。ZH-Rank 位于 S 下方 7.5000e4 bit，没有保住真实 R1 response 的收益。P1 的 retention 是 −0.0741。

Q7. 五个 seed 和八个 scenario 的方向不一致。seed 1 的平均差为 +8.333e4 bit；seed 0 与 seed 4 为 0；seed 2 为 −2.083e5 bit；seed 3 为 −8.333e4 bit。只有 1/5 个 seed 的平均收益上升。scenario 中 3 个为正（310111、310113、310115），2 个持平（310112、310114），3 个为负（310110、310116、310117）。

Q8. `310114 t=15` 的 A_opt 在 ZH-Rank 五个 seed 上的名次是 70、72、64、70、63，中位数 70。五个 seed 都没有回到 Top-3。P1 中位数是 74，S 是 53，R1 是 2。A_pred 在五个 ZH-Rank seed 上都是第 1 名。五个 seed 的 value margin 仍然反号，而且比 P1 更负。

Q9. 预先登记的 10 个 residual cases：修复 4，不变 6，恶化 0。其中两个“修复”来自同一个状态 `310111 t=15` 的两个不同登记 assignment，所以按状态去重后，selected J15 上升的状态是 3 个。optimum rank 没有整体回到高位。

Q10. 没有足够证据进入下一阶段独立确认。门控是 RANK-C，本损失路线停止。`310118–310125` 继续关闭。

## 对照与对齐

训练前记录的唯一配置差异是损失。Baseline P1 使用 Phase-5ZE 已冻结的 epoch-10 checkpoint，没有重训，也没有重选轮次。ZH-Rank 使用同一 `OneStepExecutorResponsePredictor`、同一 310100–310109 训练池、同一 Adam \(10^{-3}\)、weight decay \(10^{-4}\)、grad clip 1.0、10 个 epoch、五个 seed。checkpoint 固定为最后一个 epoch。

数据 lineage 通过：token 归一化差、五个 seed 的输入归一化差、response 均值差、调用方指纹和 parity gap 都是 0。response schema 沿用 Phase-5ZD。重算的 S、P1、R1 与 Phase-5ZE 保存分数的最大绝对差都是 0。

| 模型 | mean J15 | median J15 | Δ vs P1 | Δ vs S | retention | nreg | Top-1 | Top-3 | Top-5 | 与 R1 一致 | value Spearman | value pairwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S | 2.3417e6 | 3.0000e6 | +3.333e4 | 0 | 0 | 0.1958 | 0.6917 | 0.8417 | 0.9083 | 0.5250 | 0.7465 | 0.8250 |
| P1 | 2.3083e6 | 3.0000e6 | 0 | −3.333e4 | −0.0741 | 0.2458 | 0.5833 | 0.8250 | 0.9583 | 0.3500 | 0.7343 | 0.8076 |
| ZH-Rank | 2.2667e6 | 3.0000e6 | −4.167e4 | −7.500e4 | −0.1667 | 0.2597 | 0.5250 | 0.8333 | 0.9583 | 0.2250 | 0.7290 | 0.8023 |
| R1 | 2.7917e6 | 3.0000e6 | +4.833e5 | +4.500e5 | 1 | 0.0625 | 0.8750 | 0.9583 | 0.9583 | 1 | 1 | 1 |

S、P1、R1 的 mean selected J15 与已锁定结果一致。R1 在这里只是教师对照。

训练集上的 rank loss 从第 1 个 epoch 的约 0.57 降到第 10 个 epoch 的 0.314–0.337，response loss 降到约 0.122。这个训练下降没有变成确认集上的 teacher-value 排序提升。

## 310114 t=15

A_pred = `2;5;-1;4;-1;-1`，真实 J15 = 4e6 bit。A_opt = `-1;4;2;5;-1;-1`，真实 J15 = 5e6 bit。

| seed | S opt / pred | P1 opt / pred | ZH opt / pred | R1 opt / pred | ZH margin | 反号 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 0 | 53 / 2 | 74 / 1 | 70 / 1 | 2 / 4 | −2.161 | 是 |
| 1 | 54 / 2 | 72 / 1 | 72 / 1 | 3 / 4 | −1.930 | 是 |
| 2 | 53 / 2 | 76 / 1 | 64 / 1 | 3 / 4 | −1.998 | 是 |
| 3 | 54 / 2 | 85 / 1 | 70 / 1 | 2 / 4 | −2.312 | 是 |
| 4 | 52 / 3 | 74 / 1 | 63 / 1 | 2 / 4 | −2.358 | 是 |

五个 seed 的真实已分配均值相同：A_pred 的 SINR 是 12.174，rate 是 3.417e6 bit；A_opt 的 SINR 是 14.815，rate 是 3.977e6 bit。真实顺序是 A_opt 更高。

seed 0 的预测已分配均值：

| 模型 | A_pred SINR | A_opt SINR | A_pred rate | A_opt rate |
| --- | ---: | ---: | ---: | ---: |
| P1 | 5.577 | 2.049 | 2.508e6 | 1.183e6 |
| ZH-Rank | 9.223 | 3.313 | 3.878e6 | 1.665e6 |

ZH-Rank 把 A_pred 的 SINR 和 rate 都预测得高于 A_opt，顺序与真实相反。五个 seed 都是这个方向。R1 的 margin 在五个 seed 上都是正的（seed 0 为 +0.1093）。ZH-Rank 的 margin 更负，A_opt 的名次只比 P1 略靠前，仍然远离 Top-3。

## Residual cases

判定按五个 seed 的 mean selected J15，差值不超过 1 bit 记为不变。

| scenario | t | 登记的 predicted assignment | 方向 | P1 J15 | ZH J15 | P1 regret | ZH regret | P1 opt rank | ZH opt rank | P1 ordering error | ZH ordering error | P1 margin error | ZH margin error |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 310110 | 0 | `4;-1;-1;-1;3;5` | 不变 | 1.0e6 | 1.0e6 | 0.667 | 0.667 | 3.0 | 3.0 | 0.104 | 0.102 | −0.442 | −0.694 |
| 310111 | 15 | `-1;4;-1;-1;5;3` | 修复 | 0.6e6 | 1.2e6 | 0.800 | 0.600 | 71.4 | 75.0 | 0.588 | 0.569 | −0.542 | −0.400 |
| 310111 | 15 | `3;4;-1;-1;-1;5` | 修复 | 0.6e6 | 1.2e6 | 0.800 | 0.600 | 71.4 | 75.0 | 0.588 | 0.569 | −1.641 | −2.015 |
| 310111 | 30 | `-1;4;5;3;-1;-1` | 不变 | 1.0e6 | 1.0e6 | 0.667 | 0.667 | 2.0 | 2.0 | 0.242 | 0.223 | −0.224 | −0.131 |
| 310113 | 15 | `4;-1;5;-1;-1;3` | 修复 | 1.8e6 | 3.0e6 | 0.400 | 0.000 | 70.8 | 62.6 | 0.489 | 0.468 | −0.644 | −0.554 |
| 310114 | 0 | `-1;4;-1;3;5;-1` | 不变 | 2.0e6 | 2.0e6 | 0.333 | 0.333 | 2.8 | 2.6 | 0.133 | 0.138 | −0.683 | −1.053 |
| 310114 | 15 | `2;5;-1;4;-1;-1` | 不变 | 4.0e6 | 4.0e6 | 0.200 | 0.200 | 76.2 | 67.8 | 0.382 | 0.394 | −1.581 | −2.200 |
| 310115 | 15 | `4;-1;-1;3;-1;5` | 修复 | 2.2e6 | 2.6e6 | 0.267 | 0.133 | 1.4 | 1.2 | 0.452 | 0.470 | −0.046 | +0.006 |
| 310116 | 15 | `5;4;-1;3;-1;-1` | 不变 | 2.0e6 | 2.0e6 | 0.333 | 0.333 | 22.0 | 25.0 | 0.420 | 0.405 | −0.399 | −0.029 |
| 310117 | 0 | `4;3;-1;-1;5;-1` | 不变 | 1.0e6 | 1.0e6 | 0.500 | 0.500 | 4.0 | 4.0 | 0.128 | 0.127 | +0.176 | +0.131 |

`310111 t=15` 的两行是同一状态上的两个登记 assignment，selected J15 相同。`310113 t=15` 的 mean J15 升到 3.0e6 bit，但 optimum 平均名次仍是 62.6。`310115 t=15` 的 optimum 平均名次从 1.4 到 1.2。其余案例的 selected J15 没有变化；`310116 t=15` 的 optimum 名次从 22.0 移到 25.0。

## 推理时间

在一个确认状态上，预热 2 次后重复 20 次，测的是 predictor 前向加上冻结 Model-R1 打分：P1 4.91 ms，ZH-Rank 4.56 ms，S 3.52 ms。ZH-Rank 没有增加一层网络，耗时与 P1 同量级。

## 门控判定

RANK-A 需要平均 J15 提升、CI 下界大于 0、至少 4/5 seed 改善、retention 至少 0.70、value Spearman 与 pairwise 同时上升，并且物理检查通过。这里平均差为负，CI 下界为 −2.2500e5，上升 seed 只有 1 个，retention 为 −0.1667，两条 value 排序指标都下降。物理检查通过，所以这不是数值崩溃。CI 上界是 +1.5000e5，因此也没有把点估计的下降写成已被区间确认的退化。排序和 selected J15 都没有可靠提升，门控为 RANK-C。

产物在 `src/wt/lowlevel_v2/phase5zh_ranking_response_distillation/`。新的 predictor checkpoint 只写在该目录的 `checkpoints/` 下，没有覆盖 Phase-5ZE 或 Phase-5ZD。
