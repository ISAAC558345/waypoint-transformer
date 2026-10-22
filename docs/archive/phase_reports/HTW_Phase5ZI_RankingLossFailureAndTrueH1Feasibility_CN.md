# Phase-5ZI：排序损失失效与真实 H1 可行性审计

门控：**ZI-D（TRUE_H1_COMPUTATION_PRIORITY）**。

本阶段没有训练，没有改 λ，没有新的环境 rollout，也没有打开 `310118–310125`。下一阶段应优先审计真实 H1 的计算，而不是再训一个 Response Predictor。

```text
planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False
```

真实 H1 仍然不是已经验证过的在线 planner。部署预算没有测量。

## 直接回答

1. Ranking loss 的实现是接通的。它写进了同一次 `backward`，梯度能从冻结 Model-R1 的 response 输入回到 Predictor，R1 自身参数梯度为 0。Pair 只来自同一个 state，teacher 高分在前。\(T_v=1.0\) 与 value 标准差 0.650 处于同一量级。

2. 训练里的 \(L_{\mathrm{rank}}\) 下降，是因为每个 state 有大约 600 个 pair，损失是它们的平均。确认集上，Top-5 对长尾的损失占 69.3%，Teacher Top-10 内部只占 16.9%。平均损失下降不要求那一小部分决定名次的 pair 变好。确认集的 value Spearman 只在第 10 个 epoch 测过一次，从 P1 的 0.7343 降到 ZH-Rank 的 0.7290。第 1 到第 9 个 epoch 的排序指标没有 checkpoint，不能补画，也不能补训。

3. 真实训练样本上的 \(G_{\mathrm{resp}}\) 与 \(G_{\mathrm{rank}}\) 余弦没有测量，标记为 NOT_MEASURED。训练状态没有序列化，本阶段不允许重新 rollout。固定诊断张量上的余弦是 −0.006，这只说明随机目标下两路梯度近乎正交，不能当成训练冲突。sigmoid 饱和也不成立：确认集 pair 里 sigmoid 权重低于 0.05 的比例是 0.97%，高于 0.95 的比例是 0。

4. Teacher value 与 J15 是对齐的。120 个确认 state-seed 上，Spearman 平均 0.8008，Teacher Top-1 命中量化 J15 最优的比例是 0.875。平局 pair 比例 0.177%。Top-1 margin 平均 0.227，Top-3 margin 平均 0.078。这不是“教师分不出候选”。

5. `310114 t=15` 仍失败，是因为 ZH-Rank 把 A_opt 的 SINR 和 rate 都估得比 A_pred 更低，value margin 比 P1 更负。这个 state 属于确认集，没有进入训练损失。五个 seed 的 A_opt 名次是 70、72、64、70、63。

6. 在已经锁定的门槛下，Predictor 路线没有一条已经证实、可以直接再训练的修复。实现没有断，教师与 J15 对齐，大间隔 pair 的损失占比是 55.6%，没有达到 80% 的 ZI-B 线。P1 和 ZH-Rank 的 selected J15 都低于 S。

7. 已记录的 515.26 ms 里，环境一步占 294.85 ms（57.2%），`executor.act` 占 138.51 ms（26.9%），response 组装占 58.86 ms（11.4%），环境复制占 17.37 ms（3.4%）。Encoder 只算一次，2.14 ms。纯 MAPPO 前向没有从 `act` 里拆出来，标记为 NOT_MEASURED。

8. 候选池里 120 个 assignment 全部唯一，重复 assignment 的 state 数是 0。Encoder 已经复用。环境一步依赖联合动作之后的 SINR，不能把不同 assignment 的干扰当成独立。因此每个 state 仍要 120 次环境执行。把 120 次 `act` 改成批量前向在数学上可以不改变动作，但加速比没有测过。

9. 下一阶段应优先做真实 H1 的计算优化。继续用当前蒸馏目标训练 Predictor 没有新的已验证依据。

## 同时记下、但没有单独改写门控的事实

Top-10 内部 pair 只贡献 16.9% 的 score-space 损失，低于 0.20。ZI-B 还要求“teacher 间隔大于一个 payload block”的损失占比至少 80%。实测是 55.6%，所以 ZI-B 不成立。这些 Top-10 pair 的平均 sigmoid 权重是 0.429，长尾是 0.294，高价值 pair 不是没有梯度。

真实训练 batch 的参数梯度冲突，以及逐 epoch 的排序曲线，记录不足。它们被记在门控的 `mechanisms_recorded` 里，没有被写成另一个字母。可操作的阻塞是：两个 Predictor 都没有保住 R1−S 的 +4.5000e5 bit，而真实 H1 已经有这个收益。

## 任务 A：梯度是否接通

Phase-5ZH 的训练步是 \(L_{\mathrm{resp}}+0.2\,L_{\mathrm{rank}}\) 之后一次 `backward`。Value 网络的 `forward` 对 response 没有 `detach`，ranking 那段也不在 `no_grad` 里。R1 参数被 `requires_grad_(False)` 挡住，所以参数不更新，输入端仍能回传。

保存的训练日志里，五个 seed 第 1 个 epoch 的 rank loss 约 0.56–0.59，clip 前总梯度范数约 1.0–1.4。第 10 个 epoch 的 rank loss 是 0.314–0.337，clip 前范数约 0.72–0.74。总梯度没有变成 0。日志里没有把 \(G_{\mathrm{resp}}\) 和 \(G_{\mathrm{rank}}\) 分开。

固定诊断张量、不执行参数更新，加载的是冻结的 R1 seed 0 和 ZH-Rank seed 0：

| 量 | 值 |
| --- | ---: |
| rank 梯度 L2 | 0.271 |
| response 梯度 L2 | 0.467 |
| \(0.2\|G_{\mathrm{rank}}\|/\|G_{\mathrm{resp}}\|\) | 0.116 |
| response 输入收到梯度 | 是 |
| R1 参数梯度 | 0 |
| SINR 头上的 rank 梯度 L2 | 0.0286 |
| rate 头上的 rank 梯度 L2 | 0.0412 |
| pair 下标越出该 state | 否 |
| teacher 高分在前 | 是 |

诊断张量的余弦和通道范数只证明计算图接通、两路梯度量级可比。它们不是 310100–310109 上的训练冲突。

Pair 规则在每个 state 内生成：Teacher Top-5 对 Top-5 之外、Top-10 内部、以及固定种子的 64 个随机 pair。平局阈值 \(10^{-3}\)。确认集上平局 pair 占 0.177%，有效非平局 pair 占 99.8%。sigmoid 没有成片饱和。Top-10 内部的平均 sigmoid 权重 0.429，高于长尾的 0.294，所以高价值 pair 在分数空间里有梯度。它们不是损失的主体。

## 任务 C：Teacher 是否代表 J15

确认集已保存的 `v_r1` 与 J15，每个 state 120 个候选，五个 seed：

| 指标 | 值 |
| --- | ---: |
| Spearman | 0.8008 |
| Top-1 命中量化 J15 最优 | 0.875 |
| value 标准差 | 0.650 |
| Top-1 margin | 0.227 |
| Top-3 margin | 0.078 |
| pair 平局比例 | 0.177% |

\(T_v=1.0\) 落在预先锁定的尺度区间 [0.1, 10] 内。Teacher 能区分候选，也能对上 J15。失败点不在教师标签。

## 任务 D：损失下降和排序

可绘制的只有训练损失。五个 seed 的 response loss 从约 0.56 降到约 0.122，rank loss 从约 0.57 降到约 0.32。checkpoint 只有 `phase5zh_rank_seed0.pt` 到 `seed4.pt`，epoch 字段都是 10。第 1–9 轮的 value Spearman、pairwise、Top-1、Top-3、selected J15 在 `phase5zi_loss_vs_ranking.csv` 里记为 MISSING_CHECKPOINT。

第 10 轮在确认集上的 ZH-Rank：value Spearman 0.7290，pairwise 0.8023，Top-1 0.5250，Top-3 0.8333，selected J15 2.2667e6 bit。对照 P1 是 0.7343、0.8076、0.5833、0.8250、2.3083e6 bit。没有按这条曲线重选 checkpoint。

## 310114 t=15 与其余 residual cases

A_opt 的真实 J15 是 5e6 bit，A_pred 是 4e6 bit。真实已分配 SINR 是 14.815 对 12.174，rate 是 3.977e6 对 3.417e6。R1 的 value margin 在五个 seed 上是 +0.109、+0.012、+0.00085、+0.058、+0.062。seed 2 的 margin 低于 \(10^{-3}\)，按现有平局规则这一对不会被强制排序；另外四个 seed 会进入 Top-10 内部 pair。

P1 的 margin 约 −1.48 到 −1.58。ZH-Rank 的 margin 约 −1.93 到 −2.36，A_opt 名次中位数 70，A_pred 在五个 seed 上都是第 1 名。seed 0 的 ZH-Rank 把 A_opt 的 SINR 预测成 3.31（误差 −11.5），A_pred 预测成 9.22（误差 −2.95）；rate 分别是 1.665e6（误差 −2.31e6）和 3.878e6（误差 +4.62e5）。相对顺序与真实相反，而且比 P1 的 margin 更负。这个 state 不在 310100–310109 的训练损失里。

其余 9 个已登记 residual case 与 `310114 t=15` 一起，仍是修复 4、不变 6、恶化 0。两个修复来自同一个 `310111 t=15`。`310113 t=15` 的 selected J15 从 1.8e6 升到 3.0e6，optimum 平均名次仍是 62.6。`310116 t=15` 的 selected J15 不变，optimum 名次从 22.0 移到 25.0。`310114 t=15` 的 value-margin 误差从 −1.58 变成 −2.20。

## 真实 H1 的 515 ms

数字来自 Phase-5ZE 已保存的 100 次计时，池宽 120。本阶段没有重跑。

| 部分 | 毫秒 | 占 515.26 ms |
| --- | ---: | ---: |
| 环境一步 | 294.85 | 57.2% |
| executor.act | 138.51 | 26.9% |
| response 组装 | 58.86 | 11.4% |
| 环境复制 | 17.37 | 3.4% |
| 冻结 value | 3.52 | 0.7% |
| encoder，一次 | 2.14 | 0.4% |

`executor.act` 含 MAPPO 前向、residual lock 和安全投影。单独的网络前向是 NOT_MEASURED。

预测路径的 118.38 ms 里，120 次串行 `act` 占 109.10 ms，Predictor 本身 3.58 ms，value 3.40 ms。预测路径省下的是环境复制、环境一步和 response 组装，不是 Predictor 比 MAPPO 更轻。Phase-5ZH 另一次 20 次重复测到的约 4.6 ms 只覆盖 Predictor 加 value，没有把这 109 ms 的 `act` 算进去。

候选数只在 120 上测过。代码对每个候选循环一次，其他池宽的时间是 NOT_MEASURED。批量 `act` 的加速比也是 NOT_MEASURED。

## 两条路线

|  | 路线 A：继续 Predictor | 路线 B：真实 H1 加计算优化 |
| --- | --- | --- |
| 已有收益 | P1 比 S 低 3.33e4 bit，ZH-Rank 低 7.50e4 bit | R1 比 S 高 4.50e5 bit |
| 是否还要为当前损失再训练 | 没有已验证的修复点 | 不需要新训练 |
| 高层 encoder | 保持 | 保持 |
| MAPPO | 保持 | 保持 |
| 排序误差 | P1 与 ZH-Rank 都还在 | 真实 response 下 R1 的确认集 Top-1 是 0.875 |
| 计算 | 预测全路径约 118 ms，其中 109 ms 仍是串行 `act` | 全路径 515 ms，主体是环境一步 |
| 正式在线 planner | 否 | 否 |

ZI-D 的含义是下一阶段先处理真实 H1 的计算，不继续盲训 Predictor。它不表示 515 ms 已经满足部署预算，也不表示可以把真实 H1 直接接到正式 planner。
