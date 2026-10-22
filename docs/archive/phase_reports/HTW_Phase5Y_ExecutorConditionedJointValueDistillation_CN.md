# HTW Phase-5Y Executor-Conditioned Joint Value Distillation

**planner_ready = False**

**deployment_verified = False**

本阶段只重新初始化 assignment-level value head。Phase-5L encoder、Phase-5U 关系特征和 Phase-5V 时间特征保持冻结。监督标签是 frozen MAPPO 真实执行 15 步后的 payload，单位是 block（J15 / 1e6）。损失是 state 内平均 Huber，delta = 1 block，再对 state 平均。Adam、学习率 1e-3、weight decay 1e-4、梯度裁剪 1.0、5 个 seed、10 个 epoch、checkpoint 固定为 epoch 10，与 Phase-5S 的训练预算相同。唯一的接口改动是把 pairwise ranking loss 换成这个 Huber。短程 rollout、未来 SINR、未来 progress 和 J15 都不是网络输入。310110–310117 只评估一次。310118–310125 没有打开。

## 1. 标签

Phase-5X horizon=15 的 shadow payload 与候选池对齐。相同 state + assignment 没有出现不同的 J15，候选池没有缺失标签，池内 assignment 都合法，shadow payload 与已经保存的 J15 相差不超过 1 bit。

## 2. 排序

confirmation 上，5 个 seed 的平均：

| model | Spearman | Top-1 | Top-3 | Top-5 | normalized regret | selected J15 |
| --- | --- | --- | --- | --- | --- | --- |
| Phase-5V | 0.6181 | 0.6417 | 0.8583 | 0.9167 | 0.1903 | 2.3917e6 |
| Score-B H=5 | 0.8081 | 0.8333 | 1.0000 | 1.0000 | 0.1042 | 2.7500e6 |
| Score-B H=8 | 0.8193 | 0.9167 | 1.0000 | 1.0000 | 0.0625 | 2.8750e6 |
| Phase-5Y | 0.5785 | 0.6833 | 0.8417 | 0.9083 | 0.2003 | 2.3417e6 |
| Phase-5Y ensemble | 0.5787 | 0.6667 | 0.8333 | 0.9167 | 0.2125 | 2.2917e6 |
| J15 oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.9583e6 |

各种子的 selected J15：seed 0 为 2.2917e6，seed 1 为 2.2917e6，seed 2 为 2.3750e6，seed 3 为 2.4583e6，seed 4 为 2.2917e6。ensemble 是诊断量，不进入 Gate。

## 3. 相对 Phase-5V

Δ selected J15 的均值是 −5.0000e4 bit。scenario-cluster bootstrap 95% CI 是 [−1.8333e5, +4.1667e4]，跨过 0。

scenario：2 个上升（310110 +1.3333e5，310116 +6.6667e4），3 个持平，3 个下降（310111 −4.6667e5，310113 −6.6667e4，310114 −6.6667e4）。

seed：seed 2 为 +4.1667e4，seed 3 为 +1.2500e5，其余 3 个 seed 为负。

Top-1 从 0.6417 升到 0.6833。Top-3 从 0.8583 降到 0.8417。normalized regret 从 0.1903 升到 0.2003。多出来的 Top-1 没有带来更高的平均 payload。

## 4. 回归与校准

confirmation：MAE 0.5481 block，RMSE 0.7992，Huber 0.2654，R² 0.4097，bias −0.1158。collapsed fraction = 0。训练 epoch 10 的 state-mean Huber 大约在 0.24 到 0.25。

| 预测 bin | 样本数 | 平均预测 | 平均真实 |
| --- | --- | --- | --- |
| 0 | 6561 | 0.0840 | 0.0983 |
| 1 | 5969 | 0.9678 | 1.2320 |
| 2 | 1712 | 1.8380 | 1.8378 |
| 3 | 158 | 2.7783 | 2.7927 |
| 4+ | 0 |  |  |

模型没有把任何候选预测到 4 block 以上。1-block 这一档的真实均值是 1.23，预测均值是 0.97。

## 5. 310114 t=15

命名 optimum `-1;4;2;5;-1;-1` 的真实 J15 是 5e6。五个 seed 的预测值是 1.099、0.754、1.495、1.414、1.039 block，名次是 53、54、54、53、52。中位名次 53。没有 seed 把它排进 Top-5。

seed 0、1、4 选中 `2;-1;5;4;-1;-1`，真实 J15 = 2e6。seed 2、3 选中 Phase-5V 的 `2;5;-1;4;-1;-1`，真实 J15 = 4e6。seed 中位 selected J15 是 2e6，低于 Phase-5V 的 4e6，也低于 H=5 和 H=8 的 5e6。

## 6. Residual 与 executor information

10 个 residual case 里，Phase-5Y 的 seed 中位选择把 regret 降到 1 个 block 以下的只有 310114 t=0 这 1 个。310114 t=15 的 regret 从 Phase-5V 的 1e6 变成 3e6。310111 t=15 的中位 selected J15 是 0。

H=8 相对 Phase-5V 有正增益的 state×seed 共 38 个。其中 68.4% 的 recovery 是 0，18.4% 接近 1，13.2% 为负。均值是 0，中位数是 0，IQR 是 0。均值是正负抵消后的结果。

J15 至少差 1 block 的 assignment 配对共 95587 对。embedding cosine ≥ 0.99 的比例是 0.4674。|ΔV| < 0.1 block 的比例大约是 0.150。两者同时成立的比例是 0.1461。value 撞在一起时，embedding cosine 的均值大约是 0.997。

## 7. 时间

120 个 assignment 的 value head 前向平均 3.50 ms。这是 frozen embedding 已经算好之后的耗时。Phase-5X 的 H=8 shadow 是 5066.7 ms。加速约 1447 倍。这个速度没有进入 Gate。

## 8. 结论

Gate 是 **Y-E**。

预先登记的确认条件没有成立：平均 selected J15 是 −5.0000e4 bit，CI 下界是 −1.8333e5，scenario 上升 2/8，seed 上升 2/5，normalized regret 是 0.2003，310114 的中位名次是 53。R² 0.4097 低于回归确认线 0.5。embedding 接近且 value 分不开的比例 0.1461 低于 0.25。因此不标记为 `FROZEN_FOR_FRESH_CONFIRMATION`。

标签完整性已经通过。失败点在决策：Huber 头拟合了常见的 0–2 block，但没有把 H=8 才能分开的高价值联合分配排到前面。下一步按 Y-E 复查高价值 assignment 的标签对齐和表示差异，不加深 HGNN，不把 ranking loss 加回去，不打开 310118–310125。

`planner_ready = False`

`deployment_verified = False`

## 9. 十一个问题

Q1. 当前 frozen HGNN representation 能不能学到 executor-conditioned joint value？能学到粗的 block 回归。confirmation MAE 是 0.5481 block，R² 是 0.4097，预测没有 collapse。它没有把 4 block 以上的联合分配预测出来，310114 的 5-block optimum 被预测成大约 1 block。

Q2. 相对 Phase-5V，selected J15 提高多少？平均下降 5.0000e4 bit。95% CI 是 [−1.8333e5, +4.1667e4]。

Q3. 相对 H=8，恢复了多少 executor information？在 H=8 确实更好的 38 个 state×seed 上，recovery 均值是 0，中位数是 0。68.4% 与 Phase-5V 的 payload 相同，18.4% 追上 H=8，13.2% 往相反方向移动。

Q4. Top-1、Top-3、normalized regret 是多少？Top-1 0.6833，Top-3 0.8417，Top-5 0.9083，normalized regret 0.2003，Spearman 0.5785。

Q5. 310114 t=15 的真实 optimum `-1;4;2;5;-1;-1` 排第几？五个 seed 的名次是 53、54、54、53、52，中位数 53。没有进入 Top-5。

Q6. 10 个 residual case 修复多少？1/10。只有 310114 t=0 的中位 regret 降到 0。310114 t=15 的中位 regret 从 1e6 增加到 3e6。

Q7. 五个 seed 是否方向一致？不一致。seed 2 和 seed 3 的平均 selected J15 高于 Phase-5V，seed 0、1、4 低于 Phase-5V。

Q8. 是否存在 representation collision？存在。J15 至少差 1 block 的配对里，46.74% 的 joint embedding cosine ≥ 0.99。这些配对里，14.61% 同时满足 |ΔV| < 0.1 block。

Q9. 主要瓶颈现在在哪里？换掉 ranking supervision 之后，决策没有改善。回归停在 R² 0.4097，高 payload 的联合分配没有被分开。Gate 是 Y-E。标签文件本身没有冲突。下一步复查这些高价值样本的表示和标签是否对齐。

Q10. 120 个 candidate 的 value inference 耗时多少？value head 平均 3.50 ms，相对 H=8 的 5066.7 ms 大约快 1447 倍。这个时间不包含每个 state 一次的 frozen encoder 前向。

Q11. 是否达到 Y-A，可以冻结结构进入 fresh confirmation？没有。`frozen_for_fresh_confirmation = False`。不打开 310118–310125，不进入 deployment verification。
