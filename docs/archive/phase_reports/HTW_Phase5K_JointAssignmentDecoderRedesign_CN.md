# HTW Phase-5K Joint Assignment Decoder Redesign

**planner_ready = False**

**deployment_verified = False**

本阶段只在冻结的 HGNN encoder 上训练两个新的 decoder 头。没有修改 Residual MAPPO、低层策略、reward、B2、τ、Gate-1、Candidate-C 或 planner，没有调用 deployment `decide()`，没有使用 230000–230127。

## 1. Phase-5J 的问题

Phase-5J 的问题在高层 decoder 的分配耦合。encoder、低层 MAPPO 和 reward 不是这一步要改的对象。

在联合 MAPPO 有差别的时刻，单个效用和的 Spearman 是 0.3673，联合效用 J(A) 的 Spearman 是 0.7492。现有 OrderFreeSetDecoder 按 UAV 分别输出浮标 logit。冻结解码在 310100 和 310101 上的平均联合 MAPPO 是 0，尽管冲突率也是 0。

## 2. Decoder

Encoder 输入保持不变。UAV 侧是位置、速度、服务进度、是否已分配、已分配队列，能量通道仍为 0。Buoy 侧是位置、队列、已观测到达、服务进度、占用，以及现有的 urgency 通道。UAV-Buoy 边保留距离、当前速率和 ETA。UAV-UAV 边保留距离和干扰。没有加入未来轨迹。J(A) 只作为监督标签，不进入 encoder。

Global assignment head 用 H_UAV 和 H_Buoy 的拼接得到 M×K 配对分数，再做带虚拟行的 Sinkhorn。得到的矩阵 P 满足行和为 1，列质量至多为 1。读出时每次取整张矩阵里剩余的最大元素，并同时去掉该行和该列。不按 UAV 单独取最大列。

Matching decoder 使用同一类配对分数。训练时分配分数是被选中配对的分数和。读出使用 Hungarian matching，直接给出 one-to-one assignment。

OrderFreeSetDecoder 保持 Phase-5F 检查点，不更新参数。它的排序分数是空白分配上各 UAV logit 的和，用来表示逐 UAV 打分。

## 3. Joint loss

监督标签是 Phase-5J 的 J(A)，不是 assignment similarity。

对所有 J(A) > J(B) 的分配对，损失是 `-log sigmoid(s(A) - s(B))`。J 只决定哪一个分配必须得到更高分数。直接把 `-log sigmoid(J(A) - J(B))` 当作损失不会产生 decoder 梯度。

Global head 的 s(A) 是约束后 P 上被选配对的对数概率和。Matching decoder 的 s(A) 是配对分数和。训练场景是 310100 的 t=0 和 t=15。验证只用 310100 的 t=30 做早停，它和训练是同一 scenario 的稍后时刻。测试是 310101 的 t=0、15、30，不参与训练和选模。

Global head 选中的验证轮次是 5，验证 Spearman 是 0.4286。Matching decoder 选中的验证轮次是 5，验证 Spearman 是 0.4286。

## 4. 实验结果

下表的决策数字来自测试场景 310101。individual greedy 和 pool J(A) optimum 是参照，不是新的 decoder。duplicate target rate 是重复选择次数除以无人机数。conflict rate 是分配里处于重复浮标上的无人机比例。

### 测试场景 310101

| model | joint MAPPO | duplicate target rate | conflict rate | Spearman vs J(A) | Top-1 |
| --- | --- | --- | --- | --- | --- |
| OrderFreeSetDecoder | 0.0000e+00 | 0.0000 | 0.0000 | 0.0845 | 0.3333 |
| Global assignment head | 0.0000e+00 | 0.0000 | 0.0000 | -0.4845 | 0.3333 |
| Matching decoder | 3.3333e+05 | 0.0000 | 0.0000 | 0.0803 | 1.0000 |
| individual greedy | 6.6667e+05 | 0.0000 | 0.0000 | 0.6672 | 0.6667 |
| pool J(A) optimum | 2.0000e+06 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

### 训练场景 310100

t=0 和 t=15 参与了 decoder 训练。t=30 只用于早停。

| model | joint MAPPO | duplicate target rate | conflict rate | Spearman vs J(A) | Top-1 |
| --- | --- | --- | --- | --- | --- |
| OrderFreeSetDecoder | 0.0000e+00 | 0.0000 | 0.0000 | 0.1556 | 0.5000 |
| Global assignment head | 0.0000e+00 | 0.0000 | 0.0000 | -0.4144 | 0.5000 |
| Matching decoder | 0.0000e+00 | 0.0000 | 0.0000 | -0.0308 | 0.5000 |
| individual greedy | 5.0000e+05 | 0.0000 | 0.0000 | 0.5126 | 0.5000 |
| pool J(A) optimum | 1.5000e+06 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

### 验证时刻 310100 t=30

| model | joint MAPPO | duplicate target rate | conflict rate | Spearman vs J(A) | Top-1 |
| --- | --- | --- | --- | --- | --- |
| OrderFreeSetDecoder | 0.0000e+00 | 0.0000 | 0.0000 | 0.4286 | 1.0000 |
| Global assignment head | 0.0000e+00 | 0.0000 | 0.0000 | 0.4286 | 1.0000 |
| Matching decoder | 0.0000e+00 | 0.0000 | 0.0000 | 0.4286 | 1.0000 |
| individual greedy | 0.0000e+00 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| pool J(A) optimum | 0.0000e+00 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

### 各时刻联合 MAPPO

| scenario | t | split | baseline | global head | matching | individual greedy | pool J(A) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 310100 | 0 | train | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000e+06 | 1.0000e+06 |
| 310100 | 15 | train | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 2.0000e+06 |
| 310100 | 30 | val | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 310101 | 0 | test | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 310101 | 15 | test | 0.0000e+00 | 0.0000e+00 | 1.0000e+06 | 1.0000e+06 | 5.0000e+06 |
| 310101 | 30 | test | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000e+06 | 1.0000e+06 |

Top-1 表示模型在该时刻分配池内分数最高的分配，是否就是 J(A) 最高的分配。联合 MAPPO 是读出分配交给冻结 MAPPO 执行 15 步的 payload。目标是多机联合 payload，不是单目标 payload。

Matching decoder 在测试池内的 Top-1 是 1.0000，Spearman 是 0.0803。Top-1 只比较池内分配。Hungarian 读出可以选到池外浮标，因此联合 MAPPO 均值是 3.3333e+05，低于 individual greedy 的 6.6667e+05，也低于池内 J(A) 最优的 2.0000e+06。310101 的 t=15 上，池内 J(A) 最优分配是 [6, 10, 2, 7, 9, 8]，联合 MAPPO 是 5.0000e+06；Matching decoder 读出的分配是 [2, 4, 3, 5, 0, 1]，联合 MAPPO 是 1.0000e+06。Global assignment head 的测试 Spearman 是 -0.4845，联合 MAPPO 是 0.0000e+00。三种 decoder 读出的 duplicate target rate 和 conflict rate 都是 0。去掉重复目标没有把多机联合 payload 提升到 J(A)。

## 5. 是否重新训练整个 HGNN

在 310101 上，matching decoder 的联合 MAPPO 高于冻结 OrderFreeSetDecoder，但没有高于单个效用的排他匹配。本阶段不进入整个 HGNN 的重新训练。

本阶段不进入 deployment verification。

`planner_ready = False`

`deployment_verified = False`
