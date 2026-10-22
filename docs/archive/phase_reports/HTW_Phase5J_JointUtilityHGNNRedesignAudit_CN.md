# HTW Phase-5J Joint Utility HGNN Redesign Audit

**planner_ready = False**

**deployment_verified = False**

本阶段只核对联合分配标签和三种读出方式。没有训练 deployment 模型，没有修改 Residual MAPPO，没有修改 B2、τ、Gate-1、Candidate-C、planner 或 deployment `decide()`，没有使用 230000–230127。

## 1. Individual utility 的局限

单个效用最高的组合不是联合 MAPPO 最高的组合。标签在每个决策时刻的排他分配池内排序，对照量是该分配的联合 MAPPO payload。池来自各无人机边际 fly-toward 效用的 top-3，只保留互不重复的分配；完整的 P(12,6) 没有穷尽。

在联合 MAPPO 极差至少一个服务块的 5 个时刻上，单个效用和的 Spearman 是 0.3673，Top-1 是 0.2000，Top-3 是 0.4000。Kendall 只有 0.0311，名次经常和联合 MAPPO 不一致。

不做排他约束、直接取各无人机最高分时，平均重复目标是 2.6667，conflict rate 是 0.5556。这个分配的平均联合 MAPPO payload 是 5.0000e5。按 J(A) 选出的排他分配没有重复目标，平均联合 MAPPO payload 是 2.3333e6。

数据是已有重置 310100 和 310101，决策时刻 t=0、15、30。没有新的 scenario。310100 曾用于 utility HGNN 训练，310101 是它的测试场景；这里只评标签，不更新检查点。两个场景的 Top-1 模式相同：t=0 联合标签和单个效用都没有命中，t=15 只有联合标签命中，t=30 两者都命中。

| scenario | t | pool | joint Top-1 | individual Top-1 |
| --- | --- | --- | --- | --- |
| 310100 | 0 | 43 | 0 | 0 |
| 310100 | 15 | 24 | 1 | 0 |
| 310100 | 30 | 6 | 1 | 1 |
| 310101 | 0 | 43 | 0 | 0 |
| 310101 | 15 | 39 | 1 | 0 |
| 310101 | 30 | 16 | 1 | 1 |

## 2. Joint utility

`J(A) = sum_i payload_gain(i, a_i) - lambda * conflict_loss(A)`。

lambda = 1.0，不拟合。conflict_loss 的标量是通信收益下降：`max(0, sum_i payload_gain(i, a_i) - joint fly payload)`，单位是 bit。边际和大于联合 payload 时，J(A) 等于联合 fly-toward payload。这个量只作为高层标签，不写入低层 reward。

另外三项只作诊断，不换算后加进这个标量，避免和通信下降重复计算：同一目标的重复架数、conflict rate、联合飞行时的干扰矩阵和。

| label | subset | Spearman | Kendall | Top-1 | Top-3 |
| --- | --- | --- | --- | --- | --- |
| A assignment similarity | all | 0.3546 | 0.2728 | 0.5000 | 0.6667 |
| A assignment similarity | payload_separated | 0.5112 | 0.3274 | 0.4000 | 0.6000 |
| B individual trajectory utility | all | 0.4727 | 0.0259 | 0.3333 | 0.5000 |
| B individual trajectory utility | payload_separated | 0.3673 | 0.0311 | 0.2000 | 0.4000 |
| C joint trajectory utility | all | 0.7910 | 0.2149 | 0.6667 | 0.6667 |
| C joint trajectory utility | payload_separated | 0.7492 | 0.2579 | 0.6000 | 0.6000 |

Spearman、Kendall、Top-1、Top-3 都在每个决策时刻的分配池内部计算，再对时刻取平均。payload_separated 只保留联合 MAPPO 极差至少一个服务块的时刻，这里是 5 个时刻。联合标签的 Spearman 0.7492 和 Top-1 0.6000 都高于 assignment similarity 与单个效用。它的 Top-3 也是 0.6000，未命中的时刻里，联合 MAPPO 最高的分配也不在联合分数的前三名。

## 3. Decoder

HGNN encoder 保持冻结。三种读出都没有新的训练参数，也没有训练 deployment 模型。

OrderFreeSetDecoder 是检查点里的联合解码。oracle global J(A) 是在穷尽池里直接选取测量到的 J(A) 最大分配，用来表示 global assignment head 若完美预测 J(A) 能到达的离线上界，不是一个已经训练的头。exclusive matching 是在单个效用上做排他贪心，禁止两架飞机使用同一浮标。

| readout | joint fly | joint MAPPO | duplicate targets | conflict rate |
| --- | --- | --- | --- | --- |
| OrderFreeSetDecoder | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle global J(A) | 833333.3333 | 2333333.3333 | 0.0000 | 0.0000 |
| exclusive matching | 333333.3333 | 500000.0000 | 0.0000 | 0.0000 |
| raw individual argmax | 333333.3333 | 500000.0000 | 2.6667 | 0.5556 |

冻结 OrderFreeSetDecoder 的平均联合 fly-toward 和联合 MAPPO 都是 0，冲突率也是 0。它给出了排他分配，但这组目标在 15 步内没有形成 payload。排他匹配去掉了重复目标，联合 MAPPO 均值为 5.0000e5。按 J(A) 选取的分配联合 MAPPO 均值是 2.3333e6。

## 4. Global assignment head

不增加 global assignment head，本阶段也没有训练这个头。

排他匹配已经去掉重复目标，联合 MAPPO 仍只有 5.0000e5。按测得 J(A) 选取能到 2.3333e6。这个差距来自联合标签，不是来自一个尚未训练的新模块。若以后授权重新训练，监督对象仍是现有 HGNN encoder 和 OrderFreeSetDecoder。

## 5. Conflict-aware assignment

原始最高分的冲突率是 0.5556，重复目标 2.6667。排他匹配的冲突率是 0.0000。按 J(A) 选出的分配冲突率是 0.0000，重复目标是 0.0000。

联合标签因此去掉了原始 argmax 上的重复服务。排他匹配也能把冲突率降到 0，但联合 MAPPO 仍低于 J(A)。冲突清零本身不等于联合最优。

池内分配的平均通信下降是 2.2757e6 bit，平均干扰是 0.5650。干扰只作诊断，没有加进 J(A) 的标量，也没有写入低层 reward。

## 6. 是否重新训练 HGNN

在 payload 有差别的时刻，联合标签 Top-1 是 0.6000，单个效用是 0.2000，assignment similarity 是 0.4000。这是以后用 J(A) 重新监督现有 HGNN 的依据。

Top-1 仍是 0.6000。310100 和 310101 在 t=0 都没有命中，而且未命中时正确分配不在联合分数的前三名。本阶段不开始 HGNN 重新训练，不增加 global assignment head，不进入 deployment verification。

`planner_ready = False`

`deployment_verified = False`
