# HTW Phase-5L Joint Utility HGNN Training

**planner_ready = False**

**deployment_verified = False**

本阶段重新训练 HGNN encoder。Phase-5K 的 Matching decoder 保持冻结，没有修改 Residual MAPPO、低层策略、reward 或 planner，没有调用 deployment `decide()`，没有使用 230000–230127。

## 1. 为什么训练 encoder

Phase-5K 中 OrderFreeSetDecoder、Global assignment head 和 Matching decoder 的 duplicate target rate 与 conflict rate 都是 0。Matching decoder 能给出合法的一对一分配，但与 J(A) 的 Spearman 只有 0.0803。瓶颈是联合效用表示，不是分配耦合。

## 2. 联合表示

UAV-UAV 边保持现有的距离和干扰。UAV-Buoy 边在现有距离、速率和 ETA 之外，附上该浮标的当前队列比例。这个队列已经在浮标节点上，不使用未来轨迹。

`z_joint` 是 H_UAV 与 H_Buoy 的掩码平均，再拼上当前已被占用浮标的池化向量和原有 global 特征。它经过线性层加回每个 UAV 和每个浮标的 individual embedding，然后进入冻结的 Matching decoder。Hungarian 读出仍然是 one-to-one，decoder 参数不更新。

## 3. 训练

标签是 Phase-5J 的 J(A)，不是 assignment similarity。损失是 `-log sigmoid(s(A) - s(B))`，只在 `J(A) > J(B)` 时计算。s(A) 是冻结 Matching decoder 在联合条件嵌入上的配对分数和。

梯度只使用 310100 的 t=0、15、30。310101 的三个时刻不进入梯度，用来和 Phase-5K Matching decoder 比较。没有新的 scenario。选模使用 310100 上的 Spearman，最佳轮次是 16，训练 Spearman 是 0.2162。

## 4. 结果

### 测试场景 310101

| model | Spearman vs J(A) | Top-1 | joint MAPPO | duplicate rate | conflict rate |
| --- | --- | --- | --- | --- | --- |
| Phase-5K Matching decoder | 0.0803 | 1.0000 | 3.3333e+05 | 0.0000 | 0.0000 |
| Joint utility HGNN | 0.1194 | 1.0000 | 3.3333e+05 | 0.0000 | 0.0000 |

### 训练场景 310100

| model | Spearman vs J(A) | Top-1 | joint MAPPO | duplicate rate | conflict rate |
| --- | --- | --- | --- | --- | --- |
| Phase-5K Matching decoder | 0.1223 | 0.6667 | 0.0000e+00 | 0.0000 | 0.0000 |
| Joint utility HGNN | 0.1939 | 0.6667 | 0.0000e+00 | 0.0000 | 0.0000 |

### 各时刻

| scenario | t | split | baseline MAPPO | proposed MAPPO | baseline Spearman | proposed Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| 310100 | 0 | train | 0.0000e+00 | 0.0000e+00 | -0.6208 | -0.7027 |
| 310100 | 15 | train | 0.0000e+00 | 0.0000e+00 | 0.5591 | 0.8557 |
| 310100 | 30 | train | 0.0000e+00 | 0.0000e+00 | 0.4286 | 0.4286 |
| 310101 | 0 | test | 0.0000e+00 | 0.0000e+00 | -0.4843 | -0.5888 |
| 310101 | 15 | test | 1.0000e+06 | 1.0000e+06 | 0.1810 | 0.3441 |
| 310101 | 30 | test | 0.0000e+00 | 0.0000e+00 | 0.5441 | 0.6029 |

Top-1 是分配池里分数最高的分配是否为 J(A) 最高的分配。联合 MAPPO 是 Hungarian 读出交给冻结 MAPPO 执行 15 步的 payload。

## 5. 判断

310101 上，Joint utility HGNN 的 Spearman 是 0.1194，Phase-5K Matching decoder 是 0.0803。两者的联合 MAPPO 都是 3.3333e+05，duplicate rate 和 conflict rate 都是 0。Spearman 有提高，联合 MAPPO 没有提高。需要进一步设计 hierarchical joint planner。本阶段不开始该设计。

本阶段不进入 deployment verification。

`planner_ready = False`

`deployment_verified = False`
