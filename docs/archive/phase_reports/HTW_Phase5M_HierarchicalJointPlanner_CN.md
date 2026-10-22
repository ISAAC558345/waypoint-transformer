# HTW Phase-5M Hierarchical Joint Planner

**planner_ready = False**

**deployment_verified = False**

本阶段把一次 HGNN 分配改成候选生成加联合精炼。没有修改 Residual MAPPO、reward、低层策略或 deployment，没有调用 deployment `decide()`，没有使用 230000–230127。

## 1. 当前问题

Phase-5L 继续训练 encoder 后，310101 上的 Spearman 从 0.0803 升到 0.1194，联合 MAPPO 仍是 3.3333e5。直接分配的冲突率和重复目标率已经是 0。当前问题是高层组合优化，不是 decoder 冲突，也不是 encoder 特征不够。

## 2. 两段结构

Stage 1 保留 HGNN encoder 和冻结的 Matching decoder，只用来给每架 UAV 产生 top-3 浮标候选，不直接输出最终分配。若互斥组合超过 500，候选宽度降到仍可穷尽的 top-k。

Stage 2 在候选组合上计算 `J(A) = sum_i payload_gain(i, a_i) - lambda * conflict_loss(A)`。lambda = 1，不拟合。conflict_loss 仍是通信收益下降，因此边际和大于联合 fly-toward payload 时，J(A) 等于联合 fly-toward payload。这个量不写入低层 reward。

Greedy refinement 每次把一架尚未分配的 UAV 接到一个剩余候选浮标上，选择使当前部分分配 J(A) 最高的配对。Exhaustive search 评价该宽度下的全部互斥分配，取得分最高者。穷尽搜索只用于这个 M=6、K<=3 的验证。

## 3. 实验

数据是已有的 [310100, 310101]，时刻 t=0、15、30。Original HGNN 是冻结 utility encoder 加 Phase-5K Matching decoder 的 Hungarian 分配。Phase-5L HGNN 是训练后的联合 encoder 加同一个冻结 Matching decoder 的 Hungarian 分配。层次 planner 的候选来自 Phase-5L 的配对分数。

Spearman 是候选池内分数与 J(A) 的相关性。穷尽精炼和贪心精炼使用 J(A) 本身作为分数，因此 Spearman 为 1。Top-1 是该方法读出的分配是否等于候选池内 J(A) 最高的分配。搜索复杂度是每个决策时刻调用 J(A) 的次数。直接 HGNN 不调用 J(A)。

### 测试场景 310101

| model | joint MAPPO | Spearman vs J(A) | Top-1 | J(A) evaluations |
| --- | --- | --- | --- | --- |
| Original HGNN | 3.3333e+05 | 0.3905 | 0.0000 | 0.0 |
| Phase-5L HGNN | 3.3333e+05 | 0.0286 | 0.0000 | 0.0 |
| Hierarchical greedy | 3.3333e+05 | 1.0000 | 1.0000 | 32.0 |
| Hierarchical exhaustive | 3.3333e+05 | 1.0000 | 1.0000 | 6.0 |

### 全部决策时刻

| model | joint MAPPO | Spearman vs J(A) | Top-1 | J(A) evaluations |
| --- | --- | --- | --- | --- |
| Original HGNN | 1.6667e+05 | 0.3333 | 0.0000 | 0.0 |
| Phase-5L HGNN | 1.6667e+05 | -0.1714 | 0.0000 | 0.0 |
| Hierarchical greedy | 5.0000e+05 | 1.0000 | 0.8333 | 32.0 |
| Hierarchical exhaustive | 6.6667e+05 | 1.0000 | 1.0000 | 6.0 |

## 4. 判断

六个决策时刻上，穷尽精炼的联合 MAPPO 均值是 6.6667e+05，两种直接 HGNN 都是 1.6667e+05。穷尽精炼的 Top-1 是 1.0000，直接分配的 Top-1 是 0.0000。每个时刻的互斥候选平均是 6.0 个，穷尽搜索评价 6.0 次 J(A)，贪心评价 32.0 次。310101 上四种读出的联合 MAPPO 均值都是 3.3333e+05：t=15 直接分配更高，t=30 穷尽精炼更高。层次精炼能选中候选池内 J(A) 最高的分配，贪心精炼的全部时刻 Top-1 是 0.8333。310101 的均值还没有稳定高于直接 Hungarian。本阶段不进入 deployment verification。

`planner_ready = False`

`deployment_verified = False`
