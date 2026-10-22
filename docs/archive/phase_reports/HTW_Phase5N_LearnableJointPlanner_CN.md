# HTW Phase-5N Learnable Joint Planner

**planner_ready = False**

**deployment_verified = False**

本阶段用可学习的 Joint Assignment Head 替换 Phase-5M 的搜索式精炼。Phase-5L encoder 和 Matching decoder 保持冻结，没有修改 Residual MAPPO、reward、低层策略或 deployment，没有调用 deployment `decide()`，没有使用 230000–230127。

## 1. 结构

Stage 1 仍由冻结的 Phase-5L HGNN 给出每个 UAV 的 top-3 候选。这六个 UAV 的 top-3 在所测时刻都是同一组浮标。Stage 2 是新的 Joint Assignment Head。它的输入是 UAV embedding、Buoy embedding 和二者的 global pooled embedding，输出候选配对分数。

配对分数先做 Sinkhorn，得到 soft assignment，再做 Hungarian 解码。非候选格子不参与匹配；候选列用尽后，其余 UAV 记为 idle，不会被配到掩码外的浮标。

监督集合是该 top-3 掩码里的全部极大互斥分配：只要某个 UAV 还有未被占用的自身候选，它就不能闲置。按 UAV 顺序填满后再把后面的 UAV 全部闲置，只覆盖了这个集合的一小部分，Hungarian 可以选出集合里的其他分配。穷尽搜索在这个完整集合上计算 J(A)，A* 是 J(A) 最大的那些分配。

## 2. 分配级损失

标签是上述集合上的穷尽搜索结果。损失比较的是完整分配的分数 S(A)，不是单个 UAV-Buoy pair。J(A) 全部相等的决策没有正样本对，损失为 0。

当 J(A) > J(B) 时，损失是 `-log sigmoid(S(A) - S(B))`。S(A) 是该分配在 Sinkhorn soft plan 上的概率和。Hungarian 最大化的也是这个分配级分数，不是逐个 UAV-Buoy pair 的名次。

梯度只使用 310100 的 t=0、15、30。310101 不进入梯度和选模。最佳轮次是 10，训练 Spearman 是 0.3565。

## 3. 结果

每个决策的候选掩码含 120 个极大匹配。推理时间是每次决策生成分配的耗时，不含 MAPPO 执行。穷尽搜索的时间是这 120 次 J(A) fly-toward 展开。Phase-5M 报告里的穷尽搜索只包含按 UAV 顺序填满的 6 个前缀分配，本表的耗时和 MAPPO 属于扩大后的极大匹配集合。

### 测试场景 310101

| model | joint MAPPO | Spearman vs J(A) | Top-1 | inference ms |
| --- | --- | --- | --- | --- |
| Original HGNN | 3.3333e+05 | -0.0433 | 0.0000 | 0.2438 |
| Phase-5L HGNN | 3.3333e+05 | -0.0498 | 0.0000 | 0.1705 |
| 穷尽搜索（极大匹配） | 1.0000e+06 | 1.0000 | 1.0000 | 4039.5175 |
| Phase-5N Joint Planner | 1.3333e+06 | -0.0843 | 0.6667 | 1.0058 |

### 全部决策时刻

| model | joint MAPPO | Spearman vs J(A) | Top-1 | inference ms |
| --- | --- | --- | --- | --- |
| Original HGNN | 1.6667e+05 | -0.0857 | 0.0000 | 0.2994 |
| Phase-5L HGNN | 1.6667e+05 | -0.0735 | 0.0000 | 0.1928 |
| 穷尽搜索（极大匹配） | 8.3333e+05 | 1.0000 | 1.0000 | 4184.7795 |
| Phase-5N Joint Planner | 8.3333e+05 | 0.1361 | 0.8333 | 0.9807 |

Top-1 表示读出的分配是否属于 J(A) 最大的那一组极大匹配。J(A) 全相同时，120 个分配都是最优，Top-1 为 1 只说明解码留在掩码内。Spearman 使用处理并列的相关系数，并且只在 J(A) 和分数都不是常数的决策上平均。这样的决策只有 310100 t=30 和 310101 t=30。穷尽搜索的分数就是 J(A)，所以在这两个时刻它的 Spearman 为 1。

| scenario | t | J(A) 是否可排序 | exhaustive MAPPO | Phase-5N MAPPO | Phase-5N Spearman | Phase-5N Top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| 310100 | 0 | 否，120 个分配的 J(A) 相同 | 0.0000e+00 | 0.0000e+00 | NA | 1.0000 |
| 310100 | 15 | 否，120 个分配的 J(A) 相同 | 1.0000e+06 | 0.0000e+00 | NA | 1.0000 |
| 310100 | 30 | 是 | 1.0000e+06 | 1.0000e+06 | 0.3565 | 1.0000 |
| 310101 | 0 | 否，120 个分配的 J(A) 相同 | 0.0000e+00 | 0.0000e+00 | NA | 1.0000 |
| 310101 | 15 | 否，120 个分配的 J(A) 相同 | 0.0000e+00 | 3.0000e+06 | NA | 1.0000 |
| 310101 | 30 | 是 | 3.0000e+06 | 1.0000e+06 | -0.0843 | 0.0000 |

## 4. 判断

分配级排序没有复现穷尽搜索。310101 t=30 的 Phase-5N Top-1 是 0.0000，Spearman 是 -0.0843，联合 MAPPO 是 1.0000e+06，穷尽搜索是 3.0000e+06。310101 的平均 Top-1 是 0.6667，平均联合 MAPPO 是 1.3333e+06，穷尽搜索的平均联合 MAPPO 是 1.0000e+06。平均 MAPPO 更高来自 J(A) 完全相同的时刻：A* 只是并列中的第一个，另一个并列分配可以被 MAPPO 执行出更高载荷，但这不是对 J(A) 的更优排序。本阶段不进入 deployment verification。

`planner_ready = False`

`deployment_verified = False`
