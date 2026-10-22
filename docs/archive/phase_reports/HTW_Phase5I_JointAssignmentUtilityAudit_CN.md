# HTW Phase-5I Joint Assignment Utility Audit

**planner_ready = False**

**deployment_verified = False**

本阶段只比较单个目标效用和完整分配效用。没有改模型，没有训练 HGNN，没有改 Residual MAPPO，没有修改 B2、τ、Gate-1、Candidate-C、planner 或 deployment `decide()`，没有使用 230000–230127。

## 1. Individual utility 的局限

Phase-5F 的标签是固定其他无人机之后，一名无人机换目标的 15 步 fly-toward payload。Phase-5G 和 Phase-5H 把六架飞机的最高分拼成一条分配之后，团队 payload 下降。

搜索池是每架无人机边际 fly-toward 效用的前 3 个目标，再做排他组合。完整的 12 选 6 有 665280 条，不能逐条做 15 步执行。本阶段把池子收在可穷尽的宽度内，最宽而且不超过 500 条。六个决策时刻平均穷尽 27.6667 条。

池内 sum of individual utility 和联合 fly-toward payload 的 Spearman = 0.8308，Kendall = 0.3198。HGNN 分数之和与联合 fly-toward 的 Spearman = 0.4752。

单个效用最高的组合就是联合 fly-toward 最高组合的比例是 0.1667。

## 2. Joint mismatch

oracle joint payload 是这个搜索池里联合 fly-toward 的最大 payload。MAPPO 只在少数代表分配上执行，执行器仍是冻结的 `mappo_original.pt`。

| assignment | joint fly payload | joint MAPPO payload |
| --- | --- | --- |
| sum of individual utility | 166666.6667 | 833333.3333 |
| utility HGNN greedy | 0.0000 | 333333.3333 |
| utility HGNN decoder | 0.0000 | 166666.6667 |
| DeepSets greedy | 666666.6667 | 1500000.0000 |
| raw individual argmax | 0.0000 | 0.0000 |
| best joint fly-toward | 1000000.0000 | 1833333.3333 |

sum-individual 最大分配的 MAPPO payload = 833333.3333。联合 fly-toward 最大分配的 MAPPO payload = 1833333.3333。两者分配相同的比例是 0.1667。

## 3. HGNN decoder

现有 OrderFreeSetDecoder 的联合解码和外部贪心一样，都低于池内最好的联合 fly-toward。缺的是联合标签，不是先换一个 decoder。

## 4. Global assignment head

现有 decoder 已经能输出一条完整分配。本阶段不增加 global assignment head。

## 5. Assignment coupling

冲突矩阵是六架无人机在取各自分数最高目标时，两两选中同一浮标的比例。贪心排他会拆开这个冲突，所以 Phase-5H 看到的重复目标数是 0；冲突发生在组合之前。

| UAV | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 3 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

原始最高分分配的重复目标数平均 5.0000。贪心后的重复目标数平均 0.0000。正队列浮标被贪心分配覆盖的比例是 0.3333。原始最高分分配的联合 fly-toward payload 是 0.0000。

## 6. 新的监督接口

不改网络。三个标签只作接口：

- A assignment similarity：当前排他匹配，Phase-5 已经证明它和 payload 排序不一致。
- B individual trajectory utility：固定其他无人机后的单目标 15 步 payload 排序。它能提高单目标替换，不能代表六架同时换目标。
- C joint trajectory utility：`J(A) = sum_i payload_gain(i, A_i) - lambda * conflict_loss(A)`。

lambda 不拟合，接口取 1。conflict_loss 定义为边际 payload 之和减去这条分配真正的联合 fly-toward payload，小于 0 时记 0。这样 J(A) 等于联合 payload。服务冲突率在贪心之后是 0，所以不能再用抢同一浮标的计数充当这个损失。

单个目标的效用最大组合和联合 fly-toward 最大组合不是同一个。联合效用标签有证据。

本阶段不开始 joint utility HGNN 训练，也不进入 deployment verification。

`planner_ready = False`

`deployment_verified = False`
