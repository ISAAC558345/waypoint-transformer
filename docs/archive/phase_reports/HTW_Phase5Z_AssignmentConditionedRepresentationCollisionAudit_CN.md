# HTW Phase-5Z Assignment-Conditioned Representation Collision Audit

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练 HGNN、value head、scorer 或 MAPPO，没有重新计算 J15，没有打开 310118–310125，没有调用 deployment `decide()`。追踪的是 Phase-5Y seed 0 的冻结 value head。J15 使用已经保存的 Phase-5X horizon-15 payload。确认集是 310110–310117 的 8 个 scenario、每个 3 个时刻。统计先在 state 内计算，再按 scenario 做 cluster bootstrap（10000 次，seed 0）。

恢复检查：重新前向的 value 与已保存 Phase-5Y 预测的最大绝对差是 0。重新计算的 relational 特征与已保存 `z_rel` 的最大绝对差是 0。

Gate 是 **Z-A**。

## Q1. collision 发生在哪一层

发生在 joint pooling。selected edge、assignment-conditioned relation 和有序 pre-pool 都能分开绝大多数 `|ΔJ15| ≥ 1 block` 的 assignment。经过 mean pooling 之后，这些 pair 的 pooled embedding cosine 全部不低于 0.99。

| 层 | cosine ≥ 0.99 的比例 | scenario-cluster 95% CI |
| --- | --- | --- |
| selected-edge embedding | 0.0423 | [0.0416, 0.0431] |
| edge + pair interaction + REP | 0.0072 | [0.0062, 0.0081] |
| 有序 pre-pool | 0.0424 | [0.0416, 0.0431] |
| post-pool joint embedding | 1.0000 | [1.0000, 1.0000] |
| value-head 输入 | 0.4858 | [0.4715, 0.4998] |

pre-pool 已经不同、post-pool 却撞在一起的比例是 0.9576，CI [0.9569, 0.9584]。head 输入已经不同但 `|ΔV| < 0.1 block` 的比例只有 0.0038。value head 不是主要丢失点。

effective rank 也在这一步掉下来。confirmation 上的中位数：

| 表示 | effective rank | 第一奇异值方差占比 | assignment 间 cosine 中位数 |
| --- | --- | --- | --- |
| selected edge | 5.43 | 0.204 | 0.495 |
| REP | 5.63 | 0.608 | 0.869 |
| pre-pool | 5.08 | 0.206 | 0.496 |
| post-pool | 2.23 | 0.907 | 0.9994 |

同一个 state 里的 `H_UAV`、`H_Buoy` 本来就相同，没有把这一项算成错误。

## Q2. pooling 是否抹掉 UAV-target correspondence

是。当前 post-pool 先对选中的 `(h_u, h_b, h_edge)` 做 mean，再对 UAV pair 做 mean。buoy 集合相同的时候，buoy embedding 的和与哪一架 UAV 连到哪一个浮标无关。

确认集里、两边都在 120 候选池中、只交换两架已选 UAV 目标、并且 `|ΔJ15| ≥ 1 block` 的 swap 有 2308 对。它们的 post-pool cosine 全部 ≥ 0.99。拆开 L2：

| 分量 | pooling 前中位 L2 | pooling 后中位 L2 |
| --- | --- | --- |
| UAV embedding | 0 | 0 |
| buoy embedding | 0.192 | 8.7e-7 |
| selected edge embedding | 0.162 | 0.047 |
| pair interaction | — | 0.004 |

buoy 对应关系在求和之后数值上消失。edge 残差还在，但相对 post-pool 范数（中位数约 20.8）只剩大约 0.047。合法的 UAV 整体重编号保持 value，310114 t=15 上最大绝对差是 1.8e-6。只交换两个目标则不保持执行价值，表示却几乎不变。前者是合法的 permutation equivariance，后者是错误的 correspondence invariance。

Hamming assignment 距离和 head embedding L2 的 Spearman 只有 0.193，CI [0.188, 0.198]。

## Q3. swap 的执行价值是否不同、表示是否几乎相同

是。上述 2308 对里，post-pool cosine ≥ 0.99 的比例是 1。`|ΔV|` 的中位数是 0.684 block，其中 `|ΔV| < 0.1` 的比例是 0.178。`|ΔJ15| ≥ 2 / 3 / 4 block` 的 swap 分别有 839、271、12 对，post-pool cosine 仍然全部 ≥ 0.99。原始 selected-edge 特征能看见这些交换，中位 L2 是 1.00；丢失发生在把 buoy embedding 加到一起的时候。

整段 pre-pool 向量的 cosine 也会 ≥ 0.99，因为没被交换的 UAV 槽占了大部分范数。对应关系的差异在被交换的那两个槽里，不在整段 cosine 里。

## Q4. 310114 t=15 从哪一层开始不可分

`A_pred = 2;5;-1;4;-1;-1`，J15 = 4e6，预测 value = 2.206。

`A_opt = -1;4;2;5;-1;-1`，J15 = 5e6，预测 value = 1.099，rank = 53。

两边选中的浮标集合都是 {2, 4, 5}，只是对应到不同 UAV，并且空闲的 UAV 不同。

| 层 | cosine | L2 |
| --- | --- | --- |
| selected-edge raw | 0.380 | 2.851 |
| selected-edge embedding | 0.665 | 2.150 |
| REP | 0.900 | 1.344 |
| TEMP | 0.891 | 1.847 |
| 有序 pre-pool | 0.690 | 31.142 |
| post-pool | 0.999 | 1.126 |
| value-head 输入 | 0.958 | 6.983 |

pre-pool 的 31.14 里，buoy 分量是 29.23，UAV 分量是 10.54，edge 分量是 2.15。post-pool 的 1.13 几乎全部来自 UAV 分量 1.126；buoy 分量变成 9.1e-7，edge 分量只剩 0.021。

可分性从 mean pooling 开始丢失。value head 读到的剩余差异把真正的 optimum 放到了 53 名，而且方向是反的：它给 J15 更低的 `A_pred` 更高的 value。

当前物理量也指向 `A_pred`。立即施加该分配后的 rate 之和：`A_pred` 是 1.254e6，`A_opt` 是 1.194e6，optimum 的 PHY rank 是 85。逐 UAV 的当前反事实 SINR：

- pred：0.320，0.392，0，0.298，0，0
- opt：0，0.407，0.249，0.302，0，0

`same_target`：pred 是 `1;0;0;1;0;0`，UAV0 和 UAV3 延续当前目标；opt 是全 0。当前 service progress 是 `0;0.055;0;0.157;0;0`，当前分配是 `2;0;5;4;1;10`。延续进度和当前速率都偏向 pred，15 步执行结果却是 opt 多 1 个 block。这个 1 block 不在当前 SINR 里。pooling 先把“谁连哪一个浮标”消掉，head 再顺着剩下的 UAV 集合差异把 optimum 排到后面。

## Q5. 当前反事实 SINR/rate 能否解释 J15 的主要差异

它能解释平均排序的一部分，不能解释 310114 的这个差异。确认集上 PHY_NOW（当前链路集合的 rate 之和，不推进环境）的 Spearman 是 0.486，CI [0.383, 0.594]；Kendall 0.411；符号准确率 0.804；Top-1 0.625；Top-3 0.708。Phase-5Y value 的 Spearman 是 0.580，CI [0.499, 0.645]，高于 PHY_NOW，所以按预先写好的规则，这一项不构成 Z-C。310114 t=15 上 PHY_NOW 把真正 optimum 排到 85，比 value 的 53 更差。

## Q6. pair interaction 是否比单 edge 更能区分 J15

不能。单 edge 的当前 rate 均值 Spearman 是 0.477，CI [0.363, 0.585]，符号准确率 0.792。两两交叉干扰的均值 Spearman 是 -0.268，CI [-0.373, -0.184]，符号准确率只有 0.339。selected-edge embedding 本身的 collision 只有 4.23%。缺的不是单条 edge，而是 pooling 把 edge 和 buoy 的对应关系加没了。

## Q7. TEMP 是否被 value pipeline 使用

有使用。同一 state 内 TEMP 的平均 cosine 是 0.747，CI [0.734, 0.758]，不是常数。把 TEMP 置零后，确认集 50% 的 state 改变 argmax，value 的平均绝对变化是 0.376 block。在 state 内打乱 TEMP 后，argmax 改变 66.7%。TEMP 没有解释 post-pool collision。

## Q8. REP 是否贡献 assignment discrimination

有贡献，比 TEMP 弱。REP 置零后 20.8% 的 state 改变 argmax，value 平均绝对变化 0.336 block。state 内打乱 REP 后，argmax 改变 37.5%。REP 的 assignment 间 cosine 中位数已经是 0.869，区分度低于有序 pre-pool。

## Q9. 大价值差是否仍然 collision

仍然严重。按 pair 数加权：

| `|ΔJ15|` | pair 数 | post-pool cosine ≥ 0.99 | pooling collision | `|ΔV| < 0.1` |
| --- | --- | --- | --- | --- |
| ≥ 2 block | 35065 | 1.000 | 0.956 | 0.188 |
| ≥ 3 block | 13884 | 1.000 | 0.956 | 0.167 |
| ≥ 4 block | 721 | 1.000 | 0.957 | 0.280 |

≥ 4 block 只出现在 1 个确认 state，所以这一行不能单独做 scenario bootstrap。≥ 2 和 ≥ 3 block 覆盖 23 和 22 个 state。

head 输入上的最近邻 contradiction（cosine ≥ 0.99 且价值差达到阈值）是：≥ 1 block 为 0.252，CI [0.228, 0.277]；≥ 2 block 为 0.085；≥ 3 block 为 0.024。最近邻 cosine 中位数是 0.9998。post-pool 上同样的比例更高：0.353、0.152、0.047。

## Q10. 机制标签

十个 Phase-5W 高 regret case 全部带 POOL。pre-pool cosine 在 0.33 到 0.69，post-pool cosine 全部 ≥ 0.999。310114 t=15 只有 POOL：TEMP cosine 0.891，没有到 0.99；当前 PHY 的符号和 J15 相反，所以不是“物理量已经分开、表示没编码”。其余 case 里，TEMP cosine ≥ 0.99 的同时标了 TEMP，PHY 符号与 J15 一致且 post-pool 撞在一起的同时标了 PHY。没有任何一个 case 是 HEAD：post-pool 还没有分开。

因此主标签是 POOL。EDGE、PAIR、TEMP、PHY、STATE、HEAD 都不是这批 collision 的主因。

## Q11. 下一步改什么

只改 assignment-conditioned pooling。有序 pre-pool 已经保留 `(UAV_i, target_i)`。下一步要用保留这个对应关系的 set token，再做 permutation-equivariant 的集合聚合，不能再把 buoy embedding 按集合求和。

不改 HGNN encoder，不增加层数或宽度，不改 REP/TEMP 的定义，不改 value head，不改 candidate pool。当前反事实 SINR 有平均排序信号，但 310114 说明它解释不了这 1 个 block，而且现有 value 的 Spearman 已经更高，所以下一步不是补 PHY 特征。pair interference 弱于单 edge。TEMP 和 REP 已经被 head 使用。当前观测并不是完全没有信息，所以也不转向 learned executor dynamics。

`planner_ready = False`

`deployment_verified = False`
