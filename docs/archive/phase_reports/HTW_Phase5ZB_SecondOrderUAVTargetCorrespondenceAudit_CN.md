# HTW Phase-5ZB Second-Order UAV-Target Correspondence Audit

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练。统计只用 310100–310109 上 Phase-5ZA 已登记的 2834 个主动 swap，条件是 `|ΔJ15| ≥ 1 block`。310114 t=15 只作为预先指定的个案。没有用 310110–310117 挑选聚合，没有打开 310118–310125。

Gate 是 **CM-C**。

## 机制修正

旧代码 `assignment_latents` 已经先构造

`triple_i = concat(H_UAV_i, H_Buoy_{a_i}, H_edge_{i,a_i})`

然后才对选中 UAV 做 mean。Phase-5ZA 的 `phi(pair_i)` 再 mean，仍是这个一阶集合统计。

不能再说“旧结构没有先绑定 UAV-target”。准确名称是 `FIRST_ORDER_SET_AGGREGATION_LOSS`。

## Q1. Phase-5ZA 为什么没通过

是。`phi(pair)` 之后的 mean 仍然是一阶集合统计。它把 collision 从 1.0 降到 0.3119，但预先写定的上限是 0.10。这次用同一个 seed 0 随机初始化复现了 **0.3119**。绑定本身不是完整修复。

## Q2. 旧 pooling 是否已经先绑定再 mean

是。上面的 `triple_i` 来自实际代码。mean 的 buoy 段等于选中浮标 embedding 之和，所以对应关系在求和时被弱化，但绑定这一步是有的。

## Q3. C_UB 有没有降低 collision

没有。2834 对上，cosine ≥ 0.99 的比例是 **1.000**。cosine 中位数是 0.99999999，normalized L2 中位数是 **0.000169**。这不是高维把 cosine 抬高：两个 representation 本身几乎相同。

对照：

| 表示 | collision | cosine 中位数 | normalized L2 中位数 |
| --- | --- | --- | --- |
| 旧 post-pool | 1.000 | 1.000 | — |
| 随机 phi+mean | 0.312 | 0.966 | 0.269 |
| PAIRMEAN | 0.273 | 0.964 | 0.272 |
| mean selected-edge | 1.000 | 1.000 | 0.027 |
| C_UB | 1.000 | 1.000 | 0.000169 |

scenario-cluster 95% CI 也是 [1.000, 1.000]。重编号误差是 0，所以这不是实现把 UAV/buoy 下标弄错了。`u_hat_i b_hat_i^T` 的平均在交换目标后几乎不变，说明归一化之后的 UAV 方向对这种交换不敏感。

## Q4. 大 gap

C_UB 在更大价值差上仍然全是 collision。

| `|ΔJ15|` | pair 数 | C_UB collision | phi+mean | PAIRMEAN |
| --- | --- | --- | --- | --- |
| ≥ 2 block | 942 | 1.000 | 0.255 | 0.249 |
| ≥ 3 block | 244 | 1.000 | 0.250 | 0.238 |
| ≥ 4 block | 9 | 1.000 | 0.333 | 0.333 |

≥ 4 block 只有 9 对，不能单独下结论。≥ 2 和 ≥ 3 block 的 C_UB 仍然完全撞在一起。

## Q5. 310114 t=15

不能分开。`A_pred` 和 `A_opt` 的 C_UB cosine 是 **0.9963**，normalized L2 是 0.0864，仍按 0.99 记为 collision。外积矩阵的 Frobenius 差是 0.0943。两边 idle fraction 都是 0.5，idle UAV 均值的 L2 是 0.0471。A_opt 在 A_pred 的其他 assignment 里按 C_UB 距离排第 83。

同一个 pair 上，PAIRMEAN cosine 是 0.889，随机 phi cosine 是 0.878，mean edge cosine 是 0.9999。一阶 token 比 C_UB 更能分开这个个案。

## Q6. mean selected-edge 为什么不够

selected edge 本身是 `(UAV_i, buoy_{a_i})`，但取 mean 之后，swap 的 normalized L2 只有 0.027，cosine ≥ 0.99 的比例是 1.0。局部 edge 的变化被集合平均稀释了。

C_UB 并没有比它更好。C_UB 的 normalized L2 是 0.000169，assignment Spearman 0.116，低于 edge 的 0.191，也低于 PAIRMEAN 的 0.248。所以不能把“edge 已经含有对应关系，只是一阶平均把它抹掉，二阶矩可以找回来”当成结论。归一化外积把这点残差也抹掉了。

## Q7. C_UB 是否更接近 assignment incidence

不是。incidence 距离和 representation 距离的 Spearman：

- PAIRMEAN：0.248
- mean edge：0.191
- C_UB：0.116

C_UB 更不忠于“谁连哪一个浮标”。

## Q8. 和 `|ΔJ15|` 的关系有没有变好

没有。`|ΔJ15|` 与 normalized L2 的 Spearman：

- 随机 phi：0.450
- PAIRMEAN：0.448
- mean edge：0.334
- C_UB：**-0.002**

最近邻 contradiction（cosine ≥ 0.99 且 `|ΔJ15| ≥ 1 block`）：C_UB 0.460，PAIRMEAN 0.223，edge 0.271。C_UB 更高，不是更低。

训练集 effective rank 均值：C_UB **2.39**，PAIRMEAN **5.40**。旧确认集 post-pool 参考是 2.23，pre-pool 参考是 5.08。C_UB 回到了旧 post-pool 那种低秩，没有把 assignment variation 恢复出来。

## Q9. 要不要再加 C_UE / C_BE

加了，仍然不够。C_UB 没过门槛之后，同一次审计追加了外积，没有训练：

| 表示 | ≥ 1 block collision | 310114 cosine |
| --- | --- | --- |
| C_UB | 1.000 | 0.996 |
| C_UB + C_UE | 1.000 | 0.996 |
| C_UB + C_BE | 1.000 | 0.998 |
| CM-ALL | 1.000 | 0.997 |

单独 C_UB 不够，把 edge 外积接上也不够。

## Q10. 下一阶段能否训练 low-rank second-order pooling

不能。Gate 是 **CM-C**：二阶 UAV-target cross moment 不是当前瓶颈。它没有降低 collision，也没有更接近 assignment 或 J15。

下一阶段应转向 executor-state / dynamic observability，而不是继续改 pooling，也不是训练 low-rank bilinear pooling。

`planner_ready = False`

`deployment_verified = False`
