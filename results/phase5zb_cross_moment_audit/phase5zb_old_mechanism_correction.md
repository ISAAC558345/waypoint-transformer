# Phase-5Z / Phase-5ZA 机制修正

旧代码 `assignment_latents` 已经先构造

`triple_i = concat(H_UAV_i, H_Buoy_{a_i}, H_edge_{i,a_i})`

然后才对选中 UAV 做 mean。Phase-5ZA 的 `phi(pair_i)` 再 mean，仍是这个一阶集合统计。

因此不能再说“旧结构没有先绑定 UAV-target”。

准确名称是 `FIRST_ORDER_SET_AGGREGATION_LOSS`。

一阶求和保留的是 UAV 集合和 buoy 集合。`sum_i u_i b_i^T` 才会随具体配对改变。

Phase-5ZA 在 2834 个 `|ΔJ15| ≥ 1 block` 的训练集 swap 上，把 cosine ≥ 0.99 的比例从 1.0 降到 0.3119，仍高于 0.10。pair binding 不是完整修复。
