# HTW Phase-5ZM：Model-R1 多种子共识与部署规范审计

**Gate = MS-A SEED_NEUTRAL_AGGREGATION_VERIFIED**

阻塞项：['MS-D']。

**planner_ready = False**

**deployment_verified = False**

**FROZEN_FOR_FRESH_CONFIRMATION = False**

**fresh_confirmation_authorized = False**

本阶段只在 310110–310117 上审计预先登记的 Rank-Mean Consensus V1。没有挑选最佳 seed，没有根据 J15 调权，没有打开 310118–310125，没有调用正式决策函数。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。

## Q1. 五个 R1 seed 的真实分歧程度有多大？

24 个状态中，5/5 选中同一分配的比例是 0.9583，4/5 的比例是 0.0417，不超过 3/5 的比例是 0.0000。
consensus_strength 等于 1.0、0.8、0.6 和不超过 0.4 的状态数分别是 23、1、0、0。
seed 1–4 两两的分配一致率是 1。seed 0 与其余每个 seed 的一致率都是 0.9583，只差 310111 t=15。rank Spearman 在所有 seed 对上都在 0.9926 到 0.9988 之间。Top-3 rank 集合的平均 Jaccard 是 0.9292。候选 rank 方差的状态平均是 3.8622。第一名与第二名的平均 rank margin 是 1.0083。

这些数字只作为 confidence diagnostic，没有用来改决策。

## Q2. 310111 t=15 为什么出现 seed disagreement？

五个冻结 value head 对同一组 120 个候选给出了不同的 Top-1。seed 0 选 `-1;-1;4;-1;5;3`，seed 1–4 选 `3;-1;4;-1;5;-1`。两者都是各自 seed 的唯一 rank 1，该状态上最小 value margin 是 0.0212，不是 value 完全相同造成的。

## Q3. Rank-Mean Consensus V1 最终选择哪一个 assignment？

310111 t=15 选择 `3;-1;4;-1;5;-1`。决定阶段是 mean_rank。
- `-1;-1;4;-1;5;3`：五 seed rank [1, 2, 2, 2, 2]，mean rank 1.8，majority 1，已保存 J15 3.0000e+06 bit。
- `3;-1;4;-1;5;-1`：五 seed rank [2, 1, 1, 1, 1]，mean rank 1.2，majority 4，已保存 J15 3.0000e+06 bit。

mean rank 更低者优先。若 mean rank 相同，则 majority Top-1 更高者优先。若仍然相同，则用分配向量的字典序。

## Q4. 这个选择是否完全不依赖 J15？

不依赖。选择函数只接收五个 seed 的 value 和候选分配向量。J15、scenario id 和开发集收益都不是输入。两个竞争分配的已保存 J15 是否相同，只在选择之后读取：相同。该状态没有因为 J15 相同而被删掉。

## Q5. seed permutation 是否完全不影响结果？

seed_permutation_invariance = 1.0。原顺序、反序和 10 个固定随机置换下，rank mean、选中分配和 consensus strength 都不变。

## Q6. candidate enumeration 是否影响结果？

candidate_permutation_invariance = 1.0。10 个固定置换下，按分配对齐后的 rank sum 和最终分配都不变。完全平局时的第二层 tie-break 是分配向量字典序，不是输入下标。

## Q7. Consensus 是否保留现有 True-H1 R1 的收益？

这是已保存标签上的重放，不是新的性能结论。Consensus 平均 selected J15 是 2.7917e+06 bit，中位数是 3.0000e+06 bit。场景聚类 95% 区间是 [2.4167e+06, 3.1667e+06] bit，这个区间描述的是均值，不是中位数。
五个单 seed 的平均 selected J15 是 seed 0 2.7917e+06, seed 1 2.7917e+06, seed 2 2.7917e+06, seed 3 2.7917e+06, seed 4 2.7917e+06。Model-S 五个 seed 的总平均是 2.3417e+06 bit。
Consensus 相对 Model-S 的平均 ΔJ15 是 4.5000e+05 bit，区间 [1.6667e+05, 7.4167e+05] bit。相对单 seed R1 的平均差是 0.0000e+00 bit。
按状态计，相对 Model-S 为正、平、负的数量是 5、19、0。按 scenario 平均后是 5、3、0。
预先登记的保留条件是：Consensus 均值不低于最低单 seed，也不低于 Model-S，容差 1 bit。结果是 True。平均 normalized regret 是 0.062500。

## Q8. 五 seed ensemble 后完整推理耗时是多少？

计时边界是 encoder、候选生成、optimized True-H1、五个 Model-R1、Rank-Mean、TaskOption。中位数 334.28 ms，均值 340.10 ms，P95 450.93 ms，P99 458.34 ms。
分量均值：True-H1 207.56 ms，五个 Model-R1 20.70 ms，aggregation 0.6248 ms。
五个 Model-R1 合计 20.70 ms。上一阶段单个 value head大约 4 ms，所以多出来的四个 head 大约增加 16 ms。aggregation 只有 0.62 ms。总时间仍由 True-H1 的 207.56 ms 主导。本次中位数 334.28 ms 低于上一阶段单 head 的 342.09 ms，这是三次重复下 True-H1 本身的波动，不能解释成五个 seed 更快。没有为这次测量改写模块。deployment_latency_requirement 仍然是 NOT_SPECIFIED。调用者环境指纹差是 0，在线 value 与已保存 v_r1 的最大绝对误差是 0。

## Q9. 当前还缺哪些人工部署规范？

缺五项，都写在 pending specification 里，本阶段没有代填：最大高层时延、超时策略、推理异常 fallback、Rank-Mean 是否正式批准、fresh confirmation 是否授权。DeepSets、B2、Oracle、HGNN 和 seed 0 都没有被设为 fallback。

## Q10. 是否已经具备冻结算法结构的条件？

技术核验是 True。主 Gate 是 MS-A。即便技术核验通过，聚合规则仍是 model_r1_aggregation_approved = false，不能算已经冻结进正式系统。FROZEN_FOR_FRESH_CONFIRMATION 保持 False。

## Q11. 是否可以打开 310118–310125？

FRESH_CONFIRMATION_NOT_AUTHORIZED

人工尚未批准全部 pending specification。本阶段的答案不能是“可以”。

## 接口草案

当前状态进入冻结的 Stage-1 encoder 和候选生成，得到 120 个联合分配。optimized True-H1 只给这些分配做一步 value 评价。五个冻结 Model-R1 输出 value，Rank-Mean Consensus V1 选出一个联合分配，写成 remaining_hold=15 的 TaskOption。冻结 Residual MAPPO 只执行这个 option，不重新选择浮标。

True-H1 不是第二个 planner，不是轨迹 planner，也不是 15 步 rollout。
