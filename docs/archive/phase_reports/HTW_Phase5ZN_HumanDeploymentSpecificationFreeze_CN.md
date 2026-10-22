# HTW Phase-5ZN：人工部署规范冻结

**Gate = SPEC-A ALGORITHM_FROZEN_WAITING_HUMAN_APPROVAL**

**planner_ready = False**

**deployment_verified = False**

**FROZEN_FOR_FRESH_CONFIRMATION = False**

**fresh_confirmation_authorized = False**

本阶段没有改算法，没有训练，没有 rollout，没有打开 310118–310125。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。

## 候选架构

高层是：当前状态、冻结的 Phase-5L encoder 和 Stage-1 top-3 候选、120 个合法联合分配、shipped optimized True-H1 的一步响应、五个冻结 Model-R1、Rank-Mean Consensus V1、TaskOption。

低层是：TaskOption、冻结 Residual MAPPO、SafetyProjector、环境执行。低层不重新选择浮标。

True-H1 是价值评估器。它不是第二个 planner，不是轨迹 planner，也不是 15 步 rollout。未交付的 batch GEMM 不在冻结结构里。正式 DeepSets 权重被登记并保持不动，但它不是这条候选控制器的 encoder。

清单哈希是 `ebf61f60bfdf2d3ad2939b5fd741c25950c1ccbcdc67fce6f600c7c047f1970b`。缺失组件数是 0。

## H1. 聚合规则

技术建议是批准 RANK_MEAN_CONSENSUS_V1。理由只来自 Phase-5ZM：seed 和 candidate permutation invariance 都是 1.0，24 个已消费状态里 23 个五 seed 一致、1 个是 4/5，Consensus 的 selected J15 仍是 2.7917e6 bit，相对 Model-S 仍是 +4.5000e5 bit。选择不使用 J15、场景收益或 seed 权重。

model_r1_aggregation_approved 仍然是 false。批准状态是 HUMAN_APPROVAL_REQUIRED。RECOMMEND_APPROVAL = true 只是建议，不是批准。

## H2. 时延

原型测量是 median 334.28 ms、P95 450.93 ms、P99 458.34 ms。deployment_latency_requirement = HUMAN_SPECIFICATION_REQUIRED。

参考选项是 500 ms、750 ms、1000 ms 和 NOT_YET_SPECIFIED。本阶段没有选择。500 ms 只是当前计算平台上略高于已消费开发状态的 P99，不是真实 UAV 部署 SLA。

## H3. 超时

timeout_policy = HUMAN_SPECIFICATION_REQUIRED。

当前控制器在 hold 计数到 0 时发布新 option，并把 hold 设为 15；每个已执行步把计数减 1。

T1 KEEP_CURRENT_TASKOPTION 不发布新分配，不把 hold 重新设为 15，沿用当前合法 option，并按现有规则递减 remaining_hold。它最接近现有递减语义，但在 hold 已经到期的决策边界上，现有控制器会开始新的 15 步窗口，所以 T1 仍需人工批准。

T2 USE_LAST_VALID_ASSIGNMENT 会把上一次合法分配重新包成 TaskOption。按现有构造方式，这会重新打开任务窗口，因此不是默认安全等价行为。

T3 ABORT_NEW_DECISION 不改变当前 executor 状态，也不发布新结果。它和 T1 的差别是：T1 仍允许 hold 计数按原规则递减，T3 连 executor 状态也不改。

## H4. 运行时回退

runtime_fallback_policy = HUMAN_SPECIFICATION_REQUIRED。

True-H1 异常、Model-R1 的 NaN 或 Inf、空候选池、非法联合分配、SafetyProjector 失败、checkpoint 缺失、TaskOption 构造失败，都没有被批准的自动 fallback。DeepSets、B2、Oracle、单个 R1 seed、Phase-5L 直接分配和 fly-toward 都不是 fallback。

如果已经存在一个合法 TaskOption，KEEP_CURRENT_VALID_TASKOPTION 是非灾难性推理失败时最小的候选行为。如果当前没有合法 TaskOption，状态是 NO_VALID_FALLBACK_DEFINED。本阶段不补一条新策略。

SafetyProjector 失败时，冻结执行器内部会把动作置零。那是已经冻结的执行器行为，不是新的部署 fallback。

## H5. Fresh confirmation

预留池完整性是 PASS。检查只读了协议里的场景登记，没有列数据目录，没有加载 snapshot，没有读标签，没有 rollout，也没有 inference。

310100–310109 是训练登记，310110–310117 是已消费确认登记。310118–310125 与这些登记没有交集，因此没有参与训练、checkpoint 选择、损失设计、候选设计、阈值调整、Rank-Mean 选择、True-H1 优化或时延优化。

fresh_confirmation_authorized = false。authorization_source = HUMAN_APPROVAL_REQUIRED。

## 未来确认草案

只有人工授权之后，310118–310125 才能按这份冻结系统一次性评估。不允许根据前面几个 scenario 改模型、聚合、超时、候选、seed 或 checkpoint。

主比较是冻结的 Rank-Mean True-H1 控制器对冻结 Model-S。Oracle 不是部署基线。正式 DeepSets 只有在不生成新数据、并且已有产物能给出同一 payload 时才能作为次比较；不能为了补齐它而重新 rollout。

成功标准草案只要求平均配对 payload 改进大于 0，并同时报告场景聚类 95% 区间、中位数和改进的正、平、负数量。同时要求非法分配、状态污染、崩溃和最终决策的 NaN/Inf 都是 0。没有规定必须达到某个百分比。时延门槛写成 LATENCY_GATE_PENDING_HUMAN_SPECIFICATION。

这些标准没有被批准，也不能用 310118–310125 来调整。
