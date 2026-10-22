# HTW Phase-4D-4 Deployment Criterion Human Specification Draft

本阶段只建立供人工确认的 deployment criterion 规格草案。
未重新训练或修改 B2，未修改 τ，未修改 planner / decide()，未进入 Phase-4E。
未用 64 个 independent scenario 的 mean / median / worst-case / degradation rate 反推任何正式阈值。
未自动选择任何候选 criterion，未宣布 planner_ready=True。

**deployment_criterion = HUMAN_SPECIFICATION_REQUIRED**

**planner_ready = False**

**status = BLOCKED_ON_HUMAN_SPECIFICATION**

## Q1. 当前为什么不能直接进入 planner？

正式 `decide()` 仍是 joint greedy 解码，不是 B2 shadow score。
Phase-4C/4D 只证明：在冻结 Gate-1 下，B2 对真正 unseen scenario 相对 LOCAL 有平均收益，并能减少部分恶化。
仓库没有预注册的 deployment-safety acceptance criterion。没有该标准，就不能判断当前观测是否“足够安全”，因此不能接入正式 planner。

## Q2. 当前缺失的究竟是模型证据还是 deployment specification？

**缺失的是 deployment specification，不是模型是否存在 independent 证据。**
Phase-4C 已经在 64 个真正 unseen scenario（LLV2_train 220000–220063）上验证 B2。
Phase-4D/4D-2 已经用预先固定的 τ=1.535e6 审计 Gate-1。
Phase-4D-3 检索确认：NO PRE-REGISTERED DEPLOYMENT CRITERION FOUND。
阻塞项是人工尚未规定 C1–C8，而不是还没有 independent scenario。

## Q3. 哪些指标已经有 independent validation？

以下是已冻结的模型/gate 证据，不是 acceptance rule：

- B2 在 64 个 unseen scenario 上相对 LOCAL 的平均收益（Phase-4C：mean R 0.892 vs 0.763）
- permutation sanity：candidate-conditioned geometry 被打乱后 R 下降（Phase-4C drop=0.220）
- 固定 margin Gate-1 降低恶化（11→3）并避开 220012（Phase-4D/4D-2）
- Gate 输入只用 first-hold 可观测量，不使用 independent U_traj / C / oracle

## Q4. 哪些指标只是 observed performance？

`observed_reference` 单独记录冻结 Gate-1 表现，并明确：
“These are observed evaluation results and are NOT deployment acceptance criteria.”

- n_B2 = 26/64
- mean R = 0.874
- mean ΔR = 0.111
- degrade = 3/64
- degrade_rate = 0.047
- worst_case_delta_R = -0.214

这些数字不得填入 acceptance_fields。

## Q5. 哪些参数必须由人工/项目负责人预先规定？

C1 最大允许 degradation rate → `max_degradation_rate = null`
C2 最大允许 worst-case ΔR → `max_worst_case_delta_R = null`
C3 mean ΔR 要求：不要求 / 必须 > 0 / 必须 ≥ 预先指定值 → `min_mean_delta_R = null`
C4 mean U_traj 要求：不要求 / 必须 > LOCAL / 必须超过预先指定 improvement margin → `min_mean_U_gain = null`
C5 是否允许 scenario degradation：完全不允许 / 允许少量 / 允许但受 worst-case 限制
C6 非劣：strict（∀ scenario, R_selected ≥ R_LOCAL）或 aggregate → `require_all_scenarios_noninferior = null`
C7 是否需要第二批 deployment hold-out → `independent_holdout_required = null`
C8 跨 batch 重复验证：`required_independent_batches = null`；若需要，每批是否必须独立生成也待定

决策矩阵见 `deployment_criterion_decision_matrix.csv`。`human_decision` 全部为 PENDING_HUMAN_DECISION，未自动选择。

## Q6. criterion 确定后，下一步应该如何重新审计 64 个 independent scenario？

1. 人工把 C1–C8 写成非 null 的预注册字段，且不得用这 64 个 scenario 的统计量回填。
2. 保持 B2、特征、超参数、τ=1.535e6、Gate-1 定义不变。
3. 复用已冻结的 Phase-4C/4D Gate-1 逐 scenario 表，按新 criterion 逐条判定 PASS/FAIL。
4. 不在这 64 个 scenario 上搜索更优 τ 或更优阈值。
5. 若 C7/C8 要求新的 independent batch，则在新 batch 上重复同一审计；在此之前 planner_ready 仍为 False。
6. 即使 64 个 scenario 全部满足人工标准，也只有在预注册 criterion 被书面锁定后才能讨论 planner 接入；那属于后续指令，不是本阶段。

## 规格字段（全部未填正式数值）

- max_degradation_rate = null
- max_worst_case_delta_R = null
- min_mean_delta_R = null
- min_mean_U_gain = null
- require_all_scenarios_noninferior = null
- independent_holdout_required = null
- required_independent_batches = null

## 最终状态

- deployment_criterion = HUMAN_SPECIFICATION_REQUIRED
- planner_ready = False
- status = BLOCKED_ON_HUMAN_SPECIFICATION

本阶段停止。不修改任何模型、planner、decide()、B2、τ。不进入 Phase-4E。

