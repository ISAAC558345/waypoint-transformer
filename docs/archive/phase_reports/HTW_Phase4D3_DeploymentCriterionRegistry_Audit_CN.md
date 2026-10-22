# HTW Phase-4D-3 Deployment Criterion Registry Audit

本阶段只检索仓库里是否已有预注册 deployment criterion，并为人工预注册准备参数清单。
未训练 B2，未修改 B2，未修改 τ，未修改 planner / decide()，未进入 Phase-4E，未从 64 个 independent scenario 反推阈值。

**deployment_criterion = NOT_PRE_REGISTERED**

**planner_ready = False**

**status = BLOCKED_ON_HUMAN_SPECIFICATION**

## Q1. 仓库中是否存在预注册 deployment criterion？

**否。** NO PRE-REGISTERED DEPLOYMENT CRITERION FOUND.

- 检索文件数：49
- 含关键词文件数：17
- 含数值 acceptance 规则、且能作为 B2 部署标准的文件数：0
- 未根据 Gate-1 的 3/64、−0.214、0.111 或 26/64 反推任何阈值

## Q2. 如果存在，原始来源是什么？

不存在。`source = null`。

相关但不是 deployment criterion 的近邻文档：

- `HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md` / C. Conservative deployment gate：Gate-1/2 are scientific sanity checks (mean R > LOCAL; permutation drop>0.03). The same section states Gate-3 deployment threshold is undefined and planner_ready=False.
- `HTW_Phase4D_DeploymentGate_Audit_CN.md` / E/F：Explicitly states the project has no predefined deployment-safety threshold.
- `HTW_Phase4D2_DeploymentSafety_Audit_CN.md` / 既有工程阈值检索：Marks deployment criterion = NOT PRE-REGISTERED and leaves A-E thresholds null.

补充：Phase-3 / Phase-4A / FINAL_* / `HTW_Phase3_EvidenceMatrix_CN.md` 没有规定 B2 的 allowed degrade rate、worst-case ΔR 或 mean gain 门槛。
Phase-4B 的 Gate-1/2 只是 16-state 科学 sanity（mean R > LOCAL、permutation drop>0.03），同一节写明 Gate-3 deployment threshold **未定义**。
Phase-10.3 in-range zero-motion gate、Round-14 `deployment_ready=false`、Low-Level safety projector 属于别的系统，不能挪用为 B2 residual-credit 部署标准。

## Q3. 如果不存在，哪些参数必须由人工确定？

见 `deployment_criterion_proposal.json`。至少需要人工决定：

1. 最大允许 degradation rate
2. 最大允许 worst-case ΔR
3. 是否要求 mean ΔR > 0
4. 是否要求 mean U_traj > LOCAL
5. 是否允许少量 scenario degradation
6. 是否要求 B2 在所有使用场景都不劣于 LOCAL
7. 是否需要独立 deployment hold-out 再验证一次
8. 是否需要连续多个 independent batch 满足 criterion

proposal 只列出参数，**不把任何候选数值宣布为正式标准**。

## Q4. 当前 Gate-1 的 observed performance 是什么？

以下数字来自已冻结的 Phase-4C / Phase-4D / Phase-4D-2，本阶段不修改、不优化。

- Gate-1：margin = U_hat(argmax U_hat) − U_hat(argmax U_single)；τ = 1.535e6
- n_B2 = 26/64
- mean R = 0.874
- mean ΔR = 0.111
- degrade = 3/64
- degrade rate = 0.047
- worst-case ΔR = -0.214

这些是 observed performance，不是 acceptance rule。

## Q5. 在 criterion 未确定之前，能否宣布 planner_ready？

**不能。** `planner_ready = False`。即使 Gate-1 指标看起来不错，也不能因为指标好看自动 PASS。

## 最终状态

- deployment_criterion = NOT_PRE_REGISTERED
- planner_ready = False
- status = BLOCKED_ON_HUMAN_SPECIFICATION

本阶段停止。不训练，不改 B2，不改 τ，不改 planner，不改 decide()，不接入 shadow planner，不进入 Phase-4E。

