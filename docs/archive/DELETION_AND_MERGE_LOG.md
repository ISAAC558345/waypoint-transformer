# 删除与合并记录

只删除了两类已确认没有独立实验内容的文件：仓库 `backup/` 之外的 Python 字节码和系统垃圾文件，以及与根目录或归档副本 SHA-256 完全相同的 Phase-4 模块报告。空的 `.test-tmp`、`round_02/t*` 和已经清空的 `__pycache__` 目录一并移除。

- 缓存和空目录删除项：683
- 字节相同的重复报告删除项：15
- 内容不同、改为归档的模块副本：5

## 合并后的过程文档

- `OTHER_PROCESS_NOTES.md (25 files)`
- `FINAL_NOTES.md (4 files)`
- `LOW_LEVEL_NOTES.md (8 files)`
- `PHASE6I_PROCESS.md (25 files)`
- `PAPER_NOTES.md (10 files)`
- `table docs/archive/tables/FINAL_CODE_FREEZE.json`
- `table docs/archive/tables/PAPER_TABLE_MAIN_RESULTS.csv`
- `table docs/archive/tables/PAPER_TABLE_PAIRWISE_STATS.csv`

## 已删除的重复报告

- `src/wt/lowlevel_v2/phase4d_deployment_criterion_registry/HTW_Phase4D6_DeploymentCriterionRegistration_CN.md`
- `src/wt/lowlevel_v2/phase4d_deployment_gate_audit/outputs/HTW_Phase4D_DeploymentGate_Audit_CN.md`
- `src/wt/lowlevel_v2/phase4d7_independent_holdout/outputs/HTW_Phase4D7_IndependentHoldoutAudit_CN.md`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/HTW_Phase4D7B_IndependentHoldoutGeneration_CN.md`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/HTW_Phase4D7C_HoldoutBatchSizeSpecificationAudit_CN.md`
- `src/wt/lowlevel_v2/phase4d7a_holdout_source_audit/outputs/HTW_Phase4D7A_HoldoutSourceQualificationAudit_CN.md`
- `src/wt/lowlevel_v2/phase4d5_deployment_criterion_candidates/outputs/HTW_Phase4D5_DeploymentCriterionCandidates_CN.md`
- `src/wt/lowlevel_v2/phase4d4_deployment_criterion_specification/outputs/HTW_Phase4D4_DeploymentCriterionSpecification_CN.md`
- `src/wt/lowlevel_v2/phase4d3_deployment_criterion_registry/outputs/HTW_Phase4D3_DeploymentCriterionRegistry_Audit_CN.md`
- `src/wt/lowlevel_v2/phase4d2_deployment_safety_audit/outputs/HTW_Phase4D2_DeploymentSafety_Audit_CN.md`
- `src/wt/lowlevel_v2/phase4c_independent_validation/outputs/HTW_Phase4C_IndependentValidation_Audit_CN.md`
- `src/wt/lowlevel_v2/htw_phase4b_credit_learner/outputs/HTW_Phase4B_TrajectoryAwareCreditLearner_Audit_CN.md`
- `src/wt/lowlevel_v2/htw_phase4b_b2_validation/outputs/HTW_Phase4B_B2_IndependentValidation_Audit_CN.md`
- `src/wt/lowlevel_v2/htw_phase4b_b2_shadow/outputs/HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md`
- `src/wt/lowlevel_v2/htw_phase4a_hindsight_credit/outputs/HTW_Phase4A_HindsightCredit_DatasetAudit_CN.md`

## 内容不同而保留的副本

- `src/wt/lowlevel_v2/phase4e_development_pool_specification/outputs/HTW_Phase4E_RemediationDevelopmentPoolSpecificationAudit_CN.md -> docs/archive/merged/nonidentical_module_copies/HTW_Phase4E_RemediationDevelopmentPoolSpecificationAudit_CN.md`
- `src/wt/lowlevel_v2/phase4e_development_audit/outputs/HTW_Phase4E_RemediationDevelopmentAudit_CN.md -> docs/archive/merged/nonidentical_module_copies/HTW_Phase4E_RemediationDevelopmentAudit_CN.md`
- `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/HTW_Phase4E_DeploymentVerification_CN.md -> docs/archive/merged/nonidentical_module_copies/HTW_Phase4E_DeploymentVerification_CN.md`
- `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/HTW_Phase4E_FailureAttribution_CN.md -> docs/archive/merged/nonidentical_module_copies/HTW_Phase4E_FailureAttribution_CN.md`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/HTW_Phase4D7F_IndependentHoldoutGeneration_CN.md -> docs/archive/merged/nonidentical_module_copies/HTW_Phase4D7F_IndependentHoldoutGeneration_CN.md`
