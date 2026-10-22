# 仓库清理盘点

后续删除和合并记录见 `docs/archive/DELETION_AND_MERGE_LOG.md`。过程报告全文已并入 `docs/archive/merged/`，表格在 `docs/archive/tables/`。

本清单在移动未被代码引用的根目录过程报告之前生成规则，并在同一次运行中完成这些移动。没有删除 phase 目录、checkpoint、实验结果或 Markdown 正文。没有执行 `git reset` 或 `git clean`。

- 分支：`feature/maritime-wt-stage1a`
- HEAD：`716f514b942dfe413dd23fd51cf1c523d8211c0f`
- 工作区在盘点前已有大量未提交修改。这些修改原样保留。
- Python：`Python 3.14.2`
- 纳入清单的文件数：23878。`__pycache__` 与 `.pyc` 没有逐个列入。
- 本次从仓库根目录移入 `docs/archive/reports/` 的报告数：75。
- 删除清单：空。阶段报告都含有该阶段的 Gate、参数或失败分析；被代码引用的报告留在根目录，其余过程报告归档而不是删除。

逐文件登记在 `docs/archive/REPOSITORY_CLEANUP_INVENTORY.csv`。列为相对路径、类别、报告名是否出现在 `src` 的 Python 源码中、是否属于模型或实验记录、以及保留或已移动。

## 类别计数

| 类别 | 文件数 |
| --- | --- |
| historical_round | 11691 |
| snapshot_backup | 6809 |
| experiment_result | 3114 |
| source | 747 |
| figure | 368 |
| checkpoint | 301 |
| phase_report | 285 |
| other | 248 |
| test | 157 |
| document | 138 |
| paper_or_final_note | 20 |

## 关键文件 SHA-256

| 路径 | SHA-256 |
| --- | --- |
| `phase6if_ranking_loss/runs/c/epoch_001.pt` | `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864` |
| `phase6id_joint_utility_training/epoch_006.pt` | `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde` |
| `phase5l_joint_encoder/joint_utility_encoder.pt` | `929a5ae71ac78e63745eadfb5af018158b15e2d03de7f8b33c82a789d946663d` |
| `phase5k_joint_decoder/matching_decoder.pt` | `de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5` |
| `src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt` | `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e` |
| `phase6iaa_formal_planner/active_policy.json` | `3fc2f0b1277188e6f12084370c9c1d5acb70e897f6a4fba8b47bedf031b25ebb` |
| `phase6iab_planner_stability/summary.json` | `c4597d7e114120b3081409a91847a87503bf3026529e8f1385ebff1c9473f3d6` |
| `phase6iac_independent_holdout/summary.json` | `f10e19734792daa1b229e4403a4bf4cc0f5055a4893e6532ca89035f4050887f` |

## 已归档的根目录报告

- `benchmark_complexity_redesign_report.md` → `docs/archive/reports/benchmark_complexity_redesign_report.md`
- `benchmark_level2_design_report.md` → `docs/archive/reports/benchmark_level2_design_report.md`
- `cleanup_deletion_plan.md` → `docs/archive/reports/cleanup_deletion_plan.md`
- `Current_Hierarchical_MultiUAV_System_Analysis.md` → `docs/archive/reports/Current_Hierarchical_MultiUAV_System_Analysis.md`
- `FINAL_ABLATION_SUMMARY_CN.md` → `docs/archive/reports/FINAL_ABLATION_SUMMARY_CN.md`
- `FINAL_CODE_FREEZE.json` → `docs/archive/reports/FINAL_CODE_FREEZE.json`
- `FINAL_PROJECT_MANIFEST.md` → `docs/archive/reports/FINAL_PROJECT_MANIFEST.md`
- `FINAL_README_CN.md` → `docs/archive/reports/FINAL_README_CN.md`
- `FINAL_RESEARCH_SUMMARY_CN.md` → `docs/archive/reports/FINAL_RESEARCH_SUMMARY_CN.md`
- `HL_RL_SUPPORT_LOSS_EQUATIONS.md` → `docs/archive/reports/HL_RL_SUPPORT_LOSS_EQUATIONS.md`
- `HL_SEARCH_DECODER_EQUATIONS.md` → `docs/archive/reports/HL_SEARCH_DECODER_EQUATIONS.md`
- `HTW_Final_Code_and_Reproducibility_Report_CN.md` → `docs/archive/reports/HTW_Final_Code_and_Reproducibility_Report_CN.md`
- `HTW_LowLevelV2_Phase1A_Instability_Audit_CN.md` → `docs/archive/reports/HTW_LowLevelV2_Phase1A_Instability_Audit_CN.md`
- `HTW_LowLevelV2_Phase2A_Seed1_Audit_CN.md` → `docs/archive/reports/HTW_LowLevelV2_Phase2A_Seed1_Audit_CN.md`
- `HTW_LowLevelV2_Phase2C_Mask_and_BaseZero_Audit_CN.md` → `docs/archive/reports/HTW_LowLevelV2_Phase2C_Mask_and_BaseZero_Audit_CN.md`
- `HTW_Phase3G_TrajectoryAwareCredit_StructuralAudit_CN.md` → `docs/archive/reports/HTW_Phase3G_TrajectoryAwareCredit_StructuralAudit_CN.md`
- `HTW_Phase3H_CandidateConditionedDecisionCredit_Audit_CN.md` → `docs/archive/reports/HTW_Phase3H_CandidateConditionedDecisionCredit_Audit_CN.md`
- `HTW_Phase3_Failure_Diagnosis_CN.md` → `docs/archive/reports/HTW_Phase3_Failure_Diagnosis_CN.md`
- `HTW_Phase4D6_DeploymentCriterionRegistration_CN.md` → `docs/archive/reports/HTW_Phase4D6_DeploymentCriterionRegistration_CN.md`
- `HTW_Phase4D7A_HoldoutSourceQualificationAudit_CN.md` → `docs/archive/reports/HTW_Phase4D7A_HoldoutSourceQualificationAudit_CN.md`
- `HTW_Phase4D7B_IndependentHoldoutGeneration_CN.md` → `docs/archive/reports/HTW_Phase4D7B_IndependentHoldoutGeneration_CN.md`
- `HTW_Phase4D7C_HoldoutBatchSizeSpecificationAudit_CN.md` → `docs/archive/reports/HTW_Phase4D7C_HoldoutBatchSizeSpecificationAudit_CN.md`
- `HTW_Phase4D7E_HoldoutBatchSizeRegistration_CN.md` → `docs/archive/reports/HTW_Phase4D7E_HoldoutBatchSizeRegistration_CN.md`
- `HTW_Phase4D7_IndependentHoldoutAudit_CN.md` → `docs/archive/reports/HTW_Phase4D7_IndependentHoldoutAudit_CN.md`
- `HTW_Phase4E_FailureAttribution_CN.md` → `docs/archive/reports/HTW_Phase4E_FailureAttribution_CN.md`
- `HTW_Phase4E_RemediationDevelopmentPoolSpecificationAudit_CN.md` → `docs/archive/reports/HTW_Phase4E_RemediationDevelopmentPoolSpecificationAudit_CN.md`
- `HTW_Phase5D_TrajectoryAwareUtilityRankingDesignAudit_CN.md` → `docs/archive/reports/HTW_Phase5D_TrajectoryAwareUtilityRankingDesignAudit_CN.md`
- `HTW_Phase5_hierarchical_design_report.md` → `docs/archive/reports/HTW_Phase5_hierarchical_design_report.md`
- `HTW_Phase6I_AssignmentInteractionCapacityAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_AssignmentInteractionCapacityAudit_CN.md`
- `HTW_Phase6I_CandidateBoundaryRuntimeReduction_CN.md` → `docs/archive/reports/HTW_Phase6I_CandidateBoundaryRuntimeReduction_CN.md`
- `HTW_Phase6I_CandidateConditionedFeatureAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_CandidateConditionedFeatureAudit_CN.md`
- `HTW_Phase6I_DeploymentRequirementsAndFaultHandling_CN.md` → `docs/archive/reports/HTW_Phase6I_DeploymentRequirementsAndFaultHandling_CN.md`
- `HTW_Phase6I_DynamicSINRObservabilityAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_DynamicSINRObservabilityAudit_CN.md`
- `HTW_Phase6I_ExecutionAuditClosure_CN.md` → `docs/archive/reports/HTW_Phase6I_ExecutionAuditClosure_CN.md`
- `HTW_Phase6I_ExecutorConditionedAssignmentAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_ExecutorConditionedAssignmentAudit_CN.md`
- `HTW_Phase6I_ExecutorResponseObservabilityAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_ExecutorResponseObservabilityAudit_CN.md`
- `HTW_Phase6I_FinalDeploymentReadinessAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_FinalDeploymentReadinessAudit_CN.md`
- `HTW_Phase6I_FormalCandidatePoolAlignmentAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_FormalCandidatePoolAlignmentAudit_CN.md`
- `HTW_Phase6I_FrozenInferenceFusionAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_FrozenInferenceFusionAudit_CN.md`
- `HTW_Phase6I_HGNNCompatibleDataset_CN.md` → `docs/archive/reports/HTW_Phase6I_HGNNCompatibleDataset_CN.md`
- `HTW_Phase6I_HGNN_InputAlignment_CN.md` → `docs/archive/reports/HTW_Phase6I_HGNN_InputAlignment_CN.md`
- `HTW_Phase6I_HGNN_ValidationErrorAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_HGNN_ValidationErrorAudit_CN.md`
- `HTW_Phase6I_HighLevelInferenceIntegrationAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_HighLevelInferenceIntegrationAudit_CN.md`
- `HTW_Phase6I_IndependentHoldoutValidation_CN.md` → `docs/archive/reports/HTW_Phase6I_IndependentHoldoutValidation_CN.md`
- `HTW_Phase6I_LocalPairSwapAttributionAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_LocalPairSwapAttributionAudit_CN.md`
- `HTW_Phase6I_LocalPairSwapCounterfactualAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_LocalPairSwapCounterfactualAudit_CN.md`
- `HTW_Phase6I_ModelPathDecisionAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_ModelPathDecisionAudit_CN.md`
- `HTW_Phase6I_PairwiseAssignmentInteractionHead_CN.md` → `docs/archive/reports/HTW_Phase6I_PairwiseAssignmentInteractionHead_CN.md`
- `HTW_Phase6I_PerUAVExecutionResponseAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_PerUAVExecutionResponseAudit_CN.md`
- `HTW_Phase6I_RankingLossCandidateAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_RankingLossCandidateAudit_CN.md`
- `HTW_Phase6I_RankingLossRedesign_CN.md` → `docs/archive/reports/HTW_Phase6I_RankingLossRedesign_CN.md`
- `HTW_Phase6I_ShortHorizonExecutorResponsePilot_CN.md` → `docs/archive/reports/HTW_Phase6I_ShortHorizonExecutorResponsePilot_CN.md`
- `HTW_Phase6I_T15CandidateObservabilityAudit_CN.md` → `docs/archive/reports/HTW_Phase6I_T15CandidateObservabilityAudit_CN.md`
- `HTW_Phase8_relational_complexity_report.md` → `docs/archive/reports/HTW_Phase8_relational_complexity_report.md`
- `HTW_Project_Cleanup_Audit_CN.md` → `docs/archive/reports/HTW_Project_Cleanup_Audit_CN.md`
- `level2_causal_validation_report.md` → `docs/archive/reports/level2_causal_validation_report.md`
- `LLV2_PHASE2C_ADVANTAGE_NORMALIZATION.md` → `docs/archive/reports/LLV2_PHASE2C_ADVANTAGE_NORMALIZATION.md`
- `LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md` → `docs/archive/reports/LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md`
- `LLV2_PROTOCOL.md` → `docs/archive/reports/LLV2_PROTOCOL.md`
- `LLV2_REWARD_DRAFT.md` → `docs/archive/reports/LLV2_REWARD_DRAFT.md`
- `LLV2_SAFETY_PROJECTOR_REPORT.md` → `docs/archive/reports/LLV2_SAFETY_PROJECTOR_REPORT.md`
- `PAPER_ARCHITECTURE_FIGURE_SPEC_CN.md` → `docs/archive/reports/PAPER_ARCHITECTURE_FIGURE_SPEC_CN.md`
- `PAPER_BENCHMARK_DESCRIPTION_CN.md` → `docs/archive/reports/PAPER_BENCHMARK_DESCRIPTION_CN.md`
- `PAPER_CONTRIBUTIONS_CN.md` → `docs/archive/reports/PAPER_CONTRIBUTIONS_CN.md`
- `PAPER_EXPERIMENT_SECTION_CN.md` → `docs/archive/reports/PAPER_EXPERIMENT_SECTION_CN.md`
- `PAPER_LIMITATIONS_CN.md` → `docs/archive/reports/PAPER_LIMITATIONS_CN.md`
- `PAPER_MAIN_FINDINGS_CN.md` → `docs/archive/reports/PAPER_MAIN_FINDINGS_CN.md`
- `PAPER_METHOD_PIPELINE_CN.md` → `docs/archive/reports/PAPER_METHOD_PIPELINE_CN.md`
- `PAPER_RESULTS_DISCUSSION_CN.md` → `docs/archive/reports/PAPER_RESULTS_DISCUSSION_CN.md`
- `PAPER_TABLE_ABLATION_EVIDENCE_CN.md` → `docs/archive/reports/PAPER_TABLE_ABLATION_EVIDENCE_CN.md`
- `PAPER_TABLE_MAIN_RESULTS.csv` → `docs/archive/reports/PAPER_TABLE_MAIN_RESULTS.csv`
- `PAPER_TABLE_MAIN_RESULTS_CN.md` → `docs/archive/reports/PAPER_TABLE_MAIN_RESULTS_CN.md`
- `PAPER_TABLE_PAIRWISE_STATS.csv` → `docs/archive/reports/PAPER_TABLE_PAIRWISE_STATS.csv`
- `planner_audit_report.md` → `docs/archive/reports/planner_audit_report.md`
- `PLANS.md` → `docs/archive/reports/PLANS.md`

## 留在根目录的原因

审计脚本通过 `ROOT / "HTW_....md"` 读取或写回一部分报告。这些文件名出现在 `src` 中，移动会让旧入口找不到报告，因此保留在根目录。`README.md`、`AGENTS.md`、`requirements.txt`、`ROUND_VERSIONING.md` 和 `round_manifest.json` 也保留。

`backup/` 与 `round_XX/` 含有历史快照和轮次产物，按原位保留，不整目录删除。
