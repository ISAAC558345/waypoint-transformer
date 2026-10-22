# HTW Project Cleanup Audit

本阶段只做文件清理审计。没有删除文件，没有修改代码逻辑、B2、τ、Gate-1、Candidate-C、planner 或 `decide()`，没有重新生成 scenario，也没有改实验状态。

**status = DEPLOYMENT_VERIFICATION_FAILED**

**deployment_verified = False**

**planner_ready = False**

未进入 Phase-4F，未进入 planner integration，未宣布 deployment PASS。

## 1. 文件总数量

扫描跳过 `.git`。清单共 **26525** 个文件，合计约 **4.78 GB**。

完整清单：`src/wt/lowlevel_v2/project_cleanup_audit/outputs/cleanup_file_inventory.csv`

每行包含路径、大小、UTC 修改时间、类型、是否被代码引用、是否被报告引用、是否属于实验 lineage，以及建议分类。

另有 **231** 个空目录。`backup/snapshots` 里的空目录保持不动。

## 2. 分类统计

| 分类 | 数量 | 说明 |
| --- | --- | --- |
| A. 必须保留 | 24729 | 代码、模型、registry、manifest、审计报告、实验结果、round 输出、snapshot |
| B. 可以归档 | 19 | 与仓库根目录报告字节对应的 Phase-4 markdown 副本 |
| C. 删除候选 | 1777 | snapshot 之外的 `.pyc`，约 30.1 MB |

删除候选按目录：

| 位置 | 文件数 |
| --- | --- |
| `src` | 906 |
| `round_36` | 735 |
| `tests` | 122 |
| `experiments` | 14 |

`backup/` 中的删除候选数量为 0。

## 3. 删除候选

1777 个文件全部是 Python 字节码缓存，不在 `backup/snapshots` 中。它们不是 B2 标签、holdout、criterion 或 verification 的唯一记录。

建议只在人工确认后删除。本阶段没有执行删除。

空目录里，`.test-tmp/` 和 `round_02/t*` 看起来是测试临时目录。它们也等待确认，没有删除。snapshot 内的空目录不在此列。

## 4. 必须保留的关键文件

- `src/wt/lowlevel_v2/outputs/phase0/LLV2_DATA_REGISTRY.json`
- `src/wt/lowlevel_v2/phase4d_deployment_criterion_registry/deployment_criterion_registry.json`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/deployment_holdout_manifest.json`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_labels.csv`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_integrity_audit.csv`
- `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_verification_results.csv`
- `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_failure_attribution.json`
- `src/wt/lowlevel_v2/htw_phase4a_hindsight_credit/outputs/phase4a_hindsight_credit.csv`
- `src/wt/lowlevel_v2/htw_phase4b_b2_shadow/outputs/b2_shadow_decisions.csv`
- 仓库根目录的 `HTW_Phase4C`、`HTW_Phase4D`、`HTW_Phase4E` 报告
- `backup/snapshots/` 下的全部 snapshot

Phase-4C / 4D / 4E 的 registry、manifest、integrity audit、verification 和 failure attribution 都在保留集合里。

## 5. 风险检查

| 检查 | 结果 |
| --- | --- |
| 是否已删除文件 | 否 |
| 删除候选是否包含 snapshot | 否 |
| 删除候选是否包含唯一实验结果 | 否。候选只有 `.pyc` |
| Phase-4E audit、deployment verification、B2 标签、scenario manifest 是否仍在 | 是 |
| `deployment_verified` | False |
| `planner_ready` | False |
| status | DEPLOYMENT_VERIFICATION_FAILED |

19 份模块内 Phase-4 报告与根目录同名报告重复。归档它们的风险是中等：根目录副本还在，但模块输出目录里的副本也常被阶段脚本写回。因此只列入归档建议，不删除。

字节相同的 round 输出和 snapshot 最初被标成重复文件。它们仍是对应轮次的记录，已改回保留。

## 6. 删除建议

可以在确认后删除 snapshot 之外的 1777 个 `.pyc`。

不要删除或改写 `backup/snapshots`。不要删除 holdout、criterion、B2 标签、verification 或 failure attribution。19 份重复 Phase-4 报告如需归档，先保留根目录副本，并单独确认。

删除计划：`cleanup_deletion_plan.md`。等待人工确认。本阶段没有自动删除高风险文件。
