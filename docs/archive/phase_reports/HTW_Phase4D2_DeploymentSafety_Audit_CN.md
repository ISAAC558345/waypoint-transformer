# HTW Phase-4D-2 B2 Pre-Registered Deployment Safety Criterion Audit

本阶段只把 deployment-safe 判定标准写成显式配置，并对已冻结的 64 个真正 unseen scenario 做最终安全性审计。
未重新训练 ExtraTrees，未改变 B2 特征/超参数，未重选 τ，未删除任何 independent scenario（含 220012），
未用 independent U_traj/C/oracle 构造 gate，未修改正式 planner / decide()，未进入 Phase-4E。

**门控：PARTIAL** — B2 在 independent scenario 上泛化成立，固定 margin gate 也降低了风险，但项目尚未预先定义正式 deployment-safety criterion，因此不能宣布 planner_ready。

**planner_ready = False**。

**deployment criterion = NOT PRE-REGISTERED**。

## 数据来源与 64 个 independent scenario 身份

- Phase-4C 独立验证决策表：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_decisions.csv`
- Phase-4C 标签表：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_eval_labels.csv`
- Phase-4C 摘要（permutation 只读复用，不重训）：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_summary.json`
- 64 个 independent scenario ID：[220000, 220001, 220002, 220003, 220004, 220005, 220006, 220007, 220008, 220009, 220010, 220011, 220012, 220013, 220014, 220015, 220016, 220017, 220018, 220019, 220020, 220021, 220022, 220023, 220024, 220025, 220026, 220027, 220028, 220029, 220030, 220031, 220032, 220033, 220034, 220035, 220036, 220037, 220038, 220039, 220040, 220041, 220042, 220043, 220044, 220045, 220046, 220047, 220048, 220049, 220050, 220051, 220052, 220053, 220054, 220055, 220056, 220057, 220058, 220059, 220060, 220061, 220062, 220063]
- 这些 ID 来自 `LLV2_train`，从未进入 Phase-4B GroupKFold / feature / hyperparameter / threshold selection
- 全部 64 个 scenario 保留，包括 worst-case 220012
- B2 分数复用 Phase-4C 已写出的 C_hat / U_hat，不重新 fit ExtraTrees

## 冻结对象

- C_hat = 已冻结 ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)
- U_hat = U_single + C_hat
- τ = 1.535e+06（显示为 1.535e6）
- τ 来源：HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md changed-state mean dU_hat
- τ **不是**由 64 个 independent U_traj 反推，本阶段禁止修改

## Deployment-time 可观测量

Gate 只允许使用 first-hold 时可获得的信息：U_single、Phase-3E candidate-conditioned geometry、B2 的 C_hat/U_hat、由此得到的 ranking，以及 margin。
禁止作为 gate 输入：U_traj、C、oracle、后续 hold、remaining arrivals、未来位置/动作。

## 既有工程阈值检索

- 检索结果：NOT PRE-REGISTERED
- 找到预注册数值阈值：False
- 原因：Phase-4B/4C/4D summaries set planner_ready=false and do not define numeric acceptance rules for mean_dR / degrade_rate / worst_delta_R / mean utility gain. No prior deployment_criterion.json exists.
- 因此 **不要** 从这 64 个 independent scenario 拟合阈值。
- `deployment_criterion.json` 中 A–E 的 acceptance_rule 全部标记为 NOT PRE-REGISTERED，并列为需要人工确认的字段。
- F（gate 不得使用 independent U_traj/C/oracle）是唯一已固定的限制，且本阶段满足。

## 固定 Gate 定义

- Gate-0：永远使用 B2
- Gate-1：若 margin >= 1.535e6 则使用 B2，否则退回 LOCAL
- margin = U_hat(argmax U_hat) - U_hat(argmax U_single)
- 未改变 first-hold 时 margin=0 < τ，因此本定义与 Phase-4D 的 Gate-1/Gate-2 等价。这是定义结果，不是调参。

## 最终审计指标

对照（Phase-4C 冻结）：LOCAL mean R=0.763，mean U_traj=9.469e+06。

| 指标 | Gate-0 | Gate-1 |
|---|---:|---:|
| n_B2 | 64 | 26 |
| n_LOCAL | 0 | 38 |
| mean R | 0.892 | 0.874 |
| median R | 1.000 | 1.000 |
| std R | 0.154 | 0.159 |
| mean U_traj | 1.103e+07 | 1.081e+07 |
| median U_traj | 1.100e+07 | 1.100e+07 |
| improve / tie / degrade | 31/22/11 | 22/39/3 |
| degrade rate | 0.172 | 0.047 |
| mean ΔR | 0.129 | 0.111 |
| median ΔR | 0.000 | 0.000 |
| 5th percentile ΔR | -0.200 | 0.000 |
| 10th percentile ΔR | -0.085 | 0.000 |
| worst-case ΔR | -0.500 | -0.214 |
| worst-case scenario | 220012 | 220059 |
| top-1 | 0.391 | 0.391 |
| top-3 | 0.766 | 0.672 |
| B2-ORACLE overlap | 25/64 (0.391) | 25/64 (0.391) |
| mean residual gap to ORACLE | 1.359e+06 | 1.578e+06 |

## 逐 scenario 风险（Gate-1）

- Gate-1 仍然恶化的 scenario：[220023, 220028, 220059]
- Gate-1 改善最大的 5 个 scenario：[{'scenario_id': 220006, 'delta_R_Gate1': 0.5714285714285714}, {'scenario_id': 220044, 'delta_R_Gate1': 0.5714285714285714}, {'scenario_id': 220013, 'delta_R_Gate1': 0.5}, {'scenario_id': 220009, 'delta_R_Gate1': 0.46153846153846156}, {'scenario_id': 220037, 'delta_R_Gate1': 0.46153846153846156}]
- 220012：margin=1.389e+06，use_B2=0，ΔR_B2=-0.500，ΔR_Gate1=0.000，avoided=True
- Gate-0 恶化但 Gate-1 成功避免：[220000, 220007, 220012, 220032, 220036, 220040, 220041, 220046]
- 完整 64 行见 `phase4d2_scenario_final.csv`

## 稳定性检查（不重训）

1. Gate-1 是否仍然保留 B2 的主要收益？**yes**。 Gate-0 mean R=0.892，Gate-1 mean R=0.874，保留 Gate-0 mean ΔR 的 0.856。
2. Gate-1 是否显著减少 Gate-0 的 degradation？**yes**。 恶化数 11→3，worst ΔR -0.500→-0.214。
3. Gate-1 的收益是否主要来自少数 scenario？**concentrated_but_not_single_scenario**。 正 ΔR 质量中 top-3 份额=0.213，top-5 份额=0.333，正收益 scenario 数=22。
4. 是否存在某一个 scenario 对结果有异常支配作用？**no_single_scenario_dominates**。 最大 leave-one-out |Δ mean ΔR|=0.007（去掉 220006）。
5. B2 permutation sanity 是否仍然支持 candidate-conditioned signal？**True**。 Phase-4C：real R=0.892，perm R=0.672，drop=0.220。未重新训练。
6. Gate 是否完全由 deployment-time 可观测量构成？**是**。margin 与 τ 都不使用 independent U_traj/C/oracle。

## 已经证明 / 尚未证明

已经证明：

- B2 在真正 unseen scenario 上相对 LOCAL 具有泛化收益（Phase-4C：mean R 0.892 vs 0.763）。
- 固定 margin gate（τ 来自 Phase-4B shadow）可以降低恶化风险（11→3，worst ΔR -0.500→-0.214）。
- Gate 输入不依赖 independent 标签。

尚未证明：

- B2 是否达到项目规定的正式 deployment-safety standard。
- 因为该 standard **尚未预注册**，本阶段不能把观测值改写成 acceptance rule。
- 没有第二批 held-out scenario 证明 gate 本身的安全结论可迁移。

## PASS / PARTIAL / FAIL

判定规则（预先写死，不按结果改写）：

- PASS：仅当已有预注册 deployment criterion 且全部满足。
- PARTIAL：B2 泛化成立且固定 margin gate 降低风险，但项目尚未预定义正式 criterion。
- FAIL：固定 Gate-1 不再具有相对 LOCAL 的稳定收益，或发现泄漏。

**PARTIAL**。B2 在 independent scenario 上泛化成立，固定 margin gate 也降低了风险，但项目尚未预先定义正式 deployment-safety criterion，因此不能宣布 planner_ready。

**planner_ready = False**。

本阶段停止。不修改 planner，不修改 decide()，不训练任何模型，不修改 B2，不修改 τ，不进入 Phase-4E。

