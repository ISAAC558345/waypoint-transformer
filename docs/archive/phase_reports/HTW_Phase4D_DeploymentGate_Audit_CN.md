# HTW Phase-4D B2 Conservative Deployment Gate Audit

本阶段只审计冻结 B2 是否具备保守部署条件，并检查只依赖 first-hold 可观测量的预定义 gate。
未修改正式 planner / decide()，未修改 Low-Level，未重新训练 B2，未按 independent U_traj 调阈值，未进入 Phase-4E。

**门控：PARTIAL** — 存在预定义 margin gate 能减少部分恶化，但阈值来自 16-state Phase-4B shadow，只在同一批 64 independent scenario 上评价，没有第二批 held-out 证明 gate 本身不过拟合。工程也没有预先定义 deployment-safety threshold。不能接入正式 planner。

**planner_ready = False**。deployment_safe_gate = None。

## 数据来源

- Phase-4C 真正 unseen scenario 独立验证：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_decisions.csv`
- 64 个 independent scenario ID：[220000, 220001, 220002, 220003, 220004, 220005, 220006, 220007, 220008, 220009, 220010, 220011, 220012, 220013, 220014, 220015, 220016, 220017, 220018, 220019, 220020, 220021, 220022, 220023, 220024, 220025, 220026, 220027, 220028, 220029, 220030, 220031, 220032, 220033, 220034, 220035, 220036, 220037, 220038, 220039, 220040, 220041, 220042, 220043, 220044, 220045, 220046, 220047, 220048, 220049, 220050, 220051, 220052, 220053, 220054, 220055, 220056, 220057, 220058, 220059, 220060, 220061, 220062, 220063]
- 这些 ID 来自 `LLV2_train`，从未进入 Phase-4B GroupKFold / feature / hyperparameter / threshold selection
- 全部 64 个 scenario 保留，包括 worst-case 220012，不删除恶化场景
- B2 分数复用 Phase-4C 已写出的 C_hat / U_hat，不重新 fit ExtraTrees

## A. Deployment-time 可观测量

允许：

- U_single
- Phase-3E candidate-conditioned geometry
- C_hat / U_hat from frozen ExtraTrees
- candidate ranking by U_single or U_hat
- score margin = U_hat(B2) - U_hat(LOCAL)
- B2_changed = argmax(U_hat) != argmax(U_single)

禁止作为 gate 输入：

- U_traj
- C_abs / C / F
- oracle / argmax U_traj
- later holds
- remaining_arrivals
- future position / action

模型不确定性：frozen B2 contract only returns C_hat = ExtraTrees.predict; no tree-dispersion / uncertainty field was defined in Phase-4B/4C. Gate-3 is not constructible. Do not invent a proxy.

## B. 逐 scenario 风险（相对 LOCAL）

- 改善 / 持平 / 恶化 = 31 / 22 / 11
- B2 改变 first-hold = 46/64
- mean ΔR = 0.129
- median ΔR = 0.000
- 5th percentile ΔR = -0.200
- 10th percentile ΔR = -0.085
- worst-case ΔR = -0.500 (scenario 220012)
- B2 < LOCAL 的 scenario：[220000, 220007, 220012, 220023, 220028, 220032, 220036, 220040, 220041, 220046, 220059]
- 全表见 `phase4d_scenario_risk.csv`

## C. 预定义 gate（不优化）

- 固定阈值 τ = 1.535e+06
- 阈值来源：HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md changed-state mean dU_hat。**不是**用 64 个 independent U_traj 反选出来的。
- score margin = U_hat(argmax U_hat) - U_hat(argmax U_single)，只依赖 first-hold 分数

- Gate-0：永远使用 B2
- Gate-1：仅当 first-hold 改变且 margin ≥ τ 时用 B2，否则 LOCAL
- Gate-2：仅当 margin ≥ τ 时用 B2，否则 LOCAL
- Gate-3：uncertainty rejection — **不可构造**

说明：当 first-hold 未改变时 margin=0 < τ，因此 Gate-1 与 Gate-2 在本定义下等价。这是定义结果，不是调参。

## D. 各 gate 指标

| gate | constructible | n_B2 | mean U | mean R | median R | worst R | top-1 | top-3 | improve/worse/tie | mean gain vs LOCAL | worst ΔR vs LOCAL | gap vs ORACLE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Gate-0 | yes | 64 | 1.103e+07 | 0.892 | 1.000 | 0.500 | 0.391 | 0.766 | 31/11/22 | 1.562e+06 | -0.500 | 1.359e+06 |
| Gate-1 | yes | 26 | 1.081e+07 | 0.874 | 1.000 | 0.500 | 0.391 | 0.672 | 22/3/39 | 1.344e+06 | -0.214 | 1.578e+06 |
| Gate-2 | yes | 26 | 1.081e+07 | 0.874 | 1.000 | 0.500 | 0.391 | 0.672 | 22/3/39 | 1.344e+06 | -0.214 | 1.578e+06 |
| Gate-3 | **不可构造** |  |  |  |  |  |  |  |  |  |  |  |

Gate-1/2 使用 B2 的 scenario 数 = 26；退回 LOCAL = 38。

## E. 安全性

1. 有没有简单 gate 能保留主要平均收益并减少恶化？
   Gate-0 mean R=0.892，恶化 11，worst ΔR=-0.500。
   Gate-1/2 mean R=0.874，恶化 3，worst ΔR=-0.214。
   预定义 τ 来自 Phase-4B shadow，不是在 64 集上搜索得到。
2. gate 是否可能只在这 64 个 scenario 上过拟合？
   **是，有这个风险。** 阈值虽未用 independent U_traj 选择，但 gate 的安全结论只在同一批 64 上评价，没有第二批 unseen scenario。
3. 若没有足够证据证明 gate 安全：planner_ready = False。
   本阶段 **planner_ready = False**。工程仍未定义正式 deployment-safety threshold。
4. 未为了 PASS 修改 gate 定义，未删除 220012，未调 τ。

## F. PASS / PARTIAL / FAIL

**PARTIAL**。存在预定义 margin gate 能减少部分恶化，但阈值来自 16-state Phase-4B shadow，只在同一批 64 independent scenario 上评价，没有第二批 held-out 证明 gate 本身不过拟合。工程也没有预先定义 deployment-safety threshold。不能接入正式 planner。

本阶段停止。不修改 planner，不调用正式 decide()，不进入 Phase-4E，不训练任何模型。

