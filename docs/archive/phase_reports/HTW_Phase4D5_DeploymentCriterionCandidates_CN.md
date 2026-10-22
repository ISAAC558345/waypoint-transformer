# HTW Phase-4D-5 Deployment Criterion Candidate Design

本阶段只设计三档 deployment criterion candidate，不做 deployment PASS/FAIL。
未训练或修改 B2，未修改 τ / Gate-1 / planner / decide()，未改写 64 个 independent scenario 结果。
未把任何 candidate 写入 `deployment_criterion_registry.json`，未宣布 official criterion，未进入 Phase-4E。

**deployment_criterion = HUMAN_SPECIFICATION_REQUIRED**

**planner_ready = False**

**status = BLOCKED_ON_HUMAN_SPECIFICATION**

## Gate-1 与 deployment criterion 分离

Gate-1 只决定**某一个** scenario 是否使用 B2：

- margin = U_hat(argmax U_hat) − U_hat(argmax U_single)
- τ = 1.535e6（Phase-4B shadow 锁定，本阶段不改）
- 若 margin ≥ τ 则 B2，否则 LOCAL

Deployment criterion 决定**整套**冻结 B2 + 冻结 Gate-1 是否允许进入正式 planner。
二者不能混为一谈。改 criterion 不改 Gate-1；改 Gate-1 也不等于已经有 criterion。

## Observed reference（禁止作为阈值来源）

“These are observed evaluation results and are NOT a source of candidate thresholds.”

- mean ΔR = 0.111
- degrade rate = 3/64 = 0.047
- worst-case ΔR = -0.214
- mean R = 0.874
- n_B2 = 26/64

下列 candidate 的数值来自工程设计常数（R 定义在 [0,1] 上的十进制/二分小数、1-in-100 稀有事件配额），
**不是**由 3/64、−0.214 或 0.111 外推出来的 5%、−0.25 或同类填充。
本阶段不计算任何 candidate 在 64 集上的 PASS/FAIL。

## Candidate-A：Strict Safety

- 原则：Single-scenario safety first. Any regression versus LOCAL is a deployment defect.
- C1 max_degradation_rate = 0.0
- C2 max_worst_case_delta_R = 0.0
- C3 min_mean_delta_R = 0.0
- C4 min_mean_U_gain = 0.0
- C5 scenario_degradation_policy = forbid_all
- C6 require_all_scenarios_noninferior = True
- C7 independent_holdout_required = True
- C8 required_independent_batches = 1

理由：任何相对 LOCAL 的单场景恶化都视为部署缺陷。LOCAL fallback 不能用来接受有害替换。
额外要求 1 个独立生成的新 batch，因为严格安全性不能只建立在 220000–220063 这一张表上。

## Candidate-B：Balanced Deployment

- 原则：Allow only rare, bounded losses; require non-negative average gain and one extra unseen batch.
- C1 max_degradation_rate = 0.01（1-in-100，不是 3/64 或 5%）
- C2 max_worst_case_delta_R = -0.10（R 单位区间的 1/10，不是 −0.214 或 −0.25）
- C3 min_mean_delta_R = 0.0（平均不下降，不是 0.111）
- C4 min_mean_U_gain = 0.0（平均 payload 不低于 LOCAL，不抄当前增益）
- C5 scenario_degradation_policy = allow_only_if_worst_case_bound_holds
- C6 require_all_scenarios_noninferior = False
- C7 independent_holdout_required = True
- C8 required_independent_batches = 1

理由：允许极少数、且有上界的单场景损失；总体 rank 与 payload 不得变差；再用 1 个新 unseen batch 约束过拟合。

## Candidate-C：Conservative-with-Fallback

- 原则：Use LOCAL fallback as a risk control; allow limited degradation if average gain, bounded loss, and repeated unseen batches all hold.
- C1 max_degradation_rate = 0.02（2-in-100，不是 3/64）
- C2 max_worst_case_delta_R = -0.125（R 单位区间的 1/8，不是 −0.214 的填充）
- C3 min_mean_delta_R = 0.02（平均至少改善 R 单位区间的 0.02，才值得改默认 planner）
- C4 min_mean_U_gain = 0.0
- C5 scenario_degradation_policy = allow_some
- C6 require_all_scenarios_noninferior = False
- C7 independent_holdout_required = True
- C8 required_independent_batches = 2

理由：Gate-1 已经在 margin 不足时退回 LOCAL。系统级部署仍要求有限恶化、最坏损失有界、平均有正收益，
并且用 **2** 个额外独立 batch 确认这是系统性质而不是单表结果。

## 人工选择表

| Candidate | 核心原则 | 单场景风险 | 总体收益要求 | 独立验证要求 |
|-----------|----------|------------|--------------|--------------|
| A | Strict Safety | 不允许任何恶化；worst-case ΔR ≥ 0 | 平均 ΔR ≥ 0 且平均 U 不低于 LOCAL | 需要 1 个新 independent batch |
| B | Balanced | 最多 1/100 恶化，且单场景 ΔR ≥ −0.10 | 平均 ΔR ≥ 0 且平均 U 不低于 LOCAL | 需要 1 个新 independent batch |
| C | Conservative + Fallback | 最多 2/100 恶化，且单场景 ΔR ≥ −0.125 | 平均 ΔR ≥ 0.02 且平均 U 不低于 LOCAL | 需要 2 个新 independent batch |

本报告不选择 candidate。最终 deployment criterion 必须由项目负责人人工确认。

## 最终状态

- deployment_criterion = HUMAN_SPECIFICATION_REQUIRED
- planner_ready = False
- status = BLOCKED_ON_HUMAN_SPECIFICATION
- official_criterion = None
- 未写入 deployment_criterion_registry.json

本阶段停止。不训练、不调参、不改 B2、不改 τ、不改 planner、不改 decide()、
不重新计算 64 个 scenario、不宣布 PASS、不进入 Phase-4E。
