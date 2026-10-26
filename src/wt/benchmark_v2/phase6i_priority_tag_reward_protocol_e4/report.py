"""Chinese E4 protocol review. Numbers are copied from the committed summaries."""
from __future__ import annotations


def _fmt(value: object, digits: int = 6) -> str:
    if value is None or value == "":
        return "缺失"
    if isinstance(value, bool):
        return "是" if value else "否"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 10.0 ** (-int(digits)):
        return f"{number:.3e}"
    if abs(number) >= 100.0:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _rho(value: object) -> str:
    if value in (None, ""):
        return "无"
    return _fmt(value, 4)


def _ids(values: list[object]) -> str:
    if not values:
        return "无"
    return "、".join(str(item) for item in values)


def _file_lines(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        if not row.get("present"):
            lines.append(f"- `{row.get('path')}`：缺失")
            continue
        lines.append(f"- `{row['path']}`：`{row.get('sha256')}`，提交 `{row.get('git_commit')}`")
    return "\n".join(lines) if lines else "- 缺失"


def _table(rows: list[dict]) -> str:
    header = "| 形式 | lambda_u | rho | bonus/payload 均值 | bonus/penalty P95 | payload 实质场景 | base 实质场景 | 保留比例 | 通过 E3 标准 |"
    rule = "|---|---|---|---|---|---|---|---|---|"
    body = []
    for row in rows:
        kept = "不适用" if row["signal_retained_fraction"] in (None, "") else _fmt(row["signal_retained_fraction"])
        body.append(
            "| {form} | {lam} | {rho} | {payload} | {penalty} | {payload_ids} | {base_ids} | {kept} | {ok} |".format(
                form=row["form"],
                lam=_fmt(row["lambda_u"], 4),
                rho=_rho(row["rho"]),
                payload=_fmt(row["bonus_over_payload_mean"]),
                penalty=_fmt(row["bonus_over_penalty_p95"]),
                payload_ids=_ids(row["payload_material_scenarios"]),
                base_ids=_ids(row["base_material_scenarios"]),
                kept=kept,
                ok="是" if row["all_scenarios_meet_conservative_standard"] else "否",
            )
        )
    return "\n".join([header, rule, *body]) if body else "E3 组合尚未读到。"


def render_report(summary: dict) -> str:
    problems = summary.get("problems") or []
    problem_text = "无。" if not problems else "；".join(str(item) for item in problems)
    candidate = summary["conditional_candidate"]
    evidence = candidate.get("evidence") or {}
    penalty = _fmt(evidence.get("bonus_over_penalty_p95"))
    payload = _fmt(evidence.get("bonus_over_payload_mean"))
    kept = _fmt(evidence.get("signal_retained_fraction"))
    payload_ids = evidence.get("payload_material_scenarios") or []
    payload_sentence = "中没有 payload 项实质重排" if not payload_ids else f"的 payload 项实质重排场景是 {_ids(payload_ids)}"
    table = _table(summary.get("cited_forms") or [])
    return f"""# E4 报告：紧急奖励准入标准与训练前协议审查

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`{summary["instruction_commit"]}`
- E2 结果提交：`{summary["e2_commit"]}`，Gate：`{summary["e2_gate"]}`
- E3 结果提交：`{summary["e3_commit"]}`，Gate：`{summary["e3_gate"]}`
- 本阶段 Gate：`{summary["gate"]}`
- E3 Gate 保持原值：{_fmt(summary["e3_gate_preserved"] == summary["e3_gate"]) }。没有改写成 READY，也没有放宽预注册标准。
- 条件候选：Form B，`lambda_u = 0.0005`，`rho = 0.001`，状态 `{candidate["status"]}`
- `lambda_selected = null`，`rho_selected = null`，`lambda_H = NOT_SELECTED`
- 性质：只复核已提交的报告、summary、schema 和正式奖励代码。没有重跑环境，没有重抽 tag，没有训练，没有修改正式 Planner、奖励、动作、checkpoint 或旧数据集。

## 1. 直接结论

审查材料可以核对，E3 的原结论保持 `URGENCY_REWARD_FORM_INCONCLUSIVE`。两种奖励形式都没有在预注册标准下通过。Form B 的 `lambda_u=0.0005, rho=0.001` 只作为后续受控试验的条件候选，状态是 `CONDITIONAL_CANDIDATE_NOT_APPROVED`。它没有被选定，也没有获准训练。

这是固定轨迹上的奖励尺度审查。它不表示训练后的策略会保持同样的 payload、服务量或排序。

复核问题：{problem_text}

## 2. 源文件与正式奖励

E2 与 E3 的数据、报告都对上各自的结果提交。正式 `worker.py`、正式 Planner 和冻结 checkpoint 的当前哈希与 E2 summary 中记录的 watched hash 一致。checkpoint 记录值是 `{summary.get("checkpoint_sha256")}`。

### E2

{_file_lines(summary.get("e2_files", []))}

### E3

{_file_lines(summary.get("e3_files", []))}

### E1 schema 与正式代码

{_file_lines([summary.get("e1_schema_file", {})] + list(summary.get("formal_files", [])))}

正式低层奖励仍是：

`{summary["formal_formula"]}`

其中 `payload_term = served_bits / 1e6`，投影系数是 0.05，残差系数是 0.01，`ACTION_SCALE = 1`。`phase6g` 协议只引用这些常数，没有另写一套系数。本阶段没有改这个函数。

## 3. E3 的原门槛

门槛在 E3 之前已经固定，本阶段原样保留：

- 每个场景的 bonus/payload 均值不超过 1%；
- bonus/penalty 的 P95 不超过 1；
- payload 项相差超过 1 bit 时，奖励排序没有实质变化；
- Form B 相对未封顶 bonus 的保留比例不低于 0.5。

E3 的记录是：384 条 step、8 条 episode，Form A 复现了 E2 的 lambda 表，各形式累计 payload 相同，Form B 的 cap 没有超过对应 rho，`usable_candidates` 为空。因此 E3 Gate 继续是失败于保守标准，不是通过。

## 4. 两种形式的尺度

Form A：`r_A = r_base + lambda_u * U_t`。`U_t = urgency_served_bits / 1e6`。lambda 只取 0、0.0005、0.001、0.002。0.05 是高风险参照。

Form B：`r_B = r_base + min(lambda_u * U_t, rho * payload_term)`。rho 只取 0.001、0.0025、0.005，也就是本步成功服务 payload 项的 0.1%、0.25%、0.5%。本步 `payload_term = 0` 时 bonus 为 0。

下表数字来自 E3 已保存的 summary，不是本阶段重新计算的。

{table}

E2 对 lambda 0.0005 的记录是 bonus/payload 均值约 0.000130，bonus/penalty P95 为 11.613369，payload 排序的实质重排场景数是 0。0.001、0.002、0.05 的 payload 实质重排场景数是 1、2、8。E3 读到的 payload 实质场景数与这些记录一致。

运动惩罚是 `0.05 * c_project + 0.01 * c_residual`，用来惩罚执行动作相对期望动作、以及残差动作本身。它是正则项。payload 项才是主要任务收益。bonus/penalty P95 大于 1，表示在冻结轨迹的许多步骤上，紧急度 bonus 已经大于这个正则项；对 0.0005 而言，总体 P95 是 {penalty}，同时它只约占 payload 项的 {payload}。这不自动等于“系统目标失败”，因为研究目标看的是 payload 是否仍是主项、紧急信息是否只拿到小比例额外奖励。它仍然是策略风险：正则项原来用于限制动作变化，bonus 在大量步骤上超过它，训练时仍可能把动作从原来的低惩罚方向拉开。这个风险保留。门槛本身不放宽，E3 也不改成通过。

## 5. 为什么有界 bonus 不能保证训练后的总 payload 不下降

Form B 的 cap 是单步约束：`0 <= bonus_t <= rho * payload_term_t`，并且 `payload_term_t = 0` 时 bonus 为 0。它限制的是“这一步已经成功服务的 payload 项”里面，紧急度最多再加多少奖励。

它没有约束策略会选择哪一条轨迹。训练优化的是新策略下的累计 `r_B`。新动作会改变服务量、惩罚和紧急度服务。cap 用的是新轨迹自己的本步 payload，所以一条总 payload 更低的轨迹仍然可以满足 cap。例如少服务一块较大的普通数据、多服务一块较小的高紧急度数据时，单步 bonus 仍不超过该步新的 payload 项，但 episode 总 payload 可以下降。cap 也不会奖励没有成功交付的步骤，因为它在零 payload 上是 0；这只避免给未交付数据发奖励，并不惩罚“本可以交付但没有交付”的 payload。

E2/E3 的累计 payload 在各形式之间相同，是因为动作没有重新执行。固定轨迹重算不能推出训练后的 payload、轨迹或服务量。只有训练后的独立 benchmark 能回答闭环 payload 是否下降。本阶段不把 cap 写成性能保证。

在已保存的 384 步上，lambda 0.0005 和 0.001 的未封顶 bonus 已经低于最小 rho 0.001，所以这些 Form B 行的保留比例是 1，数字与对应 Form A 相同。cap 的设计约束存在，但在这批轨迹上没有实际截掉 bonus。只有 lambda 0.002、rho 0.001 的保留比例低于 1，截断后惩罚 P95 仍大于 1。

## 6. 条件候选

建议后续若做小规模、独立的低层试验，优先使用 Form B，而不是无上限的加法奖励。理由只是设计：bonus 被限制在本步成功服务 payload 项的固定小比例内。这不是性能结论，也不是从固定轨迹里选出的最优组合。

`lambda_u=0.0005, rho=0.001` 是这条设计里的首个条件候选，因为 cap 是 payload 项的 0.1%，而且 E3 记录{payload_sentence}。它没有通过 E3：bonus/penalty P95 是 {penalty}，大于 1；base reward 实质重排场景是 {_ids(evidence.get("base_material_scenarios", []))}；保留比例是 {kept}。状态因此是 `CONDITIONAL_CANDIDATE_NOT_APPROVED`。网格不扩大。不允许同时改 observation、正式 reward、动作、SafetyProjector 或任务分布。新模型若将来获准训练，也只能是新 checkpoint，不能覆盖 L0。

高层 Scheme C 保持冻结，继续 payload-first。低层 bonus 不加入高层目标。紧急度沿用 E1 的 `synthetic_priority_bucket_v1`，tag 必须标成 synthetic。`lambda_H` 保持 `NOT_SELECTED`。本协议不要求另定高层 `delta_payload_max`，因为高层不允许为了紧急度牺牲实质 payload。

## 7. 需要负责人确认

1. 是否接受这条准则：训练后的独立 benchmark 如果出现总 payload 下降，就停止并回滚，不把该模型当成可用策略。
2. 是否批准把 `lambda_u=0.0005, rho=0.001` 作为唯一的首个受控低层训练候选。批准也只允许训练新模型，绝不覆盖 L0。
3. 若第 2 项未批准，则维持不训练。本阶段现在就处于这个状态。

以上三项都还没有答案。没有负责人的明确批准，就不能进入独立 observation/reward builder 的 smoke，也不能训练。

## 8. 停止

本阶段不进入 E5，不启动 L1/L2，不修改正式奖励。E3 Gate 保持 `URGENCY_REWARD_FORM_INCONCLUSIVE`。
"""
