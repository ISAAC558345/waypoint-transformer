"""Chinese E3 report. Figures come only from the saved E2 rows."""
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


def _ids(values: list[object]) -> str:
    if not values:
        return "无"
    return "、".join(str(item) for item in values)


def _rho(value: object) -> str:
    if value in ("", None):
        return "无"
    return _fmt(value, 4)


def _combo_table(rows: list[dict]) -> str:
    header = "| 形式 | lambda_u | rho | bonus 均值 | P95 | 最大 | bonus/payload 均值 | bonus/payload P95 | bonus/penalty P50 | P95 | P99 | 近零惩罚步 | base实质场景 | payload实质场景 | 符号变化 | 保留比例 | 最差场景 | 八场景通过 |"
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    body = []
    for row in rows:
        kept = "不适用" if row["signal_retained_fraction"] in (None, "") else _fmt(row["signal_retained_fraction"])
        worst = "无" if row["worst_scenario_id"] in (None, "") else row["worst_scenario_id"]
        body.append(
            "| {form} | {lam} | {rho} | {mean} | {p95} | {maxv} | {payload} | {payload_p95} | {pen50} | {pen95} | {pen99} | {near} | {base_n} | {payload_n} | {sign} | {kept} | {worst} | {ok} |".format(
                form=row["form"],
                lam=_fmt(row["lambda_u"], 4),
                rho=_rho(row["rho"]),
                mean=_fmt(row["bonus_mean"]),
                p95=_fmt(row["bonus_p95"]),
                maxv=_fmt(row["bonus_max"]),
                payload=_fmt(row["bonus_over_payload_mean"]),
                payload_p95=_fmt(row["bonus_over_payload_p95"]),
                pen50=_fmt(row["bonus_over_penalty_p50"]),
                pen95=_fmt(row["bonus_over_penalty_p95"]),
                pen99=_fmt(row["bonus_over_penalty_p99"]),
                near=row["near_zero_penalty_steps"],
                base_n=row["scenarios_with_base_material_reorder"],
                payload_n=row["scenarios_with_payload_material_reorder"],
                sign=row["sign_flips"],
                kept=kept,
                worst=worst,
                ok="是" if int(row["all_scenarios_meet_conservative_standard"]) else "否",
            )
        )
    return "\n".join([header, rule, *body])


def _scenario_lines(summary: dict) -> str:
    lines = []
    for combo in summary.get("combinations", []):
        if combo["form"] == "A" and float(combo["lambda_u"]) == 0.0:
            continue
        failing = [str(item["scenario_id"]) for item in combo["scenarios"] if not item["meets_conservative_standard"]]
        weak = [str(item["scenario_id"]) for item in combo["scenarios"] if item["signal_too_weak"]]
        base_ids = [str(item["scenario_id"]) for item in combo["scenarios"] if item["material_order_changed"]]
        payload_ids = [str(item["scenario_id"]) for item in combo["scenarios"] if item["payload_material_order_changed"]]
        label = f"Form {combo['form']} lambda={_fmt(combo['lambda_u'], 4)} rho={_rho(combo['rho'])}"
        worst = "无" if combo["worst_scenario_id"] in (None, "") else combo["worst_scenario_id"]
        lines.append(
            f"- {label}：未通过场景 {_ids(failing)}；base reward 实质重排 {_ids(base_ids)}；"
            f"payload 实质重排 {_ids(payload_ids)}；信号过弱场景 {_ids(weak)}；最差场景 {worst}。"
        )
    return "\n".join(lines) if lines else "没有可列出的组合。"


def _row_lookup(summary: dict, form: str, lambda_u: float, rho: object) -> dict | None:
    for row in summary.get("pooled_rows", []):
        same_rho = (row["rho"] in ("", None) and rho is None) or (rho is not None and row["rho"] not in ("", None) and abs(float(row["rho"]) - float(rho)) < 1.0e-12)
        if row["form"] == form and abs(float(row["lambda_u"]) - float(lambda_u)) < 1.0e-12 and same_rho:
            return row
    return None


def _combo_lookup(summary: dict, form: str, lambda_u: float, rho: object) -> dict | None:
    for combo in summary.get("combinations", []):
        same_rho = (combo["rho"] is None and rho is None) or (
            combo["rho"] is not None and rho is not None and abs(float(combo["rho"]) - float(rho)) < 1.0e-12
        )
        if combo["form"] == form and abs(float(combo["lambda_u"]) - float(lambda_u)) < 1.0e-12 and same_rho:
            return combo
    return None


def _decision(summary: dict) -> str:
    if not summary.get("pooled_rows"):
        return "比较表尚未形成，不能回答四个决策问题。"
    small = _row_lookup(summary, "A", 0.0005, None)
    clipped = _row_lookup(summary, "B", 0.002, 0.001)
    small_combo = _combo_lookup(summary, "A", 0.0005, None)
    if small is None or clipped is None or small_combo is None:
        return "比较表缺少规定组合，不能回答四个决策问题。"
    payload_counts = []
    for lam in (0.0005, 0.001, 0.002, 0.05):
        row = _row_lookup(summary, "A", lam, None)
        if row is None:
            return "比较表缺少 Form A 参照组合，不能回答四个决策问题。"
        payload_counts.append(str(row["scenarios_with_payload_material_reorder"]))
    extra = _ids(
        [
            str(item["scenario_id"])
            for item in small_combo["scenarios"]
            if item["material_order_changed"] and not item["payload_material_order_changed"]
        ]
    )
    zero_urgency = int(small["positive_urgency_on_zero_payload_steps"])
    bound_rows = [row for row in summary["pooled_rows"] if row["form"] == "B"]
    clipped_rows = [
        row for row in bound_rows
        if row["signal_retained_fraction"] not in (None, "") and float(row["signal_retained_fraction"]) < 1.0 - 1.0e-12
    ]
    weak_rows = [combo for combo in summary.get("combinations", []) if combo["signal_too_weak"]]
    clipped_text = "、".join(
        f"lambda {_fmt(row['lambda_u'], 4)} rho {_fmt(row['rho'], 4)} 保留 {_fmt(row['signal_retained_fraction'])}"
        for row in clipped_rows
    ) or "无"
    positive_penalty = [
        row for row in summary["pooled_rows"]
        if row["form"] == "A" and float(row["lambda_u"]) > 0.0 and float(row["bonus_over_penalty_p95"]) > 1.0
    ]
    penalty_text = "全部正权重的总体 bonus/penalty P95 都大于 1" if len(positive_penalty) == 4 else "正权重的惩罚比例见上表"
    usable = list(summary.get("usable_candidates") or [])
    opening = "没有任何正权重组合能在八个场景同时满足预注册保守标准" if not usable else "有正组合满足预注册保守标准，但它们只是待批准候选"
    closing = "没有证据支持进入训练。标准没有放宽，也没有把某个较小组合称为最优。" if not usable else "满足标准的组合仍需负责人批准，不能称为最优。"
    return f"""1. {opening}。lambda 0 的数值界限平凡成立，但它不是正候选。0.0005 的总体 bonus/payload 均值是 {_fmt(small["bonus_over_payload_mean"])}，bonus/penalty P95 是 {_fmt(small["bonus_over_penalty_p95"])}。按 E2 的 payload 排序，0.0005、0.001、0.002、0.05 的实质重排场景数分别是 {"、".join(payload_counts)}。按 base reward 排序，0.0005 额外出现实质变化的场景是 {extra}。{penalty_text}。0.05 只作高风险参照。
2. Form B 的 cap 是否保持：{_fmt(summary.get("payload_caps_hold"))}。实际发生截断的组合是 {clipped_text}。lambda 0.002、rho 0.001 的保留比例是 {_fmt(clipped["signal_retained_fraction"])}，截断后 bonus/penalty P95 是 {_fmt(clipped["bonus_over_penalty_p95"])}。cap 只限制 bonus 相对本步 payload 的比例，不保证 bonus 小于投影惩罚或残差惩罚。
3. 信号过弱的预注册门槛是保留比例低于 0.5。低于该门槛的组合数是 {len(weak_rows)}。payload 为 0 且 urgency term 仍为正的步数是 {zero_urgency}。
4. {closing}`lambda_u` 与 `rho` 均未选。"""


def render_report(summary: dict) -> str:
    hashes = summary.get("source_sha256", {})
    hash_lines = "\n".join(f"- `{name}`：`{digest}`" for name, digest in hashes.items()) or "- 缺失"
    problems = summary.get("problems") or []
    problem_text = "无。" if not problems else "；".join(str(item) for item in problems)
    table = "尺度表尚未形成。" if not summary.get("pooled_rows") else _combo_table(summary["pooled_rows"])
    return f"""# E3 报告：加法紧急奖励与 payload 有界 bonus

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`{summary["instruction_commit"]}`
- 前置 Gate：`URGENCY_REWARD_SCALE_INCONCLUSIVE`
- 本阶段 Gate：`{summary["gate"]}`
- 原因：{summary["reason"]}
- `lambda_u` 与 `rho`：均未选。`lambda_selected = null`，`rho_selected = null`
- 性质：只重算 E2 已保存的固定轨迹。没有重新运行环境，没有重新抽样 tag，没有训练，没有修改正式 Planner、奖励、动作、checkpoint 或旧数据集。

## 1. 直接结论

{summary["conclusion"]}

这是固定轨迹上的离线奖励重算。它不表示换成该奖励后重新训练的策略会有同样的 payload、服务量或排序。不能把 cap 说成闭环性能保证，也不能从这些数字推出训练后 payload 不会下降。

## 2. 源文件

E2 结果提交是 `{summary["expected_e2_commit"]}`。四个源文件的最后提交与该提交一致：{_fmt(summary["source_commit_ok"])}。工作区没有改动这些文件：{_fmt(summary["source_clean"])}。记录数量、schema 和逐步字段复核通过：{_fmt(summary["source_ok"])}。

{hash_lines}

复核问题：{problem_text}

逐步记录 {summary.get("step_rows", "缺失")} 行，episode 记录 {summary.get("episode_rows", "缺失")} 行。Form A 与 E2 已保存的 lambda 表一致：{_fmt(summary.get("form_a_matches_e2"))}。各形式的累计 payload 完全相同：{_fmt(summary.get("payload_sums_identical_across_forms"))}。Form B 的 bonus/payload 没有超过对应 rho：{_fmt(summary.get("payload_caps_hold"))}。

## 3. 公式与字段

正式 base reward 保持不变：

`r_base = payload_term - 0.05 * c_project - 0.01 * c_residual`

E2 的 `payload_term` 是本步 `served_bits / 1e6`。同文件里的 `payload_bits` 是累计 payload，不用作本步 cap。`U_t = urgency_served_bits / 1e6`，对应列 `urgency_term`。

Form A：`r_A = r_base + lambda_u * U_t`。lambda 为 0、0.0005、0.001、0.002。0.05 只作高风险参照。

Form B：`r_B = r_base + min(lambda_u * U_t, rho * payload_term)`。rho 为 0.001、0.0025、0.005。lambda 为 0.0005、0.001、0.002。

本步 `payload_term = 0` 时，cap 为 0，bonus 也为 0。没有成功服务的步骤不会因为紧急度得到奖励。本次记录里，payload 为 0 且 urgency term 仍为正的步数见各组合的 `positive_urgency_on_zero_payload_steps`。

惩罚比例只在惩罚大于 `1e-12` 的步上计算。接近零的惩罚步单独计数，不用极小分母代替绝对 bonus。

## 4. 总体分布

保守标准在计算前固定，没有放宽：每个场景的 bonus/payload 均值不超过 1%，bonus/penalty 的 P95 不超过 1，并且 payload 项相差超过 1 bit 的步骤没有 base reward 排序变化。Form B 还要求相对未封顶 bonus 的保留比例不低于 0.5，否则视为 cap 把紧急度信号压得过弱。通过标准的组合只列为待批准候选，不称为最优。下表的“base实质场景”和“payload实质场景”分别是两种排序定义下发生变化的场景数。

{table}

## 5. 决策问题

{_decision(summary)}

## 6. 分场景

{ _scenario_lines(summary) }

最差场景按该组合 bonus/penalty 的 P95 选取。分场景的 bonus 分布、累计 payload、urgent-service bits、累计 bonus 和惩罚写在 `form_rows.csv`。同一条轨迹上，各形式的累计 payload 必须相同，因为策略没有重新执行。

## 7. 排序与限制

这里有两种排序，都只在同一条已保存轨迹上计算。payload 实质重排沿用 E2：比较 payload 项与 payload 项加 bonus，严格差距超过 1 bit 后方向反转才算。base 实质重排是本阶段额外要求：比较 `r_base` 与加上 bonus 后的奖励，payload 项相差超过 1 bit 的步骤对里，顺序反转或严格顺序变成平局都算。平局内部的顺序变化两种定义都不算实质重排。

这些数字不是不同奖励训练之后的策略表现。重新训练可能改变哪些步骤的惩罚很小，因此不能从固定轨迹上的比例推出闭环 payload 不会下降。

## 8. 停止

本阶段不进入 E4，不训练 L1/L2，不改正式奖励。最终 `lambda_u` 和 `rho` 仍等负责人确认。
"""
