"""Chinese M1.3 report. Unrun rewards stay blank rather than zero."""

from __future__ import annotations

from pathlib import Path


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "未测"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    number = float(value)
    if number != 0.0 and abs(number) < 10.0 ** (-digits):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _tier_lines(name: str, tier: dict) -> list[str]:
    if int(tier.get("pairs") or 0) == 0:
        return [f"- {name}：未运行。"]
    return [
        f"- {name}：{tier['pairs']} 对。净效用差（允许校准减关闭校准）{_fmt(tier['utility_gap'], 4)}。",
        f"  允许校准的平均净效用 {_fmt(tier['utility_on'], 4)}，关闭校准 {_fmt(tier['utility_off'], 4)}。",
        f"  按时比特差 {_fmt(tier['bits_gap'], 1)}，能量差 {_fmt(tier['energy_gap_j'], 1)} J。",
        f"  平均校准时间：允许 {_fmt(tier['calibration_s_on'], 1)} s，关闭 {_fmt(tier['calibration_s_off'], 1)} s。",
        f"  首次服务目标不同的基础场景：{_fmt(tier['choice_differences'])}。",
    ]


def _answers(summary: dict) -> list[str]:
    paired = summary["paired"]
    tight = paired["tight"]
    window = paired["window"]
    loose = paired["loose"]
    numerical = summary["numerical_status"]
    sensitive = summary.get("numerically_sensitive")
    rescore = summary["rescore"]
    lines = ["## 五个问题", ""]
    if int(tight.get("pairs") or 0) == 0:
        lines.append("- tight 上校准是否值得：对照未运行。")
    else:
        gap = tight["utility_gap"]
        lines.append(
            f"- tight 上校准是否值得：允许校准相对关闭校准的平均净效用差是 {_fmt(gap, 4)}。"
            f"首次服务目标不同的基础场景有 {_fmt(tight['choice_differences'])} 个。"
            "差为正表示这四条样本里允许校准的净效用更高，差为负表示紧预算上校准没有带来更高净效用。"
        )
    if not summary["budget"]["window_valid"]:
        lines.append("- 中间档是否把信息变成任务改变：window 没有形成 L < U 的区间，该档没有飞行。")
    elif int(window.get("pairs") or 0) == 0:
        lines.append("- 中间档是否把信息变成任务改变：window 对照未运行。")
    else:
        lines.append(
            f"- 中间档是否把信息变成任务改变：净效用差 {_fmt(window['utility_gap'], 4)}，"
            f"{_fmt(window['choice_differences'])} 个基础场景的首次服务目标不同。"
            f"允许校准的平均校准时间是 {_fmt(window['calibration_s_on'], 1)} s。"
        )
    if int(loose.get("pairs") or 0) == 0:
        lines.append("- loose 是否避免无谓校准：对照未运行。")
    else:
        lines.append(
            f"- loose 是否避免无谓校准：允许校准的平均校准时间是 {_fmt(loose['calibration_s_on'], 1)} s，"
            f"关闭校准是 {_fmt(loose['calibration_s_off'], 1)} s，净效用差 {_fmt(loose['utility_gap'], 4)}。"
        )
    if int(summary["counts"]["ablation_branches"]) == 0:
        lines.append("- 信息消融是否支持信息本身：没有 window 校准快照，消融次数为 0。")
    else:
        lines.append(
            f"- 信息消融是否支持信息本身：保留更新减去掉更新的平均净效用差是 {_fmt(paired['ablation_utility_gap'], 4)}，"
            f"续跑 {summary['counts']['ablation_branches']} 条。"
        )
    lines.append(
        f"- 结论对数值评分有多敏感：数值状态 {numerical}，开发最大 |ΔQ| = {_fmt(summary['integral'].get('measured_q_gap'), 6)}。"
        f"再评分检查了 {rescore['snapshots']} 个关键快照，其中 {rescore['rank_changes']} 个动作排名改变。"
        f"数值敏感标记：{_fmt(sensitive)}。"
        "这 16 或 24 条航次是机制样本，不是显著性检验，也不外推到其他几何或风场。"
    )
    return lines


def write_m13_report(summary: dict, path: Path) -> None:
    budget = summary["budget"]
    counts = summary["counts"]
    integral = summary["integral"]
    first = budget["uavs"][0]
    lines = [
        "# 主动功耗辨识 M1.3 结果",
        "",
        f"交接提交 `{summary['parent_commit']}`。运行时 HEAD `{summary['git']['head']}`。结果目录 `{summary['result_dir']}`。",
        "",
        "## 状态",
        "",
        f"- 单元测试：{summary['unit_tests_status']}",
        f"- 成本核对：{summary['accounting_status']}",
        f"- 数值积分：{summary['numerical_status']}",
        f"- 场景：{summary['scenario_status']}，window：{budget['window_status']}",
        f"- 同规划器对照：{summary['comparison_status']}",
        f"- 评分方案：{summary['scoring_mode']}",
        "",
        "单元测试通过不等于数值或机制通过。未运行的差写为未测。已经跑出来的零差才写 0。",
        "",
        "## 三档预算",
        "",
        f"tight 仍是 {_fmt(budget['tight_battery_j'][0], 6)} J，两架相同。",
        f"loose 复用 M1.2 登记向量 {_fmt(budget['loose_battery_j'][0], 6)} J，两架相同。",
        f"设计参数 theta_ref = {_fmt(budget['theta_ref'], 1)}，epsilon_design = {_fmt(budget['epsilon_design_j'], 4)} J。",
        f"UAV0：L = {_fmt(first['L_j'], 4)} J，U = {_fmt(first['U_j'], 4)} J，window = {_fmt(first.get('window_battery_j'), 4)} J。",
        f"window 有效：{_fmt(budget['window_valid'])}。posterior_zero_admits = {_fmt(budget['posterior_zero_admits'])}，它不决定 tight 或 loose 是否飞行。",
        budget["limiting_condition"] or "中间档区间成立，三档使用同一预算向量。",
        "",
        "## 数值",
        "",
        integral["selection"],
        f"选用参数节点：{_fmt(integral.get('theta_nodes'))}。分段内层节点：{_fmt(integral.get('segment_inner_nodes'))}。",
        f"开发物理展开 {counts['development_rollouts']} 次，规划步 {counts['development_planning_steps']}，缓存命中 {counts['development_cache_hits']}，求根 {counts['development_root_solves']}，代数积分 {counts['development_algebraic_integrals']}，代数时间 {_fmt(counts['development_algebraic_seconds'], 1)} s。",
        f"预算打满：{_fmt(integral.get('budget_exhausted'))}。开发排名变化：{_fmt(integral.get('development_rank_changed'))}。",
        "",
        "## 对照",
        "",
    ]
    lines.extend(_tier_lines("tight", summary["paired"]["tight"]))
    lines.extend(_tier_lines("window", summary["paired"]["window"]))
    lines.extend(_tier_lines("loose", summary["paired"]["loose"]))
    lines.extend(
        [
            "",
            f"核心航次 {counts['core_missions']}。消融续跑 {counts['ablation_branches']}。设计探针区间 {counts['design_probe_intervals']}。",
            f"主试规划展开 {counts['mission_rollouts']}，主试物理步 {counts['mission_physical_steps']}。",
            f"消融规划展开 {counts['ablation_rollouts']}。再评分展开 {counts['rescore_rollouts']}，上限是否打满：{_fmt(summary['rescore']['cap_hit'])}。",
            "",
        ]
    )
    lines.extend(_answers(summary))
    lines.extend(
        [
            "",
            "## 保留",
            "",
            "M1、M1.1、M1.2 的结果、锁和隔离清单没有改写。本轮没有并入主分支、默认分支或 legacy 维护分支，也没有训练神经网络或增加风浪。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
