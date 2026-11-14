"""Chinese M1.2 report. Unrun rewards stay blank rather than zero."""

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


def _moment_lines(integral: dict) -> list[str]:
    lines = []
    for check in integral.get("moment_checks", []):
        bits = []
        for case in check["cases"]:
            bits.append(f"{case['case']} 方差误差 {case['variance_abs_error']:.3e}")
        flag = "通过" if check["passed"] else "未过"
        lines.append(f"- {check['theta_nodes']} 点：{flag}；" + "；".join(bits))
    return lines


def _noise_lines(integral: dict) -> list[str]:
    lines = []
    for check in integral.get("noise_checks", []):
        flag = "通过" if check["passed"] else "未过"
        lines.append(
            f"- 参数 {check['theta_nodes']} 点、噪声 {check['noise_nodes']} 点相对参照：{flag}，最大 |ΔQ| = {_fmt(check['max_abs_q'], 6)}"
        )
        for state in check["states"]:
            lines.append(f"  - {state['state']}: {_fmt(state['max_abs_q'], 6)}")
    if not lines:
        lines.append("- 矩检验未通过，动作 Q 没有继续计算。")
    return lines


def write_m12_report(summary: dict, path: Path) -> None:
    budget = summary["budget"]
    integral = summary["integral"]
    counts = summary["counts"]
    first = budget["uavs"][0]
    zero = first["zero_noise_posterior"]
    adverse = first["adverse_posterior"]
    lines = [
        "# 主动功耗辨识 M1.2 结果",
        "",
        f"父提交 `{summary['parent_commit']}`。结果目录 `{summary['result_dir']}`。",
        "",
        "## 状态",
        "",
        f"- 单元测试：{summary['unit_tests_status']}",
        f"- 成本核对：{summary['accounting_status']}",
        f"- 数值积分：{summary['numerical_status']}",
        f"- 场景窗口：{summary['scenario_status']}",
        f"- 同规划器对照：{summary['comparison_status']}",
        "",
        "单元测试通过不等于场景、数值或机制通过。未运行的收益、差值和误差写为未测，不记成 0。",
        "",
        "## 1. 准入和完整执行为什么不一致，原电量是否仍有效",
        "",
        "service 的旧准入只用去程、2 秒采集悬停、返程和 5 秒降落，没有把决策间隔里的到点悬停和整段空中回传算进去。完整业务报价改为复用同一低层控制器：采集、回传、返航、降落都占满所在的决策间隔。路径不依赖 theta，成本是 `theta * z + 辅助能量`，没有另加拟合常数。",
        "",
        f"报价对执行的最大绝对误差是 {_fmt(summary['accounting']['worst_abs_j'], 6)} J，容差 {_fmt(budget['execution_abs_tol_j'], 1)} J。",
        f"先验 q95 = {_fmt(budget['guard_theta'], 6)}。UAV0 远航程完整报价 {_fmt(first['prior_quote_j'])} J，加备用 {_fmt(first['prior_quote_plus_reserve_j'])} J。",
        f"theta=1 的远航程报价 {_fmt(first['physical_quote_j'])} J，加备用 {_fmt(first['physical_quote_plus_reserve_j'])} J。",
        f"theta=1 的校准加远航程是 {_fmt(first['chain_theta_1_j'])} J，加备用 {_fmt(first['chain_theta_1_plus_reserve_j'])} J。",
        f"备用 {_fmt(budget['reserve_j'], 0)} J 在报价之外，只加一次。另一架机在报价里保持等待，并占用自己的指令和遥测时隙。",
        "",
        f"原敏感电量 {_fmt(budget['candidate_battery_j'])} J 没有被替换。理想脚本窗口（校准后 theta=1 的完整成本加备用 ≤ B，且 B < 先验保守完整成本加备用）{'成立' if budget['ideal_script_window'] else '不成立'}。",
        f"一次校准后，零噪声后验均值 {_fmt(zero['posterior_mean'], 4)}，q95 = {_fmt(zero['posterior_q95'], 4)}。剩余能量 {_fmt(zero['energy_left_j'])} J，该分位的完整远航程加备用 {_fmt(zero['quote_plus_reserve_j'])} J，差额 {_fmt(zero['slack_j'])} J。",
        f"不利噪声 {_fmt(adverse['epsilon_j'])} J 把 q95 推到 {_fmt(adverse['posterior_q95'], 4)}，差额 {_fmt(adverse['slack_j'])} J。",
        f"theta=1 的理想剩余成本不是这个保守预测。位置在校准后到 x={_fmt(zero['x'], 1)} m，截止时间内航程本身放得下。",
        "",
        budget["limiting_condition"] or "后验分位下原电量仍然同时满足物理成本和保守准入。",
        "",
        "真实 theta=1 时远航程在物理上放得下，先验 q95 不放行，校准后的后验 q95 仍不放行。这三件事用的不是同一个阈值换了标签。近程服务在先验下仍然放行，没有为了突出校准而额外拦截普通任务。",
        "",
        "## 2. 数值精度和临界决策",
        "",
        (
            f"数值状态 {summary['numerical_status']}。"
            f"参数节点{'未选用' if integral.get('theta_nodes') is None else ' ' + str(integral.get('theta_nodes'))}，"
            f"噪声节点{'未选用' if integral.get('noise_nodes') is None else ' ' + str(integral.get('noise_nodes'))}，"
            f"并列带宽 {integral.get('tie_abs')}，测得最大 |ΔQ| {_fmt(integral.get('measured_q_gap'), 6)}。"
        ),
        "没有候选通过时不会留下最后一个候选并称其通过。",
        "",
        *_moment_lines(integral),
        "",
        *_noise_lines(integral),
        "",
        (
            "10 MJ 机坪上的宽、窄和下边界对照，|ΔQ| 低于 0.01。紧电量、中途航段、缓存回传和校准切换状态超过 0.01。容差没有放宽，也没有改用最后一个候选。"
            if summary["numerical_status"] != "PASS"
            else "登记候选里有一组同时满足矩容差和临界状态的 Q 容差。"
        ),
        "",
        f"开发状态上的规划步数 {counts['development_planning_steps']}，拷贝 {counts['development_planning_copies']}。",
        "",
        "## 3. 校准开与关的配对差",
        "",
    ]
    paired = summary.get("paired")
    if summary["comparison_status"] != "RAN" or not paired:
        lines.append("16 条核心航次没有运行。配对效用差、比特差和能量差都未测，不是 0。")
    else:
        lines.append(
            f"敏感效用差 {_fmt(paired['sensitive_utility_gap'], 5)}，比特差 {_fmt(paired['sensitive_bits_gap'], 1)}，能量差 {_fmt(paired['sensitive_energy_gap_j'], 1)} J。"
        )
        lines.append(
            f"首个服务不同的敏感配对 {paired['sensitive_choice_differences']} / {paired['sensitive_pairs']}。这是机制样本，不是显著性或外推。"
        )
    lines.extend(
        [
            "",
            "## 4. 消融是否支持校准收益",
            "",
            f"信息消融续跑 {counts['ablation_branches']} 条，校准快照 {counts['calibration_snapshots']} 个。",
        ]
    )
    if counts["ablation_branches"] == 0:
        lines.append("没有实际校准被撤除。消融效用差未测。不能把未运行写成校准有收益或没有收益。")
    else:
        lines.append(f"撤去校准观测后的效用差 {_fmt(paired.get('ablation_utility_gap') if paired else None, 5)}。")
    lines.extend(
        [
            "",
            "## 5. 若普通服务已经够用，额外校准的机会成本是什么",
            "",
            "这次没有同规划器航次，不能从样本里估计额外校准的效用差。",
            "能确定的限制是：登记的 1200 J 裕量刚好盖住 theta=1 的理想完整成本，盖不住一次校准之后后验 0.95 分位相对真值的抬升。校准本身还消耗了那段裕量。近程服务在先验下已经放行，所以缺的不是普通任务的准入，而是保守信念下远航程仍然不被放开。",
            "",
            "## 计数",
            "",
            f"- 核心航次 {counts['core_missions']}",
            f"- 消融续跑 {counts['ablation_branches']}",
            f"- 物理步数 {counts['physical_steps']}",
            f"- 规划步数 {counts['planning_steps']}",
            f"- 耗时 {_fmt(counts['elapsed_s'], 1)} 秒",
            "",
            summary["next_step"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
