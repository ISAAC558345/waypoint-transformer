"""Chinese M1.1 report filled from the trial summary."""

from __future__ import annotations

from pathlib import Path


def _n(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write_m11_report(summary: dict, path: Path) -> None:
    counts = summary["counts"]
    answers = summary["answers"]
    lines = [
        "# 主动功耗辨识 M1.1 报告",
        "",
        "- 分支：`research/ocean-observation-isolated`",
        f"- 指令提交：`{summary['parent_commit']}`",
        f"- 审查过的 M1 提交：`{summary['reviewed_m1_commit']}`",
        f"- 结果目录：`{summary['result_dir']}`",
        f"- 接口判定：{summary['interface_status']}",
        f"- 场景判定：{summary['scenario_status']}",
        "",
        "M1 的 32 条核心航次仍在 `results/active_model_identification_m1/m1_20261113/`。本报告不覆盖那些文件，也不把旧回传零推进下的效用当作修正后的证据。",
        "",
        "## 核算",
        "",
        summary["accounting_note"],
        "",
        f"- 必要测试：{summary['tests']['status']}",
        f"- 积分设置：参数节点 {summary['integral']['theta_nodes']}，未来噪声节点 {summary['integral']['noise_nodes']}，并列带宽 {_n(summary['integral']['tie_abs'], 4)}",
        f"- 开发状态上的最大 Q 差：{_n(summary['integral']['measured_q_gap'], 6)}",
        f"- 终端近似：{summary['terminal_note']}",
        "",
        "## 运行计数",
        "",
        f"- 核心航次：{counts['core_missions']}",
        f"- 信息消融续跑：{counts['ablation_branches']}",
        f"- 核心物理步：{counts['physical_steps']}",
        f"- 消融物理步：{counts['ablation_physical_steps']}",
        f"- 规划步：{counts['planning_steps']}",
        f"- 规划副本：{counts['planning_copies']}",
        f"- 积分开发规划步：{counts['development_planning_steps']}",
        f"- 预算探测物理步：{counts['probe_physical_steps']}",
        f"- 返航失败事件：{counts['return_failures']}",
        f"- 训练更新：{counts['training_updates']}",
        "",
        "## 同规划器配对",
        "",
        summary["mechanism_note"],
        "",
    ]
    if int(counts["core_missions"]) == 0:
        diagnosis = summary.get("diagnosis") or {}
        if diagnosis:
            lines.extend(
                [
                    f"- 修正后 theta=1 的校准加远航程链：{_n(diagnosis['chain_theta_1_j'], 1)} J",
                    f"- 按登记公式得到的敏感电量：{_n(diagnosis['sensitive_battery_j'], 1)} J；先验 0.95 分位 {_n(diagnosis['guard_theta'], 3)} 下允许高价值服务：是",
                    f"- 按登记公式得到的不敏感电量：{_n(diagnosis['insensitive_battery_j'], 1)} J",
                    f"- M1 旧敏感电量 {_n(diagnosis['old_sensitive_battery_j'], 1)} J 在修正模型下不再送达该链，剩余约 {_n(diagnosis['old_chain_energy_left_j'], 1)} J",
                    f"- 等宽 7 点在窄后验上的均值：{_n(diagnosis['equal_width_narrow_mean'], 5)}；等概率节点的均值误差在窄后验上可忽略",
                    f"- 参数节点 7 与 15 都没有达到预先写下的方差容差 {diagnosis['variance_tol']}；规则因此保留最细的登记候选 15，而不是再加大节点去凑任务收益",
                    "",
                ]
            )
        lines.extend(
            [
                "16 条核心航次和信息消融都没有运行。上面不把缺失对照写成零收益。",
                "",
                "## 复现",
                "",
                "在仓库根目录执行，并把 `src` 放进 `PYTHONPATH`。",
                "",
                "```text",
                summary["commands"]["tests"],
                summary["commands"]["trial"],
                "```",
                "",
                "本轮没有统计显著或泛化结论。前瞻仍是单决策间隔加解析终端的近似。",
                "",
                "## 下一步",
                "",
                summary["next_step"],
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.extend(
        [
        f"- 敏感场景净效用差（开校准减关校准）均值：{_n(answers['sensitive_utility_gap'])}",
        f"- 不敏感场景净效用差均值：{_n(answers['insensitive_utility_gap'])}",
        f"- 敏感场景按时比特差均值：{_n(answers['sensitive_bits_gap'], 1)}",
        f"- 敏感场景能耗差均值：{_n(answers['sensitive_energy_gap_j'], 1)} J",
        f"- 首个服务浮标不同的敏感配对数：{answers['sensitive_choice_differences']} / {answers['sensitive_pairs']}",
        f"- 开校准在敏感场景的平均校准时间：{_n(answers['on_sensitive_cal_s'], 1)} 秒",
        f"- 开校准在不敏感场景的平均校准时间：{_n(answers['on_insensitive_cal_s'], 1)} 秒",
        f"- 关校准的校准时间：{_n(answers['off_cal_s'], 1)} 秒",
        f"- 实际校准快照：{answers['calibration_snapshots']}",
        f"- 消融后续净效用差（保留更新减撤去更新）均值：{_n(answers['ablation_utility_gap'])}",
        "",
        "### 方法均值",
        "",
        ]
    )
    for row in summary["method_means"]:
        lines.append(
            f"- {row['method']} / {row['condition']}：净效用 {_n(row['utility'])}，按时比特 {_n(row['on_time_bits'], 1)}，"
            f"耗能 {_n(row['energy_kj'], 3)} kJ，校准 {_n(row['calibration_s'], 1)} 秒，"
            f"RMSE {_n(row['rmse'])}，90% 覆盖 {_n(row['coverage'])}，"
            f"未返航 {row['not_returned']}，未降落 {row['not_landed']}，低于余量 {row['below_reserve']}，"
            f"能量耗尽 {row['energy_exhausted']}。"
        )
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "在仓库根目录执行，并把 `src` 放进 `PYTHONPATH`。",
            "",
            "```text",
            summary["commands"]["tests"],
            summary["commands"]["trial"],
            "```",
            "",
            "16 条核心航次和最多 8 条续跑是机制小试，不构成统计显著或泛化结论。前瞻仍是单决策间隔加解析终端的近似。",
            "",
            "## 下一步",
            "",
            summary["next_step"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
