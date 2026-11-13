"""Chinese M1 report filled from the trial summary."""

from __future__ import annotations

from pathlib import Path


def _n(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _yn(flag: bool) -> str:
    return "是" if flag else "否"


def write_report(summary: dict, path: Path) -> None:
    answers = summary["answers"]
    mechanism = "样本里有任务选择和校准时间差异；敏感场景的平均净效用没有显示额外校准的成本被交付收益抵消。"
    if not any(
        (
            answers["information_changes_choice"],
            answers["insensitive_reduces_calibration"],
            answers["calibration_adds_information"],
            answers["benefit_offsets_cost"],
        )
    ):
        mechanism = "这组样本没有显示额外校准带来交付收益。"
    lines = [
        "# 主动功耗辨识 M1 小试报告",
        "",
        f"- 分支：`research/ocean-observation-isolated`",
        f"- 父提交：`{summary['parent_commit']}`",
        f"- 结果目录：`{summary['result_dir']}`",
        f"- 核心任务：{summary['counts']['core_missions']} 条（8 个场景 × 4 个方法）",
        f"- 固定接收点诊断：{summary['counts']['fixed_receiver_diagnostics']} 条",
        f"- 实际执行任务：{summary['counts']['executed_tasks']} 条",
        f"- 直传核算：{summary['counts']['direct_accountings']} 条，不计入环境任务",
        f"- 物理步：{summary['counts']['physical_steps']}",
        f"- 规划副本步：{summary['counts']['planning_steps']}",
        f"- 规划副本次数：{summary['counts']['planning_copies']}",
        f"- 返航失败：{summary['counts']['return_failures']}",
        "",
        "本轮没有训练神经网络，没有加载旧模型权重。前瞻是单决策间隔加解析终端的坐标分离近似，不是精确贝叶斯最优。小试样本不构成统计显著或泛化结论。",
        "",
        "## 接口与机制",
        "",
        f"- 必要测试：{summary['tests']['status']}",
        f"- 接口判定：{summary['interface_status']}",
        f"- 机制判定：{mechanism}",
        "",
        "接口通过只说明计账、更新和规划副本按合同工作。机制是否有效看下面四个问题的样本差，不把接口通过写成调度收益。",
        "",
        "## 四个问题",
        "",
        f"1. 额外校准是否带来被动遥测之外的信息：{_yn(answers['calibration_adds_information'])}。固定校准首个间隔的名义激励积分均值为 {_n(answers['fixed_calibration_first_z_j'], 1)} J，被动方法为 {_n(answers['passive_first_z_j'], 1)} J。敏感场景结束时，固定校准的平均后验方差为 {_n(answers['fixed_calibration_end_variance'], 6)}，被动为 {_n(answers['passive_end_variance'], 6)}。",
        f"2. 信息是否改变任务选择：{_yn(answers['information_changes_choice'])}。敏感场景里，有限前瞻与被动方法的首个服务浮标不同的配对数为 {answers['sensitive_choice_differences']} / {answers['sensitive_pairs']}。",
        f"3. 任务收益是否抵消校准成本：{_yn(answers['benefit_offsets_cost'])}。敏感场景上，有限前瞻减被动的净效用样本差均值为 {_n(answers['sensitive_utility_gap'], 4)}。这不是置信区间。",
        f"4. 决策不敏感时是否减少不必要校准：{_yn(answers['insensitive_reduces_calibration'])}。有限前瞻在不敏感场景的平均校准时间为 {_n(answers['lookahead_insensitive_cal_s'], 1)} 秒，敏感场景为 {_n(answers['lookahead_sensitive_cal_s'], 1)} 秒；固定校准在不敏感场景仍为 {_n(answers['fixed_calibration_insensitive_cal_s'], 1)} 秒。",
        "",
        "## 方法均值",
        "",
    ]
    for row in summary["method_means"]:
        lines.append(
            f"- {row['method']} / {row['condition']}：净效用 {_n(row['utility'], 4)}，按时比特 {_n(row['on_time_bits'], 1)}，耗能 {_n(row['energy_kj'], 3)} kJ，校准 {_n(row['calibration_s'], 1)} 秒 / {_n(row['calibration_kj'], 3)} kJ，参数 RMSE {_n(row['rmse'], 4)}，90% 区间覆盖 {_n(row['coverage'], 3)}，返航失败 {row['return_failures']}。"
        )
        lines.extend(
            [
                "",
                "## 直传与固定接收点",
                "",
                f"同一短距业务合同下，直传交付 {summary['diagnostics_overview']['direct_delivered_bits']} 比特，固定接收点交付 {summary['diagnostics_overview']['fixed_receiver_delivered_bits']} 比特。LoRa 能否对岸闭合只作为未计分旁注，不把旁注比特算进交付。",
                "",
                "## 复现",
                "",
                "在仓库根目录执行，并把 `src` 放进 `PYTHONPATH`。",
                "",
                "```text",
                "set PYTHONPATH=src",
                summary["commands"]["tests"],
                summary["commands"]["trial"],
                "```",
                "",
                "预算探针使用 10 MJ 哨兵电量，只用于在试验前测量航程代价，没有计入 32 条核心任务。",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
