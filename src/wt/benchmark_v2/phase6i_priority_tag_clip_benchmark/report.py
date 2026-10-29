"""Chinese report for the authorized full-budget clip comparison."""
from __future__ import annotations

from .protocol import GATE_PASS, REPORT


def _num(value) -> str:
    if value is None:
        return "空"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def _answer(summary: dict) -> str:
    gate = summary.get("scientific_gate")
    if gate == GATE_PASS:
        return "分组裁剪提高了实际 tagged 服务，并守住了预登记的 payload、SINR 和干扰条件。通过该标准不证明总 payload 显著高于 L0。"
    if gate == "CLIP_GROUP_BENCHMARK_FAIL":
        return "分组裁剪没有通过预登记的性能标准，因此不能说它提高了实际 tagged 服务并守住 payload。"
    if gate == "CLIP_GROUP_BENCHMARK_INCONCLUSIVE":
        return "现有开发场景证据不足以判定分组裁剪提高了实际 tagged 服务并守住 payload。"
    return "对照没有完成，不能回答分组裁剪是否提高了实际 tagged 服务并守住 payload。"


def _seed_lines(summary: dict) -> list[str]:
    lines = []
    for seed in summary.get("score", {}).get("seeds", []):
        joint = seed["versus_full_j"]
        base = seed["versus_l0"]
        lines.append(
            f"- seed {seed['model_seed']}: 相对 FULL_J 的 tagged 均值 {_num(joint['tagged']['mean'])}，"
            f"CI 下界 {_num(joint['tagged']['ci_low'])}；相对 L0 的 tagged 均值 {_num(base['tagged']['mean'])}，"
            f"CI 下界 {_num(base['tagged']['ci_low'])}；payload 均值相对 FULL_J {_num(joint['payload']['mean'])}，"
            f"相对 L0 {_num(base['payload']['mean'])}；安全差 FULL_J/L0 "
            f"{_num(joint['safety_diff_sum'])}/{_num(base['safety_diff_sum'])}；"
            f"约束差 FULL_J/L0 {_num(joint['constraint_diff_sum'])}/{_num(base['constraint_diff_sum'])}；"
            f"本种子标准 {'满足' if seed['passes_preregistered_criteria'] else '不满足'}。"
        )
    if not lines:
        lines.append("- 没有完成的开发场景评分。")
    return lines


def write_report(summary: dict) -> None:
    comparison = summary.get("test_comparison") or {}
    frozen = summary.get("frozen_gates") or {}
    lines = [
        "# 裁剪分组全预算对照",
        "",
        f"- direction_status: `{summary.get('direction_status')}`",
        f"- PPO 更新: {summary.get('ppo_updates')}",
        f"- Adam 步: {summary.get('adam_steps')}",
        f"- 评估回合: {summary.get('evaluation_episodes')}",
        f"- scientific_gate: `{summary.get('scientific_gate')}`",
        "- release_eligible: false",
        f"- 性能问题: {_answer(summary)}",
        "",
        "用户已选择 A。本次从共享 LEGACY 初始化全新训练 FULL_J 与 FULL_S，各 3 个种子、每组 300 次 PPO 更新。没有从 32 次试点 checkpoint 续跑。奖励、种子、总梯度上界和判定门槛都保持预登记值。",
        "",
        f"E12 仍是 `{frozen.get('e12_gate')}`。E13 仍是 `{frozen.get('e13_gate')}`。E14 仍是 `{frozen.get('e14_gate')}` / `{frozen.get('e14_effect')}`。`merged` 为 false。",
        "",
        "## 同一起点的开发场景对照",
        "",
        "FULL_S 对同种子 FULL_J 是裁剪分组对照，对 L0 是正式系统对照。区间来自按场景配对的 bootstrap，10000 次、种子 0、分位 2.5 和 97.5。开发池已经被多轮使用，这不是未见场景泛化。",
        "",
        * _seed_lines(summary),
        "",
        f"通过种子数: {summary.get('score', {}).get('passing_seeds')}",
        f"原因记录: {summary.get('reason')}",
        "",
        "## 训练安全事件",
        "",
        f"训练安全计数合计 {summary.get('training_safety_violation_count')}，约束失败合计 {summary.get('training_constraint_failure_count')}。",
        "E14 的 4 条事件没有位置和距离，记为未知，原计数保留，没有重训补造。E15 的新事件记录执行前后位置、距离、预测距离、原始动作、投影动作和数值余量。",
        "",
        "## 测试",
        "",
    ]
    if comparison:
        lines.extend(
            [
                f"- 干净基线测试数: {comparison.get('baseline_ran')}",
                f"- 加上本阶段后的测试数: {comparison.get('repository_ran')}",
                f"- 基线失败签名数: {comparison.get('baseline_failure_count')}",
                f"- 对照失败签名数: {comparison.get('repository_failure_count')}",
                f"- new_failure_count: {comparison.get('new_failure_count')}",
                f"- 相关测试通过: {comparison.get('related_ok')}",
                "- 既有失败仍然是发布阻塞项。完整测试没有写成全通过。",
            ]
        )
    else:
        lines.append("- 全套测试对照尚未写入本摘要。")
    lines.extend(
        [
            "",
            "## 停止",
            "",
            "本轮交付后按原停止规则结束。FAIL 或 INCONCLUSIVE 不改阈值、不换种子、不追加预算，也不把结果回写成奖励系数搜索依据。",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
