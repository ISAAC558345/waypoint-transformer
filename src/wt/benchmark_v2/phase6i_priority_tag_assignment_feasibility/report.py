"""Chinese report for the frozen assignment diagnosis."""
from __future__ import annotations

from .protocol import E15_GATE, E16_GATE, REPORT


def _suggestion(summary: dict) -> str:
    if summary["scientific_gate"] != "ASSIGNMENT_FEASIBILITY_COMPLETE" or not summary["opportunity_published"]:
        return "这次核对没有给出可执行的新训练。不要把已经结案的裁剪分组换一个名字重开，也不要扩大场景或步数预算。"
    label = summary["opportunity_evidence"]
    if label == "REPLICATED_LOCAL_OPPORTUNITY":
        return "至少两个训练场景在注册的 5 个候选帽里出现了同时守住 payload、原始安全和通信条件的已知标签服务机会。若要检验它能否成为规则，需要另开预登记，并且仍然冻结低层、不使用事后最优候选。这次不启动该对照。"
    if label == "LOCAL_OR_CONSTRAINED_OPPORTUNITY":
        return "注册样本里看到了已知标签服务机会，但没有在两个场景上同时通过安全或通信条件，或者覆盖不完整。不要把事后最优候选拿去训练。"
    if label == "NOT_OBSERVED_IN_REGISTERED_POOL":
        return "这 8 个训练场景的注册候选帽内没有观察到已知标签服务机会。这不证明任务分配在所有场景都无效，也不构成重开裁剪实验的理由。"
    return "有效覆盖不足。先处理核对缺口，不开始新训练。"


def write_report(summary: dict) -> None:
    counts = summary["opportunity_counts"]
    lines = [
        "# 冻结低层任务分配可行性诊断",
        "",
        f"- direction_status: CHOSEN_A",
        f"- E15 scientific_gate: `{E15_GATE}`",
        f"- E16 scientific_gate: `{E16_GATE}`",
        f"- 本阶段 scientific_gate: `{summary['scientific_gate']}`",
        f"- opportunity_evidence: `{summary['opportunity_evidence']}`",
        f"- release_eligible: false",
        f"- merged: false",
        f"- training_updates: {summary['training_updates']}",
        "",
        "这是冻结 L0 上的任务分配反事实，不是正式 Scheme C 的端到端评测，也不是新的性能 PASS。正式 Planner、奖励、SafetyProjector 和 L0 checkpoint 没有替换。",
        "",
        "候选只来自决策时刻的 G1_RATE k=5 池。含自然抽到的基准在内最多 5 个唯一关联，按原生成顺序保留，池更短时不补候选。t=45 的不足 15 步尾部没有进入比较。事后最优和 Pareto 集只描述这个有限集合，标记为 hindsight_diagnostic，不是可部署的无未来信息高层。",
        "",
        "## 预算与核对",
        "",
        f"- 参考轨迹步数: {summary['budget']['reference_steps']}",
        f"- 候选步数: {summary['budget']['candidate_steps']}",
        f"- 基准重复步数: {summary['budget']['repeat_steps']}",
        f"- fixture 步数: {summary['budget']['fixture_steps']}",
        f"- 合计步数: {summary['budget']['total_steps']} / {summary['budget']['max_steps']}",
        f"- t=0 已知标签负对照: {summary['t0_negative_control']}",
        f"- 可解释快照: {summary['interpreted_snapshots']}",
        f"- S0 / S1 / S2 候选行: {counts['s0']} / {counts['s1']} / {counts['s2']}",
        f"- 出现 S2 的场景: {counts['s2_scenarios']}",
        "",
        "S0 只看快照当时已经存在的 tagged 服务增量是否大于 1 bit，并且 payload 增量不低于 -1 bit。之后才生成的标签服务另列，不混进这个机会。S1 再要求原始安全和约束计数不增加。S2 再要求平均 SINR 不下降、平均干扰不增加。",
        "",
        "## 缺口",
        "",
    ]
    if summary["problems"]:
        for item in summary["problems"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 预登记的来源、基准回放、重复回放、共同到达和 t=0 负对照没有发现缺项。")
    lines.extend([
        "",
        "## 一个未执行的后续问题",
        "",
        _suggestion(summary),
        "",
    ])
    if summary.get("test_comparison"):
        comparison = summary["test_comparison"]
        lines.extend([
            "## 测试",
            "",
            f"- 干净基线测试数: {comparison['baseline_ran']}",
            f"- 加上本阶段后的测试数: {comparison['repository_ran']}",
            f"- 基线失败签名数: {comparison['baseline_failure_count']}",
            f"- 对照失败签名数: {comparison['repository_failure_count']}",
            f"- new_failure_count: {comparison['new_failure_count']}",
            "- 既有失败仍然是发布阻塞项。完整测试没有写成全通过。",
            "",
        ])
    lines.extend([
        "## 停止",
        "",
        f"E15 保持 `{E15_GATE}`。E16 保持 `{E16_GATE}`。本阶段 scientific_gate 是 `{summary['scientific_gate']}`。",
        "没有合并，没有训练，也没有启动替代方案 B 或下一项实验。正式系统继续使用现有 L0。",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
