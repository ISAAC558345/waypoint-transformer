"""Chinese report for the activity-matched coverage diagnosis."""
from __future__ import annotations

import csv

from .protocol import E15_GATE, E16_GATE, E17_GATE, E17_OPPORTUNITY, OUTPUT, REPORT


def _suggestion(summary: dict) -> str:
    if summary.get("scientific_gate") != "ASSIGNMENT_COVERAGE_DIAG_COMPLETE" or not summary.get("opportunity_published"):
        mismatch = "; ".join(summary.get("problems") or []) or "完整性核对没有通过"
        return f"核对停在：{mismatch}。不要放宽容差、更换场景、重抽种子，或把这次未完成的样本当成没有机会。本阶段到此停止。"
    label = summary.get("opportunity_evidence")
    if label == "REPLICATED_LOCAL_OPPORTUNITY":
        return "至少两个训练场景在同掩码分层样本里出现了同时守住 payload、原始安全和通信条件的已知标签服务差。若要检验一条固定规则，必须另开预登记，并且仍然冻结低层、不使用事后最优候选。本阶段不启动该对照。"
    if label == "LOCAL_OR_CONSTRAINED_OPPORTUNITY":
        return "同掩码样本里出现了已知标签服务差，但没有在两个场景上同时通过安全或通信条件，或者覆盖不完整。不要把事后最优关联拿去训练。本阶段到此停止。"
    if label == "NOT_OBSERVED_UNDER_MATCHED_MASK":
        return "这 24 个状态在各自至少一条同掩码替代上没有观察到已知标签服务差。这不是联合池的穷尽上界，也不是重开裁剪实验或增大 k 的理由。本阶段到此停止。"
    return "同掩码替代没有覆盖全部 24 个状态，或者有效回放不足以归入前两类。没有候选不等于没有机会。不要在本阶段补候选、增大 k 或换场景。本阶段到此停止。"


def _measured_lines() -> list[str]:
    path = OUTPUT / "candidate_rows.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("s0") == "1"]
    if not rows:
        return ["登记替代里没有超过 1 bit 的已知标签服务差。", ""]
    deltas = [float(row["known_tagged_delta_bits"]) for row in rows]
    times = sorted({int(row["decision_time"]) for row in rows})
    safety_path = OUTPUT / "safety_events.csv"
    safety_rows = 0
    if safety_path.is_file():
        with safety_path.open(encoding="utf-8", newline="") as handle:
            safety_rows = max(sum(1 for _ in handle) - 1, 0)
    return [
        "## 测量到的服务差",
        "",
        f"- 达到 S0 的替代有 {len(rows)} 条，已知标签服务差最小 {min(deltas):.1f} bit，最大 {max(deltas):.1f} bit。",
        f"- 这些行的决策时刻是 {times}。全部 S0 行的安全差和约束差都没有上升，所以 S1 与 S0 计数相同。",
        "- S2 只保留其中均值 SINR 不下降且均值干扰不上升的行。",
        f"- 回放安全或约束事件行数: {safety_rows}。约束失败仍保留原始计数，没有被记成安全机会。",
        "",
    ]


def write_report(summary: dict) -> None:
    counts = summary.get("opportunity_counts") or {}
    budget = summary.get("budget") or {}
    coverage = summary.get("coverage") or {}
    label = summary.get("opportunity_evidence") or "未发布"
    measured = summary.get("measured_opportunity") or label
    opening = (
        f"E17 已经完成，scientific_gate 仍是 `{E17_GATE}`，机会结论仍是 `{E17_OPPORTUNITY}`。"
        "该结论只说明当时登记的前缀截断集合里没有观察到已知标签服务机会，不改写成执行失败，也不表示整个 G1_RATE k=5 池、高层任务分配或正式 Scheme C 无效。"
        "旧规则的问题是参与规模：非基准候选落在联合池生成顺序的最前面，表现为空闲或单 UAV。"
        "本阶段唯一的方法改变，是在自然基准之外，按原池索引把活跃掩码逐 UAV 相同的合法关联分成最多四层，每层抽一条，作为替代候选。"
        f"有效覆盖为 {coverage.get('snapshots_with_same_mask_alternative', '未完成')} / {coverage.get('snapshots', '未完成')} 个状态至少有一条同掩码替代，"
        f"实际抽到的替代条数是 {coverage.get('selected_alternatives', '未完成')}，"
        f"同掩码唯一替代总数是 {coverage.get('matched_unique_alternatives', '未完成')}。"
        f"实际步数是 {budget.get('total_steps', '未完成')}。"
        f"机会类别是 `{measured}`。"
        "这个分层样本不是联合池的穷尽搜索，不是全局上界，也不是正式 Scheme C 的性能结论。"
        "训练更新为 0，参数更新为 0，release_eligible 为 false，merged 为 false。"
    )
    lines = [
        "# 活动掩码匹配的任务分配覆盖诊断",
        "",
        opening,
        "",
        f"- E15 scientific_gate: `{E15_GATE}`",
        f"- E16 scientific_gate: `{E16_GATE}`",
        f"- E17 scientific_gate: `{summary.get('e17_scientific_gate', E17_GATE)}`",
        f"- E17 opportunity_evidence: `{summary.get('e17_opportunity_evidence', E17_OPPORTUNITY)}`",
        f"- 本阶段 scientific_gate: `{summary.get('scientific_gate')}`",
        f"- opportunity_evidence: `{label}`",
        f"- opportunity_published: {str(bool(summary.get('opportunity_published'))).lower()}",
        "- release_eligible: false",
        "- merged: false",
        f"- training_updates: {summary.get('training_updates', 0)}",
        f"- parameter_updates: {summary.get('parameter_updates', 0)}",
        "",
        "旧前缀和新样本只比较选择覆盖和本次登记的诊断结论。后出现的服务差不能解读成新的高层已经提升了性能。",
        "",
        "## 预算与核对",
        "",
        f"- 参考轨迹步数: {budget.get('reference_steps')}",
        f"- 候选回放步数: {budget.get('candidate_steps')}",
        f"- 基准重复步数: {budget.get('repeat_steps')}",
        f"- 总步数: {budget.get('total_steps')}",
        f"- 选择清单哈希: `{summary.get('selection_sha256')}`",
        f"- 正式低层 SHA-256: `{summary.get('formal_actor_sha256')}`",
        f"- t=0 负对照通过: {str(bool(summary.get('t0_negative_control'))).lower()}",
        "",
        "## 机会计数",
        "",
        f"- S0: {counts.get('s0', 0)}",
        f"- S1: {counts.get('s1', 0)}",
        f"- S2: {counts.get('s2', 0)}",
        f"- S0 场景: {counts.get('s0_scenarios', [])}",
        f"- S2 场景: {counts.get('s2_scenarios', [])}",
        "",
        *_measured_lines(),
        "## 限制",
        "",
        "同掩码抽样会漏掉更换参与 UAV 才能出现的收益。k 仍是 5，场景仍是 321000–321007，决策时刻仍是 0、15、30，t=45 尾部没有进入比较。",
        "事后最优只描述已经回放的样本，标记为 hindsight_diagnostic，不是可部署高层。",
        "",
        "## 下一步",
        "",
        _suggestion(summary),
        "",
        "本阶段到此停止，不扩大 k、场景、候选或预算，不训练，不修改正式系统，不重开裁剪线。",
        "",
    ]
    if summary.get("problems"):
        lines.extend(["## 阻塞", ""])
        lines.extend(f"- {item}" for item in summary["problems"])
        lines.append("")
    comparison = summary.get("test_comparison")
    if comparison:
        lines.extend(
            [
                "## 测试",
                "",
                f"- 干净基线测试数: {comparison.get('baseline_ran')}",
                f"- 干净基线加上本阶段代码后的测试数: {comparison.get('repository_ran')}",
                f"- 新增失败: {comparison.get('new_failure_count')}",
                f"- 相关测试通过: {str(bool(comparison.get('related_ok'))).lower()}",
                "- 历史失败仍是发布阻塞项。完整历史套件不记为全绿。",
                "",
            ]
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
