"""Chinese report for the fixed tag-rule diagnosis."""
from __future__ import annotations

from .protocol import E15_GATE, E16_GATE, E17_GATE, E17_OPPORTUNITY, E18_GATE, E18_OPPORTUNITY, OUTPUT, REPORT


def _interval(summary: dict, key: str) -> str:
    item = (summary.get("bootstrap") or {}).get(key)
    if not item:
        return "未计算"
    cross = "区间包含 0" if item.get("crosses_zero") else "区间不包含 0"
    return (
        f"均值 {item['mean']:.6g}，中位数 {item['median']:.6g}，"
        f"95% 区间 [{item['ci_low']:.6g}, {item['ci_high']:.6g}]，{cross}"
    )


def _suggestion(summary: dict) -> str:
    if summary.get("scientific_gate") != "FIXED_TAG_RULE_DIAG_COMPLETE" or not summary.get("mechanism_published"):
        mismatch = "; ".join(summary.get("problems") or []) or "完整性核对没有通过"
        return f"核对停在：{mismatch}。不要放宽容差、更换场景、改权重或补跑窗口。本阶段到此停止。"
    label = summary.get("mechanism_evidence")
    if label == "NO_TAG_DECISION_EFFECT":
        return "这 16 个状态上，0.001 的标签项没有改变任何选择。这只说明这条固定代理和这个权重在本样本里没有决策差，不说明标签没有用。不要在本阶段加大权重、增大 k 或改去 DEV/测试场景。本阶段到此停止。"
    if label == "REPLICATED_GUARDED_TAG_SIGNAL":
        uncertain = "描述性区间仍包含 0 或落在负侧，总体增益仍不确定。" if summary.get("overall_gain_uncertain") else "描述性区间没有跨过 0，但这仍不是采纳标准。"
        return f"至少两个训练场景上，带标签的规则相对屏蔽对照和自然基准都达到了 S2，且 16 个场景的安全和约束没有变差、平均 payload 没有低于 -1 bit。{uncertain} 若要在另一个预登记窗口上重复同一条固定规则，必须另开阶段，并且仍然冻结低层。本阶段不启动该重复。"
    if label == "LOCAL_OR_UNGUARDED_TAG_EFFECT":
        failed = "; ".join(summary.get("failed_checks") or []) or "覆盖、安全、通信或总体 payload 条件没有同时成立"
        return f"标签项改变了至少一个选择，也出现了 S0，但没有形成有防护的重复信号。未通过的核对是：{failed}。不要把这条规则拿去训练或合并。本阶段到此停止。"
    return "选择有差别，但没有任何场景在屏蔽对照上达到 S0，或者没有形成有防护的重复信号。负的差值和未通过的核对保留原样。不要改权重、增大 k、接触 DEV/测试场景或启动训练。本阶段到此停止。"


def write_report(summary: dict) -> None:
    budget = summary.get("budget") or {}
    contrasts = summary.get("contrasts") or {}
    reuse = summary.get("reuse") or {}
    label = summary.get("mechanism_evidence") or "未发布"
    tm = contrasts.get("tm") or {}
    tn = contrasts.get("tn") or {}
    mn = contrasts.get("mn") or {}
    opening = (
        "本阶段检验一条预先固定的小权重标签规则，训练更新为 0，参数更新为 0。"
        "屏蔽对照使用同一候选集合、同一数值运算和同一稳定排序，只把分数里的标签项去掉；回放环境仍使用真实合成标签。"
        "场景是训练集 321008–321023，共 16 个，它们属于训练池，不是项目级的未见保留集。"
        "每个场景只在自然参考的 t=15 保存一次可恢复快照，主比较窗口只有 [15, 30)。"
        "只看 t=15 是明确的开发选择，因为 E18 的全部 S0 行都出现在 t=15；本阶段不检验 t=0、t=30、t=45，也不评估一条新规则的完整闭环回合。"
        "步数上限是 1728，旧阶段剩余的步数不能挪用。"
        "这不是正式 Scheme C，不是新高层策略的完整闭环，也不是 E17 或 E18 的改写。"
        f"E17 的 scientific_gate 仍是 `{E17_GATE}`，机会结论仍是 `{E17_OPPORTUNITY}`。"
        f"E18 的 scientific_gate 仍是 `{E18_GATE}`，机会结论仍是 `{E18_OPPORTUNITY}`。"
        f"正式回放的步数是 {budget.get('total_steps', '未完成')}，其中参考 {budget.get('reference_steps', '未完成')}、候选 {budget.get('candidate_steps', '未完成')}、重复 {budget.get('repeat_steps', '未完成')}。"
        "第一次回放没有打开确定性算法，动作差约 1e-8，在进入机制分类前停止。那次额外使用了 15 个候选步和 15 个重复步，没有被当成第二个样本。"
        "正式回放仍读取同一份已提交的选择清单。把这 30 个作废步与正式回放相加，总量仍低于 1728。"
        f"标签规则与屏蔽对照选择不同的场景数是 {summary.get('tagged_differs_count', '未完成')}。"
        f"机制证据是 `{label}`。"
        "release_eligible 为 false，merged 为 false，adopted 为 false，performance_pass 为 false。"
    )
    lines = [
        "# 固定标签规则与屏蔽对照诊断",
        "",
        opening,
        "",
        f"- E15 scientific_gate: `{summary.get('e15_scientific_gate', E15_GATE)}`",
        f"- E16 scientific_gate: `{summary.get('e16_scientific_gate', E16_GATE)}`",
        f"- E17 scientific_gate: `{summary.get('e17_scientific_gate', E17_GATE)}`",
        f"- E17 opportunity_evidence: `{summary.get('e17_opportunity_evidence', E17_OPPORTUNITY)}`",
        f"- E18 scientific_gate: `{summary.get('e18_scientific_gate', E18_GATE)}`",
        f"- E18 opportunity_evidence: `{summary.get('e18_opportunity_evidence', E18_OPPORTUNITY)}`",
        f"- 本阶段 scientific_gate: `{summary.get('scientific_gate')}`",
        f"- mechanism_evidence: `{label}`",
        f"- mechanism_published: {str(bool(summary.get('mechanism_published'))).lower()}",
        "- performance_pass: false",
        "- release_eligible: false",
        "- merged: false",
        "- adopted: false",
        "- 训练更新: 0",
        "- 参数更新: 0",
        f"- 选择不同的场景数: {summary.get('tagged_differs_count', '未完成')}",
        f"- 复用了屏蔽对照回放的场景数: {reuse.get('scenarios_reused_m', '未完成')}",
        f"- 复用了标签规则回放的场景数: {reuse.get('scenarios_reused_t', '未完成')}",
        f"- T−M 的 S0/S1/S2: {tm.get('s0', '未完成')}/{tm.get('s1', '未完成')}/{tm.get('s2', '未完成')}",
        f"- T−N 的 S0/S1/S2: {tn.get('s0', '未完成')}/{tn.get('s1', '未完成')}/{tn.get('s2', '未完成')}",
        f"- M−N 的 S0/S1/S2: {mn.get('s0', '未完成')}/{mn.get('s1', '未完成')}/{mn.get('s2', '未完成')}",
        f"- T−M 已知标签服务差: {_interval(summary, 'known_tagged_tm')}",
        f"- T−M payload 差: {_interval(summary, 'payload_tm')}",
        f"- T−N 已知标签服务差: {_interval(summary, 'known_tagged_tn')}",
        f"- T−N payload 差: {_interval(summary, 'payload_tn')}",
        f"- M−N 已知标签服务差: {_interval(summary, 'known_tagged_mn')}",
        f"- M−N payload 差: {_interval(summary, 'payload_mn')}",
        f"- 总体增益仍不确定: {str(bool(summary.get('overall_gain_uncertain'))).lower()}",
        f"- 正式低层检查点 SHA-256: `{summary.get('formal_actor_sha256', '未完成')}`",
        f"- 选择清单 SHA-256: `{summary.get('selection_sha256', '未完成')}`",
        f"- 实现提交: `{summary.get('implementation_commit', '未完成')}`",
        f"- 选择提交: `{summary.get('selection_commit', '未完成')}`",
        "",
        "## 规则",
        "",
        "- N 是快照上的自然抽样 C0。",
        "- M 在与 C0 逐 UAV 活跃掩码相同的去重候选上最大化 payload 代理 P。",
        "- T 在同一集合上最大化 P + 0.001 K。K 是当前 FIFO 队列里、容量前缀落在 priority_tag=1 的部分。",
        "- 普通桶排在前面时会先占用容量。空队列的 P 和 K 都是 0。空闲 UAV 两项都贡献 0。",
        "- 分数完全相同时取原池最小索引。不用观测到的服务打破平局，也不加随机抖动。",
        "- P 是当前速率扣掉几何 ETA 后的有限窗代理，并被当前队列截断。它不是实际传输，也不是联合干扰预测。",
        "- M 只在打分时不使用标签。M 的回放环境不会把标签清零。",
        "- 选择器不读取 scenario_id、未来到达、E18 事后最优索引或已经实现的服务。",
        "",
        "## 未通过或保留的核对",
        "",
    ]
    failed = summary.get("failed_checks") or []
    problems = summary.get("problems") or []
    if failed:
        lines.extend(f"- {item}" for item in failed)
    else:
        lines.append("- 没有额外的机制核对失败项。")
    if problems:
        lines.extend(f"- 完整性问题: {item}" for item in problems)
    else:
        lines.append("- 没有完整性问题。")
    lines.extend(["", "## 下一步", "", _suggestion(summary), "", "## 停止", "", "本阶段到此停止。不搜索其他权重，不增大 k，不接触 DEV 或测试场景，不启动 HGNN 或强化学习，不修改正式装载器，也不合并。", ""])
    comparison = summary.get("test_comparison")
    lines.extend(["## 测试", ""])
    if not comparison:
        lines.append("- 历史全量测试对照尚未写入。")
    else:
        lines.extend(
            [
                f"- 干净基线 Ran {comparison.get('baseline_ran')}，失败签名 {comparison.get('baseline_failure_count')}。",
                f"- 叠加本阶段后 Ran {comparison.get('repository_ran')}，失败签名 {comparison.get('repository_failure_count')}。",
                f"- 新增失败 {comparison.get('new_failure_count')}。历史失败仍然是发布阻断，全量套件不是绿灯。",
                f"- 本阶段相关测试通过: {str(bool(comparison.get('related_ok'))).lower()}。",
            ]
        )
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    _ = OUTPUT
