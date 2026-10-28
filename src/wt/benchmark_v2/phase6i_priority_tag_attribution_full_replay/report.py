"""Chinese E9 report. Figures are copied from the summary."""
from __future__ import annotations

from .protocol import GATE_BLOCKED, GATE_COMPLETE, GATE_INCONCLUSIVE


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "缺失"
    return f"{float(value):,.{digits}f}"


def _interval(item: dict) -> str:
    return f"均值 {_fmt(item['mean'])}，95% 区间 [{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}]"


def _recommendation(summary: dict) -> str:
    gate = summary.get("gate")
    if gate == GATE_BLOCKED:
        return (
            "重放、哈希、划分或时间预算没有通过。"
            "不用未完成的场景解释年龄或标签，不改奖励，不训练，不合并。"
        )
    if gate == GATE_INCONCLUSIVE:
        return (
            "224 条重放都在原容差内，但三个 L2 种子对合紧急度配对变化的符号，"
            "或对年龄与标签增量哪一项更大，并不一致。"
            "这不足以支持单一归因，也不启动新的奖励设计。"
        )
    if gate != GATE_COMPLETE:
        return "本阶段没有形成可执行的模型建议。"
    letter = summary.get("locked_rule_letter")
    positive = int(summary.get("l2_tagged_positive_seeds") or 0)
    if letter == "B":
        return (
            "锁定规则给出 B：至少两个 L2 种子的年龄服务配对均值为正，且大于标签增量配对均值。"
            "合紧急度变化主要由年龄承担。以后若要改奖励，只应在新的实验分支上把年龄和直接 tagged 服务分成两个有上界的项，"
            "并预先写好对照 L0 的标准。本阶段不选择新公式，也不训练。"
        )
    if letter == "A" and positive >= 2:
        return (
            "锁定规则给出 A，因为达不到“至少两个 L2 种子的年龄服务配对均值为正且大于标签增量”。"
            f"同时有 {positive} 个 L2 种子的 tagged 服务配对均值为正，所以不能把 A 读成“tagged 服务没有增加”。"
            "本阶段仍不改公式，也不训练。"
        )
    if letter == "A":
        return (
            "锁定规则给出 A：少于两个 L2 种子表现为年龄服务配对均值为正且大于标签增量。"
            f"tagged 服务配对均值为正的 L2 种子数是 {positive}。"
            "下一步是停止这条 priority-tag 奖励，不在本分支训练。"
        )
    return "完整重放没有给出锁定规则字母。不改奖励，不训练。"


def render_report(summary: dict) -> str:
    preflight = summary.get("preflight") or {}
    replay = summary.get("replay") or {}
    reconciliation = summary.get("reconciliation") or {}
    lines = [
        "# 实验报告：已验证运行时上下文下的紧急度归因重放",
        "",
        "- 诊断分支：`experiment/priority-tag-attribution-full-replay`",
        "- 来源实验提交：`f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb`",
        f"- E8 提交：`{summary.get('e8_commit')}`",
        f"- 指令提交：`{summary.get('instruction_commit')}`",
        f"- Gate：`{summary.get('gate')}`",
        "- 是否合并：否",
        "- 是否重新训练：否",
        "- 标签性质：synthetic priority tag，不是真实海事业务标签。",
        "",
        "## 1. 种子角色",
        "",
        "模型种子是 checkpoint 和 CSV 的 model_seed。L0 是 0，L1/L2 各自使用 0、1、2。",
        "环境 reset seed 是场景编号。分配 RNG 是 7000000 加场景编号。",
        "全局 seed 2 只作为历史继承进程上下文，在每条 episode 开始前显式重新设置。",
        "它不是评估协议宣布的通用种子。规范化显式评估全局种子仍然是 null。",
        f"进程标签：`{summary.get('context_label')}`。角色：`{summary.get('process_rng_role')}`。",
        "",
        "## 2. 对账与预检",
        "",
        f"保存的 Gate 是 `{reconciliation.get('saved_gate')}`。",
        f"CSV 行数 {reconciliation.get('row_count')}，期望 {reconciliation.get('expected_row_count')}。",
        f"按保存的 CSV 重算区间后，最大绝对差是 {_fmt(reconciliation.get('max_abs_gap'), 6)}。",
        f"预检是 L0 seed 0 场景 {preflight.get('scenario_id')}，用时 {_fmt(preflight.get('seconds'), 3)} 秒。",
        f"按这个速度估计 224 条需要 {_fmt(preflight.get('estimated_seconds'), 1)} 秒，上限 600 秒。",
        f"预检 payload 差 {_fmt(preflight.get('max_bit_gap'), 6)} bit，路径差 {_fmt(preflight.get('path_gap_m'), 9)} 米，"
        f"年龄加标签增量对合紧急度的差 {_fmt(preflight.get('identity_gap_bits'), 6)} bit，"
        f"合紧急度对账本的差 {_fmt(preflight.get('ledger_urgency_gap_bits'), 6)} bit。",
        f"预检进入容差：{preflight.get('matches_saved')}。",
        "",
        "## 3. 全表重放",
        "",
        f"完成 {replay.get('completed_episodes')} 条，用时 {_fmt(replay.get('elapsed_seconds'), 1)} 秒。",
        f"最大 bit 差 {_fmt(replay.get('max_bit_gap'), 6)} bit，最大路径差 {_fmt(replay.get('max_path_gap_m'), 9)} 米。",
        f"分解恒等式最大差 {_fmt(replay.get('max_identity_gap_bits'), 6)} bit，"
        f"相对账本合紧急度最大差 {_fmt(replay.get('max_ledger_urgency_gap_bits'), 6)} bit。",
        "容差仍是记账 1 bit、路径 1e-9 米、年龄加标签增量对合紧急度 1e-6 bit。",
        f"正式 checkpoint 运行后哈希未变：{summary.get('formal_hash_unchanged')}。",
        "",
    ]
    failure = summary.get("failure")
    if failure:
        lines.append(
            f"停止于 {failure.get('variant')} seed {failure.get('model_seed')} 场景 {failure.get('scenario_id')}。"
        )
        lines.append(
            f"路径差 {_fmt(failure.get('path_gap_m'), 9)} 米，分解差 {_fmt(failure.get('identity_gap_bits'), 6)} bit。"
        )
        for key, value in (failure.get("bit_gaps") or {}).items():
            lines.append(f"- {key}：{_fmt(value, 6)} bit")
        lines.append("未完成的场景不进入年龄或标签结论。")
        lines.append("")
    attribution = summary.get("attribution")
    if attribution is None:
        lines.extend(
            [
                "## 4. 归因",
                "",
                "完整重放没有形成对照表。",
                "",
                "## 5. 原先的重放差",
                "",
                "E6 在默认线程和未设置历史进程上下文时，L0 场景 321101 的 payload 差约 1.553167 bit。",
                "那次差已经由 E8 的显式上下文复核闭合。本阶段若预检或后续场景再次超出容差，仍然保持原阈值并停止。",
                "",
                "## 6. 下一步",
                "",
                _recommendation(summary),
                "",
            ]
        )
        return "\n".join(lines)
    clipping = attribution["l0_clipping"]
    lines.extend(
        [
            "## 4. 年龄、标签增量和裁剪",
            "",
            f"L0 年龄加权服务合计 {_fmt(clipping['age_service_bits'])} bit，"
            f"标签增量合计 {_fmt(clipping['tag_increment_service_bits'])} bit，"
            f"合紧急度 {_fmt(clipping['combined_urgency_served_bits'])} bit。",
            f"L0 tagged 服务 {_fmt(clipping['tagged_served_bits'])} bit，"
            f"其中紧急度已被年龄顶满、标签增量完全饱和的比例 {_fmt(clipping['full_saturation_fraction'])}，"
            f"标签增量低于 0.25 的比例 {_fmt(clipping['below_full_boost_fraction'])}，"
            f"标签增量保留比例 {_fmt(clipping['boost_retained_fraction'])}。",
            f"L0 生成标签率均值 {_fmt(clipping['tag_draw_rate_mean'])}。"
            f"服务时普通块平均年龄 {_fmt(clipping['served_age_mean_ordinary'])} 步，"
            f"tagged 块平均年龄 {_fmt(clipping['served_age_mean_tagged'])} 步。",
            "",
        ]
    )
    for group in attribution["groups"]:
        if group["variant"] != "L2":
            continue
        lines.append(
            f"- L2 seed {group['model_seed']} 相对 L0：年龄服务 {_interval(group['intervals']['age_service_bits'])}；"
            f"标签增量 {_interval(group['intervals']['tag_increment_service_bits'])}；"
            f"合紧急度 {_interval(group['intervals']['combined_urgency_served_bits'])}；"
            f"tagged 服务 {_interval(group['intervals']['served_tagged_bits'])}；"
            f"payload {_interval(group['intervals']['payload_bits'])}。"
        )
        lines.append(
            f"- L2 seed {group['model_seed']} 的合紧急度配对变化中，普通块年龄占 {_fmt(group['ordinary_share_of_combined_delta'])}，"
            f"标签增量占 {_fmt(group['tag_share_of_combined_delta'])}。"
            f"完全饱和比例 {_fmt(group['clipping']['full_saturation_fraction'])}，"
            f"标签增量保留比例 {_fmt(group['clipping']['boost_retained_fraction'])}。"
        )
    lines.extend(["", "### L1 的 tagged 服务", ""])
    for group in attribution["groups"]:
        if group["variant"] != "L1":
            continue
        lines.append(
            f"- L1 seed {group['model_seed']} 相对 L0：年龄服务 {_interval(group['intervals']['age_service_bits'])}；"
            f"标签增量 {_interval(group['intervals']['tag_increment_service_bits'])}；"
            f"tagged 服务 {_interval(group['intervals']['served_tagged_bits'])}。"
        )
    lines.extend(
        [
            "",
            "## 5. 原先的重放差",
            "",
            "E6 停止时，L0 场景 321101 的 payload 差是 1.553167 bit，路径差约 5.745e-6 米。",
            "那次差来自未显式设置的线程数和历史进程上下文，不是归因包装改写了动作。",
            "保存的 CSV 与 summary 区间差仍按对账结果核对。本阶段重放进入原容差后，原先那组超差不再成立。",
            "标签是合成的，不能当成真实海事优先级。",
            "",
            "## 6. 下一步",
            "",
            _recommendation(summary),
            "",
            "本阶段不合并任何分支。",
            "",
        ]
    )
    return "\n".join(lines)
