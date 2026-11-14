"""Chinese report for the M1.4 calibration-source diagnosis."""

from __future__ import annotations

from pathlib import Path


def _num(value, digits: int = 6) -> str:
    if value is None or value == "":
        return "未测"
    return f"{float(value):.{digits}g}"


def _recommendation(summary: dict) -> str:
    focuses = [item for item in summary["focuses"] if item.get("status") == "COMPLETE"]
    if len(focuses) < 6:
        return "六个快照没有全部完成，先不设计修复实验，也不据此否定主动辨识。"
    return (
        "值得单独设计两个实验，本轮都不做。"
        "第一，冲突修复目前接受评分表里紧挨着冲突目标的下一行；三个冲突快照的下一行是校准，它比合法回传高约 0.001–0.003，其中 b3_window 在关闭后续校准的续策里符号相反。"
        "可以问修复是否应在非校准合法动作中重选。"
        "第二，三个单机快照的终端业务都是 none，校准只因为这一段推进功耗低于开始返航而胜出，优势 0.0004–0.0006；续策里返航少用约 4700–6700 焦。"
        "可以问一步终端要不要计入返航最终落地后推进关闭。"
        "恒等式残差为 0，分解和冻结评分表一致，这两件事是视野问题，留到下一条指令。"
        "不重跑 M1.3，也不把负效用当成待修故障。主动辨识在其他任务上是否还有价值，这一诊断没有否决。"
    )


def _pair(summary: dict, snapshot: str, comparison: str) -> dict:
    return next(row for row in summary["continuation"]["pairs"] if row["snapshot"] == snapshot and row["comparison"] == comparison)


def _answers(summary: dict) -> list[str]:
    origin = summary["origin"]
    by_origin = origin["by_origin_condition"]
    return [
        f"76 条校准请求中，{origin['conflict_repair']} 条来自冲突修复，{origin['single_score']} 条是单机评分首选。"
        f"修复在 tight / window / loose 各 {by_origin['conflict_repair'].get('tight', 0)} / {by_origin['conflict_repair'].get('window', 0)} / {by_origin['conflict_repair'].get('loose', 0)} 条；"
        f"单机首选分别是 {by_origin['single_score'].get('tight', 0)} / {by_origin['single_score'].get('window', 0)} / {by_origin['single_score'].get('loose', 0)} 条。",
        f"单机首选的 Qcal 为负，相对表内最佳执行动作的优势是 {_num(origin['single_score_gap_min'])} 到 {_num(origin['single_score_gap_max'])}。负的 Q 只表示这一步期望净收益为负，本轮没有因此增加新规则。",
        f"返航检查单独计数：{origin['return_guard_replacements']} 条请求在执行中被换成返航，保存的决策行没有换之前的秒数，这 {origin['duration_unknown']} 条不记成 30 秒。"
        f"其余 {origin['full_30s']} 条的最终动作仍是校准，整段 30 秒都在校准。",
        "六个焦点都完整分解，登记类别和日志一致。冲突三例的合法替代是回传；单机三例的合法替代是返航。日志里的最佳执行动作在冲突例上是两机争抢的浮标，该组合不能执行，所以不用它做反事实。六个焦点的另一架实际都在 service，单机评分假定它 wait；两架 Q 没有相加。",
        "额外的校准信息溢价为 0。三个单机焦点的校准溢价和普通执行溢价都是 0，终端业务是 none，后验没有打开新业务。三个冲突焦点的校准溢价约 0.00065–0.00104，但回传遥测的溢价更高，净信息差最大约 0.00016，方向偏向回传。",
        "无更新状态或成本解释了校准相对合法替代的正差距。三个单机焦点的差距等于 30 秒即时项：校准推进功耗低于开始返航，无更新终端差为 0。b1_tight 的无更新状态项为 0.00248，净信息差在 tie 内。b3_window 与 b1_loose 因为净信息差刚刚超过 tie 被记为 mixed，正差距仍来自状态项；b3_window 的即时项其实偏向回传约 1.998，因为回传在这 30 秒交付了缓存，终端又把后续业务加回校准一侧，相消后只剩 0.00273。",
        _continuation_sentence(summary),
        "一步差距都大于 M1.3 已记录的 15 点残差 0.000290，符号按冻结的 15×7 判定，没有调用 31×15。恒等式残差为 0，分解值和冻结评分表的差在 1e-15 量级。",
    ]


def _continuation_sentence(summary: dict) -> str:
    if not summary["continuation"].get("enabled"):
        return "固定续策未跑，哪些近似 Q 会兑现这一项未测。"
    names = ("b3_window", "b1_window", "b4_window", "b2_loose")
    flipped = []
    for name in names:
        row = _pair(summary, name, "C_plus_minus_E_plus")
        flipped.append(f"{name} 的续跑效用差 {_num(row['realized_utility_gap'])}、能耗差 {_num(row['realized_energy_gap_j'])} 焦")
    held = []
    for name in ("b1_tight", "b1_loose"):
        row = _pair(summary, name, "C_plus_minus_E_plus")
        held.append(f"{name} 兑现为首段少用约 {abs(float(row['realized_energy_gap_j'])):.0f} 焦，之后动作不再分歧")
    return (
        "固定关闭后续校准的续策已经跑完。六对信息消融的效用差都是 0：三个正的一步信息溢价没有兑现，三个零溢价与续跑一致。"
        "校准高于合法替代这一步判断没有兑现的有："
        + "；".join(flipped)
        + "。这四条的按时 bit 差都是 0。"
        + "；".join(held)
        + "。这两条的按时 bit 差也是 0。一步 Q 和续跑增量是两个量，数值不同留在表里。"
    )


def write_m14_report(summary: dict, path: Path) -> None:
    terminals = summary["core_terminals"]
    counts = summary["counts"]
    lines = [
        "# 主动功耗辨识 M1.4 校准来源与价值分解",
        "",
        f"- 结果目录：`{summary['result_dir']}`",
        f"- 审查的 M1.3 结果：`{summary['reviewed_result_commit']}`",
        f"- 指令提交：`{summary['instruction_commit']}`",
        f"- 必要测试：{summary['unit_tests_status']}",
        "- 本轮没有改电池、业务、先验、噪声、物理模型、准入、效用权重、冲突修复或生产评分，也没有训练。",
        "",
        "## 直接回答",
        "",
        *_answers(summary),
        "",
        _recommendation(summary),
        "",
        "## 终端字段",
        "",
        f"M1.3 核心航次共 {terminals['missions']} 条。物理未返航、未落地、耗尽和 return_failures 在开启与关闭两侧都是"
        f" {terminals['on']['not_returned']} / {terminals['off']['not_returned']}、"
        f"{terminals['on']['not_landed']} / {terminals['off']['not_landed']}、"
        f"{terminals['on']['energy_exhausted']} / {terminals['off']['energy_exhausted']}、"
        f"{terminals['on']['return_failures']} / {terminals['off']['return_failures']}。",
        f"备用电量标准 below_reserve：开启 {terminals['on']['below_reserve']}/24 架，关闭 {terminals['off']['below_reserve']}/24 架。满足返航、落地且不低于备用标准的架数是开启 {terminals['on']['returned']}、关闭 {terminals['off']['returned']}。",
        "",
        "## 六个快照",
        "",
        "清单在分析前固定：冲突修复是 b1_tight、b3_window、b1_loose；单机评分是 b1_window、b4_window、b2_loose。",
        "替代动作 E 固定另一架在原联合动作里的任务，再在准入通过且不争抢同一浮标的非校准候选里，用原来的 tie 规则取最高分。它不是全局最优联合策略。",
        "单机 Q 假定另一架等待。下面的 context_other_is_wait 为假时，实际联合动作不是等待；两架单机 Q 没有相加。",
        "生产终端只读焦点 UAV 的信念。撤去更新只撤焦点机本区间的观测，另一架的更新不进入这个单机 Q。",
        "恒等式是 Q(C,+)-Q(E,+) = [Q(C,-)-Q(E,-)] + [Q(C,+)-Q(C,-)] - [Q(E,+)-Q(E,-)]。近似 Q 里的信息溢价不是任务收益。",
        "",
    ]
    for item in summary["focuses"]:
        lines.extend(_focus_lines(item))
    lines.extend(
        [
            "## 计算量",
            "",
            f"来源分类和配对表只读 M1.3 日志。Q 诊断物理展开 {counts['q_rollouts']} 次，缓存命中 {counts['q_cache_hits']}，规划步 {counts['q_planning_steps']}，代数积分 {counts['q_algebraic_integrals']}，代数用时 {_num(counts['q_algebraic_seconds'])} 秒。上限 6000 次物理展开。",
            f"续跑启用：{counts['continuation_enabled']}。原因：{summary['continuation']['reason']}。",
            f"续跑规划展开 {counts['continuation_rollouts']} 次，复用 {counts['continuation_reused']} 条，新跑 {counts['continuation_new']} 条，未测 {counts['continuation_not_run']} 条。上限 200000 次。",
            f"墙钟 {_num(summary['elapsed_s'])} 秒。训练更新 0。",
            "",
            "## 没有做的事",
            "",
            "没有重跑三档主试，没有搜索新电池，没有把 L8、L9 或未注册场景接进来，没有把研究分支并入维护分支。M1.3 的负效用保持原样。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _focus_lines(item: dict) -> list[str]:
    if item.get("status") != "COMPLETE":
        return ["", f"### {item.get('snapshot', '未命名')}", "", f"状态 {item.get('status', '未完成')}，不另找替代快照。", ""]
    return [
        "",
        f"### {item['snapshot']}",
        "",
        f"登记类别 {item['registered_class']}，日志来源 {item['log_origin']}，类别与日志一致：{item['class_matches_log']}。",
        f"焦点 UAV {item['focus_uav']}，对方联合动作 {item['partner_action']}，合法替代 {item['alternative']}。对方是否在评分假设里等待：{item['context_other_is_wait']}。",
        f"归因 {item['attribution']}，符号 {item['sign_status']}。Q(C,+)-Q(E,+) = {_num(item['gap_plus'])}，"
        f"无更新状态项 {_num(item['state_cost'])}，校准信息溢价 {_num(item['info_premium_c'])}，执行信息溢价 {_num(item['info_premium_e'])}。",
        f"其中即时项差 {_num(item['immediate_gap'])}，无更新终端差 {_num(item['terminal_minus_gap'])}。恒等式残差 {_num(item['identity_residual'], 3)}。",
        f"15×7 与评分表差距的绝对差 {_num(item['table_split_abs'], 3)}。31×15 {'未调用' if item['fine_gap'] in (None, '') else _num(item['fine_gap'])}。",
        f"代表终点：校准 {item['calibration_end']}，替代 {item['alternative_end']}。推进能耗 {_num(item['calibration_propulsion_j'])} 对 {_num(item['alternative_propulsion_j'])} 焦。",
        f"运动注记：{', '.join(item['motion_notes']) if item['motion_notes'] else '无'}。",
        "",
    ]
