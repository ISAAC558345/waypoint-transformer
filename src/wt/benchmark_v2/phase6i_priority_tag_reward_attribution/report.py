"""Chinese attribution report. Numbers come from the summary and are not rewritten by hand."""
from __future__ import annotations


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "缺失"
    return f"{float(value):,.{digits}f}"


def _interval(item: dict) -> str:
    return f"均值 {_fmt(item['mean'])}，95% 区间 [{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}]"


def _clipping_lines(title: str, clipping: dict) -> list[str]:
    return [
        f"- {title}：tagged 服务 {_fmt(clipping['tagged_served_bits'])} bit，"
        f"完全饱和比例 {_fmt(clipping['full_saturation_fraction'])}，"
        f"增量低于 0.25 的比例 {_fmt(clipping['below_full_boost_fraction'])}，"
        f"标签增量保留比例 {_fmt(clipping['boost_retained_fraction'])}，"
        f"被裁剪掉的比例 {_fmt(clipping['boost_suppressed_fraction'])}。",
        f"- {title} 的生成标签率均值 {_fmt(clipping['tag_draw_rate_mean'])}。"
        f"服务时普通块平均年龄 {_fmt(clipping['served_age_mean_ordinary'])} 步，"
        f"tagged 块平均年龄 {_fmt(clipping['served_age_mean_tagged'])} 步。",
        f"- {title} 的年龄加权服务合计 {_fmt(clipping['age_service_bits'])} bit，"
        f"标签增量合计 {_fmt(clipping['tag_increment_service_bits'])} bit，"
        f"合紧急度 {_fmt(clipping['combined_urgency_served_bits'])} bit。",
    ]


def _choice_text(choice: str | None) -> str:
    if choice == "A":
        return (
            "锁定规则给出 A：少于两个 L2 种子表现为“年龄服务配对均值为正，且大于标签增量配对均值”。"
            "在记账有效的前提下，当前结果不支持把下一步做成新的奖励公式。停止这条 priority-tag 奖励，不在本分支训练。"
        )
    if choice == "B":
        return (
            "锁定规则给出 B：至少两个 L2 种子的年龄服务配对均值为正，并且大于标签增量的配对均值。"
            "当前合紧急度主要跟着年龄走。下一步只应在新的实验分支上，把年龄和直接 tagged 服务分成两个有上界的项，"
            "再按预先写好的标准对照 L0。本阶段不选择新公式，也不训练。"
        )
    if choice == "C":
        return "锁定规则给出 C：保存的评估、置信区间或年龄/标签分解没有对上。先修正记账或仪器，不用这批数字解释奖励目标。"
    return "重放没有形成完整对照，因此不在 A、B、C 中选择。"


def render_report(summary: dict) -> str:
    lines = [
        "# 实验报告：紧急度奖励的年龄与标签归因",
        "",
        "- 诊断分支：`experiment/priority-tag-reward-attribution`",
        "- 来源实验分支：`experiment/priority-tag-urgency-reward`",
        f"- 来源结果提交：`{summary['source_result_commit']}`",
        f"- 指令提交：`{summary['instruction_commit']}`",
        f"- Gate：`{summary['gate']}`",
        "- 是否合并：否",
        "- 标签性质：synthetic priority tag，不是真实海事业务标签。",
        "",
        "## 1. 本阶段不做的事",
        "",
        "没有重新训练，没有改 lambda_u、rho、beta_tag、tag_rate、队列顺序、动作空间、观测、任务分布或训练预算。",
        "正式奖励、正式 Planner、正式观测、SafetyProjector 和 L0 checkpoint 保持原文件。",
        "来源实验分支不合并。诊断分支也不合并。",
        "",
        "## 2. 来源对账",
        "",
        f"保存的 Gate 是 `{summary['reconciliation']['saved_gate']}`。",
        f"CSV 行数 {summary['reconciliation']['row_count']}，期望 {summary['reconciliation']['expected_row_count']}。",
        f"按 CSV 重算后，区间和计数的最大绝对差是 {_fmt(summary['reconciliation']['max_abs_gap'], 6)}。",
        f"对账问题：{('无' if not summary['reconciliation']['problems'] else '；'.join(summary['reconciliation']['problems']))}。",
        "",
        "## 3. 定义",
        "",
        "服务前对每个 FIFO 桶计算 `u_age = clip(age/48, 0, 1)`，`u = clip(u_age + 0.25*synthetic_tag, 0, 1)`。",
        "年龄服务是 `u_age * served_bits`，标签增量是 `(u - u_age) * served_bits`，合紧急度是两者之和。",
        "tagged 服务只统计 `synthetic_tag = 1` 的实际出队 bit。空队列和 tag=0 不计入标签增量。",
        "FIFO 键仍是 admission step 和 insert index。标签不改变出队顺序。",
        "",
        "## 4. 重放",
        "",
        f"首个场景用时 {_fmt(summary['probe']['first_scenario_seconds'], 3)} 秒。",
        f"按这个速度估计全部 {summary['probe']['full_episode_count']} 段需要 {_fmt(summary['probe']['estimated_seconds'], 1)} 秒。",
        f"时间上限是 {_fmt(summary['probe']['time_limit_seconds'], 1)} 秒。",
        f"实际完成 {summary['replay']['completed_episodes']} 段，用时 {_fmt(summary['replay']['elapsed_seconds'], 1)} 秒。",
        f"相对已保存评估的最大差 {_fmt(summary['replay'].get('max_saved_gap_bits'), 6)} bit。",
        f"年龄加标签增量相对账本合紧急度的最大差 {_fmt(summary['replay'].get('max_decomposition_gap_bits'), 6)} bit。",
        "",
    ]
    failure = summary.get("failure")
    if failure:
        lines.extend(
            [
                f"停止于 {failure['variant']} seed {failure['model_seed']} 场景 {failure['scenario_id']}。",
                "仪器重放和已保存结果的字段差：",
            ]
        )
        for key, value in failure["field_gaps"].items():
            lines.append(f"- {key}：{_fmt(value, 6)}")
        lines.append(f"该场景的分解差是 {_fmt(failure['decomposition_gap_bits'], 6)} bit。")
        lines.append("同一检查点、同一确定性动作、不经过归因包装的原评估路径，字段差为：")
        for key, value in failure["uninstrumented_field_gaps"].items():
            lines.append(f"- {key}：{_fmt(value, 6)}")
        same = all(
            abs(float(failure["field_gaps"][key]) - float(failure["uninstrumented_field_gaps"][key])) <= 1.0e-6
            for key in failure["field_gaps"]
        )
        if same:
            lines.append("原评估路径和仪器路径的差相同，说明差不来自归因包装。")
        lines.append("这些差大于 1 bit 的记账容差。重放在这里停止，不用未完成的场景解释年龄或标签。")
        lines.append("")
    if summary.get("failure") and summary.get("next_design") == "C":
        lines.extend(
            [
                "## 5. 归因",
                "",
                "完整重放没有执行，因此没有年龄/标签对照表。",
                "",
                "## 6. 下一步",
                "",
                "字母是 C。已保存 CSV 和 summary 的区间差是 0，年龄加标签增量也回到账本里的合紧急度。",
                f"不能进入容差的是冻结评估的重新运行：payload 差 {_fmt(failure['field_gaps']['payload_bits'], 6)} bit，"
                f"普通服务差 {_fmt(failure['field_gaps']['served_ordinary_bits'], 6)} bit，"
                f"普通丢弃差 {_fmt(failure['field_gaps']['dropped_ordinary_bits'], 6)} bit，"
                f"路径差 {_fmt(failure['field_gaps']['path_length_m'], 6)} 米。",
                *(
                    ["归因包装和原评估路径给出同一组差，所以这不是包装改了动作。"]
                    if all(
                        abs(float(failure["field_gaps"][key]) - float(failure["uninstrumented_field_gaps"][key])) <= 1.0e-6
                        for key in failure["field_gaps"]
                    )
                    else ["归因包装和原评估路径的字段差不完全相同。"]
                ),
                "容差保持 1 bit，不因为重放差超过 1 bit 而放宽。",
                "在重放能够复现已保存 bit 之前，不用部分场景解释年龄或标签，不改奖励，也不训练。",
                "",
                "固定轨迹上的奖励上升不能代替 tagged 服务。正式分支和原始 checkpoint 保持不变。",
                "",
            ]
        )
        return "\n".join(lines)
    if summary.get("attribution") is None:
        lines.extend(
            [
                "## 5. 归因",
                "",
                "完整重放没有执行，因此没有年龄/标签对照表。",
                "",
                "## 6. 下一步",
                "",
                _choice_text(summary.get("next_design")),
                "",
            ]
        )
        return "\n".join(lines)
    attribution = summary["attribution"]
    lines.extend(["## 5. 年龄、标签和裁剪", ""])
    lines.extend(_clipping_lines("L0", attribution["l0_clipping"]))
    lines.append("")
    for group in attribution["groups"]:
        if group["variant"] != "L2":
            continue
        lines.append(
            f"- L2 seed {group['model_seed']} 相对 L0：年龄服务 {_interval(group['intervals']['age_service_bits'])}；"
            f"其中普通块年龄服务 {_interval(group['intervals']['age_service_ordinary_bits'])}；"
            f"标签增量 {_interval(group['intervals']['tag_increment_service_bits'])}；"
            f"合紧急度 {_interval(group['intervals']['combined_urgency_served_bits'])}；"
            f"tagged 服务 {_interval(group['intervals']['served_tagged_bits'])}；"
            f"payload {_interval(group['intervals']['payload_bits'])}。"
        )
        lines.append(
            f"- L2 seed {group['model_seed']} 的合紧急度变化里，普通块年龄占 {_fmt(group['ordinary_share_of_combined_delta'])}，"
            f"标签增量占 {_fmt(group['tag_share_of_combined_delta'])}。"
            f"完全饱和比例 {_fmt(group['clipping']['full_saturation_fraction'])}，"
            f"标签增量保留比例 {_fmt(group['clipping']['boost_retained_fraction'])}，"
            f"生成标签率 {_fmt(group['clipping']['tag_draw_rate_mean'])}，"
            f"服务时普通年龄 {_fmt(group['clipping']['served_age_mean_ordinary'])}，"
            f"tagged 年龄 {_fmt(group['clipping']['served_age_mean_tagged'])}。"
        )
        mechanism = group["mechanism_sums"]
        lines.append(
            f"- L2 seed {group['model_seed']} 的 tagged 变化拆成：浮标进出 {_fmt(mechanism['tagged_delta_buoy_entry_exit'])} bit，"
            f"共同浮标上的服务量变化 {_fmt(mechanism['tagged_delta_shared_volume'])} bit，"
            f"共同浮标上的标签比例变化 {_fmt(mechanism['tagged_delta_shared_mix'])} bit，"
            f"拆分缺口最大 {_fmt(mechanism['tagged_delta_split_gap_max'], 6)} bit。"
        )
    lines.extend(["", "### L1", ""])
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
            "## 6. 这六个问题的读法",
            "",
            "合紧急度等于年龄服务加标签增量。标签增量只来自 tag=1 且年龄还没有把紧急度顶到 1 的那部分 bit。",
            "因此合紧急度上升，可以只来自更老的普通块，而不增加 tagged bit。",
            "FIFO 不使用标签排序，所以同一个浮标内部不能优先服务 tagged 块。",
            "浮标进出和共同浮标的服务量变化对应跨浮标的目标选择；共同浮标上的标签比例变化才是桶内组成变化。",
            "生成标签率如果仍在 0.25 附近，tagged 服务变化就不是标签生成器造成的。",
            "当前奖励使用合紧急度。它直接奖励年龄，只在裁剪还没饱和时奖励标签。",
            "",
            "## 7. 下一步",
            "",
            _choice_text(summary.get("next_design")),
            "",
            "固定轨迹上的奖励上升不能代替 tagged 服务。正式分支和原始 checkpoint 保持不变。",
            "",
        ]
    )
    return "\n".join(lines)
