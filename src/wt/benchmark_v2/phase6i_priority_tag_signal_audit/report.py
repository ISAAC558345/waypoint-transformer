"""Chinese audit report. Every quoted number comes from the summary object."""
from __future__ import annotations


def _num(value, digits: int = 6) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, bool):
        return "是" if value else "否"
    return f"{float(value):.{int(digits)}f}"


def _rate(numerator, denominator) -> str:
    if denominator in (None, 0):
        return f"{numerator}/{denominator}"
    return f"{numerator}/{denominator} = {_num(float(numerator) / float(denominator), 6)}"


def render(summary: dict) -> str:
    gate = summary["gate"]
    inventory = summary["inventory"]
    recomputed = summary["recomputed_gate"]
    preflight = summary.get("preflight")
    lines = [
        "# 标签信号可见性、控制能力与安全退化诊断报告",
        "",
        f"- Gate：`{gate}`",
        "- 本 Gate 不是性能验收。E10 的 `DIRECT_TAG_REWARD_BENCHMARK_FAIL` 保持不变。",
        "- 本阶段没有训练、没有搜索系数、没有合并。",
        f"- 合并状态：{summary['merged']}",
        "- 标签是合成标签，不是外部业务优先级。",
        "- seed 2 只作为历史复现时重新施加的进程上下文。模型种子、场景 reset 种子和任务分配种子彼此分开。",
        "- 诊断轨迹使用评估分配种子 `7000000 + scenario_id`，不是训练分配种子 `321000000 + model_seed * 10000 + env_index`。",
        "",
        "## 1. E10 结果与来源",
        "",
        f"- 场景行数 {inventory['rows']}，唯一键 {inventory['unique_keys']}，问题 {inventory['problems'] or '无'}。",
        f"- 重算 Gate `{recomputed['gate']}`，与已保存 FAIL 一致：{recomputed['matches_saved_fail_gate']}。",
        f"- 通过种子数 {recomputed['passing_seeds']}。区间差异 {recomputed['interval_gaps'] or '无'}。",
        f"- 训练实现字节：{summary['source_audit']['implementation_bytes_at_training_time']}。指令提交只含指令：{summary['source_audit']['source_commit_is_instruction_only']}。",
        f"- checkpoint 问题：{summary['checkpoints']['problems'] or '无'}。",
        "",
        "重算后的 tagged 与 payload 均值仍以保存的场景表为准，负均值不是显著性结论。三个 tagged 区间是否跨零，以 E10 保存区间和本次重算差异为准。",
        "",
    ]
    if preflight is None:
        lines.append("- L0 / 321101 预检没有完成。")
    else:
        lines.append(
            f"- L0 / 321101 预检匹配 {preflight['matches_saved']}，最大 bit 差 {_num(preflight['max_bit_gap'])}，路径差 {_num(preflight['path_gap_m'], 12)} m。"
        )
    lines.extend(["", "## 2. 标签是否进入 actor 和控制窗口", ""])
    if not summary.get("exposure"):
        lines.append("自然轨迹没有运行，因此不能用本次诊断解释 E10 的失败机制。")
    else:
        lines.append("暴露率的分子和分母来自 8 条完整轨迹。生成标签的抽样率不作为可见率。")
        lines.append("")
        for item in summary["exposure"]["episodes"]:
            lines.append(
                f"- {item['variant']} seed {item['model_seed']} / {item['scenario_id']}："
                f"可见 tagged UAV 步 {_rate(item['visible_tagged_uav_steps'], item['uav_steps'])}；"
                f"其中策略掩码为正 {_rate(item['visible_and_policy_active_uav_steps'], item['visible_tagged_uav_steps'])}；"
                f"掩码范围 [{_num(item['policy_active_mask_min'], 3)}, {_num(item['policy_active_mask_max'], 3)}]；"
                f"队首为 tagged 的浮标步 {_rate(item['head_tagged_buoy_steps'], item['occupied_buoy_steps'])}；"
                f"入队 tagged 桶 {item['queued_tagged_buckets']} / 生成 tagged 事件 {item['generated_tagged_buckets']}；"
                f"正 tagged 服务步 {_rate(item['positive_tag_service_steps'], item['steps'])}；"
                f"可见延迟中位数 {_num(item['visible_delay_median'], 3)}（未删失 {item['visible_delay_observed']}，未可见 {item['never_visible_queued_buckets']}）；"
                f"服务延迟中位数 {_num(item['served_delay_median'], 3)}（未删失 {item['served_delay_observed']}，未服务 {item['never_served_queued_buckets']}）；"
                f"tagged 服务 {_num(item['served_tagged_bits'], 3)} bit，bonus_tag 合计 {_num(item['bonus_tag_sum'])}。"
            )
        lines.append("")
        lines.append("策略掩码在 worker 中是全 1。可见且策略有效因此等于可见 UAV 步，它不是学出来的活动门。")
        lines.append("新生成的 tagged 桶在生成步之后才可能进入下一步的 actor 输入。未成为队首或没有 UAV 关联的桶记为未可见，不补插延迟。")
    lines.extend(["", "## 3. 冻结 L3 对 tag 特征是否敏感", ""])
    if not summary.get("sensitivity"):
        lines.append("敏感性没有计算。")
    else:
        lines.append("比较只把 actor 输入的 tag 列置零。队列、年龄、分配和物理状态不变。动作不变只说明这些快照上不敏感。")
        lines.append("")
        for item in summary["sensitivity"]["episodes"]:
            if not item["actor_accepts_tag"]:
                lines.append(f"- {item['variant']} seed {item['model_seed']} / {item['scenario_id']}：不接受 tag 输入，未伪造置零。")
                continue
            lines.append(
                f"- L3 seed {item['model_seed']} / {item['scenario_id']}："
                f"对齐步 {item['aligned_steps']} / {item['steps']}，对齐失败 {item['alignment_failed_steps']}；"
                f"对齐且 tag 输入非零的步 {item['aligned_nonzero_tag_input_steps']}；"
                f"其中原始残差变化 {item['mu_changed_steps']}，执行动作变化 {item['executed_changed_steps']}；"
                f"残差差中位数 {_num(item['median_mu_gap'])}，最大 {_num(item['max_mu_gap'])}；"
                f"执行动作差中位数 {_num(item['median_executed_gap'])}。"
            )
    lines.extend(["", "## 4. 短时域动作反事实", ""])
    counter = summary.get("counterfactual")
    if not counter:
        lines.append("反事实没有运行。")
    else:
        lines.append(
            f"预定规则：每条轨迹 8 个时间等分左端点，共 {counter['snapshots']} 个快照；"
            f"每个快照 5 个候选、各 4 步；扰动幅度 {summary['protocol']['perturbation']}。"
            "候选是加在每架 UAV 残差上的同一偏移，经过同一裁剪和 SafetyProjector。"
        )
        lines.append(
            f"基线可复现快照 {counter['snapshots_reproduced']} / {counter['snapshots']}。"
            f"在可复现快照上，tagged 服务增加超过 1 bit、payload 下降不超过 1 bit、安全和约束失败不增加的非零扰动候选行数：{counter['opportunity_rows']}。"
        )
        statement = str(counter["statement"]).rstrip("。")
        lines.append(f"结论：{statement}。4 步筛选不是长期可控性的证明，也不是新策略的性能结果。")
    lines.extend(["", "## 5. 奖励与离线学习信号", ""])
    if not summary.get("reward_signal"):
        lines.append("奖励信号没有计算。")
    else:
        lines.append("以下比值和 GAE 都是诊断估计，不是 300 次更新期间的优势或梯度。分母为零的步单独计数。中位数使用全部分母非零的步，包含 bonus 为 0 的步，因此可以为 0；正 tag bonus 的步数和 float32 是否仍能分辨另计。")
        lines.append("")
        for item in summary["reward_signal"]["episodes"]:
            lines.append(
                f"- L3 seed {item['model_seed']} / {item['scenario_id']}："
                f"bonus_age 合计 {_num(item['bonus_age_sum'])}，bonus_tag 合计 {_num(item['bonus_tag_sum'])}，"
                f"payload 项合计 {_num(item['payload_term_sum'])}，运动惩罚合计 {_num(item['motion_penalty_sum'])}；"
                f"payload 项为零的步 {item['zero_payload_steps']}，运动惩罚为零的步 {item['zero_motion_steps']}；"
                f"正 tag bonus 步 {item['positive_tag_bonus_steps']}，其中 float32 仍能分辨 {item['float32_distinguishable_tag_steps']}；"
                f"tag/payload 中位数 {_num(item['tag_over_payload_median'])}，tag/运动惩罚中位数 {_num(item['tag_over_motion_median'])}。"
            )
        lines.append("")
        for item in summary["gae"]["episodes"]:
            if not item["computed"]:
                lines.append(f"- {item['variant']} seed {item['model_seed']} / {item['scenario_id']}：{item['reason']}")
                continue
            lines.append(
                f"- L3 seed {item['model_seed']} / {item['scenario_id']} 诊断估计："
                f"归一化优势里 tag 增量的平均绝对差 {_num(item['mean_abs_normalized_tag_effect'])}，"
                f"最大 {_num(item['max_abs_normalized_tag_effect'])}；"
                f"年龄增量的平均绝对差 {_num(item['mean_abs_normalized_age_effect'])}；"
                f"原始 GAE 的 tag 平均绝对差 {_num(item['mean_abs_raw_tag_effect'])}。"
            )
        lines.append("")
        lines.append("历史训练日志没有逐步 bonus_tag。不能从 mean_bonus 回推。")
        for item in summary["training_logs"]["seeds"]:
            lines.append(
                f"- 训练 seed {item['model_seed']}，{item['updates']} 次更新："
                f"KL 均值 {_num(item.get('approx_kl_mean'))}，最大 {_num(item.get('approx_kl_max'))}；"
                f"clip fraction 均值 {_num(item.get('clip_fraction_mean'))}；"
                f"value loss 均值 {_num(item.get('value_loss_mean'), 3)}；"
                f"explained variance 均值 {_num(item.get('explained_variance_mean'))}；"
                f"动作标准差均值 {_num(item.get('action_std_mean'))}；"
                f"投影率均值 {_num(item.get('projection_rate_mean'))}；"
                f"裁剪前梯度均值 {_num(item.get('gradient_norm_preclip_mean'), 3)}，最大 {_num(item.get('gradient_norm_preclip_max'), 3)}；"
                f"安全频率均值 {_num(item.get('safety_violation_rate_mean'))}。"
                "较大的裁剪前梯度本身不构成失稳结论。"
            )
    lines.extend(["", "## 6. 两处安全事件", ""])
    if not summary.get("safety"):
        lines.append("安全重放没有运行，不对 E10 的两处事件做因果归类。")
    else:
        lines.append("计数器在执行后的真实间距上使用 1e-9；投影器在预测间距上使用 1e-6。阈值都是 1.0 m。同一间距可以同时通过投影器检查并被计数器记为违规。证据标签可以并列，不改 SafetyProjector。")
        lines.append("")
        for item in summary["safety"]:
            lines.append(
                f"- {item['case']['variant']} seed {item['case']['model_seed']} / {item['case']['scenario_id']}："
                f"与 E10 保存行匹配 {item['matched_saved_rows']}，解释状态 {item['causal_interpretation']}，"
                f"当前 bit 差 {item['current_match']['bit_gaps']}，路径差 {_num(item['current_match']['path_gap_m'], 12)}。"
            )
            if not item["events"]:
                lines.append("  - 没有可归类的违规步。")
            for event in item["events"]:
                lines.append(
                    f"  - 步 {event['step']}，UAV {event['uav_i']} 与 {event['uav_j']}，"
                    f"间距 {_num(event['distance_m'], 9)} m，前一步间距 {_num(event['distance_before_m'], 9)} m，"
                    f"阈值 {_num(event['threshold_m'], 3)}，预测间距 {_num(event['predicted_distance_m'], 9)}，"
                    f"投影状态 {event['projection_status']}，约束失败 {event['constraint_failure']}。"
                )
                lines.append(f"    证据标签：{', '.join(event['labels'])}。")
    lines.extend(["", "## 7. 测试、证据缺口与下一步建议", ""])
    tests = summary["tests"]
    lines.append(f"- 相关测试命令：`{tests['related']['command']}`，退出码 {tests['related']['exit_code']}。")
    lines.append(f"- 相关测试失败：{tests['related']['failures'] or '无'}。")
    suite = tests.get("repository_suite")
    if suite is None:
        lines.append("- 全量现有测试没有运行。")
    else:
        lines.append(f"- 全量测试命令：`{suite['command']}`，退出码 {suite['exit_code']}。")
        lines.append(f"- 全量测试失败 {len(suite['failures'])} 项：{suite['failures']}。这些失败来自日志中的历史锁定文件缺失，审计相关测试本身退出码为 0。失败项已保留。")
    lines.append(f"- 环境：{summary['runtime']['python']}；NumPy {summary['runtime']['numpy']}；Torch {summary['runtime']['torch']}；CUDA {summary['runtime']['cuda']}。")
    lines.append(f"- 证据缺口：{summary['evidence_gaps']}")
    lines.append("")
    lines.append(summary["recommendation"])
    lines.append("")
    lines.append("本报告到此停止，不启动下一阶段实验。")
    lines.append("")
    return "\n".join(lines)
