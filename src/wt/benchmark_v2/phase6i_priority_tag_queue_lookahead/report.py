"""Chinese report for the registered queue-lookahead comparison."""
from __future__ import annotations

from .protocol import DISTANCE_THRESHOLD_M, E11_RESULT_COMMIT, FORMAL_ACTOR_SHA256, INSTRUCTION_COMMIT, PAYLOAD_CI_FLOOR_FRACTION


def _num(value) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return repr(float(value))


def _interval(item: dict) -> str:
    return f"均值 {_num(item['mean'])}，95% CI [{_num(item['ci_low'])}, {_num(item['ci_high'])}]"


def _seed_block(item: dict) -> list[str]:
    lines = [f"### 模型种子 {item['model_seed']}", ""]
    lines.append(f"- 是否同时满足五条预注册正向标准：{_num(bool(item['passes_preregistered_criteria']))}")
    for key, title in (("versus_l4_c", "相对同种子 L4_C"), ("versus_l0", "相对 L0"), ("versus_l3", "相对同种子 L3（描述性历史参考）")):
        view = item.get(key)
        if not view:
            continue
        lines.append(f"- {title} tagged 服务：{_interval(view['tagged'])}")
        lines.append(
            f"- {title} payload：{_interval(view['payload'])}；该参考平均 payload 的 -0.5% 下界为 {_num(view['payload_ci_floor'])}"
        )
        lines.append(
            f"- {title} 安全配对差之和 {_num(view['safety_diff_sum'])}，约束失败配对差之和 {_num(view['constraint_diff_sum'])}"
        )
        lines.append(f"- {title} SINR 平均差 {_num(view['sinr_mean'])}，干扰平均差 {_num(view['interference_mean'])}")
        lines.append(
            f"- {title} 留一 tagged 均值全部大于 0：{_num(bool(view['leave_one_tagged']))}；去掉两条最大 tagged 增益后均值仍大于 0：{_num(bool(view['leave_two_tagged']))}"
        )
        lines.append(f"- {title} tagged 为负的场景：{view['tagged_negative_scenarios']}")
        lines.append(f"- {title} payload 为负的场景：{view['payload_negative_scenarios']}")
    lines.append("")
    return lines


def render_report(summary: dict) -> str:
    gate = str(summary["scientific_gate"])
    supported = "得到支持" if gate.endswith("PASS") else "没有得到支持"
    if gate.endswith("INCONCLUSIVE"):
        supported = "证据不足以支持"
    if gate.endswith("BLOCKED"):
        supported = "没有完成可判定的检验"
    lines = [
        "# 队列前瞻观测实验报告",
        "",
        f"- 科学 Gate：`{gate}`",
        f"- 发布资格：{'是' if summary.get('release_eligible') else '否'}",
        f"- 研究假设：{supported}",
        "- 适用范围：只覆盖本轮注册的开发池 321096–321127。该池已被前期实验使用，不能外推到未见场景。标签是入队时已知的 synthetic tag，不表示真实业务已经拥有该标记。",
        f"- 指令提交：`{INSTRUCTION_COMMIT}`",
        f"- 实现提交：`{summary.get('implementation_commit', '尚未形成')}`",
        f"- 前序审计提交：`{E11_RESULT_COMMIT}`，Gate 保持 DIRECT_TAG_SIGNAL_AUDIT_COMPLETE。E10 Gate 保持 DIRECT_TAG_REWARD_BENCHMARK_FAIL，奖励系数没有改动。",
        "- 合并状态：未合并。`merged` 为 false。",
        "",
        "## 假设与对照",
        "",
        "本轮只检验一件事：奖励、任务分配、FIFO 和训练算法保持不变时，让 actor 在 tagged 桶成为队首之前看到已入队标签积压和 FIFO 阻挡，是否能提高实际 tagged 服务，同时守住 payload 与安全。",
        "",
        "L4_C 与 L4_Q 使用同一五列 actor、同一初始化和同一 E10 L3 奖励。L4_C 的两列前瞻输入始终为零。L4_Q 使用服务前的 T/Q 与 B/Q。两组 critic 都保持 L3 的三列队首输入。这个对照不能解释成 critic 信息实验。",
        "",
        "低可见率不是已经证明的唯一失败原因。本结果无论成败，都不作为加大奖励或改高层的依据。",
        "",
        "## 阶段 A",
        "",
    ]
    stage = summary.get("stage_a") or {}
    lines.append(f"- 可行性：{stage.get('feasibility_status', '缺失')}。8 条冻结轨迹上，有效关联、队首为普通桶且后方已有 tagged 积压的 UAV-step 数为 {stage.get('ordinary_head_with_tagged_backlog_uav_steps', '缺失')}。")
    lines.append(f"- 安全复现：保存行对齐 {stage.get('safety_matched_saved_rows', '缺失')}。正式 checkpoint SHA-256 为 `{stage.get('formal_actor_sha256', FORMAL_ACTOR_SHA256)}`，核对 {stage.get('formal_actor_hash_ok', '缺失')}。")
    lines.append(f"- 安全距离阈值 { _num(DISTANCE_THRESHOLD_M) } m。计数器容差 1e-9 m，投影器容差 1e-6 m。数值边界只作诊断，不免除 E10 的违规计数。")
    for case in stage.get("safety_cases") or []:
        identity = case["case"]
        lines.append(
            f"- {identity['variant']} seed {identity['model_seed']} / {identity['scenario_id']} 对齐保存行为 {_num(bool(case['matched_saved_rows']))}。"
        )
        for event in case.get("events") or []:
            lines.append(
                f"  - step {event['step']}，UAV {event['uav_i']} 与 {event['uav_j']}。上一时刻该对距离 {_num(event['before']['distance_m'])} m，有符号余量 {_num(event['before']['signed_margin_m'])} m。"
            )
            lines.append(
                f"  - 执行后距离 {_num(event['after']['distance_m'])} m，余量 {_num(event['after']['signed_margin_m'])} m。计数器判违规 {_num(bool(event['after']['counter_violation']))}，按 1e-6 m 容差该距离会被接受 {_num(bool(event['after']['projector_would_accept_distance']))}。"
            )
            lines.append(
                f"  - 同一对的预测距离 {_num(event['predicted_pair']['distance_m'])} m，预测减执行 {_num(event['predicted_minus_after_m'])} m。投影器记录的全局最小预测距离 {_num(event['projector_minimum_predicted_distance_m'])} m，投影状态 {event['projection_status']}，约束失败 {event['constraint_failure']}。"
            )
    lines.extend(["", "三种分母按轨迹分别列出。全部 UAV-step 包含没有有效任务关联的步，不能与约 25% 的生成标签率直接比较。", ""])
    lines.append("| 轨迹 | 全部 UAV-step | 有效关联 | 有效关联且有 tagged 积压 | 队首普通且有 tagged 积压 | 双侧可见桶 | 提前步数中位数 | 前瞻删失 | 队首删失 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for episode in stage.get("episodes") or []:
        lines.append(
            f"| {episode['variant']} seed {episode['model_seed']} / {episode['scenario_id']} | {episode['uav_steps']} | {episode['valid_association_uav_steps']} | {episode['tagged_backlog_valid_uav_steps']} | {episode['ordinary_head_with_tagged_backlog_uav_steps']} | {episode['lookahead_before_head_observed']} | {episode['lookahead_advance_median']} | {episode['censored_lookahead']} | {episode['censored_head']} |"
        )
    lines.extend(
        [
            "",
            "提前步数只在同一已入队桶既进入过分配 UAV 的队列视野、后来又成为队首时计算。从未进入该视野或从未成为队首的桶保留为删失，不插补。",
            "",
            "## 训练与评估",
            "",
            f"- 奖励保持 `r_base + min(0.0005*A, 0.00025*P) + min(0.0010*T, 0.00075*P)`。payload CI 下限比例为 {_num(PAYLOAD_CI_FLOOR_FRACTION)}。",
            "- 初始化来自 legacy warm start，再把新增两列权重置零。不从训练后的 L3 热启动。",
            "- 评估进程种子角色是 historical_inherited_process_context。它不是训练种子。评估分配种子是 7000000 + scenario_id。环境重置种子是 scenario_id。",
            "",
        ]
    )
    score = summary.get("score")
    if score:
        lines.append(f"通过全部正向标准的种子数：{score['passing_seeds']}。科学 PASS 要求至少 2 个。")
        lines.append("")
        for item in score["seeds"]:
            lines.extend(_seed_block(item))
    else:
        lines.append("没有形成完整的配对评分。")
        lines.append("")
    lines.extend(["## 测试", ""])
    tests = summary.get("tests") or {}
    lines.append(f"- 相关测试通过：{tests.get('related_ok', '缺失')}。完整测试新增失败数：{tests.get('new_failure_count', '缺失')}。")
    lines.append(f"- 干净基线 Ran：{tests.get('baseline_ran', '缺失')}。新代码完整测试 Ran：{tests.get('repository_ran', '缺失')}。")
    if tests.get("historical_ids"):
        lines.append(f"- 与干净基线相同的既有失败：{tests['historical_ids']}")
    if tests.get("new_ids"):
        lines.append(f"- 基线中没有的失败：{tests['new_ids']}")
    lines.append("- 发布条件不因既有缺失产物而满足。没有伪造锁文件，也没有删除测试。")
    lines.extend(["", "## 发布阻塞", ""])
    for item in summary.get("release_blockers") or []:
        lines.append(f"- {item}")
    if summary.get("evidence_gaps"):
        lines.extend(["", "## 证据边界", ""])
        for item in summary["evidence_gaps"]:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 停止",
            "",
            "本报告交付科学结果和发布阻塞项后停止。不改奖励、不延长训练、不更换种子，也不启动下一组实验。",
            "",
        ]
    )
    return "\n".join(lines)
