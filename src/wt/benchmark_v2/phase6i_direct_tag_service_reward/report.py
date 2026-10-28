"""Chinese L3 report. Numbers come from the summary."""
from __future__ import annotations

from .protocol import GATE_BLOCKED, GATE_INCONCLUSIVE, GATE_PASS


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "缺失"
    return f"{float(value):,.{digits}f}"


def _interval(item: dict | None) -> str:
    if not item:
        return "缺失"
    return f"均值 {_fmt(item.get('mean'))}，95% 区间 [{_fmt(item.get('ci_low'))}, {_fmt(item.get('ci_high'))}]"


def _next(summary: dict) -> str:
    gate = summary.get("gate")
    if gate == GATE_BLOCKED:
        return "实现、训练、重放或哈希没有通过。不调 lambda_tag 或 rho_tag，不延长训练，不合并。"
    if gate == GATE_PASS:
        return "L3 达到第 6 节的全部预注册条件。只允许包含本实验独立代码、测试、报告和结果的差异进入正式分支。"
    if summary.get("metrics", {}).get("merge") and int(summary.get("unexpected_formal_diff_count") or 0) > 0:
        return "指标达到合并门槛，但当前分支相对正式分支还包含此前未合并的 priority-tag 实验历史。本次不合并，也不把这条历史整支并入正式分支。"
    return "L3 没有同时满足 tagged 服务、payload、安全、非单场景和相对 L1 的预注册条件。不改权重再跑，不合并。"


def render_report(summary: dict) -> str:
    metrics = summary.get("metrics") or {}
    lines = [
        "# 实验报告：直接 tagged-service 有界奖励",
        "",
        "- 实验分支：`experiment/direct-tag-service-reward`",
        f"- 指令提交：`{summary.get('instruction_commit')}`",
        f"- Gate：`{summary.get('gate')}`",
        "- 是否合并：否" if not summary.get("merged") else "- 是否合并：是",
        "- 是否重新训练 L0/L1/L2：否",
        "- 标签性质：synthetic priority tag，不是真实海事业务标签。",
        "",
        "## 1. 奖励",
        "",
        "基线奖励保持 `served_bits/1e6 - 0.05*c_project - 0.01*c_residual`。",
        "L3 只在此基础上增加 `min(0.002 * served_tagged_bits/1e6, 0.001 * served_bits/1e6)`。",
        "served_tagged_bits 是当前步从 tag=1 FIFO 桶实际服务的 bit。年龄加权量和合紧急度不进入 L3 bonus。",
        "因为 tagged bit 不超过总服务量，bonus 位于 0 和 `0.001 * served_bits/1e6` 之间。服务量为 0 时 bonus 为 0。",
        "这个上界不能保证训练后的 episode payload 不下降。",
        "",
        "## 2. 运行上下文",
        "",
        "训练沿用 L1/L2 的入口：每个模型种子调用 `seed_everything(model_seed)`，线程数为 1，初始化是 legacy checkpoint 的显式零列移植，不从 L1/L2 热启动。",
        "评估使用已验证的历史继承上下文：线程数 1，确定性算法打开，全局 seed 2。这个 2 不是评估协议种子。",
        "环境 reset seed 是场景编号，分配 RNG 是 7000000 加场景编号。",
        "",
        "## 3. Smoke、预检和训练",
        "",
        f"Smoke 通过：{summary.get('smoke', {}).get('passed')}。问题：{summary.get('smoke', {}).get('problems') or '无'}。",
        f"L0 场景 321101 预检进入原容差：{summary.get('preflight', {}).get('matches_saved')}。"
        f"bit 差 {_fmt(summary.get('preflight', {}).get('max_bit_gap'), 6)}，路径差 {_fmt(summary.get('preflight', {}).get('path_gap_m'), 9)} 米。",
        f"训练完成：{summary.get('train_ok')}。正式哈希未变：{summary.get('hashes_ok')}。",
        "",
    ]
    for item in summary.get("training") or []:
        lines.append(
            f"- L3 seed {item['model_seed']} sha256 `{item['sha256']}`，用时 {_fmt(item['elapsed_seconds'], 1)} 秒，"
            f"最终 policy loss {_fmt(item['final_policy_loss'], 6)}，value loss {_fmt(item['final_value_loss'], 3)}，"
            f"梯度范数 {_fmt(item['final_gradient_norm'], 3)}。"
        )
    failure = summary.get("failure")
    if failure:
        lines.extend(["", f"停止原因：{failure}。", ""])
    replay = summary.get("replay") or {}
    lines.extend(
        [
            "",
            "## 4. 已保存评估的复核",
            "",
            f"L0/L1/L2 复核条数 {replay.get('checked', 0)}，失配 {replay.get('mismatches', '缺失')}。",
            f"最大 bit 差 {_fmt(replay.get('max_bit_gap'), 6)}，最大路径差 {_fmt(replay.get('max_path_gap_m'), 9)} 米。",
            "",
            "## 5. L3 配对结果",
            "",
        ]
    )
    if not metrics.get("seeds"):
        lines.append("没有形成完整的 L3 配对表。")
    for seed in metrics.get("seeds") or []:
        lines.append(
            f"- L3 seed {seed['model_seed']} 相对 L0：tagged {_interval(seed.get('tagged_versus_l0'))}；"
            f"payload {_interval(seed.get('payload_versus_l0'))}；"
            f"年龄服务 {_interval(seed.get('age_versus_l0'))}；"
            f"标签增量 {_interval(seed.get('tag_increment_versus_l0'))}。"
        )
        lines.append(
            f"- 同一种子相对 L1 的 tagged {_interval(seed.get('tagged_versus_l1'))}；"
            f"相对 L2 的 tagged {_interval(seed.get('tagged_versus_l2'))}。"
        )
        lines.append(
            f"- 安全差合计 {_fmt(seed.get('safety_diff_sum'), 3)}，约束差合计 {_fmt(seed.get('constraint_diff_sum'), 3)}，"
            f"丢弃均值 {_fmt(seed.get('dropped_versus_l0_mean'), 3)}，队列均值 {_fmt(seed.get('queue_versus_l0_mean'), 3)}，"
            f"SINR 均值 {_fmt(seed.get('sinr_versus_l0_mean'), 6)}，干扰均值 {_fmt(seed.get('interference_versus_l0_mean'), 6)}。"
        )
        negative = seed.get("payload_negative_scenarios") or []
        lines.append(
            f"- payload 劣于 L0 的场景数 {len(negative)}。"
            f"预注册是否通过：{seed.get('passes_preregistered_criteria')}。"
        )
    lines.extend(
        [
            "",
            f"通过种子数 {metrics.get('passing_seeds', 0)}，合并所需至少 2。",
            "通过条件是：相对 L0 的 tagged 均值大于 0 且 95% 区间下界大于 0；payload 均值非负且区间下界非负；"
            "安全与约束差合计不增加；丢弃、最终队列和干扰的配对均值不增加；SINR 配对均值不下降；"
            "去掉任一场景后 tagged 均值仍大于 0；相对同一序号的 L1，tagged 均值大于 0 且区间下界大于 0。",
            "",
            "## 6. 下一步",
            "",
            _next(summary),
            "",
        ]
    )
    return "\n".join(lines)
