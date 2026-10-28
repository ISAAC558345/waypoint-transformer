"""Chinese report for the clip-group pilot. Numbers come from the summary."""
from __future__ import annotations

from .protocol import EFFECT_MEASURABLE, GATE_COMPLETE, REPORT


def _fmt(value) -> str:
    if value is None:
        return "空"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def _comparison_lines(summary: dict) -> list[str]:
    lines = []
    for seed in summary.get("effect", {}).get("per_seed", []):
        for row in seed.get("comparisons", []):
            lines.append(
                f"- seed {seed['model_seed']} / {row['source_variant']} / update {row['ppo_update']}: "
                f"R={_fmt(row.get('r'))}，可用={row.get('usable')}"
            )
    if not lines:
        lines.append("- 没有进入效应规则的 16/32 同源比较。")
    return lines


def _run_lines(summary: dict) -> list[str]:
    lines = []
    for run in summary.get("runs", []):
        lines.append(
            "- {variant} seed {model_seed}: PPO 更新 {ppo_updates}，Adam 步 {adam_steps}，"
            "影子 Adam 步 {shadow_adam_steps}，安全计数 {safety_violation_count}，"
            "约束失败 {constraint_failure_count}，payload 均值 {mean_served_bits}，"
            "标记服务均值 {mean_served_tagged_bits}，KL 均值 {mean_approx_kl}，KL 最大 {max_approx_kl}".format(
                variant=run.get("variant"),
                model_seed=run.get("model_seed"),
                ppo_updates=run.get("ppo_updates"),
                adam_steps=run.get("adam_steps"),
                shadow_adam_steps=run.get("shadow_adam_steps"),
                safety_violation_count=run.get("safety_violation_count"),
                constraint_failure_count=run.get("constraint_failure_count"),
                mean_served_bits=_fmt(run.get("mean_served_bits")),
                mean_served_tagged_bits=_fmt(run.get("mean_served_tagged_bits")),
                mean_approx_kl=_fmt(run.get("mean_approx_kl")),
                max_approx_kl=_fmt(run.get("max_approx_kl")),
            )
        )
    if not lines:
        lines.append("- 没有完成的运行记录。")
    return lines


def _recommendation(summary: dict) -> str:
    safety = int(summary.get("safety_violation_count") or 0)
    constraints = int(summary.get("constraint_failure_count") or 0)
    listed = f"安全计数合计 {safety}，约束失败合计 {constraints}。"
    if summary.get("recommendation") == "register_one_full_budget_clip_group_comparison":
        return (
            "唯一的下一步建议是另开一次预注册的全预算对照，只比较同一种裁剪分组、同一奖励和同一组种子。"
            "本次没有实现该对照，也没有把更新数加到 300。"
            + listed
        )
    return "本次不建议扩大训练或把试点延长到 300 次更新。" + listed


def write_report(summary: dict) -> None:
    comparison = summary.get("test_comparison") or {}
    frozen = summary.get("frozen_gates") or {}
    lines = [
        "# 优化器裁剪分组试验",
        "",
        f"- trial_gate: `{summary.get('scientific_gate')}`",
        f"- effect_assessment: `{summary.get('effect_assessment')}`",
        f"- PPO 更新: {summary.get('ppo_updates')}",
        f"- Adam 步: {summary.get('adam_steps')}",
        f"- 影子 Adam 步: {summary.get('shadow_adam_steps')}",
        "- 性能结论: 无。本报告不给出性能提升结论。",
        "",
        "COMPLETE 只表示这组机制试验跑完。MEASURABLE 只表示在同一源状态上，改变 actor/critic 的裁剪分组后，实际 Adam 参数更新发生了预注册屏幕上的变化。它不解释 E12 的失败，也不表示策略变好。",
        "",
        f"E12 仍是 `{frozen.get('e12_gate')}`，passing_seeds 仍是 {frozen.get('e12_passing_seeds')}。E13 仍是 `{frozen.get('e13_gate')}`，建议仍是 `{frozen.get('e13_recommendation')}`。`release_eligible` 为 false，`merged` 为 false。",
        "",
        "## 干预",
        "",
        "两组都使用 E12 的 L4_Q 输入：actor 是原来的 3 列头部特征加上两列真实队列前瞻，critic 仍是 3 列头部状态。初始化是同一次 LEGACY 热启动，新列权重从 0 开始。奖励、GAE、优势归一化、策略/价值/熵损失和团队广播都保持原样。",
        "",
        "PILOT_J 按原来的方式把 actor 和 critic 放在一起，用注册配置里的 G=0.5 做一次裁剪。PILOT_S 把 actor、critic 分开裁剪，每组上界是 G/sqrt(2)。分开之后的联合范数上界仍然是 G。两组各用一个 Adam，超参数和参数顺序相同。",
        "",
        "注册预算是 3 个模型种子、每组 32 次 PPO 更新，一共 6 次运行。每次 PPO 更新内部有多个 Adam 步。影子干预只在第 1、16、32 次更新的第一个 epoch、第一个 minibatch，于真实裁剪和真实 step 之前，对三份独立副本各做一次 Adam 步，然后丢弃。第 1 次更新只记录，不进入效应规则。",
        "",
        "## 运行计数",
        "",
        * _run_lines(summary),
        "",
        f"原因记录: {summary.get('reason')}",
        "",
        "## 同一源状态上的 Adam 更新差",
        "",
        "R 只在 J 副本的 actor 更新范数非零、且 J 重复副本通过时定义。描述性屏幕是 R >= 0.01。这 1% 只用于选择后续研究方向，不是显著性、性能收益或安全界。零范数不计算比值。",
        "",
        * _comparison_lines(summary),
        "",
        f"达到屏幕的种子数: {summary.get('effect', {}).get('measurable_seed_count')}",
        f"有效比较数: {summary.get('effect', {}).get('valid_effect_comparisons')}",
        f"覆盖完整: {summary.get('effect', {}).get('complete_coverage')}",
        "",
        "## 训练过程中的安全、约束、KL 和 payload",
        "",
        "下面的数值来自训练池上的 32 次滚动，只作描述。它们不是公平 benchmark，也不是独立场景样本。本试验没有为 KL 或 payload 另设数值门槛。非有限的损失、梯度或 KL 会直接使试验 BLOCKED。",
        "",
        * _run_lines(summary),
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
                "- 对照使用 2bce391 的干净工作区，不使用主工作区里残留文件的那次执行。既有失败仍然是发布阻塞项。",
            ]
        )
    else:
        lines.append("- 全套测试对照尚未写入本摘要。")
    lines.extend(
        [
            "",
            "## 建议",
            "",
            _recommendation(summary),
            "",
            "## 证据边界",
            "",
            "历史静态梯度比值不能代替这次实际 Adam 更新。联合梯度被 critic 主导、以及单独显示的裁剪系数，都没有被当成历史 Adam 已经缩小了更新的证据。",
            "",
            "这次试点不能判断 300 次更新后的策略质量。低于屏幕的结果只覆盖本次预算和观察到的源状态。影子步骤没有环境滚动，也不能叫作额外的 PPO 更新。",
            "",
        ]
    )
    if summary.get("scientific_gate") == GATE_COMPLETE and summary.get("effect_assessment") == EFFECT_MEASURABLE:
        lines.append("效应判断只说明裁剪分组改变了同一源状态上的实际更新。")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
