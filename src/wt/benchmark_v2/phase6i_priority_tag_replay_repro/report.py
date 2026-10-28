"""Chinese report for the single-scenario replay check."""
from __future__ import annotations

from .protocol import ACCOUNTING_TOLERANCE_BITS, PATH_EQUAL_METERS


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "缺失"
    return f"{float(value):,.{digits}f}"


def _cause_text(letter: str) -> str:
    texts = {
        "A": "原因字母 A：同一套代码和同一个 checkpoint 下，重放进程没有继承训练结束时留下的 torch 线程数和确定性算法开关。",
        "B": "原因字母 B：reset 后的队列或位置已经不同，差异从初始状态进入。",
        "C": "原因字母 C：动作张量或 SafetyProjector 出现数值差，但还没有证明它来自未记录的进程开关。",
        "D": "原因字母 D：动作一致时，队列、snap 或累计 payload 先分叉。",
        "E": "原因字母 E：只有汇总写盘精度不同。路径差不支持这个判断。",
        "F": "原因字母 F：现有证据还不能确定直接原因。",
    }
    return texts.get(letter, texts["F"])


def _next_text(gate: str) -> str:
    if gate == "REPLAY_DIVERGENCE_LOCALIZED":
        return (
            "下一步只记录并恢复评估进程的 torch 线程数和确定性算法开关，再对这一条场景复核 1 bit 容差。"
            "不要据此重训，不要改奖励，也不要恢复 224 条归因。"
        )
    if gate == "REPLAY_DIVERGENCE_BLOCKED":
        return "下一步先补齐 checkpoint、代码哈希或 split 证据。不要解释年龄和标签，不要合并。"
    return "下一步仍然是补齐可复现证据。不要用这次未闭合的差异解释年龄和标签，不要合并。"


def render_report(summary: dict) -> str:
    runtime = summary.get("runtime_before", {})
    lines = [
        "# 实验报告：冻结 L0 场景 321101 的重放复现",
        "",
        f"- Gate：`{summary.get('gate')}`",
        f"- 原因字母：`{summary.get('cause_letter')}`",
        "- 诊断分支：`experiment/priority-tag-replay-repro`",
        f"- 指令提交：`{summary.get('instruction_commit')}`",
        f"- 来源实验提交：`{summary.get('source_experiment_commit')}`",
        f"- 来源诊断提交：`{summary.get('source_diagnostic_commit')}`",
        "- 是否合并：否",
        "- 是否重新训练：否",
        "",
        "## 1. 范围",
        "",
        "只检查 L0、模型种子 0、development 场景 321101 的 48 步。",
        "没有重训 L1/L2，没有改奖励、动作、标签、队列顺序或 SafetyProjector。",
        f"记账容差保持 {ACCOUNTING_TOLERANCE_BITS:g} bit。路径相等检查是 {PATH_EQUAL_METERS:g} 米，用来判断轨迹是否同一条，不代替 1 bit 容差。",
        "",
        "## 2. 来源与环境",
        "",
        f"正式 checkpoint SHA-256：`{summary.get('formal_actor_sha256')}`。",
        f"来源实验提交中的相关文件哈希一致：{summary.get('blobs_match_source_experiment')}。",
        f"场景 split 校验通过：{summary.get('split_ok')}。",
        f"Python：{runtime.get('python', '缺失')}",
        f"NumPy：{runtime.get('numpy')}；仓库 requirements 写的是 {runtime.get('requirements_numpy')}。",
        f"PyTorch：{runtime.get('torch')}；仓库 requirements 写的是 {runtime.get('requirements_torch')}。",
        f"CUDA 可用：{runtime.get('cuda_available')}。",
        f"重放开始前 torch 线程数：{runtime.get('torch_threads_before')}；确定性算法开关：{runtime.get('deterministic_algorithms_before')}。",
        "已保存的评估 CSV 和 summary 没有逐步动作、观测、snap 或上述进程开关。",
        "",
        "## 3. 对照",
        "",
        "fresh 是新进程的默认 torch 状态，走未包装的原评估路径。",
        "instrumented 使用同一默认状态和归因包装。",
        "inherited 在同一进程中补上训练结束时留下的 `torch.set_num_threads(1)` 和 `seed_everything(2)`，再走未包装路径。",
        "这三趟都只跑场景 321101，不扩大到 224 条。",
        "",
        f"reset 状态一致：{summary.get('reset_equal')}。",
        f"包装路径和 fresh 逐步一致：{summary.get('instrumented_same_as_fresh')}。",
        f"fresh 相对已保存结果进入 1 bit 且路径相等：{summary.get('fresh_matches_saved')}。",
        f"inherited 相对已保存结果进入 1 bit 且路径相等：{summary.get('inherited_matches_saved')}。",
        "",
    ]
    for label, gaps in (
        ("fresh", summary.get("fresh_gaps")),
        ("instrumented", summary.get("instrumented_gaps")),
        ("inherited", summary.get("inherited_gaps")),
    ):
        if not gaps:
            continue
        lines.append(
            f"- {label} 差：payload {_fmt(gaps['payload_bits'])} bit，普通服务 {_fmt(gaps['served_ordinary_bits'])} bit，"
            f"tagged 服务 {_fmt(gaps['served_tagged_bits'])} bit，普通丢弃 {_fmt(gaps['dropped_ordinary_bits'])} bit，"
            f"tagged 丢弃 {_fmt(gaps['dropped_tagged_bits'])} bit，紧急度 {_fmt(gaps['urgency_served_bits'])} bit，"
            f"路径 {_fmt(gaps['path_length_m'])} 米。"
        )
    found = summary.get("fresh_vs_inherited")
    lines.extend(["", "## 4. 首次分叉", ""])
    if not found:
        lines.append("fresh 和 inherited 的逐步记录没有数值差，因此没有首次分叉步骤。")
    else:
        lines.append(
            f"fresh 相对 inherited 的第一处差异在 step {found.get('step_index')}，字段 `{found.get('field')}`。"
        )
        if found.get("field") != "observation":
            lines.append("该步的观测哈希一致，分叉发生在同一观测上的后续数值。")
        nonzero = found.get("nonzero") or {}
        for key, value in nonzero.items():
            lines.append(f"- {key} 绝对差 {_fmt(value, 12)}")
    lines.extend(
        [
            "",
            "## 5. 判定",
            "",
            _cause_text(str(summary.get("cause_letter"))),
            "线程数和确定性算法开关是一起恢复的，本阶段没有拆开单独测试。",
            "当前安装的 PyTorch 在恢复这些开关后能够对上已保存结果，因此这次 1 bit 以外的差不是 requirements.txt 里旧版本号本身造成的。",
            "包装路径如果和 fresh 逐步相同，E6 里两条路径一起偏离保存结果就不是包装造成的。",
            _next_text(str(summary.get("gate"))),
            "",
            "正式 checkpoint 和正式代码没有改。历史评估文件没有改。诊断分支不合并。",
            "",
        ]
    )
    return "\n".join(lines)
