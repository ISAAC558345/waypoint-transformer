"""Chinese report for the explicit evaluation-context check."""
from __future__ import annotations

from .protocol import ACCOUNTING_TOLERANCE_BITS, PATH_TOLERANCE_M


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "缺失"
    return f"{float(value):,.{digits}f}"


def _gap_line(name: str, match: dict) -> str:
    gaps = match["gaps"]
    return (
        f"- {name} 相对已保存结果：payload {_fmt(gaps['payload_bits'])} bit，"
        f"普通服务 {_fmt(gaps['served_ordinary_bits'])} bit，tagged 服务 {_fmt(gaps['served_tagged_bits'])} bit，"
        f"普通丢弃 {_fmt(gaps['dropped_ordinary_bits'])} bit，tagged 丢弃 {_fmt(gaps['dropped_tagged_bits'])} bit，"
        f"紧急度 {_fmt(gaps['urgency_served_bits'])} bit，路径 {_fmt(gaps['path_length_m'])} 米，"
        f"进入容差：{match['matches']}。"
    )


def _next(gate: str) -> str:
    if gate == "EVAL_CONTEXT_REPRO_PASS":
        return "下一步才可以制定带有这份显式 manifest 的完整归因重放方案。本阶段不启动那次重放，不重训，不合并。"
    if gate == "EVAL_CONTEXT_REPRO_INCONCLUSIVE":
        return "两次新进程彼此一致，但没有在原容差内对上历史结果。先继续记录缺失的历史进程元数据，不要把当前可重复写成历史已复现，也不要解释年龄和标签。"
    return "两次进程不一致或完整性检查失败。先修复运行环境证据，不要解释年龄和标签，不要合并。"


def render_report(summary: dict) -> str:
    lines = [
        "# 实验报告：显式评估运行时上下文的单场景复核",
        "",
        f"- Gate：`{summary.get('gate')}`",
        "- 分支：`experiment/priority-tag-eval-context`",
        f"- 指令提交：`{summary.get('instruction_commit')}`",
        f"- 来源 E7 提交：`{summary.get('source_e7_commit')}`",
        f"- 运行时 HEAD：`{summary.get('head')}`",
        "- 是否合并：否",
        "- 是否重新训练：否",
        "",
        "## 1. 种子角色",
        "",
        "报告里的模型种子 0 是 L0 checkpoint 和已保存 CSV 的 model_seed 列。",
        "环境 reset seed 是场景编号 321101。分配随机数种子是评估 worker 使用的 7000000+321101。",
        "全局 Python/NumPy/Torch seed 的数值 2 只代表历史进程里最后一次 `train_variant` 留下的继承上下文，不是评估协议宣布的通用种子。",
        "规范化的显式评估全局种子没有在历史协议里找到，因此 manifest 里该字段是 null，没有用另一个新种子冒充历史设置。",
        "",
        "## 2. 容差与完整性",
        "",
        f"payload 和记账容差是 {ACCOUNTING_TOLERANCE_BITS:g} bit。路径容差是 {PATH_TOLERANCE_M:g} 米。",
        f"正式 checkpoint SHA-256：`{summary.get('formal_actor_sha256')}`。",
        f"运行后 checkpoint SHA-256：`{summary.get('formal_actor_sha256_after', summary.get('formal_actor_sha256'))}`。",
        f"manifest SHA-256：`{summary.get('manifest_sha256')}`。",
        f"manifest 保持历史继承标签：{summary.get('manifest_explicit')}。",
        f"DEV split 校验：{summary.get('split_ok')}。",
        "",
        "## 3. 两次独立进程",
        "",
    ]
    comparison = summary.get("process_comparison") or {}
    lines.append(f"逐步轨迹在原容差内一致：{comparison.get('consistent')}。")
    lines.append(f"进程间 payload 最大差 {_fmt(comparison.get('max_payload_abs'))} bit，路径差 {_fmt(comparison.get('path_abs'))} 米。")
    lines.append(f"reset 哈希一致：{summary.get('reset_equal')}。")
    for name, match in (summary.get("saved_match") or {}).items():
        lines.append(_gap_line(name, match))
    ledger = summary.get("ledger_ok") or {}
    lines.append(f"账本闭合：run_a={ledger.get('run_a')}，run_b={ledger.get('run_b')}。")
    lines.extend(["", "## 4. 判定", "", _next(str(summary.get("gate"))), ""])
    return "\n".join(lines)
