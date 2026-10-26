"""Chinese report for the offline label-rule design. Numbers come from summary.json."""
from __future__ import annotations

from .protocol import DELTA_PAYLOAD_MAX, EPS_BITS, FRACTIONS, LAMBDA_H


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return "缺失"
    number = float(value)
    if abs(number) >= 100.0:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _time_block(title: str, block: dict) -> str:
    lines = [
        f"### {title}",
        "",
        (
            f"状态 {block['states']} 个，horizon {block['horizon']}，标签族 `{block['label_family']}`。"
            f"候选数 {block['candidate_count']} / 全池 {block['pool_count_text']}。"
            f"Pareto 前沿大小 {block['pareto_min']}–{block['pareto_max']}（均值 {_fmt(block['pareto_mean'], 2)}）。"
            f"1 bit payload 并列候选占比均值 {_pct(block['payload_tie_fraction_mean'])}。"
            + ("" if block["horizon_source"] == "labels_csv" else " 该 horizon 来自旧 schema，标签表没有这一列。")
        ),
        "",
        (
            f"Rule P 的 payload regret 都是 0（不超过 {EPS_BITS:g} bit）。"
            f"相对“只看 payload、用更小候选 ID 打破平局”，Rule P 的 urgent 增益均值为 {_fmt(block['tiebreak_gain_mean'])} bit，"
            f"超过 1 bit 的状态 {block['material_tiebreak_states']}/{block['states']}。"
            f"Rule P 与抽样子集 urgent-first 相同的状态 {block['matches_urgent_first']}/{block['states']}。"
        ),
        "",
        (
            f"urgent-first 相对该状态最高 payload 的 regret 均值 {_fmt(block['urgent_regret_mean'])} bit，"
            f"范围 {_fmt(block['urgent_regret_min'])}–{_fmt(block['urgent_regret_max'])} bit。"
            f"其中超过 1 bit 的状态 {block['material_regret_states']}/{block['states']}。"
            f"这些 urgent-first 相对 Rule P 的 urgent 增益均值 {_fmt(block['urgent_gain_vs_rule_p_mean'])} bit。"
        ),
        "",
    ]
    if block["exchange_count"]:
        lines.append(
            f"描述性交换比（payload regret / urgent gain，两边都超过 1 bit）有 {block['exchange_count']} 个状态，"
            f"范围 {_fmt(block['exchange_min'], 6)}–{_fmt(block['exchange_max'], 6)}。"
            "这不是选定的 `lambda_H`。"
        )
        lines.append("")
    else:
        lines.append("没有状态同时出现超过 1 bit 的 payload regret 和超过 1 bit 的 urgent 增益，因此不计算交换比。")
        lines.append("")
    if block.get("rows"):
        lines.append(
            _table(
                ["场景", "Pareto", "并列数", "Rule P", "tie-break 增益", "urgent-first", "payload regret", "相对 Rule P 的 urgent 增益"],
                [
                    [
                        row["scenario_id"],
                        row["pareto_count"],
                        row["payload_tie_count"],
                        row["rule_p_candidate_id"],
                        _fmt(row["rule_p_urgent_gain_vs_payload_id_bits"]),
                        row["urgent_first_candidate_id"],
                        _fmt(row["urgent_first_payload_regret_bits"]),
                        _fmt(row["urgent_first_urgent_gain_vs_rule_p_bits"]),
                    ]
                    for row in block["rows"]
                ],
            )
        )
        lines.append("")
    return "\n".join(lines)


def _sensitivity_table(rows: list[dict]) -> str:
    return _table(
        ["span 比例", "状态数", "assignment 改变", "到达 urgent-first", "payload regret 均值", "urgent 增益均值", "regret 范围"],
        [
            [
                f"{float(row['fraction']):.2f}",
                row["states"],
                f"{row['changed_states']}/{row['states']}",
                f"{row['matched_urgent_states']}/{row['states']}",
                _fmt(row["regret_mean"]),
                _fmt(row["gain_mean"]),
                f"{_fmt(row['regret_min'])}–{_fmt(row['regret_max'])}",
            ]
            for row in rows
        ],
    )


def render_report(summary: dict) -> str:
    recommendation = summary["recommendation"]
    decision = recommendation["detail"]
    source_rows = [
        [
            item["source"],
            item["summary_gate"],
            item["commit"][:12],
            item["labels_sha256"][:16],
            item["summary_sha256"][:16],
        ]
        for item in summary["sources"]
    ]
    gap_rows = [[item["source"], item["gaps"] or "无"] for item in summary["field_gap_groups"]]
    fraction_text = ", ".join(f"{fraction:.2f}" for fraction in FRACTIONS)
    text = f"""# Stage D7 报告：高层双目标标签决策规则

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`{summary['instruction_commit']}`
- 前置 Gate：`ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE`，本阶段没有修改
- 本阶段 Gate：`{summary['gate']}`
- 性质：只读 D0/D1/D5/D6 已有标签。`rollouts_run = 0`，`learned_policy = false`，没有训练 HGNN 或 MAPPO

## 1. 直接结论

本阶段没有重跑环境，没有新增候选，也没有改 Planner、奖励、动作、checkpoint 或旧数据集。`lambda_H = {LAMBDA_H}`，`delta_payload_max = {DELTA_PAYLOAD_MAX}`。敏感性网格的每个点都标了 `budget_selected = 0`。

{decision}

t=0 和 t=45 继续用 payload-first。这里的 payload-first 就是 Rule P：先取 1 bit 内的最高 payload，urgent 只在这个并列组里打破平局。不为这两个时刻增加会牺牲 payload 的紧急度奖励。15 步标签和 t=45 的 3 步标签没有合并，不同来源的 bit 值也没有相减。

8 个 D6 场景、每状态 128 个候选只说明这批抽样子集上的规则形状。不宣称全池最优、统计显著或泛化。

## 2. 数据来源

每份来源都记录了路径、最后一次提交和 SHA-256。完整哈希在 `summary.json`。

{_table(["来源", "原 Gate", "提交", "labels SHA 前缀", "summary SHA 前缀"], source_rows)}

D0、D1、D5、D6 都沿 `MAX_RATE_INDEPENDENT` 数据集重放和冻结 MAPPO-V2 seed 0。它们不是 Scheme C 轨迹。D0/D1 的标签在 scratch 副本上关闭了到达。D5/D6 的规则使用真实到达标签，零到达列只作 legacy 对照。

## 3. 分时刻语义

- t=0：D0 是 322000–322003 的零到达 15 步标签。D6 是 322016–322023 的真实到达 15 步标签；D6 报告里两臂数值相同，新到达没有进入这 128 个候选的服务量。本规则分析只用各自文件里的列，不把 D0 的 bit 减去 D6。
- t=15：D0 仍是零到达标签，没有真实到达列。D6 的真实到达 15 步标签是这个时刻的主数据。
- t=30：D0/D1 是零到达标签。D1 覆盖 322004–322015。D5 是 322000–322003 的真实到达 15 步标签，和 D6 的 322016–322023 真实到达标签分开汇总。
- t=45：只有 D6 有这个时刻。horizon 来自标签表，值为 3。正式到达窗口的生成量在 D6 中为 0。它不进入 15 步平均，也不使用旧 J15 这个名称。

## 4. 覆盖与字段缺口

主规则数据是 D6 的 t=15 和 t=30，各 8 个状态、每状态 128 个候选。D6 summary 里的合法池大小是 13,327。抽样比例是 128/13,327。Pareto 数因此是抽样子集的非支配候选数，不是全池前沿。

{_table(["来源", "字段缺口"], gap_rows)}

缺口只说明旧表没有该列。本阶段没有重算标签，也没有填入猜测值。D0/D1 没有 `real_payload_bits`，所以它们不进入 Rule C 网格。

## 5. 主数据上的 Rule P

Rule P 在每个状态内部独立计算。payload 最高值 1 bit 以内视为并列，组内再最大化 urgent-service，仍在 1 bit 内则取更小的候选 ID。

{_time_block("D6 t=15，horizon 15，真实到达", summary["blocks"]["D6:15"])}
{_time_block("D6 t=30，horizon 15，真实到达", summary["blocks"]["D6:30"])}

## 6. 其他时刻和来源

这些块与上一节使用同一套规则，但不是同一个状态样本，均值不能相加或相减。

{_time_block("D6 t=0，horizon 15，真实到达标签", summary["blocks"]["D6:0"])}
{_time_block("D6 t=45，horizon 3，真实到达标签", summary["blocks"]["D6:45"])}
{_time_block("D5 t=30，horizon 15，真实到达标签", summary["blocks"]["D5:30"])}
{_time_block("D0 t=0，零到达标签", summary["blocks"]["D0:0"])}
{_time_block("D0 t=15，零到达标签", summary["blocks"]["D0:15"])}
{_time_block("D0 t=30，零到达标签", summary["blocks"]["D0:30"])}
{_time_block("D1 t=30，零到达标签", summary["blocks"]["D1:30"])}

D0 的三个时刻分开列出。零到达标签上的 Rule P 不能解释 D6 的真实到达权衡，两组 bit 值也不相减。

## 7. Rule C 敏感性

网格是每个状态自己的 payload 极差乘以固定比例 {fraction_text}。比例 0 的预算是 0 bit，再加上“不超过 1 bit 的差距记为 0 损失”，所以它等于 Rule P。比例 1 允许整个抽样子集，因此会选中该子集的 urgent-first，除非 urgent 的 1 bit 并列被更小 ID 打破后仍是同一个候选。中间比例只展示权衡。没有任何比例被写成最终 `delta_payload_max`。

### D6 t=15

{_sensitivity_table(summary["sensitivity"]["D6:15"])}

### D6 t=30

{_sensitivity_table(summary["sensitivity"]["D6:30"])}

t=15 在比例 0.50、t=30 在比例 0.50 时，这 8 个状态的抽样 urgent-first 都已经进入预算。这只说明抽样 payload 极差比前沿上的 regret 更宽，不是选定的预算。

### D6 t=0 与 t=45

t=0 与 t=45 的网格写在同一张 CSV 里，`primary_sensitivity = 0`。它们不参与 t=15/t=30 的均值。t=0 的推荐仍然是 Rule P。t=45 的推荐也是 Rule P，并且保持 3 步 horizon。

#### t=0

{_sensitivity_table(summary["sensitivity"]["D6:0"])}

#### t=45

{_sensitivity_table(summary["sensitivity"]["D6:45"])}

### D5 t=30

D5 只有 4 个更早的训练场景，状态路径与 D6 不同。下表单独列出，不并进 D6。

{_sensitivity_table(summary["sensitivity"]["D5:30"])}

## 8. 三种监督形式

1. 只按真实 payload 排序，urgent 仅在 1 bit 并列时打破平局。这就是 Rule P。它不需要 `lambda_H` 或 `delta_payload_max`。D6 的 t=15/t=30 显示，抽样 urgent-first 经常在并列组之外，所以这种形式留住了 payload，也留下了 D6 已经测到的 urgent 差距。
2. 两个 head 分别预测真实 payload 和 urgent-service，推理时再套 Rule P 或 Rule C。预测和可接受损失因此可以分开，现在不必猜 `lambda_H`。oracle 标签上的 Pareto 前沿不能证明 HGNN 学得会这两个量，两个 head 以后都要单独看预测误差。
3. 单一效用 `payload + lambda_H * urgent`。上面的描述性交换比是各状态 urgent-first 相对 Rule P 的 bit 比值；状态之间并不相同，所以现有标签给不出一个全局 `lambda_H`。

在 `delta_payload_max` 未定之前，形式 2 是以后可以审计的目标形状，但本阶段不授权收集正式全量数据或开始训练。

## 9. Gate

`{summary['gate']}`。

- t=0/15/30/45 的标签语义已经分开。
- D6 t=15 与 t=30 的 Rule P、Rule C 网格完整，状态数分别为 {summary['primary_state_counts'].get('15', 0)} 和 {summary['primary_state_counts'].get('30', 0)}。
- `delta_payload_max` 保持为空。`lambda_H` 保持为空。
- 是否构建正式数据集：否。是否开始双头试验：{"是，仅限独立小规模试验" if recommendation["worth_dual_head_trial"] else "否"}。
- 还缺的验证：研究负责人给出 payload 损失预算；在该预算下重读这批标签，确认规则仍只使用当前状态的真实到达标签；用更大的候选覆盖检查 128/13,327 子集是否改变前沿形状；双头模型必须分别报告两个标签的预测误差；闭环 payload 与紧急服务尚未测量，不能把离线标签写成系统改进。

## 10. 停止

本阶段停在这里。不训练 HGNN，不重跑 rollout，不修改正式系统，也不进入 D8。
"""
    return text
