"""Chinese experiment report. Figures are copied from the saved summary."""
from __future__ import annotations


def _fmt(value: object, digits: int = 6) -> str:
    if value is None or value == "":
        return "缺失"
    if isinstance(value, bool):
        return "是" if value else "否"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) >= 1000.0:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _seed_lines(rows: list[dict]) -> str:
    if not rows:
        return "尚无评估。"
    lines = []
    for row in rows:
        lines.append(
            f"- {row['variant']} seed {row['model_seed']}：通过={_fmt(row['passes_preregistered_criteria'])}，"
            f"紧急度差均值 {_fmt(row['urgency']['mean'])}，95% 区间 [{_fmt(row['urgency']['ci_low'])}, {_fmt(row['urgency']['ci_high'])}]，"
            f"tagged 差均值 {_fmt(row['tagged']['mean'])}，95% 区间 [{_fmt(row['tagged']['ci_low'])}, {_fmt(row['tagged']['ci_high'])}]，"
            f"payload 差均值 {_fmt(row['payload']['mean'])}，95% 区间 [{_fmt(row['payload']['ci_low'])}, {_fmt(row['payload']['ci_high'])}]，"
            f"payload 变差场景 {row['payload_negative_scenarios']}，安全差 {_fmt(row['safety_diff_sum'])}，"
            f"约束失败差 {_fmt(row['constraint_diff_sum'])}，丢弃差均值 {_fmt(row['dropped_diff_mean'])}，"
            f"SINR 差均值 {_fmt(row['sinr_diff_mean'])}，干扰差均值 {_fmt(row['interference_diff_mean'])}。"
        )
    return "\n".join(lines)


def render_report(summary: dict) -> str:
    decision = summary.get("decision") or {}
    smoke = summary.get("smoke") or {}
    problems = summary.get("problems") or smoke.get("problems") or []
    problem_text = "无。" if not problems else "；".join(str(item) for item in problems)
    hashes = summary.get("watched_sha256") or smoke.get("watched_sha256") or {}
    hash_lines = "\n".join(f"- `{name}`：`{digest}`" for name, digest in hashes.items()) or "- 尚未记录"
    return f"""# 实验报告：合成 priority-tag 的有界紧急奖励

- 分支：`experiment/priority-tag-urgency-reward`
- 基线分支：`feature/maritime-wt-stage1a`
- Gate：`{summary.get("gate")}`
- 是否合并：{_fmt(summary.get("merged"))}
- 标签性质：synthetic priority tag，不是真实海事业务标签。

## 1. 预注册设置

本报告不改验收标准。L1 只增加观测，奖励仍是正式 `r_base`。L2 使用同一观测，并加上预注册候选：

`r = r_base + min(0.0005 * urgency_served_bits/1e6, 0.001 * served_bits/1e6)`

`r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual`。`u = clip(clip(age/48,0,1)+0.25*synthetic_tag,0,1)`，在服务前读取。服务量为 0 时 bonus 为 0。

训练种子是 0、1、2。优化器、rollout、更新次数沿用正式 MAPPO：300 次更新，4 个环境，每次 48 步。初始化是 legacy checkpoint 的显式移植，新输入列权重为 0，不是静默部分加载。L0 使用冻结的 `phase6g_seed0.pt`，不重新训练。

合并需要至少 2 个 L2 种子同时满足：紧急度加权服务和 tagged 服务的配对 95% 下界都大于 0；payload 配对均值不小于 0 且下界不小于 0；安全违规和约束失败不增加；丢弃、SINR、干扰的配对均值不变差；紧急度增益和 payload 结论不被单个场景带动。

## 2. Smoke

Smoke Gate：`{smoke.get("gate", "未运行")}`。首个场景用时 {_fmt(smoke.get("first_scenario_seconds"))} 秒，总用时 {_fmt(smoke.get("elapsed_seconds"))} 秒。问题：{problem_text}

## 3. 正式文件哈希

{hash_lines}

正式 L0 checkpoint：`{summary.get("formal_actor_sha256", smoke.get("formal_actor_sha256", "缺失"))}`。

## 4. 训练

{summary.get("training_text", "训练尚未运行。")}

## 5. 配对评估

评估使用登记的 development pool，确定性动作，同一个场景重置种子。没有读取 reserved 或 test split。

### L2

{_seed_lines(decision.get("l2_seeds", []))}

### L1

{_seed_lines(decision.get("l1_seeds", []))}

L2 通过种子数：{decision.get("l2_passing_seeds", "缺失")}。L1 通过种子数：{decision.get("l1_passing_seeds", "缺失")}。

## 6. 合并决定

{summary.get("decision_text", "评估尚未完成，不合并。")}

固定轨迹上的奖励上升不能代替这些标准。正式分支、原始 checkpoint 和 E0–E4 结果保持不变。
"""
