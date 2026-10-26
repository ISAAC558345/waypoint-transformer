"""Chinese E2 report rendered from the summary. It does not choose lambda_u."""
from __future__ import annotations


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, bool):
        return "是" if value else "否"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 10.0 ** (-int(digits)):
        return f"{number:.3e}"
    if abs(number) >= 100.0:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _lambda_table(scale: dict) -> str:
    header = "| lambda_u | 角色 | reward 均值 | P50 | P95 | P99 | 最大 | bonus/payload 均值 | bonus/penalty P95 | 无惩罚上界步数 | 实质重排集数 | 符号变化 |"
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    body = []
    for item in scale["by_lambda"]:
        penalty = item["bonus_over_penalty"]
        payload = item["bonus_over_payload"]
        role = "高风险参照" if item["role"] == "high_risk_reference" else "候选"
        body.append(
            "| {lam} | {role} | {mean} | {p50} | {p95} | {p99} | {maxv} | {payload} | {penalty} | {unbounded} | {material} | {sign} |".format(
                lam=_fmt(item["lambda_u"], 4),
                role=role,
                mean=_fmt(item["reward_mean"]),
                p50=_fmt(item["reward_p50"]),
                p95=_fmt(item["reward_p95"]),
                p99=_fmt(item["reward_p99"]),
                maxv=_fmt(item["reward_max"]),
                payload=_fmt(payload["mean"]),
                penalty=_fmt(penalty["p95"]),
                unbounded=penalty["unbounded_steps"],
                material=item["episodes_with_material_reorder"],
                sign=item["sign_flips"],
            )
        )
    return "\n".join([header, rule, *body])


def _episode_table(summary: dict) -> str:
    header = "| 场景 | 耗时 s | payload bit | 普通服务 | tag=1 服务 | urgency/served | tag 抽样 | tag=1 | 丢弃 tag=1 | 账本差距 | 安全违规 | 投影步 |"
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    body = []
    fractions = {int(item["scenario_id"]): item["urgency_over_served"] for item in (summary.get("scale") or {}).get("urgency_fraction_by_scenario", [])}
    for episode in summary["episodes"]:
        scenario = int(episode["scenario_id"])
        body.append(
            "| {scenario} | {elapsed} | {payload} | {ordinary} | {tagged} | {fraction} | {draws} | {urgent} | {drop} | {gap} | {safety} | {project} |".format(
                scenario=scenario,
                elapsed=_fmt(episode["elapsed_s"], 3),
                payload=_fmt(episode["payload_bits"], 1),
                ordinary=_fmt(episode["served_ordinary_bits"], 1),
                tagged=_fmt(episode["served_tagged_bits"], 1),
                fraction=_fmt(fractions.get(scenario), 4),
                draws=episode["tag_draws"],
                urgent=episode["tagged_draws"],
                drop=_fmt(episode["dropped_tagged_bits"], 1),
                gap=_fmt(episode["max_ledger_gap_bits"]),
                safety=episode["safety_violations"],
                project=episode["projection_steps"],
            )
        )
    return "\n".join([header, rule, *body])


def _suggestion(summary: dict) -> str:
    scale = summary.get("scale") or {}
    suggested = scale.get("suggested_upper_candidate")
    if summary["gate"] == "URGENCY_REWARD_SCALE_BLOCKED":
        return "完整性或预算没有通过，不能从这次运行推荐权重。"
    if summary["gate"] == "URGENCY_REWARD_SCALE_REPLICATED" and suggested is not None:
        return (
            f"预注册界限内，最大的正候选是 `{_fmt(suggested, 4)}`。这只是证据上的保守上沿，不是选定值。"
            " `0.05` 仍是高风险参照。最终 `lambda_u` 需要负责人确认。"
        )
    smallest = None
    if scale:
        for item in scale["by_lambda"]:
            if float(item["lambda_u"]) == 0.0005:
                smallest = item
    penalty_note = ""
    if smallest is not None:
        penalty_note = (
            f"最小正候选 0.0005 的逐步 bonus/payload 均值是 {_fmt(smallest['bonus_over_payload']['mean'])}，"
            f"bonus/penalty 的 P95 是 {_fmt(smallest['bonus_over_penalty']['p95'])}。"
        )
    reasons = []
    if scale and scale.get("ratio_span_ok"):
        reasons.append(f"八个场景的 urgency/served 跨度是 {_fmt(scale.get('urgency_fraction_span'))}，没有超过预注册上限 2。")
    elif scale:
        reasons.append(f"场景间 urgency/served 的最大最小比是 {_fmt(scale.get('urgency_fraction_span'))}，超过预注册上限 2。")
    if scale and not scale.get("positive_candidate_supported"):
        reasons.append("没有任何正候选在全部场景上同时满足 bonus/penalty 的 P95 不超过 1、相对 payload 均值不超过 1%，并且不造成超过 1 bit 的 payload 重排。")
    if scale and not scale.get("e1_scale_close"):
        reasons.append(
            f"冻结策略的平均 urgency/served 是 {_fmt(scale.get('frozen_urgency_over_served_mean'))}，"
            f"E1 脚本 smoke 是 {_fmt((scale.get('e1_smoke') or {}).get('urgency_over_payload'))}，相对差距 {_fmt(scale.get('e1_relative_gap'))}，超过 0.50。"
            "E1 的 16 步脚本不能代表这条冻结策略轨迹。"
        )
    detail = penalty_note + "".join(reasons) if reasons or penalty_note else "样本还不足以把某个正候选当成保守选择。"
    return detail + "因此不推荐进入训练。`lambda_selected` 保持为空，`0.05` 不是候选。"


def render_report(summary: dict) -> str:
    scale = summary.get("scale")
    scale_text = "尺度表尚未形成。" if not scale else _lambda_table(scale)
    episode_text = "没有完成的 episode。" if not summary["episodes"] else _episode_table(summary)
    integrity = summary.get("integrity") or {}
    return f"""# E2 报告：冻结低层策略上的紧急奖励尺度

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`{summary["instruction_commit"]}`
- 前置 Gate：`PRIORITY_LEDGER_PROTOTYPE_PASS`、`URGENCY_REWARD_SCALE_READY`
- 本阶段 Gate：`{summary["gate"]}`
- 原因：{summary["reason"]}
- `lambda_u` 最终值：未选。`lambda_selected = null`
- 性质：冻结 `phase6g_seed0.pt` 的均值动作，合法 G1_RATE 任务每 15 步抽样一次。没有训练，没有更新权重，没有修改正式 Planner、奖励、动作、SafetyProjector、checkpoint 或旧数据集。

## 1. 直接结论

{_suggestion(summary)}

第一个场景耗时 {_fmt(summary["elapsed_first_s"], 3)} 秒，外推 8 集为 {_fmt(summary["estimated_eight_s"], 3)} 秒。预算是 {_fmt(summary["budget_seconds"], 1)} 秒。实际完成 {summary["episodes_completed"]} 集。

## 2. 场景、checkpoint 与正式文件

场景来自 phase6g `TRAIN_POOL` 的前 8 个，并且都通过 `assert_allowed(split="train")`：{summary["scenarios"]}。

checkpoint SHA-256 是 `{summary["checkpoint_sha256"]}`。与 E1 记录一致：{_fmt(summary["checkpoint_matches_e1"])}。被监视的正式文件与 E1 哈希一致：{_fmt(summary["watched_matches_e1"])}。运行前后 actor 权重摘要一致：{_fmt(summary["actor_digest_unchanged"])}。

动作使用 actor 的均值，不从高斯分布抽样。任务抽样种子是 `6000000 + scenario_id`，因此单个场景可以独立重跑。观测沿用正式 `build_actor_observation`，不加入 tag，也不读取未来到达。

## 3. 完整性

- 账本与计账通过：{_fmt(integrity.get("accounting_ok"))}
- 预注册场景重跑后 tag、动作、服务和 payload 一致：{_fmt(integrity.get("repeat_trace_match"))}
- 关闭账本后的正式指纹差：{_fmt(integrity.get("ledger_off_fingerprint_gap"))}
- 关闭账本后的动作一致：{_fmt(integrity.get("ledger_off_action_match"))}
- 观测字段：{integrity.get("observation_keys", "缺失")}

惩罚按正式 `low_level_reward` 的公式拆成 `0.05 * c_project` 与 `0.01 * c_residual`。八集投影惩罚合计 {_fmt(sum(float(item["sum_projection_penalty"]) for item in summary["episodes"]))}，残差惩罚合计 {_fmt(sum(float(item["sum_residual_penalty"]) for item in summary["episodes"]))}。包装内重算的 base reward 与正式函数的最大差距写在每集记录的 `max_official_reward_gap`。

## 4. 各场景服务与队列

{episode_text}

账本差距不超过 1 bit 才算通过。tag 在每个正到达块上抽一次。丢弃仍只从该步新到达桶扣除。snap 不计入 urgency 服务。

## 5. 权重尺度

下表把 8 条轨迹的逐步 reward 放在一起。bonus 等于 `lambda_u * urgency_served_bits / 1e6`。bonus/penalty 只统计惩罚大于 `1e-12` 的步；惩罚接近 0 但 bonus 为正的步计入“无惩罚上界步数”。实质重排指 payload 项相差超过 1 bit 的步被对调。符号变化比较的是正式 base reward 与加上 bonus 之后的符号。

{scale_text}

预注册保守界限是：每个场景的 bonus/penalty P95 不超过 1，bonus/payload 均值不超过 0.01，并且没有实质重排。`0.05` 不参与这条界限，只作参照。旧 buoy-level urgency 比例没有使用。

冻结策略平均 urgency/served 为 {_fmt(None if not scale else scale.get("frozen_urgency_over_served_mean"))}。E1 的 16 步脚本比值是 {_fmt(None if not scale else (scale.get("e1_smoke") or {}).get("urgency_over_payload"))}。场景间跨度是 {_fmt(None if not scale else scale.get("urgency_fraction_span"))}。

这些离线数字不能推出训练后 payload 不会下降。不同 lambda 的闭环结果仍要单独训练和评估。

## 6. 停止

本阶段不训练 L1/L2，不改正式奖励，不启动完整 benchmark。下一阶段必须等负责人明确 `lambda_u` 之后，才能做独立 observation/reward builder 和短程 smoke。
"""
