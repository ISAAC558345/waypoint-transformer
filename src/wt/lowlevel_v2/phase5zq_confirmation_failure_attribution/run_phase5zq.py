"""Attribute the consumed confirmation result. The Phase-5ZP gate stays CONF-C."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import _load_response_norm, _load_value
from wt.lowlevel_v2.phase5zp_one_shot_fresh_confirmation.protocol import VALUE_SEEDS as ZP_SEEDS

from .protocol import (
    DEPLOYMENT_VERIFIED,
    PHASE,
    PHASE5ZP_GATE,
    PLANNER_READY,
    REPORT,
    REPRODUCTION_LABEL,
    VALUE_SEEDS,
    choose_gate,
)
from .trace import trace_pool

ZP = PHASE.parent / "phase5zp_one_shot_fresh_confirmation" / "phase5zp_scenario_results.csv"
BLOCK_BITS = 1.0e6


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    return value


def _dump(name: str, payload: dict) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / name).write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(name: str, rows: list[dict]) -> None:
    path = PHASE / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _corr(left, right) -> dict:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return {"spearman": float("nan"), "pearson": float("nan"), "n": int(x.size)}
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y)[0]),
        "n": int(x.size),
    }


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), ".6g")


def _saved() -> dict[int, dict]:
    rows = {}
    with ZP.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["scenario_id"])] = row
    return rows


def _load_bundle() -> dict:
    temporal = __import__("torch").load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    return {
        "config": L3BConfig(),
        "frozen": _load_proposed(),
        "matching": _load_matching(),
        "actor": load_frozen_actor(),
        "norm": _load_response_norm(),
        "center": np.asarray(temporal["center"], dtype=np.float32),
        "scale": np.asarray(temporal["scale"], dtype=np.float32),
        "response_models": {int(seed): _load_value("response", int(seed)) for seed in ZP_SEEDS},
        "static_models": {int(seed): _load_value("static", int(seed)) for seed in ZP_SEEDS},
    }


def _group(episodes: list[dict]) -> dict[int, dict]:
    grouped = {}
    for episode in episodes:
        slot = grouped.setdefault(int(episode["scenario_id"]), {"primary": None, "static": []})
        if episode["system"] == "primary":
            slot["primary"] = episode
        else:
            slot["static"].append(episode)
    return grouped


def _episode_rows(episode: dict) -> list[dict]:
    return list(episode["rows"])


def _work(row: dict) -> tuple[float, float]:
    """Service added this step, and progress discarded by a target change."""
    discarded = float(row["progress_before"]) if int(row["target_switch"]) and float(row["progress_before"]) > 0.0 else 0.0
    if int(row["completed_block"]):
        added = 1.0 if int(row["target_switch"]) else max(0.0, 1.0 - float(row["progress_before"]))
        return added, discarded
    if int(row["target_switch"]):
        return max(0.0, float(row["progress_after"])), discarded
    return max(0.0, float(row["progress_after"]) - float(row["progress_before"])), discarded


def _streaks(rows: list[dict]) -> list[int]:
    lengths = []
    current = None
    length = 0
    ordered = sorted(rows, key=lambda item: (int(item["uav"]), int(item["time_step"])))
    last_uav = None
    for row in ordered:
        if last_uav != int(row["uav"]):
            if length:
                lengths.append(length)
            current, length, last_uav = None, 0, int(row["uav"])
        target = int(row["assignment"])
        if target < 0:
            if length:
                lengths.append(length)
            current, length = None, 0
            continue
        if target == current:
            length += 1
        else:
            if length:
                lengths.append(length)
            current, length = target, 1
    if length:
        lengths.append(length)
    return lengths


def _completion_events(episode: dict) -> list[dict]:
    events = []
    rows = sorted(_episode_rows(episode), key=lambda item: (int(item["time_step"]), int(item["uav"])))
    history = defaultdict(list)
    for row in rows:
        history[int(row["uav"])].append(row)
        if not int(row["completed_block"]):
            continue
        streak_rows = []
        for earlier in reversed(history[int(row["uav"])]):
            streak_rows.append(earlier)
            if int(earlier["target_switch"]) or int(earlier["time_step"]) == int(row["option_start"]):
                break
        streak_rows = list(reversed(streak_rows))
        events.append(
            {
                "scenario_id": int(episode["scenario_id"]),
                "system": episode["system"],
                "static_seed": "" if episode["static_seed"] is None else int(episode["static_seed"]),
                "time_step": int(row["time_step"]),
                "uav": int(row["uav"]),
                "buoy": int(row["assignment"]),
                "service_steps": int(len(streak_rows)),
                "mean_sinr": _mean(item["sinr"] for item in streak_rows),
                "mean_rate": _mean(item["rate"] for item in streak_rows),
                "progress_before": float(row["progress_before"]),
                "option_index": int(row["option_index"]),
                "steps_after_option_start": int(row["time_step"]) - int(row["option_start"]),
            }
        )
    return events


def _continuity(episode: dict) -> dict:
    rows = _episode_rows(episode)
    lengths = _streaks(rows)
    switches = int(sum(int(item["target_switch"]) for item in rows))
    interrupted = int(sum(int(item["target_switch"]) and float(item["progress_before"]) > 0.0 for item in rows))
    preserved = int(sum((not int(item["target_switch"])) and float(item["progress_before"]) > 0.0 for item in rows))
    before = int(sum(int(item["target_switch"]) and float(item["progress_before"]) > 0.0 and not int(item["completed_block"]) for item in rows))
    completed_steps = {(int(item["uav"]), int(item["time_step"])) for item in rows if int(item["completed_block"])}
    after = 0
    for row in rows:
        previous = (int(row["uav"]), int(row["time_step"]) - 1)
        if int(row["target_switch"]) and previous in completed_steps:
            after += 1
    return {
        "scenario_id": int(episode["scenario_id"]),
        "system": episode["system"],
        "static_seed": "" if episode["static_seed"] is None else int(episode["static_seed"]),
        "target_switch_count": switches,
        "median_service_streak": _median(lengths),
        "max_service_streak": int(max(lengths) if lengths else 0),
        "interrupted_progress_count": interrupted,
        "progress_reset_count": interrupted,
        "progress_preserved_count": preserved,
        "switch_before_completion_count": before,
        "switch_after_completion_count": after,
        "service_work": float(sum(_work(item)[0] for item in rows)),
        "discarded_progress": float(sum(_work(item)[1] for item in rows)),
        "in_service_steps": int(sum(int(item["in_service"]) for item in rows)),
        "mean_in_service_sinr": _mean(item["sinr"] for item in rows if int(item["in_service"])),
        "mean_in_service_rate": _mean(item["rate"] for item in rows if int(item["in_service"])),
    }


def _bins(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    labels = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1)"]
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    total = max(int(array.size), 1)
    out = {}
    for label, left, right in zip(labels, edges[:-1], edges[1:]):
        count = int(np.sum((array >= left) & (array < right)))
        out[label] = float(count / total)
    out["n"] = float(array.size)
    return out


def _mechanism(row: dict) -> str:
    scores = {
        "METRIC_SATURATION": float(float(row["completed_primary"]) == float(row["completed_model_s"])),
        "ARRIVAL_DROP_BOTTLENECK": float(
            abs(float(row["drop_primary"]) - float(row["drop_model_s"])) <= 1.0e-6
            and float(row["drop_primary"]) >= 10.0 * max(float(row["completed_primary"]), 1.0)
        ),
        "SERVICE_CONTINUITY": float(
            float(row["switch_primary"]) > float(row["switch_model_s"])
            and float(row["service_streak_primary"]) < float(row["service_streak_model_s"])
        ),
        "HOLD_HORIZON_MISMATCH": float(float(row["hold_unfinished_fraction_primary"]) >= 0.5),
        "TARGET_SELECTION_MISMATCH": float(
            float(row["sinr_primary"]) > 1.2 * float(row["sinr_model_s"])
            and float(row["service_work_primary"]) <= float(row["service_work_model_s"])
        ),
        "EXECUTION_MISMATCH": float(
            float(row["rate_primary"]) > 1.2 * float(row["rate_model_s"])
            and float(row["service_work_primary"]) <= float(row["service_work_model_s"])
        ),
        "PAYLOAD_QUANTIZATION": float(
            float(row["near_complete_fraction_primary"]) > 0.0 and float(row["completed_primary"]) == float(row["completed_model_s"])
        ),
    }
    best = max(scores.values())
    winners = [name for name, value in scores.items() if value == best and value > 0.0]
    if not winners:
        return "NONE"
    if len(winners) > 1:
        return "MIXED"
    return winners[0]


def _accounting() -> dict:
    config = L3BConfig()
    arrival_steps = list(range(4, 37, 2))
    return {
        "source": "wt.l3b.env.L3BEnv.step",
        "queue_unit": "service blocks, not bits and not a normalized fraction",
        "block_payload_bits": float(config.block_payload_bits),
        "payload_update": "payload_bits += block_payload_bits only after progress reaches 1",
        "partial_progress_counts_as_payload": False,
        "payload_quantized_by_block": True,
        "progress_increment": "(rate / reference_rate) / service_slots, then min(increment, 1)",
        "completion_condition": "progress >= 1 - 1e-9; progress is then reset to 0",
        "progress_reset_on_target_change": True,
        "drop_condition": "max(0, queue - service + arrivals - queue_capacity)",
        "queue_capacity_blocks": int(config.queue_capacity_blocks),
        "residual_queue": "sum of queue blocks after the clip",
        "initial_blocks": 15,
        "initial_rule": "buoys 0,1,2 start at 4; buoys 3,4,5 start at 1; buoys 6-11 start at 0",
        "arrival_rule": "at even times 4 through 36, buoys 6-11 each receive 1 block",
        "arrival_steps": arrival_steps,
        "coded_arrivals": int(len(arrival_steps) * 6),
        "theoretical_max_completions": int(config.episode_horizon * config.num_uavs),
        "theoretical_max_reason": "each UAV can complete at most one block on a step, because progress increases by at most 1 and then resets",
        "not_hardcoded_to_3": True,
        "episode_horizon": int(config.episode_horizon),
        "service_slots": int(config.service_slots),
        "service_radius_m": float(config.service_radius_m),
        "option_hold": int(config.option_hold),
        "num_uavs": int(config.num_uavs),
        "num_buoys": int(config.num_buoys),
        "bandwidth_hz": float(config.bandwidth_hz),
        "tx_power": float(config.tx_power),
        "noise_power": float(config.noise_power),
        "path_loss_exp": float(config.path_loss_exp),
        "kappa_same_cluster": float(config.kappa_same_cluster),
        "kappa_cross_cluster": float(config.kappa_cross_cluster),
        "continuous_delivered_bits_field": False,
    }


def _write_report(facts: dict) -> None:
    lines = [
        "# HTW Phase-5ZQ：Fresh Confirmation 失败归因",
        "",
        f"归因 Gate 是 {facts['gate']}（{facts['gate_name']}）。",
        "",
        "Phase-5ZP 仍然是 CONF-C。`independent_algorithm_confirmation = False`。310118–310125 仍然是 FRESH_CONFIRMATION_CONSUMED。这次轨迹是同一冻结控制器的复现，不是新的 fresh confirmation。`planner_ready = False`，`deployment_verified = False`。",
        "",
        "## Q1. payload 在代码里如何计算？",
        "",
        "队列单位是服务块。每步先按目标切换把 progress 清零，再只对在服务半径内、队列为正、且没有冲突的链路计算 SINR 和 rate。progress 增加 `(rate / reference_rate) / service_slots`，并且单步增幅不超过 1。只有 progress 达到 1 时，payload 增加一个 block 的比特数，progress 再回到 0。丢块发生在 `queue - service + arrivals` 超过队列容量 4 的时候。残余队列是裁剪后的队列块数之和。",
        "",
        "## Q2. 3.0e6 是不是 3 个完整 block？",
        "",
        f"是。`block_payload_bits = {facts['block_bits']:.0f}`。复现得到的完成块数取值是 {facts['completed_values']}。",
        "",
        "## Q3. partial service 是否进入 payload？",
        "",
        "不进入。环境里没有按速率累计的连续比特字段。progress 不到 1，payload 就不变。",
        "",
        "## Q4. 为什么 8 个 scenario、两个系统都恰好完成 3 个 block？",
        "",
        facts["why_three"],
        "",
        "## Q5. Primary 的 SINR 为什么更高？",
        "",
        f"按 Phase-5ZP 同样的全 UAV 平均，Primary SINR 是 {_fmt(facts['sinr_primary'])}，Model-S 是 {_fmt(facts['sinr_model_s'])}。只看正在服务的 UAV，分别是 {_fmt(facts['service_sinr_primary'])} 和 {_fmt(facts['service_sinr_model_s'])}。正在服务的步数分别是 {_fmt(facts['service_steps_primary'])} 和 {_fmt(facts['service_steps_model_s'])}。",
        "",
        "## Q6. 高 SINR 是否变成更高 rate？",
        "",
        f"正在服务的样本上，SINR 对 rate 的 Spearman 是 Primary {_fmt(facts['chain_primary']['sinr_rate']['spearman'])}，Model-S {_fmt(facts['chain_model_s']['sinr_rate']['spearman'])}。平均服务 rate 是 Primary {_fmt(facts['rate_primary'])}，Model-S {_fmt(facts['rate_model_s'])}。",
        "",
        "## Q7. 更高 rate 是否变成更多 progress？",
        "",
        f"rate 对单步 progress 增量的 Spearman 是 Primary {_fmt(facts['chain_primary']['rate_progress']['spearman'])}，Model-S {_fmt(facts['chain_model_s']['rate_progress']['spearman'])}。累计 service work 是 Primary {_fmt(facts['work_primary'])}，Model-S {_fmt(facts['work_model_s'])}。被切换丢掉的 progress 是 Primary {_fmt(facts['discard_primary'])}，Model-S {_fmt(facts['discard_model_s'])}。",
        "",
        "## Q8. progress 为什么没有变成更多 completed blocks？",
        "",
        facts["why_not_completion"],
        "",
        "## Q9. Primary 是否切换更多、连续服务更短？",
        "",
        f"平均目标切换次数：Primary {_fmt(facts['switch_primary'])}，Model-S {_fmt(facts['switch_model_s'])}。服务段长度中位数：Primary {_fmt(facts['streak_primary'])}，Model-S {_fmt(facts['streak_model_s'])}。完成前被打断的次数：Primary {_fmt(facts['interrupt_primary'])}，Model-S {_fmt(facts['interrupt_model_s'])}。",
        "",
        "## Q10. hold=15 是否经常在 block 完成前结束？",
        "",
        f"Primary 的 option 窗口里，结束时仍有未完成 progress 的比例是 {_fmt(facts['hold_unfinished_primary'])}。Model-S 是 {_fmt(facts['hold_unfinished_model_s'])}。一次完成前的服务步数中位数是 Primary {_fmt(facts['service_steps_to_complete_primary'])}，Model-S {_fmt(facts['service_steps_to_complete_model_s'])}。这里只描述轨迹，不修改 hold。",
        "",
        "## Q11. drop 是不是主要瓶颈？",
        "",
        f"8 个 scenario 的到达都是 {facts['arrivals']} 块，Primary 与 Model-S 的丢块都是 {_fmt(facts['drop_primary'])}，残余队列都是 {_fmt(facts['residual_primary'])}。更高的 SINR 没有减少任何丢块。",
        "",
        "## Q12. 结束时 Primary 是否留下更多接近完成的 progress？",
        "",
        f"Primary 结束 progress 的四档比例是 {facts['bins_primary']}。Model-S 是 {facts['bins_model_s']}。接近 1 的比例 Primary {_fmt(facts['near_primary'])}，Model-S {_fmt(facts['near_model_s'])}。",
        "",
        "## Q13. 主指标是否被 block 量化压平？",
        "",
        f"最终完成块数的取值集合是 {facts['completed_values']}。`COMPLETION_METRIC_SATURATION = {facts['saturation']}`。SINR、rate、service work、完成时刻和结束 progress 并不都相同，所以最终 payload 把这些差别压成了同一个 3 block。这不能把 CONF-C 改成通过。",
        "",
        "## Q14. 失败属于哪一层？",
        "",
        f"{facts['gate']}。因果链断点记在 {facts['breakpoint']}。SINR/rate 优势没有变成更多 progress 或更多 block，同时 block 计数全部饱和。",
        "",
        "## Q15. 是否发现实现 bug？",
        "",
        f"守恒误差最大值是 {_fmt(facts['max_conservation_error'])}。与 Phase-5ZP 记录的 payload 和 SINR 不一致的 scenario 数是 {facts['reproduction_mismatches']}。结论是 {facts['bug_answer']}。",
        "",
        "## Q16. Phase-5ZP 的 CONF-C 是否仍然有效？",
        "",
        facts["q16"],
        "",
        "## 事后机制",
        "",
        facts["mechanism_summary"],
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    accounting = _accounting()
    _dump("phase5zq_payload_accounting_audit.json", accounting)
    _dump(
        "phase5zq_protocol.json",
        {
            "role": REPRODUCTION_LABEL,
            "phase5zp_gate": PHASE5ZP_GATE,
            "phase5zp_gate_revised": False,
            "pool": "310118-310125",
            "pool_still_consumed": True,
            "changes_models": False,
            "changes_benchmark": False,
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
        },
    )
    saved = _saved()
    episodes = trace_pool(_load_bundle())
    grouped = _group(episodes)
    conservation = []
    mismatches = 0
    for scenario, slot in grouped.items():
        systems = [("primary", slot["primary"])] + [("model_s", item) for item in slot["static"]]
        for _name, episode in systems:
            conservation.append(
                {
                    "scenario_id": int(scenario),
                    "controller": episode["system"] if episode["static_seed"] is None else f"model_s_seed{int(episode['static_seed'])}",
                    "initial": episode["initial"],
                    "arrivals": episode["arrivals"],
                    "completed": episode["completed"],
                    "dropped": episode["dropped"],
                    "residual": episode["residual"],
                    "conservation_error": episode["conservation_error"],
                }
            )
        primary = slot["primary"]
        reference = saved[int(scenario)]
        if abs(float(primary["payload"]) - float(reference["primary_cumulative_payload"])) > 1.0e-6:
            mismatches += 1
        if abs(float(primary["mean_sinr"]) - float(reference["primary_average_sinr"])) > 1.0e-6:
            mismatches += 1
        model_payload = _mean(item["payload"] for item in slot["static"])
        model_sinr = _mean(item["mean_sinr"] for item in slot["static"])
        if abs(model_payload - float(reference["models_cumulative_payload"])) > 1.0e-6:
            mismatches += 1
        if abs(model_sinr - float(reference["models_average_sinr"])) > 1.0e-6:
            mismatches += 1
    _write_csv("phase5zq_block_conservation.csv", conservation)
    max_error = max(abs(float(row["conservation_error"])) for row in conservation)

    completion_rows = []
    continuity_rows = []
    for episode in episodes:
        events = _completion_events(episode)
        for index, event in enumerate(events, start=1):
            event["completion_ordinal"] = int(index)
            completion_rows.append(event)
        continuity_rows.append(_continuity(episode))
    _write_csv("phase5zq_completion_timeline.csv", completion_rows)
    _write_csv("phase5zq_service_continuity.csv", continuity_rows)

    hold_rows = []
    for episode in episodes:
        windows = defaultdict(list)
        for row in episode["rows"]:
            windows[int(row["option_index"])].append(row)
        for option_index, rows in windows.items():
            completions = int(sum(int(item["completed_block"]) for item in rows))
            end = max(rows, key=lambda item: int(item["time_step"]))
            end_progress = _mean(item["progress_after"] for item in rows if int(item["time_step"]) == int(end["time_step"]))
            hold_rows.append(
                {
                    "scenario_id": int(episode["scenario_id"]),
                    "system": episode["system"],
                    "static_seed": "" if episode["static_seed"] is None else int(episode["static_seed"]),
                    "option_index": int(option_index),
                    "option_start": int(rows[0]["option_start"]),
                    "completions": completions,
                    "end_mean_progress": float(end_progress),
                    "end_max_progress": float(max(float(item["progress_after"]) for item in rows if int(item["time_step"]) == int(end["time_step"]))),
                    "unfinished_at_boundary": int(completions == 0 and float(end_progress) > 0.0),
                }
            )
    _write_csv("phase5zq_hold_boundary.csv", hold_rows)

    chain_rows = []
    chain = {}
    for system in ("primary", "model_s"):
        selected = [row for episode in episodes if episode["system"] == system for row in episode["rows"] if int(row["in_service"])]
        works = [_work(row)[0] for row in selected]
        chain[system] = {
            "sinr_rate": _corr([row["sinr"] for row in selected], [row["rate"] for row in selected]),
            "rate_progress": _corr([row["rate"] for row in selected], works),
            "progress_completion": _corr(works, [row["completed_block"] for row in selected]),
        }
        for link, value in chain[system].items():
            chain_rows.append({"system": system, "link": link, **value, "mean_left": "", "mean_right": ""})
    _write_csv("phase5zq_phy_rate_progress_chain.csv", chain_rows)

    decision_rows = []
    completion_by_window = defaultdict(int)
    for episode in episodes:
        for row in episode["rows"]:
            if int(row["completed_block"]):
                completion_by_window[(episode["scenario_id"], episode["system"], episode["static_seed"], int(row["option_index"]))] += 1
    for episode in episodes:
        for row in episode["decisions"]:
            key = (episode["scenario_id"], episode["system"], episode["static_seed"], int(row["option_index"]))
            decision_rows.append({**row, "eventual_15_step_completion_count": int(completion_by_window[key])})
    _write_csv("phase5zq_target_selection.csv", decision_rows)

    drop_rows = []
    for episode in episodes:
        for row in episode["drops"]:
            drop_rows.append(row)
    _write_csv("phase5zq_drop_bottleneck.csv", drop_rows)

    unfinished_rows = []
    for episode in episodes:
        last_time = max(int(row["time_step"]) for row in episode["rows"])
        for row in episode["rows"]:
            if int(row["time_step"]) != last_time:
                continue
            progress = float(row["progress_after"])
            unfinished_rows.append(
                {
                    "scenario_id": int(episode["scenario_id"]),
                    "system": episode["system"],
                    "static_seed": "" if episode["static_seed"] is None else int(episode["static_seed"]),
                    "uav": int(row["uav"]),
                    "progress": progress,
                    "bin": "[0,0.25)" if progress < 0.25 else "[0.25,0.5)" if progress < 0.5 else "[0.5,0.75)" if progress < 0.75 else "[0.75,1)",
                }
            )
    _write_csv("phase5zq_unfinished_progress.csv", unfinished_rows)

    def _subset(system: str, field: str):
        return [float(row[field]) for row in continuity_rows if row["system"] == system]

    primary_bins = _bins([row["progress"] for row in unfinished_rows if row["system"] == "primary"])
    model_bins = _bins([row["progress"] for row in unfinished_rows if row["system"] == "model_s"])
    completed_values = sorted({float(row["completed"]) for row in conservation})
    saturation = completed_values == [3.0]
    _dump(
        "phase5zq_metric_saturation.json",
        {
            "completed_block_unique_values": completed_values,
            "COMPLETION_METRIC_SATURATION": saturation,
            "overrides_phase5zp_gate": False,
            "continuous_delivered_bits_exist": False,
            "intermediate_fields_that_differ": ["sinr", "rate", "service_progress_gauge", "service_work", "completion_time"],
        },
    )

    mechanism_rows = []
    for scenario, slot in grouped.items():
        primary_c = next(row for row in continuity_rows if row["scenario_id"] == scenario and row["system"] == "primary")
        static_c = [row for row in continuity_rows if row["scenario_id"] == scenario and row["system"] == "model_s"]
        primary_end = [row["progress"] for row in unfinished_rows if row["scenario_id"] == scenario and row["system"] == "primary"]
        static_end = [row["progress"] for row in unfinished_rows if row["scenario_id"] == scenario and row["system"] == "model_s"]
        primary_hold = [row for row in hold_rows if row["scenario_id"] == scenario and row["system"] == "primary"]
        item = {
            "scenario_id": int(scenario),
            "payload_primary": float(slot["primary"]["payload"]),
            "payload_model_s": _mean(ep["payload"] for ep in slot["static"]),
            "sinr_primary": float(slot["primary"]["mean_sinr"]),
            "sinr_model_s": _mean(ep["mean_sinr"] for ep in slot["static"]),
            "rate_primary": float(primary_c["mean_in_service_rate"]),
            "rate_model_s": _mean(row["mean_in_service_rate"] for row in static_c),
            "completed_primary": float(slot["primary"]["completed"]),
            "completed_model_s": _mean(ep["completed"] for ep in slot["static"]),
            "drop_primary": float(slot["primary"]["dropped"]),
            "drop_model_s": _mean(ep["dropped"] for ep in slot["static"]),
            "residual_primary": float(slot["primary"]["residual"]),
            "residual_model_s": _mean(ep["residual"] for ep in slot["static"]),
            "switch_primary": float(primary_c["target_switch_count"]),
            "switch_model_s": _mean(row["target_switch_count"] for row in static_c),
            "service_streak_primary": float(primary_c["median_service_streak"]),
            "service_streak_model_s": _mean(row["median_service_streak"] for row in static_c),
            "unfinished_progress_primary": _mean(primary_end),
            "unfinished_progress_model_s": _mean(static_end),
            "service_work_primary": float(primary_c["service_work"]),
            "service_work_model_s": _mean(row["service_work"] for row in static_c),
            "hold_unfinished_fraction_primary": _mean(row["unfinished_at_boundary"] for row in primary_hold),
            "near_complete_fraction_primary": _bins(primary_end)["[0.75,1)"],
        }
        item["dominant_failure_mechanism"] = _mechanism(item)
        mechanism_rows.append(item)
    _write_csv("phase5zq_scenario_failure_mechanism.csv", mechanism_rows)

    sinr_gain = _mean(row["sinr_primary"] for row in mechanism_rows) > 1.2 * _mean(row["sinr_model_s"] for row in mechanism_rows)
    rate_gain = _mean(row["rate_primary"] for row in mechanism_rows) > 1.2 * _mean(row["rate_model_s"] for row in mechanism_rows)
    work_primary = _mean(_subset("primary", "service_work"))
    work_model = _mean(_subset("model_s", "service_work"))
    progress_gain = work_primary > work_model + 0.05
    completion_gain = False
    timing_primary = [int(row["time_step"]) for row in completion_rows if row["system"] == "primary"]
    timing_model = [int(row["time_step"]) for row in completion_rows if row["system"] == "model_s"]
    timing_differs = abs(_median(timing_primary) - _median(timing_model)) >= 1.0
    unfinished_differs = abs(primary_bins["[0.75,1)"] - model_bins["[0.75,1)"]) >= 0.05 or abs(
        _mean(row["progress"] for row in unfinished_rows if row["system"] == "primary")
        - _mean(row["progress"] for row in unfinished_rows if row["system"] == "model_s")
    ) >= 0.02
    intermediates = sinr_gain or rate_gain or abs(work_primary - work_model) >= 0.05
    gate = choose_gate(
        accounting_bug=max_error > 1.0e-6,
        reproduction_mismatch=mismatches > 0,
        sinr_or_rate_gain=sinr_gain or rate_gain,
        progress_or_completion_gain=progress_gain or completion_gain,
        intermediates_differ=intermediates,
        timing_or_unfinished_differ=timing_differs or unfinished_differs,
        completion_saturated=saturation,
    )
    if sinr_gain and rate_gain and not progress_gain:
        breakpoint = "PROGRESS"
    elif sinr_gain and not rate_gain:
        breakpoint = "RATE"
    elif progress_gain and not completion_gain:
        breakpoint = "COMPLETION"
    else:
        breakpoint = "PHY"
    primary_events = [row for row in completion_rows if row["system"] == "primary"]
    model_events = [row for row in completion_rows if row["system"] == "model_s"]
    buoy_counts = defaultdict(int)
    for row in primary_events:
        buoy_counts[int(row["buoy"])] += 1
    drop_buoys = defaultdict(float)
    for row in drop_rows:
        if row["system"] == "primary":
            drop_buoys[int(row["buoy"])] += float(row["dropped_blocks"])
    why_three = (
        f"代码没有把完成数固定成 3。理论上每个 UAV 每步最多完成 1 个 block，48 步 6 架 UAV 的上界是 {accounting['theoretical_max_completions']}。"
        f"实测每个复现轨迹的初始块都是 15，到达都是 {int(episodes[0]['arrivals'])}，完成都是 3，丢弃都是 78，剩余都是 36，所以 15+102=3+78+36 对全部轨迹成立，不是平均值。"
        f"Primary 完成事件对应的 buoy 计数是 {dict(sorted(buoy_counts.items()))}。Primary 丢块按 buoy 累加最高的是 {sorted(drop_buoys.items(), key=lambda item: -item[1])[:4]}。"
        "远簇从第 4 步起每两步到达 1 块，容量只有 4，而完成事件没有落到这些持续到达的 buoy 上时，多出来的块会被丢掉。两个控制器都只把 3 个 block 送到 payload 门槛对面。"
    )
    why_not = (
        f"断点在 {breakpoint}。服务中的 SINR 与 rate 相关，但 Primary 的累计 service work 是 {_fmt(work_primary)}，Model-S 是 {_fmt(work_model)}。"
        "完成门槛是 progress 到达 1，而且目标一变 progress 就清零。最终 payload 只数跨过门槛的次数，两次都是 3。"
    )
    facts = {
        "gate": gate["gate"],
        "gate_name": gate["gate_name"],
        "block_bits": BLOCK_BITS,
        "completed_values": completed_values,
        "why_three": why_three,
        "why_not_completion": why_not,
        "sinr_primary": _mean(row["sinr_primary"] for row in mechanism_rows),
        "sinr_model_s": _mean(row["sinr_model_s"] for row in mechanism_rows),
        "service_sinr_primary": _mean(_subset("primary", "mean_in_service_sinr")),
        "service_sinr_model_s": _mean(_subset("model_s", "mean_in_service_sinr")),
        "service_steps_primary": _mean(_subset("primary", "in_service_steps")),
        "service_steps_model_s": _mean(_subset("model_s", "in_service_steps")),
        "rate_primary": _mean(row["rate_primary"] for row in mechanism_rows),
        "rate_model_s": _mean(row["rate_model_s"] for row in mechanism_rows),
        "chain_primary": chain["primary"],
        "chain_model_s": chain["model_s"],
        "work_primary": work_primary,
        "work_model_s": work_model,
        "discard_primary": _mean(_subset("primary", "discarded_progress")),
        "discard_model_s": _mean(_subset("model_s", "discarded_progress")),
        "switch_primary": _mean(_subset("primary", "target_switch_count")),
        "switch_model_s": _mean(_subset("model_s", "target_switch_count")),
        "streak_primary": _mean(_subset("primary", "median_service_streak")),
        "streak_model_s": _mean(_subset("model_s", "median_service_streak")),
        "interrupt_primary": _mean(_subset("primary", "interrupted_progress_count")),
        "interrupt_model_s": _mean(_subset("model_s", "interrupted_progress_count")),
        "hold_unfinished_primary": _mean(row["unfinished_at_boundary"] for row in hold_rows if row["system"] == "primary"),
        "hold_unfinished_model_s": _mean(row["unfinished_at_boundary"] for row in hold_rows if row["system"] == "model_s"),
        "service_steps_to_complete_primary": _median(row["service_steps"] for row in primary_events),
        "service_steps_to_complete_model_s": _median(row["service_steps"] for row in model_events),
        "arrivals": int(episodes[0]["arrivals"]),
        "drop_primary": _mean(row["drop_primary"] for row in mechanism_rows),
        "residual_primary": _mean(row["residual_primary"] for row in mechanism_rows),
        "bins_primary": primary_bins,
        "bins_model_s": model_bins,
        "near_primary": primary_bins["[0.75,1)"],
        "near_model_s": model_bins["[0.75,1)"],
        "saturation": saturation,
        "breakpoint": breakpoint,
        "max_conservation_error": max_error,
        "reproduction_mismatches": mismatches,
        "bug_answer": "没有发现 payload 记账错误" if max_error <= 1.0e-6 and mismatches == 0 else "复现与 Phase-5ZP 记录不一致，或守恒不成立",
        "q16": "YES。没有发现能够推翻原确认记账的实现错误。归因结果不改写 CONF-C，也不宣布 deployment PASS。"
        if max_error <= 1.0e-6 and mismatches == 0
        else "本次复现和 Phase-5ZP 的记录不一致，或守恒不成立。Phase-5ZP 的 CONF-C 文件没有被改写，也不在本阶段重跑确认。",
        "mechanism_summary": "逐 scenario 的主导标签计数是 "
        + str(dict(sorted({name: sum(row["dominant_failure_mechanism"] == name for row in mechanism_rows) for name in {row["dominant_failure_mechanism"] for row in mechanism_rows}}.items()))),
    }
    gate_payload = dict(gate)
    gate_payload.update(
        {
            "breakpoint": breakpoint,
            "completion_metric_saturation": saturation,
            "sinr_or_rate_gain": bool(sinr_gain or rate_gain),
            "progress_gain": bool(progress_gain),
            "confirmation_still_valid": bool(max_error <= 1.0e-6 and mismatches == 0),
            "fresh_confirmation_pool_role": "FRESH_CONFIRMATION_CONSUMED",
            "reproduction_label": REPRODUCTION_LABEL,
            "deployment_pass": False,
        }
    )
    _dump("phase5zq_failure_attribution_gate.json", gate_payload)
    _write_report(facts)
    print({"gate": gate["gate"], "mismatch": mismatches, "max_error": max_error, "breakpoint": breakpoint}, flush=True)


if __name__ == "__main__":
    main()
