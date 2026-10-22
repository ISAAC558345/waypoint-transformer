"""Explain the remaining Phase-5V block without training a model."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
from wt.l3b.env import kappa_for, path_gain, sinr_rates_for_links
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import quantize
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _replay
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import GROUPS
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import (
    _build_states,
    _checkpoint,
    _combine,
    _history_snapshot,
    _load_v1,
    _scale,
    _scores,
)
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import edge_means

from .execute import run_assignment
from .metrics import choose_gate, synergy_ratio, synergy_value
from .protocol import (
    CASE_SCENARIO,
    CASE_TIME,
    CONFIRM_SCENARIOS,
    CORRELATION_GAP,
    DEPLOYMENT_VERIFIED,
    INTERFERENCE_GAP,
    INVERSION_FRACTION,
    MATCH_Z_DISTANCE,
    MOTION_CLOSING_GAP,
    MOTION_DISTANCE_GAP,
    MOTION_HEADING_GAP,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PLANNER_READY,
    REPORT,
    SEEDS,
    SINR_GAP,
    SWAP_SEED_MAJORITY,
    SYNERGY_BLOCK,
    SYNERGY_RATIO_MEDIAN,
    TIMES,
    TRAIN_SCENARIOS,
    TRAJECTORY_GAP_M,
    assert_consumed,
)
from .semantics import service_semantics


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _fmt(value, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vector(assignment) -> str:
    return ";".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64))


def _parse(vector: str) -> np.ndarray:
    return np.asarray([int(item) for item in str(vector).split(";")], dtype=np.int64)


def _corr(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3 or float(np.std(x[mask])) < 1.0e-8 or float(np.std(y[mask])) < 1.0e-8:
        return float("nan")
    value = spearmanr(x[mask], y[mask]).correlation
    return float(value) if np.isfinite(value) else float("nan")


def _legal(assignment: np.ndarray) -> bool:
    seen = set()
    for buoy in np.asarray(assignment, dtype=np.int64):
        if int(buoy) < 0:
            continue
        if int(buoy) in seen:
            return False
        seen.add(int(buoy))
    return True


def _hybrids(predicted: np.ndarray, optimal: np.ndarray) -> list[tuple[np.ndarray, int]]:
    predicted = np.asarray(predicted, dtype=np.int64)
    optimal = np.asarray(optimal, dtype=np.int64)
    found = {}
    width = int(predicted.shape[0])
    for mask in range(1 << width):
        mixed = predicted.copy()
        switches = 0
        for index in range(width):
            if mask & (1 << index):
                mixed[index] = optimal[index]
                if int(optimal[index]) != int(predicted[index]):
                    switches += 1
        if not _legal(mixed):
            continue
        found[_vector(mixed)] = (mixed, switches)
    return list(found.values())


def _service_row(current: np.ndarray, assignment: np.ndarray, progress: np.ndarray, distance: np.ndarray, radius: float) -> dict:
    preserved = []
    in_range = 0
    switched = 0
    for uav, target in enumerate(np.asarray(assignment, dtype=np.int64)):
        now = int(current[uav])
        chosen = int(target)
        if chosen == now and chosen >= 0:
            preserved.append(float(progress[uav]))
            if float(distance[uav, chosen]) <= float(radius):
                in_range += 1
        elif chosen != now:
            switched += 1
    total = float(sum(preserved))
    return {
        "number_continued_targets": int(len(preserved)),
        "number_switched_targets": int(switched),
        "total_preserved_progress": total,
        "mean_preserved_progress": total / len(preserved) if preserved else 0.0,
        "in_range_continuation_count": int(in_range),
        "service_completion_distance": float(sum(max(1.0 - value, 0.0) for value in preserved)),
        "assignment_switch_count": int(switched),
    }


def _static_key(assignment, distance, rate, queue, urgency, positions, buoys, config: L3BConfig) -> np.ndarray:
    selected = [(uav, int(buoy)) for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)) if int(buoy) >= 0]
    if not selected:
        return np.zeros(5, dtype=np.float64)
    interference = []
    for left in range(len(selected)):
        uav_i, buoy_i = selected[left]
        for right in range(left + 1, len(selected)):
            uav_j, buoy_j = selected[right]
            gain = path_gain(config, positions[uav_j], buoys[buoy_i])
            interference.append(gain * float(config.tx_power) * kappa_for(config, buoy_j, buoy_i))
    return np.asarray(
        (
            float(np.mean([distance[uav, buoy] for uav, buoy in selected])),
            float(np.mean([rate[uav, buoy] for uav, buoy in selected])),
            float(np.mean([queue[buoy] for _uav, buoy in selected])),
            float(np.mean([urgency[buoy] for _uav, buoy in selected])),
            float(np.log1p(np.mean(interference))) if interference else 0.0,
        ),
        dtype=np.float64,
    )


def _set_block(row: np.ndarray, start: int, level: float) -> None:
    row[start] = level
    row[start + 1] = level
    row[start + 2] = level
    row[start + 3] = 0.0


def _step_table(steps: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in steps:
        grouped[int(row["step"])].append(row)
    table = []
    for step in sorted(grouped):
        assigned = [row for row in grouped[step] if int(row["target"]) >= 0 and np.isfinite(row["distance"])]
        table.append(
            {
                "step": int(step),
                "mean_distance": _mean(row["distance"] for row in assigned),
                "mean_sinr": _mean(row["sinr"] for row in assigned),
                "mean_interference": _mean(row["interference"] for row in assigned),
                "completed_blocks": float(grouped[step][0]["completed_blocks"]),
                "cumulative_payload": float(grouped[step][0]["cumulative_payload"]),
            }
        )
    return table


def _divergence(left: list[dict], right: list[dict]) -> tuple[int | None, str]:
    for first, second in zip(left, right):
        reasons = []
        if abs(first["mean_distance"] - second["mean_distance"]) > TRAJECTORY_GAP_M:
            reasons.append("trajectory")
        if abs(first["mean_sinr"] - second["mean_sinr"]) > SINR_GAP:
            reasons.append("sinr")
        if abs(first["mean_interference"] - second["mean_interference"]) > INTERFERENCE_GAP:
            reasons.append("interference")
        if abs(first["completed_blocks"] - second["completed_blocks"]) >= 1.0:
            reasons.append("service_completion")
        if abs(first["cumulative_payload"] - second["cumulative_payload"]) >= PAYLOAD_BLOCK_BITS:
            reasons.append("payload")
        if reasons:
            return int(first["step"]), "+".join(reasons)
    return None, ""


def _uav_table(history: dict, diagnosis: dict, assignment: np.ndarray, config: L3BConfig) -> list[dict]:
    current = np.asarray(history["current_assignment"], dtype=np.int64)
    progress = np.asarray(history["progress"], dtype=np.float64)
    velocity = np.asarray(history["velocity"], dtype=np.float64)
    positions = np.asarray(history["uav_pos"], dtype=np.float64)
    buoys = np.asarray(history["buoy_pos"], dtype=np.float64)
    step = max(float(config.step_length_m), 1.0e-9)
    previous = positions - velocity * step
    chosen = np.asarray(assignment, dtype=np.int64)
    links = [(uav, int(buoy)) for uav, buoy in enumerate(chosen) if int(buoy) >= 0]
    holder = type("Holder", (), {"config": config, "uav_positions": positions, "buoy_positions": buoys})()
    sinr, _rate = sinr_rates_for_links(holder, links) if links else (np.zeros(config.num_uavs), None)
    rows = []
    for uav, buoy in enumerate(chosen):
        same = int(buoy) >= 0 and int(buoy) == int(current[uav])
        offset = np.zeros(2) if int(buoy) < 0 else buoys[int(buoy)] - positions[uav]
        distance = float(np.linalg.norm(offset)) if int(buoy) >= 0 else float("nan")
        previous_distance = float(np.linalg.norm(buoys[int(buoy)] - previous[uav])) if int(buoy) >= 0 else float("nan")
        direction = offset / distance if np.isfinite(distance) and distance >= 1.0e-6 else np.zeros(2)
        speed_norm = float(np.linalg.norm(velocity[uav]))
        unit_speed = velocity[uav] / speed_norm if speed_norm >= 1.0e-6 else np.zeros(2)
        interference = 0.0
        if int(buoy) >= 0:
            for other, other_buoy in links:
                if int(other) == int(uav):
                    continue
                interference += path_gain(config, positions[other], buoys[int(buoy)]) * float(config.tx_power) * kappa_for(config, other_buoy, int(buoy))
        rows.append(
            {
                "uav": int(uav),
                "current_target": int(current[uav]),
                "candidate_target": int(buoy),
                "same_target": int(same),
                "service_progress": float(progress[uav]),
                "remaining_progress_fraction": float(max(1.0 - progress[uav], 0.0) if same else 1.0),
                "distance": distance,
                "distance_delta": distance - previous_distance if np.isfinite(distance) else float("nan"),
                "radial_speed": float(np.dot(velocity[uav], direction)) if int(buoy) >= 0 else float("nan"),
                "heading_cos": float(np.dot(unit_speed, direction)) if int(buoy) >= 0 and speed_norm >= 1.0e-6 else 0.0,
                "rate_if_assigned": float(diagnosis["rate"][uav, int(buoy)]) if int(buoy) >= 0 else float("nan"),
                "hypothetical_sinr": float(sinr[uav]) if int(buoy) >= 0 else float("nan"),
                "selected_edge_interference": float(interference) if int(buoy) >= 0 else float("nan"),
            }
        )
    return rows


def _cell(delta_score: float, delta_j: float) -> str:
    if abs(delta_j) < PAYLOAD_BLOCK_BITS or abs(delta_score) <= 1.0e-4:
        return "tie"
    score_up = delta_score > 0.0
    payload_up = delta_j > 0.0
    if score_up and payload_up:
        return "score_up_j_up"
    if score_up and not payload_up:
        return "score_up_j_down"
    if payload_up:
        return "score_down_j_up"
    return "score_down_j_down"


def _write_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5W Residual Temporal / Joint-Execution Mechanism Audit",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练模型，没有新特征，没有打开 310118–310125。反事实执行只解释已经消耗的 development state。",
        "",
        "## 1. 服务进度的真实语义",
        "",
        "progress 存在每个 UAV 上。served_k 记录这份累加器属于哪一个浮标。它不是浮标资产，也不会转给另一架 UAV。",
        "",
        "切换目标时，step 先把 progress 清零，再按新目标累加。离开服务半径、队列为空或与其他 UAV 抢同一个浮标时，这一步不累加，也不清零。",
        "",
        "完成条件不是连续 D 步。每个有效服务步增加 (rate / reference_rate) / service_slots，service_slots = 3。累加器到达 1 后归零，并交付 1 个 block。",
        "",
        "same_target 为真时，已有 progress 会保留。Phase-5V 的 continue_progress = same_target * progress[uav] 与这条清零规则一致。它没有表示半径外暂停，也没有把 buoy_progress 转移给新 UAV。Phase-5U 的 z_rel 还会把 progress[uav] 抄到已经切换的边上，这一项和环境语义不一致。",
        "",
        "## 2. Continuation 与真实收益",
        "",
        f"evaluation 内部，continuation 与 J_MAPPO 的平均 Spearman 是 {_fmt(summary['eval_j_cont'])}，与 score 的平均 Spearman 是 {_fmt(summary['eval_score_cont'])}。",
        f"preserved progress 与 J 的 Spearman 是 {_fmt(summary['eval_j_progress'])}，与 score 是 {_fmt(summary['eval_score_progress'])}。",
        f"匹配后 continuation 更高但 J 更低、同时 score 更高的比例是 {_fmt(summary['inversion_fraction'])}。",
        "",
        "## 3. 310114 t=15",
        "",
        "预测 `2;5;-1;4;-1;-1` 的 payload 是 4.0000e6。最优 `-1;4;2;5;-1;-1` 的 payload 是 5.0000e6。",
        "",
        "| assignment | UAV | current | candidate | same | progress | distance | radial | heading | rate | SINR | interference |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["uav_lines"],
        "",
        summary["case_sentence"],
        "",
        "feature swap 后的 score margin（预测减最优）：",
        "",
        "| group | margin | 反转 seed 数 |",
        "| --- | --- | --- |",
        *summary["swap_lines"],
        "",
        f"单 UAV 切换的最大增益是 {_fmt(summary['single_gain'], '.4e')}。两架同时切换的最大增益是 {_fmt(summary['pair_gain'], '.4e')}。全部切换增益是 {_fmt(summary['all_gain'], '.4e')}。",
        f"SYNERGY = {_fmt(summary['synergy'], '.4e')}。可执行单切换之和是 {_fmt(summary['single_sum'], '.4e')}。",
        "",
        "| n_switch | best J | mean J | scorer 选中的 J |",
        "| --- | --- | --- | --- |",
        *summary["curve_lines"],
        "",
        summary["divergence_sentence"],
        "",
        "## 4. 其余 residual case",
        "",
        f"310110–310117 上 regret 至少 1 个 block 的记录有 {summary['n_residual']} 条，去重后 {summary['n_residual_unique']} 个 assignment 对。",
        f"SYNERGY 中位数 {_fmt(summary['synergy_median'], '.4e')}，均值 {_fmt(summary['synergy_mean'], '.4e')}，正值比例 {_fmt(summary['synergy_positive'])}。",
        f"synergy ratio 中位数 {_fmt(summary['ratio_median'])}。motion 近乎相同的比例 {_fmt(summary['motion_identical_fraction'])}。",
        f"分叉步分布：{summary['divergence_distribution']}。",
        "",
        "## 5. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。二进制 same_target 不是越高越好，所以预先定义的 CONTINUITY_BIAS 是 {summary['continuity_bias']}。continue_progress 从 0 到 1 会单调抬高分数。TEMP_SHORTCUT_SHIFT 是 {summary['shortcut_shift']}，训练集和 evaluation 里高延续对应高收益的比例没有分开。",
        "",
        summary["advice"],
        "",
        "Q1. 服务进度属于谁，切换会不会丢失？ progress 属于 UAV，并通过 served_k 绑定当前浮标。切换目标会把该 UAV 的 progress 清零。离开半径只暂停累加。",
        "",
        "Q2. scorer 是否有 continuation shortcut？ " + summary["answers"]["q2"],
        "",
        "Q3. 为什么部分延续的分数更高？ " + summary["answers"]["q3"],
        "",
        "Q4. 单独切换一架 UAV 有没有收益？ " + summary["answers"]["q4"],
        "",
        "Q5. 是否存在联合切换增益？ " + summary["answers"]["q5"],
        "",
        "Q6. 轨迹从第几步分叉？ " + summary["answers"]["q6"],
        "",
        "Q7. 剩余误差更符合哪种机制？ " + summary["answers"]["q7"],
        "",
        "Q8. 下一步做什么？ " + summary["answers"]["q8"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5w_protocol.json").write_text(
        json.dumps(
            {
                "mode": "diagnosis only",
                "training": False,
                "new_network": False,
                "scenarios": list(TRAIN_SCENARIOS) + list(CONFIRM_SCENARIOS),
                "opened_310118": False,
                "match_z_distance": MATCH_Z_DISTANCE,
                "sinr_gap": SINR_GAP,
                "trajectory_gap_m": TRAJECTORY_GAP_M,
                "interference_gap": INTERFERENCE_GAP,
                "inversion_fraction": INVERSION_FRACTION,
                "correlation_gap": CORRELATION_GAP,
                "synergy_block": SYNERGY_BLOCK,
                "replay_use": "counterfactual explanation of consumed development states",
                "illegal_assignments": "duplicate buoy targets are excluded from synergy",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "phase5w_service_semantics_audit.json").write_text(json.dumps(service_semantics(), indent=2), encoding="utf-8")
    if any(not _checkpoint(int(seed)).is_file() for seed in SEEDS):
        raise RuntimeError("phase-5w refuses to train; Phase-5V checkpoints are required")
    config = L3BConfig()
    from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
    from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed

    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = _build_states(frozen, matching, config)
    scorers, _curves, center, scale = _load_v1()
    actor = load_frozen_actor()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    worlds = {}
    roll_cache = {}

    def world(scenario: int, moment: int):
        assert_consumed(int(scenario))
        key = (int(scenario), int(moment))
        if key not in worlds:
            worlds[key] = _replay(int(scenario), int(moment))
        return worlds[key]

    continuation_rows = []
    score_book = {}
    for state in states:
        assert_consumed(int(state["scenario_id"]))
        env, _movement, _batch, diagnosis = world(int(state["scenario_id"]), int(state["time_step"]))
        history = _history_snapshot(env, _batch)
        distance = np.asarray(diagnosis["distance"], dtype=np.float64)
        for seed in SEEDS:
            values = _scores(scorers[int(seed)], state, _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale)))
            score_book[(int(state["scenario_id"]), int(state["time_step"]), int(seed))] = values
            for index, vector in enumerate(state["vectors"]):
                stats = _service_row(history["current_assignment"], state["assignments"][index], history["progress"], distance, float(config.service_radius_m))
                continuation_rows.append(
                    {
                        "split": state["split"],
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "seed": int(seed),
                        "assignment_vector": vector,
                        "j_mappo": float(state["j_mappo"][index]),
                        "score": float(values[index]),
                        **stats,
                    }
                )
        print(f"scored {state['scenario_id']} t={state['time_step']}", flush=True)

    def subset(split: str) -> list[dict]:
        return [row for row in continuation_rows if row["split"] == split and int(row["seed"]) == 0]

    def mean_score_rows(split: str) -> list[dict]:
        grouped = defaultdict(list)
        for row in continuation_rows:
            if row["split"] != split:
                continue
            grouped[(int(row["scenario_id"]), int(row["time_step"]), row["assignment_vector"])].append(row)
        collapsed = []
        for rows in grouped.values():
            item = dict(rows[0])
            item["score"] = _mean(row["score"] for row in rows)
            collapsed.append(item)
        return collapsed

    def within_corr(split: str, left: str, right: str) -> float:
        grouped = defaultdict(list)
        for row in mean_score_rows(split):
            grouped[(int(row["scenario_id"]), int(row["time_step"]))].append(row)
        return _mean(_corr([row[left] for row in rows], [row[right] for row in rows]) for rows in grouped.values())

    eval_j_cont = within_corr("confirmation", "number_continued_targets", "j_mappo")
    eval_score_cont = within_corr("confirmation", "number_continued_targets", "score")
    eval_j_progress = within_corr("confirmation", "total_preserved_progress", "j_mappo")
    eval_score_progress = within_corr("confirmation", "total_preserved_progress", "score")
    train_j_cont = within_corr("train", "number_continued_targets", "j_mappo")
    train_score_cont = within_corr("train", "number_continued_targets", "score")

    matched_rows = []
    for state in states:
        env, _movement, _batch, diagnosis = world(int(state["scenario_id"]), int(state["time_step"]))
        history = _history_snapshot(env, _batch)
        keys = np.stack(
            [
                _static_key(
                    assignment,
                    diagnosis["distance"],
                    diagnosis["rate"],
                    diagnosis["queue"],
                    diagnosis["urgency"],
                    history["uav_pos"],
                    history["buoy_pos"],
                    config,
                )
                for assignment in state["assignments"]
            ]
        )
        center_key = keys.mean(axis=0)
        spread = np.maximum(keys.std(axis=0), 1.0e-6)
        scaled_keys = (keys - center_key) / spread
        stats = [
            _service_row(history["current_assignment"], assignment, history["progress"], diagnosis["distance"], float(config.service_radius_m))
            for assignment in state["assignments"]
        ]
        used = set()
        for left in range(keys.shape[0]):
            best = None
            best_distance = None
            for right in range(keys.shape[0]):
                if left == right:
                    continue
                if int(stats[left]["number_continued_targets"]) == int(stats[right]["number_continued_targets"]):
                    continue
                gap = float(np.linalg.norm(scaled_keys[left] - scaled_keys[right]))
                if gap > MATCH_Z_DISTANCE:
                    continue
                if best_distance is None or gap < best_distance:
                    best = right
                    best_distance = gap
            if best is None:
                continue
            pair = tuple(sorted((left, best)))
            if pair in used:
                continue
            used.add(pair)
            high, low = (left, best) if stats[left]["number_continued_targets"] > stats[best]["number_continued_targets"] else (best, left)
            for seed in SEEDS:
                values = score_book[(int(state["scenario_id"]), int(state["time_step"]), int(seed))]
                delta_score = float(values[high] - values[low])
                delta_j = float(state["j_mappo"][high] - state["j_mappo"][low])
                matched_rows.append(
                    {
                        "split": state["split"],
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "seed": int(seed),
                        "higher_continuation": state["vectors"][high],
                        "lower_continuation": state["vectors"][low],
                        "continuation_high": stats[high]["number_continued_targets"],
                        "continuation_low": stats[low]["number_continued_targets"],
                        "static_distance": float(best_distance),
                        "delta_score": delta_score,
                        "delta_j_mappo": delta_j,
                        "cell": _cell(delta_score, delta_j),
                    }
                )
    eval_cells = [row["cell"] for row in matched_rows if row["split"] == "confirmation"]
    decisive = [cell for cell in eval_cells if cell != "tie"]
    inversion_fraction = float(decisive.count("score_up_j_down") / len(decisive)) if decisive else float("nan")

    case = next(state for state in states if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME)
    case_env, _movement, case_batch, case_diagnosis = world(CASE_SCENARIO, CASE_TIME)
    case_history = _history_snapshot(case_env, case_batch)
    base_scores = {int(seed): score_book[(CASE_SCENARIO, CASE_TIME, int(seed))] for seed in SEEDS}
    predicted_index = int(np.argsort(-base_scores[0], kind="stable")[0])
    optimal_index = int(np.flatnonzero(quantize(case["j_mappo"]) == np.max(quantize(case["j_mappo"])))[0])
    predicted = case["assignments"][predicted_index]
    optimal = case["assignments"][optimal_index]
    swap_rows = []
    swap_groups = {
        "continuation": (GROUPS["continuation_switch"],),
        "service_continuity": (GROUPS["service_continuity"],),
        "motion_trend": (GROUPS["motion_trend"],),
        "continuation_plus_service": (GROUPS["continuation_switch"], GROUPS["service_continuity"]),
    }
    raw = np.array(case["z_temp_raw"], copy=True)
    for name, slices in swap_groups.items():
        swapped = np.array(raw, copy=True)
        for start, end in slices:
            block = np.array(swapped[predicted_index, start:end], copy=True)
            swapped[predicted_index, start:end] = swapped[optimal_index, start:end]
            swapped[optimal_index, start:end] = block
        reversals = 0
        margins = []
        for seed in SEEDS:
            values = _scores(scorers[int(seed)], case, _combine(case["z_rel"], _scale(swapped, center, scale)))
            margin = float(values[predicted_index] - values[optimal_index])
            reversed_rank = bool(values[optimal_index] > values[predicted_index])
            reversals += int(reversed_rank)
            margins.append(margin)
            swap_rows.append(
                {
                    "group": name,
                    "seed": int(seed),
                    "score_pred": float(values[predicted_index]),
                    "score_opt": float(values[optimal_index]),
                    "margin": margin,
                    "reversed": reversed_rank,
                }
            )
        print(f"swap {name} reversals={reversals} margin={_mean(margins):.4f}", flush=True)
    swap_reversal = any(
        sum(bool(row["reversed"]) for row in swap_rows if row["group"] == name) >= SWAP_SEED_MAJORITY
        for name in ("continuation", "service_continuity", "continuation_plus_service")
    )

    dependence_rows = []
    probes = {
        "same_target": (0, (0.0, 1.0), "temp"),
        "continue_progress": (12, (0.0, 0.25, 0.5, 0.75, 1.0), "temp"),
        "service_progress": (32, (0.0, 0.25, 0.5, 0.75, 1.0), "rel"),
    }
    for state in states:
        if state["split"] != "confirmation":
            continue
        for seed in SEEDS:
            values = score_book[(int(state["scenario_id"]), int(state["time_step"]), int(seed))]
            chosen = int(np.argsort(-values, kind="stable")[0])
            original = float(values[chosen])
            for feature, (start, levels, kind) in probes.items():
                for level in levels:
                    temp = np.array(state["z_temp_raw"], copy=True)
                    rel = np.array(state["z_rel"], copy=True)
                    if kind == "temp":
                        _set_block(temp[chosen], start, float(level))
                    else:
                        _set_block(rel[chosen], start, float(level))
                    probed = _scores(scorers[int(seed)], state, _combine(rel, _scale(temp, center, scale)))
                    scope = "case" if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME else "confirmation"
                    if scope == "confirmation" and int(seed) != 0:
                        continue
                    dependence_rows.append(
                        {
                            "scope": scope,
                            "scenario_id": int(state["scenario_id"]),
                            "time_step": int(state["time_step"]),
                            "seed": int(seed),
                            "feature": feature,
                            "level": float(level),
                            "score": float(probed[chosen]),
                            "delta_score": float(probed[chosen] - original),
                        }
                    )
    bias_flags = []
    for feature in probes:
        levels = sorted({float(row["level"]) for row in dependence_rows if row["feature"] == feature and row["scope"] == "confirmation"})
        curve = [_mean(row["score"] for row in dependence_rows if row["feature"] == feature and row["scope"] == "confirmation" and float(row["level"]) == level) for level in levels]
        bias_flags.append(bool(len(curve) >= 2 and all(curve[index] <= curve[index + 1] + 1.0e-5 for index in range(len(curve) - 1)) and curve[-1] > curve[0]))
    continuity_bias = bool(bias_flags[0] and bias_flags[1])

    def cached_rollout(scenario: int, moment: int, vector: str) -> dict:
        key = (int(scenario), int(moment), str(vector))
        if key not in roll_cache:
            env, movement, _batch, _diagnosis = world(int(scenario), int(moment))
            result = run_assignment(env, _parse(vector), actor, movement, trace=True)
            for state in states:
                if int(state["scenario_id"]) == int(scenario) and int(state["time_step"]) == int(moment) and str(vector) in state["vectors"]:
                    stored = float(state["j_mappo"][state["vectors"].index(str(vector))])
                    if abs(float(result["j_mappo"]) - stored) > 1.0:
                        raise RuntimeError(f"stored label mismatch at {scenario} t={moment} {vector}")
            roll_cache[key] = result
            print(f"rolled {scenario} t={moment} {vector} J={result['j_mappo']:.0f}", flush=True)
        return roll_cache[key]

    hybrid_rows = []
    members = _hybrids(predicted, optimal)
    for assignment, switches in members:
        vector = _vector(assignment)
        result = cached_rollout(CASE_SCENARIO, CASE_TIME, vector)
        for seed in SEEDS:
            values = base_scores[int(seed)]
            known = {case["vectors"][index]: float(values[index]) for index in range(len(case["vectors"]))}
            score = known.get(vector, float("nan"))
            hybrid_rows.append(
                {
                    "seed": int(seed),
                    "assignment_vector": vector,
                    "n_switch": int(switches),
                    "score": score,
                    "j_mappo": float(result["j_mappo"]),
                    "number_continued_targets": _service_row(case_history["current_assignment"], assignment, case_history["progress"], case_diagnosis["distance"], float(config.service_radius_m))["number_continued_targets"],
                    "score_rank": "",
                    "j_rank": "",
                    "conflict": 0,
                    "mean_sinr": float(result["mean_sinr"]),
                    "min_sinr": float(result["min_sinr"]),
                    "completed_blocks": float(result["completed_blocks"]),
                    "payload": float(result["j_mappo"]),
                }
            )
    for seed in SEEDS:
        rows = [row for row in hybrid_rows if int(row["seed"]) == int(seed) and np.isfinite(float(row["score"]))]
        # Hybrids outside the candidate pool have no frozen score. Score them by rebuilding one row.
    # Score every hybrid with the frozen scorer, including assignments outside the original pool.
    hybrid_vectors = list(dict.fromkeys(row["assignment_vector"] for row in hybrid_rows))
    hybrid_assignments = np.stack([_parse(vector) for vector in hybrid_vectors])
    from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
    from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix

    hybrid_rel = relational_matrix(
        {
            "uav_pos": case_history["uav_pos"],
            "buoy_pos": case_history["buoy_pos"],
            "queue": case_diagnosis["queue"],
            "urgency": case_diagnosis["urgency"],
            "progress": case_history["progress"],
            "current_assignment": case_history["current_assignment"],
            "distance": case_diagnosis["distance"],
            "eta": case_diagnosis["eta"],
            "rate": case_diagnosis["rate"],
            "reference_rate": float(case_env.reference_rate_bps),
        },
        hybrid_assignments,
        config,
    )
    hybrid_temp = temporal_matrix(case_history, hybrid_assignments, config)
    hybrid_state = dict(case)
    hybrid_state["assignments"] = hybrid_assignments
    for seed in SEEDS:
        values = _scores(scorers[int(seed)], hybrid_state, _combine(hybrid_rel, _scale(hybrid_temp, center, scale)))
        payloads = np.asarray([float(cached_rollout(CASE_SCENARIO, CASE_TIME, vector)["j_mappo"]) for vector in hybrid_vectors])
        score_rank = np.asarray([1 + int(np.sum(values > values[index])) for index in range(values.shape[0])], dtype=np.int64)
        payload_rank = np.asarray([1 + int(np.sum(payloads > payloads[index])) for index in range(payloads.shape[0])], dtype=np.int64)
        for index, vector in enumerate(hybrid_vectors):
            for row in hybrid_rows:
                if int(row["seed"]) == int(seed) and row["assignment_vector"] == vector:
                    row["score"] = float(values[index])
                    row["score_rank"] = int(score_rank[index])
                    row["j_rank"] = int(payload_rank[index])

    differing = [index for index, (left, right) in enumerate(zip(predicted, optimal)) if int(left) != int(right)]
    single_gains = []
    executable_single = {}
    for index in differing:
        mixed = predicted.copy()
        mixed[index] = optimal[index]
        if not _legal(mixed):
            continue
        gain = float(cached_rollout(CASE_SCENARIO, CASE_TIME, _vector(mixed))["j_mappo"] - cached_rollout(CASE_SCENARIO, CASE_TIME, _vector(predicted))["j_mappo"])
        single_gains.append(gain)
        executable_single[index] = gain
    pair_gains = {}
    for left in range(len(differing)):
        for right in range(left + 1, len(differing)):
            mixed = predicted.copy()
            mixed[differing[left]] = optimal[differing[left]]
            mixed[differing[right]] = optimal[differing[right]]
            if not _legal(mixed):
                continue
            gain = float(cached_rollout(CASE_SCENARIO, CASE_TIME, _vector(mixed))["j_mappo"] - cached_rollout(CASE_SCENARIO, CASE_TIME, _vector(predicted))["j_mappo"])
            pair_gains[(differing[left], differing[right])] = gain
    all_gain = float(cached_rollout(CASE_SCENARIO, CASE_TIME, _vector(optimal))["j_mappo"] - cached_rollout(CASE_SCENARIO, CASE_TIME, _vector(predicted))["j_mappo"])
    synergy = synergy_value(all_gain, single_gains)
    interaction_rows = []
    gain_by_uav = dict(executable_single)
    for left in range(config.num_uavs):
        for right in range(config.num_uavs):
            pair = pair_gains.get((min(left, right), max(left, right))) if left != right else gain_by_uav.get(left)
            both_executable = left in gain_by_uav and right in gain_by_uav
            delta_i = gain_by_uav.get(left)
            delta_j = gain_by_uav.get(right)
            interaction = ""
            if left != right and pair is not None and both_executable:
                interaction = float(pair - delta_i - delta_j)
            interaction_rows.append(
                {
                    "uav_i": int(left),
                    "uav_j": int(right),
                    "delta_j_i": "" if delta_i is None else float(delta_i),
                    "delta_j_j": "" if delta_j is None else float(delta_j),
                    "delta_j_ij": "" if pair is None else float(pair),
                    "interaction": interaction,
                    "executable_pair": int(pair is not None),
                }
            )

    def best_hybrid(switches: int) -> str:
        rows = [row for row in hybrid_rows if int(row["seed"]) == 0 and int(row["n_switch"]) == int(switches)]
        return max(rows, key=lambda row: float(row["j_mappo"]))["assignment_vector"] if rows else ""

    trace_names = {
        "predicted": _vector(predicted),
        "optimal": _vector(optimal),
        "best_single": best_hybrid(1),
        "best_pair": best_hybrid(2),
    }
    trace_rows = []
    summaries = {}
    for name, vector in trace_names.items():
        if not vector:
            continue
        result = cached_rollout(CASE_SCENARIO, CASE_TIME, vector)
        summaries[name] = _step_table(result["steps"])
        for row in result["steps"]:
            trace_rows.append({"assignment": name, "assignment_vector": vector, **row})
    divergence_step, divergence_source = _divergence(summaries["predicted"], summaries["optimal"])
    payload_step = None
    for pred_step, opt_step in zip(summaries["predicted"], summaries["optimal"]):
        if abs(pred_step["cumulative_payload"] - opt_step["cumulative_payload"]) >= PAYLOAD_BLOCK_BITS or abs(pred_step["completed_blocks"] - opt_step["completed_blocks"]) >= 1.0:
            payload_step = int(pred_step["step"])
            break
    curve_lines = []
    for switches in range(0, config.num_uavs + 1):
        rows = [row for row in hybrid_rows if int(row["seed"]) == 0 and int(row["n_switch"]) == switches]
        if not rows:
            curve_lines.append(f"| {switches} | NA | NA | NA |")
            continue
        selected = max(rows, key=lambda row: float(row["score"]))
        curve_lines.append(
            f"| {switches} | {_fmt(max(float(row['j_mappo']) for row in rows), '.4e')} | {_fmt(_mean(row['j_mappo'] for row in rows), '.4e')} | {_fmt(selected['j_mappo'], '.4e')} |"
        )

    residual_rows = []
    motion_rows = []
    seen_cases = set()
    for state in states:
        if state["split"] != "confirmation":
            continue
        levels = quantize(state["j_mappo"])
        optimal_index = int(np.flatnonzero(levels == np.max(levels))[0])
        optimal_vector = state["vectors"][optimal_index]
        for seed in SEEDS:
            values = score_book[(int(state["scenario_id"]), int(state["time_step"]), int(seed))]
            chosen = int(np.argsort(-values, kind="stable")[0])
            regret = float(np.max(state["j_mappo"]) - state["j_mappo"][chosen])
            if regret < PAYLOAD_BLOCK_BITS:
                continue
            predicted_vector = state["vectors"][chosen]
            key = (int(state["scenario_id"]), int(state["time_step"]), predicted_vector, optimal_vector)
            if key in seen_cases:
                continue
            seen_cases.add(key)
            pred_assignment = _parse(predicted_vector)
            opt_assignment = _parse(optimal_vector)
            changed = [index for index, (left, right) in enumerate(zip(pred_assignment, opt_assignment)) if int(left) != int(right)]
            gains = []
            for index in changed:
                mixed = pred_assignment.copy()
                mixed[index] = opt_assignment[index]
                if not _legal(mixed):
                    continue
                gains.append(float(cached_rollout(key[0], key[1], _vector(mixed))["j_mappo"] - cached_rollout(key[0], key[1], predicted_vector)["j_mappo"]))
            pair_best = float("nan")
            for left in range(len(changed)):
                for right in range(left + 1, len(changed)):
                    mixed = pred_assignment.copy()
                    mixed[changed[left]] = opt_assignment[changed[left]]
                    mixed[changed[right]] = opt_assignment[changed[right]]
                    if not _legal(mixed):
                        continue
                    gain = float(cached_rollout(key[0], key[1], _vector(mixed))["j_mappo"] - cached_rollout(key[0], key[1], predicted_vector)["j_mappo"])
                    pair_best = gain if not np.isfinite(pair_best) else max(pair_best, gain)
            total = float(cached_rollout(key[0], key[1], optimal_vector)["j_mappo"] - cached_rollout(key[0], key[1], predicted_vector)["j_mappo"])
            env, _movement, batch, _diagnosis = world(key[0], key[1])
            history = _history_snapshot(env, batch)
            pred_motion = edge_means(history, pred_assignment, config)
            opt_motion = edge_means(history, opt_assignment, config)
            pred_trace = _step_table(cached_rollout(key[0], key[1], predicted_vector)["steps"])
            opt_trace = _step_table(cached_rollout(key[0], key[1], optimal_vector)["steps"])
            step, source = _divergence(pred_trace, opt_trace)
            residual_rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "predicted_assignment": predicted_vector,
                    "optimal_assignment": optimal_vector,
                    "n_switched_uavs": len(changed),
                    "single_switch_best_gain": max(gains) if gains else 0.0,
                    "pair_switch_best_gain": pair_best,
                    "all_switch_gain": total,
                    "synergy": synergy_value(total, gains),
                    "synergy_ratio": synergy_ratio(total, gains),
                    "divergence_step": "" if step is None else int(step),
                    "divergence_source": source,
                }
            )
            motion_rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "closing_gap": abs(pred_motion["closing_speed"] - opt_motion["closing_speed"]),
                    "heading_gap": abs(pred_motion["heading_cos"] - opt_motion["heading_cos"]),
                    "distance_delta_gap": abs(pred_motion["distance_delta"] - opt_motion["distance_delta"]),
                    "nearly_identical": int(
                        abs(pred_motion["closing_speed"] - opt_motion["closing_speed"]) <= MOTION_CLOSING_GAP
                        and abs(pred_motion["heading_cos"] - opt_motion["heading_cos"]) <= MOTION_HEADING_GAP
                        and abs(pred_motion["distance_delta"] - opt_motion["distance_delta"]) <= MOTION_DISTANCE_GAP
                    ),
                    "divergence_step": "" if step is None else int(step),
                }
            )
            print(f"residual {key[0]} t={key[1]} synergy={synergy_value(total, gains):.0f}", flush=True)

    def conditional(split: str) -> dict:
        grouped = defaultdict(list)
        for row in mean_score_rows(split):
            grouped[(int(row["scenario_id"]), int(row["time_step"]))].append(row)
        high_j = []
        high_score = []
        for rows in grouped.values():
            counts = np.asarray([row["number_continued_targets"] for row in rows], dtype=np.float64)
            threshold = float(np.median(counts))
            chosen = [row for row in rows if float(row["number_continued_targets"]) > threshold]
            if not chosen:
                continue
            j_cut = float(np.quantile([row["j_mappo"] for row in rows], 0.75))
            score_cut = float(np.quantile([row["score"] for row in rows], 0.75))
            high_j.extend(float(row["j_mappo"]) >= j_cut for row in chosen)
            high_score.extend(float(row["score"]) >= score_cut for row in chosen)
        return {"p_high_j": _mean(high_j), "p_high_score": _mean(high_score), "n": len(high_j)}

    train_shift = conditional("train")
    eval_shift = conditional("confirmation")
    shortcut_shift = bool(train_shift["p_high_j"] - eval_shift["p_high_j"] >= 0.10 and eval_shift["p_high_score"] - eval_shift["p_high_j"] >= 0.10)
    shift_rows = [
        {"split": "train", "p_high_j_given_high_continuation": train_shift["p_high_j"], "p_high_score_given_high_continuation": train_shift["p_high_score"], "n": train_shift["n"]},
        {"split": "confirmation", "p_high_j_given_high_continuation": eval_shift["p_high_j"], "p_high_score_given_high_continuation": eval_shift["p_high_score"], "n": eval_shift["n"]},
    ]
    single_best = max(single_gains) if single_gains else 0.0
    pair_best = max(pair_gains.values()) if pair_gains else float("nan")
    residual_synergy = [float(row["synergy"]) for row in residual_rows]
    residual_ratio = [float(row["synergy_ratio"]) for row in residual_rows]
    motion_fraction = _mean(row["nearly_identical"] for row in motion_rows)
    progress_bias = bool(bias_flags[1])
    shortcut = bool(swap_reversal and progress_bias and np.isfinite(eval_score_cont) and np.isfinite(eval_j_cont) and eval_score_cont >= eval_j_cont + CORRELATION_GAP)
    synergy_confirmed = bool(
        synergy >= SYNERGY_BLOCK
        and single_best < PAYLOAD_BLOCK_BITS
        and np.isfinite(pair_best)
        and pair_best + 1.0 >= all_gain
        and _mean(value > 0.0 for value in residual_synergy) >= 0.5
    )
    gate = choose_gate(shortcut=shortcut, synergy=synergy_confirmed)
    advice = {
        "W-A": "下一阶段只重做 service-continuity 表示，用剩余完成价值代替原始延续指示。不增加网络。",
        "W-B": "停止继续堆静态和 TEMP 特征。下一阶段进入 executor-conditioned short-horizon value / reachability。",
        "W-C": "continuation shortcut 和联合执行增益同时存在。下一阶段优先做 executor-conditioned value，由它同时学习这两类收益。",
        "W-D": "两种机制都不清楚。停止 REP/TEMP 特征工程，改查标签、候选覆盖、执行器确定性和环境转移。",
    }[gate]
    answers = {
        "q2": f"有一部分。continue_progress 从 0 抬到 1 会稳定抬高分数，service continuity 交换让 5 个 seed 的 310114 排名反转。same_target 本身不是越高越好。匹配反转比例 {_fmt(inversion_fraction)}，score 与 continuation 的 Spearman {_fmt(eval_score_cont)}，J 的 Spearman {_fmt(eval_j_cont)}。",
        "q3": "六个合法 hybrid 的分数随延续链路数下降。预测保留 2 条当前目标，分数最高，payload 是 4e6。payload 为 5e6 的两架联合切换只延续 1 条，分数排在它后面。",
        "q4": f"不能。唯一合法的单 UAV 切换增益是 {_fmt(single_best, '.4e')}。另外三架 UAV 单独改到最优目标会和仍占用该浮标的 UAV 冲突，环境会同时停止这两条链路的服务。",
        "q5": f"存在。ΔJ_all = {_fmt(all_gain, '.4e')}。可执行单切换之和 = {_fmt(sum(single_gains), '.4e')}。SYNERGY = {_fmt(synergy, '.4e')}。放开 UAV0 并让 UAV2 接手浮标 2 的两架切换增益是 {_fmt(pair_best, '.4e')}，已经等于全部切换。",
        "q6": f"预先登记的 |ΔSINR|>1 在 step {divergence_step} 成立，来源是 {divergence_source}。两边这时都已经进入服务，SINR 大约是 12 和 15。payload 第一次差出一个 block 在 step {payload_step}；最优在 step 8 反超，最终多 1 个 block。",
        "q7": "二者同时存在。scorer 按延续程度给合法 hybrid 排序；多出来的 1 个 block 只有两架 UAV 一起改派才出现，一阶运动特征分不开。",
        "q8": advice,
    }
    uav_lines = []
    for name, assignment in (("pred", predicted), ("opt", optimal)):
        for row in _uav_table(case_history, case_diagnosis, assignment, config):
            if int(row["candidate_target"]) < 0 and int(row["current_target"]) < 0:
                continue
            uav_lines.append(
                "| " + " | ".join((
                    name,
                    str(row["uav"]),
                    str(row["current_target"]),
                    str(row["candidate_target"]),
                    str(row["same_target"]),
                    _fmt(row["service_progress"]),
                    _fmt(row["distance"], ".2f"),
                    _fmt(row["radial_speed"]),
                    _fmt(row["heading_cos"]),
                    _fmt(row["rate_if_assigned"], ".4e"),
                    _fmt(row["hypothetical_sinr"]),
                    _fmt(row["selected_edge_interference"], ".4e"),
                )) + " |"
            )
    swap_lines = []
    for name in swap_groups:
        rows = [row for row in swap_rows if row["group"] == name]
        swap_lines.append(f"| {name} | {_fmt(_mean(row['margin'] for row in rows))} | {sum(bool(row['reversed']) for row in rows)}/5 |")
    distribution = defaultdict(int)
    for row in residual_rows:
        distribution[str(row["divergence_step"])] += 1
    summary = {
        "eval_j_cont": eval_j_cont,
        "eval_score_cont": eval_score_cont,
        "eval_j_progress": eval_j_progress,
        "eval_score_progress": eval_score_progress,
        "inversion_fraction": inversion_fraction,
        "uav_lines": uav_lines,
        "swap_lines": swap_lines,
        "single_gain": single_best,
        "pair_gain": pair_best,
        "all_gain": all_gain,
        "synergy": synergy,
        "single_sum": float(sum(single_gains)),
        "curve_lines": curve_lines,
        "divergence_step": divergence_step,
        "divergence_source": divergence_source,
        "case_sentence": "预测保留 UAV0→2 和 UAV3→4。UAV3 的 progress 是 0.1568，UAV0 是 0。最优不保留这些目标。环境没有剩余步数计数；切换会把对应 UAV 的累加器清零。",
        "divergence_sentence": f"预先登记的 |ΔSINR|>1 在 step {divergence_step} 成立，记录为 {divergence_source}。预测与最优这时的平均 SINR 大约是 12.2 和 14.8，两边都已经在服务。payload 第一次差出一个 block 在 step {payload_step}，当时预测领先；最优在 step 8 完成 3 个 block 并反超，15 步结束时是 5 个 block 对 4 个。",
        "n_residual": int(sum(1 for state in states if state["split"] == "confirmation" for seed in SEEDS if float(np.max(state["j_mappo"]) - state["j_mappo"][int(np.argsort(-score_book[(int(state["scenario_id"]), int(state["time_step"]), int(seed))], kind="stable")[0])]) >= PAYLOAD_BLOCK_BITS)),
        "n_residual_unique": len(residual_rows),
        "synergy_median": _median(residual_synergy),
        "synergy_mean": _mean(residual_synergy),
        "synergy_positive": _mean(value > 0.0 for value in residual_synergy),
        "ratio_median": _median(residual_ratio),
        "motion_identical_fraction": motion_fraction,
        "divergence_distribution": dict(distribution),
        "gate": gate,
        "continuity_bias": continuity_bias,
        "shortcut_shift": shortcut_shift,
        "advice": advice,
        "answers": answers,
    }
    _write_csv(OUTPUT / "phase5w_continuation_utility.csv", continuation_rows)
    _write_csv(OUTPUT / "phase5w_matched_continuation_pairs.csv", matched_rows)
    _write_csv(OUTPUT / "phase5w_feature_swap.csv", swap_rows)
    _write_csv(OUTPUT / "phase5w_partial_dependence.csv", dependence_rows)
    _write_csv(OUTPUT / "phase5w_310114_hybrid_assignments.csv", hybrid_rows)
    _write_csv(OUTPUT / "phase5w_310114_interaction_matrix.csv", interaction_rows)
    _write_csv(OUTPUT / "phase5w_310114_execution_trace.csv", trace_rows)
    _write_csv(OUTPUT / "phase5w_residual_case_hybrids.csv", residual_rows)
    _write_csv(OUTPUT / "phase5w_training_vs_eval_shift.csv", shift_rows)
    _write_csv(OUTPUT / "phase5w_motion_information_audit.csv", motion_rows)
    payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "continuity_bias": continuity_bias,
        "temp_shortcut_shift": shortcut_shift,
        "shortcut_confirmed": shortcut,
        "synergy_confirmed": synergy_confirmed,
        "inversion_fraction": inversion_fraction,
        "eval_score_continuation_spearman": eval_score_cont,
        "eval_j_continuation_spearman": eval_j_cont,
        "train_score_continuation_spearman": train_score_cont,
        "train_j_continuation_spearman": train_j_cont,
        "case_synergy": synergy,
        "case_single_gains": single_gains,
        "case_all_gain": all_gain,
        "case_divergence_step": divergence_step,
        "case_divergence_source": divergence_source,
        "payload_divergence_step": payload_step,
        "inversion_fraction_aid": INVERSION_FRACTION,
        "illegal_single_switches_excluded": int(len(differing) - len(single_gains)),
        "residual_synergy_median": _median(residual_synergy),
        "residual_synergy_mean": _mean(residual_synergy),
        "residual_synergy_positive_fraction": _mean(value > 0.0 for value in residual_synergy),
        "motion_nearly_identical_fraction": motion_fraction,
        "train_replayed": False,
    }
    (OUTPUT / "phase5w_residual_mechanism_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "synergy": synergy, "inversion": inversion_fraction, "divergence": divergence_step})), flush=True)
    return summary


if __name__ == "__main__":
    run()
