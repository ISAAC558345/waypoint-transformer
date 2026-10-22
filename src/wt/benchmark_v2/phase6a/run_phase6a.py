"""Benchmark-V2 development audit. It does not train or open the reserved pool."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import _load_response_norm, _load_value
from wt.lowlevel_v2.phase5zp_one_shot_fresh_confirmation.protocol import VALUE_SEEDS

from .protocol import (
    CONSERVATION_ABS,
    DEPLOYMENT_VERIFIED,
    DEVELOPMENT_POOL,
    DISTINGUISH_MIN,
    HISTORICAL_CONSUMED,
    PAIRWISE_TIE_MAX,
    PHASE,
    PHASE5ZP_GATE,
    PHASE5ZQ_GATE,
    PLANNER_READY,
    REPORT,
    RESERVED_POOL,
    SATURATION_MAX,
    SWITCH_NEGATIVE_SPEARMAN,
    choose_gate,
)
from .rollout import run_controlled_episode

CONTROLLERS = (
    "frozen_phase5",
    "model_s",
    "fly_toward",
    "queue_rate_heuristic",
    "random_legal",
    "instant_rate_diagnostic",
)
NONE_EQUIVALENT_PAIRS = (("frozen_phase5", "random_legal"), ("fly_toward", "instant_rate_diagnostic"), ("frozen_phase5", "model_s"))


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
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


def _std(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.std(array)) if array.size else float("nan")


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), ".6g")


def _spearman(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _saturation(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    rounded = np.round(array, decimals=6)
    _unique, counts = np.unique(rounded, return_counts=True)
    return float(np.max(counts) / rounded.size)


def _iqr(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, 75) - np.percentile(array, 25))


def _load_bundle() -> dict:
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    actor = load_frozen_actor()
    return {
        "config": L3BConfig(),
        "frozen": _load_proposed(),
        "matching": _load_matching(),
        "actor": actor,
        "norm": _load_response_norm(),
        "center": np.asarray(temporal["center"], dtype=np.float32),
        "scale": np.asarray(temporal["scale"], dtype=np.float32),
        "response_models": {int(seed): _load_value("response", int(seed)) for seed in VALUE_SEEDS},
        "static_models": {int(seed): _load_value("static", int(seed)) for seed in VALUE_SEEDS},
    }


def _average_model_s(rows: list[dict]) -> dict:
    base = dict(rows[0])
    numeric = [
        "initial_blocks",
        "arrival_blocks",
        "completed_blocks",
        "dropped_blocks",
        "residual_blocks",
        "e0_payload_bits",
        "e2_blocks",
        "rate_integral_bits",
        "mean_sinr",
        "mean_rate",
        "in_service_steps",
        "switch_count",
        "abandoned_progress",
        "median_streak",
        "service_work",
        "fairness_jain_completed_blocks",
    ]
    for key in numeric:
        base[key] = _mean(row[key] for row in rows)
    base["static_seed"] = "mean_of_five"
    base["counterfactuals"] = []
    return base


def _range_row(controller: str, rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    mean = _mean(values)
    std = _std(values)
    return {
        "controller": controller,
        "endpoint": key,
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean else float("nan"),
        "unique_value_count": int(len({round(value, 6) for value in values})),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "iqr": _iqr(values),
        "saturation_ratio": _saturation(values),
    }


def _write_report(facts: dict) -> None:
    lines = [
        "# HTW Phase-6A：Benchmark-V2 通信效用与服务连续性审计",
        "",
        f"Gate 是 {facts['gate']}（{facts['gate_name']}）。",
        "",
        "Phase-5ZP 仍然是 CONF-C。Phase-5ZQ 仍然是 ATTR-C。310118–310125 仍然是已消耗确认数据。320100–320131 没有打开。本阶段没有训练。`planner_ready = False`，`deployment_verified = False`。",
        "",
        "## Q1. 旧 payload 是 throughput 还是 task completion？",
        "",
        "是 task completion。队列以服务块计数。rate 只把 UAV 自己的 progress 计时器往前推，`(rate / reference_rate) / service_slots`。progress 到达 1，payload 才增加 1e6 bit，队列才减 1 块。",
        "",
        "## Q2. 未完成的已传输数据为什么被记为 0？",
        "",
        "因为环境没有把 rate 积分写进队列或 payload。没到 1 的 progress 只存在于当前 UAV。目标一变，这个 progress 被清零，已经按 rate 算过的那一段时间不留下比特。",
        "",
        "## Q3. continuous delivered bits 能否从现有环境严格计算？",
        "",
        "不能。没有 bit 队列。rate 的单位是 bps，队列的单位是 block。把 block 乘上 1e6 再去和 rate 做 min，是额外解释，不是状态转移。本阶段只把 `sum(rate) * delta_t` 记成假设积分，不把它当成环境交付量。",
        "",
        "## Q4. continuous delivered bits 和 queue reduction 是否守恒？",
        "",
        f"队列守恒成立：E2 块数减完成块数的最大绝对误差是 {_fmt(facts['max_conservation_error'])}。假设 rate 积分相对 E0 的平均差是 {_fmt(facts['rate_gap'])} bit。两者不是同一个量，所以不能互相守恒。",
        "",
        "## Q5. 旧 completed-block payload 的 saturation ratio 是多少？",
        "",
        facts["saturation_text"],
        "",
        "## Q6. 新 endpoint 是否能区分不同策略？",
        "",
        f"连续比特 endpoint 不能采用。旧 E0 上，预先指定的非等价控制器对里，最高可区分 scenario 比例是 {_fmt(facts['best_distinguish'])}。预先登记的门槛是 {DISTINGUISH_MIN:.2f}，打平比例门槛是 {PAIRWISE_TIE_MAX:.2f}。旧指标的平均打平比例是 {_fmt(facts['mean_tie'])}。",
        "",
        "## Q7. 新 development data 上是否重新出现 higher SINR/rate 但长期收益不变？",
        "",
        f"冻结 Phase-5 控制器平均 SINR {_fmt(facts['frozen_sinr'])}，Model-S {_fmt(facts['models_sinr'])}。平均 rate {_fmt(facts['frozen_rate'])} 对 {_fmt(facts['models_rate'])}。平均完成块 {_fmt(facts['frozen_blocks'])} 对 {_fmt(facts['models_blocks'])}。平均假设 rate 积分 {_fmt(facts['frozen_integral'])} 对 {_fmt(facts['models_integral'])}。",
        "",
        "## Q8. target switching 是否和长期通信效用负相关？",
        "",
        f"冻结控制器在 64 个 scenario 上，切换次数对 E0 的 Spearman 是 {_fmt(facts['spearman_switch_e0'])}，对假设 rate 积分的 Spearman 是 {_fmt(facts['spearman_switch_rate'])}。服务段长度对 E0 的 Spearman 是 {_fmt(facts['spearman_streak_e0'])}，对假设积分是 {_fmt(facts['spearman_streak_rate'])}。负相关的预先门槛是 Spearman <= {SWITCH_NEGATIVE_SPEARMAN:.2f}。跨控制器均值的切换对 E0 Spearman 是 {_fmt(facts['spearman_across_controllers'])}。",
        "",
        "## Q9. service progress 应该属于 UAV、buoy，还是 bit queue？",
        "",
        "按现在的代码，progress 属于 UAV，并且只在连续服务同一个目标时保留。这表示一段连续驻留计时，不是 buoy 上已经交付的比特，也不是可以交给下一架 UAV 的队列。若目标是最大化信息传输，物理上已经发出的比特应留在队列扣减里，不应因换 UAV 而消失。本阶段只把 P0、P1、P2 列为设计候选，不改规则。",
        "",
        "## Q10. hold=15 是否仍然是问题？",
        "",
        f"离线保持同一 assignment 时，horizon 1、3、5、10、15 的平均完成块是 {facts['hold_completed']}。平均假设 rate 积分是 {facts['hold_rate']}。这只说明更长的保持会累计更多计时，不是在线修改 hold 的依据。hold 仍是 15，没有被调整。",
        "",
        "## Q11. 真正问题是 metric saturation、continuity loss，还是两者都有？",
        "",
        f"Gate 是 {facts['gate']}。bit-level throughput 目前不是环境里的守恒量，所以不能先宣布 continuity-aware 目标已经证实。旧 block payload 在冻结控制器上的 saturation ratio 是 {_fmt(facts['frozen_saturation'])}。冻结控制器相对 Model-S 的切换差是 {_fmt(facts['switch_gap'])}，完成块差是 {_fmt(facts['block_gap'])}。",
        "",
        "## Q12. 是否有充分依据进入 continuity-aware high-level objective？",
        "",
        facts["q12"],
        "",
        "## Q13. 是否需要修改低层 MAPPO？",
        "",
        "NO。低层只执行给定 TaskOption。完成门槛和 progress 清零都发生在环境与高层换目标，不是低层重新选 buoy。",
        "",
        "## Q14. 是否需要增加中间评价层？",
        "",
        "NO。问题在主指标的物理含义，以及高层是否把未完成服务清掉。不需要再插一层评价器。",
        "",
        "## Q15. 是否需要继续增加 HGNN 复杂度？",
        "",
        "NO。HGNN 仍然只产生 Stage-1 候选。增加深度不能把 block 计数变成比特吞吐。",
        "",
        "## 设计候选",
        "",
        "P0 是现在的 UAV-owned reset，表示连续驻留计时。P1 是 buoy-owned progress，表示同一任务的计时可以换 UAV 接着走，但仍然不是比特。P2 是比特队列，rate 直接扣减队列，不再用人工 block progress。论文如果要以信息传输为主问题，P2 才和目标同单位。lambda 没有选择，状态是 HUMAN_OR_NEXT_PHASE_SELECTION_REQUIRED。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    config = L3BConfig()
    _dump(
        "phase6a_protocol.json",
        {
            "phase5zp_gate": PHASE5ZP_GATE,
            "phase5zq_gate": PHASE5ZQ_GATE,
            "development_pool": "320000-320063",
            "reserved_pool": "320100-320131",
            "historical_consumed": "310100-310125",
            "training": False,
            "lambda_status": "HUMAN_OR_NEXT_PHASE_SELECTION_REQUIRED",
            "saturation_max": SATURATION_MAX,
            "pairwise_tie_max": PAIRWISE_TIE_MAX,
            "distinguish_min": DISTINGUISH_MIN,
            "switch_negative_spearman": SWITCH_NEGATIVE_SPEARMAN,
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
            "candidate_mechanism_not_yet_a_conclusion": [
                "better PHY",
                "higher rate",
                "more service work",
                "excessive reassignment",
                "progress reset",
                "no additional completed block",
                "identical payload",
            ],
        },
    )
    _dump(
        "phase6a_data_lineage.json",
        {
            "development": {
                "ids": "320000-320063",
                "role": "BENCHMARK_V2_DEVELOPMENT",
                "allowed_uses": ["benchmark_development", "mechanism_analysis"],
            },
            "reserved": {
                "ids": "320100-320131",
                "role": "BENCHMARK_V2_FRESH_CONFIRMATION_RESERVED",
                "opened_in_phase6a": False,
                "untouched": True,
            },
            "historical_consumed": {
                "ids": "310100-310125",
                "opened_in_phase6a": False,
                "forbidden_uses": ["training", "tuning", "benchmark_redesign", "feature_design", "loss_design", "threshold_selection", "confirmation"],
            },
        },
    )
    _dump(
        "phase6a_current_payload_semantics.json",
        {
            "queue_unit": "blocks",
            "rate_unit": "bits per second",
            "delta_t_s": float(config.delta_t),
            "progress_owner": "uav",
            "progress_increment": "(rate / reference_rate) / service_slots, capped at 1",
            "block_completion": "uav progress >= 1 - 1e-9",
            "payload_increment": "block_payload_bits, currently 1e6, only on completion",
            "queue_decrement": "one block on the served buoy at completion",
            "drop": "max(0, queue - completed_service + arrivals - capacity)",
            "partial_transmission_stored": False,
            "classification": "task_completion_metric",
            "matches_information_transmission": False,
        },
    )
    _dump(
        "phase6a_progress_semantics_audit.json",
        {
            "current_behavior": "progress is stored per UAV and reset to 0 when that UAV changes target",
            "half_served_buoy_after_switch": "the progress timer is discarded; the queue block is not decremented",
            "second_uav_continues_same_buoy": "the new UAV starts progress at 0; earlier timer is not transferred",
            "meaning": "consecutive dwell timer for one UAV, not a packet-bit transfer and not a handshake",
            "candidates": {
                "P0": "current UAV-owned reset timer",
                "P1": "buoy-owned persistent timer that another UAV can continue",
                "P2": "bit queue decremented by the physical rate, no artificial block timer",
            },
            "rule_changed_in_this_phase": False,
        },
    )
    _dump(
        "phase6a_continuity_objective_design.json",
        {
            "formula": "U(A_t) = V_exec(A_t) - lambda_switch * C_switch(A_t, A_t_minus_1) - lambda_progress * C_abandon(A_t)",
            "v_exec": "executor-conditioned value already available at the decision",
            "c_switch": "count of UAV targets that differ from the current assignment; observable now",
            "c_abandon": "sum of current UAV progress on targets that would be left; observable now",
            "uses_future_information": False,
            "label_leakage": False,
            "possible_double_count": "V_exec already sees one-step progress features, so C_abandon can count progress twice",
            "normalization": "C_switch / num_uavs and C_abandon / num_uavs; progress is already on [0, 1)",
            "lambda_status": "HUMAN_OR_NEXT_PHASE_SELECTION_REQUIRED",
            "distance_penalty_added": False,
            "trained": False,
        },
    )
    bundle = _load_bundle()
    per_seed = []
    aggregated = []
    counterfactuals = []
    for scenario in DEVELOPMENT_POOL:
        print(f"development {int(scenario)}", flush=True)
        frozen = run_controlled_episode(bundle, int(scenario), "frozen_phase5")
        per_seed.append(frozen)
        aggregated.append(frozen)
        counterfactuals.extend(frozen["counterfactuals"])
        static_rows = [run_controlled_episode(bundle, int(scenario), "model_s", static_seed=int(seed)) for seed in VALUE_SEEDS]
        per_seed.extend(static_rows)
        aggregated.append(_average_model_s(static_rows))
        for name in ("fly_toward", "queue_rate_heuristic", "random_legal", "instant_rate_diagnostic"):
            episode = run_controlled_episode(bundle, int(scenario), name)
            per_seed.append(episode)
            aggregated.append(episode)
    result_rows = []
    for row in aggregated:
        result_rows.append({key: value for key, value in row.items() if key != "counterfactuals"})
    _write_csv("phase6a_reference_controller_results.csv", result_rows)
    range_rows = []
    for controller in CONTROLLERS:
        rows = [row for row in aggregated if row["controller"] == controller]
        for key in ("e0_payload_bits", "e2_blocks", "rate_integral_bits"):
            range_rows.append(_range_row(controller, rows, key))
    _write_csv("phase6a_endpoint_dynamic_range.csv", range_rows)
    continuity_rows = []
    for row in aggregated:
        if row["controller"] in {"frozen_phase5", "model_s"}:
            continuity_rows.append(
                {
                    "scenario_id": int(row["scenario_id"]),
                    "controller": row["controller"],
                    "switch_count": row["switch_count"],
                    "median_streak": row["median_streak"],
                    "abandoned_progress": row["abandoned_progress"],
                    "service_work": row["service_work"],
                    "rate_integral_bits": row["rate_integral_bits"],
                    "completed_blocks": row["completed_blocks"],
                    "dropped_blocks": row["dropped_blocks"],
                    "residual_blocks": row["residual_blocks"],
                    "mean_sinr": row["mean_sinr"],
                    "mean_rate": row["mean_rate"],
                }
            )
    _write_csv("phase6a_continuity_metrics.csv", continuity_rows)
    frozen_rows = [row for row in aggregated if row["controller"] == "frozen_phase5"]
    switch_rows = [
        {
            "scenario_id": int(row["scenario_id"]),
            "controller": "frozen_phase5",
            "switch_count": row["switch_count"],
            "median_streak": row["median_streak"],
            "e0_payload_bits": row["e0_payload_bits"],
            "completed_blocks": row["completed_blocks"],
            "rate_integral_bits": row["rate_integral_bits"],
        }
        for row in frozen_rows
    ]
    _write_csv("phase6a_switch_vs_utility.csv", switch_rows)
    _write_csv("phase6a_hold_counterfactual.csv", counterfactuals)
    max_error = max(abs(float(row["e2_blocks"]) - float(row["completed_blocks"])) for row in aggregated)
    rate_gap = _mean(float(row["rate_integral_bits"]) - float(row["e0_payload_bits"]) for row in aggregated)
    by_controller = {name: [row for row in aggregated if row["controller"] == name] for name in CONTROLLERS}
    ties = []
    distinguish = {}
    for left_name, right_name in NONE_EQUIVALENT_PAIRS:
        same = 0
        for left, right in zip(by_controller[left_name], by_controller[right_name]):
            same += int(abs(float(left["e0_payload_bits"]) - float(right["e0_payload_bits"])) <= 1.0e-6)
        fraction_same = same / max(len(by_controller[left_name]), 1)
        ties.append(fraction_same)
        distinguish[f"{left_name}_vs_{right_name}"] = 1.0 - fraction_same
    frozen_saturation = _saturation(row["e0_payload_bits"] for row in frozen_rows)
    model_rows = [row for row in aggregated if row["controller"] == "model_s"]
    spearman_switch_e0 = _spearman([row["switch_count"] for row in frozen_rows], [row["e0_payload_bits"] for row in frozen_rows])
    spearman_switch_rate = _spearman([row["switch_count"] for row in frozen_rows], [row["rate_integral_bits"] for row in frozen_rows])
    spearman_streak_e0 = _spearman([row["median_streak"] for row in frozen_rows], [row["e0_payload_bits"] for row in frozen_rows])
    spearman_streak_rate = _spearman([row["median_streak"] for row in frozen_rows], [row["rate_integral_bits"] for row in frozen_rows])
    controller_switch = [_mean(row["switch_count"] for row in by_controller[name]) for name in CONTROLLERS]
    controller_e0 = [_mean(row["e0_payload_bits"] for row in by_controller[name]) for name in CONTROLLERS]
    frozen_blocks = _mean(row["completed_blocks"] for row in frozen_rows)
    model_blocks = _mean(row["completed_blocks"] for row in model_rows)
    frozen_switches = _mean(row["switch_count"] for row in frozen_rows)
    model_switches = _mean(row["switch_count"] for row in model_rows)
    continuity_pattern = bool(
        _mean(row["mean_sinr"] for row in frozen_rows) > _mean(row["mean_sinr"] for row in model_rows)
        and frozen_switches > model_switches
        and frozen_blocks <= model_blocks + 1.0e-6
    )
    gate = choose_gate(
        bit_queue_exists=False,
        e1_matches_environment_transition=False,
        e1_e2_conserved=False,
        e0_saturated=bool(frozen_saturation >= SATURATION_MAX),
        endpoint_separates=bool(max(distinguish.values()) >= DISTINGUISH_MIN),
        switch_negatively_related=bool(np.isfinite(spearman_switch_e0) and spearman_switch_e0 <= SWITCH_NEGATIVE_SPEARMAN),
        continuity_pattern=continuity_pattern,
    )
    hold_completed = {}
    hold_rate = {}
    for horizon in (1, 3, 5, 10, 15):
        subset = [row for row in counterfactuals if int(row["horizon"]) == horizon]
        hold_completed[str(horizon)] = _mean(row["completed_blocks"] for row in subset)
        hold_rate[str(horizon)] = _mean(row["rate_integral_bits"] for row in subset)
    saturation_text = "；".join(
        f"{name} 的 E0 saturation ratio 是 {_fmt(_saturation(row['e0_payload_bits'] for row in by_controller[name]))}"
        for name in CONTROLLERS
    )
    facts = {
        "gate": gate["gate"],
        "gate_name": gate["gate_name"],
        "max_conservation_error": max_error,
        "rate_gap": rate_gap,
        "saturation_text": saturation_text,
        "best_distinguish": max(distinguish.values()),
        "mean_tie": _mean(ties),
        "frozen_sinr": _mean(row["mean_sinr"] for row in frozen_rows),
        "models_sinr": _mean(row["mean_sinr"] for row in model_rows),
        "frozen_rate": _mean(row["mean_rate"] for row in frozen_rows),
        "models_rate": _mean(row["mean_rate"] for row in model_rows),
        "frozen_blocks": frozen_blocks,
        "models_blocks": model_blocks,
        "frozen_integral": _mean(row["rate_integral_bits"] for row in frozen_rows),
        "models_integral": _mean(row["rate_integral_bits"] for row in model_rows),
        "spearman_switch_e0": spearman_switch_e0,
        "spearman_switch_rate": spearman_switch_rate,
        "spearman_streak_e0": spearman_streak_e0,
        "spearman_streak_rate": spearman_streak_rate,
        "spearman_across_controllers": _spearman(controller_switch, controller_e0),
        "hold_completed": hold_completed,
        "hold_rate": hold_rate,
        "frozen_saturation": frozen_saturation,
        "switch_gap": float(frozen_switches - model_switches),
        "block_gap": float(frozen_blocks - model_blocks),
        "q12": "NO。连续比特还不是环境里的守恒量，所以不能把 continuity-aware 高层目标当成已经证实的下一阶段。先要有可信的比特记账。lambda 仍然是 HUMAN_OR_NEXT_PHASE_SELECTION_REQUIRED。"
        if gate["gate"] == "BV2-E"
        else "见 Gate。lambda 仍然没有选择。",
    }
    _dump(
        "phase6a_endpoint_candidates.json",
        {
            "E0": "1e6 times completed blocks; this is the environment payload",
            "E1": "not computable as an environment transition; rate integral is diagnostic only",
            "E2": "initial blocks + arrivals - dropped - residual; equals completed blocks when conservation holds",
            "e1_matches_e2": False,
            "recommended_primary_metric": "NOT_ADOPTED",
            "reason": "no bit queue exists, so continuous delivered bits would be an invented state",
            "phase5zp_unchanged": PHASE5ZP_GATE,
        },
    )
    gate_payload = dict(gate)
    gate_payload.update(
        {
            "bit_queue_exists": False,
            "e0_e2_max_abs_error": max_error,
            "mean_rate_integral_minus_e0": rate_gap,
            "frozen_e0_saturation_ratio": frozen_saturation,
            "distinguish_fraction": distinguish,
            "mean_pairwise_tie_fraction_on_e0": _mean(ties),
            "continuity_pattern_on_e0": continuity_pattern,
            "spearman_switch_vs_e0": spearman_switch_e0,
            "opened_reserved_pool": False,
            "opened_historical_consumed": False,
            "deployment_pass": False,
            "lambda_status": "HUMAN_OR_NEXT_PHASE_SELECTION_REQUIRED",
        }
    )
    _dump("phase6a_benchmark_v2_gate.json", gate_payload)
    _write_report(facts)
    print({"gate": gate["gate"], "conservation": max_error, "frozen_saturation": frozen_saturation, "continuity": continuity_pattern}, flush=True)
    _ = (RESERVED_POOL, HISTORICAL_CONSUMED, per_seed)


if __name__ == "__main__":
    main()
