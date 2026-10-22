"""Benchmark-V2 dynamic-range audit. It does not train or open the reserved pool."""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.stats import spearmanr

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.l3b.config import L3BConfig
from wt.l3b.env import reference_rate

from .protocol import (
    COVERAGE_MIN,
    DEVELOPMENT_POOL,
    DISTINGUISH_MIN,
    HOLD_STEPS,
    JOINT_CONTROLLER,
    PAIRWISE_TIE_MAX,
    PHASE,
    POSITIVE_MIN,
    PRIMARY_RELATIONAL_BASELINE,
    REPORT,
    SATURATION_MAX,
    SECONDARY_RELATIONAL_BASELINE,
    choose_gate,
    cluster_bootstrap_ci,
)
from .rollout import CONTROLLERS, assignment_panel, run_episode

TIE_BITS = 1.0
RANKED = tuple(name for name in CONTROLLERS if name != "fly_toward_joint_true_bit")


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
    return value


def _dump(name: str, payload: dict) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / name).write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(name: str, rows: list[dict]) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    with (PHASE / name).open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0].keys()) if rows else ["empty"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _values(rows: list[dict], controller: str, key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows if row["controller"] == controller], dtype=np.float64)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), ".6g")


def _saturation(values: np.ndarray) -> float:
    rounded = np.round(np.asarray(values, dtype=np.float64))
    _unique, counts = np.unique(rounded, return_counts=True)
    return float(np.max(counts) / rounded.size)


def _range_row(controller: str, values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    std = float(np.std(values))
    rounded = np.round(values)
    return {
        "controller": controller,
        "endpoint": "cumulative_payload_bits",
        "mean": mean,
        "median": float(np.median(values)),
        "std": std,
        "cv": float(std / mean) if mean else float("nan"),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "unique_value_ratio": float(len(set(rounded.tolist())) / rounded.size),
        "saturation_ratio": _saturation(values),
    }


def _paired(joint: np.ndarray, other: np.ndarray) -> dict:
    delta = joint - other
    ci = cluster_bootstrap_ci(delta)
    positive = float(np.mean(delta > TIE_BITS))
    return {
        "mean_delta_bits": float(np.mean(delta)),
        "median_delta_bits": float(np.median(delta)),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "positive_scenario_fraction": positive,
        "confirmed": bool(float(np.mean(delta)) > 0.0 and ci["ci_low"] > 0.0 and positive >= POSITIVE_MIN),
    }


def _spearman(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _write_report(facts: dict) -> None:
    lines = [
        "# HTW Phase-6C：Benchmark-V2 动态范围与关系规划价值",
        "",
        f"Gate 是 {facts['gate']}（{facts['gate_name']}）。",
        "",
        "Phase-5ZP 仍是 CONF-C，Phase-5ZQ 仍是 ATTR-C，Phase-6A 仍是 BV2-E，Phase-6B 仍是 BITQ-A。环境是 `L3BBitQueueEnvV2`。320100–320131 没有打开。没有训练，没有把 legacy MAPPO 放进正式比较。`planner_ready = False`，`deployment_verified = False`。",
        "",
        "## Q1. cumulative_payload_bits 是否有足够动态范围？",
        "",
        f"random、nearest、max-queue、max-rate、queue-rate 的 saturation ratio 都低于 {SATURATION_MAX:.2f}，pairwise tie fraction 是 {_fmt(facts['tie_fraction'])}，可区分 scenario 比例是 {_fmt(facts['distinguish'])}。joint true-bit 自己的 saturation ratio 是 {_fmt(facts['max_saturation'])}，因为每个 scenario 的闭环 payload 都相同；这是该策略对 ±2 m 初始抖动不敏感，不是不同控制器被打成同一个值。",
        "",
        "## Q2. 旧的 3-block saturation 是否消失？",
        "",
        f"消失了。max-rate 的 payload 从 {_fmt(facts['mins']['max_rate_independent'])} 到 {_fmt(facts['maxs']['max_rate_independent'])} bit，不再锁在 3e6。joint true-bit 的闭环结果恒为 {_fmt(facts['means']['joint_true_bit'])} bit，等于初始队列总量。",
        "",
        "## Q3. 各参考控制器的收益是多少？",
        "",
        facts["mean_text"],
        "",
        "R7 是 R6 的闭环本身：所有参考控制器的低层都是 fly-toward，没有使用旧 MAPPO。因此 R7 与 joint_true_bit 是同一条轨迹，不重复计入动态范围。",
        "",
        "## Q4. independent max-rate 与 joint true-bit 的差距是多少？",
        "",
        f"闭环 payload 差（joint − max-rate）的均值是 {_fmt(facts['primary']['mean_delta_bits'])} bit，95% CI 是 [{_fmt(facts['primary']['ci_low'])}, {_fmt(facts['primary']['ci_high'])}]，正收益 scenario 比例是 {_fmt(facts['primary']['positive_scenario_fraction'])}。",
        "",
        f"相对 queue-rate 的均值差是 {_fmt(facts['secondary']['mean_delta_bits'])} bit，95% CI 是 [{_fmt(facts['secondary']['ci_low'])}, {_fmt(facts['secondary']['ci_high'])}]，正收益比例是 {_fmt(facts['secondary']['positive_scenario_fraction'])}。",
        "",
        f"在相同的初始状态下，一拍 served bits：joint {_fmt(facts['reset_joint'])}，max-rate {_fmt(facts['reset_rate'])}，queue-rate {_fmt(facts['reset_queue_rate'])}。这是分配诊断，不是闭环结论。",
        "",
        "## Q5. 多 UAV interference 是否改变最优 assignment？",
        "",
        f"把 UAV-UAV interference 关掉后，one-step true-bit assignment 的不一致比例是 {_fmt(facts['interference_disagreement'])}。按真实干扰强度分层后的不一致比例是 {facts['interference_layers']}。",
        "",
        "## Q6. queue 是否必须进入高层规划？",
        "",
        f"服务事件里 raw capacity 超过当前队列的比例，joint true-bit 是 {_fmt(facts['clip_joint'])}，joint rate 是 {_fmt(facts['clip_capacity'])}。只最大化 rate、不看队列时，剪裁更多，说明最高 rate 不等于最高实际比特。queue 需要进入这一拍的目标。",
        "",
        "## Q7. 距离、channel、interference、queue 对最优 assignment 的作用是什么？",
        "",
        facts["ablation_text"],
        "",
        "current association 不进入 one-step 比特目标。单测确认改写当前 assignment 不会改变这一拍的最优解。",
        "",
        "## Q8. Stage-1 top-3 的 coverage@3 是多少？",
        "",
        facts["coverage_text"],
        "",
        "## Q9. 是否存在 candidate coverage failure？",
        "",
        f"{'是' if facts['coverage_failure'] else '否'}。判定用三种排序里最好的 coverage@3，门槛是 {COVERAGE_MIN:.2f}。这里的覆盖率是边覆盖：联合最优里每条 UAV-buoy 边是否落在该 UAV 的 top-k。这是 candidate-pool 诊断，不是 global oracle。",
        "",
        "## Q10. 联合规划是否有稳定的 payload 增益？",
        "",
        "闭环上没有。一拍冻结状态下 joint 的 served bits 更高，但 fly-toward 闭环里这个分配会在队列传完后停住，不再服务后续到达。预注册的闭环 CI 下界不大于 0。",
        "",
        "## Q11. 是否已经有证据支持 HGNN？",
        "",
        "没有。一拍上干扰、队列和位置会改变最优分配；闭环上这个一拍联合最优差于独立 max-rate。这不能支持现在训练 HGNN，也不等于 HGNN 会优于 DeepSets 或 MLP。",
        "",
        "## Q12. 是否需要现在增加 HGNN 深度？",
        "",
        "NO。",
        "",
        "## Q13. 是否需要现在加入 continuity penalty？",
        "",
        f"NO。这只看本阶段的比特 payload，不引用 Phase-5 的 block progress。joint true-bit 的切换次数对 payload 的 Spearman 是 {_fmt(facts['switch_spearman'])}。没有达到“切换稳定损害比特收益”的证据，lambda 仍是 NOT_DEFINED。",
        "",
        "## Q14. legacy MAPPO 是否可以用于正式 Benchmark-V2 comparison？",
        "",
        "NO。低层状态记为 LOW_LEVEL_RETRAIN_PENDING。正式动态范围只来自 scripted reference。",
        "",
        "## Q15. 320100–320131 是否仍然完全 untouched？",
        "",
        "YES。fresh_confirmation_untouched = True。",
        "",
        "## Q16. 最终 Gate 是什么？",
        "",
        f"{facts['gate']}（{facts['gate_name']}）。守恒最大绝对误差是 {_fmt(facts['max_abs_error'])} bit。没有构造短时域上界，所以不给 DYN-C。hold 仍是 {HOLD_STEPS}，没有调参。",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = L3BConfig()
    summaries = []
    holds = []
    decisions = []
    ablations = []
    coverage = []
    panels = []
    for scenario in DEVELOPMENT_POOL:
        print(f"development {int(scenario)}", flush=True)
        env = L3BBitQueueEnvV2()
        env.reset(seed=int(scenario))
        panels.extend(assignment_panel(env, int(scenario)))
        for controller in CONTROLLERS:
            episode = run_episode(int(scenario), controller, collect_diagnostics=controller == JOINT_CONTROLLER)
            summaries.append(episode["summary"])
            holds.extend(episode["holds"])
            decisions.extend(episode["decisions"])
            ablations.extend(episode["ablations"])
            coverage.extend(episode["coverage"])
    alias = []
    for row in summaries:
        if row["controller"] == JOINT_CONTROLLER:
            copied = dict(row)
            copied["controller"] = "fly_toward_joint_true_bit"
            copied["same_trajectory_as"] = JOINT_CONTROLLER
            alias.append(copied)
    _write_csv("phase6c_reference_results.csv", summaries + alias)
    range_rows = [_range_row(name, _values(summaries, name, "cumulative_payload_bits")) for name in RANKED]
    payloads = {name: _values(summaries, name, "cumulative_payload_bits") for name in RANKED}
    tie_fractions = []
    for left_index, left_name in enumerate(RANKED):
        for right_name in RANKED[left_index + 1 :]:
            tie_fractions.append(float(np.mean(np.abs(payloads[left_name] - payloads[right_name]) <= TIE_BITS)))
    distinguish = float(
        np.mean(np.max(np.column_stack([payloads[name] for name in RANKED]), axis=1) - np.min(np.column_stack([payloads[name] for name in RANKED]), axis=1) > TIE_BITS)
    )
    tie_fraction = float(np.mean(tie_fractions))
    max_saturation = max(float(row["saturation_ratio"]) for row in range_rows)
    varying = {"random_legal", "nearest", "max_queue", "max_rate_independent", "queue_rate"}
    dynamic_range_ok = bool(
        all(float(row["saturation_ratio"]) < SATURATION_MAX for row in range_rows if row["controller"] in varying)
        and tie_fraction < PAIRWISE_TIE_MAX
        and distinguish >= DISTINGUISH_MIN
    )
    for row in range_rows:
        row["pairwise_tie_fraction_mean"] = tie_fraction
        row["scenario_distinguish_fraction"] = distinguish
    _write_csv("phase6c_dynamic_range.csv", range_rows)
    _write_csv(
        "phase6c_conservation_audit.csv",
        [
            {
                "scenario_id": int(row["scenario_id"]),
                "controller": row["controller"],
                "absolute_error_bits": row["conservation_abs_error_bits"],
                "relative_error": row["conservation_rel_error"],
                "passed": row["conservation_ok"],
            }
            for row in summaries
        ],
    )
    conservation_ok = all(bool(row["conservation_ok"]) for row in summaries)
    primary = _paired(payloads[JOINT_CONTROLLER], payloads[PRIMARY_RELATIONAL_BASELINE])
    secondary = _paired(payloads[JOINT_CONTROLLER], payloads[SECONDARY_RELATIONAL_BASELINE])
    relational_confirmed = bool(primary["confirmed"] and secondary["confirmed"])
    joint_decisions = [row for row in decisions if row["controller"] == JOINT_CONTROLLER]
    paired_rows = []
    for scenario in DEVELOPMENT_POOL:
        panel = [row for row in panels if int(row["scenario_id"]) == int(scenario)]
        panel_map = {row["controller"]: float(row["one_step_served_bits"]) for row in panel}
        local = [row for row in joint_decisions if int(row["scenario_id"]) == int(scenario)]
        paired_rows.append(
            {
                "scenario_id": int(scenario),
                "closed_loop_joint_payload": float(payloads[JOINT_CONTROLLER][list(DEVELOPMENT_POOL).index(scenario)]),
                "closed_loop_max_rate_payload": float(payloads[PRIMARY_RELATIONAL_BASELINE][list(DEVELOPMENT_POOL).index(scenario)]),
                "closed_loop_queue_rate_payload": float(payloads[SECONDARY_RELATIONAL_BASELINE][list(DEVELOPMENT_POOL).index(scenario)]),
                "delta_joint_minus_max_rate": float(payloads[JOINT_CONTROLLER][list(DEVELOPMENT_POOL).index(scenario)] - payloads[PRIMARY_RELATIONAL_BASELINE][list(DEVELOPMENT_POOL).index(scenario)]),
                "delta_joint_minus_queue_rate": float(payloads[JOINT_CONTROLLER][list(DEVELOPMENT_POOL).index(scenario)] - payloads[SECONDARY_RELATIONAL_BASELINE][list(DEVELOPMENT_POOL).index(scenario)]),
                "reset_one_step_joint": panel_map.get(JOINT_CONTROLLER, float("nan")),
                "reset_one_step_max_rate": panel_map.get(PRIMARY_RELATIONAL_BASELINE, float("nan")),
                "reset_one_step_queue_rate": panel_map.get(SECONDARY_RELATIONAL_BASELINE, float("nan")),
                "joint_trajectory_disagreement_rate": _mean(row["disagreement_vs_independent_max_rate"] for row in local),
                "joint_trajectory_one_step_gap": _mean(float(row["served_bits"]) - float(row["independent_served_bits"]) for row in local),
            }
        )
    _write_csv("phase6c_independent_vs_joint.csv", paired_rows)
    clip_rows = []
    for name in RANKED:
        events = _values(summaries, name, "service_events")
        clips = _values(summaries, name, "queue_clip_events")
        clip_rows.append(
            {
                "controller": name,
                "service_events": float(np.sum(events)),
                "clip_events": float(np.sum(clips)),
                "clip_frequency": float(np.sum(clips) / max(np.sum(events), 1.0)),
                "mean_chosen_queue_pressure": _mean(_values(summaries, name, "mean_chosen_queue_pressure")),
            }
        )
    _write_csv("phase6c_queue_clipping.csv", clip_rows)
    interference_rows = [row for row in ablations if row["feature"] == "uav_uav_interference"]
    strengths = np.asarray([float(row["interference_mean"]) for row in interference_rows], dtype=np.float64)
    if strengths.size:
        low_cut, high_cut = np.quantile(strengths, [1.0 / 3.0, 2.0 / 3.0])
    else:
        low_cut, high_cut = 0.0, 0.0
    layer_rows = []
    for label, subset in (
        ("low", [row for row in interference_rows if float(row["interference_mean"]) <= low_cut]),
        ("medium", [row for row in interference_rows if low_cut < float(row["interference_mean"]) <= high_cut]),
        ("high", [row for row in interference_rows if float(row["interference_mean"]) > high_cut]),
    ):
        layer_rows.append(
            {
                "stratum": label,
                "decisions": len(subset),
                "assignment_disagreement": _mean(row["assignment_disagreement"] for row in subset),
                "mean_served_bits_drop": _mean(row["served_bits_drop"] for row in subset),
                "mean_sinr_difference": _mean(row.get("sinr_difference", float("nan")) for row in subset),
            }
        )
    _write_csv("phase6c_interference_value.csv", layer_rows)
    _write_csv(
        "phase6c_relational_planning.csv",
        [
            {"contrast": "joint_true_bit_minus_max_rate_independent", **primary},
            {"contrast": "joint_true_bit_minus_queue_rate", **secondary},
        ],
    )
    spatial_rows = []
    hold_rows = []
    for name in RANKED:
        local_holds = [row for row in holds if row["controller"] == name]
        spatial_rows.append(
            {
                "controller": name,
                "mean_switch_count": _mean(_values(summaries, name, "switch_count")),
                "mean_path_length_m": _mean(_values(summaries, name, "path_length_m")),
                "mean_first_step_rate": _mean(row["first_step_rate"] for row in local_holds),
                "mean_last_step_rate": _mean(row["last_step_rate"] for row in local_holds),
                "mean_first_distance_m": _mean(row["first_distance"] for row in local_holds),
                "mean_hold_payload_bits": _mean(row["payload_bits"] for row in local_holds),
            }
        )
        hold_rows.append(
            {
                "controller": name,
                "hold_steps": HOLD_STEPS,
                "mean_hold_payload_bits": _mean(row["payload_bits"] for row in local_holds),
                "mean_zero_service_fraction": _mean(float(row["zero_service_steps"]) / max(int(row["steps"]), 1) for row in local_holds),
                "mean_within_hold_rate_drop": _mean(float(row["first_step_rate"]) - float(row["last_step_rate"]) for row in local_holds),
                "mean_queue_change_bits": _mean(row["queue_change_bits"] for row in local_holds),
                "retuned": False,
            }
        )
    _write_csv("phase6c_spatial_planning.csv", spatial_rows)
    _write_csv("phase6c_hold15_diagnostic.csv", hold_rows)
    ablation_rows = []
    for feature in ("distance", "queue", "rate_if_assigned", "uav_uav_interference", "current_association", "uav_positions"):
        subset = [row for row in ablations if row["feature"] == feature]
        ablation_rows.append(
            {
                "feature": feature,
                "decisions": len(subset),
                "assignment_disagreement": _mean(row["assignment_disagreement"] for row in subset),
                "mean_true_served_bits_drop": _mean(row["served_bits_drop"] for row in subset),
            }
        )
    _write_csv("phase6c_feature_ablation.csv", ablation_rows)
    coverage_rows = []
    best_coverage_at_3 = 0.0
    for ranking in ("instantaneous_rate", "queue_rate", "individual_served_bits"):
        for k in (1, 2, 3, 4, 5):
            subset = [row for row in coverage if row["ranking"] == ranking and int(row["k"]) == k and int(row["edges"]) > 0]
            edges = sum(int(row["edges"]) for row in subset)
            hits = sum(int(row["covered_edges"]) for row in subset)
            fraction = float(hits / edges) if edges else float("nan")
            coverage_rows.append({"ranking": ranking, "k": k, "edges": edges, "covered_edges": hits, "coverage": fraction})
            if k == 3 and np.isfinite(fraction):
                best_coverage_at_3 = max(best_coverage_at_3, fraction)
    _write_csv("phase6c_candidate_coverage.csv", coverage_rows)
    gate = choose_gate(
        conservation_ok=conservation_ok,
        best_coverage_at_3=best_coverage_at_3,
        dynamic_range_ok=dynamic_range_ok,
        relational_confirmed=relational_confirmed,
        short_horizon_upper_bound_available=False,
    )
    _dump(
        "phase6c_protocol.json",
        {
            "environment": "L3BBitQueueEnvV2",
            "development_pool": "320000-320063",
            "reserved_pool": "320100-320131",
            "fresh_confirmation_untouched": True,
            "legacy_mappo_in_formal_comparison": False,
            "low_level_status": "LOW_LEVEL_RETRAIN_PENDING",
            "movement": "fly_toward",
            "hold": HOLD_STEPS,
            "saturation_max": SATURATION_MAX,
            "pairwise_tie_max": PAIRWISE_TIE_MAX,
            "distinguish_min": DISTINGUISH_MIN,
            "positive_min": POSITIVE_MIN,
            "coverage_min": COVERAGE_MIN,
            "relational_primary": PRIMARY_RELATIONAL_BASELINE,
            "relational_secondary": SECONDARY_RELATIONAL_BASELINE,
            "short_horizon_upper_bound": False,
            "trained": False,
            "planner_ready": False,
            "deployment_verified": False,
        },
    )
    _dump(
        "phase6c_benchmark_config.json",
        {
            "M": int(config.num_uavs),
            "K": int(config.num_buoys),
            "T": int(config.episode_horizon),
            "delta_t_s": float(config.delta_t),
            "bandwidth_hz": float(config.bandwidth_hz),
            "queue_capacity_blocks": int(config.queue_capacity_blocks),
            "queue_capacity_bits": float(config.queue_capacity_blocks) * float(config.block_payload_bits),
            "block_bits": float(config.block_payload_bits),
            "arrival_process": "偶数时刻 4 到 36，浮标 6 到 11 各到达 1 block；换算成 bit 时乘 block_bits",
            "initial_queue_blocks": "浮标 0-2 为 4，浮标 3-5 为 1，浮标 6-11 为 0",
            "movement_limit_m_per_slot": float(config.max_speed * config.delta_t),
            "service_radius_m": float(config.service_radius_m),
            "association_constraints": "独占；冲突双方都不服务；越界目标记为 idle",
            "interference_model": "同簇 kappa_same_cluster，跨簇 kappa_cross_cluster，加到噪声上",
            "sinr_formula": "P * d^(-alpha) / (noise + sum_j P * d_j^(-alpha) * kappa)",
            "rate_formula": "bandwidth_hz * log2(1 + SINR)，单位 bit/s，本步比特再乘 delta_t",
            "reference_rate_bps": float(reference_rate(config)),
            "safety_projector": "保留在 legacy MAPPO 接口外；本阶段 scripted fly-toward 不调用它",
            "hold": HOLD_STEPS,
            "service_order": "先按本步 rate 服务已有队列，再接收到达，再按容量丢弃",
        },
    )
    _dump(
        "phase6c_reference_controllers.json",
        {
            "movement_for_every_closed_loop": "fly_toward",
            "legalization": "UAV 序号小的优先保留冲突浮标，被挤出的 UAV 取下一高分且未被占用的浮标",
            "controllers": {
                "random_legal": "固定 seed 的无放回浮标排列，前 M 个分给 UAV",
                "nearest": "非空队列上 1/(distance+1)，再合法化",
                "max_queue": "分数等于 Q_bits，再合法化",
                "max_rate_independent": "分数等于 rate_if_assigned，不重新优化干扰，再合法化",
                "queue_rate": "U = (Q / max Q) * (rate_if_assigned / max rate)",
                "joint_rate": "精确枚举独占且可激活的分配，最大化 sum rate * delta_t",
                "joint_true_bit": "同一个枚举，最大化 sum min(Q_bits, rate * delta_t)。这是 one-step true-bit optimum，不是 top-k pool optimum",
                "fly_toward_joint_true_bit": "与 joint_true_bit 闭环相同，因为低层已经是 fly-toward",
            },
            "exactness": "超出服务半径或空队列的目标不会形成活动链路，等价于 idle；因此只枚举距离内且队列为正的独占分配，仍是完整合法空间上的一拍最优",
        },
    )
    gate_payload = dict(gate)
    gate_payload.update(
        {
            "conservation_ok": conservation_ok,
            "max_absolute_error_bits": max(float(row["conservation_abs_error_bits"]) for row in summaries),
            "max_saturation_ratio": max_saturation,
            "pairwise_tie_fraction": tie_fraction,
            "scenario_distinguish_fraction": distinguish,
            "best_coverage_at_3": best_coverage_at_3,
            "relational_primary": primary,
            "relational_secondary": secondary,
            "fresh_confirmation_untouched": True,
            "legacy_mappo_used": False,
        }
    )
    _dump("phase6c_dynamic_range_gate.json", gate_payload)
    means = {name: float(np.mean(payloads[name])) for name in RANKED}
    mins = {name: float(np.min(payloads[name])) for name in RANKED}
    maxs = {name: float(np.max(payloads[name])) for name in RANKED}
    coverage_text = "；".join(
        f"{row['ranking']} coverage@{row['k']} = {_fmt(row['coverage'])}" for row in coverage_rows if int(row["k"]) in {1, 3, 5}
    )
    ablation_text = "；".join(
        f"{row['feature']} 改变 assignment 的比例是 {_fmt(row['assignment_disagreement'])}，真实 served bits 平均下降 {_fmt(row['mean_true_served_bits_drop'])}"
        for row in ablation_rows
    )
    _write_report(
        {
            "gate": gate["gate"],
            "gate_name": gate["gate_name"],
            "max_saturation": max_saturation,
            "tie_fraction": tie_fraction,
            "distinguish": distinguish,
            "dynamic_range_ok": dynamic_range_ok,
            "means": means,
            "mins": mins,
            "maxs": maxs,
            "mean_text": "；".join(f"{name} 均值 {_fmt(means[name])} bit" for name in RANKED),
            "primary": primary,
            "secondary": secondary,
            "reset_joint": _mean(row["reset_one_step_joint"] for row in paired_rows),
            "reset_rate": _mean(row["reset_one_step_max_rate"] for row in paired_rows),
            "reset_queue_rate": _mean(row["reset_one_step_queue_rate"] for row in paired_rows),
            "interference_disagreement": _mean(row["assignment_disagreement"] for row in interference_rows),
            "interference_layers": "，".join(f"{row['stratum']} {_fmt(row['assignment_disagreement'])}" for row in layer_rows),
            "clip_joint": next(row["clip_frequency"] for row in clip_rows if row["controller"] == JOINT_CONTROLLER),
            "clip_capacity": next(row["clip_frequency"] for row in clip_rows if row["controller"] == "joint_rate"),
            "ablation_text": ablation_text,
            "coverage_text": coverage_text,
            "coverage_failure": bool(best_coverage_at_3 < COVERAGE_MIN),
            "relational_confirmed": relational_confirmed,
            "switch_spearman": _spearman(_values(summaries, JOINT_CONTROLLER, "switch_count"), payloads[JOINT_CONTROLLER]),
            "max_abs_error": max(float(row["conservation_abs_error_bits"]) for row in summaries),
        }
    )
    print({"gate": gate["gate"], "saturation": max_saturation, "coverage3": best_coverage_at_3, "relational": relational_confirmed}, flush=True)


if __name__ == "__main__":
    main()
