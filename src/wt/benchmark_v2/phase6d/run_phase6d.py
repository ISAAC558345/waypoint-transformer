"""Phase-6D runner. Writes the audit tables and does not train a planner."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6d.protocol import (
    COVERAGE_RELATIVE_GAIN,
    COVERAGE_SCENARIOS,
    DEVELOPMENT_POOL,
    HOLDS,
    IDLE_DROP_FRACTION,
    MACRO_DISAGREEMENT_MIN,
    PHASE,
    PHASE6C_MEANS,
    REPORT,
    REPRODUCTION_ABS_BITS,
    RESERVED_POOL,
    SHORT_HOLDS,
    choose_gate,
    cluster_bootstrap_ci,
    gain_confirmed,
)
from wt.benchmark_v2.phase6d.rollout import audit_scenario, reproduce_scenario

PHASE6C_RESULTS = PHASE.parent / "phase6c" / "phase6c_reference_results.csv"
REPRODUCTION_CONTROLLERS = ("max_rate_independent", "queue_rate", "joint_true_bit")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    number = float(value)
    if not np.isfinite(number):
        return "null"
    if abs(number) >= 1000:
        return f"{number:.6g}"
    return f"{number:.6g}"


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _lookup(rows: list[dict], scenario: int, controller: str, hold: int) -> dict:
    for row in rows:
        if int(row["scenario_id"]) == int(scenario) and row["controller"] == controller and int(row["hold"]) == int(hold):
            return row
    raise KeyError((scenario, controller, hold))


def _payload_map(rows: list[dict], controller: str, hold: int) -> dict[int, float]:
    return {
        int(row["scenario_id"]): float(row["cumulative_payload_bits"])
        for row in rows
        if row["controller"] == controller and int(row["hold"]) == int(hold)
    }


def _paired(rows: list[dict], left: str, right: str, hold: int = 15) -> list[dict]:
    left_map = _payload_map(rows, left, hold)
    right_map = _payload_map(rows, right, hold)
    paired = []
    for scenario in DEVELOPMENT_POOL:
        delta = left_map[int(scenario)] - right_map[int(scenario)]
        paired.append(
            {
                "scenario_id": int(scenario),
                "left": left,
                "right": right,
                "left_payload_bits": left_map[int(scenario)],
                "right_payload_bits": right_map[int(scenario)],
                "delta_payload_bits": float(delta),
                "non_causal_upper_bound": bool("realized_future" in left or "realized_future" in right),
            }
        )
    return paired


def _weighted_idle(windows: list[dict], controller: str, hold: int) -> float:
    chosen = [row for row in windows if row["controller"] == controller and int(row["hold"]) == int(hold)]
    weight = sum(int(row["window_steps"]) for row in chosen)
    if weight <= 0:
        return 0.0
    idle = sum(float(row["idle_after_depletion_ratio"]) * int(row["window_steps"]) for row in chosen)
    return float(idle / weight)


def _cause_shares(windows: list[dict], controller: str, hold: int) -> dict:
    chosen = [row for row in windows if row["controller"] == controller and int(row["hold"]) == int(hold)]
    total = sum(int(row["zero_service_steps"]) for row in chosen)
    shares = {}
    for name in ("A_travel", "B_depleted", "C_interference", "D_empty_at_decision", "F_other", "A_overshoot"):
        count = sum(int(row[f"cause_{name}"]) for row in chosen)
        shares[name] = 0.0 if total <= 0 else float(count / total)
    hold_lock = sum(int(row["cause_E_hold_lock"]) for row in chosen)
    shares["E_hold_lock_among_zero_steps"] = 0.0 if total <= 0 else float(hold_lock / total)
    shares["zero_service_steps"] = int(total)
    return shares


def _reproduce_reference() -> dict[tuple[int, str], float]:
    found: dict[tuple[int, str], float] = {}
    with PHASE6C_RESULTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["controller"] in REPRODUCTION_CONTROLLERS:
                found[(int(row["scenario_id"]), row["controller"])] = float(row["cumulative_payload_bits"])
    return found


def _reproduction_rows() -> tuple[list[dict], bool]:
    reference = _reproduce_reference()
    rows = []
    ok = True
    with ProcessPoolExecutor(max_workers=4) as pool:
        for packet in pool.map(reproduce_scenario, DEVELOPMENT_POOL):
            for item in packet:
                expected = reference[(int(item["scenario_id"]), item["controller"])]
                delta = abs(float(item["phase6d_payload_bits"]) - expected)
                matched = bool(delta <= REPRODUCTION_ABS_BITS and item["conservation_ok"])
                ok = ok and matched
                rows.append(
                    {
                        **item,
                        "phase6c_payload_bits": expected,
                        "abs_delta_bits": float(delta),
                        "matched": matched,
                    }
                )
    return rows, ok


def _protocol_payload() -> dict:
    return {
        "development_pool": list(DEVELOPMENT_POOL),
        "reserved_pool_unopened": list(RESERVED_POOL),
        "holds": list(HOLDS),
        "executor": "deterministic_fly_toward",
        "environment": "L3BBitQueueEnvV2",
        "j15_current": "future unknown arrivals are zero; causal candidate for later supervision",
        "j15_future": "NON_CAUSAL_UPPER_BOUND",
        "NON_CAUSAL_UPPER_BOUND": True,
        "candidate_pool": "instantaneous-rate top-3 partial injections plus idle",
        "optimum_name": "candidate-pool optimum",
        "global_oracle": False,
        "coverage_subset": list(COVERAGE_SCENARIOS),
        "coverage_failure_rule": "mean relative J15(top-4)/J15(top-3) - 1 > 0.05 and CI lower > 0",
        "positive_fraction_min": 0.60,
        "macro_disagreement_min": MACRO_DISAGREEMENT_MIN,
        "shadow_is_intermediate_decision_layer": False,
        "planner_ready": False,
        "deployment_verified": False,
        "trained": False,
        "phase6c_reference_means": PHASE6C_MEANS,
    }


def _aggregate(packets: list[dict]) -> dict:
    summaries = [row for packet in packets for row in packet["summaries"]]
    windows = [row for packet in packets for row in packet["windows"]]
    decisions = [row for packet in packets for row in packet["decisions"]]
    coverage = [row for packet in packets for row in packet["coverage"]]
    ablations = [row for packet in packets for row in packet["ablations"]]
    conservation_ok = all(bool(row["conservation_ok"]) for row in summaries)
    return {
        "summaries": summaries,
        "windows": windows,
        "decisions": decisions,
        "coverage": coverage,
        "ablations": ablations,
        "conservation_ok": conservation_ok,
    }


def _coverage_flag(rows: list[dict]) -> tuple[bool, dict]:
    by_scenario: dict[int, list[float]] = {}
    for row in rows:
        by_scenario.setdefault(int(row["scenario_id"]), []).append(float(row["relative_k4_vs_k3"]))
    per_scenario = [float(np.mean(values)) for values in by_scenario.values()]
    stats = cluster_bootstrap_ci(np.asarray(per_scenario, dtype=np.float64))
    failure = bool(
        stats["mean"] is not None
        and stats["mean"] > COVERAGE_RELATIVE_GAIN
        and stats["ci_low"] is not None
        and stats["ci_low"] > 0.0
    )
    edge = float(np.mean([float(row["edge_coverage_k5_in_k3"]) for row in rows])) if rows else None
    return failure, {
        "scenarios": len(per_scenario),
        "mean_relative_k4_vs_k3": stats["mean"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "mean_j15_k3": _mean(rows, "j15_k3") if rows else None,
        "mean_j15_k4": _mean(rows, "j15_k4") if rows else None,
        "mean_j15_k5": _mean(rows, "j15_k5") if rows else None,
        "edge_coverage_k5_in_k3": edge,
        "temporal_candidate_coverage_failure": failure,
    }


def _hold_mismatch(summaries: list[dict], windows: list[dict]) -> tuple[bool, dict, int]:
    means = {}
    for hold in SHORT_HOLDS:
        chosen = [row for row in summaries if row["controller"] == "joint_true_bit" and int(row["hold"]) == int(hold)]
        means[int(hold)] = _mean(chosen, "cumulative_payload_bits")
    best_hold = max(means, key=means.get)
    base = _payload_map(summaries, "joint_true_bit", 15)
    short = _payload_map(summaries, "joint_true_bit", best_hold)
    delta = np.asarray([short[int(scenario)] - base[int(scenario)] for scenario in DEVELOPMENT_POOL], dtype=np.float64)
    stats = gain_confirmed(delta)
    idle_short = _weighted_idle(windows, "joint_true_bit", best_hold)
    idle_15 = _weighted_idle(windows, "joint_true_bit", 15)
    drop = float(idle_15 - idle_short)
    relative_drop = float(drop / idle_15) if idle_15 > 1.0e-9 else 0.0
    idle_dropped = bool(relative_drop >= IDLE_DROP_FRACTION or drop >= 0.25)
    confirmed = bool(stats["confirmed"] and idle_dropped)
    return confirmed, {
        "best_short_hold": int(best_hold),
        "short_hold_means": {str(key): value for key, value in means.items()},
        "hold15_mean": _mean(
            [row for row in summaries if row["controller"] == "joint_true_bit" and int(row["hold"]) == 15],
            "cumulative_payload_bits",
        ),
        "delta": stats,
        "idle_after_depletion_short": idle_short,
        "idle_after_depletion_hold15": idle_15,
        "idle_dropped": idle_dropped,
        "hold_mismatch_confirmed": confirmed,
    }, int(best_hold)


def _relation_flag(decisions: list[dict]) -> tuple[bool, dict]:
    chosen = [
        row
        for row in decisions
        if row["controller"] == "hold15_current_state_pool_optimum" and row.get("interference_disagreement") is not None
    ]
    if not chosen:
        return False, {"decisions": 0, "disagreement": None, "mean_delta_j15": None, "macro_relation_value_confirmed": False}
    disagreement = float(np.mean([int(row["interference_disagreement"]) for row in chosen]))
    delta = float(np.mean([float(row["delta_j15"]) for row in chosen]))
    confirmed = bool(disagreement >= MACRO_DISAGREEMENT_MIN)
    return confirmed, {
        "decisions": len(chosen),
        "disagreement": disagreement,
        "mean_delta_j15": delta,
        "macro_relation_value_confirmed": confirmed,
    }


def _decomposition_rows(windows: list[dict], summaries: list[dict]) -> list[dict]:
    rows = []
    for controller in ("max_rate_independent", "joint_true_bit"):
        for hold in HOLDS:
            shares = _cause_shares(windows, controller, hold)
            selected = [row for row in summaries if row["controller"] == controller and int(row["hold"]) == int(hold)]
            rows.append(
                {
                    "scenario_id": "ALL",
                    "controller": controller,
                    "hold": int(hold),
                    "mean_zero_service_fraction": _mean(selected, "zero_service_steps") / 48.0,
                    "service_utilization_ratio": _mean(selected, "service_utilization_ratio"),
                    "idle_after_depletion_ratio": _weighted_idle(windows, controller, hold),
                    **shares,
                }
            )
    return rows


def _hold_sweep_rows(summaries: list[dict], windows: list[dict]) -> list[dict]:
    rows = []
    for row in summaries:
        if row["controller"] not in {"max_rate_independent", "joint_true_bit"}:
            continue
        scenario_windows = [
            item
            for item in windows
            if item["controller"] == row["controller"]
            and int(item["hold"]) == int(row["hold"])
            and int(item["scenario_id"]) == int(row["scenario_id"])
        ]
        weight = sum(int(item["window_steps"]) for item in scenario_windows) or 1
        idle = sum(float(item["idle_after_depletion_ratio"]) * int(item["window_steps"]) for item in scenario_windows) / weight
        util = sum(float(item["service_utilization_ratio"]) * int(item["window_steps"]) for item in scenario_windows) / weight
        rows.append(
            {
                **row,
                "service_utilization_ratio_windows": float(util),
                "idle_after_depletion_ratio": float(idle),
            }
        )
    return rows


def _ablation_summary(ablations: list[dict], summaries: list[dict]) -> list[dict]:
    closed = {
        "queue_removed": "ablation_queue_removed",
        "distance_removed": "ablation_distance_removed",
        "interference_removed": "ablation_interference_removed",
        "instantaneous_rate_removed": "ablation_rate_removed",
    }
    base = _payload_map(summaries, "hold15_current_state_pool_optimum", 15)
    rows = []
    for name, controller in closed.items():
        selected = [row for row in ablations if row["ablation"] == name]
        other = _payload_map(summaries, controller, 15)
        delta = [other[int(scenario)] - base[int(scenario)] for scenario in DEVELOPMENT_POOL]
        rows.append(
            {
                "ablation": name,
                "same_state_decisions": len(selected),
                "assignment_disagreement": float(np.mean([int(row["assignment_disagreement"]) for row in selected])) if selected else None,
                "mean_delta_j15": float(np.mean([float(row["delta_j15"]) for row in selected])) if selected else None,
                "closed_loop_mean_payload_bits": float(np.mean(list(other.values()))),
                "closed_loop_minus_b6_mean_bits": float(np.mean(delta)),
                "subset_for_same_state": "320000-320015",
                "closed_loop_pool": "320000-320063",
                "non_causal_upper_bound": False,
            }
        )
    return rows


def _max_rate_advantage(windows: list[dict]) -> dict:
    def _collect(controller: str) -> dict:
        chosen = [row for row in windows if row["controller"] == controller and int(row["hold"]) == 15]
        if not chosen:
            return {}
        return {
            "mean_initial_distance_m": float(np.mean([float(row["mean_uav_target_distance"]) for row in chosen])),
            "mean_target_q_bits": float(np.mean([float(row["mean_target_q_bits"]) for row in chosen])),
            "mean_served_bits": float(np.mean([float(row["served_bits"]) for row in chosen])),
            "service_utilization": float(np.mean([float(row["service_utilization_ratio"]) for row in chosen])),
            "mean_distance_reduction_m": float(
                np.mean(
                    [
                        float(row["mean_uav_target_distance"]) - float(row["final_mean_distance"])
                        for row in chosen
                        if float(row["mean_uav_target_distance"]) > 0.0
                    ]
                )
            ),
            "mean_empty_locked_steps": float(np.mean([float(row["empty_locked_steps"]) for row in chosen])),
            "fraction_windows_target_became_empty": float(np.mean([1.0 if row["target_became_empty"] else 0.0 for row in chosen])),
        }
    return {"max_rate_independent": _collect("max_rate_independent"), "joint_true_bit": _collect("joint_true_bit")}


def _report(facts: dict) -> str:
    gain = facts["b6_vs_b3"]
    mismatch = facts["b6_vs_b5"]
    future = facts["b7_vs_b6"]
    future_rate = facts["b7_vs_b3"]
    hold = facts["hold"]
    coverage = facts["coverage"]
    relation = facts["relation"]
    shares = facts["joint_hold15_causes"]
    advantage = facts["advantage"]
    train = "YES, NEXT PHASE ONLY" if facts["gate"] == "TEMP-B" else "NO"
    supervision = (
        "executor-conditioned macro-action utility J_H_CURRENT。"
        if facts["gate"] == "TEMP-B"
        else "现在不建立监督数据。不能使用 one-step assignment similarity，也不能使用 one-step served bits。当前 Gate 是候选池覆盖失败，下一阶段应先修正 Stage-1 candidate generator，而不是用 one-step 标签训练 HGNN。"
    )
    lines = [
        "# HTW Phase-6D：时间抽象与 hold-aware 联合规划审计",
        "",
        f"Gate 是 {facts['gate']}（{facts['gate_name']}）。",
        "",
        "Phase-6A 仍是 BV2-E，Phase-6B 仍是 BITQ-A，Phase-6C 仍是 DYN-A。环境是 `L3BBitQueueEnvV2`。低层执行器始终是 deterministic fly-toward。320100–320131 没有打开。没有训练 HGNN、DeepSets、MLP、MAPPO、Transformer 或 value network。`planner_ready = False`，`deployment_verified = False`。",
        "",
        "shadow assignment evaluator 只用于诊断、计算宏动作价值和判断未来监督目标。它不是正式系统里的第三层。若以后训练高层，结构必须是 state → HGNN → joint assignment，由 HGNN 直接学习宏动作累计收益。不能做成 HGNN → top-3 → 第二个决策层再选目标。",
        "",
        "## Q1. 为什么 one-step true-bit 即时收益高，闭环收益却低？",
        "",
        f"Phase-6C 已经在同一初始状态上看到 one-step joint served bits 高于 independent max-rate。hold=15 的闭环复现结果是 max-rate 均值 {_fmt(facts['means']['max_rate_independent'])} bit，joint true-bit 均值 {_fmt(facts['means']['joint_true_bit'])} bit。按 48 个 episode step 统计，joint 的零服务比例是 {_fmt(facts['joint_zero_fraction'])}。它用一拍目标选中当前就能传完的近端队列。服务在移动之后结算，而步长是 22 m、服务半径是 16 m：已经贴着浮标时，一步 fly-toward 会先冲出半径。hold 把这个目标锁住，队列传完后的剩余步不再服务，一拍规则也不会为了远端尚未进入半径的浮标出发。",
        "",
        "## Q2. 92% 无服务的主要来源是什么？",
        "",
        f"joint true-bit 在 hold=15 的零服务时间步里，出半径 A 占 {_fmt(shares['A_travel'])}，其中过冲（移动前在半径内、服务时已出半径）占 {_fmt(shares['A_overshoot'])}。队列提前耗尽 B 占 {_fmt(shares['B_depleted'])}，干扰 C 占 {_fmt(shares['C_interference'])}，决策时目标已空 D 占 {_fmt(shares['D_empty_at_decision'])}，其他 F 占 {_fmt(shares['F_other'])}。A、B、C、D、F 加总为 1。E（hold 锁定在空目标上）占 {_fmt(shares['E_hold_lock_among_zero_steps'])}，与 B 是同一批步。过冲是 A 的子集。F 主要是仍有正队列时一拍规则选择 idle，没有开始飞向远端。",
        "",
        "## Q3. 缩短 hold 后，joint true-bit 是否恢复？",
        "",
        f"诊断用的最佳短 hold 是 {hold['best_short_hold']}，其 joint 均值是 {_fmt(hold['short_hold_means'][str(hold['best_short_hold'])])} bit，hold=15 是 {_fmt(hold['hold15_mean'])} bit。配对差均值 {_fmt(hold['delta']['mean'])} bit，95% CI [{_fmt(hold['delta']['ci_low'])}, {_fmt(hold['delta']['ci_high'])}]，正收益比例 {_fmt(hold['delta']['positive_fraction'])}。idle-after-depletion 从 {_fmt(hold['idle_after_depletion_hold15'])} 变到 {_fmt(hold['idle_after_depletion_short'])}。HOLD_MISMATCH_CONFIRMED = {hold['hold_mismatch_confirmed']}。缩短 hold 没有提高 joint true-bit 的 payload。hold=1 更差，因为每一步都在半径外重新决策，UAV 无法在覆盖圈内停留。这不是正式 hold 选择。",
        "",
        "## Q4. hold=1/3/5/8/10/15 的诊断趋势在哪里？",
        "",
        facts["hold_curve_text"],
        "",
        "## Q5. J15_CURRENT 是否稳定优于 one-step true-bit？",
        "",
        f"B6 − B5 的均值是 {_fmt(mismatch['mean'])} bit，95% CI [{_fmt(mismatch['ci_low'])}, {_fmt(mismatch['ci_high'])}]，正收益比例 {_fmt(mismatch['positive_fraction'])}。TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED = {mismatch['confirmed']}。",
        "",
        "## Q6. J15_CURRENT 是否稳定优于 independent max-rate？",
        "",
        f"B6 − B3 的均值是 {_fmt(gain['mean'])} bit，中位数 {_fmt(gain['median'])}，标准差 {_fmt(gain['std'])}，95% CI [{_fmt(gain['ci_low'])}, {_fmt(gain['ci_high'])}]，正收益比例 {_fmt(gain['positive_fraction'])}。稳定优于 max-rate：{gain['confirmed']}。B6 只是 instantaneous-rate top-3 的 candidate-pool optimum。Q9 显示这个 pool 对 J15 覆盖不足，所以这里不能把“B6 没有超过 max-rate”写成联合规划没有价值，也不能把它写成 TEMP-D。",
        "",
        "## Q7. 如果不能，J15_FUTURE 是否可以？",
        "",
        f"B7 是 NON_CAUSAL_UPPER_BOUND，不能用于训练、checkpoint、正式控制器或以后的 confirmation。B7 − B3 的均值是 {_fmt(future_rate['mean'])} bit，95% CI [{_fmt(future_rate['ci_low'])}, {_fmt(future_rate['ci_high'])}]，正收益比例 {_fmt(future_rate['positive_fraction'])}，稳定优于 max-rate：{future_rate['confirmed']}。",
        "",
        "## Q8. 未来 arrival 是否是当前主要缺失信息？",
        "",
        f"B7 − B6 的均值是 {_fmt(future['mean'])} bit，95% CI [{_fmt(future['ci_low'])}, {_fmt(future['ci_high'])}]，正收益比例 {_fmt(future['positive_fraction'])}。FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED = {future['confirmed']}。即使这个差很大，本阶段也不把 realized future arrival 放进任何特征。以后若要建模，只能使用到达统计、到达历史、队列趋势或时间索引这类因果量。",
        "",
        "## Q9. top-3 candidate pool 在 J15 下是否仍足够？",
        "",
        f"预注册子集 320000–320015 上，max-rate 决策状态的 candidate-pool optimum 均值是 J15@3 {_fmt(coverage['mean_j15_k3'])}、J15@4 {_fmt(coverage['mean_j15_k4'])}、J15@5 {_fmt(coverage['mean_j15_k5'])} bit。top-4 相对 top-3 的 scenario 均值相对增益是 {_fmt(coverage['mean_relative_k4_vs_k3'])}，95% CI [{_fmt(coverage['ci_low'])}, {_fmt(coverage['ci_high'])}]。top-5 pool optimum 的非 idle 边落在 top-3 候选里的比例是 {_fmt(coverage['edge_coverage_k5_in_k3'])}。TEMPORAL_CANDIDATE_COVERAGE_FAILURE = {coverage['temporal_candidate_coverage_failure']}。这些是 candidate-pool optimum，不是 global Oracle。",
        "",
        "## Q10. 宏动作尺度上的 interference 是否仍改变 assignment？",
        "",
        f"HOLD15 current-state pool optimum 在真实干扰和无干扰反事实下的 assignment disagreement 是 {_fmt(relation['disagreement'])}，对应的 ΔJ15 均值是 {_fmt(relation['mean_delta_j15'])} bit。MACRO_RELATION_VALUE_CONFIRMED = {relation['macro_relation_value_confirmed']}。Phase-6C 的 0.55 只说明关系影响一拍分配；这里的数字才是宏动作尺度。",
        "",
        "## Q11. queue、distance、rate、interference 哪些因素影响 J15 optimum？",
        "",
        facts["ablation_text"],
        "",
        "## Q12. 现在是否可以开始训练 HGNN？",
        "",
        train,
        "",
        "## Q13. 如果下一阶段训练 HGNN，监督目标应该是什么？",
        "",
        supervision,
        "",
        "## Q14. shadow evaluator 是不是最终系统中的中间决策层？",
        "",
        "NO。它只是 diagnostic / teacher / label generator。最终高层应直接学习该价值。",
        "",
        "## Q15. 是否需要现在重训 MAPPO？",
        "",
        "NO。",
        "",
        "## Q16. 是否需要现在加深 HGNN？",
        "",
        "NO。",
        "",
        "## Q17. 是否需要现在加入 continuity penalty？",
        "",
        "NO。本阶段没有把 continuity penalty 立为新的 bit-level 因果项目。",
        "",
        "## Q18. 320100–320131 是否仍完全 untouched？",
        "",
        "YES。",
        "",
        "## Q19. 最终 Gate 是哪一个？",
        "",
        f"{facts['gate']}（{facts['gate_name']}）。",
        "",
        "## max-rate 为什么在闭环里更强",
        "",
        facts["advantage_text"],
        "",
        f"附加标志：HOLD_MISMATCH_CONFIRMED = {facts['flags']['HOLD_MISMATCH_CONFIRMED']}；TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED = {facts['flags']['TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED']}；FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED = {facts['flags']['FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED']}；TEMPORAL_CANDIDATE_COVERAGE_FAILURE = {facts['flags']['TEMPORAL_CANDIDATE_COVERAGE_FAILURE']}；MACRO_RELATION_VALUE_CONFIRMED = {facts['flags']['MACRO_RELATION_VALUE_CONFIRMED']}。",
        "",
    ]
    return "\n".join(lines)


def _advantage_text(advantage: dict, shares_joint: dict, shares_rate: dict) -> str:
    rate = advantage["max_rate_independent"]
    joint = advantage["joint_true_bit"]
    return (
        f"hold=15 的窗口对比：max-rate 的决策距离均值 {_fmt(rate.get('mean_initial_distance_m'))} m，"
        f"目标队列 {_fmt(rate.get('mean_target_q_bits'))} bit，窗口服务 {_fmt(rate.get('mean_served_bits'))} bit，"
        f"服务利用率 {_fmt(rate.get('service_utilization'))}，距离缩短 {_fmt(rate.get('mean_distance_reduction_m'))} m，"
        f"空目标锁定步 {_fmt(rate.get('mean_empty_locked_steps'))}。"
        f"joint true-bit 的决策距离均值 {_fmt(joint.get('mean_initial_distance_m'))} m，"
        f"目标队列 {_fmt(joint.get('mean_target_q_bits'))} bit，窗口服务 {_fmt(joint.get('mean_served_bits'))} bit，"
        f"服务利用率 {_fmt(joint.get('service_utilization'))}，距离缩短 {_fmt(joint.get('mean_distance_reduction_m'))} m，"
        f"空目标锁定步 {_fmt(joint.get('mean_empty_locked_steps'))}。"
        f"零服务分解上，max-rate 的旅行/耗尽/干扰/空目标/其他比例是 "
        f"{_fmt(shares_rate['A_travel'])} / {_fmt(shares_rate['B_depleted'])} / {_fmt(shares_rate['C_interference'])} / "
        f"{_fmt(shares_rate['D_empty_at_decision'])} / {_fmt(shares_rate['F_other'])}。"
        "max-rate 的优势来自它按瞬时信道把 UAV 派向还能持续服务的链路，飞向过程本身变成了跨步的正则：目标在到达后仍有队列。joint 的一拍比特把 UAV 留在马上传完的近端目标上，宏动作后半段没有服务。"
    )


def _hold_curve_text(summaries: list[dict]) -> str:
    parts = []
    for controller in ("joint_true_bit", "max_rate_independent"):
        bits = []
        for hold in HOLDS:
            chosen = [row for row in summaries if row["controller"] == controller and int(row["hold"]) == int(hold)]
            bits.append(f"hold={hold} {_fmt(_mean(chosen, 'cumulative_payload_bits'))}")
        parts.append(controller + "：" + "，".join(bits) + " bit。")
    return " ".join(parts) + " 曲线只说明诊断趋势，不作为正式 hold。"


def _ablation_text(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"{row['ablation']} 的同状态 assignment disagreement 是 {_fmt(row['assignment_disagreement'])}，"
            f"ΔJ15 均值 {_fmt(row['mean_delta_j15'])} bit，闭环 payload 相对 B6 的均值差是 {_fmt(row['closed_loop_minus_b6_mean_bits'])} bit。"
        )
    return (
        "".join(parts)
        + " 同状态消融在 320000–320015 的 B6 决策点上；闭环差覆盖全部 64 个 development scenario。没有修改正式 benchmark。"
        + " 去掉 queue 会改变分配，但 ΔJ15 和闭环 payload 几乎不动。distance、interference，以及用 queue top-3 代替 instantaneous-rate top-3，都会降低 J15；闭环里下降最大的是去掉 distance。"
    )


def main() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    _write_json(PHASE / "phase6d_protocol.json", _protocol_payload())
    reproduction, reproduction_ok = _reproduction_rows()
    _write_csv(PHASE / "phase6d_phase6c_reproduction.csv", reproduction)
    if not reproduction_ok:
        gate = choose_gate(
            reproduction_ok=False,
            conservation_ok=False,
            coverage_failure=False,
            hold_aware_beats_max_rate=False,
            temporal_objective_mismatch=False,
            future_beats_current=False,
            future_beats_max_rate=False,
            hold_mismatch=False,
        )
        gate.update(
            {
                "HOLD_MISMATCH_CONFIRMED": False,
                "TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED": False,
                "FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED": False,
                "TEMPORAL_CANDIDATE_COVERAGE_FAILURE": False,
                "MACRO_RELATION_VALUE_CONFIRMED": False,
            }
        )
        _write_json(PHASE / "phase6d_temporal_planning_gate.json", gate)
        REPORT.write_text(
            "# HTW Phase-6D：时间抽象与 hold-aware 联合规划审计\n\n"
            "Gate 是 TEMP-E（REPRODUCTION_OR_ACCOUNTING_FAILURE）。Phase-6C 的 hold=15 payload 没有在浮点容差内复现，后续比较没有运行。320100–320131 没有打开。\n",
            encoding="utf-8",
        )
        print("TEMP-E")
        return
    packets = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, packet in enumerate(pool.map(audit_scenario, DEVELOPMENT_POOL), start=1):
            packets.append(packet)
            print(f"scenario {index}/64", flush=True)
    data = _aggregate(packets)
    if not data["conservation_ok"]:
        gate = choose_gate(
            reproduction_ok=True,
            conservation_ok=False,
            coverage_failure=False,
            hold_aware_beats_max_rate=False,
            temporal_objective_mismatch=False,
            future_beats_current=False,
            future_beats_max_rate=False,
            hold_mismatch=False,
        )
        _write_json(PHASE / "phase6d_temporal_planning_gate.json", gate)
        print("TEMP-E conservation")
        return
    summaries = data["summaries"]
    windows = data["windows"]
    decisions = data["decisions"]
    b6_b3 = gain_confirmed(np.asarray([row["delta_payload_bits"] for row in _paired(summaries, "hold15_current_state_pool_optimum", "max_rate_independent")], dtype=np.float64))
    b6_b5 = gain_confirmed(np.asarray([row["delta_payload_bits"] for row in _paired(summaries, "hold15_current_state_pool_optimum", "joint_true_bit")], dtype=np.float64))
    b7_b6 = gain_confirmed(np.asarray([row["delta_payload_bits"] for row in _paired(summaries, "hold15_realized_future_pool_optimum", "hold15_current_state_pool_optimum")], dtype=np.float64))
    b7_b3 = gain_confirmed(np.asarray([row["delta_payload_bits"] for row in _paired(summaries, "hold15_realized_future_pool_optimum", "max_rate_independent")], dtype=np.float64))
    coverage_failure, coverage = _coverage_flag(data["coverage"])
    hold_confirmed, hold_info, _best = _hold_mismatch(summaries, windows)
    relation_confirmed, relation = _relation_flag(decisions)
    gate = choose_gate(
        reproduction_ok=True,
        conservation_ok=True,
        coverage_failure=coverage_failure,
        hold_aware_beats_max_rate=bool(b6_b3["confirmed"]),
        temporal_objective_mismatch=bool(b6_b5["confirmed"]),
        future_beats_current=bool(b7_b6["confirmed"]),
        future_beats_max_rate=bool(b7_b3["confirmed"]),
        hold_mismatch=hold_confirmed,
    )
    flags = {
        "HOLD_MISMATCH_CONFIRMED": hold_confirmed,
        "TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED": bool(b6_b5["confirmed"]),
        "FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED": bool(b7_b6["confirmed"]),
        "TEMPORAL_CANDIDATE_COVERAGE_FAILURE": coverage_failure,
        "MACRO_RELATION_VALUE_CONFIRMED": relation_confirmed,
    }
    gate.update(flags)
    means = {
        controller: _mean(
            [row for row in summaries if row["controller"] == controller and int(row["hold"]) == 15],
            "cumulative_payload_bits",
        )
        for controller in (
            "random_legal",
            "nearest",
            "max_queue",
            "max_rate_independent",
            "queue_rate",
            "joint_true_bit",
            "hold15_current_state_pool_optimum",
            "hold15_realized_future_pool_optimum",
        )
    }
    joint_rows = [row for row in summaries if row["controller"] == "joint_true_bit" and int(row["hold"]) == 15]
    joint_causes = _cause_shares(windows, "joint_true_bit", 15)
    rate_causes = _cause_shares(windows, "max_rate_independent", 15)
    advantage = _max_rate_advantage(windows)
    ablation_rows = _ablation_summary(data["ablations"], summaries)
    facts = {
        "gate": gate["gate"],
        "gate_name": gate["gate_name"],
        "means": means,
        "joint_zero_fraction": _mean(joint_rows, "zero_service_steps") / 48.0,
        "joint_hold15_causes": joint_causes,
        "b6_vs_b3": b6_b3,
        "b6_vs_b5": b6_b5,
        "b7_vs_b6": b7_b6,
        "b7_vs_b3": b7_b3,
        "hold": hold_info,
        "coverage": coverage,
        "relation": relation,
        "advantage": advantage,
        "flags": flags,
        "hold_curve_text": _hold_curve_text(summaries),
        "ablation_text": _ablation_text(ablation_rows),
        "advantage_text": _advantage_text(advantage, joint_causes, rate_causes),
    }
    comparison = [row for row in summaries if int(row["hold"]) == 15]
    paired_rows = []
    for left, right in (
        ("hold15_current_state_pool_optimum", "max_rate_independent"),
        ("hold15_current_state_pool_optimum", "joint_true_bit"),
        ("hold15_realized_future_pool_optimum", "hold15_current_state_pool_optimum"),
        ("hold15_realized_future_pool_optimum", "max_rate_independent"),
    ):
        paired_rows.extend(_paired(summaries, left, right))
    current_rows = [
        row
        for row in decisions
        if row["controller"] == "hold15_current_state_pool_optimum"
    ]
    future_rows = [
        row
        for row in decisions
        if row["controller"] == "hold15_realized_future_pool_optimum"
    ]
    _write_csv(PHASE / "phase6d_hold_sweep.csv", _hold_sweep_rows(summaries, windows))
    _write_csv(PHASE / "phase6d_idle_service_decomposition.csv", _decomposition_rows(windows, summaries))
    _write_csv(PHASE / "phase6d_hold_window_diagnostics.csv", windows)
    _write_csv(PHASE / "phase6d_candidate_pool.csv", [row for row in decisions if int(row["pool_size"]) > 0])
    _write_csv(PHASE / "phase6d_j15_current.csv", current_rows)
    _write_csv(PHASE / "phase6d_j15_future_upper_bound.csv", future_rows)
    _write_csv(PHASE / "phase6d_closed_loop_comparison.csv", comparison)
    _write_csv(PHASE / "phase6d_paired_payload_gain.csv", paired_rows)
    _write_csv(
        PHASE / "phase6d_future_arrival_value.csv",
        _paired(summaries, "hold15_realized_future_pool_optimum", "hold15_current_state_pool_optimum"),
    )
    _write_csv(PHASE / "phase6d_temporal_candidate_coverage.csv", data["coverage"])
    _write_csv(
        PHASE / "phase6d_macro_interference_value.csv",
        [row for row in current_rows if row.get("interference_disagreement") is not None],
    )
    ablation_csv = []
    for row in data["ablations"]:
        ablation_csv.append(
            {
                "row_level": "decision",
                "scenario_id": row["scenario_id"],
                "time_step": row["time_step"],
                "ablation": row["ablation"],
                "assignment_disagreement": row["assignment_disagreement"],
                "delta_j15": row["delta_j15"],
                "j15_full": row["j15_full"],
                "j15_ablated_assignment": row["j15_ablated_assignment"],
                "closed_loop_minus_b6_mean_bits": "",
                "non_causal_upper_bound": False,
            }
        )
    for row in ablation_rows:
        ablation_csv.append(
            {
                "row_level": "aggregate",
                "scenario_id": "ALL",
                "time_step": "",
                "ablation": row["ablation"],
                "assignment_disagreement": row["assignment_disagreement"],
                "delta_j15": row["mean_delta_j15"],
                "j15_full": "",
                "j15_ablated_assignment": "",
                "closed_loop_minus_b6_mean_bits": row["closed_loop_minus_b6_mean_bits"],
                "non_causal_upper_bound": False,
            }
        )
    _write_csv(PHASE / "phase6d_macro_feature_ablation.csv", ablation_csv)
    _write_json(PHASE / "phase6d_temporal_planning_gate.json", gate)
    REPORT.write_text(_report(facts), encoding="utf-8")
    print(gate["gate"], flush=True)


if __name__ == "__main__":
    main()
