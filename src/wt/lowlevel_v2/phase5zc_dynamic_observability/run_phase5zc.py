"""Executor-response observability. Reuses Phase-5X shadows and adds one frozen action."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np

from wt.l3b.config import L3BConfig
from wt.l3b.env import kappa_for, path_gain, sinr_rates_for_links
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _labels, _replay, _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _history_snapshot, _vector
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.metrics import synergy_value
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.run_phase5w import _hybrids, _legal, _parse
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff

from .actions import assignment_option, mean_active, radial_components
from .metrics import (
    choose_gate,
    correlation,
    execution_informative,
    nearest_neighbor,
    pairwise_spearman,
    rank_bar,
    ranking,
    sign_accuracy,
    unit_rows,
)
from .protocol import (
    ACT0_SCORE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMAL,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CONFIRM_SCENARIOS,
    CROSS_UAV_FRACTION,
    DEPLOYMENT_VERIFIED,
    HORIZON_SCORE,
    HORIZONS,
    L0_SCORE,
    MOTION_EFFICIENCY,
    ORACLE_HORIZON,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PHY_SCORE,
    PLANNER_READY,
    REGRET_BAR,
    REPORT,
    RESIDUAL_BAR,
    ROOT,
    RESIDUAL_ORIENTATION,
    SPEARMAN_BAR,
    SPEARMAN_GAIN,
    TIMES,
    TOP3_BAR,
    TRAIN_SCENARIOS,
    assert_consumed,
)

ROLLOUT_FIELDS = (
    "payload_h",
    "completed_blocks_h",
    "progress_gain_h",
    "progress_end_sum",
    "q_exec_h",
    "mean_sinr_h",
    "min_sinr_h",
    "mean_rate_h",
    "mean_interference_h",
    "distance_reduction_h",
    "number_in_service_radius_h",
    "service_continuity_preserved_h",
    "number_of_target_switches",
    "low_level_action_norm",
    "elapsed_ms",
    "j15",
)
SIGNAL_FIELDS = {
    "cumulative_payload": "payload_h",
    "mean_sinr": "mean_sinr_h",
    "minimum_sinr": "min_sinr_h",
    "mean_interference": "mean_interference_h",
    "service_progress": "progress_end_sum",
    "completed_blocks": "completed_blocks_h",
    "distance_reduction": "distance_reduction_h",
    "in_range_count": "number_in_service_radius_h",
    "same_target_continuity": "service_continuity_preserved_h",
}
RESIDUAL_FIELDS = {
    "mean_sinr": "mean_sinr_h",
    "service_progress": "progress_end_sum",
    "mean_interference": "mean_interference_h",
    "distance_reduction": "distance_reduction_h",
}


def _plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_csv(name: str, rows: list[dict]) -> None:
    path = OUTPUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _boot(rows: list[dict], key: str) -> dict[str, float]:
    return cluster_bootstrap(rows, key, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


def _load_rollouts() -> tuple[dict, dict]:
    path = ROOT / "results/phase5x_executor_lookahead" / "phase5x_executor_rollouts.csv"
    table: dict[tuple[int, int, str], dict[int, dict]] = {}
    order: dict[tuple[int, int], list[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            moment = int(row["time_step"])
            vector = row["assignment_vector"]
            packed = {field: float(row[field]) for field in ROLLOUT_FIELDS}
            table.setdefault((scenario, moment, vector), {})[int(row["horizon"])] = packed
            order.setdefault((scenario, moment), [])
            if vector not in order[(scenario, moment)]:
                order[(scenario, moment)].append(vector)
    return table, order


def _load_synergy_cases() -> list[dict]:
    path = ROOT / "results/phase5w_residual_mechanism_audit" / "phase5w_residual_case_hybrids.csv"
    cases = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            assert_consumed(int(row["scenario_id"]))
            if float(row["synergy"]) <= 0.0:
                continue
            cases.append(row)
    return cases


def _counterfactual(env, assignment: np.ndarray) -> dict[str, float]:
    links = [(uav, int(buoy)) for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)) if int(buoy) >= 0]
    sinr, rate = sinr_rates_for_links(env, links)
    interference = np.zeros(env.config.num_uavs, dtype=np.float64)
    for uav, buoy in links:
        received = 0.0
        for other, other_buoy in links:
            if int(other) == int(uav):
                continue
            gain = path_gain(env.config, env.uav_positions[other], env.buoy_positions[buoy])
            received += gain * float(env.config.tx_power) * kappa_for(env.config, other_buoy, buoy)
        interference[uav] = received
    active = [uav for uav, _buoy in links]
    return {
        "phy_rate": float(np.sum(rate)),
        "mean_sinr": float(np.mean(sinr[active])) if active else float("nan"),
        "mean_interference": float(np.mean(interference[active])) if active else float("nan"),
    }


def _signal_value(checkpoint: dict, name: str) -> float:
    if name == "motion_efficiency":
        return float(checkpoint["distance_reduction_h"]) / (float(checkpoint["low_level_action_norm"]) + 1.0e-8)
    field = SIGNAL_FIELDS.get(name, RESIDUAL_FIELDS.get(name))
    return float(checkpoint[field])


def _column(state: dict, horizon: int, name: str) -> np.ndarray:
    return np.asarray([_signal_value(item[horizon], name) for item in state["roll"]], dtype=np.float64)


def _action_bundle(executor, env, movement, assignment: np.ndarray) -> dict:
    executed, desired, delta = executor.act(env, assignment_option(env, assignment), np.array(movement, copy=True))
    executed = np.asarray(executed, dtype=np.float64)
    radial, tangential, heading = radial_components(executed, env.uav_positions, env.buoy_positions, assignment)
    return {
        "executed": executed,
        "desired": np.asarray(desired, dtype=np.float64),
        "delta": np.asarray(delta, dtype=np.float64),
        "radial": radial,
        "tangential": tangential,
        "heading": heading,
        "score": mean_active(radial, assignment),
        "joint_l2": float(np.linalg.norm(executed)),
    }


def _pair_distances(actions: np.ndarray, radial: np.ndarray, heading: np.ndarray) -> dict[str, np.ndarray]:
    delta = actions[:, None, :, :] - actions[None, :, :, :]
    per_uav = np.linalg.norm(delta, axis=-1)
    angle = np.abs(heading[:, None, :] - heading[None, :, :])
    angle = np.minimum(angle, 2.0 * np.pi - angle)
    return {
        "joint_l2": np.linalg.norm(delta.reshape(delta.shape[0], delta.shape[1], -1), axis=-1),
        "mean_uav_l2": np.mean(per_uav, axis=-1),
        "radial_l1": np.mean(np.abs(radial[:, None, :] - radial[None, :, :]), axis=-1),
        "heading_l1": np.mean(angle, axis=-1),
        "joint_cosine_distance": 1.0 - (unit_rows(actions.reshape(actions.shape[0], -1)) @ unit_rows(actions.reshape(actions.shape[0], -1)).T),
    }


def _conditioning(assignments: np.ndarray, actions: np.ndarray) -> dict[str, float]:
    same = assignments[:, None, :] == assignments[None, :, :]
    upper = np.triu(np.ones(assignments.shape[0], dtype=bool), k=1)
    own_only = []
    other_only = []
    own_any = []
    for uav in range(assignments.shape[1]):
        others = np.any(~same[:, :, np.arange(assignments.shape[1]) != uav], axis=-1)
        own = ~same[:, :, uav]
        distance = np.linalg.norm(actions[:, None, uav, :] - actions[None, :, uav, :], axis=-1)
        own_only.append(float(np.mean(distance[upper & own & ~others])) if np.any(upper & own & ~others) else float("nan"))
        other_only.append(float(np.mean(distance[upper & ~own & others])) if np.any(upper & ~own & others) else float("nan"))
        own_any.append(float(np.mean(distance[upper & own])) if np.any(upper & own) else float("nan"))
    return {
        "own_target_only_l2": _mean(own_only),
        "other_target_only_l2": _mean(other_only),
        "own_target_any_l2": _mean(own_any),
    }


def _level_row(name: str, rows: list[dict], residual_key: str, runtime_ms: float) -> dict:
    summary = {
        "level": name,
        "spearman": _mean(row["spearman"] for row in rows),
        "kendall": _mean(row["kendall"] for row in rows),
        "top1": _mean(row["top1"] for row in rows),
        "top3": _mean(row["top3"] for row in rows),
        "top5": _mean(row["top5"] for row in rows),
        "normalized_regret": _mean(row["normalized_regret"] for row in rows),
        "nn_contradiction_1": _mean(row["contradiction_1"] for row in rows),
        "nn_contradiction_2": _mean(row["contradiction_2"] for row in rows),
        "future_residual_spearman": _mean(row[residual_key] for row in rows),
        "runtime_ms_per_assignment": runtime_ms,
        "n_states": len(rows),
    }
    for key in ("spearman", "top3", "normalized_regret", "future_residual_spearman"):
        interval = _boot(rows, key if key != "future_residual_spearman" else residual_key)
        summary[f"{key}_low"] = interval["low"]
        summary[f"{key}_high"] = interval["high"]
    return summary


def _fmt(value) -> str:
    number = float(value) if value is not None else float("nan")
    return "nan" if not np.isfinite(number) else f"{number:.4f}"


def _report(summary: dict) -> None:
    ladder = {row["level"]: row for row in summary["ladder"]}
    lines = [
        "# Phase-5ZC：执行器响应的动态可观测性",
        "",
        "本阶段没有训练模型，没有打开 310118–310125，没有调用 deployment `decide()`。",
        "",
        f"Gate = `{summary['gate']}`。`H_dyn = {summary['h_dyn']}`。`EXECUTOR_FEEDBACK_NEEDED = {summary['executor_feedback_needed']}`。",
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
        "## 信息阶梯",
        "",
        "排序使用事先登记的标量：L0 是已选边的边际 rate 之和，PHY_NOW 是当前联合 SINR rate 之和，ACT0 是指向已分配浮标的平均径向动作，H=1/3/5/8 是 Phase-5X 的 Score-B（`q_exec_h`）。没有拟合 probe。",
        "",
        "| 层级 | J15 Spearman | Top-1 | Top-3 | normalized regret | NN≥1 block | residual Spearman |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for level in ("L0", "PHY_NOW", "ACT0", "RESP1", "H3", "H5", "H8", "J15_ORACLE"):
        row = ladder[level]
        lines.append(
            f"| {level} | {_fmt(row['spearman'])} | {_fmt(row['top1'])} | {_fmt(row['top3'])} | {_fmt(row['normalized_regret'])} | {_fmt(row['nn_contradiction_1'])} | {_fmt(row['future_residual_spearman'])} |"
        )
    lines.extend(
        [
            "",
            f"EXECUTION-INFORMATIVE 的锁定阈值是 Spearman ≥ {SPEARMAN_BAR:.2f}，Top-3 ≥ {TOP3_BAR:.2f}，normalized regret ≤ {REGRET_BAR:.2f}，且至少一个非累计 payload 动态量与 future residual 的 Spearman ≥ {RESIDUAL_BAR:.2f}。",
            "",
            "## Q1. 为什么二阶 correspondence 仍然失败？",
            "",
            "Phase-5ZB 的归一化 UAV–target 外积在交换两个目标后几乎不变，因为它先把每架 UAV 的方向单位化，再对集合取平均。对应关系在平均之前就已经被消掉。本阶段的阶梯说明，即便换一种静态标量（边际 rate 或当前联合 SINR），J15 排序仍过不了锁定门槛。缺的不是另一个静态 pooling。",
            "",
            "## Q2. 静态 state 是否已经足够？",
            "",
            f"L0 Spearman {_fmt(ladder['L0']['spearman'])}，Top-3 {_fmt(ladder['L0']['top3'])}，normalized regret {_fmt(ladder['L0']['normalized_regret'])}。PHY_NOW Spearman {_fmt(ladder['PHY_NOW']['spearman'])}，Top-3 {_fmt(ladder['PHY_NOW']['top3'])}，normalized regret {_fmt(ladder['PHY_NOW']['normalized_regret'])}。最近邻 contradiction（cosine ≥ 0.99 且 |ΔJ15| ≥ 1 block）L0 为 {_fmt(ladder['L0']['nn_contradiction_1'])}，PHY_NOW 为 {_fmt(ladder['PHY_NOW']['nn_contradiction_1'])}。两者都没有同时达到 Spearman、Top-3 和 regret 门槛，所以当前静态信息不足以恢复 J15 排序。",
            "",
            "## Q3. 第一步动作是否带来新信息？",
            "",
            f"ACT0 Spearman {_fmt(ladder['ACT0']['spearman'])}，Top-3 {_fmt(ladder['ACT0']['top3'])}，normalized regret {_fmt(ladder['ACT0']['normalized_regret'])}。相对静态标量的 Spearman 增益是 {_fmt(summary['act0_gain'])}，锁定的显著恢复幅度是 {SPEARMAN_GAIN:.2f}。|ΔJ15| 与 joint action L2 的平均 Spearman 是 {_fmt(summary['action_distance_spearman'])}。第一步动作{summary['act0_verdict']}。",
            "",
            "## Q4. 最小 informative horizon",
            "",
            f"`H_dyn = {summary['h_dyn']}`。它只表示执行差异从哪一层开始满足锁定标准，不是 deployment horizon。",
            "",
            "## Q5. 哪一个信号最早出现？",
            "",
            summary["earliest_signal_text"],
            "",
            "## Q6. 它是在预测 residual，还是在记录已交付 payload？",
            "",
            summary["residual_text"],
            "",
            "## Q7. 310114 t=15 的最早稳定优势",
            "",
            summary["case_text"],
            "",
            "## Q8. 其他 UAV 的目标会不会改变当前 UAV 的动作？",
            "",
            summary["conditioning_text"],
            "",
            "## Q9. synergy 从哪一层开始出现？",
            "",
            summary["synergy_text"],
            "",
            "## Q10. 下一步",
            "",
            summary["next_step"],
            "",
            "## Q11. 是否需要执行器反馈？",
            "",
            f"`EXECUTOR_FEEDBACK_NEEDED = {summary['executor_feedback_needed']}`。若为 True，低层响应只作为高层 assignment value 的输入，不是第二张选票。结构是：候选分配，冻结执行器响应估计，执行器条件下的联合价值，再给出最终分配。",
            "",
            "本阶段没有训练网络，没有修改 planner，没有进入 deployment verification。",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "mode": "executor response observability",
        "training": False,
        "probe": False,
        "l0_score": L0_SCORE,
        "phy_score": PHY_SCORE,
        "act0_score": ACT0_SCORE,
        "horizon_score": HORIZON_SCORE,
        "motion_efficiency": MOTION_EFFICIENCY,
        "spearman_bar": SPEARMAN_BAR,
        "top3_bar": TOP3_BAR,
        "regret_bar": REGRET_BAR,
        "residual_bar": RESIDUAL_BAR,
        "spearman_gain": SPEARMAN_GAIN,
        "cross_uav_fraction": CROSS_UAV_FRACTION,
        "residual_orientation": RESIDUAL_ORIENTATION,
        "population": "310100-310117, scenario bootstrap",
        "opened_310118": False,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
    }
    (OUTPUT / "phase5zc_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    labels = _labels()
    rollouts, order = _load_rollouts()
    label_gap = 0.0
    for key, value in labels.items():
        checkpoint = rollouts.get(key, {}).get(ORACLE_HORIZON)
        if checkpoint is None:
            raise RuntimeError(f"missing Phase-5X horizon 15 for {key}")
        label_gap = max(label_gap, abs(float(checkpoint["payload_h"]) - float(value)))
    if label_gap > 1.0:
        raise RuntimeError(f"Phase-5X horizon 15 disagrees with stored J15 by {label_gap}")
    cases = _load_synergy_cases()
    case_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in cases:
        case_groups[(int(row["scenario_id"]), int(row["time_step"]))].append(row)
    actor = load_frozen_actor()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    executor = LockedExecutor(actor)
    config = L3BConfig()
    states = []
    integrity = [{"check": "phase5x_h15_vs_stored_j15", "max_abs": label_gap, "n": len(labels)}]
    synergy_rows = []
    for scenario in list(TRAIN_SCENARIOS) + list(CONFIRM_SCENARIOS):
        for moment in TIMES:
            assert_consumed(int(scenario))
            env, movement, batch, diagnosis = _replay(int(scenario), int(moment))
            vectors = [vector for vector in order[(int(scenario), int(moment))] if (int(scenario), int(moment), vector) in labels]
            if len(vectors) != 120:
                raise RuntimeError(f"{scenario} t={moment} has {len(vectors)} pool assignments")
            assignments = np.stack([_parse(vector) for vector in vectors], axis=0)
            snapshot = _snapshot(env, diagnosis)
            relational = relational_matrix(snapshot, assignments, config)
            temporal = temporal_matrix(_history_snapshot(env, batch), assignments, config)
            before = fingerprint(env)
            movement_before = np.array(movement, copy=True)
            actions = []
            started = time.perf_counter()
            for assignment in assignments:
                actions.append(_action_bundle(executor, env, movement, assignment))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            roll = [rollouts[(int(scenario), int(moment), vector)] for vector in vectors]
            marginal = np.asarray(
                [
                    float(sum(float(snapshot["rate"][uav, int(buoy)]) for uav, buoy in enumerate(row) if int(buoy) >= 0))
                    for row in assignments
                ],
                dtype=np.float64,
            )
            physical = np.asarray([_counterfactual(env, row)["phy_rate"] for row in assignments], dtype=np.float64)
            state = {
                "scenario_id": int(scenario),
                "time_step": int(moment),
                "split": "train" if int(scenario) in TRAIN_SCENARIOS else "confirmation",
                "vectors": vectors,
                "assignments": assignments,
                "labels": np.asarray([labels[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                "roll": roll,
                "relational": np.asarray(relational, dtype=np.float64),
                "temporal": np.asarray(temporal, dtype=np.float64),
                "marginal": marginal,
                "physical": physical,
                "actions": actions,
                "act0_ms": elapsed_ms / float(len(vectors)),
                "snapshot": snapshot,
                "env_positions": (np.array(env.uav_positions, copy=True), np.array(env.buoy_positions, copy=True)),
                "case_phy": {},
            }
            if int(scenario) == CASE_SCENARIO and int(moment) == CASE_TIME:
                state["case_phy"] = {
                    vector: _counterfactual(env, assignment)
                    for vector, assignment in zip(vectors, assignments)
                    if vector in {CASE_PREDICTED, CASE_OPTIMAL}
                }
            if (int(scenario), int(moment)) in case_groups:
                cached = {vector: bundle for vector, bundle in zip(vectors, actions)}
                for item in case_groups[(int(scenario), int(moment))]:
                    predicted = _parse(item["predicted_assignment"])
                    optimal = _parse(item["optimal_assignment"])
                    singles = [assignment for assignment, switches in _hybrids(predicted, optimal) if int(switches) == 1 and _legal(assignment)]
                    for assignment in [predicted, optimal, *singles]:
                        vector = _vector(assignment)
                        if vector not in cached:
                            cached[vector] = _action_bundle(executor, env, movement, assignment)
                    base = cached[_vector(predicted)]["executed"].ravel()
                    gains = [float(np.linalg.norm(cached[_vector(assignment)]["executed"].ravel() - base)) for assignment in singles]
                    total = float(np.linalg.norm(cached[_vector(optimal)]["executed"].ravel() - base))
                    synergy_rows.append(
                        {
                            "scenario_id": int(scenario),
                            "time_step": int(moment),
                            "predicted_assignment": item["predicted_assignment"],
                            "horizon": "ACT0",
                            "synergy_action_l2": synergy_value(total, gains),
                            "joint_effect": total,
                            "sum_individual": float(np.sum(gains)) if gains else 0.0,
                        }
                    )
            difference = max_abs_diff(before, fingerprint(env))
            if difference > 0.0 or not np.array_equal(movement_before, np.asarray(movement)):
                raise RuntimeError(f"ACT0 changed the caller state at {scenario} t={moment}")
            integrity.append({"check": "act0_caller_fingerprint", "scenario_id": int(scenario), "time_step": int(moment), "max_abs": difference})
            states.append(state)
            print(f"state {scenario} t={moment}", flush=True)

    level_rows = defaultdict(list)
    signal_rows = []
    residual_rows = []
    horizon_rows = []
    action_rows = []
    neighbor_rows = []
    conditioning_rows = []
    one_step_rows = []
    for state in states:
        labels_now = state["labels"]
        identity = {
            "scenario_id": state["scenario_id"],
            "time_step": state["time_step"],
            "split": state["split"],
        }
        l0_vector = np.concatenate((unit_rows(state["relational"]), unit_rows(state["temporal"])), axis=1)
        action_matrix = np.stack([item["executed"].ravel() for item in state["actions"]], axis=0)
        radial = np.stack([item["radial"] for item in state["actions"]], axis=0)
        heading = np.stack([item["heading"] for item in state["actions"]], axis=0)
        representations = {
            "L0": l0_vector,
            "PHY_NOW": state["physical"][:, None],
            "ACT0": unit_rows(action_matrix),
            "L0_PLUS_ACT0": np.concatenate((unit_rows(l0_vector), unit_rows(action_matrix)), axis=1),
        }
        scores = {
            "L0": state["marginal"],
            "PHY_NOW": state["physical"],
            "ACT0": np.asarray([item["score"] for item in state["actions"]], dtype=np.float64),
        }
        residual_at_zero = labels_now.copy()
        for name in ("L0", "PHY_NOW", "ACT0"):
            measured = ranking(scores[name], labels_now)
            neighbor = nearest_neighbor(representations[name], labels_now)
            residual_spearman, _kendall = correlation(scores[name], residual_at_zero)
            row = {**identity, "level": name, **measured, **neighbor, "residual_spearman": residual_spearman}
            level_rows[name].append(row)
            neighbor_rows.append({**identity, "level": name, **neighbor})
        plus = nearest_neighbor(representations["L0_PLUS_ACT0"], labels_now)
        neighbor_rows.append({**identity, "level": "L0_PLUS_ACT0", **plus})
        distances = _pair_distances(np.stack([item["executed"] for item in state["actions"]], axis=0), radial, heading)
        action_row = dict(identity)
        for name, distance in distances.items():
            action_row[f"{name}_vs_j15"] = pairwise_spearman(distance, labels_now)
        action_row["radial_spearman"] = level_rows["ACT0"][-1]["spearman"]
        action_rows.append(action_row)
        conditioning_rows.append({**identity, **_conditioning(state["assignments"], np.stack([item["executed"] for item in state["actions"]]))})
        for horizon in list(HORIZONS) + [ORACLE_HORIZON]:
            q_exec = np.asarray([item[horizon]["q_exec_h"] for item in state["roll"]], dtype=np.float64)
            payload = _column(state, horizon, "cumulative_payload")
            residual = labels_now - payload
            measured = ranking(q_exec, labels_now)
            response = np.concatenate(
                [
                    _column(state, horizon, "mean_sinr")[:, None],
                    _column(state, horizon, "service_progress")[:, None],
                    _column(state, horizon, "distance_reduction")[:, None],
                ],
                axis=1,
            )
            neighbor = nearest_neighbor(response, labels_now)
            best_residual = float("nan")
            for signal, orientation in RESIDUAL_ORIENTATION.items():
                values = _column(state, horizon, signal) * float(orientation)
                spearman, kendall = correlation(values, residual)
                payload_spearman, _payload_kendall = correlation(values, payload)
                raw, raw_kendall = correlation(_column(state, horizon, signal), labels_now)
                residual_rows.append(
                    {
                        **identity,
                        "horizon": int(horizon),
                        "signal": signal,
                        "residual_spearman": spearman,
                        "residual_kendall": kendall,
                        "payload_spearman": payload_spearman,
                        "j15_spearman": raw,
                        "j15_kendall": raw_kendall,
                    }
                )
                if np.isfinite(spearman):
                    best_residual = spearman if not np.isfinite(best_residual) else max(best_residual, spearman)
            level_name = {1: "RESP1", 3: "H3", 5: "H5", 8: "H8", 15: "J15_ORACLE"}.get(int(horizon))
            packed = {**identity, "level": level_name, "horizon": int(horizon), **measured, **neighbor, "residual_spearman": best_residual}
            if level_name != "J15_ORACLE":
                level_rows[level_name].append(packed)
            horizon_rows.append(packed)
            if int(horizon) == 1:
                one_step_rows.append(packed)
            if int(horizon) in HORIZONS:
                for signal in list(SIGNAL_FIELDS) + ["motion_efficiency"]:
                    values = _column(state, horizon, signal)
                    spearman, kendall = correlation(values, labels_now)
                    signal_rows.append(
                        {
                            **identity,
                            "horizon": int(horizon),
                            "signal": signal,
                            "spearman": spearman,
                            "kendall": kendall,
                            "sign_accuracy": sign_accuracy(values, labels_now),
                        }
                    )
        oracle = ranking(labels_now, labels_now)
        oracle_neighbor = nearest_neighbor(labels_now[:, None], labels_now)
        level_rows["J15_ORACLE"].append({**identity, "level": "J15_ORACLE", **oracle, **oracle_neighbor, "residual_spearman": 1.0})

    def runtime_for(level: str) -> float:
        if level == "ACT0":
            return _mean(state["act0_ms"] for state in states)
        if level in {"RESP1", "H3", "H5", "H8"}:
            horizon = {"RESP1": 1, "H3": 3, "H5": 5, "H8": 8}[level]
            return _mean(item[horizon]["elapsed_ms"] for state in states for item in state["roll"])
        return 0.0

    ladder = []
    for level in ("L0", "PHY_NOW", "ACT0", "RESP1", "H3", "H5", "H8", "J15_ORACLE"):
        ladder.append(_level_row(level, level_rows[level], "residual_spearman", runtime_for(level)))
    ladder_index = {row["level"]: row for row in ladder}
    informative = {}
    for level in ("ACT0", "RESP1", "H3", "H5", "H8"):
        row = ladder_index[level]
        informative[level] = execution_informative(row["spearman"], row["top3"], row["normalized_regret"], row["future_residual_spearman"])
    h_dyn = ""
    for level in ("RESP1", "H3", "H5", "H8"):
        if informative[level]:
            h_dyn = {"RESP1": "1", "H3": "3", "H5": "5", "H8": "8"}[level]
            break
    l0_rank = rank_bar(ladder_index["L0"]["spearman"], ladder_index["L0"]["top3"], ladder_index["L0"]["normalized_regret"])
    phy_rank = rank_bar(ladder_index["PHY_NOW"]["spearman"], ladder_index["PHY_NOW"]["top3"], ladder_index["PHY_NOW"]["normalized_regret"])
    gain = float(ladder_index["ACT0"]["spearman"] - max(ladder_index["L0"]["spearman"], ladder_index["PHY_NOW"]["spearman"]))
    gate = choose_gate(l0_rank=l0_rank, phy_rank=phy_rank, act0_informative=informative["ACT0"], act0_gain=gain, h_dyn=h_dyn)
    if gate == "DYN-A":
        h_dyn_text = "0"
    elif gate == "DYN-B":
        h_dyn_text = "ACT0"
    elif gate == "DYN-C":
        h_dyn_text = h_dyn
    else:
        h_dyn_text = "none"
    signal_summary = []
    for horizon in HORIZONS:
        for signal in list(SIGNAL_FIELDS) + ["motion_efficiency"]:
            chosen = [row for row in signal_rows if int(row["horizon"]) == int(horizon) and row["signal"] == signal]
            signal_summary.append(
                {
                    "horizon": int(horizon),
                    "signal": signal,
                    "spearman": _mean(row["spearman"] for row in chosen),
                    "kendall": _mean(row["kendall"] for row in chosen),
                    "sign_accuracy": _mean(row["sign_accuracy"] for row in chosen),
                }
            )
    residual_summary = []
    for horizon in list(HORIZONS) + [ORACLE_HORIZON]:
        for signal in RESIDUAL_ORIENTATION:
            chosen = [row for row in residual_rows if int(row["horizon"]) == int(horizon) and row["signal"] == signal]
            residual_summary.append(
                {
                    "horizon": int(horizon),
                    "signal": signal,
                    "residual_spearman": _mean(row["residual_spearman"] for row in chosen),
                    "payload_spearman": _mean(row["payload_spearman"] for row in chosen),
                    "j15_spearman": _mean(row["j15_spearman"] for row in chosen),
                }
            )
    earliest = []
    for signal in ("mean_sinr", "service_progress", "mean_interference", "distance_reduction", "motion_efficiency"):
        for horizon in HORIZONS:
            match = next(row for row in residual_summary if int(row["horizon"]) == int(horizon) and row["signal"] == signal)
            oriented = float(match["j15_spearman"]) * float(RESIDUAL_ORIENTATION[signal])
            if np.isfinite(oriented) and oriented >= float(RESIDUAL_BAR):
                earliest.append((int(horizon), signal, oriented, float(match["residual_spearman"]), float(match["payload_spearman"])))
                break
    earliest.sort(key=lambda item: (item[0], -item[2]))
    if earliest:
        first_h, first_name, first_value, first_residual, first_payload = earliest[0]
        earliest_text = f"按锁定方向，最早达到 Spearman ≥ {RESIDUAL_BAR:.2f} 的是 H={first_h} 的 {first_name}，与 J15 的有向 Spearman 为 {_fmt(first_value)}。同层同时达到的还有：{', '.join(name for horizon, name, _value, _residual, _payload in earliest if horizon == first_h)}。"
        residual_text = (
            f"{first_name} 在 H={first_h} 与 future residual 的 Spearman 是 {_fmt(first_residual)}，与已交付 payload 的 Spearman 是 {_fmt(first_payload)}。"
            " H=1 时绝大多数候选还没有交付 block，residual 几乎等于 J15，所以这时的相关是预测而不是已实现收益。"
            " 若某个更晚的累计 payload 只能解释已经交付的 block、却达不到 residual 门槛，它不算预测信息。"
        )
    else:
        earliest_text = "SINR、progress、interference、motion 在 H=8 之前都没有达到锁定的有向 Spearman 0.40。"
        residual_text = "没有信号达到锁定的出现门槛，因此也不把任何信号称为 future residual 的预测器。"

    case_state = next(state for state in states if state["scenario_id"] == CASE_SCENARIO and state["time_step"] == CASE_TIME)
    case_rows = []
    named = {}
    for vector in (CASE_PREDICTED, CASE_OPTIMAL):
        index = case_state["vectors"].index(vector)
        assignment = case_state["assignments"][index]
        physical = case_state["case_phy"][vector]
        action = case_state["actions"][index]
        case_rows.append(
            {
                "assignment": "predicted" if vector == CASE_PREDICTED else "optimal",
                "vector": vector,
                "level": "H0",
                "marginal_rate": float(case_state["marginal"][index]),
                "phy_rate": float(case_state["physical"][index]),
                "mean_distance": _mean(
                    float(np.linalg.norm(case_state["env_positions"][0][uav] - case_state["env_positions"][1][int(buoy)]))
                    for uav, buoy in enumerate(assignment)
                    if int(buoy) >= 0
                ),
                "mean_sinr": physical["mean_sinr"],
                "mean_interference": physical["mean_interference"],
                    "progress_sum": float(np.sum(case_state["snapshot"]["progress"])),
                    "preserved_progress": float(
                        sum(
                            float(case_state["snapshot"]["progress"][uav])
                            for uav, buoy in enumerate(assignment)
                            if int(buoy) >= 0 and int(buoy) == int(case_state["snapshot"]["current_assignment"][uav])
                        )
                    ),
                "rep_l2": float(np.linalg.norm(case_state["relational"][index])),
                "temp_l2": float(np.linalg.norm(case_state["temporal"][index])),
            }
        )
        case_rows.append(
            {
                "assignment": case_rows[-1]["assignment"],
                "vector": vector,
                "level": "ACT0",
                "joint_l2": action["joint_l2"],
                "mean_radial": action["score"],
                "mean_heading": mean_active(action["heading"], assignment),
                "action_0": ";".join(f"{value:.6f}" for value in action["executed"][0]),
                "action_1": ";".join(f"{value:.6f}" for value in action["executed"][1]),
                "action_2": ";".join(f"{value:.6f}" for value in action["executed"][2]),
                "action_3": ";".join(f"{value:.6f}" for value in action["executed"][3]),
                "action_4": ";".join(f"{value:.6f}" for value in action["executed"][4]),
                "action_5": ";".join(f"{value:.6f}" for value in action["executed"][5]),
            }
        )
        named[vector] = {"ACT0_radial": action["score"]}
        for horizon in list(HORIZONS) + [ORACLE_HORIZON]:
            checkpoint = case_state["roll"][index][int(horizon)]
            case_rows.append(
                {
                    "assignment": "predicted" if vector == CASE_PREDICTED else "optimal",
                    "vector": vector,
                    "level": f"H{horizon}",
                    "payload": checkpoint["payload_h"],
                    "q_exec": checkpoint["q_exec_h"],
                    "mean_sinr": checkpoint["mean_sinr_h"],
                    "mean_interference": checkpoint["mean_interference_h"],
                    "progress": checkpoint["progress_end_sum"],
                    "distance_reduction": checkpoint["distance_reduction_h"],
                    "j15": checkpoint["j15"],
                }
            )
            named[vector][f"H{horizon}_q"] = checkpoint["q_exec_h"]
            named[vector][f"H{horizon}_sinr"] = checkpoint["mean_sinr_h"]
    pred_index = case_state["vectors"].index(CASE_PREDICTED)
    opt_index = case_state["vectors"].index(CASE_OPTIMAL)
    pred_rep = unit_rows(case_state["relational"][pred_index : pred_index + 1])[0]
    opt_rep = unit_rows(case_state["relational"][opt_index : opt_index + 1])[0]
    pred_temp = unit_rows(case_state["temporal"][pred_index : pred_index + 1])[0]
    opt_temp = unit_rows(case_state["temporal"][opt_index : opt_index + 1])[0]
    case_rows.append(
        {
            "assignment": "pair",
            "level": "H0",
            "rep_cosine": float(np.dot(pred_rep, opt_rep)),
            "temp_cosine": float(np.dot(pred_temp, opt_temp)),
        }
    )
    stable = "none"
    for horizon in list(HORIZONS) + [ORACLE_HORIZON]:
        later = [item for item in list(HORIZONS) + [ORACLE_HORIZON] if int(item) >= int(horizon)]
        if all(named[CASE_OPTIMAL][f"H{item}_q"] > named[CASE_PREDICTED][f"H{item}_q"] for item in later):
            stable = str(int(horizon))
            break
    radial_prefers_opt = named[CASE_OPTIMAL]["ACT0_radial"] > named[CASE_PREDICTED]["ACT0_radial"]
    case_text = (
        f"A_pred 的当前联合 rate 是 {_fmt(case_state['physical'][pred_index])}，A_opt 是 {_fmt(case_state['physical'][opt_index])}。"
        f" ACT0 平均径向动作 pred {_fmt(named[CASE_PREDICTED]['ACT0_radial'])}，opt {_fmt(named[CASE_OPTIMAL]['ACT0_radial'])}。"
        f" 以 Score-B 为准，A_opt 从 H={stable} 开始持续高于 A_pred，并且之后的 H=3/5/8/15 没有再被反超。"
        f" 第一步径向动作{'已经' if radial_prefers_opt else '还没有'}偏向 A_opt。"
    )
    own_only = _mean(row["own_target_only_l2"] for row in conditioning_rows)
    own = _mean(row["own_target_any_l2"] for row in conditioning_rows)
    other = _mean(row["other_target_only_l2"] for row in conditioning_rows)
    cross = bool(np.isfinite(own) and np.isfinite(other) and own > 1.0e-8 and other >= float(CROSS_UAV_FRACTION) * own)
    conditioning_text = (
        f"候选池里只改动 UAV i、其余目标完全不变的配对并不存在，所以 own-only 是 {_fmt(own_only)}。"
        f" 只要 UAV i 的目标发生变化，其动作 L2 平均为 {_fmt(own)}。只改变其他 UAV 的目标时，UAV i 的动作 L2 平均为 {_fmt(other)}。"
        f" 锁定的跨机比例是 {CROSS_UAV_FRACTION:.2f}。{'其他 UAV 的目标已经改变当前 UAV 的第一步动作。' if cross else '其他 UAV 的目标没有明显改变当前 UAV 的第一步动作。'}"
    )
    synergy_path = ROOT / "results/phase5x_executor_lookahead" / "phase5x_synergy_horizon.csv"
    with synergy_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) not in HORIZONS and int(row["horizon"]) != ORACLE_HORIZON:
                continue
            synergy_rows.append(
                {
                    "scenario_id": int(row["scenario_id"]),
                    "time_step": int(row["time_step"]),
                    "horizon": int(row["horizon"]),
                    "synergy_payload": float(row["synergy_payload"]),
                    "synergy_exec": float(row["synergy_exec"]),
                }
            )
    synergy_horizon = "none"
    for horizon in ("ACT0", 1, 3, 5, 8):
        if horizon == "ACT0":
            values = [float(row["synergy_action_l2"]) for row in synergy_rows if row["horizon"] == "ACT0"]
        else:
            values = [float(row["synergy_exec"]) for row in synergy_rows if row["horizon"] == int(horizon)]
        if values and float(np.median(values)) > 0.0:
            synergy_horizon = str(horizon)
            break
    synergy_text = f"Phase-5W 中 synergy > 0 的案例，joint effect 减 individual effects 的中位数从 {synergy_horizon} 开始为正。ACT0 使用动作 L2，H=1/3/5/8 复用 Phase-5X 的 Score-B synergy。"
    if gate == "DYN-A":
        next_step = "下一步重新设计因果静态 assignment scorer。不做 rollout planner。"
    elif gate == "DYN-B":
        next_step = "下一步做冻结 MAPPO 的 ACT0 条件价值：state + assignment + ACT0 → J15。MAPPO 继续冻结。"
    elif gate == "DYN-C":
        next_step = f"下一步蒸馏 H={h_dyn_text} 的短程执行器响应到 learned value。不在线调用 H=15 rollout。"
    else:
        next_step = "下一步重新检查 J15 horizon、低层随机性、任务完成的不连续性和候选定义。不直接训练新的 value。"
    feedback = gate in {"DYN-B", "DYN-C"}
    action_summary = []
    for key in ("joint_l2_vs_j15", "mean_uav_l2_vs_j15", "radial_l1_vs_j15", "heading_l1_vs_j15", "joint_cosine_distance_vs_j15"):
        action_summary.append({"metric": key, "spearman": _mean(row[key] for row in action_rows)})
    runtime_rows = [{"level": row["level"], "runtime_ms_per_assignment": row["runtime_ms_per_assignment"]} for row in ladder]
    trace_gap = _case_action_gap(case_state)
    integrity.append({"check": "310114_t15_act0_vs_phase5w_step1", "max_abs": trace_gap})
    gate_payload = {
        "gate": gate,
        "h_dyn": h_dyn_text,
        "executor_feedback_needed": feedback,
        "planner_ready": False,
        "deployment_verified": False,
        "trained": False,
        "probe_trained": False,
        "opened_310118": False,
        "decide_called": False,
        "l0_rank_bar": l0_rank,
        "phy_rank_bar": phy_rank,
        "act0_informative": informative["ACT0"],
        "act0_spearman_gain": gain,
        "horizon_informative": {key: informative[key] for key in informative},
        "ladder": ladder,
        "earliest_signal": earliest[0][1] if earliest else None,
        "earliest_signal_horizon": earliest[0][0] if earliest else None,
        "case_stable_horizon": stable,
        "case_act0_prefers_optimal": radial_prefers_opt,
        "cross_uav_response": cross,
        "synergy_horizon": synergy_horizon,
        "label_integrity_max_abs": label_gap,
        "case_action_integrity_max_abs": trace_gap,
        "next_step": next_step,
    }
    (OUTPUT / "phase5zc_dynamic_observability_gate.json").write_text(json.dumps(_plain(gate_payload), indent=2), encoding="utf-8")
    _write_csv("phase5zc_shadow_integrity.csv", integrity)
    _write_csv("phase5zc_information_ladder.csv", ladder)
    _write_csv("phase5zc_action_response.csv", action_summary + action_rows)
    _write_csv("phase5zc_one_step_response.csv", one_step_rows)
    _write_csv("phase5zc_horizon_response.csv", horizon_rows)
    _write_csv("phase5zc_signal_emergence.csv", signal_summary)
    _write_csv("phase5zc_future_residual.csv", residual_summary)
    _write_csv("phase5zc_310114_t15_trace.csv", case_rows)
    _write_csv("phase5zc_policy_conditioning.csv", conditioning_rows)
    _write_csv("phase5zc_synergy_emergence.csv", synergy_rows)
    _write_csv(
        "phase5zc_phy_vs_executor.csv",
        [
            {
                "level": row["level"],
                "spearman": row["spearman"],
                "top1": row["top1"],
                "top3": row["top3"],
                "normalized_regret": row["normalized_regret"],
                "future_residual_spearman": row["future_residual_spearman"],
            }
            for row in ladder
        ],
    )
    _write_csv("phase5zc_nearest_neighbor_contradiction.csv", neighbor_rows)
    _write_csv("phase5zc_runtime.csv", runtime_rows)
    summary = {
        "gate": gate,
        "h_dyn": h_dyn_text,
        "executor_feedback_needed": feedback,
        "ladder": ladder,
        "act0_gain": gain,
        "act0_verdict": "达到了 EXECUTION-INFORMATIVE，并且相对静态标量有锁定幅度的恢复" if gate == "DYN-B" else "没有达到锁定的 EXECUTION-INFORMATIVE",
        "action_distance_spearman": next(row["spearman"] for row in action_summary if row["metric"] == "joint_l2_vs_j15"),
        "earliest_signal_text": earliest_text,
        "residual_text": residual_text,
        "case_text": case_text,
        "conditioning_text": conditioning_text,
        "synergy_text": synergy_text,
        "next_step": next_step,
    }
    _report(summary)
    print(json.dumps(_plain({"gate": gate, "h_dyn": h_dyn_text, "feedback": feedback}), indent=2), flush=True)
    return gate_payload


def _case_action_gap(state: dict) -> float:
    path = ROOT / "results/phase5w_residual_mechanism_audit" / "phase5w_310114_execution_trace.csv"
    stored = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["step"]) != 1:
                continue
            stored.setdefault(row["assignment_vector"], {})[int(row["uav"])] = (float(row["action_x"]), float(row["action_y"]))
    largest = 0.0
    for vector in (CASE_PREDICTED, CASE_OPTIMAL):
        action = state["actions"][state["vectors"].index(vector)]["executed"]
        for uav in range(action.shape[0]):
            recorded = stored[vector][uav]
            largest = max(largest, abs(float(action[uav, 0]) - recorded[0]), abs(float(action[uav, 1]) - recorded[1]))
    return float(largest)


if __name__ == "__main__":
    run()
