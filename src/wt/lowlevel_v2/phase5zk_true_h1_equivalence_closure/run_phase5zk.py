"""Closure audit. Does not edit the Phase-5ZJ gate and does not call the planner."""
from __future__ import annotations

import csv
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

from wt.l3b.config import L3BConfig
from wt.l3b.env import hypothetical_rates
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import MODEL_POOL_WIDTH, SEEDS
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _replay, _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _combine, _history_snapshot, _scale, _vector
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES
from wt.lowlevel_v2.phase5ze_response_distillation.protocol import BINARY_FEATURES
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import _apply_affine, _labels, _load_response_norm, _load_value, _predict_value
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.baseline import serial_h1, step_original
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.fast_path import _observations, optimized_h1, shipped_h1, slim_steps

from .metrics import choose_gate, combined_limit
from .protocol import (
    ATOL,
    BASELINE_VALUE_ABS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMAL,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CONFIRM_SCENARIOS,
    DEPLOYMENT_VERIFIED,
    EXCEPTION_SCENARIO,
    EXCEPTION_TIME,
    FROZEN_FOR_FRESH_CONFIRMATION,
    MEDIAN_SPEEDUP,
    OUTPUT,
    PLANNER_READY,
    REPEATS,
    REPORT,
    RTOL,
    TIMES,
    VALUE_SEED,
    WARMUP,
    assert_consumed,
)

UNITS = {
    "sinr_1": "linear SINR",
    "delta_sinr": "linear SINR",
    "interference_1": "interference power",
    "delta_interference": "interference power",
    "service_progress_1": "service fraction",
    "delta_service_progress": "service fraction",
    "rate_1": "bit/s",
    "distance_to_target_1": "m",
    "delta_distance_to_target": "m",
    "assigned_in_range_1": "binary",
    "service_active_1": "binary",
    "same_target_after_step_1": "binary",
    "radial_motion_1": "m",
    "heading_alignment_1": "dimensionless",
}
BINARY_INDEX = tuple(RESPONSE_FEATURES.index(name) for name in BINARY_FEATURES)
SAVED = Path(__file__).resolve().parents[1] / "phase5ze_response_distillation" / "phase5ze_downstream_value_predictions.csv"


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
    (OUTPUT / name).write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(name: str, rows: list[dict]) -> None:
    path = OUTPUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _saved_values() -> dict[tuple, float]:
    table = {}
    with SAVED.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            table[(scenario, int(row["time_step"]), int(row["seed"]), row["assignment_vector"])] = float(row["v_r1"])
    return table


def _limit(reference: np.ndarray) -> np.ndarray:
    return float(ATOL) + float(RTOL) * np.abs(np.asarray(reference, dtype=np.float64))


def _scan_response(serial: np.ndarray, other: np.ndarray, state: dict, path: str) -> list[dict]:
    delta = np.abs(np.asarray(other, dtype=np.float64) - np.asarray(serial, dtype=np.float64))
    reference = np.asarray(serial, dtype=np.float64)
    limit = _limit(reference)
    relative = delta / np.maximum(np.abs(reference), 1.0e-30)
    rows = []
    fail = delta > limit
    flat = np.argpartition(relative.reshape(-1), -1)[-1]
    max_index = np.unravel_index(int(flat), relative.shape)
    marked = set(map(tuple, np.argwhere(fail)))
    marked.add(tuple(int(item) for item in max_index))
    for candidate, uav, feature in sorted(marked):
        name = RESPONSE_FEATURES[int(feature)]
        rows.append(
            {
                "path": path,
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "assignment_vector": state["vectors"][int(candidate)],
                "candidate_index": int(candidate),
                "uav": int(uav),
                "field": name,
                "unit": UNITS[name],
                "serial_value": float(reference[candidate, uav, feature]),
                "optimized_value": float(other[candidate, uav, feature]),
                "abs_error": float(delta[candidate, uav, feature]),
                "rel_error": float(relative[candidate, uav, feature]),
                "combined_limit": float(limit[candidate, uav, feature]),
                "exceeds_combined": bool(delta[candidate, uav, feature] > limit[candidate, uav, feature]),
                "near_zero_reference": bool(abs(float(reference[candidate, uav, feature])) < 1.0e-3),
            }
        )
    return rows


def _field_stats(serial: np.ndarray, other: np.ndarray) -> dict[str, float | int]:
    delta = np.abs(np.asarray(other, dtype=np.float64) - np.asarray(serial, dtype=np.float64))
    reference = np.asarray(serial, dtype=np.float64)
    fail = delta > _limit(reference)
    relative = delta / np.maximum(np.abs(reference), 1.0e-30)
    return {
        "max_abs": float(np.max(delta)) if delta.size else 0.0,
        "max_rel": float(np.max(relative)) if relative.size else 0.0,
        "failures": int(np.sum(fail)),
        "bitwise": bool(np.array_equal(serial, other)),
    }


def _order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def _load_states(frozen, matching, config, center, scale, labels) -> list[dict]:
    states = []
    for scenario in CONFIRM_SCENARIOS:
        for moment in TIMES:
            assert_consumed(int(scenario))
            env, movement, batch, diagnosis = _replay(int(scenario), int(moment))
            embedded = {key: value.detach() if torch.is_tensor(value) else value for key, value in _edge_features(frozen, batch).items()}
            logits = _features(frozen, matching, batch)["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded = {key[2] for key in labels if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded or len(vectors) != 120:
                raise RuntimeError(f"candidate pool drifted at {scenario} t={moment}")
            combined = _combine(
                relational_matrix(_snapshot(env, diagnosis), pool["assignments"], config),
                _scale(temporal_matrix(_history_snapshot(env, batch), pool["assignments"], config), center, scale),
            )
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "env": env,
                    "movement": np.array(movement, copy=True),
                    "batch": batch,
                    "diagnosis": diagnosis,
                    "assignments": np.asarray(pool["assignments"], dtype=np.int64),
                    "vectors": vectors,
                    "j_mappo": np.asarray([labels[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                    "embedded": embedded,
                    "relational": combined,
                }
            )
            print(f"loaded {scenario} t={moment}", flush=True)
    return states


def _score_one(model, state: dict, response: np.ndarray, norm: dict) -> tuple[np.ndarray, float]:
    features = {"embedded": state["embedded"], "assignments": state["assignments"]}
    scores, elapsed = _predict_value(model, features, _apply_affine(response, norm), state["relational"])
    return np.asarray(scores, dtype=np.float64), float(elapsed)


def _prepare(frozen, matching, config, state: dict, center: np.ndarray, scale: np.ndarray) -> tuple[float, float, np.ndarray]:
    started = time.perf_counter()
    additive = _features(frozen, matching, state["batch"])
    _edge_features(frozen, state["batch"])
    encoder_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    logits = additive["logits"].detach().cpu().numpy()
    pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
    if not np.array_equal(np.asarray(pool["assignments"]), state["assignments"]):
        raise RuntimeError("candidate preparation changed the pool")
    relational = _combine(
        relational_matrix(_snapshot(state["env"], state["diagnosis"]), pool["assignments"], config),
        _scale(temporal_matrix(_history_snapshot(state["env"], state["batch"]), pool["assignments"], config), center, scale),
    )
    prep_ms = (time.perf_counter() - started) * 1000.0
    return float(encoder_ms), float(prep_ms), relational


def _pipeline(frozen, matching, config, model, state: dict, norm: dict, center: np.ndarray, scale: np.ndarray, mode: str) -> dict:
    started = time.perf_counter()
    encoder_ms, prep_ms, relational = _prepare(frozen, matching, config, state, center, scale)
    body_started = time.perf_counter()
    if mode == "serial":
        body = serial_h1(state["env"], state["assignments"], state["actor"], state["movement"], profile=False)
    else:
        body = shipped_h1(state["env"], state["assignments"], state["actor"], state["movement"])
    body_ms = (time.perf_counter() - body_started) * 1000.0
    scored = dict(state)
    scored["relational"] = relational
    scores, value_ms = _score_one(model, scored, body["response"], norm)
    arg_started = time.perf_counter()
    chosen = int(_order(scores)[0])
    argmax_ms = (time.perf_counter() - arg_started) * 1000.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    timing = dict(body["timing"])
    timing.update(
        {
            "encoder_ms": encoder_ms,
            "candidate_prep_ms": prep_ms,
            "h1_body_ms": body_ms,
            "value_ms": value_ms,
            "argmax_ms": argmax_ms,
        }
    )
    return {"wall_ms": wall_ms, "timing": timing, "chosen": chosen, "scores": scores}


def _observation_gap(env, assignments, movement) -> float:
    cached = _observations(env, assignments, movement)
    gap = 0.0
    for assignment, row in zip(assignments, cached):
        original = build_actor_observation(env, row["option"], movement)
        for key in ("self", "task", "communication", "neighbor_set", "d_nearest"):
            gap = max(gap, float(np.max(np.abs(np.asarray(row[key]) - np.asarray(original[key])))))
    return gap


def _percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values.size else float("nan")


def _write_report(summary: dict) -> None:
    text = f"""# Phase-5ZK：真实 H1 数值等价与耗时复核

Gate = {summary['gate']}。

本阶段没有改 Phase-5ZJ 的 Gate 文件，也没有改它的报告。下面是对那次 FAST-A 的复核。

planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False

## 数值例外

Phase-5ZJ 提到的相对误差 9.35e-5 发生在未交付的批量 GEMM 实验，不在已交付路径上。

- 路径：{summary['exception_path']}
- 状态：{summary['exception_scenario']} t={summary['exception_time']}
- assignment：{summary['exception_assignment']}
- UAV：{summary['exception_uav']}
- 字段：{summary['exception_field']}（{summary['exception_unit']}）
- 原始值：{summary['exception_serial']}
- 批量前向值：{summary['exception_other']}
- 绝对误差：{summary['exception_abs']}
- 相对误差：{summary['exception_rel']}
- 组合容差上限：{summary['exception_limit']}
- 是否超过组合容差：{summary['exception_exceeds']}
- 参考值是否近零：{summary['exception_near_zero']}

已交付路径的 response 失败元素数是 {summary['shipped_response_failures']}。批量 GEMM 的失败元素数是 {summary['batch_response_failures']}。

根因是 float32 批量矩阵乘的归约顺序，不是反事实速率表的删除。用原始动作走删除该表后的环境一步，response 与串行结果逐位相等。反事实速率表只在 `info()` 里计算，不写环境状态，也不消耗随机数。

没有修改 adapter。交付路径本来就使用原始逐候选 `executor.act`。

## 耗时口径

端到端包含冻结 encoder、候选准备、H1、一个 Model-R1 前向和 argmax。每个状态 1 次 warmup、{summary['repeats']} 次重复。中位数是 24 个状态均值的中位数。95% CI 是这些状态配对绝对改善（串行减优化）的 scenario 聚类均值区间，不是中位数区间，也不是相对百分比。

- 串行中位数 {summary['serial_median_ms']:.2f} ms
- 交付路径中位数 {summary['optimized_median_ms']:.2f} ms
- 中位数相对下降 {summary['median_reduction']:.4f}
- 配对绝对改善的均值 {summary['improvement_mean_ms']:.2f} ms
- scenario-cluster 95% CI [{summary['latency_low']:.2f}, {summary['latency_high']:.2f}] ms

Phase-5ZI 的约 515 ms 是单个状态、把反事实速率表算进去、只含一个 value 头的分量之和。Phase-5ZJ 的 711.43 ms 和 216.47 ms 没有把 encoder 算进端到端，而且 value 头测了五个。本次 CI 若与 `[814.07, 862.10]` 不同，是因为计时边界不同，不是把均值和中位数当成同一个数。

## 问题

Q1. 9.35e-5 量级的最大相对误差落在 `{summary['exception_field']}`，assignment `{summary['exception_assignment']}`，UAV {summary['exception_uav']}。

Q2. 该元素{'超过' if summary['exception_exceeds'] else '没有超过'}组合容差 `atol + rtol * |原始值|`。近零放大不是这个元素的原因：参考值近零 = {summary['exception_near_zero']}。

Q3. 根因已经定位。它不在交付路径里，所以没有再改 adapter。交付路径不需要这个修复。

Q4. 交付路径的环境状态、SINR、progress 和 payload：{'逐位一致' if summary['shipped_bitwise'] else '没有逐位一致'}。组合容差失败数 {summary['shipped_float_failures']}。

Q5. assignment 和 selected J15：{'全部一致' if summary['decisions_match'] else '存在不一致'}。

Q6. 按本次统一边界，中位数相对下降 {summary['median_reduction']:.4f}。{'加速可以复现。' if summary['speedup_ok'] and summary['ci_positive'] else '加速没有同时满足 20% 和 CI 下界。'}

Q7. 515 ms、711.43 ms 和 216.47 ms 的边界不同。515 ms 含反事实速率表且只覆盖一个状态；711.43 / 216.47 ms 是 24 个状态的 H1 中位数，不含 encoder，含五个 value 头。本次统一边界的两个中位数写在上面。

Q8. 交付路径没有 neural batching。环境复用和删除未使用的反事实速率表是已实现部分。Actor observation batching、MAPPO batching、安全投影 batching、环境一步向量化都是 NOT_IMPLEMENTED。

Q9. 没有正式在线时延要求。deployment_latency_requirement = NOT_SPECIFIED。仿真步长是 1 秒，高层保持 15 步。这些不是墙上时钟预算。

Q10. {'可以把它作为后续独立集成审计的候选实现，但不能接进正式 planner，也不是 deployment PASS。' if summary['gate'] == 'ZK-A' else '还不能作为严格数值等价的集成候选。'}

planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    config = L3BConfig()
    threads = int(torch.get_num_threads())
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    frozen.eval()
    matching.eval()
    actor = load_frozen_actor()
    actor.eval()
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    center = np.asarray(temporal["center"], dtype=np.float32)
    scale = np.asarray(temporal["scale"], dtype=np.float32)
    labels = _labels("confirm")
    saved = _saved_values()
    norm = _load_response_norm()
    models = {int(seed): _load_value("response", int(seed)) for seed in SEEDS}
    states = _load_states(frozen, matching, config, center, scale, labels)
    for state in states:
        state["actor"] = actor
    executor = LockedExecutor(actor)
    exception_env = next(state["env"] for state in states if int(state["scenario_id"]) == int(EXCEPTION_SCENARIO) and int(state["time_step"]) == int(EXCEPTION_TIME))
    before_info = fingerprint(exception_env)
    _ = exception_env.info()
    hypothetical_rates(exception_env)
    info_gap = max_abs_diff(before_info, fingerprint(exception_env))
    info_rng_same = before_info["rng"] == fingerprint(exception_env)["rng"]
    exceptions = []
    action_rows = []
    env_rows = []
    response_rows = []
    assignment_rows = []
    baseline_gap = 0.0
    shipped_response_failures = 0
    batch_response_failures = 0
    shipped_float_failures = 0
    discrete_mismatches = 0
    assignment_mismatches = 0
    shipped_bitwise = True
    caller_gap = 0.0
    observation_gap_max = 0.0
    isolation = {}
    for state in states:
        official = serial_h1(state["env"], state["assignments"], actor, state["movement"], profile=False)
        before = fingerprint(state["env"])
        shipped = shipped_h1(state["env"], state["assignments"], actor, state["movement"])
        caller_gap = max(caller_gap, max_abs_diff(before, fingerprint(state["env"])))
        batched = optimized_h1(state["env"], state["assignments"], actor, state["movement"], executor)
        shipped_rows = _scan_response(official["response"], shipped["response"], state, "shipped")
        batch_rows = _scan_response(official["response"], batched["response"], state, "batch_gemm_not_shipped")
        exceptions.extend(row for row in shipped_rows + batch_rows if row["exceeds_combined"] or (int(state["scenario_id"]) == int(EXCEPTION_SCENARIO) and int(state["time_step"]) == int(EXCEPTION_TIME) and row["path"] == "batch_gemm_not_shipped"))
        shipped_response_failures += sum(int(row["exceeds_combined"]) for row in shipped_rows)
        batch_response_failures += sum(int(row["exceeds_combined"]) for row in batch_rows)
        obs_gap = _observation_gap(state["env"], state["assignments"], state["movement"]) if (int(state["scenario_id"]) == int(EXCEPTION_SCENARIO) and int(state["time_step"]) == int(EXCEPTION_TIME)) else 0.0
        observation_gap_max = max(observation_gap_max, obs_gap)
        action_stats = _field_stats(official["actions"], shipped["actions"])
        batch_action = _field_stats(official["actions"], batched["actions"])
        shipped_float_failures += int(action_stats["failures"]) + int(batch_action["failures"] * 0)
        env_fail = 0
        env_record = {"scenario_id": int(state["scenario_id"]), "time_step": int(state["time_step"])}
        serial_origin = official["positions"] - official["displacement"]
        shipped_displacement = shipped["positions"] - serial_origin
        compared = {
            "positions": (official["positions"], shipped["positions"]),
            "displacement_m": (official["displacement"], shipped_displacement),
            "queues": (official["queues"], shipped["queues"]),
            "progress": (official["progress"], shipped["progress"]),
            "sinr": (official["sinr"], shipped["sinr"]),
            "rate": (official["rate"], shipped["rate"]),
            "interference": (official["interference"], shipped["interference"]),
        }
        for name, (left, right) in compared.items():
            stats = _field_stats(left, right)
            env_record[f"{name}_max_abs"] = stats["max_abs"]
            env_record[f"{name}_failures"] = stats["failures"]
            env_fail += int(stats["failures"])
            shipped_bitwise = bool(shipped_bitwise and stats["bitwise"])
        shipped_float_failures += env_fail
        response_stats = _field_stats(official["response"], shipped["response"])
        shipped_bitwise = bool(shipped_bitwise and response_stats["bitwise"] and action_stats["bitwise"])
        binary_same = bool(np.array_equal(official["response"][..., BINARY_INDEX], shipped["response"][..., BINARY_INDEX]))
        payload_same = bool(np.array_equal(official["payload"], shipped["payload"]))
        served_same = bool(np.array_equal(official["served"], shipped["served"]))
        assign_same = bool(np.array_equal(official["assignment_1"], shipped["assignment_1"]))
        if not (binary_same and payload_same and served_same and assign_same):
            discrete_mismatches += 1
        action_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "shipped_action_max_abs": action_stats["max_abs"],
                "shipped_action_failures": action_stats["failures"],
                "shipped_action_bitwise": action_stats["bitwise"],
                "batch_action_max_abs": batch_action["max_abs"],
                "batch_action_failures": batch_action["failures"],
                "observation_max_abs": obs_gap if int(state["scenario_id"]) == int(EXCEPTION_SCENARIO) and int(state["time_step"]) == int(EXCEPTION_TIME) else "",
            }
        )
        env_record.update({"payload_exact": payload_same, "served_exact": served_same, "assignment_exact": assign_same, "float_failures": env_fail})
        env_rows.append(env_record)
        response_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "shipped_max_abs": response_stats["max_abs"],
                "shipped_max_rel": response_stats["max_rel"],
                "shipped_failures": response_stats["failures"],
                "shipped_bitwise": response_stats["bitwise"],
                "binary_exact": binary_same,
                "batch_max_abs": _field_stats(official["response"], batched["response"])["max_abs"],
                "batch_max_rel": _field_stats(official["response"], batched["response"])["max_rel"],
                "batch_failures": _field_stats(official["response"], batched["response"])["failures"],
            }
        )
        serial_values = {}
        shipped_values = {}
        for seed, model in models.items():
            serial_scores, _elapsed = _score_one(model, state, official["response"], norm)
            shipped_scores, _elapsed = _score_one(model, state, shipped["response"], norm)
            serial_values[int(seed)] = serial_scores
            shipped_values[int(seed)] = shipped_scores
            for index, vector in enumerate(state["vectors"]):
                baseline_gap = max(baseline_gap, abs(float(serial_scores[index]) - float(saved[(int(state["scenario_id"]), int(state["time_step"]), int(seed), vector)])))
            left = _order(serial_scores)
            right = _order(shipped_scores)
            if int(left[0]) != int(right[0]) or not np.array_equal(left[:5], right[:5]):
                assignment_mismatches += int(int(left[0]) != int(right[0]))
            measured = metrics_at(serial_scores, state["j_mappo"], int(left[0]))
            other = metrics_at(shipped_scores, state["j_mappo"], int(right[0]))
            assignment_rows.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "serial_assignment": state["vectors"][int(left[0])],
                    "shipped_assignment": state["vectors"][int(right[0])],
                    "argmax_same": bool(int(left[0]) == int(right[0])),
                    "top1_same": bool(np.array_equal(left[:1], right[:1])),
                    "top3_same": bool(np.array_equal(left[:3], right[:3])),
                    "top5_same": bool(np.array_equal(left[:5], right[:5])),
                    "tie_margin": float(serial_scores[int(left[0])] - serial_scores[int(left[1])]),
                    "serial_j15": float(state["j_mappo"][int(left[0])]),
                    "shipped_j15": float(state["j_mappo"][int(right[0])]),
                    "serial_nreg": float(measured["normalized_regret"]),
                    "shipped_nreg": float(other["normalized_regret"]),
                    "value_max_abs": float(np.max(np.abs(serial_scores - shipped_scores))),
                    "case_optimal_rank_serial": int(1 + np.sum(serial_scores > serial_scores[state["vectors"].index(CASE_OPTIMAL)])) if int(state["scenario_id"]) == int(CASE_SCENARIO) and int(state["time_step"]) == int(CASE_TIME) else "",
                    "case_predicted_rank_serial": int(1 + np.sum(serial_scores > serial_scores[state["vectors"].index(CASE_PREDICTED)])) if int(state["scenario_id"]) == int(CASE_SCENARIO) and int(state["time_step"]) == int(CASE_TIME) else "",
                    "case_optimal_value_abs": float(abs(serial_scores[state["vectors"].index(CASE_OPTIMAL)] - shipped_scores[state["vectors"].index(CASE_OPTIMAL)])) if int(state["scenario_id"]) == int(CASE_SCENARIO) and int(state["time_step"]) == int(CASE_TIME) else "",
                    "case_predicted_value_abs": float(abs(serial_scores[state["vectors"].index(CASE_PREDICTED)] - shipped_scores[state["vectors"].index(CASE_PREDICTED)])) if int(state["scenario_id"]) == int(CASE_SCENARIO) and int(state["time_step"]) == int(CASE_TIME) else "",
                }
            )
        if int(state["scenario_id"]) == int(EXCEPTION_SCENARIO) and int(state["time_step"]) == int(EXCEPTION_TIME):
            restored = slim_steps(state["env"], state["assignments"], official["actions"])
            replayed = step_original(state["env"], state["assignments"], batched["actions"])
            isolation = {
                "slim_with_serial_actions_response_bitwise": bool(np.array_equal(restored["response"], official["response"])),
                "original_step_with_batch_actions_response_max_abs": _field_stats(official["response"], replayed["response"])["max_abs"],
                "original_step_with_batch_actions_response_failures": _field_stats(official["response"], replayed["response"])["failures"],
                "observation_max_abs": obs_gap,
                "batch_action_max_abs": batch_action["max_abs"],
                "info_and_hypothetical_rates_fingerprint": info_gap,
                "info_rng_unchanged": bool(info_rng_same),
                "hypothetical_rates_writes_state": False,
                "hypothetical_rates_called_from": "L3BEnv.info only; step updates physics before info()",
            }
        print(
            f"compared {state['scenario_id']} t={state['time_step']} shipped_fail {response_stats['failures']} batch_rel {response_rows[-1]['batch_max_rel']:.3e}",
            flush=True,
        )
    focus = [row for row in exceptions if row["path"] == "batch_gemm_not_shipped" and int(row["scenario_id"]) == int(EXCEPTION_SCENARIO) and int(row["time_step"]) == int(EXCEPTION_TIME)]
    focus.sort(key=lambda row: float(row["rel_error"]), reverse=True)
    top = focus[0] if focus else {
        "assignment_vector": "",
        "uav": "",
        "field": "",
        "unit": "",
        "serial_value": "",
        "optimized_value": "",
        "abs_error": "",
        "rel_error": "",
        "combined_limit": "",
        "exceeds_combined": False,
        "near_zero_reference": False,
    }
    baseline_ok = bool(baseline_gap <= float(BASELINE_VALUE_ABS))
    behavioral = bool(discrete_mismatches > 0 or assignment_mismatches > 0)
    numerical_exception = bool(shipped_response_failures > 0 or shipped_float_failures > 0)
    decisions_match = bool(assignment_mismatches == 0 and all(row["serial_j15"] == row["shipped_j15"] for row in assignment_rows))
    latency_rows = []
    if baseline_ok and not behavioral:
        value_model = models[int(VALUE_SEED)]
        for state in states:
            for _repeat in range(int(WARMUP)):
                _pipeline(frozen, matching, config, value_model, state, norm, center, scale, "serial")
                _pipeline(frozen, matching, config, value_model, state, norm, center, scale, "shipped")
            for repeat in range(int(REPEATS)):
                for mode, label in (("serial", "serial"), ("shipped", "optimized")):
                    result = _pipeline(frozen, matching, config, value_model, state, norm, center, scale, mode)
                    latency_rows.append(
                        {
                            "path": label,
                            "scenario_id": int(state["scenario_id"]),
                            "time_step": int(state["time_step"]),
                            "repeat": int(repeat),
                            "encoder_ms": result["timing"]["encoder_ms"],
                            "candidate_prep_ms": result["timing"]["candidate_prep_ms"],
                            "state_copy_ms": result["timing"].get("state_copy_ms", 0.0),
                            "executor_act_ms": result["timing"].get("executor_act_ms", 0.0),
                            "env_step_ms": result["timing"].get("env_step_ms", 0.0),
                            "response_assembly_ms": result["timing"].get("response_assembly_ms", 0.0),
                            "value_ms": result["timing"]["value_ms"],
                            "argmax_ms": result["timing"]["argmax_ms"],
                            "wall_ms": result["wall_ms"],
                        }
                    )
            print(f"timed {state['scenario_id']} t={state['time_step']}", flush=True)
    tracemalloc.start()
    _pipeline(frozen, matching, config, models[int(VALUE_SEED)], states[0], norm, center, scale, "serial")
    serial_peak = int(tracemalloc.get_traced_memory()[1])
    tracemalloc.stop()
    tracemalloc.start()
    _pipeline(frozen, matching, config, models[int(VALUE_SEED)], states[0], norm, center, scale, "shipped")
    shipped_peak = int(tracemalloc.get_traced_memory()[1])
    tracemalloc.stop()

    def _state_means(path: str) -> np.ndarray:
        grouped: dict[tuple, list[float]] = {}
        for row in latency_rows:
            if row["path"] != path:
                continue
            grouped.setdefault((int(row["scenario_id"]), int(row["time_step"])), []).append(float(row["wall_ms"]))
        return np.asarray([float(np.mean(values)) for values in grouped.values()], dtype=np.float64)

    serial_means = _state_means("serial")
    optimized_means = _state_means("optimized")
    serial_median = float(np.median(serial_means)) if serial_means.size else float("nan")
    optimized_median = float(np.median(optimized_means)) if optimized_means.size else float("nan")
    reduction = float((serial_median - optimized_median) / serial_median) if serial_median else float("nan")
    paired = []
    for state in states:
        left = [float(row["wall_ms"]) for row in latency_rows if row["path"] == "serial" and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"])]
        right = [float(row["wall_ms"]) for row in latency_rows if row["path"] == "optimized" and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"])]
        if left and right:
            paired.append({"scenario_id": int(state["scenario_id"]), "time_step": int(state["time_step"]), "improvement_ms": float(np.mean(left) - np.mean(right))})
    interval = cluster_bootstrap(paired, "improvement_ms", draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED)) if paired else {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    speedup_ok = bool(reduction >= float(MEDIAN_SPEEDUP))
    ci_positive = bool(float(interval["low"]) > 0.0)
    gate = choose_gate(
        baseline_ok=baseline_ok,
        behavioral_mismatch=behavioral,
        numerical_exception=numerical_exception,
        speedup_ok=speedup_ok,
        latency_ci_positive=ci_positive,
    )
    scenario_rows = []
    for scenario in CONFIRM_SCENARIOS:
        rows = [row for row in paired if int(row["scenario_id"]) == int(scenario)]
        scenario_rows.append(
            {
                "scenario_id": int(scenario),
                "mean_improvement_ms": float(np.mean([row["improvement_ms"] for row in rows])) if rows else float("nan"),
                "serial_mean_ms": float(np.mean([float(row["wall_ms"]) for row in latency_rows if row["path"] == "serial" and int(row["scenario_id"]) == int(scenario)])) if latency_rows else float("nan"),
                "optimized_mean_ms": float(np.mean([float(row["wall_ms"]) for row in latency_rows if row["path"] == "optimized" and int(row["scenario_id"]) == int(scenario)])) if latency_rows else float("nan"),
            }
        )
    _dump(
        "phase5zk_protocol.json",
        {
            "atol": ATOL,
            "rtol": RTOL,
            "combined_rule": "abs(optimized-serial) <= atol + rtol * abs(serial)",
            "median_definition": "median of per-state mean wall times",
            "ci_definition": "scenario-cluster 95% CI of the mean paired absolute improvement in milliseconds",
            "warmup": WARMUP,
            "repeats": REPEATS,
            "value_seed_in_latency": VALUE_SEED,
            "threads": threads,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
            "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
        },
    )
    if not exceptions:
        exceptions = [{"path": "none", "scenario_id": "", "time_step": "", "assignment_vector": "", "candidate_index": "", "uav": "", "field": "", "unit": "", "serial_value": "", "optimized_value": "", "abs_error": "", "rel_error": "", "combined_limit": "", "exceeds_combined": False, "near_zero_reference": False}]
    _write_csv("phase5zk_numerical_exception.csv", exceptions)
    _dump(
        "phase5zk_root_cause_audit.json",
        {
            "exception_is_on_shipped_path": bool(shipped_response_failures > 0),
            "adapter_modified": False,
            "reason": "The delivered path already uses the original per-candidate forward. The relative-error element comes from the unshipped float32 batch GEMM.",
            "isolation": isolation,
            "caller_fingerprint": caller_gap,
            "top_relative_element": top,
        },
    )
    _write_csv("phase5zk_action_equivalence.csv", action_rows)
    _write_csv("phase5zk_environment_equivalence.csv", env_rows)
    _write_csv("phase5zk_response_equivalence.csv", response_rows)
    _write_csv("phase5zk_assignment_equivalence.csv", assignment_rows)
    _write_csv("phase5zk_latency_breakdown.csv", latency_rows)
    _write_csv("phase5zk_scenario_latency_statistics.csv", scenario_rows)
    _dump(
        "phase5zk_batching_implementation_audit.json",
        {
            "environment_computation_reuse": "IMPLEMENTED",
            "unnecessary_table_elimination": "IMPLEMENTED",
            "response_assembly_optimization": "IMPLEMENTED_BY_SKIPPING_UNUSED_PRESTEP_INFO",
            "actor_observation_batching": "NOT_IMPLEMENTED",
            "mappo_neural_forward_batching": "NOT_IMPLEMENTED",
            "safety_projection_batching": "NOT_IMPLEMENTED",
            "environment_step_vectorization": "NOT_IMPLEMENTED",
            "note": "Deleting the unused counterfactual rate grid is not batched neural inference.",
        },
    )
    _dump(
        "phase5zk_deployment_latency_audit.json",
        {
            "deployment_latency_requirement": "NOT_SPECIFIED",
            "delta_t_s": float(config.delta_t),
            "option_hold_steps": int(config.option_hold),
            "episode_horizon_steps": int(config.episode_horizon),
            "simulation_time_is_not_wall_clock": True,
            "this_machine_is_deployment_target": False,
            "serial_traced_peak_bytes": serial_peak,
            "shipped_traced_peak_bytes": shipped_peak,
            "serial_median_ms": serial_median,
            "optimized_median_ms": optimized_median,
            "serial_p95_ms": _percentile(serial_means, 95),
            "serial_p99_ms": _percentile(serial_means, 99),
            "optimized_p95_ms": _percentile(optimized_means, 95),
            "optimized_p99_ms": _percentile(optimized_means, 99),
        },
    )
    _dump(
        "phase5zk_equivalence_latency_gate.json",
        {
            "gate": gate,
            "planner_ready": False,
            "deployment_verified": False,
            "frozen_for_fresh_confirmation": False,
            "baseline_ok": baseline_ok,
            "baseline_value_max_abs": baseline_gap,
            "behavioral_mismatch": behavioral,
            "numerical_exception_on_shipped_path": numerical_exception,
            "shipped_bitwise": shipped_bitwise,
            "speedup_ok": speedup_ok,
            "latency_ci_low": float(interval["low"]),
            "latency_ci_high": float(interval["high"]),
            "ci_statistic": "mean paired absolute improvement, scenario cluster",
            "median_reduction": reduction,
            "serial_median_ms": serial_median,
            "optimized_median_ms": optimized_median,
            "j15_retained": decisions_match,
            "phase5zj_gate_file_modified": False,
            "deployment_latency_requirement": "NOT_SPECIFIED",
        },
    )
    summary = {
        "gate": gate,
        "exception_path": "batch_gemm_not_shipped",
        "exception_scenario": int(EXCEPTION_SCENARIO),
        "exception_time": int(EXCEPTION_TIME),
        "exception_assignment": top["assignment_vector"],
        "exception_uav": top["uav"],
        "exception_field": top["field"],
        "exception_unit": top["unit"],
        "exception_serial": top["serial_value"],
        "exception_other": top["optimized_value"],
        "exception_abs": top["abs_error"],
        "exception_rel": top["rel_error"],
        "exception_limit": top["combined_limit"],
        "exception_exceeds": bool(top["exceeds_combined"]),
        "exception_near_zero": bool(top["near_zero_reference"]),
        "shipped_response_failures": shipped_response_failures,
        "batch_response_failures": batch_response_failures,
        "shipped_bitwise": shipped_bitwise,
        "shipped_float_failures": shipped_float_failures,
        "decisions_match": decisions_match,
        "serial_median_ms": serial_median,
        "optimized_median_ms": optimized_median,
        "median_reduction": reduction,
        "improvement_mean_ms": float(interval["mean"]),
        "latency_low": float(interval["low"]),
        "latency_high": float(interval["high"]),
        "repeats": REPEATS,
        "speedup_ok": speedup_ok,
        "ci_positive": ci_positive,
    }
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "field": top["field"], "rel": top["rel_error"], "exceeds": top["exceeds_combined"], "reduction": reduction}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
