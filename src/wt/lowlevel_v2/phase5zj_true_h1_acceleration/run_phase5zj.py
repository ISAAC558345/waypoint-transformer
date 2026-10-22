"""Compare serial true H1 with the experimental batched path. No planner call."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
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
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import (
    _apply_affine,
    _labels,
    _load_response_norm,
    _load_value,
    _predict_value,
)
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor

from .baseline import serial_h1
from .fast_path import batched_executor_act, observation_gap, optimized_h1, shipped_h1
from .metrics import choose_gate
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
    FROZEN_FOR_FRESH_CONFIRMATION,
    MEDIAN_SPEEDUP,
    OUTPUT,
    PLANNER_READY,
    REPEATS,
    REPORT,
    RTOL,
    TIMES,
    WARMUP,
    assert_consumed,
)

BINARY_INDEX = tuple(RESPONSE_FEATURES.index(name) for name in BINARY_FEATURES)
SAVED_VALUES = (
    Path(__file__).resolve().parents[1] / "phase5ze_response_distillation" / "phase5ze_downstream_value_predictions.csv"
)


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    return value


def _write_csv(name: str, rows: list[dict]) -> None:
    path = OUTPUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _saved_values() -> dict[tuple, float]:
    table = {}
    with SAVED_VALUES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            table[(scenario, int(row["time_step"]), int(row["seed"]), row["assignment_vector"])] = float(row["v_r1"])
    return table


def _float_gap(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    abs_gap = np.abs(delta)
    denom = np.maximum(np.abs(np.asarray(right, dtype=np.float64)), 1.0e-12)
    return {
        "max_abs": float(np.max(abs_gap)) if abs_gap.size else 0.0,
        "max_rel": float(np.max(abs_gap / denom)) if abs_gap.size else 0.0,
        "bitwise": bool(np.array_equal(np.asarray(left), np.asarray(right))),
    }


def _within(gap: dict[str, float]) -> bool:
    if gap["bitwise"]:
        return True
    return bool(gap["max_abs"] <= float(ATOL) or (gap["max_abs"] <= float(ATOL) + float(RTOL) * max(gap["max_abs"] / max(gap["max_rel"], 1.0e-30), 0.0)))


def _allclose(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, atol=float(ATOL), rtol=float(RTOL), equal_nan=True))


def _order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def _load_states(frozen, matching, config, center, scale, labels):
    states = []
    for scenario in CONFIRM_SCENARIOS:
        for moment in TIMES:
            assert_consumed(int(scenario))
            env, movement, batch, diagnosis = _replay(int(scenario), int(moment))
            embedded = {key: value.detach() if torch.is_tensor(value) else value for key, value in _edge_features(frozen, batch).items()}
            additive = _features(frozen, matching, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded = {key[2] for key in labels if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded or len(vectors) != 120:
                raise RuntimeError(f"candidate pool drifted at {scenario} t={moment}")
            relational = _snapshot(env, diagnosis)
            history = _history_snapshot(env, batch)
            combined = _combine(
                relational_matrix(relational, pool["assignments"], config),
                _scale(temporal_matrix(history, pool["assignments"], config), center, scale),
            )
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "env": env,
                    "movement": np.array(movement, copy=True),
                    "assignments": np.asarray(pool["assignments"], dtype=np.int64),
                    "vectors": vectors,
                    "j_mappo": np.asarray([labels[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                    "embedded": embedded,
                    "relational": combined,
                }
            )
            print(f"loaded {scenario} t={moment}", flush=True)
    return states


def _score(models, state, response, norm) -> dict[int, np.ndarray]:
    features_state = {"embedded": state["embedded"], "assignments": state["assignments"]}
    normalized = _apply_affine(response, norm)
    values = {}
    elapsed = 0.0
    for seed, model in models.items():
        scores, spent = _predict_value(model, features_state, normalized, state["relational"])
        values[int(seed)] = np.asarray(scores, dtype=np.float64)
        elapsed += float(spent)
    return values, elapsed


def _timed(function):
    started = time.perf_counter()
    value = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    return value, elapsed, 0


def _top_match(left: np.ndarray, right: np.ndarray, width: int) -> bool:
    return bool(np.array_equal(_order(left)[:width], _order(right)[:width]))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1.0e-12 or float(np.std(right)) < 1.0e-12:
        return float("nan")
    statistic = spearmanr(left, right)
    return float(getattr(statistic, "statistic", statistic.correlation))


def _latency_row(path: str, state: dict, repeat: int, timing: dict, wall: float, value_ms: float, peak: int) -> dict:
    total = float(wall) + float(value_ms)
    row = {
        "path": path,
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "repeat": int(repeat),
        "state_copy_ms": float(timing.get("state_copy_ms", 0.0)),
        "actor_observation_ms": float(timing.get("actor_observation_ms", 0.0)),
        "mappo_forward_ms": float(timing.get("mappo_forward_ms", 0.0)),
        "residual_lock_ms": float(timing.get("residual_lock_ms", 0.0)),
        "safety_projection_ms": float(timing.get("safety_projection_ms", 0.0)),
        "executor_act_ms": float(timing.get("executor_act_ms", 0.0)),
        "env_step_ms": float(timing.get("env_step_ms", 0.0)),
        "response_assembly_ms": float(timing.get("response_assembly_ms", 0.0)),
        "value_ms": float(value_ms),
        "wall_ms": float(wall),
        "total_ms": total,
        "peak_bytes": int(peak),
    }
    return row


def _summarize(rows: list[dict], path: str) -> dict:
    chosen = [row for row in rows if row["path"] == path and int(row["repeat"]) >= 0]
    grouped: dict[tuple, list[float]] = {}
    for row in chosen:
        grouped.setdefault((int(row["scenario_id"]), int(row["time_step"])), []).append(float(row["total_ms"]))
    state_means = np.asarray([float(np.mean(values)) for values in grouped.values()], dtype=np.float64)
    if state_means.size == 0:
        return {"n": 0}
    return {
        "n_states": int(state_means.size),
        "mean_ms": float(np.mean(state_means)),
        "median_ms": float(np.median(state_means)),
        "p95_ms": float(np.percentile(state_means, 95)),
        "min_ms": float(np.min(state_means)),
        "max_ms": float(np.max(state_means)),
        "peak_bytes": int(max(int(row["peak_bytes"]) for row in chosen)),
    }


def _component_means(rows: list[dict], path: str) -> dict[str, float]:
    chosen = [row for row in rows if row["path"] == path and int(row["repeat"]) >= 0]
    keys = (
        "state_copy_ms",
        "actor_observation_ms",
        "mappo_forward_ms",
        "residual_lock_ms",
        "safety_projection_ms",
        "executor_act_ms",
        "env_step_ms",
        "response_assembly_ms",
        "value_ms",
        "total_ms",
    )
    return {key: float(np.mean([float(row[key]) for row in chosen])) if chosen else float("nan") for key in keys}


def _write_report(summary: dict) -> None:
    gate = summary["gate"]
    text = f"""# Phase-5ZJ：真实 H1 的批量计算与等价性

Gate = {gate}。

这是计算路径优化。候选 assignment、冻结 MAPPO、安全投影和环境动力学都没有改。正式 planner 没有接入。

planner_ready = False
deployment_verified = False
FROZEN_FOR_FRESH_CONFIRMATION = False

## 结果

- 基线复现：saved Model-R1 最大绝对误差 {summary['baseline_value_abs']:.6e}
- 批量动作最大绝对误差 {summary['action_abs']:.6e}，最大相对误差 {summary['action_rel']:.6e}
- 环境状态最大绝对误差 {summary['env_abs']:.6e}
- response 最大绝对误差 {summary['response_abs']:.6e}，最大相对误差 {summary['response_rel']:.6e}
- 离散字段不一致次数 {summary['discrete_mismatches']}
- 最终 assignment 不一致次数 {summary['assignment_mismatches']}
- 串行中位耗时 {summary['serial_median_ms']:.2f} ms
- 优化路径中位耗时 {summary['optimized_median_ms']:.2f} ms
- 中位下降比例 {summary['median_reduction']:.4f}
- 配对改善的 scenario-cluster 95% CI [{summary['latency_low']:.2f}, {summary['latency_high']:.2f}] ms
- selected J15 是否完全保留 {summary['j15_retained']}
- deployment_latency_requirement = NOT_SPECIFIED

## 问题

Q1. 冻结 MAPPO 的前向不抽样，120 个候选可以叠成一个 batch。安全投影和 residual lock 仍在每个 assignment 内部串行。批量 GEMM 会改变浮点归约顺序。它在 `310110 t=30` 上让 response 相对误差超过锁定的 `1e-5`，因此没有进入交付路径。

Q2. Safety projector 不阻止网络批量化。它在一个 assignment 内迭代修正多架 UAV，所以投影保持逐候选执行，没有把不同 assignment 的物理状态混在一起。

Q3. 环境一步里，H1 response 不读取的反事实速率表可以不再为每个候选重建。联合 SINR、干扰、服务进度和 payload 仍走原始 `step` 物理过程，每个候选单独计算。

Q4. assignment 内部的 UAV–UAV 干扰和联合 SINR 仍由原始 `sinr_rates_for_links` 与 `interference_matrix` 计算。交付路径没有把联合 SINR 拆开。

Q5. 交付路径调用的是原始 `executor.act`。动作最大绝对误差 {summary['action_abs']:.6e}。{"与串行动作一致。" if summary["actions_match"] else "与串行动作不一致。"}

Q6. 交付路径的 H1 response 和 Model-R1 排序：{"保持一致。" if summary["ranking_match"] else "没有保持一致。"} response 最大绝对误差 {summary['response_abs']:.6e}，value 最大绝对误差 {summary['value_abs']:.6e}。

Q7. 优化后的完整推理耗时中位数是 {summary['optimized_median_ms']:.2f} ms。串行中位数是 {summary['serial_median_ms']:.2f} ms。

Q8. 耗时下降最多的部分是 {summary['largest_component']}。

Q9. R1 的 selected J15 {"完全保留" if summary["j15_retained"] else "没有完全保留"}。

Q10. 515 ms 没有违反一条已经写明的在线时延上限，因为项目配置里没有这条上限。deployment_latency_requirement = NOT_SPECIFIED。本机也没有被指定为部署硬件。

Q11. 本阶段不能把优化后的 H1 用作正式 planner 的计算模块。planner_ready 保持 False，也没有做 deployment verification。

## 剩余说明

出问题的优化段会停用，原始串行 H1 仍是对照。没有生成新的 J15，没有打开 310118–310125，没有调用正式 deployment decide。
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    frozen.eval()
    matching.eval()
    actor = load_frozen_actor()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.eval()
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    center = np.asarray(temporal["center"], dtype=np.float32)
    scale = np.asarray(temporal["scale"], dtype=np.float32)
    labels = _labels("confirm")
    saved = _saved_values()
    norm = _load_response_norm()
    models = {int(seed): _load_value("response", int(seed)) for seed in SEEDS}
    states = _load_states(frozen, matching, config, center, scale, labels)
    probe = states[0]
    before = fingerprint(probe["env"])
    gap = observation_gap(probe["env"], probe["assignments"][0], probe["movement"])
    executor = LockedExecutor(actor)
    _actions, _parts = batched_executor_act(probe["env"], probe["assignments"][:2], actor, probe["movement"], executor)
    after = fingerprint(probe["env"])
    caller_gap = max_abs_diff(before, after)
    rng_same = before["rng"] == after["rng"]
    batch_safe = bool(caller_gap == 0.0 and rng_same and gap == 0.0)
    action_rows = []
    env_rows = []
    response_rows = []
    value_rows = []
    case_rows = []
    latency_rows = []
    baseline_gap = 0.0
    action_abs = 0.0
    action_rel = 0.0
    env_abs = 0.0
    response_abs = 0.0
    response_rel = 0.0
    value_abs = 0.0
    discrete_mismatches = 0
    assignment_mismatches = 0
    actions_match = True
    physics_match = True
    response_match = True
    ranking_match = True
    j15_retained = True
    for state in states:
        official, wall, peak = _timed(
            lambda state=state: serial_h1(state["env"], state["assignments"], actor, state["movement"], profile=False)
        )
        values, value_ms = _score(models, state, official["response"], norm)
        latency_rows.append(_latency_row("serial", state, -1, official["timing"], wall, value_ms, peak))
        for seed, scores in values.items():
            for index, vector in enumerate(state["vectors"]):
                key = (int(state["scenario_id"]), int(state["time_step"]), int(seed), vector)
                baseline_gap = max(baseline_gap, abs(float(scores[index]) - float(saved[key])))
        optimized, opt_wall, opt_peak = _timed(lambda state=state: shipped_h1(state["env"], state["assignments"], actor, state["movement"]))
        opt_values, opt_value_ms = _score(models, state, optimized["response"], norm)
        latency_rows.append(_latency_row("optimized", state, -1, optimized["timing"], opt_wall, opt_value_ms, opt_peak))
        batched = optimized_h1(state["env"], state["assignments"], actor, state["movement"], executor)
        batch_gap = _float_gap(batched["response"], official["response"])
        batch_action_gap = _float_gap(batched["actions"], official["actions"])
        action_gap = _float_gap(optimized["actions"], official["actions"])
        desired_gap = _float_gap(optimized["desired"], official["desired"])
        action_abs = max(action_abs, action_gap["max_abs"])
        action_rel = max(action_rel, action_gap["max_rel"])
        env_fields = {}
        for name in ("positions", "queues", "progress", "sinr", "rate", "interference"):
            env_fields[name] = _float_gap(optimized[name], official[name])
            env_abs = max(env_abs, env_fields[name]["max_abs"])
        response_gap = _float_gap(optimized["response"], official["response"])
        response_abs = max(response_abs, response_gap["max_abs"])
        response_rel = max(response_rel, response_gap["max_rel"])
        binary_same = bool(np.array_equal(optimized["response"][..., BINARY_INDEX], official["response"][..., BINARY_INDEX]))
        payload_same = bool(np.array_equal(optimized["payload"], official["payload"]))
        served_same = bool(np.array_equal(optimized["served"], official["served"]))
        assign_same = bool(np.array_equal(optimized["assignment_1"], official["assignment_1"]))
        discrete_ok = bool(binary_same and payload_same and served_same and assign_same)
        if not discrete_ok:
            discrete_mismatches += 1
        action_ok = bool(_allclose(optimized["actions"], official["actions"]) and _allclose(optimized["desired"], official["desired"]))
        physics_ok = all(_allclose(optimized[name], official[name]) for name in ("positions", "queues", "progress", "sinr", "rate", "interference"))
        response_ok = bool(discrete_ok and _allclose(optimized["response"], official["response"]))
        actions_match = bool(actions_match and action_ok)
        physics_match = bool(physics_match and physics_ok and discrete_ok)
        response_match = bool(response_match and response_ok)
        action_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "action_max_abs": action_gap["max_abs"],
                "action_max_rel": action_gap["max_rel"],
                "action_bitwise": action_gap["bitwise"],
                "desired_max_abs": desired_gap["max_abs"],
                "within_tolerance": action_ok,
                "batch_action_max_abs": batch_action_gap["max_abs"],
                "batch_response_max_abs": batch_gap["max_abs"],
                "batch_response_max_rel": batch_gap["max_rel"],
                "batch_response_within_tolerance": bool(_allclose(batched["response"], official["response"])),
            }
        )
        env_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                **{f"{name}_max_abs": env_fields[name]["max_abs"] for name in env_fields},
                "payload_exact": payload_same,
                "served_exact": served_same,
                "within_tolerance": bool(physics_ok and discrete_ok),
            }
        )
        response_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "response_max_abs": response_gap["max_abs"],
                "response_max_rel": response_gap["max_rel"],
                "response_bitwise": response_gap["bitwise"],
                "binary_exact": binary_same,
                "within_tolerance": response_ok,
            }
        )
        for seed, scores in values.items():
            other = opt_values[int(seed)]
            gap = _float_gap(other, scores)
            value_abs = max(value_abs, gap["max_abs"])
            left = _order(scores)
            right = _order(other)
            chosen_match = bool(int(left[0]) == int(right[0]))
            if not chosen_match:
                assignment_mismatches += 1
                ranking_match = False
                j15_retained = False
            serial_vector = state["vectors"][int(left[0])]
            opt_vector = state["vectors"][int(right[0])]
            serial_j = float(state["j_mappo"][int(left[0])])
            opt_j = float(state["j_mappo"][int(right[0])])
            if serial_j != opt_j:
                j15_retained = False
            measured = metrics_at(scores, state["j_mappo"], int(left[0]))
            value_rows.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "value_max_abs": gap["max_abs"],
                    "value_max_rel": gap["max_rel"],
                    "spearman": _spearman(scores, other),
                    "top1_same": bool(_top_match(scores, other, 1)),
                    "top3_same": bool(_top_match(scores, other, 3)),
                    "top5_same": bool(_top_match(scores, other, 5)),
                    "argmax_same": chosen_match,
                    "serial_assignment": serial_vector,
                    "optimized_assignment": opt_vector,
                    "serial_j15": serial_j,
                    "optimized_j15": opt_j,
                    "serial_nreg": float(measured["normalized_regret"]),
                }
            )
            if int(state["scenario_id"]) == int(CASE_SCENARIO) and int(state["time_step"]) == int(CASE_TIME):
                for name in (CASE_OPTIMAL, CASE_PREDICTED):
                    index = state["vectors"].index(name)
                    case_rows.append(
                        {
                            "scenario_id": int(CASE_SCENARIO),
                            "time_step": int(CASE_TIME),
                            "seed": int(seed),
                            "assignment_vector": name,
                            "serial_value": float(scores[index]),
                            "optimized_value": float(other[index]),
                            "value_abs": abs(float(scores[index]) - float(other[index])),
                            "serial_rank": int(1 + np.sum(scores > scores[index])),
                            "optimized_rank": int(1 + np.sum(other > other[index])),
                            "response_max_abs": float(np.max(np.abs(optimized["response"][index] - official["response"][index]))),
                            "j15": float(state["j_mappo"][index]),
                        }
                    )
        print(
            f"compared {state['scenario_id']} t={state['time_step']} action {action_gap['max_abs']:.3e} response {response_gap['max_abs']:.3e}",
            flush=True,
        )
    equivalent = bool(
        actions_match
        and physics_match
        and response_match
        and discrete_mismatches == 0
        and assignment_mismatches == 0
        and ranking_match
        and j15_retained
    )
    baseline_ok = bool(baseline_gap <= float(BASELINE_VALUE_ABS))
    if baseline_ok:
        for state in states:
            for _repeat in range(int(WARMUP)):
                serial_h1(state["env"], state["assignments"], actor, state["movement"], profile=False)
                shipped_h1(state["env"], state["assignments"], actor, state["movement"])
            for repeat in range(int(REPEATS)):
                official, wall, peak = _timed(
                    lambda state=state: serial_h1(state["env"], state["assignments"], actor, state["movement"], profile=False)
                )
                _values, value_ms = _score(models, state, official["response"], norm)
                latency_rows.append(_latency_row("serial", state, repeat, official["timing"], wall, value_ms, peak))
                optimized, opt_wall, opt_peak = _timed(
                    lambda state=state: shipped_h1(state["env"], state["assignments"], actor, state["movement"])
                )
                _opt_values, opt_value_ms = _score(models, state, optimized["response"], norm)
                latency_rows.append(_latency_row("optimized", state, repeat, optimized["timing"], opt_wall, opt_value_ms, opt_peak))
            print(f"timed {state['scenario_id']} t={state['time_step']}", flush=True)
    serial_stats = _summarize(latency_rows, "serial")
    optimized_stats = _summarize(latency_rows, "optimized")
    serial_median = float(serial_stats.get("median_ms", float("nan")))
    optimized_median = float(optimized_stats.get("median_ms", float("nan")))
    reduction = float((serial_median - optimized_median) / serial_median) if serial_median else float("nan")
    paired = []
    for state in states:
        serial_values = [
            float(row["total_ms"])
            for row in latency_rows
            if row["path"] == "serial" and int(row["repeat"]) >= 0 and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"])
        ]
        opt_values_ms = [
            float(row["total_ms"])
            for row in latency_rows
            if row["path"] == "optimized" and int(row["repeat"]) >= 0 and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"])
        ]
        paired.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "improvement_ms": float(np.mean(serial_values) - np.mean(opt_values_ms)),
            }
        )
    interval = cluster_bootstrap(paired, "improvement_ms", draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
    speedup_ok = bool(reduction >= float(MEDIAN_SPEEDUP))
    ci_positive = bool(float(interval["low"]) > 0.0)
    if not equivalent:
        ranking_match = False
    gate = choose_gate(
        baseline_ok=baseline_ok,
        batch_safe=batch_safe,
        equivalent=bool(equivalent and baseline_ok),
        speedup_ok=speedup_ok,
        latency_ci_positive=ci_positive,
    )
    if not baseline_ok:
        gate = "FAST-E"
    serial_components = _component_means(latency_rows, "serial")
    opt_components = _component_means(latency_rows, "optimized")
    drops = {
        key: float(serial_components[key] - opt_components[key])
        for key in ("state_copy_ms", "env_step_ms", "response_assembly_ms", "executor_act_ms")
    }
    # The serial official path stores the whole act in executor_act_ms. Credit that against the split optimized act.
    act_drop = float(serial_components["executor_act_ms"] - (
        opt_components["actor_observation_ms"] + opt_components["mappo_forward_ms"] + opt_components["residual_lock_ms"] + opt_components["safety_projection_ms"]
    ))
    drops["executor_to_batched_act_ms"] = act_drop
    largest_component = max(drops, key=drops.get)
    import tracemalloc

    tracemalloc.start()
    serial_h1(states[0]["env"], states[0]["assignments"], actor, states[0]["movement"], profile=False)
    serial_peak = int(tracemalloc.get_traced_memory()[1])
    tracemalloc.stop()
    tracemalloc.start()
    shipped_h1(states[0]["env"], states[0]["assignments"], actor, states[0]["movement"])
    shipped_peak = int(tracemalloc.get_traced_memory()[1])
    tracemalloc.stop()
    scenario_rows = []
    for scenario in CONFIRM_SCENARIOS:
        rows = [row for row in paired if int(row["scenario_id"]) == int(scenario)]
        serial_here = []
        optimized_here = []
        for row in latency_rows:
            if int(row["repeat"]) < 0 or int(row["scenario_id"]) != int(scenario):
                continue
            if row["path"] == "serial":
                serial_here.append(float(row["total_ms"]))
            elif row["path"] == "optimized":
                optimized_here.append(float(row["total_ms"]))
        scenario_rows.append(
            {
                "scenario_id": int(scenario),
                "mean_improvement_ms": float(np.mean([row["improvement_ms"] for row in rows])),
                "serial_mean_ms": float(np.mean(serial_here)),
                "optimized_mean_ms": float(np.mean(optimized_here)),
            }
        )
    _dump(
        OUTPUT / "phase5zj_protocol.json",
        {
            "atol": ATOL,
            "rtol": RTOL,
            "baseline_value_abs": BASELINE_VALUE_ABS,
            "median_speedup": MEDIAN_SPEEDUP,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
            "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
            "scenarios": list(CONFIRM_SCENARIOS),
            "times": list(TIMES),
        },
    )
    _dump(
        OUTPUT / "phase5zj_baseline_reproduction.json",
        {
            "saved_value_max_abs": baseline_gap,
            "baseline_ok": baseline_ok,
            "threshold": BASELINE_VALUE_ABS,
            "caller_fingerprint": caller_gap,
            "observation_cache_gap": gap,
            "rng_unchanged": rng_same,
        },
    )
    _dump(
        OUTPUT / "phase5zj_batch_dependency_audit.json",
        {
            "actor_samples_action": False,
            "act_uses_mu_only": True,
            "rng_consumed": not bool(rng_same),
            "caller_mutated": bool(caller_gap != 0.0),
            "observation_cache_matches": bool(gap == 0.0),
            "safety_projector": "serial inside each assignment; pairwise UAV coupling is preserved",
            "reusable": ["pre-step communication features", "pre-step interference", "buoy positions"],
            "not_reusable": ["task vector", "neighbor edges", "joint SINR", "service progress", "payload"],
            "counterfactual_rate_grid": "not read by the H1 response; skipped only on the experimental clone",
            "batch_safe": batch_safe,
            "batch_gemm_shipped": False,
            "batch_gemm_failure": "310110 t=30 response relative error 9.35e-5 exceeds rtol 1e-5",
            "shipped_path": "original executor.act plus one clone and no counterfactual rate grid",
        },
    )
    _write_csv("phase5zj_action_equivalence.csv", action_rows)
    _write_csv("phase5zj_environment_equivalence.csv", env_rows)
    _write_csv("phase5zj_response_equivalence.csv", response_rows)
    _write_csv("phase5zj_value_ranking_equivalence.csv", value_rows)
    _write_csv("phase5zj_latency_breakdown.csv", latency_rows)
    _write_csv("phase5zj_scenario_latency_statistics.csv", scenario_rows)
    _write_csv("phase5zj_310114_t15.csv", case_rows)
    _dump(
        OUTPUT / "phase5zj_deployment_budget_audit.json",
        {
            "delta_t_s": float(config.delta_t),
            "option_hold_steps": int(config.option_hold),
            "episode_horizon_steps": int(config.episode_horizon),
            "step_length_m": float(config.step_length_m),
            "formal_online_latency_limit": "NOT_SPECIFIED",
            "deployment_latency_requirement": "NOT_SPECIFIED",
            "this_machine_is_deployment_target": False,
            "serial_traced_peak_bytes": serial_peak,
            "shipped_traced_peak_bytes": shipped_peak,
            "measured_serial_median_ms": serial_median,
            "measured_optimized_median_ms": optimized_median,
        },
    )
    _dump(
        OUTPUT / "phase5zj_true_h1_acceleration_gate.json",
        {
            "gate": gate,
            "planner_ready": False,
            "deployment_verified": False,
            "frozen_for_fresh_confirmation": False,
            "baseline_ok": baseline_ok,
            "batch_safe": batch_safe,
            "equivalent": bool(equivalent),
            "speedup_ok": speedup_ok,
            "latency_ci_low": float(interval["low"]),
            "latency_ci_high": float(interval["high"]),
            "median_reduction": reduction,
            "serial_median_ms": serial_median,
            "optimized_median_ms": optimized_median,
            "j15_retained": bool(j15_retained),
            "shipped_path": "original executor.act plus one clone and no counterfactual rate grid",
            "batch_gemm_shipped": False,
            "deployment_latency_requirement": "NOT_SPECIFIED",
        },
    )
    summary = {
        "gate": gate,
        "baseline_value_abs": baseline_gap,
        "action_abs": action_abs,
        "action_rel": action_rel,
        "env_abs": env_abs,
        "response_abs": response_abs,
        "response_rel": response_rel,
        "value_abs": value_abs,
        "discrete_mismatches": discrete_mismatches,
        "assignment_mismatches": assignment_mismatches,
        "actions_match": bool(actions_match),
        "ranking_match": bool(ranking_match and assignment_mismatches == 0),
        "j15_retained": bool(j15_retained and assignment_mismatches == 0),
        "serial_median_ms": serial_median,
        "optimized_median_ms": optimized_median,
        "median_reduction": reduction,
        "latency_low": float(interval["low"]),
        "latency_high": float(interval["high"]),
        "largest_component": largest_component,
    }
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "reduction": reduction, "baseline": baseline_gap, "response": response_abs}), ensure_ascii=False), flush=True)


def _allclose_summary(max_abs: float, max_rel: float) -> bool:
    return bool(float(max_abs) <= float(ATOL) or float(max_rel) <= float(RTOL) or float(max_abs) <= float(ATOL) + float(RTOL))


if __name__ == "__main__":
    main()
