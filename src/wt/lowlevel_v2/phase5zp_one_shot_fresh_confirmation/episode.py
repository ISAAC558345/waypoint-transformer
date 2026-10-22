"""Closed-loop paired episodes. Selection does not read payload labels."""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import MODEL_POOL_WIDTH
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _combine, _history_snapshot, _scale, _vector
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import _apply_affine, _predict_value
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.fast_path import shipped_h1
from wt.lowlevel_v2.phase5zl_shadow_hierarchical_integration.shadow import assignment_ok, build_task_option
from wt.lowlevel_v2.phase5zm_multiseed_model_policy.consensus import rank_mean_select

from .protocol import HOLD_STEPS, VALUE_SEEDS


class DecisionFailure(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = str(kind)


def _score(model, state: dict, response: np.ndarray, norm: dict) -> np.ndarray:
    features = {"embedded": state["embedded"], "assignments": state["assignments"]}
    scores, _elapsed = _predict_value(model, features, _apply_affine(response, norm), state["relational"])
    return np.asarray(scores, dtype=np.float64)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    value = getattr(result, "statistic", result.correlation)
    return float(value)


def build_decision_state(env: L3BEnv, last_arrival: np.ndarray, last_positions: np.ndarray, frozen, matching, config: L3BConfig, center: np.ndarray, scale: np.ndarray) -> tuple[dict, float, float]:
    """One encoder pass and the frozen top-3 joint pool. No label is attached."""
    started = time.perf_counter()
    info = env.info()
    batch = graph_batch(info, env, last_arrival, np.asarray(last_positions, dtype=np.float32), urgency=shared_task_urgency(env.queue))
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked or "j_mappo" in batch:
        raise DecisionFailure("high_level_inference_exception")
    batch = {key: np.asarray(value).copy() for key, value in batch.items()}
    diagnosis = {
        "queue": np.asarray(env.queue, dtype=np.float64).copy(),
        "urgency": np.asarray(shared_task_urgency(env.queue), dtype=np.float64).copy(),
        "distance": np.asarray(info["distance_uav_buoy"], dtype=np.float64).copy(),
        "rate": np.asarray(info["rate_if_assigned"], dtype=np.float64).copy(),
        "eta": np.asarray(info["eta_slots"], dtype=np.float64).copy(),
        "interference": np.asarray(info["interference_uav_uav"], dtype=np.float64).copy(),
    }
    embedded = {key: value.detach() if torch.is_tensor(value) else value for key, value in _edge_features(frozen, batch).items()}
    additive = _features(frozen, matching, batch)
    encoder_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    logits = additive["logits"].detach().cpu().numpy()
    pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
    assignments = np.asarray(pool["assignments"], dtype=np.int64)
    if assignments.size == 0:
        raise DecisionFailure("candidate_pool_empty")
    relational = _combine(
        relational_matrix(_snapshot(env, diagnosis), assignments, config),
        _scale(temporal_matrix(_history_snapshot(env, batch), assignments, config), center, scale),
    )
    prep_ms = (time.perf_counter() - started) * 1000.0
    state = {
        "env": env,
        "info": info,
        "batch": batch,
        "assignments": assignments,
        "embedded": embedded,
        "relational": relational,
        "candidate_mask": np.asarray(pool["candidate_mask"], dtype=bool),
    }
    return state, float(encoder_ms), float(prep_ms)


def _valid_count(assignments: np.ndarray, num_uavs: int, num_buoys: int) -> int:
    return int(sum(assignment_ok(row, num_uavs, num_buoys)[0] for row in assignments))


def _assigned_mean(values: np.ndarray, assignment: np.ndarray) -> float:
    mask = np.asarray(assignment, dtype=np.int64) >= 0
    chosen = np.asarray(values, dtype=np.float64)[mask]
    return float(np.mean(chosen)) if chosen.size else float("nan")


def _top1_vectors(scores: np.ndarray, assignments: np.ndarray) -> list[str]:
    vectors = []
    for row in np.asarray(scores, dtype=np.float64):
        index = int(np.argsort(-row, kind="stable")[0])
        vectors.append(_vector(assignments[index]))
    return vectors


def _pairwise(scores: np.ndarray) -> float:
    values = []
    for left in range(scores.shape[0]):
        for right in range(left + 1, scores.shape[0]):
            values.append(_spearman(scores[left], scores[right]))
    finite = [item for item in values if np.isfinite(item)]
    return float(np.mean(finite)) if finite else float("nan")


def select_primary(bundle: dict, env: L3BEnv, last_arrival: np.ndarray, last_positions: np.ndarray, last_movement: np.ndarray) -> dict:
    before = fingerprint(env)
    state, encoder_ms, prep_ms = build_decision_state(
        env, last_arrival, last_positions, bundle["frozen"], bundle["matching"], bundle["config"], bundle["center"], bundle["scale"]
    )
    assignments = state["assignments"]
    valid = _valid_count(assignments, env.config.num_uavs, env.config.num_buoys)
    started = time.perf_counter()
    body = shipped_h1(env, assignments, bundle["actor"], last_movement)
    h1_ms = (time.perf_counter() - started) * 1000.0
    if int(np.asarray(body["response"]).shape[-1]) != len(RESPONSE_FEATURES):
        raise DecisionFailure("high_level_inference_exception")
    started = time.perf_counter()
    scores = np.stack([_score(bundle["response_models"][int(seed)], state, body["response"], bundle["norm"]) for seed in VALUE_SEEDS], axis=0)
    value_ms = (time.perf_counter() - started) * 1000.0
    if not np.all(np.isfinite(scores)):
        kind = "model_r1_inf" if np.any(np.isinf(scores)) else "model_r1_nan"
        raise DecisionFailure(kind)
    started = time.perf_counter()
    selected = rank_mean_select(scores, assignments)
    agg_ms = (time.perf_counter() - started) * 1000.0
    legal, reason = assignment_ok(selected["assignment"], env.config.num_uavs, env.config.num_buoys)
    if not legal:
        failure = DecisionFailure("invalid_assignment")
        failure.reason = reason  # type: ignore[attr-defined]
        raise failure
    started = time.perf_counter()
    option = build_task_option(env, state["info"], selected["assignment"])
    option_ms = (time.perf_counter() - started) * 1000.0
    if int(option.remaining_hold) != int(HOLD_STEPS) or not np.array_equal(option.association, option.targets):
        raise DecisionFailure("taskoption_construction_failure")
    after = fingerprint(env)
    mean_ranks = selected["rank_sum"].astype(np.float64) / float(scores.shape[0])
    winner = float(mean_ranks[int(selected["index"])])
    others = np.delete(mean_ranks, int(selected["index"]))
    margin = float(np.min(others) - winner) if others.size else float("nan")
    return {
        "option": option,
        "assignment": np.asarray(selected["assignment"], dtype=np.int64).copy(),
        "vector": _vector(selected["assignment"]),
        "candidate_count": int(assignments.shape[0]),
        "valid_candidate_count": int(valid),
        "encoder_ms": encoder_ms,
        "candidate_ms": prep_ms,
        "true_h1_ms": h1_ms,
        "r1_ms": value_ms,
        "aggregation_ms": agg_ms,
        "taskoption_ms": option_ms,
        "state_contamination": float(max_abs_diff(before, after)) != 0.0,
        "true_h1_steps": 1,
        "seed_top1": _top1_vectors(scores, assignments),
        "majority_top1": int(selected["majority_top1"]),
        "consensus_strength": float(selected["strength"]),
        "mean_rank": float(selected["mean_rank"]),
        "rank_margin": margin,
        "pairwise_spearman": _pairwise(scores),
        "stage": str(selected["stage"]),
        "ranks": [int(item) for item in selected["ranks"]],
        "selected_value": float(np.mean(scores[:, int(selected["index"])])),
        "h1_sinr_mean": _assigned_mean(body["response"][int(selected["index"]), :, 0], selected["assignment"]),
        "h1_rate_mean": _assigned_mean(body["response"][int(selected["index"]), :, 6], selected["assignment"]),
    }


def select_static(bundle: dict, env: L3BEnv, last_arrival: np.ndarray, last_positions: np.ndarray, seed: int) -> dict:
    before = fingerprint(env)
    state, encoder_ms, prep_ms = build_decision_state(
        env, last_arrival, last_positions, bundle["frozen"], bundle["matching"], bundle["config"], bundle["center"], bundle["scale"]
    )
    assignments = state["assignments"]
    zeros = np.zeros((assignments.shape[0], env.config.num_uavs, len(RESPONSE_FEATURES)), dtype=np.float64)
    started = time.perf_counter()
    scores = _score(bundle["static_models"][int(seed)], state, zeros, bundle["norm"])
    value_ms = (time.perf_counter() - started) * 1000.0
    if not np.all(np.isfinite(scores)):
        raise DecisionFailure("model_s_non_finite")
    started = time.perf_counter()
    index = int(np.argsort(-scores, kind="stable")[0])
    agg_ms = (time.perf_counter() - started) * 1000.0
    legal, _reason = assignment_ok(assignments[index], env.config.num_uavs, env.config.num_buoys)
    if not legal:
        raise DecisionFailure("invalid_assignment")
    started = time.perf_counter()
    option = build_task_option(env, state["info"], assignments[index])
    option_ms = (time.perf_counter() - started) * 1000.0
    after = fingerprint(env)
    return {
        "option": option,
        "assignment": np.asarray(assignments[index], dtype=np.int64).copy(),
        "vector": _vector(assignments[index]),
        "candidate_count": int(assignments.shape[0]),
        "valid_candidate_count": _valid_count(assignments, env.config.num_uavs, env.config.num_buoys),
        "encoder_ms": encoder_ms,
        "candidate_ms": prep_ms,
        "true_h1_ms": 0.0,
        "r1_ms": value_ms,
        "aggregation_ms": agg_ms,
        "taskoption_ms": option_ms,
        "state_contamination": float(max_abs_diff(before, after)) != 0.0,
        "true_h1_steps": 0,
        "selected_value": float(scores[index]),
        "h1_sinr_mean": float("nan"),
        "h1_rate_mean": float("nan"),
    }


def _publish_failure(kind: str, option) -> str:
    if option is not None and assignment_ok(option.association, 6, 12)[0]:
        return "KEEP_CURRENT_VALID_TASKOPTION"
    return "ABORT_DECISION_IF_NO_VALID_TASKOPTION"


def run_episode(bundle: dict, scenario: int, system: str, *, static_seed: int | None = None, max_steps: int | None = None) -> dict:
    """One independent environment. The caller supplies the scenario seed."""
    env = L3BEnv()
    info = env.reset(seed=int(scenario))
    executor = LockedExecutor(bundle["actor"])
    if bundle["actor"] is None:
        raise RuntimeError("frozen actor is missing; fly-toward is not a fallback")
    pending = 0
    option = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    initial = fingerprint(env)
    sinr_sum = 0.0
    conflict_sum = 0.0
    progress_sum = 0.0
    interference_sum = 0.0
    queue_sum = 0.0
    steps = 0
    latency_rows = []
    option_rows = []
    consensus_rows = []
    exception_rows = []
    fallback_rows = []
    illegal = 0
    contamination = 0
    nan_count = 0
    inf_count = 0
    unhandled = 0
    fallback_count = 0
    timeout_count = 0
    duplicate_decisions = 0
    legal_decisions = 0
    decision_count = 0
    hold_ok = True
    low_level_target_unchanged = True
    projector_failures = 0
    while int(env.t) < int(env.config.episode_horizon):
        if max_steps is not None and steps >= int(max_steps):
            break
        published = False
        decision = None
        if pending <= 0:
            decision_count += 1
            before_decision = fingerprint(env)
            try:
                if system == "primary":
                    decision = select_primary(bundle, env, last_arrival, last_positions, last_movement)
                else:
                    decision = select_static(bundle, env, last_arrival, last_positions, int(static_seed))
                option = decision["option"]
                pending = int(HOLD_STEPS)
                option.remaining_hold = int(HOLD_STEPS)
                published = True
                legal_decisions += 1
                if np.unique(option.association[option.association >= 0]).size != np.sum(option.association >= 0):
                    duplicate_decisions += 1
            except DecisionFailure as exc:
                action = _publish_failure(exc.kind, option)
                fallback_count += 1
                if exc.kind == "invalid_assignment":
                    illegal += 1
                if exc.kind in {"model_r1_nan", "model_s_non_finite"}:
                    nan_count += 1
                if exc.kind == "model_r1_inf":
                    inf_count += 1
                if action == "KEEP_CURRENT_VALID_TASKOPTION":
                    timeout_count += 1
                    hold_ok = hold_ok and int(option.remaining_hold) != int(HOLD_STEPS)
                row = {
                    "scenario_id": int(scenario),
                    "time_step": int(env.t),
                    "system": system,
                    "static_seed": "" if static_seed is None else int(static_seed),
                    "exception_type": exc.kind,
                    "fallback_action": action,
                    "episode_continued": True,
                    "hold_reset": False,
                }
                exception_rows.append(row)
                fallback_rows.append(dict(row))
            except Exception as exc:
                action = _publish_failure(type(exc).__name__, option)
                fallback_count += 1
                row = {
                    "scenario_id": int(scenario),
                    "time_step": int(env.t),
                    "system": system,
                    "static_seed": "" if static_seed is None else int(static_seed),
                    "exception_type": type(exc).__name__,
                    "fallback_action": action,
                    "episode_continued": True,
                    "hold_reset": False,
                }
                exception_rows.append(row)
                fallback_rows.append(dict(row))
            if max_abs_diff(before_decision, fingerprint(env)) != 0.0:
                contamination += 1
        if published and decision is not None:
            total_ms = float(decision["encoder_ms"] + decision["candidate_ms"] + decision["true_h1_ms"] + decision["r1_ms"] + decision["aggregation_ms"] + decision["taskoption_ms"])
            latency_rows.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(env.t),
                    "system": system,
                    "static_seed": "" if static_seed is None else int(static_seed),
                    "candidate_count": int(decision["candidate_count"]),
                    "valid_candidate_count": int(decision["valid_candidate_count"]),
                    "encoder_ms": float(decision["encoder_ms"]),
                    "candidate_ms": float(decision["candidate_ms"]),
                    "true_h1_ms": float(decision["true_h1_ms"]),
                    "r1_ms": float(decision["r1_ms"]),
                    "aggregation_ms": float(decision["aggregation_ms"]),
                    "taskoption_ms": float(decision["taskoption_ms"]),
                    "total_ms": total_ms,
                    "true_h1_steps": int(decision["true_h1_steps"]),
                }
            )
            chosen = np.asarray(decision["assignment"], dtype=np.int64)
            active = chosen[chosen >= 0]
            option_rows.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(env.t),
                    "system": system,
                    "static_seed": "" if static_seed is None else int(static_seed),
                    "assignment": decision["vector"],
                    "association_equals_assignment": bool(np.array_equal(option.association, chosen)),
                    "idle_code": -1,
                    "legal": True,
                    "duplicate_targets": int(active.size - np.unique(active).size),
                    "remaining_hold_initialized": int(option.remaining_hold),
                    "urgency_is_null": option.urgency is None,
                }
            )
            if system == "primary":
                consensus_rows.append(
                    {
                        "scenario_id": int(scenario),
                        "time_step": int(env.t),
                        "selected_assignment": decision["vector"],
                        "seed_top1": "|".join(decision["seed_top1"]),
                        "majority_top1_count": int(decision["majority_top1"]),
                        "consensus_strength": float(decision["consensus_strength"]),
                        "pairwise_spearman": float(decision["pairwise_spearman"]),
                        "selected_mean_rank": float(decision["mean_rank"]),
                        "top1_top2_rank_margin": float(decision["rank_margin"]),
                        "stage": decision["stage"],
                        "ranks": "|".join(str(item) for item in decision["ranks"]),
                    }
                )
        positions_before = np.asarray(env.uav_positions, dtype=np.float64).copy()
        if option is None:
            executed = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
            association = np.asarray(env.assignment, dtype=np.int64).copy()
        else:
            association = np.asarray(option.association, dtype=np.int64).copy()
            executed, desired, _delta = executor.act(env, option, last_movement)
            status = executor.projector.project(env, desired, option).constraint_status
            if status != "ok":
                projector_failures += 1
                executed = np.zeros_like(desired)
            if not np.array_equal(option.association, association):
                low_level_target_unchanged = False
        try:
            info = env.step(executed, association)
        except Exception as exc:
            unhandled += 1
            exception_rows.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(env.t),
                    "system": system,
                    "static_seed": "" if static_seed is None else int(static_seed),
                    "exception_type": type(exc).__name__,
                    "fallback_action": "STOP_CURRENT_DECISION",
                    "episode_continued": False,
                    "hold_reset": False,
                }
            )
            break
        if not np.array_equal(np.asarray(option.association if option is not None else association), association):
            low_level_target_unchanged = False
        if option is not None:
            pending -= 1
            option.remaining_hold = int(pending)
            if published and int(option.remaining_hold) != int(HOLD_STEPS) - 1:
                hold_ok = False
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
        last_positions = positions_before.astype(np.float32)
        last_movement = np.asarray(executed, dtype=np.float64)
        sinr_sum += float(np.mean(info["sinr"]))
        conflict_sum += float(info["conflict_rate"])
        progress_sum += float(np.mean(info["service_progress"]))
        interference_sum += float(np.mean(info["interference_uav_uav"]))
        queue_sum += float(np.sum(info["queue_blocks"]))
        steps += 1
    payload = float(env.payload_bits)
    block = float(env.config.block_payload_bits)
    return {
        "scenario_id": int(scenario),
        "system": system,
        "static_seed": None if static_seed is None else int(static_seed),
        "initial": initial,
        "payload": payload,
        "residual_queue": float(np.sum(env.queue)),
        "completed_blocks": float(payload / block),
        "average_sinr": float(sinr_sum / max(steps, 1)),
        "conflict_rate": float(conflict_sum / max(steps, 1)),
        "service_progress": float(progress_sum / max(steps, 1)),
        "mean_interference": float(interference_sum / max(steps, 1)),
        "mean_queue": float(queue_sum / max(steps, 1)),
        "dropped_blocks": float(env.dropped),
        "inference_latency_ms": float(np.mean([row["total_ms"] for row in latency_rows])) if latency_rows else float("nan"),
        "fallback_count": int(fallback_count),
        "timeout_count": int(timeout_count),
        "illegal_assignment": int(illegal),
        "state_contamination": int(contamination),
        "nan_final_decision": int(nan_count),
        "inf_final_decision": int(inf_count),
        "unhandled_runtime_exception": int(unhandled),
        "runtime_exception_count": int(len(exception_rows)),
        "duplicate_target_rate": float(duplicate_decisions / max(decision_count, 1)),
        "assignment_legality": float(legal_decisions / max(decision_count, 1)),
        "decision_count": int(decision_count),
        "hold_decrement_ok": bool(hold_ok),
        "low_level_target_unchanged": bool(low_level_target_unchanged),
        "projector_failures": int(projector_failures),
        "latency_rows": latency_rows,
        "option_rows": option_rows,
        "consensus_rows": consensus_rows,
        "exception_rows": exception_rows,
        "fallback_rows": fallback_rows,
    }
