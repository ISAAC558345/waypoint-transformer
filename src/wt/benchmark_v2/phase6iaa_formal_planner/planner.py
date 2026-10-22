"""Official high-level entry. Scheme C scores the frozen k=6 pool; MAPPO only moves."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import feature_matrices, proposal_scores
from wt.benchmark_v2.phase6f_system_bringup.episode import _contamination, _fingerprint
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.faults import (
    file_sha256,
    handle_episode_fault,
    option_is_retainable,
    verify_startup_hashes,
)
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    ACTIVE_CANDIDATE_K,
    EPOCH6_CKPT,
    HOLD,
    SCHEME_C_CKPT,
    assert_episode_scenario,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import hgnn_tensors, step_observation
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import GRAPH_KEYS
from wt.benchmark_v2.phase6iw_inference_adapter.interface import SchemeCAdapter
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import additive_assignment, bind_keys, frozen_candidate_keys
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_payload, config_sha256
from wt.htw_phase100.contract import TaskOption
from wt.htw_phase9.const import EDGE_UB_DIM
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase5l_joint_encoder.model import append_queue_edge


def _sha256(path) -> str:
    return file_sha256(path)


def startup_record(policy: str) -> dict:
    """Hashes and the candidate contract. This does not load a fallback pool."""
    if policy not in ("scheme_c", "g1_rate"):
        raise RuntimeError(f"unknown high_level_policy {policy}")
    if int(ACTIVE_CANDIDATE_K) != 6 or GENERATOR != "G1_RATE":
        raise RuntimeError("active candidate contract is not G1_RATE k=6")
    payload = {
        "high_level_policy": policy,
        "scheme_c_checkpoint": str(SCHEME_C_CKPT),
        "scheme_c_sha256": _sha256(SCHEME_C_CKPT),
        "epoch6_sha256": _sha256(EPOCH6_CKPT),
        "mappo_checkpoint": str(primary_checkpoint()),
        "mappo_sha256": _sha256(primary_checkpoint()),
        "candidate_config_sha256": config_sha256(),
        "candidate_config": config_payload(),
        "active_k": int(ACTIVE_CANDIDATE_K),
        "hold": int(HOLD),
        "low_level": "frozen MAPPO-V2 deterministic mean",
    }
    verify_startup_hashes(payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def graph_from_env(env: L3BBitQueueEnvV2, position_prev: np.ndarray, predecessor_time: int | None) -> dict:
    """Same node and edge contract as load_decision, read from the live environment."""
    arrays = state_arrays(env)
    observed = step_observation(env, position_prev, predecessor_time)
    arrays["service_progress"] = np.asarray(observed["service_progress"], dtype=np.float64)
    arrays["uav_step_displacement_m"] = np.asarray(observed["uav_step_displacement_m"], dtype=np.float64)
    built = hgnn_tensors(arrays, env.config)
    edge = np.asarray(built["edge"], dtype=np.float32)
    if int(edge.shape[-1]) != int(EDGE_UB_DIM) + 1:
        raise RuntimeError("assembled edge width does not match scheme C")
    graph = {
        "uav": np.asarray(built["uav"], dtype=np.float32)[None, ...],
        "buoy": np.asarray(built["buoy"], dtype=np.float32)[None, ...],
        "uav_mask": np.ones((1, built["uav"].shape[0]), dtype=bool),
        "buoy_mask": np.ones((1, built["buoy"].shape[0]), dtype=bool),
        "global": np.asarray(built["global"], dtype=np.float32)[None, ...],
        "edge": edge[None, ..., : int(EDGE_UB_DIM)],
        "edge_uu": np.asarray(built["edge_uu"], dtype=np.float32)[None, ...],
    }
    if tuple(graph) != GRAPH_KEYS:
        raise RuntimeError("graph contains a field scheme C does not consume")
    conditioned = append_queue_edge(graph)
    if int(conditioned["edge"].shape[-1]) != 4:
        raise RuntimeError("queue-edge width is not 4")
    if not np.allclose(conditioned["edge"], edge[None, ...], atol=1.0e-6, rtol=0.0):
        raise RuntimeError("queue-edge append does not match the assembled edge")
    return graph


def accept_scored_row(scores: np.ndarray, legal: np.ndarray) -> int | None:
    """Stable top score, or no choice when the pool contract fails."""
    values = np.asarray(scores, dtype=np.float64)
    flags = np.asarray(legal, dtype=bool)
    if values.size == 0 or values.shape != flags.shape:
        return None
    if not np.all(np.isfinite(values)) or not np.all(flags):
        return None
    return int(np.argsort(-values, kind="stable")[0])


@dataclass
class Selection:
    status: str
    reason: str
    assignment: tuple[int, ...] | None
    option: TaskOption | None
    score_index: int
    candidate_count: int
    pool_seconds: float
    score_seconds: float
    scheme_c_called: int
    pool_hash: str = ""


class FormalPlanner:
    """One frozen policy. A failed choice does not substitute another generator."""

    def __init__(self, policy: str) -> None:
        if policy not in ("scheme_c", "g1_rate"):
            raise RuntimeError(f"unknown high_level_policy {policy}")
        self.policy = policy
        self.startup = startup_record(policy)
        self.actor = load_primary()
        self.actor.eval()
        self.scorer = SchemeCAdapter() if policy == "scheme_c" else None
        if self.scorer is not None:
            self.scorer.checkpoint_sha256 = self.startup["scheme_c_sha256"]
        self.scheme_c_calls = 0

    def select_assignment(self, env: L3BBitQueueEnvV2, position_prev: np.ndarray, predecessor_time: int | None) -> Selection:
        before = _fingerprint(env, None)
        token = parameter_token(self.actor)
        try:
            selection = self._choose(env, position_prev, predecessor_time)
        except Exception as exc:
            selection = Selection("aborted", f"{type(exc).__name__}: {exc}", None, None, -1, 0, 0.0, 0.0, 0)
        if _contamination(before, _fingerprint(env, None)) != 0 or parameter_token(self.actor) != token:
            raise RuntimeError("CONTAMINATION")
        return selection

    def _choose(self, env: L3BBitQueueEnvV2, position_prev: np.ndarray, predecessor_time: int | None) -> Selection:
        pool_started = _clock()
        keys, _raw_count = frozen_candidate_keys(env)
        pool_seconds = _clock() - pool_started
        pool_hash = _pool_hash(keys)
        if not keys:
            return Selection("aborted", "no legal candidate", None, None, -1, 0, pool_seconds, 0.0, 0, pool_hash)
        bindings, _mapping = bind_keys(keys, n_uavs=int(env.config.num_uavs), n_buoys=int(env.config.num_buoys))
        if any(not binding.legal for binding in bindings):
            return Selection("aborted", "illegal candidate", None, None, -1, len(bindings), pool_seconds, 0.0, 0, pool_hash)
        matrix = np.asarray([binding.assignment for binding in bindings], dtype=np.int64)
        if self.policy == "g1_rate":
            totals = _g1_totals(env, keys)
            index = accept_scored_row(totals, np.ones(len(keys), dtype=bool))
            called = 0
            score_seconds = 0.0
            if index is None:
                return Selection("aborted", "baseline scores are not finite", None, None, -1, len(keys), pool_seconds, 0.0, 0, pool_hash)
            chosen = additive_assignment(keys, proposal_scores(feature_matrices(env))[GENERATOR])
            if chosen != keys[index]:
                return Selection("aborted", "baseline tie-break diverged", None, None, -1, len(keys), pool_seconds, 0.0, 0, pool_hash)
        else:
            if self.scorer is None:
                return Selection("aborted", "scheme C is not loaded", None, None, -1, len(keys), pool_seconds, 0.0, 0, pool_hash)
            graph = graph_from_env(env, position_prev, predecessor_time)
            scored = self.scorer.infer(graph, matrix)
            self.scheme_c_calls += 1
            called = 1
            score_seconds = float(scored.seconds)
            index = accept_scored_row(scored.scores, np.asarray([binding.legal for binding in bindings]))
            if index is None or int(index) != int(scored.chosen_index):
                return Selection("aborted", "scheme C score was rejected", None, None, -1, len(keys), pool_seconds, score_seconds, called, pool_hash)
            chosen = bindings[index].assignment
            if list(chosen) != list(scored.chosen_assignment):
                return Selection("aborted", "score index is not bound", None, None, -1, len(keys), pool_seconds, score_seconds, called, pool_hash)
        option = make_option(env, np.asarray(chosen, dtype=np.int64))
        if int(option.remaining_hold) != int(HOLD):
            return Selection("aborted", "hold is not 15", None, None, -1, len(keys), pool_seconds, score_seconds, called, pool_hash)
        if not np.array_equal(np.asarray(option.association), np.asarray(chosen, dtype=np.int64)):
            return Selection("aborted", "TaskOption association diverged", None, None, -1, len(keys), pool_seconds, score_seconds, called, pool_hash)
        return Selection("selected", "", tuple(int(item) for item in chosen), option, int(index), len(keys), pool_seconds, score_seconds, called, pool_hash)


def _pool_hash(keys: list[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(keys, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _g1_totals(env: L3BBitQueueEnvV2, keys: list[tuple[int, ...]]) -> np.ndarray:
    normalized = proposal_scores(feature_matrices(env))[GENERATOR]
    totals = np.zeros(len(keys), dtype=np.float64)
    for row, key in enumerate(keys):
        for uav, buoy in enumerate(key):
            if int(buoy) >= 0:
                totals[row] += float(normalized[uav, int(buoy)])
    return totals


def _clock() -> float:
    import time

    return time.perf_counter()


def run_closed_episode(
    scenario_id: int,
    planner: FormalPlanner,
    *,
    stop_time: int | None = None,
    allow_holdout: bool = False,
    policy_path: str | Path | None = None,
    fault_log: str | Path | None = None,
) -> dict:
    """Execute one episode. A failed new choice keeps no fresh assignment."""
    assert_episode_scenario(int(scenario_id), allow_holdout=allow_holdout)
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    rng_after_reset = str(env.rng.bit_generator.state)
    limit = int(env.config.episode_horizon if stop_time is None else stop_time)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    option: TaskOption | None = None
    pending = 0
    decisions = []
    anomalies = []
    served_sum = 0.0
    sinr = []
    interference = []
    rate = []
    per_uav_active = np.zeros(env.config.num_uavs, dtype=np.int64)
    idle_steps = 0
    interventions = 0
    illegal = 0
    accounting_failures = 0
    nan_inf = 0
    hold_ok = True
    max_conservation_error = 0.0
    started = _clock()
    while int(env.t) < limit:
        if pending <= 0:
            try:
                selection = planner.select_assignment(env, position_prev, predecessor)
            except Exception as exc:
                anomalies.append({"time": int(env.t), "reason": f"{type(exc).__name__}: {exc}"})
                break
            if selection.status != "selected" or selection.option is None or selection.assignment is None:
                anomalies.append({"time": int(env.t), "reason": selection.reason or selection.status})
                break
            option = selection.option
            pending = int(HOLD)
            decisions.append(
                {
                    "time": int(env.t),
                    "policy": planner.policy,
                    "assignment": list(selection.assignment),
                    "score_index": int(selection.score_index),
                    "candidate_count": int(selection.candidate_count),
                    "pool_seconds": float(selection.pool_seconds),
                    "score_seconds": float(selection.score_seconds),
                    "scheme_c_called": int(selection.scheme_c_called),
                    "pool_hash": selection.pool_hash,
                    "idle_slots": int(sum(int(item) < 0 for item in selection.assignment)),
                }
            )
        if option is None:
            anomalies.append({"time": int(env.t), "reason": "no TaskOption"})
            break
        if not option_is_retainable(
            option,
            decisions[-1]["assignment"],
            pending,
            n_uavs=int(env.config.num_uavs),
            n_buoys=int(env.config.num_buoys),
        ):
            anomalies.append({"time": int(env.t), "reason": "TaskOption is not retainable"})
            break
        hold_before = int(option.remaining_hold)
        position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
        predecessor = int(env.t)
        association = np.asarray(option.association, dtype=np.int64).copy()
        try:
            movement, intervened, _residual = _actor_move(env, planner.actor, option, env._phase6h_last)
            info = env.step(movement, association)
        except Exception as exc:
            accounting_failures += 1
            anomalies.append({"time": int(env.t), "reason": f"{type(exc).__name__}: {exc}"})
            break
        option_changed = not np.array_equal(np.asarray(option.association, dtype=np.int64), association)
        if option_changed:
            illegal += 1
        error = float(np.max(np.abs(env.conservation_error_bits())))
        max_conservation_error = max(max_conservation_error, error)
        if error > float(env.tol_bits) or not np.isfinite(float(env.payload_bits)):
            accounting_failures += 1
            anomalies.append({"time": int(env.t), "reason": "accounting"})
            break
        env._phase6h_last = np.asarray(movement, dtype=np.float64)
        option.remaining_hold -= 1
        pending -= 1
        interventions += int(intervened)
        if int(option.remaining_hold) != hold_before - 1:
            hold_ok = False
        served = np.asarray(info["served_bits"], dtype=np.float64)
        served_sum += float(np.sum(served))
        sinr.append(float(np.mean(info["sinr"])))
        rate.append(float(np.mean(info["rate_bps"])))
        interference.append(float(np.mean(communication_features(env)["total_interference"])))
        for uav, buoy in enumerate(association):
            if int(buoy) < 0:
                idle_steps += 1
            elif float(served[int(buoy)]) > float(env.tol_bits):
                per_uav_active[uav] += 1
        if not np.all(np.isfinite(served)) or not np.all(np.isfinite(info["sinr"])) or not np.all(np.isfinite(info["rate_bps"])):
            nan_inf += 1
            anomalies.append({"time": int(env.t), "reason": "nonfinite"})
            break
        decision_changed = not np.array_equal(association, np.asarray(decisions[-1]["assignment"], dtype=np.int64))
        if decision_changed:
            illegal += 1
        if option_changed or decision_changed:
            anomalies.append({"time": int(env.t), "reason": "TaskOption is no longer valid"})
            break
    elapsed = _clock() - started
    if not hold_ok:
        anomalies.append({"time": int(env.t), "reason": "hold contract failed"})
    fault_recovery = None
    if anomalies:
        reason = "; ".join(str(item.get("reason", "")) for item in anomalies)
        fault_recovery = handle_episode_fault(
            reason=reason,
            scenario_id=int(scenario_id),
            time_index=int(env.t),
            policy_path=None if policy_path is None else Path(policy_path),
            log_path=None if fault_log is None else Path(fault_log),
        )
    payload = float(env.payload_bits)
    payload_ok = abs(payload - served_sum) <= float(env.tol_bits)
    retargets = 0
    for previous, current in zip(decisions, decisions[1:]):
        retargets += sum(int(left) != int(right) for left, right in zip(previous["assignment"], current["assignment"]))
    rng_after = str(env.rng.bit_generator.state)
    complete = (
        int(env.t) == limit
        and not anomalies
        and illegal == 0
        and accounting_failures == 0
        and nan_inf == 0
        and hold_ok
        and payload_ok
        and bool(decisions)
    )
    return {
        "scenario_id": int(scenario_id),
        "policy": planner.policy,
        "stop_time": limit,
        "complete": bool(complete),
        "payload_bits": payload,
        "served_bits": float(served_sum),
        "payload_matches_served": int(payload_ok),
        "completed_blocks": float(np.sum(env.completed_block_equivalent)),
        "queue_bits": float(np.sum(env.q_bits)),
        "sinr_mean": float(np.mean(sinr)) if sinr else 0.0,
        "rate_mean": float(np.mean(rate)) if rate else 0.0,
        "interference_mean": float(np.mean(interference)) if interference else 0.0,
        "service_steps_per_uav": [int(item) for item in per_uav_active],
        "service_steps_mean": float(np.mean(per_uav_active)) if per_uav_active.size else 0.0,
        "idle_assignments": int(sum(int(row["idle_slots"]) for row in decisions)),
        "idle_steps": int(idle_steps),
        "decision_retargets": int(retargets),
        "max_conservation_error": float(max_conservation_error),
        "reset_seed": int(scenario_id),
        "rng_after_reset": rng_after_reset,
        "rng_after_episode": rng_after,
        "rng_unchanged": int(rng_after == rng_after_reset),
        "terminal_time": int(env.t),
        "high_level_seconds": float(sum(float(row["pool_seconds"]) + float(row["score_seconds"]) for row in decisions)),
        "decision_count": int(len(decisions)),
        "decision_times": [int(row["time"]) for row in decisions],
        "anomalies": anomalies,
        "anomaly_count": int(len(anomalies)),
        "illegal": int(illegal),
        "accounting_failures": int(accounting_failures),
        "nan_inf": int(nan_inf),
        "hold_ok": bool(hold_ok),
        "seconds": float(elapsed),
        "scheme_c_calls": int(sum(int(row["scheme_c_called"]) for row in decisions)),
        "projector_interventions": int(interventions),
        "decisions": decisions,
        "fault_recovery": fault_recovery,
    }


__all__ = [
    "FormalPlanner",
    "Selection",
    "accept_scored_row",
    "graph_from_env",
    "run_closed_episode",
    "startup_record",
]
