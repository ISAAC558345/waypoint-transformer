"""Instrumented fly-toward episodes for the Phase-6D audit."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.l3b.policies import fly_toward

from .pool import candidate_pool, top_targets
from .protocol import COVERAGE_SCENARIOS, HOLDS, assert_development
from .shadow import macro_return, score_pool

SCRIPTED = (
    "random_legal",
    "nearest",
    "max_queue",
    "max_rate_independent",
    "queue_rate",
    "joint_true_bit",
)
POOL_MODES = {
    "hold15_current_state_pool_optimum": "current",
    "hold15_realized_future_pool_optimum": "future",
    "ablation_queue_removed": "current",
    "ablation_distance_removed": "current",
    "ablation_interference_removed": "current",
    "ablation_rate_removed": "current",
}


def _strip_queue(clone: L3BBitQueueEnvV2) -> None:
    clone.q_bits[:] = clone.q_max_bits
    clone._sync_block_view()


def _strip_distance(clone: L3BBitQueueEnvV2) -> None:
    centroid = np.mean(clone.uav_positions, axis=0)
    clone.uav_positions[:] = centroid


def _strip_interference(clone: L3BBitQueueEnvV2) -> None:
    clone.config = replace(clone.config, kappa_same_cluster=0.0, kappa_cross_cluster=0.0)


def _prepare(controller: str):
    if controller == "ablation_queue_removed":
        return _strip_queue
    if controller == "ablation_distance_removed":
        return _strip_distance
    if controller == "ablation_interference_removed":
        return _strip_interference
    return None


def _ranking(controller: str) -> str:
    if controller == "ablation_rate_removed":
        return "queue"
    return "instantaneous_rate"


def _as_list(assignment: np.ndarray) -> str:
    return ",".join(str(int(item)) for item in assignment)


def run_instrumented(scenario: int, controller: str, hold: int, *, k: int = 3, diagnose_windows: bool = True) -> dict:
    assert_development(int(scenario))
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario))
    rng = np.random.default_rng(17_000_003 + int(scenario))
    radius = float(env.config.service_radius_m)
    tol = float(env.tol_bits)
    horizon_limit = int(env.config.episode_horizon)
    previous = np.full(env.config.num_uavs, -2, dtype=np.int64)
    pending = 0
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    sinr_sum = 0.0
    rate_sum = 0.0
    path = 0.0
    switches = 0
    zero_steps = 0
    windows: list[dict] = []
    decisions: list[dict] = []
    ablations: list[dict] = []
    q_decision = np.zeros(env.config.num_uavs, dtype=np.float64)
    queue_snapshot = np.zeros(env.config.num_buoys, dtype=np.float64)
    state = _fresh_window()

    def close_window() -> None:
        if state["steps"] <= 0 or not diagnose_windows:
            return
        length = int(state["steps"])
        uavs = int(env.config.num_uavs)
        windows.append(
            {
                "scenario_id": int(scenario),
                "controller": controller,
                "hold": int(hold),
                "time_step": int(state["start"]),
                "window_steps": length,
                "assignment": _as_list(assignment),
                "target_q_bits_at_decision": float(np.sum(q_decision)),
                "mean_target_q_bits": float(np.mean(q_decision)),
                "first_service_step": int(state["first_service"]),
                "queue_depletion_step": int(state["depletion_step"]),
                "active_service_steps": int(state["active_time"]),
                "zero_service_steps": int(state["zero_time"]),
                "raw_rate_mean": float(state["rate_sum"] / length),
                "served_bits": float(state["served"]),
                "mean_uav_target_distance": float(state["initial_distance"]),
                "final_mean_distance": float(state["final_distance"]),
                "mean_sinr": float(state["sinr_sum"] / length),
                "queue_clip_count": int(state["clips"]),
                "target_became_empty": bool(state["became_empty"]),
                "locked_to_empty_target": bool(state["empty_locked_steps"] > 0),
                "empty_locked_steps": int(state["empty_locked_steps"]),
                "service_utilization_ratio": float(state["active_time"] / length),
                "idle_after_depletion_ratio": float(state["idle_after_steps"] / length),
                "zero_uav_steps": int(state["zero_uav"]),
                "cause_A_travel": int(state["step_cause"]["A_travel"]),
                "cause_A_overshoot": int(state["step_cause"]["A_overshoot"]),
                "cause_B_depleted": int(state["step_cause"]["B_depleted"]),
                "cause_C_interference": int(state["step_cause"]["C_interference"]),
                "cause_D_empty_at_decision": int(state["step_cause"]["D_empty_at_decision"]),
                "cause_E_hold_lock": int(state["step_cause"]["E_hold_lock"]),
                "cause_F_other": int(state["step_cause"]["F_other"]),
                "uav_A_travel": int(state["uav_cause"]["A_travel"]),
                "uav_B_depleted": int(state["uav_cause"]["B_depleted"]),
                "uav_C_interference": int(state["uav_cause"]["C_interference"]),
                "uav_D_empty_at_decision": int(state["uav_cause"]["D_empty_at_decision"]),
                "uav_E_hold_lock": int(state["uav_cause"]["E_hold_lock"]),
                "uav_F_other": int(state["uav_cause"]["F_other"]),
                "non_causal_upper_bound": bool(controller == "hold15_realized_future_pool_optimum"),
                "uav_slots": int(length * uavs),
            }
        )

    while int(env.t) < horizon_limit:
        if pending <= 0:
            close_window()
            extra = _select(controller, env, rng, hold, horizon_limit, k)
            assignment = extra["assignment"]
            if int(previous[0]) != -2:
                switches += int(np.sum(previous != assignment))
            q_decision = np.zeros(env.config.num_uavs, dtype=np.float64)
            queue_snapshot = env.q_bits.copy()
            for uav, buoy in enumerate(assignment):
                if int(buoy) >= 0:
                    q_decision[uav] = float(env.q_bits[int(buoy)])
            decisions.append(
                {
                    "scenario_id": int(scenario),
                    "controller": controller,
                    "hold": int(hold),
                    "time_step": int(env.t),
                    "assignment": _as_list(assignment),
                    "pool_size": int(extra.get("pool_size", 0)),
                    "shadow_value": extra.get("shadow_value"),
                    "non_causal_upper_bound": bool(extra.get("non_causal_upper_bound", False)),
                    "interference_disagreement": extra.get("interference_disagreement"),
                    "j15_true": extra.get("j15_true"),
                    "j15_of_no_interference_assignment": extra.get("j15_of_no_interference_assignment"),
                    "delta_j15": extra.get("delta_j15"),
                    "no_interference_assignment": extra.get("no_interference_assignment"),
                    "global_oracle": False,
                }
            )
            if controller == "hold15_current_state_pool_optimum" and int(scenario) in COVERAGE_SCENARIOS:
                ablations.extend(_feature_ablation(env, scenario, assignment, float(extra["shadow_value"]), hold, horizon_limit, k))
            previous = np.asarray(assignment, dtype=np.int64).copy()
            pending = int(hold)
            state = _fresh_window()
            state["start"] = int(env.t)
        q_before = env.q_bits.copy()
        positions = env.uav_positions.copy()
        distances = np.full(env.config.num_uavs, -1.0, dtype=np.float64)
        for uav, buoy in enumerate(assignment):
            if int(buoy) >= 0:
                distances[uav] = float(np.linalg.norm(positions[uav] - env.buoy_positions[int(buoy)]))
        info = env.step(fly_toward(env, assignment), assignment)
        path += float(np.sum(np.linalg.norm(env.uav_positions - positions, axis=1)))
        sinr_sum += float(np.mean(info["sinr"]))
        rate_sum += float(np.mean(info["rate_bps"]))
        served_now = float(info["payload_increment_bits"])
        if served_now <= tol:
            zero_steps += 1
        if diagnose_windows:
            _record_step(
                state,
                assignment,
                q_decision,
                queue_snapshot,
                q_before,
                env.q_bits,
                distances,
                env.uav_positions,
                env.buoy_positions,
                info,
                radius,
                tol,
                float(env.config.delta_t),
            )
        pending -= 1
    close_window()
    error = np.abs(env.conservation_error_bits())
    summary = {
        "scenario_id": int(scenario),
        "controller": controller,
        "hold": int(hold),
        "cumulative_payload_bits": float(env.payload_bits),
        "residual_queue_bits": float(np.sum(env.q_bits)),
        "dropped_bits": float(np.sum(env.cumulative_dropped_bits)),
        "mean_sinr": float(sinr_sum / horizon_limit),
        "mean_rate": float(rate_sum / horizon_limit),
        "switch_count": int(switches),
        "path_length_m": float(path),
        "zero_service_steps": int(zero_steps),
        "steps": int(horizon_limit),
        "service_utilization_ratio": float((horizon_limit - zero_steps) / horizon_limit),
        "conservation_abs_error_bits": float(np.max(error)),
        "conservation_ok": bool(float(np.max(error)) <= env.tol_bits),
        "non_causal_upper_bound": bool(controller == "hold15_realized_future_pool_optimum"),
        "movement": "fly_toward",
    }
    return {"summary": summary, "windows": windows, "decisions": decisions, "ablations": ablations}


def _fresh_window() -> dict:
    blank = {name: 0 for name in ("A_travel", "B_depleted", "C_interference", "D_empty_at_decision", "E_hold_lock", "F_other", "A_overshoot")}
    return {
        "start": 0,
        "steps": 0,
        "active_time": 0,
        "zero_time": 0,
        "served": 0.0,
        "rate_sum": 0.0,
        "sinr_sum": 0.0,
        "clips": 0,
        "first_service": -1,
        "depletion_step": -1,
        "became_empty": False,
        "empty_locked_steps": 0,
        "idle_after_steps": 0,
        "zero_uav": 0,
        "initial_distance": 0.0,
        "final_distance": 0.0,
        "step_cause": dict(blank),
        "uav_cause": dict(blank),
    }


def _record_step(state, assignment, q_decision, queue_snapshot, q_before, q_after, distances, positions_after, buoy_positions, info, radius, tol, delta_t) -> None:
    state["steps"] += 1
    index = int(state["steps"] - 1)
    served_now = float(info["payload_increment_bits"])
    state["served"] += served_now
    state["rate_sum"] += float(np.mean(info["rate_bps"]))
    state["sinr_sum"] += float(np.mean(info["sinr"]))
    served_by_buoy = np.asarray(info["served_bits"], dtype=np.float64)
    labels = []
    dist_values = []
    overshoot = False
    for uav, buoy in enumerate(assignment):
        buoy_id = int(buoy)
        if buoy_id >= 0:
            dist_values.append(float(distances[uav]))
        pre_distance = float(distances[uav]) if buoy_id >= 0 else -1.0
        post_distance = (
            float(np.linalg.norm(positions_after[uav] - buoy_positions[buoy_id])) if buoy_id >= 0 else -1.0
        )
        served_u = float(served_by_buoy[buoy_id]) if buoy_id >= 0 else 0.0
        q_start = float(q_before[buoy_id]) if buoy_id >= 0 else 0.0
        label = _uav_label(
            buoy_id,
            float(q_decision[uav]),
            q_start,
            float(queue_snapshot.max()) if queue_snapshot.size else 0.0,
            post_distance,
            served_u,
            radius,
            tol,
        )
        if (
            label == "A_travel"
            and pre_distance >= 0.0
            and pre_distance <= radius
            and post_distance > radius
        ):
            overshoot = True
        if label is None:
            if float(info["rate_bps"][uav]) * delta_t > (float(q_before[buoy_id]) + tol):
                state["clips"] += 1
            if state["first_service"] < 0:
                state["first_service"] = index
        else:
            state["zero_uav"] += 1
            state["uav_cause"][label] += 1
            if label == "B_depleted":
                state["uav_cause"]["E_hold_lock"] += 1
            labels.append(label)
        if buoy_id >= 0 and float(q_before[buoy_id]) > tol and float(q_after[buoy_id]) <= tol:
            state["became_empty"] = True
            if state["depletion_step"] < 0:
                state["depletion_step"] = index
    if index == 0 and dist_values:
        state["initial_distance"] = float(np.mean(dist_values))
    finals = []
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        finals.append(float(np.linalg.norm(positions_after[uav] - buoy_positions[int(buoy)])))
    if finals:
        state["final_distance"] = float(np.mean(finals))
    if served_now > tol:
        state["active_time"] += 1
    else:
        state["zero_time"] += 1
        step_label = _step_label(labels)
        state["step_cause"][step_label] += 1
        if step_label == "A_travel" and overshoot:
            state["step_cause"]["A_overshoot"] += 1
        if step_label == "B_depleted":
            state["step_cause"]["E_hold_lock"] += 1
            state["empty_locked_steps"] += 1
            state["idle_after_steps"] += 1
        elif step_label == "D_empty_at_decision":
            state["idle_after_steps"] += 1


def _uav_label(buoy, q_decision, q_start, max_queue, distance, served, radius, tol) -> str | None:
    if served > tol:
        return None
    if buoy < 0:
        return "D_empty_at_decision" if max_queue <= tol else "F_other"
    if q_decision <= tol:
        return "D_empty_at_decision"
    if q_start <= tol:
        return "B_depleted"
    if distance > radius:
        return "A_travel"
    return "C_interference"


def _step_label(labels: list[str]) -> str:
    if not labels:
        return "F_other"
    if labels.count("A_travel") > 0 and "C_interference" not in labels:
        # A zero-service step that still has someone flying toward a live queue is travel.
        if all(item in {"A_travel", "B_depleted", "D_empty_at_decision", "F_other"} for item in labels):
            return "A_travel"
    order = ("C_interference", "A_travel", "B_depleted", "D_empty_at_decision", "F_other")
    for name in order:
        if name in labels:
            return name
    return "F_other"


def _select(controller: str, env: L3BBitQueueEnvV2, rng, hold: int, horizon_limit: int, k: int) -> dict:
    if controller in SCRIPTED:
        geo = geometry(env)
        assignment, _info = choose_assignment(controller, env, rng, geo)
        return {"assignment": assignment, "pool_size": 0, "shadow_value": None, "non_causal_upper_bound": False}
    ranking = _ranking(controller)
    pool, pool_size = candidate_pool(env, k, ranking)
    future = POOL_MODES[controller] == "future"
    remaining = min(int(hold), horizon_limit - int(env.t))
    assignment, value = score_pool(
        env,
        pool,
        remaining,
        use_realized_future_arrivals=future,
        prepare_clone=_prepare(controller),
    )
    extra = {
        "assignment": assignment,
        "pool_size": int(pool_size),
        "shadow_value": float(value),
        "non_causal_upper_bound": bool(future),
    }
    if controller == "hold15_current_state_pool_optimum":
        blinded = copy.deepcopy(env)
        _strip_interference(blinded)
        alt_pool, _alt_size = candidate_pool(blinded, k, "instantaneous_rate")
        alt, _shadow = score_pool(
            env,
            alt_pool,
            remaining,
            use_realized_future_arrivals=False,
            prepare_clone=_strip_interference,
        )
        true_alt = macro_return(env, alt, remaining, use_realized_future_arrivals=False)
        extra["interference_disagreement"] = int(not np.array_equal(assignment, alt))
        extra["j15_true"] = float(value)
        extra["j15_of_no_interference_assignment"] = float(true_alt)
        extra["delta_j15"] = float(value - true_alt)
        extra["no_interference_assignment"] = _as_list(alt)
    return extra


def _feature_ablation(env, scenario, full_assignment, full_value, hold, horizon_limit, k) -> list[dict]:
    remaining = min(int(hold), horizon_limit - int(env.t))
    specs = (
        ("queue_removed", _strip_queue, "instantaneous_rate"),
        ("distance_removed", _strip_distance, "instantaneous_rate"),
        ("interference_removed", _strip_interference, "instantaneous_rate"),
        ("instantaneous_rate_removed", None, "queue"),
    )
    rows = []
    for name, prepare, ranking in specs:
        source = env
        if name == "interference_removed":
            source = copy.deepcopy(env)
            _strip_interference(source)
        pool, _size = candidate_pool(source, k, ranking)
        alternative, _shadow = score_pool(
            env,
            pool,
            remaining,
            use_realized_future_arrivals=False,
            prepare_clone=prepare,
        )
        true_value = macro_return(env, alternative, remaining, use_realized_future_arrivals=False)
        rows.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(env.t),
                "ablation": name,
                "assignment_disagreement": int(not np.array_equal(full_assignment, alternative)),
                "j15_full": float(full_value),
                "j15_ablated_assignment": float(true_value),
                "delta_j15": float(full_value - true_value),
                "full_assignment": _as_list(full_assignment),
                "ablated_assignment": _as_list(alternative),
                "non_causal_upper_bound": False,
                "global_oracle": False,
            }
        )
    return rows


def coverage_audit(scenario: int) -> list[dict]:
    """Compare top-3/4/5 J15 on max-rate decision states. Not a global oracle."""
    assert_development(int(scenario))
    if int(scenario) not in COVERAGE_SCENARIOS:
        return []
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario))
    rng = np.random.default_rng(17_000_003 + int(scenario))
    pending = 0
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    rows: list[dict] = []
    horizon_limit = int(env.config.episode_horizon)
    while int(env.t) < horizon_limit:
        if pending <= 0:
            geo = geometry(env)
            assignment, _info = choose_assignment("max_rate_independent", env, rng, geo)
            remaining = min(15, horizon_limit - int(env.t))
            values = {}
            chosen = {}
            sizes = {}
            for k in (3, 4, 5):
                pool, size = candidate_pool(env, k, "instantaneous_rate")
                best, value = score_pool(env, pool, remaining, use_realized_future_arrivals=False)
                values[k] = float(value)
                chosen[k] = best
                sizes[k] = int(size)
            tops = [set(int(item) for item in row) for row in top_targets(env, 3, "instantaneous_rate")]
            edges = 0
            covered = 0
            for uav, buoy in enumerate(chosen[5]):
                if int(buoy) < 0:
                    continue
                edges += 1
                if int(buoy) in tops[uav]:
                    covered += 1
            j3 = values[3]
            j4 = values[4]
            if j3 > 1.0:
                relative = (j4 - j3) / j3
            elif j4 <= 1.0:
                relative = 0.0
            else:
                relative = 1.0
            rows.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(env.t),
                    "pool_size_k3": sizes[3],
                    "pool_size_k4": sizes[4],
                    "pool_size_k5": sizes[5],
                    "j15_k3": j3,
                    "j15_k4": j4,
                    "j15_k5": values[5],
                    "relative_k4_vs_k3": float(relative),
                    "edge_coverage_k5_in_k3": float(covered / edges) if edges else 1.0,
                    "assignment_k3": _as_list(chosen[3]),
                    "assignment_k5": _as_list(chosen[5]),
                    "non_causal_upper_bound": False,
                    "global_oracle": False,
                }
            )
            pending = 15
        env.step(fly_toward(env, assignment), assignment)
        pending -= 1
    return rows


def reproduce_scenario(scenario: int) -> list[dict]:
    rows = []
    for controller in ("max_rate_independent", "queue_rate", "joint_true_bit"):
        summary = run_instrumented(int(scenario), controller, 15, diagnose_windows=False)["summary"]
        rows.append(
            {
                "scenario_id": int(scenario),
                "controller": controller,
                "phase6d_payload_bits": float(summary["cumulative_payload_bits"]),
                "conservation_ok": bool(summary["conservation_ok"]),
            }
        )
    return rows


def audit_scenario(scenario: int) -> dict:
    assert_development(int(scenario))
    summaries = []
    windows = []
    decisions = []
    ablations = []
    for hold in HOLDS:
        for controller in ("max_rate_independent", "joint_true_bit"):
            episode = run_instrumented(int(scenario), controller, hold, diagnose_windows=True)
            summaries.append(episode["summary"])
            windows.extend(episode["windows"])
            decisions.extend(episode["decisions"])
            ablations.extend(episode["ablations"])
    for controller in ("random_legal", "nearest", "max_queue", "queue_rate"):
        episode = run_instrumented(int(scenario), controller, 15, diagnose_windows=False)
        summaries.append(episode["summary"])
    for controller in POOL_MODES:
        episode = run_instrumented(
            int(scenario),
            controller,
            15,
            diagnose_windows=controller == "hold15_current_state_pool_optimum",
        )
        summaries.append(episode["summary"])
        windows.extend(episode["windows"])
        decisions.extend(episode["decisions"])
        ablations.extend(episode["ablations"])
    return {
        "summaries": summaries,
        "windows": windows,
        "decisions": decisions,
        "ablations": ablations,
        "coverage": coverage_audit(int(scenario)),
    }
