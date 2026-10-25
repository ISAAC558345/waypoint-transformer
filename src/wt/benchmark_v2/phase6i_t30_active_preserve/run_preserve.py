"""t=30 counterfactual that keeps in-range assignments fixed for 15 steps."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import D_SAFE_M
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6i_execution_attribution.run_attribution import (
    DECISION_TIMES,
    HOLD,
    _distance,
    _move,
    assert_validation_scenario,
    sha256_file,
)
from wt.benchmark_v2.phase6i_t45_counterfactual.run_counterfactual import (
    P1_WINDOWS,
    _branch,
    _environment_hash,
    _paired_bootstrap,
    pool_hash,
)
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner, graph_from_env
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, SCHEME_C_CKPT, SCHEME_C_SHA256
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import DECODER_CKPT, RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import bind_keys, frozen_candidate_keys
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.observation import communication_features

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_t30_active_preserve"
RULE_VERSION = "t30_preserve_in_range_then_max_reach_then_min_distance_v1"
IDLE_DISTANCE_M = 1.0e9
MIN_CHANGED_SCENARIOS = 3
WINDOW_STEPS = 15
TAIL_STEPS = 3

SCENARIO_FIELDS = (
    "scenario_id",
    "decision_t",
    "window_steps",
    "tail_steps",
    "candidate_count",
    "preserving_candidate_count",
    "pool_hash",
    "state_hash",
    "branch_hash_match",
    "rule_version",
    "comparison_status",
    "fixed_uavs",
    "reassignable_uavs",
    "scheme_index",
    "scheme_assignment",
    "scheme_reassignable_reachable",
    "scheme_reassignable_distance_m",
    "scheme_score",
    "preserve_index",
    "preserve_assignment",
    "preserve_reassignable_reachable",
    "preserve_reassignable_distance_m",
    "preserve_score",
    "changed_uavs",
    "payload_at_branch_bits",
    "window_payload_scheme_bits",
    "window_payload_preserve_bits",
    "delta_window_bits",
    "branch_payload_scheme_bits",
    "branch_payload_preserve_bits",
    "delta_branch_bits",
    "episode_payload_scheme_bits",
    "episode_payload_preserve_bits",
    "delta_episode_bits",
    "entered_window_scheme",
    "entered_window_preserve",
    "service_steps_window_scheme",
    "service_steps_window_preserve",
    "t45_scheme_assignment",
    "t45_preserve_assignment",
    "t45_scheme_pool_hash",
    "t45_preserve_pool_hash",
    "projection_rate_window_scheme",
    "projection_rate_window_preserve",
    "safety_violation_rate_scheme",
    "safety_violation_rate_preserve",
    "accounting_error_scheme_bits",
    "accounting_error_preserve_bits",
    "queue_before_bits",
    "queue_after_window_scheme_bits",
    "queue_after_window_preserve_bits",
    "config_hash",
    "scheme_c_sha256",
    "mappo_sha256",
    "decoder_sha256",
)
UAV_FIELDS = (
    "scenario_id",
    "arm",
    "window_t",
    "uav_id",
    "role_at_t30",
    "target_buoy_id",
    "idle",
    "initial_distance_m",
    "geometrically_reachable",
    "first_entry_step",
    "entered_service_radius",
    "effective_service_steps",
    "served_bits",
    "path_length_m",
    "end_distance_m",
    "projection_count",
    "mean_sinr",
    "mean_interference",
)


def uav_roles(
    uav_positions: np.ndarray,
    buoy_positions: np.ndarray,
    assignment: tuple[int, ...] | np.ndarray,
    radius_m: float,
) -> tuple[list[str], list[float | None]]:
    """In-range and idle UAVs stay fixed. Every other assigned UAV may move."""
    roles: list[str] = []
    distances: list[float | None] = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            roles.append("fixed_idle")
            distances.append(None)
            continue
        distance = float(np.linalg.norm(uav_positions[uav] - buoy_positions[int(buoy)]))
        distances.append(distance)
        if distance <= float(radius_m):
            roles.append("fixed_in_radius")
        else:
            roles.append("reassignable_outside")
    return roles, distances


def _reassignable_score(
    uav_positions: np.ndarray,
    buoy_positions: np.ndarray,
    assignment: tuple[int, ...],
    reassignable: list[int],
    radius_m: float,
    step_length_m: float,
    steps: int,
) -> tuple[int, float]:
    budget = float(steps) * float(step_length_m)
    reachable = 0
    total = 0.0
    for uav in reassignable:
        buoy = int(assignment[uav])
        if buoy < 0:
            total += float(IDLE_DISTANCE_M)
            continue
        distance = float(np.linalg.norm(uav_positions[uav] - buoy_positions[buoy]))
        total += distance
        if max(distance - float(radius_m), 0.0) <= budget:
            reachable += 1
    return reachable, total


def _preserves(candidate: tuple[int, ...], scheme: tuple[int, ...], roles: list[str]) -> bool:
    for uav, role in enumerate(roles):
        if role != "reassignable_outside" and int(candidate[uav]) != int(scheme[uav]):
            return False
    return True


def choose_preserving_assignment(
    uav_positions: np.ndarray,
    buoy_positions: np.ndarray,
    keys: list[tuple[int, ...]],
    scheme_index: int,
    radius_m: float,
    step_length_m: float,
    steps: int = WINDOW_STEPS,
) -> dict:
    """Keep in-range and idle targets. Move outside UAVs by reach, then distance."""
    scheme = tuple(int(item) for item in keys[int(scheme_index)])
    roles, _distances = uav_roles(uav_positions, buoy_positions, scheme, radius_m)
    reassignable = [uav for uav, role in enumerate(roles) if role == "reassignable_outside"]
    fixed = [uav for uav, role in enumerate(roles) if role != "reassignable_outside"]
    base = {
        "roles": roles,
        "fixed_uavs": fixed,
        "reassignable_uavs": reassignable,
        "rule_version": RULE_VERSION,
        "preserving_candidate_count": 0,
    }
    if not reassignable:
        return {
            **base,
            "status": "no_reassignable_uav",
            "index": int(scheme_index),
            "assignment": scheme,
            "reachable_uavs": 0,
            "total_distance_m": 0.0,
        }
    eligible = [index for index, key in enumerate(keys) if _preserves(tuple(key), scheme, roles)]
    base["preserving_candidate_count"] = int(len(eligible))
    if not eligible:
        reach, total = _reassignable_score(
            uav_positions, buoy_positions, scheme, reassignable, radius_m, step_length_m, steps
        )
        return {
            **base,
            "status": "no_legal_preserving_candidate",
            "index": int(scheme_index),
            "assignment": scheme,
            "reachable_uavs": reach,
            "total_distance_m": total,
        }
    best_index = None
    best_reach = -1
    best_total = 0.0
    for index in eligible:
        reach, total = _reassignable_score(
            uav_positions, buoy_positions, tuple(keys[index]), reassignable, radius_m, step_length_m, steps
        )
        better_reach = best_index is None or reach > best_reach
        closer = best_index is not None and reach == best_reach and total < best_total
        if better_reach or closer:
            best_index = int(index)
            best_reach = int(reach)
            best_total = float(total)
    chosen = tuple(int(item) for item in keys[int(best_index)])
    status = "preserving_identical" if chosen == scheme else "selected"
    return {
        **base,
        "status": status,
        "index": int(best_index),
        "assignment": chosen,
        "reachable_uavs": int(best_reach),
        "total_distance_m": float(best_total),
    }


def _advance_until(env: L3BBitQueueEnvV2, planner: FormalPlanner, stop_t: int):
    actor = planner.actor
    horizon = int(env.config.episode_horizon)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    while int(env.t) < int(stop_t):
        decision_t = int(env.t)
        if decision_t not in DECISION_TIMES:
            raise RuntimeError(f"decision time {decision_t} is outside the registered clock")
        selection = planner.select_assignment(env, position_prev, predecessor)
        if selection.status != "selected" or selection.option is None or selection.assignment is None:
            raise RuntimeError(f"selection aborted at t={decision_t}: {selection.reason}")
        option = selection.option
        assignment = np.asarray(selection.assignment, dtype=np.int64).copy()
        for _step in range(min(HOLD, horizon - decision_t)):
            t_before = int(env.t)
            step_start = np.asarray(env.uav_positions, dtype=np.float64).copy()
            movement, _intervened, _norms = _move(env, actor, option, last)
            env.step(movement, assignment)
            option.remaining_hold -= 1
            last = np.asarray(movement, dtype=np.float64).copy()
            position_prev = step_start
            predecessor = t_before
    if int(env.t) != int(stop_t):
        raise RuntimeError(f"capture time is {env.t}, expected {stop_t}")
    return position_prev, predecessor, last


def _execute_window(env: L3BBitQueueEnvV2, actor, assignment: np.ndarray, last_movement: np.ndarray, steps: int) -> dict:
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    if not is_legal(chosen, int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("window received an illegal assignment")
    option = make_option(env, chosen)
    radius = float(env.config.service_radius_m)
    tol = float(env.tol_bits)
    budget = float(steps) * float(env.config.step_length_m)
    payload_before = float(env.payload_bits)
    queue_before = float(np.mean(env.q_bits))
    last_movement = np.asarray(last_movement, dtype=np.float64).copy()
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    rows = []
    for uav, buoy in enumerate(chosen):
        distance = None if int(buoy) < 0 else float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
        rows.append(
            {
                "uav_id": int(uav),
                "target_buoy_id": int(buoy),
                "idle": int(int(buoy) < 0),
                "initial_distance_m": distance,
                "geometrically_reachable": int(distance is not None and max(distance - radius, 0.0) <= budget),
                "served_bits": 0.0,
                "effective_service_steps": 0,
                "entered_service_radius": 0,
                "first_entry_step": None,
                "path_length_m": 0.0,
                "projection_count": 0,
                "sinr": [],
                "interf": [],
            }
        )
    served_sum = 0.0
    projections = 0
    safety = 0
    max_gap = 0.0
    sinr_vals = []
    interf_vals = []
    for step_index in range(int(steps)):
        t_before = int(env.t)
        step_start = np.asarray(env.uav_positions, dtype=np.float64).copy()
        movement, intervened, per_uav_norm = _move(env, actor, option, last_movement)
        payload_step_before = float(env.payload_bits)
        info = env.step(movement, chosen)
        served = np.asarray(info["served_bits"], dtype=np.float64)
        gap = abs(float(env.payload_bits) - payload_step_before - float(np.sum(served)))
        max_gap = max(max_gap, gap, float(np.max(np.abs(env.conservation_error_bits()))))
        comm = communication_features(env)
        served_sum += float(np.sum(served))
        projections += int(intervened)
        safety += int(minimum_pairwise_distance(env.uav_positions) + 1.0e-9 < float(D_SAFE_M))
        sinr_vals.append(float(np.mean(info["sinr"])))
        interf_vals.append(float(np.mean(comm["total_interference"])))
        for uav, buoy in enumerate(chosen):
            item = rows[uav]
            distance = _distance(env, uav, int(buoy))
            if distance is not None and distance <= radius:
                item["entered_service_radius"] = 1
                if item["first_entry_step"] is None:
                    item["first_entry_step"] = int(step_index)
            if int(buoy) >= 0 and float(served[int(buoy)]) > tol:
                item["effective_service_steps"] += 1
                item["served_bits"] += float(served[int(buoy)])
            item["path_length_m"] += float(np.linalg.norm(movement[uav]) * env.config.step_length_m)
            item["projection_count"] += int(float(per_uav_norm[uav]) > 1.0e-3)
            item["sinr"].append(float(info["sinr"][uav]))
            item["interf"].append(float(comm["total_interference"][uav]))
            item["end_distance_m"] = distance
        option.remaining_hold -= 1
        last_movement = np.asarray(movement, dtype=np.float64).copy()
        position_prev = step_start
        predecessor = t_before
    payload_delta = float(env.payload_bits) - payload_before
    if abs(payload_delta - served_sum) > tol or max_gap > tol:
        raise RuntimeError("window accounting failed")
    for item in rows:
        item["mean_sinr"] = float(np.mean(item.pop("sinr")))
        item["mean_interference"] = float(np.mean(item.pop("interf")))
    return {
        "payload_bits": payload_delta,
        "served_bits": served_sum,
        "entered_uavs": int(sum(item["entered_service_radius"] for item in rows if item["idle"] == 0)),
        "service_steps": int(sum(item["effective_service_steps"] for item in rows)),
        "projection_rate": float(projections / max(int(steps), 1)),
        "safety_violation_rate": float(safety / max(int(steps), 1)),
        "accounting_error_bits": float(max_gap),
        "queue_before_bits": queue_before,
        "queue_after_bits": float(np.mean(env.q_bits)),
        "mean_sinr": float(np.mean(sinr_vals)) if sinr_vals else None,
        "mean_interference": float(np.mean(interf_vals)) if interf_vals else None,
        "uavs": rows,
        "last_movement": last_movement,
        "position_prev": position_prev,
        "predecessor": predecessor,
    }


def _p1_value(scenario_id: int, decision_t: int, field: str) -> str:
    with P1_WINDOWS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["scenario_id"]) == int(scenario_id) and int(row["decision_t"]) == int(decision_t):
                return str(row[field])
    raise RuntimeError(f"P1 has no t={decision_t} row for {scenario_id}")


def _blank(value) -> str:
    if value is None:
        return ""
    return str(value)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _blank(row.get(key)) for key in fields})


def _arm_rows(scenario_id: int, arm: str, window_t: int, result: dict, roles: list[str]) -> list[dict]:
    rows = []
    for item in result["uavs"]:
        rows.append(
            {
                "scenario_id": int(scenario_id),
                "arm": arm,
                "window_t": int(window_t),
                "role_at_t30": roles[int(item["uav_id"])],
                **item,
            }
        )
    return rows


def _run_arm(env: L3BBitQueueEnvV2, planner: FormalPlanner, assignment: np.ndarray, last_movement: np.ndarray) -> dict:
    actor = planner.actor
    window = _execute_window(env, actor, assignment, last_movement, WINDOW_STEPS)
    if int(env.t) != 45:
        raise RuntimeError(f"window ended at t={env.t}")
    selection = planner.select_assignment(env, window["position_prev"], window["predecessor"])
    if selection.status != "selected" or selection.assignment is None:
        raise RuntimeError(f"t=45 scheme C aborted: {selection.reason}")
    tail = _execute_window(env, actor, np.asarray(selection.assignment, dtype=np.int64), window["last_movement"], TAIL_STEPS)
    if int(env.t) != int(env.config.episode_horizon):
        raise RuntimeError(f"episode ended at t={env.t}")
    return {"window": window, "tail": tail, "t45_assignment": tuple(int(item) for item in selection.assignment), "t45_pool_hash": str(selection.pool_hash), "final_payload_bits": float(env.payload_bits)}


def _figures(rows: list[dict], directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    saved = []
    for name, field, label in (
        ("window_delta.png", "delta_window_bits", "15-step payload preserve minus scheme (Mbit)"),
        ("episode_delta.png", "delta_episode_bits", "episode payload preserve minus scheme (Mbit)"),
    ):
        fig, ax = plt.subplots()
        ax.bar(range(count), [float(row[field]) / 1.0e6 for row in rows])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("validation scenario order")
        ax.set_ylabel(label)
        ax.set_title(f"{name.split('_')[0]} paired payload, n_scenarios={count}, unit=scenario")
        fig.savefig(directory / name, dpi=120)
        plt.close(fig)
        saved.append(name)
    fig, ax = plt.subplots()
    ax.scatter(
        [int(row["entered_window_scheme"]) for row in rows],
        [int(row["entered_window_preserve"]) for row in rows],
        s=28,
    )
    ax.set_xlabel("scheme C UAVs entering radius")
    ax.set_ylabel("preserve-rule UAVs entering radius")
    ax.set_title(f"15-step service entry, n_scenarios={count}, unit=scenario")
    fig.savefig(directory / "entered_uavs.png", dpi=120)
    plt.close(fig)
    saved.append("entered_uavs.png")
    return saved


def run_scenario(scenario_id: int, planner: FormalPlanner, hashes: dict) -> tuple[dict, list[dict]]:
    assert_validation_scenario(scenario_id)
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    position_prev, predecessor, last = _advance_until(env, planner, 30)
    token = parameter_token(planner.actor)
    before_hash = _environment_hash(env)
    selection = planner.select_assignment(env, position_prev, predecessor)
    if selection.status != "selected" or selection.assignment is None:
        raise RuntimeError(f"scheme C aborted at t=30: {selection.reason}")
    if _environment_hash(env) != before_hash or parameter_token(planner.actor) != token:
        raise RuntimeError("scheme C selection changed the state or the actor")
    keys, _raw = frozen_candidate_keys(env)
    bindings, _mapping = bind_keys(keys, n_uavs=int(env.config.num_uavs), n_buoys=int(env.config.num_buoys))
    if any(not binding.legal for binding in bindings) or len(keys) != int(selection.candidate_count):
        raise RuntimeError("candidate binding diverged")
    digest = pool_hash(keys)
    if digest != selection.pool_hash:
        raise RuntimeError("pool hash diverged from FormalPlanner")
    scheme_index = int(selection.score_index)
    scheme = tuple(int(item) for item in selection.assignment)
    if scheme != tuple(int(item) for item in keys[scheme_index]):
        raise RuntimeError("scheme C index is not bound to its assignment")
    official = ",".join(str(item) for item in scheme)
    if official != _p1_value(scenario_id, 30, "assignment"):
        raise RuntimeError("t=30 scheme C assignment does not match the P1 replay")
    matrix = np.asarray([binding.assignment for binding in bindings], dtype=np.int64)
    scored = planner.scorer.infer(graph_from_env(env, position_prev, predecessor), matrix)
    if int(scored.chosen_index) != scheme_index or tuple(int(item) for item in scored.chosen_assignment) != scheme:
        raise RuntimeError("repeated scheme C score does not match the official selection")
    if _environment_hash(env) != before_hash or parameter_token(planner.actor) != token:
        raise RuntimeError("scoring changed the state or the actor")
    positions = np.asarray(env.uav_positions, dtype=np.float64)
    buoys = np.asarray(env.buoy_positions, dtype=np.float64)
    choice = choose_preserving_assignment(
        positions,
        buoys,
        keys,
        scheme_index,
        float(env.config.service_radius_m),
        float(env.config.step_length_m),
        WINDOW_STEPS,
    )
    if not is_legal(np.asarray(choice["assignment"], dtype=np.int64), int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("preserving rule returned an illegal assignment")
    changed = [index for index, (left, right) in enumerate(zip(scheme, choice["assignment"])) if int(left) != int(right)]
    if any(index not in choice["reassignable_uavs"] for index in changed):
        raise RuntimeError("preserving rule changed a fixed UAV")
    scheme_reach, scheme_total = _reassignable_score(
        positions,
        buoys,
        scheme,
        choice["reassignable_uavs"],
        float(env.config.service_radius_m),
        float(env.config.step_length_m),
        WINDOW_STEPS,
    )
    payload_at_branch = float(env.payload_bits)
    arm_s = _run_arm(_branch(env, scenario_id), planner, np.asarray(scheme, dtype=np.int64), last)
    arm_p = _run_arm(_branch(env, scenario_id), planner, np.asarray(choice["assignment"], dtype=np.int64), last)
    if parameter_token(planner.actor) != token or _environment_hash(env) != before_hash:
        raise RuntimeError("branch execution changed the captured state or the actor")
    official_tail = _p1_value(scenario_id, 45, "assignment")
    if ",".join(str(item) for item in arm_s["t45_assignment"]) != official_tail:
        raise RuntimeError("scheme arm t=45 assignment does not match the P1 replay")
    p1_final = float(_p1_value(scenario_id, 45, "payload_after_bits"))
    tol = float(env.tol_bits)
    if abs(arm_s["final_payload_bits"] - p1_final) > tol:
        raise RuntimeError("scheme arm final payload does not match the P1 replay")
    for arm in (arm_s, arm_p):
        delivered = float(arm["window"]["served_bits"] + arm["tail"]["served_bits"])
        if abs(arm["final_payload_bits"] - payload_at_branch - delivered) > tol:
            raise RuntimeError("branch payload does not match the served-bit sum")
    delta_window = float(arm_p["window"]["payload_bits"] - arm_s["window"]["payload_bits"])
    delta_episode = float(arm_p["final_payload_bits"] - arm_s["final_payload_bits"])
    delta_branch = float((arm_p["final_payload_bits"] - payload_at_branch) - (arm_s["final_payload_bits"] - payload_at_branch))
    if abs(delta_episode - delta_branch) > tol:
        raise RuntimeError("episode delta and branch delta diverged")
    if choice["status"] != "selected" and abs(delta_episode) > tol:
        raise RuntimeError("unchanged assignment produced an episode payload gap")
    row = {
        "scenario_id": int(scenario_id),
        "decision_t": 30,
        "window_steps": WINDOW_STEPS,
        "tail_steps": TAIL_STEPS,
        "candidate_count": int(len(keys)),
        "preserving_candidate_count": int(choice["preserving_candidate_count"]),
        "pool_hash": digest,
        "state_hash": before_hash,
        "branch_hash_match": 1,
        "rule_version": RULE_VERSION,
        "comparison_status": choice["status"],
        "fixed_uavs": ",".join(str(item) for item in choice["fixed_uavs"]),
        "reassignable_uavs": ",".join(str(item) for item in choice["reassignable_uavs"]),
        "scheme_index": scheme_index,
        "scheme_assignment": official,
        "scheme_reassignable_reachable": int(scheme_reach),
        "scheme_reassignable_distance_m": float(scheme_total),
        "scheme_score": float(scored.scores[scheme_index]),
        "preserve_index": int(choice["index"]),
        "preserve_assignment": ",".join(str(int(item)) for item in choice["assignment"]),
        "preserve_reassignable_reachable": int(choice["reachable_uavs"]),
        "preserve_reassignable_distance_m": float(choice["total_distance_m"]),
        "preserve_score": float(scored.scores[int(choice["index"])]),
        "changed_uavs": ",".join(str(item) for item in changed),
        "payload_at_branch_bits": payload_at_branch,
        "window_payload_scheme_bits": arm_s["window"]["payload_bits"],
        "window_payload_preserve_bits": arm_p["window"]["payload_bits"],
        "delta_window_bits": delta_window,
        "branch_payload_scheme_bits": arm_s["final_payload_bits"] - payload_at_branch,
        "branch_payload_preserve_bits": arm_p["final_payload_bits"] - payload_at_branch,
        "delta_branch_bits": delta_branch,
        "episode_payload_scheme_bits": arm_s["final_payload_bits"],
        "episode_payload_preserve_bits": arm_p["final_payload_bits"],
        "delta_episode_bits": delta_episode,
        "entered_window_scheme": arm_s["window"]["entered_uavs"],
        "entered_window_preserve": arm_p["window"]["entered_uavs"],
        "service_steps_window_scheme": arm_s["window"]["service_steps"],
        "service_steps_window_preserve": arm_p["window"]["service_steps"],
        "t45_scheme_assignment": ",".join(str(item) for item in arm_s["t45_assignment"]),
        "t45_preserve_assignment": ",".join(str(item) for item in arm_p["t45_assignment"]),
        "t45_scheme_pool_hash": arm_s["t45_pool_hash"],
        "t45_preserve_pool_hash": arm_p["t45_pool_hash"],
        "projection_rate_window_scheme": arm_s["window"]["projection_rate"],
        "projection_rate_window_preserve": arm_p["window"]["projection_rate"],
        "safety_violation_rate_scheme": max(arm_s["window"]["safety_violation_rate"], arm_s["tail"]["safety_violation_rate"]),
        "safety_violation_rate_preserve": max(arm_p["window"]["safety_violation_rate"], arm_p["tail"]["safety_violation_rate"]),
        "accounting_error_scheme_bits": max(arm_s["window"]["accounting_error_bits"], arm_s["tail"]["accounting_error_bits"]),
        "accounting_error_preserve_bits": max(arm_p["window"]["accounting_error_bits"], arm_p["tail"]["accounting_error_bits"]),
        "queue_before_bits": arm_s["window"]["queue_before_bits"],
        "queue_after_window_scheme_bits": arm_s["window"]["queue_after_bits"],
        "queue_after_window_preserve_bits": arm_p["window"]["queue_after_bits"],
        **hashes,
    }
    uav_rows = []
    uav_rows.extend(_arm_rows(scenario_id, "scheme", 30, arm_s["window"], choice["roles"]))
    uav_rows.extend(_arm_rows(scenario_id, "preserve", 30, arm_p["window"], choice["roles"]))
    uav_rows.extend(_arm_rows(scenario_id, "scheme", 45, arm_s["tail"], choice["roles"]))
    uav_rows.extend(_arm_rows(scenario_id, "preserve", 45, arm_p["tail"], choice["roles"]))
    return row, uav_rows


def _side(interval: dict) -> int:
    if float(interval["ci_low"]) > 0.0:
        return 1
    if float(interval["ci_high"]) < 0.0:
        return -1
    return 0


def _gate(rows: list[dict], window_interval: dict, episode_interval: dict) -> str:
    if len(rows) != len(VALIDATION_POOL):
        return "T30_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED"
    if any(int(row["branch_hash_match"]) != 1 or int(row["candidate_count"]) != 13327 for row in rows):
        return "T30_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED"
    changed = sum(row["comparison_status"] == "selected" for row in rows)
    window_side = _side(window_interval)
    episode_side = _side(episode_interval)
    if changed < MIN_CHANGED_SCENARIOS or window_side == 0 or episode_side == 0 or window_side != episode_side:
        return "T30_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE"
    return "T30_ACTIVE_ASSIGNMENT_PRESERVED_COMPLETE"


def main() -> dict:
    torch.set_num_threads(1)
    if any(int(item) in RESERVED_TEST for item in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    if sha256_file(SCHEME_C_CKPT) != SCHEME_C_SHA256 or sha256_file(primary_checkpoint()) != MAPPO_SHA256:
        raise RuntimeError("frozen checkpoint hash mismatch")
    if not DECODER_CKPT.is_file():
        raise RuntimeError("matching decoder checkpoint is missing")
    hashes = {
        "config_hash": config_sha256(),
        "scheme_c_sha256": SCHEME_C_SHA256,
        "mappo_sha256": sha256_file(primary_checkpoint()),
        "decoder_sha256": sha256_file(DECODER_CKPT),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    planner = FormalPlanner("scheme_c")
    rows = []
    uav_rows = []
    for scenario in VALIDATION_POOL:
        print(f"t30 preserve {scenario}", flush=True)
        row, uavs = run_scenario(int(scenario), planner, hashes)
        rows.append(row)
        uav_rows.extend(uavs)
    _write_csv(OUTPUT / "scenario_pairs.csv", SCENARIO_FIELDS, rows)
    _write_csv(OUTPUT / "uav_window_rows.csv", UAV_FIELDS, uav_rows)
    window_deltas = [float(row["delta_window_bits"]) for row in rows]
    episode_deltas = [float(row["delta_episode_bits"]) for row in rows]
    changed_window = [float(row["delta_window_bits"]) for row in rows if row["comparison_status"] == "selected"]
    changed_episode = [float(row["delta_episode_bits"]) for row in rows if row["comparison_status"] == "selected"]
    window_interval = _paired_bootstrap(window_deltas)
    episode_interval = _paired_bootstrap(episode_deltas)
    summary = {
        "gate": _gate(rows, window_interval, episode_interval),
        "n_scenarios": len(rows),
        "split": "validation_322048_322063",
        "reserved_test_used": False,
        "decision_t": 30,
        "window_steps": WINDOW_STEPS,
        "tail_steps": TAIL_STEPS,
        "rule_version": RULE_VERSION,
        "min_changed_scenarios_for_complete": MIN_CHANGED_SCENARIOS,
        "changed_scenarios": int(len(changed_window)),
        "unchanged_scenarios": int(len(rows) - len(changed_window)),
        "no_reassignable_scenarios": int(sum(row["comparison_status"] == "no_reassignable_uav" for row in rows)),
        "no_legal_preserving_scenarios": int(sum(row["comparison_status"] == "no_legal_preserving_candidate" for row in rows)),
        "paired_window_all_scenarios": window_interval,
        "paired_episode_all_scenarios": episode_interval,
        "paired_window_changed_scenarios": _paired_bootstrap(changed_window) if changed_window else None,
        "paired_episode_changed_scenarios": _paired_bootstrap(changed_episode) if changed_episode else None,
        "max_accounting_error_bits": float(
            max(max(row["accounting_error_scheme_bits"], row["accounting_error_preserve_bits"]) for row in rows)
        ),
        "max_safety_violation_rate": float(
            max(max(row["safety_violation_rate_scheme"], row["safety_violation_rate_preserve"]) for row in rows)
        ),
        "source": hashes,
        "figures": _figures(rows, OUTPUT / "figures"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": summary["gate"],
                "changed_scenarios": summary["changed_scenarios"],
                "paired_window_all_scenarios": window_interval,
                "paired_episode_all_scenarios": episode_interval,
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    main()
