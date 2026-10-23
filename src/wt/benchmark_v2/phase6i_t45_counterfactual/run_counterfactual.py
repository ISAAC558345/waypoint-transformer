"""Paired t=45 counterfactual. The distance rule never sees future payload."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
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
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner, graph_from_env
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, SCHEME_C_CKPT, SCHEME_C_SHA256
from wt.benchmark_v2.phase6iab_planner_stability.protocol import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, BIT_TIE
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_hash
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import DECODER_CKPT, RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import bind_keys, frozen_candidate_keys
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.observation import communication_features

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_t45_counterfactual"
P1_WINDOWS = ROOT / "results" / "phase6i_execution_attribution" / "window_rows.csv"
TAIL_STEPS = 3
RULE_VERSION = "reachable_count_then_total_distance_then_pool_order_v1"

SCENARIO_FIELDS = (
    "scenario_id",
    "decision_t",
    "window_steps",
    "candidate_count",
    "pool_hash",
    "state_hash",
    "branch_hash_match",
    "rule_version",
    "comparison_status",
    "scheme_index",
    "scheme_assignment",
    "scheme_reachable_uavs",
    "scheme_total_distance_m",
    "scheme_score",
    "near_index",
    "near_assignment",
    "near_reachable_uavs",
    "near_total_distance_m",
    "near_score",
    "changed_uavs",
    "payload_scheme_bits",
    "payload_near_bits",
    "delta_payload_bits",
    "entered_scheme",
    "entered_near",
    "service_steps_scheme",
    "service_steps_near",
    "projection_rate_scheme",
    "projection_rate_near",
    "safety_violation_rate_scheme",
    "safety_violation_rate_near",
    "accounting_error_scheme_bits",
    "accounting_error_near_bits",
    "config_hash",
    "scheme_c_sha256",
    "mappo_sha256",
    "decoder_sha256",
)
UAV_FIELDS = (
    "scenario_id",
    "arm",
    "uav_id",
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


def pool_hash(keys: list[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(keys, dtype=np.int64).tobytes())
    return digest.hexdigest()


def assignment_geometry(
    uav_positions: np.ndarray,
    buoy_positions: np.ndarray,
    assignment: tuple[int, ...] | np.ndarray,
    radius_m: float,
    step_length_m: float,
    steps: int = TAIL_STEPS,
) -> tuple[int, float, list[float | None]]:
    """Count non-idle UAVs that can geometrically reach within the remaining steps."""
    budget = float(steps) * float(step_length_m)
    reachable = 0
    total = 0.0
    distances: list[float | None] = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            distances.append(None)
            continue
        distance = float(np.linalg.norm(uav_positions[uav] - buoy_positions[int(buoy)]))
        distances.append(distance)
        total += distance
        if max(distance - float(radius_m), 0.0) <= budget:
            reachable += 1
    return reachable, total, distances


def choose_distance_assignment(
    uav_positions: np.ndarray,
    buoy_positions: np.ndarray,
    keys: list[tuple[int, ...]],
    scheme_index: int,
    radius_m: float,
    step_length_m: float,
    steps: int = TAIL_STEPS,
) -> dict:
    """Pick a strictly more reachable legal row. Ties use distance, then pool order."""
    scheme = tuple(int(item) for item in keys[int(scheme_index)])
    scheme_reach, scheme_total, _distances = assignment_geometry(
        uav_positions, buoy_positions, scheme, radius_m, step_length_m, steps
    )
    best_index = None
    best_reach = -1
    best_total = 0.0
    for index, key in enumerate(keys):
        reach, total, _row_distances = assignment_geometry(
            uav_positions, buoy_positions, key, radius_m, step_length_m, steps
        )
        if reach <= scheme_reach:
            continue
        better_reach = best_index is None or reach > best_reach
        closer = best_index is not None and reach == best_reach and total < best_total
        if better_reach or closer:
            best_index = int(index)
            best_reach = int(reach)
            best_total = float(total)
    if best_index is None:
        return {
            "status": "no_better_geometric_candidate",
            "index": int(scheme_index),
            "assignment": scheme,
            "reachable_uavs": int(scheme_reach),
            "total_distance_m": float(scheme_total),
            "rule_version": RULE_VERSION,
        }
    return {
        "status": "selected",
        "index": int(best_index),
        "assignment": tuple(int(item) for item in keys[best_index]),
        "reachable_uavs": int(best_reach),
        "total_distance_m": float(best_total),
        "rule_version": RULE_VERSION,
    }


def _environment_hash(env: L3BBitQueueEnvV2) -> str:
    snap = capture_state(env)
    arrays = {name: np.asarray(value) for name, value in snap.items()}
    return state_hash(int(env.t), arrays)


def _advance_to_tail(env: L3BBitQueueEnvV2, planner: FormalPlanner):
    actor = planner.actor
    horizon = int(env.config.episode_horizon)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    while int(env.t) < 45:
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
    if int(env.t) != 45:
        raise RuntimeError(f"tail capture time is {env.t}")
    return position_prev, predecessor, last


def _branch(env: L3BBitQueueEnvV2, scenario_id: int) -> L3BBitQueueEnvV2:
    clone = L3BBitQueueEnvV2()
    install_light_info(clone)
    clone.reset(seed=int(scenario_id))
    restore_state(clone, capture_state(env))
    clone.rng.bit_generator.state = copy.deepcopy(env.rng.bit_generator.state)
    if not np.array_equal(clone.buoy_positions, env.buoy_positions):
        raise RuntimeError("branched buoy positions diverged")
    if _environment_hash(clone) != _environment_hash(env):
        raise RuntimeError("branched state hash diverged")
    return clone


def _execute(env: L3BBitQueueEnvV2, actor, assignment: np.ndarray, last_movement: np.ndarray) -> dict:
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    if not is_legal(chosen, int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("tail received an illegal assignment")
    option = make_option(env, chosen)
    radius = float(env.config.service_radius_m)
    tol = float(env.tol_bits)
    payload_before = float(env.payload_bits)
    rows = []
    served_sum = 0.0
    projections = 0
    safety = 0
    max_gap = 0.0
    for step_index in range(TAIL_STEPS):
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
        if step_index == 0:
            rows = [
                {
                    "uav_id": int(uav),
                    "target_buoy_id": int(buoy),
                    "idle": int(int(buoy) < 0),
                    "served_bits": 0.0,
                    "effective_service_steps": 0,
                    "entered_service_radius": 0,
                    "first_entry_step": None,
                    "path_length_m": 0.0,
                    "projection_count": 0,
                    "sinr": [],
                    "interf": [],
                }
                for uav, buoy in enumerate(chosen)
            ]
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
    payload_delta = float(env.payload_bits) - payload_before
    if abs(payload_delta - served_sum) > tol or max_gap > tol:
        raise RuntimeError("tail accounting failed")
    for item in rows:
        item["mean_sinr"] = float(np.mean(item.pop("sinr")))
        item["mean_interference"] = float(np.mean(item.pop("interf")))
    return {
        "payload_bits": payload_delta,
        "served_bits": served_sum,
        "entered_uavs": int(sum(item["entered_service_radius"] for item in rows if item["idle"] == 0)),
        "service_steps": int(sum(item["effective_service_steps"] for item in rows)),
        "projection_rate": float(projections / TAIL_STEPS),
        "safety_violation_rate": float(safety / TAIL_STEPS),
        "accounting_error_bits": float(max_gap),
        "uavs": rows,
    }


def _p1_assignment(scenario_id: int) -> str:
    with P1_WINDOWS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["scenario_id"]) == int(scenario_id) and int(row["decision_t"]) == 45:
                return str(row["assignment"])
    raise RuntimeError(f"P1 has no t=45 row for {scenario_id}")


def _paired_bootstrap(deltas: list[float]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = np.asarray(deltas, dtype=np.float64)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    count = int(data.size)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, count, count)]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": int(BOOTSTRAP_DRAWS),
        "seed": int(BOOTSTRAP_SEED),
        "positive": int(np.sum(data > float(BIT_TIE))),
        "negative": int(np.sum(data < -float(BIT_TIE))),
        "tie": int(np.sum(np.abs(data) <= float(BIT_TIE))),
    }


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


def _figures(rows: list[dict], directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    fig, ax = plt.subplots()
    ax.bar(range(count), [float(row["delta_payload_bits"]) / 1.0e6 for row in rows])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("validation scenario order")
    ax.set_ylabel("payload near minus scheme (Mbit)")
    ax.set_title(f"paired tail payload, n_scenarios={count}, unit=scenario")
    fig.savefig(directory / "paired_delta.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter(
        [int(row["scheme_reachable_uavs"]) for row in rows],
        [int(row["near_reachable_uavs"]) for row in rows],
        s=28,
    )
    ax.set_xlabel("scheme C reachable UAVs")
    ax.set_ylabel("distance-rule reachable UAVs")
    ax.set_title(f"geometric reach count, n_scenarios={count}, unit=scenario")
    fig.savefig(directory / "reachable_count.png", dpi=120)
    plt.close(fig)
    return ["paired_delta.png", "reachable_count.png"]


def run_scenario(scenario_id: int, planner: FormalPlanner, hashes: dict) -> tuple[dict, list[dict]]:
    assert_validation_scenario(scenario_id)
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    position_prev, predecessor, last = _advance_to_tail(env, planner)
    token = parameter_token(planner.actor)
    before_hash = _environment_hash(env)
    selection = planner.select_assignment(env, position_prev, predecessor)
    if selection.status != "selected" or selection.assignment is None:
        raise RuntimeError(f"scheme C aborted at t=45: {selection.reason}")
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
    if official != _p1_assignment(scenario_id):
        raise RuntimeError("t=45 scheme C assignment does not match the P1 replay")
    matrix = np.asarray([binding.assignment for binding in bindings], dtype=np.int64)
    scored = planner.scorer.infer(graph_from_env(env, position_prev, predecessor), matrix)
    if int(scored.chosen_index) != scheme_index or tuple(int(item) for item in scored.chosen_assignment) != scheme:
        raise RuntimeError("repeated scheme C score does not match the official selection")
    if _environment_hash(env) != before_hash or parameter_token(planner.actor) != token:
        raise RuntimeError("scoring changed the state or the actor")
    choice = choose_distance_assignment(
        np.asarray(env.uav_positions, dtype=np.float64),
        np.asarray(env.buoy_positions, dtype=np.float64),
        keys,
        scheme_index,
        float(env.config.service_radius_m),
        float(env.config.step_length_m),
        TAIL_STEPS,
    )
    if not is_legal(np.asarray(choice["assignment"], dtype=np.int64), int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("distance rule returned an illegal assignment")
    scheme_reach, scheme_total, scheme_distances = assignment_geometry(
        env.uav_positions, env.buoy_positions, scheme, float(env.config.service_radius_m), float(env.config.step_length_m)
    )
    arm_s = _execute(_branch(env, scenario_id), planner.actor, np.asarray(scheme, dtype=np.int64), last.copy())
    arm_n = _execute(_branch(env, scenario_id), planner.actor, np.asarray(choice["assignment"], dtype=np.int64), last.copy())
    if parameter_token(planner.actor) != token or _environment_hash(env) != before_hash:
        raise RuntimeError("tail execution changed the captured state or the actor")
    changed = [index for index, (left, right) in enumerate(zip(scheme, choice["assignment"])) if int(left) != int(right)]
    row = {
        "scenario_id": int(scenario_id),
        "decision_t": 45,
        "window_steps": TAIL_STEPS,
        "candidate_count": int(len(keys)),
        "pool_hash": digest,
        "state_hash": before_hash,
        "branch_hash_match": 1,
        "rule_version": RULE_VERSION,
        "comparison_status": choice["status"],
        "scheme_index": scheme_index,
        "scheme_assignment": official,
        "scheme_reachable_uavs": scheme_reach,
        "scheme_total_distance_m": scheme_total,
        "scheme_score": float(scored.scores[scheme_index]),
        "near_index": int(choice["index"]),
        "near_assignment": ",".join(str(int(item)) for item in choice["assignment"]),
        "near_reachable_uavs": int(choice["reachable_uavs"]),
        "near_total_distance_m": float(choice["total_distance_m"]),
        "near_score": float(scored.scores[int(choice["index"])]),
        "changed_uavs": ",".join(str(item) for item in changed),
        "payload_scheme_bits": arm_s["payload_bits"],
        "payload_near_bits": arm_n["payload_bits"],
        "delta_payload_bits": float(arm_n["payload_bits"] - arm_s["payload_bits"]),
        "entered_scheme": arm_s["entered_uavs"],
        "entered_near": arm_n["entered_uavs"],
        "service_steps_scheme": arm_s["service_steps"],
        "service_steps_near": arm_n["service_steps"],
        "projection_rate_scheme": arm_s["projection_rate"],
        "projection_rate_near": arm_n["projection_rate"],
        "safety_violation_rate_scheme": arm_s["safety_violation_rate"],
        "safety_violation_rate_near": arm_n["safety_violation_rate"],
        "accounting_error_scheme_bits": arm_s["accounting_error_bits"],
        "accounting_error_near_bits": arm_n["accounting_error_bits"],
        **hashes,
    }
    uav_rows = []
    near_distances = assignment_geometry(
        env.uav_positions,
        env.buoy_positions,
        choice["assignment"],
        float(env.config.service_radius_m),
        float(env.config.step_length_m),
    )[2]
    for arm_name, arm, distances in (("scheme", arm_s, scheme_distances), ("near", arm_n, near_distances)):
        for item, distance in zip(arm["uavs"], distances):
            uav_rows.append(
                {
                    "scenario_id": int(scenario_id),
                    "arm": arm_name,
                    "initial_distance_m": distance,
                    "geometrically_reachable": int(distance is not None and max(distance - float(env.config.service_radius_m), 0.0) <= TAIL_STEPS * float(env.config.step_length_m)),
                    **item,
                }
            )
    return row, uav_rows


def _gate(rows: list[dict], interval: dict) -> str:
    if len(rows) != len(VALIDATION_POOL):
        return "T45_COUNTERFACTUAL_BLOCKED"
    if any(int(row["branch_hash_match"]) != 1 or int(row["candidate_count"]) != 13327 for row in rows):
        return "T45_COUNTERFACTUAL_BLOCKED"
    if float(interval["ci_low"]) <= 0.0 <= float(interval["ci_high"]):
        return "T45_COUNTERFACTUAL_INCONCLUSIVE"
    return "T45_COUNTERFACTUAL_COMPLETE"


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
        print(f"t45 counterfactual {scenario}", flush=True)
        row, uavs = run_scenario(int(scenario), planner, hashes)
        rows.append(row)
        uav_rows.extend(uavs)
    _write_csv(OUTPUT / "scenario_pairs.csv", SCENARIO_FIELDS, rows)
    _write_csv(OUTPUT / "uav_pairs.csv", UAV_FIELDS, uav_rows)
    deltas = [float(row["delta_payload_bits"]) for row in rows]
    changed_deltas = [float(row["delta_payload_bits"]) for row in rows if row["comparison_status"] == "selected"]
    interval = _paired_bootstrap(deltas)
    summary = {
        "gate": _gate(rows, interval),
        "n_scenarios": len(rows),
        "split": "validation_322048_322063",
        "reserved_test_used": False,
        "decision_t": 45,
        "window_steps": TAIL_STEPS,
        "rule_version": RULE_VERSION,
        "better_geometric_scenarios": int(sum(row["comparison_status"] == "selected" for row in rows)),
        "unchanged_scenarios": int(sum(row["comparison_status"] != "selected" for row in rows)),
        "paired_all_scenarios": interval,
        "paired_changed_scenarios": _paired_bootstrap(changed_deltas) if changed_deltas else None,
        "mean_scheme_reachable_uavs": float(np.mean([row["scheme_reachable_uavs"] for row in rows])),
        "mean_near_reachable_uavs": float(np.mean([row["near_reachable_uavs"] for row in rows])),
        "max_accounting_error_bits": float(max(max(row["accounting_error_scheme_bits"], row["accounting_error_near_bits"]) for row in rows)),
        "max_safety_violation_rate": float(max(max(row["safety_violation_rate_scheme"], row["safety_violation_rate_near"]) for row in rows)),
        "source": hashes,
        "figures": _figures(rows, OUTPUT / "figures"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gate": summary["gate"], "paired_all_scenarios": interval, "better_geometric_scenarios": summary["better_geometric_scenarios"]}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
