"""t=45 counterfactual that keeps in-range assignments fixed."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6i_execution_attribution.run_attribution import assert_validation_scenario, sha256_file
from wt.benchmark_v2.phase6i_t45_counterfactual.run_counterfactual import (
    TAIL_STEPS,
    _advance_to_tail,
    _branch,
    _environment_hash,
    _execute,
    _paired_bootstrap,
    _p1_assignment,
    pool_hash,
)
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner, graph_from_env
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, SCHEME_C_CKPT, SCHEME_C_SHA256
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import DECODER_CKPT, RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import bind_keys, frozen_candidate_keys
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_t45_active_preserve"
RULE_VERSION = "preserve_fixed_then_max_reach_then_min_distance_v1"
IDLE_DISTANCE_M = 1.0e9
MIN_CHANGED_SCENARIOS = 3

SCENARIO_FIELDS = (
    "scenario_id",
    "decision_t",
    "window_steps",
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
    "payload_scheme_bits",
    "payload_preserve_bits",
    "delta_payload_bits",
    "fixed_in_radius_served_scheme_bits",
    "fixed_in_radius_served_preserve_bits",
    "entered_scheme",
    "entered_preserve",
    "projection_rate_scheme",
    "projection_rate_preserve",
    "safety_violation_rate_scheme",
    "safety_violation_rate_preserve",
    "accounting_error_scheme_bits",
    "accounting_error_preserve_bits",
    "config_hash",
    "scheme_c_sha256",
    "mappo_sha256",
    "decoder_sha256",
)
UAV_FIELDS = (
    "scenario_id",
    "arm",
    "uav_id",
    "role",
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
    step_length_m: float,
    steps: int = TAIL_STEPS,
) -> tuple[list[str], list[float | None]]:
    """Classify each UAV from the current target. In-range and idle stay fixed."""
    budget = float(steps) * float(step_length_m)
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
        elif max(distance - float(radius_m), 0.0) <= budget:
            roles.append("fixed_outside_reachable")
        else:
            roles.append("reassignable_unreachable")
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
        if role != "reassignable_unreachable" and int(candidate[uav]) != int(scheme[uav]):
            return False
    return True


def choose_preserving_assignment(
    uav_positions: np.ndarray,
    buoy_positions: np.ndarray,
    keys: list[tuple[int, ...]],
    scheme_index: int,
    radius_m: float,
    step_length_m: float,
    steps: int = TAIL_STEPS,
) -> dict:
    """Reassign only UAVs that cannot geometrically arrive. Fixed targets stay put."""
    scheme = tuple(int(item) for item in keys[int(scheme_index)])
    roles, _distances = uav_roles(uav_positions, buoy_positions, scheme, radius_m, step_length_m, steps)
    reassignable = [uav for uav, role in enumerate(roles) if role == "reassignable_unreachable"]
    fixed = [uav for uav, role in enumerate(roles) if role != "reassignable_unreachable"]
    base = {
        "roles": roles,
        "fixed_uavs": fixed,
        "reassignable_uavs": reassignable,
        "rule_version": RULE_VERSION,
        "preserving_candidate_count": 0,
    }
    if not reassignable:
        reach, total = 0, 0.0
        return {
            **base,
            "status": "no_reassignable_uav",
            "index": int(scheme_index),
            "assignment": scheme,
            "reachable_uavs": reach,
            "total_distance_m": total,
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
    ax.set_ylabel("payload preserve minus scheme (Mbit)")
    ax.set_title(f"paired tail payload, n_scenarios={count}, unit=scenario")
    fig.savefig(directory / "paired_delta.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter(
        [float(row["fixed_in_radius_served_scheme_bits"]) / 1.0e6 for row in rows],
        [float(row["fixed_in_radius_served_preserve_bits"]) / 1.0e6 for row in rows],
        s=28,
    )
    ax.set_xlabel("in-range UAV served bits, scheme (Mbit)")
    ax.set_ylabel("in-range UAV served bits, preserve (Mbit)")
    ax.set_title(f"fixed in-range service, n_scenarios={count}, unit=scenario")
    fig.savefig(directory / "fixed_service.png", dpi=120)
    plt.close(fig)
    return ["paired_delta.png", "fixed_service.png"]


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
    positions = np.asarray(env.uav_positions, dtype=np.float64)
    buoys = np.asarray(env.buoy_positions, dtype=np.float64)
    choice = choose_preserving_assignment(
        positions,
        buoys,
        keys,
        scheme_index,
        float(env.config.service_radius_m),
        float(env.config.step_length_m),
        TAIL_STEPS,
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
        TAIL_STEPS,
    )
    arm_s = _execute(_branch(env, scenario_id), planner.actor, np.asarray(scheme, dtype=np.int64), last.copy())
    arm_p = _execute(_branch(env, scenario_id), planner.actor, np.asarray(choice["assignment"], dtype=np.int64), last.copy())
    if parameter_token(planner.actor) != token or _environment_hash(env) != before_hash:
        raise RuntimeError("tail execution changed the captured state or the actor")
    in_radius = [uav for uav, role in enumerate(choice["roles"]) if role == "fixed_in_radius"]
    row = {
        "scenario_id": int(scenario_id),
        "decision_t": 45,
        "window_steps": TAIL_STEPS,
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
        "payload_scheme_bits": arm_s["payload_bits"],
        "payload_preserve_bits": arm_p["payload_bits"],
        "delta_payload_bits": float(arm_p["payload_bits"] - arm_s["payload_bits"]),
        "fixed_in_radius_served_scheme_bits": float(sum(arm_s["uavs"][uav]["served_bits"] for uav in in_radius)),
        "fixed_in_radius_served_preserve_bits": float(sum(arm_p["uavs"][uav]["served_bits"] for uav in in_radius)),
        "entered_scheme": arm_s["entered_uavs"],
        "entered_preserve": arm_p["entered_uavs"],
        "projection_rate_scheme": arm_s["projection_rate"],
        "projection_rate_preserve": arm_p["projection_rate"],
        "safety_violation_rate_scheme": arm_s["safety_violation_rate"],
        "safety_violation_rate_preserve": arm_p["safety_violation_rate"],
        "accounting_error_scheme_bits": arm_s["accounting_error_bits"],
        "accounting_error_preserve_bits": arm_p["accounting_error_bits"],
        **hashes,
    }
    uav_rows = []
    for arm_name, arm, assignment in (
        ("scheme", arm_s, scheme),
        ("preserve", arm_p, choice["assignment"]),
    ):
        for item in arm["uavs"]:
            uav = int(item["uav_id"])
            buoy = int(assignment[uav])
            distance = None if buoy < 0 else float(np.linalg.norm(positions[uav] - buoys[buoy]))
            uav_rows.append(
                {
                    "scenario_id": int(scenario_id),
                    "arm": arm_name,
                    "role": choice["roles"][uav],
                    "initial_distance_m": distance,
                    "geometrically_reachable": int(
                        distance is not None and max(distance - float(env.config.service_radius_m), 0.0) <= TAIL_STEPS * float(env.config.step_length_m)
                    ),
                    **item,
                }
            )
    return row, uav_rows


def _gate(rows: list[dict], interval: dict) -> str:
    if len(rows) != len(VALIDATION_POOL):
        return "T45_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED"
    if any(int(row["branch_hash_match"]) != 1 or int(row["candidate_count"]) != 13327 for row in rows):
        return "T45_ACTIVE_ASSIGNMENT_PRESERVED_BLOCKED"
    changed = sum(row["comparison_status"] == "selected" for row in rows)
    crosses = float(interval["ci_low"]) <= 0.0 <= float(interval["ci_high"])
    if crosses or changed < MIN_CHANGED_SCENARIOS:
        return "T45_ACTIVE_ASSIGNMENT_PRESERVED_INCONCLUSIVE"
    return "T45_ACTIVE_ASSIGNMENT_PRESERVED_COMPLETE"


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
        print(f"t45 preserve {scenario}", flush=True)
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
        "min_changed_scenarios_for_complete": MIN_CHANGED_SCENARIOS,
        "changed_scenarios": int(len(changed_deltas)),
        "unchanged_scenarios": int(len(rows) - len(changed_deltas)),
        "paired_all_scenarios": interval,
        "paired_changed_scenarios": _paired_bootstrap(changed_deltas) if changed_deltas else None,
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
                "paired_all_scenarios": interval,
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    main()
