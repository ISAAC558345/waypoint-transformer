"""Replay registered validation episodes for attribution. No training."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL, D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    MAPPO_SHA256,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    assert_episode_scenario,
)
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import DECODER_CKPT, RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.safety_projector import SafetyProjector

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_execution_attribution"
DECISION_TIMES = (0, 15, 30, 45)
HOLD = 15
LATE_ENTRY_FRACTION = 0.5
HIGH_PROJECTION_RATE = 0.5

WINDOW_FIELDS = (
    "scenario_id",
    "decision_t",
    "window_steps",
    "previous_assignment",
    "assignment",
    "candidate_count",
    "pool_hash",
    "selection_status",
    "selection_seconds",
    "payload_before_bits",
    "payload_after_bits",
    "served_bits_sum",
    "arrival_bits_sum",
    "accounting_error_bits",
    "mean_sinr",
    "mean_interference",
    "projection_rate",
    "safety_violation_rate",
    "mean_queue_before_bits",
    "mean_queue_after_bits",
    "episode_complete",
    "source_log",
    "config_hash",
    "checkpoint_hash",
)
UAV_FIELDS = (
    "scenario_id",
    "decision_t",
    "uav_id",
    "target_buoy_id",
    "idle",
    "initial_distance_m",
    "first_entry_step",
    "entered_service_radius",
    "effective_service_steps",
    "served_bits",
    "path_length_m",
    "net_displacement_m",
    "end_distance_m",
    "projection_count",
    "projection_norm_mean",
    "mean_sinr",
    "mean_interference",
    "hold_remaining",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_validation_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or not (322064 > value >= 322048):
        raise RuntimeError(f"attribution refused scenario {value}")
    assert_episode_scenario(value, allow_holdout=False)


def blank(value) -> str:
    if value is None:
        return ""
    return str(value)


def _move(env, actor, option, last_movement):
    obs = build_actor_observation(env, option, last_movement)
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(obs["self"], dtype=torch.float32),
            torch.as_tensor(obs["task"], dtype=torch.float32),
            torch.as_tensor(obs["communication"], dtype=torch.float32),
            torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
        )
    pre = mu.detach().cpu().numpy().astype(np.float64)
    delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
    desired = clip_unit_step(fly_toward(env, option.association) + delta)
    projected = SafetyProjector(
        CollisionConfig(0.4, 0.2, placeholder=False),
        safety_margin=float(SERVICE_SAFETY_MARGIN_M),
    ).project(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    executed = clip_unit_step(executed)
    per_uav_norm = np.linalg.norm(desired - executed, axis=1)
    intervened = bool(failure or float(np.max(per_uav_norm)) > 1.0e-3)
    return executed, intervened, per_uav_norm


def _distance(env, uav: int, buoy: int) -> float | None:
    if int(buoy) < 0:
        return None
    return float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))


def observation_flags(row: dict) -> dict[str, int]:
    """Fixed labels. A row may match more than one observation."""
    if int(row["idle"]) == 1:
        return {"idle": 1}
    window_steps = int(row["window_steps"])
    served = float(row["served_bits"])
    entered = int(row["entered_service_radius"]) == 1
    zero = served <= float(row["tol_bits"])
    first = row["first_entry_step"]
    late = entered and first is not None and int(first) >= int(window_steps * LATE_ENTRY_FRACTION)
    short_service = int(row["effective_service_steps"]) * 2 < window_steps
    high_projection = float(row["projection_count"]) / max(window_steps, 1) >= HIGH_PROJECTION_RATE
    return {
        "reachability": int(not entered),
        "communication_observation": int(entered and zero),
        "utilization": int(late and short_service and not zero),
        "projection": int(high_projection and (not entered or zero)),
        "delivered": int(entered and not zero and not (late and short_service)),
    }


def run_episode(scenario_id: int, planner: FormalPlanner, config_hash: str, checkpoint_hash: str) -> tuple[list[dict], list[dict], dict]:
    assert_validation_scenario(scenario_id)
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    actor = planner.actor
    horizon = int(env.config.episode_horizon)
    radius = float(env.config.service_radius_m)
    tol = float(env.tol_bits)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    previous = np.full(env.config.num_uavs, -1, dtype=np.int64)
    windows: list[dict] = []
    uav_rows: list[dict] = []
    anomalies = []
    max_error = 0.0
    while int(env.t) < horizon:
        decision_t = int(env.t)
        if decision_t not in DECISION_TIMES:
            raise RuntimeError(f"decision time {decision_t} is outside 0,15,30,45")
        clock = time.perf_counter()
        selection = planner.select_assignment(env, position_prev, predecessor)
        elapsed = time.perf_counter() - clock
        if selection.status != "selected" or selection.option is None or selection.assignment is None:
            raise RuntimeError(f"selection aborted at {scenario_id} t={decision_t}: {selection.reason}")
        option = selection.option
        assignment = np.asarray(selection.assignment, dtype=np.int64).copy()
        window_steps = min(HOLD, horizon - decision_t)
        payload_before = float(env.payload_bits)
        queue_before = float(np.mean(env.q_bits))
        start_pos = np.asarray(env.uav_positions, dtype=np.float64).copy()
        per = []
        for uav, buoy in enumerate(assignment):
            per.append(
                {
                    "uav_id": int(uav),
                    "target": int(buoy),
                    "idle": int(int(buoy) < 0),
                    "initial_distance": _distance(env, uav, int(buoy)),
                    "first_entry": None,
                    "entered": 0,
                    "service_steps": 0,
                    "served": 0.0,
                    "path": 0.0,
                    "proj_count": 0,
                    "proj_norm": 0.0,
                    "sinr": [],
                    "interf": [],
                    "start": start_pos[uav].copy(),
                }
            )
        served_sum = 0.0
        arrival_sum = 0.0
        sinr_vals = []
        interf_vals = []
        projections = 0
        safety = 0
        step_errors = []
        for step_index in range(window_steps):
            t_before = int(env.t)
            step_start = np.asarray(env.uav_positions, dtype=np.float64).copy()
            movement, intervened, per_uav_norm = _move(env, actor, option, last)
            payload_step_before = float(env.payload_bits)
            info = env.step(movement, assignment)
            served = np.asarray(info["served_bits"], dtype=np.float64)
            gap = abs(float(env.payload_bits) - payload_step_before - float(np.sum(served)))
            step_errors.append(gap)
            max_error = max(max_error, float(np.max(np.abs(env.conservation_error_bits()))))
            if gap > tol:
                anomalies.append({"scenario_id": int(scenario_id), "t": t_before, "payload_gap_bits": gap})
            comm = communication_features(env)
            served_sum += float(np.sum(served))
            arrival_sum += float(np.sum(info["generated_arrival_bits"]))
            sinr_vals.append(float(np.mean(info["sinr"])))
            interf_vals.append(float(np.mean(comm["total_interference"])))
            projections += int(intervened)
            safety += int(minimum_pairwise_distance(env.uav_positions) + 1.0e-9 < float(D_SAFE_M))
            for uav, buoy in enumerate(assignment):
                item = per[uav]
                dist = _distance(env, uav, int(buoy))
                if dist is not None and dist <= radius:
                    item["entered"] = 1
                    if item["first_entry"] is None:
                        item["first_entry"] = int(step_index)
                if int(buoy) >= 0 and float(served[int(buoy)]) > tol:
                    item["service_steps"] += 1
                    item["served"] += float(served[int(buoy)])
                item["path"] += float(np.linalg.norm(movement[uav]) * env.config.step_length_m)
                item["proj_count"] += int(float(per_uav_norm[uav]) > 1.0e-3)
                item["proj_norm"] += float(per_uav_norm[uav])
                item["sinr"].append(float(info["sinr"][uav]))
                item["interf"].append(float(comm["total_interference"][uav]))
                item["end"] = np.asarray(env.uav_positions[uav], dtype=np.float64).copy()
                item["end_distance"] = dist
            option.remaining_hold -= 1
            last = np.asarray(movement, dtype=np.float64).copy()
            position_prev = step_start
            predecessor = t_before
        hold_remaining = int(option.remaining_hold)
        window = {
            "scenario_id": int(scenario_id),
            "decision_t": decision_t,
            "window_steps": int(window_steps),
            "previous_assignment": ",".join(str(int(item)) for item in previous),
            "assignment": ",".join(str(int(item)) for item in assignment),
            "candidate_count": int(selection.candidate_count),
            "pool_hash": str(selection.pool_hash),
            "selection_status": str(selection.status),
            "selection_seconds": float(elapsed),
            "payload_before_bits": payload_before,
            "payload_after_bits": float(env.payload_bits),
            "served_bits_sum": float(served_sum),
            "arrival_bits_sum": float(arrival_sum),
            "accounting_error_bits": float(max(step_errors) if step_errors else 0.0),
            "mean_sinr": float(np.mean(sinr_vals)),
            "mean_interference": float(np.mean(interf_vals)),
            "projection_rate": float(projections / window_steps),
            "safety_violation_rate": float(safety / window_steps),
            "mean_queue_before_bits": queue_before,
            "mean_queue_after_bits": float(np.mean(env.q_bits)),
            "episode_complete": 0,
            "source_log": "fresh_validation_replay",
            "config_hash": config_hash,
            "checkpoint_hash": checkpoint_hash,
        }
        if abs(window["payload_after_bits"] - payload_before - served_sum) > tol:
            anomalies.append({"scenario_id": int(scenario_id), "t": decision_t, "window_gap_bits": abs(window["payload_after_bits"] - payload_before - served_sum)})
        windows.append(window)
        for item in per:
            end = item.get("end", item["start"])
            uav_rows.append(
                {
                    "scenario_id": int(scenario_id),
                    "decision_t": decision_t,
                    "uav_id": item["uav_id"],
                    "target_buoy_id": item["target"],
                    "idle": item["idle"],
                    "initial_distance_m": item["initial_distance"],
                    "first_entry_step": item["first_entry"],
                    "entered_service_radius": item["entered"],
                    "effective_service_steps": item["service_steps"],
                    "served_bits": item["served"],
                    "path_length_m": item["path"],
                    "net_displacement_m": float(np.linalg.norm(end - item["start"])),
                    "end_distance_m": item.get("end_distance"),
                    "projection_count": item["proj_count"],
                    "projection_norm_mean": item["proj_norm"] / window_steps,
                    "mean_sinr": float(np.mean(item["sinr"])) if item["sinr"] else None,
                    "mean_interference": float(np.mean(item["interf"])) if item["interf"] else None,
                    "hold_remaining": hold_remaining,
                    "window_steps": window_steps,
                    "tol_bits": tol,
                }
            )
        previous = assignment.copy()
    payload = float(env.payload_bits)
    served_total = float(sum(row["served_bits_sum"] for row in windows))
    if abs(payload - served_total) > tol:
        anomalies.append({"scenario_id": int(scenario_id), "episode_gap_bits": abs(payload - served_total)})
    complete = int(int(env.t) == horizon and not anomalies)
    for row in windows:
        row["episode_complete"] = complete
    return windows, uav_rows, {"scenario_id": int(scenario_id), "payload_bits": payload, "anomalies": anomalies, "max_accounting_error_bits": max_error, "complete": complete}


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: blank(row.get(key)) for key in fields})


def _figures(windows: list[dict], uav_rows: list[dict], directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    assigned = [row for row in uav_rows if int(row["idle"]) == 0]
    n_scen = len({int(row["scenario_id"]) for row in windows})
    saved = []
    fig, ax = plt.subplots()
    ax.scatter([row["initial_distance_m"] for row in assigned], [row["end_distance_m"] for row in assigned], s=12)
    ax.set_xlabel("initial distance (m)")
    ax.set_ylabel("end distance (m)")
    ax.set_title(f"target distance, n_scenarios={n_scen}, unit=UAV-window")
    fig.savefig(directory / "target_distance.png", dpi=120)
    plt.close(fig)
    saved.append("target_distance.png")
    entered = [row["first_entry_step"] for row in assigned if row["first_entry_step"] is not None]
    fig, ax = plt.subplots()
    ax.hist(entered, bins=range(0, 16), align="left")
    ax.set_xlabel("first entry step within window")
    ax.set_ylabel("UAV-windows")
    ax.set_title(f"service entry, n_scenarios={n_scen}, unit=UAV-window")
    fig.savefig(directory / "entry_time.png", dpi=120)
    plt.close(fig)
    saved.append("entry_time.png")
    fig, ax = plt.subplots()
    ax.hist([row["served_bits_sum"] / 1.0e6 for row in windows], bins=12)
    ax.set_xlabel("window served bits (Mbit)")
    ax.set_ylabel("windows")
    ax.set_title(f"window payload, n_scenarios={n_scen}, unit=window")
    fig.savefig(directory / "window_served.png", dpi=120)
    plt.close(fig)
    saved.append("window_served.png")
    by_scene = {}
    for row in windows:
        by_scene.setdefault(int(row["scenario_id"]), []).append(row)
    fig, ax = plt.subplots()
    ax.scatter(
        [float(np.mean([item["projection_rate"] for item in rows])) for rows in by_scene.values()],
        [float(rows[-1]["payload_after_bits"]) / 1.0e6 for rows in by_scene.values()],
        s=20,
    )
    ax.set_xlabel("mean projection rate")
    ax.set_ylabel("episode payload (Mbit)")
    ax.set_title(f"projection vs payload, n_scenarios={n_scen}, unit=scenario")
    fig.savefig(directory / "projection_vs_payload.png", dpi=120)
    plt.close(fig)
    saved.append("projection_vs_payload.png")
    fig, ax = plt.subplots()
    ax.scatter([row["mean_sinr"] for row in assigned], [row["served_bits"] / 1.0e6 for row in assigned], s=12)
    ax.set_xlabel("mean SINR")
    ax.set_ylabel("served bits (Mbit)")
    ax.set_title(f"SINR vs service, n_scenarios={n_scen}, unit=UAV-window")
    fig.savefig(directory / "sinr_vs_served.png", dpi=120)
    plt.close(fig)
    saved.append("sinr_vs_served.png")
    return saved


def summarize(windows: list[dict], uav_rows: list[dict], episodes: list[dict], source: dict) -> dict:
    flags = [observation_flags(row) for row in uav_rows]
    names = ("idle", "reachability", "communication_observation", "utilization", "projection", "delivered")
    counts = {name: int(sum(item.get(name, 0) for item in flags)) for name in names}
    candidates = {name: counts[name] for name in ("reachability", "communication_observation", "utilization", "projection")}
    reach_times = [int(row["decision_t"]) for row, flag in zip(uav_rows, flags) if flag.get("reachability")]
    full_hold_reach_miss = any(time_step != 45 for time_step in reach_times)
    primary = "unconfirmed"
    if full_hold_reach_miss and candidates["reachability"] == max(candidates.values()):
        primary = "reachability"
    payloads = [float(row["payload_bits"]) for row in episodes]
    return {
        "gate": "EXECUTION_ATTRIBUTION_COMPLETE" if all(int(row["complete"]) == 1 for row in episodes) else "EXECUTION_ATTRIBUTION_BLOCKED",
        "n_scenarios": len(episodes),
        "n_windows": len(windows),
        "n_uav_rows": len(uav_rows),
        "scenario_ids": [int(row["scenario_id"]) for row in episodes],
        "split": "validation_322048_322063",
        "reserved_test_used": False,
        "historical_step_logs_present": False,
        "missing_fields": [
            "No saved full-episode step log remained after the repository cleanup, so previous assignment traces, per-step positions, and counterfactual scores were not available to reuse.",
            "high_level_counterfactual: not present. Scheme C alternatives at the same state were not re-scored, so a high-level assignment error is unconfirmed.",
        ],
        "accounting_anomaly_count": int(sum(len(row["anomalies"]) for row in episodes)),
        "anomalies": [item for row in episodes for item in row["anomalies"]],
        "max_accounting_error_bits": float(max(row["max_accounting_error_bits"] for row in episodes)),
        "payload_bits_mean": float(np.mean(payloads)),
        "payload_bits_median": float(np.median(payloads)),
        "label_counts": counts,
        "primary_bottleneck_candidate": primary,
        "reachability_decision_times": sorted(set(reach_times)),
        "high_level_assignment_error": "unconfirmed",
        "source": source,
    }


def main() -> dict:
    torch.set_num_threads(1)
    if sha256_file(SCHEME_C_CKPT) != SCHEME_C_SHA256 or sha256_file(primary_checkpoint()) != MAPPO_SHA256:
        raise RuntimeError("frozen checkpoint hash mismatch")
    if not DECODER_CKPT.is_file():
        raise RuntimeError("matching decoder checkpoint is missing")
    if any(int(item) in RESERVED_TEST for item in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_hash = config_sha256()
    checkpoint_hash = sha256_file(primary_checkpoint())
    planner = FormalPlanner("scheme_c")
    windows: list[dict] = []
    uav_rows: list[dict] = []
    episodes = []
    for scenario in VALIDATION_POOL:
        print(f"attribution scenario {scenario}", flush=True)
        w, u, episode = run_episode(int(scenario), planner, config_hash, checkpoint_hash)
        windows.extend(w)
        uav_rows.extend(u)
        episodes.append(episode)
    _write_csv(OUTPUT / "window_rows.csv", WINDOW_FIELDS, windows)
    _write_csv(OUTPUT / "uav_window_rows.csv", UAV_FIELDS, uav_rows)
    summary = summarize(
        windows,
        uav_rows,
        episodes,
        {
            "policy_file": "results/phase6iaa_formal_planner/active_policy.json",
            "scheme_c_sha256": SCHEME_C_SHA256,
            "matching_decoder_sha256": sha256_file(DECODER_CKPT),
            "mappo_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "replay": "FormalPlanner scheme_c + frozen phase6g actor, validation pool only",
        },
    )
    summary["figures"] = _figures(windows, uav_rows, OUTPUT / "figures")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("gate", "n_scenarios", "primary_bottleneck_candidate", "payload_bits_mean", "accounting_anomaly_count")}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
