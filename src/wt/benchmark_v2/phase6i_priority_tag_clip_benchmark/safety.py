"""Online geometry for new safety events. E14 events stay without invented positions."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase1.protocol import D_SAFE_M


def closest_pair(positions: np.ndarray) -> dict:
    points = np.asarray(positions, dtype=np.float64)
    best_i = 0
    best_j = 1
    best = float("inf")
    for i in range(int(points.shape[0])):
        for j in range(i + 1, int(points.shape[0])):
            distance = float(np.linalg.norm(points[i] - points[j]))
            if distance < best:
                best = distance
                best_i = i
                best_j = j
    return {"uav_i": int(best_i), "uav_j": int(best_j), "distance_m": float(best), "margin_m": float(best - float(D_SAFE_M))}


def install_geometry(worker) -> tuple:
    project = worker.projector.project
    step = worker.env.step

    def _project(env, desired, option):
        result = project(env, desired, option)
        worker.geometry_projection = {
            "positions_before": np.asarray(env.uav_positions, dtype=np.float64).copy(),
            "desired": np.asarray(desired, dtype=np.float64).copy(),
            "projected": np.asarray(result.projected_actions, dtype=np.float64).copy(),
            "constraint_status": str(result.constraint_status),
            "predicted_distance_m": float(result.minimum_predicted_distance),
            "intervention_norm": float(result.intervention_norm),
            "finite": bool(result.finite),
            "envelope_ok": bool(result.envelope_ok),
        }
        return result

    def _step(movement, association):
        before = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
        info = step(movement, association)
        worker.geometry_after = {
            "positions_before": before,
            "positions_after": np.asarray(worker.env.uav_positions, dtype=np.float64).copy(),
            "executed": np.asarray(movement, dtype=np.float64).copy(),
        }
        return info

    worker.projector.project = _project
    worker.env.step = _step
    return project, step


def restore_geometry(worker, originals) -> None:
    project, step = originals
    worker.projector.project = project
    worker.env.step = step


def event_from(worker, record: dict, holder: dict) -> dict | None:
    safety = int(record["safety_violation"])
    failure = int(record["constraint_failure"])
    if not safety and not failure:
        return None
    projection = getattr(worker, "geometry_projection", None)
    after = getattr(worker, "geometry_after", None)
    event = {
        "variant": holder["variant"],
        "model_seed": int(holder["model_seed"]),
        "ppo_update": int(holder["ppo_update"]),
        "env_index": int(worker.env_index),
        "safety_violation": safety,
        "constraint_failure": failure,
        "safe_distance_m": float(D_SAFE_M),
    }
    if projection is None or after is None:
        event["geometry"] = "unknown"
        return event
    after_pair = closest_pair(after["positions_after"])
    i = int(after_pair["uav_i"])
    j = int(after_pair["uav_j"])
    before_i = np.asarray(projection["positions_before"][i], dtype=np.float64)
    before_j = np.asarray(projection["positions_before"][j], dtype=np.float64)
    before_distance = float(np.linalg.norm(before_i - before_j))
    event.update(
        {
            "geometry": "measured",
            "uav_i": i,
            "uav_j": j,
            "position_before_i": before_i.tolist(),
            "position_before_j": before_j.tolist(),
            "position_after_i": np.asarray(after["positions_after"][i], dtype=np.float64).tolist(),
            "position_after_j": np.asarray(after["positions_after"][j], dtype=np.float64).tolist(),
            "distance_before_m": before_distance,
            "distance_after_m": after_pair["distance_m"],
            "margin_after_m": after_pair["margin_m"],
            "predicted_distance_m": float(projection["predicted_distance_m"]),
            "predicted_margin_m": float(projection["predicted_distance_m"]) - float(D_SAFE_M),
            "desired_action": projection["desired"].reshape(-1).tolist(),
            "projected_action": projection["projected"].reshape(-1).tolist(),
            "executed_action": after["executed"].reshape(-1).tolist(),
            "constraint_status": projection["constraint_status"],
            "intervention_norm": float(projection["intervention_norm"]),
            "projection_finite": bool(projection["finite"]),
            "envelope_ok": bool(projection["envelope_ok"]),
        }
    )
    return event
