"""Localize the two registered E10 safety events. Thresholds stay unchanged."""
from __future__ import annotations

import numpy as np

from .episode import run_episode
from .protocol import BIT_TOLERANCE, PATH_TOLERANCE_M, SAFETY_DISTANCE_TOLERANCE_M


def _totals(episode: dict) -> dict:
    steps = episode["steps"]
    return {
        "steps": len(steps),
        "payload_bits": float(steps[-1]["payload_bits"]),
        "served_tagged_bits": float(sum(step["served_tagged_bits"] for step in steps)),
        "served_ordinary_bits": float(sum(step["served_ordinary_bits"] for step in steps)),
        "dropped_bits": float(sum(float(np.sum(step["dropped"])) for step in steps)),
        "path_length_m": float(sum(step["path_step"] for step in steps)),
        "safety_violations": float(sum(step["safety_violation"] for step in steps)),
        "constraint_failures": float(sum(step["constraint_failure"] for step in steps)),
    }


def _matches(fresh: dict, saved: dict) -> dict:
    gaps = {
        key: abs(float(fresh[key]) - float(saved[key]))
        for key in ("payload_bits", "served_tagged_bits", "served_ordinary_bits", "dropped_bits", "safety_violations", "constraint_failures")
    }
    path_gap = abs(float(fresh["path_length_m"]) - float(saved["path_length_m"]))
    ok = bool(all(gap <= float(BIT_TOLERANCE) for gap in gaps.values()) and path_gap <= float(PATH_TOLERANCE_M) and int(fresh["steps"]) == int(saved["steps"]))
    return {"matches_saved": ok, "bit_gaps": gaps, "path_gap_m": float(path_gap)}


def _labels(step: dict, baseline: dict | None) -> list[str]:
    labels = ["counter_tolerance_1e-9_and_projector_tolerance_1e-6"]
    distance = float(step["distance_after_m"])
    threshold = float(step["counter_threshold_m"])
    predicted = float(step["predicted_distance_m"])
    if baseline is not None:
        gap = float(np.max(np.abs(step["positions_after"] - baseline["positions_after"])))
        if gap > float(PATH_TOLERANCE_M):
            labels.append("policy_changed_positions_at_event")
        else:
            labels.append("positions_match_baseline_at_event")
        if int(baseline["safety_violation"]) == 1:
            labels.append("baseline_also_violates_this_step")
    if str(step["projection_status"]) == "ok" and distance + float(SAFETY_DISTANCE_TOLERANCE_M) < threshold:
        labels.append("accepted_projection_then_realized_violation")
    if str(step["projection_status"]) != "ok":
        labels.append("projector_rejected_action")
    if int(step["constraint_failure"]) == 1 and float(np.max(np.abs(step["action"]))) == 0.0:
        labels.append("executed_action_zeroed")
    if predicted + 1.0e-6 >= float(step["projector_threshold_m"]) and distance + float(SAFETY_DISTANCE_TOLERANCE_M) < threshold:
        labels.append("predicted_safe_realized_violation")
    if abs(distance - threshold) <= 1.0e-6:
        labels.append("near_numeric_boundary")
    if abs(float(step["projector_threshold_m"]) - threshold) > 0.0:
        labels.append("projector_and_counter_thresholds_differ")
    return labels


def _view(step: dict) -> dict:
    return {
        "step": int(step["step"]),
        "uav_i": int(step["uav_i"]),
        "uav_j": int(step["uav_j"]),
        "distance_m": float(step["distance_after_m"]),
        "distance_before_m": float(step["distance_before_m"]),
        "predicted_distance_m": float(step["predicted_distance_m"]),
        "threshold_m": float(step["counter_threshold_m"]),
        "projector_threshold_m": float(step["projector_threshold_m"]),
        "projection_status": step["projection_status"],
        "projection_intervention": float(step["projection_intervention"]),
        "constraint_failure": int(step["constraint_failure"]),
        "safety_violation": int(step["safety_violation"]),
        "pre_tanh": np.asarray(step["pre_tanh"], dtype=np.float64).reshape(-1).tolist(),
        "desired": np.asarray(step["desired"], dtype=np.float64).reshape(-1).tolist(),
        "action": np.asarray(step["action"], dtype=np.float64).reshape(-1).tolist(),
    }


def localize(saved_index: dict, case: dict) -> dict:
    current = run_episode(case["variant"], int(case["model_seed"]), int(case["scenario_id"]))
    baseline = run_episode(case["baseline_variant"], int(case["baseline_seed"]), int(case["scenario_id"]))
    current_saved = saved_index[(case["variant"], int(case["model_seed"]), int(case["scenario_id"]))]
    baseline_saved = saved_index[(case["baseline_variant"], int(case["baseline_seed"]), int(case["scenario_id"]))]
    current_match = _matches(_totals(current), current_saved)
    baseline_match = _matches(_totals(baseline), baseline_saved)
    matched = bool(current_match["matches_saved"] and baseline_match["matches_saved"])
    events = []
    if matched:
        baseline_steps = {int(step["step"]): step for step in baseline["steps"]}
        current_steps = {int(step["step"]): step for step in current["steps"]}
        for step in current["steps"]:
            if int(step["safety_violation"]) != 1:
                continue
            previous = current_steps.get(int(step["step"]) - 1)
            events.append(
                {
                    "variant": case["variant"],
                    "model_seed": int(case["model_seed"]),
                    "scenario_id": int(case["scenario_id"]),
                    "step": int(step["step"]),
                    "uav_i": int(step["uav_i"]),
                    "uav_j": int(step["uav_j"]),
                    "distance_m": float(step["distance_after_m"]),
                    "distance_before_m": float(step["distance_before_m"]),
                    "threshold_m": float(step["counter_threshold_m"]),
                    "counter_tolerance_m": float(SAFETY_DISTANCE_TOLERANCE_M),
                    "projector_tolerance_m": 1.0e-6,
                    "predicted_distance_m": float(step["predicted_distance_m"]),
                    "projection_status": step["projection_status"],
                    "projection_intervention": float(step["projection_intervention"]),
                    "constraint_failure": int(step["constraint_failure"]),
                    "labels": _labels(step, baseline_steps.get(int(step["step"]))),
                    "event_step": _view(step),
                    "previous_step": None if previous is None else _view(previous),
                }
            )
    return {
        "case": case,
        "matched_saved_rows": matched,
        "current_match": current_match,
        "baseline_match": baseline_match,
        "events": events,
        "causal_interpretation": "reported" if matched else "stopped_because_replay_mismatch",
    }
