"""Recompute the two registered safety events for the same UAV pair."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_signal_audit.episode import run_episode
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.integrity import load_e10_rows
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.safety import _matches, _totals
from wt.lowlevel_v2.contracts import pairwise_distances

from .protocol import COUNTER_TOLERANCE_M, DISTANCE_THRESHOLD_M, FORMAL_ACTOR_SHA256, PROJECTOR_TOLERANCE_M, SAFETY_CASES
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256


def _pair_distance(positions: np.ndarray, left: int, right: int) -> float:
    return float(pairwise_distances(np.asarray(positions, dtype=np.float64))[int(left), int(right)])


def _judgment(distance: float) -> dict:
    return {
        "distance_m": float(distance),
        "signed_margin_m": float(distance) - float(DISTANCE_THRESHOLD_M),
        "counter_violation": bool(float(distance) + float(COUNTER_TOLERANCE_M) < float(DISTANCE_THRESHOLD_M)),
        "projector_would_accept_distance": bool(float(distance) + float(PROJECTOR_TOLERANCE_M) >= float(DISTANCE_THRESHOLD_M)),
    }


def _predicted_positions(before: np.ndarray, action: np.ndarray, step_length: float, width: float, height: float) -> np.ndarray:
    move = np.asarray(action, dtype=np.float64)
    norms = np.linalg.norm(move, axis=-1, keepdims=True)
    move = np.where(norms > 1.0, move / np.clip(norms, 1.0e-9, None), move)
    positions = np.asarray(before, dtype=np.float64) + move * float(step_length)
    return np.clip(positions, [0.0, 0.0], [float(width), float(height)])


def localize() -> dict:
    formal = file_sha256(FORMAL_ACTOR)
    saved = {(row["variant"], int(row["model_seed"]), int(row["scenario_id"])): row for row in load_e10_rows()}
    cases = []
    for case in SAFETY_CASES:
        current = run_episode(case["variant"], int(case["model_seed"]), int(case["scenario_id"]))
        baseline = run_episode(case["baseline_variant"], int(case["baseline_seed"]), int(case["scenario_id"]))
        current_match = _matches(_totals(current), saved[(case["variant"], int(case["model_seed"]), int(case["scenario_id"]))])
        baseline_match = _matches(_totals(baseline), saved[(case["baseline_variant"], int(case["baseline_seed"]), int(case["scenario_id"]))])
        worker = current["worker"]
        step_length = float(worker.env.config.step_length_m)
        width = float(worker.env.config.area_width)
        height = float(worker.env.config.area_height)
        events = []
        if current_match["matches_saved"] and baseline_match["matches_saved"]:
            for step in current["steps"]:
                if int(step["safety_violation"]) != 1:
                    continue
                left = int(step["uav_i"])
                right = int(step["uav_j"])
                before = _pair_distance(step["positions_before"], left, right)
                after = _pair_distance(step["positions_after"], left, right)
                predicted_positions = _predicted_positions(step["positions_before"], step["action"], step_length, width, height)
                predicted = _pair_distance(predicted_positions, left, right)
                events.append(
                    {
                        "step": int(step["step"]),
                        "uav_i": left,
                        "uav_j": right,
                        "before": _judgment(before),
                        "after": _judgment(after),
                        "predicted_pair": _judgment(predicted),
                        "predicted_minus_after_m": float(predicted - after),
                        "projector_minimum_predicted_distance_m": float(step["predicted_distance_m"]),
                        "projection_status": step["projection_status"],
                        "constraint_failure": int(step["constraint_failure"]),
                        "numeric_boundary": bool(abs(after - float(DISTANCE_THRESHOLD_M)) <= float(PROJECTOR_TOLERANCE_M)),
                        "same_pair_not_global_previous": True,
                    }
                )
        cases.append(
            {
                "case": case,
                "matched_saved_rows": bool(current_match["matches_saved"] and baseline_match["matches_saved"]),
                "current_match": current_match,
                "baseline_match": baseline_match,
                "events": events,
            }
        )
    return {
        "formal_actor_sha256": formal,
        "formal_actor_hash_ok": formal == FORMAL_ACTOR_SHA256,
        "cases": cases,
        "matched": all(item["matched_saved_rows"] for item in cases) and formal == FORMAL_ACTOR_SHA256,
    }
