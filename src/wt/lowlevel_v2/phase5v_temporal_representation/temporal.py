"""Candidate-conditioned history that the current env can actually recover."""
from __future__ import annotations

import numpy as np

from wt.l3b.config import L3BConfig

from .protocol import EDGE_FEATURES, GLOBAL_FEATURES, TEMP_INPUT_DIM


def observability_rows() -> list[dict]:
    """Code audit. A quantity is allowed only when the current step can recover it."""
    rows = [
        ("episode_step", 1, 1, 0, 0, 0, 0, 1),
        ("episode_fraction", 1, 1, 0, 0, 0, 0, 1),
        ("remaining_episode_fraction", 1, 1, 0, 0, 0, 0, 1),
        ("decision_index", 1, 1, 0, 0, 0, 0, 1),
        ("hold_remaining", 0, 0, 0, 0, 0, 0, 0),
        ("option_age", 0, 0, 0, 0, 0, 1, 0),
        ("previous_or_current_target", 1, 1, 1, 1, 1, 0, 1),
        ("candidate_equals_current_target", 1, 1, 1, 0, 1, 0, 1),
        ("consecutive_steps_on_target", 0, 1, 1, 0, 0, 1, 0),
        ("macro_periods_retained", 0, 1, 1, 0, 0, 1, 0),
        ("consecutive_steps_in_radius", 0, 1, 1, 0, 0, 1, 0),
        ("current_service_progress", 1, 1, 1, 1, 1, 0, 1),
        ("service_steps_remaining_counter", 0, 0, 1, 0, 0, 1, 0),
        ("switch_resets_progress", 1, 1, 1, 0, 0, 0, 1),
        ("current_velocity", 1, 1, 1, 1, 0, 0, 1),
        ("previous_velocity", 0, 1, 1, 0, 0, 1, 0),
        ("previous_position", 1, 1, 1, 1, 0, 0, 1),
        ("radial_closing_velocity", 1, 1, 1, 0, 0, 0, 1),
        ("heading_alignment", 1, 1, 1, 0, 0, 0, 1),
        ("distance_delta", 1, 1, 1, 0, 0, 0, 1),
        ("previous_step_sinr", 1, 1, 1, 0, 0, 0, 1),
        ("previous_step_rate", 1, 1, 0, 0, 0, 0, 0),
        ("recent_sinr_change", 0, 1, 1, 0, 0, 1, 0),
        ("recent_rate_change", 0, 1, 1, 0, 0, 1, 0),
        ("assignment_switch_indicator", 1, 1, 1, 0, 1, 0, 1),
        ("waiting_time_aoi_deadline", 0, 0, 0, 0, 0, 1, 0),
    ]
    payload = []
    for name, available, causal, specific, encoder, rep, buffer, allowed in rows:
        payload.append(
            {
                "feature": name,
                "available_now": available,
                "causal": causal,
                "candidate_specific": specific,
                "already_in_encoder": encoder,
                "already_in_rep_aug": rep,
                "needs_history_buffer": buffer,
                "allowed": allowed,
            }
        )
    return payload


def _stats(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros(4 * width, dtype=np.float32)
    stacked = np.stack(rows, axis=0)
    parts = []
    for feature in range(width):
        column = stacked[:, feature]
        parts.extend((float(np.mean(column)), float(np.max(column)), float(np.min(column)), float(np.std(column))))
    return np.asarray(parts, dtype=np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-6:
        return np.zeros_like(vector)
    return vector / norm


def temporal_matrix(snapshot: dict, assignments: np.ndarray, config: L3BConfig) -> np.ndarray:
    """z_temp plus three global context values. No future channel or payload is read."""
    uav = np.asarray(snapshot["uav_pos"], dtype=np.float64)
    buoy = np.asarray(snapshot["buoy_pos"], dtype=np.float64)
    progress = np.asarray(snapshot["progress"], dtype=np.float64)
    current = np.asarray(snapshot["current_assignment"], dtype=np.int64)
    velocity = np.asarray(snapshot["velocity"], dtype=np.float64)
    last_sinr = np.asarray(snapshot["last_sinr"], dtype=np.float64)
    step = max(float(config.step_length_m), 1.0e-9)
    previous = uav - velocity * step
    buoy_progress = np.zeros(buoy.shape[0], dtype=np.float64)
    for uav_id, target in enumerate(current):
        if int(target) >= 0:
            buoy_progress[int(target)] = max(float(buoy_progress[int(target)]), float(progress[uav_id]))
    chosen = np.asarray(assignments, dtype=np.int64)
    output = np.zeros((chosen.shape[0], TEMP_INPUT_DIM), dtype=np.float32)
    horizon = max(float(config.episode_horizon), 1.0)
    hold = max(float(config.option_hold), 1.0)
    time_step = float(snapshot["time_step"])
    global_context = np.asarray(
        (
            time_step / horizon,
            (horizon - time_step) / horizon,
            time_step / hold,
        ),
        dtype=np.float32,
    )
    for index, assignment in enumerate(chosen):
        edges = []
        for uav_id, target in enumerate(assignment):
            if int(target) < 0:
                continue
            buoy_id = int(target)
            offset = buoy[buoy_id] - uav[uav_id]
            distance = float(np.linalg.norm(offset))
            previous_distance = float(np.linalg.norm(buoy[buoy_id] - previous[uav_id]))
            direction = _unit(offset)
            speed = _unit(velocity[uav_id])
            valid = float(np.linalg.norm(velocity[uav_id]) >= 1.0e-6)
            same = float(int(current[uav_id]) == buoy_id)
            closing = float(np.dot(velocity[uav_id], direction))
            heading = float(np.dot(speed, direction)) if valid else 0.0
            edges.append(
                np.asarray(
                    (
                        same,
                        1.0 - same,
                        float(np.log1p(max(float(last_sinr[uav_id]), 0.0))) * same,
                        float(progress[uav_id]) * same,
                        float(buoy_progress[buoy_id]),
                        closing,
                        heading,
                        valid,
                        (distance - previous_distance) / step,
                    ),
                    dtype=np.float64,
                )
            )
        output[index] = np.concatenate((_stats(edges, len(EDGE_FEATURES)), global_context))
    return output


def edge_means(snapshot: dict, assignment: np.ndarray, config: L3BConfig) -> dict[str, float]:
    """Unaggregated means for the registered case study."""
    matrix = temporal_matrix(snapshot, np.asarray(assignment, dtype=np.int64).reshape(1, -1), config)
    values = {}
    cursor = 0
    for name in EDGE_FEATURES:
        values[name] = float(matrix[0, cursor])
        cursor += 4
    for name in GLOBAL_FEATURES:
        values[name] = float(matrix[0, cursor])
        cursor += 1
    return values


__all__ = ["edge_means", "observability_rows", "temporal_matrix"]
