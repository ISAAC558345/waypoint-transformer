"""Steer and speed around fly_toward. Maximum turn and speed are round-1 limits."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.phase1.protocol import SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.service_safe_envelope import envelope_active, inner_radius

from .protocol import EXCESSIVE_HEADING_DEG, EXCESSIVE_SPEED_FRACTION, MAX_TURN_DEG

SERVICE_MARGIN = float(SERVICE_SAFETY_MARGIN_M)


def adaptive_command(
    base: np.ndarray,
    pre_tanh: np.ndarray,
    *,
    steer_center: float,
    steer_temperature: float,
    speed_bias: float,
    max_turn_deg: float = MAX_TURN_DEG,
) -> np.ndarray:
    """World-frame unit-step command. +angle rotates the base direction counterclockwise.

    A zero base (idle or coincident target) produces a hover. Nonfinite inputs hover.
    """
    direction = np.asarray(base, dtype=np.float64)
    pre = np.asarray(pre_tanh, dtype=np.float64)
    if direction.ndim != 2 or direction.shape[1] != 2 or pre.shape != direction.shape:
        raise ValueError("adaptive command expects base and pre-activation shaped (M, 2)")
    command = np.zeros_like(direction)
    if not np.all(np.isfinite(direction)) or not np.all(np.isfinite(pre)):
        return command
    if not np.isfinite(steer_center) or not np.isfinite(speed_bias) or float(steer_temperature) <= 0.0:
        return command
    turn = np.tanh((pre[:, 0] - float(steer_center)) / float(steer_temperature)) * float(max_turn_deg)
    turn = np.clip(turn, -float(max_turn_deg), float(max_turn_deg))
    speed = np.clip((np.tanh(pre[:, 1] + float(speed_bias)) + 1.0) * 0.5, 0.0, 1.0)
    radians = np.deg2rad(turn)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    rotated = np.column_stack(
        (
            cosine * direction[:, 0] - sine * direction[:, 1],
            sine * direction[:, 0] + cosine * direction[:, 1],
        )
    )
    norms = np.linalg.norm(direction, axis=-1)
    active = norms > 1.0e-8
    rotated[active] /= norms[active, None]
    command[active] = rotated[active] * speed[active, None]
    return clip_unit_step(command)


def heading_delta_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Signed degrees from `right` to `left`. Positive is counterclockwise. Zero vectors contribute 0."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    cross = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]
    dot = np.sum(a * b, axis=-1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    angle = np.degrees(np.arctan2(cross, dot))
    return np.where(norms > 1.0e-8, angle, 0.0)


def projection_report(
    env,
    option,
    commanded: np.ndarray,
    executed: np.ndarray,
    *,
    d_safe: float,
) -> dict[str, float]:
    """Classify the projector change without relaxing the constraint."""
    command = np.asarray(commanded, dtype=np.float64)
    done = np.asarray(executed, dtype=np.float64)
    step = float(env.config.step_length_m)
    start = np.asarray(env.uav_positions, dtype=np.float64)
    commanded_next = start + command * step
    heading = heading_delta_deg(done, command)
    speed_gap = np.linalg.norm(done, axis=-1) - np.linalg.norm(command, axis=-1)
    separation = _pair_distances(commanded_next)
    collision = bool(np.isfinite(separation) and separation < float(d_safe) - 1.0e-9)
    flags = envelope_active(env, option, float(SERVICE_MARGIN))
    radius = inner_radius(env, float(SERVICE_MARGIN))
    envelope = False
    for uav, active in enumerate(flags):
        if not active:
            continue
        target = int(option.association[uav])
        if target < 0:
            continue
        dist = float(np.linalg.norm(commanded_next[uav] - env.buoy_positions[target]))
        if dist > radius + 1.0e-9:
            envelope = True
            break
    base = _safe_base(env, option)
    active = np.linalg.norm(base, axis=-1) > 1.0e-8
    if np.any(active):
        command_turn = float(np.mean(np.abs(heading_delta_deg(command[active], base[active]))))
        executed_turn = float(np.mean(np.abs(heading_delta_deg(done[active], base[active]))))
    else:
        command_turn = 0.0
        executed_turn = 0.0
    abs_heading = np.abs(heading)
    abs_speed = np.abs(speed_gap)
    return {
        "heading_correction_deg": float(np.mean(abs_heading)),
        "max_heading_correction_deg": float(np.max(abs_heading)),
        "speed_correction": float(np.mean(abs_speed)),
        "max_speed_correction": float(np.max(abs_speed)),
        "excessive_heading": int(float(np.max(abs_heading)) >= float(EXCESSIVE_HEADING_DEG)),
        "excessive_speed": int(float(np.max(abs_speed)) >= float(EXCESSIVE_SPEED_FRACTION)),
        "collision_risk": int(collision),
        "envelope_risk": int(envelope),
        "command_turn_deg": command_turn,
        "executed_turn_deg": executed_turn,
        "command_speed": float(np.mean(np.linalg.norm(command, axis=-1))),
        "executed_speed": float(np.mean(np.linalg.norm(done, axis=-1))),
    }


def _safe_base(env, option) -> np.ndarray:
    from wt.l3b.policies import fly_toward

    return fly_toward(env, np.asarray(option.association, dtype=np.int64))


def _pair_distances(positions: np.ndarray) -> float:
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) < 2:
        return float("inf")
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    mask = np.triu(np.ones(dist.shape, dtype=bool), k=1)
    return float(np.min(dist[mask]))


__all__ = ["adaptive_command", "heading_delta_deg", "projection_report"]
