"""Original residual map with an explicit scale. The scale multiplies tanh(pre) only."""
from __future__ import annotations

import math

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL, D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.safety_projector import SafetyProjector


def scale_limits(alpha: float) -> dict[str, float]:
    """Reachable set of clip(fly_toward + alpha * tanh(pre)).

    Each tanh component lies in (-1, 1), so the residual is a world-frame box
    of half-width `alpha`, not a heading or a speed command.
    """
    value = float(alpha)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("residual scale must be a positive finite number")
    forward = 1.0 - value
    heading = math.degrees(math.atan2(value, forward)) if forward > 0.0 else 180.0
    return {
        "alpha": value,
        "max_heading_deg": float(heading if forward > 0.0 else 180.0),
        "min_speed_fraction": float(max(forward, 0.0)),
        "speed_at_max_heading": float(math.hypot(max(forward, 0.0), value) if forward > 0.0 else 0.0),
        "can_stop": bool(value >= 1.0),
        "can_reverse": bool(value > 1.0),
        "max_step_fraction": 1.0,
        "component_limit": value,
    }


def residual_command(base: np.ndarray, pre_tanh: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (clipped command, world-frame delta).

    Order: tanh, multiply by alpha, add to fly_toward, then clip the Euclidean
    norm to one step. Alpha is not applied after the clip.
    """
    direction = np.asarray(base, dtype=np.float64)
    pre = np.asarray(pre_tanh, dtype=np.float64)
    if direction.ndim != 2 or direction.shape[1] != 2 or pre.shape != direction.shape:
        raise ValueError("residual command expects base and pre-activation shaped (M, 2)")
    value = float(alpha)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("residual scale must be a positive finite number")
    delta = value * np.tanh(pre)
    return clip_unit_step(direction + delta), delta


def heading_delta_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Signed degrees from `right` to `left`. Positive is counterclockwise."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    cross = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]
    dot = np.sum(a * b, axis=-1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    angle = np.degrees(np.arctan2(cross, dot))
    return np.where(norms > 1.0e-8, angle, 0.0)


def make_projector() -> SafetyProjector:
    return SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))


def scaled_move(env, actor, option, last_movement: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    """Frozen deterministic mean, original residual map, then the original projector."""
    if actor.training:
        raise RuntimeError("residual actor left eval mode")
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
    base = fly_toward(env, np.asarray(option.association, dtype=np.int64))
    desired, delta = residual_command(base, pre, float(alpha))
    projected = make_projector().project(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    executed = clip_unit_step(executed)
    active = np.linalg.norm(base, axis=-1) > 1.0e-8
    if np.any(active):
        command_turn = float(np.mean(np.abs(heading_delta_deg(desired[active], base[active]))))
        executed_turn = float(np.mean(np.abs(heading_delta_deg(executed[active], base[active]))))
    else:
        command_turn = 0.0
        executed_turn = 0.0
    heading_fix = heading_delta_deg(executed, desired)
    speed_gap = np.linalg.norm(executed, axis=-1) - np.linalg.norm(desired, axis=-1)
    report = {
        "failure": bool(failure),
        "intervened": bool(failure or float(projected.intervention_norm) > 1.0e-3),
        "intervention_norm": float(projected.intervention_norm),
        "correction_norm": float(np.linalg.norm(executed - desired)),
        "command_turn_deg": command_turn,
        "executed_turn_deg": executed_turn,
        "heading_correction_deg": float(np.mean(np.abs(heading_fix))),
        "max_heading_correction_deg": float(np.max(np.abs(heading_fix))),
        "speed_correction": float(np.mean(np.abs(speed_gap))),
        "command_speed": float(np.mean(np.linalg.norm(desired, axis=-1))),
        "executed_speed": float(np.mean(np.linalg.norm(executed, axis=-1))),
        "residual_norm": float(np.mean(np.linalg.norm(delta, axis=-1))),
        "pre_abs_mean": float(np.mean(np.abs(pre))),
    }
    return executed, report


def assert_official_match(env, actor, option, last_movement: np.ndarray) -> None:
    """Alpha 0.25 must reproduce the frozen executor before any other scale runs."""
    if abs(float(ALPHA_RESIDUAL) - 0.25) > 1.0e-12:
        raise RuntimeError("official residual scale is no longer 0.25")
    official, intervened, _norm = _actor_move(env, actor, option, last_movement)
    ours, report = scaled_move(env, actor, option, last_movement, float(ALPHA_RESIDUAL))
    if not np.allclose(official, ours, atol=1.0e-8, equal_nan=False):
        raise RuntimeError("alpha 0.25 does not match the frozen residual executor")
    if bool(intervened) != bool(report["intervened"]):
        raise RuntimeError("alpha 0.25 projector flag does not match the frozen executor")


__all__ = [
    "assert_official_match",
    "heading_delta_deg",
    "make_projector",
    "residual_command",
    "scale_limits",
    "scaled_move",
]
