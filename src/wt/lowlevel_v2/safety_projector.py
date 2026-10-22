"""Deterministic iterative safety projector. Does not silently claim safety on failure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv

from .contracts import CollisionConfig, clip_unit_step, env_target, minimum_pairwise_distance
from .protocol import D_SAFE_PLACEHOLDER_M, SERVICE_SAFETY_MARGIN_PLACEHOLDER_M, UAV_PHYSICAL_RADIUS_PLACEHOLDER_M, COLLISION_MARGIN_PLACEHOLDER_M
from .service_safe_envelope import envelope_active, inner_radius, next_in_envelope


@dataclass
class ProjectionResult:
    projected_actions: np.ndarray
    intervention_norm: float
    constraint_status: str
    minimum_predicted_distance: float
    envelope_ok: bool
    finite: bool


def _default_collision() -> CollisionConfig:
    return CollisionConfig(
        uav_physical_radius=float(UAV_PHYSICAL_RADIUS_PLACEHOLDER_M),
        collision_margin=float(COLLISION_MARGIN_PLACEHOLDER_M),
        placeholder=True,
    )


def _apply_step(env: L3BEnv, positions: np.ndarray, movement: np.ndarray) -> np.ndarray:
    move = clip_unit_step(movement)
    nxt = positions + move * float(env.config.step_length_m)
    lo = np.array([0.0, 0.0])
    hi = np.array([float(env.config.area_width), float(env.config.area_height)])
    return np.clip(nxt, lo, hi)


class SafetyProjector:
    def __init__(
        self,
        collision: CollisionConfig | None = None,
        safety_margin: float | None = None,
        max_iters: int = 64,
    ) -> None:
        self.collision = collision or _default_collision()
        self.safety_margin = float(SERVICE_SAFETY_MARGIN_PLACEHOLDER_M if safety_margin is None else safety_margin)
        self.max_iters = int(max_iters)

    def project(
        self,
        env: L3BEnv,
        desired_actions: np.ndarray,
        option: TaskOption,
    ) -> ProjectionResult:
        desired = clip_unit_step(desired_actions)
        start = np.asarray(env.uav_positions, dtype=np.float64).copy()
        d_safe = float(self.collision.d_safe)
        if abs(d_safe - D_SAFE_PLACEHOLDER_M) > 1.0e-12 and self.collision.placeholder:
            pass
        nxt = _apply_step(env, start, desired)
        active = envelope_active(env, option, self.safety_margin)
        radius = inner_radius(env, self.safety_margin)
        finite = bool(np.all(np.isfinite(desired)) and np.all(np.isfinite(nxt)))
        if not finite:
            return ProjectionResult(desired, 0.0, "failure_nonfinite", float("nan"), False, False)
        for _ in range(self.max_iters):
            changed = False
            m_count = int(start.shape[0])
            for i in range(m_count):
                for j in range(i + 1, m_count):
                    delta = nxt[j] - nxt[i]
                    dist = float(np.linalg.norm(delta))
                    if dist >= d_safe - 1.0e-9:
                        continue
                    axis = delta / dist if dist > 1.0e-12 else np.array([1.0, 0.0])
                    corr = 0.5 * (d_safe - dist) * axis
                    nxt[i] -= corr
                    nxt[j] += corr
                    changed = True
            nxt = np.clip(nxt, [0.0, 0.0], [env.config.area_width, env.config.area_height])
            for uav, flag in enumerate(active):
                if not flag:
                    continue
                tgt = env_target(int(option.association[uav]), int(env.config.num_buoys))
                rel = nxt[uav] - env.buoy_positions[tgt]
                dist = float(np.linalg.norm(rel))
                if dist > radius:
                    scale = radius / max(dist, 1.0e-12)
                    nxt[uav] = env.buoy_positions[tgt] + rel * scale
                    changed = True
            delta_pos = nxt - start
            max_step = float(env.config.step_length_m)
            norms = np.linalg.norm(delta_pos, axis=-1, keepdims=True)
            over = norms[:, 0] > max_step + 1.0e-9
            if np.any(over):
                delta_pos[over] *= max_step / np.clip(norms[over], 1.0e-12, None)
                nxt = start + delta_pos
                changed = True
            if not changed:
                break
        actions = clip_unit_step((nxt - start) / max(float(env.config.step_length_m), 1.0e-12))
        predicted = _apply_step(env, start, actions)
        min_d = minimum_pairwise_distance(predicted)
        env_ok = bool(np.all(next_in_envelope(env, option, predicted, self.safety_margin)))
        sep_ok = bool(min_d + 1.0e-6 >= d_safe)
        finite = bool(np.all(np.isfinite(actions)) and np.all(np.isfinite(predicted)))
        status = "ok" if finite and sep_ok and env_ok else "failure"
        intervention = float(np.linalg.norm(actions - desired))
        return ProjectionResult(
            projected_actions=actions,
            intervention_norm=intervention,
            constraint_status=status,
            minimum_predicted_distance=min_d,
            envelope_ok=env_ok,
            finite=finite,
        )


__all__ = ["ProjectionResult", "SafetyProjector"]
