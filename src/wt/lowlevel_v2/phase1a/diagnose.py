"""Classify projector reasons from desired-action geometry. Does not modify SafetyProjector."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import clip_unit_step, env_target, minimum_pairwise_distance
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.service_safe_envelope import next_in_envelope


def predicted_next(env: L3BEnv, movement: np.ndarray) -> np.ndarray:
    move = clip_unit_step(movement)
    nxt = env.uav_positions + move * float(env.config.step_length_m)
    return np.clip(nxt, [0.0, 0.0], [env.config.area_width, env.config.area_height])


def desired_constraint_flags(env: L3BEnv, option: TaskOption, a_des: np.ndarray) -> dict[str, bool]:
    raw_over = bool(np.any(np.linalg.norm(np.asarray(a_des, dtype=np.float64), axis=-1) > 1.0 + 1.0e-9))
    nxt = predicted_next(env, a_des)
    pair = bool(minimum_pairwise_distance(nxt) + 1.0e-9 < D_SAFE_M)
    envl = bool(not np.all(next_in_envelope(env, option, nxt, SERVICE_SAFETY_MARGIN_M)))
    return {"PAIRWISE_COLLISION": pair, "SERVICE_SAFE_ENVELOPE": envl, "STEP_LIMIT": raw_over}


def intervention_label(flags: dict[str, bool], intervened: bool) -> str:
    if not intervened:
        return "NONE"
    active = [name for name, hit in flags.items() if hit]
    if len(active) >= 2:
        return "MULTI_CONSTRAINT"
    if len(active) == 1:
        return active[0]
    return "OTHER"


def predict_sinr_rate(env: L3BEnv, movement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from wt.l3b.env import active_links, sinr_rates_for_links

    saved = env.uav_positions.copy()
    env.uav_positions = predicted_next(env, movement)
    links = active_links(env)
    sinr, rate = sinr_rates_for_links(env, links)
    env.uav_positions = saved
    return sinr, rate
