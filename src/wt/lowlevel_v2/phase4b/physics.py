"""One-step and H=3 communication outcomes using the real L3-B env model."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv, active_links
from wt.lowlevel_v2.contracts import clip_unit_step, env_target, minimum_pairwise_distance
from wt.lowlevel_v2.observation import communication_features as comm_feat
from wt.lowlevel_v2.phase1a.diagnose import predicted_next
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.phase4b.snapshot import restore_env
from wt.lowlevel_v2.service_safe_envelope import next_in_envelope


def feasible_joint(env: L3BEnv, option, movement: np.ndarray) -> dict:
    move = clip_unit_step(movement)
    legal = bool(np.all(np.linalg.norm(np.asarray(movement, dtype=np.float64), axis=-1) <= 1.0 + 1.0e-6))
    nxt = predicted_next(env, move)
    sep = minimum_pairwise_distance(nxt)
    sep_ok = bool(sep + 1.0e-6 >= D_SAFE_M)
    env_ok = bool(np.all(next_in_envelope(env, option, nxt, SERVICE_SAFETY_MARGIN_M)))
    return {"legal": legal, "sep": float(sep), "sep_ok": sep_ok, "envelope_ok": env_ok, "feasible": bool(legal and sep_ok and env_ok)}


def evaluate_joint_step(env: L3BEnv, snap: dict, option, movement: np.ndarray, uav: int) -> dict:
    restore_env(env, snap)
    flags = feasible_joint(env, option, movement)
    assoc = np.asarray(option.association, dtype=np.int64)
    pay0 = float(env.payload_bits)
    prog0 = float(np.sum(env.progress))
    restore_env(env, snap)
    env.step(clip_unit_step(movement), assoc)
    comm = comm_feat(env)
    tgt = env_target(int(assoc[uav]), int(env.config.num_buoys))
    in_range = bool(tgt >= 0 and env._in_range(uav, tgt))
    links = active_links(env)
    team_rate = float(np.sum(env.last_rate))
    others = float(np.sum(env.last_rate) - env.last_rate[uav])
    return {
        **flags,
        "sinr_u": float(env.last_sinr[uav]),
        "rate_u": float(env.last_rate[uav]),
        "team_rate": team_rate,
        "others_rate": others,
        "payload_inc": float(env.payload_bits - pay0),
        "progress_inc": float(np.sum(env.progress) - prog0),
        "total_interference_u": float(comm["total_interference"][uav]),
        "inr_u": float(comm["interference_to_noise"][uav]),
        "in_range": in_range,
        "n_active_links": len(links),
        "min_sep_after": float(minimum_pairwise_distance(env.uav_positions)),
    }


def evaluate_h3(env: L3BEnv, snap: dict, option, first_move: np.ndarray, base_fn, uav: int) -> dict:
    restore_env(env, snap)
    assoc = np.asarray(option.association, dtype=np.int64)
    pay0 = float(env.payload_bits)
    prog0 = float(np.sum(env.progress))
    team_int = 0.0
    move = clip_unit_step(first_move)
    for h in range(3):
        if h == 0:
            env.step(move, assoc)
        else:
            a_base, _ = base_fn(env, option)
            env.step(clip_unit_step(a_base), assoc)
        team_int += float(np.sum(env.last_rate))
    return {
        "h3_team_rate_integral": team_int,
        "h3_payload_inc": float(env.payload_bits - pay0),
        "h3_progress_inc": float(np.sum(env.progress) - prog0),
        "h3_min_sep": float(minimum_pairwise_distance(env.uav_positions)),
    }


__all__ = ["evaluate_h3", "evaluate_joint_step", "feasible_joint"]
