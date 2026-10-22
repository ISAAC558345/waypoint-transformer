"""L3-B goal-conditioned low-level features. Dims come from the live env schema."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase9.const import HOLD
from wt.htw_phase9.features import feature_dict_l3b, stack_feature_dicts
from wt.l3b.env import L3BEnv


def build_schema(env: L3BEnv | None = None) -> dict[str, Any]:
    env = env or L3BEnv()
    probe = feature_dict_l3b(env.reset(seed=0), env)
    m_count = int(env.config.num_uavs)
    k_count = int(env.config.num_buoys)
    option_dim = k_count + 1
    extra_uav = option_dim + 2 + 1 + 1 + 1  # one-hot, rel_xy, dist_steps, progress, target_valid
    extra_global = 2  # remaining_hold, remaining_horizon
    return {
        "max_uavs": m_count,
        "max_buoys": k_count,
        "idle_index": k_count,
        "idle_assignment": -1,
        "base_uav_dim": int(probe["uav"].shape[-1]),
        "base_buoy_dim": int(probe["buoy"].shape[-1]),
        "base_global_dim": int(probe["global"].shape[-1]),
        "option_onehot_dim": option_dim,
        "uav_dim": int(probe["uav"].shape[-1]) + extra_uav,
        "buoy_dim": int(probe["buoy"].shape[-1]),
        "global_dim": int(probe["global"].shape[-1]) + extra_global,
        "action_dim": 2,
        "action_range": "tanh in [-1,1]^2 then env unit-disk clip times step_length_m",
        "normalization": {
            "uav_xy": "2 * p / area - 1 (same as feature_dict_l3b)",
            "rel_xy": " (target - uav) / (area_width, area_height); idle -> 0",
            "distance": "euclidean_m / step_length_m; idle -> 0",
            "progress": "env service_progress in [0,1)",
            "remaining_hold": "pending / HOLD",
            "remaining_horizon": "(T - t) / T",
            "onehot": "buoy k in 0..K-1 or idle token K; never index buoy[-1] on idle",
        },
        "units": {
            "position": "meters, then area-normalized",
            "distance_feature": "movement steps (meters / step_length_m)",
            "d_out_reward": "max(dist_m - service_radius_m, 0) / step_length_m",
            "hold": int(HOLD),
            "horizon": int(env.config.episode_horizon),
            "step_length_m": float(env.config.step_length_m),
            "service_radius_m": float(env.config.service_radius_m),
            "area_width_m": float(env.config.area_width),
            "area_height_m": float(env.config.area_height),
        },
        "forbids_future_arrivals": True,
        "hardcoded_5x21": False,
    }


def attach_option(base: Mapping[str, np.ndarray], env: L3BEnv, info: Mapping[str, Any], option: TaskOption, schema: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not np.array_equal(option.association, option.targets):
        raise RuntimeError("low-level features refused rematching: association != targets")
    m_count = int(schema["max_uavs"])
    k_count = int(schema["max_buoys"])
    idle_index = int(schema["idle_index"])
    positions = np.asarray(info["uav_positions_m"], dtype=np.float32)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float32)
    progress = np.asarray(info["service_progress"], dtype=np.float32)
    step = max(float(env.config.step_length_m), 1.0e-9)
    width = max(float(env.config.area_width), 1.0e-9)
    height = max(float(env.config.area_height), 1.0e-9)
    onehot = np.zeros((m_count, idle_index + 1), dtype=np.float32)
    rel = np.zeros((m_count, 2), dtype=np.float32)
    dist = np.zeros((m_count, 1), dtype=np.float32)
    prog = np.zeros((m_count, 1), dtype=np.float32)
    valid = np.zeros((m_count, 1), dtype=np.float32)
    targets = np.asarray(option.targets, dtype=np.int64)
    uav_mask = np.asarray(base["uav_mask"], dtype=bool).copy()
    buoy_mask = np.asarray(base["buoy_mask"], dtype=bool).copy()
    for uav in range(int(env.config.num_uavs)):
        prog[uav, 0] = float(progress[uav]) if uav < progress.shape[0] else 0.0
        tgt = int(targets[uav]) if uav < targets.shape[0] else -1
        if tgt < 0 or tgt >= k_count or not bool(buoy_mask[tgt]):
            onehot[uav, idle_index] = 1.0
            continue
        onehot[uav, tgt] = 1.0
        valid[uav, 0] = 1.0
        delta = buoys[tgt] - positions[uav]
        rel[uav, 0] = float(delta[0]) / width
        rel[uav, 1] = float(delta[1]) / height
        dist[uav, 0] = float(np.linalg.norm(delta)) / step
    extra = np.concatenate((onehot, rel, dist, prog, valid), axis=-1)
    uav = np.concatenate((np.asarray(base["uav"], dtype=np.float32), extra), axis=-1)
    remaining_hold = float(option.remaining_hold) / float(max(int(HOLD), 1))
    remaining_horizon = float(max(int(env.config.episode_horizon) - int(info["time_step"]), 0)) / float(max(int(env.config.episode_horizon), 1))
    global_ = np.concatenate((np.asarray(base["global"], dtype=np.float32), np.asarray((remaining_hold, remaining_horizon), dtype=np.float32)), axis=0)
    if uav.shape != (m_count, int(schema["uav_dim"])) or global_.shape != (int(schema["global_dim"]),):
        raise ValueError(f"goal-conditioned feature shape mismatch: uav={uav.shape} global={global_.shape}")
    return {
        "uav": uav,
        "buoy": np.asarray(base["buoy"], dtype=np.float32).copy(),
        "uav_mask": uav_mask,
        "buoy_mask": buoy_mask,
        "global": global_,
    }


def encode_obs(env: L3BEnv, info: Mapping[str, Any], option: TaskOption, schema: Mapping[str, Any], last_arrival: np.ndarray, last_positions: np.ndarray) -> dict[str, np.ndarray]:
    base = feature_dict_l3b(info, env, last_arrival=last_arrival, last_positions=last_positions)
    return attach_option(base, env, info, option, schema)


__all__ = ["attach_option", "build_schema", "encode_obs", "stack_feature_dicts"]
