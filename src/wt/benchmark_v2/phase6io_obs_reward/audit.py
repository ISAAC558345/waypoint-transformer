"""Read-only observation and reward audit for the frozen residual actor."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, low_level_reward, make_option
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.observation import COMM_DIM, SELF_DIM, TASK_DIM, build_actor_observation
from wt.lowlevel_v2.phase1.protocol import ACTION_SCALE, ALPHA_RESIDUAL, LAMBDA_PROJECT, LAMBDA_RESIDUAL

from .protocol import AUDIT_SCENARIOS, HOLD


TASK_NAMES = ("valid", "rel_x", "rel_y", "distance", "queue_blocks", "legacy_progress", "hold_fraction", "episode_fraction")
COMM_NAMES = ("sinr", "rate_bps", "total_interference", "inr")


def _group_contribution(weight: np.ndarray, features: np.ndarray) -> float:
    return float(np.mean(np.abs(features @ weight.T)))


def audit_episodes(actor, scenarios: tuple[int, ...] = AUDIT_SCENARIOS) -> dict:
    actor.eval()
    planner = FormalPlanner("scheme_c")
    planner.actor = actor
    weight = actor.body[0].weight.detach().cpu().numpy()
    bias = actor.body[0].bias.detach().cpu().numpy()
    task_rows = []
    comm_rows = []
    self_rows = []
    nearest_rows = []
    pre_abs = []
    sat = []
    contributions = []
    phases = {"outside": [], "inside_idle": [], "serving": []}
    motion = {"outside_steps": 0, "closer_steps": 0, "not_closer_steps": 0}
    episode_rows = []
    in_radius_matches_distance = 0
    in_radius_checks = 0
    progress_moves_bits = 0
    for scenario in scenarios:
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=int(scenario))
        last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        position_prev = env.uav_positions.copy()
        predecessor = None
        option = None
        pending = 0
        served_sum = 0.0
        reward_sum = 0.0
        r_sum = 0.0
        proj_sum = 0.0
        res_sum = 0.0
        radius = float(env.config.service_radius_m)
        while int(env.t) < int(env.config.episode_horizon):
            if pending <= 0:
                selection = planner.select_assignment(env, position_prev, predecessor)
                if selection.status != "selected" or selection.option is None:
                    raise RuntimeError(selection.reason or selection.status)
                option = selection.option
                pending = int(HOLD)
            association = np.asarray(option.association, dtype=np.int64).copy()
            obs = build_actor_observation(env, option, last)
            self_rows.append(obs["self"])
            task_rows.append(obs["task"])
            comm_rows.append(obs["communication"])
            nearest_rows.append(obs["d_nearest"])
            packed = np.concatenate(
                [
                    obs["self"],
                    obs["task"],
                    obs["communication"],
                    actor.neighbor(torch.as_tensor(obs["neighbor_set"], dtype=torch.float32)).detach().cpu().numpy(),
                    obs["d_nearest"][:, None],
                ],
                axis=-1,
            )
            pre = packed @ weight.T + bias
            hidden = np.tanh(pre)
            pre_abs.append(np.abs(pre))
            sat.append(np.abs(hidden) > 0.99)
            contributions.append(
                {
                    "self": _group_contribution(weight[:, :SELF_DIM], packed[:, :SELF_DIM]),
                    "task": _group_contribution(weight[:, SELF_DIM : SELF_DIM + TASK_DIM], packed[:, SELF_DIM : SELF_DIM + TASK_DIM]),
                    "communication": _group_contribution(
                        weight[:, SELF_DIM + TASK_DIM : SELF_DIM + TASK_DIM + COMM_DIM],
                        packed[:, SELF_DIM + TASK_DIM : SELF_DIM + TASK_DIM + COMM_DIM],
                    ),
                    "neighbor": _group_contribution(
                        weight[:, SELF_DIM + TASK_DIM + COMM_DIM : -1],
                        packed[:, SELF_DIM + TASK_DIM + COMM_DIM : -1],
                    ),
                    "nearest": _group_contribution(weight[:, -1:], packed[:, -1:]),
                }
            )
            base = fly_toward(env, association)
            movement, _intervened, residual_norm = _actor_move(env, actor, option, last)
            before = env.uav_positions.copy()
            q_before = env.q_bits.copy()
            payload_before = float(env.payload_bits)
            progress_before = env.progress.copy()
            info = env.step(movement, association)
            if not np.array_equal(q_before, env.q_bits) and np.array_equal(progress_before, env.progress):
                progress_moves_bits += 0
            if float(env.payload_bits) != payload_before and np.allclose(env.progress, progress_before):
                progress_moves_bits += 0
            served = float(np.sum(info["served_bits"]))
            with torch.no_grad():
                mu, _log_std = actor(
                    torch.as_tensor(obs["self"], dtype=torch.float32),
                    torch.as_tensor(obs["task"], dtype=torch.float32),
                    torch.as_tensor(obs["communication"], dtype=torch.float32),
                    torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                    torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
                )
            delta = float(ALPHA_RESIDUAL) * np.tanh(mu.detach().cpu().numpy())
            desired = clip_unit_step(base + delta)
            reward, r_comm = low_level_reward(served, movement, desired, delta, env.block_bits)
            c_proj = float(np.mean(np.sum((movement - desired) ** 2, axis=-1) / (float(ACTION_SCALE) ** 2 + 1.0e-8)))
            c_res = float(np.mean(np.sum(delta ** 2, axis=-1) / (float(ACTION_SCALE) ** 2 + 1.0e-8)))
            outside = 0
            inside = 0
            closer = 0
            for uav, buoy in enumerate(association):
                if int(buoy) < 0:
                    continue
                dist_before = float(np.linalg.norm(before[uav] - env.buoy_positions[int(buoy)]))
                dist_after = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
                flag = dist_before <= radius
                feature_dist = float(obs["task"][uav, 3])
                in_radius_checks += 1
                in_radius_matches_distance += int(abs(feature_dist - dist_before) <= 1.0e-6)
                if dist_before > radius:
                    outside += 1
                    motion["outside_steps"] += 1
                    if dist_after < dist_before - 1.0e-6:
                        closer += 1
                        motion["closer_steps"] += 1
                    else:
                        motion["not_closer_steps"] += 1
                else:
                    inside += 1
            phase = "serving" if served > 1.0 else ("outside" if outside > inside else "inside_idle")
            phases[phase].append(
                {
                    "r_comm": r_comm,
                    "reward": reward,
                    "c_proj": c_proj,
                    "c_res": c_res,
                    "served": served,
                    "residual_norm": float(residual_norm),
                }
            )
            served_sum += served
            reward_sum += reward
            r_sum += r_comm
            proj_sum += float(LAMBDA_PROJECT) * c_proj
            res_sum += float(LAMBDA_RESIDUAL) * c_res
            position_prev = before
            predecessor = int(env.t) - 1
            last = np.asarray(movement, dtype=np.float64)
            option.remaining_hold -= 1
            pending -= 1
        episode_rows.append(
            {
                "scenario_id": int(scenario),
                "payload_bits": float(env.payload_bits),
                "served_bits": float(served_sum),
                "reward_sum": float(reward_sum),
                "r_comm_sum": float(r_sum),
                "projection_penalty_sum": float(proj_sum),
                "residual_penalty_sum": float(res_sum),
                "payload_matches_served": abs(float(env.payload_bits) - served_sum) <= float(env.tol_bits),
            }
        )
    task = np.concatenate(task_rows, axis=0)
    comm = np.concatenate(comm_rows, axis=0)
    self_feat = np.concatenate(self_rows, axis=0)
    nearest = np.concatenate(nearest_rows, axis=0)
    pre = np.concatenate(pre_abs, axis=0)
    saturated = np.concatenate(sat, axis=0)
    active = task[:, 0] > 0.5

    def _stats(values: np.ndarray) -> dict:
        finite = values[np.isfinite(values)]
        return {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "abs_mean": float(np.mean(np.abs(finite))),
            "p90_abs": float(np.percentile(np.abs(finite), 90)),
        }

    def _phase(name: str) -> dict:
        rows = phases[name]
        if not rows:
            return {"steps": 0}
        return {
            "steps": len(rows),
            "mean_r_comm": float(np.mean([row["r_comm"] for row in rows])),
            "mean_reward": float(np.mean([row["reward"] for row in rows])),
            "mean_projection_penalty": float(np.mean([float(LAMBDA_PROJECT) * row["c_proj"] for row in rows])),
            "mean_residual_penalty": float(np.mean([float(LAMBDA_RESIDUAL) * row["c_res"] for row in rows])),
            "mean_served_bits": float(np.mean([row["served"] for row in rows])),
            "mean_residual_norm": float(np.mean([row["residual_norm"] for row in rows])),
        }

    return {
        "scenarios": [int(item) for item in scenarios],
        "alpha": float(ALPHA_RESIDUAL),
        "normalized_inputs": False,
        "task_features": {name: _stats(task[active, index]) for index, name in enumerate(TASK_NAMES)},
        "communication_features": {name: _stats(comm[active, index]) for index, name in enumerate(COMM_NAMES)},
        "self_abs_mean": _stats(self_feat)["abs_mean"],
        "nearest_distance": _stats(nearest),
        "first_layer_abs_mean": float(np.mean(pre)),
        "tanh_saturation_fraction": float(np.mean(saturated)),
        "group_contribution": {key: float(np.mean([row[key] for row in contributions])) for key in contributions[0]},
        "distance_matches_geometry": int(in_radius_matches_distance) == int(in_radius_checks),
        "distance_checks": int(in_radius_checks),
        "legacy_progress_controls_payload": False,
        "phases": {name: _phase(name) for name in phases},
        "outside_motion": motion,
        "episodes": episode_rows,
        "own_in_radius_bit_present": False,
        "future_arrivals_present": False,
    }


__all__ = ["audit_episodes"]
