"""Read-only scale audit of the frozen MAPPO-V2 actor. Does not train or save weights."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker, low_level_reward
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary
from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.contracts import clip_unit_step
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.residual_policy import CentralizedCritic
from wt.lowlevel_v2.phase1.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    SERVICE_SAFETY_MARGIN_M,
)
from wt.lowlevel_v2.contracts import CollisionConfig
from wt.lowlevel_v2.safety_projector import SafetyProjector
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.residual_policy import ResidualActor


def main() -> None:
    actor = load_primary()
    actor.eval()
    body = actor.body[0]
    weight = body.weight.detach().cpu().numpy()
    groups = {
        "self": (0, 5),
        "task": (5, 13),
        "comm": (13, 17),
        "neighbor": (17, 33),
        "nearest": (33, 34),
    }
    critic = CentralizedCritic(expected_critic_dim(6, 12))
    worker = TrainWorker(0, 0, actor, critic)
    projector = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))
    feature_abs = {name: [] for name in ("self", "task_rel", "task_dist", "task_queue", "task_progress", "task_hold", "comm_sinr", "comm_rate", "comm_interf", "neigh_pos", "nearest")}
    contrib = {name: [] for name in groups}
    pre_abs = []
    tanh_sat = []
    reward_parts = []
    in_radius = []
    first_hit = []
    intervention = []
    correction = []
    served_steps = []
    cfg = L3BConfig()
    radius = float(cfg.service_radius_m)
    for scenario in (321000, 321001, 321002, 321003):
        worker.scenario = scenario
        worker.cursor = 0
        worker.env.reset(seed=scenario)
        worker.option = None
        worker.pending = 0
        worker.last_movement = np.zeros((6, 2), dtype=np.float64)
        hit = np.full(6, -1, dtype=np.int64)
        for step_index in range(int(cfg.episode_horizon)):
            if worker.pending <= 0:
                from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment

                worker.option = make_option(worker.env, sample_assignment(worker.env, worker.rng))
                worker.pending = 15
            association = np.asarray(worker.option.association, dtype=np.int64).copy()
            obs = __import__("wt.lowlevel_v2.observation", fromlist=["build_actor_observation"]).build_actor_observation(
                worker.env, worker.option, worker.last_movement
            )
            task = obs["task"]
            comm = obs["communication"]
            neigh = obs["neighbor_set"]
            feature_abs["self"].append(np.mean(np.abs(obs["self"][:, 0:2])))
            feature_abs["task_rel"].append(np.mean(np.abs(task[:, 1:3])))
            feature_abs["task_dist"].append(np.mean(np.abs(task[:, 3])))
            feature_abs["task_queue"].append(np.mean(np.abs(task[:, 4])))
            feature_abs["task_progress"].append(np.mean(np.abs(task[:, 5])))
            feature_abs["task_hold"].append(np.mean(np.abs(task[:, 6])))
            feature_abs["comm_sinr"].append(np.mean(np.abs(comm[:, 0])))
            feature_abs["comm_rate"].append(np.mean(np.abs(comm[:, 1])))
            feature_abs["comm_interf"].append(np.mean(np.abs(comm[:, 2])))
            feature_abs["neigh_pos"].append(np.mean(np.abs(neigh[:, :, 0:2])))
            feature_abs["nearest"].append(np.mean(np.abs(obs["d_nearest"])))
            with torch.no_grad():
                packed = actor._pack(
                    torch.as_tensor(obs["self"], dtype=torch.float32),
                    torch.as_tensor(obs["task"], dtype=torch.float32),
                    torch.as_tensor(obs["communication"], dtype=torch.float32),
                    torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                    torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
                ).cpu().numpy()
                mu, _log_std = actor(
                    torch.as_tensor(obs["self"], dtype=torch.float32),
                    torch.as_tensor(obs["task"], dtype=torch.float32),
                    torch.as_tensor(obs["communication"], dtype=torch.float32),
                    torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                    torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
                )
            mean_x = np.mean(np.abs(packed), axis=0)
            for name, (start, stop) in groups.items():
                piece = np.abs(weight[:, start:stop]) * mean_x[start:stop]
                contrib[name].append(float(np.sum(piece) / weight.shape[0]))
            pre = mu.detach().cpu().numpy()
            pre_abs.append(float(np.mean(np.abs(pre))))
            tanh_sat.append(float(np.mean(np.abs(np.tanh(pre)) > 0.9)))
            base = fly_toward(worker.env, association)
            delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
            desired = clip_unit_step(base + delta)
            projected = projector.project(worker.env, desired, worker.option)
            failure = projected.constraint_status != "ok" or not projected.finite
            executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
            executed = clip_unit_step(executed)
            info = worker.env.step(executed, association)
            served = float(np.sum(info["served_bits"]))
            reward, r_comm = low_level_reward(served, executed, desired, delta, worker.env.block_bits)
            c_proj = float(np.mean(np.sum((executed - desired) ** 2, axis=-1))) / (float(ACTION_SCALE) ** 2)
            c_res = float(np.mean(np.sum(delta ** 2, axis=-1))) / (float(ACTION_SCALE) ** 2)
            reward_parts.append((r_comm, float(LAMBDA_PROJECT) * c_proj, float(LAMBDA_RESIDUAL) * c_res, reward, served))
            distances = []
            for uav, buoy in enumerate(association):
                if int(buoy) < 0:
                    continue
                dist = float(np.linalg.norm(worker.env.uav_positions[uav] - worker.env.buoy_positions[int(buoy)]))
                distances.append(dist)
                if dist <= radius and hit[uav] < 0:
                    hit[uav] = step_index + 1
            in_radius.append(float(np.mean([item <= radius for item in distances])) if distances else 0.0)
            intervention.append(float(failure or float(projected.intervention_norm) > 1.0e-3))
            correction.append(float(projected.intervention_norm))
            served_steps.append(served)
            worker.pending -= 1
            worker.option.remaining_hold = int(worker.pending)
            worker.last_movement = executed.copy()
        first_hit.append(hit)
    parts = np.asarray(reward_parts, dtype=np.float64)
    print("feature_mean_abs")
    for name, values in feature_abs.items():
        print(f"  {name} {float(np.mean(values)):.6g}")
    print("first_layer_mean_abs_contribution")
    for name, values in contrib.items():
        print(f"  {name} {float(np.mean(values)):.6g}")
    print(f"pre_abs_mean {float(np.mean(pre_abs)):.6g}")
    print(f"tanh_sat_fraction {float(np.mean(tanh_sat)):.6g}")
    print(f"log_std {actor.log_std.detach().cpu().numpy().tolist()}")
    print(f"r_comm {parts[:, 0].mean():.6g} proj_penalty {parts[:, 1].mean():.6g} res_penalty {parts[:, 2].mean():.6g} reward {parts[:, 3].mean():.6g}")
    print(f"served_bits_mean {parts[:, 4].mean():.6g}")
    print(f"in_radius_fraction {float(np.mean(in_radius)):.6g}")
    print(f"projection_rate {float(np.mean(intervention)):.6g}")
    print(f"intervention_norm_mean {float(np.mean(correction)):.6g}")
    print(f"service_radius_m {radius}")
    print(f"first_hit_times {np.concatenate(first_hit).tolist()}")
    outside = parts[np.asarray(in_radius) < 0.5]
    inside = parts[np.asarray(in_radius) >= 0.5]
    if len(outside):
        print(f"outside_majority r_comm {outside[:, 0].mean():.6g} reward {outside[:, 3].mean():.6g} n {len(outside)}")
    if len(inside):
        print(f"inside_majority r_comm {inside[:, 0].mean():.6g} reward {inside[:, 3].mean():.6g} n {len(inside)}")


if __name__ == "__main__":
    main()
