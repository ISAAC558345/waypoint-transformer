"""Movement-only residual step on L3BBitQueueEnvV2. Assignment is an input."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import build_pools, is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    D_SAFE_M,
    HOLD,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    PAYLOAD_SCALE,
    SERVICE_SAFETY_MARGIN_M,
    TRAIN_POOL,
    assert_allowed,
)
from wt.htw_phase100.contract import TaskOption
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector


def install_light_info(env: L3BBitQueueEnvV2) -> None:
    def info() -> dict:
        return {
            "sinr": env.last_sinr.copy(),
            "rate_bps": env.last_rate.copy(),
            "time_step": int(env.t),
        }

    env.info = info


def make_option(env: L3BBitQueueEnvV2, assignment: np.ndarray) -> TaskOption:
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1).copy()
    if not is_legal(chosen, int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("sampler produced an illegal assignment")
    return TaskOption(
        targets=chosen.copy(),
        decision_time=int(env.t),
        remaining_hold=int(HOLD),
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=chosen.copy(),
    )


def sample_assignment(env: L3BBitQueueEnvV2, rng: np.random.Generator) -> np.ndarray:
    pools, _packed, illegal = build_pools(env, (("G1_RATE", 5),))
    if int(illegal) != 0:
        raise RuntimeError("candidate pool contains an illegal assignment")
    pool = pools[("G1_RATE", 5)]
    if not pool:
        raise RuntimeError("candidate pool is empty")
    return np.asarray(pool[int(rng.integers(0, len(pool)))], dtype=np.int64).copy()


def low_level_reward(served_bits: float, a_exec: np.ndarray, a_des: np.ndarray, delta_used: np.ndarray, block_bits: float) -> tuple[float, float]:
    if abs(float(block_bits) - float(PAYLOAD_SCALE)) > 1.0e-6:
        raise RuntimeError("block scale does not match the frozen payload scale")
    r_comm = float(served_bits) / float(block_bits)
    c_proj = float(np.mean(np.sum((a_exec - a_des) ** 2, axis=-1) / (float(ACTION_SCALE) ** 2 + 1.0e-8)))
    c_res = float(np.mean(np.sum(delta_used ** 2, axis=-1) / (float(ACTION_SCALE) ** 2 + 1.0e-8)))
    reward = r_comm - float(LAMBDA_PROJECT) * c_proj - float(LAMBDA_RESIDUAL) * c_res
    if not np.isfinite(reward):
        raise RuntimeError("NONFINITE")
    return float(reward), float(r_comm)


class TrainWorker:
    def __init__(self, model_seed: int, env_index: int, actor: ResidualActor, critic: CentralizedCritic) -> None:
        self.model_seed = int(model_seed)
        self.env_index = int(env_index)
        self.actor = actor
        self.critic = critic
        self.env = L3BBitQueueEnvV2()
        install_light_info(self.env)
        self.projector = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))
        self.rng = np.random.default_rng(321_000_000 + self.model_seed * 10_000 + self.env_index)
        self.cursor = int(self.model_seed + self.env_index) % len(TRAIN_POOL)
        self.option: TaskOption | None = None
        self.pending = 0
        self.last_movement = np.zeros((6, 2), dtype=np.float64)
        self.scenario = int(TRAIN_POOL[self.cursor])

    def reset(self) -> None:
        self.scenario = int(TRAIN_POOL[self.cursor % len(TRAIN_POOL)])
        self.cursor += 1
        assert_allowed(self.scenario, split="train")
        self.env.reset(seed=self.scenario)
        self.option = None
        self.pending = 0
        self.last_movement = np.zeros((self.env.config.num_uavs, 2), dtype=np.float64)

    def _act(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        self_f = torch.as_tensor(obs["self"], dtype=torch.float32)
        task_f = torch.as_tensor(obs["task"], dtype=torch.float32)
        comm_f = torch.as_tensor(obs["communication"], dtype=torch.float32)
        neigh = torch.as_tensor(obs["neighbor_set"], dtype=torch.float32)
        near = torch.as_tensor(obs["d_nearest"], dtype=torch.float32)
        mu, log_std = self.actor(self_f, task_f, comm_f, neigh, near)
        pre = Normal(mu, log_std.exp()).sample()
        log_uav, _entropy = tanh_log_prob_uav(mu, log_std, pre)
        return pre.detach().cpu().numpy().astype(np.float64), log_uav.detach().cpu().numpy().astype(np.float32)

    def step(self) -> dict:
        if self.pending <= 0:
            self.option = make_option(self.env, sample_assignment(self.env, self.rng))
            self.pending = int(HOLD)
        assert self.option is not None
        if not np.array_equal(self.option.association, self.option.targets):
            raise RuntimeError("INTERFACE")
        association = np.asarray(self.option.association, dtype=np.int64).copy()
        obs = build_actor_observation(self.env, self.option, self.last_movement)
        glob = build_critic_state(self.env, self.option, self.last_movement)
        with torch.no_grad():
            value = float(self.critic(torch.as_tensor(glob["vector"], dtype=torch.float32)).cpu())
            pre, log_uav = self._act(obs)
        base = fly_toward(self.env, association)
        delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
        desired = clip_unit_step(base + delta)
        projected = self.projector.project(self.env, desired, self.option)
        failure = projected.constraint_status != "ok" or not projected.finite
        executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
        executed = clip_unit_step(executed)
        info = self.env.step(executed, association)
        if not np.array_equal(self.option.association, association):
            raise RuntimeError("INTERFACE")
        error = float(np.max(np.abs(self.env.conservation_error_bits())))
        if error > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        served = float(np.sum(info["served_bits"]))
        reward, r_comm = low_level_reward(served, executed, desired, delta, self.env.block_bits)
        comm = communication_features(self.env)
        self.pending -= 1
        self.option.remaining_hold = int(self.pending)
        self.last_movement = executed.copy()
        done = bool(int(self.env.t) >= int(self.env.config.episode_horizon) or failure)
        record = {
            "obs": obs,
            "global_state": np.asarray(glob["vector"], dtype=np.float32),
            "pre_tanh": pre.astype(np.float32),
            "log_prob_uav": log_uav,
            "policy_active_mask": np.ones(self.env.config.num_uavs, dtype=np.float32),
            "value": value,
            "reward": reward,
            "r_comm": r_comm,
            "served_bits": served,
            "sinr": float(np.mean(info["sinr"])),
            "interference": float(np.mean(comm["total_interference"])),
            "path_step": float(np.sum(np.linalg.norm(executed, axis=-1) * self.env.config.step_length_m)),
            "residual_norm": float(np.mean(np.linalg.norm(delta, axis=-1))),
            "projected": int(failure or float(projected.intervention_norm) > 1.0e-3),
            "safety_violation": int(minimum_pairwise_distance(self.env.uav_positions) + 1.0e-9 < float(D_SAFE_M)),
            "done": done,
            "payload": float(self.env.payload_bits),
        }
        if done:
            self.reset()
        return record
