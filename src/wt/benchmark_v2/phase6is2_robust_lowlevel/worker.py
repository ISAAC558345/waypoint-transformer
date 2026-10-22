"""Continuous episodes. A new high-level task is not an environment termination."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL, HOLD, SERVICE_SAFETY_MARGIN_M, TRAIN_POOL
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, low_level_reward
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import FORBIDDEN_KEYS, build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.safety_projector import SafetyProjector
from torch.distributions import Normal

from .protocol import TRAIN_SOURCES, assert_split_scenario


class RobustTrainWorker:
    def __init__(self, model_seed: int, env_index: int, actor, critic, selector, num_envs: int) -> None:
        self.model_seed = int(model_seed)
        self.env_index = int(env_index)
        self.actor = actor
        self.critic = critic
        self.selector = selector
        self.num_envs = int(num_envs)
        self.env = L3BBitQueueEnvV2()
        install_light_info(self.env)
        self.projector = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))
        self.rng = np.random.default_rng(421_000_000 + self.model_seed * 10_000 + self.env_index)
        self.episode_index = int(env_index)
        self.option = None
        self.pending = 0
        self.last_movement = np.zeros((6, 2), dtype=np.float64)
        self.position_prev = np.zeros((6, 2), dtype=np.float64)
        self.predecessor = None
        self.scenario = int(TRAIN_POOL[0])
        self.source = TRAIN_SOURCES[0]
        self.window_served = 0.0
        self.open_record = None
        self.decisions: list[dict] = []

    def reset(self) -> None:
        index = int(self.episode_index)
        self.episode_index += int(self.num_envs)
        self.scenario = int(TRAIN_POOL[(index // len(TRAIN_SOURCES)) % len(TRAIN_POOL)])
        self.source = TRAIN_SOURCES[index % len(TRAIN_SOURCES)]
        assert_split_scenario(self.scenario, "train")
        if self.source not in TRAIN_SOURCES:
            raise RuntimeError("held-out task source entered training")
        self.env.reset(seed=self.scenario)
        self.option = None
        self.pending = 0
        self.window_served = 0.0
        self.open_record = None
        self.last_movement = np.zeros((self.env.config.num_uavs, 2), dtype=np.float64)
        self.position_prev = np.asarray(self.env.uav_positions, dtype=np.float64).copy()
        self.predecessor = None

    def _close_window(self) -> None:
        if self.open_record is None:
            return
        self.open_record["realized_served_bits"] = float(self.window_served)
        self.decisions.append(self.open_record)
        self.open_record = None
        self.window_served = 0.0

    def _act(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        mu, log_std = self.actor(
            torch.as_tensor(obs["self"], dtype=torch.float32),
            torch.as_tensor(obs["task"], dtype=torch.float32),
            torch.as_tensor(obs["communication"], dtype=torch.float32),
            torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
        )
        pre = Normal(mu, log_std.exp()).sample()
        log_uav, _entropy = tanh_log_prob_uav(mu, log_std, pre)
        return pre.detach().cpu().numpy().astype(np.float64), log_uav.detach().cpu().numpy().astype(np.float32)

    def _ensure_task(self) -> bool:
        if self.pending > 0:
            return False
        self._close_window()
        chosen = self.selector.choose(self.env, self.source, self.rng, self.position_prev, self.predecessor)
        self.option = chosen["option"]
        if not np.array_equal(self.option.association, self.option.targets):
            raise RuntimeError("INTERFACE")
        self.pending = int(HOLD)
        distance, queues = [], []
        for uav, buoy in enumerate(self.option.association):
            if int(buoy) < 0:
                continue
            distance.append(float(np.linalg.norm(self.env.uav_positions[uav] - self.env.buoy_positions[int(buoy)])))
            queues.append(float(self.env.q_bits[int(buoy)]))
        self.open_record = {
            "scenario_id": int(self.scenario),
            "scenario_seed": int(self.scenario),
            "model_seed": int(self.model_seed),
            "task_source": str(self.source),
            "time": int(self.env.t),
            "assignment": [int(item) for item in self.option.association],
            "targets": [int(item) for item in self.option.targets],
            "remaining_hold": int(self.option.remaining_hold),
            "config_sha256": str(self.selector.config_sha256),
            "n_active": int(len(distance)),
            "distance_m": float(np.mean(distance)) if distance else 0.0,
            "queue_bits": float(np.mean(queues)) if queues else 0.0,
            "realized_served_bits": None,
        }
        return True

    def step(self) -> dict:
        task_boundary = self._ensure_task()
        assert self.option is not None
        association = np.asarray(self.option.association, dtype=np.int64).copy()
        obs = build_actor_observation(self.env, self.option, self.last_movement)
        if any(key in obs for key in FORBIDDEN_KEYS) or "realized_served_bits" in obs or "payload" in obs:
            raise RuntimeError("future information reached the actor observation")
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
        self.position_prev = np.asarray(self.env.uav_positions, dtype=np.float64).copy()
        self.predecessor = int(self.env.t)
        payload_before = float(self.env.payload_bits)
        info = self.env.step(executed, association)
        if float(np.max(np.abs(self.env.conservation_error_bits()))) > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        served = float(np.sum(info["served_bits"]))
        if abs(float(self.env.payload_bits) - payload_before - served) > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        reward, r_comm = low_level_reward(served, executed, desired, delta, self.env.block_bits)
        if abs(float(r_comm) * float(self.env.block_bits) - served) > 1.0e-3:
            raise RuntimeError("reward used a value other than this step's served bits")
        self.window_served += served
        self.pending -= 1
        self.option.remaining_hold -= 1
        self.last_movement = executed.copy()
        step_time = int(self.env.t)
        comm = communication_features(self.env)
        safety = int(minimum_pairwise_distance(self.env.uav_positions) + 1.0e-9 < float(D_SAFE_M))
        episode_done = bool(failure or step_time >= int(self.env.config.episode_horizon))
        if episode_done:
            self._close_window()
            self.reset()
        elif self.pending <= 0:
            self._ensure_task()
        record = {
            "obs": obs,
            "global_state": np.asarray(glob["vector"], dtype=np.float32),
            "pre_tanh": pre.astype(np.float32),
            "log_prob_uav": log_uav,
            "policy_active_mask": np.ones(self.env.config.num_uavs, dtype=np.float32),
            "value": value,
            "reward": float(reward),
            "r_comm": float(r_comm),
            "served_bits": served,
            "sinr": float(np.mean(info["sinr"])),
            "interference": float(np.mean(comm["total_interference"])),
            "projected": int(failure or float(projected.intervention_norm) > 1.0e-3),
            "safety_violation": safety,
            "done": episode_done,
            "task_boundary": bool(task_boundary),
            "time": step_time,
        }
        return record

    def bootstrap_value(self) -> float:
        if self.option is None:
            return 0.0
        glob = build_critic_state(self.env, self.option, self.last_movement)
        with torch.no_grad():
            return float(self.critic(torch.as_tensor(glob["vector"], dtype=torch.float32)).cpu())


__all__ = ["RobustTrainWorker"]
