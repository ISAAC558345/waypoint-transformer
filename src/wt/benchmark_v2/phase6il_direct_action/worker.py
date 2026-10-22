"""Direct velocity worker. Reward coefficients stay on the Phase-6G function."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import HOLD
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker, low_level_reward, make_option, sample_assignment
from wt.benchmark_v2.phase6ik_lowlevel_opt.scale import scale_actor_observation, scale_critic_vector
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .action import direct_command, direct_reward_terms


def make_projector() -> SafetyProjector:
    return SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))


def execute_command(env, option, commanded: np.ndarray) -> tuple[np.ndarray, dict]:
    projector = make_projector()
    projected = projector.project(env, commanded, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(commanded) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    executed = clip_unit_step(executed)
    correction = float(np.linalg.norm(executed - commanded))
    return executed, {
        "failure": bool(failure),
        "intervened": bool(failure or float(projected.intervention_norm) > 1.0e-3),
        "intervention_norm": float(projected.intervention_norm),
        "correction_norm": correction,
        "status": projected.constraint_status,
    }


class DirectTrainWorker(TrainWorker):
    """Samples a direct velocity. Assignment still comes only from the TaskOption."""

    def __init__(self, model_seed: int, env_index: int, actor, critic) -> None:
        super().__init__(model_seed, env_index, actor, critic)
        self.projector = make_projector()

    def _scaled(self) -> tuple[dict, np.ndarray]:
        obs = scale_actor_observation(build_actor_observation(self.env, self.option, self.last_movement))
        vector = scale_critic_vector(build_critic_state(self.env, self.option, self.last_movement)["vector"])
        return obs, vector.astype(np.float32)

    def step(self) -> dict:
        if self.pending <= 0:
            self.option = make_option(self.env, sample_assignment(self.env, self.rng))
            self.pending = int(HOLD)
        assert self.option is not None
        if not np.array_equal(self.option.association, self.option.targets):
            raise RuntimeError("INTERFACE")
        association = np.asarray(self.option.association, dtype=np.int64).copy()
        obs, vector = self._scaled()
        with torch.no_grad():
            value = float(self.critic(torch.as_tensor(vector, dtype=torch.float32)).cpu())
            pre, log_uav = self._act(obs)
        commanded = direct_command(pre)
        executed, safety = execute_command(self.env, self.option, commanded)
        info = self.env.step(executed, association)
        if not np.array_equal(np.asarray(self.option.association, dtype=np.int64), association):
            raise RuntimeError("INTERFACE")
        error = float(np.max(np.abs(self.env.conservation_error_bits())))
        if error > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        served = float(np.sum(info["served_bits"]))
        done_action, desired, delta = direct_reward_terms(commanded, executed)
        reward, r_comm = low_level_reward(served, done_action, desired, delta, self.env.block_bits)
        comm = communication_features(self.env)
        self.pending -= 1
        self.option.remaining_hold = int(self.pending)
        self.last_movement = executed.copy()
        done = bool(int(self.env.t) >= int(self.env.config.episode_horizon) or safety["failure"])
        record = {
            "obs": obs,
            "global_state": vector,
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
            "residual_norm": float(np.mean(np.linalg.norm(commanded, axis=-1))),
            "projected": int(safety["intervened"]),
            "correction_norm": float(safety["correction_norm"]),
            "safety_violation": int(minimum_pairwise_distance(self.env.uav_positions) + 1.0e-9 < float(D_SAFE_M)),
            "done": done,
            "payload": float(self.env.payload_bits),
        }
        if done:
            self.reset()
        return record

    def bootstrap_value(self) -> float:
        if self.option is None:
            return 0.0
        _obs, vector = self._scaled()
        with torch.no_grad():
            return float(self.critic(torch.as_tensor(vector, dtype=torch.float32)).cpu())


__all__ = ["DirectTrainWorker", "execute_command", "make_projector"]
