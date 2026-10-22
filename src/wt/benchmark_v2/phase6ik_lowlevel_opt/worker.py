"""Same residual step and reward as Phase-6G, with physically scaled network inputs."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import HOLD, TRAIN_POOL, assert_allowed
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker, low_level_reward, make_option, sample_assignment
from wt.lowlevel_v2.contracts import clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL

from .scale import scale_actor_observation, scale_critic_vector


class ScaledTrainWorker(TrainWorker):
    """TrainWorker whose actor and critic see divided inputs. The reward function is unchanged."""

    def _scaled(self) -> tuple[dict, np.ndarray]:
        obs = scale_actor_observation(build_actor_observation(self.env, self.option, self.last_movement))
        glob = build_critic_state(self.env, self.option, self.last_movement)
        return obs, scale_critic_vector(glob["vector"]).astype(np.float32)

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
            "residual_norm": float(np.mean(np.linalg.norm(delta, axis=-1))),
            "projected": int(failure or float(projected.intervention_norm) > 1.0e-3),
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


def assert_train_scenario(scenario_id: int) -> None:
    assert_allowed(int(scenario_id), split="train")
    if int(scenario_id) not in TRAIN_POOL:
        raise RuntimeError(f"phase-6ik train refuses scenario {scenario_id}")


__all__ = ["ScaledTrainWorker", "assert_train_scenario"]
