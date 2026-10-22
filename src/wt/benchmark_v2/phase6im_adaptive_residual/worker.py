"""Adaptive residual step. The reward function and safety bounds stay on Phase-6G."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import HOLD
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker, low_level_reward, make_option, sample_assignment
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .action import adaptive_command, projection_report


def make_projector() -> SafetyProjector:
    return SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))


def apply_command(env, option, commanded: np.ndarray) -> tuple[np.ndarray, dict]:
    projected = make_projector().project(env, commanded, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(commanded) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    executed = clip_unit_step(executed)
    report = projection_report(env, option, commanded, executed, d_safe=float(D_SAFE_M))
    report["failure"] = bool(failure)
    report["intervened"] = bool(failure or float(projected.intervention_norm) > 1.0e-3)
    report["intervention_norm"] = float(projected.intervention_norm)
    report["correction_norm"] = float(np.linalg.norm(executed - commanded))
    return executed, report


class AdaptiveTrainWorker(TrainWorker):
    def __init__(self, model_seed: int, env_index: int, actor, critic, calibration: dict) -> None:
        super().__init__(model_seed, env_index, actor, critic)
        self.calibration = dict(calibration)
        self.projector = make_projector()

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
        commanded = adaptive_command(base, pre, **_map_args(self.calibration))
        executed, safety = apply_command(self.env, self.option, commanded)
        info = self.env.step(executed, association)
        if not np.array_equal(np.asarray(self.option.association, dtype=np.int64), association):
            raise RuntimeError("INTERFACE")
        error = float(np.max(np.abs(self.env.conservation_error_bits())))
        if error > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        served = float(np.sum(info["served_bits"]))
        delta = commanded - base
        reward, r_comm = low_level_reward(served, executed, commanded, delta, self.env.block_bits)
        comm = communication_features(self.env)
        self.pending -= 1
        self.option.remaining_hold = int(self.pending)
        self.last_movement = executed.copy()
        done = bool(int(self.env.t) >= int(self.env.config.episode_horizon) or safety["failure"])
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
            "projected": int(safety["intervened"]),
            "correction_norm": float(safety["correction_norm"]),
            "command_turn_deg": float(safety["command_turn_deg"]),
            "executed_turn_deg": float(safety["executed_turn_deg"]),
            "excessive_heading": int(safety["excessive_heading"]),
            "excessive_speed": int(safety["excessive_speed"]),
            "collision_risk": int(safety["collision_risk"]),
            "envelope_risk": int(safety["envelope_risk"]),
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
        glob = build_critic_state(self.env, self.option, self.last_movement)
        with torch.no_grad():
            return float(self.critic(torch.as_tensor(glob["vector"], dtype=torch.float32)).cpu())


def _map_args(calibration: dict) -> dict:
    return {
        "steer_center": float(calibration["steer_center"]),
        "steer_temperature": float(calibration["steer_temperature"]),
        "speed_bias": float(calibration["speed_bias"]),
        "max_turn_deg": float(calibration["max_turn_deg"]),
    }


__all__ = ["AdaptiveTrainWorker", "apply_command", "make_projector"]
