"""Train on recorded scheme-C windows. Each step uses that step's served bits."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import TrainWorker, low_level_reward, make_option
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M

from .dataset import _restore


class SchemeCTrainWorker(TrainWorker):
    def bind_records(self, records: list[dict], stride: int) -> None:
        self.records = records
        self.stride = int(stride)
        self.cursor = int(self.env_index)
        self.steps_left = 0

    def reset(self) -> None:
        self._load_window()

    def _load_window(self) -> None:
        row = self.records[int(self.cursor) % len(self.records)]
        self.cursor += int(self.stride)
        self.env.reset(seed=int(row["scenario_id"]))
        if not np.allclose(self.env.buoy_positions, row["buoy_positions"]):
            raise RuntimeError("INTERFACE")
        _restore(self.env, row["snap"], row["initial_q_bits"])
        self.option = make_option(self.env, row["assignment"])
        if not np.array_equal(self.option.association, self.option.targets):
            raise RuntimeError("INTERFACE")
        if not np.array_equal(self.option.association, row["assignment"]):
            raise RuntimeError("INTERFACE")
        self.last_movement = np.asarray(row["last_movement"], dtype=np.float64).copy()
        self.steps_left = int(row["window_steps"])
        self.window_id = (int(row["scenario_id"]), int(row["time"]))

    def step(self) -> dict:
        if self.steps_left <= 0:
            self._load_window()
        assert self.option is not None
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
        payload_before = float(self.env.payload_bits)
        info = self.env.step(executed, association)
        if not np.array_equal(self.option.association, association):
            raise RuntimeError("INTERFACE")
        if float(np.max(np.abs(self.env.conservation_error_bits()))) > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        served = float(np.sum(info["served_bits"]))
        if abs(float(self.env.payload_bits) - payload_before - served) > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        reward, r_comm = low_level_reward(served, executed, desired, delta, self.env.block_bits)
        if abs(float(r_comm) * float(self.env.block_bits) - served) > 1.0e-3:
            raise RuntimeError("reward used a value other than this step's served bits")
        comm = communication_features(self.env)
        self.steps_left -= 1
        self.option.remaining_hold -= 1
        self.last_movement = executed.copy()
        done = bool(self.steps_left <= 0 or failure or int(self.env.t) >= int(self.env.config.episode_horizon))
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
            "safety_violation": int(minimum_pairwise_distance(self.env.uav_positions) + 1.0e-9 < float(D_SAFE_M)),
            "done": done,
            "window": self.window_id,
        }
        if done:
            self._load_window()
        return record

    def bootstrap_value(self) -> float:
        if self.option is None or self.steps_left <= 0:
            return 0.0
        glob = build_critic_state(self.env, self.option, self.last_movement)
        with torch.no_grad():
            return float(self.critic(torch.as_tensor(glob["vector"], dtype=torch.float32)).cpu())


__all__ = ["SchemeCTrainWorker"]
