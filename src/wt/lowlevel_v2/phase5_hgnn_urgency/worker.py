"""Phase-5 residual worker. Same reward, projector, and hold. Planner is injected."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.base_executor import BaseExecutor
from wt.lowlevel_v2.contracts import CollisionConfig, ResidualConfig, clip_unit_step, env_target, minimum_pairwise_distance, refuse_rematch
from wt.lowlevel_v2.critic_state import build_critic_state, expected_critic_dim
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.phase1b.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    D_SAFE_M,
    LOG_STD_INIT,
    SERVICE_SAFETY_MARGIN_M,
)
from wt.lowlevel_v2.phase1b.service_lock import apply_service_residual_lock, service_lock_mask
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector
from wt.round30.ppo import tanh_log_prob

from .actor import UrgencyResidualActor
from .planner import Phase5Planner, make_phase5_option
from .protocol import HOLD_STEPS, assert_phase5_seed
from .reward import unchanged_step_reward


def assigned_urgency(env: L3BEnv, option: TaskOption) -> np.ndarray:
    m_count = int(env.config.num_uavs)
    column = np.zeros((m_count, 1), dtype=np.float64)
    if option.urgency is None:
        return column
    shared = np.asarray(option.urgency, dtype=np.float64)
    for uav in range(m_count):
        target = env_target(int(option.association[uav]), int(env.config.num_buoys))
        if target >= 0:
            column[uav, 0] = float(shared[target])
    return column


def critic_vector(env: L3BEnv, option: TaskOption, last_movement: np.ndarray, *, use_urgency: bool) -> np.ndarray:
    vector = np.asarray(build_critic_state(env, option, last_movement)["vector"], dtype=np.float64)
    if not use_urgency:
        return vector
    shared = np.zeros(int(env.config.num_buoys), dtype=np.float64) if option.urgency is None else np.asarray(option.urgency, dtype=np.float64)
    return np.concatenate((vector, shared), axis=0)


def critic_dim(*, use_urgency: bool) -> int:
    base = expected_critic_dim(6, 12)
    return int(base + (12 if use_urgency else 0))


class Phase5Worker:
    def __init__(self, planner: Phase5Planner, actor, critic: CentralizedCritic, *, use_urgency: bool, seeds: tuple[int, ...]) -> None:
        self.planner = planner
        self.actor = actor
        self.critic = critic
        self.use_urgency = bool(use_urgency)
        self.seeds = tuple(int(seed) for seed in seeds)
        for seed in self.seeds:
            assert_phase5_seed(seed)
        self.env = L3BEnv()
        self.base = BaseExecutor(gated=True)
        self.projector = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=SERVICE_SAFETY_MARGIN_M)
        self.residual_cfg = ResidualConfig(alpha_residual=ALPHA_RESIDUAL, placeholder=False)
        self.pending = 0
        self.option: TaskOption | None = None
        self.last_arrival = np.zeros(12, dtype=np.float32)
        self.last_positions = np.zeros((6, 2), dtype=np.float32)
        self.last_movement = np.zeros((6, 2), dtype=np.float64)
        self.cursor = 0
        self.seed = int(self.seeds[0])

    def reset(self, seed: int) -> dict:
        assert_phase5_seed(int(seed))
        self.seed = int(seed)
        info = self.env.reset(seed=int(seed))
        self.pending = 0
        self.option = None
        self.last_arrival = np.zeros(self.env.config.num_buoys, dtype=np.float32)
        self.last_positions = np.asarray(self.env.uav_positions, dtype=np.float32).copy()
        self.last_movement = np.zeros((self.env.config.num_uavs, 2), dtype=np.float64)
        return info

    def _next_seed(self) -> int:
        self.cursor += 1
        return int(self.seeds[self.cursor % len(self.seeds)])

    def _refresh_option(self, info: dict) -> None:
        if self.pending <= 0:
            assignment, urgency = self.planner.decide(self.env, info, self.last_arrival, self.last_positions)
            self.pending = int(HOLD_STEPS)
            self.option = make_phase5_option(
                self.env, info, assignment, self.pending, urgency, attach_urgency=self.use_urgency
            )
        assert self.option is not None
        self.option.remaining_hold = int(self.pending)
        refuse_rematch(self.option)

    def _act(self, obs: dict, urgency: np.ndarray, *, deterministic: bool):
        self_f = torch.as_tensor(obs["self"], dtype=torch.float32)
        task_f = torch.as_tensor(obs["task"], dtype=torch.float32)
        comm_f = torch.as_tensor(obs["communication"], dtype=torch.float32)
        neigh = torch.as_tensor(obs["neighbor_set"], dtype=torch.float32)
        near = torch.as_tensor(obs["d_nearest"], dtype=torch.float32)
        if isinstance(self.actor, UrgencyResidualActor):
            urg = torch.as_tensor(urgency, dtype=torch.float32)
            mu, log_std = self.actor(self_f, task_f, comm_f, neigh, near, urg)
        elif isinstance(self.actor, ResidualActor):
            mu, log_std = self.actor(self_f, task_f, comm_f, neigh, near)
        else:
            raise TypeError(type(self.actor).__name__)
        pre = mu if deterministic else Normal(mu, log_std.exp()).sample()
        log_uav, _ = tanh_log_prob_uav(mu, log_std, pre)
        log_joint, _ = tanh_log_prob(mu, log_std, pre)
        return pre, log_joint, log_uav

    def step(self, info: dict, *, deterministic: bool) -> tuple[dict, dict]:
        self._refresh_option(info)
        assert self.option is not None
        frozen_urgency = None if self.option.urgency is None else np.asarray(self.option.urgency, dtype=np.float64).copy()
        min_now = minimum_pairwise_distance(self.env.uav_positions)
        preexisting = bool(min_now + 1.0e-9 < D_SAFE_M)
        obs = build_actor_observation(self.env, self.option, self.last_movement)
        urgency = assigned_urgency(self.env, self.option)
        glob = critic_vector(self.env, self.option, self.last_movement, use_urgency=self.use_urgency)
        a_base, _ = self.base.act(self.env, self.option)
        lock = service_lock_mask(self.env, self.option)
        if preexisting:
            zeros = np.zeros((6, 2), dtype=np.float64)
            record = _record(obs, urgency, glob, np.zeros((6, 2), dtype=np.float32), 0.0, np.zeros(6, dtype=np.float32), 0.0, 0.0, True, a_base, zeros, zeros, a_base, a_base, np.zeros((6, 2)), False, True, 0.0, min_now, 0.0, lock)
            return info, record
        with torch.no_grad():
            value = float(self.critic(torch.as_tensor(glob, dtype=torch.float32)).cpu())
            pre, log_joint, log_uav = self._act(obs, urgency, deterministic=deterministic)
        pre_np = pre.detach().cpu().numpy().astype(np.float64)
        a_des, delta_used = _compose(a_base, pre_np, lock)
        proj = self.projector.project(self.env, a_des, self.option)
        failure = proj.constraint_status != "ok"
        if not failure:
            a_exec = np.asarray(proj.projected_actions, dtype=np.float64)
            a_projected = a_exec.copy()
        else:
            a_projected = np.asarray(proj.projected_actions, dtype=np.float64)
            a_exec = np.zeros((6, 2), dtype=np.float64)
        a_exec = clip_unit_step(a_exec)
        payload_before = float(self.env.payload_bits)
        after = self.env.step(a_exec, self.option.association)
        self.pending -= 1
        payload_inc = float(self.env.payload_bits - payload_before)
        reward = unchanged_step_reward(payload_inc, a_exec, a_des, delta_used)
        if frozen_urgency is not None:
            if self.option.urgency is None or not np.allclose(self.option.urgency, frozen_urgency):
                raise RuntimeError("low-level recomputed urgency inside a hold")
        self.last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        self.last_positions = np.asarray(self.env.uav_positions, dtype=np.float32).copy()
        self.last_movement = a_exec.copy()
        done = bool(after.get("terminated", False) or failure)
        log_uav_np = log_uav.detach().cpu().numpy().astype(np.float32)
        mask = (~lock).astype(np.float32)
        record = _record(
            obs,
            urgency,
            glob,
            pre_np.astype(np.float32),
            float(log_joint.detach().cpu()),
            log_uav_np,
            value,
            reward,
            done,
            a_base,
            np.tanh(pre_np),
            delta_used,
            a_des,
            a_projected,
            a_exec,
            bool(failure),
            False,
            float(proj.intervention_norm),
            minimum_pairwise_distance(self.env.uav_positions),
            payload_inc,
            lock,
            mask,
        )
        return after, record


def _compose(a_base: np.ndarray, pre_np: np.ndarray, lock: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_des, delta_locked = apply_service_residual_lock(a_base, pre_np, lock, ALPHA_RESIDUAL)
    return a_des, np.asarray(delta_locked, dtype=np.float64)


def _record(obs, urgency, glob, pre, log_joint, log_uav, value, reward, done, a_base, delta_tanh, delta_used, a_des, a_projected, a_exec, failure, preexisting, intervention, min_sep, payload_inc, lock, mask=None) -> dict:
    if mask is None:
        mask = (~np.asarray(lock, dtype=bool)).astype(np.float32)
    return {
        "obs": obs,
        "urgency": np.asarray(urgency, dtype=np.float32),
        "global_state": np.asarray(glob, dtype=np.float32),
        "pre_tanh": np.asarray(pre, dtype=np.float32),
        "log_prob": float(log_joint),
        "log_prob_uav": np.asarray(log_uav, dtype=np.float32),
        "value": float(value),
        "reward": float(reward),
        "done": bool(done),
        "a_base": np.asarray(a_base, dtype=np.float64),
        "delta_tanh": np.asarray(delta_tanh, dtype=np.float64),
        "delta_used": np.asarray(delta_used, dtype=np.float64),
        "a_des": np.asarray(a_des, dtype=np.float64),
        "a_projected": np.asarray(a_projected, dtype=np.float64),
        "a_exec": np.asarray(a_exec, dtype=np.float64),
        "projector_failure": bool(failure),
        "preexisting_safety_violation": bool(preexisting),
        "intervention_norm": float(intervention),
        "min_sep": float(min_sep),
        "payload_inc": float(payload_inc),
        "service_lock": np.asarray(lock, dtype=bool),
        "policy_active_mask": np.asarray(mask, dtype=np.float32),
    }


def make_critic(use_urgency: bool) -> CentralizedCritic:
    return CentralizedCritic(critic_dim(use_urgency=use_urgency))


def make_actor(use_urgency: bool):
    if use_urgency:
        return UrgencyResidualActor(log_std_init=float(LOG_STD_INIT))
    from .actor import original_actor

    return original_actor(float(LOG_STD_INIT))


__all__ = [
    "ACTION_SCALE",
    "Phase5Worker",
    "assigned_urgency",
    "critic_dim",
    "critic_vector",
    "make_actor",
    "make_critic",
]
