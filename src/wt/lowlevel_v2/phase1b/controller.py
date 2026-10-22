"""Phase-1B residual worker: ServiceResidualLock then SafetyProjector. High-level frozen."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Normal

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.base_executor import BaseExecutor
from wt.lowlevel_v2.contracts import CollisionConfig, ResidualConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.residual_action import residual_desired
from wt.lowlevel_v2.residual_policy import ResidualActor, CentralizedCritic
from wt.lowlevel_v2.safety_projector import SafetyProjector
from wt.round30.ppo import tanh_log_prob

from .mappo import tanh_log_prob_uav
from .protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    D_SAFE_M,
    HOLD_STEPS,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    PAYLOAD_SCALE,
    SERVICE_SAFETY_MARGIN_M,
    refuse_untouched,
)
from .service_lock import apply_service_residual_lock, service_lock_mask


@dataclass
class StepRecord:
    obs: dict
    global_state: np.ndarray
    pre_tanh: np.ndarray
    log_prob: float
    log_prob_uav: np.ndarray
    value: float
    reward: float
    done: bool
    a_base: np.ndarray
    delta_tanh: np.ndarray
    delta_used: np.ndarray
    a_des: np.ndarray
    a_projected: np.ndarray
    a_exec: np.ndarray
    projector_failure: bool
    preexisting_safety_violation: bool
    intervention_norm: float
    min_sep: float
    payload_inc: float
    service_lock: np.ndarray
    policy_active_mask: np.ndarray
    lock_violation: float
    r_raw: np.ndarray = field(default_factory=lambda: np.zeros((6, 2), dtype=np.float64))
    cap_active: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))
    base_zero_lock: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))
    base_norm: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))


class ResidualWorker:
    def __init__(
        self,
        high,
        actor: ResidualActor,
        critic: CentralizedCritic,
        *,
        service_lock: bool = True,
        trust_region: bool = False,
    ) -> None:
        self.high = high
        self.actor = actor
        self.critic = critic
        self.service_lock = bool(service_lock)
        self.trust_region = bool(trust_region)
        self.env = L3BEnv()
        self.base = BaseExecutor(gated=True)
        self.projector = SafetyProjector(
            CollisionConfig(0.4, 0.2, placeholder=False),
            safety_margin=SERVICE_SAFETY_MARGIN_M,
        )
        self.residual_cfg = ResidualConfig(alpha_residual=ALPHA_RESIDUAL, placeholder=False)
        self.pending = 0
        self.option = None
        self.last_arrival = np.zeros(12, dtype=np.float32)
        self.last_positions = np.zeros((6, 2), dtype=np.float32)
        self.last_movement = np.zeros((6, 2), dtype=np.float64)
        self.seed = 0

    def reset(self, seed: int) -> dict:
        refuse_untouched(int(seed))
        self.seed = int(seed)
        from wt.htw_phase100.high_level import decide, make_option

        info = self.env.reset(seed=int(seed))
        self.pending = 0
        self.option = None
        self.last_arrival = np.zeros(self.env.config.num_buoys, dtype=np.float32)
        self.last_positions = np.asarray(self.env.uav_positions, dtype=np.float32).copy()
        self.last_movement = np.zeros((self.env.config.num_uavs, 2), dtype=np.float64)
        _ = decide, make_option
        return info

    def _refresh_option(self, info: dict) -> None:
        from wt.htw_phase100.high_level import decide, make_option

        if self.pending <= 0:
            assignment = decide(self.high, self.env, info, self.last_arrival, self.last_positions, self.pending)
            self.pending = int(HOLD_STEPS)
            self.option = make_option(self.env, info, assignment, self.pending)
        assert self.option is not None
        self.option.remaining_hold = int(self.pending)

    def act_tensors(self, obs: dict, *, deterministic: bool):
        self_f = torch.as_tensor(obs["self"], dtype=torch.float32)
        task_f = torch.as_tensor(obs["task"], dtype=torch.float32)
        comm_f = torch.as_tensor(obs["communication"], dtype=torch.float32)
        neigh = torch.as_tensor(obs["neighbor_set"], dtype=torch.float32)
        near = torch.as_tensor(obs["d_nearest"], dtype=torch.float32)
        mu, log_std = self.actor(self_f, task_f, comm_f, neigh, near)
        if deterministic:
            pre = mu
        else:
            pre = Normal(mu, log_std.exp()).sample()
        log_uav, _ = tanh_log_prob_uav(mu, log_std, pre)
        log_joint, _ = tanh_log_prob(mu, log_std, pre)
        return pre, log_joint, log_uav

    def step(self, info: dict, *, deterministic: bool, zero_residual: bool = False) -> tuple[dict, StepRecord]:
        self._refresh_option(info)
        min_now = minimum_pairwise_distance(self.env.uav_positions)
        preexisting = bool(min_now + 1.0e-9 < D_SAFE_M)
        obs = build_actor_observation(self.env, self.option, self.last_movement)
        glob = build_critic_state(self.env, self.option, self.last_movement)
        a_base, _ = self.base.act(self.env, self.option)
        lock = service_lock_mask(self.env, self.option) if self.service_lock else np.zeros(6, dtype=bool)
        if preexisting:
            zeros = np.zeros((6, 2), dtype=np.float64)
            record = StepRecord(
                obs=obs,
                global_state=glob["vector"],
                pre_tanh=np.zeros((6, 2), dtype=np.float32),
                log_prob=0.0,
                log_prob_uav=np.zeros(6, dtype=np.float32),
                value=0.0,
                reward=0.0,
                done=True,
                a_base=a_base,
                delta_tanh=zeros,
                delta_used=zeros,
                a_des=a_base,
                a_projected=a_base,
                a_exec=np.zeros((6, 2)),
                projector_failure=False,
                preexisting_safety_violation=True,
                intervention_norm=0.0,
                min_sep=min_now,
                payload_inc=0.0,
                service_lock=lock,
                policy_active_mask=(~lock).astype(np.float32),
                lock_violation=0.0,
            )
            return info, record
        with torch.no_grad():
            value = float(self.critic(torch.as_tensor(glob["vector"], dtype=torch.float32)).cpu())
            if zero_residual:
                pre = torch.zeros(6, 2)
                log_joint = torch.zeros(())
                log_uav = torch.zeros(6)
            else:
                pre, log_joint, log_uav = self.act_tensors(obs, deterministic=deterministic)
        pre_np = pre.detach().cpu().numpy().astype(np.float64)
        a_des, delta_used, tr = self._compose_desired(a_base, pre_np, lock)
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
        r_payload = payload_inc / float(PAYLOAD_SCALE)
        delta_tanh = np.tanh(pre_np)
        c_proj = float(np.mean(np.sum((a_exec - a_des) ** 2, axis=-1) / (ACTION_SCALE ** 2 + 1.0e-8)))
        c_res = float(np.mean(np.sum(delta_used ** 2, axis=-1) / (ACTION_SCALE ** 2 + 1.0e-8)))
        reward = r_payload - LAMBDA_PROJECT * c_proj - LAMBDA_RESIDUAL * c_res
        if not np.isfinite(reward):
            raise RuntimeError("non-finite train reward")
        self.last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        self.last_positions = np.asarray(self.env.uav_positions, dtype=np.float32).copy()
        self.last_movement = a_exec.copy()
        done = bool(after.get("terminated", False) or failure)
        violation = float(np.max(np.linalg.norm(delta_used[lock], axis=-1))) if np.any(lock) else 0.0
        log_uav_np = log_uav.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(log_uav) else np.asarray(log_uav, dtype=np.float32)
        mask = self._policy_active_mask(lock, tr["base_zero_lock"])
        record = StepRecord(
            obs=obs,
            global_state=glob["vector"],
            pre_tanh=pre_np.astype(np.float32),
            log_prob=float(log_joint.detach().cpu()) if torch.is_tensor(log_joint) else float(log_joint),
            log_prob_uav=log_uav_np,
            value=value,
            reward=float(reward),
            done=done,
            a_base=np.asarray(a_base, dtype=np.float64),
            delta_tanh=delta_tanh,
            delta_used=delta_used,
            a_des=a_des,
            a_projected=a_projected,
            a_exec=a_exec,
            projector_failure=bool(failure),
            preexisting_safety_violation=False,
            intervention_norm=float(proj.intervention_norm),
            min_sep=minimum_pairwise_distance(self.env.uav_positions),
            payload_inc=payload_inc,
            service_lock=lock,
            policy_active_mask=mask,
            lock_violation=violation,
            r_raw=tr["r_raw"],
            cap_active=tr["cap_active"],
            base_zero_lock=tr["base_zero_lock"],
            base_norm=tr["base_norm"],
        )
        return after, record

    def _compose_desired(self, a_base: np.ndarray, pre_np: np.ndarray, lock: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        r_raw = float(ALPHA_RESIDUAL) * np.tanh(np.asarray(pre_np, dtype=np.float64))
        if self.service_lock:
            a_des, delta_locked = apply_service_residual_lock(a_base, pre_np, lock, ALPHA_RESIDUAL)
            r_after_lock = np.asarray(delta_locked, dtype=np.float64)
        else:
            r_after_lock = r_raw.copy()
            a_des = clip_unit_step(np.asarray(a_base, dtype=np.float64) + r_after_lock)
        base_norm = np.linalg.norm(np.asarray(a_base, dtype=np.float64), axis=-1)
        extra = {
            "r_raw": r_raw,
            "cap_active": np.zeros(base_norm.shape[0], dtype=bool),
            "base_zero_lock": np.zeros(base_norm.shape[0], dtype=bool),
            "base_norm": base_norm,
        }
        if not self.trust_region:
            return a_des, r_after_lock, extra
        from wt.lowlevel_v2.phase2b.trust_region import apply_base_relative_residual_trust_region

        r_used, extra = apply_base_relative_residual_trust_region(a_base, r_after_lock)
        a_des = clip_unit_step(np.asarray(a_base, dtype=np.float64) + r_used)
        extra["r_raw"] = r_raw
        return a_des, r_used, extra

    @staticmethod
    def _policy_active_mask(lock: np.ndarray, base_zero_lock: np.ndarray) -> np.ndarray:
        return (~(np.asarray(lock, dtype=bool) | np.asarray(base_zero_lock, dtype=bool))).astype(np.float32)


__all__ = ["ResidualWorker", "StepRecord", "load_frozen_high", "make_networks"]
