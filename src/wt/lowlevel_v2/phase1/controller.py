"""Frozen high-level + residual low-level step. High-level never trains."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Normal

from wt.htw_phase100.high_level import decide, load_deepsets_util, make_option
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.base_executor import BaseExecutor
from wt.lowlevel_v2.contracts import CollisionConfig, ResidualConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.residual_action import residual_desired
from wt.lowlevel_v2.residual_policy import ResidualActor, CentralizedCritic
from wt.lowlevel_v2.safety_projector import SafetyProjector
from wt.round30.ppo import tanh_log_prob

from .protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    D_SAFE_M,
    EXPECTED_HIGH_SHA,
    FINAL_CKPT,
    HOLD_STEPS,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LOG_STD_INIT,
    PAYLOAD_SCALE,
    SERVICE_SAFETY_MARGIN_M,
    high_sha,
    refuse_untouched,
)


def load_frozen_high() -> tuple[object, str]:
    digest = high_sha()
    if digest != EXPECTED_HIGH_SHA:
        raise RuntimeError(f"high-level SHA mismatch {digest}")
    model = load_deepsets_util(0, FINAL_CKPT)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, digest


def make_networks() -> tuple[ResidualActor, CentralizedCritic]:
    from wt.lowlevel_v2.critic_state import expected_critic_dim

    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    critic = CentralizedCritic(expected_critic_dim(6, 12))
    return actor, critic


@dataclass
class StepRecord:
    obs: dict
    global_state: np.ndarray
    pre_tanh: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    a_base: np.ndarray
    delta_tanh: np.ndarray
    a_des: np.ndarray
    a_projected: np.ndarray
    a_exec: np.ndarray
    projector_failure: bool
    preexisting_safety_violation: bool
    intervention_norm: float
    min_sep: float
    payload_inc: float


class ResidualWorker:
    def __init__(self, high, actor: ResidualActor, critic: CentralizedCritic) -> None:
        self.high = high
        self.actor = actor
        self.critic = critic
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
        info = self.env.reset(seed=int(seed))
        self.pending = 0
        self.option = None
        self.last_arrival = np.zeros(self.env.config.num_buoys, dtype=np.float32)
        self.last_positions = np.asarray(self.env.uav_positions, dtype=np.float32).copy()
        self.last_movement = np.zeros((self.env.config.num_uavs, 2), dtype=np.float64)
        return info

    def _refresh_option(self, info: dict) -> None:
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
        log_prob, _ = tanh_log_prob(mu, log_std, pre)
        return pre, log_prob

    def step(self, info: dict, *, deterministic: bool, zero_residual: bool = False) -> tuple[dict, StepRecord]:
        self._refresh_option(info)
        min_now = minimum_pairwise_distance(self.env.uav_positions)
        preexisting = bool(min_now + 1.0e-9 < D_SAFE_M)
        obs = build_actor_observation(self.env, self.option, self.last_movement)
        glob = build_critic_state(self.env, self.option, self.last_movement)
        a_base, _ = self.base.act(self.env, self.option)
        if preexisting:
            record = StepRecord(
                obs=obs,
                global_state=glob["vector"],
                pre_tanh=np.zeros((6, 2), dtype=np.float32),
                log_prob=0.0,
                value=0.0,
                reward=0.0,
                done=True,
                a_base=a_base,
                delta_tanh=np.zeros((6, 2)),
                a_des=a_base,
                a_projected=a_base,
                a_exec=np.zeros((6, 2)),
                projector_failure=False,
                preexisting_safety_violation=True,
                intervention_norm=0.0,
                min_sep=min_now,
                payload_inc=0.0,
            )
            return info, record
        with torch.no_grad():
            value = float(self.critic(torch.as_tensor(glob["vector"], dtype=torch.float32)).cpu())
            if zero_residual:
                pre = torch.zeros(6, 2)
                log_prob = torch.zeros(())
            else:
                pre, log_prob = self.act_tensors(obs, deterministic=deterministic)
        pre_np = pre.detach().cpu().numpy().astype(np.float64)
        a_des = residual_desired(a_base, pre_np, self.residual_cfg)
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
        pos_before = np.asarray(self.env.uav_positions, dtype=np.float64)
        after = self.env.step(a_exec, self.option.association)
        self.pending -= 1
        payload_inc = float(self.env.payload_bits - payload_before)
        r_payload = payload_inc / float(PAYLOAD_SCALE)
        delta_tanh = np.tanh(pre_np)
        scaled = float(ALPHA_RESIDUAL) * delta_tanh
        c_proj = float(np.mean(np.sum((a_exec - a_des) ** 2, axis=-1) / (ACTION_SCALE ** 2 + 1.0e-8)))
        c_res = float(np.mean(np.sum(scaled ** 2, axis=-1) / (ACTION_SCALE ** 2 + 1.0e-8)))
        reward = r_payload - LAMBDA_PROJECT * c_proj - LAMBDA_RESIDUAL * c_res
        if not np.isfinite(reward):
            raise RuntimeError("non-finite train reward")
        self.last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        self.last_positions = np.asarray(self.env.uav_positions, dtype=np.float32).copy()
        self.last_movement = a_exec.copy()
        done = bool(after.get("terminated", False) or failure)
        record = StepRecord(
            obs=obs,
            global_state=glob["vector"],
            pre_tanh=pre_np.astype(np.float32),
            log_prob=float(log_prob.detach().cpu()) if torch.is_tensor(log_prob) else float(log_prob),
            value=value,
            reward=float(reward),
            done=done,
            a_base=np.asarray(a_base, dtype=np.float64),
            delta_tanh=delta_tanh,
            a_des=a_des,
            a_projected=a_projected,
            a_exec=a_exec,
            projector_failure=bool(failure),
            preexisting_safety_violation=False,
            intervention_norm=float(proj.intervention_norm),
            min_sep=minimum_pairwise_distance(self.env.uav_positions),
            payload_inc=payload_inc,
        )
        _ = pos_before
        return after, record
