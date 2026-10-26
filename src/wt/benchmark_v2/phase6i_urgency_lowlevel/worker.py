"""Scheme C task worker. The low level only executes the frozen high-level assignment."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    DECISION_TIMES,
    HOLD,
    SERVICE_SAFETY_MARGIN_M,
    TRAIN_POOL,
)
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.protocol import AGE_SCALE_STEPS
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import ScoreTap
from wt.benchmark_v2.phase6i_urgency_stage_b.conditions import scheme_c_decision
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.htw_phase100.contract import TaskOption
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.residual_policy import CentralizedCritic
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .models import UrgencyActor
from .observation import build_urgency_actor_observation, build_urgency_critic_state
from .protocol import require_train_scenario
from .reward import l1_reward, l2_reward


class SchemeCWorker:
    def __init__(
        self,
        model_seed: int,
        env_index: int,
        actor: UrgencyActor,
        critic: CentralizedCritic,
        planner: FormalPlanner,
        tap: ScoreTap,
        variant: str,
    ) -> None:
        if variant not in ("L1", "L2"):
            raise RuntimeError(f"training variant must be L1 or L2, received {variant}")
        if planner.policy != "scheme_c":
            raise RuntimeError("training planner is not frozen scheme C")
        self.model_seed = int(model_seed)
        self.env_index = int(env_index)
        self.actor = actor
        self.critic = critic
        self.variant = str(variant)
        self.decide_assignment = scheme_c_decision(planner, tap)
        self.env = L3BBitQueueEnvV2()
        install_light_info(self.env)
        self.projector = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))
        self.ledger = AdmissionAgeLedger(int(self.env.config.num_buoys), AGE_SCALE_STEPS)
        self.cursor = int(self.model_seed + self.env_index) % len(TRAIN_POOL)
        self.option: TaskOption | None = None
        self.pending = 0
        self.last_movement = np.zeros((int(self.env.config.num_uavs), 2), dtype=np.float64)
        self.position_prev = np.zeros((int(self.env.config.num_uavs), 2), dtype=np.float64)
        self.predecessor = None
        self.scenario = int(TRAIN_POOL[self.cursor])
        self.resets = 0

    def reset(self) -> None:
        self.scenario = require_train_scenario(int(TRAIN_POOL[self.cursor % len(TRAIN_POOL)]))
        self.cursor += 1
        self.env.reset(seed=self.scenario)
        self.ledger.reset_from_queue(self.env.q_bits)
        self.ledger.assert_matches(self.env.q_bits, self.env.tol_bits)
        self.option = None
        self.pending = 0
        self.last_movement = np.zeros((self.env.config.num_uavs, 2), dtype=np.float64)
        self.position_prev = np.asarray(self.env.uav_positions, dtype=np.float64).copy()
        self.predecessor = None
        self.resets += 1

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

    def step(self) -> dict:
        decision_meta = None
        if self.pending <= 0:
            if int(self.env.t) not in DECISION_TIMES:
                raise RuntimeError(f"scheme C was requested at t={int(self.env.t)}")
            chosen, decision_meta = self.decide_assignment(self.env, self.position_prev, self.predecessor)
            if int(decision_meta.get("scheme_c_called", 0)) != 1:
                raise RuntimeError("training step did not call scheme C")
            self.option = make_option(self.env, np.asarray(chosen, dtype=np.int64))
            self.pending = int(HOLD)
        assert self.option is not None
        association = np.asarray(self.option.association, dtype=np.int64).copy()
        t_before = int(self.env.t)
        queue_before = np.asarray(self.env.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(self.env.cumulative_snap_bits, dtype=np.float64).copy()
        view = self.ledger.view(t_before, queue_before, float(self.env.tol_bits))
        obs = build_urgency_actor_observation(self.env, self.option, view, self.last_movement)
        glob = build_urgency_critic_state(self.env, self.option, view, self.last_movement)
        if int(obs["task"].shape[-1]) != 10 or int(glob["vector"].size) != 200:
            raise RuntimeError("urgency observation width drifted during training")
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
        step_start = np.asarray(self.env.uav_positions, dtype=np.float64).copy()
        info = self.env.step(executed, association)
        snap = snap_vector(self.env, snap_before, info)
        self.ledger.apply_transition(
            info["served_bits"],
            info["generated_arrival_bits"],
            info["admitted_arrival_bits"],
            info["dropped_bits_step"],
            snap,
            t_before,
            float(self.env.tol_bits),
        )
        gap = self.ledger.assert_matches(self.env.q_bits, float(self.env.tol_bits))
        account = float(np.max(np.abs(self.env.conservation_error_bits())))
        if account > float(self.env.tol_bits) or float(np.max(np.abs(gap))) > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        if self.variant == "L1":
            terms = l1_reward(info["served_bits"], executed, desired, delta, self.env.block_bits)
        else:
            terms = l2_reward(info["served_bits"], view["urgency"], executed, desired, delta, self.env.block_bits)
        self.pending -= 1
        self.option.remaining_hold = int(self.pending)
        self.last_movement = executed.copy()
        self.position_prev = step_start
        self.predecessor = t_before
        done = bool(int(self.env.t) >= int(self.env.config.episode_horizon) or failure)
        idle = 0 if decision_meta is None else int(np.sum(np.asarray(association) < 0))
        record = {
            "obs": obs,
            "global_state": np.asarray(glob["vector"], dtype=np.float32),
            "pre_tanh": pre.astype(np.float32),
            "log_prob_uav": log_uav,
            "policy_active_mask": np.ones(self.env.config.num_uavs, dtype=np.float32),
            "value": value,
            "reward": float(terms["reward"]),
            "communication_reward": float(terms["communication_reward"]),
            "urgency_reward": float(terms["urgency_reward"]),
            "served_bits": float(np.sum(info["served_bits"])),
            "urgent_service_bits": float(np.sum(np.asarray(view["urgency"]) * np.asarray(info["served_bits"], dtype=np.float64))),
            "projected": int(failure or float(projected.intervention_norm) > 1.0e-3),
            "safety_violation": int(minimum_pairwise_distance(self.env.uav_positions) + 1.0e-9 < float(D_SAFE_M)),
            "done": done,
            "payload": float(self.env.payload_bits),
            "decision": decision_meta,
            "idle_slots": idle,
            "scenario_id": int(self.scenario),
        }
        if done:
            self.reset()
        return record
