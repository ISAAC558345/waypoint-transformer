"""Experiment rollouts. The formal environment step is called, not replaced."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    HOLD,
    SERVICE_SAFETY_MARGIN_M,
    TRAIN_POOL,
)
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment
from wt.benchmark_v2.phase6i_priority_tag.follow import follow_step
from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.critic_state import build_critic_state, expected_critic_dim
from wt.lowlevel_v2.observation import FORBIDDEN_KEYS, build_actor_observation, communication_features
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .features import assigned_features, queue_age
from .protocol import EVAL_ASSIGNMENT_SEED, FEATURE_NAMES, L1_LAMBDA_U, LAMBDA_U, RHO, TOL_BITS
from .reward import experiment_reward


def critic_dim(num_uavs: int = 6, num_buoys: int = 12) -> int:
    return int(expected_critic_dim(int(num_uavs), int(num_buoys)) + int(num_uavs) * len(FEATURE_NAMES))


def critic_vector(env, option, last_movement, priority: np.ndarray) -> np.ndarray:
    base = build_critic_state(env, option, last_movement)
    extra = np.asarray(priority, dtype=np.float64).reshape(-1)
    vector = np.concatenate([base["vector"], extra], axis=0)
    if int(vector.size) != critic_dim(int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("experiment critic width does not match the schema")
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("NONFINITE")
    return vector


def _fingerprint(env: L3BBitQueueEnvV2) -> dict:
    return {
        "t": int(env.t),
        "payload_bits": float(env.payload_bits),
        "q_bits": np.asarray(env.q_bits, dtype=np.float64).copy(),
        "uav_positions": np.asarray(env.uav_positions, dtype=np.float64).copy(),
    }


class ExperimentWorker:
    """One environment. Ledger following is explicit and can be switched off."""

    def __init__(
        self,
        model_seed: int,
        env_index: int,
        actor,
        critic: CentralizedCritic | None,
        *,
        lambda_u: float,
        rho: float,
        use_ledger: bool,
        deterministic: bool,
        scenario_id: int | None = None,
    ) -> None:
        self.model_seed = int(model_seed)
        self.env_index = int(env_index)
        self.actor = actor
        self.critic = critic
        self.lambda_u = float(lambda_u)
        self.rho = float(rho)
        self.use_ledger = bool(use_ledger)
        self.deterministic = bool(deterministic)
        self.fixed_scenario = None if scenario_id is None else int(scenario_id)
        self.env = L3BBitQueueEnvV2()
        self.projector = SafetyProjector(
            CollisionConfig(0.4, 0.2, placeholder=False),
            safety_margin=float(SERVICE_SAFETY_MARGIN_M),
        )
        self.rng = np.random.default_rng(321_000_000 + self.model_seed * 10_000 + self.env_index)
        self.cursor = int(self.model_seed + self.env_index) % len(TRAIN_POOL)
        self.option = None
        self.pending = 0
        self.last_movement = np.zeros((6, 2), dtype=np.float64)
        self.ledger: PriorityTagLedger | None = None
        self.scenario = 0

    def reset(self) -> None:
        if self.fixed_scenario is None:
            self.scenario = int(TRAIN_POOL[self.cursor % len(TRAIN_POOL)])
            self.cursor += 1
            from .protocol import assert_train_scenario

            assert_train_scenario(self.scenario)
        else:
            from .protocol import DEV_POOL, assert_dev_scenario, assert_train_scenario

            self.scenario = int(self.fixed_scenario)
            if self.scenario in TRAIN_POOL:
                assert_train_scenario(self.scenario)
            elif self.scenario in DEV_POOL:
                assert_dev_scenario(self.scenario)
            else:
                raise RuntimeError(f"scenario {self.scenario} is outside the registered train and development pools")
        self.env.reset(seed=int(self.scenario))
        self.option = None
        self.pending = 0
        self.last_movement = np.zeros((self.env.config.num_uavs, 2), dtype=np.float64)
        self.ledger = PriorityTagLedger(int(self.env.config.num_buoys)) if self.use_ledger else None
        if self.ledger is not None:
            self.ledger.reset_from_queue(self.env.q_bits)

    def _priority(self) -> np.ndarray:
        if self.ledger is None or self.option is None:
            return np.zeros((int(self.env.config.num_uavs), len(FEATURE_NAMES)), dtype=np.float64)
        return assigned_features(self.ledger, int(self.env.t), self.option.association, int(self.env.config.num_buoys))

    def _act(self, obs: dict, priority: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self_f = torch.as_tensor(obs["self"], dtype=torch.float32)
        task_f = torch.as_tensor(obs["task"], dtype=torch.float32)
        comm_f = torch.as_tensor(obs["communication"], dtype=torch.float32)
        neigh = torch.as_tensor(obs["neighbor_set"], dtype=torch.float32)
        near = torch.as_tensor(obs["d_nearest"], dtype=torch.float32)
        priority_t = torch.as_tensor(priority, dtype=torch.float32)
        if isinstance(self.actor, ResidualActor):
            mu, log_std = self.actor(self_f, task_f, comm_f, neigh, near)
        else:
            mu, log_std = self.actor(self_f, task_f, comm_f, neigh, near, priority_t)
        if self.deterministic:
            pre = mu.detach()
            zeros = torch.zeros_like(pre[..., 0])
            return pre.cpu().numpy().astype(np.float64), zeros.cpu().numpy().astype(np.float32)
        pre = Normal(mu, log_std.exp()).sample()
        log_uav, _entropy = tanh_log_prob_uav(mu, log_std, pre)
        return pre.detach().cpu().numpy().astype(np.float64), log_uav.detach().cpu().numpy().astype(np.float32)

    def step(self) -> dict:
        if self.pending <= 0:
            self.option = make_option(self.env, sample_assignment(self.env, self.rng))
            self.pending = int(HOLD)
        if self.option is None or not np.array_equal(self.option.association, self.option.targets):
            raise RuntimeError("INTERFACE")
        association = np.asarray(self.option.association, dtype=np.int64).copy()
        obs = build_actor_observation(self.env, self.option, self.last_movement)
        if obs.get("leaks_future_arrivals") or any(key in obs for key in FORBIDDEN_KEYS):
            raise RuntimeError("future arrival information entered the observation")
        priority = self._priority()
        vector = critic_vector(self.env, self.option, self.last_movement, priority)
        with torch.no_grad():
            value = 0.0 if self.critic is None else float(self.critic(torch.as_tensor(vector, dtype=torch.float32)).cpu())
            pre, log_uav = self._act(obs, priority)
        base = fly_toward(self.env, association)
        delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
        desired = clip_unit_step(base + delta)
        projected = self.projector.project(self.env, desired, self.option)
        failure = projected.constraint_status != "ok" or not projected.finite
        executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
        executed = clip_unit_step(executed)
        if self.ledger is None:
            info = self.env.step(executed, association)
            served = float(np.sum(info["served_bits"]))
            urgency_bits = 0.0
            served_tagged = 0.0
            served_ordinary = served
            dropped_tagged = 0.0
            dropped_ordinary = float(np.sum(info["dropped_bits_step"]))
            ledger_gap = 0.0
            mean_age = 0.0
            max_age = 0.0
            remaining_tagged = 0.0
        else:
            followed = follow_step(self.env, self.ledger, executed, association, int(self.scenario))
            served = float(followed["served_bits"])
            urgency_bits = float(followed["urgency_served_bits"])
            served_tagged = float(followed["served_tagged_bits"])
            served_ordinary = float(followed["served_ordinary_bits"])
            dropped_tagged = float(followed["dropped_tagged_bits"])
            dropped_ordinary = float(followed["dropped_ordinary_bits"])
            ledger_gap = float(followed["ledger_gap_bits"])
            mean_age, max_age = queue_age(self.ledger, int(self.env.t))
            _ordinary, tagged = self.ledger.bits_by_tag()
            remaining_tagged = float(np.sum(tagged))
            if abs(served_ordinary + served_tagged - served) > float(TOL_BITS):
                raise RuntimeError("served tag split does not match served bits")
        error = float(np.max(np.abs(self.env.conservation_error_bits())))
        if error > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        reward, base_reward, bonus = experiment_reward(
            served,
            urgency_bits,
            executed,
            desired,
            delta,
            self.env.block_bits,
            lambda_u=self.lambda_u,
            rho=self.rho,
        )
        comm = communication_features(self.env)
        self.pending -= 1
        self.option.remaining_hold = int(self.pending)
        self.last_movement = executed.copy()
        done = bool(int(self.env.t) >= int(self.env.config.episode_horizon) or failure)
        record = {
            "obs": obs,
            "priority": priority.astype(np.float32),
            "global_state": vector.astype(np.float32),
            "pre_tanh": pre.astype(np.float32),
            "log_prob_uav": log_uav,
            "policy_active_mask": np.ones(self.env.config.num_uavs, dtype=np.float32),
            "value": value,
            "reward": reward,
            "base_reward": base_reward,
            "bonus": bonus,
            "served_bits": served,
            "urgency_served_bits": urgency_bits,
            "served_tagged_bits": served_tagged,
            "served_ordinary_bits": served_ordinary,
            "dropped_tagged_bits": dropped_tagged,
            "dropped_ordinary_bits": dropped_ordinary,
            "ledger_gap_bits": ledger_gap,
            "queue_age_mean": mean_age,
            "queue_age_max": max_age,
            "remaining_tagged_bits": remaining_tagged,
            "sinr": float(np.mean(self.env.last_sinr)),
            "interference": float(np.mean(comm["total_interference"])),
            "path_step": float(np.sum(np.linalg.norm(executed, axis=-1) * self.env.config.step_length_m)),
            "projected": int(failure or float(projected.intervention_norm) > 1.0e-3),
            "constraint_failure": int(failure),
            "safety_violation": int(minimum_pairwise_distance(self.env.uav_positions) + 1.0e-9 < float(D_SAFE_M)),
            "done": done,
            "payload": float(self.env.payload_bits),
            "action": executed.copy(),
            "fingerprint": _fingerprint(self.env),
            "q_bits": float(np.sum(self.env.q_bits)),
        }
        if done and self.fixed_scenario is None:
            self.reset()
        return record


def run_fixed_episode(worker: ExperimentWorker, steps: int) -> list[dict]:
    worker.reset()
    rows = []
    for _ in range(int(steps)):
        if int(worker.env.t) >= int(worker.env.config.episode_horizon):
            break
        rows.append(worker.step())
    return rows


def eval_worker(actor, scenario_id: int, *, lambda_u: float, use_ledger: bool) -> ExperimentWorker:
    from .protocol import assert_dev_scenario, assert_train_scenario

    scenario = int(scenario_id)
    if int(scenario) in set(TRAIN_POOL):
        assert_train_scenario(scenario)
    else:
        assert_dev_scenario(scenario)
    worker = ExperimentWorker(
        EVAL_ASSIGNMENT_SEED + scenario,
        0,
        actor,
        None,
        lambda_u=float(lambda_u),
        rho=float(RHO),
        use_ledger=bool(use_ledger),
        deterministic=True,
        scenario_id=scenario,
    )
    worker.rng = np.random.default_rng(EVAL_ASSIGNMENT_SEED + scenario)
    return worker


def capture_bundle(worker: ExperimentWorker) -> dict:
    bundle = {"env": capture_state(worker.env), "pending": int(worker.pending), "last_movement": worker.last_movement.copy()}
    bundle["ledger"] = None if worker.ledger is None else worker.ledger.capture()
    bundle["association"] = None if worker.option is None else np.asarray(worker.option.association, dtype=np.int64).copy()
    bundle["remaining_hold"] = None if worker.option is None else int(worker.option.remaining_hold)
    return bundle


def restore_bundle(worker: ExperimentWorker, bundle: dict) -> None:
    restore_state(worker.env, bundle["env"])
    worker.pending = int(bundle["pending"])
    worker.last_movement = np.asarray(bundle["last_movement"], dtype=np.float64).copy()
    if worker.ledger is not None:
        worker.ledger.restore(bundle["ledger"])
    if bundle["association"] is not None:
        worker.option = make_option(worker.env, bundle["association"])
        worker.option.remaining_hold = int(bundle["remaining_hold"])
        worker.pending = int(bundle["remaining_hold"])


def variant_lambda(name: str) -> float:
    if name == "L1":
        return float(L1_LAMBDA_U)
    if name == "L2":
        return float(LAMBDA_U)
    if name == "L0":
        return 0.0
    raise RuntimeError(f"unknown variant {name}")
