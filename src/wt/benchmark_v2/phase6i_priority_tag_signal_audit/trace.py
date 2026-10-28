"""Deterministic diagnostic steps. Offset zero follows the frozen worker."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL, D_SAFE_M, HOLD
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment
from wt.benchmark_v2.phase6i_priority_tag.follow import arrival_tags
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.worker import DirectServiceWorker
from wt.benchmark_v2.phase6i_priority_tag_reward.features import head_bucket
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import TOL_BITS
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import (
    ExperimentWorker,
    _fingerprint,
    capture_bundle,
    critic_vector,
    restore_bundle,
)
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import clip_unit_step, minimum_pairwise_distance, pairwise_distances
from wt.lowlevel_v2.observation import FORBIDDEN_KEYS, build_actor_observation, communication_features
from wt.lowlevel_v2.phase1.protocol import ACTION_SCALE, LAMBDA_PROJECT, LAMBDA_RESIDUAL
from wt.lowlevel_v2.residual_policy import ResidualActor

from .protocol import PROJECTOR_DISTANCE_TOLERANCE_M, SAFETY_DISTANCE_TOLERANCE_M, TAG_COLUMN


def motion_terms(served_bits: float, executed: np.ndarray, desired: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    scale = float(ACTION_SCALE) ** 2 + 1.0e-8
    projection = float(np.mean(np.sum((executed - desired) ** 2, axis=-1) / scale))
    residual = float(np.mean(np.sum(delta ** 2, axis=-1) / scale))
    payload = float(served_bits) / 1.0e6
    penalty = float(LAMBDA_PROJECT) * projection + float(LAMBDA_RESIDUAL) * residual
    return {"payload_term": payload, "motion_penalty": penalty, "projection_cost": projection, "residual_cost": residual}


def _mu(actor, obs: dict, priority: np.ndarray) -> np.ndarray:
    self_f = torch.as_tensor(obs["self"], dtype=torch.float32)
    task_f = torch.as_tensor(obs["task"], dtype=torch.float32)
    comm_f = torch.as_tensor(obs["communication"], dtype=torch.float32)
    neigh = torch.as_tensor(obs["neighbor_set"], dtype=torch.float32)
    near = torch.as_tensor(obs["d_nearest"], dtype=torch.float32)
    with torch.no_grad():
        if isinstance(actor, ResidualActor):
            mu, _log_std = actor(self_f, task_f, comm_f, neigh, near)
        else:
            priority_t = torch.as_tensor(priority, dtype=torch.float32)
            mu, _log_std = actor(self_f, task_f, comm_f, neigh, near, priority_t)
    return mu.detach().cpu().numpy().astype(np.float64)


def _desired(env, association: np.ndarray, pre: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    delta = float(ALPHA_RESIDUAL) * np.tanh(pre) + np.asarray(offset, dtype=np.float64).reshape(1, 2)
    raw = fly_toward(env, association) + delta
    desired = clip_unit_step(raw)
    clipped = bool(np.max(np.linalg.norm(raw, axis=-1)) > 1.0 + 1.0e-12)
    return desired, delta, clipped


def _execute(env, desired: np.ndarray, option, projector) -> tuple[np.ndarray, object, bool]:
    projected = projector(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    return clip_unit_step(executed), projected, bool(failure)


def closest_pair(positions: np.ndarray) -> dict[str, float | int]:
    distances = pairwise_distances(positions)
    count = int(distances.shape[0])
    best = (0, 1, float("inf"))
    for left in range(count):
        for right in range(left + 1, count):
            if float(distances[left, right]) < best[2]:
                best = (left, right, float(distances[left, right]))
    return {"uav_i": int(best[0]), "uav_j": int(best[1]), "distance_m": float(best[2])}


def capture_worker(worker: ExperimentWorker) -> dict:
    return {
        "bundle": capture_bundle(worker),
        "rng": copy.deepcopy(worker.rng.bit_generator.state),
        "torch": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_worker(worker: ExperimentWorker, snap: dict) -> None:
    restore_bundle(worker, snap["bundle"])
    worker.rng.bit_generator.state = copy.deepcopy(snap["rng"])
    torch.set_rng_state(snap["torch"].clone())
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])


def take_step(worker: ExperimentWorker, offset: np.ndarray) -> dict:
    """One frozen step with a residual offset. A zero offset is the worker action."""
    if worker.pending <= 0:
        worker.option = make_option(worker.env, sample_assignment(worker.env, worker.rng))
        worker.pending = int(HOLD)
    if worker.option is None or not np.array_equal(worker.option.association, worker.option.targets):
        raise RuntimeError("INTERFACE")
    association = np.asarray(worker.option.association, dtype=np.int64).copy()
    obs = build_actor_observation(worker.env, worker.option, worker.last_movement)
    if obs.get("leaks_future_arrivals") or any(key in obs for key in FORBIDDEN_KEYS):
        raise RuntimeError("future arrival information entered the observation")
    priority = worker._priority()
    vector = critic_vector(worker.env, worker.option, worker.last_movement, priority)
    with torch.no_grad():
        value = 0.0 if worker.critic is None else float(worker.critic(torch.as_tensor(vector, dtype=torch.float32)).cpu())
    pre = _mu(worker.actor, obs, priority)
    if worker.deterministic:
        log_uav = np.zeros(worker.env.config.num_uavs, dtype=np.float32)
    else:
        _pre_sampled, log_uav = worker._act(obs, priority)
        pre = np.asarray(_pre_sampled, dtype=np.float64)
    desired, delta, clipped = _desired(worker.env, association, pre, offset)
    executed, projected, failure = _execute(worker.env, desired, worker.option, worker.projector.project)
    generated_before = np.asarray(worker.env.cumulative_generated_arrival_bits, dtype=np.float64).copy()
    dropped_before = np.asarray(worker.env.cumulative_dropped_bits, dtype=np.float64).copy()
    from wt.benchmark_v2.phase6i_priority_tag_reward import rollout as rollout_module

    followed = rollout_module.follow_step(worker.env, worker.ledger, executed, association, int(worker.scenario))
    served = float(followed["served_bits"])
    reward_inputs = motion_terms(served, executed, desired, delta)
    from wt.benchmark_v2.phase6i_priority_tag_reward.reward import experiment_reward

    reward, base_reward, bonus = experiment_reward(
        served,
        float(followed["urgency_served_bits"]),
        executed,
        desired,
        delta,
        worker.env.block_bits,
        lambda_u=float(worker.lambda_u),
        rho=float(worker.rho),
    )
    attribution = getattr(worker.ledger, "step_attribution", None)
    if isinstance(worker, DirectServiceWorker):
        if not isinstance(attribution, dict):
            raise RuntimeError("L3 counterfactual is missing pre-service attribution")
        from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.reward import split_bonus

        if worker.use_bonus:
            bonus_age, bonus_tag = split_bonus(
                float(attribution["age_service_bits"]),
                float(attribution["served_tagged_bits"]),
                served,
            )
        else:
            bonus_age, bonus_tag = 0.0, 0.0
        bonus = float(bonus_age + bonus_tag)
        reward = float(base_reward) + bonus
    else:
        bonus_age, bonus_tag = 0.0, 0.0
    error = float(np.max(np.abs(worker.env.conservation_error_bits())))
    if error > float(worker.env.tol_bits):
        raise RuntimeError("ACCOUNTING")
    if abs(float(followed["served_ordinary_bits"]) + float(followed["served_tagged_bits"]) - served) > float(TOL_BITS):
        raise RuntimeError("served tag split does not match served bits")
    worker.pending -= 1
    worker.option.remaining_hold = int(worker.pending)
    worker.last_movement = executed.copy()
    done = bool(int(worker.env.t) >= int(worker.env.config.episode_horizon) or failure)
    comm = communication_features(worker.env)
    generated = np.asarray(worker.env.cumulative_generated_arrival_bits, dtype=np.float64) - generated_before
    dropped = np.asarray(worker.env.cumulative_dropped_bits, dtype=np.float64) - dropped_before
    pair = closest_pair(worker.env.uav_positions)
    record = {
        "reward": float(reward),
        "base_reward": float(base_reward),
        "bonus": float(bonus),
        "bonus_age": float(bonus_age),
        "bonus_tag": float(bonus_tag),
        "served_bits": served,
        "served_tagged_bits": float(followed["served_tagged_bits"]),
        "served_ordinary_bits": float(followed["served_ordinary_bits"]),
        "payload": float(worker.env.payload_bits),
        "action": executed.copy(),
        "desired": desired.copy(),
        "delta": delta.copy(),
        "pre_tanh": pre.copy(),
        "priority": priority.astype(np.float32),
        "clipped": bool(clipped),
        "projection_intervention": float(projected.intervention_norm),
        "constraint_failure": int(failure),
        "constraint_status": str(projected.constraint_status),
        "predicted_distance_m": float(projected.minimum_predicted_distance),
        "safety_violation": int(float(pair["distance_m"]) + float(SAFETY_DISTANCE_TOLERANCE_M) < float(D_SAFE_M)),
        "distance_m": float(pair["distance_m"]),
        "uav_i": int(pair["uav_i"]),
        "uav_j": int(pair["uav_j"]),
        "positions": np.asarray(worker.env.uav_positions, dtype=np.float64).copy(),
        "done": done,
        "value": float(value),
        "log_prob_uav": log_uav,
        "association": association.copy(),
        "generated": generated.copy(),
        "dropped": dropped.copy(),
        "arrival_tags": arrival_tags(int(worker.scenario), int(followed["time_step"]), generated),
        "time_step": int(followed["time_step"]),
        "payload_term": float(reward_inputs["payload_term"]),
        "motion_penalty": float(reward_inputs["motion_penalty"]),
        "fingerprint": _fingerprint(worker.env),
        "sinr": float(np.mean(worker.env.last_sinr)),
        "interference": float(np.mean(comm["total_interference"])),
        "path_step": float(np.sum(np.linalg.norm(executed, axis=-1) * worker.env.config.step_length_m)),
        "ledger_gap_bits": float(followed["ledger_gap_bits"]),
        "attribution": attribution,
        "projector_tolerance_m": float(PROJECTOR_DISTANCE_TOLERANCE_M),
        "counter_tolerance_m": float(SAFETY_DISTANCE_TOLERANCE_M),
        "projector_threshold_m": float(worker.projector.collision.d_safe),
        "counter_threshold_m": float(D_SAFE_M),
    }
    if float(DUST_BITS) < 0.0:
        raise RuntimeError("dust threshold drifted")
    return record


def zero_tag_sensitivity(worker: ExperimentWorker, obs: dict, priority: np.ndarray, association: np.ndarray, projector) -> dict | None:
    """Compare the frozen actor with the tag column cleared. L0 has no tag input."""
    if isinstance(worker.actor, ResidualActor):
        return None
    real = _mu(worker.actor, obs, priority)
    cleared = np.array(priority, dtype=np.float64, copy=True)
    cleared[:, int(TAG_COLUMN)] = 0.0
    alt = _mu(worker.actor, obs, cleared)
    real_desired, _delta, _clipped = _desired(worker.env, association, real, np.zeros(2))
    alt_desired, _alt_delta, _alt_clipped = _desired(worker.env, association, alt, np.zeros(2))
    real_executed, _real_projected, _real_failure = _execute(worker.env, real_desired, worker.option, projector)
    alt_executed, _alt_projected, _alt_failure = _execute(worker.env, alt_desired, worker.option, projector)
    return {
        "mu_gap": float(np.max(np.abs(real - alt))),
        "desired_gap": float(np.max(np.abs(real_desired - alt_desired))),
        "executed_gap": float(np.max(np.abs(real_executed - alt_executed))),
        "tag_input_nonzero": bool(np.any(np.asarray(priority)[:, int(TAG_COLUMN)] != 0.0)),
        "real_mu": real,
        "real_executed": real_executed,
    }


def bucket_snapshot(worker: ExperimentWorker) -> list[list[dict]]:
    rows = []
    for buoy, buckets in enumerate(worker.ledger.buckets):
        alive = []
        for bucket in buckets:
            if float(bucket.bits_remaining) <= float(DUST_BITS):
                continue
            alive.append(
                {
                    "buoy": int(buoy),
                    "admission_step": int(bucket.admission_step),
                    "insert_index": int(bucket.insert_index),
                    "priority_tag": int(bucket.priority_tag),
                    "bits_remaining": float(bucket.bits_remaining),
                    "head": head_bucket(worker.ledger, buoy) is bucket,
                }
            )
        alive.sort(key=lambda item: fifo_key_from(item))
        if alive:
            alive[0]["head"] = True
        rows.append(alive)
    return rows


def fifo_key_from(item: dict) -> tuple[int, int]:
    return (int(item["admission_step"]), int(item["insert_index"]))


def head_identity(buckets: list[dict]) -> tuple[int, int] | None:
    if not buckets:
        return None
    head = min(buckets, key=fifo_key_from)
    return (int(head["admission_step"]), int(head["insert_index"]))
