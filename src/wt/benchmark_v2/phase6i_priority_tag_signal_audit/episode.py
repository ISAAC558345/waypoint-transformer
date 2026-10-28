"""Natural frozen episodes. Assignment and dynamics stay on the existing worker."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL, D_SAFE_M, LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag.follow import arrival_tags
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.context import apply_historical_context, context_is_historical
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.evaluate import load_l3, load_reference
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.worker import DirectServiceWorker
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import EVAL_ASSIGNMENT_SEED, FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import ExperimentWorker, critic_dim, eval_worker
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor

from .protocol import E10_OUTPUT, snapshot_indices
from .trace import bucket_snapshot, capture_worker, closest_pair, motion_terms, zero_tag_sensitivity


def make_worker(variant: str, seed: int, scenario: int) -> ExperimentWorker:
    apply_historical_context()
    if variant == "L0":
        actor = load_reference("L0", 0)
        worker = eval_worker(actor, int(scenario), lambda_u=0.0, use_ledger=True)
        return worker
    if variant != "L3":
        raise RuntimeError(f"diagnostic loader does not train or reload {variant}")
    actor = load_l3(int(seed))
    path = E10_OUTPUT / "checkpoints" / f"l3_seed{int(seed)}.pt"
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    critic = CentralizedCritic(critic_dim())
    critic.load_state_dict(payload["critic"], strict=True)
    critic.eval()
    worker = DirectServiceWorker(
        int(EVAL_ASSIGNMENT_SEED) + int(scenario),
        0,
        actor,
        critic,
        deterministic=True,
        scenario_id=int(scenario),
    )
    worker.rng = np.random.default_rng(int(EVAL_ASSIGNMENT_SEED) + int(scenario))
    return worker


def _observe(worker: ExperimentWorker, variant: str, seed: int) -> dict:
    before = bucket_snapshot(worker)
    generated_before = np.asarray(worker.env.cumulative_generated_arrival_bits, dtype=np.float64).copy()
    dropped_before = np.asarray(worker.env.cumulative_dropped_bits, dtype=np.float64).copy()
    positions_before = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
    seen: dict = {}
    original = worker.projector.project

    def _project(env, desired, option):
        association = np.asarray(option.association, dtype=np.int64).copy()
        obs = build_actor_observation(env, option, worker.last_movement)
        priority = worker._priority()
        seen["sensitivity"] = zero_tag_sensitivity(worker, obs, priority, association, original)
        seen["desired"] = np.asarray(desired, dtype=np.float64).copy()
        seen["association"] = association
        seen["priority"] = np.asarray(priority, dtype=np.float64).copy()
        result = original(env, desired, option)
        seen["projection"] = result
        return result

    worker.projector.project = _project
    try:
        record = worker.step()
    finally:
        worker.projector.project = original
    time_step = int(worker.env.t) - 1
    generated = np.asarray(worker.env.cumulative_generated_arrival_bits, dtype=np.float64) - generated_before
    dropped = np.asarray(worker.env.cumulative_dropped_bits, dtype=np.float64) - dropped_before
    pre = np.asarray(record["pre_tanh"], dtype=np.float64)
    delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
    terms = motion_terms(float(record["served_bits"]), np.asarray(record["action"]), seen["desired"], delta)
    pair_after = closest_pair(worker.env.uav_positions)
    pair_before = closest_pair(positions_before)
    projection = seen["projection"]
    attribution = getattr(worker.ledger, "step_attribution", None)
    age_bits = float(attribution["age_service_bits"]) if isinstance(attribution, dict) else float("nan")
    sensitivity = seen.get("sensitivity")
    if sensitivity is not None:
        sensitivity = dict(sensitivity)
        sensitivity["worker_mu_gap"] = float(np.max(np.abs(sensitivity["real_mu"] - pre)))
        sensitivity["worker_action_gap"] = float(np.max(np.abs(sensitivity["real_executed"] - np.asarray(record["action"]))))
    return {
        "variant": variant,
        "model_seed": int(seed),
        "scenario_id": int(worker.scenario),
        "step": time_step,
        "before_buckets": before,
        "after_buckets": bucket_snapshot(worker),
        "generated": generated,
        "dropped": dropped,
        "arrival_tags": arrival_tags(int(worker.scenario), time_step, generated),
        "association": seen["association"],
        "priority": seen["priority"],
        "desired": seen["desired"],
        "pre_tanh": pre,
        "action": np.asarray(record["action"], dtype=np.float64).copy(),
        "sensitivity": sensitivity,
        "projection_status": str(projection.constraint_status),
        "projection_intervention": float(projection.intervention_norm),
        "predicted_distance_m": float(projection.minimum_predicted_distance),
        "positions_before": positions_before,
        "positions_after": np.asarray(worker.env.uav_positions, dtype=np.float64).copy(),
        "distance_before_m": float(pair_before["distance_m"]),
        "distance_after_m": float(pair_after["distance_m"]),
        "uav_i": int(pair_after["uav_i"]),
        "uav_j": int(pair_after["uav_j"]),
        "served_bits": float(record["served_bits"]),
        "served_tagged_bits": float(record["served_tagged_bits"]),
        "served_ordinary_bits": float(record["served_ordinary_bits"]),
        "served_tagged_by_buoy": None if not isinstance(attribution, dict) else np.asarray(attribution["served_tagged_by_buoy"], dtype=np.float64).copy(),
        "served_ordinary_by_buoy": None if not isinstance(attribution, dict) else np.asarray(attribution["served_ordinary_by_buoy"], dtype=np.float64).copy(),
        "age_service_bits": age_bits,
        "bonus_age": float(record.get("bonus_age", 0.0)),
        "bonus_tag": float(record.get("bonus_tag", 0.0)),
        "base_reward": float(record["base_reward"]),
        "reward": float(record["reward"]),
        "payload_term": float(terms["payload_term"]),
        "motion_penalty": float(terms["motion_penalty"]),
        "payload_bits": float(record["payload"]),
        "path_step": float(record["path_step"]),
        "safety_violation": int(record["safety_violation"]),
        "constraint_failure": int(record["constraint_failure"]),
        "policy_active_mask": np.asarray(record["policy_active_mask"], dtype=np.float64).copy(),
        "value": float(record["value"]),
        "done": bool(record["done"]),
        "sinr": float(record["sinr"]),
        "ledger_gap_bits": float(record["ledger_gap_bits"]),
        "dust_bits": float(DUST_BITS),
        "counter_threshold_m": float(D_SAFE_M),
        "projector_threshold_m": float(worker.projector.collision.d_safe),
        "actor_accepts_tag": not isinstance(worker.actor, ResidualActor),
        "formal_actor": str(FORMAL_ACTOR),
        "log_std_init": float(LOG_STD_INIT),
    }


def run_episode(variant: str, seed: int, scenario: int) -> dict:
    rollout, original = _install()
    try:
        apply_historical_context()
        worker = make_worker(variant, int(seed), int(scenario))
        applied = apply_historical_context()
        worker.reset()
        horizon = int(worker.env.config.episode_horizon)
        selected = set(snapshot_indices(horizon))
        steps = []
        snapshots = []
        for index in range(horizon):
            if index in selected:
                snapshots.append({"step": int(index), "capture": capture_worker(worker)})
            steps.append(_observe(worker, variant, int(seed)))
        return {
            "variant": variant,
            "model_seed": int(seed),
            "scenario_id": int(scenario),
            "steps": steps,
            "snapshots": snapshots,
            "worker": worker,
            "historical_context": applied,
            "context_is_historical": bool(context_is_historical(applied)),
            "assignment_seed": int(EVAL_ASSIGNMENT_SEED) + int(scenario),
            "assignment_protocol": "evaluation_rng_not_training_rng",
        }
    finally:
        _restore(rollout, original)
