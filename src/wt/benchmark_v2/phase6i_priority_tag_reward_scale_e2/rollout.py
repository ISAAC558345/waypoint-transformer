"""One frozen episode. Assignment uses the legal Phase-6G sampler; the actor mean is not trained."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    D_SAFE_M,
    HOLD,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LOG_STD_INIT,
    SERVICE_SAFETY_MARGIN_M,
    TRAIN_POOL,
    assert_allowed,
)
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import low_level_reward, make_option, sample_assignment
from wt.benchmark_v2.phase6i_priority_tag.follow import follow_step
from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.observation import FORBIDDEN_KEYS, build_actor_observation
from wt.lowlevel_v2.residual_policy import ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .protocol import ASSIGNMENT_SEED_BASE, CHECKPOINT, EPISODE_COUNT, EPISODE_STEPS


def train_scenarios() -> tuple[int, ...]:
    """First eight registry train ids. Each one must pass the Phase-6G train guard."""
    chosen = tuple(int(item) for item in tuple(TRAIN_POOL)[: int(EPISODE_COUNT)])
    if len(chosen) != int(EPISODE_COUNT):
        raise RuntimeError("train registry does not have eight scenarios")
    for scenario in chosen:
        assert_allowed(int(scenario), split="train")
    return chosen


def checkpoint_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(CHECKPOINT.read_bytes())
    return digest.hexdigest()


def load_frozen_actor() -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["actor"] if isinstance(payload, dict) and "actor" in payload else payload
    actor.load_state_dict(state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _penalties(executed: np.ndarray, desired: np.ndarray, delta: np.ndarray) -> tuple[float, float]:
    scale = float(ACTION_SCALE) ** 2 + 1.0e-8
    c_project = float(np.mean(np.sum((executed - desired) ** 2, axis=-1) / scale))
    c_residual = float(np.mean(np.sum(delta ** 2, axis=-1) / scale))
    return c_project, c_residual


def _act(env, actor: ResidualActor, projector: SafetyProjector, option, association: np.ndarray, last: np.ndarray) -> dict:
    observation = build_actor_observation(env, option, last)
    leaked = [str(key) for key in observation if str(key) in FORBIDDEN_KEYS or "tag" in str(key)]
    if bool(observation.get("leaks_future_arrivals", False)):
        leaked.append("leaks_future_arrivals")
    if leaked:
        raise RuntimeError(f"observation exposes {leaked}")
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(observation["self"], dtype=torch.float32),
            torch.as_tensor(observation["task"], dtype=torch.float32),
            torch.as_tensor(observation["communication"], dtype=torch.float32),
            torch.as_tensor(observation["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(observation["d_nearest"], dtype=torch.float32),
        )
    pre = mu.detach().cpu().numpy().astype(np.float64)
    delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
    desired = clip_unit_step(fly_toward(env, association) + delta)
    projected = projector.project(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    executed = clip_unit_step(executed)
    return {
        "executed": executed,
        "desired": desired,
        "delta": delta,
        "failure": bool(failure),
        "projected": bool(failure or float(projected.intervention_norm) > 1.0e-3),
        "observation_keys": sorted(str(key) for key in observation),
    }


def _fingerprint(env) -> dict[str, float]:
    return {
        "q_bits": float(np.sum(env.q_bits)),
        "payload_bits": float(env.payload_bits),
        "served_sum": float(np.sum(env.cumulative_served_bits)),
        "dropped_sum": float(np.sum(env.cumulative_dropped_bits)),
        "position_sum": float(np.sum(env.uav_positions)),
        "t": float(env.t),
    }


def roll_episode(actor: ResidualActor, scenario_id: int, *, use_ledger: bool) -> dict:
    """Run one horizon. The ledger reads the formal step and does not choose the action."""
    assert_allowed(int(scenario_id), split="train")
    if int(scenario_id) not in train_scenarios():
        raise RuntimeError("scenario is outside the eight-episode train prefix")
    env = L3BBitQueueEnvV2()
    if int(env.config.episode_horizon) != int(EPISODE_STEPS):
        raise RuntimeError("episode horizon is not 48")
    env.reset(seed=int(scenario_id))
    ledger = PriorityTagLedger(int(env.config.num_buoys))
    if use_ledger:
        ledger.reset_from_queue(env.q_bits)
        ledger.assert_matches(env.q_bits, float(env.tol_bits))
    projector = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M))
    rng = np.random.default_rng(int(ASSIGNMENT_SEED_BASE) + int(scenario_id))
    option = None
    pending = 0
    last = np.zeros((int(env.config.num_uavs), 2), dtype=np.float64)
    rows: list[dict] = []
    tag_rows: list[list] = []
    action_checksums: list[float] = []
    official_gaps: list[float] = []
    conservation: list[float] = []
    observation_keys: list[str] = []
    started = time.perf_counter()
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            option = make_option(env, sample_assignment(env, rng))
            pending = int(HOLD)
        assert option is not None
        association = np.asarray(option.association, dtype=np.int64).copy()
        if not np.array_equal(option.association, option.targets):
            raise RuntimeError("task option association does not match its targets")
        acted = _act(env, actor, projector, option, association, last)
        observation_keys = list(acted["observation_keys"])
        executed = acted["executed"]
        if use_ledger:
            followed = follow_step(env, ledger, executed, association, int(scenario_id))
            served = float(followed["served_bits"])
        else:
            info = env.step(executed, association)
            served = float(np.sum(info["served_bits"]))
            followed = None
        c_project, c_residual = _penalties(executed, acted["desired"], acted["delta"])
        official, communication = low_level_reward(served, executed, acted["desired"], acted["delta"], float(env.block_bits))
        recomputed = float(communication) - float(LAMBDA_PROJECT) * c_project - float(LAMBDA_RESIDUAL) * c_residual
        official_gaps.append(abs(float(official) - recomputed))
        conservation.append(float(np.max(np.abs(env.conservation_error_bits()))))
        safety = int(minimum_pairwise_distance(env.uav_positions) + 1.0e-9 < float(D_SAFE_M))
        action_checksums.append(float(np.sum(executed)))
        if followed is not None:
            drawn = [tag for tag in followed["tags"] if tag is not None]
            rows.append(
                {
                    "scenario_id": int(scenario_id),
                    "time_step": int(followed["time_step"]),
                    "assignment": "-".join(str(int(item)) for item in association),
                    "served_bits": served,
                    "payload_term": float(communication),
                    "urgency_served_bits": float(followed["urgency_served_bits"]),
                    "urgency_term": float(followed["urgency_served_bits"]) / 1.0e6,
                    "c_project": c_project,
                    "c_residual": c_residual,
                    "projection_penalty": float(LAMBDA_PROJECT) * c_project,
                    "residual_penalty": float(LAMBDA_RESIDUAL) * c_residual,
                    "penalty": float(LAMBDA_PROJECT) * c_project + float(LAMBDA_RESIDUAL) * c_residual,
                    "base_reward": recomputed,
                    "served_ordinary_bits": float(followed["served_ordinary_bits"]),
                    "served_tagged_bits": float(followed["served_tagged_bits"]),
                    "generated_bits": float(followed["generated_bits"]),
                    "admitted_bits": float(followed["admitted_bits"]),
                    "dropped_bits": float(followed["dropped_bits"]),
                    "admitted_ordinary_bits": float(followed["admitted_ordinary_bits"]),
                    "admitted_tagged_bits": float(followed["admitted_tagged_bits"]),
                    "dropped_ordinary_bits": float(followed["dropped_ordinary_bits"]),
                    "dropped_tagged_bits": float(followed["dropped_tagged_bits"]),
                    "snap_bits": float(followed["snap_bits"]),
                    "ledger_gap_bits": float(followed["ledger_gap_bits"]),
                    "q_bits": float(followed["q_bits"]),
                    "payload_bits": float(followed["payload_bits"]),
                    "safety_violation": safety,
                    "projected": int(acted["projected"]),
                    "constraint_failure": int(acted["failure"]),
                    "tag_draws": int(len(drawn)),
                    "tagged_draws": int(sum(int(tag) for tag in drawn)),
                    "action_checksum": float(np.sum(executed)),
                }
            )
            tag_rows.append(list(followed["tags"]))
        pending -= 1
        option.remaining_hold = int(pending)
        last = np.asarray(executed, dtype=np.float64).copy()
    ordinary = tagged = None
    if use_ledger:
        ordinary, tagged = ledger.bits_by_tag()
    return {
        "scenario_id": int(scenario_id),
        "elapsed_s": float(time.perf_counter() - started),
        "use_ledger": bool(use_ledger),
        "rows": rows,
        "tag_rows": tag_rows,
        "action_checksums": action_checksums,
        "fingerprint": _fingerprint(env),
        "observation_keys": observation_keys,
        "max_official_reward_gap": float(max(official_gaps) if official_gaps else 0.0),
        "max_conservation_error_bits": float(max(conservation) if conservation else 0.0),
        "remaining_ordinary_bits": None if ordinary is None else float(np.sum(ordinary)),
        "remaining_tagged_bits": None if tagged is None else float(np.sum(tagged)),
    }
