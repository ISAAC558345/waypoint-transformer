"""Decision states for 310102 and 310103. Labels are rollout results, not model inputs."""
from __future__ import annotations

import copy

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase1b.protocol import LOG_STD_INIT
from wt.lowlevel_v2.phase5_hgnn_urgency.actor import original_actor
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch, observable_exclusive_assignment
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5e_utility_dataset.build import _legal_buoys
from wt.lowlevel_v2.phase5e_utility_dataset.utility import descending_rank
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS, MODEL_INPUT_KEYS

from .protocol import DECISION_TIMES, MAPPO_CKPT, assert_phase5g_seed


def load_frozen_actor():
    if not MAPPO_CKPT.is_file():
        raise FileNotFoundError(str(MAPPO_CKPT))
    actor = original_actor(float(LOG_STD_INIT))
    actor.load_state_dict(torch.load(MAPPO_CKPT, map_location="cpu"))
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def mappo_h15(env: L3BEnv, assignment: np.ndarray, actor, last_movement: np.ndarray) -> float:
    """Fifteen steps of the frozen Residual MAPPO. The actor weights are not updated."""
    clone = copy.deepcopy(env)
    executor = LockedExecutor(actor)
    targets = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=targets,
        decision_time=int(clone.t),
        remaining_hold=15,
        uav_mask=np.ones(clone.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(clone.config.num_buoys, dtype=bool),
        association=targets.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    start = float(clone.payload_bits)
    movement = np.asarray(last_movement, dtype=np.float64).copy()
    for _ in range(15):
        if int(clone.t) >= int(clone.config.episode_horizon):
            break
        executed, _desired, _delta = executor.act(clone, option, movement)
        clone.step(executed, option.association)
        movement = np.asarray(executed, dtype=np.float64)
    return float(clone.payload_bits - start)


def _assert_inputs(batch: dict) -> None:
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked:
        raise RuntimeError(f"model input contains label fields: {sorted(leaked)}")
    missing = [key for key in MODEL_INPUT_KEYS if key not in batch]
    if missing:
        raise RuntimeError(f"model input missing {missing}")


def _advance(env: L3BEnv, target: int, last_positions, last_arrival, last_movement):
    while int(env.t) < int(target):
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        assignment = observable_exclusive_assignment(env)
        movement = fly_toward(env, assignment)
        info = env.step(movement, assignment)
        last_positions = positions
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
        last_movement = np.asarray(movement, dtype=np.float64)
    if int(env.t) != int(target):
        raise RuntimeError(f"phase-5g stopped at t={env.t}, expected {target}")
    return last_positions, last_arrival, last_movement


def collect_scenario(seed: int, actor) -> list[dict]:
    assert_phase5g_seed(int(seed))
    env = L3BEnv()
    env.reset(seed=int(seed))
    last_positions = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    rows = []
    for moment in DECISION_TIMES:
        last_positions, last_arrival, last_movement = _advance(env, int(moment), last_positions, last_arrival, last_movement)
        info = env.info()
        base = observable_exclusive_assignment(env)
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        buoys = np.asarray(env.buoy_positions, dtype=np.float64).copy()
        graph_last = positions if last_positions is None else np.asarray(last_positions, dtype=np.float32)
        urgency = shared_task_urgency(env.queue)
        batch = graph_batch(info, env, last_arrival, graph_last, urgency=None)
        urgency_batch = graph_batch(info, env, last_arrival, graph_last, urgency=urgency)
        _assert_inputs(batch)
        _assert_inputs(urgency_batch)
        for focal in range(int(env.config.num_uavs)):
            legal = _legal_buoys(base, focal, int(env.config.num_buoys))
            utility_payload = []
            executed_payload = []
            assignment_scores = []
            queue = np.asarray(env.queue, dtype=np.float64)
            distance = np.asarray(info["distance_uav_buoy"], dtype=np.float64)
            rate = np.asarray(info["rate_if_assigned"], dtype=np.float64)
            for buoy in legal:
                trial = np.asarray(base, dtype=np.int64).copy()
                trial[focal] = int(buoy)
                utility_payload.append(float(shadow_h15_payload(env, trial)))
                executed_payload.append(float(mappo_h15(env, trial, actor, last_movement)))
                assignment_scores.append(float(queue[buoy]) * float(rate[focal, buoy]) / (float(distance[focal, buoy]) + 1.0))
            rows.append(
                {
                    "scenario_id": int(seed),
                    "time_step": int(moment),
                    "uav_id": int(focal),
                    "batch": batch,
                    "urgency_batch": urgency_batch,
                    "uav_positions": positions,
                    "buoy_positions": buoys,
                    "context": np.asarray(base, dtype=np.int64).copy(),
                    "candidate_ids": [int(item) for item in legal],
                    "utility_payload": utility_payload,
                    "utility_rank": descending_rank(utility_payload),
                    "executed_payload": executed_payload,
                    "assignment_score": assignment_scores,
                    "last_movement": np.asarray(last_movement, dtype=np.float64).copy(),
                }
            )
        print(f"scenario {seed} t={moment} groups={env.config.num_uavs}", flush=True)
    return rows


def replay_env(seed: int, moment: int) -> tuple[L3BEnv, np.ndarray]:
    assert_phase5g_seed(int(seed))
    env = L3BEnv()
    env.reset(seed=int(seed))
    last_positions = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    last_positions, last_arrival, last_movement = _advance(env, int(moment), last_positions, last_arrival, last_movement)
    return env, last_movement


__all__ = ["collect_scenario", "load_frozen_actor", "mappo_h15", "replay_env"]
