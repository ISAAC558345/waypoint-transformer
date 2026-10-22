"""Original serial H1. Calls LockedExecutor.act and the original environment step."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv, active_links
from wt.lowlevel_v2.contracts import clip_unit_step, refuse_rematch
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1b.protocol import ALPHA_RESIDUAL
from wt.lowlevel_v2.phase1b.service_lock import apply_service_residual_lock, service_lock_mask
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5zd_executor_response_value.response import assemble_response


def _option(env: L3BEnv, assignment: np.ndarray) -> TaskOption:
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=chosen.copy(),
        decision_time=int(env.t),
        remaining_hold=15,
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=chosen.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def _response_from(clone: L3BEnv, info: dict, chosen: np.ndarray, sinr_0, interference_0, progress_0, pos_0, assignment_0, buoy_pos) -> np.ndarray:
    active = {int(uav) for uav, _buoy in active_links(clone)}
    in_range = np.asarray(
        [float(clone._in_range(uav, int(buoy))) if int(buoy) >= 0 else 0.0 for uav, buoy in enumerate(chosen)],
        dtype=np.float64,
    )
    service_active = np.asarray([float(uav in active) for uav in range(clone.config.num_uavs)], dtype=np.float64)
    return assemble_response(
        sinr_0=sinr_0,
        sinr_1=np.asarray(info["sinr"], dtype=np.float64),
        interference_0=interference_0,
        interference_1=np.asarray(info["interference_uav_uav"], dtype=np.float64).sum(axis=1),
        progress_0=progress_0,
        progress_1=np.asarray(info["service_progress"], dtype=np.float64),
        rate_1=np.asarray(info["rate_bps"], dtype=np.float64),
        pos_0=pos_0,
        pos_1=np.asarray(info["uav_positions_m"], dtype=np.float64),
        buoy_pos=buoy_pos,
        assignment_0=assignment_0,
        assignment_1=np.asarray(info["assignment"], dtype=np.int64),
        in_range=in_range,
        service_active=service_active,
    )


def _empty(env: L3BEnv, count: int) -> dict:
    uavs = int(env.config.num_uavs)
    return {
        "actions": np.zeros((count, uavs, 2), dtype=np.float64),
        "desired": np.zeros((count, uavs, 2), dtype=np.float64),
        "projected": np.zeros((count, uavs, 2), dtype=np.float64),
        "mu": np.zeros((count, uavs, 2), dtype=np.float64),
        "positions": np.zeros((count, uavs, 2), dtype=np.float64),
        "displacement": np.zeros((count, uavs, 2), dtype=np.float64),
        "queues": np.zeros((count, env.config.num_buoys), dtype=np.float64),
        "progress": np.zeros((count, uavs), dtype=np.float64),
        "sinr": np.zeros((count, uavs), dtype=np.float64),
        "rate": np.zeros((count, uavs), dtype=np.float64),
        "interference": np.zeros((count, uavs), dtype=np.float64),
        "payload": np.zeros(count, dtype=np.float64),
        "served": np.zeros((count, uavs), dtype=np.int64),
        "assignment_1": np.zeros((count, uavs), dtype=np.int64),
        "response": [],
        "constraint_status": [],
    }


def serial_h1(env: L3BEnv, assignments: np.ndarray, actor, movement: np.ndarray, *, profile: bool) -> dict:
    """One original step per candidate. profile=False calls executor.act as one bucket."""
    executor = LockedExecutor(actor)
    count = int(len(assignments))
    packed = _empty(env, count)
    timing = {
        "state_copy_ms": 0.0,
        "actor_observation_ms": 0.0,
        "mappo_forward_ms": 0.0,
        "residual_lock_ms": 0.0,
        "safety_projection_ms": 0.0,
        "env_step_ms": 0.0,
        "response_assembly_ms": 0.0,
        "executor_act_ms": 0.0,
    }
    responses = []
    for index, assignment in enumerate(assignments):
        started = time.perf_counter()
        clone = copy.deepcopy(env)
        timing["state_copy_ms"] += (time.perf_counter() - started) * 1000.0
        chosen = np.asarray(assignment, dtype=np.int64).copy()
        option = _option(clone, chosen)
        sinr_0 = np.array(clone.last_sinr, copy=True)
        progress_0 = np.array(clone.progress, copy=True)
        pos_0 = np.array(clone.uav_positions, copy=True)
        assignment_0 = np.array(clone.assignment, copy=True)
        buoy_pos = np.array(clone.buoy_positions, copy=True)
        started = time.perf_counter()
        interference_0 = np.asarray(clone.info()["interference_uav_uav"], dtype=np.float64).sum(axis=1)
        timing["response_assembly_ms"] += (time.perf_counter() - started) * 1000.0
        if profile:
            started = time.perf_counter()
            obs = build_actor_observation(clone, option, np.array(movement, copy=True))
            timing["actor_observation_ms"] += (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            with torch.no_grad():
                mu, _log_std = actor(
                    torch.as_tensor(obs["self"], dtype=torch.float32),
                    torch.as_tensor(obs["task"], dtype=torch.float32),
                    torch.as_tensor(obs["communication"], dtype=torch.float32),
                    torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                    torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing["mappo_forward_ms"] += (time.perf_counter() - started) * 1000.0
            pre = mu.detach().cpu().numpy().astype(np.float64)
            started = time.perf_counter()
            base, _flags = executor.base.act(clone, option)
            lock = service_lock_mask(clone, option)
            desired, _delta = apply_service_residual_lock(base, pre, lock, ALPHA_RESIDUAL)
            timing["residual_lock_ms"] += (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            projected = executor.projector.project(clone, desired, option)
            timing["safety_projection_ms"] += (time.perf_counter() - started) * 1000.0
            if projected.constraint_status != "ok":
                executed = np.zeros_like(desired)
            else:
                executed = np.asarray(projected.projected_actions, dtype=np.float64)
            executed = clip_unit_step(executed)
            packed["mu"][index] = pre
            packed["desired"][index] = np.asarray(desired, dtype=np.float64)
            packed["projected"][index] = np.asarray(projected.projected_actions, dtype=np.float64)
            packed["constraint_status"].append(str(projected.constraint_status))
        else:
            started = time.perf_counter()
            executed, desired, _delta = executor.act(clone, option, np.array(movement, copy=True))
            timing["executor_act_ms"] += (time.perf_counter() - started) * 1000.0
            packed["desired"][index] = np.asarray(desired, dtype=np.float64)
            packed["constraint_status"].append("official_act")
        packed["actions"][index] = np.asarray(executed, dtype=np.float64)
        started = time.perf_counter()
        info = clone.step(np.asarray(executed, dtype=np.float64), option.association)
        timing["env_step_ms"] += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        response = _response_from(clone, info, chosen, sinr_0, interference_0, progress_0, pos_0, assignment_0, buoy_pos)
        timing["response_assembly_ms"] += (time.perf_counter() - started) * 1000.0
        responses.append(response)
        packed["positions"][index] = np.asarray(info["uav_positions_m"], dtype=np.float64)
        packed["displacement"][index] = packed["positions"][index] - pos_0
        packed["queues"][index] = np.asarray(info["queue_blocks"], dtype=np.float64)
        packed["progress"][index] = np.asarray(info["service_progress"], dtype=np.float64)
        packed["sinr"][index] = np.asarray(info["sinr"], dtype=np.float64)
        packed["rate"][index] = np.asarray(info["rate_bps"], dtype=np.float64)
        packed["interference"][index] = np.asarray(info["interference_uav_uav"], dtype=np.float64).sum(axis=1)
        packed["payload"][index] = float(info["cumulative_payload_bits"])
        packed["served"][index] = np.array(clone.served_k, copy=True)
        packed["assignment_1"][index] = np.asarray(info["assignment"], dtype=np.int64)
    packed["response"] = np.stack(responses)
    packed["timing"] = timing
    return packed


def step_original(env: L3BEnv, assignments: np.ndarray, actions: np.ndarray) -> dict:
    """Original environment step for actions that were already computed."""
    count = int(len(assignments))
    packed = _empty(env, count)
    timing = {"state_copy_ms": 0.0, "env_step_ms": 0.0, "response_assembly_ms": 0.0}
    responses = []
    for index, assignment in enumerate(assignments):
        started = time.perf_counter()
        clone = copy.deepcopy(env)
        timing["state_copy_ms"] += (time.perf_counter() - started) * 1000.0
        chosen = np.asarray(assignment, dtype=np.int64).copy()
        sinr_0 = np.array(clone.last_sinr, copy=True)
        progress_0 = np.array(clone.progress, copy=True)
        pos_0 = np.array(clone.uav_positions, copy=True)
        assignment_0 = np.array(clone.assignment, copy=True)
        buoy_pos = np.array(clone.buoy_positions, copy=True)
        started = time.perf_counter()
        interference_0 = np.asarray(clone.info()["interference_uav_uav"], dtype=np.float64).sum(axis=1)
        timing["response_assembly_ms"] += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        info = clone.step(np.asarray(actions[index], dtype=np.float64), chosen)
        timing["env_step_ms"] += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        response = _response_from(clone, info, chosen, sinr_0, interference_0, progress_0, pos_0, assignment_0, buoy_pos)
        timing["response_assembly_ms"] += (time.perf_counter() - started) * 1000.0
        responses.append(response)
        packed["actions"][index] = np.asarray(actions[index], dtype=np.float64)
        packed["positions"][index] = np.asarray(info["uav_positions_m"], dtype=np.float64)
        packed["displacement"][index] = packed["positions"][index] - pos_0
        packed["queues"][index] = np.asarray(info["queue_blocks"], dtype=np.float64)
        packed["progress"][index] = np.asarray(info["service_progress"], dtype=np.float64)
        packed["sinr"][index] = np.asarray(info["sinr"], dtype=np.float64)
        packed["rate"][index] = np.asarray(info["rate_bps"], dtype=np.float64)
        packed["interference"][index] = np.asarray(info["interference_uav_uav"], dtype=np.float64).sum(axis=1)
        packed["payload"][index] = float(info["cumulative_payload_bits"])
        packed["served"][index] = np.array(clone.served_k, copy=True)
        packed["assignment_1"][index] = np.asarray(info["assignment"], dtype=np.int64)
    packed["response"] = np.stack(responses)
    packed["timing"] = timing
    return packed


__all__ = ["serial_h1", "step_original"]
