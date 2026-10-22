"""Experimental H1 path. The original executor and environment modules stay unchanged."""
from __future__ import annotations

import time
import types

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.l3a.env import competition_pairs, conflict_mask
from wt.l3b.env import L3BEnv, active_links, interference_matrix
from wt.lowlevel_v2.contracts import clip_unit_step, pairwise_distances, refuse_rematch
from wt.lowlevel_v2.observation import (
    _comm_vector,
    _self_vector,
    _task_vector,
    build_actor_observation,
    communication_features,
    neighbor_edges,
)
from wt.lowlevel_v2.phase1b.service_lock import apply_service_residual_lock, service_lock_mask
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


def info_without_counterfactual(self: L3BEnv) -> dict:
    """Same post-step record as the environment, without the unused counterfactual rate grid."""
    eta = np.linalg.norm(self.uav_positions[:, None, :] - self.buoy_positions[None, :, :], axis=-1) / max(
        self.config.step_length_m, 1.0e-9
    )
    blocked = conflict_mask(self.assignment, self.config.num_buoys)
    pairs = competition_pairs(self.assignment, self.config.num_buoys)
    distance_ub = np.linalg.norm(self.uav_positions[:, None, :] - self.buoy_positions[None, :, :], axis=-1)
    distance_uu = np.linalg.norm(self.uav_positions[:, None, :] - self.uav_positions[None, :, :], axis=-1)
    pressure = self.queue / max(float(self.config.queue_capacity_blocks), 1.0e-9)
    return {
        "time_step": int(self.t),
        "num_uavs": int(self.config.num_uavs),
        "num_buoys": int(self.config.num_buoys),
        "uav_positions_m": self.uav_positions.copy(),
        "buoy_positions_m": self.buoy_positions.copy(),
        "queue_blocks": self.queue.copy(),
        "service_progress": self.progress.copy(),
        "assignment": self.assignment.copy(),
        "eta_slots": eta,
        "cumulative_payload_bits": float(self.payload_bits),
        "dropped_blocks": float(self.dropped),
        "conflict_uav_mask": blocked,
        "conflict_rate": float(np.mean(blocked)) if blocked.size else 0.0,
        "n_conflict_pairs": int(len(pairs)),
        "competition_pairs": pairs,
        "exclusive_ok": bool(len(pairs) == 0),
        "sinr": self.last_sinr.copy(),
        "rate_bps": self.last_rate.copy(),
        "distance_uav_buoy": distance_ub,
        "distance_uav_uav": distance_uu,
        "rate_if_assigned": np.zeros((self.config.num_uavs, self.config.num_buoys)),
        "interference_uav_uav": interference_matrix(self),
        "queue_pressure": pressure,
    }


def _capture(env: L3BEnv) -> dict:
    return {
        "uav_positions": np.array(env.uav_positions, copy=True),
        "queue": np.array(env.queue, copy=True),
        "progress": np.array(env.progress, copy=True),
        "served_k": np.array(env.served_k, copy=True),
        "assignment": np.array(env.assignment, copy=True),
        "payload_bits": float(env.payload_bits),
        "dropped": float(env.dropped),
        "t": int(env.t),
        "last_sinr": np.array(env.last_sinr, copy=True),
        "last_rate": np.array(env.last_rate, copy=True),
    }


def _restore(env: L3BEnv, snap: dict) -> None:
    env.uav_positions = np.array(snap["uav_positions"], copy=True)
    env.queue = np.array(snap["queue"], copy=True)
    env.progress = np.array(snap["progress"], copy=True)
    env.served_k = np.array(snap["served_k"], copy=True)
    env.assignment = np.array(snap["assignment"], copy=True)
    env.payload_bits = float(snap["payload_bits"])
    env.dropped = float(snap["dropped"])
    env.t = int(snap["t"])
    env.last_sinr = np.array(snap["last_sinr"], copy=True)
    env.last_rate = np.array(snap["last_rate"], copy=True)


def _observations(env: L3BEnv, assignments: np.ndarray, movement: np.ndarray) -> list[dict]:
    """Assignment-specific task and neighbor features. The pre-step channel summary is computed once."""
    comm = communication_features(env)
    last = np.asarray(movement, dtype=np.float64)
    count = int(env.config.num_uavs)
    self_all = np.stack([_self_vector(env, last, uav) for uav in range(count)])
    distances = pairwise_distances(env.uav_positions)
    nearest = np.asarray(
        [float(np.min(np.delete(distances[uav], uav))) for uav in range(count)],
        dtype=np.float64,
    )
    rows = []
    for assignment in assignments:
        option = _option(env, assignment)
        rows.append(
            {
                "self": self_all,
                "task": np.stack([_task_vector(env, option, uav) for uav in range(count)]),
                "communication": np.stack([_comm_vector(env, option, comm, uav) for uav in range(count)]),
                "neighbor_set": np.stack([neighbor_edges(env, option, last, uav) for uav in range(count)]),
                "d_nearest": nearest,
                "option": option,
            }
        )
    return rows


def observation_gap(env: L3BEnv, assignment: np.ndarray, movement: np.ndarray) -> float:
    cached = _observations(env, np.asarray(assignment, dtype=np.int64).reshape(1, -1), movement)[0]
    original = build_actor_observation(env, _option(env, assignment), movement)
    gap = 0.0
    for key in ("self", "task", "communication", "neighbor_set", "d_nearest"):
        gap = max(gap, float(np.max(np.abs(np.asarray(cached[key]) - np.asarray(original[key])))))
    return gap


def batched_executor_act(env: L3BEnv, assignments: np.ndarray, actor, movement: np.ndarray, executor) -> tuple[np.ndarray, dict[str, float]]:
    """One frozen forward over candidates. Lock and safety projection stay inside each assignment."""
    started = time.perf_counter()
    observations = _observations(env, assignments, movement)
    observation_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(np.stack([row["self"] for row in observations]), dtype=torch.float32),
            torch.as_tensor(np.stack([row["task"] for row in observations]), dtype=torch.float32),
            torch.as_tensor(np.stack([row["communication"] for row in observations]), dtype=torch.float32),
            torch.as_tensor(np.stack([row["neighbor_set"] for row in observations]), dtype=torch.float32),
            torch.as_tensor(np.stack([row["d_nearest"] for row in observations]), dtype=torch.float32),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - started) * 1000.0
    mus = mu.detach().cpu().numpy().astype(np.float64)
    actions = np.zeros((len(observations), env.config.num_uavs, 2), dtype=np.float64)
    desired_all = np.zeros_like(actions)
    projected_all = np.zeros_like(actions)
    status = []
    lock_ms = 0.0
    project_ms = 0.0
    for index, row in enumerate(observations):
        started = time.perf_counter()
        base, _flags = executor.base.act(env, row["option"])
        lock = service_lock_mask(env, row["option"])
        desired, _delta = apply_service_residual_lock(base, mus[index], lock, executor_alpha())
        desired_all[index] = np.asarray(desired, dtype=np.float64)
        lock_ms += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        projected = executor.projector.project(env, desired, row["option"])
        projected_all[index] = np.asarray(projected.projected_actions, dtype=np.float64)
        status.append(str(projected.constraint_status))
        if projected.constraint_status != "ok":
            executed = np.zeros_like(desired)
        else:
            executed = np.asarray(projected.projected_actions, dtype=np.float64)
        actions[index] = clip_unit_step(executed)
        project_ms += (time.perf_counter() - started) * 1000.0
    return actions, {
        "actor_observation_ms": observation_ms,
        "mappo_forward_ms": forward_ms,
        "residual_lock_ms": lock_ms,
        "safety_projection_ms": project_ms,
        "mu": mus,
        "desired": desired_all,
        "projected": projected_all,
        "constraint_status": status,
    }


def executor_alpha() -> float:
    from wt.lowlevel_v2.phase1b.protocol import ALPHA_RESIDUAL

    return float(ALPHA_RESIDUAL)


def slim_steps(env: L3BEnv, assignments: np.ndarray, actions: np.ndarray) -> dict:
    """Original step physics on one restored clone. The unused counterfactual rate grid is not built."""
    import copy

    started = time.perf_counter()
    snap = _capture(env)
    work = copy.deepcopy(env)
    work.info = types.MethodType(info_without_counterfactual, work)
    copy_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    interference_0 = np.asarray(env.info()["interference_uav_uav"], dtype=np.float64).sum(axis=1)
    extract_ms = (time.perf_counter() - started) * 1000.0
    sinr_0 = np.array(snap["last_sinr"], copy=True)
    progress_0 = np.array(snap["progress"], copy=True)
    pos_0 = np.array(snap["uav_positions"], copy=True)
    assignment_0 = np.array(snap["assignment"], copy=True)
    buoy_pos = np.array(env.buoy_positions, copy=True)
    step_ms = 0.0
    responses = []
    positions = []
    queues = []
    progresses = []
    sinrs = []
    rates = []
    interferences = []
    payloads = []
    served = []
    assignments_1 = []
    for index, assignment in enumerate(assignments):
        _restore(work, snap)
        started = time.perf_counter()
        info = work.step(np.asarray(actions[index], dtype=np.float64), assignment)
        step_ms += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        active = {int(uav) for uav, _buoy in active_links(work)}
        in_range = np.asarray(
            [float(work._in_range(uav, int(buoy))) if int(buoy) >= 0 else 0.0 for uav, buoy in enumerate(assignment)],
            dtype=np.float64,
        )
        service_active = np.asarray([float(uav in active) for uav in range(work.config.num_uavs)], dtype=np.float64)
        interference_1 = np.asarray(info["interference_uav_uav"], dtype=np.float64).sum(axis=1)
        response = assemble_response(
            sinr_0=sinr_0,
            sinr_1=np.asarray(info["sinr"], dtype=np.float64),
            interference_0=interference_0,
            interference_1=interference_1,
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
        extract_ms += (time.perf_counter() - started) * 1000.0
        responses.append(response)
        positions.append(np.asarray(info["uav_positions_m"], dtype=np.float64))
        queues.append(np.asarray(info["queue_blocks"], dtype=np.float64))
        progresses.append(np.asarray(info["service_progress"], dtype=np.float64))
        sinrs.append(np.asarray(info["sinr"], dtype=np.float64))
        rates.append(np.asarray(info["rate_bps"], dtype=np.float64))
        interferences.append(interference_1)
        payloads.append(float(info["cumulative_payload_bits"]))
        served.append(np.array(work.served_k, copy=True))
        assignments_1.append(np.asarray(info["assignment"], dtype=np.int64))
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "response": np.stack(responses),
        "positions": np.stack(positions),
        "queues": np.stack(queues),
        "progress": np.stack(progresses),
        "sinr": np.stack(sinrs),
        "rate": np.stack(rates),
        "interference": np.stack(interferences),
        "payload": np.asarray(payloads, dtype=np.float64),
        "served": np.stack(served),
        "assignment_1": np.stack(assignments_1),
        "timing": {
            "state_copy_ms": copy_ms,
            "actor_observation_ms": 0.0,
            "mappo_forward_ms": 0.0,
            "residual_lock_ms": 0.0,
            "safety_projection_ms": 0.0,
            "executor_act_ms": 0.0,
            "env_step_ms": step_ms,
            "response_assembly_ms": extract_ms,
        },
    }


def shipped_h1(env: L3BEnv, assignments: np.ndarray, actor, movement: np.ndarray) -> dict:
    """Original executor.act, then the slim one-step path. Batch GEMM is not used."""
    from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor

    executor = LockedExecutor(actor)
    actions = np.zeros((len(assignments), env.config.num_uavs, 2), dtype=np.float64)
    desired = np.zeros_like(actions)
    act_ms = 0.0
    for index, assignment in enumerate(assignments):
        option = _option(env, assignment)
        started = time.perf_counter()
        executed, wanted, _delta = executor.act(env, option, np.array(movement, copy=True))
        act_ms += (time.perf_counter() - started) * 1000.0
        actions[index] = np.asarray(executed, dtype=np.float64)
        desired[index] = np.asarray(wanted, dtype=np.float64)
    packed = slim_steps(env, assignments, actions)
    packed["desired"] = desired
    packed["timing"]["executor_act_ms"] = act_ms
    return packed


def optimized_h1(env: L3BEnv, assignments: np.ndarray, actor, movement: np.ndarray, executor) -> dict:
    """Reuse one clone. Skip the counterfactual rate grid. Keep interference inside each assignment."""
    import copy

    started = time.perf_counter()
    snap = _capture(env)
    work = copy.deepcopy(env)
    work.info = types.MethodType(info_without_counterfactual, work)
    copy_ms = (time.perf_counter() - started) * 1000.0
    actions, parts = batched_executor_act(env, assignments, actor, movement, executor)
    mus = parts.pop("mu")
    desired = parts.pop("desired")
    projected = parts.pop("projected")
    status = parts.pop("constraint_status")
    interference_0 = interference_matrix(env).sum(axis=1)
    sinr_0 = np.array(snap["last_sinr"], copy=True)
    progress_0 = np.array(snap["progress"], copy=True)
    pos_0 = np.array(snap["uav_positions"], copy=True)
    assignment_0 = np.array(snap["assignment"], copy=True)
    buoy_pos = np.array(env.buoy_positions, copy=True)
    step_ms = 0.0
    extract_ms = 0.0
    responses = []
    positions = []
    queues = []
    progresses = []
    sinrs = []
    rates = []
    interferences = []
    payloads = []
    served = []
    assignments_1 = []
    for index, assignment in enumerate(assignments):
        _restore(work, snap)
        started = time.perf_counter()
        info = work.step(actions[index], assignment)
        step_ms += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        active = {int(uav) for uav, _buoy in active_links(work)}
        in_range = np.asarray(
            [float(work._in_range(uav, int(buoy))) if int(buoy) >= 0 else 0.0 for uav, buoy in enumerate(assignment)],
            dtype=np.float64,
        )
        service_active = np.asarray([float(uav in active) for uav in range(work.config.num_uavs)], dtype=np.float64)
        interference_1 = np.asarray(info["interference_uav_uav"], dtype=np.float64).sum(axis=1)
        response = assemble_response(
            sinr_0=sinr_0,
            sinr_1=np.asarray(info["sinr"], dtype=np.float64),
            interference_0=interference_0,
            interference_1=interference_1,
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
        extract_ms += (time.perf_counter() - started) * 1000.0
        responses.append(response)
        positions.append(np.asarray(info["uav_positions_m"], dtype=np.float64))
        queues.append(np.asarray(info["queue_blocks"], dtype=np.float64))
        progresses.append(np.asarray(info["service_progress"], dtype=np.float64))
        sinrs.append(np.asarray(info["sinr"], dtype=np.float64))
        rates.append(np.asarray(info["rate_bps"], dtype=np.float64))
        interferences.append(interference_1)
        payloads.append(float(info["cumulative_payload_bits"]))
        served.append(np.array(work.served_k, copy=True))
        assignments_1.append(np.asarray(info["assignment"], dtype=np.int64))
    parts.update(
        {
            "state_copy_ms": copy_ms,
            "env_step_ms": step_ms,
            "response_assembly_ms": extract_ms,
        }
    )
    return {
        "actions": actions,
        "mu": mus,
        "desired": desired,
        "projected": projected,
        "constraint_status": status,
        "response": np.stack(responses),
        "positions": np.stack(positions),
        "queues": np.stack(queues),
        "progress": np.stack(progresses),
        "sinr": np.stack(sinrs),
        "rate": np.stack(rates),
        "interference": np.stack(interferences),
        "payload": np.asarray(payloads, dtype=np.float64),
        "served": np.stack(served),
        "assignment_1": np.stack(assignments_1),
        "timing": parts,
    }


__all__ = ["batched_executor_act", "info_without_counterfactual", "observation_gap", "optimized_h1"]
