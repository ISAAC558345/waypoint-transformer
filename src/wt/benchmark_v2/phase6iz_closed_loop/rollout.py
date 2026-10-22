"""One 15-step frozen-MAPPO window from a saved t=15 snapshot. No planner call."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import feature_matrices, is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import REVISION_DATASET
from wt.benchmark_v2.phase6iz_closed_loop.protocol import HORIZON, RATE_ATOL
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.observation import communication_features


def _revision_state(scenario: int, time_step: int) -> dict[str, np.ndarray]:
    path = REVISION_DATASET / "validation" / f"scenario_{int(scenario)}" / f"t{int(time_step):03d}_state.npz"
    with np.load(path) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def restore_snapshot(scenario: int, time_step: int) -> tuple[L3BBitQueueEnvV2, dict]:
    """Copy the saved decision snapshot. Missing remainder does not enter served bits."""
    saved = _revision_state(scenario, time_step)
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario))
    env.uav_positions = np.asarray(saved["uav_position_m"], dtype=np.float64).copy()
    env.buoy_positions = np.asarray(saved["buoy_position_m"], dtype=np.float64).copy()
    env.q_bits = np.asarray(saved["q_bits"], dtype=np.float64).copy()
    env.queue = env.q_bits / float(env.block_bits)
    env.assignment = np.asarray(saved["association"], dtype=np.int64).copy()
    env.served_k = env.assignment.copy()
    env.progress = np.asarray(saved["service_progress"], dtype=np.float64).copy()
    env.t = int(np.asarray(saved["time_index"]).reshape(-1)[0])
    env.service_remainder_bits[:] = 0.0
    env.completed_block_equivalent[:] = 0.0
    env.cumulative_served_bits[:] = 0.0
    env.cumulative_dropped_bits[:] = 0.0
    env.cumulative_generated_arrival_bits[:] = 0.0
    env.cumulative_snap_bits[:] = 0.0
    env.legacy_progress_completions[:] = 0.0
    env.initial_q_bits = env.q_bits.copy()
    env.payload_bits = 0.0
    env.dropped = 0.0
    env.last_sinr = np.asarray(saved["current_sinr"], dtype=np.float64).copy()
    env.last_rate = np.asarray(saved["current_rate_bps"], dtype=np.float64).copy()
    rate_gap = float(np.max(np.abs(feature_matrices(env)["rate"] - np.asarray(saved["uav_buoy_hypothetical_rate"], dtype=np.float64))))
    report = {
        "time_step": int(env.t),
        "rate_gap": rate_gap,
        "rate_match": int(rate_gap <= RATE_ATOL),
        "progress": [float(item) for item in env.progress],
        "served_k_set_from_association": 1,
        "service_remainder_bits_stored": 0,
        "cumulative_payload_stored": 0,
        "arrival_rule": "deterministic arrivals_at(time); future arrivals are not zeroed",
        "policy_sampling": "deterministic mean action; no Normal sample",
        "rng_state": _rng_token(env),
    }
    if int(env.t) != int(time_step) or rate_gap > RATE_ATOL:
        raise RuntimeError(f"t={time_step} snapshot for {scenario} was not restored")
    return env, report


def _rng_token(env: L3BBitQueueEnvV2) -> str:
    return str(env.rng.bit_generator.state)


def _target_distance(env: L3BBitQueueEnvV2, assignment: np.ndarray) -> list[float]:
    distances = []
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            distances.append(-1.0)
        else:
            distances.append(float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)])))
    return distances


def execute_assignment(env: L3BBitQueueEnvV2, actor, assignment: np.ndarray) -> dict:
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1).copy()
    if not is_legal(chosen, int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("execution received an illegal assignment")
    option = make_option(env, chosen)
    if int(option.remaining_hold) != HORIZON:
        raise RuntimeError("hold is not 15")
    rng_before = _rng_token(env)
    q_before = env.q_bits.copy()
    progress_before = env.progress.copy()
    payload_before = float(env.payload_bits)
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    steps = []
    first_radius = np.full(env.config.num_uavs, -1, dtype=np.int64)
    per_uav_bits = np.zeros(env.config.num_uavs, dtype=np.float64)
    active_steps = np.zeros(env.config.num_uavs, dtype=np.int64)
    interventions = 0
    switches = 0
    for index in range(HORIZON):
        association = np.asarray(option.association, dtype=np.int64).copy()
        movement, intervened, _residual = _actor_move(env, actor, option, last)
        info = env.step(movement, association)
        if not np.array_equal(np.asarray(option.association, dtype=np.int64), association):
            switches += 1
        served = np.asarray(info["served_bits"], dtype=np.float64)
        distances = _target_distance(env, association)
        for uav, buoy in enumerate(association):
            if int(buoy) < 0:
                continue
            if first_radius[uav] < 0 and distances[uav] <= float(env.config.service_radius_m):
                first_radius[uav] = int(index)
            if float(served[int(buoy)]) > float(env.tol_bits):
                active_steps[uav] += 1
            per_uav_bits[uav] += float(served[int(buoy)])
        comm = communication_features(env)
        steps.append(
            {
                "sinr_mean": float(np.mean(info["sinr"])),
                "rate_mean": float(np.mean(info["rate_bps"])),
                "interference_mean": float(np.mean(comm["total_interference"])),
                "served_bits": float(np.sum(served)),
                "intervention": int(intervened),
                "distance": distances,
                "progress": [float(item) for item in env.progress],
                "pairwise_distance_m": float(minimum_pairwise_distance(env.uav_positions)),
            }
        )
        interventions += int(intervened)
        option.remaining_hold -= 1
        last = movement
    increments = float(sum(step["served_bits"] for step in steps))
    payload_after = float(env.payload_bits)
    error = float(np.max(np.abs(env.conservation_error_bits())))
    rng_after = _rng_token(env)
    return {
        "assignment": [int(item) for item in chosen],
        "payload_before": payload_before,
        "payload_after": payload_after,
        "delivered_bits": increments,
        "payload_matches_sum": int(abs(payload_after - payload_before - increments) <= float(env.tol_bits)),
        "completed_blocks": float(np.sum(env.completed_block_equivalent)),
        "queue_before": float(np.sum(q_before)),
        "queue_after": float(np.sum(env.q_bits)),
        "dropped_bits": float(np.sum(env.cumulative_dropped_bits)),
        "arrival_bits": float(np.sum(env.cumulative_generated_arrival_bits)),
        "conservation_error": error,
        "accounting_ok": int(error <= float(env.tol_bits)),
        "steps_executed": int(len(steps)),
        "idle_uavs": int(np.sum(chosen < 0)),
        "target_switches": int(switches),
        "projector_interventions": int(interventions),
        "active_service_steps": [int(item) for item in active_steps],
        "first_radius_step": [int(item) for item in first_radius],
        "per_uav_bits": [float(item) for item in per_uav_bits],
        "per_uav_bits_reliable": int(abs(float(np.sum(per_uav_bits)) - increments) <= float(env.tol_bits)),
        "progress_before": [float(item) for item in progress_before],
        "sinr_mean": float(np.mean([step["sinr_mean"] for step in steps])),
        "interference_mean": float(np.mean([step["interference_mean"] for step in steps])),
        "rng_unchanged": int(rng_before == rng_after),
        "series": steps,
    }


__all__ = ["execute_assignment", "restore_snapshot"]
