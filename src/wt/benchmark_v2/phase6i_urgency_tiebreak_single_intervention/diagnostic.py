"""Real-arrival diagnostic. It never chooses the assignment."""
from __future__ import annotations

import copy

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import restore_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import frozen_mappo_action

PAIRING = "arrivals_at is a deterministic function of time, so both copies share the exogenous sequence"


def real_horizon(env_snap: dict, ledger_snap: dict, actor, assignment: np.ndarray, *, scenario_id: int, horizon: int) -> dict:
    """Execute one assignment on a restored copy without removing arrivals."""
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    restore_state(env, env_snap)
    ledger = AdmissionAgeLedger(int(env.config.num_buoys))
    ledger.restore(copy.deepcopy(ledger_snap))
    ledger.assert_matches(env.q_bits, float(env.tol_bits))
    option = make_option(env, np.asarray(assignment, dtype=np.int64))
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    payload = 0.0
    urgent = 0.0
    generated = 0.0
    steps = 0
    while steps < int(horizon) and int(env.t) < int(env.config.episode_horizon):
        t_before = int(env.t)
        queue_before = np.asarray(env.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
        view = ledger.view(t_before, queue_before, float(env.tol_bits))
        action = frozen_mappo_action(env, actor, option, last)
        info = env.step(action.executed, option.association)
        served = np.asarray(info["served_bits"], dtype=np.float64)
        payload += float(info["payload_increment_bits"])
        urgent += float(np.sum(np.asarray(view["urgency"], dtype=np.float64) * served))
        generated += float(np.sum(info["generated_arrival_bits"]))
        snap = snap_vector(env, snap_before, info)
        ledger.apply_transition(
            served,
            info["generated_arrival_bits"],
            info["admitted_arrival_bits"],
            info["dropped_bits_step"],
            snap,
            t_before,
            float(env.tol_bits),
        )
        ledger.assert_matches(env.q_bits, float(env.tol_bits))
        option.remaining_hold -= 1
        last = np.asarray(action.executed, dtype=np.float64).copy()
        steps += 1
    if steps != int(horizon):
        raise RuntimeError("real-arrival diagnostic did not finish the requested horizon")
    return {"payload_bits": float(payload), "urgent_service_bits": float(urgent), "generated_arrival_bits": float(generated), "steps": int(steps)}
