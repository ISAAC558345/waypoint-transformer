"""Zero-arrival and real-arrival labels. The horizon is an argument, not a hidden constant."""
from __future__ import annotations

import copy

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.value import _prepare, _zero_arrivals
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import deterministic_move
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector


def _snap_equal(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    for key, value in left.items():
        other = right[key]
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, np.asarray(other)):
                return False
        elif value != other:
            return False
    return True


def expected_generated_bits(env_snap: dict, scenario_id: int, time_step: int, horizon: int) -> float:
    """Sum the formal arrival function over the label window. Candidates do not enter this sum."""
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    restore_state(env, env_snap)
    total = 0.0
    block = float(env.block_bits)
    for offset in range(int(horizon)):
        total += float(np.sum(env.arrivals_at(int(time_step) + int(offset)))) * block
    return float(total)


def score_label(
    env_snap: dict,
    ledger_snap: dict,
    actor,
    assignment: np.ndarray,
    *,
    scenario_id: int,
    zero_arrivals: bool,
    horizon: int,
) -> dict:
    """Score one hold. Arrivals are removed only when the zero-arrival arm asks for it."""
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    restore_state(env, env_snap)
    held = capture_state(env)
    arrival_fn = getattr(env.arrivals, "__func__", env.arrivals)
    arrival_at = getattr(env.arrivals_at, "__func__", env.arrivals_at)
    if zero_arrivals:
        scratch = _prepare(env, blind_interference=False)
    else:
        scratch = copy.deepcopy(env)
        install_light_info(scratch)
    if getattr(env.arrivals, "__func__", env.arrivals) is not arrival_fn or getattr(env.arrivals_at, "__func__", env.arrivals_at) is not arrival_at:
        raise RuntimeError("label copy replaced arrivals on the saved environment")
    if not _snap_equal(capture_state(env), held):
        raise RuntimeError("label copy changed the saved environment")
    baseline = capture_state(scratch)
    ledger = AdmissionAgeLedger(int(scratch.config.num_buoys))
    ledger.restore(copy.deepcopy(ledger_snap))
    ledger.assert_matches(scratch.q_bits, float(scratch.tol_bits))
    key = np.asarray(assignment, dtype=np.int64).reshape(-1)
    restore_state(scratch, baseline)
    if zero_arrivals:
        _zero_arrivals(scratch)
    ledger.restore(copy.deepcopy(ledger_snap))
    option = make_option(scratch, key)
    last = np.zeros((scratch.config.num_uavs, 2), dtype=np.float64)
    payload = 0.0
    urgent = 0.0
    generated = 0.0
    admitted = 0.0
    dropped = 0.0
    snap_bits = 0.0
    served = np.zeros(scratch.config.num_buoys, dtype=np.float64)
    initial_queue = float(np.sum(scratch.q_bits))
    max_gap = 0.0
    max_account = 0.0
    steps = 0
    while steps < int(horizon) and int(scratch.t) < int(scratch.config.episode_horizon):
        t_before = int(scratch.t)
        queue_before = np.asarray(scratch.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(scratch.cumulative_snap_bits, dtype=np.float64).copy()
        view = ledger.view(t_before, queue_before, float(scratch.tol_bits))
        movement = deterministic_move(scratch, actor, option, last)
        info = scratch.step(movement, option.association)
        if not np.array_equal(option.association, key):
            raise RuntimeError("assignment changed during the label rollout")
        served_now = np.asarray(info["served_bits"], dtype=np.float64)
        step_snap = snap_vector(scratch, snap_before, info)
        payload += float(info["payload_increment_bits"])
        urgent += float(np.sum(np.asarray(view["urgency"], dtype=np.float64) * served_now))
        generated += float(np.sum(info["generated_arrival_bits"]))
        admitted += float(np.sum(info["admitted_arrival_bits"]))
        dropped += float(np.sum(info["dropped_bits_step"]))
        snap_bits += float(np.sum(step_snap))
        served += served_now
        ledger.apply_transition(
            served_now,
            info["generated_arrival_bits"],
            info["admitted_arrival_bits"],
            info["dropped_bits_step"],
            step_snap,
            t_before,
            float(scratch.tol_bits),
        )
        gap = ledger.assert_matches(scratch.q_bits, float(scratch.tol_bits))
        max_gap = max(max_gap, float(np.max(np.abs(gap))))
        max_account = max(max_account, float(np.max(np.abs(scratch.conservation_error_bits()))))
        option.remaining_hold -= 1
        last = movement
        steps += 1
    if zero_arrivals and float(generated) > float(scratch.tol_bits):
        raise RuntimeError("zero-arrival label admitted arrivals")
    if steps != int(horizon):
        raise RuntimeError(f"label rollout finished {steps} steps, not the requested horizon {int(horizon)}")
    if not _snap_equal(capture_state(env), held):
        raise RuntimeError("label rollout changed the saved environment")
    final_view = ledger.view(int(scratch.t), np.asarray(scratch.q_bits, dtype=np.float64), float(scratch.tol_bits))
    served_gap = abs(float(payload) - float(np.sum(served)))
    return {
        "payload_bits": float(payload),
        "urgent_service_bits": float(urgent),
        "served_bits": [float(item) for item in served],
        "dropped_bits": float(dropped),
        "admitted_bits": float(admitted),
        "final_queue_bits": float(np.sum(scratch.q_bits)),
        "initial_queue_bits": float(initial_queue),
        "generated_bits": float(generated),
        "snap_bits": float(snap_bits),
        "final_age_weighted_backlog_bits": float(np.sum(final_view["age_weighted_backlog"])),
        "ledger_gap_bits": float(max_gap),
        "accounting_error_bits": float(max_account),
        "payload_served_gap_bits": float(served_gap),
        "steps": int(steps),
    }
