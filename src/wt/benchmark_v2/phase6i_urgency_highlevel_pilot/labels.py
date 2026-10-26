"""Age-weighted service labels on a zero-arrival copy of one frozen state."""
from __future__ import annotations

import copy
import time

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.value import _prepare, _zero_arrivals
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import deterministic_move
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import HOLD
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector

from .protocol import PAYLOAD_SCALE


def score_assignment(env_snap: dict, ledger_snap: dict, actor, assignment: np.ndarray, *, scenario_id: int) -> dict:
    """Score one assignment. Arrivals are removed only on the scratch environment."""
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    restore_state(env, env_snap)
    held = capture_state(env)
    arrival_fn = getattr(env.arrivals, "__func__", env.arrivals)
    arrival_at = getattr(env.arrivals_at, "__func__", env.arrivals_at)
    scratch = _prepare(env, blind_interference=False)
    if getattr(env.arrivals, "__func__", env.arrivals) is not arrival_fn:
        raise RuntimeError("zero-arrival copy replaced arrivals on the saved environment")
    if getattr(env.arrivals_at, "__func__", env.arrivals_at) is not arrival_at:
        raise RuntimeError("zero-arrival copy replaced arrivals_at on the saved environment")
    if not _snap_equal(capture_state(env), held):
        raise RuntimeError("zero-arrival copy changed the saved environment")
    baseline = capture_state(scratch)
    ledger = AdmissionAgeLedger(int(scratch.config.num_buoys))
    ledger.restore(copy.deepcopy(ledger_snap))
    ledger.assert_matches(scratch.q_bits, float(scratch.tol_bits))
    key = tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))
    restore_state(scratch, baseline)
    _zero_arrivals(scratch)
    ledger.restore(copy.deepcopy(ledger_snap))
    option = make_option(scratch, np.asarray(key, dtype=np.int64))
    last = np.zeros((scratch.config.num_uavs, 2), dtype=np.float64)
    payload = 0.0
    urgent = 0.0
    generated = 0.0
    served = np.zeros(scratch.config.num_buoys, dtype=np.float64)
    max_gap = 0.0
    max_account = 0.0
    oldest_start = None
    started = time.perf_counter()
    steps = 0
    while steps < int(HOLD) and int(scratch.t) < int(scratch.config.episode_horizon):
        t_before = int(scratch.t)
        queue_before = np.asarray(scratch.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(scratch.cumulative_snap_bits, dtype=np.float64).copy()
        view = ledger.view(t_before, queue_before, float(scratch.tol_bits))
        if oldest_start is None:
            oldest_start = np.asarray(view["oldest_wait_age"], dtype=np.float64).copy()
        movement = deterministic_move(scratch, actor, option, last)
        info = scratch.step(movement, option.association)
        if not np.array_equal(option.association, np.asarray(key, dtype=np.int64)):
            raise RuntimeError("assignment changed during the label rollout")
        served_now = np.asarray(info["served_bits"], dtype=np.float64)
        urgency = np.asarray(view["urgency"], dtype=np.float64)
        payload += float(info["payload_increment_bits"])
        urgent += float(np.sum(urgency * served_now))
        generated += float(np.sum(info["generated_arrival_bits"]))
        served += served_now
        snap = snap_vector(scratch, snap_before, info)
        ledger.apply_transition(
            served_now,
            info["generated_arrival_bits"],
            info["admitted_arrival_bits"],
            info["dropped_bits_step"],
            snap,
            t_before,
            float(scratch.tol_bits),
        )
        gap = ledger.assert_matches(scratch.q_bits, float(scratch.tol_bits))
        max_gap = max(max_gap, float(np.max(np.abs(gap))))
        max_account = max(max_account, float(np.max(np.abs(scratch.conservation_error_bits()))))
        option.remaining_hold -= 1
        last = movement
        steps += 1
    if float(generated) > float(scratch.tol_bits):
        raise RuntimeError("label rollout admitted arrivals that should have stayed at zero")
    if steps != int(HOLD):
        raise RuntimeError("label rollout did not finish the 15-step hold")
    if not _snap_equal(capture_state(env), held):
        raise RuntimeError("label rollout changed the saved environment")
    return {
        "payload_bits": float(payload),
        "urgent_service_bits": float(urgent),
        "payload_term": float(payload / PAYLOAD_SCALE),
        "urgent_service_term": float(urgent / PAYLOAD_SCALE),
        "ledger_gap_bits": float(max_gap),
        "accounting_error_bits": float(max_account),
        "served_bits": [float(item) for item in served],
        "generated_bits": float(generated),
        "oldest_age_start": [float(item) for item in oldest_start],
        "steps": int(steps),
        "seconds": float(time.perf_counter() - started),
    }


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
