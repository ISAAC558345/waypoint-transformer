"""Replay the dataset path through t=45 without writing a shard."""
from __future__ import annotations

import time

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs import _live_step
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import HOLD, PRIMARY_TIMES
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays, state_hash
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import assert_train_scenario
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.replay import _capture_decision, _choose, _restore_roundtrip
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor

from .protocol import DECISION_TIMES


def replay_scenario(scenario_id: int) -> dict:
    """Replay one train scenario through t=45 on the MAX_RATE_INDEPENDENT path."""
    scenario = assert_train_scenario(scenario_id)
    started = time.perf_counter()
    actor = _frozen_actor()
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=scenario)
    env._phase6h_last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    ledger = AdmissionAgeLedger(int(env.config.num_buoys))
    ledger.reset_from_queue(env.q_bits)
    ledger.assert_matches(env.q_bits, float(env.tol_bits))
    option = None
    pending = 0
    states = []
    max_gap = 0.0
    max_account = 0.0
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if int(env.t) in DECISION_TIMES:
                if int(env.t) in PRIMARY_TIMES:
                    states.append(_capture_decision(env, ledger, scenario))
                    states[-1]["shard_checked"] = True
                else:
                    states.append(_capture_open(env, ledger, scenario))
                max_gap = max(max_gap, float(states[-1]["ledger_gap_bits"]))
            if int(env.t) >= int(DECISION_TIMES[-1]):
                break
            option = _choose(env)
            pending = int(HOLD)
        t_before = int(env.t)
        snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
        info = _live_step(env, actor, option, use_fly=False)
        snap = snap_vector(env, snap_before, info)
        ledger.apply_transition(
            info["served_bits"],
            info["generated_arrival_bits"],
            info["admitted_arrival_bits"],
            info["dropped_bits_step"],
            snap,
            t_before,
            float(env.tol_bits),
        )
        gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
        max_gap = max(max_gap, float(np.max(np.abs(gap))))
        max_account = max(max_account, float(np.max(np.abs(env.conservation_error_bits()))))
        pending -= 1
    found = [int(row["time_step"]) for row in states]
    if found != list(DECISION_TIMES):
        raise RuntimeError(f"replay reached {found}, not every expansion decision time")
    return {
        "scenario_id": scenario,
        "seconds": float(time.perf_counter() - started),
        "ledger_gap_bits": float(max_gap),
        "accounting_error_bits": float(max_account),
        "states": states,
    }


def _capture_open(env, ledger: AdmissionAgeLedger, scenario: int) -> dict:
    """t=45 has no 15-step shard. The protocol identity is this replay, not a frozen file."""
    arrays = state_arrays(env)
    gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
    return {
        "scenario_id": int(scenario),
        "time_step": int(env.t),
        "state_hash": state_hash(scenario, arrays),
        "manifest_state_hash": "",
        "state_hash_match": False,
        "state_mismatch_key": "",
        "checksum_state": "",
        "checksum_state_match": False,
        "shard_checked": False,
        "q_bits_max_abs_gap": 0.0,
        "q_bits_mean": float(np.mean(arrays["q_bits"])),
        "manifest_q_bits_mean": None,
        "ledger_gap_bits": float(np.max(np.abs(gap))),
        "ledger_restore_independent": bool(_restore_roundtrip(env, ledger)),
        "env_snap": capture_state(env),
        "ledger_snap": ledger.capture(),
        "tol_bits": float(env.tol_bits),
        "manifest_candidate_count": None,
        "manifest_checksum_candidates": "",
        "manifest_illegal": 0,
    }
