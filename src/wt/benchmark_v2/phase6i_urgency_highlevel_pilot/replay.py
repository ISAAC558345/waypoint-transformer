"""Replay the frozen dataset path and keep an admission-age ledger beside it."""
from __future__ import annotations

import time

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.benchmark_v2.phase6d.shadow import capture_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs import _live_step
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor, _key
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import HOLD, PRIMARY_TIMES
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays, state_hash
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, AgeBucket, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.protocol import AGE_ORIGIN_RESET

from .protocol import assert_train_scenario
from .shards import ShardError, load_manifest, load_npz


def _same_state(live: dict, stored: dict) -> tuple[bool, str]:
    if set(live) != set(stored):
        return False, "keys"
    for key in live:
        if not np.array_equal(np.asarray(live[key]), np.asarray(stored[key])):
            return False, str(key)
    return True, ""


def _restore_roundtrip(env, ledger: AdmissionAgeLedger) -> bool:
    saved = ledger.capture()
    env_snap = capture_state(env)
    probe = AdmissionAgeLedger(ledger.num_buoys)
    probe.restore(saved)
    probe.buckets[0].append(AgeBucket(0, AGE_ORIGIN_RESET, 3.0))
    probe.restore(saved)
    restored = probe.capture()
    return restored == saved and _snap_equal(capture_state(env), env_snap)


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


def _choose(env):
    geo = geometry(env)
    max_rate, _meta = choose_assignment("max_rate_independent", env, np.random.default_rng(0), geo)
    return make_option(env, np.asarray(_key(max_rate), dtype=np.int64))


def replay_scenario(scenario_id: int) -> dict:
    """Replay one train scenario through t=30. Real arrivals stay on this path."""
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
            if int(env.t) in PRIMARY_TIMES:
                states.append(_capture_decision(env, ledger, scenario))
                max_gap = max(max_gap, float(states[-1]["ledger_gap_bits"]))
            if int(env.t) >= int(PRIMARY_TIMES[-1]):
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
    if [int(row["time_step"]) for row in states] != list(PRIMARY_TIMES):
        raise RuntimeError("replay did not reach every primary decision time")
    return {
        "scenario_id": scenario,
        "seconds": float(time.perf_counter() - started),
        "ledger_gap_bits": float(max_gap),
        "accounting_error_bits": float(max_account),
        "states": states,
    }


def _capture_decision(env, ledger: AdmissionAgeLedger, scenario: int) -> dict:
    arrays = state_arrays(env)
    digest = state_hash(scenario, arrays)
    manifest = load_manifest(scenario, int(env.t))
    stored, state_sha = load_npz(scenario, int(env.t), "state")
    equal, mismatch_key = _same_state(arrays, stored)
    gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
    checksum_ok = state_sha == str(manifest["checksum_state"])
    hash_ok = digest == str(manifest["state_hash"])
    if not checksum_ok:
        raise ShardError(f"state checksum mismatch at {scenario} t={int(env.t)}")
    return {
        "scenario_id": int(scenario),
        "time_step": int(env.t),
        "state_hash": digest,
        "manifest_state_hash": str(manifest["state_hash"]),
        "state_hash_match": bool(hash_ok and equal),
        "state_mismatch_key": mismatch_key,
        "checksum_state": state_sha,
        "checksum_state_match": bool(checksum_ok),
        "q_bits_max_abs_gap": float(np.max(np.abs(np.asarray(arrays["q_bits"]) - np.asarray(stored["q_bits"])))),
        "q_bits_mean": float(np.mean(arrays["q_bits"])),
        "manifest_q_bits_mean": float(manifest["q_bits_mean"]),
        "ledger_gap_bits": float(np.max(np.abs(gap))),
        "ledger_restore_independent": bool(_restore_roundtrip(env, ledger)),
        "env_snap": capture_state(env),
        "ledger_snap": ledger.capture(),
        "tol_bits": float(env.tol_bits),
        "manifest_candidate_count": int(manifest["candidate_count"]),
        "manifest_checksum_candidates": str(manifest["checksum_candidates"]),
        "manifest_illegal": int(manifest["illegal"]),
    }
