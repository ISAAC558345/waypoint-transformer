"""Deterministic bit-accounting checks. They do not train or rank controllers."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2, apply_service_then_arrival
from wt.l3b.policies import exclusive_matching, fly_toward

from .protocol import assert_sanity_scenario


def _fresh(scenario: int = 320000) -> L3BBitQueueEnvV2:
    assert_sanity_scenario(int(scenario))
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario))
    env.q_bits[:] = 0.0
    env.cumulative_served_bits[:] = 0.0
    env.cumulative_dropped_bits[:] = 0.0
    env.cumulative_generated_arrival_bits[:] = 0.0
    env.cumulative_snap_bits[:] = 0.0
    env.completed_block_equivalent[:] = 0.0
    env.service_remainder_bits[:] = 0.0
    env.initial_q_bits = np.zeros_like(env.q_bits)
    env._sync_block_view()
    return env


def _arm(env: L3BBitQueueEnvV2, uav: int, buoy: int, q_bits: float) -> None:
    env.q_bits[int(buoy)] = float(q_bits)
    env.initial_q_bits = env.q_bits.copy()
    env.uav_positions[int(uav)] = np.asarray(env.buoy_positions[int(buoy)], dtype=np.float64)
    env._sync_block_view()


def _assign(env: L3BBitQueueEnvV2, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    for uav, buoy in pairs:
        assignment[int(uav)] = int(buoy)
        env.uav_positions[int(uav)] = np.asarray(env.buoy_positions[int(buoy)], dtype=np.float64)
    return assignment


def _rates(env: L3BBitQueueEnvV2, values: dict[int, float]) -> np.ndarray:
    rate = np.zeros(env.config.num_uavs, dtype=np.float64)
    for uav, value in values.items():
        rate[int(uav)] = float(value)
    return rate


def _close(actual: float, expected: float, tol: float) -> bool:
    return abs(float(actual) - float(expected)) <= float(tol)


def run_accounting_tests() -> dict:
    env = _fresh()
    block = float(env.block_bits)
    tol = float(env.tol_bits)
    zeros = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    rows = []

    env = _fresh()
    _arm(env, 0, 0, block)
    info = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: 0.0}))
    rows.append(
        {
            "test": "Test-1-zero-rate",
            "passed": bool(
                _close(info["payload_increment_bits"], 0.0, tol)
                and _close(float(np.sum(info["served_bits"])), 0.0, tol)
                and _close(float(env.q_bits[0]), block, tol)
            ),
        }
    )

    env = _fresh()
    _arm(env, 0, 0, 0.0)
    info = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: block}))
    rows.append(
        {
            "test": "Test-2-empty-queue",
            "passed": bool(_close(float(np.sum(info["served_bits"])), 0.0, tol) and _close(float(env.q_bits[0]), 0.0, tol)),
        }
    )

    env = _fresh()
    _arm(env, 0, 0, block)
    info = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: 0.4 * block}))
    rows.append(
        {
            "test": "Test-3-partial-transmission",
            "passed": bool(
                _close(float(env.q_bits[0]), 0.6 * block, tol)
                and _close(info["payload_increment_bits"], 0.4 * block, tol)
                and info["payload_increment_bits"] > tol
            ),
        }
    )

    env = _fresh()
    _arm(env, 0, 0, block)
    env.q_bits[1] = block
    env.initial_q_bits = env.q_bits.copy()
    first = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: 0.4 * block}))
    second = env.step(zeros, _assign(env, ((0, 1),)), rate_bps_override=_rates(env, {0: 0.0}))
    held = float(env.payload_bits)
    third = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: 0.6 * block}))
    switch_record = {
        "first_payload_increment_bits": float(first["payload_increment_bits"]),
        "payload_after_switch_away": float(held),
        "served_on_other_buoy_during_switch": float(second["served_bits"][0]),
        "third_payload_increment_bits": float(third["payload_increment_bits"]),
        "buoy_a_cumulative_served_bits": float(env.cumulative_served_bits[0]),
        "buoy_a_final_queue_bits": float(env.q_bits[0]),
        "payload_was_not_rewound": bool(float(env.payload_bits) + tol >= held),
    }
    rows.append(
        {
            "test": "Test-4-switch-persistence",
            "passed": bool(
                _close(switch_record["buoy_a_cumulative_served_bits"], block, tol)
                and _close(switch_record["buoy_a_final_queue_bits"], 0.0, tol)
                and switch_record["payload_was_not_rewound"]
                and _close(switch_record["first_payload_increment_bits"], 0.4 * block, tol)
                and _close(switch_record["served_on_other_buoy_during_switch"], 0.0, tol)
                and _close(switch_record["third_payload_increment_bits"], 0.6 * block, tol)
            ),
        }
    )

    env = _fresh()
    _arm(env, 0, 0, block)
    env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: 0.4 * block}))
    env.step(zeros, _assign(env, ((1, 0),)), rate_bps_override=_rates(env, {1: 0.6 * block}))
    cross_record = {
        "cumulative_served_bits": float(env.cumulative_served_bits[0]),
        "final_queue_bits": float(env.q_bits[0]),
        "completed_block_equivalent": float(env.completed_block_equivalent[0]),
        "remainder_owner": "buoy",
    }
    rows.append(
        {
            "test": "Test-5-cross-uav-continuation",
            "passed": bool(
                _close(cross_record["cumulative_served_bits"], block, tol)
                and _close(cross_record["final_queue_bits"], 0.0, tol)
                and _close(cross_record["completed_block_equivalent"], 1.0, tol)
            ),
        }
    )

    env = _fresh()
    _arm(env, 0, 0, 0.2 * block)
    info = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: block}))
    rows.append(
        {
            "test": "Test-6-queue-clipping",
            "passed": bool(_close(float(np.sum(info["served_bits"])), 0.2 * block, tol) and _close(float(env.q_bits[0]), 0.0, tol)),
        }
    )

    before = np.array([0.5 * block, 0.0], dtype=np.float64)
    raw = np.array([0.2 * block, 0.0], dtype=np.float64)
    arrival = np.array([block, 0.0], dtype=np.float64)
    limit = np.array([4.0 * block, 4.0 * block], dtype=np.float64)
    out = apply_service_then_arrival(before, raw, arrival, limit, np.zeros(2), block, tol)
    lhs = before + out["generated_arrival_bits"]
    rhs = out["q_after"] + out["served_bits"] + out["dropped_bits"] + out["float_snap_bits"]
    rows.append({"test": "Test-7-arrival-conservation", "passed": bool(np.max(np.abs(lhs - rhs)) <= tol)})

    before = np.full(2, 4.0 * block, dtype=np.float64)
    arrival = np.array([block, 0.5 * block], dtype=np.float64)
    out = apply_service_then_arrival(before, np.zeros(2), arrival, before.copy(), np.zeros(2), block, tol)
    lhs = before + out["generated_arrival_bits"]
    rhs = out["q_after"] + out["served_bits"] + out["dropped_bits"] + out["float_snap_bits"]
    rows.append(
        {
            "test": "Test-8-capacity-overflow",
            "passed": bool(
                np.max(np.abs(lhs - rhs)) <= tol
                and _close(float(out["admitted_arrival_bits"][0]), 0.0, tol)
                and _close(float(out["dropped_bits"][0]), block, tol)
                and _close(float(out["q_after"][0]), 4.0 * block, tol)
            ),
        }
    )

    env = _fresh()
    initial = 2.5 * block
    rate = 0.4 * block
    _arm(env, 0, 0, initial)
    served = 0.0
    for _ in range(5):
        info = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: rate}))
        served += float(info["payload_increment_bits"])
    rows.append(
        {
            "test": "Test-9-multi-step-rate-integral",
            "passed": bool(_close(served, min(initial, 5.0 * rate), tol) and _close(float(env.cumulative_served_bits[0]), served, tol)),
        }
    )

    env = _fresh()
    _arm(env, 0, 0, 2.0 * block)
    increments = []
    completions = []
    for fraction in (0.3, 0.2, 0.5):
        amount = float(np.round(fraction * block))
        info = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: amount}))
        increments.append(float(info["payload_increment_bits"]))
        completions.append(float(np.sum(info["completed_block_equivalent_increment"])))
    rows.append(
        {
            "test": "Test-10-block-diagnostic",
            "passed": bool(
                _close(increments[0], np.round(0.3 * block), tol)
                and _close(increments[1], np.round(0.2 * block), tol)
                and _close(increments[2], np.round(0.5 * block), tol)
                and _close(sum(completions), 1.0, tol)
                and completions[0] == 0.0
                and completions[1] == 0.0
            ),
        }
    )

    env = _fresh()
    _arm(env, 0, 0, 2.0 * block)
    info = env.step(zeros, _assign(env, ((0, 0),)), rate_bps_override=_rates(env, {0: 1.5 * block}))
    expected_q = 0.5 * block
    double_record = {
        "queue_before_bits": 2.0 * block,
        "served_bits": float(info["payload_increment_bits"]),
        "queue_after_bits": float(env.q_bits[0]),
        "completed_block_equivalent": float(env.completed_block_equivalent[0]),
        "extra_block_was_not_subtracted": bool(not _close(float(env.q_bits[0]), expected_q - block, tol)),
    }
    rows.append(
        {
            "test": "NO_DOUBLE_DECREMENT",
            "passed": bool(
                _close(double_record["queue_after_bits"], expected_q, tol)
                and _close(double_record["served_bits"], 1.5 * block, tol)
                and _close(double_record["completed_block_equivalent"], 1.0, tol)
                and double_record["extra_block_was_not_subtracted"]
            ),
        }
    )
    return {
        "rows": rows,
        "switch_persistence": switch_record,
        "cross_uav": cross_record,
        "no_double_decrement": double_record,
        "block_bits": block,
        "tolerance_bits": tol,
        "dtype": "float64",
    }


def run_scripted_conservation() -> list[dict]:
    from .protocol import SANITY_SCENARIOS

    rows = []
    for scenario in SANITY_SCENARIOS:
        assert_sanity_scenario(int(scenario))
        env = L3BBitQueueEnvV2()
        env.reset(seed=int(scenario))
        while int(env.t) < int(env.config.episode_horizon):
            assignment = exclusive_matching(env)
            env.step(fly_toward(env, assignment), assignment)
        error = np.abs(env.conservation_error_bits())
        payload_gap = abs(float(env.payload_bits) - float(np.sum(env.cumulative_served_bits)))
        rows.append(
            {
                "scenario_id": int(scenario),
                "max_abs_error_bits": float(np.max(error)),
                "relative_error": float(np.max(error) / env.block_bits),
                "payload_matches_served": bool(payload_gap <= env.tol_bits),
                "passed": bool(float(np.max(error)) <= env.tol_bits and payload_gap <= env.tol_bits),
                "final_payload_bits": float(env.payload_bits),
                "completed_block_equivalent": float(np.sum(env.completed_block_equivalent)),
            }
        )
    return rows
