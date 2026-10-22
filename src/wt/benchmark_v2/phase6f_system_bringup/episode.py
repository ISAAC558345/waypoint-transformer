"""One Benchmark-V2 episode. Candidate search and J15 stay in Phase-6E."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.benchmark_v2.phase6e.generators import build_pools, is_legal
from wt.benchmark_v2.phase6e.value import argmax_value, batch_j15
from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K, GENERATOR, HOLD
from wt.htw_phase100.contract import TaskOption
from wt.htw_phase9.const import assignment_to_indices
from wt.l3b.policies import fly_toward


@dataclass
class EpisodeResult:
    scenario_id: int
    mode: str
    complete: bool
    illegal: int
    contamination: int
    accounting_failures: int
    nan_inf: int
    hold_ok: bool
    chain_ok: bool
    payload_bits: float
    decision_count: int
    mean_candidate_count: float
    mean_j15_seconds: float
    elapsed_seconds: float
    runtime: dict
    decisions: list[dict]
    steps: list[dict]
    uav_rows: list[dict]
    buoy_rows: list[dict]
    audits: dict


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _count_nonfinite(values) -> int:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0
    return int(np.size(array) - np.sum(np.isfinite(array)))


def _rng_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_rng_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_rng_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right))
    return left == right


def _fingerprint(env: L3BBitQueueEnvV2, option: TaskOption | None) -> dict:
    snap = {
        "q_bits": env.q_bits.copy(),
        "uav_positions": env.uav_positions.copy(),
        "assignment": env.assignment.copy(),
        "progress": env.progress.copy(),
        "served_k": env.served_k.copy(),
        "payload_bits": float(env.payload_bits),
        "dropped": float(env.dropped),
        "t": int(env.t),
        "last_sinr": env.last_sinr.copy(),
        "last_rate": env.last_rate.copy(),
        "cumulative_served_bits": env.cumulative_served_bits.copy(),
        "cumulative_dropped_bits": env.cumulative_dropped_bits.copy(),
        "cumulative_generated_arrival_bits": env.cumulative_generated_arrival_bits.copy(),
        "arrivals": np.asarray(env.arrivals(), dtype=np.float64).copy(),
        "rng": copy.deepcopy(env.rng.bit_generator.state),
        "velocity_field_absent": not hasattr(env, "uav_velocity_mps"),
    }
    if option is not None:
        snap["option_association"] = np.asarray(option.association, dtype=np.int64).copy()
        snap["option_hold"] = int(option.remaining_hold)
    return snap


def _contamination(before: dict, after: dict) -> int:
    mismatches = 0
    for key, value in before.items():
        other = after[key]
        if isinstance(value, np.ndarray):
            same = bool(np.array_equal(value, other))
        elif key == "rng":
            same = _rng_equal(value, other)
        else:
            same = value == other
        mismatches += int(not same)
    return mismatches


def _make_option(assignment: tuple[int, ...], decision_time: int, env: L3BBitQueueEnvV2) -> TaskOption:
    chosen = np.asarray(assignment, dtype=np.int64)
    return TaskOption(
        targets=assignment_to_indices(chosen),
        decision_time=int(decision_time),
        remaining_hold=int(HOLD),
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=chosen.copy(),
    )


def _select_j15(env: L3BBitQueueEnvV2, option: TaskOption | None) -> dict:
    started = time.perf_counter()
    pools, _packed, illegal = build_pools(env, ((GENERATOR, CANDIDATE_K),))
    generation_seconds = time.perf_counter() - started
    pool = pools[(GENERATOR, CANDIDATE_K)]
    duplicates = 0
    for assignment in pool:
        chosen = [int(item) for item in np.asarray(assignment).reshape(-1) if int(item) >= 0]
        duplicates += int(len(chosen) != len(set(chosen)))
    horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))
    before = _fingerprint(env, option)
    values, evaluation_seconds, shadow_error = batch_j15(env, pool, horizon, blind_interference=False)
    after = _fingerprint(env, option)
    best_key, best_value = argmax_value(pool, values)
    second = None
    for assignment in pool:
        key = tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))
        if key == best_key:
            continue
        value = float(values[key])
        if second is None or value > second:
            second = value
    margin = 0.0 if second is None else float(best_value - second)
    return {
        "assignment": best_key,
        "selected_j15": float(best_value),
        "second_best_j15": float(best_value if second is None else second),
        "j15_margin": margin,
        "candidate_count": int(len(pool)),
        "illegal_assignment_count": int(illegal),
        "duplicate_target_count": int(duplicates),
        "candidate_seconds": float(generation_seconds),
        "j15_seconds": float(evaluation_seconds),
        "shadow_conservation_error": float(shadow_error),
        "shadow_state_contamination": _contamination(before, after),
        "horizon": int(horizon),
    }


def _select_max_rate(env: L3BBitQueueEnvV2) -> dict:
    started = time.perf_counter()
    chosen_array, _detail = choose_assignment(
        "max_rate_independent",
        env,
        np.random.default_rng(0),
        geometry(env),
    )
    chosen = tuple(int(item) for item in np.asarray(chosen_array, dtype=np.int64).reshape(-1))
    return {
        "assignment": chosen,
        "selected_j15": 0.0,
        "second_best_j15": 0.0,
        "j15_margin": 0.0,
        "candidate_count": 1,
        "illegal_assignment_count": int(not is_legal(np.asarray(chosen, dtype=np.int64), env.config.num_uavs, env.config.num_buoys)),
        "duplicate_target_count": 0,
        "candidate_seconds": float(time.perf_counter() - started),
        "j15_seconds": 0.0,
        "shadow_conservation_error": 0.0,
        "shadow_state_contamination": 0,
        "horizon": 0,
    }


def run_episode(scenario_id: int, mode: str) -> EpisodeResult:
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario_id))
    option: TaskOption | None = None
    pending = 0
    decision_id = -1
    decisions: list[dict] = []
    steps: list[dict] = []
    uav_rows: list[dict] = []
    buoy_rows: list[dict] = []
    audits = {"candidate": [], "shadow": [], "accounting": [], "taskoption": []}
    runtime = {
        "candidate_generation": 0.0,
        "j15_exhaustive": 0.0,
        "taskoption_construction": 0.0,
        "low_level_action": 0.0,
        "environment_step": 0.0,
    }
    illegal = 0
    contamination = 0
    accounting_failures = 0
    nan_inf = 0
    hold_ok = True
    chain_ok = True
    started = time.perf_counter()
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            decision_id += 1
            selected = _select_j15(env, option) if mode == "j15" else _select_max_rate(env)
            illegal += int(selected["illegal_assignment_count"])
            contamination += int(selected["shadow_state_contamination"])
            if float(selected["shadow_conservation_error"]) > float(env.tol_bits):
                accounting_failures += 1
            built = time.perf_counter()
            option = _make_option(selected["assignment"], int(env.t), env)
            runtime["taskoption_construction"] += time.perf_counter() - built
            pending = int(HOLD)
            option.remaining_hold = int(HOLD)
            chain_ok = chain_ok and option.association.shape == (env.config.num_uavs,)
            decisions.append({"decision_id": decision_id, "episode_step": int(env.t), **selected})
            audits["candidate"].append(selected)
            audits["shadow"].append(selected)
            runtime["candidate_generation"] += float(selected["candidate_seconds"])
            runtime["j15_exhaustive"] += float(selected["j15_seconds"])
            if contamination > 0 or accounting_failures > 0:
                break
        assert option is not None
        if int(option.remaining_hold) != int(pending):
            hold_ok = False
        hold_before = int(option.remaining_hold)
        q_before = env.q_bits.copy()
        action_started = time.perf_counter()
        movement = fly_toward(env, option.association)
        runtime["low_level_action"] += time.perf_counter() - action_started
        for uav, buoy in enumerate(option.association):
            buoy_id = int(buoy)
            if buoy_id < 0:
                chain_ok = chain_ok and float(np.linalg.norm(movement[uav])) <= 1.0e-8
                continue
            delta = env.buoy_positions[buoy_id] - env.uav_positions[uav]
            if float(np.linalg.norm(delta)) > 1.0e-6:
                chain_ok = chain_ok and float(np.dot(movement[uav], delta)) >= -1.0e-8
        step_started = time.perf_counter()
        info = env.step(movement, option.association)
        runtime["environment_step"] += time.perf_counter() - step_started
        option.remaining_hold -= 1
        pending -= 1
        if int(option.remaining_hold) != hold_before - 1:
            hold_ok = False
        error = env.conservation_error_bits()
        step_fail = int(float(np.max(np.abs(error))) > float(env.tol_bits))
        accounting_failures += step_fail
        served = np.asarray(info["served_bits"], dtype=np.float64)
        generated = np.asarray(info["generated_arrival_bits"], dtype=np.float64)
        dropped = np.asarray(info["dropped_bits_step"], dtype=np.float64)
        nan_inf += _count_nonfinite([float(env.payload_bits), float(np.sum(env.q_bits)), *served, *generated, *dropped, *info["sinr"], *info["rate_bps"]])
        step_index = int(env.t) - 1
        steps.append(
            {
                "scenario_id": int(scenario_id),
                "episode_step": step_index,
                "high_level_decision_id": decision_id,
                "candidate_count": int(decisions[-1]["candidate_count"]),
                "selected_assignment": ",".join(str(int(item)) for item in option.association),
                "selected_J15": float(decisions[-1]["selected_j15"]),
                "second_best_J15": float(decisions[-1]["second_best_j15"]),
                "J15_margin": float(decisions[-1]["j15_margin"]),
                "remaining_hold": int(option.remaining_hold),
                "cumulative_payload_bits": float(env.payload_bits),
                "total_q_bits": float(np.sum(env.q_bits)),
                "served_bits": float(np.sum(served)),
            }
        )
        audits["taskoption"].append(
            {
                "scenario_id": int(scenario_id),
                "episode_step": step_index,
                "high_level_decision_id": decision_id,
                "remaining_hold_before": hold_before,
                "remaining_hold_after": int(option.remaining_hold),
                "reset_mid_hold": 0,
            }
        )
        audits["accounting"].append(
            {
                "scenario_id": int(scenario_id),
                "episode_step": step_index,
                "max_abs_error_bits": float(np.max(np.abs(error))),
                "pass": int(step_fail == 0),
            }
        )
        for uav, buoy in enumerate(option.association):
            buoy_id = int(buoy)
            distance = -1.0 if buoy_id < 0 else float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy_id]))
            uav_rows.append(
                {
                    "scenario_id": int(scenario_id),
                    "episode_step": step_index,
                    "uav": uav,
                    "position_x": float(env.uav_positions[uav, 0]),
                    "position_y": float(env.uav_positions[uav, 1]),
                    "target": buoy_id,
                    "distance": distance,
                    "sinr": float(info["sinr"][uav]),
                    "rate": float(info["rate_bps"][uav]),
                    "served_bits": 0.0 if buoy_id < 0 else float(served[buoy_id]),
                }
            )
        for buoy in range(env.config.num_buoys):
            buoy_rows.append(
                {
                    "scenario_id": int(scenario_id),
                    "episode_step": step_index,
                    "buoy": buoy,
                    "q_bits": float(env.q_bits[buoy]),
                    "arrival_bits": float(generated[buoy]),
                    "dropped_bits": float(dropped[buoy]),
                }
            )
        if accounting_failures > 0:
            break
    elapsed = time.perf_counter() - started
    complete = int(env.t) == int(env.config.episode_horizon) and accounting_failures == 0 and contamination == 0
    j15_times = [float(row["j15_seconds"]) for row in decisions]
    counts = [int(row["candidate_count"]) for row in decisions]
    times = [int(row["episode_step"]) for row in decisions]
    hold_ok = bool(hold_ok and len(decisions) == 4 and times == [0, 15, 30, 45])
    return EpisodeResult(
        scenario_id=int(scenario_id),
        mode=mode,
        complete=bool(complete),
        illegal=int(illegal),
        contamination=int(contamination),
        accounting_failures=int(accounting_failures),
        nan_inf=int(nan_inf),
        hold_ok=hold_ok,
        chain_ok=bool(chain_ok and (mode != "j15" or all(row["candidate_count"] > 0 and row["horizon"] > 0 for row in decisions))),
        payload_bits=float(env.payload_bits),
        decision_count=int(len(decisions)),
        mean_candidate_count=float(np.mean(counts) if counts else 0.0),
        mean_j15_seconds=float(np.mean(j15_times) if j15_times else 0.0),
        elapsed_seconds=float(elapsed),
        runtime={**runtime, "total_episode": float(elapsed)},
        decisions=decisions,
        steps=steps,
        uav_rows=uav_rows,
        buoy_rows=buoy_rows,
        audits=audits,
    )


def run_active_episode(scenario_id: int, *, stop_time: int | None = None, policy_path: str | None = None, allow_holdout: bool = False) -> dict:
    """Official MAPPO loop. The historical j15 mode above still uses k=5."""
    from pathlib import Path

    from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner, run_closed_episode
    from wt.benchmark_v2.phase6iaa_formal_planner.protocol import read_policy

    policy_file = None if policy_path is None else Path(policy_path)
    return run_closed_episode(
        int(scenario_id),
        FormalPlanner(read_policy(policy_file)),
        stop_time=stop_time,
        allow_holdout=allow_holdout,
        policy_path=policy_file,
    )
