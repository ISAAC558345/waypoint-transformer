"""Reference rollouts on the development pool. They do not change the environment."""
from __future__ import annotations

import copy

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv, active_links

_ORIGINAL_INFO = L3BEnv.info
_ORIGINAL_STEP = L3BEnv.step


class _FastStep:
    """Skip only the post-step hypothetical graph. The physical update stays in env.step."""

    def __enter__(self):
        def _fast(self, movement, association):
            def _skipped(inner):
                inner._skip_graph = True
                try:
                    return _ORIGINAL_INFO(inner)
                finally:
                    inner._skip_graph = False

            L3BEnv.info = _skipped
            try:
                return _ORIGINAL_STEP(self, movement, association)
            finally:
                L3BEnv.info = _ORIGINAL_INFO

        L3BEnv.step = _fast
        return self

    def __exit__(self, exc_type, exc, tb):
        L3BEnv.step = _ORIGINAL_STEP
        L3BEnv.info = _ORIGINAL_INFO
        return False
from wt.l3b.policies import exclusive_matching, fly_toward
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5zp_one_shot_fresh_confirmation.episode import select_primary, select_static
from wt.lowlevel_v2.phase5zp_one_shot_fresh_confirmation.protocol import HOLD_STEPS, VALUE_SEEDS

from .protocol import assert_allowed


def _option(env: L3BEnv, assignment: np.ndarray) -> TaskOption:
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=chosen.copy(),
        decision_time=int(env.t),
        remaining_hold=int(HOLD_STEPS),
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=chosen.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def random_legal(env: L3BEnv, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(env.config.num_buoys)
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    for uav, buoy in enumerate(order[: env.config.num_uavs]):
        assignment[uav] = int(buoy)
    return assignment


def instant_rate_exclusive(env: L3BEnv) -> np.ndarray:
    """Greedy exclusive score from the current rate table. Not a deployed policy."""
    info = env.info()
    rates = np.asarray(info["rate_if_assigned"], dtype=np.float64)
    weights = rates * (np.asarray(env.queue, dtype=np.float64) > 0.0)
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    taken: set[int] = set()
    for uav in range(env.config.num_uavs):
        for buoy in np.argsort(-weights[uav], kind="stable"):
            if int(buoy) in taken or float(weights[uav, buoy]) <= 0.0:
                continue
            assignment[uav] = int(buoy)
            taken.add(int(buoy))
            break
    return assignment


def _movement(env: L3BEnv, option: TaskOption, executor: LockedExecutor | None, last_movement: np.ndarray, mode: str) -> np.ndarray:
    if mode == "fly":
        return fly_toward(env, option.association)
    assert executor is not None
    executed, desired, _delta = executor.act(env, option, last_movement)
    if executor.projector.project(env, desired, option).constraint_status != "ok":
        return np.zeros_like(desired)
    return executed


def _summarize_prefix() -> dict:
    return {
        "switches": 0,
        "abandoned_progress": 0.0,
        "service_work": 0.0,
        "rate_integral_bits": 0.0,
        "sinr_sum": 0.0,
        "rate_sum": 0.0,
        "steps": 0,
        "in_service_steps": 0,
        "completions_by_buoy": None,
    }


def run_controlled_episode(bundle: dict | None, scenario: int, controller: str, *, static_seed: int | None = None) -> dict:
    assert_allowed(int(scenario))
    env = L3BEnv()
    env.reset(seed=int(scenario))
    mode = "fly" if controller == "fly_toward" else "mappo"
    executor = None if mode == "fly" else LockedExecutor(bundle["actor"])
    rng = np.random.default_rng(int(scenario) + 1009)
    initial = float(np.sum(env.queue))
    arrivals = 0.0
    pending = 0
    option = None
    previous = np.full(env.config.num_uavs, -2, dtype=np.int64)
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    completions = np.zeros(env.config.num_buoys, dtype=np.float64)
    switches = 0
    abandoned = 0.0
    streaks = []
    open_streak = np.zeros(env.config.num_uavs, dtype=np.int64)
    open_target = np.full(env.config.num_uavs, -2, dtype=np.int64)
    rate_integral = 0.0
    sinr_sum = 0.0
    rate_sum = 0.0
    in_service = 0
    service_work = 0.0
    counterfactuals = []
    with _FastStep():
        while int(env.t) < int(env.config.episode_horizon):
            if pending <= 0:
                progress_now = np.asarray(env.progress, dtype=np.float64).copy()
                if controller == "frozen_phase5":
                    decision = select_primary(bundle, env, last_arrival, last_positions, last_movement)
                    assignment = np.asarray(decision["assignment"], dtype=np.int64)
                elif controller == "model_s":
                    decision = select_static(bundle, env, last_arrival, last_positions, int(static_seed))
                    assignment = np.asarray(decision["assignment"], dtype=np.int64)
                elif controller in {"fly_toward", "queue_rate_heuristic"}:
                    assignment = exclusive_matching(env)
                elif controller == "random_legal":
                    assignment = random_legal(env, rng)
                elif controller == "instant_rate_diagnostic":
                    assignment = instant_rate_exclusive(env)
                else:
                    raise RuntimeError(f"unknown controller {controller}")
                option = _option(env, assignment)
                if int(previous[0]) != -2:
                    changed = previous != assignment
                    switches += int(np.sum(changed))
                    abandoned += float(np.sum(progress_now[changed & (progress_now > 0.0)]))
                if controller == "frozen_phase5":
                    for item in hold_curve(env, assignment, bundle["actor"], last_movement):
                        item = dict(item)
                        item["scenario_id"] = int(scenario)
                        item["time_step"] = int(env.t)
                        counterfactuals.append(item)
                pending = int(HOLD_STEPS)
            assert option is not None
            progress_before = np.asarray(env.progress, dtype=np.float64).copy()
            served_before = np.asarray(env.served_k, dtype=np.int64).copy()
            positions_before = np.asarray(env.uav_positions, dtype=np.float64).copy()
            executed = _movement(env, option, executor, last_movement, mode)
            info = env.step(executed, option.association)
            arrivals += float(np.sum(info["arrivals_blocks"]))
            completions += np.asarray(info["step_service_blocks"], dtype=np.float64)
            rate_integral += float(np.sum(info["rate_bps"]) * float(env.config.delta_t))
            sinr_sum += float(np.mean(info["sinr"]))
            rate_sum += float(np.mean(info["rate_bps"]))
            in_service += int(np.sum(np.asarray(info["rate_bps"]) > 0.0))
            for uav in range(env.config.num_uavs):
                target = int(option.association[uav])
                if target != int(open_target[uav]):
                    if int(open_streak[uav]) > 0:
                        streaks.append(int(open_streak[uav]))
                    open_target[uav] = target
                    open_streak[uav] = 0 if target < 0 else 1
                elif target >= 0:
                    open_streak[uav] += 1
                completed_here = bool(target >= 0 and float(info["step_service_blocks"][target]) > 0.0)
                same_target = bool(target >= 0 and int(served_before[uav]) == target)
                if completed_here and same_target:
                    service_work += max(0.0, 1.0 - float(progress_before[uav]))
                elif completed_here:
                    service_work += 1.0
                elif target >= 0 and same_target:
                    service_work += max(0.0, float(env.progress[uav]) - float(progress_before[uav]))
                elif target >= 0:
                    service_work += max(0.0, float(env.progress[uav]))
            pending -= 1
            option.remaining_hold = int(pending)
            previous = np.asarray(option.association, dtype=np.int64).copy()
            last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
            last_positions = positions_before.astype(np.float32)
            last_movement = np.asarray(executed, dtype=np.float64)
    for length in open_streak:
        if int(length) > 0:
            streaks.append(int(length))
    completed = float(env.payload_bits / float(env.config.block_payload_bits))
    residual = float(np.sum(env.queue))
    dropped = float(env.dropped)
    positive = np.asarray(completions, dtype=np.float64)
    fairness = 0.0
    if float(np.sum(positive)) > 0.0:
        fairness = float((np.sum(positive) ** 2) / (positive.size * np.sum(positive ** 2)))
    return {
        "scenario_id": int(scenario),
        "controller": controller,
        "static_seed": None if static_seed is None else int(static_seed),
        "initial_blocks": initial,
        "arrival_blocks": arrivals,
        "completed_blocks": completed,
        "dropped_blocks": dropped,
        "residual_blocks": residual,
        "e0_payload_bits": float(env.payload_bits),
        "e2_blocks": float(initial + arrivals - dropped - residual),
        "rate_integral_bits": float(rate_integral),
        "mean_sinr": float(sinr_sum / max(int(env.config.episode_horizon), 1)),
        "mean_rate": float(rate_sum / max(int(env.config.episode_horizon), 1)),
        "in_service_steps": int(in_service),
        "switch_count": int(switches),
        "abandoned_progress": float(abandoned),
        "median_streak": float(np.median(streaks)) if streaks else 0.0,
        "service_work": float(service_work),
        "fairness_jain_completed_blocks": fairness,
        "active_links_definition": "in range, positive queue, no conflict",
        "counterfactuals": counterfactuals,
        "env_seed_state_used_only_for_initial_jitter": True,
    }


def hold_curve(env_snapshot: L3BEnv, assignment: np.ndarray, actor, movement: np.ndarray, horizons: tuple[int, ...] = (1, 3, 5, 10, 15)) -> list[dict]:
    """One offline clone. Checkpoints do not change the caller environment."""
    clone = copy.deepcopy(env_snapshot)
    executor = LockedExecutor(actor)
    option = _option(clone, assignment)
    last_movement = np.asarray(movement, dtype=np.float64).copy()
    payload_0 = float(clone.payload_bits)
    queue_0 = float(np.sum(clone.queue))
    dropped_0 = float(clone.dropped)
    arrivals = 0.0
    rate_integral = 0.0
    wanted = set(int(item) for item in horizons)
    rows = []
    steps = 0
    limit = max(wanted) if wanted else 0
    while steps < limit and int(clone.t) < int(clone.config.episode_horizon):
        executed = _movement(clone, option, executor, last_movement, "mappo")
        info = clone.step(executed, option.association)
        arrivals += float(np.sum(info["arrivals_blocks"]))
        rate_integral += float(np.sum(info["rate_bps"]) * float(clone.config.delta_t))
        last_movement = np.asarray(executed, dtype=np.float64)
        steps += 1
        if steps in wanted:
            completed = float((clone.payload_bits - payload_0) / float(clone.config.block_payload_bits))
            conserved = float(queue_0 + arrivals - (clone.dropped - dropped_0) - np.sum(clone.queue))
            rows.append(
                {
                    "horizon": int(steps),
                    "completed_blocks": completed,
                    "queue_reduction_blocks": conserved,
                    "conservation_error": float(conserved - completed),
                    "rate_integral_bits": float(rate_integral),
                }
            )
    return rows


def hold_counterfactual(env_snapshot: L3BEnv, assignment: np.ndarray, actor, movement: np.ndarray, horizon: int) -> dict:
    """Offline clone. The caller environment is not stepped by this function."""
    clone = copy.deepcopy(env_snapshot)
    executor = LockedExecutor(actor)
    option = _option(clone, assignment)
    last_movement = np.asarray(movement, dtype=np.float64).copy()
    payload_0 = float(clone.payload_bits)
    queue_0 = float(np.sum(clone.queue))
    dropped_0 = float(clone.dropped)
    arrivals = 0.0
    rate_integral = 0.0
    steps = 0
    while steps < int(horizon) and int(clone.t) < int(clone.config.episode_horizon):
        executed = _movement(clone, option, executor, last_movement, "mappo")
        info = clone.step(executed, option.association)
        arrivals += float(np.sum(info["arrivals_blocks"]))
        rate_integral += float(np.sum(info["rate_bps"]) * float(clone.config.delta_t))
        last_movement = np.asarray(executed, dtype=np.float64)
        steps += 1
    completed = float((clone.payload_bits - payload_0) / float(clone.config.block_payload_bits))
    conserved = float(queue_0 + arrivals - (clone.dropped - dropped_0) - np.sum(clone.queue))
    return {
        "steps": int(steps),
        "completed_blocks": completed,
        "queue_reduction_blocks": conserved,
        "conservation_error": float(conserved - completed),
        "rate_integral_bits": float(rate_integral),
    }


__all__ = [
    "VALUE_SEEDS",
    "active_links",
    "hold_counterfactual",
    "instant_rate_exclusive",
    "random_legal",
    "run_controlled_episode",
]
