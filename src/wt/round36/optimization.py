"""Exact-only runtime optimization audits for the stopped Round-36 harness."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
import os
from time import perf_counter
from typing import Any

import numpy as np

from .harness import (
    Round36AuditState, assignment_service_matrix, audit_legacy_execute,
    direct_execute_assignment, enumerate_legal_assignments, first_action_assessment,
    first_action_assessment_with_feasibility, frozen_legacy_movement,
)


def prefix_trajectory(state: Round36AuditState, movement: np.ndarray, gamma: np.ndarray, config: Any, eta: float) -> dict[int, dict[str, Any]]:
    """One H10 path; all shorter utilities are strict prefixes of it."""
    current = direct_execute_assignment(state, movement, gamma, config, interference_overlap_factor=eta)
    utility = 0.0
    service = np.zeros((config.num_uavs, config.num_buoys), dtype=np.float64)
    result: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        if step > 1:
            action = frozen_legacy_movement(current.next_state, config)
            current = audit_legacy_execute(current.next_state, action, config, interference_overlap_factor=eta)
        utility += float(np.sum(current.transition.payload_bits))
        service += assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys)
        if step in (1, 5, 10):
            result[step] = {
                "utility": utility, "service": service.copy(), "queue": current.next_state.queue_bits.copy(),
                "positions": current.next_state.positions_m.copy(), "interference": current.transition.interference_power_w.copy(),
                "reward": float(current.reward), "assignment": current.transition.assignment.copy(),
            }
    return result


def separately_recomputed_prefixes(state: Round36AuditState, movement: np.ndarray, gamma: np.ndarray, config: Any, eta: float) -> dict[int, dict[str, Any]]:
    """The pre-reuse reference: independently roll H1, H5, and H10."""
    result: dict[int, dict[str, Any]] = {}
    for horizon in (1, 5, 10):
        current = direct_execute_assignment(state, movement, gamma, config, interference_overlap_factor=eta)
        utility = 0.0
        service = np.zeros((config.num_uavs, config.num_buoys), dtype=np.float64)
        for step in range(1, horizon + 1):
            if step > 1:
                action = frozen_legacy_movement(current.next_state, config)
                current = audit_legacy_execute(current.next_state, action, config, interference_overlap_factor=eta)
            utility += float(np.sum(current.transition.payload_bits))
            service += assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys)
        result[horizon] = {
            "utility": utility, "service": service, "queue": current.next_state.queue_bits.copy(),
            "positions": current.next_state.positions_m.copy(), "interference": current.transition.interference_power_w.copy(),
            "reward": float(current.reward), "assignment": current.transition.assignment.copy(),
        }
    return result


def equivalent_prefixes(old: dict[int, dict[str, Any]], new: dict[int, dict[str, Any]], atol: float = 1.0e-9) -> bool:
    fields = ("utility", "service", "queue", "positions", "interference", "reward", "assignment")
    return all(np.allclose(np.asarray(old[h][field]), np.asarray(new[h][field]), rtol=0.0, atol=atol) for h in (1, 5, 10) for field in fields)


def _assessment_worker(args: tuple[Round36AuditState, np.ndarray, np.ndarray, Any, float]) -> tuple[float, float, float, tuple[int, ...]]:
    state, movement, gamma, config, eta = args
    result = first_action_assessment(state, movement, gamma, config, interference_overlap_factor=eta)
    return float(result[1]["utility"]), float(result[5]["utility"]), float(result[10]["utility"]), tuple(int(x) for x in gamma)


def _feasibility_worker(args: tuple[Round36AuditState, np.ndarray, np.ndarray, Any, float]) -> tuple[float, float, float, bool, bool, bool, tuple[int, ...]]:
    state, movement, gamma, config, eta = args
    result, valid, _ = first_action_assessment_with_feasibility(state, movement, gamma, config, interference_overlap_factor=eta)
    return float(result[1]["utility"]), float(result[5]["utility"]), float(result[10]["utility"]), bool(valid[1]), bool(valid[5]), bool(valid[10]), tuple(int(x) for x in gamma)


def all_candidate_utilities(state: Round36AuditState, movement: np.ndarray, config: Any, eta: float, *, workers: int = 1, executor: str = "serial", record_continuation_infeasible: bool = False) -> list[tuple]:
    """Deterministic ordered exact enumeration; workers never share state."""
    catalogue = enumerate_legal_assignments(config.num_uavs, config.num_buoys)
    inputs = [(state, movement, gamma, config, eta) for gamma in catalogue]
    worker = _feasibility_worker if record_continuation_infeasible else _assessment_worker
    if workers <= 1 or executor == "serial":
        return [worker(item) for item in inputs]
    pool_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    with pool_class(max_workers=int(workers)) as pool:
        return list(pool.map(worker, inputs, chunksize=16))


def safe_worker_count() -> int:
    """Leave one logical core unused; cap workers to bound process memory."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def profile_candidate(state: Round36AuditState, movement: np.ndarray, gamma: np.ndarray, config: Any, eta: float) -> dict[str, float | int]:
    """Wall-clock decomposition of one exact H10 candidate trajectory."""
    times = {"first_step_direct_executor": 0.0, "continuation_movement": 0.0, "continuation_transition_and_matching": 0.0, "prefix_utility_service": 0.0}
    started = perf_counter(); current = direct_execute_assignment(state, movement, gamma, config, interference_overlap_factor=eta); times["first_step_direct_executor"] = perf_counter() - started
    utility = float(np.sum(current.transition.payload_bits)); service = assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys)
    for _ in range(2, 11):
        started = perf_counter(); action = frozen_legacy_movement(current.next_state, config); times["continuation_movement"] += perf_counter() - started
        started = perf_counter(); current = audit_legacy_execute(current.next_state, action, config, interference_overlap_factor=eta); times["continuation_transition_and_matching"] += perf_counter() - started
        started = perf_counter(); utility += float(np.sum(current.transition.payload_bits)); service += assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys); times["prefix_utility_service"] += perf_counter() - started
    times["transition_calls"] = 10
    times["utility"] = utility
    return times


__all__ = [
    "all_candidate_utilities", "equivalent_prefixes", "prefix_trajectory", "safe_worker_count",
    "separately_recomputed_prefixes", "profile_candidate",
]
