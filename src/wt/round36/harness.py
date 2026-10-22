"""Audit-only exact-transition helpers for the Round-36 protocol.

Nothing in this module is a production environment API.  The direct executor
accepts an already-validated legal assignment and mirrors the locked
Round-35 physical transition without re-running its preference search.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from wt.maritime.config import MaritimeEnvConfig
from wt.round18.multi_env import EPSILON, POSITION_DIM, deterministic_matching, process_joint_action, project_uav_separation, rate_matrix
from wt.round19.multi_env import SharedShadowStep, _assignment_candidates, assignment_link_metrics, gain_matrix
from wt.round19.planner import independent_queue_rate_action
from wt.round35.level2_env import association_candidate_mask


@dataclass(frozen=True)
class Round36AuditState:
    """Minimal current-state carrier; it deliberately contains no future trace."""

    positions_m: np.ndarray
    remaining_energy: np.ndarray
    queue_bits: np.ndarray
    buoy_positions_m: np.ndarray
    time_step: int
    arrival_bits: np.ndarray
    max_team_rate_bps: float

    def copy(self) -> "Round36AuditState":
        return Round36AuditState(
            self.positions_m.copy(), self.remaining_energy.copy(), self.queue_bits.copy(),
            self.buoy_positions_m.copy(), int(self.time_step), self.arrival_bits.copy(),
            float(self.max_team_rate_bps),
        )


@dataclass(frozen=True)
class DirectExecution:
    transition: SharedShadowStep
    next_state: Round36AuditState
    candidate_mask: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


@lru_cache(maxsize=None)
def _candidate_table(m_count: int, k_count: int) -> np.ndarray:
    return np.asarray(tuple(_assignment_candidates(m_count, k_count)), dtype=np.int64)


def enumerate_legal_assignments(num_uavs: int, num_buoys: int) -> np.ndarray:
    """Return the frozen partial-injective assignment universe (3393 for 4/8)."""
    return _candidate_table(int(num_uavs), int(num_buoys)).copy()


def _matched(assignment: np.ndarray, num_buoys: int) -> np.ndarray:
    result = np.full(int(num_buoys), -1, dtype=np.int64)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if buoy >= 0:
            result[buoy] = uav
    return result


def _validate_assignment(assignment: np.ndarray, queue_bits: np.ndarray, gains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    choices = np.asarray(assignment, dtype=np.int64)
    m_count, k_count = gains.shape
    if choices.shape != (m_count,) or np.any((choices < -1) | (choices >= k_count)):
        raise ValueError("assignment has an invalid shape or buoy index")
    selected = choices[choices >= 0]
    if len(np.unique(selected)) != len(selected):
        raise ValueError("assignment must be partial-injective")
    mask = association_candidate_mask(queue_bits, gains)
    if any(not mask[uav, buoy] for uav, buoy in enumerate(choices) if buoy >= 0):
        raise ValueError("assignment contains a masked or infeasible edge")
    return choices.copy(), mask


def frozen_legacy_movement(state: Round36AuditState, config: MaritimeEnvConfig) -> np.ndarray:
    """The frozen current-state Round-19 independent H1 movement action."""
    return independent_queue_rate_action(state.positions_m, state.queue_bits, state.buoy_positions_m, config)


def audit_legacy_assignment(queue_bits: np.ndarray, legacy_rates_bps: np.ndarray, queue_capacity_bits: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized audit-only equivalent of frozen deterministic matching.

    It is used solely to avoid repeatedly calculating every assignment's score
    in Python during counterfactual continuation.  Its equality with the
    formal decoder is regression-tested before any scientific outcome.
    """
    queues = np.asarray(queue_bits, dtype=np.float64)
    rates = np.asarray(legacy_rates_bps, dtype=np.float64)
    if rates.ndim != 2 or queues.shape != (rates.shape[1],):
        raise ValueError("queue/rate dimensions are inconsistent")
    if not (np.all(np.isfinite(queues)) and np.all(np.isfinite(rates))) or np.any(queues < 0.0) or np.any(rates < 0.0):
        raise ValueError("queue/rate arrays must be finite and nonnegative")
    m_count, k_count = rates.shape
    scores = (queues[None, :] / float(queue_capacity_bits)) * rates
    # In every registered Round-36 state all queue/rate edges are strictly
    # positive, so a maximum-weight injective assignment necessarily serves
    # every UAV.  The exact rectangular assignment solver is therefore the
    # same optimization as the frozen exhaustive search, without materializing
    # its 3393 candidates.  Degenerate masked/zero cases retain the exhaustive
    # vector path below so the adapter remains a faithful regression target.
    if k_count >= m_count and np.all(queues > EPSILON) and np.all(rates > EPSILON):
        rows, columns = linear_sum_assignment(-scores)
        assignment = np.full(m_count, -1, dtype=np.int64)
        assignment[rows] = columns
        return assignment, _matched(assignment, k_count), scores.copy()
    candidates = _candidate_table(m_count, k_count)
    active = candidates >= 0
    indices = np.where(active, candidates, 0)
    picked = scores[np.arange(m_count)[None, :], indices]
    feasible = np.all(~active | ((queues[indices] > EPSILON) & (rates[np.arange(m_count)[None, :], indices] > EPSILON)), axis=1)
    totals = np.sum(np.where(active, picked, 0.0), axis=1)
    totals[~feasible] = -np.inf
    best_score = float(np.max(totals))
    if not math.isfinite(best_score):
        raise RuntimeError("audit legacy adapter found no feasible assignment")
    tied = np.flatnonzero(np.isclose(totals, best_score, rtol=1.0e-12, atol=1.0e-9))
    cardinality = np.sum(active[tied], axis=1)
    tied = tied[cardinality == np.max(cardinality)]
    lexical = np.where(candidates[tied] < 0, k_count, candidates[tied])
    order = np.lexsort(tuple(lexical[:, column] for column in range(m_count - 1, -1, -1)))
    assignment = candidates[tied[int(order[0])]].copy()
    return assignment, _matched(assignment, k_count), scores.copy()


def _motion_terms(state: Round36AuditState, movement_action: np.ndarray, config: MaritimeEnvConfig):
    count = int(config.num_uavs)
    raw, component, normalized, component_adjusted, norm_adjusted = process_joint_action(movement_action, count)
    displacement = normalized * float(config.max_speed * config.delta_t)
    movement = np.linalg.norm(displacement, axis=1)
    candidate = state.positions_m + displacement
    upper = np.array((config.area_width, config.area_height), dtype=np.float64)
    boundary = np.any(candidate < 0.0, axis=1) | np.any(candidate > upper, axis=1)
    bounded = np.clip(candidate, 0.0, upper)
    safe, raw_risk, safety_correction, minimum_distance = project_uav_separation(
        bounded, float(config.multi_uav_min_separation_m), float(config.area_width), float(config.area_height)
    )
    applied_displacement = safe - state.positions_m
    velocities = applied_displacement / float(config.delta_t)
    applied_movement = np.linalg.norm(applied_displacement, axis=1)
    energy_cost = float(config.hover_energy_per_step) + float(config.move_energy_per_meter) * movement
    return (raw, component, normalized, component_adjusted, norm_adjusted, movement, boundary, safe,
            raw_risk, safety_correction, minimum_distance, velocities, applied_movement, energy_cost)


def direct_execute_assignment(
    state: Round36AuditState,
    movement_action: np.ndarray,
    assignment: np.ndarray,
    config: MaritimeEnvConfig,
    *,
    interference_overlap_factor: float,
) -> DirectExecution:
    """Execute an already legal assignment with frozen Round-35 physics."""
    count = int(config.num_uavs)
    (raw, component, normalized, component_adjusted, norm_adjusted, movement, boundary, safe,
     raw_risk, safety_correction, minimum_distance, velocities, applied_movement, energy_cost) = _motion_terms(state, movement_action, config)
    next_energy = np.maximum(0.0, state.remaining_energy - energy_cost)
    gains, ranges = gain_matrix(safe, state.buoy_positions_m, config)
    legacy_rates, _ = rate_matrix(safe, state.buoy_positions_m, config)
    choices, mask = _validate_assignment(assignment, state.queue_bits, gains)
    desired, interference, sinr, isr, selected_rates = assignment_link_metrics(choices, gains, config, float(interference_overlap_factor))
    rates = np.zeros_like(gains)
    capacity = np.zeros(int(config.num_buoys), dtype=np.float64)
    payload_by_uav = np.zeros(count, dtype=np.float64)
    for uav, buoy in enumerate(choices):
        if buoy >= 0:
            rates[uav, buoy] = selected_rates[uav]
            capacity[buoy] = selected_rates[uav] * float(config.delta_t)
            payload_by_uav[uav] = min(state.queue_bits[buoy], capacity[buoy])
    payload = np.minimum(state.queue_bits, capacity)
    preclip = state.queue_bits - payload + state.arrival_bits
    dropped = np.maximum(preclip - float(config.queue_capacity_bits), 0.0)
    next_queue = np.clip(preclip, 0.0, float(config.queue_capacity_bits))
    residual = next_queue - state.queue_bits - state.arrival_bits + payload + dropped
    if float(np.max(np.abs(residual))) > 1.0e-6:
        raise RuntimeError("Round-36 direct executor queue conservation failed")
    transition = SharedShadowStep(
        safe, velocities, next_energy, next_queue, rates, legacy_rates, ranges, gains,
        desired, interference, sinr, isr, choices, _matched(choices, int(config.num_buoys)),
        np.zeros_like(gains), 0, capacity, payload, payload_by_uav, state.arrival_bits.copy(),
        dropped, boundary, raw_risk, safety_correction, minimum_distance, movement,
        applied_movement, energy_cost, raw, component, normalized, component_adjusted, norm_adjusted,
    )
    next_state = Round36AuditState(
        transition.positions_m.copy(), transition.remaining_energy.copy(), transition.queue_bits.copy(),
        state.buoy_positions_m.copy(), int(state.time_step) + 1, state.arrival_bits.copy(), state.max_team_rate_bps,
    )
    energy_depleted = bool(np.any(next_energy <= EPSILON))
    truncated = bool(next_state.time_step >= int(config.episode_horizon))
    maximum_step_energy = float(config.hover_energy_per_step + config.move_energy_per_meter * config.max_speed * config.delta_t)
    normalized_energy = float(np.mean(energy_cost / maximum_step_energy)) if maximum_step_energy > EPSILON else 0.0
    payload_utility = float(np.clip(np.sum(payload) / (state.max_team_rate_bps * config.delta_t), 0.0, 1.0))
    adjusted = component_adjusted | norm_adjusted
    # At all frozen anchors and horizons this never invokes terminal-success
    # logic; retaining terminal penalties preserves exact ordinary termination.
    residual_ratio = float(np.sum(next_queue) / (int(config.num_buoys) * config.queue_capacity_bits))
    reward = float(
        config.payload_reward_weight * payload_utility
        - config.energy_penalty_weight * normalized_energy
        - (config.boundary_penalty if bool(np.any(boundary)) else 0.0)
        - (config.invalid_action_penalty if bool(np.any(adjusted)) else 0.0)
        - (config.energy_depletion_penalty if energy_depleted else 0.0)
        - (config.unfinished_queue_penalty_weight * residual_ratio if (energy_depleted or truncated) else 0.0)
    )
    return DirectExecution(transition, next_state, mask, reward, energy_depleted, truncated)


def audit_legacy_execute(
    state: Round36AuditState,
    movement_action: np.ndarray,
    config: MaritimeEnvConfig,
    *,
    interference_overlap_factor: float,
) -> DirectExecution:
    """Apply the frozen current-state legacy association via its audit adapter."""
    terms = _motion_terms(state, movement_action, config)
    moved_positions = terms[7]
    legacy_rates, _ = rate_matrix(moved_positions, state.buoy_positions_m, config)
    assignment, _, _ = audit_legacy_assignment(state.queue_bits, legacy_rates, config.queue_capacity_bits)
    return direct_execute_assignment(
        state, movement_action, assignment, config,
        interference_overlap_factor=interference_overlap_factor,
    )


def assignment_service_matrix(transition: SharedShadowStep, num_uavs: int, num_buoys: int) -> np.ndarray:
    """Per-UAV/per-buoy actual payload for frozen Task-Waypoint semantics."""
    result = np.zeros((int(num_uavs), int(num_buoys)), dtype=np.float64)
    for uav, buoy in enumerate(transition.assignment):
        if buoy >= 0:
            result[uav, buoy] = transition.payload_by_uav_bits[uav]
    return result


def first_action_assessment(
    state: Round36AuditState,
    first_movement: np.ndarray,
    first_assignment: np.ndarray,
    config: MaritimeEnvConfig,
    *,
    interference_overlap_factor: float,
    maximum_horizon: int = 10,
) -> dict[int, dict[str, np.ndarray | float]]:
    """One exact FAO candidate followed by the identical frozen continuation."""
    horizons = (1, 5, 10)
    if maximum_horizon != 10:
        raise ValueError("Round-36 frozen assessment maximum horizon is 10")
    current = direct_execute_assignment(
        state, first_movement, first_assignment, config,
        interference_overlap_factor=interference_overlap_factor,
    )
    payload_total = float(np.sum(current.transition.payload_bits))
    services = assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys)
    result: dict[int, dict[str, np.ndarray | float]] = {
        1: {"utility": payload_total, "service": services.copy(), "final_queue": current.next_state.queue_bits.copy()},
    }
    for step in range(2, maximum_horizon + 1):
        action = frozen_legacy_movement(current.next_state, config)
        current = audit_legacy_execute(
            current.next_state, action, config, interference_overlap_factor=interference_overlap_factor,
        )
        payload_total += float(np.sum(current.transition.payload_bits))
        services += assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys)
        if step in horizons:
            result[step] = {"utility": payload_total, "service": services.copy(), "final_queue": current.next_state.queue_bits.copy()}
    return result


def first_action_assessment_with_feasibility(
    state: Round36AuditState,
    first_movement: np.ndarray,
    first_assignment: np.ndarray,
    config: MaritimeEnvConfig,
    *,
    interference_overlap_factor: float,
) -> tuple[dict[int, dict[str, np.ndarray | float]], dict[int, bool], str | None]:
    """Exact FAO prefixes plus an explicit frozen-projection feasibility record.

    This does not relax the hard safety projection.  If the inherited
    continuation cannot be projected, later horizons are marked invalid and
    receive only finite storage placeholders; callers must never rank them.
    """
    result: dict[int, dict[str, np.ndarray | float]] = {}
    valid = {1: False, 5: False, 10: False}
    current = direct_execute_assignment(state, first_movement, first_assignment, config, interference_overlap_factor=interference_overlap_factor)
    utility = float(np.sum(current.transition.payload_bits))
    service = assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys)
    result[1] = {"utility": utility, "service": service.copy(), "final_queue": current.next_state.queue_bits.copy()}; valid[1] = True
    failure: str | None = None
    for step in range(2, 11):
        try:
            action = frozen_legacy_movement(current.next_state, config)
            current = audit_legacy_execute(current.next_state, action, config, interference_overlap_factor=interference_overlap_factor)
        except RuntimeError as error:
            if "hard UAV separation projection did not converge" not in str(error):
                raise
            failure = str(error); break
        utility += float(np.sum(current.transition.payload_bits)); service += assignment_service_matrix(current.transition, config.num_uavs, config.num_buoys)
        if step in valid:
            result[step] = {"utility": utility, "service": service.copy(), "final_queue": current.next_state.queue_bits.copy()}; valid[step] = True
    for horizon in valid:
        if horizon not in result:
            result[horizon] = {"utility": 0.0, "service": np.zeros((config.num_uavs, config.num_buoys), dtype=np.float64), "final_queue": np.zeros(config.num_buoys, dtype=np.float64)}
    return result, valid, failure


def formal_legacy_assignment(queue_bits: np.ndarray, legacy_rates_bps: np.ndarray, queue_capacity_bits: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit regression oracle for the frozen production decoder."""
    return deterministic_matching(queue_bits, legacy_rates_bps, queue_capacity_bits)


__all__ = [
    "DirectExecution", "Round36AuditState", "assignment_service_matrix", "audit_legacy_assignment", "audit_legacy_execute", "direct_execute_assignment", "first_action_assessment",
    "enumerate_legal_assignments", "formal_legacy_assignment", "first_action_assessment_with_feasibility", "frozen_legacy_movement",
]
