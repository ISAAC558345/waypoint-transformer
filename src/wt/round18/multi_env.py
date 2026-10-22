"""Deterministic minimal multi-UAV maritime environment for Round-18.

The M=1 branch deliberately delegates to :class:`wt.maritime.env.MaritimeEnv`.
The M>1 branch adds only shared queues, exclusive matching, and hard spatial
separation around the project's existing static channel, arrival, payload,
energy, and reward primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Optional

import numpy as np

from wt.maritime.arrival import materialize_arrival_trace
from wt.maritime.channel import compute_channel_metrics, theoretical_max_rates
from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import ContinuousBox, MaritimeEnv


EPSILON = 1.0e-12
POSITION_DIM = 2
UAV_FEATURE_DIM = 7
BUOY_FEATURE_DIM = 5
PAIR_FEATURE_DIM = 3


def _channel_kwargs(config: MaritimeEnvConfig) -> dict[str, float]:
    return {
        "uav_altitude_m": float(config.uav_altitude),
        "total_bandwidth_hz": float(config.total_bandwidth_hz),
        "buoy_tx_power_w": float(config.buoy_tx_power_w),
        "reference_channel_gain": float(config.reference_channel_gain),
        "pathloss_exponent": float(config.pathloss_exponent),
        "noise_power_w": float(config.noise_power_w),
    }


def _check_finite_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result.copy()


def process_joint_action(
    action: Any, num_uavs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the existing component clip and radial projection per UAV."""
    raw = _check_finite_matrix(action, (int(num_uavs), POSITION_DIM), "joint action")
    component_clipped = np.clip(raw, -1.0, 1.0)
    component_adjusted = np.any(component_clipped != raw, axis=1)
    norms = np.linalg.norm(component_clipped, axis=1)
    norm_adjusted = norms > 1.0
    applied = component_clipped.copy()
    if np.any(norm_adjusted):
        applied[norm_adjusted] /= norms[norm_adjusted, None]
    return raw, component_clipped, applied, component_adjusted, norm_adjusted


def minimum_pairwise_distance(positions_m: np.ndarray) -> float:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != POSITION_DIM:
        raise ValueError("positions_m must have shape (M, 2)")
    if positions.shape[0] < 2:
        return math.inf
    distances = [
        float(np.linalg.norm(positions[j] - positions[i]))
        for i in range(positions.shape[0])
        for j in range(i + 1, positions.shape[0])
    ]
    return min(distances)


def project_uav_separation(
    positions_m: np.ndarray,
    min_separation_m: float,
    area_width: float,
    area_height: float,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """Enforce hard separation with equal-and-opposite pair corrections.

    For the measure-zero exact-coincidence case, a fixed global-axis tie rule
    supplies equal and opposite corrections.  It has no magnitude or reward
    preference for either UAV; non-coincident primary M=2 projections are
    permutation-equivariant.
    """
    positions = np.asarray(positions_m, dtype=np.float64).copy()
    if positions.ndim != 2 or positions.shape[1] != POSITION_DIM:
        raise ValueError("positions_m must have shape (M, 2)")
    if min_separation_m < 0.0:
        raise ValueError("min_separation_m must be nonnegative")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    corrections = np.zeros(positions.shape[0], dtype=np.float64)
    raw_risk_pairs = 0
    if positions.shape[0] < 2 or min_separation_m <= 0.0:
        return positions, raw_risk_pairs, corrections, minimum_pairwise_distance(positions)
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            if float(np.linalg.norm(positions[j] - positions[i])) < min_separation_m - 1.0e-9:
                raw_risk_pairs += 1
    # The finite repeat count is sufficient for the small Round-18 setting and
    # avoids an unbounded projection loop for arbitrary future M.
    for _ in range(max(4, 4 * positions.shape[0])):
        changed = False
        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                delta = positions[j] - positions[i]
                distance = float(np.linalg.norm(delta))
                if distance >= min_separation_m - 1.0e-9:
                    continue
                axis = (
                    delta / distance
                    if distance > EPSILON
                    else np.array((1.0, 0.0), dtype=np.float64)
                )
                correction = 0.5 * (min_separation_m - distance) * axis
                positions[i] -= correction
                positions[j] += correction
                positions = np.clip(
                    positions,
                    np.zeros(POSITION_DIM, dtype=np.float64),
                    np.array((area_width, area_height), dtype=np.float64),
                )
                corrections[i] += float(np.linalg.norm(correction))
                corrections[j] += float(np.linalg.norm(correction))
                changed = True
        if not changed:
            break
    minimum = minimum_pairwise_distance(positions)
    if minimum < min_separation_m - 1.0e-6:
        raise RuntimeError("hard UAV separation projection did not converge")
    return positions, raw_risk_pairs, corrections, minimum


def rate_matrix(
    uav_positions_m: np.ndarray,
    buoy_positions_m: np.ndarray,
    config: MaritimeEnvConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the existing static per-UAV/per-buoy rates and horizontal range."""
    uavs = np.asarray(uav_positions_m, dtype=np.float64)
    buoys = np.asarray(buoy_positions_m, dtype=np.float64)
    if uavs.ndim != 2 or uavs.shape[1] != POSITION_DIM:
        raise ValueError("uav_positions_m must have shape (M, 2)")
    if buoys.shape != (int(config.num_buoys), POSITION_DIM):
        raise ValueError("buoy_positions_m must have shape (num_buoys, 2)")
    rates, ranges = [], []
    for position in uavs:
        metrics = compute_channel_metrics(position, buoys, **_channel_kwargs(config))
        rates.append(np.asarray(metrics.rates_bps, dtype=np.float64))
        ranges.append(np.asarray(metrics.horizontal_distances_m, dtype=np.float64))
    result_rates = np.vstack(rates)
    result_ranges = np.vstack(ranges)
    if not np.all(np.isfinite(result_rates)) or np.any(result_rates < 0.0):
        raise RuntimeError("multi-UAV rate matrix is not finite and nonnegative")
    return result_rates, result_ranges


def deterministic_matching(
    queue_bits: np.ndarray,
    rates_bps: np.ndarray,
    queue_capacity_bits: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the frozen exhaustive current-state injective assignment."""
    queues = np.asarray(queue_bits, dtype=np.float64)
    rates = np.asarray(rates_bps, dtype=np.float64)
    if rates.ndim != 2 or queues.shape != (rates.shape[1],):
        raise ValueError("queue/rate dimensions are inconsistent")
    if not np.all(np.isfinite(queues)) or np.any(queues < 0.0):
        raise ValueError("queue_bits must be finite and nonnegative")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("rates_bps must be finite and nonnegative")
    scores = (queues[None, :] / float(queue_capacity_bits)) * rates
    m_count, k_count = rates.shape
    best_assignment: Optional[tuple[int, ...]] = None
    best_score = -math.inf
    best_cardinality = -1
    for candidate in itertools.product(range(-1, k_count), repeat=m_count):
        assigned = [index for index in candidate if index >= 0]
        if len(set(assigned)) != len(assigned):
            continue
        if any(queues[index] <= EPSILON or rates[row, index] <= EPSILON for row, index in enumerate(candidate) if index >= 0):
            continue
        score = float(sum(scores[row, index] for row, index in enumerate(candidate) if index >= 0))
        cardinality = len(assigned)
        # -1 is sorted after every real buoy index in the frozen tie rule.
        lexical = tuple(k_count if index < 0 else index for index in candidate)
        if (
            score > best_score + 1.0e-9
            or (
                math.isclose(score, best_score, rel_tol=1.0e-12, abs_tol=1.0e-9)
                and (cardinality > best_cardinality or (cardinality == best_cardinality and (best_assignment is None or lexical < tuple(k_count if index < 0 else index for index in best_assignment))))
            )
        ):
            best_assignment = tuple(candidate)
            best_score = score
            best_cardinality = cardinality
    if best_assignment is None:
        raise RuntimeError("matching solver found no feasible assignment")
    assignment = np.asarray(best_assignment, dtype=np.int64)
    matched_by_buoy = np.full(k_count, -1, dtype=np.int64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            matched_by_buoy[buoy] = uav
    return assignment, matched_by_buoy, scores


def expected_arrival_bits(config: MaritimeEnvConfig) -> np.ndarray:
    """Causal arrival expectation; it never accesses a realized future trace."""
    if config.arrival_mode == "constant":
        return np.full(config.num_buoys, float(config.constant_arrival_bits_per_step), dtype=np.float64)
    if config.arrival_mode == "poisson":
        value = float(config.poisson_arrival_packets_per_second * config.delta_t * config.arrival_packet_size_bits)
        return np.full(config.num_buoys, value, dtype=np.float64)
    raise ValueError("causal multi-UAV planner does not support trace-mode arrivals")


@dataclass(frozen=True)
class ShadowStep:
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    remaining_energy: np.ndarray
    queue_bits: np.ndarray
    rates_bps: np.ndarray
    horizontal_distances_m: np.ndarray
    assignment: np.ndarray
    matched_uav_by_buoy: np.ndarray
    matching_scores: np.ndarray
    link_capacity_bits: np.ndarray
    payload_bits: np.ndarray
    payload_by_uav_bits: np.ndarray
    arrival_bits: np.ndarray
    dropped_arrival_bits: np.ndarray
    boundary_violation: np.ndarray
    safety_raw_risk_pairs: int
    safety_correction_m: np.ndarray
    min_uav_distance_m: float
    movement_distance_m: np.ndarray
    applied_movement_distance_m: np.ndarray
    energy_cost: np.ndarray
    raw_action: np.ndarray
    component_clipped_action: np.ndarray
    normalized_applied_action: np.ndarray
    action_component_clipped: np.ndarray
    action_norm_projected: np.ndarray


def shadow_step(
    *,
    positions_m: np.ndarray,
    remaining_energy: np.ndarray,
    queue_bits: np.ndarray,
    buoy_positions_m: np.ndarray,
    action: np.ndarray,
    arrival_bits: np.ndarray,
    config: MaritimeEnvConfig,
) -> ShadowStep:
    """One joint current-state transition shared by environment and planner."""
    count = int(config.num_uavs)
    previous = _check_finite_matrix(positions_m, (count, POSITION_DIM), "positions_m")
    energy = _check_finite_matrix(np.asarray(remaining_energy).reshape(-1, 1), (count, 1), "remaining_energy").reshape(-1)
    queues = np.asarray(queue_bits, dtype=np.float64).copy()
    arrivals = np.asarray(arrival_bits, dtype=np.float64).copy()
    if queues.shape != (config.num_buoys,) or arrivals.shape != (config.num_buoys,):
        raise ValueError("queue and arrival vectors must have num_buoys entries")
    raw, component, normalized, component_adjusted, norm_adjusted = process_joint_action(action, count)
    maximum_step = float(config.max_speed * config.delta_t)
    displacement = normalized * maximum_step
    movement = np.linalg.norm(displacement, axis=1)
    candidate = previous + displacement
    upper = np.array((config.area_width, config.area_height), dtype=np.float64)
    boundary = np.any(candidate < 0.0, axis=1) | np.any(candidate > upper, axis=1)
    bounded = np.clip(candidate, 0.0, upper)
    safe, raw_risk, safety_correction, minimum_distance = project_uav_separation(
        bounded,
        float(config.multi_uav_min_separation_m),
        float(config.area_width),
        float(config.area_height),
    )
    applied_displacement = safe - previous
    velocities = applied_displacement / float(config.delta_t)
    applied_movement = np.linalg.norm(applied_displacement, axis=1)
    energy_cost = float(config.hover_energy_per_step) + float(config.move_energy_per_meter) * movement
    next_energy = np.maximum(0.0, energy - energy_cost)
    rates, ranges = rate_matrix(safe, buoy_positions_m, config)
    assignment, matched_by_buoy, scores = deterministic_matching(queues, rates, float(config.queue_capacity_bits))
    capacity = np.zeros(config.num_buoys, dtype=np.float64)
    payload_by_uav = np.zeros(count, dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            capacity[buoy] = rates[uav, buoy] * float(config.delta_t)
            payload_by_uav[uav] = min(queues[buoy], capacity[buoy])
    payload = np.minimum(queues, capacity)
    preclip = queues - payload + arrivals
    dropped = np.maximum(preclip - float(config.queue_capacity_bits), 0.0)
    next_queue = np.clip(preclip, 0.0, float(config.queue_capacity_bits))
    residual = next_queue - queues - arrivals + payload + dropped
    if float(np.max(np.abs(residual))) > 1.0e-6:
        raise RuntimeError("multi-UAV queue update violates bit conservation")
    if np.any(payload > queues + 1.0e-6) or np.any(payload > capacity + 1.0e-6):
        raise RuntimeError("multi-UAV payload violates queue or capacity bounds")
    return ShadowStep(
        positions_m=safe, velocities_mps=velocities, remaining_energy=next_energy,
        queue_bits=next_queue, rates_bps=rates, horizontal_distances_m=ranges,
        assignment=assignment, matched_uav_by_buoy=matched_by_buoy,
        matching_scores=scores, link_capacity_bits=capacity, payload_bits=payload,
        payload_by_uav_bits=payload_by_uav, arrival_bits=arrivals, dropped_arrival_bits=dropped,
        boundary_violation=boundary, safety_raw_risk_pairs=raw_risk,
        safety_correction_m=safety_correction, min_uav_distance_m=minimum_distance,
        movement_distance_m=movement, applied_movement_distance_m=applied_movement,
        energy_cost=energy_cost, raw_action=raw, component_clipped_action=component,
        normalized_applied_action=normalized, action_component_clipped=component_adjusted,
        action_norm_projected=norm_adjusted,
    )


class MultiUAVMaritimeEnv:
    """Configurable multi-UAV task environment with exact M=1 delegation."""

    def __init__(
        self,
        config: Optional[MaritimeEnvConfig] = None,
        *,
        buoy_positions_m: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config if config is not None else MaritimeEnvConfig()
        if not isinstance(self.config, MaritimeEnvConfig):
            raise TypeError("config must be a MaritimeEnvConfig instance or None")
        self._single = MaritimeEnv(self.config) if self.config.num_uavs == 1 else None
        self._provided_buoy_positions = None if buoy_positions_m is None else _check_finite_matrix(buoy_positions_m, (self.config.num_buoys, POSITION_DIM), "buoy_positions_m")
        if self.num_uavs == 1:
            return
        if self.config.task_semantics_version != "v1":
            raise ValueError("multi-UAV task coupling requires task_semantics_version='v1'")
        if self.config.buoy_drift_mode != "static" or self.config.channel_variation_mode != "static":
            raise ValueError("Round-18 multi-UAV environment requires static buoy drift and static channel")
        self._prototype = MaritimeEnv(self.config)
        self._has_reset = False
        self._episode_done = False
        self._time_step = 0
        self._uav_positions_m = np.zeros((self.num_uavs, POSITION_DIM), dtype=np.float64)
        self._uav_velocities_mps = np.zeros_like(self._uav_positions_m)
        self._remaining_energy = np.full(self.num_uavs, float(self.config.initial_energy), dtype=np.float64)
        self._last_action = np.zeros_like(self._uav_positions_m)
        self._buoy_positions_m = np.zeros((self.num_buoys, POSITION_DIM), dtype=np.float64)
        self._queue_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._initial_queue_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._arrival_trace_bits = np.zeros((self.config.episode_horizon, self.num_buoys), dtype=np.float64)
        self._arrival_trace_id = ""
        self._last_arrival_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._last_link_capacity_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._last_payload_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._last_payload_by_uav_bits = np.zeros(self.num_uavs, dtype=np.float64)
        self._last_assignment = np.full(self.num_uavs, -1, dtype=np.int64)
        self._last_matching_scores = np.zeros((self.num_uavs, self.num_buoys), dtype=np.float64)
        self._last_rates_bps = np.zeros((self.num_uavs, self.num_buoys), dtype=np.float64)
        self._last_ranges_m = np.zeros((self.num_uavs, self.num_buoys), dtype=np.float64)
        self._cumulative_payload_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._cumulative_arrival_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._cumulative_dropped_bits = np.zeros(self.num_buoys, dtype=np.float64)
        self._cumulative_distance_m = np.zeros(self.num_uavs, dtype=np.float64)
        self._cumulative_energy_m = np.zeros(self.num_uavs, dtype=np.float64)
        self._last_safety_correction_m = np.zeros(self.num_uavs, dtype=np.float64)
        self._last_raw_collision_risk_pairs = 0
        max_single, max_total = theoretical_max_rates(num_buoys=self.num_buoys, **_channel_kwargs(self.config))
        self._max_single_rate_bps = float(max_single)
        self._max_team_rate_bps = float(max_total) * self.num_uavs

    @property
    def num_uavs(self) -> int:
        return int(self.config.num_uavs)

    @property
    def num_buoys(self) -> int:
        return int(self.config.num_buoys)

    @property
    def state_dim(self) -> int:
        if self._single is not None:
            return self._single.state_dim
        return self.num_uavs * UAV_FEATURE_DIM + self.num_buoys * BUOY_FEATURE_DIM + self.num_uavs * self.num_buoys * PAIR_FEATURE_DIM + self.num_uavs + self.num_buoys

    @property
    def action_dim(self) -> int:
        return self._single.action_dim if self._single is not None else self.num_uavs * POSITION_DIM

    @property
    def action_shape(self) -> tuple[int, int]:
        return (self.num_uavs, POSITION_DIM)

    @property
    def observation_space(self) -> ContinuousBox:
        if self._single is not None:
            return self._single.observation_space
        return ContinuousBox(low=np.full(self.state_dim, -1.0, dtype=np.float32), high=np.ones(self.state_dim, dtype=np.float32), shape=(self.state_dim,), dtype=np.float32)

    @property
    def action_space(self) -> ContinuousBox:
        if self._single is not None:
            return self._single.action_space
        return ContinuousBox(low=np.full(self.action_shape, -1.0, dtype=np.float64), high=np.ones(self.action_shape, dtype=np.float64), shape=self.action_shape, dtype=np.float64)

    @property
    def observation_schema(self) -> Mapping[str, Any]:
        if self._single is not None:
            return self._single.observation_schema
        uav_end = self.num_uavs * UAV_FEATURE_DIM
        buoy_end = uav_end + self.num_buoys * BUOY_FEATURE_DIM
        pair_end = buoy_end + self.num_uavs * self.num_buoys * PAIR_FEATURE_DIM
        return {
            "schema_version": "round18_multi_uav_flat_v1", "state_dim": self.state_dim,
            "num_uavs": self.num_uavs, "num_buoys": self.num_buoys,
            "offsets": {"uav_states": [0, uav_end], "buoy_states": [uav_end, buoy_end], "uav_buoy_relations": [buoy_end, pair_end], "uav_mask": [pair_end, pair_end + self.num_uavs], "buoy_mask": [pair_end + self.num_uavs, self.state_dim]},
            "uav_state_shape": [self.num_uavs, UAV_FEATURE_DIM],
            "buoy_state_shape": [self.num_buoys, BUOY_FEATURE_DIM],
            "uav_buoy_relation_shape": [self.num_uavs, self.num_buoys, PAIR_FEATURE_DIM],
            "joint_action_shape": [self.num_uavs, POSITION_DIM],
        }

    @property
    def time_step(self) -> int:
        return self._single.time_step if self._single is not None else int(self._time_step)

    @property
    def uav_positions_m(self) -> np.ndarray:
        if self._single is not None:
            return self._single.uav_position_m.reshape(1, POSITION_DIM)
        return self._uav_positions_m.copy()

    @property
    def buoy_positions_m(self) -> np.ndarray:
        return self._single.buoy_positions_m if self._single is not None else self._buoy_positions_m.copy()

    def _initial_uav_positions(self) -> np.ndarray:
        if self.config.initial_uav_positions_m is not None:
            positions = np.asarray(self.config.initial_uav_positions_m, dtype=np.float64)
        else:
            center = np.array((self.config.initial_uav_x, self.config.initial_uav_y), dtype=np.float64)
            offsets = np.linspace(-(self.num_uavs - 1), self.num_uavs - 1, self.num_uavs) * max(float(self.config.multi_uav_min_separation_m), 10.0)
            positions = np.column_stack((center[0] + offsets, np.full(self.num_uavs, center[1])))
        positions = np.clip(positions, 0.0, np.array((self.config.area_width, self.config.area_height), dtype=np.float64))
        safe, _, _, _ = project_uav_separation(positions, float(self.config.multi_uav_min_separation_m), float(self.config.area_width), float(self.config.area_height))
        return safe

    def reset(self, seed: Optional[int] = None):
        if self._single is not None:
            return self._single.reset(seed=seed)
        _, base_info = self._prototype.reset(seed=seed)
        self._time_step = 0
        self._uav_positions_m = self._initial_uav_positions()
        self._uav_velocities_mps = np.zeros_like(self._uav_positions_m)
        self._remaining_energy = np.full(self.num_uavs, float(self.config.initial_energy), dtype=np.float64)
        self._last_action = np.zeros_like(self._uav_positions_m)
        self._buoy_positions_m = (self._provided_buoy_positions.copy() if self._provided_buoy_positions is not None else np.asarray(base_info["buoy_positions_m"], dtype=np.float64).copy())
        self._initial_queue_bits = np.asarray(base_info["raw_queue_bits"], dtype=np.float64).copy()
        self._queue_bits = self._initial_queue_bits.copy()
        trace_seed = int(self.config.seed if seed is None else seed)
        self._arrival_trace_bits, self._arrival_trace_id = materialize_arrival_trace(self.config, seed=trace_seed)
        self._last_arrival_bits.fill(0.0)
        self._last_link_capacity_bits.fill(0.0)
        self._last_payload_bits.fill(0.0)
        self._last_payload_by_uav_bits.fill(0.0)
        self._last_assignment.fill(-1)
        self._last_matching_scores.fill(0.0)
        self._last_rates_bps, self._last_ranges_m = rate_matrix(self._uav_positions_m, self._buoy_positions_m, self.config)
        self._cumulative_payload_bits.fill(0.0)
        self._cumulative_arrival_bits.fill(0.0)
        self._cumulative_dropped_bits.fill(0.0)
        self._cumulative_distance_m.fill(0.0)
        self._cumulative_energy_m.fill(0.0)
        self._last_safety_correction_m.fill(0.0)
        self._last_raw_collision_risk_pairs = 0
        self._has_reset, self._episode_done = True, False
        return self._build_observation(), self._build_info()

    def _build_observation(self) -> np.ndarray:
        # A hard safety projection can add bounded corrective displacement to
        # a legal command.  Use an odd, information-preserving bounded map for
        # the structured velocity field while exposing raw velocity in info.
        velocity_normalized = self._uav_velocities_mps / (
            float(self.config.max_speed) + np.abs(self._uav_velocities_mps)
        )
        uav = np.column_stack((
            2.0 * self._uav_positions_m[:, 0] / self.config.area_width - 1.0,
            2.0 * self._uav_positions_m[:, 1] / self.config.area_height - 1.0,
            velocity_normalized[:, 0],
            velocity_normalized[:, 1],
            self._remaining_energy / self.config.initial_energy,
            np.clip(self._cumulative_distance_m / (self.config.episode_horizon * self.config.max_speed * self.config.delta_t), 0.0, 1.0),
            np.clip(self._cumulative_energy_m / self.config.initial_energy, 0.0, 1.0),
        ))
        completion = self._completion_ratio_per_buoy()
        buoy = np.column_stack((
            2.0 * self._buoy_positions_m[:, 0] / self.config.area_width - 1.0,
            2.0 * self._buoy_positions_m[:, 1] / self.config.area_height - 1.0,
            self._queue_bits / self.config.queue_capacity_bits,
            self._last_arrival_bits / (self._last_arrival_bits + self.config.queue_capacity_bits),
            completion,
        ))
        rel = self._buoy_positions_m[None, :, :] - self._uav_positions_m[:, None, :]
        relations = np.stack((rel[:, :, 0] / self.config.area_width, rel[:, :, 1] / self.config.area_height, self._last_rates_bps / self._max_single_rate_bps), axis=-1)
        observation = np.concatenate((uav.reshape(-1), buoy.reshape(-1), relations.reshape(-1), np.ones(self.num_uavs), np.ones(self.num_buoys))).astype(np.float32)
        if observation.shape != (self.state_dim,) or not np.all(np.isfinite(observation)) or not self.observation_space.contains(observation):
            raise RuntimeError("multi-UAV observation violates its declared schema")
        return observation.copy()

    def _completion_ratio_per_buoy(self) -> np.ndarray:
        offered = self._initial_queue_bits + self._cumulative_arrival_bits
        result = np.ones(self.num_buoys, dtype=np.float64)
        positive = offered > EPSILON
        result[positive] = self._cumulative_payload_bits[positive] / offered[positive]
        return np.clip(result, 0.0, 1.0)

    def _task_completion_ratio(self) -> float:
        offered = float(np.sum(self._initial_queue_bits + self._cumulative_arrival_bits))
        return 1.0 if offered <= EPSILON else float(np.clip(np.sum(self._cumulative_payload_bits) / offered, 0.0, 1.0))

    def _build_info(self) -> dict[str, Any]:
        matched = np.full(self.num_buoys, -1, dtype=np.int64)
        for uav, buoy in enumerate(self._last_assignment):
            if buoy >= 0:
                matched[buoy] = uav
        return {
            "environment_schema": "round18_multi_uav_v1", "observation_schema": self.observation_schema,
            "state_dim": self.state_dim, "action_dim": self.action_dim, "joint_action_shape": self.action_shape,
            "num_uavs": self.num_uavs, "num_buoys": self.num_buoys, "time_step": int(self._time_step),
            "uav_positions_m": self._uav_positions_m.copy(), "uav_velocity_mps": self._uav_velocities_mps.copy(),
            "remaining_energy_per_uav": self._remaining_energy.copy(), "uav_last_action": self._last_action.copy(),
            "buoy_positions_m": self._buoy_positions_m.copy(), "raw_queue_bits": self._queue_bits.copy(),
            "queue_values": self._queue_bits.copy(), "queue_capacity_bits": float(self.config.queue_capacity_bits),
            "raw_arrival_bits": self._last_arrival_bits.copy(), "rates_bps": self._last_rates_bps.copy(),
            "uav_buoy_rates_bps": self._last_rates_bps.copy(), "horizontal_distances_m": self._last_ranges_m.copy(),
            "link_capacity_bits": self._last_link_capacity_bits.copy(), "transmitted_payload_bits": self._last_payload_bits.copy(),
            "transmitted_payload_bits_by_uav": self._last_payload_by_uav_bits.copy(),
            "uav_to_buoy_assignment": self._last_assignment.copy(), "matched_uav_by_buoy": matched,
            "matching_scores": self._last_matching_scores.copy(), "valid_uav_mask": np.ones(self.num_uavs, dtype=np.bool_),
            "valid_node_mask": np.ones(self.num_buoys, dtype=np.bool_), "min_uav_distance_m": minimum_pairwise_distance(self._uav_positions_m),
            "safety_correction_m": self._last_safety_correction_m.copy(), "raw_collision_risk_pairs": int(self._last_raw_collision_risk_pairs),
            "cumulative_payload_bits": float(np.sum(self._cumulative_payload_bits)),
            "cumulative_payload_bits_per_buoy": self._cumulative_payload_bits.copy(),
            "cumulative_distance_m_per_uav": self._cumulative_distance_m.copy(),
            "cumulative_energy_consumption_per_uav": self._cumulative_energy_m.copy(),
            "arrival_trace_id": str(self._arrival_trace_id), "buoy_drift_mode": "static", "channel_variation_mode": "static",
            "primary_couplings": ("shared_queue", "exclusive_matching", "hard_separation"),
        }

    def step(self, action: np.ndarray):
        if self._single is not None:
            value = np.asarray(action)
            if value.shape == (1, POSITION_DIM):
                value = value[0]
            return self._single.step(value)
        if not self._has_reset:
            raise RuntimeError("step() called before reset(); call reset() first")
        if self._episode_done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        queue_before = self._queue_bits.copy()
        transition = shadow_step(
            positions_m=self._uav_positions_m, remaining_energy=self._remaining_energy,
            queue_bits=self._queue_bits, buoy_positions_m=self._buoy_positions_m,
            action=action, arrival_bits=self._arrival_trace_bits[self._time_step], config=self.config,
        )
        self._uav_positions_m = transition.positions_m.copy()
        self._uav_velocities_mps = transition.velocities_mps.copy()
        self._remaining_energy = transition.remaining_energy.copy()
        self._queue_bits = transition.queue_bits.copy()
        self._last_action = transition.normalized_applied_action.copy()
        self._last_arrival_bits = transition.arrival_bits.copy()
        self._last_link_capacity_bits = transition.link_capacity_bits.copy()
        self._last_payload_bits = transition.payload_bits.copy()
        self._last_payload_by_uav_bits = transition.payload_by_uav_bits.copy()
        self._last_assignment = transition.assignment.copy()
        self._last_matching_scores = transition.matching_scores.copy()
        self._last_rates_bps = transition.rates_bps.copy()
        self._last_ranges_m = transition.horizontal_distances_m.copy()
        self._last_safety_correction_m = transition.safety_correction_m.copy()
        self._last_raw_collision_risk_pairs = int(transition.safety_raw_risk_pairs)
        self._cumulative_payload_bits += transition.payload_bits
        self._cumulative_arrival_bits += transition.arrival_bits
        self._cumulative_dropped_bits += transition.dropped_arrival_bits
        self._cumulative_distance_m += transition.applied_movement_distance_m
        self._cumulative_energy_m += transition.energy_cost
        self._time_step += 1
        energy_depleted = bool(np.any(self._remaining_energy <= EPSILON))
        residual_ratio = float(np.sum(self._queue_bits) / (self.num_buoys * self.config.queue_capacity_bits))
        task_success = bool(self._task_completion_ratio() >= self.config.target_task_completion_ratio and residual_ratio <= self.config.maximum_allowed_residual_queue_ratio and float(np.min(self._remaining_energy / self.config.initial_energy)) >= self.config.minimum_terminal_energy_ratio)
        terminated = bool(energy_depleted or (self.config.terminate_on_task_success and task_success))
        truncated = bool(self._time_step >= self.config.episode_horizon)
        maximum_step_energy = float(self.config.hover_energy_per_step + self.config.move_energy_per_meter * self.config.max_speed * self.config.delta_t)
        normalized_energy = float(np.mean(transition.energy_cost / maximum_step_energy)) if maximum_step_energy > EPSILON else 0.0
        payload_utility = float(np.clip(np.sum(transition.payload_bits) / (self._max_team_rate_bps * self.config.delta_t), 0.0, 1.0))
        adjusted = transition.action_component_clipped | transition.action_norm_projected
        episode_end = terminated or truncated
        reward_components = {
            "payload_reward": float(self.config.payload_reward_weight * payload_utility),
            "energy_penalty": float(-self.config.energy_penalty_weight * normalized_energy),
            "boundary_penalty": float(-self.config.boundary_penalty if bool(np.any(transition.boundary_violation)) else 0.0),
            "invalid_action_penalty": float(-self.config.invalid_action_penalty if bool(np.any(adjusted)) else 0.0),
            "energy_depletion_penalty": float(-self.config.energy_depletion_penalty if energy_depleted else 0.0),
            "unfinished_queue_penalty": float(-self.config.unfinished_queue_penalty_weight * residual_ratio if episode_end and not task_success else 0.0),
            "task_success_bonus": float(self.config.task_success_bonus if task_success and self.config.terminate_on_task_success else 0.0),
        }
        reward = float(sum(reward_components.values()))
        info = self._build_info()
        info.update({
            "raw_action": transition.raw_action.copy(), "component_clipped_action": transition.component_clipped_action.copy(),
            "normalized_applied_action": transition.normalized_applied_action.copy(),
            "action_component_clipped": transition.action_component_clipped.copy(), "action_norm_projected": transition.action_norm_projected.copy(),
            "boundary_violation": transition.boundary_violation.copy(), "queue_before_bits": queue_before,
            "queue_conservation_residual_bits": self._queue_bits - queue_before - transition.arrival_bits + transition.payload_bits + transition.dropped_arrival_bits,
            "payload_utility": payload_utility, "normalized_energy_cost": normalized_energy,
            "energy_cost_per_uav": transition.energy_cost.copy(), "movement_distance_m_per_uav": transition.movement_distance_m.copy(),
            "applied_movement_distance_m_per_uav": transition.applied_movement_distance_m.copy(),
            "task_success": task_success, "terminated": terminated, "truncated": truncated,
            "reward_components": reward_components,
        })
        self._episode_done = bool(terminated or truncated)
        return self._build_observation(), reward, terminated, truncated, info


__all__ = [
    "BUOY_FEATURE_DIM", "EPSILON", "MultiUAVMaritimeEnv", "PAIR_FEATURE_DIM",
    "POSITION_DIM", "ShadowStep", "UAV_FEATURE_DIM", "deterministic_matching",
    "expected_arrival_bits", "minimum_pairwise_distance", "process_joint_action",
    "project_uav_separation", "rate_matrix", "shadow_step",
]
