"""Round-19 shared-spectrum buoy-uplink extension.

The orthogonal branch delegates to the locked Round-18 environment exactly.
Shared mode keeps the same state, queue, motion, energy, reward, and safety
semantics while replacing only the selected-link rate calculation and matching
value with the documented buoy-to-UAV cross-uplink model.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Optional

import numpy as np

from wt.maritime.channel import compute_channel_metrics
from wt.maritime.config import MaritimeEnvConfig
from wt.round18.multi_env import (
    EPSILON,
    POSITION_DIM,
    MultiUAVMaritimeEnv,
    deterministic_matching,
    minimum_pairwise_distance,
    process_joint_action,
    project_uav_separation,
    rate_matrix,
)


SPECTRUM_MODES = ("orthogonal", "shared")
MATCHING_MODES = ("legacy", "joint_aware")


def _channel_kwargs(config: MaritimeEnvConfig) -> dict[str, float]:
    return {
        "uav_altitude_m": float(config.uav_altitude),
        "total_bandwidth_hz": float(config.total_bandwidth_hz),
        "buoy_tx_power_w": float(config.buoy_tx_power_w),
        "reference_channel_gain": float(config.reference_channel_gain),
        "pathloss_exponent": float(config.pathloss_exponent),
        "noise_power_w": float(config.noise_power_w),
    }


@dataclass(frozen=True)
class SharedSpectrumOptions:
    """Protocol-only options; physical configuration remains MaritimeEnvConfig."""

    spectrum_mode: str = "orthogonal"
    interference_overlap_factor: float = 0.0
    matching_mode: str = "legacy"
    conflict_isr_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.spectrum_mode not in SPECTRUM_MODES:
            raise ValueError(f"spectrum_mode must be one of {SPECTRUM_MODES}")
        if self.matching_mode not in MATCHING_MODES:
            raise ValueError(f"matching_mode must be one of {MATCHING_MODES}")
        for name in ("interference_overlap_factor", "conflict_isr_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(self.interference_overlap_factor) > 1.0:
            raise ValueError("interference_overlap_factor must not exceed 1")
        if self.spectrum_mode == "orthogonal" and float(self.interference_overlap_factor) != 0.0:
            raise ValueError("orthogonal mode requires interference_overlap_factor=0")


@dataclass(frozen=True)
class SharedShadowStep:
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    remaining_energy: np.ndarray
    queue_bits: np.ndarray
    rates_bps: np.ndarray
    legacy_rates_bps: np.ndarray
    horizontal_distances_m: np.ndarray
    channel_gains: np.ndarray
    desired_power_w: np.ndarray
    interference_power_w: np.ndarray
    sinr_linear: np.ndarray
    isr: np.ndarray
    assignment: np.ndarray
    matched_uav_by_buoy: np.ndarray
    matching_scores: np.ndarray
    matching_assignments_evaluated: int
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


def gain_matrix(
    uav_positions_m: np.ndarray,
    buoy_positions_m: np.ndarray,
    config: MaritimeEnvConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Current static path gains and horizontal ranges for every receiver."""
    positions = np.asarray(uav_positions_m, dtype=np.float64)
    buoys = np.asarray(buoy_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != POSITION_DIM:
        raise ValueError("uav_positions_m must have shape (M, 2)")
    if buoys.shape != (int(config.num_buoys), POSITION_DIM):
        raise ValueError("buoy_positions_m has an incompatible shape")
    metrics = [compute_channel_metrics(position, buoys, **_channel_kwargs(config)) for position in positions]
    gains = np.vstack([np.asarray(item.channel_gains, dtype=np.float64) for item in metrics])
    ranges = np.vstack([np.asarray(item.horizontal_distances_m, dtype=np.float64) for item in metrics])
    if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
        raise RuntimeError("shared-spectrum gains are non-finite or negative")
    return gains, ranges


def _assignment_candidates(m_count: int, k_count: int):
    for candidate in itertools.product(range(-1, k_count), repeat=m_count):
        assigned = [item for item in candidate if item >= 0]
        if len(set(assigned)) == len(assigned):
            yield candidate


def assignment_link_metrics(
    assignment: np.ndarray,
    channel_gains: np.ndarray,
    config: MaritimeEnvConfig,
    overlap_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute selected shared-link signal, interference, SINR, ISR, and rate."""
    choices = np.asarray(assignment, dtype=np.int64)
    gains = np.asarray(channel_gains, dtype=np.float64)
    m_count, k_count = gains.shape
    if choices.shape != (m_count,) or np.any((choices < -1) | (choices >= k_count)):
        raise ValueError("assignment has an invalid shape or buoy index")
    desired = np.zeros(m_count, dtype=np.float64)
    interference = np.zeros(m_count, dtype=np.float64)
    sinr = np.zeros(m_count, dtype=np.float64)
    isr = np.zeros(m_count, dtype=np.float64)
    rates = np.zeros(m_count, dtype=np.float64)
    tx_power = float(config.buoy_tx_power_w)
    noise = float(config.noise_power_w)
    bandwidth = float(config.total_bandwidth_hz)
    eta = float(overlap_factor)
    for receiver, desired_buoy in enumerate(choices):
        if desired_buoy < 0:
            continue
        desired[receiver] = tx_power * gains[receiver, desired_buoy]
        for other, other_buoy in enumerate(choices):
            if other != receiver and other_buoy >= 0:
                interference[receiver] += eta * tx_power * gains[receiver, other_buoy]
        sinr[receiver] = desired[receiver] / (noise + interference[receiver])
        isr[receiver] = interference[receiver] / desired[receiver] if desired[receiver] > EPSILON else 0.0
        rates[receiver] = bandwidth * math.log1p(sinr[receiver]) / math.log(2.0)
    return desired, interference, sinr, isr, rates


def joint_aware_matching(
    queue_bits: np.ndarray,
    channel_gains: np.ndarray,
    config: MaritimeEnvConfig,
    overlap_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Deterministic exhaustive current-only injective shared-rate matching."""
    queues = np.asarray(queue_bits, dtype=np.float64)
    gains = np.asarray(channel_gains, dtype=np.float64)
    if gains.ndim != 2 or queues.shape != (gains.shape[1],):
        raise ValueError("queue/gain dimensions are inconsistent")
    m_count, k_count = gains.shape
    best: Optional[tuple[int, ...]] = None
    best_rates = np.zeros(m_count, dtype=np.float64)
    best_score, best_cardinality, evaluated = -math.inf, -1, 0
    for candidate in _assignment_candidates(m_count, k_count):
        evaluated += 1
        feasible = all(queues[buoy] > EPSILON for buoy in candidate if buoy >= 0)
        if not feasible:
            continue
        choice = np.asarray(candidate, dtype=np.int64)
        _, _, _, _, rates = assignment_link_metrics(choice, gains, config, overlap_factor)
        score = float(sum((queues[buoy] / float(config.queue_capacity_bits)) * rates[m] for m, buoy in enumerate(candidate) if buoy >= 0))
        cardinality = int(sum(buoy >= 0 for buoy in candidate))
        lexical = tuple(k_count if buoy < 0 else buoy for buoy in candidate)
        old_lexical = tuple(k_count if buoy < 0 else buoy for buoy in best) if best is not None else None
        if score > best_score + 1.0e-9 or (math.isclose(score, best_score, rel_tol=1.0e-12, abs_tol=1.0e-9) and (cardinality > best_cardinality or (cardinality == best_cardinality and (old_lexical is None or lexical < old_lexical)))):
            best, best_rates, best_score, best_cardinality = candidate, rates, score, cardinality
    if best is None:
        raise RuntimeError("joint-aware matching found no feasible assignment")
    assignment = np.asarray(best, dtype=np.int64)
    matched = np.full(k_count, -1, dtype=np.int64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            matched[buoy] = uav
    scores = np.zeros((m_count, k_count), dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            scores[uav, buoy] = (queues[buoy] / float(config.queue_capacity_bits)) * best_rates[uav]
    return assignment, matched, scores, evaluated


def shared_shadow_step(
    *, positions_m: np.ndarray, remaining_energy: np.ndarray, queue_bits: np.ndarray,
    buoy_positions_m: np.ndarray, action: np.ndarray, arrival_bits: np.ndarray,
    config: MaritimeEnvConfig, options: SharedSpectrumOptions,
) -> SharedShadowStep:
    """One causal shared-mode transition; it receives no future trace/state."""
    if options.spectrum_mode != "shared":
        raise ValueError("shared_shadow_step requires spectrum_mode='shared'")
    count = int(config.num_uavs)
    previous = np.asarray(positions_m, dtype=np.float64).reshape(count, POSITION_DIM).copy()
    energy = np.asarray(remaining_energy, dtype=np.float64).reshape(count).copy()
    queues = np.asarray(queue_bits, dtype=np.float64).reshape(config.num_buoys).copy()
    arrivals = np.asarray(arrival_bits, dtype=np.float64).reshape(config.num_buoys).copy()
    if not (np.all(np.isfinite(previous)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(queues)) and np.all(np.isfinite(arrivals))):
        raise ValueError("shared transition inputs must be finite")
    raw, component, normalized, component_adjusted, norm_adjusted = process_joint_action(action, count)
    displacement = normalized * float(config.max_speed * config.delta_t)
    movement = np.linalg.norm(displacement, axis=1)
    candidate = previous + displacement
    upper = np.array((config.area_width, config.area_height), dtype=np.float64)
    boundary = np.any(candidate < 0.0, axis=1) | np.any(candidate > upper, axis=1)
    bounded = np.clip(candidate, 0.0, upper)
    safe, raw_risk, safety_correction, minimum_distance = project_uav_separation(bounded, float(config.multi_uav_min_separation_m), float(config.area_width), float(config.area_height))
    applied_displacement = safe - previous
    velocities = applied_displacement / float(config.delta_t)
    applied_movement = np.linalg.norm(applied_displacement, axis=1)
    energy_cost = float(config.hover_energy_per_step) + float(config.move_energy_per_meter) * movement
    next_energy = np.maximum(0.0, energy - energy_cost)
    gains, ranges = gain_matrix(safe, buoy_positions_m, config)
    legacy_rates, _ = rate_matrix(safe, buoy_positions_m, config)
    if options.matching_mode == "legacy":
        assignment, matched, legacy_scores = deterministic_matching(queues, legacy_rates, float(config.queue_capacity_bits))
        matching_scores = legacy_scores
        evaluated = sum(1 for _ in _assignment_candidates(count, config.num_buoys))
    else:
        assignment, matched, matching_scores, evaluated = joint_aware_matching(queues, gains, config, float(options.interference_overlap_factor))
    desired, interference, sinr, isr, selected_rates = assignment_link_metrics(assignment, gains, config, float(options.interference_overlap_factor))
    rates = np.zeros_like(gains)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            rates[uav, buoy] = selected_rates[uav]
    capacity = np.zeros(config.num_buoys, dtype=np.float64)
    payload_by_uav = np.zeros(count, dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            capacity[buoy] = selected_rates[uav] * float(config.delta_t)
            payload_by_uav[uav] = min(queues[buoy], capacity[buoy])
    payload = np.minimum(queues, capacity)
    preclip = queues - payload + arrivals
    dropped = np.maximum(preclip - float(config.queue_capacity_bits), 0.0)
    next_queue = np.clip(preclip, 0.0, float(config.queue_capacity_bits))
    residual = next_queue - queues - arrivals + payload + dropped
    if float(np.max(np.abs(residual))) > 1.0e-6 or np.any(payload > queues + 1.0e-6) or np.any(payload > capacity + 1.0e-6):
        raise RuntimeError("shared-spectrum queue/payload conservation failed")
    return SharedShadowStep(safe, velocities, next_energy, next_queue, rates, legacy_rates, ranges, gains, desired, interference, sinr, isr, assignment, matched, matching_scores, evaluated, capacity, payload, payload_by_uav, arrivals, dropped, boundary, raw_risk, safety_correction, minimum_distance, movement, applied_movement, energy_cost, raw, component, normalized, component_adjusted, norm_adjusted)


class SharedSpectrumMultiUAVEnv:
    """Round-19 facade with exact R18 orthogonal/M=1 delegation."""

    def __init__(self, config: Optional[MaritimeEnvConfig] = None, *, buoy_positions_m: Optional[np.ndarray] = None, options: Optional[SharedSpectrumOptions] = None) -> None:
        self.config = config if config is not None else MaritimeEnvConfig()
        self.options = options if options is not None else SharedSpectrumOptions()
        if not isinstance(self.config, MaritimeEnvConfig):
            raise TypeError("config must be a MaritimeEnvConfig instance")
        self._base = MultiUAVMaritimeEnv(self.config, buoy_positions_m=buoy_positions_m)
        if self.options.spectrum_mode == "shared" and self.config.num_uavs > 1:
            if self.config.task_semantics_version != "v1" or self.config.buoy_drift_mode != "static" or self.config.channel_variation_mode != "static":
                raise ValueError("Round-19 shared primary environment requires static v1 task semantics")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def num_uavs(self) -> int: return self._base.num_uavs

    @property
    def num_buoys(self) -> int: return self._base.num_buoys

    @property
    def state_dim(self) -> int: return self._base.state_dim

    @property
    def action_dim(self) -> int: return self._base.action_dim

    @property
    def action_shape(self) -> tuple[int, int]: return self._base.action_shape

    @property
    def observation_space(self) -> Any: return self._base.observation_space

    @property
    def action_space(self) -> Any: return self._base.action_space

    @property
    def time_step(self) -> int: return self._base.time_step

    def reset(self, seed: Optional[int] = None):
        observation, info = self._base.reset(seed=seed)
        if self.options.spectrum_mode == "shared" and self.num_uavs > 1:
            observation = self._shared_observation()
        return observation, self._decorate_info(info)

    def _shared_observation(self) -> np.ndarray:
        """Keep the R18 layout while normalizing its rate feature for B links."""
        base = self._base
        velocity = base._uav_velocities_mps / (float(self.config.max_speed) + np.abs(base._uav_velocities_mps))
        uav = np.column_stack((2.0 * base._uav_positions_m[:, 0] / self.config.area_width - 1.0, 2.0 * base._uav_positions_m[:, 1] / self.config.area_height - 1.0, velocity[:, 0], velocity[:, 1], base._remaining_energy / self.config.initial_energy, np.clip(base._cumulative_distance_m / (self.config.episode_horizon * self.config.max_speed * self.config.delta_t), 0.0, 1.0), np.clip(base._cumulative_energy_m / self.config.initial_energy, 0.0, 1.0)))
        completion = base._completion_ratio_per_buoy()
        buoy = np.column_stack((2.0 * base._buoy_positions_m[:, 0] / self.config.area_width - 1.0, 2.0 * base._buoy_positions_m[:, 1] / self.config.area_height - 1.0, base._queue_bits / self.config.queue_capacity_bits, base._last_arrival_bits / (base._last_arrival_bits + self.config.queue_capacity_bits), completion))
        rel = base._buoy_positions_m[None, :, :] - base._uav_positions_m[:, None, :]
        shared_rate_upper = float(base._max_single_rate_bps) * self.num_buoys
        relations = np.stack((rel[:, :, 0] / self.config.area_width, rel[:, :, 1] / self.config.area_height, np.clip(base._last_rates_bps / shared_rate_upper, 0.0, 1.0)), axis=-1)
        observation = np.concatenate((uav.reshape(-1), buoy.reshape(-1), relations.reshape(-1), np.ones(self.num_uavs), np.ones(self.num_buoys))).astype(np.float32)
        if observation.shape != (self.state_dim,) or not np.all(np.isfinite(observation)) or not self.observation_space.contains(observation):
            raise RuntimeError("shared-spectrum observation violates the inherited schema bounds")
        return observation.copy()

    def _decorate_info(self, info: Mapping[str, Any], transition: Optional[SharedShadowStep] = None) -> dict[str, Any]:
        result = dict(info)
        result.update({"environment_schema": "round19_shared_spectrum_multi_uav_v1", "spectrum_mode": self.options.spectrum_mode, "interference_overlap_factor": float(self.options.interference_overlap_factor), "matching_mode": self.options.matching_mode, "interference_conflict_isr_threshold": float(self.options.conflict_isr_threshold)})
        if transition is None:
            positions = (np.asarray(result["uav_position_m"], dtype=np.float64).reshape(1, POSITION_DIM) if self.num_uavs == 1 else np.asarray(result["uav_positions_m"], dtype=np.float64))
            gains, _ = gain_matrix(positions, np.asarray(result["buoy_positions_m"]), self.config)
            result.update({"uav_buoy_channel_gains": gains, "desired_signal_power_w": np.zeros(self.num_uavs), "cross_interference_power_w": np.zeros(self.num_uavs), "sinr_linear": np.zeros(self.num_uavs), "interference_to_signal_ratio": np.zeros(self.num_uavs), "interference_conflict": False, "matching_assignments_evaluated": 0})
        else:
            result.update({"uav_buoy_channel_gains": transition.channel_gains.copy(), "legacy_uav_buoy_rates_bps": transition.legacy_rates_bps.copy(), "desired_signal_power_w": transition.desired_power_w.copy(), "cross_interference_power_w": transition.interference_power_w.copy(), "sinr_linear": transition.sinr_linear.copy(), "interference_to_signal_ratio": transition.isr.copy(), "interference_conflict": bool(np.max(transition.isr) >= float(self.options.conflict_isr_threshold)), "matching_assignments_evaluated": int(transition.matching_assignments_evaluated)})
        return result

    def step(self, action: np.ndarray, *, matching_mode: Optional[str] = None):
        if self.options.spectrum_mode == "orthogonal" or self.num_uavs == 1:
            observation, reward, terminated, truncated, info = self._base.step(action)
            return observation, reward, terminated, truncated, self._decorate_info(info)
        active_options = self.options if matching_mode is None else SharedSpectrumOptions("shared", self.options.interference_overlap_factor, matching_mode, self.options.conflict_isr_threshold)
        base = self._base
        if not base._has_reset:
            raise RuntimeError("step() called before reset(); call reset() first")
        if base._episode_done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        queue_before = base._queue_bits.copy()
        transition = shared_shadow_step(positions_m=base._uav_positions_m, remaining_energy=base._remaining_energy, queue_bits=base._queue_bits, buoy_positions_m=base._buoy_positions_m, action=action, arrival_bits=base._arrival_trace_bits[base._time_step], config=self.config, options=active_options)
        base._uav_positions_m, base._uav_velocities_mps, base._remaining_energy = transition.positions_m.copy(), transition.velocities_mps.copy(), transition.remaining_energy.copy()
        base._queue_bits, base._last_action = transition.queue_bits.copy(), transition.normalized_applied_action.copy()
        base._last_arrival_bits, base._last_link_capacity_bits = transition.arrival_bits.copy(), transition.link_capacity_bits.copy()
        base._last_payload_bits, base._last_payload_by_uav_bits = transition.payload_bits.copy(), transition.payload_by_uav_bits.copy()
        base._last_assignment, base._last_matching_scores = transition.assignment.copy(), transition.matching_scores.copy()
        base._last_rates_bps, base._last_ranges_m = transition.rates_bps.copy(), transition.horizontal_distances_m.copy()
        base._last_safety_correction_m, base._last_raw_collision_risk_pairs = transition.safety_correction_m.copy(), int(transition.safety_raw_risk_pairs)
        base._cumulative_payload_bits += transition.payload_bits; base._cumulative_arrival_bits += transition.arrival_bits; base._cumulative_dropped_bits += transition.dropped_arrival_bits
        base._cumulative_distance_m += transition.applied_movement_distance_m; base._cumulative_energy_m += transition.energy_cost; base._time_step += 1
        energy_depleted = bool(np.any(base._remaining_energy <= EPSILON))
        residual_ratio = float(np.sum(base._queue_bits) / (self.num_buoys * self.config.queue_capacity_bits))
        task_success = bool(base._task_completion_ratio() >= self.config.target_task_completion_ratio and residual_ratio <= self.config.maximum_allowed_residual_queue_ratio and float(np.min(base._remaining_energy / self.config.initial_energy)) >= self.config.minimum_terminal_energy_ratio)
        terminated = bool(energy_depleted or (self.config.terminate_on_task_success and task_success)); truncated = bool(base._time_step >= self.config.episode_horizon)
        maximum_step_energy = float(self.config.hover_energy_per_step + self.config.move_energy_per_meter * self.config.max_speed * self.config.delta_t)
        normalized_energy = float(np.mean(transition.energy_cost / maximum_step_energy)) if maximum_step_energy > EPSILON else 0.0
        payload_utility = float(np.clip(np.sum(transition.payload_bits) / (base._max_team_rate_bps * self.config.delta_t), 0.0, 1.0))
        adjusted = transition.action_component_clipped | transition.action_norm_projected; episode_end = terminated or truncated
        reward_components = {"payload_reward": float(self.config.payload_reward_weight * payload_utility), "energy_penalty": float(-self.config.energy_penalty_weight * normalized_energy), "boundary_penalty": float(-self.config.boundary_penalty if bool(np.any(transition.boundary_violation)) else 0.0), "invalid_action_penalty": float(-self.config.invalid_action_penalty if bool(np.any(adjusted)) else 0.0), "energy_depletion_penalty": float(-self.config.energy_depletion_penalty if energy_depleted else 0.0), "unfinished_queue_penalty": float(-self.config.unfinished_queue_penalty_weight * residual_ratio if episode_end and not task_success else 0.0), "task_success_bonus": float(self.config.task_success_bonus if task_success and self.config.terminate_on_task_success else 0.0)}
        reward = float(sum(reward_components.values()))
        info = base._build_info(); info.update({"raw_action": transition.raw_action.copy(), "component_clipped_action": transition.component_clipped_action.copy(), "normalized_applied_action": transition.normalized_applied_action.copy(), "action_component_clipped": transition.action_component_clipped.copy(), "action_norm_projected": transition.action_norm_projected.copy(), "boundary_violation": transition.boundary_violation.copy(), "queue_before_bits": queue_before, "queue_conservation_residual_bits": base._queue_bits - queue_before - transition.arrival_bits + transition.payload_bits + transition.dropped_arrival_bits, "payload_utility": payload_utility, "normalized_energy_cost": normalized_energy, "energy_cost_per_uav": transition.energy_cost.copy(), "movement_distance_m_per_uav": transition.movement_distance_m.copy(), "applied_movement_distance_m_per_uav": transition.applied_movement_distance_m.copy(), "task_success": task_success, "terminated": terminated, "truncated": truncated, "reward_components": reward_components})
        base._episode_done = bool(terminated or truncated)
        return self._shared_observation(), reward, terminated, truncated, self._decorate_info(info, transition)


__all__ = ["MATCHING_MODES", "SPECTRUM_MODES", "SharedShadowStep", "SharedSpectrumMultiUAVEnv", "SharedSpectrumOptions", "assignment_link_metrics", "gain_matrix", "joint_aware_matching", "shared_shadow_step"]
