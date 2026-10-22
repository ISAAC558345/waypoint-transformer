"""Level-2 shared-spectrum environment with an external association action.

This module is deliberately an additive facade over the locked Round-19
physics.  ``legacy_deterministic`` delegates to that environment unchanged.
``external_preference`` retains its movement, queue, safety, reward, and
shared-spectrum equations while replacing only the ownership of association
selection: finite external edge logits select amongst feasible assignments.
No model, optimizer, planner, future state, or heuristic queue/rate objective
is used by the external-preference decoder.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round18.multi_env import EPSILON, POSITION_DIM, process_joint_action, project_uav_separation, rate_matrix
from wt.round19.multi_env import (
    SharedShadowStep,
    SharedSpectrumMultiUAVEnv,
    SharedSpectrumOptions,
    _assignment_candidates,
    assignment_link_metrics,
    gain_matrix,
)


ASSOCIATION_MODES = ("legacy_deterministic", "external_preference")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite with shape {shape}") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


def association_candidate_mask(queue_bits: np.ndarray, channel_gains: np.ndarray) -> np.ndarray:
    """Return the current-state, individual-edge feasibility mask.

    The injectivity, one-service-per-UAV capacity, and deterministic tie rule
    remain constraints of the structured decoder.  Positive current queue and
    finite positive physical gain are the only per-edge mask predicates.
    """
    queues = np.asarray(queue_bits, dtype=np.float64)
    gains = np.asarray(channel_gains, dtype=np.float64)
    if gains.ndim != 2 or queues.shape != (gains.shape[1],):
        raise ValueError("queue/gain dimensions are inconsistent")
    if not np.all(np.isfinite(queues)) or np.any(queues < 0.0):
        raise ValueError("queue_bits must be finite and nonnegative")
    if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
        raise ValueError("channel_gains must be finite and nonnegative")
    return ((queues[None, :] > EPSILON) & (gains > EPSILON)).copy()


def external_preference_matching(
    queue_bits: np.ndarray,
    channel_gains: np.ndarray,
    config: MaritimeEnvConfig,
    association_logits: np.ndarray,
    *,
    candidate_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode external edge preferences with exact masked injective search.

    Crucially, the comparison objective below is *only* the sum of supplied
    logits.  Physical quantities are used only to construct/validate the
    feasible candidate mask.  Thus the older queue-weighted-rate candidate
    score cannot silently override the controller's preference ownership.
    """
    gains = np.asarray(channel_gains, dtype=np.float64)
    if gains.ndim != 2:
        raise ValueError("channel_gains must be a matrix")
    m_count, k_count = gains.shape
    queues = _finite_array(queue_bits, (k_count,), "queue_bits")
    if np.any(queues < 0.0):
        raise ValueError("queue_bits must be nonnegative")
    logits = _finite_array(association_logits, (m_count, k_count), "association_logits")
    mask = association_candidate_mask(queues, gains) if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    if mask.shape != (m_count, k_count):
        raise ValueError("candidate_mask has an incompatible shape")

    best: Optional[tuple[int, ...]] = None
    best_score = -math.inf
    best_cardinality = -1
    evaluated = 0
    for candidate in _assignment_candidates(m_count, k_count):
        evaluated += 1
        if any(not mask[uav, buoy] for uav, buoy in enumerate(candidate) if buoy >= 0):
            continue
        score = float(sum(logits[uav, buoy] for uav, buoy in enumerate(candidate) if buoy >= 0))
        cardinality = int(sum(buoy >= 0 for buoy in candidate))
        lexical = tuple(k_count if buoy < 0 else buoy for buoy in candidate)
        old_lexical = tuple(k_count if buoy < 0 else buoy for buoy in best) if best is not None else None
        if (
            score > best_score + 1.0e-12
            or (
                math.isclose(score, best_score, rel_tol=1.0e-12, abs_tol=1.0e-12)
                and (
                    cardinality > best_cardinality
                    or (cardinality == best_cardinality and (old_lexical is None or lexical < old_lexical))
                )
            )
        ):
            best, best_score, best_cardinality = candidate, score, cardinality
    if best is None:
        raise RuntimeError("external preference decoder found no feasible assignment")
    assignment = np.asarray(best, dtype=np.int64)
    matched = np.full(k_count, -1, dtype=np.int64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            matched[buoy] = uav
    # Scores are reported as controller preferences, never as an internal
    # queue/rate objective.  Invalid edges are explicitly masked.
    reported = np.where(mask, logits, 0.0)
    return assignment, matched, reported, mask.copy(), evaluated


def _external_shadow_step(
    *,
    positions_m: np.ndarray,
    remaining_energy: np.ndarray,
    queue_bits: np.ndarray,
    buoy_positions_m: np.ndarray,
    movement_action: np.ndarray,
    association_logits: np.ndarray,
    arrival_bits: np.ndarray,
    config: MaritimeEnvConfig,
    options: SharedSpectrumOptions,
) -> tuple[SharedShadowStep, np.ndarray]:
    """Round-19 shared transition with only association ownership replaced."""
    if options.spectrum_mode != "shared":
        raise ValueError("Level-2 external preference requires shared spectrum")
    count = int(config.num_uavs)
    previous = _finite_array(positions_m, (count, POSITION_DIM), "positions_m")
    energy = _finite_array(remaining_energy, (count,), "remaining_energy")
    queues = _finite_array(queue_bits, (int(config.num_buoys),), "queue_bits")
    arrivals = _finite_array(arrival_bits, (int(config.num_buoys),), "arrival_bits")
    if np.any(queues < 0.0) or np.any(arrivals < 0.0):
        raise ValueError("queue and arrival bits must be nonnegative")
    raw, component, normalized, component_adjusted, norm_adjusted = process_joint_action(movement_action, count)
    displacement = normalized * float(config.max_speed * config.delta_t)
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
    gains, ranges = gain_matrix(safe, buoy_positions_m, config)
    legacy_rates, _ = rate_matrix(safe, buoy_positions_m, config)
    assignment, matched, matching_scores, mask, evaluated = external_preference_matching(
        queues, gains, config, association_logits
    )
    desired, interference, sinr, isr, selected_rates = assignment_link_metrics(
        assignment, gains, config, float(options.interference_overlap_factor)
    )
    rates = np.zeros_like(gains)
    capacity = np.zeros(int(config.num_buoys), dtype=np.float64)
    payload_by_uav = np.zeros(count, dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            rates[uav, buoy] = selected_rates[uav]
            capacity[buoy] = selected_rates[uav] * float(config.delta_t)
            payload_by_uav[uav] = min(queues[buoy], capacity[buoy])
    payload = np.minimum(queues, capacity)
    preclip = queues - payload + arrivals
    dropped = np.maximum(preclip - float(config.queue_capacity_bits), 0.0)
    next_queue = np.clip(preclip, 0.0, float(config.queue_capacity_bits))
    residual = next_queue - queues - arrivals + payload + dropped
    if (
        float(np.max(np.abs(residual))) > 1.0e-6
        or np.any(payload > queues + 1.0e-6)
        or np.any(payload > capacity + 1.0e-6)
    ):
        raise RuntimeError("Level-2 shared-spectrum queue/payload conservation failed")
    transition = SharedShadowStep(
        safe, velocities, next_energy, next_queue, rates, legacy_rates, ranges, gains,
        desired, interference, sinr, isr, assignment, matched, matching_scores,
        evaluated, capacity, payload, payload_by_uav, arrivals, dropped, boundary,
        raw_risk, safety_correction, minimum_distance, movement, applied_movement,
        energy_cost, raw, component, normalized, component_adjusted, norm_adjusted,
    )
    return transition, mask


class Level2SharedSpectrumMultiUAVEnv:
    """M/K-configurable Level-2 facade with movement plus edge preferences."""

    def __init__(
        self,
        config: MaritimeEnvConfig,
        *,
        buoy_positions_m: Optional[np.ndarray] = None,
        interference_overlap_factor: float = 0.25,
        association_mode: str = "external_preference",
    ) -> None:
        if association_mode not in ASSOCIATION_MODES:
            raise ValueError(f"association_mode must be one of {ASSOCIATION_MODES}")
        self.association_mode = association_mode
        self._shared = SharedSpectrumMultiUAVEnv(
            config,
            buoy_positions_m=buoy_positions_m,
            options=SharedSpectrumOptions(
                spectrum_mode="shared",
                interference_overlap_factor=float(interference_overlap_factor),
                matching_mode="legacy",
            ),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._shared, name)

    @property
    def config(self) -> MaritimeEnvConfig:
        return self._shared.config

    @property
    def num_uavs(self) -> int:
        return self._shared.num_uavs

    @property
    def num_buoys(self) -> int:
        return self._shared.num_buoys

    @property
    def state_dim(self) -> int:
        return self._shared.state_dim

    @property
    def action_dim(self) -> int:
        return self._shared.action_dim

    @property
    def action_shape(self) -> tuple[int, int]:
        return (self.num_uavs, POSITION_DIM)

    @property
    def association_action_shape(self) -> tuple[int, int]:
        return (self.num_uavs, self.num_buoys)

    @property
    def action_spec(self) -> Mapping[str, tuple[int, int]]:
        return {"move": self.action_shape, "association": self.association_action_shape}

    def reset(self, seed: Optional[int] = None):
        return self._shared.reset(seed=seed)

    def relational_observation(self) -> dict[str, np.ndarray]:
        """Current, variable-size relational view; it contains no future field."""
        base = self._shared._base
        positions = base._uav_positions_m
        buoys = base._buoy_positions_m
        queues = base._queue_bits
        gains, ranges = gain_matrix(positions, buoys, self.config)
        direct_rates, _ = rate_matrix(positions, buoys, self.config)
        relative = buoys[None, :, :] - positions[:, None, :]
        assignment_relation = np.zeros((self.num_uavs, self.num_buoys), dtype=np.float64)
        for uav, buoy in enumerate(base._last_assignment):
            if buoy >= 0:
                assignment_relation[uav, buoy] = 1.0
        uav = np.column_stack((
            positions,
            base._uav_velocities_mps,
            base._remaining_energy,
            base._cumulative_distance_m,
            base._last_payload_by_uav_bits,
            np.any(assignment_relation > 0.0, axis=1).astype(np.float64),
        ))
        buoy = np.column_stack((
            buoys,
            queues,
            base._last_arrival_bits,
            base._completion_ratio_per_buoy(),
            base._last_payload_bits,
            np.any(assignment_relation > 0.0, axis=0).astype(np.float64),
        ))
        queue_weighted_rate = (queues[None, :] / float(self.config.queue_capacity_bits)) * direct_rates
        interference_potential = float(self._shared.options.interference_overlap_factor * self.config.buoy_tx_power_w) * gains
        ub = np.stack((
            relative[:, :, 0], relative[:, :, 1], ranges, gains, direct_rates,
            queue_weighted_rate, interference_potential, assignment_relation,
        ), axis=-1)
        uu = np.zeros((self.num_uavs, self.num_uavs, 7), dtype=np.float64)
        for source in range(self.num_uavs):
            for target in range(self.num_uavs):
                if source == target:
                    continue
                delta = positions[target] - positions[source]
                distance = float(np.linalg.norm(delta))
                uu[source, target] = (
                    distance, delta[0], delta[1],
                    base._uav_velocities_mps[target, 0] - base._uav_velocities_mps[source, 0],
                    base._uav_velocities_mps[target, 1] - base._uav_velocities_mps[source, 1],
                    max(0.0, float(self.config.multi_uav_min_separation_m) - distance),
                    float(np.sum(gains[source]) - np.sum(gains[target])),
                )
        return {
            "uav_features": uav.copy(),
            "buoy_features": buoy.copy(),
            "uav_buoy_relations": ub.copy(),
            "uav_uav_relations": uu.copy(),
            "uav_mask": np.ones(self.num_uavs, dtype=bool),
            "buoy_mask": np.ones(self.num_buoys, dtype=bool),
        }

    def step(self, action: Mapping[str, Any] | np.ndarray):
        """Apply ``{'move': (M,2), 'association': (M,K)}`` in Level-2 mode."""
        if self.association_mode == "legacy_deterministic":
            movement = action["move"] if isinstance(action, Mapping) else action
            return self._shared.step(movement, matching_mode="legacy")
        if not isinstance(action, Mapping) or set(action) != {"move", "association"}:
            raise ValueError("external_preference action must contain exactly move and association")
        movement = _finite_array(action["move"], self.action_shape, "move action")
        logits = _finite_array(action["association"], self.association_action_shape, "association action")
        base = self._shared._base
        if not base._has_reset:
            raise RuntimeError("step() called before reset(); call reset() first")
        if base._episode_done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        queue_before = base._queue_bits.copy()
        transition, mask = _external_shadow_step(
            positions_m=base._uav_positions_m,
            remaining_energy=base._remaining_energy,
            queue_bits=base._queue_bits,
            buoy_positions_m=base._buoy_positions_m,
            movement_action=movement,
            association_logits=logits,
            arrival_bits=base._arrival_trace_bits[base._time_step],
            config=self.config,
            options=self._shared.options,
        )
        base._uav_positions_m = transition.positions_m.copy()
        base._uav_velocities_mps = transition.velocities_mps.copy()
        base._remaining_energy = transition.remaining_energy.copy()
        base._queue_bits, base._last_action = transition.queue_bits.copy(), transition.normalized_applied_action.copy()
        base._last_arrival_bits, base._last_link_capacity_bits = transition.arrival_bits.copy(), transition.link_capacity_bits.copy()
        base._last_payload_bits, base._last_payload_by_uav_bits = transition.payload_bits.copy(), transition.payload_by_uav_bits.copy()
        base._last_assignment, base._last_matching_scores = transition.assignment.copy(), transition.matching_scores.copy()
        base._last_rates_bps, base._last_ranges_m = transition.rates_bps.copy(), transition.horizontal_distances_m.copy()
        base._last_safety_correction_m, base._last_raw_collision_risk_pairs = transition.safety_correction_m.copy(), int(transition.safety_raw_risk_pairs)
        base._cumulative_payload_bits += transition.payload_bits
        base._cumulative_arrival_bits += transition.arrival_bits
        base._cumulative_dropped_bits += transition.dropped_arrival_bits
        base._cumulative_distance_m += transition.applied_movement_distance_m
        base._cumulative_energy_m += transition.energy_cost
        base._time_step += 1
        energy_depleted = bool(np.any(base._remaining_energy <= EPSILON))
        residual_ratio = float(np.sum(base._queue_bits) / (self.num_buoys * self.config.queue_capacity_bits))
        task_success = bool(
            base._task_completion_ratio() >= self.config.target_task_completion_ratio
            and residual_ratio <= self.config.maximum_allowed_residual_queue_ratio
            and float(np.min(base._remaining_energy / self.config.initial_energy)) >= self.config.minimum_terminal_energy_ratio
        )
        terminated = bool(energy_depleted or (self.config.terminate_on_task_success and task_success))
        truncated = bool(base._time_step >= self.config.episode_horizon)
        maximum_step_energy = float(self.config.hover_energy_per_step + self.config.move_energy_per_meter * self.config.max_speed * self.config.delta_t)
        normalized_energy = float(np.mean(transition.energy_cost / maximum_step_energy)) if maximum_step_energy > EPSILON else 0.0
        payload_utility = float(np.clip(np.sum(transition.payload_bits) / (base._max_team_rate_bps * self.config.delta_t), 0.0, 1.0))
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
        info = base._build_info()
        info.update({
            "raw_action": transition.raw_action.copy(),
            "component_clipped_action": transition.component_clipped_action.copy(),
            "normalized_applied_action": transition.normalized_applied_action.copy(),
            "action_component_clipped": transition.action_component_clipped.copy(),
            "action_norm_projected": transition.action_norm_projected.copy(),
            "boundary_violation": transition.boundary_violation.copy(),
            "queue_before_bits": queue_before,
            "queue_conservation_residual_bits": base._queue_bits - queue_before - transition.arrival_bits + transition.payload_bits + transition.dropped_arrival_bits,
            "payload_utility": payload_utility,
            "normalized_energy_cost": normalized_energy,
            "energy_cost_per_uav": transition.energy_cost.copy(),
            "movement_distance_m_per_uav": transition.movement_distance_m.copy(),
            "applied_movement_distance_m_per_uav": transition.applied_movement_distance_m.copy(),
            "task_success": task_success,
            "terminated": terminated,
            "truncated": truncated,
            "reward_components": reward_components,
            "association_mode": "external_preference",
            "association_logits": logits.copy(),
            "association_candidate_mask": mask.copy(),
            "matching_assignments_evaluated": int(transition.matching_assignments_evaluated),
            "uav_buoy_channel_gains": transition.channel_gains.copy(),
            "legacy_uav_buoy_rates_bps": transition.legacy_rates_bps.copy(),
            "desired_signal_power_w": transition.desired_power_w.copy(),
            "cross_interference_power_w": transition.interference_power_w.copy(),
            "sinr_linear": transition.sinr_linear.copy(),
            "interference_to_signal_ratio": transition.isr.copy(),
            "interference_conflict": bool(np.max(transition.isr) >= float(self._shared.options.conflict_isr_threshold)),
        })
        base._episode_done = bool(terminated or truncated)
        return self._shared._shared_observation(), reward, terminated, truncated, info


__all__ = [
    "ASSOCIATION_MODES",
    "Level2SharedSpectrumMultiUAVEnv",
    "association_candidate_mask",
    "external_preference_matching",
]
