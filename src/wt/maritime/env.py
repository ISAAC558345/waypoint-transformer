"""Minimal single-UAV maritime communication environment.

The environment deliberately has no dependency on Gym.  Its ``reset`` and
``step`` return values follow the Gymnasium API shape so that a later data
collector can wrap it without changing the environment dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .arrival import materialize_arrival_trace
from .channel import compute_channel_metrics, theoretical_max_rates
from .config import MaritimeEnvConfig


_ACTION_DIM = 2
_POSITION_DIM = 2
_BASE_STATE_DIM = 6
_BUOY_FEATURE_DIM = 4
_V1_BASE_STATE_DIM = 9
_V1_BUOY_FEATURE_DIM = 7
_V1_DRIFT_BUOY_FEATURE_DIM = 9
_V1_CHANNEL_BUOY_FEATURE_DIM = 10
_V1_DRIFT_CHANNEL_BUOY_FEATURE_DIM = 12
_NORMALIZATION_TOLERANCE = 1.0e-9
_DEFAULT_FOUR_BUOY_POSITIONS_M = (
    (50.0, 50.0),
    (50.0, 150.0),
    (150.0, 50.0),
    (150.0, 150.0),
)
_CHANNEL_ARRAY_KEYS = (
    "horizontal_distances_m",
    "distances_3d_m",
    "channel_gains",
    "snr_linear",
    "rates_bps",
)


def reflect_buoy_motion(
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    delta_t: float,
    config: MaritimeEnvConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance buoy positions with deterministic margin-rectangle reflection.

    The function is pure so a causal planner can use the exact same geometric
    transition without reading a future environment trace. Every odd boundary
    crossing reverses the associated velocity component, including repeated
    crossings within one time step.
    """
    positions = np.asarray(positions_m, dtype=np.float64)
    velocities = np.asarray(velocities_mps, dtype=np.float64)
    expected = (int(config.num_buoys), _POSITION_DIM)
    if positions.shape != expected or velocities.shape != expected:
        raise ValueError("buoy positions and velocities must have shape (num_buoys, 2)")
    lower = np.full(_POSITION_DIM, float(config.buoy_margin), dtype=np.float64)
    upper = np.array(
        (config.area_width - config.buoy_margin, config.area_height - config.buoy_margin),
        dtype=np.float64,
    )
    span = upper - lower
    candidate = positions + velocities * float(delta_t)
    translated = candidate - lower
    crossings = np.floor(translated / span).astype(np.int64)
    period = 2.0 * span
    phase = np.mod(translated, period)
    reflected = lower + np.where(phase <= span, phase, period - phase)
    signs = np.where(np.mod(crossings, 2) == 0, 1.0, -1.0)
    reflected_velocity = velocities * signs
    if not np.all(np.isfinite(reflected)) or not np.all(np.isfinite(reflected_velocity)):
        raise RuntimeError("buoy reflection produced NaN or Inf")
    if np.any(reflected < lower - 1.0e-9) or np.any(reflected > upper + 1.0e-9):
        raise RuntimeError("buoy reflection left the configured margin rectangle")
    return reflected.astype(np.float64, copy=True), reflected_velocity.astype(np.float64, copy=True)


@dataclass(frozen=True)
class ContinuousBox:
    """Dependency-free public description of a continuous vector space."""

    low: np.ndarray
    high: np.ndarray
    shape: Tuple[int, ...]
    dtype: Any = np.float32

    def contains(self, value: Any) -> bool:
        """Return whether ``value`` has the declared shape and finite bounds."""
        try:
            array = np.asarray(value, dtype=self.dtype)
        except (TypeError, ValueError):
            return False
        return bool(
            array.shape == self.shape
            and np.all(np.isfinite(array))
            and np.all(array >= self.low)
            and np.all(array <= self.high)
        )


class MaritimeEnv:
    """Single-UAV maritime communication environment with optional buoy drift.

    The action is a normalized two-vector with shape ``(2,)``.  The
    observation is a ``float32`` vector with shape ``(state_dim,)``, where
    ``state_dim = 6 + 4 * num_buoys``.

    Parameters
    ----------
    config:
        Immutable environment configuration.  When omitted, the default
        :class:`MaritimeEnvConfig` is used.
    """

    def __init__(self, config: Optional[MaritimeEnvConfig] = None) -> None:
        """Initialize an environment without starting an episode."""
        self.config = config if config is not None else MaritimeEnvConfig()
        if not isinstance(self.config, MaritimeEnvConfig):
            raise TypeError(
                "config must be a MaritimeEnvConfig instance or None, "
                f"got {type(self.config).__name__}"
            )

        self._rng = np.random.default_rng(self.config.seed)
        self._time_step = 0
        self._uav_position_m = np.zeros(_POSITION_DIM, dtype=np.float64)
        self._uav_velocity_mps = np.zeros(_POSITION_DIM, dtype=np.float64)
        self._remaining_energy = float(self.config.initial_energy)
        self._buoy_positions_m = np.empty(
            (self.config.num_buoys, _POSITION_DIM), dtype=np.float64
        )
        self._buoy_drift_velocity_mps = np.zeros(
            (self.config.num_buoys, _POSITION_DIM), dtype=np.float64
        )
        self._buoy_drift_mean_velocity_mps = np.zeros(
            (self.config.num_buoys, _POSITION_DIM), dtype=np.float64
        )
        self._channel_phase_rad = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._channel_frequency_rad_per_slot = float(
            2.0 * math.pi / self.config.channel_period_slots
        )
        self._channel_multipliers = np.ones(
            self.config.num_buoys, dtype=np.float64
        )
        self._channel_metrics: Dict[str, Any] = {}
        self._has_reset = False
        self._episode_done = False
        self._initial_queue_bits = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._queue_bits = np.zeros(self.config.num_buoys, dtype=np.float64)
        self._arrival_trace_bits = np.zeros(
            (self.config.episode_horizon, self.config.num_buoys),
            dtype=np.float64,
        )
        self._arrival_trace_id = ""
        self._last_arrival_bits = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._last_link_capacity_bits = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._last_transmitted_payload_bits = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._last_dropped_arrival_bits = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._cumulative_arrival_bits = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._cumulative_payload_bits_by_buoy = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._cumulative_dropped_bits = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._service_completion_ratio = np.zeros(
            self.config.num_buoys, dtype=np.float64
        )
        self._cumulative_distance_m = 0.0
        self._cumulative_energy_consumption = 0.0
        self._task_completion_ratio = 0.0
        self._task_success = False
        self._terminal_reason = "not_reset"

        channel_kwargs = self._channel_keyword_arguments()
        if self._channel_enabled():
            channel_kwargs = dict(channel_kwargs)
            channel_kwargs["reference_channel_gain"] *= self._maximum_channel_multiplier()
        max_single, max_total = theoretical_max_rates(
            num_buoys=self.config.num_buoys,
            **channel_kwargs,
        )
        self._max_single_rate_bps = self._require_positive_finite_scalar(
            max_single, "theoretical maximum single-buoy rate"
        )
        self._max_total_rate_bps = self._require_positive_finite_scalar(
            max_total, "theoretical maximum total rate"
        )
        self._maximum_3d_distance_m = math.sqrt(
            self.config.uav_altitude**2
            + self.config.area_width**2
            + self.config.area_height**2
        )

    def _channel_enabled(self) -> bool:
        return self.config.channel_variation_mode != "static"

    def _v1_per_buoy_feature_dim(self) -> int:
        if self.config.buoy_drift_mode != "static" and self._channel_enabled():
            return _V1_DRIFT_CHANNEL_BUOY_FEATURE_DIM
        if self.config.buoy_drift_mode != "static":
            return _V1_DRIFT_BUOY_FEATURE_DIM
        if self._channel_enabled():
            return _V1_CHANNEL_BUOY_FEATURE_DIM
        return _V1_BUOY_FEATURE_DIM

    def _maximum_channel_multiplier(self) -> float:
        return float(math.exp(self.config.channel_log_amplitude))

    @property
    def state_dim(self) -> int:
        """Return the fixed one-dimensional observation size."""
        if self.config.task_semantics_version == "v1":
            per_buoy = self._v1_per_buoy_feature_dim()
            return _V1_BASE_STATE_DIM + per_buoy * int(
                self.config.num_buoys
            )
        return _BASE_STATE_DIM + _BUOY_FEATURE_DIM * int(self.config.num_buoys)

    @property
    def action_dim(self) -> int:
        """Return the planar movement-action size."""
        return _ACTION_DIM

    @property
    def observation_space(self) -> ContinuousBox:
        """Return finite flat observation bounds without importing Gym."""
        low = np.zeros(self.state_dim, dtype=np.float32)
        high = np.ones(self.state_dim, dtype=np.float32)
        low[:4] = -1.0
        if self.config.task_semantics_version == "v0":
            features = low[_BASE_STATE_DIM:].reshape(
                self.num_buoys, _BUOY_FEATURE_DIM
            )
            features[:, :2] = -1.0
        else:
            per_buoy = self._v1_per_buoy_feature_dim()
            features = low[_V1_BASE_STATE_DIM:].reshape(
                self.num_buoys, per_buoy
            )
            features[:, :2] = -1.0
            signed_end = 2
            if self.config.buoy_drift_mode != "static":
                features[:, signed_end:signed_end + 2] = -1.0
                signed_end += 2
            if self._channel_enabled():
                features[:, signed_end:signed_end + 2] = -1.0
        return ContinuousBox(
            low=low,
            high=high,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> ContinuousBox:
        """Return component bounds for the normalized planar action."""
        return ContinuousBox(
            low=np.full((_ACTION_DIM,), -1.0, dtype=np.float64),
            high=np.full((_ACTION_DIM,), 1.0, dtype=np.float64),
            shape=(_ACTION_DIM,),
            dtype=np.float64,
        )

    @property
    def observation_schema(self) -> Dict[str, Any]:
        """Describe the active flat observation layout and normalization."""
        if self.config.task_semantics_version == "v0":
            return {
                "schema_version": "maritime_observation_v0",
                "task_semantics_version": "v0",
                "state_dim": self.state_dim,
                "num_buoys": self.num_buoys,
                "base_feature_count": _BASE_STATE_DIM,
                "per_buoy_feature_count": _BUOY_FEATURE_DIM,
                "per_buoy_feature_names": [
                    "relative_x_over_area_width",
                    "relative_y_over_area_height",
                    "distance_3d_over_maximum_3d_distance",
                    "rate_over_theoretical_max_single_buoy_rate",
                ],
            }
        drift_enabled = self.config.buoy_drift_mode != "static"
        channel_enabled = self._channel_enabled()
        per_buoy_names = [
            "relative_x_over_area_width",
            "relative_y_over_area_height",
        ]
        if drift_enabled:
            per_buoy_names.extend(
                [
                    "drift_velocity_x_over_drift_speed_max",
                    "drift_velocity_y_over_drift_speed_max",
                ]
            )
        if channel_enabled:
            per_buoy_names.extend(
                [
                    "channel_phase_sin",
                    "channel_phase_cos",
                    "channel_frequency_over_registered_max",
                ]
            )
        per_buoy_names.extend(
            [
                "distance_3d_over_maximum_3d_distance",
                "allocated_rate_over_theoretical_max_single_buoy_rate",
                "raw_queue_bits_over_queue_capacity_bits",
                "last_arrival_bits_over_last_arrival_plus_queue_capacity",
                "service_completion_ratio",
            ]
        )
        return {
            "schema_version": (
                "v1_drift_channel" if drift_enabled and channel_enabled
                else "v1_channel" if channel_enabled
                else "v1_drift" if drift_enabled else "v1"
            ),
            "schema_id": (
                "maritime_observation_v1_drift_channel" if drift_enabled and channel_enabled
                else "maritime_observation_v1_channel" if channel_enabled
                else "maritime_observation_v1_drift" if drift_enabled else "maritime_observation_v1"
            ),
            "task_semantics_version": "v1",
            "buoy_drift_mode": self.config.buoy_drift_mode,
            "channel_variation_mode": self.config.channel_variation_mode,
            "state_dim": self.state_dim,
            "num_buoys": self.num_buoys,
            "flattening_order": "nine global features, then dynamic per-buoy features",
            "base_feature_count": _V1_BASE_STATE_DIM,
            "base_feature_names": [
                "normalized_uav_x",
                "normalized_uav_y",
                "normalized_uav_velocity_x",
                "normalized_uav_velocity_y",
                "remaining_energy_ratio",
                "remaining_horizon_fraction",
                "normalized_cumulative_distance",
                "normalized_cumulative_energy_consumption",
                "task_completion_ratio",
            ],
            "per_buoy_feature_count": len(per_buoy_names),
            "per_buoy_feature_names": per_buoy_names,
            "units": {
                "queue": "bit",
                "arrival": "bit",
                "rate": "bit/s",
                "position": "m",
                "drift_velocity": "m/s",
                "channel_phase": "radian encoded as sine/cosine",
                "channel_frequency": "radian/slot normalized by registered maximum",
                "energy": "abstract energy unit",
            },
            "arrival_alignment": (
                "state s_t contains only A[t-1], the most recent realized "
                "arrival; A[t] is not exposed before action_t"
            ),
        }

    @property
    def num_buoys(self) -> int:
        """Return the number of static buoys in the environment."""
        return int(self.config.num_buoys)

    @property
    def time_step(self) -> int:
        """Return the number of completed steps in the current episode."""
        return self._time_step

    @property
    def uav_position_m(self) -> np.ndarray:
        """Return a copy of the physical UAV position with shape ``(2,)``."""
        return self._uav_position_m.copy()

    @property
    def uav_velocity_mps(self) -> np.ndarray:
        """Return a copy of the physical UAV velocity with shape ``(2,)``."""
        return self._uav_velocity_mps.copy()

    @property
    def remaining_energy(self) -> float:
        """Return the remaining energy in the placeholder energy model."""
        return float(self._remaining_energy)

    @property
    def buoy_positions_m(self) -> np.ndarray:
        """Return current buoy positions with shape ``(K, 2)``."""
        return self._buoy_positions_m.copy()

    @property
    def buoy_drift_velocity_mps(self) -> np.ndarray:
        """Return current observable buoy drift velocities with shape ``(K, 2)``."""
        return self._buoy_drift_velocity_mps.copy()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode and return ``(observation, info)``.

        Parameters
        ----------
        seed:
            Optional seed used to reconstruct this environment's private
            ``numpy.random.Generator``.  No global random state is changed.

        Returns
        -------
        observation:
            Normalized ``float32`` array with shape ``(state_dim,)``.
        info:
            Dictionary containing copied, unnormalized physical arrays and
            current communication metrics.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._time_step = 0
        self._uav_position_m = np.array(
            (self.config.initial_uav_x, self.config.initial_uav_y),
            dtype=np.float64,
        )
        self._uav_velocity_mps = np.zeros(_POSITION_DIM, dtype=np.float64)
        self._remaining_energy = float(self.config.initial_energy)
        self._buoy_positions_m = self._generate_buoy_positions()
        self._reset_buoy_drift()
        self._reset_channel_dynamics()
        self._update_channel_metrics()
        if self.config.task_semantics_version == "v1":
            self._reset_task_state(seed)
        self._has_reset = True
        self._episode_done = False

        observation = self._build_observation()
        info = self._build_physical_info()
        return observation.copy(), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one slot.

        Parameters
        ----------
        action:
            Array-like normalized movement command with exact shape ``(2,)``.
            Components are clipped to ``[-1, 1]`` and the resulting vector is
            radially projected onto the unit disk before it is converted to a
            physical displacement.

        Returns
        -------
        observation:
            Next normalized ``float32`` state with shape ``(state_dim,)``.
        reward:
            Python ``float`` containing the decomposed communication reward.
        terminated:
            ``True`` when the placeholder energy budget is depleted.
        truncated:
            ``True`` when ``episode_horizon`` steps have completed.
        info:
            Copied physical state, channel data, action processing details,
            boundary processing details, and signed reward components.

        Raises
        ------
        RuntimeError:
            If called before ``reset`` or after the episode has ended.
        ValueError:
            If the action has an invalid shape or contains a non-finite value.
        """
        self._ensure_step_is_available()
        if self.config.task_semantics_version == "v1":
            return self._step_v1(action)
        (
            raw_action,
            component_clipped_action,
            normalized_applied_action,
            action_component_clipped,
            action_norm_projected,
        ) = self._process_action(action)

        physical_displacement_m = (
            normalized_applied_action
            * self.config.max_speed
            * self.config.delta_t
        )
        movement_distance_m = float(np.linalg.norm(physical_displacement_m))
        maximum_step_distance_m = self.config.max_speed * self.config.delta_t
        if movement_distance_m > maximum_step_distance_m + _NORMALIZATION_TOLERANCE:
            raise RuntimeError(
                "processed action exceeds the configured maximum displacement"
            )

        previous_position_m = self._uav_position_m.copy()
        candidate_position_m = previous_position_m + physical_displacement_m
        area_lower_bound_m = np.zeros(_POSITION_DIM, dtype=np.float64)
        area_upper_bound_m = np.array(
            (self.config.area_width, self.config.area_height), dtype=np.float64
        )
        boundary_violation = bool(
            np.any(candidate_position_m < area_lower_bound_m)
            or np.any(candidate_position_m > area_upper_bound_m)
        )
        applied_position_m = np.clip(
            candidate_position_m, area_lower_bound_m, area_upper_bound_m
        )
        boundary_projection_distance_m = float(
            np.linalg.norm(candidate_position_m - applied_position_m)
        )
        applied_displacement_m = applied_position_m - previous_position_m

        self._uav_position_m = applied_position_m.copy()
        self._uav_velocity_mps = (
            applied_displacement_m / self.config.delta_t
        ).astype(np.float64, copy=True)

        energy_cost = float(
            self.config.hover_energy_per_step
            + self.config.move_energy_per_meter * movement_distance_m
        )
        previous_energy = self._remaining_energy
        self._remaining_energy = float(max(0.0, previous_energy - energy_cost))
        actual_energy_consumed = float(previous_energy - self._remaining_energy)

        self._time_step += 1
        self._update_channel_metrics()

        energy_depleted = bool(self._remaining_energy <= 0.0)
        rate_utility = float(
            self._channel_metrics["total_rate_bps"] / self._max_total_rate_bps
        )
        self._require_interval(rate_utility, 0.0, 1.0, "rate utility")
        movement_cost = float(movement_distance_m / maximum_step_distance_m)
        self._require_interval(movement_cost, 0.0, 1.0, "movement cost")
        boundary_cost = float(
            self.config.boundary_penalty if boundary_violation else 0.0
        )
        energy_depletion_cost = float(
            self.config.energy_depletion_penalty if energy_depleted else 0.0
        )

        reward_components = {
            "rate_reward": float(self.config.rate_reward_weight * rate_utility),
            "movement_penalty": float(
                -self.config.movement_penalty_weight * movement_cost
            ),
            "boundary_penalty": float(-boundary_cost),
            "energy_depletion_penalty": float(-energy_depletion_cost),
        }
        reward = float(sum(reward_components.values()))
        if not math.isfinite(reward):
            raise RuntimeError("reward calculation produced a non-finite value")

        terminated = energy_depleted
        truncated = bool(self._time_step >= self.config.episode_horizon)
        observation = self._build_observation()
        info = self._build_physical_info()
        info.update(
            {
                "raw_action": raw_action.copy(),
                "component_clipped_action": component_clipped_action.copy(),
                "normalized_applied_action": normalized_applied_action.copy(),
                "physical_displacement_m": physical_displacement_m.copy(),
                "applied_displacement_m": applied_displacement_m.copy(),
                "action_component_clipped": action_component_clipped,
                "action_norm_projected": action_norm_projected,
                "boundary_violation": boundary_violation,
                "candidate_position_m": candidate_position_m.copy(),
                "applied_position_m": applied_position_m.copy(),
                "boundary_projection_distance_m": boundary_projection_distance_m,
                "movement_distance_m": movement_distance_m,
                "applied_movement_distance_m": float(
                    np.linalg.norm(applied_displacement_m)
                ),
                "rate_utility": rate_utility,
                "movement_cost": movement_cost,
                "boundary_cost": boundary_cost,
                "energy_cost": energy_cost,
                "actual_energy_consumed": actual_energy_consumed,
                "energy_depleted": energy_depleted,
                "energy_depletion_cost": energy_depletion_cost,
                "reward_components": dict(reward_components),
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        self._episode_done = terminated or truncated
        return observation.copy(), reward, terminated, truncated, info

    def _reset_task_state(self, seed: Optional[int]) -> None:
        """Initialize all task-semantics-v1 state without exposing the future."""
        if self.config.initial_queue_bits_by_buoy is None:
            initial = np.full(
                self.num_buoys,
                float(self.config.initial_queue_bits),
                dtype=np.float64,
            )
        else:
            initial = np.asarray(
                self.config.initial_queue_bits_by_buoy, dtype=np.float64
            )
        self._initial_queue_bits = initial.copy()
        self._queue_bits = initial.copy()
        trace_seed = int(self.config.seed if seed is None else seed)
        trace, identifier = materialize_arrival_trace(
            self.config, seed=trace_seed
        )
        self._arrival_trace_bits = trace.copy()
        self._arrival_trace_id = identifier
        zeros = np.zeros(self.num_buoys, dtype=np.float64)
        self._last_arrival_bits = zeros.copy()
        self._last_link_capacity_bits = zeros.copy()
        self._last_transmitted_payload_bits = zeros.copy()
        self._last_dropped_arrival_bits = zeros.copy()
        self._cumulative_arrival_bits = zeros.copy()
        self._cumulative_payload_bits_by_buoy = zeros.copy()
        self._cumulative_dropped_bits = zeros.copy()
        self._service_completion_ratio = self._completion_ratios()
        self._cumulative_distance_m = 0.0
        self._cumulative_energy_consumption = 0.0
        self._task_completion_ratio = self._total_completion_ratio()
        self._task_success = False
        self._terminal_reason = "ongoing"

    def _step_v1(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one service-before-arrival task slot.

        The channel already assigns each buoy a fixed orthogonal ``1/K``
        bandwidth share. Consequently ``rates_bps`` is the allocated rate and
        the per-slot capacity is exactly ``rates_bps * delta_t``; multiplying
        by ``1/K`` again would double-apply the allocation.
        """
        (
            raw_action,
            component_clipped_action,
            normalized_applied_action,
            action_component_clipped,
            action_norm_projected,
        ) = self._process_action(action)
        action_was_adjusted = bool(
            action_component_clipped or action_norm_projected
        )

        physical_displacement_m = (
            normalized_applied_action
            * self.config.max_speed
            * self.config.delta_t
        )
        movement_distance_m = float(np.linalg.norm(physical_displacement_m))
        maximum_step_distance_m = self.config.max_speed * self.config.delta_t
        previous_position_m = self._uav_position_m.copy()
        candidate_position_m = previous_position_m + physical_displacement_m
        lower = np.zeros(_POSITION_DIM, dtype=np.float64)
        upper = np.array(
            (self.config.area_width, self.config.area_height), dtype=np.float64
        )
        boundary_violation = bool(
            np.any(candidate_position_m < lower)
            or np.any(candidate_position_m > upper)
        )
        applied_position_m = np.clip(candidate_position_m, lower, upper)
        applied_displacement_m = applied_position_m - previous_position_m
        boundary_projection_distance_m = float(
            np.linalg.norm(candidate_position_m - applied_position_m)
        )
        applied_movement_distance_m = float(
            np.linalg.norm(applied_displacement_m)
        )
        self._uav_position_m = applied_position_m.copy()
        self._uav_velocity_mps = (
            applied_displacement_m / self.config.delta_t
        ).astype(np.float64, copy=True)

        energy_cost = float(
            self.config.hover_energy_per_step
            + self.config.move_energy_per_meter * movement_distance_m
        )
        previous_energy = float(self._remaining_energy)
        self._remaining_energy = float(max(0.0, previous_energy - energy_cost))
        actual_energy_consumed = float(
            previous_energy - self._remaining_energy
        )
        self._cumulative_distance_m += applied_movement_distance_m
        self._cumulative_energy_consumption += actual_energy_consumed

        # Service uses Q[t] only. A[t] is realized after this service and can
        # first influence the following state/action.
        queue_before_bits = self._queue_bits.copy()
        slot_index = int(self._time_step)
        self._update_channel_metrics()
        service_total_rate_bps = float(self._channel_metrics["total_rate_bps"])
        link_capacity_bits = (
            np.asarray(self._channel_metrics["rates_bps"], dtype=np.float64)
            * float(self.config.delta_t)
        )
        transmitted_payload_bits = np.minimum(
            queue_before_bits, link_capacity_bits
        )
        transmitted_payload_bits = np.maximum(
            transmitted_payload_bits, 0.0
        )
        arrival_bits = self._arrival_trace_bits[slot_index].copy()
        preclip_queue_bits = (
            queue_before_bits - transmitted_payload_bits + arrival_bits
        )
        dropped_arrival_bits = np.maximum(
            preclip_queue_bits - float(self.config.queue_capacity_bits), 0.0
        )
        next_queue_bits = np.clip(
            preclip_queue_bits,
            0.0,
            float(self.config.queue_capacity_bits),
        )
        conservation_residual_bits = (
            next_queue_bits
            - queue_before_bits
            - arrival_bits
            + transmitted_payload_bits
            + dropped_arrival_bits
        )
        if float(np.max(np.abs(conservation_residual_bits))) > 1.0e-6:
            raise RuntimeError("queue update violates bit conservation")
        if np.any(transmitted_payload_bits > queue_before_bits + 1.0e-6):
            raise RuntimeError("payload exceeds available queue data")
        if np.any(transmitted_payload_bits > link_capacity_bits + 1.0e-6):
            raise RuntimeError("payload exceeds allocated link capacity")

        self._queue_bits = next_queue_bits.copy()
        self._last_arrival_bits = arrival_bits.copy()
        self._last_link_capacity_bits = link_capacity_bits.copy()
        self._last_transmitted_payload_bits = transmitted_payload_bits.copy()
        self._last_dropped_arrival_bits = dropped_arrival_bits.copy()
        self._cumulative_arrival_bits += arrival_bits
        self._cumulative_payload_bits_by_buoy += transmitted_payload_bits
        self._cumulative_dropped_bits += dropped_arrival_bits
        self._service_completion_ratio = self._completion_ratios()
        self._task_completion_ratio = self._total_completion_ratio()
        self._time_step += 1
        drift_before_positions_m = self._buoy_positions_m.copy()
        drift_before_velocity_mps = self._buoy_drift_velocity_mps.copy()
        channel_phase_before_rad = self._channel_phase_rad.copy()
        channel_multiplier_before = self._channel_multipliers.copy()
        drift_reflection_axes = self._advance_buoy_drift()
        self._advance_channel_dynamics()
        if self.config.buoy_drift_mode != "static" or self._channel_enabled():
            self._update_channel_metrics()

        energy_depleted = bool(self._remaining_energy <= 0.0)
        residual_queue_ratio = self._residual_queue_ratio()
        energy_ratio = self._remaining_energy_ratio()
        self._task_success = bool(
            self._task_completion_ratio
            >= self.config.target_task_completion_ratio
            and residual_queue_ratio
            <= self.config.maximum_allowed_residual_queue_ratio
            and energy_ratio >= self.config.minimum_terminal_energy_ratio
        )
        terminated = bool(
            energy_depleted
            or (
                self.config.terminate_on_task_success
                and self._task_success
            )
        )
        truncated = bool(self._time_step >= self.config.episode_horizon)
        if self._task_success and energy_depleted:
            terminal_reason = "task_success_energy_depleted"
        elif self._task_success and self.config.terminate_on_task_success:
            terminal_reason = "task_success"
        elif energy_depleted:
            terminal_reason = "energy_depleted"
        elif truncated:
            terminal_reason = "horizon"
        else:
            terminal_reason = "ongoing"
        self._terminal_reason = terminal_reason

        maximum_step_energy = float(
            self.config.hover_energy_per_step
            + self.config.move_energy_per_meter * maximum_step_distance_m
        )
        normalized_energy_cost = (
            actual_energy_consumed / maximum_step_energy
            if maximum_step_energy > 0.0
            else 0.0
        )
        normalized_energy_cost = float(
            np.clip(normalized_energy_cost, 0.0, 1.0)
        )
        payload_utility = float(
            np.sum(transmitted_payload_bits)
            / (self._max_total_rate_bps * self.config.delta_t)
        )
        payload_utility = float(np.clip(payload_utility, 0.0, 1.0))
        episode_end = bool(terminated or truncated)
        reward_components = {
            "payload_reward": float(
                self.config.payload_reward_weight * payload_utility
            ),
            "energy_penalty": float(
                -self.config.energy_penalty_weight * normalized_energy_cost
            ),
            "boundary_penalty": float(
                -self.config.boundary_penalty
                if boundary_violation
                else 0.0
            ),
            "invalid_action_penalty": float(
                -self.config.invalid_action_penalty
                if action_was_adjusted
                else 0.0
            ),
            "energy_depletion_penalty": float(
                -self.config.energy_depletion_penalty
                if energy_depleted
                else 0.0
            ),
            "unfinished_queue_penalty": float(
                -self.config.unfinished_queue_penalty_weight
                * residual_queue_ratio
                if episode_end and not self._task_success
                else 0.0
            ),
            "task_success_bonus": float(
                self.config.task_success_bonus
                if self._task_success
                and self.config.terminate_on_task_success
                else 0.0
            ),
        }
        reward = float(sum(reward_components.values()))
        if not math.isfinite(reward):
            raise RuntimeError("reward calculation produced a non-finite value")

        observation = self._build_observation()
        info = self._build_physical_info()
        info.update(
            {
                "raw_action": raw_action.copy(),
                "component_clipped_action": (
                    component_clipped_action.copy()
                ),
                "normalized_applied_action": (
                    normalized_applied_action.copy()
                ),
                "physical_displacement_m": physical_displacement_m.copy(),
                "applied_displacement_m": applied_displacement_m.copy(),
                "action_component_clipped": action_component_clipped,
                "action_norm_projected": action_norm_projected,
                "action_was_adjusted": action_was_adjusted,
                "boundary_violation": boundary_violation,
                "candidate_position_m": candidate_position_m.copy(),
                "applied_position_m": applied_position_m.copy(),
                "boundary_projection_distance_m": (
                    boundary_projection_distance_m
                ),
                "movement_distance_m": movement_distance_m,
                "applied_movement_distance_m": applied_movement_distance_m,
                "movement_cost": float(
                    movement_distance_m / maximum_step_distance_m
                ),
                "energy_cost": energy_cost,
                "actual_energy_consumed": actual_energy_consumed,
                "normalized_energy_cost": normalized_energy_cost,
                "energy_depleted": energy_depleted,
                "energy_depletion_cost": float(
                    self.config.energy_depletion_penalty
                    if energy_depleted
                    else 0.0
                ),
                "queue_before_bits": queue_before_bits.copy(),
                "available_data_bits": queue_before_bits.copy(),
                "preclip_queue_bits": preclip_queue_bits.copy(),
                "queue_conservation_residual_bits": (
                    conservation_residual_bits.copy()
                ),
                "payload_utility": payload_utility,
                # Kept for v0-shaped diagnostics; this is not used in v1 reward.
                "rate_utility": float(
                    service_total_rate_bps / self._max_total_rate_bps
                ),
                "service_total_rate_bps": service_total_rate_bps,
                "buoy_positions_before_drift_m": drift_before_positions_m.copy(),
                "buoy_drift_velocity_before_mps": drift_before_velocity_mps.copy(),
                "buoy_drift_reflection_axes": drift_reflection_axes.copy(),
                "boundary_cost": float(
                    self.config.boundary_penalty
                    if boundary_violation
                    else 0.0
                ),
                "reward_components": dict(reward_components),
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if self._channel_enabled():
            info.update(
                {
                    "channel_phase_before_rad": channel_phase_before_rad.copy(),
                    "channel_multiplier_before": channel_multiplier_before.copy(),
                }
            )
        self._episode_done = bool(terminated or truncated)
        return observation.copy(), reward, terminated, truncated, info

    def _channel_keyword_arguments(self) -> Dict[str, float]:
        return {
            "uav_altitude_m": self.config.uav_altitude,
            "total_bandwidth_hz": self.config.total_bandwidth_hz,
            "buoy_tx_power_w": self.config.buoy_tx_power_w,
            "reference_channel_gain": self.config.reference_channel_gain,
            "pathloss_exponent": self.config.pathloss_exponent,
            "noise_power_w": self.config.noise_power_w,
        }

    def _reset_channel_dynamics(self) -> None:
        """Initialize current-only predictable oscillator state.

        The static branch deliberately does not sample the generator, preserving
        prior static reset trajectories exactly.
        """
        if not self._channel_enabled():
            self._channel_phase_rad = np.zeros(self.num_buoys, dtype=np.float64)
            self._channel_multipliers = np.ones(self.num_buoys, dtype=np.float64)
            return
        supplied = self.config.channel_initial_phase_by_buoy
        phases = (
            np.asarray(supplied, dtype=np.float64)
            if supplied is not None
            else self._rng.uniform(-math.pi, math.pi, size=self.num_buoys)
        )
        self._channel_phase_rad = np.mod(phases, 2.0 * math.pi).astype(
            np.float64, copy=True
        )
        self._refresh_channel_multipliers()

    def _refresh_channel_multipliers(self) -> None:
        if not self._channel_enabled():
            self._channel_multipliers = np.ones(self.num_buoys, dtype=np.float64)
            return
        values = np.exp(
            float(self.config.channel_log_amplitude)
            * np.sin(self._channel_phase_rad)
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise RuntimeError("predictable channel multiplier must be finite and positive")
        self._channel_multipliers = np.asarray(values, dtype=np.float64).copy()

    def _advance_channel_dynamics(self) -> None:
        """Advance deterministic phases after service in the completed slot."""
        if not self._channel_enabled():
            return
        self._channel_phase_rad = np.mod(
            self._channel_phase_rad + self._channel_frequency_rad_per_slot,
            2.0 * math.pi,
        )
        self._refresh_channel_multipliers()

    def _reset_buoy_drift(self) -> None:
        """Initialize episode-constant or slowly varying observable drift."""
        shape = (self.config.num_buoys, _POSITION_DIM)
        if self.config.buoy_drift_mode == "static":
            self._buoy_drift_velocity_mps = np.zeros(shape, dtype=np.float64)
            self._buoy_drift_mean_velocity_mps = np.zeros(shape, dtype=np.float64)
            return
        angles = self._rng.uniform(0.0, 2.0 * math.pi, size=self.config.num_buoys)
        maximum_speed = float(
            self.config.buoy_drift_speed_fraction * self.config.max_speed
        )
        velocities = maximum_speed * np.column_stack((np.cos(angles), np.sin(angles)))
        self._buoy_drift_velocity_mps = velocities.astype(np.float64, copy=True)
        self._buoy_drift_mean_velocity_mps = velocities.astype(np.float64, copy=True)

    def _advance_buoy_drift(self) -> np.ndarray:
        """Advance buoy state after service and return reflected-axis flags."""
        shape = (self.config.num_buoys, _POSITION_DIM)
        if self.config.buoy_drift_mode == "static":
            return np.zeros(shape, dtype=np.bool_)
        prior_velocity = self._buoy_drift_velocity_mps.copy()
        positions, reflected_velocity = reflect_buoy_motion(
            self._buoy_positions_m, prior_velocity, self.config.delta_t, self.config
        )
        reflection_axes = np.not_equal(np.signbit(prior_velocity), np.signbit(reflected_velocity))
        if self.config.buoy_drift_mode == "constant":
            next_velocity = reflected_velocity
        else:
            maximum_speed = float(
                self.config.buoy_drift_speed_fraction * self.config.max_speed
            )
            innovation_scale = float(
                self.config.buoy_drift_innovation_fraction * maximum_speed
            )
            innovation = self._rng.normal(0.0, innovation_scale, size=shape)
            next_velocity = (
                self.config.buoy_drift_rho * reflected_velocity
                + (1.0 - self.config.buoy_drift_rho) * self._buoy_drift_mean_velocity_mps
                + innovation
            )
            norms = np.linalg.norm(next_velocity, axis=1, keepdims=True)
            next_velocity = np.where(
                norms > maximum_speed,
                next_velocity * (maximum_speed / np.maximum(norms, 1.0e-12)),
                next_velocity,
            )
        self._buoy_positions_m = positions.copy()
        self._buoy_drift_velocity_mps = np.asarray(next_velocity, dtype=np.float64).copy()
        return reflection_axes.astype(np.bool_, copy=True)

    def _generate_buoy_positions(self) -> np.ndarray:
        if self.config.randomize_buoys:
            x_positions = self._rng.uniform(
                self.config.buoy_margin,
                self.config.area_width - self.config.buoy_margin,
                size=self.config.num_buoys,
            )
            y_positions = self._rng.uniform(
                self.config.buoy_margin,
                self.config.area_height - self.config.buoy_margin,
                size=self.config.num_buoys,
            )
            positions = np.column_stack((x_positions, y_positions))
        else:
            positions = self._deterministic_buoy_positions()

        positions = np.asarray(positions, dtype=np.float64)
        expected_shape = (self.config.num_buoys, _POSITION_DIM)
        if positions.shape != expected_shape:
            raise RuntimeError(
                f"generated buoy positions must have shape {expected_shape}, "
                f"got {positions.shape}"
            )
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("generated buoy positions contain NaN or Inf")

        lower = self.config.buoy_margin
        upper = np.array(
            (
                self.config.area_width - self.config.buoy_margin,
                self.config.area_height - self.config.buoy_margin,
            ),
            dtype=np.float64,
        )
        if np.any(positions < lower) or np.any(positions > upper):
            raise RuntimeError(
                "generated buoy positions violate the configured buoy margin"
            )
        return positions.copy()

    def _deterministic_buoy_positions(self) -> np.ndarray:
        if self.config.num_buoys == 4 and self._default_four_layout_is_valid():
            return np.asarray(
                _DEFAULT_FOUR_BUOY_POSITIONS_M, dtype=np.float64
            ).copy()

        aspect_ratio = self.config.area_width / self.config.area_height
        num_x = int(math.ceil(math.sqrt(self.config.num_buoys * aspect_ratio)))
        num_x = max(1, min(self.config.num_buoys, num_x))
        num_y = int(math.ceil(self.config.num_buoys / num_x))
        x_coordinates = self._grid_axis_coordinates(
            self.config.area_width, self.config.buoy_margin, num_x
        )
        y_coordinates = self._grid_axis_coordinates(
            self.config.area_height, self.config.buoy_margin, num_y
        )
        positions = np.array(
            [(x, y) for x in x_coordinates for y in y_coordinates],
            dtype=np.float64,
        )
        return positions[: self.config.num_buoys].copy()

    def _default_four_layout_is_valid(self) -> bool:
        default_positions = np.asarray(
            _DEFAULT_FOUR_BUOY_POSITIONS_M, dtype=np.float64
        )
        lower = self.config.buoy_margin
        upper = np.array(
            (
                self.config.area_width - self.config.buoy_margin,
                self.config.area_height - self.config.buoy_margin,
            ),
            dtype=np.float64,
        )
        return bool(
            np.all(default_positions >= lower)
            and np.all(default_positions <= upper)
        )

    @staticmethod
    def _grid_axis_coordinates(
        extent: float, margin: float, count: int
    ) -> np.ndarray:
        if count == 1:
            return np.array((extent / 2.0,), dtype=np.float64)
        return np.linspace(margin, extent - margin, num=count, dtype=np.float64)

    def _process_action(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
        try:
            untyped_action = np.asarray(action)
        except (TypeError, ValueError) as error:
            raise ValueError("action must be a numeric array with shape (2,)") from error
        if np.iscomplexobj(untyped_action):
            raise ValueError("action must contain only real values")
        try:
            raw_action = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("action must be a numeric array with shape (2,)") from error

        if raw_action.shape != (_ACTION_DIM,):
            raise ValueError(
                f"action must have shape ({_ACTION_DIM},), got {raw_action.shape}"
            )
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("action must contain only finite values")

        raw_action = raw_action.copy()
        component_clipped_action = np.clip(raw_action, -1.0, 1.0)
        action_component_clipped = bool(
            np.any(component_clipped_action != raw_action)
        )
        clipped_norm = float(np.linalg.norm(component_clipped_action))
        action_norm_projected = bool(clipped_norm > 1.0)
        if action_norm_projected:
            normalized_applied_action = component_clipped_action / clipped_norm
        else:
            normalized_applied_action = component_clipped_action.copy()
        return (
            raw_action,
            component_clipped_action.copy(),
            normalized_applied_action.astype(np.float64, copy=True),
            action_component_clipped,
            action_norm_projected,
        )

    def _update_channel_metrics(self) -> None:
        geometric_metrics = compute_channel_metrics(
            self._uav_position_m,
            self._buoy_positions_m,
            **self._channel_keyword_arguments(),
        )
        if self._channel_enabled():
            multipliers = np.asarray(self._channel_multipliers, dtype=np.float64)
            if multipliers.shape != (self.num_buoys,):
                raise RuntimeError("channel multiplier has invalid buoy shape")
            gains = np.asarray(geometric_metrics["channel_gains"], dtype=np.float64) * multipliers
            snr = (
                float(self.config.buoy_tx_power_w) * gains
                / float(self.config.noise_power_w)
            )
            rates = (
                float(self.config.total_bandwidth_hz) / float(self.num_buoys)
                * np.log1p(snr) / math.log(2.0)
            )
            metrics: Dict[str, Any] = {
                "horizontal_distances_m": geometric_metrics["horizontal_distances_m"],
                "distances_3d_m": geometric_metrics["distances_3d_m"],
                "channel_gains": gains,
                "snr_linear": snr,
                "rates_bps": rates,
                "total_rate_bps": float(np.sum(rates, dtype=np.float64)),
            }
        else:
            # Keep the completed static calculation path intact.
            metrics = geometric_metrics
        owned_metrics: Dict[str, Any] = {}
        expected_shape = (self.config.num_buoys,)
        for key in _CHANNEL_ARRAY_KEYS:
            if key not in metrics:
                raise RuntimeError(f"channel calculation did not return {key!r}")
            values = np.asarray(metrics[key], dtype=np.float64)
            if values.shape != expected_shape:
                raise RuntimeError(
                    f"channel metric {key!r} must have shape {expected_shape}, "
                    f"got {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"channel metric {key!r} contains NaN or Inf")
            if np.any(values < 0.0):
                raise RuntimeError(f"channel metric {key!r} contains negatives")
            owned_metrics[key] = values.copy()

        if self._channel_enabled():
            owned_metrics["geometric_channel_gains"] = np.asarray(
                geometric_metrics["channel_gains"], dtype=np.float64
            ).copy()
            owned_metrics["channel_multipliers"] = np.asarray(
                self._channel_multipliers, dtype=np.float64
            ).copy()

        if "total_rate_bps" not in metrics:
            raise RuntimeError("channel calculation did not return 'total_rate_bps'")
        total_rate_bps = float(metrics["total_rate_bps"])
        if not math.isfinite(total_rate_bps) or total_rate_bps < 0.0:
            raise RuntimeError("total channel rate must be finite and non-negative")
        if not math.isclose(
            total_rate_bps,
            float(np.sum(owned_metrics["rates_bps"])),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("total channel rate does not equal the sum of rates")
        owned_metrics["total_rate_bps"] = total_rate_bps
        self._channel_metrics = owned_metrics

    def _build_observation(self) -> np.ndarray:
        if self.config.task_semantics_version == "v1":
            return self._build_observation_v1()
        position_normalized = np.array(
            (
                2.0 * self._uav_position_m[0] / self.config.area_width - 1.0,
                2.0 * self._uav_position_m[1] / self.config.area_height - 1.0,
            ),
            dtype=np.float64,
        )
        velocity_normalized = self._uav_velocity_mps / self.config.max_speed
        if self.config.initial_energy > 0.0:
            energy_normalized = (
                self._remaining_energy / self.config.initial_energy
            )
        else:
            energy_normalized = 0.0
        time_remaining = (
            self.config.episode_horizon - self._time_step
        ) / self.config.episode_horizon

        relative_positions = self._buoy_positions_m - self._uav_position_m
        relative_x = relative_positions[:, 0] / self.config.area_width
        relative_y = relative_positions[:, 1] / self.config.area_height
        distance_normalized = (
            self._channel_metrics["distances_3d_m"]
            / self._maximum_3d_distance_m
        )
        rate_normalized = (
            self._channel_metrics["rates_bps"] / self._max_single_rate_bps
        )

        self._require_array_interval(
            position_normalized, -1.0, 1.0, "normalized UAV position"
        )
        self._require_array_interval(
            velocity_normalized, -1.0, 1.0, "normalized UAV velocity"
        )
        self._require_interval(
            float(energy_normalized), 0.0, 1.0, "normalized energy"
        )
        self._require_interval(
            float(time_remaining), 0.0, 1.0, "normalized remaining time"
        )
        self._require_array_interval(
            relative_x, -1.0, 1.0, "normalized buoy relative x positions"
        )
        self._require_array_interval(
            relative_y, -1.0, 1.0, "normalized buoy relative y positions"
        )
        self._require_array_interval(
            distance_normalized, 0.0, 1.0, "normalized buoy distances"
        )
        self._require_array_interval(
            rate_normalized, 0.0, 1.0, "normalized buoy rates"
        )

        buoy_features = np.column_stack(
            (relative_x, relative_y, distance_normalized, rate_normalized)
        ).reshape(-1)
        observation = np.concatenate(
            (
                position_normalized,
                velocity_normalized,
                np.array(
                    (energy_normalized, time_remaining), dtype=np.float64
                ),
                buoy_features,
            )
        ).astype(np.float32)
        if observation.shape != (self.state_dim,):
            raise RuntimeError(
                f"observation must have shape ({self.state_dim},), "
                f"got {observation.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("observation contains NaN or Inf")
        return observation.copy()

    def _build_observation_v1(self) -> np.ndarray:
        """Build the leak-free Round-3 observation from current information."""
        position_normalized = np.array(
            (
                2.0 * self._uav_position_m[0] / self.config.area_width - 1.0,
                2.0 * self._uav_position_m[1] / self.config.area_height - 1.0,
            ),
            dtype=np.float64,
        )
        velocity_normalized = self._uav_velocity_mps / self.config.max_speed
        remaining_horizon_fraction = float(
            (self.config.episode_horizon - self._time_step)
            / self.config.episode_horizon
        )
        maximum_episode_distance = float(
            self.config.episode_horizon
            * self.config.max_speed
            * self.config.delta_t
        )
        normalized_distance = float(
            self._cumulative_distance_m / maximum_episode_distance
        )
        normalized_energy_consumption = float(
            self._cumulative_energy_consumption / self.config.initial_energy
            if self.config.initial_energy > 0.0
            else 0.0
        )

        relative_positions = self._buoy_positions_m - self._uav_position_m
        relative_x = relative_positions[:, 0] / self.config.area_width
        relative_y = relative_positions[:, 1] / self.config.area_height
        distance_normalized = (
            self._channel_metrics["distances_3d_m"]
            / self._maximum_3d_distance_m
        )
        rate_normalized = (
            self._channel_metrics["rates_bps"] / self._max_single_rate_bps
        )
        queue_normalized = (
            self._queue_bits / self.config.queue_capacity_bits
        )
        arrival_normalized = self._last_arrival_bits / (
            self._last_arrival_bits + self.config.queue_capacity_bits
        )
        global_features = np.array(
            (
                position_normalized[0],
                position_normalized[1],
                velocity_normalized[0],
                velocity_normalized[1],
                self._remaining_energy_ratio(),
                remaining_horizon_fraction,
                normalized_distance,
                normalized_energy_consumption,
                self._task_completion_ratio,
            ),
            dtype=np.float64,
        )
        buoy_columns = [relative_x, relative_y]
        if self.config.buoy_drift_mode != "static":
            maximum_drift_speed = float(
                self.config.buoy_drift_speed_fraction * self.config.max_speed
            )
            drift_normalized = self._buoy_drift_velocity_mps / maximum_drift_speed
            buoy_columns.extend([drift_normalized[:, 0], drift_normalized[:, 1]])
        if self._channel_enabled():
            frequency_normalized = float(
                self._channel_frequency_rad_per_slot
                / self.config.channel_frequency_normalization_max_rad_per_slot
            )
            self._require_interval(
                frequency_normalized, 0.0, 1.0, "normalized channel frequency"
            )
            buoy_columns.extend(
                [
                    np.sin(self._channel_phase_rad),
                    np.cos(self._channel_phase_rad),
                    np.full(self.num_buoys, frequency_normalized, dtype=np.float64),
                ]
            )
        buoy_columns.extend(
            [
                distance_normalized,
                rate_normalized,
                queue_normalized,
                arrival_normalized,
                self._service_completion_ratio,
            ]
        )
        buoy_features = np.column_stack(tuple(buoy_columns)).reshape(-1)
        observation = np.concatenate((global_features, buoy_features))
        self._require_array_interval(
            observation[:4], -1.0, 1.0, "v1 signed global features"
        )
        self._require_array_interval(
            observation[4:], -1.0, 1.0, "v1 observation features"
        )
        # Every feature after velocity is either [0,1] or a per-node relative
        # coordinate. Check the structured form to avoid accepting negatives in
        # queue/arrival/completion fields.
        per_buoy = self._v1_per_buoy_feature_dim()
        nodes = observation[_V1_BASE_STATE_DIM:].reshape(self.num_buoys, per_buoy)
        nonnegative_start = 2
        if self.config.buoy_drift_mode != "static":
            nonnegative_start += 2
        if self._channel_enabled():
            nonnegative_start += 3
        self._require_array_interval(
            nodes[:, nonnegative_start:], 0.0, 1.0, "v1 nonnegative buoy features"
        )
        self._require_array_interval(
            observation[4:_V1_BASE_STATE_DIM],
            0.0,
            1.0,
            "v1 nonnegative global features",
        )
        result = observation.astype(np.float32)
        if result.shape != (self.state_dim,):
            raise RuntimeError(
                f"v1 observation must have shape ({self.state_dim},)"
            )
        if not self.observation_space.contains(result):
            raise RuntimeError("v1 observation violates observation_space")
        return result.copy()

    def _build_physical_info(self) -> Dict[str, Any]:
        info = {
            "time_step": int(self._time_step),
            "uav_position_m": self._uav_position_m.copy(),
            "uav_velocity_mps": self._uav_velocity_mps.copy(),
            "uav_altitude_m": float(self.config.uav_altitude),
            "remaining_energy": float(self._remaining_energy),
            "buoy_positions_m": self._buoy_positions_m.copy(),
            "buoy_drift_velocity_mps": self._buoy_drift_velocity_mps.copy(),
            "buoy_drift_mode": str(self.config.buoy_drift_mode),
            "buoy_drift_speed_max_mps": float(
                self.config.buoy_drift_speed_fraction * self.config.max_speed
            ),
            "horizontal_distances_m": self._channel_metrics[
                "horizontal_distances_m"
            ].copy(),
            "distances_3d_m": self._channel_metrics["distances_3d_m"].copy(),
            "rates_bps": self._channel_metrics["rates_bps"].copy(),
            "total_rate_bps": float(self._channel_metrics["total_rate_bps"]),
            "channel_gains": self._channel_metrics["channel_gains"].copy(),
            "snr_linear": self._channel_metrics["snr_linear"].copy(),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "num_buoys": int(self.num_buoys),
        }
        if self._channel_enabled():
            info.update(
                {
                    "channel_variation_mode": str(self.config.channel_variation_mode),
                    "channel_log_amplitude": float(self.config.channel_log_amplitude),
                    "channel_period_slots": int(self.config.channel_period_slots),
                    "channel_phase_rad": self._channel_phase_rad.copy(),
                    "channel_phase_sin": np.sin(self._channel_phase_rad).copy(),
                    "channel_phase_cos": np.cos(self._channel_phase_rad).copy(),
                    "channel_frequency_rad_per_slot": float(
                        self._channel_frequency_rad_per_slot
                    ),
                    "channel_frequency_normalized": float(
                        self._channel_frequency_rad_per_slot
                        / self.config.channel_frequency_normalization_max_rad_per_slot
                    ),
                    "channel_multipliers": self._channel_multipliers.copy(),
                    "geometric_channel_gains": self._channel_metrics[
                        "geometric_channel_gains"
                    ].copy(),
                }
            )
        if self.config.task_semantics_version == "v1":
            queue_normalized = (
                self._queue_bits / self.config.queue_capacity_bits
            )
            arrival_normalized = self._last_arrival_bits / (
                self._last_arrival_bits + self.config.queue_capacity_bits
            )
            normalized_position = np.array(
                (
                    2.0
                    * self._uav_position_m[0]
                    / self.config.area_width
                    - 1.0,
                    2.0
                    * self._uav_position_m[1]
                    / self.config.area_height
                    - 1.0,
                ),
                dtype=np.float64,
            )
            desired_goal = np.array(
                (
                    self.config.target_task_completion_ratio,
                    self.config.maximum_allowed_residual_queue_ratio,
                    self.config.minimum_terminal_energy_ratio,
                ),
                dtype=np.float64,
            )
            info.update(
                {
                    "task_semantics_version": "v1",
                    "dataset_schema_version": (
                        "v1_channel_diagnostic" if self._channel_enabled()
                        and self.config.buoy_drift_mode == "static"
                        else "v1_drift_channel_diagnostic" if self._channel_enabled()
                        else "v1" if self.config.buoy_drift_mode == "static"
                        else "v1_drift_diagnostic"
                    ),
                    "goal_spec_version": "v1",
                    "observation_schema": self.observation_schema,
                    "observation_space_shape": self.observation_space.shape,
                    "raw_queue_bits": self._queue_bits.copy(),
                    "queue_values": self._queue_bits.copy(),
                    "remaining_queue": self._queue_bits.copy(),
                    "normalized_queue": queue_normalized.copy(),
                    "raw_arrival_bits": self._last_arrival_bits.copy(),
                    "normalized_arrival": arrival_normalized.copy(),
                    "instantaneous_rate_bps": self._channel_metrics[
                        "rates_bps"
                    ].copy(),
                    "allocated_rate_bps": self._channel_metrics[
                        "rates_bps"
                    ].copy(),
                    "bandwidth_allocation_fraction": np.full(
                        self.num_buoys,
                        1.0 / self.num_buoys,
                        dtype=np.float64,
                    ),
                    "link_capacity_bits": (
                        self._last_link_capacity_bits.copy()
                    ),
                    "transmitted_payload_bits": (
                        self._last_transmitted_payload_bits.copy()
                    ),
                    "dropped_arrival_bits": (
                        self._last_dropped_arrival_bits.copy()
                    ),
                    "service_completion_ratio": (
                        self._service_completion_ratio.copy()
                    ),
                    "valid_node_mask": np.ones(
                        self.num_buoys, dtype=np.bool_
                    ),
                    "service_eligible_mask": (
                        self._queue_bits > 0.0
                    ).astype(np.bool_),
                    "normalized_position": normalized_position.copy(),
                    "raw_position_m": self._uav_position_m.copy(),
                    "area_bounds_m": np.array(
                        (
                            self.config.area_width,
                            self.config.area_height,
                        ),
                        dtype=np.float64,
                    ),
                    "max_step_distance_m": float(
                        self.config.max_speed * self.config.delta_t
                    ),
                    "remaining_energy_ratio": self._remaining_energy_ratio(),
                    "raw_remaining_energy": float(self._remaining_energy),
                    "cumulative_distance_m": float(
                        self._cumulative_distance_m
                    ),
                    "cumulative_energy_consumption": float(
                        self._cumulative_energy_consumption
                    ),
                    "cumulative_arrival_bits": float(
                        np.sum(self._cumulative_arrival_bits)
                    ),
                    "cumulative_arrival_bits_per_buoy": (
                        self._cumulative_arrival_bits.copy()
                    ),
                    "cumulative_payload_bits": float(
                        np.sum(self._cumulative_payload_bits_by_buoy)
                    ),
                    "cumulative_payload_bits_per_buoy": (
                        self._cumulative_payload_bits_by_buoy.copy()
                    ),
                    "cumulative_dropped_bits": float(
                        np.sum(self._cumulative_dropped_bits)
                    ),
                    "cumulative_dropped_bits_per_buoy": (
                        self._cumulative_dropped_bits.copy()
                    ),
                    "elapsed_fraction": float(
                        self._time_step / self.config.episode_horizon
                    ),
                    "remaining_horizon_fraction": float(
                        (
                            self.config.episode_horizon
                            - self._time_step
                        )
                        / self.config.episode_horizon
                    ),
                    "task_completion_ratio": float(
                        self._task_completion_ratio
                    ),
                    "residual_queue_ratio": self._residual_queue_ratio(),
                    "task_success": bool(self._task_success),
                    "terminal_reason": str(self._terminal_reason),
                    "arrival_trace_id": str(self._arrival_trace_id),
                    "arrival_mode": str(self.config.arrival_mode),
                    "queue_capacity_bits": float(
                        self.config.queue_capacity_bits
                    ),
                    "initial_queue_bits": self._initial_queue_bits.copy(),
                    "desired_goal": desired_goal,
                    "desired_goal_field_names": (
                        "target_task_completion_ratio",
                        "maximum_allowed_residual_queue_ratio",
                        "minimum_terminal_energy_ratio",
                    ),
                    "communication_energy_available": False,
                    "communication_energy_status": (
                        "not defined by the current project energy model"
                    ),
                    "service_allocation_semantics": (
                        "fixed parallel orthogonal 1/K bandwidth shares are "
                        "already included in rates_bps"
                    ),
                    "one_step_movement_energy_proxy": (
                        self._one_step_movement_energy_proxy()
                    ),
                }
            )
        else:
            info["task_semantics_version"] = "v0"
        return info

    def _completion_ratios(self) -> np.ndarray:
        offered = self._initial_queue_bits + self._cumulative_arrival_bits
        completed = np.ones(self.num_buoys, dtype=np.float64)
        nonzero = offered > 0.0
        completed[nonzero] = (
            self._cumulative_payload_bits_by_buoy[nonzero] / offered[nonzero]
        )
        return np.clip(completed, 0.0, 1.0)

    def _total_completion_ratio(self) -> float:
        offered = float(
            np.sum(self._initial_queue_bits + self._cumulative_arrival_bits)
        )
        if offered <= 0.0:
            return 1.0
        return float(
            np.clip(
                np.sum(self._cumulative_payload_bits_by_buoy) / offered,
                0.0,
                1.0,
            )
        )

    def _residual_queue_ratio(self) -> float:
        denominator = float(
            self.num_buoys * self.config.queue_capacity_bits
        )
        return float(np.sum(self._queue_bits) / denominator)

    def _remaining_energy_ratio(self) -> float:
        if self.config.initial_energy <= 0.0:
            return 0.0
        return float(
            np.clip(
                self._remaining_energy / self.config.initial_energy,
                0.0,
                1.0,
            )
        )

    def _one_step_movement_energy_proxy(self) -> np.ndarray:
        """Return current, normalized candidate movement-energy proxies."""
        horizontal = np.asarray(
            self._channel_metrics["horizontal_distances_m"], dtype=np.float64
        )
        maximum_distance = self.config.max_speed * self.config.delta_t
        reachable_distance = np.minimum(horizontal, maximum_distance)
        costs = (
            self.config.hover_energy_per_step
            + self.config.move_energy_per_meter * reachable_distance
        )
        maximum_cost = (
            self.config.hover_energy_per_step
            + self.config.move_energy_per_meter * maximum_distance
        )
        if maximum_cost <= 0.0:
            return np.zeros(self.num_buoys, dtype=np.float64)
        return np.clip(costs / maximum_cost, 0.0, 1.0)

    def _ensure_step_is_available(self) -> None:
        if not self._has_reset:
            raise RuntimeError("step() called before reset(); call reset() first")
        if self._episode_done:
            raise RuntimeError(
                "step() called after the episode ended; call reset() first"
            )

    @staticmethod
    def _require_positive_finite_scalar(value: Any, name: str) -> float:
        scalar = float(value)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise RuntimeError(f"{name} must be finite and positive, got {scalar}")
        return scalar

    @staticmethod
    def _require_interval(
        value: float, lower: float, upper: float, name: str
    ) -> None:
        if not math.isfinite(value):
            raise RuntimeError(f"{name} must be finite")
        if (
            value < lower - _NORMALIZATION_TOLERANCE
            or value > upper + _NORMALIZATION_TOLERANCE
        ):
            raise RuntimeError(
                f"{name} must be in [{lower}, {upper}], got {value}"
            )

    @classmethod
    def _require_array_interval(
        cls,
        values: np.ndarray,
        lower: float,
        upper: float,
        name: str,
    ) -> None:
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} must contain only finite values")
        if np.any(values < lower - _NORMALIZATION_TOLERANCE) or np.any(
            values > upper + _NORMALIZATION_TOLERANCE
        ):
            raise RuntimeError(
                f"{name} must contain values in [{lower}, {upper}]"
            )
