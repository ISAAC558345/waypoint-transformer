"""Semantically structured, variable-scale maritime episode trajectories.

The round-02 collector stores complete episodes rather than a shuffled table of
transitions.  This keeps terminal boundaries explicit and lets later code form
``(s_t, s_{t+K})`` pairs without crossing episodes.

The current environment does *not* define queues, actual delivered payload,
action masks, or an episode desired goal.  Those fields are represented by
zero-width arrays and explicit availability flags; no synthetic physical
quantity is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TRAJECTORY_SCHEMA_VERSION = "1.0"
GOAL_SPEC_VERSION = "v0"
ACHIEVED_GOAL_NAMES = (
    "final_uav_x_m",
    "final_uav_y_m",
    "final_remaining_energy",
)


def default_observation_schema(num_buoys: int) -> Dict[str, Any]:
    """Return the documented observation layout for a variable buoy count."""
    if isinstance(num_buoys, bool) or not isinstance(num_buoys, Integral):
        raise ValueError("num_buoys must be an integer")
    validated_num_buoys = int(num_buoys)
    if validated_num_buoys < 1:
        raise ValueError("num_buoys must be positive")
    return {
        "schema_version": "maritime_observation_v1",
        "state_dim": 6 + 4 * validated_num_buoys,
        "num_buoys": validated_num_buoys,
        "flattening_order": "base features, then four features per buoy",
        "segments": [
            {
                "name": "uav_position_normalized",
                "start": 0,
                "stop": 2,
                "normalization": "[2*x/area_width-1, 2*y/area_height-1]",
            },
            {
                "name": "uav_velocity_normalized",
                "start": 2,
                "stop": 4,
                "normalization": "uav_velocity_mps/max_speed",
            },
            {
                "name": "remaining_energy_normalized",
                "start": 4,
                "stop": 5,
                "normalization": "remaining_energy/initial_energy",
            },
            {
                "name": "remaining_time_normalized",
                "start": 5,
                "stop": 6,
                "normalization": "(episode_horizon-time_step)/episode_horizon",
            },
            {
                "name": "buoy_features",
                "start": 6,
                "stop": 6 + 4 * validated_num_buoys,
                "shape_before_flatten": [validated_num_buoys, 4],
                "feature_names": [
                    "relative_x_over_area_width",
                    "relative_y_over_area_height",
                    "distance_3d_over_maximum_3d_distance",
                    "rate_over_theoretical_max_single_buoy_rate",
                ],
            },
        ],
        "raw_counterparts": {
            "uav_position_normalized": "uav_positions_m",
            "uav_velocity_normalized": "uav_velocities_mps",
            "remaining_energy_normalized": "remaining_energy",
            "buoy_features": [
                "buoy_positions_m",
                "distances_3d_m",
                "rates_bps",
            ],
        },
        "next_observation_raw_counterparts": {
            "uav_position_normalized": "next_uav_positions_m",
            "uav_velocity_normalized": "next_uav_velocities_mps",
            "remaining_energy_normalized": "next_remaining_energy",
            "buoy_features": [
                "next_buoy_positions_m",
                "next_distances_3d_m",
                "next_rates_bps",
            ],
        },
    }


@dataclass
class EpisodeTrajectory:
    """One complete, unpadded episode with semantically named arrays.

    All leading dimensions equal ``episode_length``.  Per-buoy fields use the
    episode's own ``num_buoys``; different episodes may therefore have
    different shapes and are deliberately saved in separate NPZ files.
    """

    episode_id: str
    policy_id: str
    seed: int
    scenario_id: str
    num_buoys: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    time_steps: np.ndarray
    valid_transition_mask: np.ndarray
    valid_node_mask: np.ndarray
    action_mask: np.ndarray
    action_mask_available: bool
    uav_positions_m: np.ndarray
    next_uav_positions_m: np.ndarray
    uav_velocities_mps: np.ndarray
    next_uav_velocities_mps: np.ndarray
    buoy_positions_m: np.ndarray
    next_buoy_positions_m: np.ndarray
    remaining_energy: np.ndarray
    next_remaining_energy: np.ndarray
    horizontal_distances_m: np.ndarray
    next_horizontal_distances_m: np.ndarray
    distances_3d_m: np.ndarray
    next_distances_3d_m: np.ndarray
    channel_gains: np.ndarray
    next_channel_gains: np.ndarray
    snr_linear: np.ndarray
    next_snr_linear: np.ndarray
    rates_bps: np.ndarray
    next_rates_bps: np.ndarray
    total_rate_bps: np.ndarray
    next_total_rate_bps: np.ndarray
    raw_actions: np.ndarray
    component_clipped_actions: np.ndarray
    normalized_applied_actions: np.ndarray
    physical_displacements_m: np.ndarray
    applied_displacements_m: np.ndarray
    candidate_positions_m: np.ndarray
    applied_positions_m: np.ndarray
    action_component_clipped: np.ndarray
    action_norm_projected: np.ndarray
    boundary_violation: np.ndarray
    constraint_violation: np.ndarray
    boundary_projection_distance_m: np.ndarray
    commanded_movement_distance_m: np.ndarray
    movement_distance_m: np.ndarray
    rate_utility: np.ndarray
    movement_cost: np.ndarray
    boundary_cost: np.ndarray
    energy_cost: np.ndarray
    actual_energy_consumed: np.ndarray
    energy_depleted: np.ndarray
    energy_depletion_cost: np.ndarray
    reward_components: np.ndarray
    queue_values: np.ndarray
    queue_values_available: bool
    transmitted_payload_bits: np.ndarray
    transmitted_payload_available: bool
    desired_goal: np.ndarray
    desired_goal_available: bool
    achieved_goal: np.ndarray
    achieved_goal_available: bool
    episode_return: float
    episode_length: int
    terminal_reason: str
    observation_schema: Dict[str, Any] = field(default_factory=dict)
    goal_spec_version: str = GOAL_SPEC_VERSION
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate shape, boundary, availability, and finite-value invariants."""
        self.episode_id = _nonempty_string("episode_id", self.episode_id)
        self.policy_id = _nonempty_string("policy_id", self.policy_id)
        self.scenario_id = _nonempty_string("scenario_id", self.scenario_id)
        self.terminal_reason = _nonempty_string(
            "terminal_reason", self.terminal_reason
        )
        self.seed = _nonnegative_integer("seed", self.seed)
        self.num_buoys = _positive_integer("num_buoys", self.num_buoys)
        self.episode_length = _positive_integer(
            "episode_length", self.episode_length
        )
        if self.goal_spec_version != GOAL_SPEC_VERSION:
            raise ValueError(
                "round-02 trajectories require goal_spec_version='v0'"
            )
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported trajectory schema_version "
                f"{self.schema_version!r}"
            )
        if not isinstance(self.observation_schema, dict):
            raise TypeError("observation_schema must be a dictionary")
        if not self.observation_schema:
            self.observation_schema = default_observation_schema(self.num_buoys)
        self.observation_schema = _canonical_metadata(self.observation_schema)
        schema_num_buoys = self.observation_schema.get("num_buoys")
        if (
            schema_num_buoys is not None
            and _positive_integer(
                "observation_schema.num_buoys", schema_num_buoys
            )
            != self.num_buoys
        ):
            raise ValueError(
                "observation_schema num_buoys contradicts episode metadata"
            )

        t_count = self.episode_length
        self.observations = _finite_array(
            "observations", self.observations, np.float32, 2
        )
        state_dim = self.observations.shape[1]
        expected_state_dim = _positive_integer(
            "observation_schema.state_dim",
            self.observation_schema.get("state_dim"),
        )
        if state_dim != expected_state_dim:
            raise ValueError(
                f"observations state dimension must be {expected_state_dim}, "
                f"got {state_dim}"
            )
        self.next_observations = _finite_array(
            "next_observations", self.next_observations, np.float32, 2
        )
        self.actions = _finite_array("actions", self.actions, np.float64, 2)
        action_dim = self.actions.shape[1]
        if action_dim < 1:
            raise ValueError("actions must have at least one column")
        self.rewards = _finite_array("rewards", self.rewards, np.float64, 1)

        self.terminated = _bool_array("terminated", self.terminated, 1)
        self.truncated = _bool_array("truncated", self.truncated, 1)
        self.valid_transition_mask = _bool_array(
            "valid_transition_mask", self.valid_transition_mask, 1
        )
        self.valid_node_mask = _bool_array(
            "valid_node_mask", self.valid_node_mask, 2
        )
        self.action_mask = _bool_array("action_mask", self.action_mask, 2)
        self.time_steps = np.asarray(self.time_steps, dtype=np.int64)

        exact_shapes = {
            "observations": (t_count, state_dim),
            "next_observations": (t_count, state_dim),
            "actions": (t_count, action_dim),
            "rewards": (t_count,),
            "terminated": (t_count,),
            "truncated": (t_count,),
            "time_steps": (t_count,),
            "valid_transition_mask": (t_count,),
            "valid_node_mask": (t_count, self.num_buoys),
        }
        for name, shape in exact_shapes.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not np.array_equal(self.time_steps, np.arange(t_count)):
            raise ValueError("time_steps must be contiguous integers [0, T)")
        if not np.all(self.valid_transition_mask):
            raise ValueError(
                "unpadded per-episode storage requires every transition to be valid"
            )
        if self.action_mask_available:
            if self.action_mask.shape[0] != t_count or self.action_mask.shape[1] < 1:
                raise ValueError(
                    "available action_mask must have shape [T, positive_width]"
                )
        elif self.action_mask.shape != (t_count, 0):
            raise ValueError(
                "unavailable action_mask must be represented by shape [T, 0]"
            )

        action_vector_names = (
            "raw_actions",
            "component_clipped_actions",
            "normalized_applied_actions",
        )
        for name in action_vector_names:
            value = _finite_array(name, getattr(self, name), np.float64, 2)
            if value.shape != (t_count, action_dim):
                raise ValueError(
                    f"{name} must have shape {(t_count, action_dim)}, "
                    f"got {value.shape}"
                )
            setattr(self, name, value)

        vector_2_names = (
            "uav_positions_m",
            "next_uav_positions_m",
            "uav_velocities_mps",
            "next_uav_velocities_mps",
            "physical_displacements_m",
            "applied_displacements_m",
            "candidate_positions_m",
            "applied_positions_m",
        )
        for name in vector_2_names:
            value = _finite_array(name, getattr(self, name), np.float64, 2)
            if value.shape != (t_count, 2):
                raise ValueError(
                    f"{name} must have shape {(t_count, 2)}, got {value.shape}"
                )
            setattr(self, name, value)

        buoy_matrix_names = ("buoy_positions_m", "next_buoy_positions_m")
        for name in buoy_matrix_names:
            value = _finite_array(name, getattr(self, name), np.float64, 3)
            expected = (t_count, self.num_buoys, 2)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            setattr(self, name, value)

        buoy_vector_names = (
            "horizontal_distances_m",
            "next_horizontal_distances_m",
            "distances_3d_m",
            "next_distances_3d_m",
            "channel_gains",
            "next_channel_gains",
            "snr_linear",
            "next_snr_linear",
            "rates_bps",
            "next_rates_bps",
        )
        for name in buoy_vector_names:
            value = _finite_array(name, getattr(self, name), np.float64, 2)
            expected = (t_count, self.num_buoys)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            setattr(self, name, value)

        scalar_float_names = (
            "remaining_energy",
            "next_remaining_energy",
            "total_rate_bps",
            "next_total_rate_bps",
            "boundary_projection_distance_m",
            "commanded_movement_distance_m",
            "movement_distance_m",
            "rate_utility",
            "movement_cost",
            "boundary_cost",
            "energy_cost",
            "actual_energy_consumed",
            "energy_depletion_cost",
        )
        for name in scalar_float_names:
            value = _finite_array(name, getattr(self, name), np.float64, 1)
            if value.shape != (t_count,):
                raise ValueError(
                    f"{name} must have shape {(t_count,)}, got {value.shape}"
                )
            setattr(self, name, value)

        scalar_bool_names = (
            "action_component_clipped",
            "action_norm_projected",
            "boundary_violation",
            "constraint_violation",
            "energy_depleted",
        )
        for name in scalar_bool_names:
            value = _bool_array(name, getattr(self, name), 1)
            if value.shape != (t_count,):
                raise ValueError(
                    f"{name} must have shape {(t_count,)}, got {value.shape}"
                )
            setattr(self, name, value)

        self.reward_components = _finite_array(
            "reward_components", self.reward_components, np.float64, 2
        )
        if self.reward_components.shape != (t_count, 4):
            raise ValueError(
                "reward_components must have shape [T, 4] in the documented order"
            )
        reward_sums = np.sum(self.reward_components, axis=1, dtype=np.float64)
        if not np.allclose(
            reward_sums, self.rewards, rtol=1.0e-12, atol=1.0e-12
        ):
            raise ValueError(
                "reward_components must sum to the stored reward at every step"
            )
        if not np.allclose(
            np.sum(self.rates_bps, axis=1, dtype=np.float64),
            self.total_rate_bps,
            rtol=1.0e-12,
            atol=1.0e-6,
        ):
            raise ValueError(
                "total_rate_bps must equal the sum of per-buoy rates"
            )
        if not np.allclose(
            np.sum(self.next_rates_bps, axis=1, dtype=np.float64),
            self.next_total_rate_bps,
            rtol=1.0e-12,
            atol=1.0e-6,
        ):
            raise ValueError(
                "next_total_rate_bps must equal the sum of next per-buoy rates"
            )
        expected_constraint = np.logical_or.reduce(
            (
                self.boundary_violation,
                self.action_component_clipped,
                self.action_norm_projected,
            )
        )
        if not np.array_equal(
            self.constraint_violation, expected_constraint
        ):
            raise ValueError(
                "constraint_violation must combine boundary and action flags"
            )
        self.queue_values = _finite_array(
            "queue_values", self.queue_values, np.float64, 2
        )
        self.transmitted_payload_bits = _finite_array(
            "transmitted_payload_bits",
            self.transmitted_payload_bits,
            np.float64,
            2,
        )
        if self.queue_values_available:
            raise ValueError(
                "queue_values must remain unavailable in the current environment"
            )
        if self.transmitted_payload_available:
            raise ValueError(
                "actual transmitted payload must remain unavailable in the "
                "current environment"
            )
        _validate_unavailable_matrix(
            "queue_values", self.queue_values, False, t_count
        )
        _validate_unavailable_matrix(
            "transmitted_payload_bits",
            self.transmitted_payload_bits,
            False,
            t_count,
        )

        self.desired_goal = _finite_array(
            "desired_goal", self.desired_goal, np.float64, 1
        )
        if self.desired_goal_available or self.desired_goal.shape != (0,):
            raise ValueError(
                "goal_spec v0 has no desired goal; use an unavailable [0] array"
            )
        self.achieved_goal = _finite_array(
            "achieved_goal", self.achieved_goal, np.float64, 1
        )
        if not self.achieved_goal_available or self.achieved_goal.shape != (3,):
            raise ValueError(
                "achieved_goal must contain final x, final y, and energy"
            )
        expected_achieved = np.concatenate(
            (self.next_uav_positions_m[-1], self.next_remaining_energy[-1:])
        )
        if not np.array_equal(self.achieved_goal, expected_achieved):
            raise ValueError(
                "achieved_goal must be derived exactly from the final physical state"
            )

        self.episode_return = float(self.episode_return)
        if not math.isfinite(self.episode_return):
            raise ValueError("episode_return must be finite")
        summed_return = float(np.sum(self.rewards, dtype=np.float64))
        if not math.isclose(
            self.episode_return, summed_return, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError("episode_return must equal the sum of rewards")
        if np.any(self.terminated[:-1]) or np.any(self.truncated[:-1]):
            raise ValueError("done flags may occur only on the final transition")
        if self.terminal_reason in (
            "terminated_energy_depleted",
            "terminated_and_truncated",
        ) and not bool(self.terminated[-1]):
            raise ValueError("terminal_reason contradicts terminated flag")
        if self.terminal_reason in (
            "truncated_horizon",
            "terminated_and_truncated",
        ) and not bool(self.truncated[-1]):
            raise ValueError("terminal_reason contradicts truncated flag")

    @property
    def state_dim(self) -> int:
        """Return this episode's flattened observation dimension."""
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        """Return this episode's action dimension."""
        return int(self.actions.shape[1])

    @property
    def final_info(self) -> Dict[str, Any]:
        """Return the final physical state used to derive ``achieved_goal``."""
        return {
            "uav_position_m": self.next_uav_positions_m[-1].copy(),
            "uav_velocity_mps": self.next_uav_velocities_mps[-1].copy(),
            "remaining_energy": float(self.next_remaining_energy[-1]),
            "buoy_positions_m": self.next_buoy_positions_m[-1].copy(),
            "rates_bps": self.next_rates_bps[-1].copy(),
            "total_rate_bps": float(self.next_total_rate_bps[-1]),
            "terminated": bool(self.terminated[-1]),
            "truncated": bool(self.truncated[-1]),
        }

    def arrays(self) -> Dict[str, np.ndarray]:
        """Return copied array fields suitable for ``numpy.savez_compressed``."""
        result: Dict[str, np.ndarray] = {}
        for name in _ARRAY_FIELD_NAMES:
            result[name] = np.asarray(getattr(self, name)).copy()
        return result


class EpisodeTrajectoryBuilder:
    """Incrementally construct one :class:`EpisodeTrajectory` from a rollout."""

    def __init__(
        self,
        episode_id: str,
        policy_id: str,
        seed: int,
        scenario_id: str,
        num_buoys: int,
        observation_schema: Optional[Mapping[str, Any]] = None,
        initial_info: Optional[Mapping[str, Any]] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> None:
        self.episode_id = _nonempty_string("episode_id", episode_id)
        self.policy_id = _nonempty_string("policy_id", policy_id)
        self.seed = _nonnegative_integer("seed", seed)
        self.scenario_id = _nonempty_string("scenario_id", scenario_id)
        self.num_buoys = _positive_integer("num_buoys", num_buoys)
        self.observation_schema = _canonical_metadata(
            observation_schema
            if observation_schema is not None
            else default_observation_schema(self.num_buoys)
        )
        schema_num_buoys = self.observation_schema.get("num_buoys")
        if (
            schema_num_buoys is not None
            and _positive_integer(
                "observation_schema.num_buoys", schema_num_buoys
            )
            != self.num_buoys
        ):
            raise ValueError(
                "observation_schema num_buoys contradicts builder metadata"
            )
        self._initial_info = (
            _copy_info(initial_info) if initial_info is not None else None
        )
        schema_state_dim = self.observation_schema.get("state_dim")
        info_state_dim = (
            self._initial_info.get("state_dim")
            if self._initial_info is not None
            else None
        )
        info_action_dim = (
            self._initial_info.get("action_dim")
            if self._initial_info is not None
            else None
        )
        self.state_dim = _resolve_dimension(
            "state_dim", state_dim, info_state_dim, schema_state_dim
        )
        self.action_dim = _resolve_dimension(
            "action_dim", action_dim, info_action_dim
        )
        if (
            schema_state_dim is not None
            and int(schema_state_dim) != self.state_dim
        ):
            raise ValueError(
                "observation_schema state_dim contradicts resolved state_dim"
            )
        self.observation_schema["state_dim"] = self.state_dim
        self._last_info = self._initial_info
        self._records: List[Dict[str, Any]] = []
        self._done = False
        self._action_mask_width: Optional[int] = None

    def append(
        self,
        observation: Any,
        action: Any,
        reward: Any,
        next_observation: Any,
        terminated: Any,
        truncated: Any,
        info_before: Optional[Mapping[str, Any]] = None,
        info_after: Optional[Mapping[str, Any]] = None,
        valid_node_mask: Optional[Any] = None,
        action_mask: Optional[Any] = None,
    ) -> None:
        """Append one transition while preserving pre/post physical semantics."""
        if self._done:
            raise RuntimeError("cannot append after a terminated/truncated transition")
        before = (
            _copy_info(info_before)
            if info_before is not None
            else _copy_info(self._last_info)
            if self._last_info is not None
            else None
        )
        if before is None:
            raise ValueError(
                "info_before or initial_info is required for the first transition"
            )
        if info_after is None:
            raise ValueError("info_after from MaritimeEnv.step is required")
        after = _copy_info(info_after)
        _validate_info_scale(before, self.num_buoys, "info_before")
        _validate_info_scale(after, self.num_buoys, "info_after")
        for label, info in (("info_before", before), ("info_after", after)):
            if int(info.get("state_dim", self.state_dim)) != self.state_dim:
                raise ValueError(f"{label} state_dim changed within the episode")
            if int(info.get("action_dim", self.action_dim)) != self.action_dim:
                raise ValueError(f"{label} action_dim changed within the episode")

        obs = _one_dimensional_finite("observation", observation, np.float32)
        next_obs = _one_dimensional_finite(
            "next_observation", next_observation, np.float32
        )
        if obs.shape != (self.state_dim,) or next_obs.shape != (
            self.state_dim,
        ):
            raise ValueError(
                f"observations must have shape ({self.state_dim},)"
            )
        act = _one_dimensional_finite("action", action, np.float64)
        if act.shape != (self.action_dim,):
            raise ValueError(
                f"action must have shape ({self.action_dim},)"
            )
        scalar_reward = float(reward)
        if not math.isfinite(scalar_reward):
            raise ValueError("reward must be finite")
        terminated_flag = _strict_bool("terminated", terminated)
        truncated_flag = _strict_bool("truncated", truncated)

        if valid_node_mask is None:
            node_mask = np.ones(self.num_buoys, dtype=np.bool_)
        else:
            node_mask = _bool_array(
                "valid_node_mask", valid_node_mask, 1
            )
            if node_mask.shape != (self.num_buoys,):
                raise ValueError(
                    f"valid_node_mask must have shape ({self.num_buoys},)"
                )
        if action_mask is None:
            act_mask = np.empty((0,), dtype=np.bool_)
        else:
            act_mask = _bool_array("action_mask", action_mask, 1)
            if act_mask.ndim != 1 or act_mask.size < 1:
                raise ValueError("action_mask must be a non-empty one-dimensional array")
        if self._action_mask_width is None:
            self._action_mask_width = int(act_mask.size)
        elif self._action_mask_width != int(act_mask.size):
            raise ValueError("action_mask width must remain constant within an episode")

        self._records.append(
            {
                "observation": obs,
                "action": act,
                "reward": scalar_reward,
                "next_observation": next_obs,
                "terminated": terminated_flag,
                "truncated": truncated_flag,
                "info_before": before,
                "info_after": after,
                "valid_node_mask": node_mask.copy(),
                "action_mask": act_mask.copy(),
            }
        )
        self._last_info = after
        self._done = terminated_flag or truncated_flag

    def finalize(
        self, terminal_reason: Optional[str] = None
    ) -> EpisodeTrajectory:
        """Validate and materialize the accumulated variable-length episode."""
        if not self._records:
            raise ValueError("cannot finalize an empty episode")
        last = self._records[-1]
        if terminal_reason is None:
            terminal_reason = _infer_terminal_reason(
                bool(last["terminated"]), bool(last["truncated"])
            )
        t_count = len(self._records)
        before_infos = [record["info_before"] for record in self._records]
        after_infos = [record["info_after"] for record in self._records]

        kwargs: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "policy_id": self.policy_id,
            "seed": self.seed,
            "scenario_id": self.scenario_id,
            "num_buoys": self.num_buoys,
            "observations": _stack_records(self._records, "observation"),
            "actions": _stack_records(self._records, "action"),
            "rewards": np.asarray(
                [record["reward"] for record in self._records], dtype=np.float64
            ),
            "next_observations": _stack_records(
                self._records, "next_observation"
            ),
            "terminated": np.asarray(
                [record["terminated"] for record in self._records], dtype=np.bool_
            ),
            "truncated": np.asarray(
                [record["truncated"] for record in self._records], dtype=np.bool_
            ),
            "time_steps": np.arange(t_count, dtype=np.int64),
            "valid_transition_mask": np.ones(t_count, dtype=np.bool_),
            "valid_node_mask": _stack_records(
                self._records, "valid_node_mask"
            ),
            "action_mask": _stack_records(self._records, "action_mask"),
            "action_mask_available": bool(
                self._action_mask_width is not None
                and self._action_mask_width > 0
            ),
            "uav_positions_m": _stack_info(before_infos, "uav_position_m"),
            "next_uav_positions_m": _stack_info(
                after_infos, "uav_position_m"
            ),
            "uav_velocities_mps": _stack_info(
                before_infos, "uav_velocity_mps"
            ),
            "next_uav_velocities_mps": _stack_info(
                after_infos, "uav_velocity_mps"
            ),
            "buoy_positions_m": _stack_info(before_infos, "buoy_positions_m"),
            "next_buoy_positions_m": _stack_info(
                after_infos, "buoy_positions_m"
            ),
            "remaining_energy": _scalar_info(before_infos, "remaining_energy"),
            "next_remaining_energy": _scalar_info(
                after_infos, "remaining_energy"
            ),
            "horizontal_distances_m": _stack_info(
                before_infos, "horizontal_distances_m"
            ),
            "next_horizontal_distances_m": _stack_info(
                after_infos, "horizontal_distances_m"
            ),
            "distances_3d_m": _stack_info(before_infos, "distances_3d_m"),
            "next_distances_3d_m": _stack_info(
                after_infos, "distances_3d_m"
            ),
            "channel_gains": _stack_info(before_infos, "channel_gains"),
            "next_channel_gains": _stack_info(after_infos, "channel_gains"),
            "snr_linear": _stack_info(before_infos, "snr_linear"),
            "next_snr_linear": _stack_info(after_infos, "snr_linear"),
            "rates_bps": _stack_info(before_infos, "rates_bps"),
            "next_rates_bps": _stack_info(after_infos, "rates_bps"),
            "total_rate_bps": _scalar_info(before_infos, "total_rate_bps"),
            "next_total_rate_bps": _scalar_info(
                after_infos, "total_rate_bps"
            ),
            "raw_actions": _stack_info(after_infos, "raw_action"),
            "component_clipped_actions": _stack_info(
                after_infos, "component_clipped_action"
            ),
            "normalized_applied_actions": _stack_info(
                after_infos, "normalized_applied_action"
            ),
            "physical_displacements_m": _stack_info(
                after_infos, "physical_displacement_m"
            ),
            "applied_displacements_m": _stack_info(
                after_infos, "applied_displacement_m"
            ),
            "candidate_positions_m": _stack_info(
                after_infos, "candidate_position_m"
            ),
            "applied_positions_m": _stack_info(
                after_infos, "applied_position_m"
            ),
            "action_component_clipped": _bool_info(
                after_infos, "action_component_clipped"
            ),
            "action_norm_projected": _bool_info(
                after_infos, "action_norm_projected"
            ),
            "boundary_violation": _bool_info(
                after_infos, "boundary_violation"
            ),
            "constraint_violation": np.logical_or.reduce(
                (
                    _bool_info(after_infos, "boundary_violation"),
                    _bool_info(after_infos, "action_component_clipped"),
                    _bool_info(after_infos, "action_norm_projected"),
                )
            ),
            "boundary_projection_distance_m": _scalar_info(
                after_infos, "boundary_projection_distance_m"
            ),
            "commanded_movement_distance_m": _scalar_info(
                after_infos, "movement_distance_m"
            ),
            # True path length uses the applied post-boundary displacement.
            "movement_distance_m": _scalar_info(
                after_infos, "applied_movement_distance_m"
            ),
            "rate_utility": _scalar_info(after_infos, "rate_utility"),
            "movement_cost": _scalar_info(after_infos, "movement_cost"),
            "boundary_cost": _scalar_info(after_infos, "boundary_cost"),
            "energy_cost": _scalar_info(after_infos, "energy_cost"),
            "actual_energy_consumed": _scalar_info(
                after_infos, "actual_energy_consumed"
            ),
            "energy_depleted": _bool_info(after_infos, "energy_depleted"),
            "energy_depletion_cost": _scalar_info(
                after_infos, "energy_depletion_cost"
            ),
            "reward_components": _reward_components(after_infos),
            "queue_values": np.empty((t_count, 0), dtype=np.float64),
            "queue_values_available": False,
            "transmitted_payload_bits": np.empty(
                (t_count, 0), dtype=np.float64
            ),
            "transmitted_payload_available": False,
            "desired_goal": np.empty((0,), dtype=np.float64),
            "desired_goal_available": False,
            "achieved_goal": np.concatenate(
                (
                    np.asarray(after_infos[-1]["uav_position_m"], dtype=np.float64),
                    np.array(
                        (float(after_infos[-1]["remaining_energy"]),),
                        dtype=np.float64,
                    ),
                )
            ),
            "achieved_goal_available": True,
            "episode_return": float(
                np.sum(
                    [record["reward"] for record in self._records],
                    dtype=np.float64,
                )
            ),
            "episode_length": t_count,
            "terminal_reason": terminal_reason,
            "observation_schema": dict(self.observation_schema),
        }
        return EpisodeTrajectory(**kwargs)


def trajectory_from_episode_dict(record: Mapping[str, Any]) -> EpisodeTrajectory:
    """Adapt a runner's generic episode dictionary to the canonical structure."""
    required = (
        "episode_id",
        "policy_id",
        "seed",
        "scenario_id",
        "num_buoys",
        "transitions",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"episode record is missing keys: {missing}")
    builder = EpisodeTrajectoryBuilder(
        episode_id=record["episode_id"],
        policy_id=record["policy_id"],
        seed=record["seed"],
        scenario_id=record["scenario_id"],
        num_buoys=record["num_buoys"],
        observation_schema=record.get("observation_schema"),
        initial_info=record.get("initial_info"),
    )
    transitions = record["transitions"]
    if not isinstance(transitions, Sequence):
        raise TypeError("transitions must be a sequence")
    for transition in transitions:
        if not isinstance(transition, Mapping):
            raise TypeError("each transition must be a mapping")
        builder.append(
            observation=transition["observation"],
            action=transition["action"],
            reward=transition["reward"],
            next_observation=transition["next_observation"],
            terminated=transition["terminated"],
            truncated=transition["truncated"],
            info_before=transition.get("info_before"),
            info_after=transition.get("info_after", transition.get("info")),
            valid_node_mask=transition.get("valid_node_mask"),
            action_mask=transition.get("action_mask"),
        )
    return builder.finalize(record.get("terminal_reason"))


def trajectories_equal(
    left: EpisodeTrajectory, right: EpisodeTrajectory
) -> bool:
    """Return whether two episodes are exactly equal, including array dtypes."""
    scalar_names = (
        "episode_id",
        "policy_id",
        "seed",
        "scenario_id",
        "num_buoys",
        "action_mask_available",
        "queue_values_available",
        "transmitted_payload_available",
        "desired_goal_available",
        "achieved_goal_available",
        "episode_return",
        "episode_length",
        "terminal_reason",
        "goal_spec_version",
        "schema_version",
        "observation_schema",
    )
    for name in scalar_names:
        if getattr(left, name) != getattr(right, name):
            return False
    for name in _ARRAY_FIELD_NAMES:
        left_array = np.asarray(getattr(left, name))
        right_array = np.asarray(getattr(right, name))
        if left_array.dtype != right_array.dtype:
            return False
        if not np.array_equal(left_array, right_array):
            return False
    return True


_ARRAY_FIELD_NAMES = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
    "time_steps",
    "valid_transition_mask",
    "valid_node_mask",
    "action_mask",
    "uav_positions_m",
    "next_uav_positions_m",
    "uav_velocities_mps",
    "next_uav_velocities_mps",
    "buoy_positions_m",
    "next_buoy_positions_m",
    "remaining_energy",
    "next_remaining_energy",
    "horizontal_distances_m",
    "next_horizontal_distances_m",
    "distances_3d_m",
    "next_distances_3d_m",
    "channel_gains",
    "next_channel_gains",
    "snr_linear",
    "next_snr_linear",
    "rates_bps",
    "next_rates_bps",
    "total_rate_bps",
    "next_total_rate_bps",
    "raw_actions",
    "component_clipped_actions",
    "normalized_applied_actions",
    "physical_displacements_m",
    "applied_displacements_m",
    "candidate_positions_m",
    "applied_positions_m",
    "action_component_clipped",
    "action_norm_projected",
    "boundary_violation",
    "constraint_violation",
    "boundary_projection_distance_m",
    "commanded_movement_distance_m",
    "movement_distance_m",
    "rate_utility",
    "movement_cost",
    "boundary_cost",
    "energy_cost",
    "actual_energy_consumed",
    "energy_depleted",
    "energy_depletion_cost",
    "reward_components",
    "queue_values",
    "transmitted_payload_bits",
    "desired_goal",
    "achieved_goal",
)


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _resolve_dimension(name: str, *candidates: Optional[Any]) -> int:
    present = [
        _positive_integer(name, value)
        for value in candidates
        if value is not None
    ]
    if not present:
        raise ValueError(
            f"{name} must be provided by explicit metadata, reset info, or schema"
        )
    if any(value != present[0] for value in present[1:]):
        raise ValueError(f"conflicting {name} metadata: {present}")
    return present[0]


def _canonical_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical_metadata(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("observation_schema contains NaN or Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "observation_schema contains unsupported value "
        f"{type(value).__name__}"
    )


def _finite_array(
    name: str, value: Any, dtype: Any, ndim: int
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def _bool_array(name: str, value: Any, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {raw.ndim}")
    if raw.dtype.kind not in ("b",):
        if not np.all(np.logical_or(raw == 0, raw == 1)):
            raise ValueError(f"{name} must contain only boolean values")
    return np.asarray(raw, dtype=np.bool_).copy()


def _one_dimensional_finite(
    name: str, value: Any, dtype: Any
) -> np.ndarray:
    return _finite_array(name, value, dtype, 1)


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _validate_unavailable_matrix(
    name: str, value: np.ndarray, available: bool, t_count: int
) -> None:
    if not isinstance(available, (bool, np.bool_)):
        raise TypeError(f"{name}_available must be boolean")
    if value.shape[0] != t_count:
        raise ValueError(f"{name} leading dimension must be T")
    if available:
        if value.shape[1] < 1:
            raise ValueError(f"available {name} must have positive width")
    elif value.shape != (t_count, 0):
        raise ValueError(
            f"unavailable {name} must be represented by shape [T, 0]"
        )


def _copy_info(info: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if info is None:
        raise ValueError("physical info is required")
    result: Dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            result[str(key)] = value.copy()
        elif isinstance(value, Mapping):
            result[str(key)] = dict(value)
        else:
            result[str(key)] = value
    return result


def _validate_info_scale(
    info: Mapping[str, Any], num_buoys: int, label: str
) -> None:
    if int(info.get("num_buoys", num_buoys)) != num_buoys:
        raise ValueError(f"{label} num_buoys contradicts episode metadata")
    required = (
        "uav_position_m",
        "uav_velocity_mps",
        "remaining_energy",
        "buoy_positions_m",
        "horizontal_distances_m",
        "distances_3d_m",
        "channel_gains",
        "snr_linear",
        "rates_bps",
        "total_rate_bps",
    )
    missing = [name for name in required if name not in info]
    if missing:
        raise ValueError(f"{label} is missing physical keys: {missing}")
    expected_buoy_vector = (num_buoys,)
    for name in (
        "horizontal_distances_m",
        "distances_3d_m",
        "channel_gains",
        "snr_linear",
        "rates_bps",
    ):
        if np.asarray(info[name]).shape != expected_buoy_vector:
            raise ValueError(
                f"{label}[{name!r}] must have shape {expected_buoy_vector}"
            )
    if np.asarray(info["buoy_positions_m"]).shape != (num_buoys, 2):
        raise ValueError(
            f"{label}['buoy_positions_m'] must have shape {(num_buoys, 2)}"
        )


def _stack_records(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.stack([np.asarray(record[key]) for record in records], axis=0)


def _stack_info(infos: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    missing = [index for index, info in enumerate(infos) if key not in info]
    if missing:
        raise ValueError(f"step info is missing {key!r} at indices {missing}")
    return np.stack([np.asarray(info[key]) for info in infos], axis=0)


def _scalar_info(infos: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    missing = [index for index, info in enumerate(infos) if key not in info]
    if missing:
        raise ValueError(f"step info is missing {key!r} at indices {missing}")
    values = np.asarray([info[key] for info in infos], dtype=np.float64)
    if values.shape != (len(infos),):
        raise ValueError(f"step info {key!r} must be scalar")
    return values


def _bool_info(infos: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    missing = [index for index, info in enumerate(infos) if key not in info]
    if missing:
        raise ValueError(f"step info is missing {key!r} at indices {missing}")
    return np.asarray([bool(info[key]) for info in infos], dtype=np.bool_)


def _reward_components(infos: Sequence[Mapping[str, Any]]) -> np.ndarray:
    order = (
        "rate_reward",
        "movement_penalty",
        "boundary_penalty",
        "energy_depletion_penalty",
    )
    rows: List[List[float]] = []
    for index, info in enumerate(infos):
        components = info.get("reward_components")
        if not isinstance(components, Mapping):
            raise ValueError(
                f"step info reward_components missing at index {index}"
            )
        rows.append([float(components[name]) for name in order])
    return np.asarray(rows, dtype=np.float64)


def _infer_terminal_reason(terminated: bool, truncated: bool) -> str:
    if terminated and truncated:
        return "terminated_and_truncated"
    if terminated:
        return "terminated_energy_depleted"
    if truncated:
        return "truncated_horizon"
    return "collector_limit"


__all__ = [
    "ACHIEVED_GOAL_NAMES",
    "EpisodeTrajectory",
    "EpisodeTrajectoryBuilder",
    "GOAL_SPEC_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "default_observation_schema",
    "trajectories_equal",
    "trajectory_from_episode_dict",
]
