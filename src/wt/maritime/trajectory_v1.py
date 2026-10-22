"""Strict schema-v1 trajectories for the maritime queueing task.

This module is intentionally separate from :mod:`wt.maritime.trajectory`.
The latter is the immutable round-02/v0 compatibility path, while this module
stores the real queue, arrival, capacity, delivered-payload, completion, and
goal semantics introduced in round 03.  No v0 trajectory can be silently
adapted to this representation.

Slot alignment follows the documented service-before-arrival convention:

``D_t <= min(Q_t, C_t)`` and
``Q_(t+1) = clip(Q_t - D_t + A_t, 0, Q_max)``.

Every per-step array has leading dimension ``T``.  The two goal arrays are
episode-level values and are deliberately not broadcast over transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TRAJECTORY_SCHEMA_VERSION_V1 = "v1"
GOAL_SPEC_VERSION_V1 = "v1"

DESIRED_GOAL_NAMES_V1 = (
    "target_task_completion_ratio",
    "maximum_allowed_residual_queue_ratio",
    "minimum_terminal_energy_ratio",
)
ACHIEVED_GOAL_GLOBAL_NAMES_V1 = (
    "final_task_completion_ratio",
    "final_residual_queue_ratio",
    "final_energy_ratio",
    "final_uav_x_normalized",
    "final_uav_y_normalized",
)
REWARD_COMPONENT_NAMES_V1 = (
    "payload_reward",
    "energy_penalty",
    "boundary_penalty",
    "invalid_action_penalty",
    "energy_depletion_penalty",
    "unfinished_queue_penalty",
    "task_success_bonus",
)

_FLOAT_ATOL = 1.0e-6
_FLOAT_RTOL = 1.0e-10


@dataclass
class MaritimeEpisodeV1:
    """One complete, unpadded schema-v1 maritime episode.

    ``num_buoys`` is the number of real node columns in this episode.  Scale
    padding belongs in the later supervised-sample builder, not in this
    per-episode physical record.
    """

    # Stable identity and episode metadata.
    episode_id: str
    policy_id: str
    seed: int
    scenario_id: str
    num_buoys: int
    arrival_trace_id: str
    terminal_reason: str

    # Transition core.
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    time_steps: np.ndarray
    valid_transition_mask: np.ndarray
    valid_node_mask: np.ndarray
    next_valid_node_mask: np.ndarray

    # Physical state.
    uav_positions_m: np.ndarray
    next_uav_positions_m: np.ndarray
    buoy_positions_m: np.ndarray
    rates_bps: np.ndarray
    next_rates_bps: np.ndarray
    remaining_energy: np.ndarray
    next_remaining_energy: np.ndarray

    # Queueing/task state and transition quantities.
    raw_queue_bits: np.ndarray
    next_raw_queue_bits: np.ndarray
    raw_arrival_bits: np.ndarray
    link_capacity_bits: np.ndarray
    transmitted_payload_bits: np.ndarray
    dropped_arrival_bits: np.ndarray
    service_completion_ratio: np.ndarray
    next_service_completion_ratio: np.ndarray
    cumulative_distance_m: np.ndarray
    next_cumulative_distance_m: np.ndarray
    cumulative_energy_consumed: np.ndarray
    next_cumulative_energy_consumed: np.ndarray
    cumulative_payload_bits: np.ndarray
    next_cumulative_payload_bits: np.ndarray
    task_completion_ratio: np.ndarray
    next_task_completion_ratio: np.ndarray

    # Diagnostics.
    boundary_violation: np.ndarray
    action_was_adjusted: np.ndarray
    policy_selected_buoy_index: np.ndarray
    reward_components: np.ndarray

    # Episode-level goals.
    desired_goal: np.ndarray
    achieved_goal_global: np.ndarray
    achieved_goal_per_buoy_queue_ratio: np.ndarray
    achieved_goal_per_buoy_completion: np.ndarray

    # Configuration needed to interpret normalization exactly.
    area_width: float
    area_height: float
    initial_energy: float
    queue_capacity_bits: float

    observation_schema: Dict[str, Any] = field(default_factory=dict)
    episode_return: Optional[float] = None
    episode_length: Optional[int] = None
    goal_spec_version: str = GOAL_SPEC_VERSION_V1
    schema_version: str = TRAJECTORY_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        """Canonicalize arrays and enforce the v1 physical invariants."""
        self.episode_id = _nonempty_string("episode_id", self.episode_id)
        self.policy_id = _nonempty_string("policy_id", self.policy_id)
        self.scenario_id = _nonempty_string("scenario_id", self.scenario_id)
        self.arrival_trace_id = _nonempty_string(
            "arrival_trace_id", self.arrival_trace_id
        )
        self.terminal_reason = _nonempty_string(
            "terminal_reason", self.terminal_reason
        )
        self.seed = _nonnegative_integer("seed", self.seed)
        self.num_buoys = _positive_integer("num_buoys", self.num_buoys)
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION_V1:
            raise ValueError(
                "MaritimeEpisodeV1 requires schema_version='v1'; "
                "v0 data must use the round-02 loader"
            )
        if self.goal_spec_version != GOAL_SPEC_VERSION_V1:
            raise ValueError(
                "MaritimeEpisodeV1 requires goal_spec_version='v1'"
            )

        self.area_width = _positive_finite("area_width", self.area_width)
        self.area_height = _positive_finite("area_height", self.area_height)
        self.initial_energy = _positive_finite(
            "initial_energy", self.initial_energy
        )
        self.queue_capacity_bits = _positive_finite(
            "queue_capacity_bits", self.queue_capacity_bits
        )

        self.observation_schema = _canonical_metadata(
            self.observation_schema
        )
        if not isinstance(self.observation_schema, dict):
            raise TypeError("observation_schema must be a dictionary")

        # Core arrays establish T, S, and A.
        self.observations = _finite_array(
            "observations", self.observations, np.float32, 2
        )
        if self.observations.shape[0] < 1 or self.observations.shape[1] < 1:
            raise ValueError("observations must have shape [positive T, positive S]")
        t_count, state_dim = self.observations.shape
        self.next_observations = _finite_array(
            "next_observations", self.next_observations, np.float32, 2
        )
        self.actions = _finite_array("actions", self.actions, np.float64, 2)
        if self.actions.shape[1] < 1:
            raise ValueError("actions must have at least one column")
        action_dim = int(self.actions.shape[1])
        self.rewards = _finite_array("rewards", self.rewards, np.float64, 1)
        self.terminated = _bool_array("terminated", self.terminated, 1)
        self.truncated = _bool_array("truncated", self.truncated, 1)
        self.valid_transition_mask = _bool_array(
            "valid_transition_mask", self.valid_transition_mask, 1
        )
        self.valid_node_mask = _bool_array(
            "valid_node_mask", self.valid_node_mask, 2
        )
        self.next_valid_node_mask = _bool_array(
            "next_valid_node_mask", self.next_valid_node_mask, 2
        )
        self.time_steps = np.asarray(self.time_steps, dtype=np.int64)
        if self.time_steps.ndim != 1:
            raise ValueError("time_steps must be one-dimensional")

        exact_core_shapes = {
            "next_observations": (t_count, state_dim),
            "actions": (t_count, action_dim),
            "rewards": (t_count,),
            "terminated": (t_count,),
            "truncated": (t_count,),
            "time_steps": (t_count,),
            "valid_transition_mask": (t_count,),
            "valid_node_mask": (t_count, self.num_buoys),
            "next_valid_node_mask": (t_count, self.num_buoys),
        }
        for name, expected in exact_core_shapes.items():
            actual = getattr(self, name).shape
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")
        if not np.array_equal(self.time_steps, np.arange(t_count, dtype=np.int64)):
            raise ValueError("time_steps must be contiguous integers [0, T)")
        if not np.all(self.valid_transition_mask):
            raise ValueError(
                "per-episode v1 storage is unpadded; all transitions must be valid"
            )
        if np.any(np.logical_and(self.terminated, self.truncated)[:-1]):
            raise ValueError("done flags may not occur before the final transition")
        if np.any(self.terminated[:-1]) or np.any(self.truncated[:-1]):
            raise ValueError("terminated/truncated may only be true at the end")
        if not (bool(self.terminated[-1]) or bool(self.truncated[-1])):
            raise ValueError("a complete episode must end terminated or truncated")
        if self.terminal_reason == "ongoing":
            raise ValueError("a complete episode cannot have terminal_reason='ongoing'")
        if self.terminal_reason in ("horizon", "horizon_reached"):
            if not bool(self.truncated[-1]):
                raise ValueError("horizon terminal_reason requires truncation")
        if (
            self.terminal_reason.startswith("task_success")
            or self.terminal_reason in ("energy_depleted", "terminated")
        ) and not bool(self.terminated[-1]):
            raise ValueError(
                f"{self.terminal_reason!r} terminal_reason requires termination"
            )

        schema_state_dim = self.observation_schema.get("state_dim")
        if schema_state_dim is not None and _positive_integer(
            "observation_schema.state_dim", schema_state_dim
        ) != state_dim:
            raise ValueError("observation_schema state_dim contradicts arrays")
        schema_action_dim = self.observation_schema.get("action_dim")
        if schema_action_dim is not None and _positive_integer(
            "observation_schema.action_dim", schema_action_dim
        ) != action_dim:
            raise ValueError("observation_schema action_dim contradicts arrays")
        schema_num_buoys = self.observation_schema.get("num_buoys")
        if schema_num_buoys is not None and _positive_integer(
            "observation_schema.num_buoys", schema_num_buoys
        ) != self.num_buoys:
            raise ValueError("observation_schema num_buoys contradicts metadata")
        self.observation_schema.setdefault("schema_version", "v1")
        self.observation_schema.setdefault("state_dim", state_dim)
        self.observation_schema.setdefault("action_dim", action_dim)
        self.observation_schema.setdefault("num_buoys", self.num_buoys)
        self.observation_schema.setdefault(
            "reward_component_names", list(REWARD_COMPONENT_NAMES_V1)
        )
        if self.observation_schema.get("schema_version") != "v1":
            raise ValueError(
                "v1 trajectories require observation_schema.schema_version='v1'"
            )

        # Physical matrices/vectors.
        for name in ("uav_positions_m", "next_uav_positions_m"):
            value = _finite_array(name, getattr(self, name), np.float64, 2)
            expected = (t_count, 2)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if (
                np.any(value[:, 0] < -_FLOAT_ATOL)
                or np.any(value[:, 0] > self.area_width + _FLOAT_ATOL)
                or np.any(value[:, 1] < -_FLOAT_ATOL)
                or np.any(value[:, 1] > self.area_height + _FLOAT_ATOL)
            ):
                raise ValueError(f"{name} lies outside the configured area")
            setattr(self, name, value)
        self.buoy_positions_m = _finite_array(
            "buoy_positions_m", self.buoy_positions_m, np.float64, 3
        )
        expected_buoy_positions = (t_count, self.num_buoys, 2)
        if self.buoy_positions_m.shape != expected_buoy_positions:
            raise ValueError(
                "buoy_positions_m must have shape "
                f"{expected_buoy_positions}, got {self.buoy_positions_m.shape}"
            )
        if (
            np.any(self.buoy_positions_m[:, :, 0] < -_FLOAT_ATOL)
            or np.any(
                self.buoy_positions_m[:, :, 0]
                > self.area_width + _FLOAT_ATOL
            )
            or np.any(self.buoy_positions_m[:, :, 1] < -_FLOAT_ATOL)
            or np.any(
                self.buoy_positions_m[:, :, 1]
                > self.area_height + _FLOAT_ATOL
            )
        ):
            raise ValueError("buoy_positions_m lies outside the configured area")
        for name in ("rates_bps", "next_rates_bps"):
            value = _finite_array(name, getattr(self, name), np.float64, 2)
            expected = (t_count, self.num_buoys)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if np.any(value < -_FLOAT_ATOL):
                raise ValueError(f"{name} must be nonnegative")
            setattr(self, name, value)
        for name in ("remaining_energy", "next_remaining_energy"):
            value = _finite_array(name, getattr(self, name), np.float64, 1)
            if value.shape != (t_count,):
                raise ValueError(f"{name} must have shape {(t_count,)}")
            if np.any(value < -_FLOAT_ATOL) or np.any(
                value > self.initial_energy + _FLOAT_ATOL
            ):
                raise ValueError(
                    f"{name} must lie in [0, initial_energy]"
                )
            setattr(self, name, value)

        per_node_names = (
            "raw_queue_bits",
            "next_raw_queue_bits",
            "raw_arrival_bits",
            "link_capacity_bits",
            "transmitted_payload_bits",
            "dropped_arrival_bits",
            "service_completion_ratio",
            "next_service_completion_ratio",
        )
        for name in per_node_names:
            value = _finite_array(name, getattr(self, name), np.float64, 2)
            expected = (t_count, self.num_buoys)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            setattr(self, name, value)

        for name in (
            "raw_queue_bits",
            "next_raw_queue_bits",
            "raw_arrival_bits",
            "link_capacity_bits",
            "transmitted_payload_bits",
            "dropped_arrival_bits",
        ):
            if np.any(getattr(self, name) < -_FLOAT_ATOL):
                raise ValueError(f"{name} must be nonnegative")
        for name in ("raw_queue_bits", "next_raw_queue_bits"):
            if np.any(getattr(self, name) > self.queue_capacity_bits + _FLOAT_ATOL):
                raise ValueError(f"{name} exceeds queue_capacity_bits")
        for name in ("service_completion_ratio", "next_service_completion_ratio"):
            _require_array_interval(name, getattr(self, name), 0.0, 1.0)

        scalar_series = (
            "cumulative_distance_m",
            "next_cumulative_distance_m",
            "cumulative_energy_consumed",
            "next_cumulative_energy_consumed",
            "cumulative_payload_bits",
            "next_cumulative_payload_bits",
            "task_completion_ratio",
            "next_task_completion_ratio",
        )
        for name in scalar_series:
            value = _finite_array(name, getattr(self, name), np.float64, 1)
            if value.shape != (t_count,):
                raise ValueError(f"{name} must have shape {(t_count,)}")
            setattr(self, name, value)
        for name in (
            "cumulative_distance_m",
            "next_cumulative_distance_m",
            "cumulative_energy_consumed",
            "next_cumulative_energy_consumed",
            "cumulative_payload_bits",
            "next_cumulative_payload_bits",
        ):
            if np.any(getattr(self, name) < -_FLOAT_ATOL):
                raise ValueError(f"{name} must be nonnegative")
        for name in ("task_completion_ratio", "next_task_completion_ratio"):
            _require_array_interval(name, getattr(self, name), 0.0, 1.0)

        self.boundary_violation = _bool_array(
            "boundary_violation", self.boundary_violation, 1
        )
        self.action_was_adjusted = _bool_array(
            "action_was_adjusted", self.action_was_adjusted, 1
        )
        self.policy_selected_buoy_index = np.asarray(
            self.policy_selected_buoy_index, dtype=np.int64
        )
        for name in (
            "boundary_violation",
            "action_was_adjusted",
            "policy_selected_buoy_index",
        ):
            if getattr(self, name).shape != (t_count,):
                raise ValueError(f"{name} must have shape {(t_count,)}")
        selected = self.policy_selected_buoy_index
        if np.any(selected < -1) or np.any(selected >= self.num_buoys):
            raise ValueError(
                "policy_selected_buoy_index must be -1 or a valid buoy index"
            )
        selected_rows = np.flatnonzero(selected >= 0)
        if selected_rows.size and np.any(
            np.logical_not(
                self.valid_node_mask[selected_rows, selected[selected_rows]]
            )
        ):
            raise ValueError("policy selected an invalid node")

        self.reward_components = _finite_array(
            "reward_components", self.reward_components, np.float64, 2
        )
        expected_reward_shape = (t_count, len(REWARD_COMPONENT_NAMES_V1))
        if self.reward_components.shape != expected_reward_shape:
            raise ValueError(
                "reward_components must have shape "
                f"{expected_reward_shape} in order {REWARD_COMPONENT_NAMES_V1}"
            )
        if not np.allclose(
            np.sum(self.reward_components, axis=1, dtype=np.float64),
            self.rewards,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("reward_components must sum exactly to rewards")

        self.desired_goal = _finite_array(
            "desired_goal", self.desired_goal, np.float64, 1
        )
        if self.desired_goal.shape != (3,):
            raise ValueError("desired_goal must have shape [3]")
        _require_array_interval("desired_goal", self.desired_goal, 0.0, 1.0)
        self.achieved_goal_global = _finite_array(
            "achieved_goal_global",
            self.achieved_goal_global,
            np.float64,
            1,
        )
        if self.achieved_goal_global.shape != (5,):
            raise ValueError("achieved_goal_global must have shape [5]")
        _require_array_interval(
            "achieved_goal_global[:3]",
            self.achieved_goal_global[:3],
            0.0,
            1.0,
        )
        _require_array_interval(
            "achieved_goal_global[3:]",
            self.achieved_goal_global[3:],
            -1.0,
            1.0,
        )
        self.achieved_goal_per_buoy_queue_ratio = _finite_array(
            "achieved_goal_per_buoy_queue_ratio",
            self.achieved_goal_per_buoy_queue_ratio,
            np.float64,
            1,
        )
        self.achieved_goal_per_buoy_completion = _finite_array(
            "achieved_goal_per_buoy_completion",
            self.achieved_goal_per_buoy_completion,
            np.float64,
            1,
        )
        for name in (
            "achieved_goal_per_buoy_queue_ratio",
            "achieved_goal_per_buoy_completion",
        ):
            if getattr(self, name).shape != (self.num_buoys,):
                raise ValueError(
                    f"{name} must have shape {(self.num_buoys,)}"
                )
            _require_array_interval(name, getattr(self, name), 0.0, 1.0)

        self.episode_length = (
            t_count
            if self.episode_length is None
            else _positive_integer("episode_length", self.episode_length)
        )
        if self.episode_length != t_count:
            raise ValueError("episode_length contradicts array leading dimensions")
        summed_return = float(np.sum(self.rewards, dtype=np.float64))
        self.episode_return = (
            summed_return
            if self.episode_return is None
            else _finite_scalar("episode_return", self.episode_return)
        )
        if not math.isclose(
            self.episode_return, summed_return, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError("episode_return must equal the sum of rewards")

        self._validate_physical_semantics()
        self._validate_achieved_goal()

    @property
    def state_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def identity(self) -> Dict[str, Any]:
        """Return the stable episode identity saved in JSON metadata."""
        return {
            "episode_id": self.episode_id,
            "policy_id": self.policy_id,
            "seed": self.seed,
            "scenario_id": self.scenario_id,
            "num_buoys": self.num_buoys,
        }

    @property
    def achieved_goal(self) -> np.ndarray:
        """Compatibility shorthand for the five global achieved-goal values."""
        return self.achieved_goal_global.copy()

    def arrays(self) -> Dict[str, np.ndarray]:
        """Return independent copies suitable for safe NPZ persistence."""
        return {
            name: np.asarray(getattr(self, name)).copy()
            for name in _ARRAY_FIELD_NAMES_V1
        }

    def _validate_physical_semantics(self) -> None:
        """Audit service-before-arrival conservation and cumulative counters."""
        valid = np.logical_and(self.valid_node_mask, self.next_valid_node_mask)
        invalid = np.logical_not(valid)
        for name in (
            "raw_queue_bits",
            "next_raw_queue_bits",
            "raw_arrival_bits",
            "link_capacity_bits",
            "transmitted_payload_bits",
            "dropped_arrival_bits",
            "service_completion_ratio",
            "next_service_completion_ratio",
        ):
            values = getattr(self, name)
            if np.any(np.abs(values[invalid]) > _FLOAT_ATOL):
                raise ValueError(f"{name} must be zero at invalid node positions")

        payload = self.transmitted_payload_bits
        if np.any(
            payload > self.raw_queue_bits + _FLOAT_ATOL
        ):
            raise ValueError(
                "transmitted payload exceeds service-before-arrival queue Q_t"
            )
        if np.any(payload > self.link_capacity_bits + _FLOAT_ATOL):
            raise ValueError("transmitted payload exceeds link capacity")

        unclipped = (
            self.raw_queue_bits
            - payload
            + self.raw_arrival_bits
        )
        expected_next = np.clip(
            unclipped, 0.0, self.queue_capacity_bits
        )
        expected_drop = np.maximum(
            unclipped - self.queue_capacity_bits, 0.0
        )
        if not np.allclose(
            self.next_raw_queue_bits,
            expected_next,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError(
                "queue conservation failed: expected "
                "clip(Q_t-D_t+A_t, 0, Q_max)"
            )
        if not np.allclose(
            self.dropped_arrival_bits,
            expected_drop,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError("dropped_arrival_bits contradict queue overflow")

        expected_payload_delta = np.sum(payload, axis=1, dtype=np.float64)
        if not np.allclose(
            self.next_cumulative_payload_bits - self.cumulative_payload_bits,
            expected_payload_delta,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError(
                "cumulative payload increments must equal actual transmitted payload"
            )
        expected_distance_delta = np.linalg.norm(
            self.next_uav_positions_m - self.uav_positions_m, axis=1
        )
        if not np.allclose(
            self.next_cumulative_distance_m - self.cumulative_distance_m,
            expected_distance_delta,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError(
                "cumulative distance increments must equal applied UAV displacement"
            )
        expected_energy_delta = self.remaining_energy - self.next_remaining_energy
        if np.any(expected_energy_delta < -_FLOAT_ATOL):
            raise ValueError("remaining energy may not increase")
        if not np.allclose(
            (
                self.next_cumulative_energy_consumed
                - self.cumulative_energy_consumed
            ),
            expected_energy_delta,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError(
                "cumulative energy increments must equal remaining-energy decrease"
            )

    def _validate_achieved_goal(self) -> None:
        final_mask = self.next_valid_node_mask[-1]
        valid_count = int(np.count_nonzero(final_mask))
        if valid_count < 1:
            raise ValueError("final state must contain at least one valid buoy")
        final_queue_ratio = np.where(
            final_mask,
            self.next_raw_queue_bits[-1] / self.queue_capacity_bits,
            0.0,
        )
        final_completion = np.where(
            final_mask, self.next_service_completion_ratio[-1], 0.0
        )
        residual_ratio = float(
            np.sum(final_queue_ratio, dtype=np.float64) / valid_count
        )
        final_position = self.next_uav_positions_m[-1]
        expected_global = np.array(
            (
                self.next_task_completion_ratio[-1],
                residual_ratio,
                self.next_remaining_energy[-1] / self.initial_energy,
                2.0 * final_position[0] / self.area_width - 1.0,
                2.0 * final_position[1] / self.area_height - 1.0,
            ),
            dtype=np.float64,
        )
        if not np.allclose(
            self.achieved_goal_global,
            expected_global,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError(
                "achieved_goal_global is not derived from the final state"
            )
        if not np.allclose(
            self.achieved_goal_per_buoy_queue_ratio,
            final_queue_ratio,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError(
                "per-buoy achieved queue ratios contradict the final queues"
            )
        if not np.allclose(
            self.achieved_goal_per_buoy_completion,
            final_completion,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        ):
            raise ValueError(
                "per-buoy achieved completion contradicts the final state"
            )


class MaritimeEpisodeBuilderV1:
    """Incrementally build one strict :class:`MaritimeEpisodeV1`.

    The builder consumes only ordinary mappings and NumPy-compatible values;
    it deliberately does not import the environment implementation.  This
    makes stored schema semantics independently testable.
    """

    def __init__(
        self,
        episode_id: str,
        policy_id: str,
        seed: int,
        scenario_id: str,
        num_buoys: int,
        initial_info: Mapping[str, Any],
        observation_schema: Optional[Mapping[str, Any]] = None,
        arrival_trace_id: Optional[str] = None,
        desired_goal: Optional[Any] = None,
        area_width: Optional[float] = None,
        area_height: Optional[float] = None,
        initial_energy: Optional[float] = None,
        queue_capacity_bits: Optional[float] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> None:
        self.episode_id = _nonempty_string("episode_id", episode_id)
        self.policy_id = _nonempty_string("policy_id", policy_id)
        self.seed = _nonnegative_integer("seed", seed)
        self.scenario_id = _nonempty_string("scenario_id", scenario_id)
        self.num_buoys = _positive_integer("num_buoys", num_buoys)
        if not isinstance(initial_info, Mapping):
            raise TypeError("initial_info must be a mapping")
        self._initial_info = _copy_info(initial_info)
        _require_v1_info(self._initial_info, "initial_info")
        _validate_info_scale(self._initial_info, self.num_buoys, "initial_info")

        source_schema = (
            observation_schema
            if observation_schema is not None
            else self._initial_info.get("observation_schema", {})
        )
        if not isinstance(source_schema, Mapping):
            raise TypeError("observation_schema must be a mapping")
        self.observation_schema = _canonical_metadata(dict(source_schema))

        self.state_dim = _resolve_dimension(
            "state_dim",
            state_dim,
            self._initial_info.get("state_dim"),
            self.observation_schema.get("state_dim"),
        )
        self.action_dim = _resolve_dimension(
            "action_dim",
            action_dim,
            self._initial_info.get("action_dim"),
            self.observation_schema.get("action_dim"),
        )
        self.observation_schema.setdefault("schema_version", "v1")
        self.observation_schema.setdefault("state_dim", self.state_dim)
        self.observation_schema.setdefault("action_dim", self.action_dim)
        self.observation_schema.setdefault("num_buoys", self.num_buoys)
        self.observation_schema.setdefault(
            "reward_component_names", list(REWARD_COMPONENT_NAMES_V1)
        )
        if self.observation_schema.get("schema_version") != "v1":
            raise ValueError(
                "observation schema must be explicitly versioned 'v1'"
            )

        trace_candidate = (
            arrival_trace_id
            if arrival_trace_id is not None
            else self._initial_info.get("arrival_trace_id")
        )
        self.arrival_trace_id = _nonempty_string(
            "arrival_trace_id", trace_candidate
        )
        self.area_width = _positive_finite(
            "area_width",
            _first_defined(
                area_width,
                self._initial_info.get("area_width"),
                self._initial_info.get("area_width_m"),
            ),
        )
        self.area_height = _positive_finite(
            "area_height",
            _first_defined(
                area_height,
                self._initial_info.get("area_height"),
                self._initial_info.get("area_height_m"),
            ),
        )
        self.initial_energy = _positive_finite(
            "initial_energy",
            _first_defined(
                initial_energy, self._initial_info.get("initial_energy")
            ),
        )
        self.queue_capacity_bits = _positive_finite(
            "queue_capacity_bits",
            _first_defined(
                queue_capacity_bits,
                self._initial_info.get("queue_capacity_bits"),
            ),
        )
        goal_candidate = (
            desired_goal
            if desired_goal is not None
            else self._initial_info.get("desired_goal")
        )
        self.desired_goal = _desired_goal_array(goal_candidate)
        self._records: List[Dict[str, Any]] = []
        self._last_info = self._initial_info
        self._done = False

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
        next_valid_node_mask: Optional[Any] = None,
        policy_selected_buoy_index: Optional[Any] = None,
    ) -> None:
        """Append transition ``s_t, a_t, r_t, s_(t+1)``."""
        if self._done:
            raise RuntimeError("cannot append after episode termination")
        before = (
            _copy_info(info_before)
            if info_before is not None
            else _copy_info(self._last_info)
        )
        if info_after is None:
            raise ValueError("info_after is required for every v1 transition")
        after = _copy_info(info_after)
        for label, info in (("info_before", before), ("info_after", after)):
            _require_v1_info(info, label)
            _validate_info_scale(info, self.num_buoys, label)
            if int(info.get("state_dim", self.state_dim)) != self.state_dim:
                raise ValueError(f"{label} state_dim changed within episode")
            if int(info.get("action_dim", self.action_dim)) != self.action_dim:
                raise ValueError(f"{label} action_dim changed within episode")
            info_trace = info.get("arrival_trace_id")
            if info_trace is not None and str(info_trace) != self.arrival_trace_id:
                raise ValueError(f"{label} arrival_trace_id changed within episode")

        obs = _one_dimensional_finite("observation", observation, np.float32)
        next_obs = _one_dimensional_finite(
            "next_observation", next_observation, np.float32
        )
        if obs.shape != (self.state_dim,) or next_obs.shape != (self.state_dim,):
            raise ValueError(
                f"observation and next_observation must have shape "
                f"{(self.state_dim,)}"
            )
        act = _one_dimensional_finite("action", action, np.float64)
        if act.shape != (self.action_dim,):
            raise ValueError(f"action must have shape {(self.action_dim,)}")
        scalar_reward = _finite_scalar("reward", reward)
        terminated_flag = _strict_bool("terminated", terminated)
        truncated_flag = _strict_bool("truncated", truncated)

        before_mask = _node_mask(
            valid_node_mask,
            before.get("valid_node_mask"),
            self.num_buoys,
            "valid_node_mask",
        )
        after_mask = _node_mask(
            next_valid_node_mask,
            after.get("valid_node_mask"),
            self.num_buoys,
            "next_valid_node_mask",
        )
        selected_value = (
            policy_selected_buoy_index
            if policy_selected_buoy_index is not None
            else after.get("policy_selected_buoy_index", -1)
        )
        selected_index = _selected_index(
            selected_value, self.num_buoys, before_mask
        )
        adjusted = bool(
            after.get(
                "action_was_adjusted",
                bool(after.get("action_component_clipped", False))
                or bool(after.get("action_norm_projected", False))
                or bool(after.get("service_action_adjusted", False)),
            )
        )
        boundary = _strict_bool(
            "boundary_violation", after.get("boundary_violation", False)
        )
        reward_components = _reward_component_vector(
            after.get("reward_components"), scalar_reward
        )

        self._records.append(
            {
                "observation": obs,
                "action": act,
                "reward": scalar_reward,
                "next_observation": next_obs,
                "terminated": terminated_flag,
                "truncated": truncated_flag,
                "valid_node_mask": before_mask,
                "next_valid_node_mask": after_mask,
                "policy_selected_buoy_index": selected_index,
                "boundary_violation": boundary,
                "action_was_adjusted": adjusted,
                "reward_components": reward_components,
                "info_before": before,
                "info_after": after,
            }
        )
        self._last_info = after
        self._done = terminated_flag or truncated_flag

    def finalize(
        self, terminal_reason: Optional[str] = None
    ) -> MaritimeEpisodeV1:
        """Materialize and fully validate the accumulated complete episode."""
        if not self._records:
            raise ValueError("cannot finalize an empty episode")
        last = self._records[-1]
        if terminal_reason is None:
            terminal_reason = last["info_after"].get("terminal_reason")
        if terminal_reason is None or str(terminal_reason) in ("", "ongoing"):
            terminal_reason = _infer_terminal_reason(
                bool(last["terminated"]), bool(last["truncated"])
            )

        before_infos = [item["info_before"] for item in self._records]
        after_infos = [item["info_after"] for item in self._records]
        t_count = len(self._records)
        final_info = after_infos[-1]
        final_mask = np.asarray(
            last["next_valid_node_mask"], dtype=np.bool_
        )
        final_queue = _node_info(
            final_info, "raw_queue_bits", self.num_buoys
        )
        final_completion = _node_info(
            final_info, "service_completion_ratio", self.num_buoys
        )
        final_queue_ratio = np.where(
            final_mask, final_queue / self.queue_capacity_bits, 0.0
        )
        final_completion = np.where(final_mask, final_completion, 0.0)
        valid_count = int(np.count_nonzero(final_mask))
        if valid_count < 1:
            raise ValueError("final valid_node_mask must contain a real buoy")
        residual_ratio = float(
            np.sum(final_queue_ratio, dtype=np.float64) / valid_count
        )
        final_position = _vector_info(final_info, "uav_position_m", 2)
        achieved_global = np.array(
            (
                _scalar_info_value(final_info, "task_completion_ratio"),
                residual_ratio,
                _scalar_info_value(final_info, "remaining_energy")
                / self.initial_energy,
                2.0 * final_position[0] / self.area_width - 1.0,
                2.0 * final_position[1] / self.area_height - 1.0,
            ),
            dtype=np.float64,
        )

        return MaritimeEpisodeV1(
            episode_id=self.episode_id,
            policy_id=self.policy_id,
            seed=self.seed,
            scenario_id=self.scenario_id,
            num_buoys=self.num_buoys,
            arrival_trace_id=self.arrival_trace_id,
            terminal_reason=str(terminal_reason),
            observations=_stack_records(self._records, "observation"),
            actions=_stack_records(self._records, "action"),
            rewards=np.asarray(
                [item["reward"] for item in self._records], dtype=np.float64
            ),
            next_observations=_stack_records(
                self._records, "next_observation"
            ),
            terminated=np.asarray(
                [item["terminated"] for item in self._records], dtype=np.bool_
            ),
            truncated=np.asarray(
                [item["truncated"] for item in self._records], dtype=np.bool_
            ),
            time_steps=np.arange(t_count, dtype=np.int64),
            valid_transition_mask=np.ones(t_count, dtype=np.bool_),
            valid_node_mask=_stack_records(
                self._records, "valid_node_mask"
            ),
            next_valid_node_mask=_stack_records(
                self._records, "next_valid_node_mask"
            ),
            uav_positions_m=_stack_info(before_infos, "uav_position_m"),
            next_uav_positions_m=_stack_info(after_infos, "uav_position_m"),
            buoy_positions_m=_stack_info(before_infos, "buoy_positions_m"),
            rates_bps=_stack_info(before_infos, "rates_bps"),
            next_rates_bps=_stack_info(after_infos, "rates_bps"),
            remaining_energy=_scalar_info(before_infos, "remaining_energy"),
            next_remaining_energy=_scalar_info(
                after_infos, "remaining_energy"
            ),
            raw_queue_bits=_stack_info(before_infos, "raw_queue_bits"),
            next_raw_queue_bits=_stack_info(after_infos, "raw_queue_bits"),
            raw_arrival_bits=_stack_info(after_infos, "raw_arrival_bits"),
            link_capacity_bits=_stack_info(after_infos, "link_capacity_bits"),
            transmitted_payload_bits=_stack_info(
                after_infos, "transmitted_payload_bits"
            ),
            dropped_arrival_bits=_stack_info(
                after_infos, "dropped_arrival_bits"
            ),
            service_completion_ratio=_stack_info(
                before_infos, "service_completion_ratio"
            ),
            next_service_completion_ratio=_stack_info(
                after_infos, "service_completion_ratio"
            ),
            cumulative_distance_m=_scalar_info(
                before_infos, "cumulative_distance_m"
            ),
            next_cumulative_distance_m=_scalar_info(
                after_infos, "cumulative_distance_m"
            ),
            cumulative_energy_consumed=_scalar_info_alias(
                before_infos,
                ("cumulative_energy_consumption", "cumulative_energy_consumed"),
            ),
            next_cumulative_energy_consumed=_scalar_info_alias(
                after_infos,
                ("cumulative_energy_consumption", "cumulative_energy_consumed"),
            ),
            cumulative_payload_bits=_scalar_info(
                before_infos, "cumulative_payload_bits"
            ),
            next_cumulative_payload_bits=_scalar_info(
                after_infos, "cumulative_payload_bits"
            ),
            task_completion_ratio=_scalar_info(
                before_infos, "task_completion_ratio"
            ),
            next_task_completion_ratio=_scalar_info(
                after_infos, "task_completion_ratio"
            ),
            boundary_violation=np.asarray(
                [item["boundary_violation"] for item in self._records],
                dtype=np.bool_,
            ),
            action_was_adjusted=np.asarray(
                [item["action_was_adjusted"] for item in self._records],
                dtype=np.bool_,
            ),
            policy_selected_buoy_index=np.asarray(
                [
                    item["policy_selected_buoy_index"]
                    for item in self._records
                ],
                dtype=np.int64,
            ),
            reward_components=_stack_records(
                self._records, "reward_components"
            ),
            desired_goal=self.desired_goal.copy(),
            achieved_goal_global=achieved_global,
            achieved_goal_per_buoy_queue_ratio=final_queue_ratio,
            achieved_goal_per_buoy_completion=final_completion,
            area_width=self.area_width,
            area_height=self.area_height,
            initial_energy=self.initial_energy,
            queue_capacity_bits=self.queue_capacity_bits,
            observation_schema=dict(self.observation_schema),
            episode_return=float(
                np.sum(
                    [item["reward"] for item in self._records],
                    dtype=np.float64,
                )
            ),
            episode_length=t_count,
        )


def trajectory_v1_from_episode_dict(
    record: Mapping[str, Any],
) -> MaritimeEpisodeV1:
    """Adapt a runner episode mapping without importing the environment."""
    if not isinstance(record, Mapping):
        raise TypeError("episode record must be a mapping")
    required = (
        "episode_id",
        "policy_id",
        "seed",
        "scenario_id",
        "num_buoys",
        "initial_info",
        "transitions",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"episode record is missing keys: {missing}")
    builder = MaritimeEpisodeBuilderV1(
        episode_id=record["episode_id"],
        policy_id=record["policy_id"],
        seed=record["seed"],
        scenario_id=record["scenario_id"],
        num_buoys=record["num_buoys"],
        initial_info=record["initial_info"],
        observation_schema=record.get("observation_schema"),
        arrival_trace_id=record.get("arrival_trace_id"),
        desired_goal=record.get("desired_goal"),
        area_width=record.get("area_width"),
        area_height=record.get("area_height"),
        initial_energy=record.get("initial_energy"),
        queue_capacity_bits=record.get("queue_capacity_bits"),
        state_dim=record.get("state_dim"),
        action_dim=record.get("action_dim"),
    )
    transitions = record["transitions"]
    if isinstance(transitions, (str, bytes)) or not isinstance(
        transitions, Sequence
    ):
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
            next_valid_node_mask=transition.get("next_valid_node_mask"),
            policy_selected_buoy_index=transition.get(
                "policy_selected_buoy_index"
            ),
        )
    return builder.finalize(record.get("terminal_reason"))


def trajectories_v1_equal(
    left: MaritimeEpisodeV1, right: MaritimeEpisodeV1
) -> bool:
    """Return exact equality, including scalar metadata and array dtypes."""
    if not isinstance(left, MaritimeEpisodeV1) or not isinstance(
        right, MaritimeEpisodeV1
    ):
        return False
    scalar_names = (
        "episode_id",
        "policy_id",
        "seed",
        "scenario_id",
        "num_buoys",
        "arrival_trace_id",
        "terminal_reason",
        "area_width",
        "area_height",
        "initial_energy",
        "queue_capacity_bits",
        "episode_return",
        "episode_length",
        "goal_spec_version",
        "schema_version",
        "observation_schema",
    )
    if any(getattr(left, name) != getattr(right, name) for name in scalar_names):
        return False
    left_arrays = left.arrays()
    right_arrays = right.arrays()
    if set(left_arrays) != set(right_arrays):
        return False
    return all(
        left_arrays[name].dtype == right_arrays[name].dtype
        and left_arrays[name].shape == right_arrays[name].shape
        and np.array_equal(left_arrays[name], right_arrays[name])
        for name in left_arrays
    )


def _desired_goal_array(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        missing = [name for name in DESIRED_GOAL_NAMES_V1 if name not in value]
        if missing:
            raise ValueError(f"desired_goal mapping is missing keys: {missing}")
        value = [value[name] for name in DESIRED_GOAL_NAMES_V1]
    result = _one_dimensional_finite("desired_goal", value, np.float64)
    if result.shape != (3,):
        raise ValueError("desired_goal must have shape [3]")
    _require_array_interval("desired_goal", result, 0.0, 1.0)
    return result


def _reward_component_vector(value: Any, reward: float) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ValueError(
            "info_after.reward_components must be a named mapping in schema v1"
        )
    unknown = sorted(set(value) - set(REWARD_COMPONENT_NAMES_V1))
    if unknown:
        raise ValueError(f"unknown v1 reward component names: {unknown}")
    result = np.asarray(
        [float(value.get(name, 0.0)) for name in REWARD_COMPONENT_NAMES_V1],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("reward_components contains NaN or Inf")
    if not math.isclose(
        float(np.sum(result, dtype=np.float64)),
        reward,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("reward_components do not sum to reward")
    return result


def _require_v1_info(info: Mapping[str, Any], label: str) -> None:
    semantics = info.get("task_semantics_version")
    if semantics is not None and semantics != "v1":
        raise ValueError(f"{label} is not task_semantics_version='v1'")


def _validate_info_scale(
    info: Mapping[str, Any], num_buoys: int, label: str
) -> None:
    if "num_buoys" in info and int(info["num_buoys"]) != num_buoys:
        raise ValueError(f"{label} num_buoys changed within episode")
    required_vectors = (
        "uav_position_m",
        "buoy_positions_m",
        "rates_bps",
        "raw_queue_bits",
        "service_completion_ratio",
        "valid_node_mask",
    )
    missing = [name for name in required_vectors if name not in info]
    if missing:
        raise ValueError(f"{label} is missing v1 state fields: {missing}")
    if np.asarray(info["uav_position_m"]).shape != (2,):
        raise ValueError(f"{label}.uav_position_m must have shape [2]")
    if np.asarray(info["buoy_positions_m"]).shape != (num_buoys, 2):
        raise ValueError(
            f"{label}.buoy_positions_m must have shape [{num_buoys}, 2]"
        )
    for name in (
        "rates_bps",
        "raw_queue_bits",
        "service_completion_ratio",
        "valid_node_mask",
    ):
        if np.asarray(info[name]).shape != (num_buoys,):
            raise ValueError(
                f"{label}.{name} must have shape [{num_buoys}]"
            )


def _node_mask(
    explicit: Any, fallback: Any, num_buoys: int, name: str
) -> np.ndarray:
    value = explicit if explicit is not None else fallback
    result = _bool_array(name, value, 1)
    if result.shape != (num_buoys,):
        raise ValueError(f"{name} must have shape {(num_buoys,)}")
    return result


def _selected_index(value: Any, num_buoys: int, mask: np.ndarray) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("policy_selected_buoy_index must be an integer")
    result = int(value)
    if result < -1 or result >= num_buoys:
        raise ValueError("policy_selected_buoy_index is out of range")
    if result >= 0 and not bool(mask[result]):
        raise ValueError("policy_selected_buoy_index selects an invalid node")
    return result


def _infer_terminal_reason(terminated: bool, truncated: bool) -> str:
    if terminated and truncated:
        return "terminated_at_horizon"
    if terminated:
        return "terminated"
    if truncated:
        return "horizon_reached"
    raise ValueError("complete episode has no terminal flag")


def _stack_records(
    records: Sequence[Mapping[str, Any]], key: str
) -> np.ndarray:
    return np.stack([np.asarray(record[key]) for record in records], axis=0)


def _stack_info(infos: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    try:
        return np.stack([np.asarray(info[key]) for info in infos], axis=0)
    except KeyError as error:
        raise ValueError(f"required info field is missing: {key}") from error


def _scalar_info(
    infos: Sequence[Mapping[str, Any]], key: str
) -> np.ndarray:
    try:
        values = [float(info[key]) for info in infos]
    except KeyError as error:
        raise ValueError(f"required info field is missing: {key}") from error
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{key} contains NaN or Inf")
    return result


def _scalar_info_alias(
    infos: Sequence[Mapping[str, Any]], keys: Tuple[str, ...]
) -> np.ndarray:
    values = []
    for info in infos:
        for key in keys:
            if key in info:
                values.append(float(info[key]))
                break
        else:
            raise ValueError(
                f"required info field is missing; expected one of {keys}"
            )
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{keys[0]} contains NaN or Inf")
    return result


def _scalar_info_value(info: Mapping[str, Any], key: str) -> float:
    if key not in info:
        raise ValueError(f"required info field is missing: {key}")
    return _finite_scalar(key, info[key])


def _node_info(
    info: Mapping[str, Any], key: str, num_buoys: int
) -> np.ndarray:
    if key not in info:
        raise ValueError(f"required info field is missing: {key}")
    value = _one_dimensional_finite(key, info[key], np.float64)
    if value.shape != (num_buoys,):
        raise ValueError(f"{key} must have shape {(num_buoys,)}")
    return value


def _vector_info(info: Mapping[str, Any], key: str, width: int) -> np.ndarray:
    if key not in info:
        raise ValueError(f"required info field is missing: {key}")
    value = _one_dimensional_finite(key, info[key], np.float64)
    if value.shape != (width,):
        raise ValueError(f"{key} must have shape {(width,)}")
    return value


def _copy_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(info, Mapping):
        raise TypeError("info must be a mapping")
    copied: Dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            copied[str(key)] = value.copy()
        elif isinstance(value, Mapping):
            copied[str(key)] = _copy_info(value)
        elif isinstance(value, list):
            copied[str(key)] = list(value)
        else:
            copied[str(key)] = value
    return copied


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    raise ValueError("required scalar metadata was not provided")


def _resolve_dimension(name: str, *values: Any) -> int:
    present = [_positive_integer(name, value) for value in values if value is not None]
    if not present:
        raise ValueError(f"{name} is required in arguments, info, or schema")
    if len(set(present)) != 1:
        raise ValueError(f"contradictory {name} values: {present}")
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("metadata contains NaN or Inf")
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metadata contains NaN or Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata cannot encode {type(value).__name__}")


def _finite_array(
    name: str, value: Any, dtype: Any, ndim: int
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def _one_dimensional_finite(name: str, value: Any, dtype: Any) -> np.ndarray:
    return _finite_array(name, value, dtype, 1)


def _bool_array(name: str, value: Any, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {raw.ndim}")
    if raw.dtype.kind not in ("b",):
        if not np.all(np.logical_or(raw == 0, raw == 1)):
            raise ValueError(f"{name} must contain only boolean/0/1 values")
    return raw.astype(np.bool_, copy=True)


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a bool")
    return bool(value)


def _finite_scalar(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(name: str, value: Any) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_array_interval(
    name: str, value: np.ndarray, lower: float, upper: float
) -> None:
    if np.any(value < lower - _FLOAT_ATOL) or np.any(
        value > upper + _FLOAT_ATOL
    ):
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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


_ARRAY_FIELD_NAMES_V1 = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
    "time_steps",
    "valid_transition_mask",
    "valid_node_mask",
    "next_valid_node_mask",
    "uav_positions_m",
    "next_uav_positions_m",
    "buoy_positions_m",
    "rates_bps",
    "next_rates_bps",
    "remaining_energy",
    "next_remaining_energy",
    "raw_queue_bits",
    "next_raw_queue_bits",
    "raw_arrival_bits",
    "link_capacity_bits",
    "transmitted_payload_bits",
    "dropped_arrival_bits",
    "service_completion_ratio",
    "next_service_completion_ratio",
    "cumulative_distance_m",
    "next_cumulative_distance_m",
    "cumulative_energy_consumed",
    "next_cumulative_energy_consumed",
    "cumulative_payload_bits",
    "next_cumulative_payload_bits",
    "task_completion_ratio",
    "next_task_completion_ratio",
    "boundary_violation",
    "action_was_adjusted",
    "policy_selected_buoy_index",
    "reward_components",
    "desired_goal",
    "achieved_goal_global",
    "achieved_goal_per_buoy_queue_ratio",
    "achieved_goal_per_buoy_completion",
)


# Explicit compatibility aliases within the v1-only module.  They do not touch
# or shadow the round-02 classes in ``trajectory.py``.
EpisodeTrajectoryV1 = MaritimeEpisodeV1
EpisodeTrajectoryBuilderV1 = MaritimeEpisodeBuilderV1
trajectory_from_episode_dict_v1 = trajectory_v1_from_episode_dict


__all__ = [
    "ACHIEVED_GOAL_GLOBAL_NAMES_V1",
    "DESIRED_GOAL_NAMES_V1",
    "EpisodeTrajectoryBuilderV1",
    "EpisodeTrajectoryV1",
    "GOAL_SPEC_VERSION_V1",
    "MaritimeEpisodeBuilderV1",
    "MaritimeEpisodeV1",
    "REWARD_COMPONENT_NAMES_V1",
    "TRAJECTORY_SCHEMA_VERSION_V1",
    "trajectories_v1_equal",
    "trajectory_from_episode_dict_v1",
    "trajectory_v1_from_episode_dict",
]
