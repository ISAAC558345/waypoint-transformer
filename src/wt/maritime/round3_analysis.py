"""Round-3 metrics, data-quality audit, and scientific visualizations.

This module is intentionally independent from the Round-2 metrics and plotting
modules.  It consumes completed task-semantics-v1 episodes and reports actual
delivered payload, rather than an integral of unconstrained link capacity.
Nothing in this module trains, fits, or updates a learning model.

The public functions accept either dataclass-like objects or mappings.  This
keeps the analysis layer usable for both in-memory ``MaritimeEpisodeV1``
instances and records loaded from the versioned dataset storage.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ROUND3_ANALYSIS_SCHEMA_VERSION = "round3_analysis_v1"
QUALITY_AUDIT_SCHEMA_VERSION = "data_quality_audit_v1"
ACTION_AGREEMENT_SCHEMA_VERSION = "action_agreement_v1"
EXPECTED_WAYPOINT_HORIZONS = (1, 5, 10, 20, 30)

QUALITY_HIGH = "high"
QUALITY_MID = "mid"
QUALITY_LOW = "low"

_DEFAULT_REWARD_COMPONENT_NAMES = (
    "payload_reward",
    "energy_penalty",
    "boundary_penalty",
    "invalid_action_penalty",
    "energy_depletion_penalty",
    "unfinished_queue_penalty",
    "task_success_bonus",
)
_NUMERIC_TOLERANCE = 1.0e-8
_ACTION_ATOL = 1.0e-8
_ACTION_RTOL = 1.0e-5


EPISODE_METRIC_FIELDS_V1 = (
    "schema_version",
    "episode_id",
    "scenario_id",
    "seed",
    "policy_id",
    "num_buoys",
    "scale_id",
    "split",
    "arrival_trace_id",
    "episode_return",
    "episode_length",
    "initial_queue_bits",
    "final_queue_bits",
    "initial_queue_ratio",
    "final_queue_ratio",
    "cumulative_arrival_bits",
    "cumulative_dropped_arrival_bits",
    "cumulative_actual_payload_bits",
    "cumulative_energy_consumed",
    "initial_remaining_energy",
    "final_remaining_energy",
    "final_remaining_energy_ratio",
    "total_distance_m",
    "final_task_completion_ratio",
    "task_success",
    "goal_satisfied",
    "quality_class",
    "terminal_reason",
    "terminated",
    "truncated",
    "reward_component_sums",
    "reward_component_sum_max_abs_error",
    "final_per_buoy_service_completion",
    "jain_service_completion_index",
    "nonhover_action_count",
    "nonhover_action_ratio",
    "selected_service_action_count",
    "action_adjustment_count",
    "boundary_violation_count",
    "nan_inf_count",
    "negative_queue_count",
    "queue_capacity_violation_count",
    "negative_arrival_count",
    "negative_payload_count",
    "payload_available_data_violation_count",
    "payload_capacity_violation_count",
    "negative_dropped_arrival_count",
    "queue_conservation_violation_count",
    "queue_conservation_max_abs_error_bits",
    "invalid_mask_count",
    "masked_value_violation_count",
    "completion_range_violation_count",
    "cumulative_payload_consistency_error_bits",
    "cumulative_energy_consistency_error",
    "cumulative_distance_consistency_error_m",
    "goal_consistency_ok",
    "finite_ok",
    "conservation_ok",
    "mask_ok",
    "integrity_ok",
)


SUMMARY_FIELDS_V1 = (
    "schema_version",
    "policy_id",
    "episode_count",
    "scenario_count",
    "seed_count",
    "scale_values",
    "episode_return_mean",
    "episode_return_std",
    "episode_length_mean",
    "episode_length_min",
    "episode_length_max",
    "cumulative_actual_payload_bits_mean",
    "cumulative_actual_payload_bits_std",
    "initial_queue_bits_mean",
    "final_queue_bits_mean",
    "final_queue_ratio_mean",
    "cumulative_arrival_bits_mean",
    "cumulative_dropped_arrival_bits_mean",
    "cumulative_energy_consumed_mean",
    "final_remaining_energy_mean",
    "final_remaining_energy_ratio_mean",
    "total_distance_m_mean",
    "final_task_completion_ratio_mean",
    "final_task_completion_ratio_std",
    "task_success_rate",
    "high_quality_count",
    "mid_quality_count",
    "low_quality_count",
    "high_quality_ratio",
    "mid_quality_ratio",
    "low_quality_ratio",
    "nonhover_action_ratio_mean",
    "reward_component_sums_mean",
    "action_adjustment_count_total",
    "boundary_violation_count_total",
    "nan_inf_count_total",
    "queue_conservation_violation_count_total",
    "invalid_mask_count_total",
    "integrity_failure_count",
)


QUALITY_AUDIT_FIELDS = (
    "schema_version",
    "audit_item",
    "metric",
    "value",
    "status",
    "detail",
)


ACTION_AGREEMENT_FIELDS = (
    "schema_version",
    "policy_a",
    "policy_b",
    "matched_case_count",
    "matched_episode_pair_count",
    "compared_step_count",
    "agreement_step_count",
    "agreement_rate",
    "identical_trajectory_pair_count",
    "identical_trajectory_rate",
)


ROUND3_FIGURE_FILENAMES = {
    "episode_return": "round3_episode_return_by_policy.png",
    "actual_payload": "round3_actual_cumulative_payload_by_policy.png",
    "final_queue": "round3_final_remaining_queue_by_policy.png",
    "energy": "round3_cumulative_energy_by_policy.png",
    "completion": "round3_task_completion_by_policy.png",
    "queue_series": "round3_representative_queue_time_series.png",
    "payload_series": "round3_cumulative_actual_payload_time_series.png",
    "uav_trajectories": "round3_uav_trajectories.png",
    "k_sample_counts": "round3_waypoint_k_sample_counts.png",
    "episode_lengths": "round3_episode_length_distribution.png",
    "achieved_goals": "round3_achieved_goal_distributions.png",
    "action_agreement": "round3_action_agreement_heatmap.png",
}


def compute_episode_metrics_v1(episode: Any) -> Dict[str, Any]:
    """Compute truthful, finite metrics for one task-semantics-v1 episode.

    The queue conservation residual uses the stored dropped-arrival amount:

    ``Q[t] + A[t] - D[t] - drop[t] - Q[t+1]``.

    Thus buffer overflow is not silently treated as transmitted payload.  The
    function also checks both physical payload bounds: available data and
    allocated link capacity.
    """

    rewards = _required_numeric_array(episode, ("rewards",), ndim=1)
    episode_length = _positive_int(
        _value_alias(episode, ("episode_length",), default=rewards.size),
        "episode_length",
    )
    if rewards.shape != (episode_length,):
        raise ValueError(
            "rewards must have one value per transition, "
            f"got {rewards.shape} for T={episode_length}"
        )
    num_buoys = _positive_int(
        _value_alias(episode, ("num_buoys",)), "num_buoys"
    )
    tk_shape = (episode_length, num_buoys)

    queue_before = _required_numeric_array(
        episode, ("raw_queue_bits", "queue_before_bits"), ndim=2
    )
    queue_after = _required_numeric_array(
        episode, ("next_raw_queue_bits", "queue_after_bits"), ndim=2
    )
    arrivals = _required_numeric_array(
        episode, ("raw_arrival_bits", "arrival_bits"), ndim=2
    )
    capacities = _required_numeric_array(
        episode, ("link_capacity_bits", "allocated_capacity_bits"), ndim=2
    )
    payload = _required_numeric_array(
        episode, ("transmitted_payload_bits", "actual_payload_bits"), ndim=2
    )
    dropped = _required_numeric_array(
        episode, ("dropped_arrival_bits", "overflow_dropped_bits"), ndim=2
    )
    completion_before = _required_numeric_array(
        episode, ("service_completion_ratio",), ndim=2
    )
    completion_after = _required_numeric_array(
        episode, ("next_service_completion_ratio",), ndim=2
    )
    for name, array in (
        ("raw_queue_bits", queue_before),
        ("next_raw_queue_bits", queue_after),
        ("raw_arrival_bits", arrivals),
        ("link_capacity_bits", capacities),
        ("transmitted_payload_bits", payload),
        ("dropped_arrival_bits", dropped),
        ("service_completion_ratio", completion_before),
        ("next_service_completion_ratio", completion_after),
    ):
        if array.shape != tk_shape:
            raise ValueError(f"{name} must have shape {tk_shape}, got {array.shape}")

    actions = _required_numeric_array(episode, ("actions",), ndim=2)
    if actions.shape[0] != episode_length or actions.shape[1] < 2:
        raise ValueError(
            "actions must have shape [T, A] with A >= 2, "
            f"got {actions.shape}"
        )
    terminated = _required_bool_vector(
        episode, ("terminated",), episode_length
    )
    truncated = _required_bool_vector(
        episode, ("truncated",), episode_length
    )
    if np.any(terminated[:-1]) or np.any(truncated[:-1]):
        raise ValueError("terminated/truncated may be true only at episode end")

    remaining_energy = _required_numeric_array(
        episode, ("remaining_energy", "raw_remaining_energy"), ndim=1
    )
    next_remaining_energy = _required_numeric_array(
        episode, ("next_remaining_energy", "next_raw_remaining_energy"), ndim=1
    )
    cumulative_distance = _required_numeric_array(
        episode, ("cumulative_distance_m",), ndim=1
    )
    next_cumulative_distance = _required_numeric_array(
        episode, ("next_cumulative_distance_m",), ndim=1
    )
    cumulative_energy = _required_numeric_array(
        episode, ("cumulative_energy_consumed",), ndim=1
    )
    next_cumulative_energy = _required_numeric_array(
        episode, ("next_cumulative_energy_consumed",), ndim=1
    )
    cumulative_payload = _required_numeric_array(
        episode, ("cumulative_payload_bits",), ndim=1
    )
    next_cumulative_payload = _required_numeric_array(
        episode, ("next_cumulative_payload_bits",), ndim=1
    )
    task_completion = _required_numeric_array(
        episode, ("task_completion_ratio",), ndim=1
    )
    next_task_completion = _required_numeric_array(
        episode, ("next_task_completion_ratio",), ndim=1
    )
    for name, array in (
        ("remaining_energy", remaining_energy),
        ("next_remaining_energy", next_remaining_energy),
        ("cumulative_distance_m", cumulative_distance),
        ("next_cumulative_distance_m", next_cumulative_distance),
        ("cumulative_energy_consumed", cumulative_energy),
        ("next_cumulative_energy_consumed", next_cumulative_energy),
        ("cumulative_payload_bits", cumulative_payload),
        ("next_cumulative_payload_bits", next_cumulative_payload),
        ("task_completion_ratio", task_completion),
        ("next_task_completion_ratio", next_task_completion),
    ):
        if array.shape != (episode_length,):
            raise ValueError(f"{name} must have shape ({episode_length},)")

    positions = _required_numeric_array(
        episode, ("uav_positions_m", "raw_uav_positions_m"), ndim=2
    )
    next_positions = _required_numeric_array(
        episode, ("next_uav_positions_m", "next_raw_uav_positions_m"), ndim=2
    )
    if positions.shape != (episode_length, 2):
        raise ValueError("uav_positions_m must have shape [T, 2]")
    if next_positions.shape != (episode_length, 2):
        raise ValueError("next_uav_positions_m must have shape [T, 2]")

    valid_node_mask, invalid_mask_count = _required_binary_mask(
        episode, ("valid_node_mask",), tk_shape
    )
    action_adjusted = _optional_bool_vector(
        episode,
        ("action_was_adjusted", "constraint_violation"),
        episode_length,
        default=False,
    )
    boundary = _optional_bool_vector(
        episode, ("boundary_violation",), episode_length, default=False
    )
    selected_service = _optional_integer_vector(
        episode,
        ("policy_selected_buoy_index", "selected_buoy_index"),
        episode_length,
        default=-1,
    )

    reward_matrix, reward_names = _reward_component_matrix(
        episode, episode_length
    )
    reward_step_sum = np.sum(reward_matrix, axis=1, dtype=np.float64)
    reward_component_error = float(
        np.max(np.abs(reward_step_sum - rewards), initial=0.0)
    )
    reward_component_sums = {
        name: float(np.sum(reward_matrix[:, index], dtype=np.float64))
        for index, name in enumerate(reward_names)
    }

    numeric_arrays = (
        rewards,
        queue_before,
        queue_after,
        arrivals,
        capacities,
        payload,
        dropped,
        completion_before,
        completion_after,
        actions,
        remaining_energy,
        next_remaining_energy,
        cumulative_distance,
        next_cumulative_distance,
        cumulative_energy,
        next_cumulative_energy,
        cumulative_payload,
        next_cumulative_payload,
        task_completion,
        next_task_completion,
        positions,
        next_positions,
        reward_matrix,
    )
    nan_inf_count = int(
        sum(
            int(array.size - np.count_nonzero(np.isfinite(array)))
            for array in numeric_arrays
        )
    )
    if nan_inf_count:
        raise ValueError(
            "task-semantics-v1 episode contains NaN or Inf; refusing to "
            "produce misleading physical metrics"
        )

    queue_capacity = _queue_capacity_bits(episode)
    if queue_capacity is not None:
        queue_capacity_violation_count = int(
            np.count_nonzero(queue_before > queue_capacity + _NUMERIC_TOLERANCE)
            + np.count_nonzero(queue_after > queue_capacity + _NUMERIC_TOLERANCE)
        )
        initial_queue_ratio: Optional[float] = float(
            np.sum(queue_before[0], dtype=np.float64)
            / (num_buoys * queue_capacity)
        )
        final_queue_ratio: Optional[float] = float(
            np.sum(queue_after[-1], dtype=np.float64)
            / (num_buoys * queue_capacity)
        )
    else:
        queue_capacity_violation_count = 0
        initial_queue_ratio = _initial_queue_ratio_from_record(episode)
        final_queue_ratio = _final_queue_ratio_from_record(episode)

    available_data = queue_before + arrivals
    if _service_precedes_arrival(episode):
        available_data = queue_before
    payload_tolerance = _scaled_tolerance(
        queue_before, arrivals, capacities, payload
    )
    payload_available_violation = int(
        np.count_nonzero(payload > available_data + payload_tolerance)
    )
    payload_capacity_violation = int(
        np.count_nonzero(payload > capacities + payload_tolerance)
    )

    conservation_residual = (
        queue_before + arrivals - payload - dropped - queue_after
    )
    conservation_tolerance = _scaled_tolerance(
        queue_before, arrivals, payload, dropped, queue_after
    )
    conservation_violation_count = int(
        np.count_nonzero(np.abs(conservation_residual) > conservation_tolerance)
    )
    conservation_max_error = float(
        np.max(np.abs(conservation_residual), initial=0.0)
    )

    invalid_mask_values = ~valid_node_mask
    masked_value_violation_count = int(
        sum(
            np.count_nonzero(
                np.abs(array[invalid_mask_values]) > _NUMERIC_TOLERANCE
            )
            for array in (
                queue_before,
                queue_after,
                arrivals,
                capacities,
                payload,
                dropped,
                completion_before,
                completion_after,
            )
        )
    )

    completion_range_violation_count = int(
        np.count_nonzero(
            (completion_before < -_NUMERIC_TOLERANCE)
            | (completion_before > 1.0 + _NUMERIC_TOLERANCE)
        )
        + np.count_nonzero(
            (completion_after < -_NUMERIC_TOLERANCE)
            | (completion_after > 1.0 + _NUMERIC_TOLERANCE)
        )
        + np.count_nonzero(
            (task_completion < -_NUMERIC_TOLERANCE)
            | (task_completion > 1.0 + _NUMERIC_TOLERANCE)
        )
        + np.count_nonzero(
            (next_task_completion < -_NUMERIC_TOLERANCE)
            | (next_task_completion > 1.0 + _NUMERIC_TOLERANCE)
        )
    )

    actual_payload_sum = float(np.sum(payload, dtype=np.float64))
    final_cumulative_payload = float(next_cumulative_payload[-1])
    cumulative_payload_error = abs(final_cumulative_payload - actual_payload_sum)

    energy_step = np.diff(
        np.concatenate(
            (np.asarray([cumulative_energy[0]], dtype=np.float64), next_cumulative_energy)
        )
    )
    # The first transition begins at its stored "before" cumulative value.
    energy_step[0] = next_cumulative_energy[0] - cumulative_energy[0]
    energy_sum_from_curve = float(
        next_cumulative_energy[-1] - cumulative_energy[0]
    )
    optional_actual_energy = _optional_numeric_array(
        episode, ("actual_energy_consumed",), ndim=1
    )
    if optional_actual_energy is not None:
        if optional_actual_energy.shape != (episode_length,):
            raise ValueError("actual_energy_consumed must have shape [T]")
        if not np.all(np.isfinite(optional_actual_energy)):
            raise ValueError("actual_energy_consumed contains NaN or Inf")
        energy_reference = float(np.sum(optional_actual_energy, dtype=np.float64))
    else:
        energy_reference = energy_sum_from_curve
    final_cumulative_energy = float(next_cumulative_energy[-1])
    cumulative_energy_error = abs(
        (final_cumulative_energy - float(cumulative_energy[0]))
        - energy_reference
    )

    step_distance = np.linalg.norm(next_positions - positions, axis=1)
    path_distance = float(np.sum(step_distance, dtype=np.float64))
    final_cumulative_distance = float(next_cumulative_distance[-1])
    cumulative_distance_error = abs(
        (final_cumulative_distance - float(cumulative_distance[0]))
        - path_distance
    )

    initial_energy = float(remaining_energy[0])
    final_energy = float(next_remaining_energy[-1])
    configured_initial_energy = _configuration_number(
        episode, "initial_energy"
    )
    energy_normalizer = (
        configured_initial_energy
        if configured_initial_energy is not None
        else initial_energy
    )
    final_energy_ratio = (
        float(final_energy / energy_normalizer)
        if energy_normalizer is not None and energy_normalizer > 0.0
        else _final_energy_ratio_from_record(episode)
    )

    initial_queue_bits = float(np.sum(queue_before[0], dtype=np.float64))
    final_queue_bits = float(np.sum(queue_after[-1], dtype=np.float64))
    final_completion = float(next_task_completion[-1])
    final_per_buoy_completion = completion_after[-1].astype(float).tolist()
    jain_completion = _jain_index(completion_after[-1][valid_node_mask[-1]])

    desired_goal = _optional_numeric_array(episode, ("desired_goal",), ndim=1)
    recomputed_goal = _goal_satisfied(
        desired_goal,
        completion_ratio=final_completion,
        residual_queue_ratio=final_queue_ratio,
        energy_ratio=final_energy_ratio,
    )
    stored_success_value = _value_alias(
        episode, ("task_success", "goal_satisfied"), default=None
    )
    terminal_reason = str(
        _value_alias(episode, ("terminal_reason",), default="unknown")
    )
    terminal_success = terminal_reason in {
        "task_success",
        "terminated_task_success",
        "success",
    }
    if recomputed_goal is not None:
        goal_satisfied = recomputed_goal
    elif stored_success_value is not None:
        goal_satisfied = bool(stored_success_value)
    else:
        goal_satisfied = terminal_success
    task_success = (
        bool(stored_success_value)
        if stored_success_value is not None
        else bool(goal_satisfied)
    )
    goal_consistency_ok = bool(
        stored_success_value is None
        or recomputed_goal is None
        or bool(stored_success_value) == bool(recomputed_goal)
    )
    quality_class = classify_episode_quality(
        {
            "goal_satisfied": bool(goal_satisfied),
            "final_task_completion_ratio": final_completion,
        }
    )

    movement_norm = np.linalg.norm(actions[:, :2], axis=1)
    nonhover = movement_norm > _NUMERIC_TOLERANCE
    selected_service_count = int(
        np.count_nonzero(
            (selected_service >= 0) & (selected_service < num_buoys)
        )
    )
    episode_return = float(np.sum(rewards, dtype=np.float64))
    stored_return = float(
        _value_alias(episode, ("episode_return",), default=episode_return)
    )
    if not math.isclose(
        episode_return, stored_return, rel_tol=1.0e-10, abs_tol=1.0e-8
    ):
        raise ValueError("episode_return does not equal sum(rewards)")

    negative_queue_count = int(
        np.count_nonzero(queue_before < -_NUMERIC_TOLERANCE)
        + np.count_nonzero(queue_after < -_NUMERIC_TOLERANCE)
    )
    negative_arrival_count = int(
        np.count_nonzero(arrivals < -_NUMERIC_TOLERANCE)
    )
    negative_payload_count = int(
        np.count_nonzero(payload < -_NUMERIC_TOLERANCE)
    )
    negative_dropped_count = int(
        np.count_nonzero(dropped < -_NUMERIC_TOLERANCE)
    )
    finite_ok = nan_inf_count == 0
    conservation_ok = conservation_violation_count == 0
    mask_ok = invalid_mask_count == 0 and masked_value_violation_count == 0
    integrity_ok = bool(
        finite_ok
        and conservation_ok
        and mask_ok
        and negative_queue_count == 0
        and queue_capacity_violation_count == 0
        and negative_arrival_count == 0
        and negative_payload_count == 0
        and negative_dropped_count == 0
        and payload_available_violation == 0
        and payload_capacity_violation == 0
        and completion_range_violation_count == 0
        and reward_component_error <= _NUMERIC_TOLERANCE
        and cumulative_payload_error <= conservation_tolerance
        and cumulative_energy_error <= _scaled_scalar_tolerance(
            final_cumulative_energy, energy_reference
        )
        and cumulative_distance_error <= _scaled_scalar_tolerance(
            final_cumulative_distance, path_distance
        )
        and goal_consistency_ok
    )

    row: Dict[str, Any] = {
        "schema_version": ROUND3_ANALYSIS_SCHEMA_VERSION,
        "episode_id": str(_value_alias(episode, ("episode_id",))),
        "scenario_id": str(_value_alias(episode, ("scenario_id",))),
        "seed": int(_value_alias(episode, ("seed",))),
        "policy_id": str(_value_alias(episode, ("policy_id",))),
        "num_buoys": num_buoys,
        "scale_id": str(
            _value_alias(
                episode,
                ("scale_id", "buoy_scale_id"),
                default=f"{num_buoys}_buoys",
            )
        ),
        "split": _optional_string(
            _value_alias(episode, ("split",), default=None)
        ),
        "arrival_trace_id": str(
            _value_alias(episode, ("arrival_trace_id",))
        ),
        "episode_return": episode_return,
        "episode_length": episode_length,
        "initial_queue_bits": initial_queue_bits,
        "final_queue_bits": final_queue_bits,
        "initial_queue_ratio": initial_queue_ratio,
        "final_queue_ratio": final_queue_ratio,
        "cumulative_arrival_bits": float(np.sum(arrivals, dtype=np.float64)),
        "cumulative_dropped_arrival_bits": float(
            np.sum(dropped, dtype=np.float64)
        ),
        "cumulative_actual_payload_bits": actual_payload_sum,
        "cumulative_energy_consumed": final_cumulative_energy,
        "initial_remaining_energy": initial_energy,
        "final_remaining_energy": final_energy,
        "final_remaining_energy_ratio": final_energy_ratio,
        "total_distance_m": final_cumulative_distance,
        "final_task_completion_ratio": final_completion,
        "task_success": bool(task_success),
        "goal_satisfied": bool(goal_satisfied),
        "quality_class": quality_class,
        "terminal_reason": terminal_reason,
        "terminated": bool(terminated[-1]),
        "truncated": bool(truncated[-1]),
        "reward_component_sums": reward_component_sums,
        "reward_component_sum_max_abs_error": reward_component_error,
        "final_per_buoy_service_completion": final_per_buoy_completion,
        "jain_service_completion_index": jain_completion,
        "nonhover_action_count": int(np.count_nonzero(nonhover)),
        "nonhover_action_ratio": float(np.mean(nonhover)),
        "selected_service_action_count": selected_service_count,
        "action_adjustment_count": int(np.count_nonzero(action_adjusted)),
        "boundary_violation_count": int(np.count_nonzero(boundary)),
        "nan_inf_count": nan_inf_count,
        "negative_queue_count": negative_queue_count,
        "queue_capacity_violation_count": queue_capacity_violation_count,
        "negative_arrival_count": negative_arrival_count,
        "negative_payload_count": negative_payload_count,
        "payload_available_data_violation_count": payload_available_violation,
        "payload_capacity_violation_count": payload_capacity_violation,
        "negative_dropped_arrival_count": negative_dropped_count,
        "queue_conservation_violation_count": conservation_violation_count,
        "queue_conservation_max_abs_error_bits": conservation_max_error,
        "invalid_mask_count": invalid_mask_count,
        "masked_value_violation_count": masked_value_violation_count,
        "completion_range_violation_count": completion_range_violation_count,
        "cumulative_payload_consistency_error_bits": cumulative_payload_error,
        "cumulative_energy_consistency_error": cumulative_energy_error,
        "cumulative_distance_consistency_error_m": cumulative_distance_error,
        "goal_consistency_ok": goal_consistency_ok,
        "finite_ok": finite_ok,
        "conservation_ok": conservation_ok,
        "mask_ok": mask_ok,
        "integrity_ok": integrity_ok,
    }
    _validate_json_finite(row)
    return row


def compute_episode_metric_rows_v1(
    episodes: Iterable[Any],
) -> List[Dict[str, Any]]:
    """Compute one deterministic metric row per episode."""

    return [compute_episode_metrics_v1(episode) for episode in episodes]


def classify_episode_quality(metric_row: Mapping[str, Any]) -> str:
    """Classify physical trajectory quality using the documented thresholds.

    ``high`` means the desired terminal goal is satisfied.  Otherwise,
    completion of at least 0.5 is ``mid`` and all remaining episodes are
    ``low``.  Reward is deliberately not used as a completion proxy.
    """

    if bool(metric_row.get("goal_satisfied", False)):
        return QUALITY_HIGH
    completion = float(metric_row["final_task_completion_ratio"])
    if not math.isfinite(completion):
        raise ValueError("final_task_completion_ratio must be finite")
    return QUALITY_MID if completion >= 0.5 else QUALITY_LOW


def summarize_episode_metrics_v1(
    rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate task-semantics-v1 episode rows by policy."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy_id"]), []).append(row)
    if not grouped:
        raise ValueError("at least one episode metric row is required")

    summaries: List[Dict[str, Any]] = []
    for policy in sorted(grouped):
        group = grouped[policy]
        count = len(group)
        quality_counts = {
            label: sum(
                str(row["quality_class"]) == label for row in group
            )
            for label in (QUALITY_HIGH, QUALITY_MID, QUALITY_LOW)
        }
        component_names = sorted(
            {
                name
                for row in group
                for name in _mapping_value(row, "reward_component_sums")
            }
        )
        component_means = {
            name: float(
                np.mean(
                    [
                        float(
                            _mapping_value(row, "reward_component_sums").get(
                                name, 0.0
                            )
                        )
                        for row in group
                    ]
                )
            )
            for name in component_names
        }
        summary_row: Dict[str, Any] = {
            "schema_version": ROUND3_ANALYSIS_SCHEMA_VERSION,
            "policy_id": policy,
            "episode_count": count,
            "scenario_count": len(
                {str(row["scenario_id"]) for row in group}
            ),
            "seed_count": len({int(row["seed"]) for row in group}),
            "scale_values": sorted(
                {str(row["scale_id"]) for row in group}
            ),
            "episode_return_mean": _row_mean(group, "episode_return"),
            "episode_return_std": _row_std(group, "episode_return"),
            "episode_length_mean": _row_mean(group, "episode_length"),
            "episode_length_min": int(
                min(int(row["episode_length"]) for row in group)
            ),
            "episode_length_max": int(
                max(int(row["episode_length"]) for row in group)
            ),
            "cumulative_actual_payload_bits_mean": _row_mean(
                group, "cumulative_actual_payload_bits"
            ),
            "cumulative_actual_payload_bits_std": _row_std(
                group, "cumulative_actual_payload_bits"
            ),
            "initial_queue_bits_mean": _row_mean(
                group, "initial_queue_bits"
            ),
            "final_queue_bits_mean": _row_mean(group, "final_queue_bits"),
            "final_queue_ratio_mean": _optional_row_mean(
                group, "final_queue_ratio"
            ),
            "cumulative_arrival_bits_mean": _row_mean(
                group, "cumulative_arrival_bits"
            ),
            "cumulative_dropped_arrival_bits_mean": _row_mean(
                group, "cumulative_dropped_arrival_bits"
            ),
            "cumulative_energy_consumed_mean": _row_mean(
                group, "cumulative_energy_consumed"
            ),
            "final_remaining_energy_mean": _row_mean(
                group, "final_remaining_energy"
            ),
            "final_remaining_energy_ratio_mean": _optional_row_mean(
                group, "final_remaining_energy_ratio"
            ),
            "total_distance_m_mean": _row_mean(group, "total_distance_m"),
            "final_task_completion_ratio_mean": _row_mean(
                group, "final_task_completion_ratio"
            ),
            "final_task_completion_ratio_std": _row_std(
                group, "final_task_completion_ratio"
            ),
            "task_success_rate": float(
                np.mean([bool(row["task_success"]) for row in group])
            ),
            "high_quality_count": quality_counts[QUALITY_HIGH],
            "mid_quality_count": quality_counts[QUALITY_MID],
            "low_quality_count": quality_counts[QUALITY_LOW],
            "high_quality_ratio": quality_counts[QUALITY_HIGH] / count,
            "mid_quality_ratio": quality_counts[QUALITY_MID] / count,
            "low_quality_ratio": quality_counts[QUALITY_LOW] / count,
            "nonhover_action_ratio_mean": _row_mean(
                group, "nonhover_action_ratio"
            ),
            "reward_component_sums_mean": component_means,
            "action_adjustment_count_total": _row_int_sum(
                group, "action_adjustment_count"
            ),
            "boundary_violation_count_total": _row_int_sum(
                group, "boundary_violation_count"
            ),
            "nan_inf_count_total": _row_int_sum(group, "nan_inf_count"),
            "queue_conservation_violation_count_total": _row_int_sum(
                group, "queue_conservation_violation_count"
            ),
            "invalid_mask_count_total": _row_int_sum(
                group, "invalid_mask_count"
            ),
            "integrity_failure_count": sum(
                not bool(row["integrity_ok"]) for row in group
            ),
        }
        _validate_json_finite(summary_row)
        summaries.append(summary_row)
    return summaries


def summary(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Short public alias for :func:`summarize_episode_metrics_v1`."""

    return summarize_episode_metrics_v1(rows)


def write_episode_metrics_csv_v1(
    rows: Sequence[Mapping[str, Any]], path: Union[str, Path]
) -> Path:
    """Write ``baseline_episode_metrics.csv`` with stable v1 columns."""

    return _write_csv(rows, EPISODE_METRIC_FIELDS_V1, path)


def write_baseline_summary_csv_v1(
    rows: Sequence[Mapping[str, Any]], path: Union[str, Path]
) -> Path:
    """Write ``baseline_summary.csv`` with stable v1 columns."""

    return _write_csv(rows, SUMMARY_FIELDS_V1, path)


def compute_action_agreement_matrix(
    episodes: Sequence[Any],
    *,
    rtol: float = _ACTION_RTOL,
    atol: float = _ACTION_ATOL,
) -> List[Dict[str, Any]]:
    """Compare policy actions on matched scenario/seed/scale prefixes.

    For each policy pair, only episodes sharing ``scenario_id``, ``seed``, and
    buoy scale are compared.  Each pair uses the common temporal prefix and
    applies :func:`numpy.allclose` independently at every timestep.
    """

    if not episodes:
        raise ValueError("at least one episode is required")
    if not math.isfinite(float(rtol)) or float(rtol) < 0.0:
        raise ValueError("rtol must be finite and nonnegative")
    if not math.isfinite(float(atol)) or float(atol) < 0.0:
        raise ValueError("atol must be finite and nonnegative")

    indexed: Dict[Tuple[str, int, int], Dict[str, List[Any]]] = {}
    policies = sorted(
        {str(_value_alias(episode, ("policy_id",))) for episode in episodes}
    )
    for episode in episodes:
        key = _episode_case_key(episode)
        policy = str(_value_alias(episode, ("policy_id",)))
        indexed.setdefault(key, {}).setdefault(policy, []).append(episode)
    for policy_map in indexed.values():
        for policy in policy_map:
            policy_map[policy].sort(
                key=lambda item: str(_value_alias(item, ("episode_id",)))
            )

    rows: List[Dict[str, Any]] = []
    for policy_a in policies:
        for policy_b in policies:
            matched_cases = 0
            matched_pairs = 0
            compared_steps = 0
            agreement_steps = 0
            identical_trajectories = 0
            for policy_map in indexed.values():
                if policy_a not in policy_map or policy_b not in policy_map:
                    continue
                pairs = list(
                    zip(policy_map[policy_a], policy_map[policy_b])
                )
                if not pairs:
                    continue
                matched_cases += 1
                for episode_a, episode_b in pairs:
                    actions_a = _required_numeric_array(
                        episode_a, ("actions",), ndim=2
                    )
                    actions_b = _required_numeric_array(
                        episode_b, ("actions",), ndim=2
                    )
                    if actions_a.shape[1] != actions_b.shape[1]:
                        raise ValueError(
                            "matched policy episodes have different action dimensions"
                        )
                    prefix = min(actions_a.shape[0], actions_b.shape[0])
                    if prefix < 1:
                        continue
                    close = np.isclose(
                        actions_a[:prefix],
                        actions_b[:prefix],
                        rtol=float(rtol),
                        atol=float(atol),
                        equal_nan=False,
                    )
                    step_equal = np.all(close, axis=1)
                    compared_steps += prefix
                    agreement_steps += int(np.count_nonzero(step_equal))
                    matched_pairs += 1
                    if _trajectory_exactly_equal(episode_a, episode_b):
                        identical_trajectories += 1

            agreement_rate = (
                float(agreement_steps / compared_steps)
                if compared_steps
                else None
            )
            trajectory_rate = (
                float(identical_trajectories / matched_pairs)
                if matched_pairs
                else None
            )
            row = {
                "schema_version": ACTION_AGREEMENT_SCHEMA_VERSION,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "matched_case_count": matched_cases,
                "matched_episode_pair_count": matched_pairs,
                "compared_step_count": compared_steps,
                "agreement_step_count": agreement_steps,
                "agreement_rate": agreement_rate,
                "identical_trajectory_pair_count": identical_trajectories,
                "identical_trajectory_rate": trajectory_rate,
            }
            _validate_json_finite(row)
            rows.append(row)
    return rows


def write_action_agreement_matrix_csv(
    rows: Sequence[Mapping[str, Any]], path: Union[str, Path]
) -> Path:
    """Write the long-form action-agreement matrix."""

    return _write_csv(rows, ACTION_AGREEMENT_FIELDS, path)


def build_data_quality_audit(
    episodes: Sequence[Any],
    *,
    episode_metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    split_assignments: Optional[Any] = None,
    k_statistics: Optional[Sequence[Mapping[str, Any]]] = None,
    action_agreement: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run the 17 required pilot-data audits plus readiness checks.

    ``k_statistics`` may contain aggregate rows (for example
    ``valid_sample_count`` and ``candidate_sample_count``) or one row per
    waypoint sample.  Missing grouping dimensions are reported as unavailable,
    never inferred from unrelated episode counts.
    """

    if not episodes:
        raise ValueError("at least one episode is required")
    metrics = (
        list(episode_metrics)
        if episode_metrics is not None
        else compute_episode_metric_rows_v1(episodes)
    )
    if len(metrics) != len(episodes):
        raise ValueError("episode_metrics must contain one row per episode")
    agreements = (
        list(action_agreement)
        if action_agreement is not None
        else compute_action_agreement_matrix(episodes)
    )
    k_rows = list(k_statistics or ())
    split_map, split_conflicts = _canonical_split_assignments(
        split_assignments
    )
    k_totals = _aggregate_k_statistics(k_rows)
    policy_sample_counts = _aggregate_k_dimension(k_rows, "policy_id")
    scale_sample_counts = _aggregate_k_dimension(
        k_rows, ("scale_id", "num_buoys")
    )
    split_sample_counts = _aggregate_k_dimension(k_rows, "split")
    k_split_conflicts = _k_row_split_conflicts(k_rows)

    rows: List[Dict[str, Any]] = []

    def add(
        item: str,
        metric: str,
        value: Any,
        status: str,
        detail: str,
    ) -> None:
        row = {
            "schema_version": QUALITY_AUDIT_SCHEMA_VERSION,
            "audit_item": item,
            "metric": metric,
            "value": value,
            "status": status,
            "detail": detail,
        }
        _validate_json_finite(row)
        rows.append(row)

    expected_counts = {
        str(k): int(k_totals.get(k, {}).get("valid", 0))
        for k in EXPECTED_WAYPOINT_HORIZONS
    }
    expected_ratios = {
        str(k): _safe_ratio(
            k_totals.get(k, {}).get("valid", 0),
            k_totals.get(k, {}).get("candidate", 0),
        )
        for k in EXPECTED_WAYPOINT_HORIZONS
    }
    all_k_present = all(value > 0 for value in expected_counts.values())
    add(
        "01",
        "valid_waypoint_samples_by_k",
        {"counts": expected_counts, "valid_ratios": expected_ratios},
        "PASS" if all_k_present else "FAIL",
        "Valid and candidate counts are taken from waypoint K rows; "
        "required K={1,5,10,20,30}.",
    )

    add(
        "02",
        "waypoint_sample_contribution_by_policy",
        policy_sample_counts or {"unavailable": True},
        "PASS" if len(policy_sample_counts) >= 4 else "WARN",
        "Counts are waypoint samples, not episode counts; a missing policy "
        "dimension is reported as unavailable.",
    )
    add(
        "03",
        "waypoint_sample_count_by_buoy_scale",
        scale_sample_counts or {"unavailable": True},
        "PASS" if len(scale_sample_counts) >= 2 else "WARN",
        "Variable-scale coverage is grouped by stored scale_id or num_buoys.",
    )
    required_splits = {"train", "validation", "test"}
    observed_splits = {
        _canonical_split_name(name)
        for name, count in split_sample_counts.items()
        if int(count) > 0
    }
    add(
        "04",
        "waypoint_sample_count_by_split",
        split_sample_counts or {"unavailable": True},
        "PASS" if required_splits.issubset(observed_splits) else "FAIL",
        "Training, validation, and test samples must each be non-empty and "
        "are counted from stored waypoint rows.",
    )

    lengths = np.asarray(
        [int(row["episode_length"]) for row in metrics], dtype=np.float64
    )
    add(
        "05",
        "episode_length_distribution",
        _distribution(lengths, unit="step"),
        "PASS" if int(np.max(lengths)) > 30 else "FAIL",
        "At least one episode must be longer than K=30.",
    )

    initial_queue = np.asarray(
        [float(row["initial_queue_bits"]) for row in metrics], dtype=np.float64
    )
    final_queue = np.asarray(
        [float(row["final_queue_bits"]) for row in metrics], dtype=np.float64
    )
    add(
        "06",
        "initial_and_final_total_queue_distribution",
        {
            "initial": _distribution(initial_queue, unit="bit"),
            "final": _distribution(final_queue, unit="bit"),
        },
        "PASS",
        "Both quantities are physical buffer occupancy in bit.",
    )

    completion = np.asarray(
        [float(row["final_task_completion_ratio"]) for row in metrics],
        dtype=np.float64,
    )
    add(
        "07",
        "task_completion_distribution",
        _distribution(completion, unit="ratio"),
        "PASS" if float(np.ptp(completion)) > _NUMERIC_TOLERANCE else "WARN",
        "Completion is read from task state; reward is not used as a proxy.",
    )

    actual_payload = np.asarray(
        [float(row["cumulative_actual_payload_bits"]) for row in metrics],
        dtype=np.float64,
    )
    add(
        "08",
        "actual_transmitted_payload_distribution",
        _distribution(actual_payload, unit="bit"),
        "PASS" if np.any(actual_payload > 0.0) else "FAIL",
        "This is queue-bounded transmitted payload, not link capacity.",
    )

    remaining_energy = np.asarray(
        [float(row["final_remaining_energy"]) for row in metrics],
        dtype=np.float64,
    )
    add(
        "09",
        "final_remaining_energy_distribution",
        _distribution(remaining_energy, unit="energy_unit"),
        "PASS",
        "Raw terminal energy is reported separately from its normalized ratio.",
    )

    distances = np.asarray(
        [float(row["total_distance_m"]) for row in metrics], dtype=np.float64
    )
    add(
        "10",
        "uav_distance_distribution",
        _distribution(distances, unit="m"),
        "PASS" if np.any(distances > _NUMERIC_TOLERANCE) else "WARN",
        "Distance is the stored cumulative physical UAV path length.",
    )

    off_diagonal = [
        float(row["agreement_rate"])
        for row in agreements
        if row["policy_a"] != row["policy_b"]
        and row.get("agreement_rate") is not None
    ]
    duplicate_info = _duplicate_trajectory_statistics(episodes)
    action_diverse = bool(
        off_diagonal
        and float(np.mean(off_diagonal)) < 0.99
        and duplicate_info["duplicate_episode_ratio"] <= 0.25
    )
    add(
        "11",
        "policy_action_and_trajectory_diversity",
        {
            "mean_off_diagonal_action_agreement": (
                float(np.mean(off_diagonal)) if off_diagonal else None
            ),
            **duplicate_info,
        },
        "PASS" if action_diverse else "WARN",
        "Actions use matched-case common prefixes; trajectories use exact "
        "action and physical-position bytes.",
    )

    nearest_qre = _agreement_value(
        agreements, "nearest_buoy", "queue_rate_energy"
    )
    add(
        "12",
        "nearest_vs_queue_rate_energy_action_agreement",
        nearest_qre,
        (
            "PASS"
            if nearest_qre is not None and nearest_qre < 0.95
            else "WARN"
        ),
        "Agreement uses np.allclose per action on matched "
        "scenario/seed/scale/t common prefixes.",
    )

    max_queue_rows = [
        row for row in metrics if str(row["policy_id"]) == "max_queue"
    ]
    max_queue_nonhover = (
        float(
            sum(int(row["nonhover_action_count"]) for row in max_queue_rows)
            / sum(int(row["episode_length"]) for row in max_queue_rows)
        )
        if max_queue_rows
        else None
    )
    add(
        "13",
        "max_queue_nonhover_action_ratio",
        max_queue_nonhover,
        (
            "PASS"
            if max_queue_nonhover is not None and max_queue_nonhover > 0.0
            else "FAIL"
        ),
        "A zero ratio indicates the old degenerate always-hover behavior.",
    )

    success_by_policy = {
        policy: float(
            np.mean(
                [
                    bool(row["task_success"])
                    for row in metrics
                    if str(row["policy_id"]) == policy
                ]
            )
        )
        for policy in sorted({str(row["policy_id"]) for row in metrics})
    }
    add(
        "14",
        "task_success_rate_by_policy",
        success_by_policy,
        "PASS" if any(value > 0.0 for value in success_by_policy.values()) else "WARN",
        "Success is determined by desired-goal constraints, not episode return.",
    )

    quality_by_episode = {
        str(row["episode_id"]): str(row["quality_class"]) for row in metrics
    }
    all_policies = sorted({str(row["policy_id"]) for row in metrics})
    high_by_policy = {policy: 0 for policy in all_policies}
    high_sample_rows_available = False
    for k_row in k_rows:
        if "episode_id" not in k_row or "policy_id" not in k_row:
            continue
        episode_id = str(k_row["episode_id"])
        if quality_by_episode.get(episode_id) != QUALITY_HIGH:
            continue
        high_sample_rows_available = True
        policy = str(k_row["policy_id"])
        valid_count, _ = _k_valid_candidate(k_row)
        high_by_policy[policy] = high_by_policy.get(policy, 0) + valid_count
    if not high_sample_rows_available:
        # Standalone aggregate K tables may omit episode_id.  In that case the
        # fallback is explicitly episode-level rather than pretending those
        # aggregate rows can be attributed to quality classes.
        high_by_policy = {
            policy: sum(
                row["quality_class"] == QUALITY_HIGH
                for row in metrics
                if str(row["policy_id"]) == policy
            )
            for policy in all_policies
        }
    high_total = sum(high_by_policy.values())
    high_dominance = (
        max(high_by_policy.values()) / high_total
        if high_total > 0 and high_by_policy
        else None
    )
    add(
        "15",
        "high_quality_sample_policy_dominance",
        {
            (
                "high_waypoint_sample_count_by_policy"
                if high_sample_rows_available
                else "high_episode_count_by_policy_fallback"
            ): high_by_policy,
            "maximum_policy_share": high_dominance,
        },
        (
            "PASS"
            if high_dominance is not None and high_dominance <= 0.80
            else "WARN"
        ),
        "High quality means desired goal satisfied. Waypoint samples are "
        "attributed through episode_id when available; otherwise the value is "
        "explicitly episode-level. A policy share above 0.80 is dominance.",
    )

    add(
        "16",
        "exact_duplicate_trajectory_rate",
        duplicate_info,
        (
            "PASS"
            if duplicate_info["duplicate_episode_ratio"] <= 0.25
            else "WARN"
        ),
        "More than 25% duplicate episodes is considered a large duplicate set.",
    )

    integrity_totals = {
        key: int(sum(int(row[key]) for row in metrics))
        for key in (
            "nan_inf_count",
            "negative_queue_count",
            "queue_capacity_violation_count",
            "negative_arrival_count",
            "negative_payload_count",
            "payload_available_data_violation_count",
            "payload_capacity_violation_count",
            "negative_dropped_arrival_count",
            "queue_conservation_violation_count",
            "invalid_mask_count",
            "masked_value_violation_count",
            "completion_range_violation_count",
        )
    }
    integrity_failures = sum(integrity_totals.values())
    add(
        "17",
        "numeric_mask_and_conservation_integrity",
        integrity_totals,
        "PASS" if integrity_failures == 0 else "FAIL",
        "Requires finite values, legal masks, nonnegative bounded queues and "
        "payload, and exact queue conservation within scaled tolerance.",
    )

    quality_counts = {
        label: sum(row["quality_class"] == label for row in metrics)
        for label in (QUALITY_HIGH, QUALITY_MID, QUALITY_LOW)
    }
    quality_ratios = {
        label: count / len(metrics) for label, count in quality_counts.items()
    }
    add(
        "18",
        "trajectory_quality_classes",
        {"counts": quality_counts, "ratios": quality_ratios},
        (
            "PASS"
            if quality_counts[QUALITY_HIGH] > 0
            and (
                quality_counts[QUALITY_MID] + quality_counts[QUALITY_LOW]
                > 0
            )
            else "WARN"
        ),
        "Thresholds: goal satisfied=high; otherwise completion>=0.5=mid; "
        "otherwise low. Reward is never used.",
    )

    trace_audit = _shared_trace_audit(episodes)
    add(
        "19",
        "shared_arrival_trace_consistency",
        trace_audit,
        "PASS" if trace_audit["mismatch_group_count"] == 0 else "FAIL",
        "Policies in the same scenario/seed/scale group must store one "
        "identical non-empty arrival_trace_id.",
    )

    overlap = _trajectory_overlap_score(episodes)
    add(
        "20",
        "cross_policy_state_overlap_for_stitching",
        overlap,
        (
            "PASS"
            if overlap["mean_pairwise_occupancy_jaccard"] is not None
            and overlap["mean_pairwise_occupancy_jaccard"] >= 0.05
            else "WARN"
        ),
        "Overlap uses coarse occupied bins of position, total queue ratio, "
        "and remaining-energy ratio within each buoy scale; 0.05 is the "
        "minimum evidence threshold, not a learning guarantee.",
    )

    leakage_count = split_conflicts + k_split_conflicts
    add(
        "21",
        "episode_and_waypoint_split_leakage",
        {
            "assignment_conflict_count": split_conflicts,
            "waypoint_row_conflict_count": k_split_conflicts,
            "known_episode_assignment_count": len(split_map),
        },
        "PASS" if leakage_count == 0 else "FAIL",
        "An episode_id may belong to exactly one of train/validation/test.",
    )

    total_transitions = int(sum(int(row["episode_length"]) for row in metrics))
    total_waypoint_samples = int(
        sum(entry.get("valid", 0) for entry in k_totals.values())
    )
    policy_count = len({str(row["policy_id"]) for row in metrics})
    scale_count = len({int(row["num_buoys"]) for row in metrics})
    blockers: List[str] = []
    if len(metrics) < 80:
        blockers.append("fewer_than_80_pilot_episodes")
    if total_transitions < 1000:
        blockers.append("fewer_than_1000_transitions")
    if total_waypoint_samples < 1000:
        blockers.append("fewer_than_1000_waypoint_samples")
    if not all_k_present:
        blockers.append("missing_required_K_samples")
    if not required_splits.issubset(observed_splits):
        blockers.append("missing_nonempty_split")
    if leakage_count:
        blockers.append("split_leakage")
    if integrity_failures:
        blockers.append("physical_or_numeric_integrity_failure")
    if trace_audit["mismatch_group_count"]:
        blockers.append("shared_arrival_trace_mismatch")
    if policy_count < 4:
        blockers.append("fewer_than_four_policies")
    if scale_count < 2:
        blockers.append("fewer_than_two_buoy_scales")
    if quality_counts[QUALITY_HIGH] == 0:
        blockers.append("no_high_quality_episode")
    if quality_counts[QUALITY_MID] + quality_counts[QUALITY_LOW] == 0:
        blockers.append("no_suboptimal_episode")
    if high_dominance is not None and high_dominance > 0.80:
        blockers.append("high_quality_dominated_by_one_policy")
    if nearest_qre is None or nearest_qre >= 0.95:
        blockers.append("nearest_and_queue_rate_energy_insufficiently_distinct")
    if max_queue_nonhover is None or max_queue_nonhover <= 0.0:
        blockers.append("max_queue_degenerate_hover")
    if (
        overlap["mean_pairwise_occupancy_jaccard"] is None
        or overlap["mean_pairwise_occupancy_jaccard"] < 0.05
    ):
        blockers.append("insufficient_cross_policy_overlap")

    add(
        "22",
        "waypoint_training_readiness",
        {
            "ready": not blockers,
            "blockers": blockers,
            "episode_count": len(metrics),
            "transition_count": total_transitions,
            "waypoint_sample_count": total_waypoint_samples,
            "training_performed": False,
        },
        "PASS" if not blockers else "FAIL",
        "Readiness requires volume, all K values, non-leaking splits, "
        "integrity, shared traces, four distinct policies, two scales, both "
        "goal-satisfying and suboptimal data, and stitching overlap. It is "
        "not inferred merely from a nonzero sample count.",
    )
    return rows


def write_data_quality_audit_csv(
    rows: Sequence[Mapping[str, Any]], path: Union[str, Path]
) -> Path:
    """Write the structured Round-3 data-quality audit."""

    return _write_csv(rows, QUALITY_AUDIT_FIELDS, path)


def run_round3_analysis(
    episodes: Sequence[Any],
    split_assignments: Any,
    waypoint_batches: Any,
    output_dir: Union[str, Path],
    figures_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Run the complete non-learning Round-3 analysis pipeline.

    Four CSV artifacts are written to ``output_dir`` and the twelve required
    PNGs are written to ``figures_dir``.  K statistics are reconstructed from
    the supplied waypoint batches and, when available, reconciled with the
    already persisted ``waypoint_k_statistics.csv`` so the valid ratio retains
    rejected candidate anchors.  Existing target artifacts are never
    overwritten.
    """

    episode_values = list(episodes)
    if not episode_values:
        raise ValueError("at least one MaritimeEpisodeV1 is required")
    root = Path(output_dir)
    figure_root = Path(figures_dir)
    root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "baseline_episode_metrics": root / "baseline_episode_metrics.csv",
        "baseline_summary": root / "baseline_summary.csv",
        "data_quality_audit": root / "data_quality_audit.csv",
        "action_agreement_matrix": root / "action_agreement_matrix.csv",
    }
    expected_figure_paths = {
        key: figure_root / filename
        for key, filename in ROUND3_FIGURE_FILENAMES.items()
    }
    collisions = [
        path
        for path in (
            list(artifact_paths.values()) + list(expected_figure_paths.values())
        )
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing Round-3 analysis artifacts: "
            + ", ".join(str(path) for path in collisions)
        )

    split_map, split_conflicts = _canonical_split_assignments(
        split_assignments
    )
    if split_conflicts:
        raise ValueError("split_assignments contains conflicting episode splits")
    episode_ids = {
        str(_value_alias(episode, ("episode_id",)))
        for episode in episode_values
    }
    if set(split_map) != episode_ids:
        missing = sorted(episode_ids - set(split_map))
        extra = sorted(set(split_map) - episode_ids)
        raise ValueError(
            "split_assignments must cover exactly the supplied episodes; "
            f"missing={missing}, extra={extra}"
        )

    metric_rows = compute_episode_metric_rows_v1(episode_values)
    for row in metric_rows:
        row["split"] = split_map[str(row["episode_id"])]
    summary_rows = summarize_episode_metrics_v1(metric_rows)
    agreement_rows = compute_action_agreement_matrix(episode_values)
    k_rows = _waypoint_rows_for_analysis(
        waypoint_batches, root / "waypoint_k_statistics.csv"
    )
    audit_rows = build_data_quality_audit(
        episode_values,
        episode_metrics=metric_rows,
        split_assignments=split_map,
        k_statistics=k_rows,
        action_agreement=agreement_rows,
    )

    write_episode_metrics_csv_v1(
        metric_rows, artifact_paths["baseline_episode_metrics"]
    )
    write_baseline_summary_csv_v1(
        summary_rows, artifact_paths["baseline_summary"]
    )
    write_data_quality_audit_csv(
        audit_rows, artifact_paths["data_quality_audit"]
    )
    write_action_agreement_matrix_csv(
        agreement_rows, artifact_paths["action_agreement_matrix"]
    )
    figure_paths = generate_round3_figures(
        episode_values,
        metric_rows,
        figure_root,
        k_statistics=k_rows,
        action_agreement=agreement_rows,
        split_assignments=split_map,
    )
    return {
        "schema_version": ROUND3_ANALYSIS_SCHEMA_VERSION,
        "artifact_paths": dict(artifact_paths),
        "figure_paths": dict(figure_paths),
        "episode_metrics": metric_rows,
        "baseline_summary": summary_rows,
        "data_quality_audit": audit_rows,
        "action_agreement_matrix": agreement_rows,
        "waypoint_k_rows": k_rows,
        "training_performed": False,
    }


def generate_round3_figures(
    episodes: Sequence[Any],
    episode_metrics: Sequence[Mapping[str, Any]],
    output_dir: Union[str, Path],
    *,
    k_statistics: Optional[Sequence[Mapping[str, Any]]] = None,
    action_agreement: Optional[Sequence[Mapping[str, Any]]] = None,
    split_assignments: Optional[Any] = None,
) -> Dict[str, Path]:
    """Generate the twelve required Round-3 PNG figures.

    Existing target files are never overwritten.  Pale points/lines retain
    raw episode variation and darker, thicker marks show per-policy means.
    """

    del split_assignments  # accepted for a uniform pipeline API
    if not episodes or not episode_metrics:
        raise ValueError("episodes and episode_metrics must be non-empty")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        name: destination / filename
        for name, filename in ROUND3_FIGURE_FILENAMES.items()
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing Round-3 figures: "
            + ", ".join(str(path) for path in existing)
        )

    agreements = (
        list(action_agreement)
        if action_agreement is not None
        else compute_action_agreement_matrix(episodes)
    )
    plt = _load_pyplot()
    with plt.rc_context(_scientific_style()):
        _plot_scalar_by_policy(
            plt,
            episode_metrics,
            "episode_return",
            "Episode return",
            "Non-learning baseline episode return",
            paths["episode_return"],
        )
        _plot_scalar_by_policy(
            plt,
            episode_metrics,
            "cumulative_actual_payload_bits",
            "Actual cumulative payload [bit]",
            "Queue-bounded delivered payload",
            paths["actual_payload"],
        )
        _plot_scalar_by_policy(
            plt,
            episode_metrics,
            "final_queue_bits",
            "Final total queue [bit]",
            "Terminal remaining buoy data",
            paths["final_queue"],
        )
        _plot_scalar_by_policy(
            plt,
            episode_metrics,
            "cumulative_energy_consumed",
            "Cumulative energy consumed [energy unit]",
            "UAV energy consumption",
            paths["energy"],
        )
        _plot_scalar_by_policy(
            plt,
            episode_metrics,
            "final_task_completion_ratio",
            "Task completion ratio",
            "Terminal task completion",
            paths["completion"],
            ylim=(0.0, 1.05),
        )
        _plot_representative_queue_series(
            plt, episodes, paths["queue_series"]
        )
        _plot_cumulative_payload_series(
            plt, episodes, paths["payload_series"]
        )
        _plot_uav_trajectories(
            plt, episodes, paths["uav_trajectories"]
        )
        _plot_k_sample_counts(
            plt, list(k_statistics or ()), paths["k_sample_counts"]
        )
        _plot_scalar_by_policy(
            plt,
            episode_metrics,
            "episode_length",
            "Episode length [step]",
            "Pilot episode-length distribution",
            paths["episode_lengths"],
        )
        _plot_achieved_goal_distributions(
            plt, episode_metrics, paths["achieved_goals"]
        )
        _plot_action_agreement_heatmap(
            plt, agreements, paths["action_agreement"]
        )

    for path in paths.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"failed to create Round-3 figure: {path}")
    return paths


def _plot_scalar_by_policy(
    plt: Any,
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    ylabel: str,
    title: str,
    destination: Path,
    *,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    policies = sorted({str(row["policy_id"]) for row in rows})
    colors = _palette(plt, len(policies))
    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    for index, (policy, color) in enumerate(zip(policies, colors)):
        values = np.asarray(
            [
                float(row[value_key])
                for row in rows
                if str(row["policy_id"]) == policy
            ],
            dtype=np.float64,
        )
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid {value_key} values for {policy}")
        jitter = (
            np.linspace(-0.16, 0.16, values.size)
            if values.size > 1
            else np.zeros(1)
        )
        axis.scatter(
            index + jitter,
            values,
            s=16,
            color=color,
            alpha=0.25,
            edgecolors="none",
            zorder=2,
        )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        axis.errorbar(
            index,
            mean,
            yerr=std,
            fmt="o",
            color=color,
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            markersize=6,
            linewidth=2.4,
            capsize=4,
            zorder=3,
        )
    axis.set_xticks(np.arange(len(policies)))
    axis.set_xticklabels(policies, rotation=18, ha="right")
    axis.set_xlabel("Policy")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.text(
        0.99,
        0.02,
        "Pale points: episodes; dark markers: mean ± population SD",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    _save_close(plt, figure, destination)


def _plot_representative_queue_series(
    plt: Any, episodes: Sequence[Any], destination: Path
) -> None:
    representatives = _representative_episodes(episodes)
    policies = sorted(representatives)
    rows, columns = _subplot_grid(len(policies), max_columns=2)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.0 * columns, 4.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, policy in zip(axes.flat, policies):
        episode = representatives[policy]
        queue_before = _required_numeric_array(
            episode, ("raw_queue_bits", "queue_before_bits"), ndim=2
        )
        queue_after = _required_numeric_array(
            episode, ("next_raw_queue_bits", "queue_after_bits"), ndim=2
        )
        series = np.vstack((queue_before[0], queue_after))
        times = np.arange(series.shape[0])
        buoy_colors = _palette(plt, series.shape[1])
        for buoy_index, color in enumerate(buoy_colors):
            axis.plot(
                times,
                series[:, buoy_index],
                color=color,
                alpha=0.45,
                linewidth=1.0,
                label=f"Buoy {buoy_index}",
            )
        axis.plot(
            times,
            np.sum(series, axis=1),
            color="#1B1B1B",
            linewidth=2.5,
            label="Total queue",
        )
        axis.set_xlabel("Environment step")
        axis.set_ylabel("Queue [bit]")
        episode_seed = int(_value_alias(episode, ("seed",)))
        num_buoys = int(_value_alias(episode, ("num_buoys",)))
        episode_length = int(
            _value_alias(
                episode,
                ("episode_length",),
                default=queue_before.shape[0],
            )
        )
        axis.set_title(
            f"{policy} | seed={episode_seed}, "
            f"buoys={num_buoys}, T={episode_length}"
        )
        axis.legend(loc="best", frameon=True, ncol=2)
    for axis in list(axes.flat)[len(policies) :]:
        axis.set_axis_off()
    figure.suptitle("Per-buoy queue evolution (actual buffer state)")
    _save_close(plt, figure, destination)


def _plot_cumulative_payload_series(
    plt: Any, episodes: Sequence[Any], destination: Path
) -> None:
    policies = sorted(
        {str(_value_alias(episode, ("policy_id",))) for episode in episodes}
    )
    colors = _palette(plt, len(policies))
    figure, axis = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    for policy, color in zip(policies, colors):
        curves: List[np.ndarray] = []
        for episode in episodes:
            if str(_value_alias(episode, ("policy_id",))) != policy:
                continue
            payload = _required_numeric_array(
                episode,
                ("transmitted_payload_bits", "actual_payload_bits"),
                ndim=2,
            )
            curve = np.concatenate(
                (
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(np.sum(payload, axis=1), dtype=np.float64),
                )
            )
            curves.append(curve)
            axis.plot(
                np.arange(curve.size),
                curve,
                color=color,
                alpha=0.16,
                linewidth=0.7,
            )
        mean_curve = _nanmean_variable_curves(curves)
        axis.plot(
            np.arange(mean_curve.size),
            mean_curve,
            color=color,
            linewidth=2.5,
            label=f"{policy} mean",
        )
    axis.set_xlabel("Environment step")
    axis.set_ylabel("Cumulative actual payload [bit]")
    axis.set_title("Queue-bounded delivered payload over time")
    axis.legend(loc="best", frameon=True)
    axis.text(
        0.99,
        0.02,
        "Pale lines: episodes; dark lines: per-policy mean",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    _save_close(plt, figure, destination)


def _plot_uav_trajectories(
    plt: Any, episodes: Sequence[Any], destination: Path
) -> None:
    policies = sorted(
        {str(_value_alias(episode, ("policy_id",))) for episode in episodes}
    )
    rows, columns = _subplot_grid(len(policies), max_columns=2)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.2 * columns, 5.1 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    colors = _palette(plt, len(policies))
    for axis, policy, color in zip(axes.flat, policies, colors):
        policy_episodes = sorted(
            [
                episode
                for episode in episodes
                if str(_value_alias(episode, ("policy_id",))) == policy
            ],
            key=lambda episode: str(_value_alias(episode, ("episode_id",))),
        )
        curves_x: List[np.ndarray] = []
        curves_y: List[np.ndarray] = []
        all_points: List[np.ndarray] = []
        for episode in policy_episodes:
            positions = _episode_path(episode)
            curves_x.append(positions[:, 0])
            curves_y.append(positions[:, 1])
            all_points.append(positions)
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                alpha=0.15,
                linewidth=0.65,
            )
        mean_x = _nanmean_variable_curves(curves_x)
        mean_y = _nanmean_variable_curves(curves_y)
        axis.plot(
            mean_x,
            mean_y,
            color=color,
            linewidth=2.4,
            label="Mean path",
        )
        representative = policy_episodes[0]
        buoy_positions = _required_numeric_array(
            representative, ("buoy_positions_m",), ndim=None
        )
        if buoy_positions.ndim == 3:
            buoy_positions = buoy_positions[0]
        if buoy_positions.ndim != 2 or buoy_positions.shape[1] != 2:
            raise ValueError("buoy_positions_m must have shape [K,2] or [T,K,2]")
        all_points.append(buoy_positions)
        axis.scatter(
            buoy_positions[:, 0],
            buoy_positions[:, 1],
            marker="^",
            s=42,
            color="#315C9B",
            edgecolor="white",
            linewidth=0.6,
            label="Buoy (representative case)",
            zorder=3,
        )
        concatenated = np.vstack(all_points)
        x_min, x_max = float(np.min(concatenated[:, 0])), float(
            np.max(concatenated[:, 0])
        )
        y_min, y_max = float(np.min(concatenated[:, 1])), float(
            np.max(concatenated[:, 1])
        )
        span = max(x_max - x_min, y_max - y_min, 1.0)
        x_mid, y_mid = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
        axis.set_xlim(x_mid - 0.58 * span, x_mid + 0.58 * span)
        axis.set_ylim(y_mid - 0.58 * span, y_mid + 0.58 * span)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x position [m]")
        axis.set_ylabel("y position [m]")
        axis.set_title(policy)
        axis.legend(loc="best", frameon=True)
    for axis in list(axes.flat)[len(policies) :]:
        axis.set_axis_off()
    figure.suptitle("UAV trajectories and static buoy locations")
    _save_close(plt, figure, destination)


def _plot_k_sample_counts(
    plt: Any,
    k_statistics: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    totals = _aggregate_k_statistics(k_statistics)
    counts = [
        int(totals.get(k, {}).get("valid", 0))
        for k in EXPECTED_WAYPOINT_HORIZONS
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    bars = axis.bar(
        [str(k) for k in EXPECTED_WAYPOINT_HORIZONS],
        counts,
        color="#356AA0",
        alpha=0.84,
    )
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + 0.5 * bar.get_width(),
            bar.get_height(),
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xlabel("Waypoint horizon K [step]")
    axis.set_ylabel("Valid supervised sample count")
    axis.set_title("Boundary-safe waypoint samples by horizon")
    _save_close(plt, figure, destination)


def _plot_achieved_goal_distributions(
    plt: Any,
    metrics: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    policies = sorted({str(row["policy_id"]) for row in metrics})
    specifications = (
        ("final_task_completion_ratio", "Completion ratio", (0.0, 1.05)),
        ("final_queue_ratio", "Residual queue ratio", (0.0, None)),
        ("final_remaining_energy_ratio", "Remaining energy ratio", (0.0, None)),
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(14.2, 4.2), squeeze=False, constrained_layout=True
    )
    colors = _palette(plt, len(policies))
    for axis, (key, ylabel, limits) in zip(axes.flat, specifications):
        for index, (policy, color) in enumerate(zip(policies, colors)):
            values = [
                row.get(key)
                for row in metrics
                if str(row["policy_id"]) == policy
                and row.get(key) is not None
            ]
            numeric = np.asarray(values, dtype=np.float64)
            if not numeric.size:
                continue
            jitter = (
                np.linspace(-0.14, 0.14, numeric.size)
                if numeric.size > 1
                else np.zeros(1)
            )
            axis.scatter(
                index + jitter,
                numeric,
                s=15,
                color=color,
                alpha=0.24,
                edgecolors="none",
            )
            axis.plot(
                index,
                float(np.mean(numeric)),
                marker="o",
                color=color,
                markersize=6,
                markeredgecolor="#222222",
                markeredgewidth=0.5,
            )
        axis.set_xticks(np.arange(len(policies)))
        axis.set_xticklabels(policies, rotation=23, ha="right")
        axis.set_xlabel("Policy")
        axis.set_ylabel(ylabel)
        lower, upper = limits
        if upper is None:
            axis.set_ylim(bottom=lower)
        else:
            axis.set_ylim(lower, upper)
    figure.suptitle("Achieved terminal goal distributions")
    _save_close(plt, figure, destination)


def _plot_action_agreement_heatmap(
    plt: Any,
    agreements: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    policies = sorted(
        {
            str(row[key])
            for row in agreements
            for key in ("policy_a", "policy_b")
        }
    )
    matrix = np.full((len(policies), len(policies)), np.nan, dtype=np.float64)
    index = {policy: position for position, policy in enumerate(policies)}
    for row in agreements:
        value = row.get("agreement_rate")
        if value is None:
            continue
        matrix[index[str(row["policy_a"])], index[str(row["policy_b"])]] = float(
            value
        )
    figure, axis = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axis, label="Action agreement ratio")
    axis.set_xticks(np.arange(len(policies)))
    axis.set_yticks(np.arange(len(policies)))
    axis.set_xticklabels(policies, rotation=25, ha="right")
    axis.set_yticklabels(policies)
    axis.set_xlabel("Policy B")
    axis.set_ylabel("Policy A")
    axis.set_title("Matched scenario/seed/scale action agreement")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "N/A" if not math.isfinite(float(value)) else f"{value:.2f}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if math.isfinite(float(value)) and value >= 0.62
                    else "#222222"
                ),
            )
    _save_close(plt, figure, destination)


def _field(record: Any, name: str, default: Any = ...) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is not ...:
        return default
    raise KeyError(f"record is missing required field {name!r}")


def _value_alias(
    record: Any, names: Sequence[str], default: Any = ...
) -> Any:
    for name in names:
        if isinstance(record, Mapping):
            if name in record:
                return record[name]
        elif hasattr(record, name):
            return getattr(record, name)
    if default is not ...:
        return default
    raise KeyError(
        "record is missing required field; expected one of "
        + ", ".join(repr(name) for name in names)
    )


def _required_numeric_array(
    record: Any, names: Sequence[str], *, ndim: Optional[int]
) -> np.ndarray:
    value = _value_alias(record, names)
    try:
        untyped = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{names[0]} must be a numeric array") from error
    if np.iscomplexobj(untyped):
        raise ValueError(f"{names[0]} must contain real values")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{names[0]} must be a numeric array") from error
    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{names[0]} must have rank {ndim}, got shape {array.shape}"
        )
    return array.copy()


def _optional_numeric_array(
    record: Any, names: Sequence[str], *, ndim: Optional[int]
) -> Optional[np.ndarray]:
    value = _value_alias(record, names, default=None)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{names[0]} must have rank {ndim}, got shape {array.shape}"
        )
    return array.copy()


def _required_bool_vector(
    record: Any, names: Sequence[str], length: int
) -> np.ndarray:
    value = np.asarray(_value_alias(record, names))
    if value.shape != (length,):
        raise ValueError(f"{names[0]} must have shape ({length},)")
    if value.dtype == np.bool_:
        return value.astype(bool, copy=True)
    numeric = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{names[0]} must be a boolean/binary vector")
    return numeric.astype(bool)


def _optional_bool_vector(
    record: Any,
    names: Sequence[str],
    length: int,
    *,
    default: bool,
) -> np.ndarray:
    value = _value_alias(record, names, default=None)
    if value is None:
        return np.full(length, default, dtype=bool)
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{names[0]} must have shape ({length},)")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=True)
    numeric = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{names[0]} must be boolean/binary")
    return numeric.astype(bool)


def _optional_integer_vector(
    record: Any,
    names: Sequence[str],
    length: int,
    *,
    default: int,
) -> np.ndarray:
    value = _value_alias(record, names, default=None)
    if value is None:
        return np.full(length, default, dtype=np.int64)
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{names[0]} must have shape ({length},)")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(
        numeric == np.floor(numeric)
    ):
        raise ValueError(f"{names[0]} must contain finite integers")
    return numeric.astype(np.int64)


def _required_binary_mask(
    record: Any, names: Sequence[str], shape: Tuple[int, ...]
) -> Tuple[np.ndarray, int]:
    raw = np.asarray(_value_alias(record, names))
    if raw.shape != shape:
        raise ValueError(f"{names[0]} must have shape {shape}, got {raw.shape}")
    if raw.dtype == np.bool_:
        return raw.astype(bool, copy=True), 0
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{names[0]} must be boolean/binary") from error
    finite = np.isfinite(numeric)
    binary = finite & ((numeric == 0.0) | (numeric == 1.0))
    invalid_count = int(np.count_nonzero(~binary))
    return (numeric == 1.0), invalid_count


def _reward_component_matrix(
    episode: Any, episode_length: int
) -> Tuple[np.ndarray, List[str]]:
    value = _value_alias(episode, ("reward_components",))
    if isinstance(value, Mapping):
        names = [str(name) for name in value]
        columns = [
            np.asarray(value[name], dtype=np.float64) for name in value
        ]
        if not names:
            raise ValueError("reward_components mapping must not be empty")
        if any(column.shape != (episode_length,) for column in columns):
            raise ValueError(
                "each reward_components mapping value must have shape [T]"
            )
        return np.column_stack(columns), names
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, np.ndarray))
        and len(value) > 0
        and isinstance(value[0], Mapping)
    ):
        names = sorted(
            {
                str(name)
                for step in value
                for name in step
            }
        )
        if len(value) != episode_length:
            raise ValueError("reward_components must contain T mappings")
        matrix = np.asarray(
            [
                [float(step.get(name, 0.0)) for name in names]
                for step in value
            ],
            dtype=np.float64,
        )
        return matrix, names

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != episode_length:
        raise ValueError("reward_components must have shape [T, C]")
    supplied_names = _value_alias(
        episode, ("reward_component_names",), default=None
    )
    if supplied_names is not None:
        names = [str(name) for name in supplied_names]
        if len(names) != matrix.shape[1]:
            raise ValueError(
                "reward_component_names width differs from reward_components"
            )
    elif matrix.shape[1] == len(_DEFAULT_REWARD_COMPONENT_NAMES):
        names = list(_DEFAULT_REWARD_COMPONENT_NAMES)
    else:
        names = [f"component_{index}" for index in range(matrix.shape[1])]
    if len(set(names)) != len(names):
        raise ValueError("reward_component_names must be unique")
    return matrix.copy(), names


def _queue_capacity_bits(episode: Any) -> Optional[float]:
    value = _configuration_number(episode, "queue_capacity_bits")
    if value is None:
        return None
    if value <= 0.0:
        raise ValueError("queue_capacity_bits must be positive")
    return value


def _configuration_number(record: Any, name: str) -> Optional[float]:
    direct = _field(record, name, default=None)
    if direct is not None:
        return _finite_float(direct, name)
    for container_name in (
        "environment_config",
        "env_config",
        "config",
        "metadata",
        "observation_schema",
    ):
        container = _field(record, container_name, default=None)
        if container is None:
            continue
        nested = _field(container, name, default=None)
        if nested is not None:
            return _finite_float(nested, name)
        if isinstance(container, Mapping):
            for nested_name in ("environment", "normalization", "raw_config"):
                nested_container = container.get(nested_name)
                if isinstance(nested_container, Mapping) and name in nested_container:
                    return _finite_float(nested_container[name], name)
    return None


def _initial_queue_ratio_from_record(episode: Any) -> Optional[float]:
    normalized = _optional_numeric_array(
        episode, ("normalized_queue", "normalized_queue_values"), ndim=2
    )
    if normalized is None:
        return None
    return float(np.mean(normalized[0]))


def _final_queue_ratio_from_record(episode: Any) -> Optional[float]:
    normalized = _optional_numeric_array(
        episode,
        ("next_normalized_queue", "next_normalized_queue_values"),
        ndim=2,
    )
    if normalized is not None:
        return float(np.mean(normalized[-1]))
    achieved = _optional_numeric_array(
        episode, ("achieved_goal_global",), ndim=1
    )
    if achieved is not None and achieved.size >= 2:
        return float(achieved[1])
    return None


def _final_energy_ratio_from_record(episode: Any) -> Optional[float]:
    achieved = _optional_numeric_array(
        episode, ("achieved_goal_global",), ndim=1
    )
    if achieved is not None and achieved.size >= 3:
        return float(achieved[2])
    return None


def _service_precedes_arrival(episode: Any) -> bool:
    value = str(
        _value_alias(
            episode,
            ("queue_update_order", "slot_update_order"),
            default="service_before_arrival",
        )
    ).lower()
    return "service_before_arrival" in value or "available_data=q[t]" in value


def _goal_satisfied(
    desired_goal: Optional[np.ndarray],
    *,
    completion_ratio: float,
    residual_queue_ratio: Optional[float],
    energy_ratio: Optional[float],
) -> Optional[bool]:
    if (
        desired_goal is None
        or desired_goal.size < 3
        or residual_queue_ratio is None
        or energy_ratio is None
    ):
        return None
    if not np.all(np.isfinite(desired_goal[:3])):
        raise ValueError("desired_goal contains NaN or Inf")
    return bool(
        completion_ratio + _NUMERIC_TOLERANCE >= float(desired_goal[0])
        and residual_queue_ratio
        <= float(desired_goal[1]) + _NUMERIC_TOLERANCE
        and energy_ratio + _NUMERIC_TOLERANCE >= float(desired_goal[2])
    )


def _scaled_tolerance(*arrays: np.ndarray) -> float:
    maximum = max(
        [
            float(np.max(np.abs(array), initial=0.0))
            for array in arrays
        ]
        + [1.0]
    )
    return max(_NUMERIC_TOLERANCE, 1.0e-10 * maximum)


def _scaled_scalar_tolerance(*values: float) -> float:
    return max(
        _NUMERIC_TOLERANCE,
        1.0e-10 * max([abs(float(value)) for value in values] + [1.0]),
    )


def _jain_index(values: np.ndarray) -> Optional[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return None
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("Jain index values must be finite and nonnegative")
    denominator = float(array.size * np.sum(np.square(array)))
    if denominator <= 0.0:
        return None
    return float(np.sum(array) ** 2 / denominator)


def _row_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _row_std(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.std([float(row[key]) for row in rows], ddof=0))


def _optional_row_mean(
    rows: Sequence[Mapping[str, Any]], key: str
) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _row_int_sum(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(int(row[key]) for row in rows))


def _mapping_value(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _canonical_split_assignments(
    assignments: Optional[Any],
) -> Tuple[Dict[str, str], int]:
    if assignments is None:
        return {}, 0
    pairs: List[Tuple[str, str]] = []
    if isinstance(assignments, Mapping):
        for episode_id, value in assignments.items():
            if isinstance(value, Mapping):
                split = value.get("split")
            else:
                split = value
            pairs.append((str(episode_id), _canonical_split_name(split)))
    else:
        for row in assignments:
            pairs.append(
                (
                    str(_value_alias(row, ("episode_id",))),
                    _canonical_split_name(_value_alias(row, ("split",))),
                )
            )
    result: Dict[str, str] = {}
    conflicts = 0
    for episode_id, split in pairs:
        if episode_id in result and result[episode_id] != split:
            conflicts += 1
        else:
            result[episode_id] = split
    return result, conflicts


def _canonical_split_name(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "val":
        return "validation"
    return text


def _waypoint_rows_for_analysis(
    waypoint_batches: Any, persisted_statistics_path: Path
) -> List[Mapping[str, Any]]:
    """Create grouped valid-sample rows and reconcile candidate counts."""

    if hasattr(waypoint_batches, "batches"):
        batches = getattr(waypoint_batches, "batches")
        supplied_statistics = list(
            getattr(waypoint_batches, "k_statistics", ())
        )
    else:
        batches = waypoint_batches
        supplied_statistics = []
    if not isinstance(batches, Mapping):
        raise TypeError(
            "waypoint_batches must be a split-to-WaypointSampleBatch mapping "
            "or a WaypointSampleBuildResult"
        )

    grouped: Dict[Tuple[str, int, str, int, str], int] = {}
    for split_name, batch in batches.items():
        canonical_split = _canonical_split_name(split_name)
        k_values = np.asarray(_field(batch, "k"), dtype=np.int64)
        policies = np.asarray(_field(batch, "policy_id")).astype(str)
        scales = np.asarray(_field(batch, "num_buoys"), dtype=np.int64)
        episode_ids = np.asarray(_field(batch, "episode_id")).astype(str)
        lengths = {
            k_values.shape[0],
            policies.shape[0],
            scales.shape[0],
            episode_ids.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError(
                f"waypoint batch {canonical_split!r} arrays have unequal lengths"
            )
        for k_value, policy, scale, episode_id in zip(
            k_values, policies, scales, episode_ids
        ):
            key = (
                canonical_split,
                int(k_value),
                str(policy),
                int(scale),
                str(episode_id),
            )
            grouped[key] = grouped.get(key, 0) + 1

    rows: List[Mapping[str, Any]] = [
        {
            "split": split,
            "k": k_value,
            "policy_id": policy,
            "num_buoys": scale,
            "episode_id": episode_id,
            "valid_sample_count": count,
            "candidate_sample_count": count,
        }
        for (
            split,
            k_value,
            policy,
            scale,
            episode_id,
        ), count in sorted(grouped.items())
    ]
    valid_by_split_k: Dict[Tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["split"]), int(row["k"]))
        valid_by_split_k[key] = valid_by_split_k.get(key, 0) + int(
            row["valid_sample_count"]
        )

    persisted_rows: List[Mapping[str, Any]] = supplied_statistics
    if not persisted_rows and persisted_statistics_path.is_file():
        with persisted_statistics_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            persisted_rows = list(csv.DictReader(handle))

    if persisted_rows:
        observed_keys = set()
        for stored in persisted_rows:
            split = _canonical_split_name(stored.get("split"))
            k_value = _k_value(stored)
            valid, candidate = _k_valid_candidate(stored)
            key = (split, k_value)
            observed_keys.add(key)
            batch_valid = valid_by_split_k.get(key, 0)
            if batch_valid != valid:
                raise ValueError(
                    "waypoint batch sample count disagrees with persisted K "
                    f"statistics for split={split}, K={k_value}: "
                    f"{batch_valid} != {valid}"
                )
            rows.append(
                {
                    "split": split,
                    "k": k_value,
                    "valid_sample_count": 0,
                    "candidate_sample_count": candidate - valid,
                    "candidate_adjustment_only": True,
                }
            )
        missing_statistics = sorted(set(valid_by_split_k) - observed_keys)
        if missing_statistics:
            raise ValueError(
                "persisted waypoint K statistics omit populated batch groups: "
                f"{missing_statistics}"
            )
    return rows


def _k_value(row: Mapping[str, Any]) -> int:
    value = row.get("K", row.get("k", row.get("horizon_k")))
    return _positive_int(value, "K")


def _k_valid_candidate(row: Mapping[str, Any]) -> Tuple[int, int]:
    valid_keys = (
        "valid_sample_count",
        "sample_count",
        "valid_count",
        "count",
    )
    candidate_keys = (
        "candidate_sample_count",
        "candidate_count",
        "candidate_anchor_count",
        "eligible_start_count",
        "total_candidate_count",
    )
    valid = next(
        (int(row[key]) for key in valid_keys if key in row), 1
    )
    candidate = next(
        (int(row[key]) for key in candidate_keys if key in row),
        max(1, valid),
    )
    if valid < 0 or candidate < 0 or valid > candidate:
        raise ValueError("K statistics require 0 <= valid <= candidate")
    return valid, candidate


def _aggregate_k_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, int]]:
    result: Dict[int, Dict[str, int]] = {}
    for row in rows:
        k = _k_value(row)
        valid, candidate = _k_valid_candidate(row)
        entry = result.setdefault(k, {"valid": 0, "candidate": 0})
        entry["valid"] += valid
        entry["candidate"] += candidate
    return result


def _aggregate_k_dimension(
    rows: Sequence[Mapping[str, Any]],
    dimension: Union[str, Sequence[str]],
) -> Dict[str, int]:
    names = (dimension,) if isinstance(dimension, str) else tuple(dimension)
    result: Dict[str, int] = {}
    found = False
    for row in rows:
        value = next((row[name] for name in names if name in row), None)
        if value is None:
            continue
        found = True
        valid, _ = _k_valid_candidate(row)
        key = str(value)
        result[key] = result.get(key, 0) + valid
    return result if found else {}


def _k_row_split_conflicts(rows: Sequence[Mapping[str, Any]]) -> int:
    seen: Dict[str, str] = {}
    conflicts = 0
    for row in rows:
        if "episode_id" not in row or "split" not in row:
            continue
        episode_id = str(row["episode_id"])
        split = _canonical_split_name(row["split"])
        if episode_id in seen and seen[episode_id] != split:
            conflicts += 1
        else:
            seen[episode_id] = split
    return conflicts


def _safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    denominator_value = int(denominator)
    return (
        float(int(numerator) / denominator_value)
        if denominator_value > 0
        else None
    )


def _distribution(values: np.ndarray, *, unit: str) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be a finite nonempty vector")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
        "std": float(np.std(array, ddof=0)),
        "unit": unit,
    }


def _episode_case_key(episode: Any) -> Tuple[str, int, int]:
    return (
        str(_value_alias(episode, ("scenario_id",))),
        int(_value_alias(episode, ("seed",))),
        int(_value_alias(episode, ("num_buoys",))),
    )


def _trajectory_exactly_equal(episode_a: Any, episode_b: Any) -> bool:
    arrays_a = (
        _required_numeric_array(episode_a, ("actions",), ndim=2),
        _required_numeric_array(
            episode_a, ("next_uav_positions_m",), ndim=2
        ),
        _required_numeric_array(
            episode_a, ("next_raw_queue_bits",), ndim=2
        ),
        _required_numeric_array(
            episode_a, ("transmitted_payload_bits",), ndim=2
        ),
    )
    arrays_b = (
        _required_numeric_array(episode_b, ("actions",), ndim=2),
        _required_numeric_array(
            episode_b, ("next_uav_positions_m",), ndim=2
        ),
        _required_numeric_array(
            episode_b, ("next_raw_queue_bits",), ndim=2
        ),
        _required_numeric_array(
            episode_b, ("transmitted_payload_bits",), ndim=2
        ),
    )
    return all(
        left.shape == right.shape and np.array_equal(left, right)
        for left, right in zip(arrays_a, arrays_b)
    )


def _duplicate_trajectory_statistics(
    episodes: Sequence[Any],
) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], int] = {}
    for episode in episodes:
        actions = _required_numeric_array(episode, ("actions",), ndim=2)
        positions = _required_numeric_array(
            episode, ("next_uav_positions_m",), ndim=2
        )
        queues = _required_numeric_array(
            episode, ("next_raw_queue_bits",), ndim=2
        )
        payload = _required_numeric_array(
            episode, ("transmitted_payload_bits",), ndim=2
        )
        key = (
            actions.shape,
            actions.dtype.str,
            actions.tobytes(order="C"),
            positions.shape,
            positions.dtype.str,
            positions.tobytes(order="C"),
            queues.shape,
            queues.dtype.str,
            queues.tobytes(order="C"),
            payload.shape,
            payload.dtype.str,
            payload.tobytes(order="C"),
        )
        groups[key] = groups.get(key, 0) + 1
    duplicate_episode_count = sum(
        count for count in groups.values() if count > 1
    )
    duplicate_pair_count = sum(
        count * (count - 1) // 2 for count in groups.values() if count > 1
    )
    return {
        "unique_trajectory_count": len(groups),
        "duplicate_episode_count": duplicate_episode_count,
        "duplicate_episode_ratio": float(
            duplicate_episode_count / len(episodes)
        ),
        "duplicate_pair_count": duplicate_pair_count,
    }


def _agreement_value(
    rows: Sequence[Mapping[str, Any]], policy_a: str, policy_b: str
) -> Optional[float]:
    for row in rows:
        if (
            str(row["policy_a"]) == policy_a
            and str(row["policy_b"]) == policy_b
        ):
            value = row.get("agreement_rate")
            return None if value is None else float(value)
    return None


def _shared_trace_audit(episodes: Sequence[Any]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, int, int], List[Any]] = {}
    for episode in episodes:
        groups.setdefault(_episode_case_key(episode), []).append(episode)
    comparable = 0
    mismatch = 0
    missing = 0
    for group in groups.values():
        policies = {
            str(_value_alias(episode, ("policy_id",))) for episode in group
        }
        if len(policies) < 2:
            continue
        comparable += 1
        identifiers = [
            _value_alias(episode, ("arrival_trace_id",), default=None)
            for episode in group
        ]
        if any(identifier is None or not str(identifier).strip() for identifier in identifiers):
            missing += 1
            mismatch += 1
        elif len({str(identifier) for identifier in identifiers}) != 1:
            mismatch += 1
    return {
        "comparable_group_count": comparable,
        "mismatch_group_count": mismatch,
        "missing_trace_id_group_count": missing,
    }


def _trajectory_overlap_score(episodes: Sequence[Any]) -> Dict[str, Any]:
    """Return cross-policy coarse-state occupancy overlap.

    Bins use dataset-wide x/y extents plus physically normalized total queue
    and remaining energy.  Pairwise comparisons are restricted to equal buoy
    scales, so padding does not create overlap.
    """

    positions_all = [
        _required_numeric_array(episode, ("uav_positions_m",), ndim=2)
        for episode in episodes
    ]
    flat_positions = np.vstack(positions_all)
    x_min, x_max = float(np.min(flat_positions[:, 0])), float(
        np.max(flat_positions[:, 0])
    )
    y_min, y_max = float(np.min(flat_positions[:, 1])), float(
        np.max(flat_positions[:, 1])
    )
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    occupancy: Dict[Tuple[int, str], set] = {}
    for episode, positions in zip(episodes, positions_all):
        policy = str(_value_alias(episode, ("policy_id",)))
        scale = int(_value_alias(episode, ("num_buoys",)))
        queue = _required_numeric_array(
            episode, ("raw_queue_bits", "queue_before_bits"), ndim=2
        )
        energy = _required_numeric_array(
            episode, ("remaining_energy",), ndim=1
        )
        queue_capacity = _queue_capacity_bits(episode)
        initial_energy = _configuration_number(episode, "initial_energy")
        if queue_capacity is None or initial_energy is None or initial_energy <= 0.0:
            continue
        queue_ratio = np.sum(queue, axis=1) / (scale * queue_capacity)
        energy_ratio = energy / initial_energy
        length = min(
            positions.shape[0], queue_ratio.size, energy_ratio.size
        )
        tokens = occupancy.setdefault((scale, policy), set())
        for index in range(length):
            token = (
                int(np.clip((positions[index, 0] - x_min) / x_span * 7, 0, 7)),
                int(np.clip((positions[index, 1] - y_min) / y_span * 7, 0, 7)),
                int(np.clip(queue_ratio[index] * 9, 0, 9)),
                int(np.clip(energy_ratio[index] * 9, 0, 9)),
            )
            tokens.add(token)
    scores: List[float] = []
    scales = sorted({scale for scale, _ in occupancy})
    for scale in scales:
        policies = sorted(
            policy for entry_scale, policy in occupancy if entry_scale == scale
        )
        for left_index, left in enumerate(policies):
            for right in policies[left_index + 1 :]:
                left_tokens = occupancy[(scale, left)]
                right_tokens = occupancy[(scale, right)]
                union = left_tokens | right_tokens
                if union:
                    scores.append(len(left_tokens & right_tokens) / len(union))
    return {
        "compared_policy_scale_pair_count": len(scores),
        "mean_pairwise_occupancy_jaccard": (
            float(np.mean(scores)) if scores else None
        ),
        "minimum_pairwise_occupancy_jaccard": (
            float(np.min(scores)) if scores else None
        ),
        "overlap_threshold": 0.05,
    }


def _representative_episodes(episodes: Sequence[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for episode in sorted(
        episodes,
        key=lambda item: (
            str(_value_alias(item, ("policy_id",))),
            int(_value_alias(item, ("seed",))),
            int(_value_alias(item, ("num_buoys",))),
            str(_value_alias(item, ("episode_id",))),
        ),
    ):
        result.setdefault(str(_value_alias(episode, ("policy_id",))), episode)
    return result


def _episode_path(episode: Any) -> np.ndarray:
    positions = _required_numeric_array(
        episode, ("uav_positions_m",), ndim=2
    )
    next_positions = _required_numeric_array(
        episode, ("next_uav_positions_m",), ndim=2
    )
    if positions.shape != next_positions.shape or positions.shape[1] != 2:
        raise ValueError("position arrays must both have shape [T,2]")
    return np.vstack((positions[0], next_positions))


def _nanmean_variable_curves(curves: Sequence[np.ndarray]) -> np.ndarray:
    if not curves:
        raise ValueError("at least one curve is required")
    maximum = max(curve.size for curve in curves)
    padded = np.full((len(curves), maximum), np.nan, dtype=np.float64)
    for index, curve in enumerate(curves):
        values = np.asarray(curve, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("curves must be finite one-dimensional arrays")
        padded[index, : values.size] = values
    return np.nanmean(padded, axis=0)


def _subplot_grid(count: int, *, max_columns: int) -> Tuple[int, int]:
    if count < 1:
        raise ValueError("subplot count must be positive")
    columns = min(max_columns, count)
    return int(math.ceil(count / columns)), columns


def _load_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for Round-3 figures"
        ) from error
    return plt


def _scientific_style() -> Dict[str, Any]:
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.7,
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "lines.linewidth": 1.4,
    }


def _palette(plt: Any, count: int) -> List[Any]:
    colormap = plt.get_cmap("tab10")
    return [colormap(index % 10) for index in range(count)]


def _save_close(plt: Any, figure: Any, destination: Path) -> None:
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    path: Union[str, Path],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {destination}"
        )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            missing = [field for field in fields if field not in row]
            if missing:
                raise ValueError(f"CSV row is missing fields: {missing}")
            _validate_json_finite(row)
            writer.writerow(
                {field: _csv_value(row[field]) for field in fields}
            )
    return destination


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, np.ndarray):
        return json.dumps(
            value.tolist(), ensure_ascii=False, separators=(",", ":")
        )
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_json_finite(value: Any, path: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{path}[{index}]")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(
            np.isfinite(value)
        ):
            raise ValueError(f"{path} contains NaN or Inf")
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} is NaN or Inf")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if numeric < 1 or numeric != float(value):
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real")
    return numeric


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ACTION_AGREEMENT_FIELDS",
    "ACTION_AGREEMENT_SCHEMA_VERSION",
    "EPISODE_METRIC_FIELDS_V1",
    "EXPECTED_WAYPOINT_HORIZONS",
    "QUALITY_AUDIT_FIELDS",
    "QUALITY_AUDIT_SCHEMA_VERSION",
    "QUALITY_HIGH",
    "QUALITY_LOW",
    "QUALITY_MID",
    "ROUND3_ANALYSIS_SCHEMA_VERSION",
    "ROUND3_FIGURE_FILENAMES",
    "SUMMARY_FIELDS_V1",
    "build_data_quality_audit",
    "classify_episode_quality",
    "compute_action_agreement_matrix",
    "compute_episode_metric_rows_v1",
    "compute_episode_metrics_v1",
    "generate_round3_figures",
    "run_round3_analysis",
    "summarize_episode_metrics_v1",
    "summary",
    "write_action_agreement_matrix_csv",
    "write_baseline_summary_csv_v1",
    "write_data_quality_audit_csv",
    "write_episode_metrics_csv_v1",
]
