"""Metrics for non-learning maritime baseline rollouts.

The stage-1A environment exposes link capacity, energy, position, and action
processing information.  It does *not* expose buoy queues, task completion, or
actual transmitted payload.  This module therefore reports the former using
their physical names and marks the latter explicitly unavailable instead of
manufacturing proxy queue or service values.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np


UNAVAILABLE_STATUS = "current MaritimeEnv does not provide this variable"

EPISODE_METRIC_FIELDS = (
    "episode_id",
    "policy_id",
    "seed",
    "scenario_id",
    "num_buoys",
    "episode_return",
    "episode_length",
    "terminal_reason",
    "terminated",
    "truncated",
    "cumulative_link_capacity_bits",
    "jain_link_capacity_index",
    "cumulative_energy_consumed",
    "initial_energy",
    "final_remaining_energy",
    "total_distance_m",
    "average_total_rate_bps",
    "constraint_violation_count",
    "boundary_violation_count",
    "invalid_action_count",
    "action_component_clipping_count",
    "action_norm_projection_count",
    "nan_inf_count",
    "task_success_available",
    "task_success",
    "task_completion_available",
    "task_completion_rate",
    "cumulative_collected_data_available",
    "cumulative_collected_data_bits",
    "final_remaining_queue_available",
    "final_remaining_queue",
    "per_buoy_service_completion_available",
    "per_buoy_service_completion",
    "jain_service_fairness_available",
    "jain_service_fairness_index",
    "unavailable_metric_status",
)

SUMMARY_METRIC_FIELDS = (
    "policy_id",
    "episode_count",
    "scenario_ids",
    "num_buoy_values",
    "episode_return_mean",
    "episode_return_std",
    "episode_length_mean",
    "cumulative_link_capacity_bits_mean",
    "jain_link_capacity_index_mean",
    "cumulative_energy_consumed_mean",
    "cumulative_energy_consumed_std",
    "final_remaining_energy_mean",
    "total_distance_m_mean",
    "average_total_rate_bps_mean",
    "constraint_violation_count_total",
    "boundary_violation_count_total",
    "invalid_action_count_total",
    "nan_inf_count_total",
    "task_success_available",
    "task_success_rate",
    "task_completion_available",
    "task_completion_rate_mean",
    "cumulative_collected_data_available",
    "cumulative_collected_data_bits_mean",
    "final_remaining_queue_available",
    "final_remaining_queue_mean",
    "per_buoy_service_completion_available",
    "jain_service_fairness_available",
    "jain_service_fairness_index_mean",
    "unavailable_metric_status",
)


def jain_index(
    values: Union[Sequence[float], np.ndarray]
) -> Optional[float]:
    """Return Jain's index for finite nonnegative values.

    ``None`` is returned for an empty vector or an all-zero vector because the
    mathematical ratio is then undefined.  Invalid negative or non-finite
    input is rejected rather than silently repaired.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Jain index input must be one-dimensional, got {array.shape}")
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        raise ValueError("Jain index input contains NaN or Inf")
    if np.any(array < 0.0):
        raise ValueError("Jain index input contains negative values")
    squared_sum = float(np.sum(array) ** 2)
    denominator = float(array.size * np.sum(np.square(array)))
    if denominator == 0.0:
        return None
    value = squared_sum / denominator
    return float(min(1.0, max(0.0, value)))


def compute_episode_metrics(
    episode: Any,
    *,
    delta_t: float,
) -> Dict[str, Any]:
    """Compute one row of real, finite metrics from an episode trajectory.

    Link rates are integrated over ``delta_t`` and deliberately named
    ``cumulative_link_capacity_bits``.  They are a channel-capacity proxy, not
    proof that a payload was queued or transmitted.
    """

    delta_t_value = float(delta_t)
    if not math.isfinite(delta_t_value) or delta_t_value <= 0.0:
        raise ValueError("delta_t must be finite and positive")

    rewards = _as_array(episode, "rewards", dtype=np.float64, ndim=1)
    episode_length = int(_value(episode, "episode_length", rewards.size))
    if episode_length != rewards.size:
        raise ValueError(
            "episode_length must equal the number of rewards: "
            f"{episode_length} != {rewards.size}"
        )
    if episode_length < 1:
        raise ValueError("an episode must contain at least one transition")

    rates = _as_array(
        episode, "next_rates_bps", dtype=np.float64, ndim=2
    )
    if rates.shape[0] != episode_length:
        raise ValueError(
            "next_rates_bps first dimension must equal episode_length"
        )
    num_buoys = int(_value(episode, "num_buoys"))
    if rates.shape[1] != num_buoys:
        raise ValueError(
            f"next_rates_bps must have shape ({episode_length}, {num_buoys}), "
            f"got {rates.shape}"
        )
    total_rates = _as_array(
        episode, "next_total_rate_bps", dtype=np.float64, ndim=1
    )
    if total_rates.shape != (episode_length,):
        raise ValueError(
            "next_total_rate_bps must have one value per transition"
        )
    if not np.allclose(
        np.sum(rates, axis=1),
        total_rates,
        rtol=1.0e-10,
        atol=1.0e-6,
    ):
        raise ValueError(
            "next_total_rate_bps does not equal the next per-buoy rate sum"
        )

    energy_consumed = _as_array(
        episode, "actual_energy_consumed", dtype=np.float64, ndim=1
    )
    remaining_energy = _as_array(
        episode, "remaining_energy", dtype=np.float64, ndim=1
    )
    next_remaining_energy = _as_array(
        episode, "next_remaining_energy", dtype=np.float64, ndim=1
    )
    positions = _as_array(
        episode, "uav_positions_m", dtype=np.float64, ndim=2
    )
    next_positions = _as_array(
        episode, "next_uav_positions_m", dtype=np.float64, ndim=2
    )
    for name, array in (
        ("actual_energy_consumed", energy_consumed),
        ("remaining_energy", remaining_energy),
        ("next_remaining_energy", next_remaining_energy),
    ):
        if array.shape != (episode_length,):
            raise ValueError(f"{name} must have one value per transition")
    for name, array in (
        ("uav_positions_m", positions),
        ("next_uav_positions_m", next_positions),
    ):
        if array.shape != (episode_length, 2):
            raise ValueError(
                f"{name} must have shape ({episode_length}, 2), got {array.shape}"
            )

    boundary = _as_bool_array(episode, "boundary_violation", episode_length)
    component_clipped = _as_bool_array(
        episode, "action_component_clipped", episode_length
    )
    norm_projected = _as_bool_array(
        episode, "action_norm_projected", episode_length
    )
    invalid_action = np.logical_or(component_clipped, norm_projected)
    constraint_violation = _as_bool_array(
        episode, "constraint_violation", episode_length
    )
    terminated_values = _as_bool_array(episode, "terminated", episode_length)
    truncated_values = _as_bool_array(episode, "truncated", episode_length)

    arrays_for_finiteness = (
        rewards,
        rates,
        total_rates,
        energy_consumed,
        remaining_energy,
        next_remaining_energy,
        positions,
        next_positions,
        _as_array(episode, "observations", dtype=np.float64),
        _as_array(episode, "actions", dtype=np.float64),
        _as_array(episode, "next_observations", dtype=np.float64),
    )
    nan_inf_count = int(
        sum(np.size(array) - np.count_nonzero(np.isfinite(array)) for array in arrays_for_finiteness)
    )

    cumulative_per_buoy_capacity = np.sum(rates, axis=0) * delta_t_value
    link_capacity_jain = jain_index(cumulative_per_buoy_capacity)
    cumulative_capacity = float(np.sum(cumulative_per_buoy_capacity))
    path_distance = float(
        np.sum(np.linalg.norm(next_positions - positions, axis=1))
    )
    episode_return = float(np.sum(rewards))
    stored_return = float(_value(episode, "episode_return", episode_return))
    if not math.isclose(
        episode_return, stored_return, rel_tol=1.0e-10, abs_tol=1.0e-10
    ):
        raise ValueError("stored episode_return does not equal sum(rewards)")

    row: Dict[str, Any] = {
        "episode_id": str(_value(episode, "episode_id")),
        "policy_id": str(_value(episode, "policy_id")),
        "seed": int(_value(episode, "seed")),
        "scenario_id": str(_value(episode, "scenario_id")),
        "num_buoys": num_buoys,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "terminal_reason": str(_value(episode, "terminal_reason")),
        "terminated": bool(terminated_values[-1]),
        "truncated": bool(truncated_values[-1]),
        "cumulative_link_capacity_bits": cumulative_capacity,
        "jain_link_capacity_index": link_capacity_jain,
        "cumulative_energy_consumed": float(np.sum(energy_consumed)),
        "initial_energy": float(remaining_energy[0]),
        "final_remaining_energy": float(next_remaining_energy[-1]),
        "total_distance_m": path_distance,
        "average_total_rate_bps": float(np.mean(total_rates)),
        "constraint_violation_count": int(
            np.count_nonzero(constraint_violation)
        ),
        "boundary_violation_count": int(np.count_nonzero(boundary)),
        "invalid_action_count": int(np.count_nonzero(invalid_action)),
        "action_component_clipping_count": int(
            np.count_nonzero(component_clipped)
        ),
        "action_norm_projection_count": int(np.count_nonzero(norm_projected)),
        "nan_inf_count": nan_inf_count,
        "task_success_available": False,
        "task_success": None,
        "task_completion_available": False,
        "task_completion_rate": None,
        "cumulative_collected_data_available": False,
        "cumulative_collected_data_bits": None,
        "final_remaining_queue_available": False,
        "final_remaining_queue": None,
        "per_buoy_service_completion_available": False,
        "per_buoy_service_completion": None,
        "jain_service_fairness_available": False,
        "jain_service_fairness_index": None,
        "unavailable_metric_status": UNAVAILABLE_STATUS,
    }
    _validate_metric_row(row)
    return row


def summarize_episode_metrics(
    rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate episode rows by policy without converting unavailable data."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        policy_id = str(row["policy_id"])
        grouped.setdefault(policy_id, []).append(row)

    summaries: List[Dict[str, Any]] = []
    for policy_id in sorted(grouped):
        group = grouped[policy_id]
        summary: Dict[str, Any] = {
            "policy_id": policy_id,
            "episode_count": len(group),
            "scenario_ids": "|".join(
                sorted({str(row["scenario_id"]) for row in group})
            ),
            "num_buoy_values": "|".join(
                str(value)
                for value in sorted({int(row["num_buoys"]) for row in group})
            ),
            "episode_return_mean": _mean(group, "episode_return"),
            "episode_return_std": _std(group, "episode_return"),
            "episode_length_mean": _mean(group, "episode_length"),
            "cumulative_link_capacity_bits_mean": _mean(
                group, "cumulative_link_capacity_bits"
            ),
            "jain_link_capacity_index_mean": _optional_mean(
                group, "jain_link_capacity_index"
            ),
            "cumulative_energy_consumed_mean": _mean(
                group, "cumulative_energy_consumed"
            ),
            "cumulative_energy_consumed_std": _std(
                group, "cumulative_energy_consumed"
            ),
            "final_remaining_energy_mean": _mean(
                group, "final_remaining_energy"
            ),
            "total_distance_m_mean": _mean(group, "total_distance_m"),
            "average_total_rate_bps_mean": _mean(
                group, "average_total_rate_bps"
            ),
            "constraint_violation_count_total": _integer_sum(
                group, "constraint_violation_count"
            ),
            "boundary_violation_count_total": _integer_sum(
                group, "boundary_violation_count"
            ),
            "invalid_action_count_total": _integer_sum(
                group, "invalid_action_count"
            ),
            "nan_inf_count_total": _integer_sum(group, "nan_inf_count"),
            "task_success_available": False,
            "task_success_rate": None,
            "task_completion_available": False,
            "task_completion_rate_mean": None,
            "cumulative_collected_data_available": False,
            "cumulative_collected_data_bits_mean": None,
            "final_remaining_queue_available": False,
            "final_remaining_queue_mean": None,
            "per_buoy_service_completion_available": False,
            "jain_service_fairness_available": False,
            "jain_service_fairness_index_mean": None,
            "unavailable_metric_status": UNAVAILABLE_STATUS,
        }
        _validate_metric_row(summary)
        summaries.append(summary)
    return summaries


def write_episode_metrics_csv(
    rows: Sequence[Mapping[str, Any]], path: Union[str, Path]
) -> Path:
    """Write stable per-episode columns without pandas-specific NaN values."""

    return _write_csv(rows, EPISODE_METRIC_FIELDS, path)


def write_baseline_summary_csv(
    rows: Sequence[Mapping[str, Any]], path: Union[str, Path]
) -> Path:
    """Write stable per-policy aggregate columns."""

    return _write_csv(rows, SUMMARY_METRIC_FIELDS, path)


def _value(record: Any, name: str, default: Any = ...) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is not ...:
        return default
    raise KeyError(f"episode record is missing required field {name!r}")


def _as_array(
    record: Any,
    name: str,
    *,
    dtype: Any,
    ndim: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(_value(record, name), dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    return array


def _as_bool_array(record: Any, name: str, length: int) -> np.ndarray:
    array = _as_array(record, name, dtype=np.bool_, ndim=1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _std(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.std([float(row[key]) for row in rows], ddof=0))


def _optional_mean(
    rows: Sequence[Mapping[str, Any]], key: str
) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _integer_sum(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(int(row[key]) for row in rows))


def _validate_metric_row(row: Mapping[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(
            float(value)
        ):
            raise ValueError(f"metric {key!r} is NaN or Inf")


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    path: Union[str, Path],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            missing = [field for field in fields if field not in row]
            if missing:
                raise ValueError(f"CSV row is missing fields: {missing}")
            _validate_metric_row(row)
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
        return json.dumps(value.tolist(), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value
