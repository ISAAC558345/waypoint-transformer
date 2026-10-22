"""Round-5 offline two-dimensional action evaluation.

This module is deliberately independent of the policy implementation.  It
accepts recorded action labels and predictions in normalized units, converts
them to metres using the environment movement limit, applies the radial
deployment projection, and produces finite CSV/JSON-friendly records.

No function in this file advances an environment or performs model selection.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class ActionProjection:
    """Raw and radially projected actions in normalized and physical units."""

    raw_normalized: np.ndarray
    raw_m: np.ndarray
    projected_normalized: np.ndarray
    projected_m: np.ndarray
    activation_mask: np.ndarray
    raw_valid_mask: np.ndarray
    projected_valid_mask: np.ndarray


@dataclass(frozen=True)
class ActionEvaluation:
    """Complete offline evaluation result with no non-serializable values."""

    raw_metrics: Mapping[str, Any]
    projected_metrics: Mapping[str, Any]
    grouped_metrics: Tuple[Mapping[str, Any], ...]
    prediction_rows: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "raw_metrics": dict(self.raw_metrics),
            "projected_metrics": dict(self.projected_metrics),
            "grouped_metrics": [dict(row) for row in self.grouped_metrics],
            "prediction_rows": [dict(row) for row in self.prediction_rows],
        }
        assert_finite_tree(result)
        return result


_METADATA_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "episode_id": ("episode_id", "trajectory_id", "episode"),
    "timestep": ("timestep", "t", "step", "transition_index"),
    "policy": ("policy", "policy_id", "collection_policy", "heuristic_policy"),
    "num_buoys": ("num_buoys", "buoy_count", "n_buoys"),
    "quality": ("quality", "quality_tier", "trajectory_quality"),
    "episode_length": ("episode_length", "trajectory_length", "length"),
}


def radial_action_projection(
    predicted_normalized: ArrayLike,
    max_step_distance_m: float,
    *,
    tolerance_m: float = 1e-9,
) -> ActionProjection:
    """Project predicted 2-D displacements onto the movement-radius disk.

    The input is not clipped component-wise.  Predictions outside the radial
    action disk are scaled along their original direction.  Thus this function
    exactly implements the interface projection, not a second training label.
    """

    maximum = _positive_finite(max_step_distance_m, "max_step_distance_m")
    tolerance = _nonnegative_finite(tolerance_m, "tolerance_m")
    normalized = _action_array(predicted_normalized, "predicted_normalized")
    raw_m = normalized * maximum
    magnitude = np.linalg.norm(raw_m, axis=1)
    activation = magnitude > maximum + tolerance
    scale = np.ones_like(magnitude)
    nonzero = magnitude > 0.0
    scale[activation & nonzero] = maximum / magnitude[activation & nonzero]
    projected_m = raw_m * scale[:, None]
    projected_normalized = projected_m / maximum
    raw_valid = magnitude <= maximum + tolerance
    projected_valid = np.linalg.norm(projected_m, axis=1) <= maximum + tolerance
    for name, values in (
        ("raw_m", raw_m),
        ("projected_m", projected_m),
        ("projected_normalized", projected_normalized),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or Inf")
    return ActionProjection(
        raw_normalized=normalized.copy(),
        raw_m=raw_m,
        projected_normalized=projected_normalized,
        projected_m=projected_m,
        activation_mask=activation,
        raw_valid_mask=raw_valid,
        projected_valid_mask=projected_valid,
    )


def compute_action_metrics(
    target_normalized: ArrayLike,
    predicted_normalized: ArrayLike,
    max_step_distance_m: float,
    *,
    near_zero_epsilon_m: float = 1e-6,
    prediction_space: str = "raw",
    projection_activation_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Compute finite aggregate metrics for one prediction representation.

    Direction metrics exclude a sample unless *both* true and predicted action
    magnitudes exceed ``near_zero_epsilon_m``.  Empty direction/sign subsets
    report a finite value of zero together with an eligibility count of zero;
    consumers must use the count/availability field when interpreting them.
    """

    maximum = _positive_finite(max_step_distance_m, "max_step_distance_m")
    epsilon = _nonnegative_finite(near_zero_epsilon_m, "near_zero_epsilon_m")
    target = _action_array(target_normalized, "target_normalized")
    prediction = _action_array(predicted_normalized, "predicted_normalized")
    if target.shape != prediction.shape:
        raise ValueError(
            f"target and prediction shapes differ: {target.shape} versus {prediction.shape}"
        )
    if target.shape[0] == 0:
        raise ValueError("action metrics require at least one sample")

    target_m = target * maximum
    prediction_m = prediction * maximum
    error_normalized = prediction - target
    error_m = prediction_m - target_m
    error_l2_m = np.linalg.norm(error_m, axis=1)
    target_magnitude = np.linalg.norm(target_m, axis=1)
    prediction_magnitude = np.linalg.norm(prediction_m, axis=1)
    magnitude_error = np.abs(prediction_magnitude - target_magnitude)

    direction_eligible = (target_magnitude > epsilon) & (prediction_magnitude > epsilon)
    cosine = np.zeros(target.shape[0], dtype=np.float64)
    angular = np.zeros(target.shape[0], dtype=np.float64)
    if np.any(direction_eligible):
        denominator = target_magnitude[direction_eligible] * prediction_magnitude[direction_eligible]
        cosine_values = np.sum(
            target_m[direction_eligible] * prediction_m[direction_eligible], axis=1
        ) / denominator
        cosine_values = np.clip(cosine_values, -1.0, 1.0)
        cosine[direction_eligible] = cosine_values
        angular[direction_eligible] = np.degrees(np.arccos(cosine_values))

    activation = (
        np.zeros(target.shape[0], dtype=bool)
        if projection_activation_mask is None
        else np.asarray(projection_activation_mask, dtype=bool).reshape(-1)
    )
    if activation.shape != (target.shape[0],):
        raise ValueError("projection_activation_mask length does not match action samples")

    result: Dict[str, Any] = {
        "prediction_space": str(prediction_space),
        "sample_count": int(target.shape[0]),
        "action_dimension": int(target.shape[1]),
        "max_step_distance_m": maximum,
        # Round-5 policy loss is mean_t ||a_hat_t - a_t||_2^2, not the
        # component-wise mean.  Keep the latter under an explicit name only.
        "normalized_action_mse": float(
            np.mean(np.sum(np.square(error_normalized), axis=1))
        ),
        "normalized_action_component_mse": float(np.mean(np.square(error_normalized))),
        "normalized_action_l2_rmse": float(
            np.sqrt(np.mean(np.sum(np.square(error_normalized), axis=1)))
        ),
        "action_component_mae_m": float(np.mean(np.abs(error_m))),
        "action_l2_rmse_m": float(np.sqrt(np.mean(np.square(error_l2_m)))),
        "action_l2_mae_m": float(np.mean(error_l2_m)),
        "endpoint_error_m": float(np.mean(error_l2_m)),
        "endpoint_rmse_m": float(np.sqrt(np.mean(np.square(error_l2_m)))),
        "direction_metric_eligible_count": int(np.sum(direction_eligible)),
        "direction_metric_eligible_rate": float(np.mean(direction_eligible)),
        "direction_metric_available": bool(np.any(direction_eligible)),
        "action_cosine_similarity": _mean_or_zero(cosine[direction_eligible]),
        "action_angular_error_deg": _mean_or_zero(angular[direction_eligible]),
        "action_angular_rmse_deg": _rms_or_zero(angular[direction_eligible]),
        "action_magnitude_mae_m": float(np.mean(magnitude_error)),
        "action_magnitude_rmse_m": float(np.sqrt(np.mean(np.square(magnitude_error)))),
        # Recorded environment displacements are persisted as float32.  A
        # nominally exact boundary action can therefore exceed ``maximum`` by
        # a few 1e-7 metres after the position difference is reconstructed.
        # Reuse the configured numerical epsilon for the validity *metric* so
        # those executed boundary actions are not falsely reported invalid.
        # Deployment projection remains a separate, explicit operation.
        "action_validity_rate": float(
            np.mean(prediction_magnitude <= maximum + max(1e-9, epsilon))
        ),
        "target_action_validity_rate": float(
            np.mean(target_magnitude <= maximum + max(1e-9, epsilon))
        ),
        "projection_activation_count": int(np.sum(activation)),
        "projection_activation_rate": float(np.mean(activation)),
    }
    for component, label in enumerate(("x", "y")):
        eligible = np.abs(target_m[:, component]) > epsilon
        correct = np.sign(prediction_m[:, component]) == np.sign(target_m[:, component])
        result[f"sign_{label}_eligible_count"] = int(np.sum(eligible))
        result[f"sign_{label}_accuracy"] = _mean_or_zero(correct[eligible].astype(np.float64))

    result.update(_distribution("action_error_l2_m", error_l2_m))
    result.update(_distribution("action_error_abs_component_m", np.abs(error_m).reshape(-1)))
    result.update(_distribution("target_action_magnitude_m", target_magnitude))
    result.update(_distribution("predicted_action_magnitude_m", prediction_magnitude))
    result.update(_distribution("action_magnitude_error_m", magnitude_error))
    result.update(_distribution("angular_error_deg_eligible", angular[direction_eligible]))
    assert_finite_tree(result)
    return result


def evaluate_action_predictions(
    target_normalized: ArrayLike,
    predicted_normalized: ArrayLike,
    max_step_distance_m: float,
    *,
    metadata: Optional[Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]]]] = None,
    near_zero_epsilon_m: float = 1e-6,
    magnitude_bands_m: Optional[Sequence[float]] = None,
    base_fields: Optional[Mapping[str, Any]] = None,
) -> ActionEvaluation:
    """Evaluate raw/projected actions overall, by group, and per sample."""

    target = _action_array(target_normalized, "target_normalized")
    prediction = _action_array(predicted_normalized, "predicted_normalized")
    if target.shape != prediction.shape:
        raise ValueError(
            f"target and prediction shapes differ: {target.shape} versus {prediction.shape}"
        )
    rows = _metadata_rows(metadata, target.shape[0])
    projection = radial_action_projection(prediction, max_step_distance_m)
    raw = compute_action_metrics(
        target,
        projection.raw_normalized,
        max_step_distance_m,
        near_zero_epsilon_m=near_zero_epsilon_m,
        prediction_space="raw",
        projection_activation_mask=projection.activation_mask,
    )
    projected = compute_action_metrics(
        target,
        projection.projected_normalized,
        max_step_distance_m,
        near_zero_epsilon_m=near_zero_epsilon_m,
        prediction_space="projected",
        projection_activation_mask=projection.activation_mask,
    )
    prefix = dict(base_fields or {})
    raw = {**prefix, **raw}
    projected = {**prefix, **projected}
    grouped = grouped_action_metrics(
        target,
        prediction,
        max_step_distance_m,
        metadata=rows,
        near_zero_epsilon_m=near_zero_epsilon_m,
        magnitude_bands_m=magnitude_bands_m,
        base_fields=prefix,
    )
    predictions = build_action_prediction_rows(
        target,
        prediction,
        max_step_distance_m,
        metadata=rows,
        near_zero_epsilon_m=near_zero_epsilon_m,
        base_fields=prefix,
    )
    result = ActionEvaluation(
        raw_metrics=raw,
        projected_metrics=projected,
        grouped_metrics=tuple(grouped),
        prediction_rows=tuple(predictions),
    )
    result.to_dict()  # eager finite-value contract
    return result


def grouped_action_metrics(
    target_normalized: ArrayLike,
    predicted_normalized: ArrayLike,
    max_step_distance_m: float,
    *,
    metadata: Optional[Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]]]] = None,
    near_zero_epsilon_m: float = 1e-6,
    magnitude_bands_m: Optional[Sequence[float]] = None,
    base_fields: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return raw/projected metrics for all required Round-5 strata."""

    maximum = _positive_finite(max_step_distance_m, "max_step_distance_m")
    target = _action_array(target_normalized, "target_normalized")
    prediction = _action_array(predicted_normalized, "predicted_normalized")
    if target.shape != prediction.shape:
        raise ValueError("target and prediction arrays must have identical shapes")
    if target.shape[0] == 0:
        raise ValueError("grouped metrics require at least one sample")
    rows = _metadata_rows(metadata, target.shape[0])
    canonical = _canonical_metadata(rows)
    bands = _magnitude_band_labels(
        np.linalg.norm(target * maximum, axis=1),
        magnitude_bands_m,
        maximum,
    )
    dimensions: List[Tuple[str, np.ndarray]] = [
        ("overall", np.full(target.shape[0], "all", dtype=object)),
        ("policy", canonical["policy"]),
        ("num_buoys", canonical["num_buoys"]),
        ("quality", canonical["quality"]),
        ("episode_length", canonical["episode_length"]),
        ("action_magnitude_band", bands),
    ]
    projection = radial_action_projection(prediction, maximum)
    prefix = dict(base_fields or {})
    output: List[Dict[str, Any]] = []
    for dimension, values in dimensions:
        for group_value in _stable_unique(values.tolist()):
            selected = values == group_value
            if not np.any(selected):
                continue
            for space, predictions in (
                ("raw", projection.raw_normalized),
                ("projected", projection.projected_normalized),
            ):
                metrics = compute_action_metrics(
                    target[selected],
                    predictions[selected],
                    maximum,
                    near_zero_epsilon_m=near_zero_epsilon_m,
                    prediction_space=space,
                    projection_activation_mask=projection.activation_mask[selected],
                )
                record = {
                    **prefix,
                    "group_dimension": dimension,
                    "group_value": str(group_value),
                    **metrics,
                }
                assert_finite_tree(record)
                output.append(record)
    return output


def build_action_prediction_rows(
    target_normalized: ArrayLike,
    predicted_normalized: ArrayLike,
    max_step_distance_m: float,
    *,
    metadata: Optional[Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]]]] = None,
    near_zero_epsilon_m: float = 1e-6,
    base_fields: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build finite per-transition records used by replay and figures."""

    maximum = _positive_finite(max_step_distance_m, "max_step_distance_m")
    epsilon = _nonnegative_finite(near_zero_epsilon_m, "near_zero_epsilon_m")
    target = _action_array(target_normalized, "target_normalized")
    prediction = _action_array(predicted_normalized, "predicted_normalized")
    if target.shape != prediction.shape:
        raise ValueError("target and prediction arrays must have identical shapes")
    metadata_rows = _metadata_rows(metadata, target.shape[0])
    projection = radial_action_projection(prediction, maximum)
    target_m = target * maximum
    raw_m = projection.raw_m
    projected_m = projection.projected_m
    prefix = dict(base_fields or {})
    result: List[Dict[str, Any]] = []
    for index in range(target.shape[0]):
        true_magnitude = float(np.linalg.norm(target_m[index]))
        raw_magnitude = float(np.linalg.norm(raw_m[index]))
        eligible = true_magnitude > epsilon and raw_magnitude > epsilon
        if eligible:
            cosine = float(
                np.clip(
                    np.dot(target_m[index], raw_m[index]) / (true_magnitude * raw_magnitude),
                    -1.0,
                    1.0,
                )
            )
            angle = float(np.degrees(np.arccos(cosine)))
        else:
            cosine = 0.0
            angle = 0.0
        record: Dict[str, Any] = {**metadata_rows[index], **prefix}
        record.update(
            {
                "sample_index": index,
                "true_action_normalized_x": float(target[index, 0]),
                "true_action_normalized_y": float(target[index, 1]),
                "predicted_action_raw_normalized_x": float(prediction[index, 0]),
                "predicted_action_raw_normalized_y": float(prediction[index, 1]),
                "predicted_action_projected_normalized_x": float(
                    projection.projected_normalized[index, 0]
                ),
                "predicted_action_projected_normalized_y": float(
                    projection.projected_normalized[index, 1]
                ),
                "true_action_x_m": float(target_m[index, 0]),
                "true_action_y_m": float(target_m[index, 1]),
                "predicted_action_raw_x_m": float(raw_m[index, 0]),
                "predicted_action_raw_y_m": float(raw_m[index, 1]),
                "predicted_action_projected_x_m": float(projected_m[index, 0]),
                "predicted_action_projected_y_m": float(projected_m[index, 1]),
                "true_action_magnitude_m": true_magnitude,
                "predicted_action_raw_magnitude_m": raw_magnitude,
                "predicted_action_projected_magnitude_m": float(
                    np.linalg.norm(projected_m[index])
                ),
                "raw_action_error_l2_m": float(np.linalg.norm(raw_m[index] - target_m[index])),
                "projected_action_error_l2_m": float(
                    np.linalg.norm(projected_m[index] - target_m[index])
                ),
                "direction_metric_eligible": bool(eligible),
                "raw_action_cosine_similarity": cosine,
                "raw_action_angular_error_deg": angle,
                "raw_action_valid": bool(projection.raw_valid_mask[index]),
                "projected_action_valid": bool(projection.projected_valid_mask[index]),
                "projection_activated": bool(projection.activation_mask[index]),
            }
        )
        assert_finite_tree(record)
        result.append(record)
    return result


def write_records_csv(path: PathLike, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write flat records after enforcing the finite-value contract."""

    destination = Path(path)
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("cannot write an empty CSV")
    for row in rows:
        assert_finite_tree(row)
        if any(isinstance(value, (Mapping, list, tuple, np.ndarray)) for value in row.values()):
            raise TypeError("CSV records must be flat")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_finite_json(path: PathLike, payload: Any) -> Path:
    """Write strict JSON (``allow_nan=False``) after recursive validation."""

    destination = Path(path)
    assert_finite_tree(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return destination


def assert_finite_tree(value: Any, *, path: str = "root") -> None:
    """Reject NaN/Inf and unsupported numpy containers before serialization."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, path=f"{path}[{index}]")
        return
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{path} contains NaN or Inf")
        return
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} is not finite")
        return
    if isinstance(value, np.integer):
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    raise TypeError(f"{path} has unsupported serializable type {type(value).__name__}")


def _action_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _metadata_rows(
    metadata: Optional[Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]]]],
    count: int,
) -> List[Dict[str, Any]]:
    if metadata is None:
        return [{} for _ in range(count)]
    if isinstance(metadata, Mapping):
        rows = [{} for _ in range(count)]
        for key, raw_values in metadata.items():
            if isinstance(raw_values, (str, bytes)) or np.asarray(raw_values).ndim == 0:
                values = [raw_values] * count
            else:
                values = list(raw_values)
            if len(values) != count:
                raise ValueError(f"metadata column {key!r} has {len(values)} rows, expected {count}")
            for index, value in enumerate(values):
                rows[index][str(key)] = _python_scalar(value)
        return rows
    rows = [dict(row) for row in metadata]
    if len(rows) != count:
        raise ValueError(f"metadata has {len(rows)} rows, expected {count}")
    for row in rows:
        assert_finite_tree(row)
    return rows


def _canonical_metadata(rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    episode_ids = np.asarray(
        [_metadata_value(row, "episode_id", f"unknown_episode_{index}") for index, row in enumerate(rows)],
        dtype=object,
    )
    timesteps = np.asarray(
        [_metadata_value(row, "timestep", index) for index, row in enumerate(rows)], dtype=object
    )
    inferred_lengths: Dict[str, int] = {}
    for episode in _stable_unique(episode_ids.tolist()):
        selected = episode_ids == episode
        numeric_steps: List[int] = []
        for value in timesteps[selected]:
            try:
                numeric_steps.append(int(value))
            except (TypeError, ValueError):
                pass
        inferred_lengths[str(episode)] = max(numeric_steps) + 1 if numeric_steps else int(np.sum(selected))

    values: Dict[str, np.ndarray] = {
        "episode_id": episode_ids,
        "timestep": timesteps,
    }
    for key in ("policy", "num_buoys", "quality"):
        values[key] = np.asarray(
            [_display_group(_metadata_value(row, key, "unknown")) for row in rows], dtype=object
        )
    lengths: List[str] = []
    for index, row in enumerate(rows):
        explicit = _metadata_value(row, "episode_length", None)
        length = inferred_lengths[str(episode_ids[index])] if explicit is None else int(explicit)
        lengths.append(str(length))
    values["episode_length"] = np.asarray(lengths, dtype=object)
    return values


def _metadata_value(row: Mapping[str, Any], canonical: str, default: Any) -> Any:
    for alias in _METADATA_ALIASES[canonical]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return default


def _magnitude_band_labels(
    magnitudes_m: np.ndarray,
    edges: Optional[Sequence[float]],
    maximum: float,
) -> np.ndarray:
    if edges is None:
        boundaries = np.asarray([0.0, maximum / 3.0, 2.0 * maximum / 3.0, maximum + 1e-9])
    else:
        boundaries = np.asarray(edges, dtype=np.float64).reshape(-1)
    if boundaries.size < 2 or not np.all(np.isfinite(boundaries)):
        raise ValueError("magnitude_bands_m must contain at least two finite edges")
    if np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("magnitude_bands_m must be strictly increasing")
    if boundaries[0] > 0.0 or boundaries[-1] < float(np.max(magnitudes_m)):
        raise ValueError("magnitude_bands_m do not cover all target action magnitudes")
    indices = np.searchsorted(boundaries[1:-1], magnitudes_m, side="right")
    labels = [
        f"[{_compact_number(boundaries[index])},{_compact_number(boundaries[index + 1])}{']' if index == boundaries.size - 2 else ')'}m"
        for index in indices
    ]
    return np.asarray(labels, dtype=object)


def _distribution(prefix: str, values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{prefix} distribution contains NaN or Inf")
    if array.size == 0:
        statistics = {name: 0.0 for name in ("mean", "std", "median", "q25", "q75", "min", "max")}
    else:
        statistics = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
    return {f"{prefix}_{name}": value for name, value in statistics.items()} | {
        f"{prefix}_count": int(array.size)
    }


def _mean_or_zero(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.mean(values))


def _rms_or_zero(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.sqrt(np.mean(np.square(values))))


def _stable_unique(values: Sequence[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        token = (type(value).__name__, str(value))
        if token not in seen:
            seen.add(token)
            result.append(value)
    return result


def _display_group(value: Any) -> str:
    return "unknown" if value in (None, "") else str(_python_scalar(value))


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _compact_number(value: float) -> str:
    return f"{float(value):.6g}"


def _positive_finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _nonnegative_finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return number


# Compact aliases used by runners and tests.
evaluate_actions = evaluate_action_predictions
project_actions_radially = radial_action_projection
group_action_metrics = grouped_action_metrics


__all__ = [
    "ActionEvaluation",
    "ActionProjection",
    "assert_finite_tree",
    "build_action_prediction_rows",
    "compute_action_metrics",
    "evaluate_action_predictions",
    "evaluate_actions",
    "group_action_metrics",
    "grouped_action_metrics",
    "project_actions_radially",
    "radial_action_projection",
    "write_finite_json",
    "write_records_csv",
]
