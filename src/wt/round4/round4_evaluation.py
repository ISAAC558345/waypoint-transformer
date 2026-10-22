"""Pure NumPy evaluation utilities for Round-4 spatial waypoint models.

The utilities in this module deliberately operate on arrays and metadata
mappings only.  They do not load datasets, train models, select checkpoints,
or write artifacts.  In particular, :func:`fit_train_mean_baseline` requires
an explicit all-``train`` split vector so validation or test targets cannot be
silently used to fit that baseline.

The spatial target is the two-dimensional relative displacement

``delta_normalized = (q[t + H] - q[t]) / (H * maximum_step_distance_m)``.

Consequently the unit disk is the physical reachability set.  The canonical
``normalized_squared_norm_mse`` below is the mean squared Euclidean norm,
``mean(||prediction - target||_2 ** 2)``, rather than a component-wise MSE.
Raw-displacement and endpoint RMSE/MAE use Euclidean sample errors.  Explicit
component-wise variants are also returned so the convention is unambiguous.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]
Metadata = Mapping[str, Any]

DEFAULT_GROUP_FIELDS: Tuple[str, ...] = (
    "policy_id",
    "num_buoys",
    "quality_tier",
    "episode_length_band",
    "goal_source",
)

SUMMARY_STATISTICS: Tuple[str, ...] = (
    "mean",
    "std",
    "median",
    "q25",
    "q75",
)

_DISTRIBUTION_METRICS: Tuple[str, ...] = (
    "normalized_squared_norm_error",
    "normalized_component_rmse",
    "normalized_component_mae",
    "raw_delta_l2_error_m",
    "raw_delta_component_rmse_m",
    "raw_delta_component_mae_m",
    "endpoint_l2_error_m",
    "endpoint_component_rmse_m",
    "endpoint_component_mae_m",
    "direction_cosine_similarity",
    "direction_cosine_error",
    "angular_error_rad",
    "angular_error_deg",
    "predicted_magnitude_normalized",
    "target_magnitude_normalized",
    "normalized_magnitude_error",
    "normalized_magnitude_absolute_error",
    "predicted_magnitude_m",
    "target_magnitude_m",
    "magnitude_error_m",
    "magnitude_absolute_error_m",
    "prediction_reachable",
    "target_reachable",
)


def radial_project_unit_disk(
    predictions_normalized: ArrayLike,
    *,
    radius: float = 1.0,
    activation_tolerance: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Project two-dimensional predictions radially onto a closed disk.

    Points whose norm is at most ``radius + activation_tolerance`` are
    returned exactly unchanged.  The tolerance prevents float32 round-off at
    the physical reachability boundary from being reported as a meaningful
    projection.  The mapping contains the projected predictions, the original
    norms, the projection scale, and an activation mask.
    """

    predictions = _as_xy("predictions_normalized", predictions_normalized)
    radius_value = _positive_scalar("radius", radius)
    tolerance_value = _nonnegative_scalar(
        "activation_tolerance", activation_tolerance
    )
    norms = np.linalg.norm(predictions, axis=1)
    activated = norms > radius_value + tolerance_value
    scale = np.ones_like(norms)
    np.divide(
        radius_value,
        norms,
        out=scale,
        where=activated,
    )
    projected = predictions * scale[:, None]
    return {
        "predictions_normalized": projected,
        "original_norm": norms,
        "projection_scale": scale,
        "projection_activated": activated,
    }


def compute_spatial_waypoint_metrics(
    predictions_normalized: ArrayLike,
    targets_normalized: ArrayLike,
    *,
    horizon_steps: ArrayLike,
    maximum_step_distance_m: ArrayLike,
    current_positions_m: Optional[ArrayLike] = None,
    target_deltas_m: Optional[ArrayLike] = None,
    target_endpoints_m: Optional[ArrayLike] = None,
    sample_mask: Optional[ArrayLike] = None,
    direction_near_zero_threshold: float = 1.0e-8,
    reachability_radius: float = 1.0,
    reachability_tolerance: float = 1.0e-9,
    physical_consistency_atol_m: float = 1.0e-5,
) -> Dict[str, np.ndarray]:
    """Compute finite per-sample metrics for normalized 2-D displacements.

    ``horizon_steps`` and ``maximum_step_distance_m`` may be scalars or
    length-``N`` vectors.  ``current_positions_m`` is optional because the
    endpoint error equals the raw displacement error when both endpoints use
    the same current position.  Supplying it additionally returns predicted
    and target endpoints for auditing.  ``target_deltas_m`` and
    ``target_endpoints_m`` may provide the authoritative raw trajectory values;
    their consistency with the normalized label is checked rather than using
    one representation to silently regenerate all evaluation targets.

    Direction is undefined if either vector has norm at or below
    ``direction_near_zero_threshold``.  Such rows receive finite zero
    placeholders and ``direction_valid=False``; grouping functions exclude
    them from cosine and angular summaries and report a defined rate.
    """

    predictions = _as_xy("predictions_normalized", predictions_normalized)
    targets = _as_xy("targets_normalized", targets_normalized)
    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions_normalized and targets_normalized must have the same "
            f"shape, got {predictions.shape} and {targets.shape}"
        )
    count = predictions.shape[0]
    valid = _as_sample_mask(sample_mask, count)
    horizons = _positive_vector("horizon_steps", horizon_steps, count)
    if not np.all(np.equal(horizons, np.floor(horizons))):
        raise ValueError("horizon_steps must contain positive integer values")
    maximum_steps = _positive_vector(
        "maximum_step_distance_m", maximum_step_distance_m, count
    )
    normalization_distance = horizons * maximum_steps

    near_zero = _nonnegative_scalar(
        "direction_near_zero_threshold", direction_near_zero_threshold
    )
    radius = _positive_scalar("reachability_radius", reachability_radius)
    tolerance = _nonnegative_scalar(
        "reachability_tolerance", reachability_tolerance
    )
    consistency_atol = _nonnegative_scalar(
        "physical_consistency_atol_m", physical_consistency_atol_m
    )

    normalized_error = predictions - targets
    normalized_squared_norm_error = np.einsum(
        "ij,ij->i", normalized_error, normalized_error
    )
    normalized_component_rmse = np.sqrt(
        np.mean(np.square(normalized_error), axis=1)
    )
    normalized_component_mae = np.mean(np.abs(normalized_error), axis=1)

    raw_predictions = predictions * normalization_distance[:, None]
    reconstructed_raw_targets = targets * normalization_distance[:, None]
    if target_deltas_m is None:
        raw_targets = reconstructed_raw_targets
        raw_targets_from_input = np.zeros(count, dtype=bool)
    else:
        raw_targets = _as_xy("target_deltas_m", target_deltas_m)
        if raw_targets.shape != predictions.shape:
            raise ValueError(
                "target_deltas_m must match prediction shape, got "
                f"{raw_targets.shape} and {predictions.shape}"
            )
        if not np.allclose(
            raw_targets[valid],
            reconstructed_raw_targets[valid],
            rtol=1.0e-7,
            atol=consistency_atol,
        ):
            maximum_error = float(
                np.max(
                    np.abs(
                        raw_targets[valid] - reconstructed_raw_targets[valid]
                    )
                )
            )
            raise ValueError(
                "target_deltas_m is inconsistent with targets_normalized and "
                f"H*dmax (max error {maximum_error:.9g} m)"
            )
        raw_targets_from_input = np.ones(count, dtype=bool)
    raw_error = raw_predictions - raw_targets
    raw_l2_error = np.linalg.norm(raw_error, axis=1)
    raw_component_rmse = np.sqrt(np.mean(np.square(raw_error), axis=1))
    raw_component_mae = np.mean(np.abs(raw_error), axis=1)

    if current_positions_m is None:
        if target_endpoints_m is not None:
            raise ValueError(
                "target_endpoints_m requires current_positions_m for alignment"
            )
        current_positions = np.zeros_like(raw_predictions)
        endpoints_available = np.zeros(count, dtype=bool)
    else:
        current_positions = _as_xy("current_positions_m", current_positions_m)
        if current_positions.shape != predictions.shape:
            raise ValueError(
                "current_positions_m must match prediction shape, got "
                f"{current_positions.shape} and {predictions.shape}"
            )
        endpoints_available = np.ones(count, dtype=bool)
    predicted_endpoints = current_positions + raw_predictions
    reconstructed_target_endpoints = current_positions + raw_targets
    if target_endpoints_m is None:
        target_endpoints = reconstructed_target_endpoints
        target_endpoints_from_input = np.zeros(count, dtype=bool)
    else:
        target_endpoints = _as_xy("target_endpoints_m", target_endpoints_m)
        if target_endpoints.shape != predictions.shape:
            raise ValueError(
                "target_endpoints_m must match prediction shape, got "
                f"{target_endpoints.shape} and {predictions.shape}"
            )
        if not np.allclose(
            target_endpoints[valid],
            reconstructed_target_endpoints[valid],
            rtol=1.0e-7,
            atol=consistency_atol,
        ):
            maximum_error = float(
                np.max(
                    np.abs(
                        target_endpoints[valid]
                        - reconstructed_target_endpoints[valid]
                    )
                )
            )
            raise ValueError(
                "target_endpoints_m is inconsistent with current_positions_m "
                f"and target_deltas_m (max error {maximum_error:.9g} m)"
            )
        target_endpoints_from_input = np.ones(count, dtype=bool)
    endpoint_error = predicted_endpoints - target_endpoints
    endpoint_l2_error = np.linalg.norm(endpoint_error, axis=1)
    endpoint_component_rmse = np.sqrt(
        np.mean(np.square(endpoint_error), axis=1)
    )
    endpoint_component_mae = np.mean(np.abs(endpoint_error), axis=1)

    predicted_magnitude_normalized = np.linalg.norm(predictions, axis=1)
    target_magnitude_normalized = np.linalg.norm(targets, axis=1)
    magnitude_error_normalized = (
        predicted_magnitude_normalized - target_magnitude_normalized
    )
    predicted_magnitude_m = np.linalg.norm(raw_predictions, axis=1)
    target_magnitude_m = np.linalg.norm(raw_targets, axis=1)
    magnitude_error_m = predicted_magnitude_m - target_magnitude_m

    direction_valid = np.logical_and(
        predicted_magnitude_normalized > near_zero,
        target_magnitude_normalized > near_zero,
    )
    direction_valid = np.logical_and(direction_valid, valid)
    cosine = np.zeros(count, dtype=np.float64)
    denominator = predicted_magnitude_normalized * target_magnitude_normalized
    dot_product = np.einsum("ij,ij->i", predictions, targets)
    np.divide(dot_product, denominator, out=cosine, where=direction_valid)
    cosine[direction_valid] = np.clip(cosine[direction_valid], -1.0, 1.0)
    cosine_error = np.zeros(count, dtype=np.float64)
    cosine_error[direction_valid] = 1.0 - cosine[direction_valid]
    angular_error_rad = np.zeros(count, dtype=np.float64)
    angular_error_rad[direction_valid] = np.arccos(cosine[direction_valid])
    angular_error_deg = np.degrees(angular_error_rad)

    prediction_reachable = predicted_magnitude_normalized <= radius + tolerance
    target_reachable = target_magnitude_normalized <= radius + tolerance

    return {
        "sample_valid": valid,
        "direction_valid": direction_valid,
        "endpoints_available": endpoints_available,
        "raw_targets_from_input": raw_targets_from_input,
        "target_endpoints_from_input": target_endpoints_from_input,
        "horizon_steps": horizons.astype(np.int64),
        "maximum_step_distance_m": maximum_steps,
        "normalization_distance_m": normalization_distance,
        "predictions_normalized": predictions.copy(),
        "targets_normalized": targets.copy(),
        "normalized_error": normalized_error,
        "normalized_squared_norm_error": normalized_squared_norm_error,
        "normalized_component_rmse": normalized_component_rmse,
        "normalized_component_mae": normalized_component_mae,
        "predicted_delta_m": raw_predictions,
        "target_delta_m": raw_targets,
        "raw_delta_error_m": raw_error,
        "raw_delta_l2_error_m": raw_l2_error,
        "raw_delta_component_rmse_m": raw_component_rmse,
        "raw_delta_component_mae_m": raw_component_mae,
        "predicted_endpoint_m": predicted_endpoints,
        "target_endpoint_m": target_endpoints,
        "endpoint_error_m": endpoint_error,
        "endpoint_l2_error_m": endpoint_l2_error,
        "endpoint_component_rmse_m": endpoint_component_rmse,
        "endpoint_component_mae_m": endpoint_component_mae,
        "direction_cosine_similarity": cosine,
        "direction_cosine_error": cosine_error,
        "angular_error_rad": angular_error_rad,
        "angular_error_deg": angular_error_deg,
        "predicted_magnitude_normalized": predicted_magnitude_normalized,
        "target_magnitude_normalized": target_magnitude_normalized,
        "normalized_magnitude_error": magnitude_error_normalized,
        "normalized_magnitude_absolute_error": np.abs(
            magnitude_error_normalized
        ),
        "predicted_magnitude_m": predicted_magnitude_m,
        "target_magnitude_m": target_magnitude_m,
        "magnitude_error_m": magnitude_error_m,
        "magnitude_absolute_error_m": np.abs(magnitude_error_m),
        "prediction_reachable": prediction_reachable,
        "target_reachable": target_reachable,
    }


def summarize_spatial_waypoint_metrics(
    metrics: Mapping[str, np.ndarray],
    *,
    metadata: Optional[Metadata] = None,
    group_fields: Optional[
        Sequence[Union[str, Sequence[str]]]
    ] = None,
) -> Sequence[Dict[str, Any]]:
    """Summarize overall and requested metadata groups deterministically.

    By default the function creates one-dimensional groups for every present
    field in :data:`DEFAULT_GROUP_FIELDS`.  Pass strings for one-dimensional
    groups or tuples such as ``("policy_id", "num_buoys")`` for joint groups.
    Every distribution reports population ``mean/std/median/q25/q75``.
    """

    sample_valid = np.asarray(metrics.get("sample_valid"), dtype=bool)
    if sample_valid.ndim != 1:
        raise ValueError("metrics['sample_valid'] must be one-dimensional")
    count = sample_valid.size
    metadata_arrays = _metadata_arrays(metadata, count)
    specs = _normalize_group_specs(group_fields, metadata_arrays)

    rows = [_summarize_group(metrics, sample_valid, (), ())]
    for fields in specs:
        arrays = [metadata_arrays[field] for field in fields]
        keys = sorted(
            set(zip(*(array.tolist() for array in arrays))),
            key=lambda item: tuple(str(value) for value in item),
        )
        for key in keys:
            group_mask = sample_valid.copy()
            for array, value in zip(arrays, key):
                group_mask &= _metadata_equal(array, value)
            rows.append(_summarize_group(metrics, group_mask, fields, key))
    return rows


def evaluate_spatial_waypoint_predictions(
    predictions_normalized: ArrayLike,
    targets_normalized: ArrayLike,
    *,
    horizon_steps: ArrayLike,
    maximum_step_distance_m: ArrayLike,
    current_positions_m: Optional[ArrayLike] = None,
    target_deltas_m: Optional[ArrayLike] = None,
    target_endpoints_m: Optional[ArrayLike] = None,
    sample_mask: Optional[ArrayLike] = None,
    metadata: Optional[Metadata] = None,
    group_fields: Optional[
        Sequence[Union[str, Sequence[str]]]
    ] = None,
    apply_radial_projection: bool = True,
    direction_near_zero_threshold: float = 1.0e-8,
    reachability_radius: float = 1.0,
    reachability_tolerance: float = 1.0e-9,
    physical_consistency_atol_m: float = 1.0e-5,
) -> Dict[str, Any]:
    """Evaluate unprojected and optionally radially projected predictions.

    Projection changes are defined as ``projected - unprojected``; therefore
    a negative error change denotes an improvement.  The function has no file
    I/O and does not fit anything from the evaluated targets.
    """

    common = {
        "horizon_steps": horizon_steps,
        "maximum_step_distance_m": maximum_step_distance_m,
        "current_positions_m": current_positions_m,
        "target_deltas_m": target_deltas_m,
        "target_endpoints_m": target_endpoints_m,
        "sample_mask": sample_mask,
        "direction_near_zero_threshold": direction_near_zero_threshold,
        "reachability_radius": reachability_radius,
        "reachability_tolerance": reachability_tolerance,
        "physical_consistency_atol_m": physical_consistency_atol_m,
    }
    unprojected = compute_spatial_waypoint_metrics(
        predictions_normalized,
        targets_normalized,
        **common,
    )
    result: Dict[str, Any] = {
        "unprojected": {
            "per_sample": unprojected,
            "groups": summarize_spatial_waypoint_metrics(
                unprojected,
                metadata=metadata,
                group_fields=group_fields,
            ),
        }
    }
    if not apply_radial_projection:
        return result

    projection = radial_project_unit_disk(
        predictions_normalized,
        radius=reachability_radius,
        activation_tolerance=reachability_tolerance,
    )
    projected = compute_spatial_waypoint_metrics(
        projection["predictions_normalized"],
        targets_normalized,
        **common,
    )
    change = _projection_change_metrics(unprojected, projected, projection)
    result["projected"] = {
        "per_sample": projected,
        "groups": summarize_spatial_waypoint_metrics(
            projected,
            metadata=metadata,
            group_fields=group_fields,
        ),
    }
    result["projection"] = {
        "per_sample": change,
        "groups": summarize_projection_changes(
            change,
            metadata=metadata,
            group_fields=group_fields,
        ),
    }
    return result


def summarize_projection_changes(
    changes: Mapping[str, np.ndarray],
    *,
    metadata: Optional[Metadata] = None,
    group_fields: Optional[
        Sequence[Union[str, Sequence[str]]]
    ] = None,
) -> Sequence[Dict[str, Any]]:
    """Summarize radial-projection activation and error deltas by group."""

    sample_valid = np.asarray(changes.get("sample_valid"), dtype=bool)
    if sample_valid.ndim != 1:
        raise ValueError("changes['sample_valid'] must be one-dimensional")
    count = sample_valid.size
    metadata_arrays = _metadata_arrays(metadata, count)
    specs = _normalize_group_specs(group_fields, metadata_arrays)

    rows = [_summarize_projection_group(changes, sample_valid, (), ())]
    for fields in specs:
        arrays = [metadata_arrays[field] for field in fields]
        keys = sorted(
            set(zip(*(array.tolist() for array in arrays))),
            key=lambda item: tuple(str(value) for value in item),
        )
        for key in keys:
            group_mask = sample_valid.copy()
            for array, value in zip(arrays, key):
                group_mask &= _metadata_equal(array, value)
            rows.append(
                _summarize_projection_group(changes, group_mask, fields, key)
            )
    return rows


def zero_baseline(num_samples: int) -> np.ndarray:
    """Return a target-independent zero-displacement baseline."""

    count = _nonnegative_integer("num_samples", num_samples)
    return np.zeros((count, 2), dtype=np.float64)


def fit_train_mean_baseline(
    train_targets_normalized: ArrayLike,
    *,
    train_metadata: Metadata,
    horizon_steps: Optional[ArrayLike] = None,
    sample_mask: Optional[ArrayLike] = None,
    split_key: str = "split",
) -> Dict[str, Any]:
    """Fit a global or per-horizon mean using *only* declared train rows.

    The split field is mandatory and every row, including masked rows, must be
    labelled exactly ``"train"`` (case-insensitive).  Passing validation or
    test targets raises ``ValueError``.  When ``horizon_steps`` is supplied,
    independent means are stored for each horizon.
    """

    targets = _as_xy("train_targets_normalized", train_targets_normalized)
    count = targets.shape[0]
    metadata_arrays = _metadata_arrays(train_metadata, count)
    if split_key not in metadata_arrays:
        raise KeyError(
            f"train_metadata must contain the mandatory {split_key!r} field"
        )
    split_values = np.asarray(
        [str(value).strip().lower() for value in metadata_arrays[split_key]],
        dtype=object,
    )
    if np.any(split_values != "train"):
        offending = sorted(set(split_values[split_values != "train"].tolist()))
        raise ValueError(
            "train-mean fitting accepts train rows only; found split values "
            f"{offending}"
        )
    valid = _as_sample_mask(sample_mask, count)
    if not np.any(valid):
        raise ValueError("train-mean baseline requires at least one valid row")

    if horizon_steps is None:
        mean = np.mean(targets[valid], axis=0)
        return {
            "baseline_name": "train_mean",
            "fit_split": "train",
            "fit_count": int(np.count_nonzero(valid)),
            "per_horizon": False,
            "mean_delta_normalized": mean,
        }

    horizons = _positive_vector("horizon_steps", horizon_steps, count)
    if not np.all(np.equal(horizons, np.floor(horizons))):
        raise ValueError("horizon_steps must contain positive integer values")
    horizon_values = np.unique(horizons[valid].astype(np.int64))
    means = np.empty((horizon_values.size, 2), dtype=np.float64)
    counts = np.empty(horizon_values.size, dtype=np.int64)
    for index, horizon in enumerate(horizon_values):
        selected = valid & (horizons == horizon)
        counts[index] = np.count_nonzero(selected)
        means[index] = np.mean(targets[selected], axis=0)
    return {
        "baseline_name": "train_mean",
        "fit_split": "train",
        "fit_count": int(np.count_nonzero(valid)),
        "per_horizon": True,
        "horizon_steps": horizon_values,
        "mean_delta_normalized": means,
        "fit_count_by_horizon": counts,
    }


def predict_train_mean_baseline(
    baseline: Mapping[str, Any],
    *,
    num_samples: Optional[int] = None,
    horizon_steps: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Predict from a mapping returned by :func:`fit_train_mean_baseline`."""

    if str(baseline.get("fit_split", "")).lower() != "train":
        raise ValueError("baseline was not fitted from a declared train split")
    per_horizon = bool(baseline.get("per_horizon", False))
    means = np.asarray(baseline.get("mean_delta_normalized"), dtype=np.float64)
    if not np.all(np.isfinite(means)):
        raise ValueError("baseline mean contains NaN or Inf")

    if not per_horizon:
        if means.shape != (2,):
            raise ValueError("global train mean must have shape (2,)")
        if num_samples is None:
            if horizon_steps is None:
                raise ValueError("num_samples or horizon_steps is required")
            count = _vector_length("horizon_steps", horizon_steps)
        else:
            count = _nonnegative_integer("num_samples", num_samples)
            if horizon_steps is not None:
                horizon_count = _vector_length("horizon_steps", horizon_steps)
                if horizon_count != count:
                    raise ValueError("num_samples and horizon_steps disagree")
        return np.broadcast_to(means, (count, 2)).copy()

    if horizon_steps is None:
        raise ValueError("per-horizon train mean requires horizon_steps")
    requested = _one_dimensional("horizon_steps", horizon_steps)
    if requested.size == 1 and num_samples is not None:
        requested = np.full(
            _nonnegative_integer("num_samples", num_samples),
            requested.item(),
            dtype=np.float64,
        )
    elif num_samples is not None and requested.size != int(num_samples):
        raise ValueError("num_samples and horizon_steps disagree")
    if not np.all(np.isfinite(requested)) or np.any(requested <= 0.0):
        raise ValueError("horizon_steps must be finite and positive")
    if not np.all(np.equal(requested, np.floor(requested))):
        raise ValueError("horizon_steps must contain integer values")

    fitted_horizons = np.asarray(baseline.get("horizon_steps"), dtype=np.int64)
    if fitted_horizons.ndim != 1 or means.shape != (fitted_horizons.size, 2):
        raise ValueError("invalid per-horizon baseline mapping")
    lookup = {int(horizon): means[index] for index, horizon in enumerate(fitted_horizons)}
    missing = sorted(set(requested.astype(np.int64).tolist()) - set(lookup))
    if missing:
        raise ValueError(f"train mean is unavailable for horizons {missing}")
    return np.stack(
        [lookup[int(horizon)] for horizon in requested],
        axis=0,
    )


def constant_velocity_baseline(
    current_positions_m: ArrayLike,
    current_velocities_mps: ArrayLike,
    *,
    horizon_steps: ArrayLike,
    delta_t_s: ArrayLike,
    maximum_step_distance_m: ArrayLike,
    area_lower_bound_m: ArrayLike,
    area_upper_bound_m: ArrayLike,
) -> Dict[str, np.ndarray]:
    """Predict displacement using current velocity and rectangular clipping.

    Only the current measured velocity is used: no future action, position,
    target, terminal flag, or trajectory length enters the calculation.
    Clipping is applied to the predicted physical endpoint before converting
    the displacement back to the reachability-normalized representation.
    """

    positions = _as_xy("current_positions_m", current_positions_m)
    velocities = _as_xy("current_velocities_mps", current_velocities_mps)
    if positions.shape != velocities.shape:
        raise ValueError(
            "current_positions_m and current_velocities_mps must have the same "
            f"shape, got {positions.shape} and {velocities.shape}"
        )
    count = positions.shape[0]
    horizons = _positive_vector("horizon_steps", horizon_steps, count)
    if not np.all(np.equal(horizons, np.floor(horizons))):
        raise ValueError("horizon_steps must contain positive integer values")
    delta_t = _positive_vector("delta_t_s", delta_t_s, count)
    maximum_steps = _positive_vector(
        "maximum_step_distance_m", maximum_step_distance_m, count
    )
    lower = _xy_bound("area_lower_bound_m", area_lower_bound_m)
    upper = _xy_bound("area_upper_bound_m", area_upper_bound_m)
    if np.any(upper <= lower):
        raise ValueError("area_upper_bound_m must exceed area_lower_bound_m")
    if np.any(positions < lower) or np.any(positions > upper):
        raise ValueError("current_positions_m contains positions outside the area")

    elapsed = horizons * delta_t
    endpoint_unclipped = positions + velocities * elapsed[:, None]
    endpoint_clipped = np.clip(endpoint_unclipped, lower, upper)
    clipping_activated = np.any(endpoint_clipped != endpoint_unclipped, axis=1)
    displacement_m = endpoint_clipped - positions
    normalization_distance = horizons * maximum_steps
    predictions_normalized = displacement_m / normalization_distance[:, None]
    current_speed = np.linalg.norm(velocities, axis=1)
    speed_limit = maximum_steps / delta_t
    speed_limit_violation = current_speed > speed_limit + 1.0e-9
    return {
        "predictions_normalized": predictions_normalized,
        "predicted_delta_m": displacement_m,
        "predicted_endpoint_unclipped_m": endpoint_unclipped,
        "predicted_endpoint_m": endpoint_clipped,
        "boundary_clipping_activated": clipping_activated,
        "current_speed_mps": current_speed,
        "speed_limit_mps": speed_limit,
        "speed_limit_violation": speed_limit_violation,
        "horizon_steps": horizons.astype(np.int64),
        "normalization_distance_m": normalization_distance,
    }


def assign_episode_length_bands(
    episode_lengths: ArrayLike,
    *,
    edges: ArrayLike,
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Assign deterministic bands using caller-supplied, pre-registered edges.

    Edges are never inferred from the evaluated values, avoiding accidental
    test-set-dependent bin design.  With edges ``[a, b]`` the intervals are
    ``(-inf, a)``, ``[a, b)``, and ``[b, inf)``.
    """

    lengths = _one_dimensional("episode_lengths", episode_lengths)
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 1):
        raise ValueError("episode_lengths must contain finite positive values")
    edge_values = _one_dimensional("edges", edges).astype(np.float64)
    if edge_values.size and (
        not np.all(np.isfinite(edge_values))
        or np.any(np.diff(edge_values) <= 0.0)
    ):
        raise ValueError("edges must be finite and strictly increasing")
    band_count = edge_values.size + 1
    if labels is None:
        label_values = _default_band_labels(edge_values)
    else:
        if len(labels) != band_count:
            raise ValueError(
                f"labels must contain {band_count} entries for these edges"
            )
        label_values = [str(label) for label in labels]
        if len(set(label_values)) != len(label_values):
            raise ValueError("labels must be unique")
    indices = np.searchsorted(edge_values, lengths, side="right")
    return np.asarray([label_values[index] for index in indices], dtype=str)


def _projection_change_metrics(
    unprojected: Mapping[str, np.ndarray],
    projected: Mapping[str, np.ndarray],
    projection: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        "sample_valid": np.asarray(unprojected["sample_valid"], dtype=bool),
        "projection_activated": np.asarray(
            projection["projection_activated"], dtype=bool
        ),
        "projection_scale": np.asarray(projection["projection_scale"], dtype=np.float64),
        "normalized_squared_norm_error_change": (
            np.asarray(projected["normalized_squared_norm_error"])
            - np.asarray(unprojected["normalized_squared_norm_error"])
        ),
        "raw_delta_l2_error_change_m": (
            np.asarray(projected["raw_delta_l2_error_m"])
            - np.asarray(unprojected["raw_delta_l2_error_m"])
        ),
        "endpoint_l2_error_change_m": (
            np.asarray(projected["endpoint_l2_error_m"])
            - np.asarray(unprojected["endpoint_l2_error_m"])
        ),
        "magnitude_absolute_error_change_m": (
            np.asarray(projected["magnitude_absolute_error_m"])
            - np.asarray(unprojected["magnitude_absolute_error_m"])
        ),
        "reachability_change": (
            np.asarray(projected["prediction_reachable"], dtype=np.float64)
            - np.asarray(unprojected["prediction_reachable"], dtype=np.float64)
        ),
    }


def _summarize_group(
    metrics: Mapping[str, np.ndarray],
    mask: np.ndarray,
    fields: Tuple[str, ...],
    values: Tuple[Any, ...],
) -> Dict[str, Any]:
    row = _group_identity(fields, values)
    row["sample_count"] = int(np.count_nonzero(mask))
    direction_valid = np.asarray(metrics["direction_valid"], dtype=bool) & mask
    row["direction_defined_count"] = int(np.count_nonzero(direction_valid))
    row["direction_defined_rate"] = _safe_rate(direction_valid, mask)

    for name in _DISTRIBUTION_METRICS:
        array = np.asarray(metrics[name])
        if array.shape != mask.shape:
            raise ValueError(f"metrics[{name!r}] must have shape {mask.shape}")
        metric_mask = mask
        if name in {
            "direction_cosine_similarity",
            "direction_cosine_error",
            "angular_error_rad",
            "angular_error_deg",
        }:
            metric_mask = direction_valid
        values_array = array[metric_mask].astype(np.float64)
        for statistic, value in _distribution_summary(values_array).items():
            row[f"{name}_{statistic}"] = value

    normalized_squared = np.asarray(
        metrics["normalized_squared_norm_error"], dtype=np.float64
    )[mask]
    raw_l2 = np.asarray(metrics["raw_delta_l2_error_m"], dtype=np.float64)[mask]
    endpoint_l2 = np.asarray(metrics["endpoint_l2_error_m"], dtype=np.float64)[mask]
    raw_component_rmse = np.asarray(
        metrics["raw_delta_component_rmse_m"], dtype=np.float64
    )[mask]
    raw_component_mae = np.asarray(
        metrics["raw_delta_component_mae_m"], dtype=np.float64
    )[mask]
    endpoint_component_rmse = np.asarray(
        metrics["endpoint_component_rmse_m"], dtype=np.float64
    )[mask]
    endpoint_component_mae = np.asarray(
        metrics["endpoint_component_mae_m"], dtype=np.float64
    )[mask]
    prediction_reachable = np.asarray(
        metrics["prediction_reachable"], dtype=bool
    )[mask]
    target_reachable = np.asarray(metrics["target_reachable"], dtype=bool)[mask]

    row["normalized_squared_norm_mse"] = _mean_or_none(normalized_squared)
    row["raw_delta_rmse_m"] = _root_mean_square(raw_l2)
    row["raw_delta_mae_m"] = _mean_or_none(raw_l2)
    row["endpoint_rmse_m"] = _root_mean_square(endpoint_l2)
    row["endpoint_mae_m"] = _mean_or_none(endpoint_l2)
    row["raw_component_rmse_m"] = _root_mean_square(raw_component_rmse)
    row["raw_component_mae_m"] = _mean_or_none(raw_component_mae)
    row["endpoint_component_rmse_m"] = _root_mean_square(
        endpoint_component_rmse
    )
    row["endpoint_component_mae_m"] = _mean_or_none(endpoint_component_mae)
    row["reachable_rate"] = (
        float(np.mean(prediction_reachable)) if prediction_reachable.size else None
    )
    row["target_reachable_rate"] = (
        float(np.mean(target_reachable)) if target_reachable.size else None
    )
    return row


def _summarize_projection_group(
    changes: Mapping[str, np.ndarray],
    mask: np.ndarray,
    fields: Tuple[str, ...],
    values: Tuple[Any, ...],
) -> Dict[str, Any]:
    row = _group_identity(fields, values)
    row["sample_count"] = int(np.count_nonzero(mask))
    activated = np.asarray(changes["projection_activated"], dtype=bool)
    if activated.shape != mask.shape:
        raise ValueError("projection_activated shape does not match sample_valid")
    row["projection_activation_count"] = int(np.count_nonzero(activated & mask))
    row["projection_activation_rate"] = _safe_rate(activated & mask, mask)
    for name in (
        "projection_scale",
        "normalized_squared_norm_error_change",
        "raw_delta_l2_error_change_m",
        "endpoint_l2_error_change_m",
        "magnitude_absolute_error_change_m",
        "reachability_change",
    ):
        array = np.asarray(changes[name], dtype=np.float64)
        if array.shape != mask.shape:
            raise ValueError(f"changes[{name!r}] must have shape {mask.shape}")
        for statistic, value in _distribution_summary(array[mask]).items():
            row[f"{name}_{statistic}"] = value
    return row


def _group_identity(
    fields: Tuple[str, ...], values: Tuple[Any, ...]
) -> Dict[str, Any]:
    if not fields:
        return {
            "group_kind": "overall",
            "group_fields": (),
            "group_value": "overall",
        }
    python_values = tuple(_python_scalar(value) for value in values)
    row: Dict[str, Any] = {
        "group_kind": "+".join(fields),
        "group_fields": fields,
        "group_value": python_values[0] if len(fields) == 1 else python_values,
    }
    for field, value in zip(fields, python_values):
        row[field] = value
    return row


def _normalize_group_specs(
    group_fields: Optional[Sequence[Union[str, Sequence[str]]]],
    metadata: Mapping[str, np.ndarray],
) -> Tuple[Tuple[str, ...], ...]:
    if group_fields is None:
        return tuple((field,) for field in DEFAULT_GROUP_FIELDS if field in metadata)
    specs = []
    for spec in group_fields:
        if isinstance(spec, str):
            fields = (spec,)
        else:
            fields = tuple(str(field) for field in spec)
        if not fields:
            raise ValueError("group specifications may not be empty")
        if len(set(fields)) != len(fields):
            raise ValueError(f"duplicate fields in group specification {fields}")
        missing = [field for field in fields if field not in metadata]
        if missing:
            raise KeyError(f"metadata is missing requested group fields {missing}")
        if fields not in specs:
            specs.append(fields)
    return tuple(specs)


def _metadata_arrays(
    metadata: Optional[Metadata], count: int
) -> Dict[str, np.ndarray]:
    if metadata is None:
        return {}
    arrays: Dict[str, np.ndarray] = {}
    for name, values in metadata.items():
        array = np.asarray(values)
        if array.ndim == 0:
            array = np.full(count, array.item(), dtype=array.dtype)
        # Evaluation metadata commonly also carries two-dimensional audit
        # arrays such as q_current_m.  They are valid inputs to the surrounding
        # pipeline but are not categorical grouping columns, so ignore them.
        if array.ndim != 1:
            continue
        if array.shape[0] != count:
            raise ValueError(
                f"metadata[{name!r}] must be scalar or shape ({count},), "
                f"got {array.shape}"
            )
        arrays[str(name)] = array
    if "quality_class" in arrays:
        if "quality_tier" in arrays and not np.array_equal(
            arrays["quality_tier"], arrays["quality_class"]
        ):
            raise ValueError(
                "metadata quality_tier and quality_class aliases disagree"
            )
        arrays.setdefault("quality_tier", arrays["quality_class"])
    return arrays


def _metadata_equal(array: np.ndarray, value: Any) -> np.ndarray:
    if isinstance(value, float) and np.isnan(value):
        return np.asarray(
            [isinstance(item, float) and np.isnan(item) for item in array],
            dtype=bool,
        )
    return np.asarray(array == value, dtype=bool)


def _distribution_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
    if values.ndim != 1:
        raise ValueError("distribution values must be one-dimensional")
    if values.size == 0:
        return {name: None for name in SUMMARY_STATISTICS}
    if not np.all(np.isfinite(values)):
        raise ValueError("metric distribution contains NaN or Inf")
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
    }


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _root_mean_square(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _safe_rate(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> Optional[float]:
    denominator = int(np.count_nonzero(denominator_mask))
    if denominator == 0:
        return None
    return float(np.count_nonzero(numerator_mask) / denominator)


def _as_xy(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _as_sample_mask(values: Optional[ArrayLike], count: int) -> np.ndarray:
    if values is None:
        return np.ones(count, dtype=bool)
    array = np.asarray(values)
    if array.shape == (count, 2):
        if not np.all(array[:, :1] == array[:, 1:]):
            raise ValueError(
                "a 2-D spatial sample mask must agree across x/y components"
            )
        array = array[:, 0]
    if array.shape != (count,):
        raise ValueError(
            f"sample_mask must have shape ({count},) or ({count}, 2), "
            f"got {array.shape}"
        )
    if array.dtype != np.bool_:
        if not np.all(np.isin(array, (0, 1))):
            raise ValueError("sample_mask must contain only boolean/0/1 values")
    return array.astype(bool)


def _positive_vector(name: str, values: ArrayLike, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(count, float(array), dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape ({count},), got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _one_dimensional(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def _vector_length(name: str, values: ArrayLike) -> int:
    return int(_one_dimensional(name, values).size)


def _xy_bound(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _positive_scalar(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _nonnegative_scalar(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    integer = int(value)
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _default_band_labels(edges: np.ndarray) -> Sequence[str]:
    if edges.size == 0:
        return ("all",)
    labels = [f"lt_{_format_edge(edges[0])}"]
    for lower, upper in zip(edges[:-1], edges[1:]):
        labels.append(f"{_format_edge(lower)}_to_lt_{_format_edge(upper)}")
    labels.append(f"ge_{_format_edge(edges[-1])}")
    return labels


def _format_edge(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(float(value), "g")


__all__ = [
    "DEFAULT_GROUP_FIELDS",
    "SUMMARY_STATISTICS",
    "assign_episode_length_bands",
    "compute_spatial_waypoint_metrics",
    "constant_velocity_baseline",
    "evaluate_spatial_waypoint_predictions",
    "fit_train_mean_baseline",
    "predict_train_mean_baseline",
    "radial_project_unit_disk",
    "summarize_projection_changes",
    "summarize_spatial_waypoint_metrics",
    "zero_baseline",
]
