"""Round-4 supervised spatial Waypoint training and evaluation CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

# Set before Torch/CUDA discovery so deterministic cuBLAS kernels are valid.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from .round4_data import (
    PREDICTOR_LEAKAGE_DENYLIST,
    load_round4_spatial_dataset,
)
from .round4_evaluation import (
    assign_episode_length_bands,
    constant_velocity_baseline,
    evaluate_spatial_waypoint_predictions,
    fit_train_mean_baseline,
    predict_train_mean_baseline,
    radial_project_unit_disk,
    zero_baseline,
)
from .round4_training import (
    SpatialTrainingConfig,
    load_spatial_waypoint_checkpoint,
    predict_spatial_waypoints,
    train_spatial_waypoint,
)


RUN_MANIFEST_VERSION = "round04_run_manifest_v1"
DEFAULT_CONFIG = "round_04/configs/spatial_waypoint_v1.json"
GROUP_FIELDS = (
    "policy_id",
    "num_buoys",
    "quality_tier",
    "episode_length",
    "episode_length_band",
    "goal_source",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _timestamp() -> str:
    return _now_local().strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_is_within(path: Path, root: Path) -> bool:
    """Python-3.8-compatible resolved-path containment check."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty required CSV: {path.name}")
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row: Dict[str, Any] = {}
            for key in columns:
                value = source.get(key, "")
                if isinstance(value, (list, tuple, dict, np.ndarray)):
                    value = json.dumps(
                        _json_value(value),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                elif isinstance(value, np.generic):
                    value = value.item()
                row[key] = value
            writer.writerow(row)


def _resolve_config(config_path: str | Path) -> tuple[Path, Dict[str, Any]]:
    repo = _repository_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = _read_json(path)
    return path, config


def _training_config(
    config: Mapping[str, Any],
    *,
    device_override: str | None = None,
    smoke: bool = False,
) -> SpatialTrainingConfig:
    model = config["model"]
    training = config["training"]
    resolved = SpatialTrainingConfig(
        hidden_dim=int(model["hidden_dim"]),
        num_hidden_layers=int(model["num_hidden_layers"]),
        dropout=float(model["dropout"]),
        learning_rate=float(training["learning_rate"]),
        batch_size=int(training["batch_size"]),
        maximum_epochs=(
            min(3, int(training["maximum_epochs"]))
            if smoke
            else int(training["maximum_epochs"])
        ),
        early_stopping_patience=(
            min(2, int(training["early_stopping_patience"]))
            if smoke
            else int(training["early_stopping_patience"])
        ),
        weight_decay=float(training["weight_decay"]),
        gradient_clipping=float(training["gradient_clipping"]),
        device=(
            str(device_override)
            if device_override is not None
            else str(training["device"])
        ),
        deterministic=bool(training["deterministic"]),
        num_workers=int(training.get("num_workers", 0)),
    )
    return resolved.validate()


def _model_inputs(split: Any, model_kind: str) -> np.ndarray:
    if model_kind == "goal_conditioned":
        return np.asarray(split.inputs(goal_conditioned=True), dtype=np.float32)
    if model_kind == "state_only":
        return np.asarray(split.inputs(goal_conditioned=False), dtype=np.float32)
    raise ValueError(f"unknown model kind: {model_kind}")


def _metadata(
    split: Any,
    evaluation_config: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    metadata = {
        key: np.asarray(value)
        for key, value in split.evaluation_metadata().items()
    }
    if "quality_tier" not in metadata:
        if "quality_class" not in metadata:
            raise KeyError("evaluation metadata is missing quality tier")
        metadata["quality_tier"] = metadata["quality_class"]
    metadata.setdefault(
        "goal_source",
        np.full(
            len(split),
            str(split.goal_source),
            dtype=f"<U{max(1, len(str(split.goal_source)))}",
        ),
    )
    metadata.setdefault(
        "split",
        np.full(
            len(split),
            str(split.split_name),
            dtype=f"<U{max(1, len(str(split.split_name)))}",
        ),
    )
    metadata.setdefault(
        "episode_length_band",
        assign_episode_length_bands(
            metadata["episode_length"],
            edges=np.asarray(
                evaluation_config["episode_length_band_edges"],
                dtype=np.int64,
            ),
            labels=tuple(
                str(value)
                for value in evaluation_config[
                    "episode_length_band_labels"
                ]
            ),
        ),
    )
    for field in GROUP_FIELDS:
        if field not in metadata:
            raise KeyError(f"evaluation metadata is missing {field}")
    return metadata


def _canonical_metric_row(
    row: Mapping[str, Any],
    *,
    split: str,
    horizon: int,
    seed: int | str,
    model_kind: str,
    projection: str,
) -> Dict[str, Any]:
    normalized_mse = row.get("normalized_squared_norm_mse")
    normalized_endpoint = (
        math.sqrt(float(normalized_mse))
        if normalized_mse is not None
        else None
    )
    result = dict(row)
    result.update({
        "split": split,
        "horizon": int(horizon),
        "seed": seed,
        "model_kind": model_kind,
        "projection": projection,
        "group_dimension": row.get("group_kind", "overall"),
        "group_value": row.get("group_value", "overall"),
        "sample_count": row.get("sample_count"),
        "normalized_mse": normalized_mse,
        "normalized_endpoint_error": normalized_endpoint,
        "raw_displacement_rmse_m": row.get("raw_delta_rmse_m"),
        "raw_displacement_mae_m": row.get("raw_delta_mae_m"),
        "future_endpoint_rmse_m": row.get("endpoint_rmse_m"),
        "future_endpoint_mae_m": row.get("endpoint_mae_m"),
        "direction_cosine_similarity": row.get(
            "direction_cosine_similarity_mean"
        ),
        "direction_angular_error_deg": row.get("angular_error_deg_mean"),
        "direction_defined_rate": row.get("direction_defined_rate"),
        "displacement_magnitude_error_m": row.get(
            "magnitude_absolute_error_m_mean"
        ),
        "reachable_prediction_rate": row.get("reachable_rate"),
        "target_reachable_rate": row.get("target_reachable_rate"),
    })
    for field in GROUP_FIELDS:
        if field in row:
            result[field] = row[field]
    return result


def _evaluate(
    *,
    predictions: np.ndarray,
    split: Any,
    dataset: Any,
    model_kind: str,
    seed: int | str,
    evaluation_config: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    metadata = _metadata(split, evaluation_config)
    evaluation = evaluate_spatial_waypoint_predictions(
        predictions,
        split.normalized_delta_xy,
        horizon_steps=np.full(len(split), dataset.horizon, dtype=np.int64),
        maximum_step_distance_m=np.full(
            len(split), dataset.max_step_distance_m, dtype=np.float64
        ),
        current_positions_m=split.q_current_m,
        target_deltas_m=split.raw_delta_xy_m,
        target_endpoints_m=split.q_future_m,
        metadata=metadata,
        group_fields=GROUP_FIELDS,
        apply_radial_projection=True,
        direction_near_zero_threshold=float(
            evaluation_config["near_zero_displacement_m"]
        )
        / dataset.horizon_normalization_distance_m,
        reachability_tolerance=float(
            evaluation_config["reachability_tolerance_m"]
        )
        / dataset.horizon_normalization_distance_m,
    )
    metric_rows: List[Dict[str, Any]] = []
    for key, projection in (
        ("unprojected", "raw"),
        ("projected", "projected"),
    ):
        for row in evaluation[key]["groups"]:
            metric_rows.append(
                _canonical_metric_row(
                    row,
                    split=split.split_name,
                    horizon=dataset.horizon,
                    seed=seed,
                    model_kind=model_kind,
                    projection=projection,
                )
            )
    projection_rows: List[Dict[str, Any]] = []
    for row in evaluation["projection"]["groups"]:
        normalized = dict(row)
        normalized.update({
            "split": split.split_name,
            "horizon": int(dataset.horizon),
            "seed": seed,
            "model_kind": model_kind,
            "group_dimension": row.get("group_kind", "overall"),
            "group_value": row.get("group_value", "overall"),
            "sample_count": row.get("sample_count"),
            "projection_activation_count": row.get(
                "projection_activation_count"
            ),
            "projection_activation_rate": row.get(
                "projection_activation_rate"
            ),
            "normalized_mse_change_mean": row.get(
                "normalized_squared_norm_error_change_mean"
            ),
            "raw_error_change_m_mean": row.get(
                "raw_delta_l2_error_change_m_mean"
            ),
            "endpoint_error_change_m_mean": row.get(
                "endpoint_l2_error_change_m_mean"
            ),
            "magnitude_error_change_m_mean": row.get(
                "magnitude_absolute_error_change_m_mean"
            ),
            "reachability_change_mean": row.get("reachability_change_mean"),
        })
        for field in GROUP_FIELDS:
            if field in row:
                normalized[field] = row[field]
        projection_rows.append(normalized)
    return metric_rows, projection_rows, evaluation


def _overall_raw(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["projection"] == "raw"
        and row["group_dimension"] == "overall"
    ]
    if len(selected) != 1:
        raise RuntimeError("expected exactly one overall raw metric row")
    return dict(selected[0])


def _baseline_predictions(
    dataset: Any,
    split: Any,
    baseline_name: str,
    mean_baseline: Mapping[str, Any],
    environment_config: Mapping[str, Any],
) -> np.ndarray:
    if baseline_name == "zero_waypoint":
        return zero_baseline(len(split))
    if baseline_name == "train_mean_waypoint":
        return predict_train_mean_baseline(
            mean_baseline, num_samples=len(split)
        )
    if baseline_name == "constant_velocity":
        result = constant_velocity_baseline(
            split.q_current_m,
            split.current_velocity_mps,
            horizon_steps=np.full(len(split), dataset.horizon),
            delta_t_s=np.full(len(split), dataset.delta_t_seconds),
            maximum_step_distance_m=np.full(
                len(split), dataset.max_step_distance_m
            ),
            area_lower_bound_m=np.asarray([0.0, 0.0]),
            area_upper_bound_m=np.asarray(
                [
                    float(environment_config["area_width"]),
                    float(environment_config["area_height"]),
                ]
            ),
        )
        return np.asarray(result["predictions_normalized"], dtype=np.float64)
    raise ValueError(f"unknown baseline: {baseline_name}")


def _fit_mean(dataset: Any) -> Dict[str, Any]:
    return fit_train_mean_baseline(
        dataset.train.normalized_delta_xy,
        train_metadata={
            "split": np.full(len(dataset.train), "train", dtype="<U5")
        },
    )


def _selection_rows(
    validation_rows: Sequence[Mapping[str, Any]],
    maximum_step_by_horizon: Mapping[int, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for horizon in sorted(maximum_step_by_horizon):
        selected = [
            row
            for row in validation_rows
            if int(row["horizon"]) == int(horizon)
            and row["model_kind"] == "goal_conditioned"
            and row["projection"] == "raw"
            and row["group_dimension"] == "overall"
        ]
        if not selected:
            raise RuntimeError(f"missing goal-conditioned validation rows for H={horizon}")
        endpoint = np.asarray(
            [row["future_endpoint_rmse_m"] for row in selected], dtype=float
        )
        raw_displacement = np.asarray(
            [row["raw_displacement_rmse_m"] for row in selected], dtype=float
        )
        normalized = np.asarray(
            [row["normalized_endpoint_error"] for row in selected], dtype=float
        )
        reachable = np.asarray(
            [row["reachable_prediction_rate"] for row in selected], dtype=float
        )
        if not (
            np.all(np.isfinite(endpoint))
            and np.all(np.isfinite(raw_displacement))
            and np.all(np.isfinite(normalized))
            and np.all(np.isfinite(reachable))
        ):
            raise ValueError(
                f"validation selection metrics are non-finite for H={horizon}"
            )
        if not np.allclose(
            endpoint, raw_displacement, rtol=1.0e-9, atol=1.0e-9
        ):
            raise ValueError(
                "future endpoint RMSE and raw displacement RMSE must be "
                f"identical for relative targets, H={horizon}"
            )
        cosine_values = [
            float(row["direction_cosine_similarity"])
            for row in selected
            if row.get("direction_cosine_similarity") is not None
            and math.isfinite(float(row["direction_cosine_similarity"]))
        ]
        direction_available = bool(cosine_values)
        # Undefined direction for every sample is not allowed to win through a
        # NaN sort.  Use the worst finite cosine as an explicit penalty.
        cosine_mean = (
            float(np.mean(cosine_values)) if direction_available else -1.0
        )
        normalized_std = float(np.std(normalized))
        rows.append(
            {
                "horizon": int(horizon),
                "seed_count": len(selected),
                "validation_endpoint_rmse_m_mean": float(np.mean(endpoint)),
                "validation_endpoint_rmse_m_std": float(np.std(endpoint)),
                "validation_endpoint_rmse_m_median": float(
                    np.median(endpoint)
                ),
                "validation_endpoint_rmse_m_q25": float(
                    np.quantile(endpoint, 0.25)
                ),
                "validation_endpoint_rmse_m_q75": float(
                    np.quantile(endpoint, 0.75)
                ),
                "validation_raw_displacement_rmse_m_mean": float(
                    np.mean(raw_displacement)
                ),
                "validation_raw_displacement_rmse_m_std": float(
                    np.std(raw_displacement)
                ),
                "validation_normalized_endpoint_error_mean": float(
                    np.mean(normalized)
                ),
                "validation_normalized_endpoint_error_std": normalized_std,
                "validation_seed_stability_normalized_std": normalized_std,
                "validation_direction_cosine_mean": cosine_mean,
                "validation_direction_metric_available": direction_available,
                "validation_reachable_rate_mean": float(np.mean(reachable)),
                "test_metrics_used": False,
            }
        )
    endpoint_scale = max(
        row["validation_endpoint_rmse_m_mean"] for row in rows
    )
    if not math.isfinite(endpoint_scale) or endpoint_scale <= 0.0:
        endpoint_scale = 1.0
    for row in rows:
        endpoint_fraction = (
            row["validation_endpoint_rmse_m_mean"] / endpoint_scale
        )
        # Endpoint and raw-displacement RMSE are mathematically identical for
        # q[t+H]-q[t].  Record both, but include their shared dimensional signal
        # only once so short horizons cannot win through duplicated weighting.
        row["validation_endpoint_rmse_candidate_scale_fraction"] = (
            endpoint_fraction
        )
        row["selection_score"] = (
            0.55 * row["validation_normalized_endpoint_error_mean"]
            + 0.10 * endpoint_fraction
            + 0.15 * row["validation_seed_stability_normalized_std"]
            + 0.10 * (1.0 - row["validation_direction_cosine_mean"])
            + 0.10 * (1.0 - row["validation_reachable_rate_mean"])
        )
        row["selection_score_definition"] = (
            "0.55*mean normalized endpoint error + "
            "0.10*validation endpoint/raw-displacement RMSE divided by "
            "largest candidate validation endpoint RMSE + "
            "0.15*seed std(normalized endpoint error) + "
            "0.10*(1-cosine) + 0.10*(1-reachable)"
        )
    ordered = sorted(rows, key=lambda row: (row["selection_score"], row["horizon"]))
    for rank, row in enumerate(ordered, start=1):
        row["validation_rank"] = rank
        row["recommended"] = rank == 1
    return sorted(ordered, key=lambda row: row["horizon"])


def _prediction_examples(
    *,
    split: Any,
    dataset: Any,
    predictions: np.ndarray,
    seed: int,
    model_kind: str,
    limit: int,
    projection_tolerance: float = 0.0,
) -> List[Dict[str, Any]]:
    projection = radial_project_unit_disk(
        predictions,
        activation_tolerance=projection_tolerance,
    )
    projected = projection["predictions_normalized"]
    if limit <= 0:
        return []
    indices = np.unique(
        np.linspace(0, len(split) - 1, min(limit, len(split)), dtype=int)
    )
    scale = dataset.horizon_normalization_distance_m
    rows: List[Dict[str, Any]] = []
    for index in indices:
        true_delta = np.asarray(split.normalized_delta_xy[index]) * scale
        pred_delta = np.asarray(predictions[index]) * scale
        projected_delta = np.asarray(projected[index]) * scale
        q_current = np.asarray(split.q_current_m[index])
        q_true = q_current + true_delta
        q_pred = q_current + pred_delta
        q_projected = q_current + projected_delta
        rows.append(
            {
                "split": split.split_name,
                "horizon": dataset.horizon,
                "seed": seed,
                "model_kind": model_kind,
                "sample_id": str(split.sample_id[index]),
                "episode_id": str(split.episode_id[index]),
                "policy_id": str(split.policy_id[index]),
                "num_buoys": int(split.num_buoys[index]),
                "quality_tier": str(split.quality_class[index]),
                "goal_source": str(split.goal_source),
                "t": int(split.t[index]),
                "q_current_x_m": float(q_current[0]),
                "q_current_y_m": float(q_current[1]),
                "q_true_future_x_m": float(q_true[0]),
                "q_true_future_y_m": float(q_true[1]),
                "q_pred_future_x_m": float(q_pred[0]),
                "q_pred_future_y_m": float(q_pred[1]),
                "q_projected_future_x_m": float(q_projected[0]),
                "q_projected_future_y_m": float(q_projected[1]),
                "true_delta_x_m": float(true_delta[0]),
                "true_delta_y_m": float(true_delta[1]),
                "pred_delta_x_m": float(pred_delta[0]),
                "pred_delta_y_m": float(pred_delta[1]),
                "projected_delta_x_m": float(projected_delta[0]),
                "projected_delta_y_m": float(projected_delta[1]),
                "prediction_projected": bool(
                    projection["projection_activated"][index]
                ),
            }
        )
    return rows


def _source_hashes(source_dir: Path) -> Dict[str, str]:
    paths = sorted(
        path for path in source_dir.rglob("*") if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(
            f"locked Round-3 source contains no files: {source_dir}"
        )
    return {
        path.relative_to(source_dir).as_posix(): _sha256(path)
        for path in paths
    }


def _round4_implementation_hashes(
    repo: Path, config_path: Path | None = None
) -> Dict[str, str]:
    paths = sorted((repo / "src" / "wt" / "round4").glob("*.py"))
    config = config_path or (repo / DEFAULT_CONFIG)
    if config.is_file():
        paths.append(config)
    result: Dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        key = (
            path.relative_to(repo).as_posix()
            if _path_is_within(path.resolve(), repo.resolve())
            else f"external_config::{path.resolve()}"
        )
        result[key] = _sha256(path)
    return result


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _training_provenance(
    dataset: Any,
    config_path: Path,
) -> Dict[str, Any]:
    """Bind checkpoints to the exact data, goal protocol, and config."""

    tensor_hashes: Dict[str, str] = {}
    for split_name in ("train", "validation"):
        split = dataset.split(split_name)
        for field in (
            "masked_state_t",
            "masked_goal",
            "valid_node_mask_t",
            "normalized_delta_xy",
        ):
            tensor_hashes[f"{split_name}.{field}"] = _array_sha256(
                getattr(split, field)
            )
    repo = _repository_root()
    return {
        **dict(dataset.provenance),
        "source_hashes": dict(dataset.source_hashes),
        "training_validation_tensor_hashes": tensor_hashes,
        "round4_implementation_hashes": _round4_implementation_hashes(
            repo, config_path
        ),
        "goal_source": dataset.goal_source,
        "horizon": int(dataset.horizon),
        "max_speed_mps": float(dataset.max_speed_mps),
        "delta_t_seconds": float(dataset.delta_t_seconds),
        "maximum_step_distance_m": float(dataset.max_step_distance_m),
        "round4_config_sha256": _sha256(config_path),
    }


def _save_spatial_target_audit(dataset: Any, horizon_dir: Path) -> Dict[str, Any]:
    """Persist the physical/normalized target contract outside Round-3."""

    audit_dir = horizon_dir / "spatial_target_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        split = dataset.split(split_name)
        path = audit_dir / f"{split_name}.npz"
        np.savez_compressed(
            path,
            sample_id=np.asarray(split.sample_id),
            episode_id=np.asarray(split.episode_id),
            split=np.full(
                len(split), split_name, dtype=f"<U{len(split_name)}"
            ),
            horizon_steps=np.full(
                len(split), dataset.horizon, dtype=np.int64
            ),
            maximum_reachable_distance_m=np.full(
                len(split),
                dataset.horizon_normalization_distance_m,
                dtype=np.float64,
            ),
            q_current_xy_m=np.asarray(split.q_current_m, dtype=np.float64),
            q_future_xy_m=np.asarray(split.q_future_m, dtype=np.float64),
            raw_delta_xy_m=np.asarray(
                split.raw_delta_xy_m, dtype=np.float64
            ),
            normalized_delta_xy=np.asarray(
                split.normalized_delta_xy, dtype=np.float32
            ),
        )
        with np.load(path, allow_pickle=False) as archive:
            if any(archive[name].dtype.hasobject for name in archive.files):
                raise RuntimeError("target audit unexpectedly used object dtype")
            if archive["normalized_delta_xy"].shape != (len(split), 2):
                raise RuntimeError("target audit roundtrip shape changed")
        files[split_name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "sample_count": len(split),
        }
    manifest = {
        "schema_version": "round04_spatial_target_audit_v1",
        "horizon": dataset.horizon,
        "max_speed_mps": dataset.max_speed_mps,
        "delta_t_seconds": dataset.delta_t_seconds,
        "maximum_step_distance_m": dataset.max_step_distance_m,
        "normalization": "raw_delta_xy_m / maximum_reachable_distance_m",
        "source_read_only": True,
        "files": files,
    }
    _write_json(audit_dir / "manifest.json", manifest)
    return manifest


def _write_per_horizon_artifacts(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    horizons: Sequence[int],
    seeds: Sequence[int],
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
    example_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Write an independent resolved config and metric bundle for every H."""

    artifacts: Dict[str, Any] = {}
    tables = {
        "waypoint_training_summary.csv": training_rows,
        "waypoint_validation_metrics.csv": validation_rows,
        "waypoint_test_metrics.csv": test_rows,
        "waypoint_baseline_metrics.csv": baseline_rows,
        "waypoint_projection_metrics.csv": projection_rows,
        "waypoint_prediction_examples.csv": example_rows,
        "waypoint_horizon_selection.csv": selection_rows,
    }
    for horizon in horizons:
        horizon_dir = output_dir / f"horizon_{int(horizon):03d}"
        metrics_dir = horizon_dir / "metrics"
        resolved = dict(config)
        resolved.update(
            {
                "resolved_horizon": int(horizon),
                "candidate_horizons": [int(horizon)],
                "training_seeds": [int(seed) for seed in seeds],
                "mixed_horizon_training": False,
            }
        )
        config_path = horizon_dir / "resolved_horizon_config.json"
        _write_json(config_path, resolved)
        written: Dict[str, Any] = {
            "resolved_config": {
                "path": str(config_path.relative_to(output_dir)),
                "sha256": _sha256(config_path),
            },
            "metrics": {},
        }
        for filename, rows in tables.items():
            selected = [
                dict(row)
                for row in rows
                if int(row["horizon"]) == int(horizon)
            ]
            if not selected:
                raise RuntimeError(
                    f"no rows available for per-horizon artifact {filename}, "
                    f"H={horizon}"
                )
            if filename == "waypoint_training_summary.csv":
                for row in selected:
                    root_relative = Path(str(row["history_file"]))
                    horizon_relative = root_relative.relative_to(
                        f"horizon_{int(horizon):03d}"
                    )
                    row["history_file"] = str(Path("..") / horizon_relative)
            path = metrics_dir / filename
            _write_csv(path, selected)
            written["metrics"][filename] = {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "row_count": len(selected),
            }
        artifacts[f"H{int(horizon)}"] = written
    return artifacts


def _run_manifest(
    *,
    run_id: str,
    mode: str,
    config_path: Path,
    source_dir: Path,
    horizons: Sequence[int],
    seeds: Sequence[int],
    status: str,
) -> Dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "created_at": _now_local().isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_data_path": str(source_dir),
        "source_hashes": _source_hashes(source_dir),
        "round4_implementation_hashes": _round4_implementation_hashes(
            _repository_root(), config_path
        ),
        "candidate_horizons": [int(value) for value in horizons],
        "training_seeds": [int(value) for value in seeds],
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "training_performed": status == "completed",
        "waypoint_training_performed": status == "completed",
        "transformer_training_performed": False,
        "policy_training_performed": False,
        "closed_loop_evaluation_performed": False,
        "test_used_for_model_selection": False,
    }


def _run_round4_impl(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    mode: str,
    run_id: str | None = None,
    device: str | None = None,
    resume: bool = False,
) -> Path:
    """Implement the fixed-horizon experiment behind the audited wrapper."""

    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    repo = _repository_root()
    resolved_config_path, config = _resolve_config(config_path)
    if config.get("schema_version") != "round04_spatial_waypoint_config_v1":
        raise ValueError("unsupported Round-4 config schema")
    if config.get("output_representation") != "relative_delta":
        raise ValueError("Round-4 output representation must be relative_delta")
    if config.get("goal_source") not in {
        "achieved_terminal_goal",
        "desired_goal",
    }:
        raise ValueError("unsupported goal_source")
    model_config = config.get("model")
    training_section = config.get("training")
    if not isinstance(model_config, Mapping) or not isinstance(
        training_section, Mapping
    ):
        raise ValueError("model and training config sections are required")
    if str(model_config.get("activation", "")).lower() != "relu":
        raise ValueError("Round-4 MLP activation must be ReLU")
    if int(model_config.get("output_dim", -1)) != 2:
        raise ValueError("Round-4 MLP output_dim must be 2")
    if str(training_section.get("optimizer", "")).lower() != "adam":
        raise ValueError("Round-4 optimizer must be Adam")
    evaluation_config = config.get("evaluation")
    if not isinstance(evaluation_config, Mapping):
        raise ValueError("config evaluation section must be a mapping")
    if (
        evaluation_config.get("checkpoint_selection_metric")
        != "validation_future_endpoint_rmse_m"
    ):
        raise ValueError(
            "checkpoint selection must use validation future endpoint RMSE"
        )
    if (
        evaluation_config.get("horizon_selection_metric")
        != "validation_composite_spatial_waypoint_v1"
    ):
        raise ValueError("unsupported horizon selection metric")
    for name in (
        "near_zero_displacement_m",
        "reachability_tolerance_m",
    ):
        value = float(evaluation_config[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"evaluation {name} must be finite and nonnegative")
    source_dir = (repo / config["data_path"]).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if mode == "smoke":
        horizons = [5]
        seeds = [0]
    else:
        horizons = [int(value) for value in config["candidate_horizons"]]
        seeds = [int(value) for value in config["training_seeds"]]
    if not horizons or len(set(horizons)) != len(horizons) or any(
        value <= 0 for value in horizons
    ):
        raise ValueError("candidate horizons must be unique and positive")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        value < 0 for value in seeds
    ):
        raise ValueError("training seeds must be nonempty, unique, and nonnegative")
    default_prefix = "round04_smoke" if mode == "smoke" else "round04_formal"
    run_id = run_id or f"{default_prefix}_{_timestamp()}"
    if (
        not run_id
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("run_id must be one safe path component")
    output_root = (repo / config["output_root"]).resolve()
    figure_root = (repo / config["figure_root"]).resolve()
    if not _path_is_within(output_root, repo) or not _path_is_within(
        figure_root, repo
    ):
        raise ValueError("configured output roots must stay inside the repository")
    output_dir = (output_root / run_id).resolve()
    figure_dir = (figure_root / run_id).resolve()
    if not _path_is_within(output_dir, output_root) or not _path_is_within(
        figure_dir, figure_root
    ):
        raise ValueError("run output path escaped its configured root")
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"run directory exists; pass --resume to continue: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    environment_config = _read_json(source_dir / "run_config.json")[
        "base_environment_config"
    ]
    training_config = _training_config(
        config, device_override=device, smoke=mode == "smoke"
    )
    _write_json(output_dir / "source_config.json", config)
    manifest = _run_manifest(
        run_id=run_id,
        mode=mode,
        config_path=resolved_config_path,
        source_dir=source_dir,
        horizons=horizons,
        seeds=seeds,
        status="in_progress",
    )
    manifest["goal_source"] = config["goal_source"]
    manifest["evaluation_config"] = dict(evaluation_config)
    _write_json(output_dir / "run_manifest.json", manifest)

    training_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    datasets: Dict[int, Any] = {}
    best_checkpoints: Dict[tuple[int, int, str], str] = {}
    maximum_step_by_horizon: Dict[int, float] = {}
    overall_start = time.perf_counter()

    # Stage 1: training and validation only.  Test tensors are not passed to
    # the trainer and no test evaluation occurs before horizon selection.
    for horizon in horizons:
        dataset = load_round4_spatial_dataset(
            source_dir,
            horizon=horizon,
            goal_source=str(config["goal_source"]),
        )
        datasets[horizon] = dataset
        checkpoint_provenance = _training_provenance(
            dataset, resolved_config_path
        )
        maximum_step_by_horizon[horizon] = dataset.max_step_distance_m
        _write_json(
            output_dir / f"horizon_{horizon:03d}" / "dataset_summary.json",
            dataset.summary(),
        )
        _save_spatial_target_audit(
            dataset, output_dir / f"horizon_{horizon:03d}"
        )
        train_limit = min(1024, len(dataset.train)) if mode == "smoke" else None
        val_limit = (
            min(512, len(dataset.validation)) if mode == "smoke" else None
        )
        mean_baseline = _fit_mean(dataset)
        for baseline_name in (
            "zero_waypoint",
            "train_mean_waypoint",
            "constant_velocity",
        ):
            predictions = _baseline_predictions(
                dataset,
                dataset.validation,
                baseline_name,
                mean_baseline,
                environment_config,
            )
            metrics, projections, _ = _evaluate(
                predictions=predictions,
                split=dataset.validation,
                dataset=dataset,
                model_kind=baseline_name,
                seed="baseline",
                evaluation_config=config["evaluation"],
            )
            overall = _overall_raw(metrics)
            overall["baseline"] = baseline_name
            baseline_rows.append(overall)
            projection_rows.extend(projections)

        for model_kind in ("goal_conditioned", "state_only"):
            train_inputs = _model_inputs(dataset.train, model_kind)
            val_inputs = _model_inputs(dataset.validation, model_kind)
            train_targets = dataset.train.normalized_delta_xy
            val_targets = dataset.validation.normalized_delta_xy
            if train_limit is not None:
                train_inputs = train_inputs[:train_limit]
                train_targets = train_targets[:train_limit]
            if val_limit is not None:
                val_inputs = val_inputs[:val_limit]
                val_targets = val_targets[:val_limit]
            for seed in seeds:
                relative_run_dir = (
                    Path(f"horizon_{horizon:03d}")
                    / model_kind
                    / f"seed_{seed:03d}"
                )
                model_output = output_dir / relative_run_dir
                try:
                    result = train_spatial_waypoint(
                        train_inputs=train_inputs,
                        train_targets_normalized=train_targets,
                        validation_inputs=val_inputs,
                        validation_targets_normalized=val_targets,
                        horizon=horizon,
                        maximum_step_distance_m=dataset.max_step_distance_m,
                        seed=seed,
                        model_kind=model_kind,
                        output_dir=model_output,
                        config=training_config,
                        provenance=checkpoint_provenance,
                        valid_node_mask_dim=dataset.max_num_buoys,
                        resume=resume,
                    )
                except Exception as error:
                    manifest.update(
                        {
                            "status": "incomplete",
                            "failed_stage": "training",
                            "failed_horizon": int(horizon),
                            "failed_seed": int(seed),
                            "failed_model_kind": model_kind,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "elapsed_seconds": float(
                                time.perf_counter() - overall_start
                            ),
                        }
                    )
                    _write_json(output_dir / "run_manifest.json", manifest)
                    raise
                row = asdict(result)
                row["run_id"] = run_id
                row["goal_source"] = dataset.goal_source
                row["history_file"] = str(
                    Path(result.history_file).resolve().relative_to(output_dir)
                )
                row["best_checkpoint"] = str(
                    Path(result.best_checkpoint).resolve().relative_to(repo)
                )
                row["last_checkpoint"] = str(
                    Path(result.last_checkpoint).resolve().relative_to(repo)
                )
                row["config_file"] = str(
                    Path(result.config_file).resolve().relative_to(repo)
                )
                training_rows.append(row)
                best_checkpoints[(horizon, seed, model_kind)] = str(
                    Path(result.best_checkpoint)
                )
                model, _ = load_spatial_waypoint_checkpoint(
                    result.best_checkpoint,
                    device=training_config.device,
                    expected_horizon=horizon,
                    expected_model_kind=model_kind,
                    expected_input_dim=_model_inputs(
                        dataset.validation, model_kind
                    ).shape[1],
                    expected_provenance=checkpoint_provenance,
                )
                predictions = predict_spatial_waypoints(
                    model,
                    _model_inputs(dataset.validation, model_kind),
                    batch_size=training_config.batch_size,
                    device=training_config.device,
                )
                metrics, projections, _ = _evaluate(
                    predictions=predictions,
                    split=dataset.validation,
                    dataset=dataset,
                    model_kind=model_kind,
                    seed=seed,
                    evaluation_config=config["evaluation"],
                )
                validation_rows.extend(
                    metric
                    for metric in metrics
                    if metric["group_dimension"] == "overall"
                )
                projection_rows.extend(projections)
                if model_kind == "state_only":
                    state_overall = _overall_raw(metrics)
                    state_overall["baseline"] = "state_only_mlp"
                    baseline_rows.append(state_overall)

    selection = _selection_rows(
        validation_rows, maximum_step_by_horizon
    )
    recommended_horizon = int(
        next(row["horizon"] for row in selection if row["recommended"])
    )
    _write_csv(output_dir / "waypoint_training_summary.csv", training_rows)
    _write_csv(output_dir / "waypoint_validation_metrics.csv", validation_rows)
    _write_csv(output_dir / "waypoint_horizon_selection.csv", selection)

    # Stage 2: selection is now immutable.  Every best checkpoint is evaluated
    # on test; smoke has one seed and one horizon but retains both matched MLPs.
    for horizon in horizons:
        dataset = datasets[horizon]
        checkpoint_provenance = _training_provenance(
            dataset, resolved_config_path
        )
        mean_baseline = _fit_mean(dataset)
        for baseline_name in (
            "zero_waypoint",
            "train_mean_waypoint",
            "constant_velocity",
        ):
            predictions = _baseline_predictions(
                dataset,
                dataset.test,
                baseline_name,
                mean_baseline,
                environment_config,
            )
            metrics, projections, _ = _evaluate(
                predictions=predictions,
                split=dataset.test,
                dataset=dataset,
                model_kind=baseline_name,
                seed="baseline",
                evaluation_config=config["evaluation"],
            )
            overall = _overall_raw(metrics)
            overall["baseline"] = baseline_name
            baseline_rows.append(overall)
            projection_rows.extend(projections)
        for model_kind in ("goal_conditioned", "state_only"):
            for seed in seeds:
                path = best_checkpoints[(horizon, seed, model_kind)]
                model, _ = load_spatial_waypoint_checkpoint(
                    path,
                    device=training_config.device,
                    expected_horizon=horizon,
                    expected_model_kind=model_kind,
                    expected_input_dim=_model_inputs(
                        dataset.test, model_kind
                    ).shape[1],
                    expected_provenance=checkpoint_provenance,
                )
                predictions = predict_spatial_waypoints(
                    model,
                    _model_inputs(dataset.test, model_kind),
                    batch_size=training_config.batch_size,
                    device=training_config.device,
                )
                metrics, projections, _ = _evaluate(
                    predictions=predictions,
                    split=dataset.test,
                    dataset=dataset,
                    model_kind=model_kind,
                    seed=seed,
                    evaluation_config=config["evaluation"],
                )
                test_rows.extend(metrics)
                projection_rows.extend(projections)
                if model_kind == "state_only":
                    state_overall = _overall_raw(metrics)
                    state_overall["baseline"] = "state_only_mlp"
                    baseline_rows.append(state_overall)
                example_rows.extend(
                    _prediction_examples(
                        split=dataset.test,
                        dataset=dataset,
                        predictions=predictions,
                        seed=seed,
                        model_kind=model_kind,
                        limit=int(
                            config["evaluation"][
                                "prediction_examples_per_run"
                            ]
                        ),
                        projection_tolerance=float(
                            config["evaluation"]["reachability_tolerance_m"]
                        )
                        / dataset.horizon_normalization_distance_m,
                    )
                )

    selected_config = {
        "schema_version": "round04_selected_waypoint_config_v1",
        "recommended_horizon": recommended_horizon,
        "selection_split": "validation",
        "checkpoint_selection_metric": evaluation_config[
            "checkpoint_selection_metric"
        ],
        "horizon_selection_metric": evaluation_config[
            "horizon_selection_metric"
        ],
        "test_metrics_used_for_selection": False,
        "goal_source": config["goal_source"],
        "output_representation": "relative_delta",
        "training_config": asdict(training_config),
        "evaluation_config": dict(evaluation_config),
        "best_checkpoints": {
            f"H{horizon}_seed{seed}_{model_kind}": str(
                Path(path).resolve().relative_to(repo)
            )
            for (horizon, seed, model_kind), path in sorted(
                best_checkpoints.items()
            )
        },
    }
    _write_json(output_dir / "selected_waypoint_config.json", selected_config)
    first_dataset = datasets[horizons[0]]
    input_schema = {
        "schema_version": "round04_spatial_waypoint_input_v1",
        "source_schema": first_dataset.schema,
        "state_dim": first_dataset.state_dim,
        "goal_dim": first_dataset.goal_dim,
        "valid_node_mask_dim": first_dataset.max_num_buoys,
        "goal_conditioned_input_dim": first_dataset.goal_conditioned_input_dim,
        "state_only_input_dim": first_dataset.state_only_input_dim,
        "goal_conditioned_order": [
            "masked_state_t",
            "masked_goal",
            "valid_node_mask_t",
        ],
        "state_only_order": ["masked_state_t", "valid_node_mask_t"],
        "predictor_field_denylist": sorted(PREDICTOR_LEAKAGE_DENYLIST),
        "goal_protocol": "explicit hindsight achieved-terminal conditioning",
    }
    output_schema = {
        "schema_version": "round04_spatial_waypoint_output_v1",
        "shape": ["B", 2],
        "output_dim": 2,
        "semantics": "q[t+H]-q[t]",
        "raw_unit": "m",
        "training_representation": "normalized relative displacement",
        "normalization": "raw_delta_xy_m / (H * max_speed * delta_t)",
        "maximum_step_distance_source": "locked Round-3 run_config.json",
        "training_projection_applied": False,
        "evaluation_projection": "radial unit-disk projection",
    }
    _write_json(output_dir / "input_schema.json", input_schema)
    _write_json(output_dir / "output_schema.json", output_schema)
    _write_csv(output_dir / "waypoint_test_metrics.csv", test_rows)
    _write_csv(output_dir / "waypoint_baseline_metrics.csv", baseline_rows)
    _write_csv(output_dir / "waypoint_prediction_examples.csv", example_rows)
    _write_csv(output_dir / "waypoint_projection_metrics.csv", projection_rows)
    per_horizon_artifacts = _write_per_horizon_artifacts(
        output_dir=output_dir,
        config=config,
        horizons=horizons,
        seeds=seeds,
        training_rows=training_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        baseline_rows=baseline_rows,
        projection_rows=projection_rows,
        example_rows=example_rows,
        selection_rows=selection,
    )
    source_hashes_after = _source_hashes(source_dir)
    if source_hashes_after != manifest["source_hashes"]:
        manifest.update(
            {
                "status": "incomplete",
                "failed_stage": "source_integrity_recheck",
                "source_hashes_after": source_hashes_after,
                "source_hashes_unchanged": False,
                "elapsed_seconds": float(time.perf_counter() - overall_start),
            }
        )
        _write_json(output_dir / "run_manifest.json", manifest)
        raise RuntimeError(
            "locked Round-3 source hashes changed during Round-4 execution"
        )

    manifest.update(
        {
            "status": "figures_pending",
            "recommended_horizon": recommended_horizon,
            "training_run_count": len(training_rows),
            "goal_conditioned_run_count": sum(
                row["model_kind"] == "goal_conditioned"
                for row in training_rows
            ),
            "state_only_run_count": sum(
                row["model_kind"] == "state_only" for row in training_rows
            ),
            "training_performed": True,
            "waypoint_training_performed": True,
            "transformer_training_performed": False,
            "policy_training_performed": False,
            "closed_loop_evaluation_performed": False,
            "test_used_for_model_selection": False,
            "per_horizon_artifacts": per_horizon_artifacts,
            "source_hashes_after": source_hashes_after,
            "source_hashes_unchanged": True,
            "required_outputs": [
                "waypoint_training_summary.csv",
                "waypoint_validation_metrics.csv",
                "waypoint_test_metrics.csv",
                "waypoint_baseline_metrics.csv",
                "waypoint_horizon_selection.csv",
                "waypoint_prediction_examples.csv",
                "waypoint_projection_metrics.csv",
                "selected_waypoint_config.json",
                "input_schema.json",
                "output_schema.json",
                "run_manifest.json",
            ],
        }
    )
    _write_json(output_dir / "run_manifest.json", manifest)

    from .round4_figures import (
        generate_per_horizon_loss_figures,
        generate_round4_figures,
    )

    try:
        figures = generate_round4_figures(
            output_dir,
            figure_dir,
            selected_horizon=recommended_horizon,
            overwrite=resume,
        )
        figures.update(
            generate_per_horizon_loss_figures(
                output_dir, figure_dir, overwrite=resume
            )
        )
    except Exception as error:
        manifest.update(
            {
                "status": "incomplete",
                "failed_stage": "figure_generation",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "elapsed_seconds": float(time.perf_counter() - overall_start),
            }
        )
        _write_json(output_dir / "run_manifest.json", manifest)
        raise
    manifest.update(
        {
            "status": "completed",
            "completed_at": _now_local().isoformat(),
            "elapsed_seconds": float(time.perf_counter() - overall_start),
            "figures": figures,
            "figure_file_count": len(figures),
            "figure_category_count": 13,
            "per_horizon_loss_figure_count": 2 * len(horizons),
        }
    )
    _write_json(output_dir / "run_manifest.json", manifest)
    return output_dir


def _record_unhandled_run_failure(
    *,
    config_path: str | Path,
    run_id: str,
    error: BaseException,
    interrupted: bool,
) -> None:
    """Best-effort failure closure for stages outside local error guards."""

    try:
        repo = _repository_root()
        _, config = _resolve_config(config_path)
        output_root = (repo / config["output_root"]).resolve()
        output_dir = (output_root / run_id).resolve()
        if not _path_is_within(output_dir, output_root):
            return
        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return
        manifest = _read_json(manifest_path)
        if manifest.get("status") not in {"in_progress", "figures_pending"}:
            return
        manifest.update(
            {
                "status": "incomplete",
                "failed_stage": (
                    "interrupted" if interrupted else "unhandled_exception"
                ),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "resume_supported": True,
                "failed_at": _now_local().isoformat(),
            }
        )
        _write_json(manifest_path, manifest)
    except Exception:
        # Preserve the original exception even if failure recording itself is
        # impossible (for example, a malformed config or unwritable disk).
        return


def run_round4(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    mode: str,
    run_id: str | None = None,
    device: str | None = None,
    resume: bool = False,
) -> Path:
    """Run Round-4 and ensure non-successful CLI exits remain auditable."""

    if run_id is None:
        prefix = "round04_smoke" if mode == "smoke" else "round04_formal"
        run_id = f"{prefix}_{_timestamp()}"
    try:
        return _run_round4_impl(
            config_path=config_path,
            mode=mode,
            run_id=run_id,
            device=device,
            resume=resume,
        )
    except KeyboardInterrupt as error:
        _record_unhandled_run_failure(
            config_path=config_path,
            run_id=run_id,
            error=error,
            interrupted=True,
        )
        raise
    except Exception as error:
        if not isinstance(error, FileExistsError):
            _record_unhandled_run_failure(
                config_path=config_path,
                run_id=run_id,
                error=error,
                interrupted=False,
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the Round-4 two-dimensional spatial "
            "Waypoint Network only."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = "smoke" if args.smoke else "formal"
    output = run_round4(
        config_path=args.config,
        mode=mode,
        run_id=args.run_id,
        device=args.device,
        resume=args.resume,
    )
    print(f"round4_status=completed")
    print(f"round4_mode={mode}")
    print(f"round4_output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_round4"]
