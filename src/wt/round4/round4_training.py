"""Deterministic trainer for Round-4 spatial Waypoint networks.

This module intentionally uses plain PyTorch.  The official training entry
point is coupled to D4RL, Gym, Weights & Biases, and an old Lightning API that
cannot consume the locked maritime NPZ files.  The model topology and PyTorch
initialization remain aligned with the official goal network, while this thin
trainer adds the fixed-split, local-checkpoint semantics needed by Round-4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# CUDA deterministic matrix multiplication requires this before the first
# cuBLAS operation.  ``setdefault`` preserves an explicit operator choice.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, TensorDataset

from .spatial_waypoint import (
    SpatialWaypointMLP,
    normalized_displacement_mse,
    set_deterministic_seed,
)


CHECKPOINT_FORMAT_VERSION = "round04_spatial_waypoint_checkpoint_v1"
MODEL_KINDS = ("goal_conditioned", "state_only")


@dataclass(frozen=True)
class SpatialTrainingConfig:
    """Resolved training settings saved verbatim in every checkpoint."""

    hidden_dim: int = 256
    num_hidden_layers: int = 3
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    maximum_epochs: int = 300
    early_stopping_patience: int = 30
    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    device: str = "auto"
    deterministic: bool = True
    num_workers: int = 0

    def validate(self) -> "SpatialTrainingConfig":
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.maximum_epochs <= 0:
            raise ValueError("maximum_epochs must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be nonnegative")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if (
            not math.isfinite(self.gradient_clipping)
            or self.gradient_clipping <= 0
        ):
            raise ValueError("gradient_clipping must be positive and finite")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        return self


@dataclass(frozen=True)
class SpatialTrainingResult:
    """Serializable summary of one fixed-horizon/model-kind/seed run."""

    horizon: int
    seed: int
    model_kind: str
    input_dim: int
    output_dim: int
    parameter_count: int
    train_sample_count: int
    validation_sample_count: int
    best_epoch: int
    completed_epochs: int
    best_validation_endpoint_rmse_m: float
    best_validation_normalized_mse: float
    total_training_time_seconds: float
    device: str
    stopped_early: bool
    resumed: bool
    best_checkpoint: str
    last_checkpoint: str
    history_file: str
    config_file: str


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit CPU/CUDA device without silently changing requests."""

    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        index = int(value.split(":", 1)[1])
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {index}")
        return torch.device(value)
    raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")


def _as_float32_matrix(
    name: str,
    values: np.ndarray,
    *,
    rows: Optional[int] = None,
    columns: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-two matrix")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count is inconsistent")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have width {columns}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array)


def _make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    config: SpatialTrainingConfig,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset = TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )


def predict_spatial_waypoints(
    model: SpatialWaypointMLP,
    inputs: np.ndarray,
    *,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return finite float32 normalized displacement predictions."""

    matrix = _as_float32_matrix(
        "inputs", inputs, columns=int(model.input_dim)
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved = (
        device if isinstance(device, torch.device) else resolve_device(device)
    )
    model = model.to(resolved)
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            batch = torch.from_numpy(matrix[start : start + batch_size]).to(
                resolved
            )
            prediction = model(batch).detach().cpu().numpy()
            chunks.append(np.asarray(prediction, dtype=np.float32))
    result = np.concatenate(chunks, axis=0)
    if result.shape != (matrix.shape[0], 2):
        raise RuntimeError("spatial model returned an invalid prediction shape")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("spatial model prediction contains NaN or Inf")
    return result


def _validation_metrics(
    model: SpatialWaypointMLP,
    inputs: np.ndarray,
    normalized_targets: np.ndarray,
    displacement_scale_m: float,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float]:
    prediction = predict_spatial_waypoints(
        model,
        inputs,
        batch_size=batch_size,
        device=device,
    )
    squared_norm = np.sum(
        np.square(prediction - normalized_targets, dtype=np.float64),
        axis=1,
    )
    normalized_mse = float(np.mean(squared_norm))
    endpoint_rmse_m = float(
        math.sqrt(normalized_mse) * displacement_scale_m
    )
    return normalized_mse, endpoint_rmse_m


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "epoch",
        "train_normalized_mse",
        "validation_normalized_mse",
        "validation_endpoint_rmse_m",
        "learning_rate",
        "epoch_seconds",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row[name] for name in columns})


def _checkpoint_payload(
    *,
    model: SpatialWaypointMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_validation_endpoint_rmse_m: float,
    best_validation_normalized_mse: float,
    epochs_without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    horizon: int,
    displacement_scale_m: float,
    seed: int,
    model_kind: str,
    training_config: SpatialTrainingConfig,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_class": "SpatialWaypointMLP",
        "architecture": model.architecture_metadata(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "optimizer_step": int(
            sum(
                int(state.get("step", 0))
                for state in optimizer.state.values()
                if isinstance(state, Mapping)
            )
        ),
        "best_epoch": int(best_epoch),
        "best_validation_endpoint_rmse_m": float(
            best_validation_endpoint_rmse_m
        ),
        "best_validation_normalized_mse": float(
            best_validation_normalized_mse
        ),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": [dict(row) for row in history],
        "horizon": int(horizon),
        "displacement_scale_m": float(displacement_scale_m),
        "maximum_step_distance_m": float(displacement_scale_m / horizon),
        "seed": int(seed),
        "model_kind": str(model_kind),
        "output_representation": "relative_delta",
        "output_unit": "normalized_by_horizon_times_maximum_step_distance",
        "training_config": asdict(training_config),
        "provenance": dict(provenance),
        "status": {
            "training_performed": True,
            "waypoint_training_performed": True,
            "transformer_training_performed": False,
            "policy_training_performed": False,
            "closed_loop_evaluation_performed": False,
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def _torch_load_checkpoint(path: Path, device: torch.device) -> Any:
    """Load across the supported Torch 1.13 and current APIs."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location=device)


def load_spatial_waypoint_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_horizon: Optional[int] = None,
    expected_model_kind: Optional[str] = None,
    expected_input_dim: Optional[int] = None,
    expected_provenance: Optional[Mapping[str, Any]] = None,
) -> Tuple[SpatialWaypointMLP, Dict[str, Any]]:
    """Strictly rebuild a model and restore a Round-4 checkpoint."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = (
        device if isinstance(device, torch.device) else resolve_device(device)
    )
    payload = _torch_load_checkpoint(path, resolved)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported spatial waypoint checkpoint format")
    if payload.get("output_representation") != "relative_delta":
        raise ValueError("checkpoint does not represent relative displacement")
    if expected_horizon is not None and int(payload["horizon"]) != int(
        expected_horizon
    ):
        raise ValueError("checkpoint horizon does not match requested horizon")
    if expected_model_kind is not None and payload.get(
        "model_kind"
    ) != expected_model_kind:
        raise ValueError("checkpoint model kind does not match request")
    if expected_provenance is not None and payload.get("provenance") != dict(
        expected_provenance
    ):
        raise ValueError("checkpoint provenance does not match current data")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint architecture metadata is missing")
    model = SpatialWaypointMLP.from_metadata(architecture)
    if expected_input_dim is not None and model.input_dim != int(
        expected_input_dim
    ):
        raise ValueError("checkpoint input dimension does not match schema")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved)
    model.eval()
    return model, payload


def train_spatial_waypoint(
    *,
    train_inputs: np.ndarray,
    train_targets_normalized: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets_normalized: np.ndarray,
    horizon: int,
    maximum_step_distance_m: float,
    seed: int,
    model_kind: str,
    output_dir: str | Path,
    config: SpatialTrainingConfig,
    provenance: Mapping[str, Any],
    valid_node_mask_dim: int = 0,
    resume: bool = False,
) -> SpatialTrainingResult:
    """Train one independent fixed-horizon spatial Waypoint MLP.

    Checkpoint selection uses validation future-endpoint RMSE only.  Test data
    is intentionally absent from this function's interface.
    """

    config = config.validate()
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    horizon = int(horizon)
    if (
        not math.isfinite(maximum_step_distance_m)
        or maximum_step_distance_m <= 0
    ):
        raise ValueError("maximum_step_distance_m must be positive and finite")
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"unsupported model_kind: {model_kind}")
    seed = int(seed)
    valid_node_mask_dim = int(valid_node_mask_dim)
    if valid_node_mask_dim < 0:
        raise ValueError("valid_node_mask_dim must be nonnegative")
    train_x = _as_float32_matrix("train_inputs", train_inputs)
    train_y = _as_float32_matrix(
        "train_targets_normalized",
        train_targets_normalized,
        rows=train_x.shape[0],
        columns=2,
    )
    val_x = _as_float32_matrix(
        "validation_inputs",
        validation_inputs,
        columns=train_x.shape[1],
    )
    val_y = _as_float32_matrix(
        "validation_targets_normalized",
        validation_targets_normalized,
        rows=val_x.shape[0],
        columns=2,
    )
    displacement_scale_m = float(horizon * maximum_step_distance_m)
    run_dir = Path(output_dir)
    checkpoint_dir = run_dir / "checkpoints"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    history_path = run_dir / "training_history.csv"
    config_path = run_dir / "resolved_config.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(config.device)
    set_deterministic_seed(seed, deterministic=config.deterministic)
    model = SpatialWaypointMLP(
        input_dim=train_x.shape[1],
        output_dim=2,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        dropout=config.dropout,
        valid_node_mask_dim=valid_node_mask_dim,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, Any]] = []
    start_epoch = 0
    best_epoch = -1
    best_endpoint_rmse = math.inf
    best_normalized_mse = math.inf
    resumed = False

    if resume and last_path.is_file():
        restored_model, payload = load_spatial_waypoint_checkpoint(
            last_path,
            device=resolved_device,
            expected_horizon=horizon,
            expected_model_kind=model_kind,
            expected_input_dim=train_x.shape[1],
            expected_provenance=provenance,
        )
        restored_architecture = restored_model.architecture_metadata()
        if restored_architecture != model.architecture_metadata():
            raise ValueError("resume checkpoint architecture conflicts with config")
        saved_config = payload.get("training_config")
        if saved_config != asdict(config):
            raise ValueError("resume checkpoint training config conflicts")
        if float(payload["maximum_step_distance_m"]) != float(
            maximum_step_distance_m
        ):
            raise ValueError("resume checkpoint displacement scale conflicts")
        model.load_state_dict(restored_model.state_dict(), strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_endpoint_rmse = float(
            payload["best_validation_endpoint_rmse_m"]
        )
        best_normalized_mse = float(
            payload["best_validation_normalized_mse"]
        )
        history = [dict(row) for row in payload.get("history", [])]
        epochs_without_improvement = int(
            payload.get("epochs_without_improvement", 0)
        )
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("resume checkpoint is missing RNG state")
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        resumed = True

    resolved = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "horizon": horizon,
        "maximum_step_distance_m": float(maximum_step_distance_m),
        "displacement_scale_m": displacement_scale_m,
        "seed": seed,
        "model_kind": model_kind,
        "architecture": model.architecture_metadata(),
        "parameter_count": model.parameter_count(),
        "valid_node_mask_dim": valid_node_mask_dim,
        "training": asdict(config),
        "train_sample_count": int(train_x.shape[0]),
        "validation_sample_count": int(val_x.shape[0]),
        "device": str(resolved_device),
        "provenance": dict(provenance),
        "selection_rule": "minimum validation future endpoint RMSE",
        "test_data_used": False,
    }
    _write_json(config_path, resolved)

    total_start = time.perf_counter()
    if not resumed:
        epochs_without_improvement = 0
    stopped_early = bool(
        resumed
        and config.early_stopping_patience > 0
        and epochs_without_improvement >= config.early_stopping_patience
    )
    if stopped_early:
        start_epoch = config.maximum_epochs
    for epoch in range(start_epoch, config.maximum_epochs):
        epoch_start = time.perf_counter()
        # Recreate the loader with an epoch-specific seed.  This makes a
        # resumed run use exactly the same shuffle as an uninterrupted run.
        train_loader = _make_loader(
            train_x,
            train_y,
            config,
            seed=seed * 1_000_003 + epoch,
            shuffle=True,
        )
        model.train()
        weighted_loss = 0.0
        examples = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = normalized_displacement_mse(
                predictions, batch_targets
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss became NaN or Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clipping
            )
            optimizer.step()
            batch_count = int(batch_inputs.shape[0])
            weighted_loss += float(loss.detach().cpu()) * batch_count
            examples += batch_count
        train_loss = weighted_loss / examples
        val_normalized_mse, val_endpoint_rmse = _validation_metrics(
            model,
            val_x,
            val_y,
            displacement_scale_m,
            resolved_device,
            config.batch_size,
        )
        row = {
            "epoch": int(epoch),
            "train_normalized_mse": float(train_loss),
            "validation_normalized_mse": float(val_normalized_mse),
            "validation_endpoint_rmse_m": float(val_endpoint_rmse),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
        }
        history.append(row)
        improved = val_endpoint_rmse < best_endpoint_rmse
        if improved:
            best_endpoint_rmse = val_endpoint_rmse
            best_normalized_mse = val_normalized_mse
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation_endpoint_rmse_m=best_endpoint_rmse,
            best_validation_normalized_mse=best_normalized_mse,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            horizon=horizon,
            displacement_scale_m=displacement_scale_m,
            seed=seed,
            model_kind=model_kind,
            training_config=config,
            provenance=provenance,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        _write_history(history_path, history)
        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement
            >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    total_seconds = time.perf_counter() - total_start
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("training did not produce required checkpoints")
    if best_epoch < 0 or not math.isfinite(best_endpoint_rmse):
        raise RuntimeError("training did not produce a finite validation result")
    best_model, best_payload = load_spatial_waypoint_checkpoint(
        best_path,
        device=resolved_device,
        expected_horizon=horizon,
        expected_model_kind=model_kind,
        expected_input_dim=train_x.shape[1],
    )
    check_mse, check_endpoint = _validation_metrics(
        best_model,
        val_x,
        val_y,
        displacement_scale_m,
        resolved_device,
        config.batch_size,
    )
    if not math.isclose(
        check_endpoint,
        float(best_payload["best_validation_endpoint_rmse_m"]),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError("best checkpoint reload changed validation metric")
    result = SpatialTrainingResult(
        horizon=horizon,
        seed=seed,
        model_kind=model_kind,
        input_dim=int(train_x.shape[1]),
        output_dim=2,
        parameter_count=int(model.parameter_count()),
        train_sample_count=int(train_x.shape[0]),
        validation_sample_count=int(val_x.shape[0]),
        best_epoch=int(best_epoch),
        completed_epochs=len(history),
        best_validation_endpoint_rmse_m=float(check_endpoint),
        best_validation_normalized_mse=float(check_mse),
        total_training_time_seconds=float(total_seconds),
        device=str(resolved_device),
        stopped_early=bool(stopped_early),
        resumed=bool(resumed),
        best_checkpoint=str(best_path),
        last_checkpoint=str(last_path),
        history_file=str(history_path),
        config_file=str(config_path),
    )
    _write_json(run_dir / "training_result.json", asdict(result))
    return result


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "MODEL_KINDS",
    "SpatialTrainingConfig",
    "SpatialTrainingResult",
    "load_spatial_waypoint_checkpoint",
    "predict_spatial_waypoints",
    "resolve_device",
    "train_spatial_waypoint",
]
