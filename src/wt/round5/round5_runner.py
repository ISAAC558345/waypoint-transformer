"""Configuration-driven Round-5 offline trajectory-policy experiment runner.

The runner is intentionally the only orchestration layer for Round-5.  It
keeps the locked Round-3 data and Round-4 waypoint checkpoint read-only,
finishes every validation-only choice before touching test sequences, and
labels the bounded environment integration check separately from formal
closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

# Must be set before CUDA is initialized when deterministic algorithms are used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader

from .causal_policy import (
    CausalPolicyConfig,
    MLPPolicyConfig,
    build_policy,
    count_parameters,
)
from .frozen_waypoint import (
    SELECTED_WAYPOINT_CHECKPOINT_SHA256,
    FrozenWaypointCache,
    FrozenWaypointPredictor,
    build_frozen_waypoint_cache,
    load_frozen_waypoint_cache,
    load_frozen_waypoint_predictor,
    materialize_frozen_waypoint_cache,
)
from .round5_data import (
    CANONICAL_COLLECTION_POLICIES,
    Round5SequenceSplit,
    Round5TransitionDataset,
    build_round5_sequence_split,
    load_round5_transition_dataset,
)
from .round5_diagnostics import (
    build_replay_diagnostics,
    run_interface_smoke,
    write_replay_diagnostics,
)
from .round5_evaluation import (
    ActionEvaluation,
    evaluate_action_predictions,
    write_finite_json,
)
from .round5_figures import generate_round5_figures
from .round5_training import (
    Round5Trainer,
    TrainingConfig,
    TrainingResult,
    load_policy_checkpoint,
    make_policy_balanced_loader,
    seed_everything,
)


RUN_MANIFEST_VERSION = "round05_run_manifest_v1"
DEFAULT_CONFIG = "round_05/configs/causal_transformer_v1.json"
FORMAL_RUN_COUNT = 27
STATUS_FIELDS: Mapping[str, bool] = {
    "waypoint_training_performed": False,
    "waypoint_checkpoint_frozen": True,
    "closed_loop_evaluation_performed": False,
    "ppo_training_performed": False,
    "hgnn_training_performed": False,
}


def build_seeded_policy(
    model_config: CausalPolicyConfig | MLPPolicyConfig,
    training_config: TrainingConfig,
) -> torch.nn.Module:
    """Build a policy only after its run-specific RNG state is fixed.

    ``Round5Trainer`` also resets RNG state before optimization so dropout and
    sampling start reproducibly.  Seeding here is separately necessary because
    PyTorch parameter initialization happens inside ``build_policy``.
    """

    seed_everything(training_config.seed, training_config.deterministic_algorithms)
    return build_policy(model_config)

# The configuration and output tables use the audited module's canonical
# ``wt_transformer`` label throughout.
_INTERNAL_VARIANT = {
    "wt_transformer": "wt_transformer",
    "direct_goal_transformer": "direct_goal_transformer",
    "state_only_transformer": "state_only_transformer",
    "oracle_waypoint_transformer": "oracle_waypoint_transformer",
    "mlp_state_only": "mlp_state_only",
    "mlp_waypoint": "mlp_waypoint",
}
_CONDITION_SOURCE = {
    "wt_transformer": "predicted_waypoint",
    "direct_goal_transformer": "direct_goal",
    "state_only_transformer": "state_only",
    "oracle_waypoint_transformer": "oracle_waypoint",
    "mlp_state_only": "state_only",
    "mlp_waypoint": "predicted_waypoint",
}
_TRANSFORMERS = frozenset(
    {
        "wt_transformer",
        "direct_goal_transformer",
        "state_only_transformer",
        "oracle_waypoint_transformer",
    }
)
_NON_DEPLOYABLE = frozenset({"oracle_waypoint_transformer"})


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def _module_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


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
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Round-5 artifacts may not contain NaN or Inf")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    return write_finite_json(path, _json_value(value))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError(f"refusing to write empty required CSV: {path.name}")
    fields: List[str] = []
    for row in records:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in records:
            row: Dict[str, Any] = {}
            for key in fields:
                value = _json_value(source.get(key, ""))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                row[key] = value
            writer.writerow(row)
    return path


def _write_prediction_npz(path: Path, prediction: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, prediction=np.asarray(prediction, dtype=np.float32))
    os.replace(temporary, path)
    return path


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_round5_config(config_path: str | Path = DEFAULT_CONFIG) -> Tuple[Path, Dict[str, Any]]:
    root = repository_root()
    path = _resolve_path(config_path, root)
    config = _read_json(path)
    required = {
        "dataset_root",
        "waypoint_checkpoint",
        "waypoint_checkpoint_sha256",
        "waypoint_horizon",
        "goal_source",
        "output_root",
        "figure_root",
        "formal_matrix",
        "transformer",
        "mlp",
        "training",
        "evaluation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Round-5 config misses required fields: {missing}")
    if str(config["waypoint_checkpoint_sha256"]).upper() != SELECTED_WAYPOINT_CHECKPOINT_SHA256:
        raise ValueError("config does not select the locked validation-chosen H=30 checkpoint")
    if int(config["waypoint_horizon"]) != 30:
        raise ValueError("Round-5 is locked to waypoint horizon H_w=30")
    return path, config


def build_formal_run_matrix(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand and validate the exact formal 27-run matrix."""

    matrix = config.get("formal_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("formal_matrix must be a JSON object")
    if set(matrix) != set(_INTERNAL_VARIANT):
        raise ValueError("formal_matrix must contain exactly the six Round-5 variants")
    result: List[Dict[str, Any]] = []
    for model_type in _INTERNAL_VARIANT:
        entry = matrix[model_type]
        if not isinstance(entry, Mapping):
            raise ValueError(f"formal_matrix.{model_type} must be an object")
        contexts = [int(value) for value in entry.get("contexts", [])]
        seeds = [int(value) for value in entry.get("seeds", [])]
        expected_contexts = (
            [1, 10]
            if model_type in {
                "wt_transformer",
                "direct_goal_transformer",
                "state_only_transformer",
            }
            else [10] if model_type == "oracle_waypoint_transformer" else [1]
        )
        if contexts != expected_contexts or seeds != [0, 1, 2]:
            raise ValueError(
                f"formal matrix changed for {model_type}: contexts={contexts}, seeds={seeds}"
            )
        for context_length in contexts:
            for seed in seeds:
                result.append(
                    {
                        "model_type": model_type,
                        "internal_variant": _INTERNAL_VARIANT[model_type],
                        "condition_source": _CONDITION_SOURCE[model_type],
                        "context_length": context_length,
                        "seed": seed,
                        "deployable": model_type not in _NON_DEPLOYABLE,
                        "non_deployable_diagnostic": model_type in _NON_DEPLOYABLE,
                    }
                )
    keys = {(row["model_type"], row["context_length"], row["seed"]) for row in result}
    if len(result) != FORMAL_RUN_COUNT or len(keys) != FORMAL_RUN_COUNT:
        raise ValueError(f"formal matrix must contain exactly {FORMAL_RUN_COUNT} unique runs")
    return result


def build_smoke_run_matrix(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    formal = build_formal_run_matrix(config)
    return [
        row
        for row in formal
        if row["model_type"] == "wt_transformer"
        and row["context_length"] == 1
        and row["seed"] == 0
    ]


def _run_name(row: Mapping[str, Any]) -> str:
    return f"{row['model_type']}__hc{int(row['context_length']):02d}__seed{int(row['seed']):03d}"


def _model_config(
    row: Mapping[str, Any], sequence: Round5SequenceSplit, config: Mapping[str, Any]
) -> CausalPolicyConfig | MLPPolicyConfig:
    common = {
        "variant": str(row["internal_variant"]),
        "state_dim": sequence.state_dim,
        "node_mask_dim": sequence.node_mask_dim,
        "condition_dim": sequence.condition_dim,
        "action_dim": sequence.action_dim,
        "time_dim": int(sequence.remaining_time.shape[2]),
    }
    if row["model_type"] in _TRANSFORMERS:
        transformer = config["transformer"]
        return CausalPolicyConfig(
            **common,
            context_length=int(row["context_length"]),
            n_embd=int(transformer["d_model"]),
            n_layer=int(transformer["n_layer"]),
            n_head=int(transformer["n_head"]),
            dropout=float(transformer["dropout"]),
        )
    mlp = config["mlp"]
    return MLPPolicyConfig(
        **common,
        hidden_dims=tuple(int(value) for value in mlp["hidden_dims"]),
        dropout=float(mlp.get("dropout", config["transformer"]["dropout"])),
    )


def _training_config(
    row: Mapping[str, Any],
    dataset: Round5TransitionDataset,
    config: Mapping[str, Any],
    *,
    mode: str,
    device: Optional[str],
    max_epochs: Optional[int],
    patience: Optional[int],
) -> TrainingConfig:
    training = config["training"]
    epochs = int(max_epochs) if max_epochs is not None else (
        1 if mode == "smoke" else int(training["max_epochs"])
    )
    early = int(patience) if patience is not None else (
        1 if mode == "smoke" else int(training["patience"])
    )
    return TrainingConfig(
        seed=int(row["seed"]),
        action_scale=float(dataset.maximum_step_distance_m),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        batch_size=int(training["batch_size"]),
        max_epochs=epochs,
        patience=early,
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        expected_policy_count=len(CANONICAL_COLLECTION_POLICIES),
        samples_per_epoch=(int(training["batch_size"]) if mode == "smoke" else None),
        device=str(device or config.get("device", "auto")),
    )


def _loader(sequence: Round5SequenceSplit, batch_size: int) -> DataLoader:
    return DataLoader(sequence, batch_size=int(batch_size), shuffle=False, num_workers=0)


def _resolve_torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def predict_actions(
    model: torch.nn.Module,
    sequence: Round5SequenceSplit,
    *,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    """Predict one action per sequence anchor in deterministic dataset order."""

    resolved = device if isinstance(device, torch.device) else _resolve_torch_device(device)
    model.to(resolved)
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequence), int(batch_size)):
            stop = min(len(sequence), start + int(batch_size))
            inputs: Dict[str, torch.Tensor] = {
                "states": torch.from_numpy(sequence.states[start:stop]).to(resolved),
                "remaining_time": torch.from_numpy(sequence.remaining_time[start:stop]).to(resolved),
                "attention_mask": torch.from_numpy(sequence.attention_mask[start:stop]).to(resolved),
            }
            if sequence.node_mask_dim:
                inputs["node_masks"] = torch.from_numpy(sequence.node_masks[start:stop]).to(resolved)
            if sequence.condition_dim:
                inputs["conditions"] = torch.from_numpy(sequence.conditions[start:stop]).to(resolved)
            prediction = model(**inputs)
            if prediction.ndim != 3 or prediction.shape[:2] != (stop - start, sequence.context_length):
                raise RuntimeError("Round-5 policy returned an invalid sequence action shape")
            anchor = prediction[:, -1].detach().to("cpu", dtype=torch.float32).numpy()
            chunks.append(anchor)
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(sequence), sequence.action_dim):
        raise RuntimeError("Round-5 policy prediction shape changed")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Round-5 action predictions contain NaN or Inf")
    return result


def _target_anchor(sequence: Round5SequenceSplit) -> np.ndarray:
    return np.asarray(sequence.actions[:, -1], dtype=np.float32)


def _metadata_rows(split: Any) -> List[Dict[str, Any]]:
    columns = split.evaluation_metadata()
    rows: List[Dict[str, Any]] = []
    for index in range(len(split)):
        row = {
            name: _json_value(np.asarray(values)[index])
            for name, values in columns.items()
        }
        # Evaluation accepts several historical aliases; expose the canonical
        # names explicitly so quality/policy groups can never fall back to an
        # "unknown" bucket.
        row["policy"] = str(row["policy_name"])
        row["quality"] = str(row["quality_class"])
        row["timestep"] = int(row["t"])
        rows.append(row)
    return rows


def _evaluate(
    target: np.ndarray,
    prediction: np.ndarray,
    dataset: Round5TransitionDataset,
    metadata: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    base_fields: Mapping[str, Any],
) -> ActionEvaluation:
    evaluation = config["evaluation"]
    bands = [float(value) for value in evaluation["action_magnitude_bands_m"]]
    # Sequence targets are float32, so a mathematically unit-norm action may
    # round a few ulps above the configured physical radius.  Expand only the
    # final reporting edge (never a target or model value) to cover that benign
    # representation error.
    observed_maximum = float(
        np.max(np.linalg.norm(np.asarray(target, dtype=np.float64), axis=1))
        * dataset.maximum_step_distance_m
    )
    bands[-1] = max(bands[-1], observed_maximum + 1.0e-6)
    return evaluate_action_predictions(
        target,
        prediction,
        dataset.maximum_step_distance_m,
        metadata=metadata,
        near_zero_epsilon_m=float(evaluation["action_near_zero_epsilon_m"]),
        magnitude_bands_m=bands,
        base_fields=base_fields,
    )


def _metric_rows(evaluation: ActionEvaluation) -> List[Dict[str, Any]]:
    return [dict(evaluation.raw_metrics), dict(evaluation.projected_metrics)]


def _sequence_manifest_row(
    sequence: Round5SequenceSplit,
    transitions: Any,
    dataset: Round5TransitionDataset,
    cache: FrozenWaypointCache,
) -> Dict[str, Any]:
    result = sequence.summary()
    result.update(
        {
            "episode_count": len(set(map(str, transitions.episode_id.tolist()))),
            "policy_distribution": {
                str(value): int(np.sum(transitions.policy_name == value))
                for value in sorted(set(transitions.policy_name.tolist()))
            },
            "buoy_scale_distribution": {
                str(value): int(np.sum(transitions.num_buoys == value))
                for value in sorted(set(int(v) for v in transitions.num_buoys.tolist()))
            },
            "source_dataset_hashes": dict(dataset.source_hashes),
            "waypoint_checkpoint_sha256": cache.checkpoint_sha256,
            "split_crossing": False,
            "episode_crossing": False,
            "history_action_input": False,
        }
    )
    return result


def _summary_columns(prefix: str, values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{prefix} statistics require finite non-empty values")
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q25": float(np.percentile(array, 25)),
        f"{prefix}_q75": float(np.percentile(array, 75)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def _dataset_policy_statistics(
    dataset: Round5TransitionDataset,
    waypoint_cache: FrozenWaypointCache,
    *,
    zero_epsilon_m: float,
) -> List[Dict[str, Any]]:
    """Audit policy heterogeneity before any Round-5 optimizer is created."""

    from wt.maritime.dataset_v1 import load_dataset_v1

    episodes = load_dataset_v1(dataset.source_dir, split=None, verify_checksums=True)
    returns = {str(episode.episode_id): float(episode.episode_return) for episode in episodes}
    rows: List[Dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        split = dataset.split(split_name)
        predicted_waypoint_m = waypoint_cache.split(split_name).predicted_waypoints_m
        waypoint_error_m = np.linalg.norm(
            predicted_waypoint_m - split.oracle_waypoints_m, axis=1
        )
        for policy in sorted(set(map(str, split.policy_name.tolist()))):
            selected = split.policy_name == policy
            episode_ids = sorted(set(map(str, split.episode_id[selected].tolist())))
            action_m = np.asarray(split.actions_m[selected], dtype=np.float64)
            magnitudes = np.linalg.norm(action_m, axis=1)
            distances = []
            episode_lengths = []
            for episode_id in episode_ids:
                episode_rows = split.episode_id == episode_id
                distances.append(
                    float(np.sum(np.linalg.norm(split.actions_m[episode_rows], axis=1)))
                )
                episode_lengths.append(int(np.sum(episode_rows)))
            x = action_m[:, 0]
            y = action_m[:, 1]
            epsilon = float(zero_epsilon_m)
            row: Dict[str, Any] = {
                "split": split_name,
                "policy_name": policy,
                "episode_count": len(episode_ids),
                "transition_count": int(np.sum(selected)),
                "zero_action_ratio": float(np.mean(magnitudes <= epsilon)),
                "x_negative_ratio": float(np.mean(x < -epsilon)),
                "x_near_zero_ratio": float(np.mean(np.abs(x) <= epsilon)),
                "x_positive_ratio": float(np.mean(x > epsilon)),
                "y_negative_ratio": float(np.mean(y < -epsilon)),
                "y_near_zero_ratio": float(np.mean(np.abs(y) <= epsilon)),
                "y_positive_ratio": float(np.mean(y > epsilon)),
                "average_voyage_distance_m": float(np.mean(distances)),
                "average_episode_length": float(np.mean(episode_lengths)),
                "average_episode_return": float(np.mean([returns[value] for value in episode_ids])),
                "waypoint_error_definition": "predicted_H30_endpoint_delta_vs_terminal_clamped_oracle",
                "computed_before_policy_training": True,
                "policy_id_used_as_model_input": False,
            }
            row.update(_summary_columns("action_magnitude_m", magnitudes))
            row.update(_summary_columns("waypoint_error_m", waypoint_error_m[selected]))
            rows.append(row)
    return rows


def _selection_rows(training_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select context and recommended seed using validation metrics only."""

    output: List[Dict[str, Any]] = []
    deployable_transformers = (
        "wt_transformer",
        "direct_goal_transformer",
        "state_only_transformer",
    )
    for model_type in deployable_transformers:
        members = [row for row in training_rows if row["model_type"] == model_type]
        contexts = sorted(set(int(row["context_length"]) for row in members))
        means: Dict[int, float] = {}
        for context in contexts:
            values = np.asarray(
                [
                    float(row["best_validation_raw_action_rmse_m"])
                    for row in members
                    if int(row["context_length"]) == context
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            means[context] = float(np.mean(values))
            output.append(
                {
                    "model_type": model_type,
                    "context_length": context,
                    "validation_seed_count": int(values.size),
                    "validation_action_rmse_m_mean": float(np.mean(values)),
                    "validation_action_rmse_m_std": float(np.std(values)),
                    "validation_action_rmse_m_median": float(np.median(values)),
                    "validation_action_rmse_m_q25": float(np.percentile(values, 25)),
                    "validation_action_rmse_m_q75": float(np.percentile(values, 75)),
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "selected_context": False,
                }
            )
        if means:
            chosen = min(means, key=lambda context: (means[context], context))
            for row in output:
                if row["model_type"] == model_type and int(row["context_length"]) == chosen:
                    row["selected_context"] = True

    wt_rows = [row for row in output if row["model_type"] == "wt_transformer"]
    if not wt_rows:
        # Smoke still has one WT validation configuration.
        members = [row for row in training_rows if row["model_type"] == "wt_transformer"]
        if not members:
            raise RuntimeError("no WT validation result is available for selection")
        contexts = sorted(set(int(row["context_length"]) for row in members))
        for context in contexts:
            values = np.asarray(
                [float(row["best_validation_raw_action_rmse_m"]) for row in members if int(row["context_length"]) == context]
            )
            output.append(
                {
                    "model_type": "wt_transformer",
                    "context_length": context,
                    "validation_seed_count": int(values.size),
                    "validation_action_rmse_m_mean": float(np.mean(values)),
                    "validation_action_rmse_m_std": float(np.std(values)),
                    "validation_action_rmse_m_median": float(np.median(values)),
                    "validation_action_rmse_m_q25": float(np.percentile(values, 25)),
                    "validation_action_rmse_m_q75": float(np.percentile(values, 75)),
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "selected_context": context == contexts[0],
                }
            )
        wt_rows = [row for row in output if row["model_type"] == "wt_transformer"]
    selected_context = int(next(row["context_length"] for row in wt_rows if row["selected_context"]))
    candidates = [
        row
        for row in training_rows
        if row["model_type"] == "wt_transformer"
        and int(row["context_length"]) == selected_context
    ]
    recommended = min(
        candidates,
        key=lambda row: (float(row["best_validation_raw_action_rmse_m"]), int(row["seed"])),
    )
    selection = {
        "schema_version": "round05_selected_transformer_config_v1",
        "selection_frozen": True,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "model_type": "wt_transformer",
        "context_length": selected_context,
        "training_seed": int(recommended["seed"]),
        "best_epoch": int(recommended["best_epoch"]),
        "validation_action_rmse_m": float(recommended["best_validation_raw_action_rmse_m"]),
        "checkpoint_path": str(recommended["best_checkpoint"]),
        "checkpoint_sha256": str(recommended["best_checkpoint_sha256"]),
        "hindsight_goal_limitation": True,
        "deployable_online_goal_mechanism_established": False,
        "frozen_at": _now(),
    }
    return output, selection


def _completed_training_result(path: Path) -> Optional[TrainingResult]:
    if not path.is_file():
        return None
    raw = _read_json(path)
    return TrainingResult(
        best_epoch=int(raw["best_epoch"]),
        best_validation_raw_action_rmse_m=float(raw["best_validation_raw_action_rmse_m"]),
        last_epoch=int(raw["last_epoch"]),
        stopped_early=bool(raw["stopped_early"]),
        best_checkpoint=str(raw["best_checkpoint"]),
        last_checkpoint=str(raw["last_checkpoint"]),
        history=list(raw.get("history", [])),
        sampler_counts_by_epoch=list(raw.get("sampler_counts_by_epoch", [])),
    )


def _assert_waypoint_still_frozen(
    predictor: FrozenWaypointPredictor, checkpoint_sha256: str
) -> None:
    if predictor.model.training:
        raise RuntimeError("frozen waypoint network left eval mode")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in predictor.model.parameters()):
        raise RuntimeError("frozen waypoint parameters acquired gradients")
    if _sha256(predictor.checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("locked Round-4 waypoint checkpoint changed during policy training")
    if _module_sha256(predictor.model) != predictor.parameter_sha256.upper():
        raise RuntimeError("frozen waypoint parameters changed during policy training")


def run_bounded_interface_smoke(
    *,
    env_runner: Any,
    state_callback: Callable[[Any, Mapping[str, Any], int], Any],
    waypoint_callback: Callable[[Any, Mapping[str, Any], int], Sequence[float]],
    action_callback: Callable[[Tuple[Mapping[str, Any], ...], Any, np.ndarray, int], Sequence[float]],
    max_step_distance_m: float,
    output_dir: str | Path,
    steps: int,
    context_length: int,
    reset_kwargs: Optional[Mapping[str, Any]] = None,
    environment_action_callback: Optional[Callable[[np.ndarray, np.ndarray, Any, np.ndarray, int], Any]] = None,
) -> Dict[str, Any]:
    """Run and persist the generic, disposable 1--5 step interface check."""

    result = run_interface_smoke(
        env_runner,
        state_callback,
        waypoint_callback,
        action_callback,
        max_step_distance_m,
        steps=steps,
        context_length=context_length,
        reset_kwargs=reset_kwargs,
        environment_action_callback=environment_action_callback,
    )
    if not result["closed_loop_interface_smoke_performed"]:
        raise RuntimeError("interface smoke helper did not execute")
    if result["closed_loop_evaluation_performed"] or result["formal_closed_loop_evaluation"]:
        raise RuntimeError("bounded interface smoke was mislabeled as evaluation")
    root = Path(output_dir)
    _write_json(root / "interface_smoke.json", result)
    _write_csv(root / "interface_smoke.csv", result["rows"])
    return result


def _default_interface_smoke_factory(
    *,
    dataset: Round5TransitionDataset,
    waypoint_cache: FrozenWaypointCache,
    predictor: FrozenWaypointPredictor,
    selection: Mapping[str, Any],
    selected_model: torch.nn.Module,
    output_dir: Path,
    steps: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reconstruct one disposable test case and exercise only 1--5 steps.

    The achieved terminal goal is copied from the recorded test episode solely
    for this engineering interface check.  No return is accumulated, no full
    episode is run, and the result is never compared with a baseline.
    """

    from dataclasses import replace

    from wt.maritime.config import MaritimeEnvConfig
    from wt.maritime.env import MaritimeEnv
    from wt.maritime.waypoint_samples import (
        _extract_waypoint,
        _pad_nodes,
        _pad_observation,
    )

    source = dataset.source_dir
    run_config = _read_json(source / "run_config.json")
    base_raw = run_config.get("base_environment_config")
    cases = run_config.get("case_metadata")
    if not isinstance(base_raw, Mapping) or not isinstance(cases, Mapping):
        raise ValueError("Round-3 run config lacks environment reconstruction metadata")

    # Select the first complete test episode deterministically and bind every
    # callback to its recorded hindsight goal.  The same case seed/config is
    # enough to reproduce the pre-action observation exactly.
    first_index = 0
    episode_id = str(dataset.test.episode_id[first_index])
    if int(dataset.test.t[first_index]) != 0:
        raise RuntimeError("test transition order does not begin at episode timestep zero")
    marker = "__policy-"
    if marker not in episode_id:
        raise ValueError("test episode ID cannot be mapped to a Round-3 case")
    case_id = episode_id.split(marker, 1)[0]
    case = cases.get(case_id)
    if not isinstance(case, Mapping):
        raise ValueError(f"missing Round-3 case metadata for {case_id}")

    trace_index = _read_json(source / "arrival_traces.json").get("trace_index")
    if not isinstance(trace_index, Mapping):
        raise ValueError("arrival trace index is missing")
    keys = [key for key, value in trace_index.items() if str(value) == case_id]
    if len(keys) != 1:
        raise ValueError("test case does not map to exactly one arrival trace")
    with np.load(str(source / "arrival_traces.npz"), allow_pickle=False) as archive:
        trace = np.asarray(archive[str(keys[0])], dtype=np.float64).copy()

    base = MaritimeEnvConfig(**dict(base_raw))
    environment_config = replace(
        base,
        task_semantics_version="v1",
        num_buoys=int(case["num_buoys"]),
        episode_horizon=int(case["horizon"]),
        randomize_buoys=True,
        initial_queue_bits_by_buoy=tuple(float(value) for value in case["initial_queue_bits"]),
        arrival_mode="trace",
        arrival_trace_bits=tuple(tuple(float(value) for value in row) for row in trace),
        arrival_trace_id=str(case["arrival_trace_id"]),
        target_task_completion_ratio=max(0.985, float(base.target_task_completion_ratio)),
        maximum_allowed_residual_queue_ratio=float(base.maximum_allowed_residual_queue_ratio),
    )
    env = MaritimeEnv(environment_config)
    max_nodes = dataset.node_mask_dim
    expected_initial_state = np.asarray(dataset.test.states[first_index], dtype=np.float32)
    hindsight_goal = np.asarray(dataset.test.goals[first_index], dtype=np.float32)
    context = int(selection["context_length"])
    initial_state_verified = {"value": False}

    def state_callback(observation: Any, info: Mapping[str, Any], step_index: int) -> np.ndarray:
        # ``run_interface_smoke`` passes a mutable dictionary copy.  Recording
        # this public, configured-horizon feature lets the subsequent token use
        # exactly the same remaining-time definition as offline training.
        if isinstance(info, MutableMapping):
            info["remaining_time"] = float(info["remaining_horizon_fraction"])
        raw_observation = np.asarray(observation, dtype=np.float32)
        node_mask = np.asarray(info["valid_node_mask"], dtype=np.bool_)
        current_waypoint, _ = _extract_waypoint(
            raw_observation, node_mask, int(env.num_buoys), max_nodes
        )
        padded_observation, _ = _pad_observation(
            raw_observation, node_mask, int(env.num_buoys), max_nodes
        )
        padded_nodes = _pad_nodes(node_mask, max_nodes)
        state = np.concatenate(
            (current_waypoint, padded_observation, padded_nodes.astype(np.float32))
        ).astype(np.float32, copy=False)
        if state.shape != (dataset.state_dim,):
            raise RuntimeError("interface state adapter changed schema dimension")
        if step_index == 0:
            if not np.array_equal(state, expected_initial_state):
                maximum = float(np.max(np.abs(state - expected_initial_state)))
                raise RuntimeError(
                    f"disposable test environment did not reproduce recorded state (max diff={maximum})"
                )
            initial_state_verified["value"] = True
        return state

    def waypoint_callback(state: Any, _info: Mapping[str, Any], _step_index: int) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        node_mask = state_array[-max_nodes:].astype(np.float32, copy=False)
        waypoint_input = np.concatenate((state_array, hindsight_goal, node_mask)).astype(
            np.float32, copy=False
        )
        if waypoint_input.shape != (predictor.input_dim,):
            raise RuntimeError("interface frozen-waypoint input dimension changed")
        predictor.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(waypoint_input[None, :]).to(predictor.device)
            value = predictor.model(tensor)[0].detach().to("cpu", dtype=torch.float32).numpy()
        if value.shape != (dataset.action_dim,) or not np.all(np.isfinite(value)):
            raise RuntimeError("interface waypoint prediction is invalid")
        return value

    model_device = next(selected_model.parameters()).device

    def action_callback(
        history: Tuple[Mapping[str, Any], ...],
        _state: Any,
        _waypoint: np.ndarray,
        _step_index: int,
    ) -> np.ndarray:
        if len(history) > context:
            raise RuntimeError("interface history exceeded selected context")
        states = np.zeros((1, context, dataset.state_dim), dtype=np.float32)
        nodes = np.zeros((1, context, max_nodes), dtype=np.float32)
        conditions = np.zeros((1, context, dataset.action_dim), dtype=np.float32)
        remaining = np.zeros((1, context, 1), dtype=np.float32)
        attention = np.zeros((1, context), dtype=np.bool_)
        start = context - len(history)
        for offset, token in enumerate(history):
            destination = start + offset
            token_state = np.asarray(token["state"], dtype=np.float32)
            states[0, destination] = token_state
            nodes[0, destination] = token_state[-max_nodes:]
            conditions[0, destination] = np.asarray(
                token["predicted_waypoint"], dtype=np.float32
            )
            remaining[0, destination, 0] = float(token["remaining_time"])
            attention[0, destination] = True
        selected_model.eval()
        with torch.no_grad():
            action = selected_model(
                states=torch.from_numpy(states).to(model_device),
                node_masks=torch.from_numpy(nodes).to(model_device),
                conditions=torch.from_numpy(conditions).to(model_device),
                remaining_time=torch.from_numpy(remaining).to(model_device),
                attention_mask=torch.from_numpy(attention).to(model_device),
            )[0, -1]
        result = action.detach().to("cpu", dtype=torch.float32).numpy()
        if result.shape != (dataset.action_dim,) or not np.all(np.isfinite(result)):
            raise RuntimeError("interface policy action is invalid")
        return result

    result = run_bounded_interface_smoke(
        env_runner=env,
        state_callback=state_callback,
        waypoint_callback=waypoint_callback,
        action_callback=action_callback,
        max_step_distance_m=dataset.maximum_step_distance_m,
        output_dir=output_dir,
        steps=min(5, max(1, int(steps))),
        context_length=context,
        reset_kwargs={"seed": int(dataset.test.seed[first_index])},
        # MaritimeEnv.step consumes normalized actions, not metres.
        environment_action_callback=(
            lambda _projected_m, projected_normalized, _state, _waypoint, _step: projected_normalized
        ),
    )
    result.update(
        {
            "disposable_environment_copy": True,
            "source_test_episode_id": episode_id,
            "source_test_case_id": case_id,
            "recorded_initial_state_reproduced_exactly": bool(initial_state_verified["value"]),
            "goal_source": "achieved_terminal_goal",
            "hindsight_goal_known_online_claimed": False,
            "complete_episode_executed": False,
            "episode_return_computed": False,
            "baseline_ranking_performed": False,
            "waypoint_recomputed_after_each_environment_step": True,
        }
    )
    _write_json(output_dir / "interface_smoke.json", result)
    return result


def _initial_manifest(
    *,
    run_id: str,
    mode: str,
    config_path: Path,
    matrix: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "round": 5,
        "mode": mode,
        "status": "in_progress",
        "started_at": _now(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "formal_expected_run_count": FORMAL_RUN_COUNT,
        "scheduled_run_count": len(matrix),
        "completed_training_run_count": 0,
        "completed_test_run_count": 0,
        "validation_selection_frozen": False,
        "test_sequences_accessed": False,
        "test_used_for_selection": False,
        "training_performed": False,
        "transformer_training_performed": False,
        "policy_training_performed": False,
        "closed_loop_interface_smoke_performed": False,
        "formal_closed_loop_evaluation": False,
        "resume_supported": True,
        "run_matrix": [dict(row) for row in matrix],
        **STATUS_FIELDS,
    }


def _run_round5_impl(
    *,
    config_path: str | Path,
    mode: str,
    run_id: str,
    device: Optional[str],
    resume: bool,
    max_epochs: Optional[int],
    patience: Optional[int],
    interface_smoke_factory: Optional[Callable[..., Mapping[str, Any]]],
) -> Path:
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be 'smoke' or 'formal'")
    root = repository_root()
    resolved_config, config = load_round5_config(config_path)
    matrix = build_smoke_run_matrix(config) if mode == "smoke" else build_formal_run_matrix(config)
    output_root = _resolve_path(config["output_root"], root)
    figure_root = _resolve_path(config["figure_root"], root)
    output_dir = (output_root / run_id).resolve()
    figures_dir = (figure_root / run_id).resolve()
    round_root = (root / "round_05").resolve()
    try:
        output_dir.relative_to(round_root)
        figures_dir.relative_to(round_root)
    except ValueError as error:
        raise ValueError("all Round-5 outputs must remain under round_05") from error
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Round-5 run already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("run_id") != run_id or manifest.get("mode") != mode:
            raise ValueError("resume run manifest does not match requested run")
        if manifest.get("config_sha256") != _sha256(resolved_config):
            raise ValueError("resume config differs from the original Round-5 run")
        if manifest.get("status") in {"completed", "smoke_completed"}:
            # A completed run is immutable.  In particular, returning here
            # prevents a second access to its test-once evaluation.
            stale_failure_fields = (
                "failed_at",
                "failed_stage",
                "error_type",
                "error_message",
            )
            if any(key in manifest for key in stale_failure_fields):
                for key in stale_failure_fields:
                    manifest.pop(key, None)
                _write_json(manifest_path, manifest)
            return output_dir
    else:
        manifest = _initial_manifest(
            run_id=run_id, mode=mode, config_path=resolved_config, matrix=matrix
        )
        _write_json(manifest_path, manifest)

    started = time.monotonic()
    dataset_dir = _resolve_path(config["dataset_root"], root)
    checkpoint = _resolve_path(config["waypoint_checkpoint"], root)
    dataset = load_round5_transition_dataset(
        dataset_dir,
        waypoint_horizon=int(config["waypoint_horizon"]),
        goal_source=str(config["goal_source"]),
        verify_round3_lock=True,
    )
    if dataset.action_dim != 2:
        raise ValueError("Round-5 scope is exactly the two-dimensional displacement action")
    predictor = load_frozen_waypoint_predictor(
        checkpoint,
        dataset,
        expected_sha256=str(config["waypoint_checkpoint_sha256"]),
        device=str(device or config.get("device", "auto")),
    )
    cache_dir = output_dir / "frozen_waypoints"
    if resume and (cache_dir / "cache_manifest.json").is_file():
        waypoint_cache = load_frozen_waypoint_cache(
            cache_dir,
            dataset,
            expected_checkpoint_sha256=str(config["waypoint_checkpoint_sha256"]),
        )
    else:
        waypoint_cache = build_frozen_waypoint_cache(dataset, predictor)
        materialize_frozen_waypoint_cache(waypoint_cache, cache_dir)
    frozen_manifest = {
        "schema_version": "round05_frozen_waypoint_manifest_v1",
        "summary": waypoint_cache.summary(),
        "provenance": dict(waypoint_cache.provenance),
        "materialized_cache_manifest": str(cache_dir / "cache_manifest.json"),
        "materialized_cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
        "waypoint_checkpoint_frozen": True,
        "waypoint_training_performed": False,
    }
    _write_json(output_dir / "frozen_waypoint_manifest.json", frozen_manifest)
    _assert_waypoint_still_frozen(predictor, waypoint_cache.checkpoint_sha256)
    # This table is deliberately materialized before the policy trainer or
    # optimizer exists.  It documents the collection-policy heterogeneity that
    # motivates balanced sampling without exposing policy IDs to a model.
    policy_statistics = _dataset_policy_statistics(
        dataset,
        waypoint_cache,
        zero_epsilon_m=float(config["evaluation"]["action_near_zero_epsilon_m"]),
    )
    _write_csv(output_dir / "dataset_policy_statistics.csv", policy_statistics)

    for stale in ("failed_at", "failed_stage", "error_type", "error_message"):
        manifest.pop(stale, None)
    manifest.update(
        {
            "dataset": dataset.summary(),
            "frozen_waypoint": waypoint_cache.summary(),
            "waypoint_checkpoint_path": str(checkpoint),
            "waypoint_checkpoint_sha256": waypoint_cache.checkpoint_sha256,
            "device_requested": str(device or config.get("device", "auto")),
            "status": "training_in_progress",
        }
    )
    _write_json(manifest_path, manifest)

    sequence_cache: Dict[Tuple[str, int, str], Round5SequenceSplit] = {}
    sequence_rows: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    def sequence_for(split_name: str, context: int, condition: str) -> Round5SequenceSplit:
        key = (split_name, int(context), condition)
        if key not in sequence_cache:
            transition = dataset.split(split_name)
            cache_split = waypoint_cache.split(split_name) if condition == "predicted_waypoint" else None
            sequence = build_round5_sequence_split(
                transition,
                context_length=int(context),
                condition_source=condition,
                waypoint_cache_split=cache_split,
            )
            sequence_cache[key] = sequence
            sequence_rows[key] = _sequence_manifest_row(
                sequence, transition, dataset, waypoint_cache
            )
        return sequence_cache[key]

    training_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    grouped_rows: List[Dict[str, Any]] = []
    sampler_rows: List[Dict[str, Any]] = []
    parameter_rows: List[Dict[str, Any]] = []
    trained_models: Dict[str, Tuple[torch.nn.Module, Any, TrainingConfig]] = {}

    # Test split is deliberately absent from this entire loop.
    for index, row in enumerate(matrix):
        run_name = _run_name(row)
        run_dir = output_dir / "runs" / run_name
        train_sequence = sequence_for("train", int(row["context_length"]), str(row["condition_source"]))
        validation_sequence = sequence_for("validation", int(row["context_length"]), str(row["condition_source"]))
        model_config = _model_config(row, train_sequence, config)
        training_config = _training_config(
            row,
            dataset,
            config,
            mode=mode,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
        )
        model = build_seeded_policy(model_config, training_config)
        provenance = {
            "round": 5,
            "run_id": run_id,
            "run_name": run_name,
            "model_type": row["model_type"],
            "condition_source": row["condition_source"],
            "context_length": row["context_length"],
            "seed": row["seed"],
            "source_run_id": dataset.source_run_id,
            "source_hashes": dict(dataset.source_hashes),
            "waypoint_checkpoint_sha256": waypoint_cache.checkpoint_sha256,
            "waypoint_checkpoint_frozen": True,
            "test_used_for_selection": False,
            "action_labels_used_as_model_input": False,
            "history_action_input": False,
        }
        trainer = Round5Trainer(model, training_config, run_dir, provenance=provenance)
        result_path = run_dir / "training_result.json"
        result = _completed_training_result(result_path) if resume else None
        if result is None:
            train_loader = make_policy_balanced_loader(train_sequence, training_config)
            validation_loader = _loader(validation_sequence, training_config.batch_size)
            resume_path = trainer.last_path if resume and trainer.last_path.is_file() else None
            result = trainer.fit(train_loader, validation_loader, resume_from=resume_path)
            _write_json(result_path, result.to_dict())
        else:
            load_policy_checkpoint(result.best_checkpoint, model, map_location=trainer.device)

        # Strict reload into a fresh architecture before any reported metric.
        first_prediction = predict_actions(
            model,
            validation_sequence,
            device=trainer.device,
            batch_size=training_config.batch_size,
        )
        reloaded = build_policy(model_config)
        checkpoint_payload = load_policy_checkpoint(
            result.best_checkpoint, reloaded, map_location=trainer.device
        )
        second_prediction = predict_actions(
            reloaded,
            validation_sequence,
            device=trainer.device,
            batch_size=training_config.batch_size,
        )
        if not np.array_equal(first_prediction, second_prediction):
            raise RuntimeError(f"checkpoint reload changed validation predictions: {run_name}")
        base = {
            "run_name": run_name,
            "model_type": row["model_type"],
            "context_length": int(row["context_length"]),
            "seed": int(row["seed"]),
            "split": "validation",
            "deployable": bool(row["deployable"]),
            "non_deployable_diagnostic": bool(row["non_deployable_diagnostic"]),
            "condition_source": row["condition_source"],
        }
        validation_evaluation = _evaluate(
            _target_anchor(validation_sequence),
            second_prediction,
            dataset,
            _metadata_rows(dataset.validation),
            config,
            base,
        )
        validation_rows.extend(_metric_rows(validation_evaluation))
        grouped_rows.extend(dict(value) for value in validation_evaluation.grouped_metrics)
        best_checkpoint_path = Path(result.best_checkpoint).resolve()
        training_row = {
            **base,
            "split": "train_validation",
            "best_epoch": int(result.best_epoch),
            "last_epoch": int(result.last_epoch),
            "stopped_early": bool(result.stopped_early),
            "best_validation_raw_action_rmse_m": float(result.best_validation_raw_action_rmse_m),
            "best_checkpoint": str(best_checkpoint_path),
            "best_checkpoint_sha256": _sha256(best_checkpoint_path),
            "last_checkpoint": str(Path(result.last_checkpoint).resolve()),
            "history_file": "transformer_training_history.csv",
            "parameter_count": count_parameters(reloaded),
            "checkpoint_reload_prediction_identical": True,
            "validation_prediction_sha256": _array_sha256(second_prediction),
            "selection_split": "validation",
            "test_used_for_selection": False,
            "training_performed": True,
        }
        training_rows.append(training_row)
        parameter_rows.append(
            {
                **base,
                "split": "architecture",
                "parameter_count": count_parameters(reloaded),
                "backbone_d_model": int(config["transformer"]["d_model"]) if row["model_type"] in _TRANSFORMERS else 0,
                "backbone_layers": int(config["transformer"]["n_layer"]) if row["model_type"] in _TRANSFORMERS else 0,
                "backbone_heads": int(config["transformer"]["n_head"]) if row["model_type"] in _TRANSFORMERS else 0,
                "model_config": asdict(model_config),
            }
        )
        for history in result.history:
            history_rows.append(
                {
                    "run_name": run_name,
                    "model_type": row["model_type"],
                    "context_length": int(row["context_length"]),
                    "seed": int(row["seed"]),
                    "epoch": int(history["epoch"]),
                    "train_loss": float(history["train_normalized_action_mse"]),
                    "validation_loss": float(history["validation_normalized_action_mse"]),
                    "validation_action_rmse_m": float(history["validation_raw_action_rmse_m"]),
                    "selected_as_best": bool(history["selected_as_best"]),
                }
            )
            counts = history.get("train_sampler_counts", {})
            for policy in sorted(counts):
                sampler_rows.append(
                    {
                        "run_name": run_name,
                        "model_type": row["model_type"],
                        "context_length": int(row["context_length"]),
                        "seed": int(row["seed"]),
                        "epoch": int(history["epoch"]),
                        "policy_name": policy,
                        "sample_count": int(counts[policy]),
                        "sampler": "policy_balanced",
                        "sampler_seed": int(row["seed"]) + int(history["epoch"]),
                    }
                )
        trained_models[run_name] = (reloaded, model_config, training_config)
        _assert_waypoint_still_frozen(predictor, waypoint_cache.checkpoint_sha256)
        manifest["completed_training_run_count"] = index + 1
        manifest["training_performed"] = True
        manifest["transformer_training_performed"] = any(
            item["model_type"] in _TRANSFORMERS for item in matrix[: index + 1]
        )
        manifest["policy_training_performed"] = True
        _write_json(manifest_path, manifest)

    _write_csv(output_dir / "transformer_training_summary.csv", training_rows)
    _write_csv(output_dir / "transformer_training_history.csv", history_rows)
    _write_csv(output_dir / "transformer_validation_metrics.csv", validation_rows)
    _write_csv(output_dir / "policy_sampling_audit.csv", sampler_rows)
    _write_csv(output_dir / "model_parameter_counts.csv", parameter_rows)

    context_rows, computed_selection = _selection_rows(training_rows)
    selection_path = output_dir / "selected_transformer_config.json"
    if resume and selection_path.is_file():
        selection = _read_json(selection_path)
        for key in (
            "model_type",
            "context_length",
            "training_seed",
            "best_epoch",
            "checkpoint_sha256",
        ):
            if selection.get(key) != computed_selection.get(key):
                raise RuntimeError("frozen validation selection changed during resume")
        if not math.isclose(
            float(selection["validation_action_rmse_m"]),
            float(computed_selection["validation_action_rmse_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("frozen validation metric changed during resume")
    else:
        selection = computed_selection
    selected_run_name = _run_name(
        {
            "model_type": selection["model_type"],
            "context_length": selection["context_length"],
            "seed": selection["training_seed"],
        }
    )
    selected_model_config = trained_models[selected_run_name][1]
    selection_enrichment = {
            "policy_model_config": asdict(selected_model_config),
            "policy_input_schema": {
                "state_dim": dataset.state_dim,
                "node_mask_dim": dataset.node_mask_dim,
                "condition": "predicted_waypoint",
                "condition_dim": waypoint_cache.output_dim,
                "remaining_time_dim": 1,
                "action_dim": dataset.action_dim,
                "history_action_input": False,
                "global_goal_direct_input": False,
                "oracle_waypoint_input": False,
            },
            "waypoint_horizon": dataset.waypoint_horizon,
            "waypoint_checkpoint_path": str(waypoint_cache.checkpoint_path),
            "waypoint_checkpoint_sha256": waypoint_cache.checkpoint_sha256,
            "waypoint_checkpoint_frozen": True,
            "goal_source": str(config["goal_source"]),
            "offline_hindsight_action_prediction": True,
            "action_normalization_distance_m": dataset.maximum_step_distance_m,
            "action_output_activation": "tanh",
        }
    if resume and selection_path.is_file():
        for key, value in selection_enrichment.items():
            if _json_value(selection.get(key)) != _json_value(value):
                raise RuntimeError(f"frozen selected transformer field changed: {key}")
    else:
        selection.update(selection_enrichment)
    _write_csv(output_dir / "context_length_selection.csv", context_rows)
    if not selection_path.exists():
        _write_json(selection_path, selection)
    manifest.update(
        {
            "status": "validation_selection_frozen",
            "validation_selection_frozen": True,
            "selection_frozen_at": selection["frozen_at"],
            "selected_transformer": selection,
            "test_sequences_accessed": False,
            "test_used_for_selection": False,
        }
    )
    _write_json(manifest_path, manifest)

    # Test-once starts only after the immutable validation selection above.
    test_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    test_predictions_by_run: Dict[str, Tuple[np.ndarray, ActionEvaluation]] = {}
    manifest["status"] = "test_once_in_progress"
    manifest["test_sequences_accessed"] = True
    manifest["test_accessed_after_selection_frozen"] = True
    _write_json(manifest_path, manifest)
    for index, row in enumerate(matrix):
        run_name = _run_name(row)
        run_dir = output_dir / "runs" / run_name
        test_record_path = run_dir / "test_once_evaluation.json"
        test_prediction_path = run_dir / "test_once_predictions.npz"
        test_sequence = sequence_for(
            "test", int(row["context_length"]), str(row["condition_source"])
        )
        model, _, training_config = trained_models[run_name]
        base = {
            "run_name": run_name,
            "model_type": row["model_type"],
            "context_length": int(row["context_length"]),
            "seed": int(row["seed"]),
            "split": "test",
            "deployable": bool(row["deployable"]),
            "non_deployable_diagnostic": bool(row["non_deployable_diagnostic"]),
            "condition_source": row["condition_source"],
            "test_once": True,
            "selection_already_frozen": True,
        }
        if resume and test_record_path.exists() != test_prediction_path.exists():
            raise RuntimeError(
                f"partial test-once artifact for {run_name}; refusing to evaluate test twice"
            )
        if resume and test_record_path.is_file() and test_prediction_path.is_file():
            saved = _read_json(test_record_path)
            if saved.get("schema_version") != "round05_test_once_evaluation_v1":
                raise ValueError("unsupported saved test-once evaluation")
            if saved.get("run_name") != run_name or saved.get("selection_frozen") is not True:
                raise ValueError("saved test-once evaluation has invalid provenance")
            with np.load(str(test_prediction_path), allow_pickle=False) as archive:
                if set(archive.files) != {"prediction"}:
                    raise ValueError("saved test-once prediction has unexpected arrays")
                prediction = np.asarray(archive["prediction"], dtype=np.float32).copy()
            if prediction.shape != (len(test_sequence), dataset.action_dim):
                raise ValueError("saved test-once prediction shape changed")
            if _array_sha256(prediction) != str(saved["prediction_sha256"]):
                raise ValueError("saved test-once prediction checksum changed")
            result = saved.get("evaluation")
            if not isinstance(result, Mapping):
                raise ValueError("saved test-once evaluation payload is missing")
            evaluation = ActionEvaluation(
                raw_metrics=dict(result["raw_metrics"]),
                projected_metrics=dict(result["projected_metrics"]),
                grouped_metrics=tuple(dict(value) for value in result["grouped_metrics"]),
                prediction_rows=tuple(dict(value) for value in result["prediction_rows"]),
            )
        else:
            prediction = predict_actions(
                model,
                test_sequence,
                device=training_config.device,
                batch_size=training_config.batch_size,
            )
            evaluation = _evaluate(
                _target_anchor(test_sequence),
                prediction,
                dataset,
                _metadata_rows(dataset.test),
                config,
                base,
            )
            _write_prediction_npz(test_prediction_path, prediction)
            _write_json(
                test_record_path,
                {
                    "schema_version": "round05_test_once_evaluation_v1",
                    "run_name": run_name,
                    "model_type": row["model_type"],
                    "context_length": int(row["context_length"]),
                    "seed": int(row["seed"]),
                    "selection_frozen": True,
                    "selection_frozen_at": selection["frozen_at"],
                    "test_used_for_selection": False,
                    "test_evaluated_once": True,
                    "prediction_file": test_prediction_path.name,
                    "prediction_sha256": _array_sha256(prediction),
                    "evaluation": evaluation.to_dict(),
                },
            )
        test_rows.extend(_metric_rows(evaluation))
        grouped_rows.extend(dict(value) for value in evaluation.grouped_metrics)
        projection_rows.extend(_metric_rows(evaluation))
        test_predictions_by_run[run_name] = (prediction, evaluation)
        manifest["completed_test_run_count"] = index + 1
        _write_json(manifest_path, manifest)

    _write_csv(output_dir / "transformer_test_metrics.csv", test_rows)
    _write_csv(output_dir / "transformer_grouped_metrics.csv", grouped_rows)
    _write_csv(output_dir / "action_projection_metrics.csv", projection_rows)
    baseline_rows = [
        row
        for row in [*validation_rows, *test_rows]
        if row["model_type"] in {"oracle_waypoint_transformer", "mlp_state_only", "mlp_waypoint"}
    ]
    if not baseline_rows:  # A one-run smoke still needs an honest non-empty audit table.
        baseline_rows = [
            {
                "model_type": "not_executed_in_smoke",
                "context_length": 0,
                "seed": 0,
                "split": "smoke",
                "prediction_space": "not_applicable",
                "sample_count": 0,
                "formal_baselines_scheduled": mode == "formal",
                "test_used_for_selection": False,
            }
        ]
    _write_csv(output_dir / "transformer_baseline_metrics.csv", baseline_rows)

    recommended_name = selected_run_name
    recommended_prediction, recommended_evaluation = test_predictions_by_run[recommended_name]
    prediction_rows = [dict(value) for value in recommended_evaluation.prediction_rows]
    predicted_wp_m = waypoint_cache.test.predicted_waypoints_m
    oracle_wp_m = dataset.test.oracle_waypoints_m
    for index, prediction_row in enumerate(prediction_rows):
        prediction_row["predicted_waypoint_x_m"] = float(predicted_wp_m[index, 0])
        prediction_row["predicted_waypoint_y_m"] = float(predicted_wp_m[index, 1])
        prediction_row["oracle_waypoint_x_m"] = float(oracle_wp_m[index, 0])
        prediction_row["oracle_waypoint_y_m"] = float(oracle_wp_m[index, 1])
        prediction_row["waypoint_error_l2_m"] = float(
            np.linalg.norm(predicted_wp_m[index] - oracle_wp_m[index])
        )
    _write_csv(output_dir / "action_prediction_examples.csv", prediction_rows)
    _write_csv(output_dir / "transformer_test_predictions.csv", prediction_rows)

    replay_metadata = [
        {
            key: row[key]
            for key in (
                "split",
                "sample_id",
                "episode_id",
                "scenario_id",
                "seed",
                "policy",
                "policy_name",
                "num_buoys",
                "quality",
                "quality_class",
                "t",
                "timestep",
                "episode_length",
                "terminated",
                "truncated",
            )
        }
        for row in _metadata_rows(dataset.test)
    ]
    replay = build_replay_diagnostics(
        replay_metadata,
        dataset.test.actions,
        recommended_prediction,
        dataset.maximum_step_distance_m,
        recorded_states=dataset.test.states,
        predicted_waypoints=predicted_wp_m,
        true_waypoints=oracle_wp_m,
        near_zero_epsilon_m=float(config["evaluation"]["action_near_zero_epsilon_m"]),
        base_fields={
            "model_type": selection["model_type"],
            "context_length": selection["context_length"],
            "seed": selection["training_seed"],
            "split": "test",
        },
    )
    for replay_row in replay["rows"]:
        replay_row["waypoint_error_l2_m"] = float(replay_row.get("waypoint_error_l2", 0.0))
    write_replay_diagnostics(
        replay,
        output_dir / "replay_diagnostics.csv",
        output_dir / "replay_diagnostics.json",
    )

    sequence_manifest = {
        "schema_version": "round05_sequence_dataset_manifest_v1",
        "source_run_id": dataset.source_run_id,
        "source_dataset_hashes": dict(dataset.source_hashes),
        "waypoint_checkpoint_sha256": waypoint_cache.checkpoint_sha256,
        "test_sequences_built_after_selection_frozen": True,
        "rows": [sequence_rows[key] for key in sorted(sequence_rows)],
    }
    _write_json(output_dir / "sequence_dataset_manifest.json", sequence_manifest)

    factory = interface_smoke_factory or _default_interface_smoke_factory
    interface_result = dict(
        factory(
            dataset=dataset,
            waypoint_cache=waypoint_cache,
            predictor=predictor,
            selection=selection,
            selected_model=trained_models[recommended_name][0],
            output_dir=output_dir,
            steps=int(config["evaluation"]["interface_smoke_max_steps"]),
            config=config,
        )
    )
    if interface_result.get("formal_closed_loop_evaluation") is not False:
        raise RuntimeError("interface factory returned an unsafe evaluation label")
    if interface_result.get("closed_loop_interface_smoke_performed") is not True:
        raise RuntimeError("Round-5 requires the bounded interface smoke to execute")

    figure_paths = generate_round5_figures(
        output_dir,
        figures_dir,
        overwrite=resume,
        strict=mode == "formal",
    )
    for stale in ("failed_at", "failed_stage", "error_type", "error_message"):
        manifest.pop(stale, None)
    manifest.update(
        {
            "status": "completed" if mode == "formal" else "smoke_completed",
            "completed_at": _now(),
            "elapsed_seconds": float(time.monotonic() - started),
            "training_performed": True,
            "transformer_training_performed": True,
            "policy_training_performed": True,
            "waypoint_training_performed": False,
            "waypoint_checkpoint_frozen": True,
            "closed_loop_interface_smoke_performed": bool(
                interface_result.get("closed_loop_interface_smoke_performed", False)
            ),
            "closed_loop_evaluation_performed": False,
            "formal_closed_loop_evaluation": False,
            "ppo_training_performed": False,
            "hgnn_training_performed": False,
            "test_used_for_selection": False,
            "formal_matrix_complete": mode == "formal" and len(matrix) == FORMAL_RUN_COUNT,
            "replay_mode_performed": True,
            "replay_environment_advanced": False,
            "figure_category_count": 16,
            "figure_files": {name: str(path) for name, path in figure_paths.items()},
            "required_outputs": sorted(
                path.name
                for path in output_dir.iterdir()
                if path.is_file()
            ),
            **STATUS_FIELDS,
        }
    )
    _write_json(manifest_path, manifest)
    _assert_waypoint_still_frozen(predictor, waypoint_cache.checkpoint_sha256)
    return output_dir


def _record_failure(config_path: str | Path, run_id: str, error: BaseException) -> None:
    try:
        _, config = load_round5_config(config_path)
        output_dir = _resolve_path(config["output_root"], repository_root()) / run_id
        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return
        manifest = _read_json(manifest_path)
        if manifest.get("status") in {"completed", "smoke_completed"}:
            return
        manifest.update(
            {
                "status": "incomplete",
                "failed_at": _now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "resume_supported": True,
                "closed_loop_evaluation_performed": False,
                "formal_closed_loop_evaluation": False,
            }
        )
        _write_json(manifest_path, manifest)
    except Exception:
        return


def run_round5(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    mode: str,
    run_id: Optional[str] = None,
    device: Optional[str] = None,
    resume: bool = False,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    interface_smoke_factory: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Path:
    """Run a smoke or the exact formal matrix with auditable failure state."""

    if run_id is None:
        run_id = f"round05_{mode}_{_timestamp()}"
    try:
        return _run_round5_impl(
            config_path=config_path,
            mode=mode,
            run_id=run_id,
            device=device,
            resume=resume,
            max_epochs=max_epochs,
            patience=patience,
            interface_smoke_factory=interface_smoke_factory,
        )
    except BaseException as error:
        if not isinstance(error, FileExistsError):
            _record_failure(config_path, run_id, error)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Round-5 frozen-waypoint causal 2-D offline policy experiments"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    mode = "smoke" if args.smoke else "formal"
    output = run_round5(
        config_path=args.config,
        mode=mode,
        run_id=args.run_id,
        device=args.device,
        resume=args.resume,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(f"round5_status={'smoke_completed' if mode == 'smoke' else 'completed'}")
    print(f"round5_mode={mode}")
    print(f"round5_output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "FORMAL_RUN_COUNT",
    "RUN_MANIFEST_VERSION",
    "build_seeded_policy",
    "build_formal_run_matrix",
    "build_parser",
    "build_smoke_run_matrix",
    "load_round5_config",
    "main",
    "predict_actions",
    "repository_root",
    "run_bounded_interface_smoke",
    "run_round5",
]
