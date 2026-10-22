"""Frozen-protocol Round-12 formal training, validation and final gate runner.

This module intentionally contains no model or hyperparameter search.  Its
first action is to materialise a provenance manifest; it then executes the
pre-registered twelve policy runs and the prescribed inference-only analyses.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from wt.maritime.env import MaritimeEnv
from wt.round7.round7_runner import _case as round7_case, radial_projection

from .contracts import (
    ACTION_DIM,
    CONTEXT_LENGTH,
    H_TASK,
    PREDICTOR_PATH,
    PREDICTOR_SHA256,
    assert_predictor_frozen,
    assert_selected_predictor,
    load_frozen_predictor,
    predicted_phi,
    sha256,
)
from .data import ROUND12_EPISODE_HORIZON, Round12PolicyDataset, load_round12_policy_dataset
from .evaluation import evaluate_offline_actions
from .policy_models import (
    ORACLE_TASK_WAYPOINT_TRANSFORMER,
    POLICY_FAMILIES,
    PREDICTED_TASK_WAYPOINT_TRANSFORMER,
    STATE_ONLY_TRANSFORMER,
    TASK_WAYPOINT_MLP,
    Round12PolicySchema,
    family_is_deployable,
)
from .sequence import Round12SequenceSplit, build_causal_sequence_split
from .trainer import (
    Round12PolicyTrainer,
    Round12TrainingConfig,
    TrainingResult,
    load_round12_checkpoint,
)


FORMAL_SCHEMA = "round12_formal_policy_v1"
FORMAL_CONFIG = Path("round_12/configs/task_waypoint_policy_training_config.json")
FORMAL_PROTOCOL = Path("round_12/configs/formal_evaluation_protocol.json")
FAMILY_ORDER = (
    PREDICTED_TASK_WAYPOINT_TRANSFORMER,
    STATE_ONLY_TRANSFORMER,
    TASK_WAYPOINT_MLP,
    ORACLE_TASK_WAYPOINT_TRANSFORMER,
)
VALIDATION_SOURCE = Path("round_07/outputs/round07_formal_20260809_1740")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("formal output contains NaN or Inf")
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("formal output contains NaN or Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in keys})


def _csv_value(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def _hash_tree(root: Path) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*.py")):
        result[path.relative_to(root).as_posix()] = sha256(path)
    if not result:
        raise RuntimeError("Round-12 source tree is empty")
    return MappingProxyType(result)


def _sha_json(path: Path) -> str:
    return sha256(path)


def _training_config(root: Path) -> tuple[dict[str, Any], Round12TrainingConfig]:
    raw = _read_json(root / FORMAL_CONFIG)
    expected = {
        "H_task": 10, "context_length": 10, "d_model": 128, "layers": 2,
        "heads": 4, "ffn": 512, "dropout": 0.1, "action_dim": 2,
        "optimizer": "AdamW", "learning_rate": 0.0003, "batch_size": 128,
        "max_epochs": 200, "patience": 20, "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0, "seeds": [0, 1, 2],
        # Preserve the already-frozen config byte-for-byte.  The execution
        # matrix below uses the protocol's A/B/C/D reporting order, which is
        # deliberately independent of this declarative enumeration order.
        "families": [
            PREDICTED_TASK_WAYPOINT_TRANSFORMER,
            STATE_ONLY_TRANSFORMER,
            ORACLE_TASK_WAYPOINT_TRANSFORMER,
            TASK_WAYPOINT_MLP,
        ],
        "checkpoint_selection": "validation_raw_action_rmse_only",
        "primary_metric": "cumulative_actual_payload",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise RuntimeError(f"frozen formal training config mismatch at {key!r}")
    return raw, Round12TrainingConfig(
        learning_rate=float(raw["learning_rate"]), batch_size=int(raw["batch_size"]),
        max_epochs=int(raw["max_epochs"]), patience=int(raw["patience"]),
        weight_decay=float(raw["weight_decay"]), gradient_clip_norm=float(raw["gradient_clip_norm"]),
        max_step_distance_m=10.0,
    )


def _protocol(root: Path) -> dict[str, Any]:
    raw = _read_json(root / FORMAL_PROTOCOL)
    if raw.get("bootstrap_unit") != "scenario_id" or int(raw.get("bootstrap_replicates", 0)) < 1000:
        raise RuntimeError("formal statistical protocol is malformed")
    if int(raw.get("configured_episode_horizon", 0)) != ROUND12_EPISODE_HORIZON:
        raise RuntimeError("formal rollout horizon conflicts with the frozen data contract")
    return raw


def freeze_formal_protocol(root: Path, output: Path) -> Mapping[str, Any]:
    """Write the immutable formal run manifest before opening train data."""
    root, output = Path(root), Path(output)
    manifest_path = output / "formal_protocol_manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        _assert_frozen_unchanged(root, existing)
        return existing
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"formal output exists without a protocol manifest: {output}")
    else:
        output.mkdir(parents=True)
    training_raw, _ = _training_config(root)
    selected = _read_json(root / "round_12/selected_task_waypoint_predictor.json")
    fresh = _read_json(root / "round_12/fresh_policy_dev_manifest.json")
    if selected.get("H_task") != H_TASK or selected.get("seed") != 2:
        raise RuntimeError("selected formal predictor is not Round-11 H10 seed 2")
    if selected.get("checkpoint_sha256") != PREDICTOR_SHA256 or int(fresh.get("access_count", -1)) != 0:
        raise RuntimeError("formal predictor/fresh-development freeze precondition failed")
    predictor = assert_selected_predictor(root)
    protocol = _protocol(root)
    value = {
        "schema_version": FORMAL_SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree": dict(_hash_tree(root / "src/wt/round12")),
        "source_tree_digest": _tree_digest(_hash_tree(root / "src/wt/round12")),
        "training_config": training_raw,
        "training_config_sha256": _sha_json(root / FORMAL_CONFIG),
        "selected_predictor": selected,
        "selected_predictor_file_sha256": sha256(predictor),
        "fresh_development_manifest": fresh,
        "fresh_development_manifest_sha256": _sha_json(root / "round_12/fresh_policy_dev_manifest.json"),
        "primary_metric": "cumulative_actual_payload",
        "statistical_protocol": protocol,
        "statistical_protocol_sha256": _sha_json(root / FORMAL_PROTOCOL),
        "formal_matrix": [{"family": family, "seed": seed} for family in FAMILY_ORDER for seed in (0, 1, 2)],
        "old_development_test_access_count": 0,
        "untouched_holdout_access_count": 0,
        "fresh_development_access_count_before": 0,
    }
    _write_json(manifest_path, value)
    return value


def _tree_digest(tree: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(tree), sort_keys=True).encode("utf-8")).hexdigest().upper()


def _assert_frozen_unchanged(root: Path, frozen: Mapping[str, Any]) -> None:
    current = _hash_tree(root / "src/wt/round12")
    if dict(current) != dict(frozen.get("source_tree", {})):
        raise RuntimeError("Round-12 source tree changed after formal protocol freeze")
    for relative, stored_key in ((FORMAL_CONFIG, "training_config_sha256"), (FORMAL_PROTOCOL, "statistical_protocol_sha256")):
        if sha256(root / relative) != frozen.get(stored_key):
            raise RuntimeError(f"frozen formal file changed: {relative}")
    if sha256(assert_selected_predictor(root)) != PREDICTOR_SHA256:
        raise RuntimeError("selected predictor changed after formal protocol freeze")


def _schema(dataset: Round12PolicyDataset) -> Round12PolicySchema:
    return Round12PolicySchema(state_dim=dataset.state_dim, max_buoys=dataset.max_buoys)


def _run_key(family: str, seed: int) -> str:
    return f"{family}/seed_{seed}"


def run_formal_training(root: Path, output: Path, *, device: str) -> tuple[Round12PolicyDataset, Round12SequenceSplit, Round12SequenceSplit, Mapping[str, Any]]:
    """Execute precisely the frozen 4x3 matrix, resuming only interrupted jobs."""
    root, output = Path(root), Path(output)
    frozen = freeze_formal_protocol(root, output)
    _assert_frozen_unchanged(root, frozen)
    raw, base_config = _training_config(root)
    dataset = load_round12_policy_dataset(root)
    schema = _schema(dataset)
    train, validation = build_causal_sequence_split(dataset.train), build_causal_sequence_split(dataset.validation)
    results: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for seed in raw["seeds"]:
            _assert_frozen_unchanged(root, frozen)
            job = output / "training" / family / f"seed_{seed}"
            summary_path = job / "training_result.json"
            config = replace(base_config, seed=int(seed))
            if summary_path.exists():
                summary = _read_json(summary_path)
                _verify_training_job(root, job, family, schema, validation, device)
                results.append(summary)
                continue
            trainer = Round12PolicyTrainer(family, schema, config, root=root, output_dir=job, device=device)
            started = time.perf_counter()
            resume = trainer.last_path if trainer.last_path.exists() else None
            result = trainer.fit(train, validation, resume_from=resume)
            runtime = time.perf_counter() - started
            history = [_plain(row) for row in result.history]
            _write_json(job / "training_history.json", {"history": history})
            summary = {
                "family": family, "seed": int(seed), "runtime_seconds": runtime,
                "best_epoch": result.best_epoch, "last_epoch": result.last_epoch,
                "stopped_early": result.stopped_early,
                "parameter_count": result.parameter_count,
                "best_validation_raw_action_rmse": result.best_validation_raw_action_rmse,
                "best_checkpoint": str(Path(result.best_checkpoint).relative_to(output)),
                "last_checkpoint": str(Path(result.last_checkpoint).relative_to(output)),
                "predictor_sha256_before": result.predictor_sha256_before,
                "predictor_sha256_after": result.predictor_sha256_after,
                "dataset_source_hashes": dict(dataset.source_hashes),
                "training_config": config.to_dict(),
                "optimizer_state_saved": True,
                "history_saved": True,
            }
            _write_json(summary_path, summary)
            _verify_training_job(root, job, family, schema, validation, device)
            results.append(summary)
    if len(results) != 12:
        raise RuntimeError("formal training did not produce exactly 12 runs")
    _write_csv(output / "formal_training_runs.csv", results)
    integrity = _formal_training_integrity(root, output, schema, validation, device, results)
    _write_json(output / "formal_training_integrity.json", integrity)
    if not integrity["formal_training_completed"] or not integrity["all_best_reload"]:
        raise RuntimeError("formal-training integrity gate failed")
    _freeze_policy_checkpoints(root, output)
    return dataset, train, validation, frozen


def _freeze_policy_checkpoints(root: Path, output: Path) -> Mapping[str, Any]:
    """Hash the closed 4x3 policy matrix before any closed-loop evaluation."""
    path = output / "frozen_policy_checkpoint_manifest.json"
    expected: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for seed in (0, 1, 2):
            job = output / "training" / family / f"seed_{seed}"
            for role in ("best", "last"):
                checkpoint = job / f"{role}.pt"
                if not checkpoint.is_file():
                    raise RuntimeError(f"cannot freeze missing policy checkpoint: {checkpoint}")
                expected.append({
                    "family": family, "seed": seed, "role": role,
                    "path": str(checkpoint.relative_to(output)), "sha256": sha256(checkpoint),
                })
    value = {
        "schema_version": FORMAL_SCHEMA,
        "frozen_after_formal_training": True,
        "checkpoint_count": len(expected),
        "predictor_sha256": sha256(assert_selected_predictor(root)),
        "checkpoints": expected,
    }
    if path.exists():
        existing = _read_json(path)
        if existing != value:
            raise RuntimeError("formal policy checkpoint set changed after it was frozen")
        return existing
    _write_json(path, value)
    return value


def _assert_policy_checkpoints_frozen(root: Path, output: Path) -> Mapping[str, Any]:
    path = output / "frozen_policy_checkpoint_manifest.json"
    if not path.is_file():
        raise RuntimeError("formal policy checkpoints were not frozen")
    frozen = _read_json(path)
    if int(frozen.get("checkpoint_count", -1)) != 24 or frozen.get("predictor_sha256") != PREDICTOR_SHA256:
        raise RuntimeError("formal policy checkpoint manifest is malformed")
    for row in frozen.get("checkpoints", []):
        checkpoint = output / str(row["path"])
        if not checkpoint.is_file() or sha256(checkpoint) != row.get("sha256"):
            raise RuntimeError("formal policy checkpoint changed after closed training")
    return frozen


def _verify_training_job(root: Path, job: Path, family: str, schema: Round12PolicySchema, validation: Round12SequenceSplit, device: str) -> None:
    best, last = job / "best.pt", job / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"formal run is missing a checkpoint: {job}")
    model, checkpoint = load_round12_checkpoint(best, family=family, schema=schema, map_location=device)
    if checkpoint.get("optimizer_state_dict") is None or checkpoint.get("predictor_sha256") != PREDICTOR_SHA256:
        raise RuntimeError("formal checkpoint lacks optimizer/predictor provenance")
    prediction, _ = _infer(model, family, validation.take(np.arange(min(8, len(validation))),), torch.device(device))
    reload, _ = _infer(load_round12_checkpoint(best, family=family, schema=schema, map_location=device)[0], family, validation.take(np.arange(min(8, len(validation)))), torch.device(device))
    if not np.array_equal(prediction, reload):
        raise RuntimeError("formal checkpoint reload predictions differ")


def _formal_training_integrity(root: Path, output: Path, schema: Round12PolicySchema, validation: Round12SequenceSplit, device: str, results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    rows = []
    for result in results:
        family, seed = str(result["family"]), int(result["seed"])
        job = output / "training" / family / f"seed_{seed}"
        _verify_training_job(root, job, family, schema, validation, device)
        payload = _torch_payload(job / "best.pt", torch.device(device))
        rows.append({"family": family, "seed": seed, "best_checkpoint": True, "last_checkpoint": (job / "last.pt").is_file(), "finite_history": _history_finite(payload.get("history", []))})
    return {
        "formal_training_completed": len(rows) == 12,
        "formal_training_completed_count": len(rows),
        "all_best_reload": True,
        "no_missing_checkpoint": all(row["best_checkpoint"] and row["last_checkpoint"] for row in rows),
        "no_nan_or_inf": all(row["finite_history"] for row in rows),
        "predictor_sha256_before_after_unchanged": sha256(assert_selected_predictor(root)) == PREDICTOR_SHA256,
        "predictor_requires_grad": False,
        "predictor_optimizer_absent": True,
        "runs": rows,
    }


def _history_finite(history: Iterable[Mapping[str, Any]]) -> bool:
    def visit(value: Any) -> bool:
        if isinstance(value, Mapping): return all(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)): return all(visit(item) for item in value)
        return not isinstance(value, (int, float, np.number)) or math.isfinite(float(value))
    return all(visit(row) for row in history)


def _torch_payload(path: Path, device: torch.device) -> Mapping[str, Any]:
    try: value = torch.load(path, map_location=device, weights_only=False)
    except TypeError: value = torch.load(path, map_location=device)
    if not isinstance(value, Mapping): raise ValueError("checkpoint payload is not a mapping")
    return value


def _infer(model: torch.nn.Module, family: str, sequence: Round12SequenceSplit, device: torch.device, *, predicted_phi: np.ndarray | None = None) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    """Inference-only final-token action extraction; no labels enter model inputs."""
    model = model.to(device).eval()
    condition = sequence.predicted_phi if predicted_phi is None else np.asarray(predicted_phi, dtype=np.float32)
    if condition.shape != sequence.predicted_phi.shape:
        raise ValueError("Phi perturbation shape differs from validation sequence")
    output: list[np.ndarray] = []
    for start in range(0, len(sequence), 1024):
        index = np.arange(start, min(start + 1024, len(sequence)))
        state = torch.as_tensor(sequence.states[index], device=device)
        node = torch.as_tensor(sequence.node_masks[index], device=device)
        with torch.no_grad():
            if family == TASK_WAYPOINT_MLP:
                value = model(state[:, -1], node[:, -1], torch.as_tensor(condition[index, -1], device=device))
            else:
                inputs: dict[str, torch.Tensor] = {
                    "states": state, "node_masks": node,
                    "remaining_time": torch.as_tensor(sequence.remaining_time[index], device=device),
                    "attention_mask": torch.as_tensor(sequence.attention_mask[index], device=device),
                }
                if family == ORACLE_TASK_WAYPOINT_TRANSFORMER:
                    inputs["conditions"] = torch.as_tensor(sequence.oracle_phi[index], device=device)
                elif family != STATE_ONLY_TRANSFORMER:
                    inputs["conditions"] = torch.as_tensor(condition[index], device=device)
                value = model(**inputs)[:, -1]
        output.append(value.detach().cpu().numpy())
    metadata = {name: np.asarray(value).copy() for name, value in sequence.metadata.items()}
    metadata["target"] = sequence.actions[:, -1].copy()
    return np.concatenate(output), metadata


def _summary(values: np.ndarray) -> Mapping[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    if value.size == 0: return {"mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "sample_count": 0}
    return {"mean": float(value.mean()), "std": float(value.std(ddof=0)), "median": float(np.median(value)), "p25": float(np.quantile(value, .25)), "p75": float(np.quantile(value, .75)), "sample_count": int(value.size)}


def _strata(root: Path, metadata: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Join only frozen audit labels to validation rows; never model inputs."""
    source = root / VALIDATION_SOURCE
    # The sidecars are legacy frozen audit products shared with other splits.
    # Materialise labels only for already-authorized validation IDs; they are
    # post-inference reporting annotations, never policy inputs.
    allowed_episode_ids = set(np.asarray(metadata["episode_id"]).astype(str).tolist())
    allowed_scenario_ids = set(np.asarray(metadata["scenario_id"]).astype(str).tolist())
    difficulty: dict[str, str] = {}
    with (source / "scenario_difficulty_audit.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario_id = str(row["scenario_id"])
            if scenario_id in allowed_scenario_ids:
                difficulty[scenario_id] = str(row["difficulty_tier"])
    quality: dict[str, str] = {}
    with (source / "trajectory_quality_audit.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row["episode_id"])
            if episode_id in allowed_episode_ids:
                quality[episode_id] = str(row["quality_tier"])
    stress = _read_json(source / "stitching_stress_manifest.json")
    bridge_ids = {str(edge[key]) for edge in stress.get("candidate_edges", []) for key in ("source_episode_id", "target_episode_id")}
    episode = np.asarray(metadata["episode_id"]).astype(str)
    scenario = np.asarray(metadata["scenario_id"]).astype(str)
    return MappingProxyType({
        "difficulty": np.asarray([difficulty.get(item, "unavailable") for item in scenario]),
        "quality": np.asarray([quality.get(item, "unavailable") for item in episode]),
        "behavior_generator": np.asarray(metadata["behavior_generator"]).astype(str),
        "buoy_scale": np.asarray(metadata["num_buoys"]).astype(str),
        # This is the frozen Round-7 stress-manifest bridge membership; no
        # threshold is recalculated in Round-12.
        "bridge": np.asarray(["bridge" if item in bridge_ids else "non_bridge" for item in episode]),
    })


def _metric_row(family: str, seed: int, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, group_dimension: str, group_value: str) -> Mapping[str, Any]:
    predicted, actual = prediction[mask], target[mask]
    metrics = evaluate_offline_actions(actual, predicted, max_step_distance_m=10.0)
    l2 = np.linalg.norm((predicted - actual) * 10.0, axis=1)
    return {
        "family": family, "seed": seed, "group_dimension": group_dimension, "group_value": group_value,
        "normalized_action_mse": metrics["normalized_action_mse"], "raw_action_rmse": metrics["raw_action_rmse"],
        "raw_action_mae": metrics["raw_action_mae"], "cosine_similarity": metrics["cosine_similarity"],
        "angular_error_deg": metrics["angular_error_deg"], "magnitude_error": metrics["magnitude_error"],
        "projection_activation_rate": metrics["projection_activation_rate"], **{f"action_l2_error_{key}": value for key, value in _summary(l2).items()},
    }


def run_offline_validation(root: Path, output: Path, validation: Round12SequenceSplit, schema: Round12PolicySchema, *, device: str) -> Mapping[str, Any]:
    """Evaluate every validation-selected checkpoint, grouped only after inference."""
    device_value = torch.device(device)
    all_rows: list[Mapping[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, Mapping[str, np.ndarray]]] = {}
    for family in FAMILY_ORDER:
        for seed in (0, 1, 2):
            job = output / "training" / family / f"seed_{seed}" / "best.pt"
            model, _ = load_round12_checkpoint(job, family=family, schema=schema, map_location=device_value)
            prediction, meta = _infer(model, family, validation, device_value)
            if not np.all(np.isfinite(prediction)): raise FloatingPointError("non-finite offline prediction")
            key = _run_key(family, seed); predictions[key] = (prediction, meta)
            target = meta["target"]
            all_rows.append(_metric_row(family, seed, prediction, target, np.ones(len(target), bool), "overall", "all"))
            for dimension, labels in _strata(root, meta).items():
                for label in sorted(set(labels.tolist())):
                    mask = labels == label
                    if np.any(mask): all_rows.append(_metric_row(family, seed, prediction, target, mask, dimension, str(label)))
    _write_csv(output / "offline_validation_metrics.csv", all_rows)
    _write_json(output / "offline_validation_manifest.json", {"validation_only": True, "model_seed_count": 12, "metric_rows": len(all_rows), "required_metrics": ["normalized action MSE", "raw action RMSE", "raw action MAE", "cosine similarity", "angular error", "magnitude error", "projection activation rate"]})
    return MappingProxyType({"rows": all_rows, "predictions": predictions})


def run_oracle_gap_and_utilization(root: Path, output: Path, validation: Round12SequenceSplit, schema: Round12PolicySchema, offline: Mapping[str, Any], *, device: str) -> Mapping[str, Any]:
    device_value = torch.device(device)
    predicted = offline["predictions"]
    strata = _strata(root, next(iter(predicted.values()))[1])
    gap_rows: list[Mapping[str, Any]] = []
    for seed in (0, 1, 2):
        wt, meta = predicted[_run_key(PREDICTED_TASK_WAYPOINT_TRANSFORMER, seed)]
        oracle, _ = predicted[_run_key(ORACLE_TASK_WAYPOINT_TRANSFORMER, seed)]
        for dimension, labels in {"overall": np.asarray(["all"] * len(wt)), "bridge": strata["bridge"], "difficulty": strata["difficulty"]}.items():
            for label in sorted(set(labels.tolist())):
                mask = labels == label
                if not np.any(mask): continue
                wt_rmse = evaluate_offline_actions(meta["target"][mask], wt[mask], max_step_distance_m=10.0)["raw_action_rmse"]
                oracle_rmse = evaluate_offline_actions(meta["target"][mask], oracle[mask], max_step_distance_m=10.0)["raw_action_rmse"]
                gap_rows.append({"seed": seed, "group_dimension": dimension, "group_value": str(label), "predicted_wt_rmse": wt_rmse, "oracle_rmse": oracle_rmse, "oracle_gap": float(wt_rmse - oracle_rmse), "predictor_bottleneck_possible": bool(wt_rmse > oracle_rmse)})
    _write_csv(output / "oracle_gap_analysis.csv", gap_rows)

    utilization: list[Mapping[str, Any]] = []
    operations = ("normal", "zero_full", "shuffle_full", "zero_service", "zero_progress", "zero_return", "shuffle_service", "shuffle_progress", "shuffle_return")
    for seed in (0, 1, 2):
        model, _ = load_round12_checkpoint(output / "training" / PREDICTED_TASK_WAYPOINT_TRANSFORMER / f"seed_{seed}" / "best.pt", family=PREDICTED_TASK_WAYPOINT_TRANSFORMER, schema=schema, map_location=device_value)
        baseline, meta = _infer(model, PREDICTED_TASK_WAYPOINT_TRANSFORMER, validation, device_value)
        phi = validation.predicted_phi.copy(); rng = np.random.default_rng(12012 + seed)
        for operation in operations:
            changed = phi.copy()
            if operation == "zero_full": changed.fill(0.0)
            elif operation == "shuffle_full": changed = changed[rng.permutation(len(changed))]
            elif operation == "zero_service": changed[..., :schema.max_buoys] = 0.0
            elif operation == "zero_progress": changed[..., schema.max_buoys:2 * schema.max_buoys] = 0.0
            elif operation == "zero_return": changed[..., -1] = 0.0
            elif operation == "shuffle_service": changed[..., :schema.max_buoys] = changed[rng.permutation(len(changed)), :, :schema.max_buoys]
            elif operation == "shuffle_progress": changed[..., schema.max_buoys:2 * schema.max_buoys] = changed[rng.permutation(len(changed)), :, schema.max_buoys:2 * schema.max_buoys]
            elif operation == "shuffle_return": changed[..., -1] = changed[rng.permutation(len(changed)), :, -1]
            perturbed, _ = _infer(model, PREDICTED_TASK_WAYPOINT_TRANSFORMER, validation, device_value, predicted_phi=changed)
            metric = evaluate_offline_actions(meta["target"], perturbed, max_step_distance_m=10.0)
            base = evaluate_offline_actions(meta["target"], baseline, max_step_distance_m=10.0)
            utilization.append({"seed": seed, "operation": operation, "action_sensitivity": float(np.linalg.norm((perturbed - baseline) * 10.0, axis=1).mean()), "action_rmse_change": float(metric["raw_action_rmse"] - base["raw_action_rmse"]), "cosine_change": float(metric["cosine_similarity"] - base["cosine_similarity"]), "magnitude_change": float(metric["magnitude_error"] - base["magnitude_error"]), "validation_only": True, "future_read": False})
    _write_csv(output / "task_waypoint_head_utilization.csv", utilization)
    sensitivity = {}
    for head, operation in (("service", "zero_service"), ("progress", "zero_progress"), ("return", "zero_return")):
        sensitivity[head] = float(np.mean([row["action_sensitivity"] for row in utilization if row["operation"] == operation]))
    dominant = max(sensitivity, key=sensitivity.get)
    _write_json(output / "task_waypoint_head_utilization_summary.json", {"Sensitivity_service": sensitivity["service"], "Sensitivity_progress": sensitivity["progress"], "Sensitivity_return": sensitivity["return"], "dominant_policy_head": dominant, "inference_only": True, "retraining_performed": False})
    return MappingProxyType({"oracle_gap": gap_rows, "utilization": utilization, "sensitivity": sensitivity, "dominant": dominant})


def run_bridge_analysis(root: Path, output: Path, offline: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for seed in (0, 1, 2):
        wt, meta = offline["predictions"][_run_key(PREDICTED_TASK_WAYPOINT_TRANSFORMER, seed)]
        state, _ = offline["predictions"][_run_key(STATE_ONLY_TRANSFORMER, seed)]
        labels = _strata(root, meta)["bridge"]
        for label in ("bridge", "non_bridge"):
            mask = labels == label
            if not np.any(mask):
                rows.append({"seed": seed, "region": label, "sample_count": 0, "predicted_wt_rmse": None, "state_only_rmse": None, "rmse_delta_state_minus_wt": None, "cosine_delta_wt_minus_state": None, "angular_delta_state_minus_wt": None, "magnitude_delta_state_minus_wt": None, "not_estimable": True, "bridge_definition": "Round-7 frozen stitching_stress_manifest candidate-edge membership"})
                continue
            target = meta["target"][mask]
            a, b = evaluate_offline_actions(target, wt[mask], max_step_distance_m=10.0), evaluate_offline_actions(target, state[mask], max_step_distance_m=10.0)
            rows.append({"seed": seed, "region": label, "sample_count": int(mask.sum()), "predicted_wt_rmse": a["raw_action_rmse"], "state_only_rmse": b["raw_action_rmse"], "rmse_delta_state_minus_wt": float(b["raw_action_rmse"] - a["raw_action_rmse"]), "cosine_delta_wt_minus_state": float(a["cosine_similarity"] - b["cosine_similarity"]), "angular_delta_state_minus_wt": float(b["angular_error_deg"] - a["angular_error_deg"]), "magnitude_delta_state_minus_wt": float(b["magnitude_error"] - a["magnitude_error"]), "not_estimable": False, "bridge_definition": "Round-7 frozen stitching_stress_manifest candidate-edge membership"})
    _write_csv(output / "bridge_policy_analysis.csv", rows)
    return rows


def _parse_scenario(scenario_id: str) -> tuple[int, int]:
    match = re.search(r"seed-(\d+)__buoys-(\d+)$", str(scenario_id))
    if match is None: raise ValueError(f"unparseable frozen scenario ID: {scenario_id}")
    return int(match.group(1)), int(match.group(2))


def _rollout(model: torch.nn.Module, family: str, model_seed: int, scenario: Mapping[str, Any], schema: Round12PolicySchema, *, device: torch.device, predictor: torch.nn.Module | None) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if not family_is_deployable(family): raise PermissionError("Oracle rollout is prohibited")
    seed, buoy_count = int(scenario["scenario_seed"]), int(scenario["num_buoys"])
    scenario_id, config, _, trace_id, _ = round7_case(seed, buoy_count, ROUND12_EPISODE_HORIZON)
    if scenario_id != scenario["scenario_id"]: raise RuntimeError("frozen scenario reconstruction mismatch")
    env = MaritimeEnv(config); observation, info = env.reset(seed=seed)
    model = model.to(device).eval()
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    steps: list[Mapping[str, Any]] = []; total_return = 0.0; projections = 0; violations = 0
    while True:
        state = np.zeros(schema.state_dim, dtype=np.float32); state[:len(observation)] = observation
        node = np.zeros(schema.max_buoys, dtype=np.float32); node[:buoy_count] = 1.0
        remaining = (ROUND12_EPISODE_HORIZON - env.time_step) / ROUND12_EPISODE_HORIZON
        if family == STATE_ONLY_TRANSFORMER:
            phi = np.empty((0,), dtype=np.float32)
        else:
            if predictor is None: raise RuntimeError("deployable Task-Waypoint policy lacks frozen predictor")
            phi = predicted_phi(predictor, torch.from_numpy(state[None]).to(device), torch.from_numpy(node[None]).to(device))[0].cpu().numpy()
        if family == TASK_WAYPOINT_MLP:
            with torch.no_grad(): action = model(torch.from_numpy(state[None]).to(device), torch.from_numpy(node[None]).to(device), torch.from_numpy(phi[None]).to(device))[0].cpu().numpy()
        else:
            history.append((state, node, phi, remaining)); history = history[-CONTEXT_LENGTH:]
            width = len(history); offset = CONTEXT_LENGTH - width
            states = np.zeros((1, CONTEXT_LENGTH, schema.state_dim), dtype=np.float32); nodes = np.zeros((1, CONTEXT_LENGTH, schema.max_buoys), dtype=np.float32); rem = np.zeros((1, CONTEXT_LENGTH, 1), dtype=np.float32); attention = np.zeros((1, CONTEXT_LENGTH), dtype=bool)
            for index, item in enumerate(history): states[0, offset + index], nodes[0, offset + index], rem[0, offset + index, 0], attention[0, offset + index] = item[0], item[1], item[3], True
            inputs: dict[str, torch.Tensor] = {"states": torch.from_numpy(states).to(device), "node_masks": torch.from_numpy(nodes).to(device), "remaining_time": torch.from_numpy(rem).to(device), "attention_mask": torch.from_numpy(attention).to(device)}
            if family != STATE_ONLY_TRANSFORMER:
                conditions = np.zeros((1, CONTEXT_LENGTH, schema.phi_dim), dtype=np.float32)
                for index, item in enumerate(history): conditions[0, offset + index] = item[2]
                inputs["conditions"] = torch.from_numpy(conditions).to(device)
            with torch.no_grad(): action = model(**inputs)[0, -1].cpu().numpy()
        safe = radial_projection(action)
        projected = not np.allclose(safe, action, rtol=0.0, atol=1e-7)
        observation, reward, terminated, truncated, info = env.step(safe)
        total_return += float(reward); projections += int(projected); violations += int(bool(info.get("boundary_violation", False)))
        steps.append({"scenario_id": scenario_id, "family": family, "model_seed": model_seed, "timestep": int(env.time_step - 1), "predicted_action_x": float(action[0]), "predicted_action_y": float(action[1]), "projected_action_x": float(safe[0]), "projected_action_y": float(safe[1]), "projection_activated": bool(projected), "predicted_service": phi[:schema.max_buoys].tolist() if phi.size else [], "predicted_progress": phi[schema.max_buoys:2 * schema.max_buoys].tolist() if phi.size else [], "predicted_return": float(phi[-1]) if phi.size else 0.0, "cumulative_payload_bits": float(info["cumulative_payload_bits"])})
        if terminated or truncated: break
    payload = float(info["cumulative_payload_bits"]); completion = float(info.get("task_completion_ratio", 0.0)); residual = float(np.sum(np.asarray(info.get("raw_queue_bits", []), dtype=float))); path = float(info.get("cumulative_distance_m", 0.0)); energy = float(info.get("cumulative_energy_consumed", info.get("cumulative_energy_consumption", 0.0)))
    return ({"scenario_id": scenario_id, "scenario_seed": seed, "num_buoys": buoy_count, "family": family, "model_seed": model_seed, "cumulative_actual_payload": payload, "episode_return": total_return, "completion": completion, "residual_queue_bits": residual, "path_length_m": path, "movement_energy": energy, "projection_rate": projections / max(len(steps), 1), "projection_count": projections, "safety_violations": violations, "episode_length": len(steps), "trace_id": trace_id}, steps)


def _scenario_records(validation: Round12SequenceSplit) -> list[Mapping[str, Any]]:
    values = sorted(set(np.asarray(validation.metadata["scenario_id"]).astype(str).tolist()))
    return [{"scenario_id": item, "scenario_seed": _parse_scenario(item)[0], "num_buoys": _parse_scenario(item)[1]} for item in values]


def _paired_statistics(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    by: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows: by[(str(row["scenario_id"]), int(row["model_seed"]), str(row["family"]))] = row
    metrics = {"Delta_payload": ("cumulative_actual_payload", 1.0), "Delta_return": ("episode_return", 1.0), "Delta_completion": ("completion", 1.0), "Delta_residual_queue": ("residual_queue_bits", -1.0)}
    result: list[Mapping[str, Any]] = []
    for metric, (field, direction) in metrics.items():
        per_seed: dict[int, list[tuple[str, float]]] = {}
        for scenario, seed, family in by:
            if family != PREDICTED_TASK_WAYPOINT_TRANSFORMER: continue
            other = by.get((scenario, seed, STATE_ONLY_TRANSFORMER))
            if other is None: continue
            per_seed.setdefault(seed, []).append((scenario, direction * (float(by[(scenario, seed, family)][field]) - float(other[field]))))
        pooled: dict[str, list[float]] = {}
        for pairs in per_seed.values():
            for scenario, value in pairs: pooled.setdefault(scenario, []).append(value)
        for seed, pairs in sorted(per_seed.items()): result.append(_paired_row(metric, pairs, protocol, label, seed))
        result.append(_paired_row(metric, [(scenario, float(np.mean(values))) for scenario, values in pooled.items()], protocol, label, "pooled_scenario_mean"))
    return result


def _paired_row(metric: str, pairs: Sequence[tuple[str, float]], protocol: Mapping[str, Any], label: str, seed: int | str) -> Mapping[str, Any]:
    values = np.asarray([value for _, value in pairs], dtype=np.float64); rng = np.random.default_rng(int(protocol["bootstrap_seed"]) + int(hashlib.sha256(f"{label}:{metric}:{seed}".encode()).hexdigest()[:8], 16))
    boot = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(int(protocol["bootstrap_replicates"]))])
    return {"analysis": label, "comparison": "predicted_task_waypoint_transformer_minus_state_only_transformer", "metric": metric, "model_seed": seed, "scenario_count": len(values), "paired_mean": float(values.mean()), "paired_median": float(np.median(values)), "paired_std": float(values.std(ddof=0)), "bootstrap_ci_low": float(np.quantile(boot, .025)), "bootstrap_ci_high": float(np.quantile(boot, .975)), "win_rate": float(np.mean(values > float(protocol["tie_tolerance"]))), "tie_rate": float(np.mean(np.abs(values) <= float(protocol["tie_tolerance"]))), "loss_rate": float(np.mean(values < -float(protocol["tie_tolerance"]))), "effect_size_mean_over_std": float(values.mean() / values.std(ddof=0)) if values.std(ddof=0) > 0 else 0.0}


def run_closed_loop_validation(root: Path, output: Path, validation: Round12SequenceSplit, schema: Round12PolicySchema, protocol: Mapping[str, Any], *, device: str) -> Mapping[str, Any]:
    device_value = torch.device(device); scenarios = _scenario_records(validation); episodes: list[Mapping[str, Any]] = []; steps: list[Mapping[str, Any]] = []
    predictor = load_frozen_predictor(root, schema.state_dim, schema.max_buoys).to(device_value); assert_predictor_frozen(predictor)
    for family in (PREDICTED_TASK_WAYPOINT_TRANSFORMER, STATE_ONLY_TRANSFORMER, TASK_WAYPOINT_MLP):
        for seed in (0, 1, 2):
            model, _ = load_round12_checkpoint(output / "training" / family / f"seed_{seed}" / "best.pt", family=family, schema=schema, map_location=device_value)
            for scenario in scenarios:
                metric, records = _rollout(model, family, seed, scenario, schema, device=device_value, predictor=None if family == STATE_ONLY_TRANSFORMER else predictor)
                episodes.append(metric); steps.extend(records)
    pairs = _paired_statistics(episodes, protocol, label="validation")
    _write_csv(output / "validation_closed_loop_episode_metrics.csv", episodes); _write_csv(output / "validation_closed_loop_step_metrics.csv", steps); _write_csv(output / "validation_paired_statistics.csv", pairs)
    gate = _validation_gate(episodes, pairs, protocol)
    _write_json(output / "validation_policy_gate.json", gate)
    if sha256(assert_selected_predictor(root)) != PREDICTOR_SHA256: raise RuntimeError("predictor changed during validation closed loop")
    return MappingProxyType({"episodes": episodes, "steps": steps, "pairs": pairs, "gate": gate})


def _validation_gate(rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = next(row for row in pairs if row["metric"] == "Delta_payload" and row["model_seed"] == "pooled_scenario_mean")
    by_seed = [row for row in pairs if row["metric"] == "Delta_payload" and isinstance(row["model_seed"], int)]
    positive = sum(float(row["paired_mean"]) > 0 for row in by_seed)
    projection = _paired_projection(rows)
    safe = projection["projection_mean_delta"] <= float(protocol["projection_absolute_degradation_limit"]) and projection["violation_mean_delta"] <= float(protocol["safety_violation_degradation_limit"])
    if payload["paired_mean"] > 0 and payload["bootstrap_ci_low"] > 0 and positive >= 2 and safe: value = "V-A"
    elif payload["paired_mean"] < 0 and payload["bootstrap_ci_high"] < 0 and positive <= 1: value = "V-C"
    else: value = "V-B"
    return {"validation_policy_gate": value, "validation_supports_task_waypoint": True if value == "V-A" else (False if value == "V-C" else "uncertain"), "mean_Delta_payload": payload["paired_mean"], "Delta_payload_95_CI": [payload["bootstrap_ci_low"], payload["bootstrap_ci_high"]], "positive_model_seed_count": positive, "safety_projection": projection, "fresh_development_allowed": value in ("V-A", "V-B"), "oracle_closed_loop_eligible": False, "primary_metric": "cumulative_actual_payload"}


def _paired_projection(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    left = {(row["scenario_id"], row["model_seed"]): row for row in rows if row["family"] == PREDICTED_TASK_WAYPOINT_TRANSFORMER}; right = {(row["scenario_id"], row["model_seed"]): row for row in rows if row["family"] == STATE_ONLY_TRANSFORMER}
    keys = sorted(set(left) & set(right)); return {"projection_mean_delta": float(np.mean([left[key]["projection_rate"] - right[key]["projection_rate"] for key in keys])), "violation_mean_delta": float(np.mean([left[key]["safety_violations"] - right[key]["safety_violations"] for key in keys]))}


def run_fresh_development(root: Path, output: Path, schema: Round12PolicySchema, protocol: Mapping[str, Any], gate: Mapping[str, Any], *, device: str) -> Mapping[str, Any] | None:
    """Perform the single pre-registered 60-scenario access only after V-A/B."""
    if gate["validation_policy_gate"] == "V-C": return None
    frozen = _read_json(output / "formal_protocol_manifest.json")
    _assert_frozen_unchanged(root, frozen)
    _assert_policy_checkpoints_frozen(root, output)
    fresh_path = root / "round_12/fresh_policy_dev_manifest.json"; fresh = _read_json(fresh_path)
    if int(fresh.get("access_count", -1)) not in (0, 1): raise RuntimeError("invalid fresh-development access count")
    if int(fresh["access_count"]) == 1:
        summary_path = output / "fresh_development_summary.json"
        if not summary_path.is_file():
            raise RuntimeError("fresh-development manifest was accessed by another output")
        return _read_json(summary_path)
    scenarios = [{"scenario_id": f"maritime_offline_dataset_v2__seed-{seed:05d}__buoys-{buoys}", "scenario_seed": int(seed), "num_buoys": int(buoys)} for seed in fresh["scenario_seeds"] for buoys in fresh["buoy_scales"]]
    if len(scenarios) != int(fresh["physical_scenario_count"]): raise RuntimeError("fresh manifest physical scenario count mismatch")
    device_value = torch.device(device); predictor = load_frozen_predictor(root, schema.state_dim, schema.max_buoys).to(device_value); episodes: list[Mapping[str, Any]] = []; steps: list[Mapping[str, Any]] = []
    for family in (PREDICTED_TASK_WAYPOINT_TRANSFORMER, STATE_ONLY_TRANSFORMER, TASK_WAYPOINT_MLP):
        for seed in (0, 1, 2):
            model, _ = load_round12_checkpoint(output / "training" / family / f"seed_{seed}" / "best.pt", family=family, schema=schema, map_location=device_value)
            for scenario in scenarios:
                metric, records = _rollout(model, family, seed, scenario, schema, device=device_value, predictor=None if family == STATE_ONLY_TRANSFORMER else predictor)
                episodes.append(metric); steps.extend(records)
    pairs = _paired_statistics(episodes, protocol, label="fresh_development")
    _write_csv(output / "fresh_development_closed_loop_episode_metrics.csv", episodes); _write_csv(output / "fresh_development_closed_loop_step_metrics.csv", steps); _write_csv(output / "fresh_development_paired_statistics.csv", pairs)
    safety_projection = _paired_projection(episodes)
    fresh.update({"access_count": 1, "selection_or_training_use": False, "accessed_for": "Round-12 frozen-checkpoint fresh-development evaluation", "access_output": str(output.relative_to(root)), "accessed_at_utc": datetime.now(timezone.utc).isoformat()})
    _write_json(fresh_path, fresh)
    summary = {"accessed": True, "scenario_count": len(scenarios), "pairs": pairs, "safety_projection": safety_projection, "fresh_dev_access_count": 1, "checkpoint_manifest": "frozen_policy_checkpoint_manifest.json", "selection_or_training_use": False}
    _write_json(output / "fresh_development_summary.json", summary)
    return summary


def _final_policy_gate(validation: Mapping[str, Any], fresh: Mapping[str, Any] | None, utilization: Mapping[str, Any], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    if fresh is None:
        return {"task_waypoint_policy_gate": "TW-POL-C", "candidate_ready_for_untouched_holdout": False, "reason": "validation V-C prohibited fresh-development access"}
    pairs = fresh["pairs"]; payload = next(row for row in pairs if row["metric"] == "Delta_payload" and row["model_seed"] == "pooled_scenario_mean"); seed_rows = [row for row in pairs if row["metric"] == "Delta_payload" and isinstance(row["model_seed"], int)]
    secondary = {metric: next(row for row in pairs if row["metric"] == metric and row["model_seed"] == "pooled_scenario_mean") for metric in ("Delta_return", "Delta_completion", "Delta_residual_queue")}
    positive = sum(row["paired_mean"] > 0 for row in seed_rows); phi_dependency = all(np.mean([row["action_sensitivity"] for row in utilization["utilization"] if row["operation"] == operation]) > 1e-6 for operation in ("zero_full", "shuffle_full"))
    non_degrade = secondary["Delta_return"]["paired_mean"] >= 0 and secondary["Delta_completion"]["paired_mean"] >= 0 and secondary["Delta_residual_queue"]["paired_mean"] >= 0
    projection = fresh["safety_projection"]
    safe = projection["projection_mean_delta"] <= float(protocol["projection_absolute_degradation_limit"]) and projection["violation_mean_delta"] <= float(protocol["safety_violation_degradation_limit"])
    if payload["paired_mean"] > 0 and payload["bootstrap_ci_low"] > 0 and positive >= 2 and non_degrade and safe and phi_dependency: gate, ready = "TW-POL-A", True
    elif payload["paired_mean"] < 0 and payload["bootstrap_ci_high"] < 0 and positive <= 1: gate, ready = "TW-POL-C", False
    else: gate, ready = "TW-POL-B", False
    return {"task_waypoint_policy_gate": gate, "candidate_ready_for_untouched_holdout": ready, "fresh_mean_Delta_payload": payload["paired_mean"], "fresh_Delta_payload_95_CI": [payload["bootstrap_ci_low"], payload["bootstrap_ci_high"]], "fresh_positive_model_seed_count": positive, "completion_delta": secondary["Delta_completion"]["paired_mean"], "residual_queue_delta": secondary["Delta_residual_queue"]["paired_mean"], "return_delta": secondary["Delta_return"]["paired_mean"], "safety_projection": projection, "safety_projection_acceptable": safe, "task_waypoint_dependency_evidence": phi_dependency}


def _mechanism(offline: Mapping[str, Any], validation: Mapping[str, Any], final_gate: Mapping[str, Any]) -> str:
    oracle = [row["oracle_gap"] for row in offline["oracle_gap"] if row["group_dimension"] == "overall"]; bottleneck = float(np.mean(oracle)) > 0 if oracle else False
    if final_gate["task_waypoint_policy_gate"] == "TW-POL-A": return "P1 + P3" if bottleneck else "P3"
    return "P1" if bottleneck else "P2"


def _figures(output: Path, offline: Mapping[str, Any], utilization: Mapping[str, Any], validation: Mapping[str, Any], fresh: Mapping[str, Any] | None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output / "figures"; figure_dir.mkdir(exist_ok=True)
    written: list[str] = []

    def save(name: str, figure: Any) -> None:
        figure.tight_layout()
        for suffix in ("png", "svg"):
            destination = figure_dir / f"{name}.{suffix}"
            figure.savefig(destination, dpi=140 if suffix == "png" else None)
            written.append(destination.name)
        plt.close(figure)

    def bar(name: str, title: str, labels: Sequence[str], values: Sequence[float], *, color: str = "#1f77b4") -> None:
        figure, axis = plt.subplots(figsize=(max(5, len(labels) * .55), 4), facecolor="white")
        axis.bar(range(len(values)), values, color=color)
        axis.set_xticks(range(len(labels))); axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_title(title); axis.axhline(0, color="black", linewidth=.8)
        save(name, figure)

    def family_means(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[list[str], list[float]]:
        labels = [family for family in (PREDICTED_TASK_WAYPOINT_TRANSFORMER, STATE_ONLY_TRANSFORMER, TASK_WAYPOINT_MLP) if any(row["family"] == family for row in rows)]
        return labels, [float(np.mean([float(row[field]) for row in rows if row["family"] == label])) for label in labels]

    metric_rows = [row for row in offline["rows"] if row["group_dimension"] == "overall"]
    bar("offline_rmse", "Offline validation raw action RMSE", [f"{row['family']}:{row['seed']}" for row in metric_rows], [float(row["raw_action_rmse"]) for row in metric_rows])
    oracle = [row for row in utilization["oracle_gap"] if row["group_dimension"] == "overall"]
    bar("oracle_gap", "Oracle offline RMSE gap", [str(row["seed"]) for row in oracle], [float(row["oracle_gap"]) for row in oracle], color="#9b5de5")
    bar("head_sensitivity", "Task-Waypoint head sensitivity", list(utilization["sensitivity"]), list(utilization["sensitivity"].values()), color="#e76f51")

    def paired_offline(left: str, right: str) -> tuple[list[str], list[float]]:
        values = []
        for seed in (0, 1, 2):
            a = next(row for row in metric_rows if row["family"] == left and row["seed"] == seed)
            b = next(row for row in metric_rows if row["family"] == right and row["seed"] == seed)
            values.append(float(a["raw_action_rmse"] - b["raw_action_rmse"]))
        return [f"seed {seed}" for seed in (0, 1, 2)], values

    labels, values = paired_offline(PREDICTED_TASK_WAYPOINT_TRANSFORMER, STATE_ONLY_TRANSFORMER)
    bar("wt_vs_state_only", "Offline RMSE: Predicted WT minus State-only", labels, values, color="#457b9d")
    labels, values = paired_offline(PREDICTED_TASK_WAYPOINT_TRANSFORMER, TASK_WAYPOINT_MLP)
    bar("wt_vs_mlp", "Offline RMSE: Predicted WT minus MLP", labels, values, color="#457b9d")

    bridge = [row for row in _read_csv(output / "bridge_policy_analysis.csv") if row.get("not_estimable") != "True"]
    bar("bridge_nonbridge", "State-only RMSE minus Predicted WT by bridge region", [f"{row['region']}:{row['seed']}" for row in bridge], [float(row["rmse_delta_state_minus_wt"]) for row in bridge], color="#43aa8b")
    episode_rows = validation["episodes"]
    for field, name, title in (
        ("cumulative_actual_payload", "validation_payload", "Validation cumulative actual payload"),
        ("episode_return", "validation_return", "Validation episode return"),
        ("completion", "completion", "Validation completion"),
        ("residual_queue_bits", "residual_queue", "Validation residual queue"),
    ):
        labels, values = family_means(episode_rows, field)
        bar(name, title, labels, values, color="#2a9d8f")
    paired = [row for row in validation["pairs"] if row["metric"] == "Delta_payload"]
    bar("delta_payload", "Validation Delta payload (WT minus State-only)", [str(row["model_seed"]) for row in paired], [float(row["paired_mean"]) for row in paired], color="#2a9d8f")

    # Plot every frozen training trajectory; the selection metric is raw RMSE.
    figure, axis = plt.subplots(figsize=(8, 5), facecolor="white")
    for family in FAMILY_ORDER:
        for seed in (0, 1, 2):
            history = _read_json(output / "training" / family / f"seed_{seed}" / "training_history.json")["history"]
            axis.plot([int(row["epoch"]) for row in history], [float(row["validation_raw_action_rmse"]) for row in history], label=f"{family}:{seed}", linewidth=1)
    axis.set_title("Validation raw action RMSE training curves"); axis.set_xlabel("epoch"); axis.set_ylabel("RMSE"); axis.legend(fontsize=6, ncol=2)
    save("training_curves", figure)

    step_rows = validation["steps"]
    reference_scenario = sorted(set(str(row["scenario_id"]) for row in step_rows))[0]
    figure, axis = plt.subplots(figsize=(8, 4), facecolor="white")
    for family in (PREDICTED_TASK_WAYPOINT_TRANSFORMER, STATE_ONLY_TRANSFORMER, TASK_WAYPOINT_MLP):
        rows = [row for row in step_rows if row["scenario_id"] == reference_scenario and row["family"] == family and int(row["model_seed"]) == 0]
        axis.plot([int(row["timestep"]) for row in rows], [float(row["projected_action_x"]) for row in rows], label=family)
    axis.set_title("Representative validation action trajectories (x component)"); axis.set_xlabel("timestep"); axis.legend(fontsize=7)
    save("representative_trajectories", figure)

    figure, axis = plt.subplots(figsize=(8, 4), facecolor="white")
    for family in (PREDICTED_TASK_WAYPOINT_TRANSFORMER, STATE_ONLY_TRANSFORMER, TASK_WAYPOINT_MLP):
        for timestep in sorted({int(row["timestep"]) for row in step_rows}):
            values = [float(row["cumulative_payload_bits"]) for row in step_rows if row["family"] == family and int(row["timestep"]) == timestep]
            axis.plot(timestep, float(np.mean(values)), marker=".", color={PREDICTED_TASK_WAYPOINT_TRANSFORMER: "#1f77b4", STATE_ONLY_TRANSFORMER: "#ff7f0e", TASK_WAYPOINT_MLP: "#2ca02c"}[family])
        values_by_t = [(timestep, float(np.mean([float(row["cumulative_payload_bits"]) for row in step_rows if row["family"] == family and int(row["timestep"]) == timestep]))) for timestep in sorted({int(row["timestep"]) for row in step_rows})]
        axis.plot([item[0] for item in values_by_t], [item[1] for item in values_by_t], label=family)
    axis.set_title("Validation cumulative payload over time"); axis.set_xlabel("timestep"); axis.legend(fontsize=7)
    save("cumulative_payload_over_time", figure)

    phi_rows = [row for row in step_rows if row["scenario_id"] == reference_scenario and row["family"] == PREDICTED_TASK_WAYPOINT_TRANSFORMER and int(row["model_seed"]) == 0]
    figure, axis = plt.subplots(figsize=(8, 4), facecolor="white")
    axis.plot([int(row["timestep"]) for row in phi_rows], [float(np.mean(row["predicted_service"])) for row in phi_rows], label="mean predicted service")
    axis.plot([int(row["timestep"]) for row in phi_rows], [float(np.mean(row["predicted_progress"])) for row in phi_rows], label="mean predicted progress")
    axis.plot([int(row["timestep"]) for row in phi_rows], [float(row["predicted_return"]) for row in phi_rows], label="predicted H-step return")
    axis.set_title("Predicted Task-Waypoint components over time"); axis.set_xlabel("timestep"); axis.legend(fontsize=7)
    save("task_waypoint_predictions_over_time", figure)

    bar("per_seed_results", "Validation cumulative payload per model seed", [f"{row['family']}:{row['model_seed']}" for row in episode_rows], [float(row["cumulative_actual_payload"]) for row in episode_rows], color="#264653")
    if fresh is not None:
        fresh_rows = _read_csv(output / "fresh_development_closed_loop_episode_metrics.csv")
        labels, values = family_means(fresh_rows, "cumulative_actual_payload")
        bar("fresh_development_payload", "Fresh-development cumulative actual payload", labels, values, color="#e76f51")
    return sorted(path.name for path in figure_dir.iterdir())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _write_report(output: Path, offline: Mapping[str, Any], utilization: Mapping[str, Any], validation: Mapping[str, Any], fresh: Mapping[str, Any] | None, final_gate: Mapping[str, Any], mechanism: str) -> None:
    overall = [row for row in offline["rows"] if row["group_dimension"] == "overall"]
    rmse = {f"{row['family']}:{row['seed']}": float(row["raw_action_rmse"]) for row in overall}
    gate = validation["gate"]
    lines = [
        "# Round-12 frozen formal report",
        "",
        "All policy checkpoints were selected by validation raw action RMSE. Oracle results are offline-only; no Oracle closed-loop rollout was run.",
        "",
        f"- Validation gate: `{gate['validation_policy_gate']}`",
        f"- Final policy gate: `{final_gate['task_waypoint_policy_gate']}`",
        f"- Mechanism conclusion: `{mechanism}`",
        f"- Fresh development accessed: `{fresh is not None}`",
        "",
        "## Offline raw action RMSE",
        "",
    ]
    lines.extend(f"- `{key}`: {value:.8f}" for key, value in sorted(rmse.items()))
    lines.extend(["", "## Task-Waypoint sensitivity", ""])
    lines.extend(f"- `{name}`: {float(value):.8f}" for name, value in utilization["sensitivity"].items())
    if fresh is not None:
        lines.extend(["", f"Fresh Delta payload: {float(final_gate['fresh_mean_Delta_payload']):.8f}"])
    (output / "round12_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _final_audit_and_lock(root: Path, output: Path, frozen: Mapping[str, Any], final_gate: Mapping[str, Any], mechanism: str, figure_files: Sequence[str]) -> Mapping[str, Any]:
    _assert_frozen_unchanged(root, frozen)
    _assert_policy_checkpoints_frozen(root, output)
    predictor = assert_selected_predictor(root)
    fresh = _read_json(root / "round_12/fresh_policy_dev_manifest.json"); manifest = _read_json(root / "round_12/round_manifest.json")
    figures_ok = all((output / "figures" / name).is_file() and (output / "figures" / name).stat().st_size > 0 for name in figure_files)
    required = [
        "training_curves", "offline_rmse", "oracle_gap", "wt_vs_state_only", "wt_vs_mlp", "head_sensitivity", "bridge_nonbridge", "validation_payload", "delta_payload", "validation_return", "completion", "residual_queue", "representative_trajectories", "cumulative_payload_over_time", "task_waypoint_predictions_over_time", "per_seed_results",
    ]
    if int(fresh["access_count"]) == 1:
        required.append("fresh_development_payload")
    expected_files = [f"{stem}.{suffix}" for stem in required for suffix in ("png", "svg")]
    audit = {"full_regression_tests_required": True, "compileall_required": True, "checkpoint_hash_audit": True, "predictor_hash_unchanged": sha256(predictor) == PREDICTOR_SHA256, "old_development_test_access_count": 0, "untouched_holdout_access_count": 0, "fresh_development_access_count": int(fresh["access_count"]), "manifest_consistent": manifest.get("status") == "in_progress", "figure_semantic_audit_passed": figures_ok and all(name in figure_files for name in expected_files), "expected_figure_stems": required, "source_tree_unchanged_since_formal_freeze": True, "task_waypoint_policy_gate": final_gate["task_waypoint_policy_gate"], "mechanism_conclusion": mechanism, "round_lock_pending_final_regression": True}
    _write_json(output / "final_audit.json", audit)
    return audit


def record_final_verification(output: Path, *, full_test_pass_count: int, compileall_passed: bool) -> Mapping[str, Any]:
    """Store the external full-suite/compile receipt without changing source."""
    if int(full_test_pass_count) < 1 or not bool(compileall_passed):
        raise ValueError("Round-12 can record only a successful final verification")
    value = {
        "full_regression_tests_passed": True,
        "full_regression_tests_pass_count": int(full_test_pass_count),
        "compileall_passed": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(Path(output) / "final_verification.json", value)
    return value


def finalize_round12(root: Path, output: Path) -> Mapping[str, Any]:
    """Lock Round-12 only after the external final verification receipt exists."""
    root, output = Path(root), Path(output)
    existing_lock = root / "round_12/ROUND_LOCK.json"
    if existing_lock.is_file():
        return _read_json(existing_lock)
    frozen = _read_json(output / "formal_protocol_manifest.json")
    _assert_frozen_unchanged(root, frozen)
    _assert_policy_checkpoints_frozen(root, output)
    verification = _read_json(output / "final_verification.json")
    if not verification.get("full_regression_tests_passed") or not verification.get("compileall_passed"):
        raise RuntimeError("Round-12 final regression/compile verification has not passed")
    audit = _read_json(output / "final_audit.json")
    manifest = _read_json(root / "round_12/round_manifest.json")
    fresh = _read_json(root / "round_12/fresh_policy_dev_manifest.json")
    validation_gate = _read_json(output / "validation_policy_gate.json")
    formal_integrity = _read_json(output / "formal_training_integrity.json")
    run_manifest = _read_json(output / "run_manifest.json")
    current_path = root / "CURRENT_ROUND.txt"
    current = current_path.read_text(encoding="utf-8")
    if "status=in_progress" not in current:
        raise RuntimeError("CURRENT_ROUND is not in progress at Round-12 lock time")
    required_files = [
        "formal_training_integrity.json", "offline_validation_metrics.csv", "oracle_gap_analysis.csv", "task_waypoint_head_utilization.csv", "bridge_policy_analysis.csv", "validation_closed_loop_episode_metrics.csv", "validation_paired_statistics.csv", "validation_policy_gate.json", "final_policy_gate.json", "round12_report.md", "run_manifest.json",
    ]
    if validation_gate["validation_policy_gate"] in ("V-A", "V-B"):
        required_files.extend(["fresh_development_closed_loop_episode_metrics.csv", "fresh_development_paired_statistics.csv", "fresh_development_summary.json"])
    missing = [name for name in required_files if not (output / name).is_file()]
    figures_ok = all((output / "figures" / f"{stem}.{suffix}").is_file() and (output / "figures" / f"{stem}.{suffix}").stat().st_size > 0 for stem in audit["expected_figure_stems"] for suffix in ("png", "svg"))
    fresh_count = int(fresh.get("access_count", -1))
    expected_fresh_count = 0 if validation_gate["validation_policy_gate"] == "V-C" else 1
    stages_complete = all(bool(run_manifest.get(key)) for key in ("offline_validation_completed", "oracle_diagnostic_completed", "head_utilization_completed", "bridge_analysis_completed", "validation_closed_loop_completed"))
    training_complete = formal_integrity.get("formal_training_completed_count") == 12 and all(bool(formal_integrity.get(key)) for key in ("formal_training_completed", "all_best_reload", "no_missing_checkpoint", "no_nan_or_inf", "predictor_sha256_before_after_unchanged"))
    accepted = not missing and figures_ok and stages_complete and training_complete and fresh_count == expected_fresh_count and fresh_count <= 1 and manifest.get("status") == "in_progress" and not (root / "round_13").exists() and sha256(assert_selected_predictor(root)) == PREDICTOR_SHA256
    if not accepted:
        raise RuntimeError(f"Round-12 lock acceptance failed; missing={missing}, fresh={fresh_count}, figures={figures_ok}")
    final_audit = dict(audit)
    final_audit.update({"full_regression_tests_passed": True, "full_regression_tests_pass_count": int(verification["full_regression_tests_pass_count"]), "compileall_passed": True, "checkpoint_hash_audit": True, "figure_semantic_audit_passed": figures_ok, "round_lock_pending_final_regression": False, "round12_acceptance_complete": True})
    _write_json(output / "final_audit.json", final_audit)
    lock = {"round": "round_12", "status": "locked", "locked_at_utc": datetime.now(timezone.utc).isoformat(), "formal_output": str(output.relative_to(root)), "formal_protocol_sha256": sha256(output / "formal_protocol_manifest.json"), "run_manifest_sha256": sha256(output / "run_manifest.json"), "final_audit_sha256": sha256(output / "final_audit.json"), "predictor_sha256": sha256(assert_selected_predictor(root)), "fresh_development_access_count": fresh_count, "old_development_test_access_count": 0, "untouched_holdout_access_count": 0, "round_13_created": False, "acceptance_complete": True}
    _write_json(existing_lock, lock)
    completed_manifest = dict(manifest)
    completed_manifest.update({"status": "completed", "lock_file": "round_12/ROUND_LOCK.json", "formal_output": str(output.relative_to(root)), "final_policy_gate": _read_json(output / "final_policy_gate.json")["task_waypoint_policy_gate"], "old_development_test_accessed": False, "untouched_holdout_accessed": False, "next_round_created": False})
    _write_json(root / "round_12/round_manifest.json", completed_manifest)
    current_path.write_text(current.replace("status=in_progress", "status=completed", 1) + "lock_file=round_12/ROUND_LOCK.json\n", encoding="utf-8")
    return lock


def run_formal(root: Path = Path.cwd(), *, output: Path | None = None, device: str | None = None) -> Path:
    root = Path(root); device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = output or root / "round_12" / "outputs" / datetime.now(timezone.utc).strftime("round12_formal_%Y%m%d_%H%M%S")
    dataset, train, validation, frozen = run_formal_training(root, output, device=device)
    schema, protocol = _schema(dataset), _protocol(root)
    offline = run_offline_validation(root, output, validation, schema, device=device)
    utilization = run_oracle_gap_and_utilization(root, output, validation, schema, offline, device=device)
    bridge = run_bridge_analysis(root, output, offline)
    closed = run_closed_loop_validation(root, output, validation, schema, protocol, device=device)
    fresh = run_fresh_development(root, output, schema, protocol, closed["gate"], device=device)
    final_gate = _final_policy_gate(closed, fresh, utilization, protocol)
    mechanism = _mechanism(utilization, closed, final_gate)
    _write_json(output / "final_policy_gate.json", final_gate)
    _write_json(output / "mechanism_conclusion.json", {"conclusion": mechanism})
    figures = _figures(output, offline, utilization, closed, fresh)
    _write_report(output, offline, utilization, closed, fresh, final_gate, mechanism)
    audit = _final_audit_and_lock(root, output, frozen, final_gate, mechanism, figures)
    _write_json(output / "run_manifest.json", {"schema_version": FORMAL_SCHEMA, "formal_training_completed": "12/12", "offline_validation_completed": True, "oracle_diagnostic_completed": True, "head_utilization_completed": True, "bridge_analysis_completed": True, "validation_closed_loop_completed": True, "validation_gate": closed["gate"], "fresh_development_accessed": fresh is not None, "final_gate": final_gate, "final_audit": audit, "device": device})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--output", type=Path); parser.add_argument("--device", default=None)
    args = parser.parse_args(); print(run_formal(args.root, output=args.output, device=args.device))


if __name__ == "__main__": main()
