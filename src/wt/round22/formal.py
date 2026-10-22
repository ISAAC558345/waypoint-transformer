"""Frozen 12-run Round-22 behavioral-cloning training matrix.

This module deliberately contains no model-selection, sweep, or dataset
construction logic.  It records the already-selected protocol, trains exactly
the registered family/seed matrix, and makes every finished run resumable
without rerunning an intact checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import FrozenDeepSetsPredictor, R20_OUTPUT, R22_OUTPUT, TRAIN, VALIDATION, load_policy_sequences, sha256
from .models import POLICY_FAMILIES
from .training import TrainingConfig, load_checkpoint, predict, train_one


ROOT = Path(".")
ROUND = Path("round_22")
CONFIG_PATH = ROUND / "configs" / "round22_policy_training_config.json"
FORMAL_DIR = R22_OUTPUT / "formal"
SEEDS = (0, 1, 2)
EXPECTED_PREDICTOR_SHA = "645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"cannot write an empty formal CSV: {path.name}")
    fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(materialized)


def _expected_config() -> dict[str, Any]:
    return {
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "batch_size": 128,
        "max_epochs": 120,
        "early_stopping_patience": 20,
    }


def _protocol() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = _expected_config()
    observed = {key: config["optimizer"][{"learning_rate": "learning_rate", "weight_decay": "weight_decay", "gradient_clip_norm": "gradient_clip_norm", "batch_size": "batch_size", "max_epochs": "max_epochs", "early_stopping_patience": "early_stopping_patience"}[key]] for key in expected}
    if observed != expected:
        raise RuntimeError(f"Round-22 frozen optimizer protocol differs: {observed}")
    if tuple(config["families"]) != POLICY_FAMILIES or tuple(config["seeds"]) != SEEDS or int(config["formal_run_count"]) != 12:
        raise RuntimeError("Round-22 formal family/seed matrix is no longer the registered 12-run matrix")
    if int(config["dataset"]["h_task"]) != 5 or int(config["architecture"]["context_length"]) != 10:
        raise RuntimeError("Round-22 horizon protocol has changed")
    required_architecture = {"hidden_dim": 128, "d_model": 128, "transformer_layers": 2, "attention_heads": 4, "ffn_dim": 512, "dropout": 0.1}
    if {key: config["architecture"][key] for key in required_architecture} != required_architecture:
        raise RuntimeError("Round-22 policy architecture protocol has changed")
    selection = json.loads((R22_OUTPUT / "selected_deepsets_task_waypoint_predictor.json").read_text(encoding="utf-8"))
    fresh = R22_OUTPUT / "fresh_multi_uav_policy_dev_manifest.json"
    cache = R22_OUTPUT / "predicted_task_waypoint_cache_manifest.json"
    if selection["checkpoint_sha256_before_policy_training"] != EXPECTED_PREDICTOR_SHA or sha256(Path(selection["checkpoint"])) != EXPECTED_PREDICTOR_SHA:
        raise RuntimeError("selected Round-21 predictor is not the registered frozen H5 checkpoint")
    cache_data = json.loads(cache.read_text(encoding="utf-8"))
    if cache_data["predictor_sha256"] != EXPECTED_PREDICTOR_SHA or cache_data["true_future_task_waypoint_stored"]:
        raise RuntimeError("policy cache provenance leaks true Task-Waypoint labels")
    return {
        "protocol_version": "round22_formal_policy_protocol_v1",
        "frozen_at": _now(),
        "training_config": CONFIG_PATH.as_posix(),
        "training_config_sha256": sha256(CONFIG_PATH),
        "round22_agents_sha256": sha256(ROUND / "AGENTS.md"),
        "round22_manifest_sha256": sha256(ROUND / "round_manifest.json"),
        "round20_lock_sha256": sha256(Path("round_20/ROUND_LOCK.json")),
        "round21_lock_sha256": sha256(Path("round_21/ROUND_LOCK.json")),
        "round20_transition_sha256": sha256(R20_OUTPUT / "multi_uav_transitions.jsonl"),
        "round20_task_samples_sha256": sha256(R20_OUTPUT / "multi_uav_task_waypoint_samples.jsonl"),
        "cache_manifest_sha256": sha256(cache),
        "fresh_manifest_sha256": sha256(fresh),
        "predictor_checkpoint": selection["checkpoint"],
        "predictor_sha256": EXPECTED_PREDICTOR_SHA,
        "families": list(POLICY_FAMILIES),
        "seeds": list(SEEDS),
        "formal_run_count": 12,
        "checkpoint_selection": "validation_raw_joint_movement_action_rmse_only",
        "train_split": TRAIN,
        "validation_split": VALIDATION,
        "fresh_dev_access_count_before_training": 0,
        "untouched_eval_model_access_count_before_training": 0,
    }


def _read_or_freeze_protocol() -> dict[str, Any]:
    path = R22_OUTPUT / "round22_protocol_freeze.json"
    observed = _protocol()
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        keys = [key for key in observed if key != "frozen_at"]
        if any(previous.get(key) != observed[key] for key in keys):
            raise RuntimeError("Round-22 protocol evidence changed after formal freeze")
        return previous
    _json(path, observed)
    return observed


def _run_id(family: str, seed: int) -> str:
    return f"{family}__seed_{seed}"


def _checkpoint_dir(family: str, seed: int) -> Path:
    return FORMAL_DIR / family / f"seed_{seed}"


def _reload_audit(checkpoint: Path, train, validation, config: TrainingConfig) -> dict[str, Any]:
    family, model, normalizer, payload = load_checkpoint(checkpoint, config.device)
    normalized = normalizer.apply(validation)
    first = predict(family, model, normalized, config)
    family2, model2, normalizer2, _ = load_checkpoint(checkpoint, config.device)
    second = predict(family2, model2, normalizer2.apply(validation), config)
    if family != family2 or not np.array_equal(first, second) or not np.isfinite(first).all():
        raise RuntimeError(f"formal checkpoint reload audit failed: {checkpoint}")
    if payload.get("historical_action_input") or payload.get("family") != family:
        raise RuntimeError("formal policy checkpoint violates the frozen model-input contract")
    return {"checkpoint_reload_exact": True, "prediction_sha256": hashlib.sha256(first.tobytes()).hexdigest().upper(), "prediction_shape": list(first.shape)}


def _existing_result(family: str, seed: int, train, validation, config: TrainingConfig) -> dict[str, Any] | None:
    directory = _checkpoint_dir(family, seed)
    checkpoint, last = directory / "best.pt", directory / "last.pt"
    if not (checkpoint.is_file() and last.is_file()):
        return None
    _, _, _, payload = load_checkpoint(checkpoint, config.device)
    if payload.get("family") != family or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"existing checkpoint identity mismatch: {checkpoint}")
    audit = _reload_audit(checkpoint, train, validation, config)
    history = list(payload.get("history", []))
    if not history:
        raise RuntimeError(f"existing checkpoint has no recorded training history: {checkpoint}")
    return {
        "run_id": _run_id(family, seed), "family": family, "seed": seed,
        "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": sha256(checkpoint),
        "last_checkpoint": last.as_posix(), "last_checkpoint_sha256": sha256(last),
        "best_epoch": int(payload["best_epoch"]),
        "best_validation_raw_joint_action_rmse": float(payload["best_validation_raw_joint_action_rmse"]),
        "stopped_epoch": len(history), "parameter_count": sum(item.numel() for item in load_checkpoint(checkpoint, config.device)[1].parameters()),
        "deployable": bool(payload["deployable"]), "finite": bool(np.isfinite([row["train_action_mse"] for row in history]).all()),
        "resume_supported": bool(payload.get("resume_supported")), "resumed_existing_checkpoint": True,
        "dataset_transition_sha256": sha256(R20_OUTPUT / "multi_uav_transitions.jsonl"),
        "dataset_task_samples_sha256": sha256(R20_OUTPUT / "multi_uav_task_waypoint_samples.jsonl"),
        "predictor_sha256": EXPECTED_PREDICTOR_SHA, **audit,
    }


def _write_progress(protocol: dict[str, Any], runs: list[dict[str, Any]], status: str) -> None:
    _json(R22_OUTPUT / "round22_formal_training_manifest.json", {
        "round": "round_22", "status": status, "formal_training_completed": f"{len(runs)}/12",
        "formal_run_count": 12, "completed_run_count": len(runs), "families": list(POLICY_FAMILIES), "seeds": list(SEEDS),
        "protocol_freeze": "round_22/outputs/round22_deepsets_task_waypoint_policy_20260829_1153/round22_protocol_freeze.json",
        "protocol_sha256": sha256(R22_OUTPUT / "round22_protocol_freeze.json"),
        "predictor_sha256_before": protocol["predictor_sha256"],
        "predictor_sha256_after": sha256(Path(protocol["predictor_checkpoint"])),
        "runs": runs, "training_started": True, "training_completed": len(runs) == 12,
    })
    _csv(R22_OUTPUT / "round22_policy_training_summary.csv", runs)
    _json(R22_OUTPUT / "policy_checkpoint_manifest.json", {"formal_training_completed": f"{len(runs)}/12", "predictor_sha256": protocol["predictor_sha256"], "checkpoints": runs})


def run_formal_training() -> list[dict[str, Any]]:
    """Train only the registered 4 x 3 policy matrix and audit each result."""
    protocol = _read_or_freeze_protocol()
    predictor = FrozenDeepSetsPredictor()
    if predictor.sha256 != protocol["predictor_sha256"]:
        raise RuntimeError("predictor changed before formal training")
    config = TrainingConfig()
    train, validation = load_policy_sequences(TRAIN, predictor), load_policy_sequences(VALIDATION, predictor)
    runs: list[dict[str, Any]] = []
    for family in POLICY_FAMILIES:
        for seed in SEEDS:
            existing = _existing_result(family, seed, train, validation, config)
            if existing is not None:
                runs.append(existing); continue
            directory = _checkpoint_dir(family, seed); checkpoint = directory / "best.pt"
            model, normalizer, history, result = train_one(family, seed, train, validation, config, checkpoint)
            _csv(directory / "training_history.csv", history)
            normalized = normalizer.apply(validation)
            expected = predict(family, model, normalized, config)
            audit = _reload_audit(checkpoint, train, validation, config)
            if hashlib.sha256(expected.tobytes()).hexdigest().upper() != audit["prediction_sha256"]:
                raise RuntimeError(f"formal checkpoint output differs from reloaded output: {checkpoint}")
            if not result["finite"] or predictor.sha256 != protocol["predictor_sha256"]:
                raise RuntimeError(f"formal run integrity failure: {_run_id(family, seed)}")
            runs.append({"run_id": _run_id(family, seed), **result, "resumed_existing_checkpoint": False,
                         "dataset_transition_sha256": protocol["round20_transition_sha256"], "dataset_task_samples_sha256": protocol["round20_task_samples_sha256"],
                         "predictor_sha256": predictor.sha256, **audit})
            _write_progress(protocol, runs, "training_in_progress")
    if predictor.sha256 != protocol["predictor_sha256"]:
        raise RuntimeError("predictor changed during formal training")
    if len(runs) != 12 or len({row["run_id"] for row in runs}) != 12:
        raise RuntimeError("formal run matrix is incomplete or duplicated")
    _write_progress(protocol, runs, "formal_training_complete")
    access = json.loads((R22_OUTPUT / "access_audit.json").read_text(encoding="utf-8"))
    access.update({"predictor_checkpoint_sha256_before": protocol["predictor_sha256"], "predictor_checkpoint_sha256_after": predictor.sha256})
    _json(R22_OUTPUT / "access_audit.json", access)
    return runs


def main() -> None:
    runs = run_formal_training()
    print(json.dumps({"formal_training_completed": f"{len(runs)}/12", "predictor_sha256": runs[0]["predictor_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
