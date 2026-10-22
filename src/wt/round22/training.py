"""Fixed twelve-run behavioral-cloning trainer for Round-22."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from .data import PolicyNormalizer, PolicySequences
from .models import ORACLE_TRANSFORMER, PREDICTED_MLP, STATE_ONLY_TRANSFORMER, DeepSetsPolicy, PolicySchema, build_policy


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    batch_size: int = 128
    max_epochs: int = 120
    early_stopping_patience: int = 20
    device: str = "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _phi(family: str, batch: Mapping[str, Tensor]) -> Tensor | None:
    if family == STATE_ONLY_TRANSFORMER:
        return None
    return batch["oracle_phi"] if family == ORACLE_TRANSFORMER else batch["predicted_phi"]


def action_mse(prediction: Tensor, target: Tensor, valid_uavs: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or valid_uavs.shape != prediction.shape[:2]:
        raise ValueError("Round-22 action loss shape mismatch")
    weight = valid_uavs.unsqueeze(-1).to(prediction.dtype)
    return (((prediction - target) ** 2) * weight).sum() / (weight.sum() * prediction.shape[-1]).clamp_min(1.0)


def _indices(sequences: PolicySequences, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    meta = sequences.metadata
    groups: dict[tuple[str, str, int, str], list[int]] = {}
    for index in range(len(sequences)):
        key = (str(meta["scenario_id"][index]), str(meta["behavior_generator"][index]), int(meta["num_buoys"][index]), str(meta["interference_tier"][index]))
        groups.setdefault(key, []).append(index)
    target = min(len(value) for value in groups.values())
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(value, size=target, replace=False) for _, value in sorted(groups.items())])
    rng.shuffle(selected)
    quality = np.asarray(meta["quality_tier"])
    audit = {f"quality_share_{tier}": float(np.mean(quality[selected] == tier)) for tier in sorted(set(quality.tolist()))}
    audit.update({"balanced_group_count": float(len(groups)), "samples_per_group": float(target), "metadata_used_as_input": 0.0})
    return selected.astype(np.int64), audit


def _forward(family: str, model: DeepSetsPolicy, batch: Mapping[str, Tensor]) -> Tensor:
    return model(uav_nodes=batch["uav_nodes"], buoy_nodes=batch["buoy_nodes"], uav_mask=batch["uav_mask"], buoy_mask=batch["buoy_mask"], global_features=batch["global_features"], attention_mask=batch["attention_mask"], phi=_phi(family, batch))


def predict(family: str, model: DeepSetsPolicy, sequences: PolicySequences, config: TrainingConfig) -> np.ndarray:
    result: list[np.ndarray] = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), config.batch_size):
            batch = sequences.batch(range(start, min(len(sequences), start + config.batch_size)), config.device)
            result.append(_forward(family, model, batch).cpu().numpy())
    return np.concatenate(result, axis=0)


def raw_rmse(prediction: np.ndarray, target: np.ndarray, valid_uavs: np.ndarray) -> float:
    valid = np.asarray(valid_uavs, dtype=bool)[..., None]
    return float(np.sqrt((((prediction - target) ** 2) * valid).sum() / max(int(valid.sum()) * prediction.shape[-1], 1)))


def train_one(family: str, seed: int, train: PolicySequences, validation: PolicySequences, config: TrainingConfig, checkpoint: Path, resume_from: Path | None = None) -> tuple[DeepSetsPolicy, PolicyNormalizer, list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    normalizer = PolicyNormalizer.fit(train)
    train_norm, validation_norm = normalizer.apply(train), normalizer.apply(validation)
    model = build_policy(family, PolicySchema()).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_rmse, best_epoch, stale, best_state = float("inf"), -1, 0, None
    history: list[dict[str, Any]] = []; start_epoch = 1
    if resume_from is not None and resume_from.is_file():
        resumed = torch.load(resume_from, map_location=config.device, weights_only=False)
        if resumed["family"] != family or int(resumed["seed"]) != seed:
            raise ValueError("resume checkpoint does not match the registered Round-22 run")
        model.load_state_dict(resumed["state_dict"]); optimizer.load_state_dict(resumed["optimizer_state_dict"])
        normalizer = PolicyNormalizer.from_serializable(resumed["normalizer"]); train_norm, validation_norm = normalizer.apply(train), normalizer.apply(validation)
        best_rmse, best_epoch, stale, best_state = float(resumed["best_validation_raw_joint_action_rmse"]), int(resumed["best_epoch"]), int(resumed["stale_epochs"]), resumed["best_state_dict"]
        history = list(resumed["history"]); start_epoch = int(resumed["epoch"]) + 1
    last_checkpoint = checkpoint.with_name("last.pt")
    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train(); indices, sampler = _indices(train_norm, seed * 10000 + epoch); losses = []
        for start in range(0, len(indices), config.batch_size):
            batch = train_norm.batch(indices[start:start + config.batch_size], config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = action_mse(_forward(family, model, batch), batch["actions"], batch["uav_mask"][:, -1])
            loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).detach().cpu()); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        predicted = predict(family, model, validation_norm, config)
        valid = validation_norm.uav_mask[:, -1]
        rmse = raw_rmse(predicted, validation_norm.actions, valid)
        row = {"epoch": epoch, "train_action_mse": float(np.mean(losses)), "validation_raw_joint_action_rmse": rmse, "gradient_norm_preclip": grad, **sampler}
        history.append(row)
        if np.isfinite(rmse) and rmse < best_rmse - 1.0e-12:
            best_rmse, best_epoch, stale = rmse, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"family": family, "seed": seed, "schema": PolicySchema().to_dict(), "training_config": config.__dict__, "normalizer": normalizer.serializable(), "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_state_dict": best_state, "best_epoch": best_epoch, "best_validation_raw_joint_action_rmse": best_rmse, "stale_epochs": stale, "epoch": epoch, "history": history, "deployable": model.deployable, "historical_action_input": False, "resume_supported": True}, last_checkpoint)
        if stale >= config.early_stopping_patience: break
    if best_state is None: raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"family": family, "seed": seed, "schema": PolicySchema().to_dict(), "training_config": config.__dict__, "normalizer": normalizer.serializable(), "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_epoch": best_epoch, "best_validation_raw_joint_action_rmse": best_rmse, "history": history, "deployable": model.deployable, "historical_action_input": False, "resume_supported": True, "last_checkpoint": last_checkpoint.as_posix()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    return model, normalizer, history, {"family": family, "seed": seed, "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": digest, "last_checkpoint": last_checkpoint.as_posix(), "last_checkpoint_sha256": hashlib.sha256(last_checkpoint.read_bytes()).hexdigest().upper(), "best_epoch": best_epoch, "best_validation_raw_joint_action_rmse": best_rmse, "stopped_epoch": len(history), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "deployable": model.deployable, "finite": bool(np.isfinite([row["train_action_mse"] for row in history]).all()), "resume_supported": True}


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[str, DeepSetsPolicy, PolicyNormalizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    schema = PolicySchema(**payload["schema"]); model = build_policy(payload["family"], schema).to(device); model.load_state_dict(payload["state_dict"]); model.eval()
    return str(payload["family"]), model, PolicyNormalizer.from_serializable(payload["normalizer"]), payload


__all__ = ["TrainingConfig", "action_mse", "load_checkpoint", "predict", "raw_rmse", "seed_everything", "train_one"]
