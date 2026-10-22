"""Fixed-config supervised training for Round-21; no policy or RL code."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .data import GraphNormalizer, TaskDataset
from .models import build_model


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 128
    graph_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    batch_size: int = 128
    max_epochs: int = 200
    early_stopping_patience: int = 20
    device: str = "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def dimensions(dataset: TaskDataset) -> dict[str, int]:
    return {
        "uav": int(dataset.uav_nodes.shape[-1]), "buoy": int(dataset.buoy_nodes.shape[-1]),
        "ub": int(dataset.ub_edges.shape[-1]), "uu": int(dataset.uu_edges.shape[-1]), "global": int(dataset.global_features.shape[-1]),
    }


def task_loss(prediction: Mapping[str, Tensor], batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    mask_pairs = batch["uav_mask"][:, :, None] & batch["buoy_mask"][:, None, :]
    service_mask = batch["service_valid"].bool() & batch["uav_mask"].bool()
    service_by_uav = -(batch["service_target"] * torch.log(prediction["service"].clamp_min(1.0e-8))).sum(dim=-1)
    service = (service_by_uav * service_mask.to(service_by_uav.dtype)).sum() / service_mask.sum().clamp_min(1)
    progress_by_pair = F.smooth_l1_loss(prediction["progress"], batch["progress_normalized"], reduction="none")
    progress = (progress_by_pair * mask_pairs.to(progress_by_pair.dtype)).sum() / mask_pairs.sum().clamp_min(1)
    team_return = F.smooth_l1_loss(prediction["team_return"], batch["return_normalized"], reduction="mean")
    total = service + progress + team_return
    return total, {"service_loss": service, "progress_loss": progress, "return_loss": team_return, "total_loss": total}


def _batches(count: int, batch_size: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, count, batch_size)]


def evaluate_loss(model: torch.nn.Module, dataset: TaskDataset, config: TrainConfig) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {name: [] for name in ("service_loss", "progress_loss", "return_loss", "total_loss")}
    with torch.no_grad():
        for indices in _batches(len(dataset), config.batch_size, shuffle=False, seed=0):
            batch = dataset.batch(indices, config.device)
            _, fields = task_loss(model(batch), batch)
            for name, value in fields.items():
                totals[name].append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def predict(model: torch.nn.Module, dataset: TaskDataset, config: TrainConfig) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {"service": [], "progress": [], "team_return": [], "h_u": [], "h_b": [], "h_g": []}
    with torch.no_grad():
        for indices in _batches(len(dataset), config.batch_size, shuffle=False, seed=0):
            output = model(dataset.batch(indices, config.device))
            for name in collected:
                collected[name].append(output[name].detach().cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in collected.items()}


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    values: dict[str, list[float]] = {"message_bu": [], "message_ub": [], "message_uu": [], "service_head": [], "progress_head": [], "return_head": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm().cpu())
        if "msg_bu" in name:
            values["message_bu"].append(value)
        elif "msg_ub" in name:
            values["message_ub"].append(value)
        elif "msg_uu" in name:
            values["message_uu"].append(value)
        elif "heads.service" in name:
            values["service_head"].append(value)
        elif "heads.progress" in name:
            values["progress_head"].append(value)
        elif "heads.team_return" in name:
            values["return_head"].append(value)
    return {name: float(np.mean(items)) if items else 0.0 for name, items in values.items()}


def train_run(
    family: str,
    seed: int,
    train: TaskDataset,
    validation: TaskDataset,
    config: TrainConfig,
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, list[dict[str, float]], GraphNormalizer, dict[str, Any]]:
    """Train exactly one prescribed run and select strictly by validation loss."""
    seed_everything(seed)
    normalizer = GraphNormalizer.fit(train)
    train_norm = normalizer.apply(train)
    validation_norm = normalizer.apply(validation)
    model = build_model(family, dimensions(train_norm), config.hidden_dim, config.graph_layers, config.dropout).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss: list[dict[str, float]] = []
        grad_values: dict[str, list[float]] = {}
        for batch_number, indices in enumerate(_batches(len(train_norm), config.batch_size, shuffle=True, seed=seed * 10000 + epoch)):
            batch = train_norm.batch(indices, config.device)
            optimizer.zero_grad(set_to_none=True)
            total, fields = task_loss(model(batch), batch)
            total.backward()
            gradient = _gradient_norms(model)
            for name, value in gradient.items():
                grad_values.setdefault(name, []).append(value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            epoch_loss.append({name: float(value.detach().cpu()) for name, value in fields.items()})
        validation_loss = evaluate_loss(model, validation_norm, config)
        record: dict[str, float] = {"epoch": float(epoch)}
        record.update({f"train_{name}": float(np.mean([row[name] for row in epoch_loss])) for name in epoch_loss[0]})
        record.update({f"validation_{name}": value for name, value in validation_loss.items()})
        record.update({f"gradient_{name}": float(np.mean(values)) for name, values in grad_values.items()})
        history.append(record)
        value = validation_loss["total_loss"]
        if np.isfinite(value) and value < best_loss - 1.0e-10:
            best_loss = value
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    head_guard = evaluate_loss(model, validation_norm, config)
    guard = {"all_head_losses_finite": all(np.isfinite(value) for value in head_guard.values()), "best_epoch": best_epoch, "best_validation_total_loss": best_loss, "stopped_epoch": len(history)}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "family": family, "seed": seed, "dimensions": dimensions(train_norm), "config": config.__dict__, "state_dict": model.state_dict(),
        "normalizer": normalizer.serializable(), "best_epoch": best_epoch, "best_validation": head_guard, "head_guard": guard,
    }, checkpoint_path)
    guard["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
    return model, history, normalizer, guard


def load_run(checkpoint_path: Path, device: str = "cpu") -> tuple[torch.nn.Module, GraphNormalizer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fields = payload["config"]
    model = build_model(payload["family"], payload["dimensions"], fields["hidden_dim"], fields["graph_layers"], fields["dropout"]).to(device)
    model.load_state_dict(payload["state_dict"])
    normalizer = GraphNormalizer(
        means={name: np.asarray(value, dtype=np.float32) for name, value in payload["normalizer"]["means"].items()},
        stds={name: np.asarray(value, dtype=np.float32) for name, value in payload["normalizer"]["stds"].items()},
    )
    return model, normalizer, payload


__all__ = ["TrainConfig", "dimensions", "evaluate_loss", "load_run", "predict", "seed_everything", "task_loss", "train_run"]
