"""Unified deterministic trainer for the four authorized Round-12 policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from wt.round5.causal_policy import masked_action_mse

from .contracts import ACTION_DIM, PREDICTOR_SHA256, assert_selected_predictor, sha256
from .evaluation import evaluate_offline_actions
from .policy_models import (
    ORACLE_TASK_WAYPOINT_TRANSFORMER,
    POLICY_FAMILIES,
    STATE_ONLY_TRANSFORMER,
    TASK_WAYPOINT_MLP,
    Round12PolicySchema,
    build_family,
    model_config,
)
from .sampler import ScenarioBehaviorBalancedSampler
from .sequence import Round12SequenceSplit


CHECKPOINT_SCHEMA = "round12_policy_checkpoint_v1"


@dataclass(frozen=True)
class Round12TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    seed: int = 0
    batches_per_epoch: int | None = None
    max_step_distance_m: float = 10.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("learning_rate, batch_size, and max_epochs must be positive")
        if self.patience < 1 or self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid Round-12 optimizer/early-stopping configuration")
        if self.batches_per_epoch is not None and self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive when specified")
        if self.max_step_distance_m <= 0.0:
            raise ValueError("max_step_distance_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    family: str
    seed: int
    best_epoch: int
    best_validation_raw_action_rmse: float
    last_epoch: int
    stopped_early: bool
    parameter_count: int
    best_checkpoint: str
    last_checkpoint: str
    history: tuple[Mapping[str, Any], ...]
    sampler_audit: Mapping[str, Any]
    predictor_sha256_before: str
    predictor_sha256_after: str


def set_deterministic_seed(seed: int) -> None:
    """Seed every random source used by a Round-12 training run."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _torch_load(path: Path, map_location: torch.device) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, Mapping):
        raise ValueError("Round-12 checkpoint payload must be a mapping")
    return value


class Round12PolicyTrainer:
    """Validation-RMSE-selected trainer with detached predictor provenance."""

    def __init__(
        self,
        family: str,
        schema: Round12PolicySchema,
        config: Round12TrainingConfig,
        *,
        root: Path,
        output_dir: Path,
        device: str | torch.device = "cpu",
    ) -> None:
        if family not in POLICY_FAMILIES:
            raise ValueError("unauthorized Round-12 policy family")
        self.family = family
        self.schema = schema
        self.config = config
        self.root = Path(root)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        set_deterministic_seed(config.seed)
        self.model = build_family(family, schema).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.predictor_path = assert_selected_predictor(self.root)
        self.predictor_sha256_before = sha256(self.predictor_path)
        if self.predictor_sha256_before != PREDICTOR_SHA256:
            raise RuntimeError("wrong predictor selected for Round-12 trainer")

    @property
    def best_path(self) -> Path:
        return self.output_dir / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.output_dir / "last.pt"

    def _assert_sequence_schema(self, sequence: Round12SequenceSplit) -> None:
        expected = (self.schema.state_dim, self.schema.max_buoys, self.schema.phi_dim, ACTION_DIM)
        actual = (sequence.state_dim, sequence.max_buoys, sequence.phi_dim, sequence.actions.shape[-1])
        if actual != expected:
            raise ValueError(f"sequence dimensions {actual} disagree with policy schema {expected}")

    def _tensor_batch(self, sequence: Round12SequenceSplit, indices: np.ndarray) -> dict[str, torch.Tensor]:
        values = sequence.numeric_arrays()
        return {
            name: torch.as_tensor(np.asarray(value)[indices], dtype=torch.float32, device=self.device)
            for name, value in values.items()
        }

    def _model_inputs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Construct the only permitted model inputs; labels never appear here."""
        if self.family == TASK_WAYPOINT_MLP:
            return {
                "states": batch["states"][:, -1],
                "node_masks": batch["node_masks"][:, -1],
                "conditions": batch["predicted_phi"][:, -1],
            }
        result: dict[str, torch.Tensor] = {
            "states": batch["states"],
            "node_masks": batch["node_masks"],
            "remaining_time": batch["remaining_time"],
            "attention_mask": batch["attention_mask"].bool(),
        }
        if self.family == STATE_ONLY_TRANSFORMER:
            # State-only policies intentionally receive no Phi tensor at all.
            return result
        if self.family == ORACLE_TASK_WAYPOINT_TRANSFORMER:
            result["conditions"] = batch["oracle_phi"]
        else:
            result["conditions"] = batch["predicted_phi"]
        return result

    def _targets_and_mask(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.family == TASK_WAYPOINT_MLP:
            return batch["actions"][:, -1], None
        return batch["actions"], batch["loss_mask"].bool()

    def _loss(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        prediction = self.model(**self._model_inputs(batch))
        target, loss_mask = self._targets_and_mask(batch)
        if prediction.shape != target.shape:
            raise ValueError("policy prediction/action-label shape mismatch")
        return masked_action_mse(prediction, target, loss_mask), prediction, loss_mask

    def _evaluate(self, sequence: Round12SequenceSplit) -> dict[str, Any]:
        self._assert_sequence_schema(sequence)
        self.model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequence), 1024):
                indices = np.arange(start, min(start + 1024, len(sequence)), dtype=np.int64)
                batch = self._tensor_batch(sequence, indices)
                prediction = self.model(**self._model_inputs(batch))
                target, loss_mask = self._targets_and_mask(batch)
                if loss_mask is None:
                    predictions.append(prediction.detach().cpu().numpy())
                    targets.append(target.detach().cpu().numpy())
                else:
                    valid = loss_mask.bool()
                    predictions.append(prediction[valid].detach().cpu().numpy())
                    targets.append(target[valid].detach().cpu().numpy())
        result = evaluate_offline_actions(
            np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0),
            max_step_distance_m=self.config.max_step_distance_m,
        )
        if not all(math.isfinite(float(value)) for value in result.values() if isinstance(value, (int, float))):
            raise FloatingPointError("non-finite Round-12 validation metric")
        return result

    def _payload(
        self,
        *,
        epoch: int,
        best_epoch: int,
        best_value: float,
        bad_epochs: int,
        history: list[dict[str, Any]],
        best_state: Mapping[str, torch.Tensor],
        sampler: ScenarioBehaviorBalancedSampler,
    ) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "family": self.family,
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_validation_raw_action_rmse": float(best_value),
            "bad_epochs": int(bad_epochs),
            "model_config": dict(model_config(self.model)),
            "parameter_count": count_parameters(self.model),
            "model_state_dict": self.model.state_dict(),
            "best_model_state_dict": dict(best_state),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_config": self.config.to_dict(),
            "history": list(history),
            "sampler_state": dict(sampler.state_dict()),
            "selection_split": "validation",
            "selection_metric": "raw_action_rmse_only",
            "old_development_test_used": False,
            "untouched_holdout_used": False,
            "predictor_sha256": PREDICTOR_SHA256,
            "historical_actions_as_input": False,
        }

    def _resume(self, path: Path, sampler: ScenarioBehaviorBalancedSampler) -> tuple[int, int, float, int, list[dict[str, Any]], Mapping[str, torch.Tensor]]:
        checkpoint = _torch_load(path, self.device)
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "family": self.family,
            "model_config": dict(model_config(self.model)),
            "training_config": self.config.to_dict(),
            "predictor_sha256": PREDICTOR_SHA256,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"resume checkpoint {key!r} differs from active trainer")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        sampler.load_state_dict(checkpoint["sampler_state"])
        return (
            int(checkpoint["epoch"]) + 1,
            int(checkpoint["best_epoch"]),
            float(checkpoint["best_validation_raw_action_rmse"]),
            int(checkpoint["bad_epochs"]),
            [dict(item) for item in checkpoint.get("history", [])],
            deepcopy(checkpoint["best_model_state_dict"]),
        )

    def fit(
        self,
        train: Round12SequenceSplit,
        validation: Round12SequenceSplit,
        *,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        self._assert_sequence_schema(train)
        self._assert_sequence_schema(validation)
        if train.split_name != "train" or validation.split_name != "validation":
            raise PermissionError("Round-12 trainer accepts exactly train and validation sequences")
        sampler = ScenarioBehaviorBalancedSampler(train, seed=self.config.seed)
        start_epoch, best_epoch, best_value, bad_epochs = 0, -1, float("inf"), 0
        history: list[dict[str, Any]] = []
        best_state: Mapping[str, torch.Tensor] = deepcopy(self.model.state_dict())
        if resume_from is not None:
            start_epoch, best_epoch, best_value, bad_epochs, history, best_state = self._resume(Path(resume_from), sampler)

        stopped_early = False
        last_epoch = start_epoch - 1
        batches = self.config.batches_per_epoch or math.ceil(len(train) / self.config.batch_size)
        for epoch in range(start_epoch, self.config.max_epochs):
            self.model.train()
            losses: list[float] = []
            gradients: list[float] = []
            sampled: list[int] = []
            for _ in range(batches):
                indices = sampler.sample_indices(self.config.batch_size)
                sampled.extend(indices.tolist())
                batch = self._tensor_batch(train, indices)
                self.optimizer.zero_grad(set_to_none=True)
                loss, _, _ = self._loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Round-12 policy loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                if not torch.isfinite(gradient):
                    raise FloatingPointError("non-finite Round-12 policy gradient")
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradients.append(float(gradient.detach().cpu()))
            validation_metrics = self._evaluate(validation)
            value = float(validation_metrics["raw_action_rmse"])
            improved = value < best_value
            if improved:
                best_epoch, best_value, bad_epochs = epoch, value, 0
                best_state = deepcopy(self.model.state_dict())
            else:
                bad_epochs += 1
            epoch_audit = sampler.audit(np.asarray(sampled, dtype=np.int64))
            history.append({
                "epoch": epoch,
                "train_normalized_action_mse": float(np.mean(losses)),
                "gradient_norm_preclip_max": float(np.max(gradients)),
                "validation": validation_metrics,
                "validation_raw_action_rmse": value,
                "selected_as_best": bool(improved),
                "sampler_audit": epoch_audit,
            })
            last_epoch = epoch
            payload = self._payload(
                epoch=epoch, best_epoch=best_epoch, best_value=best_value, bad_epochs=bad_epochs,
                history=history, best_state=best_state, sampler=sampler,
            )
            _atomic_torch_save(payload, self.last_path)
            if improved:
                best_payload = dict(payload)
                best_payload["checkpoint_role"] = "validation_selected_best"
                best_payload["model_state_dict"] = best_state
                _atomic_torch_save(best_payload, self.best_path)
            if bad_epochs >= self.config.patience:
                stopped_early = True
                break
        if best_epoch < 0 or not self.best_path.is_file():
            raise RuntimeError("Round-12 training ended without a validation-selected checkpoint")
        best_checkpoint = _torch_load(self.best_path, self.device)
        self.model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        predictor_after = sha256(self.predictor_path)
        if predictor_after != self.predictor_sha256_before:
            raise RuntimeError("frozen Round-11 predictor changed during policy training")
        return TrainingResult(
            family=self.family,
            seed=self.config.seed,
            best_epoch=best_epoch,
            best_validation_raw_action_rmse=best_value,
            last_epoch=last_epoch,
            stopped_early=stopped_early,
            parameter_count=count_parameters(self.model),
            best_checkpoint=str(self.best_path),
            last_checkpoint=str(self.last_path),
            history=tuple(history),
            sampler_audit=sampler.audit(),
            predictor_sha256_before=self.predictor_sha256_before,
            predictor_sha256_after=predictor_after,
        )

    def predict(self, sequence: Round12SequenceSplit) -> tuple[np.ndarray, np.ndarray]:
        """Return offline predictions and labels in matching [N,2] form."""
        self._assert_sequence_schema(sequence)
        self.model.eval()
        prediction_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequence), 1024):
                indices = np.arange(start, min(start + 1024, len(sequence)), dtype=np.int64)
                batch = self._tensor_batch(sequence, indices)
                predicted = self.model(**self._model_inputs(batch))
                target, loss_mask = self._targets_and_mask(batch)
                if loss_mask is None:
                    prediction_rows.append(predicted.cpu().numpy())
                    target_rows.append(target.cpu().numpy())
                else:
                    prediction_rows.append(predicted[loss_mask].cpu().numpy())
                    target_rows.append(target[loss_mask].cpu().numpy())
        return np.concatenate(prediction_rows), np.concatenate(target_rows)


def load_round12_checkpoint(
    path: Path,
    *,
    family: str,
    schema: Round12PolicySchema,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, Mapping[str, Any]]:
    """Strictly reload a policy checkpoint for smoke and formal verification."""
    device = torch.device(map_location)
    model = build_family(family, schema).to(device)
    checkpoint = _torch_load(Path(path), device)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported Round-12 policy checkpoint schema")
    if checkpoint.get("family") != family or checkpoint.get("model_config") != dict(model_config(model)):
        raise ValueError("Round-12 checkpoint model contract differs from requested model")
    if checkpoint.get("predictor_sha256") != PREDICTOR_SHA256:
        raise ValueError("Round-12 checkpoint is not tied to the selected predictor")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


__all__ = [
    "CHECKPOINT_SCHEMA",
    "Round12PolicyTrainer",
    "Round12TrainingConfig",
    "TrainingResult",
    "count_parameters",
    "load_round12_checkpoint",
    "set_deterministic_seed",
]
