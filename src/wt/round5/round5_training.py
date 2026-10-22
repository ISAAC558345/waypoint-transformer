"""Deterministic pure-PyTorch training core for Round-5 policies.

Model selection is intentionally validation-only: :meth:`Round5Trainer.fit`
accepts a training loader and a validation loader, but has no test-loader
argument.  The separate evaluation pipeline may load the resulting immutable
best checkpoint once the validation-selected configuration has been frozen.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import BatchSampler, DataLoader, Dataset

from wt.round5.causal_policy import count_parameters, masked_action_mse


@dataclass(frozen=True)
class TrainingConfig:
    """Locked optimisation defaults for the Round-5 formal matrix."""

    seed: int
    action_scale: Union[float, Tuple[float, ...]]
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 0.0
    gradient_clip_norm: float = 1.0
    expected_policy_count: int = 4
    samples_per_epoch: Optional[int] = None
    deterministic_algorithms: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        for name in ("learning_rate", "gradient_clip_norm"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2 or not all(0.0 <= float(x) < 1.0 for x in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        for name in ("batch_size", "max_epochs", "patience", "expected_policy_count"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_size % self.expected_policy_count:
            raise ValueError(
                "batch_size must be divisible by expected_policy_count for "
                "policy-balanced formal sampling"
            )
        if self.samples_per_epoch is not None:
            if int(self.samples_per_epoch) <= 0:
                raise ValueError("samples_per_epoch must be positive")
            if int(self.samples_per_epoch) % int(self.batch_size):
                raise ValueError(
                    "samples_per_epoch must be divisible by batch_size so every "
                    "formal batch remains exactly policy-balanced"
                )
        scale = self.action_scale
        values = (float(scale),) if isinstance(scale, (int, float)) else tuple(scale)
        if not values or any(float(value) <= 0.0 for value in values):
            raise ValueError("action_scale must contain positive value(s)")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not isinstance(result["action_scale"], (int, float)):
            result["action_scale"] = list(result["action_scale"])
        result["betas"] = list(result["betas"])
        return result


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_raw_action_rmse_m: float
    last_epoch: int
    stopped_early: bool
    best_checkpoint: str
    last_checkpoint: str
    history: list[dict[str, Any]] = field(default_factory=list)
    sampler_counts_by_epoch: list[dict[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyBalancedBatchSampler(BatchSampler):
    """Deterministic equal-policy batch sampler with controlled oversampling.

    Every yielded batch contains exactly ``batch_size / n_policies`` samples
    from each policy.  A policy is reshuffled independently whenever its index
    stream is exhausted, so minority policies are oversampled without silently
    dropping majority-policy transitions.  ``seed + epoch`` determines the
    complete sequence.
    """

    def __init__(
        self,
        policy_labels: Sequence[Any],
        batch_size: int,
        *,
        seed: int,
        samples_per_epoch: Optional[int] = None,
        expected_policy_count: Optional[int] = None,
    ) -> None:
        labels = np.asarray([str(value) for value in policy_labels], dtype=object)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("policy_labels must be a non-empty one-dimensional sequence")
        self.policy_names = tuple(sorted(set(labels.tolist())))
        if expected_policy_count is not None and len(self.policy_names) != int(
            expected_policy_count
        ):
            raise ValueError(
                f"Expected {expected_policy_count} collection policies, got "
                f"{len(self.policy_names)}: {self.policy_names}"
            )
        self.batch_size = int(batch_size)
        if self.batch_size <= 0 or self.batch_size % len(self.policy_names):
            raise ValueError("batch_size must be positive and divisible by policy count")
        self.per_policy_per_batch = self.batch_size // len(self.policy_names)
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.indices_by_policy = {
            name: np.flatnonzero(labels == name).astype(np.int64)
            for name in self.policy_names
        }
        if any(len(indices) == 0 for indices in self.indices_by_policy.values()):
            raise ValueError("Every policy must have at least one sample")

        if samples_per_epoch is None:
            maximum = max(len(indices) for indices in self.indices_by_policy.values())
            self.num_batches = int(math.ceil(maximum / self.per_policy_per_batch))
        else:
            samples_per_epoch = int(samples_per_epoch)
            if samples_per_epoch <= 0 or samples_per_epoch % self.batch_size:
                raise ValueError(
                    "samples_per_epoch must be positive and divisible by batch_size"
                )
            self.num_batches = samples_per_epoch // self.batch_size
        self.epoch = 0
        self.last_counts: dict[str, int] = {name: 0 for name in self.policy_names}

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        streams = {
            name: rng.permutation(indices)
            for name, indices in self.indices_by_policy.items()
        }
        cursors = {name: 0 for name in self.policy_names}
        self.last_counts = {name: 0 for name in self.policy_names}

        def take(name: str, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                stream = streams[name]
                cursor = cursors[name]
                if cursor >= len(stream):
                    stream = rng.permutation(self.indices_by_policy[name])
                    streams[name] = stream
                    cursor = 0
                amount = min(count - len(selected), len(stream) - cursor)
                selected.extend(int(value) for value in stream[cursor : cursor + amount])
                cursors[name] = cursor + amount
            return selected

        for _ in range(self.num_batches):
            batch: list[int] = []
            for name in self.policy_names:
                batch.extend(take(name, self.per_policy_per_batch))
                self.last_counts[name] += self.per_policy_per_batch
            order = rng.permutation(len(batch))
            yield [batch[int(index)] for index in order]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "round5_policy_balanced_sampler_v1",
            "epoch": self.epoch,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "policy_names": list(self.policy_names),
            "source_counts": {
                name: len(indices) for name, indices in self.indices_by_policy.items()
            },
            "last_counts": dict(self.last_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state.get("policy_names", ())) != self.policy_names:
            raise ValueError("Sampler policy names differ from the checkpoint")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("Sampler seed differs from the checkpoint")
        if int(state.get("batch_size", -1)) != self.batch_size:
            raise ValueError("Sampler batch size differs from the checkpoint")
        self.epoch = int(state["epoch"])


def _discover_policy_labels(dataset: Dataset) -> Sequence[Any]:
    for name in ("policy_labels", "policy_names", "policy_ids"):
        if hasattr(dataset, name):
            values = getattr(dataset, name)
            if len(values) == len(dataset):
                return values
    raise ValueError(
        "policy_labels were not supplied and the dataset exposes none of "
        "policy_labels/policy_names/policy_ids"
    )


def make_policy_balanced_loader(
    dataset: Dataset,
    config: TrainingConfig,
    *,
    policy_labels: Optional[Sequence[Any]] = None,
    collate_fn: Optional[Callable[[list[Any]], Any]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Construct the formal deterministic policy-balanced training loader."""

    labels = policy_labels if policy_labels is not None else _discover_policy_labels(dataset)
    if len(labels) != len(dataset):
        raise ValueError("policy_labels length must match the dataset")
    sampler = PolicyBalancedBatchSampler(
        labels,
        config.batch_size,
        seed=config.seed,
        samples_per_epoch=config.samples_per_epoch,
        expected_policy_count=config.expected_policy_count,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )


_INPUT_ALIASES = {
    "states": ("states", "state", "observations", "observation"),
    "node_masks": ("node_masks", "node_mask", "masks", "mask_features"),
    "conditions": (
        "conditions",
        "condition",
        "waypoints",
        "waypoint",
        "goals",
        "goal",
    ),
    "remaining_time": (
        "remaining_time",
        "remaining_times",
        "time_features",
        "time_feature",
    ),
    "attention_mask": (
        "attention_mask",
        "attention_masks",
        "valid_mask",
        "padding_mask",
    ),
    "targets": (
        "actions",
        "action",
        "action_targets",
        "targets",
        "normalized_actions",
        "actions_normalized",
    ),
    "loss_mask": ("loss_mask", "loss_masks", "label_mask", "target_mask"),
}


def _batch_value(
    batch: Mapping[str, Any], name: str, *, required: bool
) -> Any:
    for alias in _INPUT_ALIASES[name]:
        if alias in batch:
            return batch[alias]
    if required:
        raise KeyError(
            f"Batch is missing {name!r}; accepted keys are {_INPUT_ALIASES[name]}"
        )
    return None


def _tensor(value: Any, device: torch.device, *, boolean: bool = False) -> Tensor:
    if isinstance(value, Tensor):
        result = value.to(device=device)
    else:
        result = torch.as_tensor(value, device=device)
    return result.bool() if boolean else result.float()


def prepare_batch(
    batch: Mapping[str, Any], model: nn.Module, device: torch.device
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Separate model inputs from labels and move tensors to ``device``.

    This function is the input-leakage boundary.  It constructs a fresh
    dictionary containing only state-side fields; neither current nor historic
    action labels can reach ``model.forward``.
    """

    if not isinstance(batch, Mapping):
        raise TypeError("Round5 training batches must implement Mapping")
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise ValueError("Round5 models must expose their schema config")

    inputs: dict[str, Tensor] = {
        "states": _tensor(_batch_value(batch, "states", required=True), device),
        "remaining_time": _tensor(
            _batch_value(batch, "remaining_time", required=True), device
        ),
    }
    if int(getattr(model_config, "node_mask_dim", 0)) > 0:
        inputs["node_masks"] = _tensor(
            _batch_value(batch, "node_masks", required=True), device
        )
    if int(getattr(model_config, "condition_dim", 0)) > 0:
        inputs["conditions"] = _tensor(
            _batch_value(batch, "conditions", required=True), device
        )
    attention = _batch_value(batch, "attention_mask", required=False)
    if attention is not None:
        inputs["attention_mask"] = _tensor(attention, device, boolean=True)

    targets = _tensor(_batch_value(batch, "targets", required=True), device)
    loss_mask = _batch_value(batch, "loss_mask", required=False)
    if loss_mask is None:
        loss_mask = attention
    if loss_mask is None:
        loss_mask_tensor = torch.ones(
            targets.shape[:-1], device=device, dtype=torch.bool
        )
    else:
        loss_mask_tensor = _tensor(loss_mask, device, boolean=True)
        if loss_mask_tensor.ndim == targets.ndim and loss_mask_tensor.shape[-1] == 1:
            loss_mask_tensor = loss_mask_tensor.squeeze(-1)
    return inputs, targets, loss_mask_tensor


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _torch_load(path: Union[str, Path], map_location: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility.
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _model_config(model: nn.Module) -> dict[str, Any]:
    if hasattr(model, "export_config"):
        return dict(model.export_config())
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Round5 model does not expose export_config/config")
    return asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)


def _sampler_from_loader(loader: DataLoader) -> Optional[PolicyBalancedBatchSampler]:
    candidate = getattr(loader, "batch_sampler", None)
    if isinstance(candidate, PolicyBalancedBatchSampler):
        return candidate
    candidate = getattr(loader, "sampler", None)
    return candidate if isinstance(candidate, PolicyBalancedBatchSampler) else None


def _set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    for candidate in (
        getattr(loader, "batch_sampler", None),
        getattr(loader, "sampler", None),
        getattr(loader, "dataset", None),
    ):
        method = getattr(candidate, "set_epoch", None)
        if callable(method):
            method(epoch)


class Round5Trainer:
    """Validation-selected deterministic trainer with resumable checkpoints."""

    checkpoint_schema = "round5_policy_checkpoint_v1"

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        output_dir: Union[str, Path],
        *,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.best_path = self.checkpoint_dir / "best.pt"
        self.last_path = self.checkpoint_dir / "last.pt"
        self.provenance = dict(provenance or {})
        self.device = _resolve_device(config.device)
        seed_everything(config.seed, config.deterministic_algorithms)
        self.model.to(self.device)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            betas=tuple(float(value) for value in config.betas),
        )

    def _action_scale(self, action_dim: int, dtype: torch.dtype) -> Tensor:
        raw = self.config.action_scale
        values = [float(raw)] if isinstance(raw, (int, float)) else [float(x) for x in raw]
        if len(values) == 1:
            values *= action_dim
        if len(values) != action_dim:
            raise ValueError(
                f"action_scale has {len(values)} values but action_dim={action_dim}"
            )
        return torch.tensor(values, device=self.device, dtype=dtype)

    def evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate a loader without changing selection or optimiser state."""

        self.model.eval()
        squared_normalized = 0.0
        squared_raw_l2 = 0.0
        absolute_raw_l2 = 0.0
        valid_tokens = 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets, loss_mask = prepare_batch(batch, self.model, self.device)
                predictions = self.model(**inputs)
                if predictions.shape != targets.shape:
                    raise ValueError(
                        f"Model output {tuple(predictions.shape)} does not match "
                        f"targets {tuple(targets.shape)}"
                    )
                if loss_mask.ndim == targets.ndim and loss_mask.shape[-1] == 1:
                    loss_mask = loss_mask.squeeze(-1)
                if tuple(loss_mask.shape) != tuple(targets.shape[:-1]):
                    raise ValueError("loss_mask shape does not match action timesteps")
                valid = loss_mask.bool()
                count = int(valid.sum().item())
                if count == 0:
                    continue
                difference = predictions - targets
                selected = difference[valid]
                scale = self._action_scale(targets.shape[-1], targets.dtype)
                raw = selected * scale
                squared_normalized += float((selected ** 2).sum().item())
                squared_raw_l2 += float((raw ** 2).sum().item())
                absolute_raw_l2 += float(torch.linalg.vector_norm(raw, dim=-1).sum().item())
                valid_tokens += count
        if valid_tokens == 0:
            raise ValueError("Evaluation loader contains no valid action targets")
        return {
            # This is E[||a_hat-a||_2^2], not a per-component mean.
            "normalized_action_mse": squared_normalized / valid_tokens,
            # Vector RMSE is the root mean squared physical displacement error.
            "raw_action_rmse_m": math.sqrt(squared_raw_l2 / valid_tokens),
            "raw_action_mae_l2_m": absolute_raw_l2 / valid_tokens,
            "valid_timestep_count": float(valid_tokens),
        }

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        weighted_loss = 0.0
        valid_tokens = 0
        for batch in loader:
            inputs, targets, loss_mask = prepare_batch(batch, self.model, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(**inputs)
            loss = masked_action_mse(predictions, targets, loss_mask)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.config.gradient_clip_norm)
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite policy gradient norm")
            self.optimizer.step()
            valid = int(loss_mask.bool().sum().item())
            weighted_loss += float(loss.detach().item()) * valid
            valid_tokens += valid
        if valid_tokens == 0:
            raise ValueError("Training loader contains no valid action targets")
        return {"normalized_action_mse": weighted_loss / valid_tokens}

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        best_epoch: int,
        best_value: float,
        bad_epochs: int,
        history: list[dict[str, Any]],
        sampler: Optional[PolicyBalancedBatchSampler],
    ) -> dict[str, Any]:
        return {
            "schema_version": self.checkpoint_schema,
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_validation_raw_action_rmse_m": float(best_value),
            "bad_epochs": int(bad_epochs),
            "model_class": type(self.model).__name__,
            "model_config": _model_config(self.model),
            "parameter_count": count_parameters(self.model),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_config": self.config.to_dict(),
            "provenance": dict(self.provenance),
            "history": list(history),
            "sampler_state": sampler.state_dict() if sampler is not None else None,
            "rng_state": capture_rng_state(),
            "selection_split": "validation",
            "test_used_for_selection": False,
            "action_labels_used_as_model_input": False,
        }

    def _resume(
        self,
        path: Union[str, Path],
        sampler: Optional[PolicyBalancedBatchSampler],
    ) -> tuple[int, int, float, int, list[dict[str, Any]]]:
        checkpoint = _torch_load(path, self.device)
        if checkpoint.get("schema_version") != self.checkpoint_schema:
            raise ValueError("Unsupported Round5 checkpoint schema")
        if checkpoint.get("training_config") != self.config.to_dict():
            raise ValueError("Training config differs from the resume checkpoint")
        if checkpoint.get("model_config") != _model_config(self.model):
            raise ValueError("Model config differs from the resume checkpoint")
        if checkpoint.get("provenance") != self.provenance:
            raise ValueError("Provenance differs from the resume checkpoint")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if sampler is not None and checkpoint.get("sampler_state") is not None:
            sampler.load_state_dict(checkpoint["sampler_state"])
        restore_rng_state(checkpoint["rng_state"])
        return (
            int(checkpoint["epoch"]) + 1,
            int(checkpoint["best_epoch"]),
            float(checkpoint["best_validation_raw_action_rmse_m"]),
            int(checkpoint["bad_epochs"]),
            list(checkpoint.get("history", [])),
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> TrainingResult:
        """Train and select an epoch using validation raw-action RMSE only."""

        sampler = _sampler_from_loader(train_loader)
        start_epoch = 0
        best_epoch = -1
        best_value = float("inf")
        bad_epochs = 0
        history: list[dict[str, Any]] = []
        if resume_from is not None:
            start_epoch, best_epoch, best_value, bad_epochs, history = self._resume(
                resume_from, sampler
            )

        sampler_audit: list[dict[str, int]] = [
            dict(row.get("train_sampler_counts", {}))
            for row in history
            if row.get("train_sampler_counts") is not None
        ]
        last_epoch = start_epoch - 1
        stopped_early = False

        for epoch in range(start_epoch, int(self.config.max_epochs)):
            _set_loader_epoch(train_loader, epoch)
            train_metrics = self._train_epoch(train_loader)
            validation_metrics = self.evaluate_loader(validation_loader)
            value = float(validation_metrics["raw_action_rmse_m"])
            improved = value < best_value - float(self.config.min_delta)
            if improved:
                best_value = value
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            counts = dict(sampler.last_counts) if sampler is not None else {}
            row = {
                "epoch": epoch,
                "train_normalized_action_mse": train_metrics["normalized_action_mse"],
                "validation_normalized_action_mse": validation_metrics[
                    "normalized_action_mse"
                ],
                "validation_raw_action_rmse_m": value,
                "validation_raw_action_mae_l2_m": validation_metrics[
                    "raw_action_mae_l2_m"
                ],
                "validation_valid_timestep_count": int(
                    validation_metrics["valid_timestep_count"]
                ),
                "train_sampler_counts": counts,
                "selected_as_best": bool(improved),
            }
            history.append(row)
            sampler_audit.append(counts)
            last_epoch = epoch

            payload = self._checkpoint_payload(
                epoch=epoch,
                best_epoch=best_epoch,
                best_value=best_value,
                bad_epochs=bad_epochs,
                history=history,
                sampler=sampler,
            )
            if improved:
                _atomic_torch_save(payload, self.best_path)
            _atomic_torch_save(payload, self.last_path)

            if bad_epochs >= int(self.config.patience):
                stopped_early = True
                break

        if best_epoch < 0 or not self.best_path.exists():
            raise RuntimeError("Training completed without a validation-selected checkpoint")
        best_checkpoint = _torch_load(self.best_path, self.device)
        self.model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        return TrainingResult(
            best_epoch=best_epoch,
            best_validation_raw_action_rmse_m=best_value,
            last_epoch=last_epoch,
            stopped_early=stopped_early,
            best_checkpoint=str(self.best_path),
            last_checkpoint=str(self.last_path),
            history=history,
            sampler_counts_by_epoch=sampler_audit,
        )


def load_policy_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Strictly restore a Round-5 model (and optionally optimiser/RNG)."""

    checkpoint = _torch_load(path, map_location)
    if checkpoint.get("schema_version") != Round5Trainer.checkpoint_schema:
        raise ValueError("Unsupported Round5 checkpoint schema")
    if checkpoint.get("model_config") != _model_config(model):
        raise ValueError("Checkpoint model config does not match the model")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint


PolicyTrainer = Round5Trainer


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "PolicyBalancedBatchSampler",
    "make_policy_balanced_loader",
    "prepare_batch",
    "seed_everything",
    "capture_rng_state",
    "restore_rng_state",
    "Round5Trainer",
    "PolicyTrainer",
    "load_policy_checkpoint",
]
