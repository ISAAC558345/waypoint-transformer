"""Minimal Round-4 spatial Waypoint MLP for the maritime task.

The module intentionally contains only model-level building blocks.  It does
not depend on PyTorch Lightning and does not provide a trainer, dataset,
policy, or Transformer integration.  The prediction target is the normalized
relative displacement ``(delta_x, delta_y)``; no absolute-position residual is
added to the network output.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn


ARCHITECTURE_VERSION = "spatial_waypoint_mlp_v1"
CHECKPOINT_VERSION = "spatial_waypoint_checkpoint_v1"
OUTPUT_DIM = 2
OUTPUT_SEMANTICS = "normalized_relative_delta_xy"


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a non-negative integer") from error
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def set_deterministic_seed(
    seed: int,
    *,
    deterministic: Optional[bool] = None,
    deterministic_algorithms: Optional[bool] = None,
) -> int:
    """Seed Python, NumPy, and Torch for deterministic model construction.

    This helper must be called before constructing the model.  Enabling Torch
    deterministic algorithms also configures CuDNN deterministically when the
    backend is available.  The normalized seed is returned for convenient
    recording in run/checkpoint metadata.
    """

    if (
        deterministic is not None
        and deterministic_algorithms is not None
        and bool(deterministic) != bool(deterministic_algorithms)
    ):
        raise ValueError(
            "deterministic and deterministic_algorithms must agree when both "
            "are provided"
        )
    deterministic_flag = (
        True
        if deterministic is None and deterministic_algorithms is None
        else bool(
            deterministic
            if deterministic is not None
            else deterministic_algorithms
        )
    )

    normalized_seed = _nonnegative_integer("seed", seed)
    if deterministic_flag:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(normalized_seed)
    np.random.seed(normalized_seed % (2**32))
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)

    torch.use_deterministic_algorithms(deterministic_flag)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_flag
        torch.backends.cudnn.benchmark = False
    return normalized_seed


def spatial_input_dim(
    schema: Mapping[str, Any],
    *,
    goal_conditioned: bool = False,
) -> int:
    """Resolve the model input width from a waypoint sample schema.

    The input contract is ``[state_t, goal?, valid_node_mask]``.  State-only
    input therefore uses ``state_dim + max_num_buoys``; goal-conditioned input
    additionally concatenates the schema's waypoint/goal width. Conditioning
    changes only ``input_dim``; it does not select a different architecture or
    forward path.
    """

    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")
    dimensions = schema.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("schema must contain a dimensions mapping")

    state_dim = _positive_integer("dimensions.state_dim", dimensions.get("state_dim"))
    valid_node_mask_dim = dimensions.get(
        "max_num_buoys", schema.get("max_num_buoys")
    )
    valid_node_mask_dim = _positive_integer(
        "dimensions.max_num_buoys", valid_node_mask_dim
    )
    if not goal_conditioned:
        return state_dim + valid_node_mask_dim

    goal_dim = dimensions.get("waypoint_dim", dimensions.get("goal_dim"))
    goal_dim = _positive_integer("dimensions.waypoint_dim", goal_dim)
    return state_dim + goal_dim + valid_node_mask_dim


def spatial_waypoint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return ``mean(sum((prediction - target)**2, dim=-1))`` exactly."""

    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch.Tensor values")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    if prediction.ndim != 2 or prediction.shape[-1] != OUTPUT_DIM:
        raise ValueError(
            f"prediction and target must have shape [B, {OUTPUT_DIM}]"
        )
    if prediction.shape[0] == 0:
        raise ValueError("prediction and target batch must be non-empty")
    return torch.mean(torch.sum((prediction - target) ** 2, dim=-1))


def normalized_displacement_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Alias for the canonical normalized-displacement regression loss."""

    return spatial_waypoint_loss(prediction, target)


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count all parameters, or only parameters that require gradients."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    parameters = model.parameters()
    if trainable_only:
        parameters = (value for value in parameters if value.requires_grad)
    return int(sum(value.numel() for value in parameters))


class SpatialWaypointMLP(nn.Module):
    """Predict a normalized relative spatial delta directly from flat input."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
        output_dim: int = OUTPUT_DIM,
        valid_node_mask_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.num_hidden_layers = _positive_integer(
            "num_hidden_layers", num_hidden_layers
        )
        self.output_dim = _positive_integer("output_dim", output_dim)
        if self.output_dim != OUTPUT_DIM:
            raise ValueError(
                f"SpatialWaypointMLP output_dim must be {OUTPUT_DIM}"
            )
        self.valid_node_mask_dim = _nonnegative_integer(
            "valid_node_mask_dim", valid_node_mask_dim
        )
        if self.valid_node_mask_dim >= self.input_dim:
            raise ValueError(
                "valid_node_mask_dim must be smaller than input_dim"
            )
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        modules = []
        width = self.input_dim
        for _ in range(self.num_hidden_layers):
            # Deliberately retain nn.Linear's default PyTorch initialization,
            # matching the official KForwardGoalNetwork initialization style.
            modules.extend(
                [
                    nn.Linear(width, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout),
                ]
            )
            width = self.hidden_dim
        modules.append(nn.Linear(width, OUTPUT_DIM))
        self.net = nn.Sequential(*modules)

    @classmethod
    def from_schema(
        cls,
        schema: Mapping[str, Any],
        *,
        goal_conditioned: bool = False,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
    ) -> "SpatialWaypointMLP":
        """Construct a state-only or goal-conditioned model from schema dims."""

        dimensions = schema.get("dimensions")
        if not isinstance(dimensions, Mapping):
            raise ValueError("schema must contain a dimensions mapping")
        valid_node_mask_dim = _positive_integer(
            "dimensions.max_num_buoys",
            dimensions.get("max_num_buoys", schema.get("max_num_buoys")),
        )
        return cls(
            input_dim=spatial_input_dim(
                schema, goal_conditioned=goal_conditioned
            ),
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            valid_node_mask_dim=valid_node_mask_dim,
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        expected_input_dim: Optional[int] = None,
    ) -> "SpatialWaypointMLP":
        """Construct an uninitialized-weight model from validated metadata."""

        architecture = validate_architecture_metadata(
            metadata, expected_input_dim=expected_input_dim
        )
        return cls(
            input_dim=architecture["input_dim"],
            hidden_dim=architecture["hidden_dim"],
            num_hidden_layers=architecture["num_hidden_layers"],
            dropout=architecture["dropout"],
            output_dim=architecture["output_dim"],
            valid_node_mask_dim=architecture["valid_node_mask_dim"],
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "features must have shape "
                f"[B, {self.input_dim}], got {tuple(features.shape)}"
            )
        # Direct relative-delta regression: intentionally no residual term and
        # no absolute-position addition.
        return self.net(features)

    def loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return spatial_waypoint_loss(prediction, target)

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    def architecture_metadata(self) -> Dict[str, Any]:
        """Return the complete architecture contract needed for reconstruction."""

        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "model_class": type(self).__name__,
            "input_dim": self.input_dim,
            "valid_node_mask_dim": self.valid_node_mask_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "activation": "ReLU",
            "dropout": self.dropout,
            "output_dim": self.output_dim,
            "output_activation": "identity",
            "output_semantics": OUTPUT_SEMANTICS,
            "absolute_position_residual": False,
        }

    def checkpoint_payload(
        self,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a framework-independent checkpoint payload in memory."""

        extra = {} if metadata is None else dict(metadata)
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": self.architecture_metadata(),
            "metadata": extra,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        expected_input_dim: Optional[int] = None,
        strict: bool = True,
    ) -> "SpatialWaypointMLP":
        """Validate an in-memory checkpoint and reconstruct its model."""

        architecture = validate_checkpoint_metadata(
            checkpoint, expected_input_dim=expected_input_dim
        )
        model = cls.from_metadata(
            architecture,
            expected_input_dim=expected_input_dim,
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint must contain a state_dict mapping")
        model.load_state_dict(state_dict, strict=strict)
        return model


def validate_architecture_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_input_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate and canonicalize SpatialWaypointMLP architecture metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("architecture metadata must be a mapping")
    if metadata.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("unsupported spatial waypoint architecture_version")
    if metadata.get("model_class") != "SpatialWaypointMLP":
        raise ValueError("checkpoint model_class is not SpatialWaypointMLP")
    if metadata.get("activation") != "ReLU":
        raise ValueError("SpatialWaypointMLP requires ReLU activation")
    if metadata.get("output_activation") != "identity":
        raise ValueError("SpatialWaypointMLP requires an identity output activation")
    if metadata.get("output_semantics") != OUTPUT_SEMANTICS:
        raise ValueError("checkpoint output semantics are incompatible")
    if metadata.get("absolute_position_residual") is not False:
        raise ValueError("spatial waypoint checkpoints must not use an absolute residual")
    if metadata.get("output_dim") != OUTPUT_DIM:
        raise ValueError(f"spatial waypoint output_dim must be {OUTPUT_DIM}")

    input_dim = _positive_integer("input_dim", metadata.get("input_dim"))
    if "valid_node_mask_dim" not in metadata:
        raise ValueError(
            "architecture metadata must contain valid_node_mask_dim"
        )
    valid_node_mask_dim = _nonnegative_integer(
        "valid_node_mask_dim", metadata.get("valid_node_mask_dim")
    )
    if valid_node_mask_dim >= input_dim:
        raise ValueError(
            "valid_node_mask_dim must be smaller than input_dim"
        )
    hidden_dim = _positive_integer("hidden_dim", metadata.get("hidden_dim"))
    num_hidden_layers = _positive_integer(
        "num_hidden_layers", metadata.get("num_hidden_layers")
    )
    dropout = float(metadata.get("dropout"))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if expected_input_dim is not None:
        expected = _positive_integer("expected_input_dim", expected_input_dim)
        if input_dim != expected:
            raise ValueError(
                "checkpoint input_dim conflicts with the current schema: "
                f"{input_dim} != {expected}"
            )

    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "model_class": "SpatialWaypointMLP",
        "input_dim": input_dim,
        "valid_node_mask_dim": valid_node_mask_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "activation": "ReLU",
        "dropout": dropout,
        "output_dim": OUTPUT_DIM,
        "output_activation": "identity",
        "output_semantics": OUTPUT_SEMANTICS,
        "absolute_position_residual": False,
    }


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_input_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate checkpoint version and return canonical architecture metadata."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported spatial waypoint checkpoint_version")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint must contain architecture metadata")
    return validate_architecture_metadata(
        architecture, expected_input_dim=expected_input_dim
    )


def rebuild_spatial_waypoint_model(
    checkpoint: Mapping[str, Any],
    *,
    expected_input_dim: Optional[int] = None,
    strict: bool = True,
) -> SpatialWaypointMLP:
    """Reconstruct a model from a validated in-memory checkpoint payload."""

    return SpatialWaypointMLP.from_checkpoint_payload(
        checkpoint,
        expected_input_dim=expected_input_dim,
        strict=strict,
    )


__all__ = [
    "ARCHITECTURE_VERSION",
    "CHECKPOINT_VERSION",
    "OUTPUT_DIM",
    "OUTPUT_SEMANTICS",
    "SpatialWaypointMLP",
    "count_parameters",
    "normalized_displacement_mse",
    "rebuild_spatial_waypoint_model",
    "set_deterministic_seed",
    "spatial_input_dim",
    "spatial_waypoint_loss",
    "validate_architecture_metadata",
    "validate_checkpoint_metadata",
]
