"""Causal trajectory policies used by the Round-5 offline experiments.

The Transformer deliberately reuses :class:`wt.transformer.GPTConfig` and
:class:`wt.transformer.Block`, which contain the audited causal and key-padding
attention implementation.  It does *not* instantiate the legacy ``GPT`` class:
there is exactly one state/condition token per physical timestep and no action
embedding or action-token pathway.

All feature dimensions are supplied by configuration.  In particular, none of
the current maritime dataset dimensions are encoded in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from wt.round5.official_transformer_core import Block, GPTConfig


TRANSFORMER_VARIANTS = frozenset(
    {
        "wt_transformer",
        "direct_goal_transformer",
        "state_only_transformer",
        "oracle_waypoint_transformer",
    }
)
DEPLOYABLE_TRANSFORMER_VARIANTS = frozenset(
    {
        "wt_transformer",
        "direct_goal_transformer",
        "state_only_transformer",
    }
)
MLP_VARIANTS = frozenset({"mlp_state_only", "mlp_waypoint"})


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class CausalPolicyConfig:
    """Schema-derived configuration for a causal trajectory policy."""

    variant: str
    state_dim: int
    node_mask_dim: int
    condition_dim: int
    action_dim: int
    context_length: int
    time_dim: int = 1
    n_embd: int = 128
    n_layer: int = 2
    n_head: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.variant not in TRANSFORMER_VARIANTS:
            raise ValueError(
                f"Unknown Transformer variant {self.variant!r}; expected one "
                f"of {sorted(TRANSFORMER_VARIANTS)}"
            )
        for name in (
            "state_dim",
            "action_dim",
            "context_length",
            "time_dim",
            "n_embd",
            "n_layer",
            "n_head",
        ):
            _positive(name, getattr(self, name))
        _nonnegative("node_mask_dim", self.node_mask_dim)
        _nonnegative("condition_dim", self.condition_dim)
        if self.variant == "state_only_transformer" and self.condition_dim != 0:
            raise ValueError("state_only_transformer requires condition_dim=0")
        if self.variant != "state_only_transformer" and self.condition_dim <= 0:
            raise ValueError(f"{self.variant} requires a positive condition_dim")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def deployable(self) -> bool:
        return self.variant in DEPLOYABLE_TRANSFORMER_VARIANTS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLPPolicyConfig:
    """Configuration for the one-step MLP controls."""

    variant: str
    state_dim: int
    node_mask_dim: int
    condition_dim: int
    action_dim: int
    time_dim: int = 1
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.variant not in MLP_VARIANTS:
            raise ValueError(
                f"Unknown MLP variant {self.variant!r}; expected one of "
                f"{sorted(MLP_VARIANTS)}"
            )
        for name in ("state_dim", "action_dim", "time_dim"):
            _positive(name, getattr(self, name))
        _nonnegative("node_mask_dim", self.node_mask_dim)
        _nonnegative("condition_dim", self.condition_dim)
        if self.variant == "mlp_state_only" and self.condition_dim != 0:
            raise ValueError("mlp_state_only requires condition_dim=0")
        if self.variant == "mlp_waypoint" and self.condition_dim <= 0:
            raise ValueError("mlp_waypoint requires a positive condition_dim")
        if not self.hidden_dims or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def deployable(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hidden_dims"] = list(self.hidden_dims)
        return result


PolicyConfig = Union[CausalPolicyConfig, MLPPolicyConfig]


_BATCH_ALIASES = {
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
}


def _lookup(batch: Mapping[str, Any], name: str, *, required: bool) -> Any:
    for alias in _BATCH_ALIASES[name]:
        if alias in batch:
            return batch[alias]
    if required:
        raise KeyError(
            f"Batch is missing {name!r}; accepted keys are "
            f"{_BATCH_ALIASES[name]}"
        )
    return None


def _unpack_inputs(
    states: Union[Tensor, Mapping[str, Any]],
    node_masks: Optional[Tensor],
    conditions: Optional[Tensor],
    remaining_time: Optional[Tensor],
    attention_mask: Optional[Tensor],
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor, Optional[Tensor]]:
    if isinstance(states, Mapping):
        if any(value is not None for value in (
            node_masks,
            conditions,
            remaining_time,
            attention_mask,
        )):
            raise ValueError(
                "Pass either a batch Mapping or explicit tensors, not both"
            )
        batch = states
        states = _lookup(batch, "states", required=True)
        node_masks = _lookup(batch, "node_masks", required=False)
        conditions = _lookup(batch, "conditions", required=False)
        remaining_time = _lookup(batch, "remaining_time", required=True)
        attention_mask = _lookup(batch, "attention_mask", required=False)
    if not isinstance(states, Tensor):
        raise TypeError("states must be a torch.Tensor or a tensor Mapping")
    if not isinstance(remaining_time, Tensor):
        raise TypeError("remaining_time must be a torch.Tensor")
    return states, node_masks, conditions, remaining_time, attention_mask


def _as_sequence(tensor: Tensor, name: str) -> tuple[Tensor, bool]:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1), True
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B,D] or [B,T,D]")
    return tensor, False


def _feature(
    tensor: Optional[Tensor],
    *,
    name: str,
    batch_size: int,
    sequence_length: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if feature_dim == 0:
        if tensor is not None and tensor.numel() != 0:
            raise ValueError(f"{name} is forbidden when its configured dimension is 0")
        return torch.empty(
            (batch_size, sequence_length, 0), device=device, dtype=dtype
        )
    if tensor is None:
        raise ValueError(f"{name} is required because {name}_dim={feature_dim}")
    tensor, _ = _as_sequence(tensor, name)
    expected = (batch_size, sequence_length, feature_dim)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


def _attention(
    mask: Optional[Tensor], batch_size: int, sequence_length: int, device: torch.device
) -> Tensor:
    if mask is None:
        return torch.ones((batch_size, sequence_length), device=device, dtype=torch.bool)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if tuple(mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            "attention_mask must have shape "
            f"{(batch_size, sequence_length)}, got {tuple(mask.shape)}"
        )
    return mask.to(device=device, dtype=torch.bool)


def _init_weights(module: nn.Module) -> None:
    """Use the official GPT initialisation convention."""

    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class CausalTrajectoryPolicy(nn.Module):
    """One-token-per-timestep causal Transformer action predictor.

    The model accepts only state-side inputs.  Action labels are intentionally
    absent from this signature and from every embedding module.
    """

    def __init__(self, config: CausalPolicyConfig):
        super().__init__()
        if not isinstance(config, CausalPolicyConfig):
            config = CausalPolicyConfig(**dict(config))
        self.config = config

        gpt_config = GPTConfig(
            vocab_size=config.action_dim,
            block_size=config.context_length,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            embd_pdrop=float(config.dropout),
            resid_pdrop=float(config.dropout),
            attn_pdrop=float(config.dropout),
            # These fields document and enforce the non-legacy topology.
            feature_dim=config.state_dim + config.node_mask_dim,
            model_actions=False,
            act_dim=config.action_dim,
        )
        self.gpt_config = gpt_config

        self.state_mask_encoder = nn.Sequential(
            nn.Linear(config.state_dim + config.node_mask_dim, config.n_embd),
            nn.Tanh(),
        )
        self.condition_encoder: Optional[nn.Module]
        if config.condition_dim:
            self.condition_encoder = nn.Sequential(
                nn.Linear(config.condition_dim, config.n_embd), nn.Tanh()
            )
        else:
            self.condition_encoder = None
        self.time_encoder = nn.Sequential(
            nn.Linear(config.time_dim, config.n_embd), nn.Tanh()
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length, config.n_embd)
        )
        self.input_dropout = nn.Dropout(config.dropout)
        # Reuse the official audited causal/padding Transformer implementation.
        self.blocks = nn.ModuleList(
            [Block(gpt_config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.action_head = nn.Sequential(
            nn.Linear(config.n_embd, config.action_dim), nn.Tanh()
        )

        self.apply(_init_weights)
        nn.init.zeros_(self.position_embedding)

    @property
    def deployable(self) -> bool:
        return self.config.deployable

    @property
    def uses_action_tokens(self) -> bool:
        return False

    def export_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def forward(
        self,
        states: Union[Tensor, Mapping[str, Any]],
        node_masks: Optional[Tensor] = None,
        conditions: Optional[Tensor] = None,
        remaining_time: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        states, node_masks, conditions, remaining_time, attention_mask = _unpack_inputs(
            states, node_masks, conditions, remaining_time, attention_mask
        )
        states, squeezed = _as_sequence(states, "states")
        batch_size, sequence_length, state_dim = states.shape
        if state_dim != self.config.state_dim:
            raise ValueError(
                f"states last dimension must be {self.config.state_dim}, got {state_dim}"
            )
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )
        dtype, device = states.dtype, states.device
        node_masks = _feature(
            node_masks,
            name="node_mask",
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=self.config.node_mask_dim,
            device=device,
            dtype=dtype,
        )
        conditions = _feature(
            conditions,
            name="condition",
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=self.config.condition_dim,
            device=device,
            dtype=dtype,
        )
        remaining_time = _feature(
            remaining_time,
            name="remaining_time",
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=self.config.time_dim,
            device=device,
            dtype=dtype,
        )
        attention_mask = _attention(
            attention_mask, batch_size, sequence_length, device
        )

        token = self.state_mask_encoder(torch.cat((states, node_masks), dim=-1))
        if self.condition_encoder is not None:
            token = token + self.condition_encoder(conditions)
        token = token + self.time_encoder(remaining_time)
        token = token + self.position_embedding[:, :sequence_length]
        x = self.input_dropout(token)
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        actions = self.action_head(x)
        actions = actions * attention_mask.unsqueeze(-1).to(actions.dtype)
        return actions[:, 0] if squeezed else actions


class OneStepMLPPolicy(nn.Module):
    """One-step state-only or waypoint-conditioned MLP control."""

    def __init__(self, config: MLPPolicyConfig):
        super().__init__()
        if not isinstance(config, MLPPolicyConfig):
            raw = dict(config)
            if "hidden_dims" in raw:
                raw["hidden_dims"] = tuple(raw["hidden_dims"])
            config = MLPPolicyConfig(**raw)
        self.config = config
        input_dim = (
            config.state_dim
            + config.node_mask_dim
            + config.condition_dim
            + config.time_dim
        )
        layers: list[nn.Module] = []
        previous = input_dim
        for width in config.hidden_dims:
            layers.extend(
                [nn.Linear(previous, int(width)), nn.GELU(), nn.Dropout(config.dropout)]
            )
            previous = int(width)
        layers.extend([nn.Linear(previous, config.action_dim), nn.Tanh()])
        self.network = nn.Sequential(*layers)
        self.apply(_init_weights)

    @property
    def deployable(self) -> bool:
        return True

    @property
    def uses_action_tokens(self) -> bool:
        return False

    def export_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def forward(
        self,
        states: Union[Tensor, Mapping[str, Any]],
        node_masks: Optional[Tensor] = None,
        conditions: Optional[Tensor] = None,
        remaining_time: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        states, node_masks, conditions, remaining_time, attention_mask = _unpack_inputs(
            states, node_masks, conditions, remaining_time, attention_mask
        )
        states, squeezed = _as_sequence(states, "states")
        batch_size, sequence_length, state_dim = states.shape
        if state_dim != self.config.state_dim:
            raise ValueError(
                f"states last dimension must be {self.config.state_dim}, got {state_dim}"
            )
        dtype, device = states.dtype, states.device
        node_masks = _feature(
            node_masks,
            name="node_mask",
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=self.config.node_mask_dim,
            device=device,
            dtype=dtype,
        )
        conditions = _feature(
            conditions,
            name="condition",
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=self.config.condition_dim,
            device=device,
            dtype=dtype,
        )
        remaining_time = _feature(
            remaining_time,
            name="remaining_time",
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=self.config.time_dim,
            device=device,
            dtype=dtype,
        )
        attention_mask = _attention(
            attention_mask, batch_size, sequence_length, device
        )
        features = torch.cat(
            (states, node_masks, conditions, remaining_time), dim=-1
        )
        actions = self.network(features)
        actions = actions * attention_mask.unsqueeze(-1).to(actions.dtype)
        return actions[:, 0] if squeezed else actions


def build_policy(config: Union[PolicyConfig, Mapping[str, Any]]) -> nn.Module:
    """Build a Round-5 policy from a typed config or plain mapping."""

    if isinstance(config, (CausalPolicyConfig, MLPPolicyConfig)):
        typed = config
    else:
        raw = dict(config)
        variant = raw.get("variant")
        if variant in TRANSFORMER_VARIANTS:
            typed = CausalPolicyConfig(**raw)
        elif variant in MLP_VARIANTS:
            if "hidden_dims" in raw:
                raw["hidden_dims"] = tuple(raw["hidden_dims"])
            typed = MLPPolicyConfig(**raw)
        else:
            raise ValueError(f"Unknown policy variant {variant!r}")
    if isinstance(typed, CausalPolicyConfig):
        return CausalTrajectoryPolicy(typed)
    return OneStepMLPPolicy(typed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return a stable parameter count for manifests and audits."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if (parameter.requires_grad or not trainable_only)
    )


def masked_action_mse(
    predictions: Tensor,
    targets: Tensor,
    loss_mask: Optional[Tensor] = None,
) -> Tensor:
    """Mean squared action-vector error over valid physical timesteps only.

    Masking happens before reduction.  Thus padding cannot affect either the
    numerator or denominator, including when padded target values are extreme.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must match, got {tuple(predictions.shape)} "
            f"and {tuple(targets.shape)}"
        )
    if predictions.ndim not in (2, 3):
        raise ValueError("actions must have shape [B,A] or [B,T,A]")
    token_shape = predictions.shape[:-1]
    if loss_mask is None:
        valid = torch.ones(token_shape, device=predictions.device, dtype=torch.bool)
    else:
        if loss_mask.ndim == predictions.ndim and loss_mask.shape[-1] == 1:
            loss_mask = loss_mask.squeeze(-1)
        if tuple(loss_mask.shape) != tuple(token_shape):
            raise ValueError(
                f"loss_mask must have shape {tuple(token_shape)}, got "
                f"{tuple(loss_mask.shape)}"
            )
        valid = loss_mask.to(device=predictions.device, dtype=torch.bool)
    valid_count = valid.sum()
    if int(valid_count.item()) == 0:
        raise ValueError("loss_mask selects zero valid timesteps")
    squared_l2 = ((predictions - targets.to(predictions)) ** 2).sum(dim=-1)
    return squared_l2.masked_select(valid).sum() / valid_count.to(squared_l2.dtype)


# Concise compatibility aliases for experiment scripts.
WaypointTransformerPolicy = CausalTrajectoryPolicy
TrajectoryMLPPolicy = OneStepMLPPolicy


__all__ = [
    "CausalPolicyConfig",
    "MLPPolicyConfig",
    "CausalTrajectoryPolicy",
    "WaypointTransformerPolicy",
    "OneStepMLPPolicy",
    "TrajectoryMLPPolicy",
    "TRANSFORMER_VARIANTS",
    "DEPLOYABLE_TRANSFORMER_VARIANTS",
    "MLP_VARIANTS",
    "build_policy",
    "count_parameters",
    "masked_action_mse",
]
