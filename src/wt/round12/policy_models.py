"""The four, and only four, Round-12 policy families."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

from wt.round5.causal_policy import CausalPolicyConfig, CausalTrajectoryPolicy

from .contracts import ACTION_DIM, CONTEXT_LENGTH


PREDICTED_TASK_WAYPOINT_TRANSFORMER = "task_waypoint_transformer"
STATE_ONLY_TRANSFORMER = "state_only_transformer"
ORACLE_TASK_WAYPOINT_TRANSFORMER = "oracle_task_waypoint_transformer"
TASK_WAYPOINT_MLP = "task_waypoint_mlp"
POLICY_FAMILIES = (
    PREDICTED_TASK_WAYPOINT_TRANSFORMER,
    STATE_ONLY_TRANSFORMER,
    ORACLE_TASK_WAYPOINT_TRANSFORMER,
    TASK_WAYPOINT_MLP,
)
DEPLOYABLE_FAMILIES = frozenset(
    (PREDICTED_TASK_WAYPOINT_TRANSFORMER, STATE_ONLY_TRANSFORMER, TASK_WAYPOINT_MLP)
)


@dataclass(frozen=True)
class Round12PolicySchema:
    state_dim: int
    max_buoys: int
    action_dim: int = ACTION_DIM
    context_length: int = CONTEXT_LENGTH
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    ffn: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.max_buoys < 1:
            raise ValueError("state_dim and max_buoys must be positive")
        if self.action_dim != ACTION_DIM:
            raise ValueError("Round-12 action dimension is fixed at two")
        if self.context_length != CONTEXT_LENGTH:
            raise ValueError("Round-12 Transformer context is fixed at Hc=10")
        if (self.d_model, self.layers, self.heads, self.ffn) != (128, 2, 4, 512):
            raise ValueError("Round-12 Transformer architecture differs from its frozen config")

    @property
    def phi_dim(self) -> int:
        return self.max_buoys * 2 + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mask_phi(phi: torch.Tensor, buoy_mask: torch.Tensor, max_buoys: int) -> torch.Tensor:
    """Enforce zero service/progress values for non-existent buoys."""
    if phi.shape[-1] != max_buoys * 2 + 1:
        raise ValueError("Task-Waypoint dimension mismatch")
    if buoy_mask.shape != phi.shape[:-1] + (max_buoys,):
        raise ValueError("buoy mask does not match Task-Waypoint batch shape")
    service = phi[..., :max_buoys]
    progress = phi[..., max_buoys : 2 * max_buoys]
    ret = phi[..., -1:]
    mask = buoy_mask.to(phi.dtype)
    return torch.cat((service * mask, progress * mask, ret), dim=-1)


class TaskWaypointMLPPolicy(nn.Module):
    """Non-temporal policy: exactly current state, mask, and predicted Phi."""

    def __init__(self, schema: Round12PolicySchema) -> None:
        super().__init__()
        self.schema = schema
        width = schema.state_dim + schema.max_buoys + schema.phi_dim
        self.network = nn.Sequential(
            nn.Linear(width, 256), nn.GELU(), nn.Dropout(schema.dropout),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(schema.dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(schema.dropout),
            nn.Linear(128, schema.action_dim), nn.Tanh(),
        )

    @property
    def deployable(self) -> bool:
        return True

    @property
    def uses_action_tokens(self) -> bool:
        return False

    def export_config(self) -> dict[str, Any]:
        return {"family": TASK_WAYPOINT_MLP, **self.schema.to_dict(), "temporal": False}

    def forward(
        self,
        states: torch.Tensor,
        node_masks: torch.Tensor,
        conditions: torch.Tensor,
        **unexpected: torch.Tensor,
    ) -> torch.Tensor:
        if unexpected:
            raise ValueError("Round-12 MLP accepts no history, time feature, or attention mask")
        if states.ndim != 2 or states.shape[1] != self.schema.state_dim:
            raise ValueError("MLP states must have shape [B,state_dim]")
        if node_masks.shape != (states.shape[0], self.schema.max_buoys):
            raise ValueError("MLP node_masks have an invalid shape")
        if conditions.shape != (states.shape[0], self.schema.phi_dim):
            raise ValueError("MLP conditions have an invalid shape")
        return self.network(torch.cat((states, node_masks.to(states.dtype), conditions), dim=-1))


def build_family(name: str, schema: Round12PolicySchema) -> nn.Module:
    """Build one explicitly authorized family; no aliases are accepted."""
    if name not in POLICY_FAMILIES:
        raise ValueError(f"unauthorized Round-12 policy family: {name!r}")
    if name == TASK_WAYPOINT_MLP:
        return TaskWaypointMLPPolicy(schema)
    condition_dim = 0 if name == STATE_ONLY_TRANSFORMER else schema.phi_dim
    variant = {
        PREDICTED_TASK_WAYPOINT_TRANSFORMER: "wt_transformer",
        STATE_ONLY_TRANSFORMER: "state_only_transformer",
        ORACLE_TASK_WAYPOINT_TRANSFORMER: "oracle_waypoint_transformer",
    }[name]
    return CausalTrajectoryPolicy(
        CausalPolicyConfig(
            variant=variant,
            state_dim=schema.state_dim,
            node_mask_dim=schema.max_buoys,
            condition_dim=condition_dim,
            action_dim=schema.action_dim,
            context_length=schema.context_length,
            n_embd=schema.d_model,
            n_layer=schema.layers,
            n_head=schema.heads,
            dropout=schema.dropout,
        )
    )


def family_is_deployable(name: str) -> bool:
    if name not in POLICY_FAMILIES:
        raise ValueError(f"unknown Round-12 policy family: {name!r}")
    return name in DEPLOYABLE_FAMILIES


def rollout_action(name: str, model: nn.Module, **inputs: torch.Tensor) -> torch.Tensor:
    """Run a causal deployable policy, rejecting the Oracle by construction."""
    if not family_is_deployable(name):
        raise PermissionError("Oracle Task-Waypoint policy is offline/teacher-forced only")
    model.eval()
    with torch.no_grad():
        result = model(**inputs)
    if result.shape[-1] != ACTION_DIM:
        raise RuntimeError("Round-12 policy emitted an invalid action width")
    return result


def model_config(model: nn.Module) -> Mapping[str, Any]:
    """Stable checkpoint comparison data without serialising live modules."""
    exporter = getattr(model, "export_config", None)
    if callable(exporter):
        return dict(exporter())
    if isinstance(model, TaskWaypointMLPPolicy):
        return model.export_config()
    raise TypeError(f"Round-12 model has no serialisable configuration: {type(model).__name__}")


__all__ = [
    "DEPLOYABLE_FAMILIES",
    "ORACLE_TASK_WAYPOINT_TRANSFORMER",
    "POLICY_FAMILIES",
    "PREDICTED_TASK_WAYPOINT_TRANSFORMER",
    "Round12PolicySchema",
    "STATE_ONLY_TRANSFORMER",
    "TASK_WAYPOINT_MLP",
    "TaskWaypointMLPPolicy",
    "build_family",
    "family_is_deployable",
    "mask_phi",
    "model_config",
    "rollout_action",
]
