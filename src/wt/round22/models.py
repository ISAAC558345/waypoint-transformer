"""Edge-free, shared-parameter DeepSets policy families for Round-22."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


PREDICTED_TRANSFORMER = "predicted_task_waypoint_transformer"
STATE_ONLY_TRANSFORMER = "state_only_transformer"
PREDICTED_MLP = "predicted_task_waypoint_mlp"
ORACLE_TRANSFORMER = "oracle_task_waypoint_transformer"
POLICY_FAMILIES = (PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, PREDICTED_MLP, ORACLE_TRANSFORMER)
DEPLOYABLE_FAMILIES = frozenset((PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, PREDICTED_MLP))


def _mlp(input_dim: int, output_dim: int, hidden: int = 128, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, output_dim))


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


@dataclass(frozen=True)
class PolicySchema:
    uav_dim: int = 8
    buoy_dim: int = 7
    global_dim: int = 6
    max_uavs: int = 2
    max_buoys: int = 5
    action_dim: int = 2
    context_length: int = 10
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    ffn: int = 512
    dropout: float = 0.1

    @property
    def phi_dim(self) -> int:
        return 2 * self.max_buoys + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepSetsStateEncoder(nn.Module):
    """Round-21-style shared node encoders and masked pooling, with no edges."""

    def __init__(self, schema: PolicySchema) -> None:
        super().__init__()
        self.schema = schema
        self.uav_encoder = _mlp(schema.uav_dim, schema.d_model, schema.d_model, schema.dropout)
        self.buoy_encoder = _mlp(schema.buoy_dim, schema.d_model, schema.d_model, schema.dropout)
        self.uav_set_transform = _mlp(schema.d_model, schema.d_model, schema.d_model, schema.dropout)
        self.buoy_set_transform = _mlp(schema.d_model, schema.d_model, schema.d_model, schema.dropout)
        self.token = _mlp(3 * schema.d_model + schema.global_dim, schema.d_model, schema.d_model, schema.dropout)

    def forward(self, uav_nodes: Tensor, buoy_nodes: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_features: Tensor) -> tuple[Tensor, Tensor]:
        if uav_nodes.ndim != 4 or buoy_nodes.ndim != 4:
            raise ValueError("Round-22 state encoder expects [B,H,M/F] node tensors")
        h_u = self.uav_set_transform(self.uav_encoder(uav_nodes)) * uav_mask.unsqueeze(-1).to(uav_nodes.dtype)
        h_b = self.buoy_set_transform(self.buoy_encoder(buoy_nodes)) * buoy_mask.unsqueeze(-1).to(buoy_nodes.dtype)
        g_u = masked_mean(h_u, uav_mask, dim=2)
        g_b = masked_mean(h_b, buoy_mask, dim=2)
        count = h_u.shape[2]
        token = self.token(torch.cat((h_u, g_u.unsqueeze(2).expand(-1, -1, count, -1), g_b.unsqueeze(2).expand(-1, -1, count, -1), global_features.unsqueeze(2).expand(-1, -1, count, -1)), dim=-1))
        return token * uav_mask.unsqueeze(-1).to(token.dtype), h_b


class TaskWaypointEncoder(nn.Module):
    """Per-UAV masked buoy-set pooling for [service, progress, team return]."""

    def __init__(self, schema: PolicySchema) -> None:
        super().__init__()
        self.schema = schema
        self.buoy_embedding = _mlp(schema.buoy_dim, schema.d_model, schema.d_model, schema.dropout)
        self.task_element = _mlp(schema.d_model + 2, schema.d_model, schema.d_model, schema.dropout)
        self.return_embedding = _mlp(1, schema.d_model, schema.d_model, schema.dropout)
        self.summary = _mlp(2 * schema.d_model, schema.d_model, schema.d_model, schema.dropout)

    def forward(self, buoy_nodes: Tensor, buoy_mask: Tensor, phi: Tensor) -> Tensor:
        if phi.shape[-1] != self.schema.phi_dim:
            raise ValueError("Task-Waypoint width is not H5 service/progress/return")
        bsz, steps, uavs, _ = phi.shape
        buoy = self.buoy_embedding(buoy_nodes).unsqueeze(2).expand(-1, -1, uavs, -1, -1)
        service, progress, team_return = phi[..., :self.schema.max_buoys], phi[..., self.schema.max_buoys:2 * self.schema.max_buoys], phi[..., -1:]
        elements = self.task_element(torch.cat((buoy, service.unsqueeze(-1), progress.unsqueeze(-1)), dim=-1))
        pooled = masked_mean(elements, buoy_mask.unsqueeze(2).expand(-1, -1, uavs, -1), dim=3)
        team = self.return_embedding(team_return)
        return self.summary(torch.cat((pooled, team), dim=-1))


class DeepSetsPolicy(nn.Module):
    """Shared per-UAV causal movement policy; action labels are never inputs."""

    def __init__(self, family: str, schema: PolicySchema = PolicySchema()) -> None:
        super().__init__()
        if family not in POLICY_FAMILIES:
            raise ValueError(f"unauthorized Round-22 family: {family}")
        self.family, self.schema = family, schema
        self.state_encoder = DeepSetsStateEncoder(schema)
        self.task_encoder = None if family == STATE_ONLY_TRANSFORMER else TaskWaypointEncoder(schema)
        if family == PREDICTED_MLP:
            self.mlp = nn.Sequential(nn.Linear(schema.d_model * 2, 256), nn.GELU(), nn.Dropout(schema.dropout), nn.Linear(256, 256), nn.GELU(), nn.Dropout(schema.dropout), nn.Linear(256, schema.action_dim), nn.Tanh())
            self.transformer = None
            self.action_head = None
        else:
            layer = nn.TransformerEncoderLayer(d_model=schema.d_model, nhead=schema.heads, dim_feedforward=schema.ffn, dropout=schema.dropout, activation="gelu", batch_first=True, norm_first=False)
            self.transformer = nn.TransformerEncoder(layer, num_layers=schema.layers, enable_nested_tensor=False)
            self.action_head = nn.Sequential(nn.LayerNorm(schema.d_model), nn.Linear(schema.d_model, schema.action_dim), nn.Tanh())
            self.mlp = None

    @property
    def deployable(self) -> bool:
        return self.family in DEPLOYABLE_FAMILIES

    @property
    def uses_historical_actions(self) -> bool:
        return False

    @property
    def uses_graph_message_passing(self) -> bool:
        return False

    def export_config(self) -> dict[str, Any]:
        return {"family": self.family, "schema": self.schema.to_dict(), "deployable": self.deployable, "historical_action_input": False, "edge_features": False}

    def _condition(self, buoy_nodes: Tensor, buoy_mask: Tensor, phi: Tensor | None, uav_count: int) -> Tensor:
        if self.family == STATE_ONLY_TRANSFORMER:
            if phi is not None:
                raise ValueError("State-only policy is forbidden from receiving Task-Waypoint input")
            return torch.zeros((buoy_nodes.shape[0], buoy_nodes.shape[1], uav_count, self.schema.d_model), dtype=buoy_nodes.dtype, device=buoy_nodes.device)
        if phi is None or self.task_encoder is None:
            raise ValueError("Task-Waypoint policy requires an explicitly supplied Phi")
        return self.task_encoder(buoy_nodes, buoy_mask, phi)

    def forward(self, *, uav_nodes: Tensor, buoy_nodes: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_features: Tensor, attention_mask: Tensor | None = None, phi: Tensor | None = None) -> Tensor:
        state, _ = self.state_encoder(uav_nodes, buoy_nodes, uav_mask, buoy_mask, global_features)
        task = self._condition(buoy_nodes, buoy_mask, phi, state.shape[2])
        token = state + task
        batch, steps, uavs, width = token.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch, steps), dtype=torch.bool, device=token.device)
        if attention_mask.shape != (batch, steps) or steps > self.schema.context_length:
            raise ValueError("invalid causal context mask")
        if self.family == PREDICTED_MLP:
            return self.mlp(torch.cat((state[:, -1], task[:, -1]), dim=-1)) * uav_mask[:, -1].unsqueeze(-1).to(token.dtype)
        shared = token.permute(0, 2, 1, 3).reshape(batch * uavs, steps, width)
        valid = attention_mask[:, None, :].expand(-1, uavs, -1).reshape(batch * uavs, steps)
        # Build a per-sequence causal/padding mask.  Padded queries attend only
        # to themselves so attention never receives an all-masked row (which
        # would create NaNs); valid queries cannot attend to padded keys.
        causal_allowed = torch.tril(torch.ones((steps, steps), dtype=torch.bool, device=token.device))
        allowed = valid[:, :, None] & valid[:, None, :] & causal_allowed
        eye = torch.eye(steps, dtype=torch.bool, device=token.device).unsqueeze(0)
        allowed = torch.where(valid[:, :, None], allowed, eye)
        blocked = (~allowed).repeat_interleave(self.schema.heads, dim=0)
        encoded = self.transformer(shared, mask=blocked)
        action = self.action_head(encoded[:, -1]).reshape(batch, uavs, self.schema.action_dim)
        return action * uav_mask[:, -1].unsqueeze(-1).to(action.dtype)


def build_policy(family: str, schema: PolicySchema = PolicySchema()) -> DeepSetsPolicy:
    return DeepSetsPolicy(family, schema)


def rollout_action(family: str, model: DeepSetsPolicy, **inputs: Tensor) -> Tensor:
    if family not in DEPLOYABLE_FAMILIES or not model.deployable:
        raise PermissionError("Oracle Task-Waypoint policy is offline-only and cannot enter a rollout")
    model.eval()
    with torch.no_grad():
        return model(**inputs)


__all__ = ["DEPLOYABLE_FAMILIES", "ORACLE_TRANSFORMER", "POLICY_FAMILIES", "PREDICTED_MLP", "PREDICTED_TRANSFORMER", "PolicySchema", "STATE_ONLY_TRANSFORMER", "DeepSetsPolicy", "build_policy", "masked_mean", "rollout_action"]
