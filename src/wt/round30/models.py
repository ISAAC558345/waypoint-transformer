"""Parameter-shared DeepSets actor and centralized invariant critic."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .features import BUOY_DIM, GLOBAL_DIM, MAX_UAVS, UAV_DIM


def _mlp(inputs: int, outputs: int, width: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


@dataclass(frozen=True)
class DeepSetsPPOShape:
    uav_dim: int = UAV_DIM
    buoy_dim: int = BUOY_DIM
    global_dim: int = GLOBAL_DIM
    max_uavs: int = MAX_UAVS
    action_dim: int = 2
    width: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class SharedDeepSetsActor(nn.Module):
    """Equivariant shared actor: no IDs, graph edges, recurrence, or future input."""

    def __init__(self, shape: DeepSetsPPOShape = DeepSetsPPOShape()) -> None:
        super().__init__()
        self.shape = shape
        self.uav_encoder, self.buoy_encoder = _mlp(shape.uav_dim, shape.width), _mlp(shape.buoy_dim, shape.width)
        self.head = _mlp(3 * shape.width + shape.global_dim, 2 * shape.action_dim, shape.width)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor) -> tuple[Tensor, Tensor]:
        if uav.ndim != 3 or buoy.ndim != 3 or uav.shape[1:] != (self.shape.max_uavs, self.shape.uav_dim) or buoy.shape[2] != self.shape.buoy_dim:
            raise ValueError("Round-30 actor input has an invalid dynamic-set shape")
        h_u, h_b = self.uav_encoder(uav), self.buoy_encoder(buoy)
        g_u, g_b = masked_mean(h_u, uav_mask, 1), masked_mean(h_b, buoy_mask, 1)
        count = uav.shape[1]
        token = torch.cat((h_u, g_u[:, None].expand(-1, count, -1), g_b[:, None].expand(-1, count, -1), global_[:, None].expand(-1, count, -1)), dim=-1)
        values = self.head(token)
        mu, log_std = values[..., :2], values[..., 2:].clamp(-4.0, 1.0)
        return mu * uav_mask.unsqueeze(-1), log_std * uav_mask.unsqueeze(-1)


class CentralDeepSetsCritic(nn.Module):
    """Centralized but permutation-invariant current-state value function."""

    def __init__(self, shape: DeepSetsPPOShape = DeepSetsPPOShape()) -> None:
        super().__init__()
        self.shape = shape
        self.uav_encoder, self.buoy_encoder = _mlp(shape.uav_dim, shape.width), _mlp(shape.buoy_dim, shape.width)
        self.value = _mlp(2 * shape.width + shape.global_dim, 1, shape.width)

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor) -> Tensor:
        h_u, h_b = self.uav_encoder(uav), self.buoy_encoder(buoy)
        return self.value(torch.cat((masked_mean(h_u, uav_mask, 1), masked_mean(h_b, buoy_mask, 1), global_), dim=-1)).squeeze(-1)


class DeepSetsActorCritic(nn.Module):
    def __init__(self, shape: DeepSetsPPOShape = DeepSetsPPOShape()) -> None:
        super().__init__()
        self.shape, self.actor, self.critic = shape, SharedDeepSetsActor(shape), CentralDeepSetsCritic(shape)

    @property
    def uses_task_waypoint(self) -> bool: return False
    @property
    def uses_planner_trigger(self) -> bool: return False
    @property
    def uses_graph_message_passing(self) -> bool: return False
    @property
    def uses_transformer(self) -> bool: return False
    @property
    def uses_recurrence(self) -> bool: return False
    @property
    def shared_actor_parameters(self) -> bool: return True

    def forward(self, **features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_std = self.actor(features["uav"], features["buoy"], features["uav_mask"], features["buoy_mask"], features["global"])
        value = self.critic(features["uav"], features["buoy"], features["uav_mask"], features["buoy_mask"], features["global"])
        return mu, log_std, value


__all__ = ["CentralDeepSetsCritic", "DeepSetsActorCritic", "DeepSetsPPOShape", "SharedDeepSetsActor", "masked_mean"]
