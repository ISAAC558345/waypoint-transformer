"""Transformer Task-Waypoint V2 heads: goal softmax and H-step subgoal.

The encoder is a token Transformer over existing UAV/buoy node features.
It does not construct graphs, use edges, or call HGNN modules.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from wt.round30.features import BUOY_DIM, GLOBAL_DIM, MAX_BUOYS, MAX_UAVS, UAV_DIM

from .contract import DEFAULT_HORIZON, GOAL_DIM


@dataclass(frozen=True)
class TWPredictorConfig:
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 2
    ffn: int = 128
    dropout: float = 0.0
    waypoint_lambda: float = 1.0
    horizon: int = DEFAULT_HORIZON


class TaskWaypointV2Predictor(nn.Module):
    """state tokens -> (goal [M,K], waypoint [M,2])."""

    def __init__(self, config: TWPredictorConfig = TWPredictorConfig()) -> None:
        super().__init__()
        self.config = config
        self.uav_embed = nn.Linear(UAV_DIM + GLOBAL_DIM, config.d_model)
        self.buoy_embed = nn.Linear(BUOY_DIM + GLOBAL_DIM, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layer, enable_nested_tensor=False)
        self.goal_scorer = nn.Sequential(nn.Linear(2 * config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, 1))
        self.waypoint_head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, 2), nn.Tanh())

    def forward(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor) -> dict[str, Tensor]:
        batch = uav.shape[0]
        global_u = global_[:, None, :].expand(-1, MAX_UAVS, -1)
        global_b = global_[:, None, :].expand(-1, MAX_BUOYS, -1)
        u_tokens = self.uav_embed(torch.cat((uav, global_u), dim=-1))
        b_tokens = self.buoy_embed(torch.cat((buoy, global_b), dim=-1))
        tokens = torch.cat((u_tokens, b_tokens), dim=1)
        padding = torch.cat((uav_mask, buoy_mask), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~padding)
        h_u, h_b = encoded[:, :MAX_UAVS], encoded[:, MAX_UAVS:]
        pair = torch.cat((h_u[:, :, None, :].expand(-1, -1, MAX_BUOYS, -1), h_b[:, None, :, :].expand(-1, MAX_UAVS, -1, -1)), dim=-1)
        logits = self.goal_scorer(pair).squeeze(-1)
        logits = logits.masked_fill(~buoy_mask[:, None, :], -1.0e9)
        logits = logits.masked_fill(~uav_mask[:, :, None], -1.0e9)
        goal = torch.softmax(logits, dim=-1) * buoy_mask[:, None, :].to(logits.dtype) * uav_mask[:, :, None].to(logits.dtype)
        waypoint = self.waypoint_head(h_u) * uav_mask.unsqueeze(-1).to(h_u.dtype)
        return {
            "goal_distribution": goal,
            "waypoint_xy": waypoint,
            "goal_logits": logits,
            "horizon": torch.full((batch,), int(self.config.horizon), device=uav.device, dtype=torch.long),
        }


def task_waypoint_v2_loss(
    prediction: dict[str, Tensor],
    goal_target: Tensor,
    waypoint_target: Tensor,
    uav_mask: Tensor,
    goal_valid: Tensor,
    *,
    waypoint_lambda: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    log_goal = prediction["goal_distribution"].clamp_min(1.0e-8).log()
    valid = goal_valid.to(goal_target.dtype) * uav_mask.to(goal_target.dtype)
    goal_nll = -(goal_target * log_goal).sum(dim=-1)
    goal_loss = (goal_nll * valid).sum() / valid.sum().clamp_min(1.0)
    waypoint_err = (prediction["waypoint_xy"] - waypoint_target).square().sum(dim=-1)
    move_mask = uav_mask.to(waypoint_err.dtype)
    waypoint_loss = (waypoint_err * move_mask).sum() / move_mask.sum().clamp_min(1.0)
    total = goal_loss + float(waypoint_lambda) * waypoint_loss
    return total, {"goal_loss": goal_loss, "waypoint_loss": waypoint_loss, "total_loss": total}


__all__ = ["TWPredictorConfig", "TaskWaypointV2Predictor", "task_waypoint_v2_loss"]
