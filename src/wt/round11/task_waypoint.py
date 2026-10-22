"""Dynamic, non-spatial task-waypoint MLP and masked supervised losses."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TaskWaypointConfig:
    state_dim: int
    max_buoys: int
    hidden_dim: int = 256
    hidden_layers: int = 3
    dropout: float = 0.0
    progress_scale: float = 1.0
    return_mean: float = 0.0
    return_std: float = 1.0
    gamma: float = 0.99

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.max_buoys

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({
            "input_contract": "[observable_state, buoy_mask]",
            "goal_dim": 0,
            "constant_goal_used": False,
            "spatial_head_present": False,
            "output_dimensions": {
                "service_distribution": self.max_buoys,
                "gross_service_progress": self.max_buoys,
                "H_step_return": 1,
            },
        })
        return value


def masked_softmax(logits: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Softmax over real buoys only; padded outputs are exactly zero."""
    if logits.ndim != 2 or node_mask.shape != logits.shape:
        raise ValueError("logits and node_mask must share shape [B, max_buoys]")
    if not torch.all(node_mask.any(dim=1)):
        raise ValueError("each sample needs at least one real buoy")
    masked = logits.masked_fill(~node_mask.bool(), torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=-1).masked_fill(~node_mask.bool(), 0.0)


class TaskWaypointMLP(nn.Module):
    """Three-head MLP: service distribution, gross service, and H-return."""
    def __init__(self, config: TaskWaypointConfig) -> None:
        super().__init__()
        if config.state_dim <= 0 or config.max_buoys <= 0:
            raise ValueError("state_dim and max_buoys must be positive")
        if config.hidden_layers != 3 or config.hidden_dim != 256 or config.dropout != 0.0:
            raise ValueError("Round-11 architecture is exactly 3x Linear(256)+ReLU, dropout=0")
        if config.progress_scale <= 0 or config.return_std <= 0:
            raise ValueError("normalization scales must be positive")
        self.config = config
        layers: list[nn.Module] = []
        width = config.input_dim
        for _ in range(config.hidden_layers):
            layers += [nn.Linear(width, config.hidden_dim), nn.ReLU()]
            width = config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.service_head = nn.Linear(width, config.max_buoys)
        self.progress_head = nn.Linear(width, config.max_buoys)
        self.return_head = nn.Linear(width, 1)

    def forward(self, observable_state: torch.Tensor, node_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if observable_state.ndim != 2 or observable_state.shape[1] != self.config.state_dim:
            raise ValueError("observable_state shape does not match dynamic state_dim")
        if node_mask.ndim != 2 or node_mask.shape[1] != self.config.max_buoys:
            raise ValueError("node_mask shape does not match dynamic max_buoys")
        x = torch.cat([observable_state, node_mask.to(observable_state.dtype)], dim=-1)
        h = self.trunk(x)
        return {
            "service_distribution": masked_softmax(self.service_head(h), node_mask.bool()),
            # An H-step transmitted-payload total can exceed one instantaneous
            # queue capacity, so the configured capacity normalization is not
            # bounded to [0, 1].  Softplus gives the required non-negativity
            # without imposing a false sigmoid ceiling.
            "gross_service_progress": F.softplus(self.progress_head(h)).masked_fill(~node_mask.bool(), 0.0),
            "H_step_return": self.return_head(h).squeeze(-1),
        }


def task_waypoint_losses(
    prediction: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor], node_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """All reductions apply masks before reduction, including padding positions."""
    mask = node_mask.bool()
    service_valid = target["service_valid"].bool()
    service_pred = prediction["service_distribution"]
    service_target = target["service_distribution"]
    if service_pred.shape != service_target.shape or service_pred.shape != mask.shape:
        raise ValueError("service and node-mask shapes must agree")
    # A zero-service window never contributes a fabricated target or loss.
    if bool(service_valid.any()):
        p = service_pred[service_valid].clamp_min(1e-8)
        y = service_target[service_valid].masked_fill(~mask[service_valid], 0.0)
        service = -(y * p.log()).sum(dim=-1).mean()
    else:
        service = service_pred.sum() * 0.0
    progress_squared = (prediction["gross_service_progress"] - target["gross_service_progress"]) ** 2
    progress = progress_squared.masked_select(mask).mean()
    ret = F.mse_loss(prediction["H_step_return"], target["H_step_return"])
    total = service + progress + ret
    return {"total": total, "service": service, "progress": progress, "return": ret}


def numpy_metrics(
    prediction: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray], node_mask: np.ndarray, progress_scale: float, return_mean: float, return_std: float
) -> dict[str, float]:
    mask = node_mask.astype(bool)
    valid = target["service_valid"].astype(bool)
    p = np.clip(prediction["service_distribution"][valid], 1e-8, 1.0)
    y = target["service_distribution"][valid]
    service_ce = float(np.mean(-np.sum(y * np.log(p), axis=1))) if len(p) else 0.0
    progress_error = (prediction["gross_service_progress"] - target["gross_service_progress"])[mask]
    ret_error = prediction["H_step_return"] - target["H_step_return"]
    return {
        "service_cross_entropy": service_ce,
        "service_kl": float(np.mean(np.sum(y * (np.log(np.clip(y, 1e-8, 1.0)) - np.log(p)), axis=1))) if len(p) else 0.0,
        "service_l1": float(np.mean(np.abs(prediction["service_distribution"][valid] - y))) if len(p) else 0.0,
        "progress_mse_normalized": float(np.mean(progress_error ** 2)),
        "progress_rmse_bits": float(np.sqrt(np.mean((progress_error * progress_scale) ** 2))),
        "return_mse_standardized": float(np.mean(ret_error ** 2)),
        "return_rmse_raw": float(np.sqrt(np.mean((ret_error * return_std) ** 2))),
        "return_bias_raw": float(np.mean(ret_error * return_std)),
    }
