"""Physical input scales for the existing residual actor. Dimensions stay the same."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.observation import COMM_DIM, NEIGHBOR_EDGE_DIM, SELF_DIM, TASK_DIM
from wt.lowlevel_v2.residual_policy import ResidualActor

from .protocol import physical_scales


def _units(config: L3BConfig | None = None) -> dict[str, float]:
    return physical_scales(config)


def actor_feature_scales(config: L3BConfig | None = None) -> np.ndarray:
    """Column scales for self, task, communication, neighbor embedding, and nearest distance."""
    unit = _units(config)
    self_scale = np.array([unit["area_width_m"], unit["area_height_m"], 1.0, 1.0, 1.0], dtype=np.float64)
    task_scale = np.array(
        [
            1.0,
            unit["area_width_m"],
            unit["area_height_m"],
            unit["diagonal_m"],
            unit["queue_capacity_blocks"],
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    comm_scale = np.array(
        [unit["close_range_sinr"], unit["bandwidth_hz"], unit["tx_power"], unit["close_range_sinr"]],
        dtype=np.float64,
    )
    neighbor_embedding = np.ones(16, dtype=np.float64)
    nearest = np.array([unit["diagonal_m"]], dtype=np.float64)
    scales = np.concatenate([self_scale, task_scale, comm_scale, neighbor_embedding, nearest])
    expected = int(SELF_DIM + TASK_DIM + COMM_DIM + 16 + 1)
    if scales.shape != (expected,) or not np.all(scales > 0.0) or not np.all(np.isfinite(scales)):
        raise RuntimeError("actor feature scales are not a positive 34-vector")
    return scales


def neighbor_edge_scales(config: L3BConfig | None = None) -> np.ndarray:
    unit = _units(config)
    scales = np.array(
        [
            unit["area_width_m"],
            unit["area_height_m"],
            1.0,
            1.0,
            unit["diagonal_m"],
            1.0,
            1.0,
            unit["area_width_m"],
            unit["area_height_m"],
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    if scales.shape != (int(NEIGHBOR_EDGE_DIM),):
        raise RuntimeError("neighbor edge scale width drifted")
    return scales


def critic_feature_scales(config: L3BConfig | None = None) -> np.ndarray:
    """Same block order as build_critic_state."""
    cfg = L3BConfig() if config is None else config
    unit = _units(cfg)
    m_count = int(cfg.num_uavs)
    k_count = int(cfg.num_buoys)
    self_scale = np.tile(np.array([unit["area_width_m"], unit["area_height_m"], 1.0, 1.0, 1.0]), m_count)
    task_scale = np.tile(
        np.array(
            [
                1.0,
                unit["area_width_m"],
                unit["area_height_m"],
                unit["diagonal_m"],
                unit["queue_capacity_blocks"],
                1.0,
                1.0,
                1.0,
            ]
        ),
        m_count,
    )
    association = np.full(m_count, float(k_count), dtype=np.float64)
    queue = np.full(k_count, unit["queue_capacity_blocks"], dtype=np.float64)
    progress = np.ones(m_count, dtype=np.float64)
    sinr = np.full(m_count, unit["close_range_sinr"], dtype=np.float64)
    rate = np.full(m_count, unit["bandwidth_hz"], dtype=np.float64)
    distance = np.full(m_count * m_count, unit["diagonal_m"], dtype=np.float64)
    interference = np.full(m_count * m_count, unit["tx_power"], dtype=np.float64)
    clock = np.ones(2, dtype=np.float64)
    scales = np.concatenate(
        [self_scale, task_scale, association, queue, progress, sinr, rate, distance, interference, clock]
    )
    if not np.all(scales > 0.0):
        raise RuntimeError("critic scales must stay positive")
    return scales


def scale_actor_observation(obs: dict[str, Any], config: L3BConfig | None = None) -> dict[str, Any]:
    scaled = dict(obs)
    edge = neighbor_edge_scales(config)
    scaled["self"] = np.asarray(obs["self"], dtype=np.float64) / actor_feature_scales(config)[:SELF_DIM]
    task_scale = actor_feature_scales(config)[SELF_DIM : SELF_DIM + TASK_DIM]
    scaled["task"] = np.asarray(obs["task"], dtype=np.float64) / task_scale
    comm_scale = actor_feature_scales(config)[SELF_DIM + TASK_DIM : SELF_DIM + TASK_DIM + COMM_DIM]
    scaled["communication"] = np.asarray(obs["communication"], dtype=np.float64) / comm_scale
    scaled["neighbor_set"] = np.asarray(obs["neighbor_set"], dtype=np.float64) / edge
    scaled["d_nearest"] = np.asarray(obs["d_nearest"], dtype=np.float64) / actor_feature_scales(config)[-1]
    return scaled


def scale_critic_vector(vector: np.ndarray, config: L3BConfig | None = None) -> np.ndarray:
    scales = critic_feature_scales(config)
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != scales.shape:
        raise RuntimeError(f"critic vector width {values.shape} does not match scales {scales.shape}")
    return values / scales


def fold_scales_into_actor(actor: ResidualActor, config: L3BConfig | None = None) -> ResidualActor:
    """Make a raw-observation actor match one that was trained on divided inputs."""
    folded = ResidualActor(log_std_init=float(actor.log_std.detach().cpu().mean()))
    folded.load_state_dict(actor.state_dict())
    feature = torch.as_tensor(actor_feature_scales(config), dtype=folded.body[0].weight.dtype)
    edge = torch.as_tensor(neighbor_edge_scales(config), dtype=folded.neighbor.phi[0].weight.dtype)
    if int(folded.body[0].weight.shape[1]) != int(feature.numel()):
        raise RuntimeError("actor first layer does not match the feature scale")
    if int(folded.neighbor.phi[0].weight.shape[1]) != int(edge.numel()):
        raise RuntimeError("neighbor encoder does not match the edge scale")
    with torch.no_grad():
        folded.body[0].weight.copy_(folded.body[0].weight / feature.unsqueeze(0))
        folded.neighbor.phi[0].weight.copy_(folded.neighbor.phi[0].weight / edge.unsqueeze(0))
    return folded


def first_layer_abs_mean(actor: ResidualActor, obs: dict[str, Any]) -> float:
    packed = actor._pack(
        torch.as_tensor(obs["self"], dtype=torch.float32),
        torch.as_tensor(obs["task"], dtype=torch.float32),
        torch.as_tensor(obs["communication"], dtype=torch.float32),
        torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
        torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
    )
    layer = actor.body[0]
    pre = packed @ layer.weight.T + layer.bias
    return float(pre.detach().abs().mean().cpu())


__all__ = [
    "actor_feature_scales",
    "critic_feature_scales",
    "first_layer_abs_mean",
    "fold_scales_into_actor",
    "neighbor_edge_scales",
    "scale_actor_observation",
    "scale_critic_vector",
]
