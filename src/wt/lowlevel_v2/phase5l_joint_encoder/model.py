"""Joint-utility encoder. The matching decoder is a frozen Phase-5K module."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from wt.htw_phase9.const import EDGE_UB_DIM, EDGE_UU_DIM, GLOBAL_DIM, UAV_DIM
from wt.round30.models import masked_mean

from .protocol import WIDTH

QUEUE_INDEX = 2
OCCUPIED_INDEX = 5
EDGE_DIM = int(EDGE_UB_DIM) + 1


def _mlp(inputs: int, outputs: int, width: int = WIDTH) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


def append_queue_edge(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy the current buoy queue onto every UAV-buoy edge.

    The queue is already an encoder input on the buoy node. This does not add
    a future trajectory feature.
    """
    updated = {key: np.asarray(value).copy() for key, value in batch.items()}
    edge = np.asarray(updated["edge"], dtype=np.float32)
    buoy = np.asarray(updated["buoy"], dtype=np.float32)
    if int(edge.shape[-1]) != int(EDGE_UB_DIM):
        raise ValueError("expected the current UAV-buoy edge width")
    queue = np.broadcast_to(buoy[:, None, :, QUEUE_INDEX], edge.shape[:-1]).astype(np.float32)
    updated["edge"] = np.concatenate((edge, queue[..., None]), axis=-1)
    return updated


class JointUtilityEncoder(nn.Module):
    """Phase-5 message passing with a queue channel on the UAV-buoy edge."""

    def __init__(self, buoy_dim: int, width: int = WIDTH, layers: int = 2) -> None:
        super().__init__()
        self.buoy_dim = int(buoy_dim)
        self.edge_dim = int(EDGE_DIM)
        self.uav_enc = _mlp(UAV_DIM, width)
        self.buoy_enc = _mlp(self.buoy_dim, width)
        self.edge_enc = _mlp(self.edge_dim, width)
        self.uu_enc = _mlp(EDGE_UU_DIM, width)
        self.ub_msg = nn.ModuleList([_mlp(3 * width, width) for _ in range(layers)])
        self.u_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.b_upd = nn.ModuleList([_mlp(width, width) for _ in range(layers)])
        self.uu_msg = _mlp(3 * width, width)
        self.uu_upd = _mlp(width, width)

    def encode_nodes(
        self,
        uav: Tensor,
        buoy: Tensor,
        uav_mask: Tensor,
        buoy_mask: Tensor,
        edge: Tensor,
        edge_uu: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if int(edge.shape[-1]) != self.edge_dim:
            raise ValueError(f"edge dim {int(edge.shape[-1])} != {self.edge_dim}")
        h_u, h_b, edge_h = self.uav_enc(uav), self.buoy_enc(buoy), self.edge_enc(edge)
        pair_mask = uav_mask[:, :, None, None] & buoy_mask[:, None, :, None]
        k_count = int(buoy.shape[1])
        m_count = int(uav.shape[1])
        for msg, u_upd, b_upd in zip(self.ub_msg, self.u_upd, self.b_upd):
            h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
            h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
            messages = msg(torch.cat((h_u_exp, h_b_exp, edge_h), dim=-1)) * pair_mask.to(h_u.dtype)
            agg_u = messages.sum(dim=2) / pair_mask.to(h_u.dtype).sum(dim=2).clamp_min(1.0)
            agg_b = messages.sum(dim=1) / pair_mask.to(h_u.dtype).sum(dim=1).clamp_min(1.0)
            h_u = h_u + u_upd(agg_u)
            h_b = h_b + b_upd(agg_b)
        e_uu = self.uu_enc(edge_uu)
        uu_mask = uav_mask[:, :, None, None] & uav_mask[:, None, :, None]
        eye = torch.eye(m_count, device=uav.device, dtype=torch.bool)[None, :, :, None]
        uu_mask = uu_mask & (~eye)
        h_src = h_u.unsqueeze(1).expand(-1, m_count, -1, -1)
        h_dst = h_u.unsqueeze(2).expand(-1, -1, m_count, -1)
        uu_messages = self.uu_msg(torch.cat((h_dst, h_src, e_uu), dim=-1)) * uu_mask.to(h_u.dtype)
        agg_uu = uu_messages.sum(dim=2) / uu_mask.to(h_u.dtype).sum(dim=2).clamp_min(1.0)
        return h_u + self.uu_upd(agg_uu), h_b


class JointUtilityHGNN(nn.Module):
    """Encoder plus the set readout that conditions individual embeddings."""

    def __init__(self, buoy_dim: int, width: int = WIDTH) -> None:
        super().__init__()
        self.encoder = JointUtilityEncoder(buoy_dim=int(buoy_dim), width=int(width))
        joint_dim = int(width) * 2 + int(GLOBAL_DIM)
        self.mix = nn.Linear(joint_dim, int(width))
        self.u_proj = nn.Linear(int(width), int(width))
        self.b_proj = nn.Linear(int(width), int(width))
        self.reset_readout()

    def reset_readout(self) -> None:
        for layer in (self.mix, self.u_proj, self.b_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def load_utility_encoder(self, state: dict[str, Tensor]) -> None:
        own = self.encoder.state_dict()
        copied = 0
        for key, value in state.items():
            if not key.startswith("encoder."):
                continue
            local = key[len("encoder.") :]
            if local not in own:
                continue
            target = own[local]
            if tuple(target.shape) == tuple(value.shape):
                own[local] = value.detach().clone()
                copied += 1
            elif local == "edge_enc.0.weight" and int(target.shape[0]) == int(value.shape[0]) and int(target.shape[1]) == int(value.shape[1]) + 1:
                filled = torch.zeros_like(target)
                filled[:, : int(value.shape[1])] = value.detach()
                own[local] = filled
                copied += 1
        if copied == 0:
            raise RuntimeError("utility encoder checkpoint did not match the joint encoder")
        self.encoder.load_state_dict(own)

    def condition(self, tensors: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        h_u, h_b = self.encoder.encode_nodes(
            tensors["uav"],
            tensors["buoy"],
            tensors["uav_mask"],
            tensors["buoy_mask"],
            tensors["edge"],
            tensors["edge_uu"],
        )
        z_all = 0.5 * (masked_mean(h_u, tensors["uav_mask"], 1) + masked_mean(h_b, tensors["buoy_mask"], 1))
        occupied = tensors["buoy"][:, :, OCCUPIED_INDEX] > 0.5
        z_selected = masked_mean(h_b, occupied & tensors["buoy_mask"], 1)
        z_joint = self.mix(torch.cat((z_all, z_selected, tensors["global"]), dim=-1))
        shift_u = self.u_proj(z_joint).unsqueeze(1)
        shift_b = self.b_proj(z_joint).unsqueeze(1)
        return h_u + shift_u, h_b + shift_b


def batch_tensors(batch: dict[str, np.ndarray]) -> dict[str, Tensor]:
    tensors = {}
    for key, value in batch.items():
        dtype = torch.bool if str(key).endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(np.asarray(value), dtype=dtype)
    return tensors


__all__ = [
    "EDGE_DIM",
    "JointUtilityEncoder",
    "JointUtilityHGNN",
    "append_queue_edge",
    "batch_tensors",
]
