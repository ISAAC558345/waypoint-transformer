"""Layerwise encoder probes. Calls frozen Phase-9.3 modules without editing them."""
from __future__ import annotations

import torch
from torch import Tensor

from wt.round30.models import masked_mean


def pairwise_cosine_mean(nodes: Tensor, mask: Tensor) -> float:
    valid = nodes[mask]
    if valid.shape[0] < 2:
        return float("nan")
    normed = torch.nn.functional.normalize(valid, dim=-1, eps=1.0e-8)
    sim = normed @ normed.T
    n = int(valid.shape[0])
    off = sim.sum() - torch.diagonal(sim).sum()
    return float((off / max(n * (n - 1), 1)).cpu())


def effective_rank_np(matrix) -> float:
    import numpy as np

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if arr.ndim == 1:
        arr = arr[None, :]
    centered = arr - arr.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    mass = singular[singular > 1.0e-12]
    if mass.size == 0:
        return 0.0
    prob = mass / mass.sum()
    return float(np.exp(-(prob * np.log(prob)).sum()))


@torch.no_grad()
def encode_hgnn_layers(encoder, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, edge: Tensor, edge_uu: Tensor) -> list[dict[str, Tensor]]:
    h_u, h_b, e = encoder.uav_enc(uav), encoder.buoy_enc(buoy), encoder.edge_enc(edge)
    pair_mask = uav_mask[:, :, None, None] & buoy_mask[:, None, :, None]
    k_count = int(buoy.shape[1])
    m_count = int(uav.shape[1])
    layers = [
        {
            "name": "layer0_init",
            "h_u": h_u,
            "h_b": h_b,
            "e_ub": e,
            "e_uu": encoder.uu_enc(edge_uu),
            "ub_msg": torch.zeros_like(e),
            "uu_msg": torch.zeros(uav.shape[0], m_count, m_count, h_u.shape[-1], device=h_u.device, dtype=h_u.dtype),
        }
    ]
    for index, (msg, u_upd, b_upd) in enumerate(zip(encoder.ub_msg, encoder.u_upd, encoder.b_upd)):
        h_u_exp = h_u.unsqueeze(2).expand(-1, -1, k_count, -1)
        h_b_exp = h_b.unsqueeze(1).expand(-1, m_count, -1, -1)
        messages = msg(torch.cat((h_u_exp, h_b_exp, e), dim=-1)) * pair_mask.to(h_u.dtype)
        agg_u = messages.sum(dim=2) / pair_mask.to(h_u.dtype).sum(dim=2).clamp_min(1.0)
        agg_b = messages.sum(dim=1) / pair_mask.to(h_u.dtype).sum(dim=1).clamp_min(1.0)
        h_u = h_u + u_upd(agg_u)
        h_b = h_b + b_upd(agg_b)
        layers.append(
            {
                "name": f"ub_layer{index + 1}",
                "h_u": h_u,
                "h_b": h_b,
                "e_ub": e,
                "e_uu": layers[0]["e_uu"],
                "ub_msg": messages,
                "uu_msg": layers[0]["uu_msg"],
            }
        )
    e_uu = encoder.uu_enc(edge_uu)
    uu_mask = uav_mask[:, :, None, None] & uav_mask[:, None, :, None]
    eye = torch.eye(m_count, device=uav.device, dtype=torch.bool)[None, :, :, None]
    uu_mask = uu_mask & (~eye)
    h_src = h_u.unsqueeze(1).expand(-1, m_count, -1, -1)
    h_dst = h_u.unsqueeze(2).expand(-1, -1, m_count, -1)
    uu_messages = encoder.uu_msg(torch.cat((h_dst, h_src, e_uu), dim=-1)) * uu_mask.to(h_u.dtype)
    agg_uu = uu_messages.sum(dim=2) / uu_mask.to(h_u.dtype).sum(dim=2).clamp_min(1.0)
    h_u = h_u + encoder.uu_upd(agg_uu)
    graph = 0.5 * (masked_mean(h_u, uav_mask, 1) + masked_mean(h_b, buoy_mask, 1))
    layers.append(
        {
            "name": "uu_layer",
            "h_u": h_u,
            "h_b": h_b,
            "e_ub": e,
            "e_uu": e_uu,
            "ub_msg": layers[-1]["ub_msg"],
            "uu_msg": uu_messages,
            "agg_uu": agg_uu,
            "graph": graph,
        }
    )
    return layers


@torch.no_grad()
def encode_deepsets_layers(encoder, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, edge: Tensor, edge_uu: Tensor) -> list[dict[str, Tensor]]:
    h_u, h_b, e = encoder.uav_enc(uav), encoder.buoy_enc(buoy), encoder.edge_enc(edge)
    graph = 0.5 * (masked_mean(h_u, uav_mask, 1) + masked_mean(h_b, buoy_mask, 1))
    zeros_uu = torch.zeros(uav.shape[0], uav.shape[1], uav.shape[1], h_u.shape[-1], device=h_u.device, dtype=h_u.dtype)
    return [
        {
            "name": "layer0_init",
            "h_u": h_u,
            "h_b": h_b,
            "e_ub": e,
            "e_uu": zeros_uu,
            "ub_msg": torch.zeros_like(e),
            "uu_msg": zeros_uu,
            "graph": graph,
        }
    ]


def encode_layers(model, tensors: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    if model.encoder_name == "hgnn":
        return encode_hgnn_layers(model.encoder, tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["edge"], tensors["edge_uu"])
    return encode_deepsets_layers(model.encoder, tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["edge"], tensors["edge_uu"])
