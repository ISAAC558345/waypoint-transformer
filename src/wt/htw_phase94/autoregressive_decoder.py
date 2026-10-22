"""Masked autoregressive assignment decoder. Encoder-agnostic; no Transformer."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wt.htw_phase9.const import GLOBAL_DIM, MAX_BUOYS, MAX_UAVS
from wt.htw_phase92.models import DeepSetsAssignmentHead
from wt.htw_phase93.models import RelationalHGNNAssignmentHead
from wt.round30.models import masked_mean

from .assignment_context import PARTIAL_DIM, PHYS_DIM

WIDTH = 128
OPTION_DIM = MAX_BUOYS + 1


def _mlp(inputs: int, outputs: int, width: int = WIDTH) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


class MaskedARDecoder(nn.Module):
    """Shared decoder: embeddings + partial assignment + per-candidate SINR physics."""

    def __init__(self) -> None:
        super().__init__()
        self.partial_enc = _mlp(PARTIAL_DIM, WIDTH)
        cond = WIDTH + WIDTH + WIDTH + WIDTH + GLOBAL_DIM + WIDTH + PHYS_DIM
        self.candidate = _mlp(cond, 1)

    def logits(
        self,
        h_u_m: Tensor,
        h_b: Tensor,
        e_m: Tensor,
        graph: Tensor,
        global_: Tensor,
        partial_vec: Tensor,
        physics: Tensor,
        legal: Tensor,
    ) -> Tensor:
        # h_u_m (B,W), h_b (B,K,W), e_m (B,K,W), graph (B,W), global (B,G)
        # physics (B,K+1,P), legal (B,K+1)
        batch = h_u_m.shape[0]
        k_count = h_b.shape[1]
        partial_h = self.partial_enc(partial_vec)
        h_u_exp = h_u_m.unsqueeze(1).expand(-1, k_count, -1)
        graph_exp = graph.unsqueeze(1).expand(-1, k_count, -1)
        glob_exp = global_.unsqueeze(1).expand(-1, k_count, -1)
        part_exp = partial_h.unsqueeze(1).expand(-1, k_count, -1)
        buoy_logits = self.candidate(torch.cat((h_u_exp, h_b, e_m, graph_exp, glob_exp, part_exp, physics[:, :k_count]), dim=-1)).squeeze(-1)
        idle_zero_b = torch.zeros_like(h_b[:, 0])
        idle_zero_e = torch.zeros_like(e_m[:, 0])
        idle_logit = self.candidate(torch.cat((h_u_m, idle_zero_b, idle_zero_e, graph, global_, partial_h, physics[:, k_count]), dim=-1)).squeeze(-1)
        logits = torch.cat((buoy_logits, idle_logit.unsqueeze(-1)), dim=-1)
        logits = logits.masked_fill(~legal, -1.0e9)
        _ = batch
        return logits


class AssignmentARModel(nn.Module):
    def __init__(self, encoder_name: str) -> None:
        super().__init__()
        if encoder_name not in {"deepsets", "hgnn"}:
            raise ValueError(encoder_name)
        self.encoder_name = encoder_name
        self.encoder = DeepSetsAssignmentHead() if encoder_name == "deepsets" else RelationalHGNNAssignmentHead()
        self.decoder = MaskedARDecoder()

    def encode(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor, edge_uu: Tensor) -> dict[str, Tensor]:
        if self.encoder_name == "deepsets":
            h_u = self.encoder.uav_enc(uav)
            h_b = self.encoder.buoy_enc(buoy)
            e_ub = self.encoder.edge_enc(edge)
        else:
            h_u, h_b, e_ub, _agg_uu = self.encoder._encode(uav, buoy, uav_mask, buoy_mask, edge, edge_uu)
        graph = 0.5 * (masked_mean(h_u, uav_mask, 1) + masked_mean(h_b, buoy_mask, 1))
        return {"h_u": h_u, "h_b": h_b, "e_ub": e_ub, "graph": graph, "global": global_}

    def step_logits(self, encoded: dict[str, Tensor], uav_index: int, partial_vec: Tensor, physics: Tensor, legal: Tensor) -> Tensor:
        return self.decoder.logits(
            encoded["h_u"][:, uav_index],
            encoded["h_b"],
            encoded["e_ub"][:, uav_index],
            encoded["graph"],
            encoded["global"],
            partial_vec,
            physics,
            legal,
        )


def decoder_parameter_count() -> int:
    return sum(int(item.numel()) for item in MaskedARDecoder().parameters())


__all__ = ["AssignmentARModel", "MaskedARDecoder", "OPTION_DIM", "WIDTH", "decoder_parameter_count"]
