"""Shared order-free candidate decoder. No UAV-index-specific parameters."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from wt.htw_phase9.const import GLOBAL_DIM, IDLE_INDEX, MAX_BUOYS, MAX_UAVS
from wt.htw_phase92.models import DeepSetsAssignmentHead
from wt.htw_phase93.models import RelationalHGNNAssignmentHead
from wt.htw_phase94.assignment_context import PHYS_DIM
from wt.round30.models import masked_mean

from .protocol import TIE_ATOL
from .set_context import UNDECIDED, canonical_keys, decided_mask, legal_candidate_mask

WIDTH = 128


def _mlp(inputs: int, outputs: int, width: int = WIDTH) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


def _sorted_mean(vectors: Tensor, keys: Tensor, dim: int) -> Tensor:
    if vectors.shape[0] == 0:
        return torch.zeros(dim, device=vectors.device, dtype=vectors.dtype)
    rank = keys[:, 0] * 1.0e6 + keys[:, 1]
    order = torch.argsort(rank)
    return vectors[order].mean(dim=0)


class OrderFreeSetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.idle_edge = _mlp(WIDTH, WIDTH)
        cond = 7 * WIDTH + GLOBAL_DIM + PHYS_DIM
        self.scorer = _mlp(cond, 1)

    def _score_from_context(
        self,
        encoded: dict[str, Tensor],
        partial: Tensor,
        rem_u: Tensor,
        rem_b: Tensor,
        physics: Tensor,
        assignment: np.ndarray,
        buoy_mask: np.ndarray,
    ) -> Tensor:
        h_u, h_b, e_ub = encoded["h_u"], encoded["h_b"], encoded["e_ub"]
        graph, global_ = encoded["graph"], encoded["global"]
        h_u_exp = h_u[0].unsqueeze(1).expand(-1, MAX_BUOYS, -1)
        h_b_exp = h_b[0].unsqueeze(0).expand(MAX_UAVS, -1, -1)
        graph_exp = graph[0].unsqueeze(0).unsqueeze(0).expand(MAX_UAVS, MAX_BUOYS, -1)
        glob_exp = global_[0].unsqueeze(0).unsqueeze(0).expand(MAX_UAVS, MAX_BUOYS, -1)
        part_exp = partial.unsqueeze(0).unsqueeze(0).expand(MAX_UAVS, MAX_BUOYS, -1)
        rem_u_exp = rem_u.unsqueeze(0).unsqueeze(0).expand(MAX_UAVS, MAX_BUOYS, -1)
        rem_b_exp = rem_b.unsqueeze(0).unsqueeze(0).expand(MAX_UAVS, MAX_BUOYS, -1)
        phys = physics.to(h_u.device)
        if phys.dim() == 4:
            phys = phys[0]
        buoy_logits = self.scorer(
            torch.cat((h_u_exp, h_b_exp, e_ub[0], graph_exp, glob_exp, part_exp, rem_u_exp, rem_b_exp, phys[:, :MAX_BUOYS]), dim=-1)
        ).squeeze(-1)
        idle_zero_b = torch.zeros_like(h_b[0, 0])
        idle_zero_e = torch.zeros_like(e_ub[0, 0, 0])
        idle_logits = self.scorer(
            torch.cat(
                (
                    h_u[0],
                    idle_zero_b.unsqueeze(0).expand(MAX_UAVS, -1),
                    idle_zero_e.unsqueeze(0).expand(MAX_UAVS, -1),
                    graph[0].unsqueeze(0).expand(MAX_UAVS, -1),
                    global_[0].unsqueeze(0).expand(MAX_UAVS, -1),
                    partial.unsqueeze(0).expand(MAX_UAVS, -1),
                    rem_u.unsqueeze(0).expand(MAX_UAVS, -1),
                    rem_b.unsqueeze(0).expand(MAX_UAVS, -1),
                    phys[:, IDLE_INDEX],
                ),
                dim=-1,
            )
        ).squeeze(-1)
        logits = torch.cat((buoy_logits, idle_logits.unsqueeze(-1)), dim=-1)
        legal = torch.as_tensor(legal_candidate_mask(assignment, buoy_mask), dtype=torch.bool, device=h_u.device)
        return logits.masked_fill(~legal, -1.0e9)

    def _contexts(self, encoded: dict[str, Tensor], assignment: np.ndarray, buoy_mask: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        h_u, h_b, e_ub = encoded["h_u"], encoded["h_b"], encoded["e_ub"]
        decided = torch.as_tensor(decided_mask(assignment), dtype=torch.bool, device=h_u.device)
        tokens = np.asarray(assignment, dtype=np.int64)
        selected = []
        selected_keys = []
        xy_u = encoded["uav_xy"][0]
        xy_b = encoded["buoy_xy"][0]
        for uav, token in enumerate(tokens):
            if token == UNDECIDED:
                continue
            if token >= 0:
                selected.append(e_ub[0, uav, int(token)])
            else:
                selected.append(self.idle_edge(h_u[0, uav]))
            selected_keys.append(xy_u[uav])
        if selected:
            partial = _sorted_mean(torch.stack(selected, dim=0), torch.stack(selected_keys, dim=0), h_u.shape[-1])
        else:
            partial = torch.zeros(h_u.shape[-1], device=h_u.device, dtype=h_u.dtype)
        remain_u = ~decided
        rem_u = _sorted_mean(h_u[0, remain_u], xy_u[remain_u], h_u.shape[-1])
        occupied = np.zeros(MAX_BUOYS, dtype=bool)
        for token in tokens:
            if 0 <= int(token) < MAX_BUOYS:
                occupied[int(token)] = True
        buoy_ok = torch.as_tensor((~occupied) & np.asarray(buoy_mask, dtype=bool), dtype=torch.bool, device=h_u.device)
        rem_b = _sorted_mean(h_b[0, buoy_ok], xy_b[buoy_ok], h_u.shape[-1])
        return partial, rem_u, rem_b

    def forward(
        self,
        encoded: dict[str, Tensor],
        assignment: np.ndarray,
        physics: Tensor,
        buoy_mask: np.ndarray,
    ) -> Tensor:
        h_u = encoded["h_u"]
        if h_u.shape[0] != 1:
            raise ValueError("order-free decoder scores one graph at a time")
        partial, rem_u, rem_b = self._contexts(encoded, assignment, buoy_mask)
        return self._score_from_context(encoded, partial, rem_u, rem_b, physics, assignment, buoy_mask)

    def forward_many(
        self,
        encoded: dict[str, Tensor],
        assignments: list[np.ndarray],
        physics: Tensor,
        buoy_mask: np.ndarray,
    ) -> Tensor:
        h_u, h_b, e_ub = encoded["h_u"], encoded["h_b"], encoded["e_ub"]
        graph, global_ = encoded["graph"], encoded["global"]
        device = h_u.device
        width = h_u.shape[-1]
        count = len(assignments)
        partials = []
        rem_us = []
        rem_bs = []
        legal = []
        for assignment in assignments:
            partial, rem_u, rem_b = self._contexts(encoded, assignment, buoy_mask)
            partials.append(partial)
            rem_us.append(rem_u)
            rem_bs.append(rem_b)
            legal.append(legal_candidate_mask(assignment, buoy_mask))
        partial = torch.stack(partials, dim=0)
        rem_u = torch.stack(rem_us, dim=0)
        rem_b = torch.stack(rem_bs, dim=0)
        phys = physics.to(device)
        h_u_exp = h_u[0].unsqueeze(0).unsqueeze(2).expand(count, MAX_UAVS, MAX_BUOYS, -1)
        h_b_exp = h_b[0].unsqueeze(0).unsqueeze(1).expand(count, MAX_UAVS, MAX_BUOYS, -1)
        e_exp = e_ub[0].unsqueeze(0).expand(count, -1, -1, -1)
        graph_exp = graph[0].view(1, 1, 1, -1).expand(count, MAX_UAVS, MAX_BUOYS, -1)
        glob_exp = global_[0].view(1, 1, 1, -1).expand(count, MAX_UAVS, MAX_BUOYS, -1)
        part_exp = partial[:, None, None, :].expand(count, MAX_UAVS, MAX_BUOYS, -1)
        rem_u_exp = rem_u[:, None, None, :].expand(count, MAX_UAVS, MAX_BUOYS, -1)
        rem_b_exp = rem_b[:, None, None, :].expand(count, MAX_UAVS, MAX_BUOYS, -1)
        buoy_logits = self.scorer(
            torch.cat((h_u_exp, h_b_exp, e_exp, graph_exp, glob_exp, part_exp, rem_u_exp, rem_b_exp, phys[:, :, :MAX_BUOYS]), dim=-1)
        ).squeeze(-1)
        zeros_b = torch.zeros(count, MAX_UAVS, width, device=device, dtype=h_u.dtype)
        idle_logits = self.scorer(
            torch.cat(
                (
                    h_u[0].unsqueeze(0).expand(count, -1, -1),
                    zeros_b,
                    zeros_b,
                    graph[0].view(1, 1, -1).expand(count, MAX_UAVS, -1),
                    global_[0].view(1, 1, -1).expand(count, MAX_UAVS, -1),
                    partial[:, None, :].expand(count, MAX_UAVS, -1),
                    rem_u[:, None, :].expand(count, MAX_UAVS, -1),
                    rem_b[:, None, :].expand(count, MAX_UAVS, -1),
                    phys[:, :, IDLE_INDEX],
                ),
                dim=-1,
            )
        ).squeeze(-1)
        logits = torch.cat((buoy_logits, idle_logits.unsqueeze(-1)), dim=-1)
        legal_t = torch.as_tensor(np.stack(legal, axis=0), dtype=torch.bool, device=device)
        return logits.masked_fill(~legal_t, -1.0e9)


class OrderFreeAssignmentModel(nn.Module):
    def __init__(self, encoder_name: str) -> None:
        super().__init__()
        if encoder_name not in {"deepsets", "hgnn"}:
            raise ValueError(encoder_name)
        self.encoder_name = encoder_name
        self.encoder = DeepSetsAssignmentHead() if encoder_name == "deepsets" else RelationalHGNNAssignmentHead()
        self.decoder = OrderFreeSetDecoder()

    def encode(self, uav: Tensor, buoy: Tensor, uav_mask: Tensor, buoy_mask: Tensor, global_: Tensor, edge: Tensor, edge_uu: Tensor) -> dict[str, Tensor]:
        if self.encoder_name == "deepsets":
            h_u = self.encoder.uav_enc(uav)
            h_b = self.encoder.buoy_enc(buoy)
            e_ub = self.encoder.edge_enc(edge)
        else:
            h_u, h_b, e_ub, _agg = self.encoder._encode(uav, buoy, uav_mask, buoy_mask, edge, edge_uu)
        graph = 0.5 * (masked_mean(h_u, uav_mask, 1) + masked_mean(h_b, buoy_mask, 1))
        return {"h_u": h_u, "h_b": h_b, "e_ub": e_ub, "graph": graph, "global": global_, "uav_xy": uav[..., :2], "buoy_xy": buoy[..., :2]}


def pick_candidate(
    logits: Tensor,
    snap,
    assignment: np.ndarray,
    buoy_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    values = logits.detach().cpu().numpy()
    legal = legal_candidate_mask(assignment, buoy_mask)
    values = np.where(legal, values, -1.0e12)
    peak = float(np.max(values))
    tied = np.argwhere(legal & (values >= peak - TIE_ATOL))
    if tied.shape[0] == 0:
        raise RuntimeError("no legal candidate")
    if rng is not None:
        tied = tied[rng.permutation(tied.shape[0])]
    keys = canonical_keys(snap, assignment)
    best = None
    for uav, token in tied:
        key = tuple(keys[int(uav), int(token)].tolist())
        if best is None or key < best[0]:
            best = (key, int(uav), int(token))
    return int(best[1]), int(best[2])


def decoder_parameter_count() -> int:
    return sum(int(item.numel()) for item in OrderFreeSetDecoder().parameters())


__all__ = ["OrderFreeAssignmentModel", "OrderFreeSetDecoder", "WIDTH", "decoder_parameter_count", "pick_candidate"]
