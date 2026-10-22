"""Supervised order-marginalized set likelihood. Same optimizer/epochs as Phase-9.4."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import BATCH, EPOCHS, LR
from wt.htw_phase93.losses import ranking_loss
from wt.htw_phase94.training import AUX_RANK_WEIGHT
from wt.round30.ppo import seed_everything

from .order_free_decoder import OrderFreeAssignmentModel
from .set_context import make_snap
from .set_likelihood_dp import set_nll


def _encode_one(model: OrderFreeAssignmentModel, data: dict[str, np.ndarray], index: int, device: str) -> dict:
    tensors = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(data[key][index : index + 1], dtype=dtype, device=device)
    return model.encode(tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"])


def train_set_model(model: OrderFreeAssignmentModel, data: dict[str, np.ndarray], name: str, init_seed: int) -> OrderFreeAssignmentModel:
    seed_everything(int(init_seed))
    device = "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    count = int(data["assignment"].shape[0])
    assignment_t = torch.as_tensor(data["assignment"], dtype=torch.int64)
    model.train()
    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        losses = []
        for start in range(0, count, BATCH):
            index = order[start : start + BATCH]
            batch_loss = []
            for item in index:
                encoded = _encode_one(model, data, int(item), device)
                snap = make_snap(data["uav_positions"][item], data["buoy_positions"][item])
                target = np.asarray(data["assignment"][item], dtype=np.int64)
                nll = set_nll(model, encoded, snap, target, data["buoy_mask"][item])
                batch_loss.append(nll)
            set_loss = torch.stack(batch_loss).mean()
            scores, idle = model.encoder(
                torch.as_tensor(data["uav"][index], dtype=torch.float32, device=device),
                torch.as_tensor(data["buoy"][index], dtype=torch.float32, device=device),
                torch.as_tensor(data["uav_mask"][index], dtype=torch.bool, device=device),
                torch.as_tensor(data["buoy_mask"][index], dtype=torch.bool, device=device),
                torch.as_tensor(data["global"][index], dtype=torch.float32, device=device),
                torch.as_tensor(data["edge"][index], dtype=torch.float32, device=device),
                torch.as_tensor(data["edge_uu"][index], dtype=torch.float32, device=device),
            )
            rank, _acc = ranking_loss(
                scores,
                idle,
                assignment_t[index].to(device),
                torch.as_tensor(data["uav_mask"][index], dtype=torch.bool, device=device),
                torch.as_tensor(data["buoy_mask"][index], dtype=torch.bool, device=device),
            )
            loss = set_loss + AUX_RANK_WEIGHT * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"{name} seed={init_seed} epoch {epoch + 1}/{EPOCHS} loss={float(np.mean(losses)):.4f}", flush=True)
    return model


__all__ = ["train_set_model"]
