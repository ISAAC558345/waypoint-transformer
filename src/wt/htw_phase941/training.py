"""Seeded Phase-9.4 training copy. Loss, epochs, optimizer, and decoder stay frozen."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import BATCH, EPOCHS, LR
from wt.htw_phase93.losses import ranking_loss
from wt.htw_phase94.autoregressive_decoder import AssignmentARModel
from wt.htw_phase94.training import AUX_RANK_WEIGHT, _encode_batch, ar_teacher_loss, precompute_teacher_context
from wt.round30.ppo import seed_everything


def train_ar_model_seeded(model: AssignmentARModel, data: dict[str, np.ndarray], name: str, init_seed: int) -> AssignmentARModel:
    seed_everything(int(init_seed))
    device = "cpu"
    model = model.to(device)
    context = precompute_teacher_context(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    count = int(data["assignment"].shape[0])
    assignment_t = torch.as_tensor(data["assignment"], dtype=torch.int64)
    model.train()
    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        losses = []
        accs = []
        for start in range(0, count, BATCH):
            index = order[start : start + BATCH]
            encoded = _encode_batch(model, data, index, device)
            ar_loss, tf_acc = ar_teacher_loss(model, encoded, context, index, device)
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
            loss = ar_loss + AUX_RANK_WEIGHT * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(float(tf_acc.detach().cpu()))
        print(f"{name} seed={init_seed} epoch {epoch + 1}/{EPOCHS} loss={float(np.mean(losses)):.4f} tf_acc={float(np.mean(accs)):.3f}", flush=True)
    return model


__all__ = ["train_ar_model_seeded"]
