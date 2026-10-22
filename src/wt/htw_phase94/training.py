"""Teacher-forcing Oracle imitation for the masked AR decoder. No PPO."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from wt.htw_phase9.const import MAX_BUOYS, MAX_UAVS, SEED
from wt.htw_phase92.run_phase92 import BATCH, EPOCHS, LR
from wt.htw_phase93.losses import ranking_loss
from wt.round30.ppo import seed_everything

from .assignment_context import PARTIAL_DIM, PHYS_DIM, all_candidate_physics, legality_mask, make_snap, partial_vector
from .autoregressive_decoder import OPTION_DIM, AssignmentARModel
from .oracle_sequence import serialize_oracle_assignment

AUX_RANK_WEIGHT = 0.05


def teacher_sequences(assignment: np.ndarray) -> np.ndarray:
    rows = [serialize_oracle_assignment(row, MAX_UAVS) for row in np.asarray(assignment, dtype=np.int64)]
    return np.stack(rows, axis=0)


def precompute_teacher_context(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = int(data["assignment"].shape[0])
    sequences = teacher_sequences(data["assignment"])
    physics = np.zeros((count, MAX_UAVS, OPTION_DIM, PHYS_DIM), dtype=np.float32)
    legal = np.zeros((count, MAX_UAVS, OPTION_DIM), dtype=bool)
    partials = np.zeros((count, MAX_UAVS, PARTIAL_DIM), dtype=np.float32)
    for index in range(count):
        snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
        prefix = np.full(MAX_UAVS, -1, dtype=np.int64)
        buoy_avail = np.asarray(data["buoy_mask"][index], dtype=bool)
        for uav in range(MAX_UAVS):
            legal[index, uav] = legality_mask(prefix, uav, MAX_BUOYS, buoy_avail)
            physics[index, uav] = all_candidate_physics(snap, prefix, uav, MAX_BUOYS)
            partials[index, uav] = partial_vector(prefix, uav)
            token = int(sequences[index, uav])
            prefix[uav] = -1 if token >= MAX_BUOYS else token
    return {"physics": physics, "legal": legal, "partial": partials, "sequence": sequences}


def _encode_batch(model: AssignmentARModel, data: dict[str, np.ndarray], index: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    tensors = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(data[key][index], dtype=dtype, device=device)
    return model.encode(
        tensors["uav"],
        tensors["buoy"],
        tensors["uav_mask"],
        tensors["buoy_mask"],
        tensors["global"],
        tensors["edge"],
        tensors["edge_uu"],
    )


def ar_teacher_loss(
    model: AssignmentARModel,
    encoded: dict[str, torch.Tensor],
    context: dict[str, np.ndarray],
    index: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    nll_terms = []
    hits = []
    for uav in range(MAX_UAVS):
        partial = torch.as_tensor(context["partial"][index, uav], dtype=torch.float32, device=device)
        physics = torch.as_tensor(context["physics"][index, uav], dtype=torch.float32, device=device)
        legal = torch.as_tensor(context["legal"][index, uav], dtype=torch.bool, device=device)
        target = torch.as_tensor(context["sequence"][index, uav], dtype=torch.int64, device=device)
        logits = model.step_logits(encoded, uav, partial, physics, legal)
        nll_terms.append(F.cross_entropy(logits, target, reduction="mean"))
        hits.append((torch.argmax(logits, dim=-1) == target).to(torch.float32))
    loss = torch.stack(nll_terms).sum()
    tf_acc = torch.cat(hits, dim=0).mean()
    return loss, tf_acc


def train_ar_model(model: AssignmentARModel, data: dict[str, np.ndarray], name: str) -> AssignmentARModel:
    seed_everything(SEED)
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
        print(f"{name} epoch {epoch + 1}/{EPOCHS} loss={float(np.mean(losses)):.4f} tf_acc={float(np.mean(accs)):.3f}", flush=True)
    return model


__all__ = ["AUX_RANK_WEIGHT", "precompute_teacher_context", "teacher_sequences", "train_ar_model"]
