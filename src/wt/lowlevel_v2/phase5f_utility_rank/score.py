"""Score one UAV's legal buoys with the existing order-free decoder."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase9.const import BUOY_DIM
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import UNDECIDED, all_candidate_physics_set, make_snap
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5_hgnn_urgency.encoder import Phase5OrderFreeModel
from wt.lowlevel_v2.protocol import FINAL_CKPT

from .protocol import MODEL_INPUT_KEYS, OLD_HGNN


def _tensors(batch: dict) -> dict[str, torch.Tensor]:
    out = {}
    for key in MODEL_INPUT_KEYS:
        dtype = torch.bool if key.endswith("mask") else torch.float32
        out[key] = torch.as_tensor(batch[key], dtype=dtype)
    return out


def focal_scores(model, sample: dict, *, use_urgency: bool) -> torch.Tensor:
    batch = sample["urgency_batch"] if use_urgency else sample["batch"]
    tensors = _tensors(batch)
    encoded = model.encode(
        tensors["uav"],
        tensors["buoy"],
        tensors["uav_mask"],
        tensors["buoy_mask"],
        tensors["global"],
        tensors["edge"],
        tensors["edge_uu"],
    )
    partial = np.full(sample["context"].shape[0], -1, dtype=np.int64)
    for uav, buoy in enumerate(np.asarray(sample["context"], dtype=np.int64)):
        if int(uav) == int(sample["uav_id"]):
            partial[uav] = UNDECIDED
        else:
            partial[uav] = -1 if int(buoy) < 0 else int(buoy)
    template = L3BEnv()
    snap = make_snap(sample["uav_positions"], sample["buoy_positions"], template.config, env=template)
    physics = torch.as_tensor(all_candidate_physics_set(snap, partial)[None, ...], dtype=torch.float32)
    buoy_mask = np.asarray(batch["buoy_mask"][0], dtype=bool)
    logits = model.decoder(encoded, partial, physics, buoy_mask)
    index = torch.as_tensor(sample["candidate_ids"], dtype=torch.long)
    chosen = logits[int(sample["uav_id"]), index]
    if bool(torch.any(chosen < -1.0e8).item()):
        raise RuntimeError("phase-5f candidate was masked by the decoder")
    return chosen


def load_deepsets():
    if not FINAL_CKPT.is_file():
        raise FileNotFoundError(str(FINAL_CKPT))
    model = OrderFreeAssignmentModel("deepsets")
    model.load_state_dict(torch.load(FINAL_CKPT, map_location="cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_old_hgnn():
    if not OLD_HGNN.is_file():
        raise FileNotFoundError(str(OLD_HGNN))
    model = Phase5OrderFreeModel(buoy_dim=int(BUOY_DIM))
    model.load_state_dict(torch.load(OLD_HGNN, map_location="cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def new_hgnn() -> Phase5OrderFreeModel:
    return Phase5OrderFreeModel(buoy_dim=int(BUOY_DIM) + 1)


__all__ = ["focal_scores", "load_deepsets", "load_old_hgnn", "new_hgnn"]
