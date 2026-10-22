"""Experimental planners. Official high_level.decide() is not replaced."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase100.high_level import load_deepsets_util
from wt.htw_phase9.const import BUOY_DIM
from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase943.order_free_decoder import OrderFreeSetDecoder
from wt.htw_phase943.set_context import empty_assignment, all_candidate_physics_set
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.protocol import FINAL_CKPT

from .encoder import Phase5OrderFreeModel
from .features import current_urgency, graph_batch
from .protocol import HOLD_STEPS


def load_frozen_deepsets():
    if not FINAL_CKPT.is_file():
        raise FileNotFoundError(str(FINAL_CKPT))
    model = load_deepsets_util(0, FINAL_CKPT)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class Phase5Planner:
    def __init__(self, kind: str, *, high_urgency: bool, model) -> None:
        if kind not in {"deepsets", "hgnn"}:
            raise ValueError(kind)
        if kind == "deepsets" and high_urgency:
            raise ValueError("official DeepSets encoder is not widened")
        self.kind = kind
        self.high_urgency = bool(high_urgency)
        self.model = model
        if kind == "hgnn" and not isinstance(model.decoder, OrderFreeSetDecoder):
            raise TypeError("HGNN planner must keep OrderFreeSetDecoder")

    def decide(self, env: L3BEnv, info: dict, last_arrival: np.ndarray, last_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        urgency = current_urgency(env)
        feature_urgency = urgency if self.high_urgency else None
        batch = graph_batch(info, env, last_arrival, last_positions, urgency=feature_urgency)
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            assignment = decode_assignment(self.model, batch, env)
        if was_training:
            self.model.train()
        return np.asarray(assignment, dtype=np.int64), urgency


def make_phase5_option(
    env: L3BEnv,
    info: dict,
    assignment: np.ndarray,
    pending: int,
    urgency: np.ndarray,
    *,
    attach_urgency: bool,
) -> TaskOption:
    targets = np.asarray(assignment, dtype=np.int64).copy()
    attached = np.asarray(urgency, dtype=np.float64).copy() if attach_urgency else None
    return TaskOption(
        targets=targets,
        decision_time=int(info["time_step"]),
        remaining_hold=int(pending),
        uav_mask=np.ones(int(env.config.num_uavs), dtype=bool),
        buoy_mask=np.ones(int(env.config.num_buoys), dtype=bool),
        association=targets.copy(),
        urgency=attached,
    )


def new_hgnn(*, high_urgency: bool) -> Phase5OrderFreeModel:
    buoy_dim = int(BUOY_DIM) + (1 if high_urgency else 0)
    return Phase5OrderFreeModel(buoy_dim=buoy_dim)


def first_step_entropy(planner: Phase5Planner, env: L3BEnv, info: dict, last_arrival: np.ndarray, last_positions: np.ndarray) -> float:
    urgency = current_urgency(env) if planner.high_urgency else None
    batch = graph_batch(info, env, last_arrival, last_positions, urgency=urgency)
    tensors = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(batch[key], dtype=dtype)
    with torch.no_grad():
        encoded = planner.model.encode(
            tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"]
        )
        assignment = empty_assignment(env.config.num_uavs)
        from wt.htw_phase943.set_context import make_snap

        snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
        physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32)
        logits = planner.model.decoder(encoded, assignment, physics, np.asarray(batch["buoy_mask"][0], dtype=bool))
    flat = logits.detach().cpu().numpy().reshape(-1)
    mask = flat > -1.0e8
    shifted = np.where(mask, flat, -1.0e9)
    shifted = shifted - float(np.max(shifted))
    weights = np.exp(shifted) * mask
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    prob = weights / total
    picked = prob[mask]
    return float(-np.sum(picked * np.log(picked + 1.0e-12)))


__all__ = [
    "HOLD_STEPS",
    "Phase5Planner",
    "first_step_entropy",
    "load_frozen_deepsets",
    "make_phase5_option",
    "new_hgnn",
]
