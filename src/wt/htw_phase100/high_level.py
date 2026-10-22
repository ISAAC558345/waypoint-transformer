"""Frozen DeepSets-util high-level decode. Reuses Phase-9.4.3 inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.l3b.env import L3BEnv

from .contract import TaskOption
from .protocol import HOLD, PHASE945_OUTPUT


def load_deepsets_util(seed: int, path: Path | None = None) -> OrderFreeAssignmentModel:
    ckpt = Path(path) if path is not None else PHASE945_OUTPUT / f"deepsets_seed{int(seed)}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(str(ckpt))
    model = OrderFreeAssignmentModel("deepsets")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def decide(
    model: OrderFreeAssignmentModel,
    env: L3BEnv,
    info: dict,
    last_arrival: np.ndarray,
    last_positions: np.ndarray,
    pending: int,
) -> np.ndarray:
    batch = _batch_from_env(info, env, last_arrival, last_positions)
    with torch.no_grad():
        assignment = decode_assignment(model, batch, env)
    return np.asarray(assignment, dtype=np.int64)


def make_option(env: L3BEnv, info: dict, assignment: np.ndarray, pending: int) -> TaskOption:
    batch_mask_u = np.ones(int(env.config.num_uavs), dtype=bool)
    batch_mask_b = np.ones(int(env.config.num_buoys), dtype=bool)
    targets = np.asarray(assignment, dtype=np.int64).copy()
    return TaskOption(
        targets=targets,
        decision_time=int(info["time_step"]),
        remaining_hold=int(pending),
        uav_mask=batch_mask_u,
        buoy_mask=batch_mask_b,
        association=targets.copy(),
    )


__all__ = ["decide", "load_deepsets_util", "make_option", "HOLD"]
