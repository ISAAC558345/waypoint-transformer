"""Fixed UAV decode-order wrapper. Does not change decoder weights or architecture."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from wt.htw_phase9.const import HOLD, MAX_UAVS
from wt.htw_phase9.features import conflict_rate, interference_utility
from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase94.autoregressive_decoder import AssignmentARModel
from wt.htw_phase94.evaluation import decode_sequence
from wt.htw_phase94.metrics import idle_fraction
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward


def permute_uav_batch(batch: dict[str, np.ndarray], order: np.ndarray) -> dict[str, np.ndarray]:
    out = {key: np.array(value, copy=True) for key, value in batch.items()}
    order = np.asarray(order, dtype=np.int64)
    out["uav"][0] = batch["uav"][0][order]
    out["uav_mask"][0] = batch["uav_mask"][0][order]
    out["edge"][0] = batch["edge"][0][order]
    out["edge_uu"][0] = batch["edge_uu"][0][np.ix_(order, order)]
    return out


def inverse_assignment(permuted: np.ndarray, order: np.ndarray, num_uavs: int) -> np.ndarray:
    orig = np.full(int(num_uavs), -1, dtype=np.int64)
    for step, phys in enumerate(np.asarray(order, dtype=np.int64)):
        orig[int(phys)] = int(permuted[step])
    return orig


def decode_ordered(model: AssignmentARModel, batch: dict[str, np.ndarray], env: L3BEnv, order: np.ndarray) -> np.ndarray:
    order = np.asarray(order, dtype=np.int64)
    env_p = copy.deepcopy(env)
    env_p.uav_positions = np.asarray(env.uav_positions, dtype=np.float64)[order]
    decoded = decode_sequence(model, permute_uav_batch(batch, order), env_p, teacher_tokens=None)
    return inverse_assignment(decoded["assignment"], order, env.config.num_uavs)


def evaluate_payload_order(model: AssignmentARModel, seed: int, order: np.ndarray) -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    pending = 0
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    assignments = []
    idles, confs, rates = [], [], []
    while not info.get("terminated", False):
        if pending <= 0:
            batch = _batch_from_env(info, env, last_arrival, last_positions)
            with torch.no_grad():
                assignment = decode_ordered(model, batch, env, order)
            assignments.append(np.asarray(assignment, dtype=np.int64).copy())
            idles.append(idle_fraction(assignment))
            confs.append(conflict_rate(assignment, env.config.num_buoys))
            pending = int(HOLD)
        after = env.step(fly_toward(env, assignment), assignment)
        rates.append(interference_utility(after))
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        pending -= 1
        info = after
    stacked = np.stack(assignments, axis=0) if assignments else np.zeros((0, MAX_UAVS), dtype=np.int64)
    return {
        "payload": float(info["cumulative_payload_bits"]),
        "idle": float(np.mean(idles)) if idles else 0.0,
        "conflict": float(np.mean(confs)) if confs else 0.0,
        "sinr_utility": float(np.mean(rates)) if rates else 0.0,
        "assignments": stacked,
    }


def assignment_entropy(stacked: np.ndarray) -> float:
    if stacked.size == 0:
        return 0.0
    values, counts = np.unique(stacked.reshape(-1), return_counts=True)
    _ = values
    p = counts.astype(np.float64) / float(counts.sum())
    return float(-np.sum(p * np.log(p + 1.0e-12)))


def freeze_orders(seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(int(seed))
    orders = {"O0": list(range(MAX_UAVS)), "O1": list(reversed(range(MAX_UAVS)))}
    seen = {tuple(orders["O0"]), tuple(orders["O1"])}
    index = 2
    while index < 6:
        perm = rng.permutation(MAX_UAVS).tolist()
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        orders[f"O{index}"] = perm
        index += 1
    return orders


__all__ = [
    "assignment_entropy",
    "decode_ordered",
    "evaluate_payload_order",
    "freeze_orders",
    "inverse_assignment",
    "permute_uav_batch",
]
