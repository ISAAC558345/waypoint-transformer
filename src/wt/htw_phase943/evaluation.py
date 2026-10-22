"""Free-running order-free inference. No Oracle prefix."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wt.htw_phase9.const import HOLD, MAX_UAVS
from wt.htw_phase9.features import conflict_rate, interference_utility
from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase94.metrics import exact_match, idle_fraction, per_uav_accuracy
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment, rollout

from .order_free_decoder import OrderFreeAssignmentModel, pick_candidate
from .oracle_assignment_set import assignment_to_tokens, tokens_to_assignment
from .set_context import UNDECIDED, all_candidate_physics_set, apply_choice, assignment_set_tuple, empty_assignment, make_snap
from .set_likelihood_dp import set_nll


def encode_env(model: OrderFreeAssignmentModel, batch: dict[str, np.ndarray]) -> dict:
    tensors = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(batch[key], dtype=dtype)
    return model.encode(tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"])


def decode_assignment(
    model: OrderFreeAssignmentModel,
    batch: dict[str, np.ndarray],
    env: L3BEnv,
    n_steps: int | None = None,
    shuffle_seed: int | None = None,
) -> np.ndarray:
    encoded = encode_env(model, batch)
    snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
    assignment = empty_assignment(env.config.num_uavs)
    buoy_mask = np.asarray(batch["buoy_mask"][0], dtype=bool)
    steps = int(env.config.num_uavs if n_steps is None else n_steps)
    rng = None if shuffle_seed is None else np.random.default_rng(int(shuffle_seed))
    for _ in range(steps):
        if np.all(assignment != UNDECIDED):
            break
        physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32)
        logits = model.decoder(encoded, assignment, physics, buoy_mask)
        if not bool(torch.isfinite(logits.masked_fill(logits < -1.0e8, 0.0)).all()):
            raise RuntimeError("non-finite decoder logits")
        uav, token = pick_candidate(logits, snap, assignment, buoy_mask, rng)
        assignment = apply_choice(assignment, uav, token)
    return tokens_to_assignment(assignment_to_tokens(np.where(assignment == UNDECIDED, -1, assignment)), env.config.num_uavs)


def heuristic_reference_rollout(seeds: tuple[int, ...]) -> dict[str, Any]:
    rows = [rollout(L3BEnv(), oracle_option_assignment, int(seed), hold=HOLD, compare_oracle=False) for seed in seeds]
    payloads = [float(row["payload_bits"]) for row in rows]
    return {"mean_payload_bits": float(np.mean(payloads)), "per_seed_payload": payloads}


def evaluate_model(model: OrderFreeAssignmentModel, seeds: tuple[int, ...], heuristic_payloads: list[float]) -> list[dict[str, Any]]:
    model.eval()
    out = []
    for seed, j_ref in zip(seeds, heuristic_payloads):
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        pending = 0
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        local = {key: [] for key in ("exact", "per", "conf", "idle", "util", "nll")}
        while not info.get("terminated", False):
            if pending <= 0:
                batch = _batch_from_env(info, env, last_arrival, last_positions)
                heuristic = oracle_option_assignment(env)
                with torch.no_grad():
                    assignment = decode_assignment(model, batch, env)
                    encoded = encode_env(model, batch)
                    snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
                    nll = float(set_nll(model, encoded, snap, heuristic, batch["buoy_mask"][0]).detach().cpu())
                local["exact"].append(exact_match(assignment, heuristic))
                local["per"].append(per_uav_accuracy(assignment, heuristic))
                local["conf"].append(conflict_rate(assignment, env.config.num_buoys))
                local["idle"].append(idle_fraction(assignment))
                if not local["nll"]:
                    local["nll"].append(nll)
                pending = int(HOLD)
            after = env.step(fly_toward(env, assignment), assignment)
            local["util"].append(interference_utility(after))
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            pending -= 1
            info = after
        payload = float(info["cumulative_payload_bits"])
        hrr = float(j_ref) - payload
        out.append(
            {
                "scenario": int(seed),
                "payload": payload,
                "hrr": hrr,
                "nhrr": hrr / (abs(float(j_ref)) + 1.0e-9),
                "conflict": float(np.mean(local["conf"])),
                "idle": float(np.mean(local["idle"])),
                "sinr_utility": float(np.mean(local["util"])),
                "exact_match": float(np.mean(local["exact"])),
                "per_uav": float(np.mean(local["per"])),
                "set_nll": float(np.mean(local["nll"])),
            }
        )
    return out


__all__ = ["decode_assignment", "encode_env", "evaluate_model", "heuristic_reference_rollout"]
