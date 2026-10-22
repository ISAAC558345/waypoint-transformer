"""Confirmation evaluation with heuristic-reference regret. Does not edit Phase-9.4."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wt.htw_phase9.const import HOLD, MAX_UAVS
from wt.htw_phase9.features import interference_utility, service_success
from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase94.autoregressive_decoder import AssignmentARModel
from wt.htw_phase94.evaluation import decode_sequence
from wt.htw_phase94.metrics import exact_match, idle_fraction, per_uav_accuracy
from wt.htw_phase94.oracle_sequence import serialize_oracle_assignment
from wt.htw_phase9.features import conflict_rate
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment, rollout


def heuristic_reference_rollout(seeds: tuple[int, ...]) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        env = L3BEnv()
        row = rollout(env, oracle_option_assignment, int(seed), hold=HOLD, compare_oracle=False)
        split = getattr(env, "_oracle_split", (-1, -1))
        row["scenario"] = int(seed)
        row["n_near"] = float(split[0])
        row["n_far"] = float(split[1])
        rows.append(row)
    payloads = [float(row["payload_bits"]) for row in rows]
    return {
        "mean_payload_bits": float(np.mean(payloads)),
        "per_seed_payload": payloads,
        "rows": rows,
        "idle_fraction": float(1.0 - np.mean([row["mean_active_frac"] for row in rows])),
        "conflict_rate": float(np.mean([row["mean_conflict_rate"] for row in rows])),
        "mean_rate_bps": float(np.mean([row["mean_rate_bps"] for row in rows])),
    }


def evaluate_detailed(model: AssignmentARModel, seeds: tuple[int, ...], heuristic_payloads: list[float]) -> list[dict[str, Any]]:
    model.eval()
    out = []
    for seed, j_ref in zip(seeds, heuristic_payloads):
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        pending = 0
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        local = {key: [] for key in ("exact", "per", "conf", "util", "idle", "svc", "tf", "fr", "nll", "rate")}
        n_near = n_far = -1
        while not info.get("terminated", False):
            if pending <= 0:
                batch_np = _batch_from_env(info, env, last_arrival, last_positions)
                heuristic = oracle_option_assignment(env)
                tokens = serialize_oracle_assignment(heuristic, env.config.num_uavs)
                n_near, n_far = getattr(env, "_oracle_split", (n_near, n_far))
                with torch.no_grad():
                    fr = decode_sequence(model, batch_np, env, teacher_tokens=None)
                    tf = decode_sequence(model, batch_np, env, teacher_tokens=tokens)
                assignment = fr["assignment"]
                local["exact"].append(exact_match(assignment, heuristic))
                local["per"].append(per_uav_accuracy(assignment, heuristic))
                local["conf"].append(conflict_rate(assignment, env.config.num_buoys))
                local["idle"].append(idle_fraction(assignment))
                local["tf"].append(per_uav_accuracy(tf["tokens"], tokens))
                local["fr"].append(per_uav_accuracy(fr["tokens"], tokens))
                local["nll"].append(float(tf["nll"]))
                pending = int(HOLD)
            after = env.step(fly_toward(env, assignment), assignment)
            local["svc"].append(service_success(after, assignment))
            local["util"].append(interference_utility(after))
            local["rate"].append(float(np.mean(np.asarray(after.get("rate_bps", np.zeros(MAX_UAVS)), dtype=np.float64))))
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            pending -= 1
            info = after
        payload = float(info["cumulative_payload_bits"])
        tf_acc = float(np.mean(local["tf"]))
        fr_acc = float(np.mean(local["fr"]))
        hrr = float(j_ref) - payload
        out.append(
            {
                "scenario": int(seed),
                "payload": payload,
                "heuristic_payload": float(j_ref),
                "hrr": hrr,
                "nhrr": hrr / (abs(float(j_ref)) + 1.0e-9),
                "conflict": float(np.mean(local["conf"])),
                "idle": float(np.mean(local["idle"])),
                "sinr_utility": float(np.mean(local["util"])),
                "exact_match": float(np.mean(local["exact"])),
                "per_uav": float(np.mean(local["per"])),
                "sequence_nll": float(np.nanmean(local["nll"])),
                "tf_acc": tf_acc,
                "fr_acc": fr_acc,
                "exposure_gap": tf_acc - fr_acc,
                "n_near": int(n_near),
                "n_far": int(n_far),
                "beats_heuristic": bool(payload > float(j_ref) + 1.0),
            }
        )
    return out


__all__ = ["evaluate_detailed", "heuristic_reference_rollout"]
