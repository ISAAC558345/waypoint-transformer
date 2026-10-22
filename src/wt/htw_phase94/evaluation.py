"""Free-running closed-loop eval plus teacher-forcing snapshot metrics."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

from wt.htw_phase9.const import EVAL_SEEDS, HOLD, MAX_BUOYS, MAX_UAVS
from wt.htw_phase9.features import conflict_rate, interference_utility, service_success
from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.l3b.env import L3BEnv
from wt.l3b.policies import exclusive_matching, fly_toward, oracle_option_assignment, rollout

from .assignment_context import all_candidate_physics, legality_mask, make_snap, partial_vector
from .autoregressive_decoder import AssignmentARModel
from .metrics import exact_match, idle_fraction, normalized_regret, oracle_regret, per_uav_accuracy
from .oracle_sequence import sequence_to_assignment, serialize_oracle_assignment


def _feature_tensors(batch_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    tensors = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(batch_np[key], dtype=dtype)
    return tensors


def decode_sequence(
    model: AssignmentARModel,
    batch_np: dict[str, np.ndarray],
    env: L3BEnv,
    *,
    teacher_tokens: np.ndarray | None = None,
) -> dict[str, Any]:
    """If teacher_tokens is set, prefixes are Oracle (TF). Otherwise prefixes are predictions (FR)."""
    tensors = _feature_tensors(batch_np)
    encoded = model.encode(
        tensors["uav"],
        tensors["buoy"],
        tensors["uav_mask"],
        tensors["buoy_mask"],
        tensors["global"],
        tensors["edge"],
        tensors["edge_uu"],
    )
    snap = make_snap(env.uav_positions, env.buoy_positions, env.config)
    prefix = np.full(MAX_UAVS, -1, dtype=np.int64)
    preds = np.full(MAX_UAVS, MAX_BUOYS, dtype=np.int64)
    nll = []
    buoy_avail = np.asarray(batch_np["buoy_mask"][0], dtype=bool)
    for uav in range(MAX_UAVS):
        legal = legality_mask(prefix, uav, MAX_BUOYS, buoy_avail)
        physics = all_candidate_physics(snap, prefix, uav, MAX_BUOYS)
        logits = model.step_logits(
            encoded,
            uav,
            torch.as_tensor(partial_vector(prefix, uav)[None, ...], dtype=torch.float32),
            torch.as_tensor(physics[None, ...], dtype=torch.float32),
            torch.as_tensor(legal[None, ...], dtype=torch.bool),
        )
        finite = torch.isfinite(logits)
        if not bool(finite.all()):
            raise RuntimeError("non-finite AR logits")
        pred = int(torch.argmax(logits, dim=-1).item())
        preds[uav] = pred
        if teacher_tokens is None:
            token = pred
        else:
            token = int(teacher_tokens[uav])
            nll.append(float(F.cross_entropy(logits, torch.as_tensor([token], dtype=torch.int64))))
        prefix[uav] = -1 if token >= MAX_BUOYS else token
    if teacher_tokens is None:
        for uav in range(MAX_UAVS):
            # NLL under the free-running path uses the model's own tokens, not future Oracle.
            pass
    assignment = sequence_to_assignment(preds, env.config.num_uavs)
    return {
        "tokens": preds,
        "assignment": assignment,
        "nll": float(np.mean(nll)) if nll else float("nan"),
        "conflict": conflict_rate(assignment, env.config.num_buoys),
    }


def evaluate_ar_model(model: AssignmentARModel, seeds: tuple[int, ...] = EVAL_SEEDS) -> dict[str, Any]:
    model.eval()
    payloads = []
    exacts, per_uav, conflicts, utilities, idles, services = [], [], [], [], [], []
    tf_accs, fr_accs, tf_nlls, exact_tf = [], [], [], []
    mean_rates = []
    per_seed_payload = []
    for seed in seeds:
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        pending = 0
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        local = {key: [] for key in ("exact", "per", "conf", "util", "idle", "svc", "tf", "fr", "nll", "etf", "rate")}
        while not info.get("terminated", False):
            if pending <= 0:
                batch_np = _batch_from_env(info, env, last_arrival, last_positions)
                oracle = oracle_option_assignment(env)
                oracle_tokens = serialize_oracle_assignment(oracle, env.config.num_uavs)
                with torch.no_grad():
                    fr = decode_sequence(model, batch_np, env, teacher_tokens=None)
                    tf = decode_sequence(model, batch_np, env, teacher_tokens=oracle_tokens)
                assignment = fr["assignment"]
                local["exact"].append(exact_match(assignment, oracle))
                local["per"].append(per_uav_accuracy(assignment, oracle))
                local["conf"].append(float(fr["conflict"]))
                local["idle"].append(idle_fraction(assignment))
                local["tf"].append(per_uav_accuracy(tf["tokens"], oracle_tokens))
                local["fr"].append(per_uav_accuracy(fr["tokens"], oracle_tokens))
                local["nll"].append(float(tf["nll"]))
                local["etf"].append(exact_match(sequence_to_assignment(tf["tokens"], env.config.num_uavs), oracle))
                pending = int(HOLD)
            after = env.step(fly_toward(env, assignment), assignment)
            local["svc"].append(service_success(after, assignment))
            local["util"].append(interference_utility(after))
            local["rate"].append(float(np.mean(np.asarray(after.get("rate_bps", np.zeros(MAX_UAVS)), dtype=np.float64))))
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            pending -= 1
            info = after
        pay = float(info["cumulative_payload_bits"])
        payloads.append(pay)
        per_seed_payload.append(pay)
        exacts.append(float(np.mean(local["exact"])))
        per_uav.append(float(np.mean(local["per"])))
        conflicts.append(float(np.mean(local["conf"])))
        utilities.append(float(np.mean(local["util"])))
        idles.append(float(np.mean(local["idle"])))
        services.append(float(np.mean(local["svc"])))
        tf_accs.append(float(np.mean(local["tf"])))
        fr_accs.append(float(np.mean(local["fr"])))
        tf_nlls.append(float(np.nanmean(local["nll"])))
        exact_tf.append(float(np.mean(local["etf"])))
        mean_rates.append(float(np.mean(local["rate"])))
    tf_mean = float(np.mean(tf_accs))
    fr_mean = float(np.mean(fr_accs))
    return {
        "mean_payload_bits": float(np.mean(payloads)),
        "per_seed_payload": per_seed_payload,
        "exact_match_rate": float(np.mean(exacts)),
        "per_uav_accuracy": float(np.mean(per_uav)),
        "conflict_rate": float(np.mean(conflicts)),
        "idle_fraction": float(np.mean(idles)),
        "interference_utility": float(np.mean(utilities)),
        "mean_rate_bps": float(np.mean(mean_rates)),
        "service_success_rate": float(np.mean(services)),
        "teacher_forcing_accuracy": tf_mean,
        "free_running_accuracy": fr_mean,
        "exposure_gap": float(tf_mean - fr_mean),
        "sequence_nll": float(np.mean(tf_nlls)),
        "teacher_forcing_exact_match": float(np.mean(exact_tf)),
        "hold": HOLD,
        "seeds": list(seeds),
    }


def evaluate_rule(assign_fn: Callable[[L3BEnv], np.ndarray], seeds: tuple[int, ...] = EVAL_SEEDS) -> dict[str, Any]:
    rows = [rollout(L3BEnv(), assign_fn, int(seed), hold=HOLD, compare_oracle=True) for seed in seeds]
    payloads = [float(row["payload_bits"]) for row in rows]
    return {
        "mean_payload_bits": float(np.mean(payloads)),
        "per_seed_payload": payloads,
        "conflict_rate": float(np.mean([row["mean_conflict_rate"] for row in rows])),
        "mean_rate_bps": float(np.mean([row["mean_rate_bps"] for row in rows])),
        "idle_fraction": float(1.0 - np.mean([row["mean_active_frac"] for row in rows])),
        "assignment_accuracy": float(np.mean([row["oracle_agreement"] for row in rows])),
        "hold": HOLD,
    }


def evaluate_oracle(seeds: tuple[int, ...] = EVAL_SEEDS) -> dict[str, Any]:
    return evaluate_rule(oracle_option_assignment, seeds)


def evaluate_exclusive(seeds: tuple[int, ...] = EVAL_SEEDS) -> dict[str, Any]:
    return evaluate_rule(exclusive_matching, seeds)


def attach_regret(result: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    j_star_mean = float(oracle["mean_payload_bits"])
    j_hat_mean = float(result["mean_payload_bits"])
    result = dict(result)
    result["oracle_regret"] = oracle_regret(j_star_mean, j_hat_mean)
    result["normalized_oracle_regret"] = normalized_regret(j_star_mean, j_hat_mean)
    if "per_seed_payload" in result and "per_seed_payload" in oracle:
        star = np.asarray(oracle["per_seed_payload"], dtype=np.float64)
        hat = np.asarray(result["per_seed_payload"], dtype=np.float64)
        result["per_seed_regret"] = (star - hat).tolist()
        result["per_seed_nregret"] = ((star - hat) / (np.abs(star) + 1.0e-9)).tolist()
    return result


__all__ = [
    "attach_regret",
    "decode_sequence",
    "evaluate_ar_model",
    "evaluate_exclusive",
    "evaluate_oracle",
]
