"""Frozen-checkpoint diagnostics. No optimizer steps and no architecture edits."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn.functional import softmax

from wt.htw_phase9.const import IDLE_INDEX, MAX_UAVS
from wt.htw_phase92.run_phase92 import BATCH, LR
from wt.htw_phase943.evaluation import decode_assignment, encode_env
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel, pick_candidate
from wt.htw_phase943.set_context import UNDECIDED, all_candidate_physics_set, apply_choice, empty_assignment, legal_candidate_mask, make_snap
from wt.htw_phase943.set_likelihood_dp import set_nll
from wt.htw_phase943.training import _encode_one
from wt.htw_phase94.metrics import exact_match, idle_fraction, per_uav_accuracy
from wt.l3b.env import L3BEnv
from wt.l3b.policies import assign_split, fly_toward, oracle_option_assignment

from .hooks import encode_layers, effective_rank_np, pairwise_cosine_mean
from .protocol import N_GRAD


def item_tensors(data: dict[str, np.ndarray], index: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    out = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        out[key] = torch.as_tensor(data[key][index : index + 1], dtype=dtype, device=device)
    return out


def clone_batch(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in tensors.items()}


def param_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        total += float(param.detach().float().pow(2).sum().cpu())
    return float(np.sqrt(total))


def grad_norm_on_items(model: OrderFreeAssignmentModel, data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    model.train()
    device = "cpu"
    losses = []
    for index in indices[:N_GRAD]:
        encoded = _encode_one(model, data, int(index), device)
        snap = make_snap(data["uav_positions"][int(index)], data["buoy_positions"][int(index)])
        target = np.asarray(data["assignment"][int(index)], dtype=np.int64)
        losses.append(set_nll(model, encoded, snap, target, data["buoy_mask"][int(index)]))
    loss = torch.stack(losses).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    sq = 0.0
    absmax = 0.0
    n_zero = 0
    n_param = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        sq += float(grad.float().pow(2).sum().cpu())
        absmax = max(absmax, float(grad.abs().max().cpu()))
        n_zero += int((grad.abs() < 1.0e-12).sum().cpu())
        n_param += int(grad.numel())
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "grad_l2": float(np.sqrt(sq)),
        "grad_absmax": absmax,
        "grad_zero_frac": n_zero / max(n_param, 1),
        "nll_batch": float(loss.detach().cpu()),
        "param_l2": param_norm(model),
        "lr": LR,
        "batch": BATCH,
    }


@torch.no_grad()
def first_step_logits(model: OrderFreeAssignmentModel, tensors: dict[str, torch.Tensor], snap, assignment: np.ndarray, buoy_mask: np.ndarray) -> torch.Tensor:
    encoded = model.encode(tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"])
    physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32)
    return model.decoder(encoded, assignment, physics, buoy_mask)


def flatten_legal_logits(logits: torch.Tensor, assignment: np.ndarray, buoy_mask: np.ndarray) -> np.ndarray:
    legal = legal_candidate_mask(assignment, buoy_mask)
    values = logits.detach().cpu().numpy()
    return values[legal]


def entropy_from_logits(logits: torch.Tensor, assignment: np.ndarray, buoy_mask: np.ndarray) -> float:
    legal = torch.as_tensor(legal_candidate_mask(assignment, buoy_mask), dtype=torch.bool, device=logits.device)
    flat = logits.reshape(-1)
    masked = torch.where(legal.reshape(-1), flat, torch.full_like(flat, -1.0e9))
    prob = softmax(masked, dim=0)
    kept = prob[legal.reshape(-1)].clamp_min(1.0e-12)
    return float((-(kept * kept.log()).sum()).cpu())


@torch.no_grad()
def decode_with_teacher(model: OrderFreeAssignmentModel, data: dict[str, np.ndarray], index: int, teacher: bool) -> dict[str, Any]:
    tensors = item_tensors(data, index)
    encoded = model.encode(tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"])
    snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
    target = np.asarray(data["assignment"][index], dtype=np.int64)
    buoy_mask = np.asarray(data["buoy_mask"][index], dtype=bool)
    assignment = empty_assignment(MAX_UAVS)
    hits = []
    margins = []
    entropies = []
    idle_margins = []
    first_error = None
    after_error_hits = []
    seen_error = False
    for step in range(MAX_UAVS):
        if np.all(assignment != UNDECIDED):
            break
        physics = torch.as_tensor(all_candidate_physics_set(snap, assignment)[None, ...], dtype=torch.float32)
        logits = model.decoder(encoded, assignment, physics, buoy_mask)
        legal = legal_candidate_mask(assignment, buoy_mask)
        values = np.where(legal, logits.detach().cpu().numpy(), -1.0e12)
        flat = values.reshape(-1)
        order = np.argsort(flat)[::-1]
        top1 = float(flat[order[0]])
        top2 = float(flat[order[1]]) if order.size > 1 else float(top1)
        margins.append(top1 - top2)
        entropies.append(entropy_from_logits(logits, assignment, buoy_mask))
        idle_logit = values[:, IDLE_INDEX]
        best_buoy = np.max(values[:, :IDLE_INDEX], axis=1)
        idle_margins.append(float(np.mean(idle_logit[:6] - best_buoy[:6])))
        uav, token = pick_candidate(logits, snap, assignment, buoy_mask)
        gold_token = IDLE_INDEX if int(target[uav]) < 0 else int(target[uav])
        hit = int(token == gold_token and assignment[uav] == UNDECIDED)
        hits.append(hit)
        if not hit and first_error is None:
            first_error = step
            seen_error = True
        if seen_error:
            after_error_hits.append(hit)
        if teacher:
            assignment = apply_choice(assignment, uav, gold_token)
        else:
            assignment = apply_choice(assignment, uav, token)
    pred = np.where(assignment == UNDECIDED, -1, assignment)
    pred = np.where(pred == IDLE_INDEX, -1, pred)
    return {
        "tf_or_fr": "teacher" if teacher else "free",
        "per_uav": float(np.mean(hits)) if hits else float("nan"),
        "exact": float(exact_match(pred[: target.shape[0]], target)),
        "idle": float(idle_fraction(pred[: target.shape[0]])),
        "margin_mean": float(np.mean(margins)) if margins else float("nan"),
        "entropy_mean": float(np.mean(entropies)) if entropies else float("nan"),
        "idle_margin_mean": float(np.mean(idle_margins)) if idle_margins else float("nan"),
        "first_error_step": -1 if first_error is None else int(first_error),
        "after_error_acc": float(np.mean(after_error_hits)) if after_error_hits else float("nan"),
        "n_steps": len(hits),
        "assignment": pred[: target.shape[0]].copy(),
        "step_hits": hits,
        "step_margins": margins,
        "step_entropy": entropies,
    }


@torch.no_grad()
def snapshot_nll(model: OrderFreeAssignmentModel, data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    nlls = []
    for index in indices:
        encoded = _encode_one(model, data, int(index), "cpu")
        snap = make_snap(data["uav_positions"][int(index)], data["buoy_positions"][int(index)])
        target = np.asarray(data["assignment"][int(index)], dtype=np.int64)
        nlls.append(float(set_nll(model, encoded, snap, target, data["buoy_mask"][int(index)]).cpu()))
    return {"set_nll": float(np.mean(nlls)), "set_nll_std": float(np.std(nlls)), "n": int(len(nlls))}


def layer_stats(model: OrderFreeAssignmentModel, data: dict[str, np.ndarray], indices: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        tensors = item_tensors(data, int(index))
        layers = encode_layers(model, tensors)
        uav_mask = tensors["uav_mask"][0]
        buoy_mask = tensors["buoy_mask"][0]
        for layer in layers:
            h_u = layer["h_u"][0]
            h_b = layer["h_b"][0]
            e_ub = layer["e_ub"][0]
            u_valid = h_u[uav_mask]
            b_valid = h_b[buoy_mask]
            pair = uav_mask[:, None] & buoy_mask[None, :]
            e_valid = e_ub[pair]
            ub_msg = layer["ub_msg"][0][pair] if layer["ub_msg"].dim() == 4 else layer["ub_msg"].reshape(-1, layer["ub_msg"].shape[-1])
            uu = layer["uu_msg"][0]
            uu_off = ~torch.eye(uu.shape[0], dtype=torch.bool, device=uu.device)
            uu_valid = uu[uav_mask[:, None] & uav_mask[None, :] & uu_off]
            graph = layer.get("graph")
            rows.append(
                {
                    "index": int(index),
                    "layer": layer["name"],
                    "uav_var": float(u_valid.var().cpu()) if u_valid.numel() else float("nan"),
                    "buoy_var": float(b_valid.var().cpu()) if b_valid.numel() else float("nan"),
                    "uav_std_mean": float(u_valid.std(dim=0).mean().cpu()) if u_valid.numel() else float("nan"),
                    "buoy_std_mean": float(b_valid.std(dim=0).mean().cpu()) if b_valid.numel() else float("nan"),
                    "os_u": pairwise_cosine_mean(h_u, uav_mask),
                    "os_b": pairwise_cosine_mean(h_b, buoy_mask),
                    "erank_uav": effective_rank_np(u_valid.cpu().numpy()) if u_valid.numel() else float("nan"),
                    "erank_buoy": effective_rank_np(b_valid.cpu().numpy()) if b_valid.numel() else float("nan"),
                    "erank_edge": effective_rank_np(e_valid.cpu().numpy()) if e_valid.numel() else float("nan"),
                    "uav_norm": float(u_valid.norm(dim=-1).mean().cpu()) if u_valid.numel() else float("nan"),
                    "buoy_norm": float(b_valid.norm(dim=-1).mean().cpu()) if b_valid.numel() else float("nan"),
                    "uav_pdist": float(torch.pdist(u_valid).mean().cpu()) if u_valid.shape[0] > 1 else float("nan"),
                    "collapse_u": pairwise_cosine_mean(h_u, uav_mask),
                    "ub_msg_norm": float(ub_msg.norm(dim=-1).mean().cpu()) if ub_msg.numel() else 0.0,
                    "uu_msg_norm": float(uu_valid.norm(dim=-1).mean().cpu()) if uu_valid.numel() else 0.0,
                    "graph_norm": float(graph.norm(dim=-1).mean().cpu()) if graph is not None else float("nan"),
                }
            )
    return rows


def perturb_tensors(tensors: dict[str, torch.Tensor], kind: str, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    out = clone_batch(tensors)
    if kind == "full":
        return out
    if kind == "zero_uu":
        out["edge_uu"].zero_()
    elif kind == "shuffle_uu":
        perm = rng.permutation(int(out["edge_uu"].shape[1]))
        out["edge_uu"] = out["edge_uu"][:, perm][:, :, perm]
    elif kind == "zero_ub":
        out["edge"].zero_()
    elif kind == "shuffle_ub":
        perm = rng.permutation(int(out["edge"].shape[2]))
        out["edge"] = out["edge"][:, :, perm]
    elif kind == "zero_global":
        out["global"].zero_()
    elif kind == "shuffle_competition":
        perm = rng.permutation(int(out["buoy"].shape[1]))
        out["buoy"] = out["buoy"][:, perm]
        out["edge"] = out["edge"][:, :, perm]
        out["buoy_mask"] = out["buoy_mask"][:, perm]
    else:
        raise ValueError(kind)
    return out


def assignment_disagreement(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    n = min(left.size, right.size)
    return float(np.mean(left[:n] != right[:n]))


def rollout_assignment(seed: int, assignment: np.ndarray) -> float:
    from wt.htw_phase9.const import HOLD

    env = L3BEnv()
    info = env.reset(seed=int(seed))
    pending = 0
    chosen = np.asarray(assignment, dtype=np.int64)
    while not info.get("terminated", False):
        if pending <= 0:
            pending = int(HOLD)
        info = env.step(fly_toward(env, chosen), chosen)
        pending -= 1
    return float(info["cumulative_payload_bits"])


def env_batch(seed: int) -> tuple[L3BEnv, dict[str, np.ndarray], np.ndarray]:
    from wt.htw_phase92.run_phase92 import _batch_from_env

    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    batch = _batch_from_env(info, env, last_arrival, last_positions)
    heuristic = oracle_option_assignment(env)
    return env, batch, heuristic


def tensors_from_batch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    out = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        out[key] = torch.as_tensor(batch[key], dtype=dtype)
    return out


def shadow_split_payloads(env: L3BEnv) -> list[dict[str, Any]]:
    rows = []
    n_uav = int(env.config.num_uavs)
    from wt.l3b.policies import _cluster_work, _shadow_payload

    near_work, far_work = _cluster_work(env)
    for n_near in range(n_uav + 1):
        for n_far in range(n_uav - n_near + 1):
            if far_work <= 1.0e-9 and n_far > 0:
                continue
            if near_work <= 1.0e-9 and n_near > 0:
                continue
            assign = assign_split(env, n_near, n_far)
            value = _shadow_payload(env, n_near, n_far)
            rows.append({"n_near": n_near, "n_far": n_far, "assignment": assign.copy(), "payload": float(value)})
    return rows


def random_exclusive_assignments(env: L3BEnv, rng: np.random.Generator, count: int = 12) -> list[np.ndarray]:
    rows = []
    k = int(env.config.num_buoys)
    m = int(env.config.num_uavs)
    for _ in range(count):
        assign = np.full(m, -1, dtype=np.int64)
        buoys = rng.permutation(k).tolist()
        taken = set()
        for uav in range(m):
            if rng.random() < 0.35:
                continue
            for buoy in buoys:
                if buoy in taken:
                    continue
                assign[uav] = int(buoy)
                taken.add(int(buoy))
                break
        rows.append(assign)
    return rows
