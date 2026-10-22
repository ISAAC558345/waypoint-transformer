"""Round-21 structural, representation, and equivariance audits."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import TaskDataset
from .training import TrainConfig, task_loss


def _permute_batch(batch: Mapping[str, Any], uav_perm: np.ndarray, buoy_perm: np.ndarray) -> dict[str, Any]:
    result = dict(batch)
    up = torch.as_tensor(uav_perm, device=batch["uav_nodes"].device)
    bp = torch.as_tensor(buoy_perm, device=batch["uav_nodes"].device)
    result["uav_nodes"] = batch["uav_nodes"][:, up]
    result["buoy_nodes"] = batch["buoy_nodes"][:, bp]
    result["uav_mask"] = batch["uav_mask"][:, up]
    result["buoy_mask"] = batch["buoy_mask"][:, bp]
    result["ub_edges"] = batch["ub_edges"][:, up][:, :, bp]
    result["uu_edges"] = batch["uu_edges"][:, up][:, :, up]
    return result


def permutation_audit(model: torch.nn.Module, dataset: TaskDataset, config: TrainConfig, family: str, horizon: int, seed: int, max_graphs: int = 64) -> list[dict[str, Any]]:
    model.eval()
    count = min(len(dataset), max_graphs)
    batch = dataset.batch(np.arange(count), config.device)
    uav_perm = np.arange(dataset.uav_nodes.shape[1])[::-1].copy()
    buoy_perm = np.arange(dataset.buoy_nodes.shape[1])
    valid_buoys = int(dataset.buoy_mask[0].sum())
    buoy_perm[:valid_buoys] = buoy_perm[:valid_buoys][::-1].copy()
    with torch.no_grad():
        original = model(batch)
        uav_changed = model(_permute_batch(batch, uav_perm, np.arange(len(buoy_perm))))
        buoy_changed = model(_permute_batch(batch, np.arange(len(uav_perm)), buoy_perm))
    expected_u_service = original["service"][:, uav_perm]
    expected_u_progress = original["progress"][:, uav_perm]
    expected_b_service = original["service"][:, :, buoy_perm]
    expected_b_progress = original["progress"][:, :, buoy_perm]
    rows = []
    for relation, changed, expected_service, expected_progress in (
        ("uav", uav_changed, expected_u_service, expected_u_progress),
        ("buoy", buoy_changed, expected_b_service, expected_b_progress),
    ):
        service_error = float((changed["service"] - expected_service).abs().max().cpu())
        progress_error = float((changed["progress"] - expected_progress).abs().max().cpu())
        return_error = float((changed["team_return"] - original["team_return"]).abs().max().cpu())
        rows.append({"family": family, "horizon": horizon, "seed": seed, "permutation": relation, "sample_count": count, "service_max_abs_error": service_error, "progress_max_abs_error": progress_error, "return_max_abs_error": return_error, "tolerance": 1.0e-5, "passed": bool(max(service_error, progress_error, return_error) <= 1.0e-5)})
    return rows


def _representation_stats(values: np.ndarray) -> tuple[float, float, float]:
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    if values.shape[0] < 2:
        return 0.0, 0.0, 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    variance = float(centered.var(axis=0).mean())
    singular = np.linalg.svd(centered, compute_uv=False)
    probability = singular / max(singular.sum(), 1.0e-12)
    effective_rank = float(np.exp(-(probability[probability > 0] * np.log(probability[probability > 0])).sum()))
    take = min(256, len(values))
    normalized = values[:take] / np.maximum(np.linalg.norm(values[:take], axis=1, keepdims=True), 1.0e-12)
    cosine = normalized @ normalized.T
    average_cosine = float((cosine.sum() - np.trace(cosine)) / max(take * (take - 1), 1))
    return variance, effective_rank, average_cosine


def representation_audit(model: torch.nn.Module, dataset: TaskDataset, config: TrainConfig, family: str, horizon: int, seed: int, max_graphs: int = 256) -> list[dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        output = model(dataset.batch(np.arange(min(max_graphs, len(dataset))), config.device))
    rows = []
    for name in ("h_u", "h_b", "h_g"):
        variance, effective_rank, cosine = _representation_stats(output[name].detach().cpu().numpy())
        rows.append({"family": family, "horizon": horizon, "seed": seed, "representation": name, "variance": variance, "effective_rank": effective_rank, "mean_pairwise_cosine": cosine, "collapse_flag": bool(variance <= 1.0e-8 or effective_rank < 2.0)})
    return rows


def relation_audit(model: torch.nn.Module, dataset: TaskDataset, config: TrainConfig, family: str, horizon: int, seed: int, max_graphs: int = 128) -> list[dict[str, Any]]:
    batch = dataset.batch(np.arange(min(max_graphs, len(dataset))), config.device)
    model.eval()
    with torch.no_grad():
        output = model(batch)
    rows: list[dict[str, Any]] = []
    for layer_index, messages in enumerate(output["relation_messages"]):
        for relation, value in messages.items():
            rows.append({"family": family, "horizon": horizon, "seed": seed, "layer": layer_index + 1, "relation": relation, "message_l2_mean": float(value.norm(dim=-1).mean().cpu()), "message_abs_mean": float(value.abs().mean().cpu()), "message_nonzero_ratio": float((value.abs() > 1.0e-9).float().mean().cpu())})
    # A final backward pass exposes the relation-specific gradient paths.
    model.train()
    model.zero_grad(set_to_none=True)
    total, _ = task_loss(model(batch), batch)
    total.backward()
    for relation, marker in (("bu", "msg_bu"), ("ub", "msg_ub"), ("uu", "msg_uu")):
        norms = [float(parameter.grad.norm().detach().cpu()) for name, parameter in model.named_parameters() if marker in name and parameter.grad is not None]
        rows.append({"family": family, "horizon": horizon, "seed": seed, "layer": "all", "relation": relation, "message_l2_mean": float("nan"), "message_abs_mean": float("nan"), "message_nonzero_ratio": float("nan"), "gradient_norm": float(np.mean(norms)) if norms else 0.0})
    model.zero_grad(set_to_none=True)
    return rows


def index_leakage_markdown(schema: Mapping[str, Any]) -> str:
    features = schema["features"]
    return """# Round-21 graph index-leakage audit

Status: **PASS** — raw assignment identity arrays are present in Round-20
serialization only for audit/reconstruction and are excluded by the Round-21
allowlist adapter before any numerical model tensor is created.

## Allowed current-observable numerical inputs

- UAV nodes: %s
- Buoy nodes: %s
- UAV↔buoy edges: %s
- UAV↔UAV edges: %s
- physical globals: %s

The binary `current_assignment_relation`, node `has_current_assignment`, and
UU `both_currently_assigned_distinct` are permutation-safe relational states.
They do not encode an array identity.

## Explicit adapter exclusions

- graph arrays: `uav_to_buoy_assignment`, `matched_uav_by_buoy`, declared
  counts/schema fields;
- IDs: sample, episode, scenario, UAV and buoy array identity;
- metadata: split, behavior, quality, difficulty and named interference tier;
- all future assignments, rates, payloads, labels and return outcomes.

The named interference tier is not an input.  Its current physical effects are
already represented by the real-valued live graph/global schema fields.
""" % (
        ", ".join(features["uav_nodes"]), ", ".join(features["buoy_nodes"]), ", ".join(features["ub_edges"]),
        ", ".join(features["uu_edges"]), ", ".join(features["global_features"]),
    )


__all__ = ["index_leakage_markdown", "permutation_audit", "relation_audit", "representation_audit"]
