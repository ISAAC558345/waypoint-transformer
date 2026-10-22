"""Offline collision audit. No optimizer and no new J15 rollout."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from wt.l3b.config import L3BConfig
from wt.l3b.env import kappa_for, path_gain, sinr_rates_for_links
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _replay, _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _build_states, _combine, _pack, _scale
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5y_executor_value_distillation.value import ExecutorConditionedJointValue
from wt.lowlevel_v2.phase5z_representation_collision_audit.layers import layer_tensors
from wt.lowlevel_v2.phase5z_representation_collision_audit.metrics import choose_gate
from wt.lowlevel_v2.phase5z_representation_collision_audit.protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMUM,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CONFIRM_SCENARIOS,
    COSINE_LIMIT,
    CURRENT_SPEARMAN_WEAK,
    DEPLOYMENT_VERIFIED,
    EMBED_COLLISION_HIGH,
    HEAD_GAP_FRACTION,
    LARGE_GAP_COLLISION,
    OUTPUT,
    PAIR_SPEARMAN_FLOOR,
    PAIR_SPEARMAN_GAIN,
    PAYLOAD_BLOCK_BITS,
    PHY_GAIN,
    PHY_SPEARMAN,
    PLANNER_READY,
    POOL_FRACTION,
    PRE_SEPARATE_FRACTION,
    REL_DIM,
    REPORT,
    SWAP_WIPE_FRACTION,
    TEMP_COSINE_FLAT,
    TEMP_DISAGREEMENT,
    TEMP_INPUT_DIM,
    TRACE_SEED,
    VALUE_GAP_BLOCKS,
    assert_consumed,
)

PHASE5X = OUTPUT.parent / "phase5x_executor_lookahead" / "phase5x_executor_rollouts.csv"
PHASE5Y_PRED = OUTPUT.parent / "phase5y_executor_value_distillation" / "phase5y_value_predictions.csv"
PHASE5Y_CKPT = OUTPUT.parent / "phase5y_executor_value_distillation" / "checkpoints" / f"phase5y_seed{TRACE_SEED}.pt"
PHASE5W = OUTPUT.parent / "phase5w_residual_mechanism_audit" / "phase5w_residual_case_hybrids.csv"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits: str = ".4f") -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return format(number, digits)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def _vector(assignment) -> str:
    return ";".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64))


def _corr(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or float(np.std(x)) < 1.0e-8 or float(np.std(y)) < 1.0e-8:
        return float("nan")
    value = spearmanr(x, y).correlation
    return float(value) if np.isfinite(value) else float("nan")


def _kendall(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or float(np.std(x)) < 1.0e-8 or float(np.std(y)) < 1.0e-8:
        return float("nan")
    value = kendalltau(x, y).correlation
    return float(value) if np.isfinite(value) else float("nan")


def _gram(matrix: np.ndarray) -> np.ndarray:
    flat = np.asarray(matrix, dtype=np.float64).reshape(matrix.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(norms, 1.0e-8)
    return flat @ flat.T


def _off_mean(gram: np.ndarray) -> float:
    count = int(gram.shape[0])
    if count < 2:
        return float("nan")
    return float((np.sum(gram) - np.trace(gram)) / (count * (count - 1)))


def _effective_rank(matrix: np.ndarray) -> dict[str, float]:
    flat = np.asarray(matrix, dtype=np.float64).reshape(matrix.shape[0], -1)
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(np.sum(energy))
    if total <= 1.0e-12:
        return {"effective_rank": 0.0, "top1_variance": float("nan"), "median_pairwise_cosine": _off_mean(_gram(flat))}
    return {
        "effective_rank": float((np.sum(singular) ** 2) / np.sum(singular**2)),
        "top1_variance": float(energy[0] / total),
        "median_pairwise_cosine": _off_mean(_gram(flat)),
    }


def _load_j15() -> dict[tuple[int, int, str], float]:
    labels = defaultdict(list)
    with PHASE5X.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) != 15:
                continue
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            labels[(scenario, int(row["time_step"]), str(row["assignment_vector"]))].append(float(row["payload_h"]))
    cleaned = {}
    for key, values in labels.items():
        if float(np.max(values) - np.min(values)) > 1.0:
            raise RuntimeError(f"conflicting saved J15 at {key}")
        cleaned[key] = float(np.mean(values))
    return cleaned


def _load_saved_values() -> dict[tuple[int, int, str], float]:
    found = {}
    with PHASE5Y_PRED.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) != int(TRACE_SEED):
                continue
            found[(int(row["scenario_id"]), int(row["time_step"]), str(row["assignment_vector"]))] = float(row["v_exec"])
    return found


def _edge_raw(snapshot: dict, assignments: np.ndarray, config: L3BConfig) -> np.ndarray:
    """Same 12 current-state edge numbers used by relational_matrix, kept per UAV."""
    uav = np.asarray(snapshot["uav_pos"], dtype=np.float64)
    buoy = np.asarray(snapshot["buoy_pos"], dtype=np.float64)
    queue = np.asarray(snapshot["queue"], dtype=np.float64)
    urgency = np.asarray(snapshot["urgency"], dtype=np.float64)
    progress = np.asarray(snapshot["progress"], dtype=np.float64)
    current = np.asarray(snapshot["current_assignment"], dtype=np.int64)
    distance = np.asarray(snapshot["distance"], dtype=np.float64)
    eta = np.asarray(snapshot["eta"], dtype=np.float64)
    rate = np.asarray(snapshot["rate"], dtype=np.float64)
    diag = float(np.hypot(config.area_width, config.area_height))
    capacity = max(float(config.queue_capacity_blocks), 1.0e-9)
    horizon = max(float(config.episode_horizon), 1.0)
    reference = max(float(snapshot["reference_rate"]), 1.0e-9)
    radius = float(config.service_radius_m)
    occupied = np.zeros(queue.shape[0], dtype=np.float64)
    for target in current:
        if int(target) >= 0:
            occupied[int(target)] = 1.0
    output = np.zeros((assignments.shape[0], assignments.shape[1], 12), dtype=np.float32)
    for index, assignment in enumerate(np.asarray(assignments, dtype=np.int64)):
        for uav_id, target in enumerate(assignment):
            if int(target) < 0:
                continue
            buoy_id = int(target)
            offset = buoy[buoy_id] - uav[uav_id]
            gain = path_gain(config, uav[uav_id], buoy[buoy_id])
            output[index, uav_id] = (
                float(distance[uav_id, buoy_id]) / diag,
                float(eta[uav_id, buoy_id]) / horizon,
                float(offset[0]) / float(config.area_width),
                float(offset[1]) / float(config.area_height),
                float(rate[uav_id, buoy_id]) / reference,
                float(np.log1p(gain)),
                float(queue[buoy_id]) / capacity,
                float(urgency[buoy_id]),
                float(progress[uav_id]),
                float(distance[uav_id, buoy_id] <= radius),
                float(int(current[uav_id]) == buoy_id),
                float(occupied[buoy_id]),
            )
    return output


def _pair_interference(snapshot: dict, assignments: np.ndarray, config: L3BConfig) -> np.ndarray:
    uav = np.asarray(snapshot["uav_pos"], dtype=np.float64)
    buoy = np.asarray(snapshot["buoy_pos"], dtype=np.float64)
    output = np.zeros(assignments.shape[0], dtype=np.float64)
    for index, assignment in enumerate(np.asarray(assignments, dtype=np.int64)):
        links = [(uav_id, int(target)) for uav_id, target in enumerate(assignment) if int(target) >= 0]
        terms = []
        for left in range(len(links)):
            uav_i, buoy_i = links[left]
            for right in range(left + 1, len(links)):
                uav_j, buoy_j = links[right]
                gain_ji = path_gain(config, uav[uav_j], buoy[buoy_i])
                gain_ij = path_gain(config, uav[uav_i], buoy[buoy_j])
                terms.append(gain_ji * float(config.tx_power) * kappa_for(config, buoy_j, buoy_i))
                terms.append(gain_ij * float(config.tx_power) * kappa_for(config, buoy_i, buoy_j))
        output[index] = float(np.mean(terms)) if terms else 0.0
    return output


def _phy_now(env, assignments: np.ndarray) -> np.ndarray:
    """Joint SINR/rate if this assignment were the link set at the current positions."""
    scores = np.zeros(assignments.shape[0], dtype=np.float64)
    for index, assignment in enumerate(np.asarray(assignments, dtype=np.int64)):
        links = [(uav_id, int(target)) for uav_id, target in enumerate(assignment) if int(target) >= 0]
        _sinr, rate = sinr_rates_for_links(env, links)
        scores[index] = float(np.sum(rate))
    return scores


def _edge_mean_rate(snapshot: dict, assignments: np.ndarray) -> np.ndarray:
    rate = np.asarray(snapshot["rate"], dtype=np.float64)
    output = np.zeros(assignments.shape[0], dtype=np.float64)
    for index, assignment in enumerate(np.asarray(assignments, dtype=np.int64)):
        chosen = [float(rate[uav_id, int(target)]) for uav_id, target in enumerate(assignment) if int(target) >= 0]
        output[index] = float(np.mean(chosen)) if chosen else 0.0
    return output


def _blocks(labels: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(labels, dtype=np.float64) / PAYLOAD_BLOCK_BITS)


def _upper(count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(count, k=1)


def _cosine_at(gram: np.ndarray, left: int, right: int) -> float:
    return float(gram[int(left), int(right)])


def _rank_of(scores: np.ndarray, index: int) -> int:
    values = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(values > values[int(index)]))


def _top3(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:3]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _sign_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    left, right = _upper(int(labels.shape[0]))
    gap = _blocks(labels)[left] - _blocks(labels)[right]
    keep = np.abs(gap) >= 1.0
    if not np.any(keep):
        return float("nan")
    pred = np.asarray(scores, dtype=np.float64)[left] - np.asarray(scores, dtype=np.float64)[right]
    return float(np.mean(np.sign(pred[keep]) == np.sign(gap[keep])))


def _layer_fractions(labels: np.ndarray, grams: dict[str, np.ndarray], values: np.ndarray) -> dict[str, float]:
    left, right = _upper(int(labels.shape[0]))
    gap = np.abs(_blocks(labels)[left] - _blocks(labels)[right])
    decisive = gap >= 1.0
    count = int(np.sum(decisive))
    empty = {
        "decisive_pairs": float(count),
        "edge_cosine_high": float("nan"),
        "relation_cosine_high": float("nan"),
        "prepool_cosine_high": float("nan"),
        "postpool_cosine_high": float("nan"),
        "head_cosine_high": float("nan"),
        "pooling_collision": float("nan"),
        "head_resolution_collision": float("nan"),
        "value_gap_small": float("nan"),
        "pre_separates": float("nan"),
    }
    if count == 0:
        return empty
    pre = grams["prepool"][left, right][decisive]
    post = grams["postpool"][left, right][decisive]
    value_gap = np.abs(values[left] - values[right])[decisive]
    head = grams["head"][left, right][decisive]

    def rate(mask) -> float:
        return float(np.mean(mask))

    return {
        "decisive_pairs": float(count),
        "edge_cosine_high": rate(grams["edge"][left, right][decisive] >= COSINE_LIMIT),
        "relation_cosine_high": rate(grams["relation"][left, right][decisive] >= COSINE_LIMIT),
        "prepool_cosine_high": rate(pre >= COSINE_LIMIT),
        "postpool_cosine_high": rate(post >= COSINE_LIMIT),
        "head_cosine_high": rate(head >= COSINE_LIMIT),
        "pooling_collision": rate((pre < COSINE_LIMIT) & (post >= COSINE_LIMIT)),
        "head_resolution_collision": rate((head < COSINE_LIMIT) & (value_gap < VALUE_GAP_BLOCKS)),
        "value_gap_small": rate(value_gap < VALUE_GAP_BLOCKS),
        "pre_separates": rate(pre < COSINE_LIMIT),
    }


def _gap_rates(labels: np.ndarray, post: np.ndarray, values: np.ndarray, pre: np.ndarray, blocks: int) -> dict[str, float]:
    left, right = _upper(int(labels.shape[0]))
    keep = np.abs(_blocks(labels)[left] - _blocks(labels)[right]) >= float(blocks)
    count = int(np.sum(keep))
    if count == 0:
        return {"pairs": 0.0, "post_cosine_high": float("nan"), "value_gap_small": float("nan"), "pooling_collision": float("nan")}
    return {
        "pairs": float(count),
        "post_cosine_high": float(np.mean(post[left, right][keep] >= COSINE_LIMIT)),
        "value_gap_small": float(np.mean(np.abs(values[left] - values[right])[keep] < VALUE_GAP_BLOCKS)),
        "pooling_collision": float(np.mean((pre[left, right][keep] < COSINE_LIMIT) & (post[left, right][keep] >= COSINE_LIMIT))),
    }


def _nearest(labels: np.ndarray, gram: np.ndarray) -> dict[str, float]:
    cosine = np.array(gram, copy=True)
    np.fill_diagonal(cosine, -np.inf)
    neighbor = np.argmax(cosine, axis=1)
    rows = np.arange(labels.shape[0])
    gap = np.abs(_blocks(labels)[rows] - _blocks(labels)[neighbor])
    close = cosine[rows, neighbor] >= COSINE_LIMIT
    return {
        "assignments": float(labels.shape[0]),
        "contradiction_1": float(np.mean(close & (gap >= 1.0))),
        "contradiction_2": float(np.mean(close & (gap >= 2.0))),
        "contradiction_3": float(np.mean(close & (gap >= 3.0))),
        "median_neighbor_cosine": float(np.median(cosine[rows, neighbor])),
    }


def _scores_from(scorer, embedded: dict, relational: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return scorer(_pack(embedded, relational), torch.as_tensor(assignments, dtype=torch.long)).detach().cpu().numpy()


def _permute_features(embedded: dict, assignments: np.ndarray, perm: np.ndarray) -> tuple[dict, np.ndarray]:
    order = np.asarray(perm, dtype=np.int64)
    updated = {key: value.detach().clone() for key, value in embedded.items()}
    updated["h_u"] = embedded["h_u"][:, order].detach().clone()
    updated["h_edge"] = embedded["h_edge"][:, order].detach().clone()
    updated["h_uu"] = embedded["h_uu"][:, order][:, :, order].detach().clone()
    return updated, np.asarray(assignments, dtype=np.int64)[:, order]


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5z_protocol.json").write_text(
        json.dumps(
            {
                "training": False,
                "new_j15_rollout": False,
                "trace_seed": TRACE_SEED,
                "cosine_limit": COSINE_LIMIT,
                "value_gap_blocks": VALUE_GAP_BLOCKS,
                "pool_fraction": POOL_FRACTION,
                "phy_spearman": PHY_SPEARMAN,
                "opened_310118": False,
                "planner_ready": PLANNER_READY,
                "deployment_verified": DEPLOYMENT_VERIFIED,
                "note": "Pooling collision means ordered pre-pool cosine < 0.99 and pooled cosine >= 0.99. PHY_NOW is the sum of current joint rates, with no fitted weight.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = _build_states(frozen, matching, config)
    labels = _load_j15()
    saved_values = _load_saved_values()
    for state in states:
        assert_consumed(int(state["scenario_id"]))
        saved_j15 = np.asarray(
            [labels[(int(state["scenario_id"]), int(state["time_step"]), vector)] for vector in state["vectors"]],
            dtype=np.float64,
        )
        if float(np.max(np.abs(saved_j15 - state["j_mappo"]))) > 1.0:
            raise RuntimeError(f"Phase-5X J15 does not match the stored label at {state['scenario_id']} t={state['time_step']}")
        state["j_mappo"] = saved_j15
    payload = torch.load(PHASE5Y_CKPT, map_location="cpu", weights_only=False)
    center = np.asarray(payload["center"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    scorer = ExecutorConditionedJointValue()
    scorer.load_state_dict(payload["state_dict"])
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)

    index_rows = []
    collision_rows = []
    phy_rows = []
    pair_rows = []
    identity_rows = []
    rank_rows = []
    nn_rows = []
    large_rows = []
    swap_rows = []
    trace_edge = []
    trace_raw = []
    trace_pre = []
    trace_post = []
    trace_head = []
    trace_rep = []
    trace_temp = []
    trace_values = []
    trace_labels = []
    h_u = []
    h_b = []
    h_uu = []
    max_edge_drift = 0.0
    value_drift = 0.0
    for state_index, state in enumerate(states):
        env, _movement, _batch, diagnosis = _replay(int(state["scenario_id"]), int(state["time_step"]))
        snapshot = _snapshot(env, diagnosis)
        raw = _edge_raw(snapshot, state["assignments"], config)
        reference = relational_matrix(snapshot, state["assignments"], config)
        drift = float(np.max(np.abs(reference - state["z_rel"])))
        max_edge_drift = max(max_edge_drift, drift)
        if drift > 1.0e-4:
            raise RuntimeError(f"relational features drifted at {state['scenario_id']} t={state['time_step']}")
        relational = _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))
        with torch.no_grad():
            layers = layer_tensors(scorer, state["embedded"], torch.as_tensor(state["assignments"], dtype=torch.long))
            values = _scores_from(scorer, state["embedded"], relational, state["assignments"])
        if state["split"] == "confirmation":
            for vector, value in zip(state["vectors"], values):
                saved = saved_values[(int(state["scenario_id"]), int(state["time_step"]), vector)]
                value_drift = max(value_drift, abs(float(value) - saved))
        edge = layers["edge"].detach().cpu().numpy()
        pre = layers["prepool"].detach().cpu().numpy()
        post = layers["postpool"].detach().cpu().numpy()
        relation = np.concatenate((edge.reshape(edge.shape[0], -1), layers["pairs"].detach().cpu().numpy().reshape(edge.shape[0], -1), state["z_rel"]), axis=1)
        head = np.concatenate((post, relational), axis=1)
        grams = {
            "edge": _gram(edge),
            "relation": _gram(relation),
            "prepool": _gram(pre),
            "postpool": _gram(post),
            "head": _gram(head),
            "temp": _gram(relational[:, REL_DIM:]),
            "rep": _gram(state["z_rel"]),
        }
        fractions = _layer_fractions(state["j_mappo"], grams, values)
        fractions.update(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "split": state["split"],
                "temp_pairwise_cosine": _off_mean(grams["temp"]),
                "rep_pairwise_cosine": _off_mean(grams["rep"]),
            }
        )
        collision_rows.append(fractions)
        phy = _phy_now(env, state["assignments"])
        edge_rate = _edge_mean_rate(snapshot, state["assignments"])
        pair_term = _pair_interference(snapshot, state["assignments"], config)
        phy_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "split": state["split"],
                "spearman_phy": _corr(phy, state["j_mappo"]),
                "kendall_phy": _kendall(phy, state["j_mappo"]),
                "top1_phy": metrics_at(phy, state["j_mappo"], int(np.argmax(phy)))["top1"],
                "top3_phy": _top3(phy, state["j_mappo"]),
                "sign_phy": _sign_accuracy(phy, state["j_mappo"]),
                "spearman_value": _corr(values, state["j_mappo"]),
                "top3_value": _top3(values, state["j_mappo"]),
                "sign_value": _sign_accuracy(values, state["j_mappo"]),
            }
        )
        pair_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "split": state["split"],
                "spearman_edge_rate": _corr(edge_rate, state["j_mappo"]),
                "spearman_pair_interference": _corr(pair_term, state["j_mappo"]),
                "spearman_phy": phy_rows[-1]["spearman_phy"],
                "sign_edge_rate": _sign_accuracy(edge_rate, state["j_mappo"]),
                "sign_pair_interference": _sign_accuracy(pair_term, state["j_mappo"]),
                "sign_phy": phy_rows[-1]["sign_phy"],
                "sign_value": phy_rows[-1]["sign_value"],
            }
        )
        left, right = _upper(int(state["j_mappo"].shape[0]))
        hamming = np.asarray([float(np.sum(state["assignments"][a] != state["assignments"][b])) for a, b in zip(left, right)], dtype=np.float64)
        embedding_distance = np.linalg.norm(head.reshape(head.shape[0], -1)[left] - head.reshape(head.shape[0], -1)[right], axis=1)
        identity_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "split": state["split"],
                "spearman_hamming_vs_embedding_l2": _corr(hamming, embedding_distance),
            }
        )
        for name, matrix in (("edge", edge), ("rep", state["z_rel"]), ("prepool", pre), ("postpool", post)):
            measured = _effective_rank(matrix)
            rank_rows.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "split": state["split"],
                    "layer": name,
                    **measured,
                }
            )
        neighbor = _nearest(state["j_mappo"], grams["head"])
        neighbor.update({"scenario_id": int(state["scenario_id"]), "time_step": int(state["time_step"]), "split": state["split"]})
        nn_rows.append(neighbor)
        for blocks in (2, 3, 4):
            measured = _gap_rates(state["j_mappo"], grams["postpool"], values, grams["prepool"], blocks)
            measured.update({"scenario_id": int(state["scenario_id"]), "time_step": int(state["time_step"]), "split": state["split"], "gap_blocks": blocks})
            large_rows.append(measured)
        pool = {vector: index for index, vector in enumerate(state["vectors"])}
        seen = set()
        swapped_assignments = []
        swapped_meta = []
        for index, assignment in enumerate(state["assignments"]):
            for left_uav in range(assignment.shape[0]):
                for right_uav in range(left_uav + 1, assignment.shape[0]):
                    if int(assignment[left_uav]) == int(assignment[right_uav]):
                        continue
                    swapped = np.array(assignment, copy=True)
                    swapped[left_uav], swapped[right_uav] = swapped[right_uav], swapped[left_uav]
                    key = _vector(swapped)
                    token = tuple(sorted((state["vectors"][index], key)))
                    if token in seen:
                        continue
                    seen.add(token)
                    swapped_assignments.append(swapped)
                    swapped_meta.append((index, key, int(assignment[left_uav]) >= 0 and int(assignment[right_uav]) >= 0))
        if swapped_assignments:
            batch = np.stack(swapped_assignments, axis=0)
            swapped_relational = _combine(
                relational_matrix(snapshot, batch, config),
                _scale(temporal_matrix(state["history"], batch, config), center, scale),
            )
            with torch.no_grad():
                swapped_layers = layer_tensors(scorer, state["embedded"], torch.as_tensor(batch, dtype=torch.long))
                swapped_values = _scores_from(scorer, state["embedded"], swapped_relational, batch)
            swapped_post = swapped_layers["postpool"].detach().cpu().numpy()
            swapped_pre = swapped_layers["prepool"].detach().cpu().numpy()
            for local, (parent, key, active_pair) in enumerate(swapped_meta):
                other = pool.get(key)
                j15_gap = ""
                if other is not None:
                    j15_gap = float(state["j_mappo"][parent] - state["j_mappo"][other])
                pre_cos = float(np.dot(pre[parent].reshape(-1), swapped_pre[local].reshape(-1)) / max(np.linalg.norm(pre[parent]), 1e-8) / max(np.linalg.norm(swapped_pre[local]), 1e-8))
                post_cos = float(np.dot(post[parent].reshape(-1), swapped_post[local].reshape(-1)) / max(np.linalg.norm(post[parent]), 1e-8) / max(np.linalg.norm(swapped_post[local]), 1e-8))
                swap_rows.append(
                    {
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "split": state["split"],
                        "assignment": state["vectors"][parent],
                        "swapped": key,
                        "in_pool": int(other is not None),
                        "active_pair": int(active_pair),
                        "delta_j15": j15_gap,
                        "prepool_cosine": pre_cos,
                        "postpool_cosine": post_cos,
                        "delta_v": float(values[parent] - swapped_values[local]),
                    }
                )
        for assignment_index, vector in enumerate(state["vectors"]):
            index_rows.append(
                {
                    "state_index": state_index,
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "split": state["split"],
                    "assignment_index": assignment_index,
                    "assignment_vector": vector,
                    "j15": float(state["j_mappo"][assignment_index]),
                    "v_pred": float(values[assignment_index]),
                }
            )
        trace_edge.append(edge.astype(np.float32))
        trace_raw.append(raw)
        trace_pre.append(pre.astype(np.float32))
        trace_post.append(post.astype(np.float32))
        trace_head.append(head.astype(np.float32))
        trace_rep.append(np.asarray(state["z_rel"], dtype=np.float32))
        trace_temp.append(np.asarray(state["z_temp_raw"], dtype=np.float32))
        trace_values.append(values.astype(np.float32))
        trace_labels.append(np.asarray(state["j_mappo"], dtype=np.float32))
        h_u.append(state["embedded"]["h_u"][0].detach().cpu().numpy().astype(np.float32))
        h_b.append(state["embedded"]["h_b"][0].detach().cpu().numpy().astype(np.float32))
        h_uu.append(state["embedded"]["h_uu"][0].detach().cpu().numpy().astype(np.float32))
        state["values"] = values
        state["relational"] = relational
        state["phy"] = phy
        state["edge_rate"] = edge_rate
        state["pair_term"] = pair_term
        print(f"traced {state['scenario_id']} t={state['time_step']}", flush=True)
    if value_drift > 1.0e-4:
        raise RuntimeError(f"recovered value drifted from the saved Phase-5Y prediction by {value_drift}")

    confirmation = [state for state in states if state["split"] == "confirmation"]
    confirm_collision = [row for row in collision_rows if row["split"] == "confirmation"]
    confirm_phy = [row for row in phy_rows if row["split"] == "confirmation"]
    confirm_pair = [row for row in pair_rows if row["split"] == "confirmation"]

    def boot(rows: list[dict], key: str) -> dict[str, float]:
        return cluster_bootstrap(rows, key, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)

    edge_high = boot(confirm_collision, "edge_cosine_high")
    relation_high = boot(confirm_collision, "relation_cosine_high")
    post_high = boot(confirm_collision, "postpool_cosine_high")
    head_high = boot(confirm_collision, "head_cosine_high")
    pooling = boot(confirm_collision, "pooling_collision")
    head_resolution = boot(confirm_collision, "head_resolution_collision")
    pre_separates = boot(confirm_collision, "pre_separates")
    phy_spearman = boot(confirm_phy, "spearman_phy")
    value_spearman = boot(confirm_phy, "spearman_value")
    edge_spearman = boot(confirm_pair, "spearman_edge_rate")
    pair_spearman = boot(confirm_pair, "spearman_pair_interference")
    temp_cosine = boot(confirm_collision, "temp_pairwise_cosine")
    identity = boot([row for row in identity_rows if row["split"] == "confirmation"], "spearman_hamming_vs_embedding_l2")

    def weighted(rows: list[dict], key: str, weight: str = "decisive_pairs") -> float:
        chosen = [row for row in rows if float(row[weight]) > 0.0 and np.isfinite(float(row[key]))]
        total = sum(float(row[weight]) for row in chosen)
        if total <= 0.0:
            return float("nan")
        return float(sum(float(row[key]) * float(row[weight]) for row in chosen) / total)

    large_summary = []
    for blocks in (2, 3, 4):
        selected = [row for row in large_rows if row["split"] == "confirmation" and int(row["gap_blocks"]) == blocks]
        large_summary.append(
            {
                "gap_blocks": blocks,
                "post_cosine_high": weighted(selected, "post_cosine_high", "pairs"),
                "value_gap_small": weighted(selected, "value_gap_small", "pairs"),
                "pooling_collision": weighted(selected, "pooling_collision", "pairs"),
            }
        )
    labeled_swaps = [row for row in swap_rows if row["split"] == "confirmation" and row["delta_j15"] != "" and abs(float(row["delta_j15"])) >= PAYLOAD_BLOCK_BITS]
    active_labeled = [row for row in labeled_swaps if int(row["active_pair"]) == 1]
    swap_wipe = _mean(float(row["postpool_cosine"]) >= COSINE_LIMIT for row in active_labeled)
    nn_rate = boot([row for row in nn_rows if row["split"] == "confirmation"], "contradiction_1")

    ablation_rows = []
    for name, kind in (("temp", "zero"), ("temp", "shuffle"), ("rep", "zero"), ("rep", "shuffle")):
        changed = []
        value_shift = []
        embed_shift = []
        for state in confirmation:
            relational = np.array(state["relational"], copy=True)
            rep = relational[:, :REL_DIM]
            temp = relational[:, REL_DIM:]
            if kind == "shuffle":
                order = np.random.default_rng(0).permutation(rep.shape[0])
                if name == "temp":
                    temp = temp[order]
                else:
                    rep = rep[order]
            else:
                if name == "temp":
                    temp = np.zeros_like(temp)
                else:
                    rep = np.zeros_like(rep)
            edited = np.concatenate((rep, temp), axis=1).astype(np.float32)
            values = _scores_from(scorer, state["embedded"], edited, state["assignments"])
            base = int(np.argmax(state["values"]))
            edited_choice = int(np.argmax(values))
            changed.append(float(base != edited_choice))
            value_shift.append(float(np.mean(np.abs(values - state["values"]))))
            embed_shift.append(float(np.mean(np.sum((edited - state["relational"]) ** 2, axis=1))))
        ablation_rows.append(
            {
                "block": name,
                "mode": kind,
                "argmax_disagreement": _mean(changed),
                "mean_abs_value_change": _mean(value_shift),
                "mean_feature_l2": _mean(np.sqrt(value) for value in embed_shift),
            }
        )

    case = next(state for state in confirmation if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME)
    pred_index = case["vectors"].index(CASE_PREDICTED)
    opt_index = case["vectors"].index(CASE_OPTIMUM)
    case_relational = case["relational"]
    with torch.no_grad():
        case_layers = layer_tensors(scorer, case["embedded"], torch.as_tensor(case["assignments"], dtype=torch.long))
    case_edge = case_layers["edge"].detach().cpu().numpy()
    case_pre = case_layers["prepool"].detach().cpu().numpy()
    case_post = case_layers["postpool"].detach().cpu().numpy()
    case_head = np.concatenate((case_post, case_relational), axis=1)

    def pair_cosine(matrix: np.ndarray) -> float:
        left = matrix[pred_index].reshape(-1)
        right = matrix[opt_index].reshape(-1)
        return float(np.dot(left, right) / max(np.linalg.norm(left), 1e-8) / max(np.linalg.norm(right), 1e-8))

    case_env, _movement, _batch, case_diagnosis = _replay(CASE_SCENARIO, CASE_TIME)
    case_raw = _edge_raw(_snapshot(case_env, case_diagnosis), case["assignments"], config)
    case_rows = [
        {"layer": "selected_edge_embedding", "cosine": pair_cosine(case_edge), "l2": float(np.linalg.norm(case_edge[pred_index] - case_edge[opt_index])), "pred": "", "opt": "", "opt_rank": ""},
        {"layer": "selected_edge_raw", "cosine": pair_cosine(case_raw), "l2": float(np.linalg.norm(case_raw[pred_index] - case_raw[opt_index])), "pred": "", "opt": "", "opt_rank": ""},
        {"layer": "rep", "cosine": pair_cosine(case["z_rel"]), "l2": float(np.linalg.norm(case["z_rel"][pred_index] - case["z_rel"][opt_index])), "pred": "", "opt": "", "opt_rank": ""},
        {"layer": "temp_raw", "cosine": pair_cosine(case["z_temp_raw"]), "l2": float(np.linalg.norm(case["z_temp_raw"][pred_index] - case["z_temp_raw"][opt_index])), "pred": "", "opt": "", "opt_rank": ""},
        {"layer": "prepool", "cosine": pair_cosine(case_pre), "l2": float(np.linalg.norm(case_pre[pred_index] - case_pre[opt_index])), "pred": "", "opt": "", "opt_rank": ""},
        {"layer": "postpool", "cosine": pair_cosine(case_post), "l2": float(np.linalg.norm(case_post[pred_index] - case_post[opt_index])), "pred": "", "opt": "", "opt_rank": ""},
        {"layer": "value_head_input", "cosine": pair_cosine(case_head), "l2": float(np.linalg.norm(case_head[pred_index] - case_head[opt_index])), "pred": "", "opt": "", "opt_rank": ""},
        {"layer": "predicted_value", "cosine": "", "l2": abs(float(case["values"][pred_index] - case["values"][opt_index])), "pred": float(case["values"][pred_index]), "opt": float(case["values"][opt_index]), "opt_rank": _rank_of(case["values"], opt_index)},
        {"layer": "phy_now", "cosine": "", "l2": abs(float(case["phy"][pred_index] - case["phy"][opt_index])), "pred": float(case["phy"][pred_index]), "opt": float(case["phy"][opt_index]), "opt_rank": _rank_of(case["phy"], opt_index)},
        {"layer": "edge_mean_rate", "cosine": "", "l2": abs(float(case["edge_rate"][pred_index] - case["edge_rate"][opt_index])), "pred": float(case["edge_rate"][pred_index]), "opt": float(case["edge_rate"][opt_index]), "opt_rank": _rank_of(case["edge_rate"], opt_index)},
        {"layer": "pair_interference", "cosine": "", "l2": abs(float(case["pair_term"][pred_index] - case["pair_term"][opt_index])), "pred": float(case["pair_term"][pred_index]), "opt": float(case["pair_term"][opt_index]), "opt_rank": _rank_of(-case["pair_term"], opt_index)},
    ]
    case_snapshot = _snapshot(case_env, case_diagnosis)
    for name, index in (("pred", pred_index), ("opt", opt_index)):
        links = [(uav_id, int(target)) for uav_id, target in enumerate(case["assignments"][index]) if int(target) >= 0]
        sinr, rate = sinr_rates_for_links(case_env, links)
        case_rows.append(
            {
                "layer": f"hypothetical_sinr_{name}",
                "cosine": "",
                "l2": "",
                "pred": ";".join(f"{float(item):.6g}" for item in sinr),
                "opt": ";".join(str(int(item)) for item in case["assignments"][index]),
                "opt_rank": "",
            }
        )
        case_rows.append(
            {
                "layer": f"hypothetical_rate_{name}",
                "cosine": "",
                "l2": "",
                "pred": ";".join(f"{float(item):.6g}" for item in rate),
                "opt": "",
                "opt_rank": "",
            }
        )
    case_rows.append(
        {
            "layer": "same_target_flags",
            "cosine": "",
            "l2": "",
            "pred": ";".join(str(int(item)) for item in case_raw[pred_index, :, 10]),
            "opt": ";".join(str(int(item)) for item in case_raw[opt_index, :, 10]),
            "opt_rank": "",
        }
    )
    case_rows.append(
        {
            "layer": "service_progress",
            "cosine": "",
            "l2": "",
            "pred": ";".join(f"{float(item):.6g}" for item in case_snapshot["progress"]),
            "opt": ";".join(str(int(item)) for item in case_snapshot["current_assignment"]),
            "opt_rank": "",
        }
    )
    perm = np.array([1, 0, 2, 3, 4, 5], dtype=np.int64)
    permuted_embedded, permuted_assignments = _permute_features(case["embedded"], case["assignments"], perm)
    permuted_values = _scores_from(scorer, permuted_embedded, case_relational, permuted_assignments)
    equivariance_gap = float(np.max(np.abs(permuted_values - case["values"])))
    correspondence_rows = [
        {
            "test": "uav_permutation_equivariance",
            "max_abs_value_gap": equivariance_gap,
            "postpool_cosine_high_fraction": "",
            "labeled_swap_pairs": "",
            "wiped_fraction": "",
        },
        {
            "test": "active_swap_correspondence",
            "max_abs_value_gap": "",
            "postpool_cosine_high_fraction": swap_wipe,
            "labeled_swap_pairs": len(active_labeled),
            "wiped_fraction": swap_wipe,
        },
    ]

    temp_zero = next(row for row in ablation_rows if row["block"] == "temp" and row["mode"] == "zero")
    rep_zero = next(row for row in ablation_rows if row["block"] == "rep" and row["mode"] == "zero")
    pooling_flag = bool(float(pooling["mean"]) >= POOL_FRACTION and float(pre_separates["mean"]) >= PRE_SEPARATE_FRACTION and float(swap_wipe) >= SWAP_WIPE_FRACTION)
    head_flag = bool(float(head_high["mean"]) < EMBED_COLLISION_HIGH and float(head_resolution["mean"]) >= HEAD_GAP_FRACTION)
    pair_flag = bool(abs(float(pair_spearman["mean"])) >= abs(float(edge_spearman["mean"])) + PAIR_SPEARMAN_GAIN and abs(float(pair_spearman["mean"])) >= PAIR_SPEARMAN_FLOOR)
    phy_flag = bool(float(phy_spearman["mean"]) >= PHY_SPEARMAN and float(phy_spearman["mean"]) >= float(value_spearman["mean"]) + PHY_GAIN)
    temp_flag = bool(float(temp_zero["argmax_disagreement"]) < TEMP_DISAGREEMENT and float(temp_cosine["mean"]) >= TEMP_COSINE_FLAT)
    gap2 = next(row for row in large_summary if int(row["gap_blocks"]) == 2)
    state_flag = bool(
        float(gap2["post_cosine_high"]) >= LARGE_GAP_COLLISION
        and abs(float(phy_spearman["mean"])) < CURRENT_SPEARMAN_WEAK
        and abs(float(pair_spearman["mean"])) < CURRENT_SPEARMAN_WEAK
        and abs(float(edge_spearman["mean"])) < CURRENT_SPEARMAN_WEAK
    )
    gate = choose_gate(head=head_flag, pooling=pooling_flag, pair=pair_flag, phy=phy_flag, temp=temp_flag, state_insufficient=state_flag)
    advice = {
        "Z-A": "下一阶段只修改 assignment-conditioned pooling，保留 UAV 与 target 的有序对应。不改 HGNN encoder。",
        "Z-B": "下一阶段增加 selected-edge pair interaction。不增加 HGNN 深度。",
        "Z-C": "下一阶段只加入当前反事实的 candidate-conditioned SINR、rate 和 interference。这不是未来 rollout。",
        "Z-D": "下一阶段只修改 candidate-conditioned temporal representation。",
        "Z-E": "joint embedding 已经能分开高低 J15。下一阶段才可以调整 value head。",
        "Z-F": "当前合法观测仍然分不开大的 J15 差异。下一阶段设计 learned executor dynamics 或短程 rollout distillation，不再加静态 HGNN 特征。",
    }[gate]

    mechanism_rows = []
    residual = list(csv.DictReader(PHASE5W.open(encoding="utf-8")))
    for item in residual:
        state = next(row for row in confirmation if int(row["scenario_id"]) == int(item["scenario_id"]) and int(row["time_step"]) == int(item["time_step"]))
        if item["predicted_assignment"] not in state["vectors"] or item["optimal_assignment"] not in state["vectors"]:
            continue
        left = state["vectors"].index(item["predicted_assignment"])
        right = state["vectors"].index(item["optimal_assignment"])
        with torch.no_grad():
            layers = layer_tensors(scorer, state["embedded"], torch.as_tensor(state["assignments"], dtype=torch.long))
        pre = layers["prepool"].detach().cpu().numpy()
        post = layers["postpool"].detach().cpu().numpy()
        edge = layers["edge"].detach().cpu().numpy()

        def cos(matrix) -> float:
            a = matrix[left].reshape(-1)
            b = matrix[right].reshape(-1)
            return float(np.dot(a, b) / max(np.linalg.norm(a), 1e-8) / max(np.linalg.norm(b), 1e-8))

        pre_cos = cos(pre)
        post_cos = cos(post)
        edge_cos = cos(edge)
        value_gap = abs(float(state["values"][left] - state["values"][right]))
        phy_gap = abs(float(state["phy"][left] - state["phy"][right]))
        j_gap = float(state["j_mappo"][left] - state["j_mappo"][right])
        phy_sign = float(state["phy"][left] - state["phy"][right])
        edge_sign = float(state["edge_rate"][left] - state["edge_rate"][right])
        pair_sign = float(state["pair_term"][left] - state["pair_term"][right])
        tags = []
        if pre_cos < COSINE_LIMIT and post_cos >= COSINE_LIMIT:
            tags.append("POOL")
        if edge_cos >= COSINE_LIMIT:
            tags.append("EDGE")
        if np.sign(pair_sign) == np.sign(j_gap) and np.sign(edge_sign) != np.sign(j_gap) and abs(j_gap) >= PAYLOAD_BLOCK_BITS:
            tags.append("PAIR")
        if cos(state["z_temp_raw"]) >= COSINE_LIMIT:
            tags.append("TEMP")
        if np.sign(phy_sign) == np.sign(j_gap) and post_cos >= COSINE_LIMIT and abs(j_gap) >= PAYLOAD_BLOCK_BITS:
            tags.append("PHY")
        if value_gap < VALUE_GAP_BLOCKS and post_cos < COSINE_LIMIT:
            tags.append("HEAD")
        if abs(j_gap) >= PAYLOAD_BLOCK_BITS and np.sign(phy_sign) != np.sign(j_gap) and edge_cos >= COSINE_LIMIT and cos(state["z_temp_raw"]) >= COSINE_LIMIT:
            tags.append("STATE")
        if not tags:
            tags.append("NONE")
        mechanism_rows.append(
            {
                "scenario_id": int(item["scenario_id"]),
                "time_step": int(item["time_step"]),
                "prepool_cosine": pre_cos,
                "postpool_cosine": post_cos,
                "edge_cosine": edge_cos,
                "temp_cosine": cos(state["z_temp_raw"]),
                "rep_cosine": cos(state["z_rel"]),
                "value_gap": value_gap,
                "phy_gap": phy_gap,
                "delta_j15": float(state["j_mappo"][left] - state["j_mappo"][right]),
                "tags": "|".join(tags),
            }
        )

    np.savez_compressed(
        OUTPUT / "phase5z_representation_trace.npz",
        edge=np.stack(trace_edge),
        edge_raw=np.stack(trace_raw),
        prepool=np.stack(trace_pre),
        postpool=np.stack(trace_post),
        head=np.stack(trace_head),
        rep=np.stack(trace_rep),
        temp=np.stack(trace_temp),
        values=np.stack(trace_values),
        j15=np.stack(trace_labels),
        h_u=np.stack(h_u),
        h_b=np.stack(h_b),
        h_uu=np.stack(h_uu),
    )
    _write_csv(OUTPUT / "phase5z_representation_trace_index.csv", index_rows)
    _write_csv(OUTPUT / "phase5z_collision_by_layer.csv", collision_rows)
    _write_csv(OUTPUT / "phase5z_swap_pair_collision.csv", swap_rows)
    _write_csv(OUTPUT / "phase5z_assignment_identity.csv", identity_rows)
    _write_csv(OUTPUT / "phase5z_pair_interaction_audit.csv", pair_rows)
    _write_csv(OUTPUT / "phase5z_phy_now_audit.csv", phy_rows)
    _write_csv(OUTPUT / "phase5z_310114_t15_trace.csv", case_rows)
    _write_csv(OUTPUT / "phase5z_large_gap_collision.csv", large_rows)
    _write_csv(OUTPUT / "phase5z_nearest_neighbor_contradiction.csv", nn_rows)
    _write_csv(OUTPUT / "phase5z_effective_rank.csv", rank_rows)
    _write_csv(OUTPUT / "phase5z_correspondence_test.csv", correspondence_rows)
    _write_csv(OUTPUT / "phase5z_temp_ablation.csv", [row for row in ablation_rows if row["block"] == "temp"])
    _write_csv(OUTPUT / "phase5z_rep_ablation.csv", [row for row in ablation_rows if row["block"] == "rep"])
    _write_csv(OUTPUT / "phase5z_mechanism_classification.csv", mechanism_rows)
    gate_payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "trained": False,
        "trace_seed": TRACE_SEED,
        "value_recovery_max_abs": value_drift,
        "relational_recovery_max_abs": max_edge_drift,
        "edge_cosine_high": edge_high["mean"],
        "relation_cosine_high": relation_high["mean"],
        "postpool_cosine_high": post_high["mean"],
        "head_cosine_high": head_high["mean"],
        "pooling_collision": pooling["mean"],
        "head_resolution_collision": head_resolution["mean"],
        "pre_separates": pre_separates["mean"],
        "swap_wipe": swap_wipe,
        "phy_spearman": phy_spearman["mean"],
        "value_spearman": value_spearman["mean"],
        "edge_spearman": edge_spearman["mean"],
        "pair_spearman": pair_spearman["mean"],
        "temp_cosine": temp_cosine["mean"],
        "temp_zero_disagreement": temp_zero["argmax_disagreement"],
        "rep_zero_disagreement": rep_zero["argmax_disagreement"],
        "identity_spearman": identity["mean"],
        "nn_contradiction": nn_rate["mean"],
        "gap2_post_cosine_high": gap2["post_cosine_high"],
        "case_post_cosine": next(row["cosine"] for row in case_rows if row["layer"] == "postpool"),
        "case_pre_cosine": next(row["cosine"] for row in case_rows if row["layer"] == "prepool"),
        "case_opt_rank": next(row["opt_rank"] for row in case_rows if row["layer"] == "predicted_value"),
        "permutation_value_gap": equivariance_gap,
        "opened_310118": False,
    }
    (OUTPUT / "phase5z_representation_mechanism_gate.json").write_text(json.dumps(_plain(gate_payload), indent=2), encoding="utf-8")
    lines = [
        "# HTW Phase-5Z Assignment-Conditioned Representation Collision Audit",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练，没有重新计算 J15。追踪的是 Phase-5Y seed 0 的冻结 value head。数字来自 310110–310117，scenario-cluster bootstrap。",
        "",
        f"恢复检查：value 与已保存预测的最大绝对差是 {_fmt(value_drift, '.3e')}，relational 特征最大绝对差是 {_fmt(max_edge_drift, '.3e')}。",
        "",
        "## 分层 collision",
        "",
        f"decisive pair 上，selected-edge cosine ≥ 0.99 的比例 {_fmt(edge_high['mean'])}，CI [{_fmt(edge_high['low'])}, {_fmt(edge_high['high'])}]。",
        f"加入 pair interaction 和 REP 后仍 ≥ 0.99 的比例 {_fmt(relation_high['mean'])}。",
        f"post-pool ≥ 0.99 的比例 {_fmt(post_high['mean'])}，CI [{_fmt(post_high['low'])}, {_fmt(post_high['high'])}]。",
        f"value-head 输入 ≥ 0.99 的比例 {_fmt(head_high['mean'])}。",
        f"pre-pool 已不同但 post-pool 撞在一起的比例 {_fmt(pooling['mean'])}。pre-pool 能够分开的比例 {_fmt(pre_separates['mean'])}。",
        f"embedding 已不同但 |ΔV| < 0.1 的比例 {_fmt(head_resolution['mean'])}。",
        "",
        f"有 J15 的主动 swap pair 共 {len(active_labeled)} 个，post-pool cosine ≥ 0.99 的比例 {_fmt(swap_wipe)}。",
        f"Hamming assignment 距离与 embedding L2 的 Spearman {_fmt(identity['mean'])}。",
        f"最近邻 contradiction {_fmt(nn_rate['mean'])}。",
        "",
        "## 当前物理量与消融",
        "",
        f"PHY_NOW Spearman {_fmt(phy_spearman['mean'])}，value Spearman {_fmt(value_spearman['mean'])}。",
        f"单 edge rate Spearman {_fmt(edge_spearman['mean'])}，pair interference Spearman {_fmt(pair_spearman['mean'])}。",
        f"TEMP 置零后 argmax 改变比例 {_fmt(temp_zero['argmax_disagreement'])}。REP 置零后 {_fmt(rep_zero['argmax_disagreement'])}。",
        f"TEMP assignment 间平均 cosine {_fmt(temp_cosine['mean'])}。",
        "",
        "## 310114 t=15",
        "",
        "| layer | cosine | l2 或差值 | opt rank |",
        "| --- | --- | --- | --- |",
    ]
    for row in case_rows:
        lines.append(f"| {row['layer']} | {_fmt(row['cosine']) if row['cosine'] != '' else ''} | {_fmt(row['l2']) if row['l2'] != '' else ''} | {row['opt_rank']} |")
    lines.extend(
        [
            "",
            f"ΔJ15 ≥ 2/3/4 block 的 post-pool collision：{', '.join(_fmt(row['post_cosine_high']) for row in large_summary)}。",
            "",
            f"Gate 是 **{gate}**。",
            "",
            advice,
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_plain({"gate": gate, "post": post_high["mean"], "pool": pooling["mean"], "phy": phy_spearman["mean"]})), flush=True)
    return {"gate": gate}


if __name__ == "__main__":
    run()
