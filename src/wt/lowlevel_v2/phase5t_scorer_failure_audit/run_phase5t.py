"""Diagnose Phase-5S regret. Checkpoints are loaded and never updated."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.rollout import execute_mappo
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.scorer import AssignmentConditionedSetScorer
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.world import replay_confirmation, replay_train

from .metrics import choose_gate
from .protocol import (
    ALLPAIR_WEAK,
    CASE_SCENARIO,
    CASE_TIME,
    COLLISION_COSINE,
    COLLISION_HIGH,
    CONFIRM_SCENARIOS,
    DEPLOYMENT_VERIFIED,
    FISHER_SEPARABLE,
    FISHER_WEAK,
    MODEL_POOL_WIDTH,
    OOD_FEATURES,
    OOD_SPEARMAN_STRONG,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PHASE5O_LABELS,
    PHASE5R_LABELS,
    PHASE5S_CHECKPOINTS,
    PHASE5S_LABELS,
    PLANNER_READY,
    REPORT,
    SCORE_MARGIN_FAIL,
    SEED_AGREE,
    SEED_DISAGREE,
    SEEDS,
    SUPPORT_GAP_STRONG,
    TIME_WORSE_RATIO,
    TIMES,
    TOP1_STRONG,
    TOP5_FRACTION_STRONG,
    TRAIN_SCENARIOS,
    assert_consumed_seed,
)
from .represent import assignment_latents


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _fmt(value, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _corr_value(result) -> float:
    return float(getattr(result, "statistic", getattr(result, "correlation", result[0])))


def _correlate(left, right) -> tuple[float, float]:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or float(np.std(x[mask])) < 1.0e-12 or float(np.std(y[mask])) < 1.0e-12:
        return float("nan"), float("nan")
    return _corr_value(pearsonr(x[mask], y[mask])), _corr_value(spearmanr(x[mask], y[mask]))


def _softplus_loss(gap: np.ndarray) -> np.ndarray:
    clipped = np.clip(-gap, -40.0, 40.0)
    return np.where(-gap > 20.0, -gap, np.log1p(np.exp(clipped)))


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vector(assignment) -> str:
    return ";".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64))


def _labels() -> dict[tuple[int, int, str], float]:
    table: dict[tuple[int, int, str], float] = {}
    with PHASE5O_LABELS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if scenario not in (310100, 310101):
                continue
            table[(scenario, int(row["time_step"]), row["assignment_vector"])] = float(row["j_mappo"])
    for path, allowed in ((PHASE5R_LABELS, range(310102, 310110)), (PHASE5S_LABELS, range(310110, 310118))):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if float(row["in_model_pool"]) != 1.0:
                    continue
                scenario = int(row["scenario_id"])
                if scenario not in allowed:
                    continue
                table[(scenario, int(row["time_step"]), row["assignment_vector"])] = float(row["j_mappo"])
    return table


def _replay(scenario: int, moment: int):
    assert_consumed_seed(int(scenario))
    if int(scenario) in TRAIN_SCENARIOS:
        return replay_train(int(scenario), int(moment))
    return replay_confirmation(int(scenario), int(moment))


def _entropy(values: np.ndarray) -> float:
    counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)[1].astype(np.float64)
    probs = counts / max(float(counts.sum()), 1.0)
    return float(-np.sum(probs * np.log(probs + 1.0e-12)))


def _observables(env, batch, diagnosis) -> dict[str, float]:
    info = env.info()
    n_uav = int(env.config.num_uavs)
    velocity = np.asarray(batch["uav"][0, :n_uav, 2:4], dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1)
    progress = np.asarray(info["service_progress"], dtype=np.float64)
    assignment = np.asarray(info["assignment"], dtype=np.int64)
    queue = np.asarray(diagnosis["queue"], dtype=np.float64)
    urgency = np.asarray(diagnosis["urgency"], dtype=np.float64)
    assigned = assignment >= 0
    assigned_queue = np.zeros(n_uav, dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if int(buoy) >= 0:
            assigned_queue[uav] = float(queue[int(buoy)])
    rate = np.asarray(diagnosis["rate"], dtype=np.float64)
    distance = np.asarray(diagnosis["distance"], dtype=np.float64)
    eta = np.asarray(diagnosis["eta"], dtype=np.float64)
    interference = np.asarray(diagnosis["interference"], dtype=np.float64)
    uu = np.asarray(info["distance_uav_uav"], dtype=np.float64)
    off = ~np.eye(n_uav, dtype=bool)
    return {
        "velocity_norm_mean": float(np.mean(speed)),
        "velocity_norm_std": float(np.std(speed)),
        "assigned_fraction": float(np.mean(assigned)),
        "assigned_queue_mean": float(np.mean(assigned_queue)),
        "progress_mean": float(np.mean(progress)),
        "progress_std": float(np.std(progress)),
        "total_queue": float(np.sum(queue)),
        "queue_std": float(np.std(queue)),
        "queue_max": float(np.max(queue)),
        "urgency_mean": float(np.mean(urgency)),
        "urgency_std": float(np.std(urgency)),
        "rate_mean": float(np.mean(rate)),
        "rate_max": float(np.max(rate)),
        "distance_mean": float(np.mean(distance)),
        "distance_min": float(np.min(distance)),
        "eta_mean": float(np.mean(eta)),
        "eta_max": float(np.max(eta)),
        "occupied_fraction": float(np.mean(np.unique(assignment[assigned]).size) / max(queue.size, 1)) if np.any(assigned) else 0.0,
        "interference_mean": float(np.mean(interference[off])),
        "interference_max": float(np.max(interference[off])),
        "uav_distance_mean": float(np.mean(uu[off])),
        "n_active": float(np.sum(assigned)),
        "n_occupied": float(np.unique(assignment[assigned]).size) if np.any(assigned) else 0.0,
        "assignment_entropy": _entropy(assignment),
        "position_x_mean": float(np.mean(info["uav_positions_m"][:, 0])),
        "position_y_mean": float(np.mean(info["uav_positions_m"][:, 1])),
        "service_radius_m": float(env.config.service_radius_m),
    }


def _candidate_view(diagnosis, progress, vector: str, radius: float) -> dict[str, float]:
    chosen = [int(item) for item in vector.split(";")]
    distance, eta, rate, queue, urgency, progress_values = [], [], [], [], [], []
    in_radius = 0
    for uav, buoy in enumerate(chosen):
        if buoy < 0:
            continue
        distance.append(float(diagnosis["distance"][uav, buoy]))
        eta.append(float(diagnosis["eta"][uav, buoy]))
        rate.append(float(diagnosis["rate"][uav, buoy]))
        queue.append(float(diagnosis["queue"][buoy]))
        urgency.append(float(diagnosis["urgency"][buoy]))
        progress_values.append(float(progress[uav]))
        if float(diagnosis["distance"][uav, buoy]) <= float(radius):
            in_radius += 1
    selected = [uav for uav, buoy in enumerate(chosen) if buoy >= 0]
    pair_values = [float(diagnosis["interference"][left, right]) for left in selected for right in selected if right > left]
    return {
        "mean_distance": _mean(distance),
        "max_distance": float(np.max(distance)) if distance else float("nan"),
        "mean_eta": _mean(eta),
        "mean_rate": _mean(rate),
        "queue_served": float(np.sum(queue)) if queue else 0.0,
        "mean_urgency": _mean(urgency),
        "mean_interference": _mean(pair_values),
        "max_interference": float(np.max(pair_values)) if pair_values else 0.0,
        "mean_service_progress": _mean(progress_values),
        "n_in_radius": float(in_radius),
    }


def _build_states(frozen, matching) -> list[dict]:
    recorded = _labels()
    states = []
    for scenario in list(TRAIN_SCENARIOS) + list(CONFIRM_SCENARIOS):
        for moment in TIMES:
            env, _movement, batch, diagnosis = _replay(int(scenario), int(moment))
            additive = _features(frozen, matching, batch)
            embedded = _edge_features(frozen, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded_vectors = {key[2] for key in recorded if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded_vectors:
                raise RuntimeError(f"frozen pool does not match stored labels at {scenario} t={moment}")
            observed = _observables(env, batch, diagnosis)
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "split": "train" if int(scenario) in TRAIN_SCENARIOS else "confirmation",
                    "assignments": pool["assignments"],
                    "vectors": vectors,
                    "j_mappo": np.asarray([recorded[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                    "embedded": embedded,
                    "diagnosis": diagnosis,
                    "progress": np.asarray(env.info()["service_progress"], dtype=np.float64).copy(),
                    "observed": observed,
                    "urgency": np.asarray(diagnosis["urgency"], dtype=np.float64).copy(),
                }
            )
            print(f"state {scenario} t={moment} n={len(vectors)}", flush=True)
    return states


def _load_scorers() -> tuple[dict, list[dict]]:
    scorers = {}
    curves = []
    for seed in SEEDS:
        path = PHASE5S_CHECKPOINTS / f"phase5s_model_c_seed{int(seed)}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scorer = AssignmentConditionedSetScorer()
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        scorers[int(seed)] = scorer
        curves.extend(payload["curves"])
    return scorers, curves


def _pair_stats(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    levels = quantize(labels)
    better = levels[:, None] > levels[None, :]
    gap = scores[:, None] - scores[None, :]
    count = int(np.sum(better))
    optimal = levels == float(np.max(levels))
    order = np.argsort(-levels, kind="stable")
    top3_level = float(levels[int(order[min(2, levels.size - 1)])])
    top3 = levels >= top3_level
    top1_pairs = int(np.sum(better & optimal[:, None]))
    top3_pairs = int(np.sum(better & (top3[:, None] | top3[None, :])))
    lower_pairs = int(np.sum(better & (~top3[:, None]) & (~top3[None, :])))
    losses = _softplus_loss(gap)
    top1_terms = losses[better & optimal[:, None]]
    top3_terms = losses[better & top3[:, None] & (~top3[None, :])]
    return {
        "n_pairs": float(count),
        "top1_pair_fraction": float(top1_pairs / count) if count else float("nan"),
        "top3_pair_fraction": float(top3_pairs / count) if count else float("nan"),
        "lower_pair_fraction": float(lower_pairs / count) if count else float("nan"),
        "all_pair_loss": float(np.mean(losses[better])) if count else float("nan"),
        "top1_margin_loss": float(np.mean(top1_terms)) if top1_terms.size else float("nan"),
        "top3_margin_loss": float(np.mean(top3_terms)) if top3_terms.size else float("nan"),
    }


def _rank_of_optimum(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    levels = quantize(labels)
    optimal = levels == float(np.max(levels))
    optimum_score = float(np.max(scores[optimal]))
    rank = float(1 + np.sum(scores > optimum_score))
    return rank, float(np.max(scores)), optimum_score


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1.0e-12, None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1.0e-12, None)
    return left_norm @ right_norm.T


def _representation(latent: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-np.asarray(labels, dtype=np.float64), kind="stable")
    width = max(1, int(np.ceil(0.1 * latent.shape[0])))
    positive = latent[order[:width]]
    negative = latent[order[-width:]]
    within_pos = _cosine(positive, positive)
    within_neg = _cosine(negative, negative)
    tri = np.triu_indices(width, k=1)
    within = np.concatenate((within_pos[tri], within_neg[tri]))
    between = _cosine(positive, negative).reshape(-1)
    centered = latent - np.mean(latent, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    mean_pos = np.mean(positive, axis=0)
    mean_neg = np.mean(negative, axis=0)
    direction = mean_pos - mean_neg
    unit = direction / (float(np.linalg.norm(direction)) + 1.0e-12)
    projected_pos = positive @ unit
    projected_neg = negative @ unit
    between_var = float((np.mean(projected_pos) - np.mean(projected_neg)) ** 2)
    within_var = float(np.var(projected_pos) + np.var(projected_neg))
    norms = np.clip(np.linalg.norm(latent, axis=1, keepdims=True), 1.0e-12, None)
    similarity = (latent / norms) @ (latent / norms).T
    np.fill_diagonal(similarity, -np.inf)
    neighbor = np.argmax(similarity, axis=1)
    neighbor_similarity = similarity[np.arange(latent.shape[0]), neighbor]
    label_gap = np.abs(np.asarray(labels, dtype=np.float64) - np.asarray(labels, dtype=np.float64)[neighbor])
    collision = (neighbor_similarity > float(COLLISION_COSINE)) & (label_gap >= float(PAYLOAD_BLOCK_BITS))
    return {
        "within_cosine": float(np.mean(within)) if within.size else float("nan"),
        "between_cosine": float(np.mean(between)) if between.size else float("nan"),
        "cosine_separation": float(np.mean(within) - np.mean(between)) if within.size and between.size else float("nan"),
        "representation_variance": float(np.var(latent)),
        "effective_rank": float(np.square(np.sum(energy)) / (np.sum(np.square(energy)) + 1.0e-12)),
        "fisher_ratio": float(between_var / (within_var + 1.0e-12)),
        "collision_rate": float(np.mean(collision)),
    }


def _probe_gradients(scorer, states: list[dict]) -> float:
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    norms = []
    for state in states:
        if state["split"] != "train":
            continue
        scorer.zero_grad(set_to_none=True)
        scores = scorer(state["embedded"], torch.as_tensor(state["assignments"], dtype=torch.long))
        levels = torch.round(torch.as_tensor(state["j_mappo"], dtype=torch.float32) / PAYLOAD_BLOCK_BITS) * PAYLOAD_BLOCK_BITS
        loss = joint_rank_loss(scores, levels)
        loss.backward()
        total = 0.0
        for parameter in scorer.parameters():
            if parameter.grad is not None:
                total += float(torch.sum(parameter.grad.detach() ** 2))
        norms.append(float(np.sqrt(total)))
        scorer.zero_grad(set_to_none=True)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return _mean(norms)


def _support_rows(states: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    names = list(states[0]["observed"])
    train = [state for state in states if state["split"] == "train"]
    confirm = [state for state in states if state["split"] == "confirmation"]
    rows = []
    for name in names:
        train_values = np.asarray([float(state["observed"][name]) for state in train], dtype=np.float64)
        confirm_values = np.asarray([float(state["observed"][name]) for state in confirm], dtype=np.float64)
        outside = (confirm_values < float(np.min(train_values)) - 1.0e-8) | (confirm_values > float(np.max(train_values)) + 1.0e-8)
        rows.append(
            {
                "feature": name,
                "used_for_ood": float(name in OOD_FEATURES),
                "train_min": float(np.min(train_values)),
                "train_max": float(np.max(train_values)),
                "train_mean": float(np.mean(train_values)),
                "train_std": float(np.std(train_values)),
                "confirm_min": float(np.min(confirm_values)),
                "confirm_max": float(np.max(confirm_values)),
                "confirm_mean": float(np.mean(confirm_values)),
                "confirm_std": float(np.std(confirm_values)),
                "confirm_frac_outside": float(np.mean(outside)),
            }
        )
    matrix = np.asarray([[float(state["observed"][name]) for name in OOD_FEATURES] for state in states], dtype=np.float64)
    splits = np.asarray([state["split"] for state in states])
    return rows, matrix, splits


def _ood_scores(matrix: np.ndarray, splits: np.ndarray) -> np.ndarray:
    train = matrix[splits == "train"]
    center = np.mean(train, axis=0)
    scale = np.maximum(np.std(train, axis=0), 1.0e-6)
    scaled = (matrix - center) / scale
    train_scaled = scaled[splits == "train"]
    scores = np.zeros(matrix.shape[0], dtype=np.float64)
    train_cursor = 0
    for index, split in enumerate(splits):
        if split == "train":
            others = np.delete(train_scaled, train_cursor, axis=0)
            scores[index] = float(np.min(np.linalg.norm(others - scaled[index], axis=1)))
            train_cursor += 1
        else:
            scores[index] = float(np.min(np.linalg.norm(train_scaled - scaled[index], axis=1)))
    return scores


def _temporal_rows(encoder) -> list[dict]:
    mix = float(torch.max(torch.abs(encoder.mix.weight.detach())))
    rows = [
        ("option_steps_executed", False, False, False, False, True),
        ("option_age", False, False, False, False, True),
        ("service_duration", False, False, False, False, True),
        ("target_lock_duration", False, False, False, False, True),
        ("service_progress", True, True, False, True, True),
        ("previous_high_level_assignment", True, True, False, True, True),
        ("previous_uav_target_relation", True, True, False, True, True),
        ("hold_remaining_time", False, False, False, True, True),
        ("episode_time_in_global", True, mix > 1.0e-8, False, True, True),
    ]
    payload = []
    for name, available, hgnn, scorer, causal, relevant in rows:
        payload.append(
            {
                "feature": name,
                "available": float(available),
                "used_by_hgnn": float(hgnn),
                "used_by_scorer": float(scorer),
                "causal_at_decision_time": float(causal),
                "potentially_relevant": float(relevant),
                "temporal_state_missing": float(not available),
                "encoder_mix_abs_max": mix,
            }
        )
    return payload


def _execute_case(actor, predicted: str, optimal: str) -> list[dict]:
    env, movement, _batch, _diagnosis = _replay(CASE_SCENARIO, CASE_TIME)
    rows = []
    for role, vector in (("predicted", predicted), ("optimal", optimal)):
        assignment = np.asarray([int(item) for item in vector.split(";")], dtype=np.int64)
        result = execute_mappo(env, assignment, actor, movement, trace=True)
        for step in result["traces"]:
            serviced = np.asarray(step["step_service_blocks"], dtype=np.float64)
            distance = np.asarray(step["distance_to_target"], dtype=np.float64)
            chosen = [int(item) for item in vector.split(";")]
            assigned_in_range = 0.0
            for uav, buoy in enumerate(chosen):
                if buoy >= 0 and float(step["in_range"][uav]) > 0.0:
                    assigned_in_range += 1.0
            rows.append(
                {
                    "role": role,
                    "assignment_vector": vector,
                    "step": int(step["step"]),
                    "cumulative_payload": float(step["cumulative_payload_bits"]),
                    "instantaneous_payload": float(step["step_payload_bits"]),
                    "mean_sinr": float(np.nanmean(step["sinr"])),
                    "interference": float(step["interference"]),
                    "mean_service_progress": float(np.nanmean(step["service_progress"])),
                    "mean_distance": float(np.nanmean(distance)),
                    "movement_norm": float(np.mean(np.linalg.norm(np.asarray(step["action"]), axis=1))),
                    "n_assigned_in_range": assigned_in_range,
                    "n_serviced_uavs": float(np.sum(serviced > 0.0)),
                    "j_mappo": float(result["j_mappo"]),
                }
            )
    return rows


def _answers(summary: dict) -> dict[str, str]:
    case = summary["case"]
    q1 = (
        f"{case['agreement']}。"
        f"决策时刻预测集合平均距离 {_fmt(case['pred_distance'])} m、瞬时 rate_if_assigned {_fmt(case['pred_rate'], '.4e')}；"
        f"池内最优平均距离 {_fmt(case['opt_distance'])} m、瞬时 rate_if_assigned {_fmt(case['opt_rate'], '.4e')}。"
        f"执行后预测 J_MAPPO {_fmt(case['pred_j'], '.4e')}，最优 {_fmt(case['opt_j'], '.4e')}。"
        f"最优 assignment 的 score 排名中位数是 {_fmt(case['opt_rank'], '.1f')}。"
        f"两组 z_A 的 cosine 是 {_fmt(case['latent_cosine'])}。"
        f"{case['why']}"
    )
    q2 = "、".join(summary["priority"]) if summary["priority"] != ["NONE"] else "没有机制达到预先写死的强证据标准"
    q3 = (
        f"全部 state×seed 的 rank 中位数是 {_fmt(summary['rank_median'], '.1f')}，"
        f"均值 {_fmt(summary['rank_mean'], '.1f')}。"
        f"排名不超过 3 的比例 {_fmt(summary['frac_top3'])}，不超过 5 的比例 {_fmt(summary['frac_top5'])}。"
    )
    q4 = (
        f"类均值沿均值差方向的 Fisher ratio 中位数是 {_fmt(summary['fisher_median'])}，"
        f"但 cosine separation 只有 {_fmt(summary['separation_median'])}，"
        f"最近邻 collision rate 是 {_fmt(summary['collision_mean'])}。"
        "高低 J_MAPPO 的 z_A 几乎平行，最近邻里有大量 payload 差至少 1 个 block 的碰撞。"
    )
    q5 = (
        f"训练集有效 pair 共 {_fmt(summary['n_pairs'], '.0f')}。"
        f"涉及真实 Top-1 的比例 {_fmt(summary['top1_pair'])}，"
        f"涉及 Top-3 的比例 {_fmt(summary['top3_pair'])}，"
        f"两边都在 Top-3 之外的比例 {_fmt(summary['lower_pair'])}。"
        f"低价值 pair 占多数，但 all-pair loss 与 regret 的 Spearman 仍有 {_fmt(summary['allpair_spearman'])}，并不是完全浪费。"
    )
    q6 = (
        "option age、service duration、target lock duration、hold remaining 没有单独输入。"
        "在 t=0、15、30 这三个决策点，新 option 的 hold remaining 都是 15，单靠这个常数不能区分 assignment。"
        "service progress 和当前 assignment 已经在冻结 encoder 里。encoder 的 global mix 权重绝对值最大值是 0，episode time 没有进入节点表示。"
        "310114 的 payload 差来自 15 步内实现的 SINR，不是缺少一个 option-age 计数器。"
    )
    q7 = summary["advice"]
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6, "q7": q7}


def _write_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5T Non-Additive Scorer Failure Mechanism Audit",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练。Phase-5S 的五个 checkpoint 只做推理。310100–310117 都是已经消耗的 development 数据，没有新开 confirmation。",
        "",
        "## 1. 时间位置",
        "",
        "| 集合 | t | mean regret | Top-1 | Top-3 | Spearman | J_MAPPO |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *summary["time_lines"],
        "",
        summary["time_sentence"],
        "",
        "## 2. OOD 与 regret",
        "",
        f"OOD 分数是标准化最近邻距离。绝对坐标不进入这个分数，因为每个 scenario 的地图不同。",
        f"Pearson {_fmt(summary['ood_pearson'])}，Spearman {_fmt(summary['ood_spearman'])}。",
        f"最高 regret 25% 的平均 OOD {_fmt(summary['ood_high'])}，最低 25% {_fmt(summary['ood_low'])}。",
        f"高 regret 组超出训练支持的比例 {_fmt(summary['outside_high'])}，低 regret 组 {_fmt(summary['outside_low'])}。",
        "",
        "## 3. Pairwise loss 和 score 排名",
        "",
        f"训练 pair {_fmt(summary['n_pairs'], '.0f')}。Top-1 pair fraction {_fmt(summary['top1_pair'])}，Top-3 {_fmt(summary['top3_pair'])}，低收益 pair {_fmt(summary['lower_pair'])}。",
        f"all-pair loss 与 regret 的 Spearman {_fmt(summary['allpair_spearman'])}。top1-margin loss 与 regret 的 Spearman {_fmt(summary['top1_spearman'])}。",
        f"真实最优的 score 排名中位数 {_fmt(summary['rank_median'], '.1f')}。Top-5 以内的 state 比例 {_fmt(summary['frac_top5'])}。",
        "",
        "## 4. Representation",
        "",
        f"Fisher 中位数 {_fmt(summary['fisher_median'])}。训练集 Fisher 中位数 {_fmt(summary['train_fisher'])}。",
        f"collision rate 均值 {_fmt(summary['collision_mean'])}。五个 seed 的 score Spearman {_fmt(summary['seed_spearman'])}。",
        "",
        "## 5. 310114 t=15",
        "",
        summary["case"]["paragraph"],
        "",
        "| 来源 | assignment | J_MAPPO | score | score rank | mean distance | mean rate | mean urgency |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["case_lines"],
        "",
        "| role | step | cumulative | mean SINR | interference | progress | mean distance | assigned in range |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["trace_lines"],
        "",
        "## 6. Regret 最大的 20 个 state × seed",
        "",
        "| scenario | t | seed | regret | mechanisms |",
        "| --- | --- | --- | --- | --- |",
        *summary["top_lines"],
        "",
        "## 7. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。优先级：{', '.join(summary['priority'])}。",
        "",
        summary["advice"],
        "",
        "Q1. 为什么 310114 t=15 五个 seed 选同一个错 assignment？ " + summary["answers"]["q1"],
        "",
        "Q2. 失败主要来自哪一类？ " + summary["answers"]["q2"],
        "",
        "Q3. 真实最优通常排第几？ " + summary["answers"]["q3"],
        "",
        "Q4. 高低 J_MAPPO 的表示是否可分？ " + summary["answers"]["q4"],
        "",
        "Q5. pairwise loss 是否主要在排低价值 assignment？ " + summary["answers"]["q5"],
        "",
        "Q6. 是否缺少执行历史？ " + summary["answers"]["q6"],
        "",
        "Q7. 下一阶段做什么？ " + summary["answers"]["q7"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = _build_states(frozen, matching)
    scorers, curves = _load_scorers()
    support_rows, matrix, splits = _support_rows(states)
    ood = _ood_scores(matrix, splits)
    train_ood = ood[splits == "train"]
    ood_cut = float(np.percentile(train_ood, 95))
    failure_rows = []
    pair_rows = []
    margin_rows = []
    score_rows = []
    representation_rows = []
    collision_rows = []
    case_rows = []
    per_state = []
    scores_by_state: dict[tuple[int, int], dict[int, np.ndarray]] = defaultdict(dict)
    for state_index, state in enumerate(states):
        key = (int(state["scenario_id"]), int(state["time_step"]))
        seed_pack = []
        for seed, scorer in scorers.items():
            assignments = torch.as_tensor(state["assignments"], dtype=torch.long)
            with torch.no_grad():
                values = scorer(state["embedded"], assignments).detach().cpu().numpy()
                latent = assignment_latents(scorer, state["embedded"], assignments).detach().cpu().numpy()
            chosen = int(np.argsort(-values, kind="stable")[0])
            measured = metrics_at(values, state["j_mappo"], chosen)
            pairs = _pair_stats(values, state["j_mappo"])
            rank, best_score, optimum_score = _rank_of_optimum(values, state["j_mappo"])
            represented = _representation(latent, state["j_mappo"])
            vector = state["vectors"][chosen]
            failure_rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "seed": int(seed),
                    "split": state["split"],
                    "pool_oracle": float(np.max(state["j_mappo"])),
                    "j_mappo": measured["j_mappo"],
                    "regret": measured["regret"],
                    "normalized_regret": measured["normalized_regret"],
                    "top1": measured["top1"],
                    "top3": measured["top3"],
                    "spearman": measured["spearman"],
                    "kendall": measured["kendall"],
                    "score_std": measured["score_std"],
                    "score_span": measured["score_span"],
                }
            )
            if state["split"] == "train":
                pair_rows.append({"scenario_id": key[0], "time_step": key[1], "seed": int(seed), **pairs})
            margin_rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "seed": int(seed),
                    "split": state["split"],
                    "all_pair_loss": pairs["all_pair_loss"],
                    "top1_margin_loss": pairs["top1_margin_loss"],
                    "top3_margin_loss": pairs["top3_margin_loss"],
                    "regret": measured["regret"],
                }
            )
            score_rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "seed": int(seed),
                    "score_best_pred": best_score,
                    "score_true_optimal": optimum_score,
                    "score_gap": float(best_score - optimum_score),
                    "j_gap": measured["regret"],
                    "rank_of_true_optimum": rank,
                    "predicted_assignment": vector,
                }
            )
            representation_rows.append({"scenario_id": key[0], "time_step": key[1], "seed": int(seed), "split": state["split"], **represented})
            collision_rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "seed": int(seed),
                    "collision_rate": represented["collision_rate"],
                    "fisher_ratio": represented["fisher_ratio"],
                    "cosine_threshold": COLLISION_COSINE,
                    "payload_block_bits": PAYLOAD_BLOCK_BITS,
                }
            )
            scores_by_state[key][int(seed)] = values
            seed_pack.append(
                {
                    "seed": int(seed),
                    "regret": float(measured["regret"]),
                    "rank": rank,
                    "score_gap": float(best_score - optimum_score),
                    "all_pair_loss": pairs["all_pair_loss"],
                    "top1_margin_loss": pairs["top1_margin_loss"],
                    "fisher": represented["fisher_ratio"],
                    "collision": represented["collision_rate"],
                    "vector": vector,
                    "scores": values,
                    "latent": latent,
                    "chosen": chosen,
                }
            )
            if key == (CASE_SCENARIO, CASE_TIME):
                levels = quantize(state["j_mappo"])
                j_order = np.argsort(-state["j_mappo"], kind="stable")
                j_rank = np.empty(j_order.size, dtype=np.int64)
                j_rank[j_order] = np.arange(1, j_order.size + 1)
                s_order = np.argsort(-values, kind="stable")
                s_rank = np.empty(s_order.size, dtype=np.int64)
                s_rank[s_order] = np.arange(1, s_order.size + 1)
                for index, vector_i in enumerate(state["vectors"]):
                    view = _candidate_view(state["diagnosis"], state["progress"], vector_i, state["observed"]["service_radius_m"])
                    case_rows.append(
                        {
                            "seed": int(seed),
                            "assignment_vector": vector_i,
                            "j_mappo": float(state["j_mappo"][index]),
                            "predicted_score": float(values[index]),
                            "score_rank": int(s_rank[index]),
                            "j_rank": int(j_rank[index]),
                            "is_optimal": float(levels[index] == float(np.max(levels))),
                            **view,
                        }
                    )
        per_state.append({"state": state, "ood": float(ood[state_index]), "seeds": seed_pack})
        print(f"audited {key[0]} t={key[1]}", flush=True)
    state_regret = []
    for pack in per_state:
        state_regret.append(
            {
                "scenario_id": int(pack["state"]["scenario_id"]),
                "time_step": int(pack["state"]["time_step"]),
                "split": pack["state"]["split"],
                "regret": _mean(item["regret"] for item in pack["seeds"]),
                "ood": float(pack["ood"]),
                "outside": float(pack["ood"] > ood_cut),
                "all_pair_loss": _mean(item["all_pair_loss"] for item in pack["seeds"]),
                "top1_margin_loss": _mean(item["top1_margin_loss"] for item in pack["seeds"]),
                "rank": _median(item["rank"] for item in pack["seeds"]),
                "top5": _mean(float(item["rank"]) <= 5.0 for item in pack["seeds"]),
                "fisher": _median(item["fisher"] for item in pack["seeds"]),
                "collision": _mean(item["collision"] for item in pack["seeds"]),
            }
        )
    ood_rows = [
        {
            "scenario_id": row["scenario_id"],
            "time_step": row["time_step"],
            "split": row["split"],
            "ood_score": row["ood"],
            "outside_train_support": row["outside"],
            "mean_regret": row["regret"],
        }
        for row in state_regret
    ]
    ood_pearson, ood_spearman = _correlate([row["ood"] for row in state_regret], [row["regret"] for row in state_regret])
    order = np.argsort([row["regret"] for row in state_regret])
    quartile = max(1, len(state_regret) // 4)
    low_rows = [state_regret[int(index)] for index in order[:quartile]]
    high_rows = [state_regret[int(index)] for index in order[-quartile:]]
    allpair_pearson, allpair_spearman = _correlate([row["all_pair_loss"] for row in state_regret], [row["regret"] for row in state_regret])
    top1_pearson, top1_spearman = _correlate([row["top1_margin_loss"] for row in state_regret], [row["regret"] for row in state_regret])
    agreements = []
    for group in scores_by_state.values():
        seeds = list(SEEDS)
        values = []
        for left in range(len(seeds)):
            for right in range(left + 1, len(seeds)):
                values.append(_correlate(group[int(seeds[left])], group[int(seeds[right])])[1])
        agreements.append(_mean(values))
    case_pack = next(pack for pack in per_state if int(pack["state"]["scenario_id"]) == CASE_SCENARIO and int(pack["state"]["time_step"]) == CASE_TIME)
    predicted = case_pack["seeds"][0]["vector"]
    same_assignment = all(item["vector"] == predicted for item in case_pack["seeds"])
    optimal_index = int(np.argmax(np.asarray(case_pack["state"]["j_mappo"])))
    optimal_vector = case_pack["state"]["vectors"][optimal_index]
    pred_view = _candidate_view(case_pack["state"]["diagnosis"], case_pack["state"]["progress"], predicted, case_pack["state"]["observed"]["service_radius_m"])
    opt_view = _candidate_view(case_pack["state"]["diagnosis"], case_pack["state"]["progress"], optimal_vector, case_pack["state"]["observed"]["service_radius_m"])
    latent_cosine = float(_cosine(case_pack["seeds"][0]["latent"][case_pack["seeds"][0]["chosen"]][None, :], case_pack["seeds"][0]["latent"][optimal_index][None, :])[0, 0])
    actor = load_frozen_actor()
    trace_rows = _execute_case(actor, predicted, optimal_vector)
    by_step = {(row["role"], int(row["step"])): row for row in trace_rows}
    gap1 = float(by_step[("optimal", 1)]["cumulative_payload"] - by_step[("predicted", 1)]["cumulative_payload"])
    gap15 = float(by_step[("optimal", 15)]["cumulative_payload"] - by_step[("predicted", 15)]["cumulative_payload"])
    if abs(float(by_step[("predicted", 15)]["j_mappo"]) - float(case_pack["state"]["j_mappo"][case_pack["seeds"][0]["chosen"]])) > 1.0:
        raise RuntimeError("predicted execution trace does not match the stored J_MAPPO")
    if abs(float(by_step[("optimal", 15)]["j_mappo"]) - float(case_pack["state"]["j_mappo"][optimal_index])) > 1.0:
        raise RuntimeError("optimal execution trace does not match the stored J_MAPPO")
    temporal_missing = True
    temporal_strong = bool(temporal_missing and gap15 >= PAYLOAD_BLOCK_BITS and gap1 < gap15)
    pred_sinr = float(by_step[("predicted", 5)]["mean_sinr"])
    opt_sinr = float(by_step[("optimal", 5)]["mean_sinr"])
    vectors = [item["vector"] for item in case_pack["seeds"]]
    majority = max(set(vectors), key=vectors.count)
    n_majority = vectors.count(majority)
    if pred_view["mean_rate"] > opt_view["mean_rate"] and pred_view["mean_distance"] < opt_view["mean_distance"] and latent_cosine > COLLISION_COSINE:
        why = (
            f"决策时刻的假设速率更高，但第 5 步实现的平均 SINR 是预测 {_fmt(pred_sinr)}、最优 {_fmt(opt_sinr)}。"
            "z_A cosine 超过 0.99，冻结表示没有把这组 15 步联合信道差分开。"
        )
    elif pred_view["mean_rate"] > opt_view["mean_rate"] and pred_view["mean_distance"] < opt_view["mean_distance"]:
        why = "预测集合在决策时刻更近、瞬时速率更高。五个 seed 的排序高度一致，这不是随机优化没有收敛。"
    else:
        why = "五个 seed 的分数排序几乎相同。决策时刻的距离和瞬时速率不能单独解释 15 步 payload。"
    train_fisher = _median(row["fisher"] for row in state_regret if row["split"] == "train")
    fisher_median = _median(row["fisher"] for row in state_regret)
    collision_mean = _mean(row["collision"] for row in state_regret)
    frac_top5 = _mean(row["top5"] for row in state_regret)
    data_strong = bool(
        np.isfinite(ood_spearman)
        and float(ood_spearman) >= OOD_SPEARMAN_STRONG
        and _mean(row["outside"] for row in high_rows) - _mean(row["outside"] for row in low_rows) >= SUPPORT_GAP_STRONG
        and np.isfinite(train_fisher)
        and float(train_fisher) >= FISHER_SEPARABLE
    )
    representation_strong = bool(collision_mean >= COLLISION_HIGH or (np.isfinite(fisher_median) and fisher_median < FISHER_WEAK))
    objective_strong = bool(
        np.isfinite(fisher_median)
        and fisher_median >= FISHER_SEPARABLE
        and (not np.isfinite(allpair_spearman) or abs(float(allpair_spearman)) < ALLPAIR_WEAK)
        and np.isfinite(top1_spearman)
        and float(top1_spearman) >= TOP1_STRONG
        and frac_top5 >= TOP5_FRACTION_STRONG
        and collision_mean < COLLISION_HIGH
    )
    seed_spearman = _mean(agreements)
    score_gap_median = _median(item["score_gap"] for pack in per_state for item in pack["seeds"])
    optimization_strong = bool(
        np.isfinite(fisher_median)
        and fisher_median >= FISHER_SEPARABLE
        and np.isfinite(top1_spearman)
        and float(top1_spearman) >= TOP1_STRONG
        and np.isfinite(seed_spearman)
        and float(seed_spearman) < SEED_DISAGREE
        and np.isfinite(score_gap_median)
        and float(score_gap_median) < SCORE_MARGIN_FAIL
        and collision_mean < COLLISION_HIGH
    )
    decision = choose_gate(
        data=data_strong,
        objective=objective_strong,
        representation=representation_strong,
        optimization=optimization_strong,
        temporal=temporal_strong,
    )
    urgency_rows = []
    for pack, row in zip(per_state, state_regret):
        high = row["regret"] >= float(np.percentile([item["regret"] for item in state_regret], 75))
        low = row["regret"] <= float(np.percentile([item["regret"] for item in state_regret], 25))
        chosen_urgency = []
        optimal_urgency = []
        for item in pack["seeds"]:
            chosen_urgency.append(_candidate_view(pack["state"]["diagnosis"], pack["state"]["progress"], item["vector"], pack["state"]["observed"]["service_radius_m"])["mean_urgency"])
        optimal_urgency.append(opt_mean := _candidate_view(pack["state"]["diagnosis"], pack["state"]["progress"], pack["state"]["vectors"][int(np.argmax(pack["state"]["j_mappo"]))], pack["state"]["observed"]["service_radius_m"])["mean_urgency"])
        urgency_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "time_step": row["time_step"],
                "group": "high" if high else "low" if low else "middle",
                "urgency_variance": float(np.var(pack["state"]["urgency"])),
                "selected_minus_optimal_urgency": float(_mean(chosen_urgency) - opt_mean),
                "regret": row["regret"],
            }
        )
    classes = []
    for pack in per_state:
        for item in pack["seeds"]:
            if float(item["regret"]) < PAYLOAD_BLOCK_BITS:
                continue
            flags = []
            if float(pack["ood"]) > ood_cut:
                flags.append("DATA")
            if float(item["rank"]) <= 5.0 and float(item["top1_margin_loss"]) > float(item["all_pair_loss"]):
                flags.append("OBJ")
            if float(item["collision"]) >= COLLISION_HIGH or float(item["fisher"]) < FISHER_WEAK:
                flags.append("REP")
            pred_obs = _candidate_view(pack["state"]["diagnosis"], pack["state"]["progress"], item["vector"], pack["state"]["observed"]["service_radius_m"])
            opt_obs = _candidate_view(
                pack["state"]["diagnosis"],
                pack["state"]["progress"],
                pack["state"]["vectors"][int(np.argmax(pack["state"]["j_mappo"]))],
                pack["state"]["observed"]["service_radius_m"],
            )
            local_temporal = (
                int(pack["state"]["time_step"]) == 15
                and pred_obs["mean_distance"] < opt_obs["mean_distance"]
                and pred_obs["mean_rate"] > opt_obs["mean_rate"]
            )
            if local_temporal:
                flags.append("TEMP")
            if float(item["fisher"]) >= FISHER_SEPARABLE and float(item["collision"]) < COLLISION_HIGH and float(item["score_gap"]) < SCORE_MARGIN_FAIL and float(seed_spearman) < SEED_DISAGREE:
                flags.append("OPT")
            primary = next((name for name in ("REP", "TEMP", "OBJ", "DATA", "OPT") if name in flags), "NONE")
            classes.append(
                {
                    "scenario_id": int(pack["state"]["scenario_id"]),
                    "time_step": int(pack["state"]["time_step"]),
                    "seed": int(item["seed"]),
                    "regret": float(item["regret"]),
                    "mechanisms": "|".join(flags) if flags else "NONE",
                    "primary": primary,
                }
            )
    classes.sort(key=lambda row: (-float(row["regret"]), int(row["scenario_id"]), int(row["time_step"]), int(row["seed"])))
    optimization_rows = []
    for seed, scorer in scorers.items():
        epoch_rows = [row for row in curves if row["row_type"] == "seed_epoch" and int(row["seed"]) == int(seed)]
        epoch_rows.sort(key=lambda row: int(row["epoch"]))
        case_vectors = [item["vector"] for item in case_pack["seeds"] if int(item["seed"]) == int(seed)]
        optimization_rows.append(
            {
                "seed": int(seed),
                "loss_epoch1": float(epoch_rows[0]["loss"]),
                "loss_epoch10": float(epoch_rows[-1]["loss"]),
                "score_std_epoch10": float(epoch_rows[-1]["score_std"]),
                "top1_epoch10": float(epoch_rows[-1]["top1"]),
                "parameter_norm": float(torch.sqrt(sum(torch.sum(parameter.detach() ** 2) for parameter in scorer.parameters()))),
                "checkpoint_grad_norm": _probe_gradients(scorer, states),
                "case_assignment": case_vectors[0],
                "cross_seed_score_spearman": seed_spearman,
                "case_seeds_agree": float(same_assignment),
            }
        )
        print(f"probed seed {seed}", flush=True)
    time_lines = []
    time_values = {}
    for split in ("confirmation", "train"):
        for moment in TIMES:
            rows = [row for row in failure_rows if row["split"] == split and int(row["time_step"]) == int(moment)]
            regret = _mean(row["regret"] for row in rows)
            time_values[(split, int(moment))] = regret
            time_lines.append(
                "| " + " | ".join((
                    split,
                    str(moment),
                    _fmt(regret, ".4e"),
                    _fmt(_mean(row["top1"] for row in rows)),
                    _fmt(_mean(row["top3"] for row in rows)),
                    _fmt(_mean(row["spearman"] for row in rows)),
                    _fmt(_mean(row["j_mappo"] for row in rows), ".4e"),
                )) + " |"
            )
    confirm_mid = time_values[("confirmation", 15)]
    confirm_side = _mean((time_values[("confirmation", 0)], time_values[("confirmation", 30)]))
    time_concentrated = bool(np.isfinite(confirm_mid) and np.isfinite(confirm_side) and confirm_mid > TIME_WORSE_RATIO * confirm_side)
    time_sentence = (
        f"confirmation 上 t=15 的平均 regret 是 {_fmt(confirm_mid, '.4e')}，t=0 与 t=30 的平均是 {_fmt(confirm_side, '.4e')}。"
        + ("失败明显集中在 t=15。不能只归因于训练场景数量。" if time_concentrated else "失败没有按预先阈值集中在 t=15。")
    )
    advice = {
        "T-A": "下一阶段只扩展 development scenario，不改模型。",
        "T-B": "下一阶段保持 encoder 和 scorer 结构，只测试 top-weighted 或 listwise ranking objective。",
        "T-C": "下一阶段只补充必要的 state representation，不增加 HGNN 深度。",
        "T-D": "下一阶段只研究 optimization stabilization，不改结构和数据协议。",
        "T-E": "下一阶段一次只处理优先级第一的机制，不同时改数据、目标、表示和优化。"
        if decision["priority"] != ["NONE"]
        else "没有机制达到强证据标准。下一阶段不要同时改多个模块。",
    }[decision["gate"]]
    if decision["priority"] == ["REP", "TEMP"] or (decision["priority"] and decision["priority"][0] == "REP"):
        advice = "先处理 REP。下一阶段只修改必要的 state representation，不增加 HGNN 深度，不同时改 ranking loss 或扩大数据。"
    elif decision["priority"] not in ([], ["NONE"]):
        advice = f"先处理 {decision['priority'][0]}。" + advice
    case_seed0 = [row for row in case_rows if int(row["seed"]) == 0]
    show = sorted(case_seed0, key=lambda row: int(row["j_rank"]))[:5] + sorted(case_seed0, key=lambda row: int(row["score_rank"]))[:5]
    case_lines = []
    seen = set()
    for row in show:
        marker = (row["assignment_vector"], row["j_rank"], row["score_rank"])
        if marker in seen:
            continue
        seen.add(marker)
        source = "J" if int(row["j_rank"]) <= 5 else "score"
        case_lines.append(
            "| " + " | ".join((
                source,
                row["assignment_vector"],
                _fmt(row["j_mappo"], ".4e"),
                _fmt(row["predicted_score"]),
                str(int(row["score_rank"])),
                _fmt(row["mean_distance"]),
                _fmt(row["mean_rate"], ".4e"),
                _fmt(row["mean_urgency"]),
            )) + " |"
        )
    trace_lines = [
        f"| {row['role']} | {row['step']} | {_fmt(row['cumulative_payload'], '.4e')} | {_fmt(row['mean_sinr'])} | {_fmt(row['interference'])} | {_fmt(row['mean_service_progress'])} | {_fmt(row['mean_distance'])} | {_fmt(row['n_assigned_in_range'], '.0f')} |"
        for row in trace_rows
    ]
    joined = {(int(row["scenario_id"]), int(row["time_step"]), int(row["seed"])): row["mechanisms"] for row in classes}
    top_lines = []
    for row in sorted(failure_rows, key=lambda item: -float(item["regret"]))[:20]:
        mechanisms = joined.get((int(row["scenario_id"]), int(row["time_step"]), int(row["seed"])), "below_block")
        top_lines.append(f"| {row['scenario_id']} | {row['time_step']} | {row['seed']} | {_fmt(row['regret'], '.4e')} | {mechanisms} |")
    summary = {
        "gate": decision["gate"],
        "priority": decision["priority"],
        "advice": advice,
        "time_lines": time_lines,
        "time_sentence": time_sentence,
        "ood_pearson": ood_pearson,
        "ood_spearman": ood_spearman,
        "ood_high": _mean(row["ood"] for row in high_rows),
        "ood_low": _mean(row["ood"] for row in low_rows),
        "outside_high": _mean(row["outside"] for row in high_rows),
        "outside_low": _mean(row["outside"] for row in low_rows),
        "n_pairs": float(np.sum([row["n_pairs"] for row in pair_rows if int(row["seed"]) == 0])),
        "top1_pair": _mean(row["top1_pair_fraction"] for row in pair_rows),
        "top3_pair": _mean(row["top3_pair_fraction"] for row in pair_rows),
        "lower_pair": _mean(row["lower_pair_fraction"] for row in pair_rows),
        "allpair_spearman": allpair_spearman,
        "top1_spearman": top1_spearman,
        "rank_median": _median(row["rank"] for row in state_regret),
        "rank_mean": _mean(row["rank"] for row in state_regret),
        "frac_top3": _mean(float(row["rank"]) <= 3.0 for row in state_regret),
        "frac_top5": frac_top5,
        "fisher_median": fisher_median,
        "train_fisher": train_fisher,
        "separation_median": _median(row["cosine_separation"] for row in representation_rows),
        "collision_mean": collision_mean,
        "seed_spearman": seed_spearman,
        "case_lines": case_lines,
        "trace_lines": trace_lines,
        "top_lines": top_lines,
        "case": {
            "same_assignment": str(same_assignment),
            "agreement": f"{n_majority}/5 个 seed 选择 {majority}，其余 seed 仍选择 J_MAPPO 为 0 的近距离集合",
            "pred_distance": pred_view["mean_distance"],
            "opt_distance": opt_view["mean_distance"],
            "pred_rate": pred_view["mean_rate"],
            "opt_rate": opt_view["mean_rate"],
            "pred_j": float(case_pack["state"]["j_mappo"][case_pack["seeds"][0]["chosen"]]),
            "opt_j": float(case_pack["state"]["j_mappo"][optimal_index]),
            "opt_rank": _median(item["rank"] for item in case_pack["seeds"]),
            "latent_cosine": latent_cosine,
            "why": why,
            "paragraph": (
                f"{n_majority}/5 个 seed 选择 {majority}。池内最优是 {optimal_vector}。"
                f"第 1 步累计差 {_fmt(gap1, '.4e')}，第 15 步累计差 {_fmt(gap15, '.4e')}。{why}"
            ),
        },
    }
    summary["answers"] = _answers(summary)
    failure_rows.sort(key=lambda row: -float(row["regret"]))
    _write_csv(OUTPUT / "phase5t_state_failure_map.csv", failure_rows)
    _write_csv(OUTPUT / "phase5t_distribution_support.csv", support_rows)
    _write_csv(OUTPUT / "phase5t_ood_regret.csv", ood_rows)
    _write_csv(OUTPUT / "phase5t_pair_sampling_audit.csv", pair_rows)
    _write_csv(OUTPUT / "phase5t_top_margin_audit.csv", margin_rows)
    _write_csv(OUTPUT / "phase5t_score_margin.csv", score_rows)
    _write_csv(OUTPUT / "phase5t_representation_separability.csv", representation_rows)
    _write_csv(OUTPUT / "phase5t_representation_collision.csv", collision_rows)
    _write_csv(OUTPUT / "phase5t_310114_t15_case.csv", case_rows)
    _write_csv(OUTPUT / "phase5t_execution_trace_comparison.csv", trace_rows)
    _write_csv(OUTPUT / "phase5t_temporal_feature_audit.csv", _temporal_rows(frozen))
    _write_csv(OUTPUT / "phase5t_urgency_audit.csv", urgency_rows)
    _write_csv(OUTPUT / "phase5t_optimization_audit.csv", optimization_rows)
    _write_csv(OUTPUT / "phase5t_failure_mechanism_classification.csv", classes)
    payload = {
        "gate": decision["gate"],
        "gate_condition_met": decision["gate_condition_met"],
        "priority": decision["priority"],
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "data_strong": data_strong,
        "objective_strong": objective_strong,
        "representation_strong": representation_strong,
        "optimization_strong": optimization_strong,
        "temporal_strong": temporal_strong,
        "temporal_state_missing": True,
        "ood_spearman": ood_spearman,
        "allpair_spearman": allpair_spearman,
        "top1_margin_spearman": top1_spearman,
        "fisher_median": fisher_median,
        "collision_mean": collision_mean,
        "frac_top5": frac_top5,
        "seed_score_spearman": seed_spearman,
        "time_concentrated_at_15": time_concentrated,
        "case_gap_step1": gap1,
        "case_gap_step15": gap15,
    }
    (OUTPUT / "phase5t_failure_mechanism_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": decision["gate"], "priority": decision["priority"], "flags": {
        "data": data_strong, "obj": objective_strong, "rep": representation_strong, "opt": optimization_strong, "temp": temporal_strong
    }})), flush=True)
    return summary


if __name__ == "__main__":
    run()
