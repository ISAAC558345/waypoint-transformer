"""Inference-only causal audit. No optimizer step and no new checkpoint."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import MODEL_POOL_WIDTH
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _replay, _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _combine, _history_snapshot, _scale, _vector
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import CHECKPOINTS as PHASE5ZD_CHECKPOINTS
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import OUTPUT as PHASE5ZD_OUTPUT
from wt.lowlevel_v2.phase5zd_executor_response_value.scorer import ExecutorConditionedJointScorer
from wt.lowlevel_v2.phase5ze_response_distillation.protocol import OUTPUT as PHASE5ZE_OUTPUT

from .aggregate import (
    bootstrap_table,
    calibration_rows,
    damage_rows,
    direction_counts,
    gate_flags,
    recovery_rows,
    stability_rows,
)
from .common import group_indexes, load_tables, mean, normalize, plain, scalars, write_csv
from .protocol import (
    ALIGN_ABS,
    CASE_OPTIMAL,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    COMBINATIONS,
    CONFIRM_SCENARIOS,
    DEPLOYMENT_VERIFIED,
    FROZEN_FOR_FRESH_CONFIRMATION,
    GROUPS,
    NEAR_PROGRESS,
    OUTPUT,
    PLANNER_READY,
    PRIMARY_SCALAR,
    RESPONSE_FEATURES,
    SEEDS,
    TIMES,
    assert_consumed,
)
from .ranking import correlations, rank_block

CONTINUOUS = tuple(index for index in range(len(RESPONSE_FEATURES)) if index not in GROUPS["binary"])
CASE_FEATURES = (
    "sinr_1",
    "interference_1",
    "service_progress_1",
    "rate_1",
    "distance_to_target_1",
    "radial_motion_1",
    "service_active_1",
)


def _load_norm() -> dict[str, np.ndarray]:
    payload = json.loads((PHASE5ZD_OUTPUT / "phase5zd_response_normalization.json").read_text(encoding="utf-8"))
    return {
        "mean": np.asarray(payload["mean"], dtype=np.float64),
        "std": np.asarray(payload["std"], dtype=np.float64),
    }


def _load_scorer(seed: int) -> ExecutorConditionedJointScorer:
    module = ExecutorConditionedJointScorer()
    payload = torch.load(
        PHASE5ZD_CHECKPOINTS / f"phase5zd_response_seed{int(seed)}.pt",
        map_location="cpu",
        weights_only=False,
    )
    module.load_state_dict(payload["state_dict"])
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _score(model, embedded: dict, assignments: np.ndarray, relational: np.ndarray, response: np.ndarray) -> np.ndarray:
    features = {}
    for key, value in embedded.items():
        features[key] = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    features["z_rel"] = torch.as_tensor(relational, dtype=torch.float32)
    features["response"] = torch.as_tensor(response, dtype=torch.float32)
    with torch.no_grad():
        return model(features, torch.as_tensor(assignments, dtype=torch.long)).detach().cpu().numpy()


def _variants(true: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    packed = {"P1": pred, "R1": true}
    for name, indexes in GROUPS.items():
        oracle = pred.copy()
        injected = true.copy()
        oracle[..., list(indexes)] = true[..., list(indexes)]
        injected[..., list(indexes)] = pred[..., list(indexes)]
        packed[f"P1+TRUE_{name.upper()}"] = oracle
        packed[f"R1-PRED_{name.upper()}"] = injected
    for name, groups in COMBINATIONS.items():
        mixed = pred.copy()
        mixed[..., list(group_indexes(groups))] = true[..., list(group_indexes(groups))]
        packed[name] = mixed
    return packed


def _top5(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:5]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _named_rank(scores: np.ndarray, index: int) -> int:
    values = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(values > values[int(index)]))


def _metric(model: str, seed: int, scenario: int, moment: int, scores: np.ndarray, labels: np.ndarray, vectors: list[str]) -> dict:
    chosen = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    measured = metrics_at(scores, labels, chosen)
    optimum = int(np.argmax(quantize(labels)))
    return {
        "model": model,
        "seed": int(seed),
        "scenario_id": int(scenario),
        "time_step": int(moment),
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "top5": _top5(scores, labels),
        "normalized_regret": measured["normalized_regret"],
        "selected_j15": measured["j_mappo"],
        "optimum_rank": _named_rank(scores, optimum),
        "assignment_vector": vectors[chosen],
        "agreement_with_r1": 0.0,
        "agreement_with_p1": 0.0,
    }


def _affine_slope(truth: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(truth) & np.isfinite(prediction)
    if int(mask.sum()) < 3 or float(np.std(truth[mask])) <= 1.0e-12:
        return float("nan")
    return float(np.polyfit(truth[mask], prediction[mask], 1)[0])


def _stack(book: dict, saved: dict, scenario: int, moment: int, seed: int, vectors: list[str]):
    true = np.zeros((len(vectors), 6, len(RESPONSE_FEATURES)), dtype=np.float64)
    pred = np.zeros_like(true)
    labels = np.zeros(len(vectors), dtype=np.float64)
    static = np.zeros(len(vectors), dtype=np.float64)
    stored_p1 = np.zeros(len(vectors), dtype=np.float64)
    stored_r1 = np.zeros(len(vectors), dtype=np.float64)
    for index, vector in enumerate(vectors):
        key = (int(scenario), int(moment), int(seed), vector)
        if key not in book or key not in saved:
            raise RuntimeError(f"missing stored response at {scenario} t={moment} seed={seed}")
        true[index], pred[index] = book[key]
        labels[index], static[index], stored_p1[index], stored_r1[index] = saved[key]
    return true, pred, labels, static, stored_p1, stored_r1


def _prepare(scenario: int, moment: int, frozen, matching, config):
    assert_consumed(int(scenario))
    _env, _movement, batch, diagnosis = _replay(int(scenario), int(moment))
    embedded = {key: value.detach() if torch.is_tensor(value) else value for key, value in _edge_features(frozen, batch).items()}
    additive = _features(frozen, matching, batch)
    logits = additive["logits"].detach().cpu().numpy()
    pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
    vectors = [_vector(row) for row in pool["assignments"]]
    if len(vectors) != 120:
        raise RuntimeError(f"candidate pool width changed at {scenario} t={moment}")
    relational = relational_matrix(_snapshot(_env, diagnosis), pool["assignments"], config)
    temporal = temporal_matrix(_history_snapshot(_env, batch), pool["assignments"], config)
    return {
        "assignments": np.asarray(pool["assignments"], dtype=np.int64),
        "vectors": vectors,
        "embedded": embedded,
        "z_rel": relational,
        "z_temp_raw": temporal,
    }


def _write_flips(handle, scenario: int, moment: int, seed: int, vectors, labels, true_s, pred_s, values) -> dict[str, dict[str, int]]:
    count = len(vectors)
    left, right = np.triu_indices(count, k=1)
    value_gap = values["R1"][left] - values["R1"][right]
    strict = np.flatnonzero(np.abs(value_gap) > 1.0e-8)
    left = left[strict]
    right = right[strict]
    value_gap = value_gap[strict]
    swap = value_gap < 0.0
    first = np.where(swap, right, left)
    second = np.where(swap, left, right)
    predicted_gap = values["P1"][first] - values["P1"][second]
    flipped = predicted_gap < 0.0
    stats = {name: {"flip_rev": 0, "flip": 0, "keep_rev": 0, "keep": 0} for name in GROUPS}
    writer = csv.writer(handle)
    for name in GROUPS:
        truth = true_s[PRIMARY_SCALAR[name]]
        prediction = pred_s[PRIMARY_SCALAR[name]]
        delta_true = truth[first] - truth[second]
        delta_pred = prediction[first] - prediction[second]
        reversed_sign = delta_true * delta_pred < 0.0
        stats[name]["flip"] = int(np.sum(flipped))
        stats[name]["keep"] = int(np.sum(~flipped))
        stats[name]["flip_rev"] = int(np.sum(flipped & reversed_sign))
        stats[name]["keep_rev"] = int(np.sum(~flipped & reversed_sign))
    for row_index in np.flatnonzero(flipped):
        first_index = int(first[row_index])
        second_index = int(second[row_index])
        payload = [scenario, moment, seed, vectors[first_index], vectors[second_index]]
        for name in GROUPS:
            payload.append(float(true_s[PRIMARY_SCALAR[name]][first_index] - true_s[PRIMARY_SCALAR[name]][second_index]))
            payload.append(float(pred_s[PRIMARY_SCALAR[name]][first_index] - pred_s[PRIMARY_SCALAR[name]][second_index]))
        payload.extend(
            [
                float(labels[first_index] - labels[second_index]),
                float(values["P1"][first_index] - values["P1"][second_index]),
                float(values["R1"][first_index] - values["R1"][second_index]),
            ]
        )
        writer.writerow(payload)
    return stats


def _accumulate(total: dict, part: dict) -> None:
    for name, counts in part.items():
        for key, value in counts.items():
            total[name][key] += int(value)


def _progress_bin(response: np.ndarray, assignments: np.ndarray) -> dict:
    held = response[:, :, 4][assignments >= 0]
    if held.size == 0 or float(np.max(held)) <= 0.0:
        label = "ZERO"
    elif float(np.max(held)) >= NEAR_PROGRESS:
        label = "NEAR_COMPLETION"
    else:
        label = "PARTIAL"
    return {
        "progress_bin": label,
        "mean_progress": float(np.mean(held)) if held.size else float("nan"),
        "max_progress": float(np.max(held)) if held.size else float("nan"),
        "fraction_zero": float(np.mean(held <= 0.0)) if held.size else float("nan"),
        "fraction_partial": float(np.mean((held > 0.0) & (held < NEAR_PROGRESS))) if held.size else float("nan"),
        "fraction_near": float(np.mean(held >= NEAR_PROGRESS)) if held.size else float("nan"),
    }


def _interference_rows(metric_rows: list[dict], rank_rows: list[dict], state_rows: list[dict]) -> list[dict]:
    ordered = sorted(state_rows, key=lambda row: (float(row["mean_interference"]), int(row["scenario_id"]), int(row["time_step"])))
    width = max(len(ordered) // 4, 1)
    chunks = [ordered[index * width:(index + 1) * width] for index in range(4)] if len(ordered) >= 4 else [ordered]
    output = []
    for quartile, chunk in enumerate(chunks, start=1):
        keys = {(int(row["scenario_id"]), int(row["time_step"])) for row in chunk}
        regrets = [
            float(row["normalized_regret"])
            for row in metric_rows
            if row["model"] == "P1" and (int(row["scenario_id"]), int(row["time_step"])) in keys
        ]
        disagree = [
            float(row["agreement_with_r1"])
            for row in metric_rows
            if row["model"] == "P1" and (int(row["scenario_id"]), int(row["time_step"])) in keys
        ]
        ranking = [
            float(row["spearman"])
            for row in rank_rows
            if row["group"] == "interference"
            and row["role"] == "primary"
            and (int(row["scenario_id"]), int(row["time_step"])) in keys
        ]
        output.append(
            {
                "quartile": f"Q{quartile}",
                "n_states": len(chunk),
                "mean_true_interference": mean(row["mean_interference"] for row in chunk),
                "response_mae": mean(float(row["interference_mae_sum"]) / float(row["interference_mae_n"]) for row in chunk),
                "ranking_spearman": mean(ranking),
                "ranking_error": 1.0 - mean(ranking),
                "p1_normalized_regret": mean(regrets),
                "p1_disagreement": 1.0 - mean(disagree),
            }
        )
    return output


def _progress_rows(metric_rows: list[dict], state_rows: list[dict]) -> list[dict]:
    lookup = {(int(row["scenario_id"]), int(row["time_step"])): row for row in state_rows}
    output = []
    for label in ("ZERO", "PARTIAL", "NEAR_COMPLETION"):
        chosen = []
        for row in metric_rows:
            if row["model"] != "P1":
                continue
            state = lookup[(int(row["scenario_id"]), int(row["time_step"]))]
            if state["progress_bin"] == label:
                chosen.append(row)
        output.append(
            {
                "progress_bin": label,
                "n_state_seeds": len(chosen),
                "p1_disagreement": 1.0 - mean(row["agreement_with_r1"] for row in chosen),
                "p1_normalized_regret": mean(row["normalized_regret"] for row in chosen),
                "mean_progress": mean(lookup[(int(row["scenario_id"]), int(row["time_step"]))]["mean_progress"] for row in chosen),
            }
        )
    return output


def _protocol() -> dict:
    return {
        "phase": "5ZF",
        "mode": "inference-only diagnostic audit",
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "checkpoint_selection": False,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
        "scenarios": list(CONFIRM_SCENARIOS),
        "times": list(TIMES),
        "seeds": list(SEEDS),
        "groups": {name: list(indexes) for name, indexes in GROUPS.items()},
        "combinations": {name: list(groups) for name, groups in COMBINATIONS.items()},
        "primary_scalar": dict(PRIMARY_SCALAR),
        "align_abs": ALIGN_ABS,
        "recovery_bar": 0.70,
        "combined_recovery_bar": 0.75,
        "near_progress": NEAR_PROGRESS,
        "case_top_rank": 3,
        "affine_diagnostic_used_for_gate": False,
        "opened_310118_310125": False,
    }


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5zf_protocol.json").write_text(json.dumps(plain(_protocol()), indent=2), encoding="utf-8")
    book, saved = load_tables(
        PHASE5ZE_OUTPUT / "phase5ze_response_predictions.csv",
        PHASE5ZE_OUTPUT / "phase5ze_downstream_value_predictions.csv",
        assert_consumed,
    )
    norm = _load_norm()
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    frozen.eval()
    matching.eval()
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    center = np.asarray(temporal["center"], dtype=np.float32)
    scale = np.asarray(temporal["scale"], dtype=np.float32)
    models = {int(seed): _load_scorer(int(seed)) for seed in SEEDS}
    metric_rows: list[dict] = []
    rank_rows: list[dict] = []
    top_rows: list[dict] = []
    sinr_rows: list[dict] = []
    value_rows: list[dict] = []
    regret_rows: list[dict] = []
    case_rows: list[dict] = []
    calib_rows: list[dict] = []
    state_rows: list[dict] = []
    align_p1 = 0.0
    align_r1 = 0.0
    true_seed_gap = 0.0
    argmax_agreement = []
    flip_totals = {name: {"flip_rev": 0, "flip": 0, "keep_rev": 0, "keep": 0} for name in GROUPS}
    flat_true = []
    flat_pred = []
    flip_path = OUTPUT / "phase5zf_pairwise_decision_flips.csv"
    with flip_path.open("w", encoding="utf-8", newline="") as flip_handle:
        csv.writer(flip_handle).writerow(
            ["scenario_id", "time_step", "seed", "assignment_a", "assignment_b"]
            + [item for name in GROUPS for item in (f"delta_true_{name}", f"delta_pred_{name}")]
            + ["true_j15_gap", "p1_value_gap", "r1_value_gap"]
        )
        for scenario in CONFIRM_SCENARIOS:
            for moment in TIMES:
                assert_consumed(int(scenario))
                state = _prepare(int(scenario), int(moment), frozen, matching, config)
                relational = _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))
                reference_true = None
                progress_state = None
                interference_state = None
                print(f"state {scenario} t={moment}", flush=True)
                for seed in SEEDS:
                    true, pred, labels, static, stored_p1, stored_r1 = _stack(
                        book, saved, int(scenario), int(moment), int(seed), state["vectors"]
                    )
                    if reference_true is None:
                        reference_true = true
                        progress_state = _progress_bin(true, state["assignments"])
                        interference_state = float(np.nanmean(scalars(true, state["assignments"])["mean_interference_1"]))
                        state_rows.append(
                            {
                                "scenario_id": int(scenario),
                                "time_step": int(moment),
                                "mean_interference": interference_state,
                                "interference_mae_sum": 0.0,
                                "interference_mae_n": 0,
                                **progress_state,
                            }
                        )
                    else:
                        true_seed_gap = max(true_seed_gap, float(np.max(np.abs(true - reference_true))))
                    state_rows[-1]["interference_mae_sum"] += float(
                        np.mean(np.abs(pred[:, :, list(GROUPS["interference"])] - true[:, :, list(GROUPS["interference"])]))
                    )
                    state_rows[-1]["interference_mae_n"] += 1
                    values = {
                        name: _score(models[int(seed)], state["embedded"], state["assignments"], relational, normalize(raw, norm))
                        for name, raw in _variants(true, pred).items()
                    }
                    align_p1 = max(align_p1, float(np.max(np.abs(values["P1"] - stored_p1))))
                    align_r1 = max(align_r1, float(np.max(np.abs(values["R1"] - stored_r1))))
                    argmax_agreement.append(
                        float(
                            int(np.argsort(-values["P1"], kind="stable")[0]) == int(np.argsort(-stored_p1, kind="stable")[0])
                            and int(np.argsort(-values["R1"], kind="stable")[0]) == int(np.argsort(-stored_r1, kind="stable")[0])
                        )
                    )
                    rows = [
                        _metric(name, int(seed), int(scenario), int(moment), scores, labels, state["vectors"])
                        for name, scores in values.items()
                    ]
                    r1_choice = next(row["assignment_vector"] for row in rows if row["model"] == "R1")
                    p1_choice = next(row["assignment_vector"] for row in rows if row["model"] == "P1")
                    for row in rows:
                        row["agreement_with_r1"] = float(row["assignment_vector"] == r1_choice)
                        row["agreement_with_p1"] = float(row["assignment_vector"] == p1_choice)
                    metric_rows.extend(rows)
                    true_s = scalars(true, state["assignments"])
                    pred_s = scalars(pred, state["assignments"])
                    flat_true.append(true_s["mean_sinr_1"])
                    flat_pred.append(pred_s["mean_sinr_1"])
                    for group, scalar in PRIMARY_SCALAR.items():
                        block = rank_block(true_s[scalar], pred_s[scalar])
                        rank_rows.append(
                            {
                                "scenario_id": int(scenario),
                                "time_step": int(moment),
                                "seed": int(seed),
                                "group": group,
                                "scalar": scalar,
                                "role": "primary",
                                **block,
                            }
                        )
                    for group, indexes in GROUPS.items():
                        for index in indexes:
                            feature = RESPONSE_FEATURES[index]
                            for prefix in ("mean", "min", "max", "std"):
                                scalar = f"{prefix}_{feature}"
                                if scalar == PRIMARY_SCALAR[group] and prefix == "mean":
                                    continue
                                block = rank_block(true_s[scalar], pred_s[scalar])
                                rank_rows.append(
                                    {
                                        "scenario_id": int(scenario),
                                        "time_step": int(moment),
                                        "seed": int(seed),
                                        "group": group,
                                        "scalar": scalar,
                                        "role": "aggregate",
                                        **block,
                                    }
                                )
                    order = np.argsort(-labels, kind="stable")
                    for width in (5, 10, 20):
                        chosen = order[:width]
                        for group, scalar in PRIMARY_SCALAR.items():
                            block = rank_block(true_s[scalar][chosen], pred_s[scalar][chosen])
                            top_rows.append(
                                {
                                    "scenario_id": int(scenario),
                                    "time_step": int(moment),
                                    "seed": int(seed),
                                    "region": f"top{width}",
                                    "group": group,
                                    "scalar": scalar,
                                    "spearman": block["spearman"],
                                    "kendall": block["kendall"],
                                    "pairwise": block["pairwise"],
                                }
                            )
                    r1_top = np.argsort(-values["R1"], kind="stable")[:3]
                    true_rank = np.empty(len(labels), dtype=np.float64)
                    pred_rank = np.empty(len(labels), dtype=np.float64)
                    true_rank[np.argsort(-true_s["mean_sinr_1"], kind="stable")] = np.arange(1, len(labels) + 1)
                    pred_rank[np.argsort(-pred_s["mean_sinr_1"], kind="stable")] = np.arange(1, len(labels) + 1)
                    sinr_rows.append(
                        {
                            "scenario_id": int(scenario),
                            "time_step": int(moment),
                            "seed": int(seed),
                            "mean_sinr_spearman": rank_block(true_s["mean_sinr_1"], pred_s["mean_sinr_1"])["spearman"],
                            "mean_sinr_kendall": rank_block(true_s["mean_sinr_1"], pred_s["mean_sinr_1"])["kendall"],
                            "min_sinr_spearman": rank_block(true_s["min_sinr_1"], pred_s["min_sinr_1"])["spearman"],
                            "top10_sinr_pairwise": rank_block(true_s["mean_sinr_1"][order[:10]], pred_s["mean_sinr_1"][order[:10]])["pairwise"],
                            "r1_top3_sinr_rank_offset": float(np.mean(np.abs(true_rank[r1_top] - pred_rank[r1_top]))),
                        }
                    )
                    for group, indexes in GROUPS.items():
                        error = np.mean(np.abs(pred[:, :, list(indexes)] - true[:, :, list(indexes)]), axis=(1, 2))
                        value_gap = np.abs(values["P1"] - values["R1"])
                        rho, tau = correlations(error, value_gap)
                        value_rows.append(
                            {
                                "scenario_id": int(scenario),
                                "time_step": int(moment),
                                "seed": int(seed),
                                "group": group,
                                "spearman_vs_value_error": rho,
                                "kendall_vs_value_error": tau,
                                "mean_response_mae": float(np.mean(error)),
                                "mean_value_error": float(np.mean(value_gap)),
                            }
                        )
                    p1_index = int(np.argsort(-values["P1"], kind="stable")[0])
                    r1_index = int(np.argsort(-values["R1"], kind="stable")[0])
                    regret = {
                        "scenario_id": int(scenario),
                        "time_step": int(moment),
                        "seed": int(seed),
                        "p1_assignment": state["vectors"][p1_index],
                        "r1_assignment": state["vectors"][r1_index],
                        "j15_regret": float(np.max(quantize(labels)) - quantize(labels)[p1_index]),
                        "j15_gap_r1_minus_p1": float(labels[r1_index] - labels[p1_index]),
                    }
                    for group, scalar in PRIMARY_SCALAR.items():
                        regret[f"delta_true_{group}"] = float(true_s[scalar][r1_index] - true_s[scalar][p1_index])
                        regret[f"delta_pred_{group}"] = float(pred_s[scalar][r1_index] - pred_s[scalar][p1_index])
                        regret[f"sign_reversal_{group}"] = float(
                            regret[f"delta_true_{group}"] * regret[f"delta_pred_{group}"] < 0.0
                        )
                    regret_rows.append(regret)
                    mask = state["assignments"] >= 0
                    for index in CONTINUOUS:
                        feature = RESPONSE_FEATURES[index]
                        truth = true[:, :, index][mask]
                        prediction = pred[:, :, index][mask]
                        std_true = float(np.std(truth))
                        std_pred = float(np.std(prediction))
                        block = rank_block(true_s[f"mean_{feature}"], pred_s[f"mean_{feature}"])
                        calib_rows.append(
                            {
                                "scenario_id": int(scenario),
                                "time_step": int(moment),
                                "seed": int(seed),
                                "feature": feature,
                                "mae": float(np.mean(np.abs(prediction - truth))),
                                "bias": float(np.mean(prediction - truth)),
                                "scale_ratio": float(std_pred / std_true) if std_true > 1.0e-12 else float("nan"),
                                "std_true": std_true,
                                "spearman": block["spearman"],
                                "kendall": block["kendall"],
                                "pairwise": block["pairwise"],
                                "affine_slope": _affine_slope(true_s[f"mean_{feature}"], pred_s[f"mean_{feature}"]),
                            }
                        )
                    _accumulate(
                        flip_totals,
                        _write_flips(
                            flip_handle,
                            int(scenario),
                            int(moment),
                            int(seed),
                            state["vectors"],
                            labels,
                            true_s,
                            pred_s,
                            values,
                        ),
                    )
                    if int(scenario) == CASE_SCENARIO and int(moment) == CASE_TIME:
                        opt_index = state["vectors"].index(CASE_OPTIMAL)
                        pred_index = state["vectors"].index(CASE_PREDICTED)
                        for uav in range(6):
                            for feature in CASE_FEATURES:
                                position = RESPONSE_FEATURES.index(feature)
                                truth = float(true[pred_index, uav, position])
                                prediction = float(pred[pred_index, uav, position])
                                case_rows.append(
                                    {
                                        "row_type": "uav_A_pred",
                                        "seed": int(seed),
                                        "uav": uav,
                                        "feature": feature,
                                        "true": truth,
                                        "predicted": prediction,
                                        "absolute_error": abs(prediction - truth),
                                        "relative_error": abs(prediction - truth) / max(abs(truth), 1.0e-8),
                                    }
                                )
                                truth = float(true[opt_index, uav, position])
                                prediction = float(pred[opt_index, uav, position])
                                case_rows.append(
                                    {
                                        "row_type": "uav_A_opt",
                                        "seed": int(seed),
                                        "uav": uav,
                                        "feature": feature,
                                        "true": truth,
                                        "predicted": prediction,
                                        "absolute_error": abs(prediction - truth),
                                        "relative_error": abs(prediction - truth) / max(abs(truth), 1.0e-8),
                                    }
                                )
                        named = {"S_saved": static, **values}
                        for model, scores in named.items():
                            case_rows.append(
                                {
                                    "row_type": "rank",
                                    "seed": int(seed),
                                    "model": model,
                                    "A_opt_rank": _named_rank(scores, opt_index),
                                    "A_pred_rank": _named_rank(scores, pred_index),
                                    "A_opt_value": float(scores[opt_index]),
                                    "A_pred_value": float(scores[pred_index]),
                                }
                            )
    aligned = bool(align_p1 <= ALIGN_ABS and align_r1 <= ALIGN_ABS)
    recovered = recovery_rows(metric_rows)
    damaged = damage_rows(metric_rows)
    stable = stability_rows(metric_rows)
    intervals = bootstrap_table(metric_rows)
    overall = {
        row["model"]: float(row["recovery"])
        for row in recovered
        if row["level"] == "overall"
    }
    short = {name: overall[f"P1+TRUE_{name.upper()}"] for name in GROUPS}
    short["C3"] = overall["C3"]
    case_sinr = [
        float(row["A_opt_rank"])
        for row in case_rows
        if row["row_type"] == "rank" and row["model"] == "P1+TRUE_SINR"
    ]
    flags = gate_flags(short, intervals, float(np.median(case_sinr)) if case_sinr else float("nan"), aligned)
    flattened = spearmanr(np.concatenate(flat_true), np.concatenate(flat_pred))
    substitution = [row for row in metric_rows if row["model"] == "P1" or row["model"] == "R1" or str(row["model"]).startswith("P1+TRUE_")]
    injection = [row for row in metric_rows if row["model"] == "R1" or str(row["model"]).startswith("R1-PRED_")]
    combined = [row for row in metric_rows if row["model"] in {"P1", "R1", "C1", "C2", "C3"}]
    write_csv(OUTPUT / "phase5zf_statewise_response_ranking.csv", rank_rows)
    write_csv(OUTPUT / "phase5zf_top_region_ranking.csv", top_rows)
    write_csv(OUTPUT / "phase5zf_oracle_channel_substitution.csv", substitution)
    write_csv(OUTPUT / "phase5zf_predicted_channel_injection.csv", injection)
    write_csv(OUTPUT / "phase5zf_combined_substitution.csv", combined)
    write_csv(OUTPUT / "phase5zf_recovery_by_group.csv", recovered)
    write_csv(OUTPUT / "phase5zf_damage_by_group.csv", damaged)
    write_csv(OUTPUT / "phase5zf_sinr_ranking_audit.csv", sinr_rows)
    write_csv(OUTPUT / "phase5zf_interference_audit.csv", _interference_rows(metric_rows, rank_rows, state_rows))
    write_csv(OUTPUT / "phase5zf_progress_audit.csv", _progress_rows(metric_rows, state_rows))
    write_csv(OUTPUT / "phase5zf_calibration_vs_ordering.csv", calibration_rows(calib_rows))
    write_csv(OUTPUT / "phase5zf_response_to_value_error.csv", value_rows)
    write_csv(OUTPUT / "phase5zf_response_to_regret.csv", regret_rows)
    write_csv(OUTPUT / "phase5zf_310114_t15.csv", case_rows)
    write_csv(OUTPUT / "phase5zf_scenario_seed_stability.csv", stable)
    statistics = {
        "aligned": aligned,
        "align_abs_p1": align_p1,
        "align_abs_r1": align_r1,
        "align_tolerance": ALIGN_ABS,
        "true_response_seed_gap": true_seed_gap,
        "stored_argmax_agreement": mean(argmax_agreement),
        "flattened_mean_sinr_spearman_not_primary": float(flattened.statistic),
        "statewise_primary": {
            group: {
                "spearman": mean(row["spearman"] for row in rank_rows if row["group"] == group and row["role"] == "primary"),
                "kendall": mean(row["kendall"] for row in rank_rows if row["group"] == group and row["role"] == "primary"),
                "pairwise": mean(row["pairwise"] for row in rank_rows if row["group"] == group and row["role"] == "primary"),
            }
            for group in GROUPS
        },
        "top10_sinr_pairwise": mean(row["top10_sinr_pairwise"] for row in sinr_rows),
        "mean_sinr_spearman": mean(row["mean_sinr_spearman"] for row in sinr_rows),
        "mean_sinr_kendall": mean(row["mean_sinr_kendall"] for row in sinr_rows),
        "min_sinr_spearman": mean(row["min_sinr_spearman"] for row in sinr_rows),
        "r1_top3_sinr_rank_offset": mean(row["r1_top3_sinr_rank_offset"] for row in sinr_rows),
        "recovery_overall": overall,
        "damage_overall": {row["model"]: float(row["damage"]) for row in damaged if row["level"] == "overall"},
        "bootstrap_vs_p1": intervals,
        "flip_sign_reversal": flip_totals,
        "selected_j15": {
            model: mean(row["selected_j15"] for row in metric_rows if row["model"] == model)
            for model in sorted({row["model"] for row in metric_rows})
        },
        "gate_flags": flags,
        "optimizer_steps": 0,
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
        "opened_310118_310125": False,
    }
    (OUTPUT / "phase5zf_statistics.json").write_text(json.dumps(plain(statistics), indent=2), encoding="utf-8")
    gate = {
        "gate": flags["gate"],
        "aligned": aligned,
        "recovery": short,
        "case_p1_true_sinr_median_rank": flags["case_median_rank"],
        "bootstrap_low": {name: intervals[name]["low"] for name in ("P1+TRUE_SINR", "P1+TRUE_PROGRESS", "P1+TRUE_INTERFERENCE", "C3")},
        "scenario_direction": {model: direction_counts(stable, model, "scenario") for model in ("P1+TRUE_SINR", "P1+TRUE_PROGRESS", "P1+TRUE_INTERFERENCE", "C3")},
        "seed_direction": {model: direction_counts(stable, model, "seed") for model in ("P1+TRUE_SINR", "P1+TRUE_PROGRESS", "P1+TRUE_INTERFERENCE", "C3")},
        "optimizer_steps": 0,
        "new_checkpoint": False,
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
        "opened_310118_310125": False,
        "decide_called": False,
    }
    (OUTPUT / "phase5zf_distillation_error_gate.json").write_text(json.dumps(plain(gate), indent=2), encoding="utf-8")
    print(json.dumps(plain(gate), indent=2), flush=True)


if __name__ == "__main__":
    run()
