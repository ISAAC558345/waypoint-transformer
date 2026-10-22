"""Read saved Phase-5ZH records. One diagnostic graph, and no parameter update."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import quantize
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES
from wt.lowlevel_v2.phase5zd_executor_response_value.scorer import ExecutorConditionedJointScorer
from wt.lowlevel_v2.phase5ze_response_distillation.predictor import (
    BINARY_INDEX,
    CONTINUOUS_INDEX,
    OneStepExecutorResponsePredictor,
    response_loss,
)
from wt.lowlevel_v2.phase5ze_response_distillation.tokens import token_dim
from wt.lowlevel_v2.phase5zf_distillation_error_audit.common import mean, plain, write_csv
from wt.lowlevel_v2.phase5zh_ranking_response_distillation.loss import pairwise_rank_loss, ranking_pairs
from wt.lowlevel_v2.phase5zh_ranking_response_distillation.protocol import (
    LAMBDA_RANK,
    PAIR_RNG_SEED,
    TOP_INTERNAL,
    TOP_OUTSIDE,
    VALUE_TEMPERATURE,
    VALUE_TIE,
)

from .metrics import choose_gate
from .protocol import (
    DIAGNOSTIC_SEED,
    EASY_LOSS_SHARE,
    EASY_TEACHER_GAP,
    HARD_LOSS_SHARE,
    OUTPUT,
    SATURATION_HIGH,
    SATURATION_LOW,
    TEACHER_SPEARMAN_MIN,
    TEACHER_TOP1_MIN,
    VALUE_STD_HIGH,
    VALUE_STD_LOW,
    assert_consumed,
)

LOWLEVEL = OUTPUT.parents[0]


def _rows(path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _spearman(left, right) -> float:
    observed = np.asarray(left, dtype=np.float64)
    predicted = np.asarray(right, dtype=np.float64)
    if observed.size < 3 or float(np.std(observed)) <= 1.0e-12 or float(np.std(predicted)) <= 1.0e-12:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def _grad_vector(module) -> torch.Tensor:
    parts = []
    for parameter in module.parameters():
        if parameter.grad is None:
            parts.append(torch.zeros(parameter.numel()))
        else:
            parts.append(parameter.grad.detach().reshape(-1).cpu())
    return torch.cat(parts)


def _channel_l2(prediction: torch.Tensor, index: int) -> float:
    gradient = prediction.grad
    if gradient is None:
        return 0.0
    return float(gradient[..., index].detach().pow(2).sum().sqrt())


def _load_norm() -> dict[str, np.ndarray]:
    payload = json.loads((LOWLEVEL / "phase5zd_executor_response_value" / "phase5zd_response_normalization.json").read_text(encoding="utf-8"))
    return {"mean": np.asarray(payload["mean"], dtype=np.float64), "std": np.asarray(payload["std"], dtype=np.float64)}


def _value_tensor(prediction: torch.Tensor, norm: dict[str, np.ndarray]) -> torch.Tensor:
    value = prediction.clone()
    mean = torch.as_tensor(norm["mean"], dtype=prediction.dtype)
    std = torch.as_tensor(norm["std"], dtype=prediction.dtype)
    probability = torch.sigmoid(prediction[..., list(BINARY_INDEX)])
    value[..., list(BINARY_INDEX)] = (probability - mean[list(BINARY_INDEX)]) / std[list(BINARY_INDEX)]
    return value


def _diagnostic(norm: dict[str, np.ndarray]) -> dict:
    """Fixed tensors through the frozen modules. The optimizer is not constructed."""
    torch.manual_seed(DIAGNOSTIC_SEED)
    rng = np.random.default_rng(DIAGNOSTIC_SEED)
    scorer = ExecutorConditionedJointScorer()
    scorer.load_state_dict(torch.load(LOWLEVEL / "phase5zd_executor_response_value" / "checkpoints" / "phase5zd_response_seed0.pt", map_location="cpu", weights_only=False)["state_dict"])
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    predictor = OneStepExecutorResponsePredictor(token_dim())
    predictor.load_state_dict(torch.load(OUTPUT.parents[0] / "phase5zh_ranking_response_distillation" / "checkpoints" / "phase5zh_rank_seed0.pt", map_location="cpu", weights_only=False)["state_dict"])
    predictor.train()
    width = int(scorer.width)
    rel_dim = int(scorer.rel_dim)
    response_dim = int(scorer.response_dim)
    global_dim = int(scorer.mlp[0].in_features) - (width * 3 + response_dim + width * 2 + rel_dim)
    count, uavs, buoys = 120, 6, 12
    tokens = torch.as_tensor(rng.normal(size=(count, uavs, token_dim())), dtype=torch.float32)
    prediction = predictor(tokens)
    prediction.retain_grad()
    target = torch.as_tensor(rng.normal(size=prediction.shape), dtype=torch.float32)
    binary = torch.zeros(prediction.shape[0], prediction.shape[1], len(BINARY_INDEX))
    regression, _huber, _bce = response_loss(prediction, target, binary)
    assignments = torch.as_tensor(rng.integers(-1, buoys, size=(count, uavs)), dtype=torch.long)
    assignments[:, 0] = 0
    features = {
        "h_u": torch.randn(1, uavs, width),
        "h_b": torch.randn(1, buoys, width),
        "h_edge": torch.randn(1, uavs, buoys, width),
        "h_uu": torch.randn(1, uavs, uavs, width),
        "z_global": torch.randn(1, global_dim),
        "z_rel": torch.randn(count, rel_dim),
        "response": _value_tensor(prediction, norm),
    }
    features["response"].retain_grad()
    student = scorer(features, assignments)
    teacher = rng.normal(size=count)
    left, right = ranking_pairs(teacher, np.random.default_rng(DIAGNOSTIC_SEED), VALUE_TIE, 16)
    ranking = pairwise_rank_loss(student, torch.as_tensor(left, dtype=torch.long), torch.as_tensor(right, dtype=torch.long), VALUE_TEMPERATURE)
    predictor.zero_grad(set_to_none=True)
    regression.backward(retain_graph=True)
    response_grad = _grad_vector(predictor)
    sinr_response = _channel_l2(prediction, 0)
    rate_response = _channel_l2(prediction, RESPONSE_FEATURES.index("rate_1"))
    predictor.zero_grad(set_to_none=True)
    if prediction.grad is not None:
        prediction.grad = None
    ranking.backward(retain_graph=True)
    rank_grad = _grad_vector(predictor)
    sinr_rank = _channel_l2(prediction, 0)
    rate_rank = _channel_l2(prediction, RESPONSE_FEATURES.index("rate_1"))
    leaked = 0.0
    for parameter in scorer.parameters():
        if parameter.grad is not None:
            leaked += float(parameter.grad.detach().abs().sum())
    response_reached = features["response"].grad is not None and float(features["response"].grad.detach().abs().sum()) > 0.0
    cosine = float("nan")
    if float(response_grad.norm()) > 0.0 and float(rank_grad.norm()) > 0.0:
        cosine = float(torch.dot(response_grad, rank_grad) / (response_grad.norm() * rank_grad.norm()))
    opposite = float(torch.mean(((response_grad * rank_grad) < 0).to(torch.float32)))
    same_state = bool(np.all(left >= 0) and np.all(left < count) and np.all(right >= 0) and np.all(right < count))
    orientation = bool(np.all(teacher[left] > teacher[right] + VALUE_TIE)) if left.size else False
    return {
        "diagnostic_tensor": True,
        "optimizer_step": False,
        "same_state_indices": same_state,
        "teacher_orientation_higher_first": orientation,
        "response_grad_reached_predictor_input": response_reached,
        "rank_grad_l2": float(rank_grad.norm()),
        "response_grad_l2": float(response_grad.norm()),
        "diagnostic_cosine": cosine,
        "diagnostic_opposite_sign_fraction": opposite,
        "lambda_times_rank_over_response": float(LAMBDA_RANK) * float(rank_grad.norm()) / float(response_grad.norm()) if float(response_grad.norm()) > 0.0 else float("nan"),
        "sinr_head_l2_response": sinr_response,
        "rate_head_l2_response": rate_response,
        "sinr_head_l2_rank": sinr_rank,
        "rate_head_l2_rank": rate_rank,
        "frozen_value_parameter_grad": leaked,
        "pair_count": int(left.size),
        "global_dim": int(global_dim),
    }


def _saved_groups() -> dict[tuple[int, int, int], dict[str, np.ndarray]]:
    grouped: dict[tuple[int, int, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    path = LOWLEVEL / "phase5ze_response_distillation" / "phase5ze_downstream_value_predictions.csv"
    for row in _rows(path):
        scenario = int(row["scenario_id"])
        assert_consumed(scenario)
        key = (scenario, int(row["time_step"]), int(row["seed"]))
        grouped[key]["assignment"].append(row["assignment_vector"])
        grouped[key]["j15"].append(float(row["j15"]))
        grouped[key]["v_r1"].append(float(row["v_r1"]))
        grouped[key]["v_p1"].append(float(row["v_p1"]))
        grouped[key]["v_s"].append(float(row["v_s"]))
    output = {}
    for key, columns in grouped.items():
        output[key] = {name: np.asarray(values) for name, values in columns.items()}
    return output


def _bucket(teacher: np.ndarray, first: int, second: int) -> str:
    order = np.argsort(-teacher, kind="stable")
    rank = {int(index): place for place, index in enumerate(order)}
    if rank[first] < TOP_INTERNAL and rank[second] < TOP_INTERNAL:
        return "top10_internal"
    if min(rank[first], rank[second]) < TOP_OUTSIDE:
        return "top5_tail"
    return "other"


def _teacher_and_pairs(groups: dict) -> tuple[list[dict], dict]:
    rows = []
    shares = defaultdict(list)
    saturations = []
    hard_weights = []
    tail_weights = []
    for (scenario, moment, seed), columns in sorted(groups.items()):
        teacher = columns["v_r1"].astype(np.float64)
        student = columns["v_p1"].astype(np.float64)
        labels = columns["j15"].astype(np.float64)
        order = np.argsort(-teacher, kind="stable")
        levels = quantize(labels)
        best = float(np.max(levels))
        top1_hit = float(levels[int(np.argmax(teacher))] == best)
        gaps = np.abs(teacher[:, None] - teacher[None, :])
        upper = np.triu(np.ones(teacher.shape[0], dtype=bool), k=1)
        tie_rate = float(np.mean(gaps[upper] <= VALUE_TIE))
        rng = np.random.default_rng(PAIR_RNG_SEED + int(seed) * 10007 + int(scenario) + int(moment))
        left, right = ranking_pairs(teacher, rng, VALUE_TIE)
        losses = {"top10_internal": 0.0, "top5_tail": 0.0, "other": 0.0}
        counts = {"top10_internal": 0, "top5_tail": 0, "other": 0}
        easy = 0.0
        total = 0.0
        weights = []
        for first, second in zip(left, right):
            margin = (float(student[int(first)]) - float(student[int(second)])) / VALUE_TEMPERATURE
            loss = float(np.logaddexp(0.0, -margin))
            weight = float(np.exp(-margin) / (1.0 + np.exp(-margin))) if margin > -40.0 else (1.0 if margin < 40.0 else 0.0)
            bucket = _bucket(teacher, int(first), int(second))
            losses[bucket] += loss
            counts[bucket] += 1
            total += loss
            weights.append(weight)
            if abs(float(teacher[int(first)]) - float(teacher[int(second)])) > EASY_TEACHER_GAP:
                easy += loss
            if bucket == "top10_internal":
                hard_weights.append(weight)
            else:
                tail_weights.append(weight)
        shares["easy"].append(easy / total if total > 0.0 else float("nan"))
        shares["hard"].append(losses["top10_internal"] / total if total > 0.0 else float("nan"))
        shares["tail"].append(losses["top5_tail"] / total if total > 0.0 else float("nan"))
        saturations.extend(weights)
        rows.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(moment),
                "seed": int(seed),
                "candidate_count": int(teacher.size),
                "unique_assignments": int(len(set(columns["assignment"].tolist()))),
                "teacher_std": float(np.std(teacher)),
                "teacher_span": float(np.max(teacher) - np.min(teacher)),
                "pair_tie_rate": tie_rate,
                "top1_margin": float(teacher[order[0]] - teacher[order[1]]),
                "top3_margin": float(teacher[order[2]] - teacher[order[3]]),
                "teacher_j15_spearman": _spearman(teacher, labels),
                "teacher_top1_hits_j15": top1_hit,
                "pair_count": int(left.size),
                "loss_share_top10_internal": shares["hard"][-1],
                "loss_share_top5_tail": shares["tail"][-1],
                "loss_share_teacher_gap_above_one_block": shares["easy"][-1],
                "mean_sigmoid_weight": mean(weights),
            }
        )
    summary = {
        "teacher_spearman": mean(row["teacher_j15_spearman"] for row in rows),
        "teacher_top1": mean(row["teacher_top1_hits_j15"] for row in rows),
        "teacher_std": mean(row["teacher_std"] for row in rows),
        "pair_tie_rate": mean(row["pair_tie_rate"] for row in rows),
        "top1_margin": mean(row["top1_margin"] for row in rows),
        "top3_margin": mean(row["top3_margin"] for row in rows),
        "easy_loss_share": mean(shares["easy"]),
        "hard_loss_share": mean(shares["hard"]),
        "tail_loss_share": mean(shares["tail"]),
        "mean_sigmoid_weight": mean(saturations),
        "saturated_low_fraction": mean(float(weight < SATURATION_LOW) for weight in saturations),
        "saturated_high_fraction": mean(float(weight > SATURATION_HIGH) for weight in saturations),
        "hard_mean_sigmoid": mean(hard_weights),
        "tail_mean_sigmoid": mean(tail_weights),
        "duplicate_assignment_states": int(sum(row["unique_assignments"] != row["candidate_count"] for row in rows)),
    }
    return rows, summary


def _loss_rows() -> list[dict]:
    source = _rows(LOWLEVEL / "phase5zh_ranking_response_distillation" / "phase5zh_training_curves.csv")
    rows = []
    for row in source:
        rows.append(
            {
                "seed": int(row["seed"]),
                "epoch": int(row["epoch"]),
                "response_loss": float(row["response_loss"]),
                "rank_loss": float(row["rank_loss"]),
                "grad_norm_before_clip": float(row["grad_norm"]),
                "statewise_value_spearman": "MISSING_CHECKPOINT",
                "pairwise_ordering_accuracy": "MISSING_CHECKPOINT",
                "top1": "MISSING_CHECKPOINT",
                "top3": "MISSING_CHECKPOINT",
                "selected_j15": "MISSING_CHECKPOINT",
            }
        )
    selected = _rows(LOWLEVEL / "phase5zh_ranking_response_distillation" / "phase5zh_selected_j15.csv")
    for row in selected:
        if row["model"] != "ZH":
            continue
        rows.append(
            {
                "seed": "confirmation_mean",
                "epoch": 10,
                "response_loss": "NOT_THIS_SPLIT",
                "rank_loss": "NOT_THIS_SPLIT",
                "grad_norm_before_clip": "NOT_THIS_SPLIT",
                "statewise_value_spearman": float(row["value_spearman"]),
                "pairwise_ordering_accuracy": float(row["value_pairwise"]),
                "top1": float(row["top1"]),
                "top3": float(row["top3"]),
                "selected_j15": float(row["selected_j15_mean"]),
                "note": "epoch-10 confirmation only; epochs 1-9 were not saved",
            }
        )
    return rows


def _case_rows(groups: dict) -> list[dict]:
    saved = _rows(LOWLEVEL / "phase5zh_ranking_response_distillation" / "phase5zh_310114_t15.csv")
    rows = []
    for row in saved:
        if row.get("row_type") != "rank":
            continue
        seed = int(row["seed"])
        columns = groups[(310114, 15, seed)]
        vectors = columns["assignment"].tolist()
        opt_index = vectors.index("-1;4;2;5;-1;-1")
        pred_index = vectors.index("2;5;-1;4;-1;-1")
        rows.append(
            {
                "seed": seed,
                "model": row["model"],
                "A_opt_rank": int(float(row["A_opt_rank"])),
                "A_pred_rank": int(float(row["A_pred_rank"])),
                "saved_margin": float(row["margin_opt_minus_pred"]),
                "true_value_margin": float(columns["v_r1"][opt_index] - columns["v_r1"][pred_index]),
                "p1_value_margin": float(columns["v_p1"][opt_index] - columns["v_p1"][pred_index]),
                "true_j15_opt": float(columns["j15"][opt_index]),
                "true_j15_pred": float(columns["j15"][pred_index]),
            }
        )
    for row in saved:
        if row.get("row_type") != "channel":
            continue
        rows.append(
            {
                "seed": int(row["seed"]),
                "model": row["model"],
                "assignment": row["assignment"],
                "true_mean_sinr": float(row["true_mean_sinr"]),
                "pred_mean_sinr": float(row["pred_mean_sinr"]),
                "sinr_error": float(row["pred_mean_sinr"]) - float(row["true_mean_sinr"]),
                "true_mean_rate": float(row["true_mean_rate"]),
                "pred_mean_rate": float(row["pred_mean_rate"]),
                "rate_error": float(row["pred_mean_rate"]) - float(row["true_mean_rate"]),
            }
        )
    return rows


def _residual_rows() -> list[dict]:
    rows = []
    for row in _rows(LOWLEVEL / "phase5zh_ranking_response_distillation" / "phase5zh_residual_cases.csv"):
        if row.get("row_type") != "case":
            continue
        rows.append(
            {
                "scenario_id": int(row["scenario_id"]),
                "time_step": int(row["time_step"]),
                "predicted_assignment": row["predicted_assignment"],
                "optimal_assignment": row["optimal_assignment"],
                "case_direction": row["case_direction"],
                "selected_j15_P1": float(row["selected_j15_P1"]),
                "selected_j15_ZH": float(row["selected_j15_ZH"]),
                "optimum_rank_P1": float(row["optimum_rank_P1"]),
                "optimum_rank_ZH": float(row["optimum_rank_ZH"]),
                "response_ordering_error_P1": float(row["response_ordering_error_P1"]),
                "response_ordering_error_ZH": float(row["response_ordering_error_ZH"]),
                "value_margin_error_P1": float(row["value_margin_error_P1"]),
                "value_margin_error_ZH": float(row["value_margin_error_ZH"]),
            }
        )
    return rows


def _compute_rows() -> list[dict]:
    runtime = {row["path"]: row for row in _rows(LOWLEVEL / "phase5ze_response_distillation" / "phase5ze_runtime.csv")}
    true = runtime["TRUE-H1"]
    predicted = runtime["PREDICTED-H1"]
    total = float(true["total_ms"])
    names = (
        ("env_clone_ms", "environment copy before the one-step transition"),
        ("mappo_action_ms", "executor.act, including MAPPO forward, residual lock, and safety projection"),
        ("env_step_ms", "one environment step on the cloned state"),
        ("response_extract_ms", "response assembly after the step"),
        ("encoder_ms", "encoder features, once per timed call"),
        ("value_ms", "frozen value forward"),
    )
    rows = []
    for key, note in names:
        value = float(true[key])
        rows.append(
            {
                "path": "TRUE-H1",
                "component": key,
                "mean_ms": value,
                "share_of_515ms": value / total,
                "per_candidate_ms": value / 120.0 if key != "encoder_ms" else value,
                "status": "MEASURED",
                "note": note,
            }
        )
    rows.append(
        {
            "path": "TRUE-H1",
            "component": "pure_mappo_forward_only",
            "mean_ms": "",
            "share_of_515ms": "",
            "status": "NOT_MEASURED",
            "note": "action timer wraps the whole executor call",
        }
    )
    rows.append(
        {
            "path": "PREDICTED-H1",
            "component": "total_ms",
            "mean_ms": float(predicted["total_ms"]),
            "mappo_action_ms": float(predicted["mappo_action_ms"]),
            "response_predictor_ms": float(predicted["response_predictor_ms"]),
            "value_ms": float(predicted["value_ms"]),
            "status": "MEASURED",
            "note": "100 repeats on one saved confirmation state, 120 serial calls",
        }
    )
    rows.append(
        {
            "path": "TRUE-H1",
            "component": "pool_size_scaling",
            "status": "NOT_MEASURED",
            "note": "timed only at 120 candidates; the code loops once per candidate",
        }
    )
    rows.append(
        {
            "path": "TRUE-H1",
            "component": "batched_mappo_speedup",
            "status": "NOT_MEASURED",
            "note": "current code calls act once per assignment",
        }
    )
    return rows


def _checkpoints() -> dict:
    folder = LOWLEVEL / "phase5zh_ranking_response_distillation" / "checkpoints"
    found = sorted(path.name for path in folder.glob("*.pt"))
    epochs = []
    for name in found:
        payload = torch.load(folder / name, map_location="cpu", weights_only=False)
        epochs.append({"file": name, "epoch": int(payload["epoch"]), "seed": int(payload["seed"])})
    return {"files": found, "epochs": epochs, "intermediate_epochs_saved": False}


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    norm = _load_norm()
    diagnostic = _diagnostic(norm)
    groups = _saved_groups()
    teacher_rows, teacher = _teacher_and_pairs(groups)
    selected = {row["model"]: row for row in _rows(LOWLEVEL / "phase5zh_ranking_response_distillation" / "phase5zh_selected_j15.csv")}
    checkpoints = _checkpoints()
    implementation_error = bool(
        (not diagnostic["response_grad_reached_predictor_input"])
        or diagnostic["rank_grad_l2"] <= 0.0
        or diagnostic["frozen_value_parameter_grad"] != 0.0
        or (not diagnostic["same_state_indices"])
        or (not diagnostic["teacher_orientation_higher_first"])
        or not (VALUE_STD_LOW <= teacher["teacher_std"] <= VALUE_STD_HIGH)
    )
    teacher_mismatch = bool(teacher["teacher_spearman"] < TEACHER_SPEARMAN_MIN or teacher["teacher_top1"] < TEACHER_TOP1_MIN)
    easy_region = bool(teacher["easy_loss_share"] >= EASY_LOSS_SHARE and teacher["hard_loss_share"] < HARD_LOSS_SHARE)
    j_s = float(selected["S"]["selected_j15_mean"])
    j_p1 = float(selected["P1"]["selected_j15_mean"])
    j_zh = float(selected["ZH"]["selected_j15_mean"])
    j_r1 = float(selected["R1"]["selected_j15_mean"])
    true_h1_stronger = bool((j_r1 - j_s) > 0.0 and (j_zh - j_s) < 0.0 and (j_p1 - j_s) <= 0.0)
    letter = choose_gate(
        implementation_error=implementation_error,
        teacher_mismatch=teacher_mismatch,
        easy_region=easy_region,
        true_h1_stronger=true_h1_stronger,
    )
    protocol = {
        "phase": "5ZI",
        "training": False,
        "optimizer_step": False,
        "new_h1_rollout": False,
        "opened_310118_310125": False,
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
        "easy_teacher_gap": EASY_TEACHER_GAP,
        "easy_loss_share_bar": EASY_LOSS_SHARE,
        "hard_loss_share_bar": HARD_LOSS_SHARE,
        "teacher_spearman_min": TEACHER_SPEARMAN_MIN,
        "teacher_top1_min": TEACHER_TOP1_MIN,
    }
    integrity = {
        "ranking_loss_in_backward": True,
        "code_ranking_term": "regression + 0.2 * ranking, then backward",
        "response_not_detached_in_value_forward": True,
        "training_loop_does_not_wrap_ranking_in_no_grad": True,
        "pairs_built_per_state": True,
        "temperature": VALUE_TEMPERATURE,
        "temperature_unit": "one payload block in the frozen value output",
        "real_training_parameter_gradients": "NOT_MEASURED",
        "reason": "training states were not serialized; a new environment rollout is outside this phase",
        "saved_checkpoints": checkpoints,
        "diagnostic": diagnostic,
        "score_space_on_saved_confirmation": teacher,
    }
    conflict_rows = [
        {"item": "real_training_batch_cosine", "value": "NOT_MEASURED", "status": "NOT_MEASURED"},
        {"item": "diagnostic_response_grad_l2", "value": diagnostic["response_grad_l2"], "status": "DIAGNOSTIC_TENSOR"},
        {"item": "diagnostic_rank_grad_l2", "value": diagnostic["rank_grad_l2"], "status": "DIAGNOSTIC_TENSOR"},
        {"item": "diagnostic_cosine", "value": diagnostic["diagnostic_cosine"], "status": "DIAGNOSTIC_TENSOR"},
        {"item": "diagnostic_opposite_sign_fraction", "value": diagnostic["diagnostic_opposite_sign_fraction"], "status": "DIAGNOSTIC_TENSOR"},
        {"item": "diagnostic_lambda_rank_over_response", "value": diagnostic["lambda_times_rank_over_response"], "status": "DIAGNOSTIC_TENSOR"},
        {"item": "diagnostic_sinr_head_l2_rank", "value": diagnostic["sinr_head_l2_rank"], "status": "DIAGNOSTIC_TENSOR"},
        {"item": "diagnostic_rate_head_l2_rank", "value": diagnostic["rate_head_l2_rank"], "status": "DIAGNOSTIC_TENSOR"},
        {"item": "saved_easy_loss_share", "value": teacher["easy_loss_share"], "status": "SAVED_CONFIRMATION_SCORES"},
        {"item": "saved_hard_loss_share", "value": teacher["hard_loss_share"], "status": "SAVED_CONFIRMATION_SCORES"},
        {"item": "saved_tail_loss_share", "value": teacher["tail_loss_share"], "status": "SAVED_CONFIRMATION_SCORES"},
        {"item": "saved_saturated_low_fraction", "value": teacher["saturated_low_fraction"], "status": "SAVED_CONFIRMATION_SCORES"},
        {"item": "saved_saturated_high_fraction", "value": teacher["saturated_high_fraction"], "status": "SAVED_CONFIRMATION_SCORES"},
        {"item": "saved_hard_mean_sigmoid", "value": teacher["hard_mean_sigmoid"], "status": "SAVED_CONFIRMATION_SCORES"},
        {"item": "saved_tail_mean_sigmoid", "value": teacher["tail_mean_sigmoid"], "status": "SAVED_CONFIRMATION_SCORES"},
    ]
    route = {
        "route_A_predictor": {
            "scientific_feasibility": "the network can be trained, and both P1 and ZH-Rank finished",
            "measured_gain_vs_S": j_zh - j_s,
            "p1_gain_vs_S": j_p1 - j_s,
            "requires_new_training_for_current_loss": False,
            "changes_high_level_encoder": False,
            "keeps_mappo": True,
            "assignment_rank_error_remains": True,
        },
        "route_B_true_h1": {
            "scientific_feasibility": "true H1 response already changes the frozen value ranking",
            "measured_gain_vs_S": j_r1 - j_s,
            "requires_new_training": False,
            "changes_high_level_encoder": False,
            "keeps_mappo": True,
            "online_planner_verified": False,
            "deployment_budget_verified": False,
        },
    }
    gate = {
        "gate": letter,
        "implementation_error": implementation_error,
        "teacher_mismatch": teacher_mismatch,
        "easy_region": easy_region,
        "true_h1_stronger": true_h1_stronger,
        "mechanisms_recorded": {
            "implementation": "connected" if not implementation_error else "broken",
            "teacher_alignment": teacher,
            "easy_region": easy_region,
            "real_parameter_conflict": "NOT_MEASURED",
            "per_epoch_ranking": "MISSING_CHECKPOINT",
        },
        "selected_j15": {"S": j_s, "P1": j_p1, "ZH": j_zh, "R1": j_r1},
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
        "opened_310118_310125": False,
        "training_this_phase": False,
    }
    (OUTPUT / "phase5zi_protocol.json").write_text(json.dumps(plain(protocol), indent=2), encoding="utf-8")
    (OUTPUT / "phase5zi_gradient_integrity.json").write_text(json.dumps(plain(integrity), indent=2), encoding="utf-8")
    write_csv(OUTPUT / "phase5zi_gradient_conflict.csv", conflict_rows)
    write_csv(OUTPUT / "phase5zi_teacher_value_audit.csv", teacher_rows)
    write_csv(OUTPUT / "phase5zi_loss_vs_ranking.csv", _loss_rows())
    write_csv(OUTPUT / "phase5zi_310114_t15.csv", _case_rows(groups))
    write_csv(OUTPUT / "phase5zi_residual_cases.csv", _residual_rows())
    write_csv(OUTPUT / "phase5zi_true_h1_compute_audit.csv", _compute_rows())
    (OUTPUT / "phase5zi_route_comparison.json").write_text(json.dumps(plain(route), indent=2), encoding="utf-8")
    (OUTPUT / "phase5zi_route_selection_gate.json").write_text(json.dumps(plain(gate), indent=2), encoding="utf-8")
    print(json.dumps(plain(gate), indent=2), flush=True)


if __name__ == "__main__":
    run()
