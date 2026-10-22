"""Train the same response predictor with one added state-conditional ranking term."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch.nn.utils import clip_grad_norm_

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import (
    PAYLOAD_MATCH_ABS,
    PROGRESS_MATCH_ABS,
    SINR_MATCH_ABS,
)
from wt.lowlevel_v2.phase5ze_response_distillation.predictor import (
    BINARY_INDEX,
    CONTINUOUS_INDEX,
    OneStepExecutorResponsePredictor,
    response_loss,
)
from wt.lowlevel_v2.phase5ze_response_distillation.protocol import EDGE_MATCH_ABS, HIDDEN, LABEL_MATCH_ABS
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import (
    _apply_affine,
    _feature_block,
    _fit_input,
    _horizon_table,
    _labels,
    _load_predictor,
    _load_response_norm,
    _load_value,
    _predict_value,
    _prepare,
    _relational,
    _to_raw,
    _to_value,
)
from wt.lowlevel_v2.phase5ze_response_distillation.tokens import token_dim
from wt.lowlevel_v2.phase5zf_distillation_error_audit.common import mean, plain, write_csv
from wt.lowlevel_v2.phase5zf_distillation_error_audit.metrics import pairwise_accuracy
from wt.round30.ppo import seed_everything

from .loss import pairwise_rank_loss, ranking_pairs
from .metrics import choose_gate
from .protocol import (
    ALIGN_ABS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMAL,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CHANNEL_STD_MIN,
    CHECKPOINTS,
    CONFIRM_SCENARIOS,
    EPOCHS,
    EXTREME_FRACTION,
    GRAD_CLIP,
    LAMBDA_RANK,
    LATENCY_REPEATS,
    LATENCY_WARMUP,
    LEARNING_RATE,
    MAE_RATIO_MAX,
    MIN_SEEDS_UP,
    OUTPUT,
    PAIR_RNG_SEED,
    PAYLOAD_BLOCK_BITS,
    RANDOM_PAIRS,
    RESPONSE_FEATURES,
    RETENTION_BAR,
    SCORE_COLLAPSE_STD,
    SEEDS,
    SINR_RATE_SPEARMAN_MIN,
    TIE_BITS,
    TIMES,
    TOKEN_NORM_ABS,
    TRAIN_SCENARIOS,
    VALUE_TEMPERATURE,
    VALUE_TIE,
    WEIGHT_DECAY,
    Z_EXTREME,
    assert_consumed,
)

MODELS = ("S", "P1", "ZH", "R1")


def _checksum(module) -> float:
    total = 0.0
    for parameter in module.parameters():
        total += float(parameter.detach().abs().sum())
    return total


def _count(module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _value_tensor(prediction: torch.Tensor, norm: dict[str, np.ndarray]) -> torch.Tensor:
    value = prediction.clone()
    mean = torch.as_tensor(norm["mean"], dtype=prediction.dtype, device=prediction.device)
    std = torch.as_tensor(norm["std"], dtype=prediction.dtype, device=prediction.device)
    probability = torch.sigmoid(prediction[..., list(BINARY_INDEX)])
    value[..., list(BINARY_INDEX)] = (probability - mean[list(BINARY_INDEX)]) / std[list(BINARY_INDEX)]
    return value


def _score_tensor(scorer, state: dict, response: torch.Tensor, relational: np.ndarray) -> torch.Tensor:
    features = {}
    for key, value in state["embedded"].items():
        features[key] = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    features["z_rel"] = torch.as_tensor(relational, dtype=torch.float32)
    features["response"] = response
    return scorer(features, torch.as_tensor(state["assignments"], dtype=torch.long))


def _spearman(left, right) -> float:
    observed = np.asarray(left, dtype=np.float64)
    predicted = np.asarray(right, dtype=np.float64)
    if observed.size < 3 or float(np.std(observed)) <= 1.0e-12 or float(np.std(predicted)) <= 1.0e-12:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def _kendall(left, right) -> float:
    observed = np.asarray(left, dtype=np.float64)
    predicted = np.asarray(right, dtype=np.float64)
    if observed.size < 3 or float(np.std(observed)) <= 1.0e-12 or float(np.std(predicted)) <= 1.0e-12:
        return float("nan")
    return float(kendalltau(observed, predicted).statistic)


def _assigned_mean(raw: np.ndarray, feature: int, assignments: np.ndarray) -> np.ndarray:
    mask = assignments >= 0
    held = np.where(mask, raw[:, :, feature], 0.0)
    count = mask.sum(axis=1)
    return np.divide(held.sum(axis=1), count, out=np.full(count.shape, np.nan), where=count > 0)


def _top5(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:5]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _named_rank(scores: np.ndarray, index: int) -> int:
    values = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(values > values[int(index)]))


def _decision(model: str, seed: int, state: dict, scores: np.ndarray, teacher: np.ndarray) -> dict:
    chosen = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    measured = metrics_at(scores, state["j_mappo"], chosen)
    optimum = int(np.argmax(quantize(state["j_mappo"])))
    true_sinr = _assigned_mean(state["response_raw"], 0, state["assignments"])
    pred_sinr = _assigned_mean(state["pred_raw"][model], 0, state["assignments"]) if model in state["pred_raw"] else true_sinr
    true_rate = _assigned_mean(state["response_raw"], 6, state["assignments"])
    pred_rate = _assigned_mean(state["pred_raw"][model], 6, state["assignments"]) if model in state["pred_raw"] else true_rate
    return {
        "model": model,
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "selected_j15": measured["j_mappo"],
        "normalized_regret": measured["normalized_regret"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "top5": _top5(scores, state["j_mappo"]),
        "optimum_rank": _named_rank(scores, optimum),
        "assignment_vector": state["vectors"][chosen],
        "value_spearman": _spearman(teacher, scores),
        "value_kendall": _kendall(teacher, scores),
        "value_pairwise": pairwise_accuracy(teacher, scores, VALUE_TIE),
        "sinr_spearman": _spearman(true_sinr, pred_sinr),
        "sinr_pairwise": pairwise_accuracy(true_sinr, pred_sinr, 1.0e-8),
        "rate_spearman": _spearman(true_rate, pred_rate),
        "rate_pairwise": pairwise_accuracy(true_rate, pred_rate, 1.0e-8),
        "pred_sinr_rate_spearman": _spearman(pred_sinr, pred_rate),
        "score_std": float(np.std(scores)),
    }


def _train_one(seed: int, states: list[dict], input_norm, response_norm, scorer) -> list[dict]:
    seed_everything(int(seed))
    model = OneStepExecutorResponsePredictor(token_dim())
    if _count(model) != _count(_load_predictor(int(seed))):
        raise RuntimeError("predictor parameter count drifted")
    before = _checksum(scorer)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    binary_index = list(BINARY_INDEX)
    for epoch in range(1, int(EPOCHS) + 1):
        model.train()
        totals, responses, ranks, grads = [], [], [], []
        for state in states:
            scenario = int(state["scenario_id"])
            assert_consumed(scenario)
            if scenario not in TRAIN_SCENARIOS:
                raise RuntimeError("confirmation state entered the response gradient")
            optimizer.zero_grad(set_to_none=True)
            tokens = torch.as_tensor(_apply_affine(state["tokens"], input_norm), dtype=torch.float32)
            prediction = model(tokens)
            raw = state["response_raw"]
            normalized = torch.as_tensor(_apply_affine(raw, response_norm), dtype=torch.float32)
            binary = torch.as_tensor(raw[..., binary_index], dtype=torch.float32)
            regression, _huber, _bce = response_loss(prediction, normalized, binary)
            student = _score_tensor(scorer, state, _value_tensor(prediction, response_norm), state["relational"])
            left, right = state["pairs"][int(seed)]
            ranking = pairwise_rank_loss(
                student,
                torch.as_tensor(left, dtype=torch.long),
                torch.as_tensor(right, dtype=torch.long),
                VALUE_TEMPERATURE,
            )
            loss = regression + float(LAMBDA_RANK) * ranking
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"ranking loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            leaked = 0.0
            for parameter in scorer.parameters():
                if parameter.grad is not None:
                    leaked += float(parameter.grad.detach().abs().sum())
            if leaked != 0.0:
                raise RuntimeError("frozen Model-R1 received a gradient")
            grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)))
            optimizer.step()
            totals.append(float(loss.detach()))
            responses.append(float(regression.detach()))
            ranks.append(float(ranking.detach()))
        if _checksum(scorer) != before:
            raise RuntimeError("frozen Model-R1 weights changed")
        model.eval()
        predicted, truth = [], []
        with torch.no_grad():
            for state in states:
                head = model(torch.as_tensor(_apply_affine(state["tokens"], input_norm), dtype=torch.float32)).detach().cpu().numpy()
                predicted.append(_to_raw(head, response_norm))
                truth.append(state["response_raw"])
        block = _feature_block(np.concatenate(predicted), np.concatenate(truth))
        row = {
            "seed": int(seed),
            "epoch": int(epoch),
            "total_loss": mean(totals),
            "response_loss": mean(responses),
            "rank_loss": mean(ranks),
            "grad_norm": mean(grads),
        }
        for item in block:
            row[f"mae_{item['feature']}"] = item["mae"]
        curves.append(row)
        print(
            f"zh seed {seed} epoch {epoch} loss={row['total_loss']:.4f} resp={row['response_loss']:.4f} rank={row['rank_loss']:.4f}",
            flush=True,
        )
    model.eval()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": int(seed),
            "epoch": int(EPOCHS),
            "model": "response_predictor_rank",
            "lambda_rank": float(LAMBDA_RANK),
            "value_temperature": float(VALUE_TEMPERATURE),
            "input_dim": token_dim(),
            "hidden": int(HIDDEN),
            "state_dict": model.state_dict(),
            "input_mean": input_norm["mean"],
            "input_std": input_norm["std"],
        },
        CHECKPOINTS / f"phase5zh_rank_seed{int(seed)}.pt",
    )
    return curves


def _load_zh(seed: int) -> OneStepExecutorResponsePredictor:
    payload = torch.load(CHECKPOINTS / f"phase5zh_rank_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != int(EPOCHS):
        raise RuntimeError("ranking predictor checkpoint is not the final epoch")
    model = OneStepExecutorResponsePredictor(int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _residual_keys() -> list[dict]:
    path = OUTPUT.parents[0] / "phase5zd_executor_response_value" / "phase5zd_residual_cases.csv"
    rows, seen = [], set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            key = (scenario, int(row["time_step"]), row["predicted_assignment"], row["optimal_assignment"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "predicted_assignment": key[2],
                    "optimal_assignment": key[3],
                }
            )
    return rows


def _protocol() -> dict:
    return {
        "phase": "5ZH",
        "loss": "L_resp + 0.2 * L_rank",
        "lambda_rank": LAMBDA_RANK,
        "value_temperature": VALUE_TEMPERATURE,
        "value_temperature_unit": "one payload block in Model-R1 output units",
        "payload_block_bits": PAYLOAD_BLOCK_BITS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "hidden": HIDDEN,
        "optimizer": "Adam",
        "batch": "one state",
        "checkpoint": "final epoch only",
        "train_scenarios": list(TRAIN_SCENARIOS),
        "confirm_scenarios_not_for_selection": list(CONFIRM_SCENARIOS),
        "opened_310118_310125": False,
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
        "configuration_difference": "ranking loss only; network, data, optimizer, and normalization stay at Phase-5ZE",
    }


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5zh_protocol.json").write_text(json.dumps(plain(_protocol()), indent=2), encoding="utf-8")
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    frozen.eval()
    matching.eval()
    actor = load_frozen_actor()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.eval()
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    center = np.asarray(temporal["center"], dtype=np.float32)
    scale = np.asarray(temporal["scale"], dtype=np.float32)
    response_norm = _load_response_norm()
    train_states, train_audit = _prepare(
        TRAIN_SCENARIOS,
        _labels("train"),
        _horizon_table(TRAIN_SCENARIOS, 1),
        frozen,
        matching,
        actor,
        config,
        False,
    )
    saved_norm = torch.load(
        OUTPUT.parents[0] / "phase5ze_response_distillation" / "checkpoints" / "phase5ze_response_seed0.pt",
        map_location="cpu",
        weights_only=False,
    )
    refit = _fit_input(train_states)
    token_gap = max(
        float(np.max(np.abs(refit["mean"] - np.asarray(saved_norm["input_mean"])))),
        float(np.max(np.abs(refit["std"] - np.asarray(saved_norm["input_std"])))),
    )
    response_gap = float(np.max(np.abs(train_audit["response_mean"] - response_norm["mean"])))
    seed_norm_gap = 0.0
    norm_dir = OUTPUT.parents[0] / "phase5ze_response_distillation" / "checkpoints"
    for seed in SEEDS:
        other = torch.load(norm_dir / f"phase5ze_response_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
        seed_norm_gap = max(
            seed_norm_gap,
            float(np.max(np.abs(np.asarray(other["input_mean"]) - np.asarray(saved_norm["input_mean"])))),
            float(np.max(np.abs(np.asarray(other["input_std"]) - np.asarray(saved_norm["input_std"])))),
        )
    lineage_ok = bool(
        float(train_audit["largest"]) == 0.0
        and float(train_audit["parity_gap"]) <= LABEL_MATCH_ABS
        and float(train_audit["edge_gap"]) <= EDGE_MATCH_ABS
        and float(train_audit["sinr_gap"]) <= SINR_MATCH_ABS
        and float(train_audit["progress_gap"]) <= PROGRESS_MATCH_ABS
        and float(train_audit["payload_gap"]) <= PAYLOAD_MATCH_ABS
        and response_gap <= LABEL_MATCH_ABS
        and token_gap <= TOKEN_NORM_ABS
        and seed_norm_gap <= TOKEN_NORM_ABS
        and all(int(state["scenario_id"]) in TRAIN_SCENARIOS for state in train_states)
    )
    input_norm = {"mean": np.asarray(saved_norm["input_mean"], dtype=np.float32), "std": np.asarray(saved_norm["input_std"], dtype=np.float32)}
    lineage = {
        "ok": lineage_ok,
        "train_scenarios": list(TRAIN_SCENARIOS),
        "label_source": "phase5o and phase5r train labels, same as phase5ze",
        "response_normalization": "phase5zd_response_normalization.json",
        "input_normalization": "phase5ze checkpoint seed 0",
        "token_norm_gap": token_gap,
        "seed_norm_gap": seed_norm_gap,
        "response_mean_gap": response_gap,
        "caller_integrity": float(train_audit["largest"]),
        "parity_gap": float(train_audit["parity_gap"]),
        "confirm_used_for_checkpoint_selection": False,
        "opened_310118_310125": False,
    }
    (OUTPUT / "phase5zh_data_lineage.json").write_text(json.dumps(plain(lineage), indent=2), encoding="utf-8")
    if not lineage_ok:
        _gate({"gate": "RANK-D", "aligned": False, "lineage": lineage})
        return
    scorers = {int(seed): _load_value("response", int(seed)) for seed in SEEDS}
    statics = {int(seed): _load_value("static", int(seed)) for seed in SEEDS}
    for state in train_states:
        relational = _relational(state, center, scale)
        state["relational"] = relational
        true_response = _apply_affine(state["response_raw"], response_norm)
        state["teacher"] = {}
        state["pairs"] = {}
        for seed in SEEDS:
            teacher, _elapsed = _predict_value(scorers[int(seed)], state, true_response, relational)
            state["teacher"][int(seed)] = teacher
            rng = np.random.default_rng(PAIR_RNG_SEED + int(seed) * 10007 + int(state["scenario_id"]) + int(state["time_step"]))
            left, right = ranking_pairs(teacher, rng, VALUE_TIE, RANDOM_PAIRS)
            state["pairs"][int(seed)] = (left, right)
    curves = []
    for seed in SEEDS:
        curves.extend(_train_one(int(seed), train_states, input_norm, response_norm, scorers[int(seed)]))
    write_csv(OUTPUT / "phase5zh_training_curves.csv", curves)
    if any(not (CHECKPOINTS / f"phase5zh_rank_seed{int(seed)}.pt").exists() for seed in SEEDS):
        raise RuntimeError("final-epoch checkpoints are missing")
    confirm_states, _confirm_audit = _prepare(
        CONFIRM_SCENARIOS,
        _labels("confirm"),
        _horizon_table(CONFIRM_SCENARIOS, 1),
        frozen,
        matching,
        actor,
        config,
        False,
    )
    baselines = {int(seed): _load_predictor(int(seed)) for seed in SEEDS}
    proposed = {int(seed): _load_zh(int(seed)) for seed in SEEDS}
    rows = []
    case_rows = []
    residual_rows = []
    align_s = 0.0
    align_p1 = 0.0
    align_r1 = 0.0
    saved_table = _load_saved()
    pred_bank = {"P1": [], "ZH": []}
    true_bank = []
    latency = []
    residuals = _residual_keys()
    zero = None
    for state in confirm_states:
        assert_consumed(int(state["scenario_id"]))
        if int(state["scenario_id"]) not in CONFIRM_SCENARIOS:
            raise RuntimeError("fresh scenario entered evaluation")
        relational = _relational(state, center, scale)
        true_response = _apply_affine(state["response_raw"], response_norm)
        if zero is None:
            zero = _apply_affine(np.zeros_like(state["response_raw"]), response_norm)
        print(f"confirm {state['scenario_id']} t={state['time_step']}", flush=True)
        for seed in SEEDS:
            with torch.no_grad():
                head_p1 = baselines[int(seed)](torch.as_tensor(_apply_affine(state["tokens"], input_norm), dtype=torch.float32)).cpu().numpy()
                head_zh = proposed[int(seed)](torch.as_tensor(_apply_affine(state["tokens"], input_norm), dtype=torch.float32)).cpu().numpy()
            raw_p1 = _to_raw(head_p1, response_norm)
            raw_zh = _to_raw(head_zh, response_norm)
            pred_bank["P1"].append(raw_p1)
            pred_bank["ZH"].append(raw_zh)
            true_bank.append(state["response_raw"])
            teacher, _elapsed = _predict_value(scorers[int(seed)], state, true_response, relational)
            scores = {
                "S": _predict_value(statics[int(seed)], state, zero, relational)[0],
                "P1": _predict_value(scorers[int(seed)], state, _to_value(head_p1, response_norm), relational)[0],
                "ZH": _predict_value(scorers[int(seed)], state, _to_value(head_zh, response_norm), relational)[0],
                "R1": teacher,
            }
            state["pred_raw"] = {"P1": raw_p1, "ZH": raw_zh, "S": state["response_raw"], "R1": state["response_raw"]}
            r1_choice = state["vectors"][int(np.argsort(-scores["R1"], kind="stable")[0])]
            for model in MODELS:
                row = _decision(model, int(seed), state, scores[model], teacher)
                row["agreement_with_r1"] = float(row["assignment_vector"] == r1_choice)
                rows.append(row)
            saved = _saved_values(saved_table, int(state["scenario_id"]), int(state["time_step"]), int(seed), state["vectors"])
            align_s = max(align_s, float(np.max(np.abs(scores["S"] - saved[0]))))
            align_p1 = max(align_p1, float(np.max(np.abs(scores["P1"] - saved[1]))))
            align_r1 = max(align_r1, float(np.max(np.abs(scores["R1"] - saved[2]))))
            if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME:
                case_rows.extend(_case_rows(int(seed), state, scores, raw_p1, raw_zh))
            for item in residuals:
                if item["scenario_id"] == int(state["scenario_id"]) and item["time_step"] == int(state["time_step"]):
                    residual_rows.append(_residual_row(item, int(seed), state, scores))
        if int(state["scenario_id"]) == CONFIRM_SCENARIOS[0] and int(state["time_step"]) == TIMES[0]:
            latency.extend(_latency(state, input_norm, response_norm, baselines[0], proposed[0], scorers[0], zero, statics[0], center, scale))
    aligned = bool(align_s <= ALIGN_ABS and align_p1 <= ALIGN_ABS and align_r1 <= ALIGN_ABS)
    summary = _finish(rows, pred_bank, true_bank, response_norm, aligned, align_s, align_p1, align_r1)
    write_csv(OUTPUT / "phase5zh_response_regression.csv", regression_rows(pred_bank, true_bank))
    write_csv(OUTPUT / "phase5zh_statewise_ranking.csv", rows)
    write_csv(OUTPUT / "phase5zh_selected_j15.csv", summary["selected_rows"])
    write_csv(OUTPUT / "phase5zh_scenario_seed_statistics.csv", summary["scenario_rows"])
    write_csv(OUTPUT / "phase5zh_310114_t15.csv", case_rows)
    residual_table, residual_counts = _residual_summary(residual_rows)
    write_csv(OUTPUT / "phase5zh_residual_cases.csv", residual_table)
    write_csv(OUTPUT / "phase5zh_latency.csv", latency)
    summary["gate"]["residual_counts"] = residual_counts
    summary["gate"]["case_310114"] = _case_summary(case_rows)
    _gate(summary["gate"])


def _load_saved() -> dict[tuple[int, int, int, str], tuple[float, float, float]]:
    path = OUTPUT.parents[0] / "phase5ze_response_distillation" / "phase5ze_downstream_value_predictions.csv"
    table = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            table[(int(row["scenario_id"]), int(row["time_step"]), int(row["seed"]), row["assignment_vector"])] = (
                float(row["v_s"]),
                float(row["v_p1"]),
                float(row["v_r1"]),
            )
    return table


def _saved_values(table, scenario: int, moment: int, seed: int, vectors: list[str]):
    chosen = [table[(int(scenario), int(moment), int(seed), vector)] for vector in vectors]
    return (
        np.asarray([item[0] for item in chosen], dtype=np.float64),
        np.asarray([item[1] for item in chosen], dtype=np.float64),
        np.asarray([item[2] for item in chosen], dtype=np.float64),
    )


def _case_rows(seed: int, state: dict, scores: dict, raw_p1: np.ndarray, raw_zh: np.ndarray) -> list[dict]:
    pred_index = state["vectors"].index(CASE_PREDICTED)
    opt_index = state["vectors"].index(CASE_OPTIMAL)
    rows = []
    for model, table in scores.items():
        rows.append(
            {
                "row_type": "rank",
                "seed": int(seed),
                "model": model,
                "A_opt_rank": _named_rank(table, opt_index),
                "A_pred_rank": _named_rank(table, pred_index),
                "margin_opt_minus_pred": float(table[opt_index] - table[pred_index]),
                "reversal": float(table[opt_index] < table[pred_index]),
            }
        )
    for label, index, raw in (("P1", pred_index, raw_p1), ("P1", opt_index, raw_p1), ("ZH", pred_index, raw_zh), ("ZH", opt_index, raw_zh)):
        name = "A_pred" if index == pred_index else "A_opt"
        rows.append(
            {
                "row_type": "channel",
                "seed": int(seed),
                "model": label,
                "assignment": name,
                "true_mean_sinr": float(_assigned_mean(state["response_raw"], 0, state["assignments"])[index]),
                "pred_mean_sinr": float(_assigned_mean(raw, 0, state["assignments"])[index]),
                "true_mean_rate": float(_assigned_mean(state["response_raw"], 6, state["assignments"])[index]),
                "pred_mean_rate": float(_assigned_mean(raw, 6, state["assignments"])[index]),
            }
        )
    return rows


def _residual_row(item: dict, seed: int, state: dict, scores: dict) -> dict:
    labels = state["j_mappo"]
    rows = {
        "scenario_id": item["scenario_id"],
        "time_step": item["time_step"],
        "seed": int(seed),
        "predicted_assignment": item["predicted_assignment"],
        "optimal_assignment": item["optimal_assignment"],
    }
    for model in ("P1", "ZH"):
        chosen = int(np.argsort(-scores[model], kind="stable")[0])
        measured = metrics_at(scores[model], labels, chosen)
        rows[f"selected_j15_{model}"] = measured["j_mappo"]
        rows[f"regret_{model}"] = measured["normalized_regret"]
        rows[f"optimum_rank_{model}"] = _named_rank(scores[model], int(np.argmax(quantize(labels))))
    opt_index = state["vectors"].index(item["optimal_assignment"])
    pred_index = state["vectors"].index(item["predicted_assignment"])
    rows["value_margin_error_p1"] = float((scores["P1"][opt_index] - scores["P1"][pred_index]) - (scores["R1"][opt_index] - scores["R1"][pred_index]))
    rows["value_margin_error_zh"] = float((scores["ZH"][opt_index] - scores["ZH"][pred_index]) - (scores["R1"][opt_index] - scores["R1"][pred_index]))
    true_sinr = _assigned_mean(state["response_raw"], 0, state["assignments"])
    pred_sinr_p1 = _assigned_mean(state["pred_raw"]["P1"], 0, state["assignments"])
    pred_sinr_zh = _assigned_mean(state["pred_raw"]["ZH"], 0, state["assignments"])
    pred_rate_p1 = _assigned_mean(state["pred_raw"]["P1"], 6, state["assignments"])
    pred_rate_zh = _assigned_mean(state["pred_raw"]["ZH"], 6, state["assignments"])
    true_rate = _assigned_mean(state["response_raw"], 6, state["assignments"])
    rows["sinr_pairwise_p1"] = pairwise_accuracy(true_sinr, pred_sinr_p1, 1.0e-8)
    rows["sinr_pairwise_zh"] = pairwise_accuracy(true_sinr, pred_sinr_zh, 1.0e-8)
    rows["rate_pairwise_p1"] = pairwise_accuracy(true_rate, pred_rate_p1, 1.0e-8)
    rows["rate_pairwise_zh"] = pairwise_accuracy(true_rate, pred_rate_zh, 1.0e-8)
    rows["response_ordering_error_p1"] = float(1.0 - 0.5 * (rows["sinr_pairwise_p1"] + rows["rate_pairwise_p1"]))
    rows["response_ordering_error_zh"] = float(1.0 - 0.5 * (rows["sinr_pairwise_zh"] + rows["rate_pairwise_zh"]))
    return rows


def _residual_summary(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario_id"], row["time_step"], row["predicted_assignment"])].append(row)
    output = []
    counts = {"repaired": 0, "unchanged": 0, "worsened": 0}
    for key, chosen in grouped.items():
        p1 = mean(row["selected_j15_P1"] for row in chosen)
        zh = mean(row["selected_j15_ZH"] for row in chosen)
        if zh > p1 + TIE_BITS:
            direction = "repaired"
        elif zh + TIE_BITS < p1:
            direction = "worsened"
        else:
            direction = "unchanged"
        counts[direction] += 1
        output.append(
            {
                "row_type": "case",
                "scenario_id": key[0],
                "time_step": key[1],
                "predicted_assignment": key[2],
                "optimal_assignment": chosen[0].get("optimal_assignment", ""),
                "case_direction": direction,
                "selected_j15_P1": p1,
                "selected_j15_ZH": zh,
                "regret_P1": mean(row["regret_P1"] for row in chosen),
                "regret_ZH": mean(row["regret_ZH"] for row in chosen),
                "optimum_rank_P1": mean(row["optimum_rank_P1"] for row in chosen),
                "optimum_rank_ZH": mean(row["optimum_rank_ZH"] for row in chosen),
                "response_ordering_error_P1": mean(row["response_ordering_error_p1"] for row in chosen),
                "response_ordering_error_ZH": mean(row["response_ordering_error_zh"] for row in chosen),
                "value_margin_error_P1": mean(row["value_margin_error_p1"] for row in chosen),
                "value_margin_error_ZH": mean(row["value_margin_error_zh"] for row in chosen),
            }
        )
        for row in chosen:
            output.append({**row, "row_type": "seed", "case_direction": direction})
    return output, counts


def _latency(state, input_norm, response_norm, baseline, proposed, scorer, zero, static, center, scale) -> list[dict]:
    relational = _relational(state, center, scale)
    rows = []
    for name, model in (("P1", baseline), ("ZH", proposed)):
        for _repeat in range(LATENCY_WARMUP):
            with torch.no_grad():
                head = model(torch.as_tensor(_apply_affine(state["tokens"], input_norm), dtype=torch.float32)).cpu().numpy()
                _predict_value(scorer, state, _to_value(head, response_norm), relational)
        samples = []
        for _repeat in range(LATENCY_REPEATS):
            started = time.perf_counter()
            with torch.no_grad():
                head = model(torch.as_tensor(_apply_affine(state["tokens"], input_norm), dtype=torch.float32)).cpu().numpy()
                _predict_value(scorer, state, _to_value(head, response_norm), relational)
            samples.append((time.perf_counter() - started) * 1000.0)
        rows.append({"model": name, "mean_ms": mean(samples), "repeats": LATENCY_REPEATS})
    samples = []
    for _repeat in range(LATENCY_REPEATS):
        started = time.perf_counter()
        _predict_value(static, state, zero, relational)
        samples.append((time.perf_counter() - started) * 1000.0)
    rows.append({"model": "S", "mean_ms": mean(samples), "repeats": LATENCY_REPEATS})
    return rows


def _case_summary(case_rows: list[dict]) -> dict:
    summary = {}
    for model in MODELS:
        ranks = [int(row["A_opt_rank"]) for row in case_rows if row.get("row_type") == "rank" and row["model"] == model]
        reversals = [float(row["reversal"]) for row in case_rows if row.get("row_type") == "rank" and row["model"] == model]
        summary[model] = {
            "a_opt_ranks": ranks,
            "a_opt_median": float(np.median(ranks)) if ranks else float("nan"),
            "top3_seeds": int(sum(rank <= 3 for rank in ranks)),
            "reversal_seeds": int(sum(reversals)),
        }
    return summary


def regression_rows(pred_bank, true_bank) -> list[dict]:
    truth = np.concatenate(true_bank)
    rows = []
    for model, chunks in pred_bank.items():
        block = _feature_block(np.concatenate(chunks), truth)
        for item in block:
            rows.append({"model": model, "feature": item["feature"], "mae": item["mae"], "rmse": item["rmse"], "r2": item["r2"]})
    return rows


def _selected(rows: list[dict], model: str, **filters) -> list[float]:
    return [
        float(row["selected_j15"])
        for row in rows
        if row["model"] == model and all(row[key] == value for key, value in filters.items())
    ]


def _finish(rows, pred_bank, true_bank, response_norm, aligned, align_s, align_p1, align_r1) -> dict:
    selected_rows = []
    j_s = mean(_selected(rows, "S"))
    j_p1 = mean(_selected(rows, "P1"))
    j_zh = mean(_selected(rows, "ZH"))
    j_r1 = mean(_selected(rows, "R1"))
    gap = j_r1 - j_s
    retention = (j_zh - j_s) / gap if gap > 0.0 else float("nan")
    for model in MODELS:
        chosen = [row for row in rows if row["model"] == model]
        values = [float(row["selected_j15"]) for row in chosen]
        selected_rows.append(
            {
                "model": model,
                "selected_j15_mean": mean(values),
                "selected_j15_median": float(np.median(values)),
                "delta_vs_p1": mean(values) - j_p1,
                "delta_vs_s": mean(values) - j_s,
                "retention_vs_s": (mean(values) - j_s) / gap if gap > 0.0 else float("nan"),
                "normalized_regret": mean(row["normalized_regret"] for row in chosen),
                "top1": mean(row["top1"] for row in chosen),
                "top3": mean(row["top3"] for row in chosen),
                "top5": mean(row["top5"] for row in chosen),
                "agreement_with_r1": mean(row["agreement_with_r1"] for row in chosen),
                "value_spearman": mean(row["value_spearman"] for row in chosen),
                "value_kendall": mean(row["value_kendall"] for row in chosen),
                "value_pairwise": mean(row["value_pairwise"] for row in chosen),
                "sinr_spearman": mean(row["sinr_spearman"] for row in chosen),
                "sinr_pairwise": mean(row["sinr_pairwise"] for row in chosen),
                "rate_spearman": mean(row["rate_spearman"] for row in chosen),
                "rate_pairwise": mean(row["rate_pairwise"] for row in chosen),
                "pred_sinr_rate_spearman": mean(row["pred_sinr_rate_spearman"] for row in chosen),
            }
        )
    pairs = []
    for row in rows:
        if row["model"] != "ZH":
            continue
        base = next(
            item["selected_j15"]
            for item in rows
            if item["model"] == "P1" and item["scenario_id"] == row["scenario_id"] and item["time_step"] == row["time_step"] and item["seed"] == row["seed"]
        )
        pairs.append({"scenario_id": int(row["scenario_id"]), "delta": float(row["selected_j15"]) - float(base)})
    interval = cluster_bootstrap(pairs, "delta", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    seed_up = 0
    scenario_rows = []
    for seed in SEEDS:
        delta = mean(_selected(rows, "ZH", seed=int(seed))) - mean(_selected(rows, "P1", seed=int(seed)))
        seed_up += int(delta > TIE_BITS)
        scenario_rows.append({"unit": "seed", "unit_id": int(seed), "delta_vs_p1": delta, "j15_zh": mean(_selected(rows, "ZH", seed=int(seed))), "j15_p1": mean(_selected(rows, "P1", seed=int(seed)))})
    for scenario in CONFIRM_SCENARIOS:
        delta = mean(_selected(rows, "ZH", scenario_id=int(scenario))) - mean(_selected(rows, "P1", scenario_id=int(scenario)))
        direction = "tie" if abs(delta) <= TIE_BITS else ("positive" if delta > 0.0 else "negative")
        scenario_rows.append({"unit": "scenario", "unit_id": int(scenario), "delta_vs_p1": delta, "direction": direction})
    p1_rank = next(row for row in selected_rows if row["model"] == "P1")
    zh_rank = next(row for row in selected_rows if row["model"] == "ZH")
    ranking_ok = bool(zh_rank["value_spearman"] > p1_rank["value_spearman"] and zh_rank["value_pairwise"] > p1_rank["value_pairwise"])
    ranking_up = bool(zh_rank["value_spearman"] > p1_rank["value_spearman"] or zh_rank["value_pairwise"] > p1_rank["value_pairwise"])
    physical = _physical(pred_bank, true_bank, response_norm, rows)
    letter = choose_gate(
        aligned=aligned,
        j15_mean_up=bool(j_zh - j_p1 > 0.0),
        ci_low_positive=bool(interval["low"] > 0.0),
        seeds_ok=bool(seed_up >= MIN_SEEDS_UP),
        retention_ok=bool(np.isfinite(retention) and retention >= RETENTION_BAR),
        ranking_ok=ranking_ok,
        physical_ok=bool(physical["ok"]),
        ranking_up=ranking_up,
        j15_degraded=bool(interval["high"] < 0.0),
    )
    gate = {
        "gate": letter,
        "aligned": aligned,
        "align_abs_s": align_s,
        "align_abs_p1": align_p1,
        "align_abs_r1": align_r1,
        "delta_vs_p1": j_zh - j_p1,
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "seeds_up": seed_up,
        "retention": retention,
        "value_spearman_p1": p1_rank["value_spearman"],
        "value_spearman_zh": zh_rank["value_spearman"],
        "value_kendall_p1": p1_rank["value_kendall"],
        "value_kendall_zh": zh_rank["value_kendall"],
        "value_pairwise_p1": p1_rank["value_pairwise"],
        "value_pairwise_zh": zh_rank["value_pairwise"],
        "sinr_spearman_p1": p1_rank["sinr_spearman"],
        "sinr_spearman_zh": zh_rank["sinr_spearman"],
        "sinr_pairwise_p1": p1_rank["sinr_pairwise"],
        "sinr_pairwise_zh": zh_rank["sinr_pairwise"],
        "rate_spearman_p1": p1_rank["rate_spearman"],
        "rate_spearman_zh": zh_rank["rate_spearman"],
        "rate_pairwise_p1": p1_rank["rate_pairwise"],
        "rate_pairwise_zh": zh_rank["rate_pairwise"],
        "physical": physical,
        "selected_j15": {row["model"]: row["selected_j15_mean"] for row in selected_rows},
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
        "opened_310118_310125": False,
        "decide_called": False,
        "checkpoint_selection_on_confirm": False,
    }
    return {"selected_rows": selected_rows, "scenario_rows": scenario_rows, "gate": gate}


def _physical(pred_bank, true_bank, response_norm, rows) -> dict:
    truth = np.concatenate(true_bank)
    report = {"ok": True}
    for model in ("P1", "ZH"):
        predicted = np.concatenate(pred_bank[model])
        z = (predicted - response_norm["mean"]) / response_norm["std"]
        continuous = list(CONTINUOUS_INDEX)
        extreme = float(np.mean(np.abs(z[..., continuous]) > Z_EXTREME))
        stds = [float(np.std(predicted[..., index])) for index in range(predicted.shape[-1])]
        report[f"extreme_fraction_{model}"] = extreme
        report[f"min_channel_std_{model}"] = float(np.min(stds))
        if model == "ZH" and (extreme > EXTREME_FRACTION or float(np.min(stds)) < CHANNEL_STD_MIN):
            report["ok"] = False
    p1_block = {item["feature"]: item["mae"] for item in _feature_block(np.concatenate(pred_bank["P1"]), truth)}
    zh_block = {item["feature"]: item["mae"] for item in _feature_block(np.concatenate(pred_bank["ZH"]), truth)}
    for feature in ("sinr_1", "rate_1"):
        ratio = zh_block[feature] / p1_block[feature] if p1_block[feature] > 0.0 else float("nan")
        report[f"mae_{feature}_p1"] = p1_block[feature]
        report[f"mae_{feature}_zh"] = zh_block[feature]
        report[f"mae_ratio_{feature}"] = ratio
        if not np.isfinite(ratio) or ratio > MAE_RATIO_MAX:
            report["ok"] = False
    report["mae_mean_p1"] = mean(p1_block.values())
    report["mae_mean_zh"] = mean(zh_block.values())
    relation = mean(row["pred_sinr_rate_spearman"] for row in rows if row["model"] == "ZH")
    report["predicted_sinr_rate_spearman"] = relation
    report["predicted_sinr_rate_spearman_p1"] = mean(row["pred_sinr_rate_spearman"] for row in rows if row["model"] == "P1")
    if not np.isfinite(relation) or relation < SINR_RATE_SPEARMAN_MIN:
        report["ok"] = False
    collapsed = mean(float(row["score_std"] < SCORE_COLLAPSE_STD) for row in rows if row["model"] == "ZH")
    report["score_collapse_fraction"] = collapsed
    if collapsed > 0.5:
        report["ok"] = False
    return report


def _gate(payload: dict) -> None:
    (OUTPUT / "phase5zh_ranking_distillation_gate.json").write_text(json.dumps(plain(payload), indent=2), encoding="utf-8")
    print(json.dumps(plain(payload), indent=2), flush=True)


if __name__ == "__main__":
    run()
