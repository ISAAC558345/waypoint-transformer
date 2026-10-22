"""Inference-only value sensitivity audit. No optimizer step and no new checkpoint."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _combine, _scale
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import OUTPUT as PHASE5ZD_OUTPUT
from wt.lowlevel_v2.phase5ze_response_distillation.protocol import OUTPUT as PHASE5ZE_OUTPUT
from wt.lowlevel_v2.phase5zf_distillation_error_audit.common import load_tables, mean, normalize, plain, write_csv
from wt.lowlevel_v2.phase5zf_distillation_error_audit.metrics import pairwise_accuracy
from wt.lowlevel_v2.phase5zf_distillation_error_audit.run_phase5zf import _load_norm, _load_scorer, _prepare, _stack

from .metrics import choose_gate, group_dot, sign_agreement, trapezoid, unique_dominant
from .protocol import (
    ALIGN_ABS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMAL,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CONFIRM_SCENARIOS,
    DEPLOYMENT_VERIFIED,
    EXPLAIN_SIGN,
    EXPLAIN_SPEARMAN,
    FINITE_FRACTION,
    FROZEN_FOR_FRESH_CONFIRMATION,
    GROUPS,
    INTERACTION_RATIO,
    MARGIN_SIGN,
    ORDERING_PAIRWISE_MAX,
    OUTPUT,
    PATH_POINTS,
    PLANNER_READY,
    PRIMARY_FEATURE,
    RESPONSE_FEATURES,
    SEEDS,
    STABLE_SCENARIOS,
    STABLE_SEEDS,
    TIE_BITS,
    TIMES,
    UNIQUE_RELATIVE_GAP,
    VALUE_EPS,
    assert_consumed,
)

NEXT_STEP = {
    "SENS-A": "sensitivity-aware response loss",
    "SENS-B": "joint response prediction objective",
    "SENS-C": "state-conditional ranking-aware response loss",
    "SENS-D": "stop and repair alignment before any sensitivity claim",
    "SENS-E": "stop model changes; the mechanism is not established",
}


def _pearson(left, right) -> float:
    observed = np.asarray(left, dtype=np.float64).reshape(-1)
    predicted = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 3 or float(np.std(observed[valid])) <= 1.0e-12 or float(np.std(predicted[valid])) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(observed[valid], predicted[valid])[0, 1])


def _spearman(left, right) -> float:
    observed = np.asarray(left, dtype=np.float64).reshape(-1)
    predicted = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 3 or float(np.std(observed[valid])) <= 1.0e-12 or float(np.std(predicted[valid])) <= 1.0e-12:
        return float("nan")
    return float(spearmanr(observed[valid], predicted[valid]).statistic)


def _assigned_mean(raw: np.ndarray, feature: int, assignments: np.ndarray) -> np.ndarray:
    mask = assignments >= 0
    held = np.where(mask, raw[:, :, feature], 0.0)
    count = mask.sum(axis=1)
    return np.divide(held.sum(axis=1), count, out=np.full(count.shape, np.nan), where=count > 0)


def _splice(base: np.ndarray, donor: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    mixed = np.array(base, copy=True)
    for name in names:
        mixed[..., list(GROUPS[name])] = donor[..., list(GROUPS[name])]
    return mixed


def _base_features(embedded: dict, relational: np.ndarray) -> dict:
    features = {}
    for key, value in embedded.items():
        features[key] = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    features["z_rel"] = torch.as_tensor(relational, dtype=torch.float32)
    return features


def _forward(model, features: dict, assignments: torch.Tensor, response: np.ndarray, need_grad: bool):
    packed = dict(features)
    tensor = torch.as_tensor(np.asarray(response, dtype=np.float32))
    if need_grad:
        tensor.requires_grad_(True)
    packed["response"] = tensor
    values = model(packed, assignments)
    if not need_grad:
        return values.detach().cpu().numpy(), None
    gradient = torch.autograd.grad(values.sum(), tensor)[0]
    return values.detach().cpu().numpy(), gradient.detach().cpu().numpy().astype(np.float64)


def _grad_rows(gradient: np.ndarray, std: np.ndarray, assignments: np.ndarray, point: str, scenario: int, moment: int, seed: int) -> list[dict]:
    mask = assignments >= 0
    raw_gradient = gradient / std.reshape(1, 1, -1)
    rows = []
    for name, indexes in GROUPS.items():
        chosen = list(indexes)
        normalized = gradient[..., chosen]
        physical = raw_gradient[..., chosen]
        components = normalized[mask]
        l2 = np.sqrt(np.sum(normalized**2, axis=(1, 2)))
        l2_raw = np.sqrt(np.sum(physical**2, axis=(1, 2)))
        rows.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(moment),
                "seed": int(seed),
                "point": point,
                "group": name,
                "mean_l1_normalized": float(np.mean(np.sum(np.abs(normalized), axis=(1, 2)))),
                "mean_l2_normalized": float(np.mean(l2)),
                "max_abs_normalized": float(np.max(np.abs(normalized))),
                "std_l2_across_assignments": float(np.std(l2)),
                "positive_fraction_selected": float(np.mean(components > VALUE_EPS)),
                "negative_fraction_selected": float(np.mean(components < -VALUE_EPS)),
                "mean_l1_raw": float(np.mean(np.sum(np.abs(physical), axis=(1, 2)))),
                "mean_l2_raw": float(np.mean(l2_raw)),
                "max_abs_raw": float(np.max(np.abs(physical))),
            }
        )
    return rows


def _chosen(scores: np.ndarray) -> int:
    return int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])


def _metric(model: str, seed: int, scenario: int, moment: int, scores: np.ndarray, labels: np.ndarray, vectors: list[str], r1_vector: str) -> dict:
    chosen = _chosen(scores)
    measured = metrics_at(scores, labels, chosen)
    return {
        "model": model,
        "seed": int(seed),
        "scenario_id": int(scenario),
        "time_step": int(moment),
        "selected_j15": measured["j_mappo"],
        "normalized_regret": measured["normalized_regret"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "assignment_vector": vectors[chosen],
        "agreement_with_r1": float(vectors[chosen] == r1_vector),
    }


def _margin_recovery(margin: float, baseline: float, target: float) -> float:
    gap = float(target) - float(baseline)
    if not np.isfinite(gap) or gap <= float(VALUE_EPS):
        return float("nan")
    return (float(margin) - float(baseline)) / gap


def _residual_keys() -> list[dict]:
    rows = []
    seen = set()
    path = PHASE5ZD_OUTPUT / "phase5zd_residual_cases.csv"
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


def _touch(name: str, fields: list[str]) -> None:
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(fields)


def _protocol() -> dict:
    return {
        "phase": "5ZG",
        "mode": "inference-only value sensitivity audit",
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
        "scenarios": list(CONFIRM_SCENARIOS),
        "groups": {name: list(indexes) for name, indexes in GROUPS.items()},
        "path_points": PATH_POINTS,
        "explain_sign": EXPLAIN_SIGN,
        "explain_spearman": EXPLAIN_SPEARMAN,
        "margin_sign": MARGIN_SIGN,
        "stable_scenarios": STABLE_SCENARIOS,
        "stable_seeds": STABLE_SEEDS,
        "interaction_ratio": INTERACTION_RATIO,
        "ordering_pairwise_max": ORDERING_PAIRWISE_MAX,
        "align_abs": ALIGN_ABS,
        "hybrid_responses_are_diagnostic_counterfactuals": True,
        "opened_310118_310125": False,
    }


def _gate_payload(letter: str, aligned: bool, extra: dict | None = None) -> dict:
    payload = {
        "gate": letter,
        "aligned": bool(aligned),
        "next_step": NEXT_STEP[letter],
        "optimizer_steps": 0,
        "new_checkpoint": False,
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
        "opened_310118_310125": False,
        "decide_called": False,
    }
    if extra:
        payload.update(extra)
    return payload


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5zg_protocol.json").write_text(json.dumps(plain(_protocol()), indent=2), encoding="utf-8")
    spec = json.loads((PHASE5ZD_OUTPUT / "phase5zd_response_feature_spec.json").read_text(encoding="utf-8"))
    if list(spec["features"]) != list(RESPONSE_FEATURES):
        _finish_unaligned({"feature_order_matches_spec": False})
        return
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
    for model in models.values():
        model.eval()
    residuals = _residual_keys()
    jacobian_rows: list[dict] = []
    projection_rows: list[dict] = []
    integrated_rows: list[dict] = []
    margin_rows: list[dict] = []
    sinr_rate_rows: list[dict] = []
    cross_rows: list[dict] = []
    case_rows: list[dict] = []
    residual_rows: list[dict] = []
    decision_rows: list[dict] = []
    metric_rows: list[dict] = []
    align_p1 = 0.0
    align_r1 = 0.0
    argmax_hits = []
    quadrature_gaps = []
    linear_add_gaps = []
    for scenario in CONFIRM_SCENARIOS:
        for moment in TIMES:
            assert_consumed(int(scenario))
            state = _prepare(int(scenario), int(moment), frozen, matching, config)
            relational = _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))
            features = _base_features(state["embedded"], relational)
            assignment_tensor = torch.as_tensor(state["assignments"], dtype=torch.long)
            print(f"state {scenario} t={moment}", flush=True)
            for seed in SEEDS:
                model = models[int(seed)]
                true, pred, labels, static, stored_p1, stored_r1 = _stack(
                    book, saved, int(scenario), int(moment), int(seed), state["vectors"]
                )
                true_n = normalize(true, norm)
                pred_n = normalize(pred, norm)
                error_n = pred_n.astype(np.float64) - true_n.astype(np.float64)
                error_raw = pred - true

                def differentiate(response: np.ndarray):
                    return _forward(model, features, assignment_tensor, response, True)

                def evaluate(raw: np.ndarray) -> np.ndarray:
                    values, _gradient = _forward(model, features, assignment_tensor, normalize(raw, norm), False)
                    return values

                values_true, grad_true = differentiate(true_n)
                values_pred, grad_pred = differentiate(pred_n)
                _values_mid, grad_mid = differentiate(0.5 * (true_n + pred_n))
                align_p1 = max(align_p1, float(np.max(np.abs(values_pred - stored_p1))))
                align_r1 = max(align_r1, float(np.max(np.abs(values_true - stored_r1))))
                argmax_hits.append(
                    float(
                        _chosen(values_pred) == _chosen(stored_p1) and _chosen(values_true) == _chosen(stored_r1)
                    )
                )
                jacobian_rows.extend(_grad_rows(grad_true, norm["std"], state["assignments"], "true", int(scenario), int(moment), int(seed)))
                jacobian_rows.extend(_grad_rows(grad_pred, norm["std"], state["assignments"], "predicted", int(scenario), int(moment), int(seed)))
                jacobian_rows.extend(_grad_rows(grad_mid, norm["std"], state["assignments"], "midpoint", int(scenario), int(moment), int(seed)))
                alphas = np.linspace(0.0, 1.0, PATH_POINTS)
                path_total = []
                path_groups = {name: [] for name in GROUPS}
                for index, alpha in enumerate(alphas):
                    if index == 0:
                        gradient = grad_true
                    elif index == PATH_POINTS - 1:
                        gradient = grad_pred
                    else:
                        _values, gradient = differentiate(true_n + float(alpha) * error_n)
                    path_total.append(np.sum(gradient * error_n, axis=(1, 2)))
                    for name, indexes in GROUPS.items():
                        path_groups[name].append(group_dot(gradient, error_n, indexes))
                integral = trapezoid(np.asarray(path_total))
                integral_groups = {name: trapezoid(np.asarray(samples)) for name, samples in path_groups.items()}
                linear_true = np.sum(grad_true * error_n, axis=(1, 2))
                linear_pred = np.sum(grad_pred * error_n, axis=(1, 2))
                linear_mid = np.sum(grad_mid * error_n, axis=(1, 2))
                linear_groups = {name: group_dot(grad_true, error_n, indexes) for name, indexes in GROUPS.items()}
                linear_add_gaps.append(float(np.max(np.abs(sum(linear_groups.values()) - linear_true))))
                actual = values_pred - values_true
                quadrature_gaps.append(float(np.max(np.abs(integral - actual))))
                isolated = {name: evaluate(_splice(true, pred, (name,))) - values_true for name in GROUPS}
                pair_interaction = {}
                for left, right in combinations(GROUPS, 2):
                    paired = evaluate(_splice(true, pred, (left, right)))
                    pair_interaction[(left, right)] = paired - (isolated[left] + values_true) - (isolated[right] + values_true) + values_true
                oracle = {
                    "P1": values_pred,
                    "R1": values_true,
                    "P1+TRUE_SINR": evaluate(_splice(pred, true, ("sinr",))),
                    "P1+TRUE_RATE": evaluate(_splice(pred, true, ("rate",))),
                    "P1+TRUE_SINR+TRUE_RATE": evaluate(_splice(pred, true, ("sinr", "rate"))),
                }
                index_r = _chosen(values_true)
                index_p = _chosen(values_pred)
                vector_r = state["vectors"][index_r]
                vector_p = state["vectors"][index_p]
                for model_name, scores in oracle.items():
                    metric_rows.append(_metric(model_name, int(seed), int(scenario), int(moment), scores, labels, state["vectors"], vector_r))
                true_sinr = _assigned_mean(true, PRIMARY_FEATURE["sinr"], state["assignments"])
                pred_sinr = _assigned_mean(pred, PRIMARY_FEATURE["sinr"], state["assignments"])
                true_rate = _assigned_mean(true, PRIMARY_FEATURE["rate"], state["assignments"])
                pred_rate = _assigned_mean(pred, PRIMARY_FEATURE["rate"], state["assignments"])
                selected = state["assignments"] >= 0
                margin_true = float(values_true[index_r] - values_true[index_p])
                margin_pred = float(values_pred[index_r] - values_pred[index_p])
                j15_gap = float(labels[index_r] - labels[index_p])
                if vector_r == vector_p:
                    klass = "SAME_ASSIGNMENT"
                elif abs(j15_gap) <= TIE_BITS:
                    klass = "SAME_J15"
                elif j15_gap > 0.0:
                    klass = "LOWER_J15"
                else:
                    klass = "HIGHER_J15"
                group_margin_linear = {name: float(linear_groups[name][index_r] - linear_groups[name][index_p]) for name in GROUPS}
                group_margin_integral = {name: float(integral_groups[name][index_r] - integral_groups[name][index_p]) for name in GROUPS}
                group_margin_isolated = {name: float(isolated[name][index_r] - isolated[name][index_p]) for name in GROUPS}
                sinr_rate_margin = float(pair_interaction[("sinr", "rate")][index_r] - pair_interaction[("sinr", "rate")][index_p])
                for assignment_index, vector in enumerate(state["vectors"]):
                    projection_rows.append(
                        {
                            "scenario_id": int(scenario),
                            "time_step": int(moment),
                            "seed": int(seed),
                            "assignment_vector": vector,
                            "delta_v_actual": float(actual[assignment_index]),
                            "delta_v_linear_true": float(linear_true[assignment_index]),
                            "delta_v_linear_predicted": float(linear_pred[assignment_index]),
                            "delta_v_linear_mid": float(linear_mid[assignment_index]),
                            "delta_v_integral": float(integral[assignment_index]),
                            **{f"linear_{name}": float(linear_groups[name][assignment_index]) for name in GROUPS},
                        }
                    )
                    integrated_rows.append(
                        {
                            "scenario_id": int(scenario),
                            "time_step": int(moment),
                            "seed": int(seed),
                            "assignment_vector": vector,
                            "delta_v_actual": float(actual[assignment_index]),
                            "delta_v_integral": float(integral[assignment_index]),
                            "quadrature_gap": float(integral[assignment_index] - actual[assignment_index]),
                            "isolated_remainder": float(actual[assignment_index] - sum(float(isolated[name][assignment_index]) for name in GROUPS)),
                            **{f"integral_{name}": float(integral_groups[name][assignment_index]) for name in GROUPS},
                            **{f"isolated_{name}": float(isolated[name][assignment_index]) for name in GROUPS},
                        }
                    )
                margin_rows.append(
                    {
                        "scenario_id": int(scenario),
                        "time_step": int(moment),
                        "seed": int(seed),
                        "class": klass,
                        "assignment_r": vector_r,
                        "assignment_p": vector_p,
                        "j15_r": float(labels[index_r]),
                        "j15_p": float(labels[index_p]),
                        "j15_gap_r_minus_p": j15_gap,
                        "margin_r": margin_true,
                        "margin_p": margin_pred,
                        "delta_margin_actual": margin_pred - margin_true,
                        "delta_margin_linear": float(linear_true[index_r] - linear_true[index_p]),
                        "delta_margin_integral": float(integral[index_r] - integral[index_p]),
                        "sinr_rate_interaction_margin": sinr_rate_margin,
                        "reversal": float(vector_r != vector_p and margin_true > VALUE_EPS and margin_pred < -VALUE_EPS),
                        **{f"linear_{name}": group_margin_linear[name] for name in GROUPS},
                        **{f"integral_{name}": group_margin_integral[name] for name in GROUPS},
                        **{f"isolated_{name}": group_margin_isolated[name] for name in GROUPS},
                    }
                )
                decision_rows.append(
                    {
                        "scenario_id": int(scenario),
                        "time_step": int(moment),
                        "seed": int(seed),
                        "class": klass,
                        "j15_gap_r_minus_p": j15_gap,
                        "value_error_selected_pair": float(abs(actual[index_r]) + abs(actual[index_p])),
                        "sinr_pairwise": pairwise_accuracy(true_sinr, pred_sinr, 1.0e-8),
                        "rate_pairwise": pairwise_accuracy(true_rate, pred_rate, 1.0e-8),
                        **{
                            f"abs_error_raw_{name}": float(np.mean(np.abs(error_raw[..., list(GROUPS[name])])))
                            for name in GROUPS
                        },
                        **{f"margin_linear_{name}": group_margin_linear[name] for name in GROUPS},
                    }
                )
                cosines = []
                for assignment_index in range(len(state["vectors"])):
                    left = grad_true[assignment_index, :, PRIMARY_FEATURE["sinr"]]
                    right = grad_true[assignment_index, :, PRIMARY_FEATURE["rate"]]
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    if denominator > 1.0e-12:
                        cosines.append(float(np.dot(left, right) / denominator))
                hybrid_margins = {
                    name: float(scores[index_r] - scores[index_p])
                    for name, scores in oracle.items()
                }
                sinr_rate_rows.append(
                    {
                        "scenario_id": int(scenario),
                        "time_step": int(moment),
                        "seed": int(seed),
                        "class": klass,
                        "true_sinr_rate_spearman": _spearman(true_sinr, true_rate),
                        "pred_sinr_rate_spearman": _spearman(pred_sinr, pred_rate),
                        "sinr_ordering_spearman": _spearman(true_sinr, pred_sinr),
                        "rate_ordering_spearman": _spearman(true_rate, pred_rate),
                        "sinr_pairwise": pairwise_accuracy(true_sinr, pred_sinr, 1.0e-8),
                        "rate_pairwise": pairwise_accuracy(true_rate, pred_rate, 1.0e-8),
                        "error_pearson_selected": _pearson(error_raw[:, :, PRIMARY_FEATURE["sinr"]][selected], error_raw[:, :, PRIMARY_FEATURE["rate"]][selected]),
                        "gradient_cosine": mean(cosines),
                        "interaction_mean": float(np.mean(pair_interaction[("sinr", "rate")])),
                        "interaction_margin": sinr_rate_margin,
                        **{f"j15_{name}": float(labels[_chosen(scores)]) for name, scores in oracle.items()},
                        **{f"margin_{name}": hybrid_margins[name] for name in oracle},
                        "margin_recovery_sinr": _margin_recovery(hybrid_margins["P1+TRUE_SINR"], margin_pred, margin_true),
                        "margin_recovery_rate": _margin_recovery(hybrid_margins["P1+TRUE_RATE"], margin_pred, margin_true),
                        "margin_recovery_joint": _margin_recovery(hybrid_margins["P1+TRUE_SINR+TRUE_RATE"], margin_pred, margin_true),
                    }
                )
                for left, right in combinations(GROUPS, 2):
                    interaction = pair_interaction[(left, right)]
                    cross_rows.append(
                        {
                            "scenario_id": int(scenario),
                            "time_step": int(moment),
                            "seed": int(seed),
                            "left": left,
                            "right": right,
                            "mean_interaction": float(np.mean(interaction)),
                            "margin_interaction": float(interaction[index_r] - interaction[index_p]),
                            "mean_abs_interaction": float(np.mean(np.abs(interaction))),
                        }
                    )
                if int(scenario) == CASE_SCENARIO and int(moment) == CASE_TIME:
                    _append_case(
                        case_rows,
                        int(seed),
                        state,
                        true,
                        pred,
                        error_raw,
                        values_true,
                        values_pred,
                        oracle,
                        grad_true,
                        linear_groups,
                        integral_groups,
                        isolated,
                        pair_interaction[("sinr", "rate")],
                        static,
                    )
                for item in residuals:
                    if item["scenario_id"] == int(scenario) and item["time_step"] == int(moment):
                        residual_rows.append(
                            _residual_row(
                                item,
                                int(seed),
                                state,
                                labels,
                                static,
                                values_true,
                                values_pred,
                                error_raw,
                                group_margin_linear,
                                margin_true,
                                margin_pred,
                                vector_r,
                                vector_p,
                                klass,
                            )
                        )
    aligned = bool(align_p1 <= ALIGN_ABS and align_r1 <= ALIGN_ABS and float(np.min(argmax_hits)) == 1.0)
    alignment = {
        "aligned": aligned,
        "align_abs_p1": align_p1,
        "align_abs_r1": align_r1,
        "argmax_agreement": mean(argmax_hits),
        "feature_order_matches_spec": True,
        "normalization_reused": "phase5zd_response_normalization.json",
        "checkpoint_pairing": "response predictor seed k with Model-R1 seed k",
        "candidate_source": "frozen stage-1 top-3 pool",
        "quadrature_max_abs": float(np.max(quadrature_gaps)) if quadrature_gaps else None,
        "linear_group_additivity_max_abs": float(np.max(linear_add_gaps)) if linear_add_gaps else None,
    }
    (OUTPUT / "phase5zg_alignment_audit.json").write_text(json.dumps(plain(alignment), indent=2), encoding="utf-8")
    if not aligned:
        _write_outputs(
            jacobian_rows,
            projection_rows,
            integrated_rows,
            margin_rows,
            sinr_rate_rows,
            cross_rows,
            case_rows,
            residual_rows,
            decision_rows,
            [],
        )
        payload = _gate_payload("SENS-D", False, {"alignment": alignment})
        (OUTPUT / "phase5zg_value_sensitivity_gate.json").write_text(json.dumps(plain(payload), indent=2), encoding="utf-8")
        print(json.dumps(plain(payload), indent=2), flush=True)
        return
    summary = _summarize(projection_rows, margin_rows, sinr_rate_rows, metric_rows, cross_rows, decision_rows)
    _write_outputs(
        jacobian_rows,
        projection_rows,
        integrated_rows,
        margin_rows,
        sinr_rate_rows,
        cross_rows,
        case_rows,
        residual_rows,
        decision_rows,
        summary["scenario_rows"],
    )
    (OUTPUT / "phase5zg_statistics.json").write_text(json.dumps(plain(summary["statistics"]), indent=2), encoding="utf-8")
    payload = _gate_payload(summary["letter"], True, summary["gate_extra"])
    (OUTPUT / "phase5zg_value_sensitivity_gate.json").write_text(json.dumps(plain(payload), indent=2), encoding="utf-8")
    print(json.dumps(plain(payload), indent=2), flush=True)


def _append_case(rows, seed, state, true, pred, error_raw, values_true, values_pred, oracle, grad_true, linear_groups, integral_groups, isolated, interaction, static) -> None:
    vectors = state["vectors"]
    opt_index = vectors.index(CASE_OPTIMAL)
    pred_index = vectors.index(CASE_PREDICTED)
    for label, index in (("A_pred", pred_index), ("A_opt", opt_index)):
        for uav in range(6):
            for feature_index, feature in enumerate(RESPONSE_FEATURES):
                rows.append(
                    {
                        "row_type": "response",
                        "seed": int(seed),
                        "assignment": label,
                        "uav": uav,
                        "feature": feature,
                        "true": float(true[index, uav, feature_index]),
                        "predicted": float(pred[index, uav, feature_index]),
                        "error": float(error_raw[index, uav, feature_index]),
                    }
                )
        for name, indexes in GROUPS.items():
            rows.append(
                {
                    "row_type": "group",
                    "seed": int(seed),
                    "assignment": label,
                    "group": name,
                    "error_l1": float(np.sum(np.abs(error_raw[index, :, list(indexes)]))),
                    "jacobian_l2_normalized": float(np.sqrt(np.sum(grad_true[index, :, list(indexes)] ** 2))),
                    "linear_contribution": float(linear_groups[name][index]),
                    "integral_contribution": float(integral_groups[name][index]),
                    "isolated_contribution": float(isolated[name][index]),
                }
            )
    rows.append(
        {
            "row_type": "margin",
            "seed": int(seed),
            "margin_r": float(values_true[opt_index] - values_true[pred_index]),
            "margin_p": float(values_pred[opt_index] - values_pred[pred_index]),
            "sinr_rate_interaction": float(interaction[opt_index] - interaction[pred_index]),
            "A_opt_rank_S": int(1 + np.sum(static > static[opt_index])),
            "A_opt_rank_P1": int(1 + np.sum(values_pred > values_pred[opt_index])),
            "A_opt_rank_R1": int(1 + np.sum(values_true > values_true[opt_index])),
            "A_opt_rank_TRUE_SINR": int(1 + np.sum(oracle["P1+TRUE_SINR"] > oracle["P1+TRUE_SINR"][opt_index])),
            "A_opt_rank_TRUE_RATE": int(1 + np.sum(oracle["P1+TRUE_RATE"] > oracle["P1+TRUE_RATE"][opt_index])),
            "A_opt_rank_TRUE_SINR_RATE": int(1 + np.sum(oracle["P1+TRUE_SINR+TRUE_RATE"] > oracle["P1+TRUE_SINR+TRUE_RATE"][opt_index])),
            **{f"margin_linear_{name}": float(linear_groups[name][opt_index] - linear_groups[name][pred_index]) for name in GROUPS},
            **{f"margin_integral_{name}": float(integral_groups[name][opt_index] - integral_groups[name][pred_index]) for name in GROUPS},
            **{f"margin_isolated_{name}": float(isolated[name][opt_index] - isolated[name][pred_index]) for name in GROUPS},
        }
    )


def _residual_row(item, seed, state, labels, static, values_true, values_pred, error_raw, group_margin_linear, margin_true, margin_pred, vector_r, vector_p, klass) -> dict:
    vectors = list(state["vectors"])
    named_pred = vectors.index(item["predicted_assignment"])
    named_opt = vectors.index(item["optimal_assignment"])
    selected_s = _chosen(static)
    selected_p = _chosen(values_pred)
    selected_r = _chosen(values_true)
    j15_s = float(labels[selected_s])
    j15_p = float(labels[selected_p])
    j15_r = float(labels[selected_r])
    named_margin_r = float(values_true[named_opt] - values_true[named_pred])
    named_margin_p = float(values_pred[named_opt] - values_pred[named_pred])
    return {
        "scenario_id": int(item["scenario_id"]),
        "time_step": int(item["time_step"]),
        "seed": int(seed),
        "predicted_assignment": item["predicted_assignment"],
        "optimal_assignment": item["optimal_assignment"],
        "selected_j15_s": j15_s,
        "selected_j15_p1": j15_p,
        "selected_j15_r1": j15_r,
        "named_predicted_j15": float(labels[named_pred]),
        "named_optimal_j15": float(labels[named_opt]),
        "p1_regret": float(np.max(quantize(labels)) - quantize(labels)[selected_p]),
        "value_error_named_predicted": float(values_pred[named_pred] - values_true[named_pred]),
        "r1_repaired_vs_s": float(j15_r > j15_s + TIE_BITS),
        "p1_lost_r1_repair": float(j15_r > j15_s + TIE_BITS and j15_p + TIE_BITS < j15_r),
        "selected_class": klass,
        "selected_margin_reversal": float(vector_r != vector_p and margin_true > VALUE_EPS and margin_pred < -VALUE_EPS),
        "named_margin_r": named_margin_r,
        "named_margin_p": named_margin_p,
        "named_margin_reversal": float(named_margin_r > VALUE_EPS and named_margin_p < -VALUE_EPS),
        "dominant_selected_margin_group": unique_dominant(group_margin_linear, UNIQUE_RELATIVE_GAP) or "",
        **{
            f"error_l1_{name}": float(np.sum(np.abs(error_raw[named_pred, :, list(GROUPS[name])])))
            for name in GROUPS
        },
        **{f"selected_margin_linear_{name}": group_margin_linear[name] for name in GROUPS},
    }


def _mean_finite(values) -> float:
    return mean(values)


def _vote(rows: list[dict], key_fn) -> dict[str, str | None]:
    grouped: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        if row["class"] != "LOWER_J15":
            continue
        grouped[key_fn(row)].append(row)
    votes = {}
    for key, chosen in grouped.items():
        contrib = {name: mean(float(row[f"linear_{name}"]) for row in chosen) for name in GROUPS}
        votes[key] = unique_dominant(contrib, UNIQUE_RELATIVE_GAP)
    return votes


def _summarize(projection_rows, margin_rows, sinr_rate_rows, metric_rows, cross_rows, decision_rows) -> dict:
    state_sign = []
    state_spearman = []
    for key in {(row["scenario_id"], row["time_step"], row["seed"]) for row in projection_rows}:
        chosen = [row for row in projection_rows if (row["scenario_id"], row["time_step"], row["seed"]) == key]
        actual = [float(row["delta_v_actual"]) for row in chosen]
        linear = [float(row["delta_v_linear_true"]) for row in chosen]
        state_sign.append({"scenario_id": key[0], "sign": sign_agreement(actual, linear)})
        state_spearman.append({"scenario_id": key[0], "spearman": _spearman(actual, linear), "pearson": _pearson(actual, linear)})
    harmful = [row for row in margin_rows if row["class"] == "LOWER_J15"]
    margin_sign = sign_agreement(
        [float(row["delta_margin_actual"]) for row in harmful],
        [float(row["delta_margin_linear"]) for row in harmful],
    )
    scenario_votes = _vote(margin_rows, lambda row: int(row["scenario_id"]))
    seed_votes = _vote(margin_rows, lambda row: int(row["seed"]))
    vote_counts = {name: sum(group == name for group in scenario_votes.values()) for name in GROUPS}
    seed_counts = {name: sum(group == name for group in seed_votes.values()) for name in GROUPS}
    if sum(vote_counts.values()) == 0:
        dominant = ""
        channel_stable = False
    else:
        dominant = max(vote_counts, key=lambda name: (vote_counts[name], -list(GROUPS).index(name)))
        channel_stable = bool(
            vote_counts.get(dominant, 0) >= STABLE_SCENARIOS and seed_counts.get(dominant, 0) >= STABLE_SEEDS
        )
    finite_spearman = [row["spearman"] for row in state_spearman if np.isfinite(row["spearman"])]
    sign_mean = _mean_finite(row["sign"] for row in state_sign)
    spearman_mean = _mean_finite(finite_spearman)
    first_order_ok = bool(
        np.isfinite(sign_mean)
        and sign_mean >= EXPLAIN_SIGN
        and np.isfinite(spearman_mean)
        and spearman_mean >= EXPLAIN_SPEARMAN
        and len(finite_spearman) >= FINITE_FRACTION * max(len(state_spearman), 1)
        and np.isfinite(margin_sign)
        and margin_sign >= MARGIN_SIGN
        and channel_stable
        and len(harmful) > 0
    )
    interaction_values = [float(row["interaction_margin"]) for row in sinr_rate_rows if row["class"] == "LOWER_J15"]
    actual_margins = [float(row["delta_margin_actual"]) for row in harmful]
    interaction_mean = _mean_finite(interaction_values)
    actual_mean = _mean_finite(actual_margins)
    ratio = abs(interaction_mean) / abs(actual_mean) if np.isfinite(interaction_mean) and np.isfinite(actual_mean) and abs(actual_mean) > VALUE_EPS else float("nan")
    interaction_pairs = [
        {"scenario_id": int(row["scenario_id"]), "delta": float(row["interaction_margin"])}
        for row in sinr_rate_rows
        if row["class"] == "LOWER_J15"
    ]
    interaction_ci = cluster_bootstrap(interaction_pairs, "delta", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    scenario_interaction = {}
    for row in sinr_rate_rows:
        if row["class"] != "LOWER_J15":
            continue
        scenario_interaction.setdefault(int(row["scenario_id"]), []).append(float(row["interaction_margin"]))
    scenario_signs = []
    for values in scenario_interaction.values():
        scenario_signs.append(float(np.mean(values)))
    same_sign = 0
    if np.isfinite(interaction_mean) and abs(interaction_mean) > VALUE_EPS:
        same_sign = sum((value > 0.0) == (interaction_mean > 0.0) and abs(value) > VALUE_EPS for value in scenario_signs)
    ci_excludes_zero = bool(interaction_ci["low"] > 0.0 or interaction_ci["high"] < 0.0)
    interaction_ok = bool(
        np.isfinite(ratio) and ratio >= INTERACTION_RATIO and same_sign >= STABLE_SCENARIOS and ci_excludes_zero
    )
    sinr_pairwise = _mean_finite(row["sinr_pairwise"] for row in decision_rows)
    rate_pairwise = _mean_finite(row["rate_pairwise"] for row in decision_rows)
    ordering_failure = bool(
        len(harmful) > 0 and (sinr_pairwise < ORDERING_PAIRWISE_MAX or rate_pairwise < ORDERING_PAIRWISE_MAX)
    )
    letter = choose_gate(
        aligned=True,
        first_order_ok=first_order_ok,
        interaction_ok=interaction_ok,
        ordering_failure=ordering_failure,
    )
    comparisons = {}
    for left, right, label in (
        ("P1+TRUE_SINR", "P1", "sinr_minus_p1"),
        ("P1+TRUE_RATE", "P1", "rate_minus_p1"),
        ("P1+TRUE_SINR+TRUE_RATE", "P1", "joint_minus_p1"),
        ("P1+TRUE_SINR+TRUE_RATE", "P1+TRUE_SINR", "joint_minus_sinr"),
        ("P1+TRUE_SINR+TRUE_RATE", "P1+TRUE_RATE", "joint_minus_rate"),
        ("R1", "P1", "r1_minus_p1"),
    ):
        pairs = []
        for row in metric_rows:
            if row["model"] != left:
                continue
            base = next(
                item["selected_j15"]
                for item in metric_rows
                if item["model"] == right
                and item["scenario_id"] == row["scenario_id"]
                and item["time_step"] == row["time_step"]
                and item["seed"] == row["seed"]
            )
            pairs.append({"scenario_id": int(row["scenario_id"]), "delta": float(row["selected_j15"]) - float(base)})
        comparisons[label] = cluster_bootstrap(pairs, "delta", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    model_names = ("P1", "R1", "P1+TRUE_SINR", "P1+TRUE_RATE", "P1+TRUE_SINR+TRUE_RATE")
    selected = {model: mean(row["selected_j15"] for row in metric_rows if row["model"] == model) for model in model_names}
    regret = {model: mean(row["normalized_regret"] for row in metric_rows if row["model"] == model) for model in model_names}
    agreement = {model: mean(row["agreement_with_r1"] for row in metric_rows if row["model"] == model) for model in model_names}
    gap = selected["R1"] - selected["P1"]

    def recovery(model: str) -> float:
        if not np.isfinite(gap) or gap <= 0.0:
            return float("nan")
        return (selected[model] - selected["P1"]) / gap

    class_counts = {name: sum(row["class"] == name for row in margin_rows) for name in ("SAME_ASSIGNMENT", "SAME_J15", "LOWER_J15", "HIGHER_J15")}
    scenario_rows = []
    for scenario in CONFIRM_SCENARIOS:
        chosen = [row for row in margin_rows if int(row["scenario_id"]) == int(scenario)]
        scenario_rows.append(
            {
                "unit": "scenario",
                "unit_id": int(scenario),
                "lower_j15": sum(row["class"] == "LOWER_J15" for row in chosen),
                "same_j15": sum(row["class"] == "SAME_J15" for row in chosen),
                "higher_j15": sum(row["class"] == "HIGHER_J15" for row in chosen),
                "same_assignment": sum(row["class"] == "SAME_ASSIGNMENT" for row in chosen),
                "dominant_group": scenario_votes.get(int(scenario)) or "",
                **{
                    f"j15_{model}": mean(
                        row["selected_j15"]
                        for row in metric_rows
                        if row["model"] == model and int(row["scenario_id"]) == int(scenario)
                    )
                    for model in ("P1", "R1", "P1+TRUE_SINR", "P1+TRUE_RATE", "P1+TRUE_SINR+TRUE_RATE")
                },
            }
        )
    for seed in SEEDS:
        chosen = [row for row in margin_rows if int(row["seed"]) == int(seed)]
        scenario_rows.append(
            {
                "unit": "seed",
                "unit_id": int(seed),
                "lower_j15": sum(row["class"] == "LOWER_J15" for row in chosen),
                "same_j15": sum(row["class"] == "SAME_J15" for row in chosen),
                "higher_j15": sum(row["class"] == "HIGHER_J15" for row in chosen),
                "same_assignment": sum(row["class"] == "SAME_ASSIGNMENT" for row in chosen),
                "dominant_group": seed_votes.get(int(seed)) or "",
                **{
                    f"j15_{model}": mean(row["selected_j15"] for row in metric_rows if row["model"] == model and int(row["seed"]) == int(seed))
                    for model in ("P1", "R1", "P1+TRUE_SINR", "P1+TRUE_RATE", "P1+TRUE_SINR+TRUE_RATE")
                },
            }
        )
    cross_summary = {}
    for left, right in combinations(GROUPS, 2):
        chosen = [row for row in cross_rows if row["left"] == left and row["right"] == right and any(
            item["scenario_id"] == row["scenario_id"] and item["time_step"] == row["time_step"] and item["seed"] == row["seed"] and item["class"] == "LOWER_J15"
            for item in margin_rows
        )]
        cross_summary[f"{left}__{right}"] = mean(row["margin_interaction"] for row in chosen)
    statistics = {
        "class_counts": class_counts,
        "sign_agreement_mean": sign_mean,
        "spearman_mean": spearman_mean,
        "pearson_mean": _mean_finite(row["pearson"] for row in state_spearman),
        "margin_sign_agreement": margin_sign,
        "scenario_votes": {str(key): value for key, value in scenario_votes.items()},
        "seed_votes": {str(key): value for key, value in seed_votes.items()},
        "vote_counts": vote_counts,
        "seed_counts": seed_counts,
        "dominant_channel": dominant if channel_stable else dominant,
        "channel_stable": channel_stable,
        "interaction_mean": interaction_mean,
        "interaction_ratio": ratio,
        "interaction_ci": interaction_ci,
        "interaction_same_sign_scenarios": same_sign,
        "selected_j15": selected,
        "normalized_regret": regret,
        "agreement_with_r1": agreement,
        "recovery": {model: recovery(model) for model in selected if model != "P1"},
        "bootstrap": comparisons,
        "sinr_pairwise": sinr_pairwise,
        "rate_pairwise": rate_pairwise,
        "true_sinr_rate_spearman": _mean_finite(row["true_sinr_rate_spearman"] for row in sinr_rate_rows),
        "pred_sinr_rate_spearman": _mean_finite(row["pred_sinr_rate_spearman"] for row in sinr_rate_rows),
        "error_pearson": _mean_finite(row["error_pearson_selected"] for row in sinr_rate_rows),
        "gradient_cosine": _mean_finite(row["gradient_cosine"] for row in sinr_rate_rows),
        "margin_recovery": {
            "sinr": _mean_finite(row["margin_recovery_sinr"] for row in sinr_rate_rows if row["class"] == "LOWER_J15"),
            "rate": _mean_finite(row["margin_recovery_rate"] for row in sinr_rate_rows if row["class"] == "LOWER_J15"),
            "joint": _mean_finite(row["margin_recovery_joint"] for row in sinr_rate_rows if row["class"] == "LOWER_J15"),
        },
        "cross_margin_interaction_on_lower_j15": cross_summary,
        "first_order_ok": first_order_ok,
        "interaction_ok": interaction_ok,
        "ordering_failure": ordering_failure,
        "optimizer_steps": 0,
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": False,
    }
    gate_extra = {
        "dominant_channel": dominant,
        "channel_stable": channel_stable,
        "sign_agreement_mean": sign_mean,
        "spearman_mean": spearman_mean,
        "margin_sign_agreement": margin_sign,
        "interaction_ratio": ratio,
        "interaction_ci_low": interaction_ci["low"],
        "interaction_ci_high": interaction_ci["high"],
        "sinr_pairwise": sinr_pairwise,
        "rate_pairwise": rate_pairwise,
        "recovery": statistics["recovery"],
        "class_counts": class_counts,
        "first_order_ok": first_order_ok,
        "interaction_ok": interaction_ok,
        "ordering_failure": ordering_failure,
    }
    return {"letter": letter, "statistics": statistics, "gate_extra": gate_extra, "scenario_rows": scenario_rows}


def _write_outputs(*tables) -> None:
    names = (
        "phase5zg_value_jacobian.csv",
        "phase5zg_response_error_projection.csv",
        "phase5zg_integrated_value_attribution.csv",
        "phase5zg_assignment_margin_flips.csv",
        "phase5zg_sinr_rate_interaction.csv",
        "phase5zg_cross_channel_interactions.csv",
        "phase5zg_310114_t15.csv",
        "phase5zg_residual_cases.csv",
        "phase5zg_prediction_value_decision_error.csv",
        "phase5zg_scenario_seed_statistics.csv",
    )
    for name, rows in zip(names, tables):
        write_csv(OUTPUT / name, rows)


def _finish_unaligned(alignment: dict) -> None:
    alignment = {"aligned": False, **alignment}
    (OUTPUT / "phase5zg_alignment_audit.json").write_text(json.dumps(plain(alignment), indent=2), encoding="utf-8")
    _touch("phase5zg_value_jacobian.csv", ["aligned"])
    _touch("phase5zg_response_error_projection.csv", ["aligned"])
    _touch("phase5zg_integrated_value_attribution.csv", ["aligned"])
    _touch("phase5zg_assignment_margin_flips.csv", ["aligned"])
    _touch("phase5zg_sinr_rate_interaction.csv", ["aligned"])
    _touch("phase5zg_cross_channel_interactions.csv", ["aligned"])
    _touch("phase5zg_310114_t15.csv", ["aligned"])
    _touch("phase5zg_residual_cases.csv", ["aligned"])
    _touch("phase5zg_prediction_value_decision_error.csv", ["aligned"])
    _touch("phase5zg_scenario_seed_statistics.csv", ["aligned"])
    payload = _gate_payload("SENS-D", False, {"alignment": alignment})
    (OUTPUT / "phase5zg_value_sensitivity_gate.json").write_text(json.dumps(plain(payload), indent=2), encoding="utf-8")
    print(json.dumps(plain(payload), indent=2), flush=True)


if __name__ == "__main__":
    run()
