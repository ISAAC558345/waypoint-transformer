"""Metrics and paired-statistics helpers for the fixed Round-21 protocol."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1.0e-9


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    # Ties receive their mean rank.
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def correlation(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    if len(actual) < 2 or np.std(actual) <= EPS or np.std(predicted) <= EPS:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(actual, predicted)[0, 1])
    spearman = float(np.corrcoef(_rank(actual), _rank(predicted))[0, 1])
    return pearson, spearman


def bootstrap_mean_ci(values: Sequence[float], seed: int = 2021, draws: int = 1000) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, vector.size, size=(draws, vector.size))
    means = vector[samples].mean(axis=1)
    return float(vector.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _entropy(probability: np.ndarray) -> float:
    active = probability[probability > EPS]
    return float(-(active * np.log(active)).sum()) if active.size else 0.0


def sample_metrics(
    prediction: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    metadata: Sequence[Mapping[str, Any]],
    bridge: np.ndarray,
    conflict: np.ndarray,
    return_mean: float,
    return_std: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return sample-level values and per-(UAV, buoy) progress audit rows."""
    service_p = np.asarray(prediction["service"], dtype=np.float64)
    progress_p = np.asarray(prediction["progress"], dtype=np.float64)
    return_p = np.asarray(prediction["team_return"], dtype=np.float64)
    service_t = np.asarray(target["service"], dtype=np.float64)
    progress_t = np.asarray(target["progress_normalized"], dtype=np.float64)
    progress_raw_t = np.asarray(target["progress_raw"], dtype=np.float64)
    future_actual = np.asarray(target["future_actual_buoy_service_bits"], dtype=np.float64)
    return_t = np.asarray(target["return_normalized"], dtype=np.float64)
    return_raw_t = np.asarray(target["return_raw"], dtype=np.float64)
    u_mask = np.asarray(target["uav_mask"], dtype=bool)
    b_mask = np.asarray(target["buoy_mask"], dtype=bool)
    service_valid = np.asarray(target["service_valid"], dtype=bool)
    rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    for index, info in enumerate(metadata):
        valid_pairs = u_mask[index, :, None] & b_mask[index, None, :]
        valid_service = service_valid[index] & u_mask[index]
        ce: list[float] = []
        kl: list[float] = []
        js: list[float] = []
        acc: list[float] = []
        target_entropy: list[float] = []
        pred_entropy: list[float] = []
        sum_error: list[float] = []
        for uav in np.flatnonzero(valid_service):
            mask = b_mask[index]
            p = service_p[index, uav, mask]
            q = service_t[index, uav, mask]
            p = p / max(p.sum(), EPS)
            q = q / max(q.sum(), EPS)
            midpoint = (p + q) / 2.0
            ce.append(float(-(q * np.log(np.clip(p, EPS, 1.0))).sum()))
            kl.append(float((q * (np.log(np.clip(q, EPS, 1.0)) - np.log(np.clip(p, EPS, 1.0)))).sum()))
            js.append(float(.5 * (p * (np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(midpoint, EPS, 1.0)))).sum() + .5 * (q * (np.log(np.clip(q, EPS, 1.0)) - np.log(np.clip(midpoint, EPS, 1.0)))).sum()))
            acc.append(float(int(np.argmax(p) == np.argmax(q))))
            target_entropy.append(_entropy(q))
            pred_entropy.append(_entropy(p))
            sum_error.append(abs(float(p.sum()) - 1.0))
        pred_raw = progress_p[index] * float(info["queue_capacity_bits"])
        error_norm = progress_p[index][valid_pairs] - progress_t[index][valid_pairs]
        error_raw = pred_raw[valid_pairs] - progress_raw_t[index][valid_pairs]
        team_pred = float(pred_raw[valid_pairs].sum())
        team_true = float(progress_raw_t[index][valid_pairs].sum())
        return_raw_p = float(return_p[index] * return_std + return_mean)
        row = dict(info)
        row.update({
            "bridge": bool(bridge[index]), "interference_conflict": bool(conflict[index]),
            "service_soft_ce": float(np.mean(ce)) if ce else float("nan"),
            "service_kl": float(np.mean(kl)) if kl else float("nan"),
            "service_js": float(np.mean(js)) if js else float("nan"),
            "service_top1_accuracy": float(np.mean(acc)) if acc else float("nan"),
            "service_target_entropy": float(np.mean(target_entropy)) if target_entropy else float("nan"),
            "service_pred_entropy": float(np.mean(pred_entropy)) if pred_entropy else float("nan"),
            "service_probability_sum_error": float(np.max(sum_error)) if sum_error else 0.0,
            "progress_normalized_squared_error": float(np.mean(error_norm ** 2)),
            "progress_normalized_absolute_error": float(np.mean(np.abs(error_norm))),
            "progress_raw_squared_error": float(np.mean(error_raw ** 2)),
            "progress_raw_absolute_error": float(np.mean(np.abs(error_raw))),
            "team_progress_raw_squared_error": (team_pred - team_true) ** 2,
            "team_progress_raw_absolute_error": abs(team_pred - team_true),
            "progress_conservation_error_bits": abs(float(pred_raw[valid_pairs].sum()) - float(future_actual[index, b_mask[index]].sum())),
            "return_normalized_squared_error": (float(return_p[index]) - float(return_t[index])) ** 2,
            "return_normalized_absolute_error": abs(float(return_p[index]) - float(return_t[index])),
            "return_raw_squared_error": (return_raw_p - float(return_raw_t[index])) ** 2,
            "return_raw_absolute_error": abs(return_raw_p - float(return_raw_t[index])),
            "return_prediction_normalized": float(return_p[index]), "return_target_normalized": float(return_t[index]),
            "return_prediction_raw": return_raw_p, "return_target_raw": float(return_raw_t[index]),
            "team_progress_prediction_raw": team_pred, "team_progress_target_raw": team_true,
        })
        rows.append(row)
        for uav, buoy in zip(*np.where(valid_pairs)):
            coordinate = dict(info)
            coordinate.update({
                "bridge": bool(bridge[index]), "interference_conflict": bool(conflict[index]),
                "uav_slot": int(uav), "buoy_slot": int(buoy),
                "prediction_normalized": float(progress_p[index, uav, buoy]), "target_normalized": float(progress_t[index, uav, buoy]),
                "prediction_raw": float(pred_raw[uav, buoy]), "target_raw": float(progress_raw_t[index, uav, buoy]),
            })
            coordinate_rows.append(coordinate)
    return rows, coordinate_rows


def grouped_metric_rows(rows: Sequence[Mapping[str, Any]], run: Mapping[str, Any], prefix: str) -> list[dict[str, Any]]:
    """Aggregate each requested stratification without treating it as input."""
    metric_map = {
        "service": ("service_soft_ce", "service_kl", "service_js", "service_top1_accuracy", "service_target_entropy", "service_pred_entropy", "service_probability_sum_error"),
        "progress": ("progress_normalized_squared_error", "progress_normalized_absolute_error", "progress_raw_squared_error", "progress_raw_absolute_error", "team_progress_raw_squared_error", "team_progress_raw_absolute_error", "progress_conservation_error_bits"),
        "return": ("return_normalized_squared_error", "return_normalized_absolute_error", "return_raw_squared_error", "return_raw_absolute_error"),
    }
    group_columns = ("overall", "num_buoys", "interference_tier", "quality_tier", "behavior_generator", "bridge", "interference_conflict")
    output: list[dict[str, Any]] = []
    for group_column in group_columns:
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets["all" if group_column == "overall" else str(row[group_column])].append(row)
        for group, selected in buckets.items():
            for metric in metric_map[prefix]:
                values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                aggregate = float(values.mean())
                if metric.endswith("squared_error"):
                    aggregate = math.sqrt(max(aggregate, 0.0))
                    metric = metric.replace("squared_error", "rmse")
                elif metric.endswith("absolute_error"):
                    metric = metric.replace("absolute_error", "mae")
                output.append({**run, "group_dimension": group_column, "group": group, "metric": metric, "value": aggregate, "sample_count": int(len(values))})
    return output


def scalar_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    def mean(name: str) -> float:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        return float(np.nanmean(values))
    actual = np.asarray([float(row["return_target_raw"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([float(row["return_prediction_raw"]) for row in rows], dtype=np.float64)
    progress_actual = np.asarray([float(row["team_progress_target_raw"]) for row in rows], dtype=np.float64)
    progress_predicted = np.asarray([float(row["team_progress_prediction_raw"]) for row in rows], dtype=np.float64)
    p_pearson, p_spearman = correlation(progress_actual, progress_predicted)
    r_pearson, r_spearman = correlation(actual, predicted)
    r2 = float(1.0 - np.sum((actual - predicted) ** 2) / max(np.sum((actual - actual.mean()) ** 2), EPS))
    return {
        "service_soft_ce": mean("service_soft_ce"), "service_kl": mean("service_kl"), "service_js": mean("service_js"),
        "service_top1_accuracy": mean("service_top1_accuracy"), "service_probability_sum_error": mean("service_probability_sum_error"),
        "progress_normalized_rmse": math.sqrt(mean("progress_normalized_squared_error")),
        "progress_normalized_mae": mean("progress_normalized_absolute_error"),
        "progress_raw_rmse": math.sqrt(mean("progress_raw_squared_error")), "progress_raw_mae": mean("progress_raw_absolute_error"),
        "team_progress_raw_rmse": math.sqrt(mean("team_progress_raw_squared_error")), "team_progress_pearson": p_pearson, "team_progress_spearman": p_spearman,
        "progress_conservation_error_bits": mean("progress_conservation_error_bits"),
        "return_normalized_rmse": math.sqrt(mean("return_normalized_squared_error")), "return_normalized_mae": mean("return_normalized_absolute_error"),
        "return_raw_rmse": math.sqrt(mean("return_raw_squared_error")), "return_raw_mae": mean("return_raw_absolute_error"),
        "return_r2": r2, "return_pearson": r_pearson, "return_spearman": r_spearman,
    }


def paired_rows(
    hgnn: Sequence[Mapping[str, Any]], deepsets: Sequence[Mapping[str, Any]], run: Mapping[str, Any], seed: int,
) -> list[dict[str, Any]]:
    """Paired by scenario/episode (with per-sample records retained)."""
    h_map = {str(row["sample_id"]): row for row in hgnn}
    d_map = {str(row["sample_id"]): row for row in deepsets}
    joined = [(h_map[key], d_map[key]) for key in sorted(h_map.keys() & d_map.keys())]
    metric_pairs = {
        "service_soft_ce": "service_soft_ce", "progress_raw_rmse_proxy": "progress_raw_squared_error", "return_raw_rmse_proxy": "return_raw_squared_error",
    }
    output: list[dict[str, Any]] = []
    for metric, field in metric_pairs.items():
        by_episode: dict[tuple[str, str], list[float]] = defaultdict(list)
        for left, right in joined:
            by_episode[(str(left["scenario_id"]), str(left["episode_id"]))].append(float(right[field]) - float(left[field]))
        values = [float(np.mean(value)) for value in by_episode.values()]
        mean, low, high = bootstrap_mean_ci(values, seed=202100 + seed)
        output.append({**run, "seed": seed, "metric": metric, "paired_unit": "scenario_episode", "paired_unit_count": len(values), "deepsets_minus_hgnn_mean": mean, "bootstrap_95_low": low, "bootstrap_95_high": high})
    return output


__all__ = ["bootstrap_mean_ci", "grouped_metric_rows", "paired_rows", "sample_metrics", "scalar_summary"]
