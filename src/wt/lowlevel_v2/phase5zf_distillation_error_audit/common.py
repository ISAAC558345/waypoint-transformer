"""Shared loaders for the inference-only audit."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from .protocol import GROUPS, RESPONSE_FEATURES


def write_csv(path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def direction(value: float, tie_bits: float) -> str:
    if abs(float(value)) <= float(tie_bits):
        return "tie"
    return "positive" if float(value) > 0.0 else "negative"


def plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def normalize(raw: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((raw - norm["mean"]) / norm["std"]).astype(np.float32)


def group_indexes(names: tuple[str, ...]) -> tuple[int, ...]:
    found = []
    for name in names:
        found.extend(GROUPS[name])
    return tuple(found)


def scalars(response: np.ndarray, assignments: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(assignments) >= 0
    count = mask.sum(axis=1)

    def assigned_mean(feature: int) -> np.ndarray:
        held = np.where(mask, response[:, :, feature], 0.0)
        return np.divide(held.sum(axis=1), count, out=np.full(count.shape, np.nan), where=count > 0)

    def assigned_min(feature: int) -> np.ndarray:
        held = np.where(mask, response[:, :, feature], np.inf)
        values = held.min(axis=1)
        values[count == 0] = np.nan
        return values

    def assigned_max(feature: int) -> np.ndarray:
        held = np.where(mask, response[:, :, feature], -np.inf)
        values = held.max(axis=1)
        values[count == 0] = np.nan
        return values

    def assigned_std(feature: int) -> np.ndarray:
        held = np.where(mask, response[:, :, feature], np.nan)
        return np.nanstd(held, axis=1)

    output = {}
    for feature, name in enumerate(RESPONSE_FEATURES):
        output[f"mean_{name}"] = assigned_mean(feature)
        output[f"min_{name}"] = assigned_min(feature)
        output[f"max_{name}"] = assigned_max(feature)
        output[f"std_{name}"] = assigned_std(feature)
    output["sum_service_progress_1"] = response[:, :, 4].sum(axis=1)
    return output


def load_tables(predictions, downstream, assert_consumed) -> tuple[dict, dict]:
    responses = defaultdict(dict)
    with predictions.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            key = (scenario, int(row["time_step"]), int(row["seed"]), row["assignment_vector"])
            if key not in responses:
                responses[key] = (
                    np.zeros((6, len(RESPONSE_FEATURES)), dtype=np.float64),
                    np.zeros((6, len(RESPONSE_FEATURES)), dtype=np.float64),
                )
            uav = int(row["uav"])
            for index, name in enumerate(RESPONSE_FEATURES):
                responses[key][0][uav, index] = float(row[f"true_{name}"])
                responses[key][1][uav, index] = float(row[f"pred_{name}"])
    scores = {}
    with downstream.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            scores[(scenario, int(row["time_step"]), int(row["seed"]), row["assignment_vector"])] = (
                float(row["j15"]),
                float(row["v_s"]),
                float(row["v_p1"]),
                float(row["v_r1"]),
            )
    return responses, scores


def overlap(truth: np.ndarray, prediction: np.ndarray, width: int) -> float:
    if truth.size < width or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        return float("nan")
    true_top = set(np.argsort(-truth, kind="stable")[:width].tolist())
    pred_top = set(np.argsort(-prediction, kind="stable")[:width].tolist())
    return float(len(true_top & pred_top) / float(width))


__all__ = [
    "PAIR_EPS",
    "direction",
    "group_indexes",
    "load_tables",
    "mean",
    "normalize",
    "overlap",
    "plain",
    "scalars",
    "write_csv",
]
