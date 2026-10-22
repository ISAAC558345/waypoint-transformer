"""Ranking and paired-difference summaries."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.labels import ordered_keys
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, POSITIVE_BIT


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = 0.5 * ((start + 1) + (stop + 1))
        ranks[order[start : stop + 1]] = average
        start = stop + 1
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left - float(np.mean(left))
    y = right - float(np.mean(right))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x * y) / denom)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(left, dtype=np.float64)), _rankdata(np.asarray(right, dtype=np.float64)))


def kendall(left: np.ndarray, right: np.ndarray) -> float:
    try:
        from scipy.stats import kendalltau
    except ImportError:
        return _kendall_numpy(left, right)
    result = kendalltau(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    tau = getattr(result, "statistic", None)
    if tau is None:
        tau = result.correlation
    return 0.0 if tau is None or not np.isfinite(tau) else float(tau)


def _kendall_numpy(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    concordant = 0.0
    discordant = 0.0
    for index in range(x.size - 1):
        product = (x[index + 1 :] - x[index]) * (y[index + 1 :] - y[index])
        concordant += float(np.sum(product > 0.0))
        discordant += float(np.sum(product < 0.0))
    total = x.size * (x.size - 1) / 2.0
    if total <= 0.0:
        return 0.0
    return float((concordant - discordant) / total)


def ranking_shift(fly: dict, mappo: dict) -> dict:
    keys = ordered_keys(fly)
    fly_vector = np.asarray([fly[key] for key in keys], dtype=np.float64)
    mappo_vector = np.asarray([mappo[key] for key in keys], dtype=np.float64)
    fly_top = ordered_keys(fly)
    mappo_top = ordered_keys(mappo)
    mappo_best = mappo_top[0]
    fly_rank = fly_top.index(mappo_best) + 1
    def overlap(k: int) -> float:
        return float(len(set(fly_top[:k]) & set(mappo_top[:k])) / k)
    return {
        "spearman": spearman(fly_vector, mappo_vector),
        "kendall": kendall(fly_vector, mappo_vector),
        "top1_agreement": int(fly_top[0] == mappo_top[0]),
        "top3_overlap": overlap(3),
        "top5_overlap": overlap(5),
        "argmax_disagreement": int(fly_top[0] != mappo_top[0]),
        "mappo_top1_fly_rank": int(fly_rank),
        "fly_top1": ",".join(str(item) for item in fly_top[0]),
        "mappo_top1": ",".join(str(item) for item in mappo_best),
        "fly_top2": ",".join(str(item) for item in fly_top[1]) if len(fly_top) > 1 else "",
        "mappo_top2": ",".join(str(item) for item in mappo_top[1]) if len(mappo_top) > 1 else "",
        "mappo_top3": ",".join(str(item) for item in mappo_top[2]) if len(mappo_top) > 2 else "",
    }


def bootstrap_mean(values: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    count = int(data.size)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, count, count)]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(np.mean(data)), float(low), float(high)


def positive_fraction(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64) > POSITIVE_BIT))


def median(values: np.ndarray) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))
