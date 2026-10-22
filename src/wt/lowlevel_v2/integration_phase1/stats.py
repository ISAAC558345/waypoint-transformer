"""Rank and correlation helpers. No causal claims."""
from __future__ import annotations

import numpy as np


def _finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def spearman(x, y) -> float:
    a, b = _finite_pair(x, y)
    if a.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(a)).astype(np.float64)
    ry = np.argsort(np.argsort(b)).astype(np.float64)
    if float(np.std(rx)) < 1.0e-12 or float(np.std(ry)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau(x, y) -> float:
    a, b = _finite_pair(x, y)
    n = int(a.size)
    if n < 3:
        return float("nan")
    conc = 0
    disc = 0
    for i in range(n):
        da = np.sign(a[i + 1 :] - a[i])
        db = np.sign(b[i + 1 :] - b[i])
        prod = da * db
        conc += int(np.sum(prod > 0))
        disc += int(np.sum(prod < 0))
    denom = n * (n - 1) / 2.0
    if denom <= 0:
        return float("nan")
    return float((conc - disc) / denom)


def summarize(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "median", "p10", "p25", "p75", "p90", "n")}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "n": int(a.size),
    }


def topk_overlap(u0, u1, k: int) -> float:
    a = np.asarray(u0, dtype=np.float64)
    b = np.asarray(u1, dtype=np.float64)
    k = min(int(k), int(a.size))
    if k <= 0:
        return float("nan")
    s0 = set(np.argsort(-a)[:k].tolist())
    s1 = set(np.argsort(-b)[:k].tolist())
    return float(len(s0 & s1) / k)


def near_opt_set(u, frac: float) -> set[int]:
    a = np.asarray(u, dtype=np.float64)
    best = float(np.max(a))
    return {int(i) for i, val in enumerate(a) if float(val) >= frac * best}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return float("nan")
    return float(len(a & b) / len(union))


__all__ = ["jaccard", "kendall_tau", "near_opt_set", "spearman", "summarize", "topk_overlap"]
