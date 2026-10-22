"""Alignment metrics on a frozen candidate pool. Not confirmation payload."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase944.tables import pearson, spearman


def pool_metrics(utilities: np.ndarray, logp: np.ndarray) -> dict[str, float]:
    u = np.asarray(utilities, dtype=np.float64)
    lp = np.asarray(logp, dtype=np.float64)
    best = float(np.max(u))
    top = int(np.argmax(lp))
    p = np.exp(lp - np.max(lp))
    p = p / max(float(p.sum()), 1.0e-12)
    nom = float(p[u >= 0.95 * best].sum()) if best > 0 else float(p[np.abs(u - best) <= 1.0e-9].sum())
    order_u = np.argsort(-u)
    n = u.size
    top10 = set(order_u[: max(1, int(np.ceil(0.10 * n)))].tolist())
    top20 = set(order_u[: max(1, int(np.ceil(0.20 * n)))].tolist())
    return {
        "spearman": spearman(u, lp),
        "pearson": pearson(u, lp),
        "top1_utility": float(u[top]),
        "best_observed": best,
        "top1_regret": best - float(u[top]),
        "hit_top10": float(top in top10),
        "hit_top20": float(top in top20),
        "expected_utility": float(np.dot(p, u)),
        "nom5": nom,
    }


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in {"encoder", "model_seed", "split"}]
    return {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys}
