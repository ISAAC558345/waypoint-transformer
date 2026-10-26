"""Candidate subset fixed from existing J15 labels. Urgent-service values are not an input."""
from __future__ import annotations

import numpy as np

from .protocol import PER_STRATUM, PILOT_SEED, SAMPLE_K, STRATA, TOP_K


def select_candidates(
    candidate_id: np.ndarray,
    j15_bits: np.ndarray,
    utility_rank: np.ndarray,
    *,
    scenario_id: int,
    time_step: int,
) -> dict:
    """Take 32 stable top-J15 rows and 96 rank-quantile rows from the remainder."""
    ids = np.asarray(candidate_id, dtype=np.int64).reshape(-1)
    labels = np.asarray(j15_bits, dtype=np.float64).reshape(-1)
    ranks = np.asarray(utility_rank, dtype=np.int64).reshape(-1)
    if ids.shape != labels.shape or ids.shape != ranks.shape:
        raise RuntimeError("candidate id, J15, and rank arrays do not align")
    if ids.size == 0:
        raise RuntimeError("candidate shard is empty")
    if len(set(ids.tolist())) != int(ids.size):
        raise RuntimeError("candidate ids are not unique")
    order = np.argsort(ids, kind="mergesort")
    order = order[np.argsort(-labels[order], kind="mergesort")]
    top_n = min(int(TOP_K), int(order.size))
    top = np.asarray(order[:top_n], dtype=np.int64)
    rest = np.asarray(order[top_n:], dtype=np.int64)
    if rest.size:
        rest_order = np.argsort(ids[rest], kind="mergesort")
        rest_order = rest_order[np.argsort(ranks[rest][rest_order], kind="mergesort")]
        sorted_rest = rest[rest_order]
    else:
        sorted_rest = rest
    cuts = np.linspace(0, int(sorted_rest.size), int(STRATA) + 1, dtype=int)
    rng = np.random.default_rng(np.random.SeedSequence([int(PILOT_SEED), int(scenario_id), int(time_step)]))
    sampled: list[np.ndarray] = []
    strata = []
    shortfall = 0
    for stratum in range(int(STRATA)):
        band = sorted_rest[int(cuts[stratum]) : int(cuts[stratum + 1])]
        take = min(int(PER_STRATUM), int(band.size))
        short = int(PER_STRATUM) - int(take)
        shortfall += short
        if take:
            picked = np.asarray(rng.choice(band, size=take, replace=False), dtype=np.int64).reshape(-1)
        else:
            picked = np.zeros(0, dtype=np.int64)
        sampled.append(picked)
        strata.append(
            {
                "stratum": int(stratum),
                "band_size": int(band.size),
                "drawn": int(take),
                "shortfall": int(short),
                "rank_min": None if band.size == 0 else int(np.min(ranks[band])),
                "rank_max": None if band.size == 0 else int(np.max(ranks[band])),
            }
        )
    sample_index = np.concatenate(sampled) if sampled else np.zeros(0, dtype=np.int64)
    selected = {int(item) for item in top.tolist()} | {int(item) for item in sample_index.tolist()}
    fill: list[int] = []
    if shortfall > 0:
        for position in np.argsort(ids, kind="mergesort"):
            index = int(position)
            if index in selected:
                continue
            fill.append(index)
            selected.add(index)
            if len(fill) >= shortfall:
                break
    fill_index = np.asarray(fill, dtype=np.int64)
    rows = [_row(ids, labels, ranks, int(position), "top_j15", -1, 0) for position in top.tolist()]
    for stratum, picked in enumerate(sampled):
        rows.extend(_row(ids, labels, ranks, int(position), "stratum", int(stratum), 0) for position in picked.tolist())
    rows.extend(_row(ids, labels, ranks, int(position), "fill", -1, 1) for position in fill_index.tolist())
    if len({int(row["candidate_id"]) for row in rows}) != len(rows):
        raise RuntimeError("candidate sample contains a duplicate id")
    return {
        "rows": rows,
        "count": int(len(rows)),
        "top_count": int(top.size),
        "stratum_count": int(sample_index.size),
        "fill_count": int(fill_index.size),
        "shortfall_unfilled": int(shortfall - fill_index.size),
        "strata": strata,
        "seed": int(PILOT_SEED),
    }


def _row(ids, labels, ranks, position: int, selection: str, stratum: int, filled: int) -> dict:
    return {
        "row_index": int(position),
        "candidate_id": int(ids[position]),
        "j15_bits": float(labels[position]),
        "utility_rank": int(ranks[position]),
        "selection": selection,
        "stratum": int(stratum),
        "filled": int(filled),
    }
