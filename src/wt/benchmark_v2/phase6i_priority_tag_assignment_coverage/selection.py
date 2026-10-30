"""Stratify the joint pool by the baseline activity mask. This module does not step the environment."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.candidates import assignment_key, pool_sha256

from .protocol import ALTERNATIVE_CAP, CANDIDATE_LIMIT, candidate_seed


def active_mask(assignment) -> tuple[int, ...]:
    """One bit per UAV. Active means the target index is non-negative."""
    return tuple(1 if int(item) >= 0 else 0 for item in assignment_key(assignment))


def _dedup(pool) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int], list[int]]:
    unique: list[tuple[int, ...]] = []
    first_index: dict[tuple[int, ...], int] = {}
    dedup: list[int] = []
    width = None
    for position, assignment in enumerate(pool):
        key = assignment_key(assignment)
        if width is None:
            width = len(key)
        if len(key) != width:
            raise RuntimeError("assignment width changed inside one pool")
        if key not in first_index:
            first_index[key] = int(position)
            unique.append(key)
        dedup.append(int(first_index[key]))
    return unique, first_index, dedup


def _selected_rows(keys: list[tuple[int, ...]], first_index: dict[tuple[int, ...], int], baseline: tuple[int, ...]) -> list[dict]:
    ordered = sorted(keys, key=lambda key: int(first_index[key]))
    rows = []
    for slot, key in enumerate(ordered):
        rows.append(
            {
                "slot": int(slot),
                "pool_index": int(first_index[key]),
                "assignment": [int(item) for item in key],
                "is_baseline": bool(key == baseline),
                "active_mask": [int(item) for item in active_mask(key)],
            }
        )
    if sum(1 for row in rows if row["is_baseline"]) != 1:
        raise RuntimeError("baseline flag was lost while sorting the matched sample")
    if len(rows) > int(CANDIDATE_LIMIT):
        raise RuntimeError("matched sample exceeded the registered cap of five")
    return rows


def select_matched(pool, baseline_index: int, scenario_id: int, decision_time: int) -> dict:
    """Keep C0 and at most four same-mask alternatives. Do not pad and do not read returns."""
    if not pool:
        raise RuntimeError("candidate pool is empty")
    index = int(baseline_index)
    if index < 0 or index >= len(pool):
        raise RuntimeError("baseline index is outside the candidate pool")
    seed = candidate_seed(scenario_id, decision_time)
    unique, first_index, dedup = _dedup(pool)
    baseline = assignment_key(pool[index])
    mask = active_mask(baseline)
    alternatives = [key for key in unique if key != baseline and active_mask(key) == mask]
    alternatives.sort(key=lambda key: int(first_index[key]))
    count = len(alternatives)
    prior_numpy = np.random.get_state()
    layer_sizes: list[int] = []
    offsets: list[int] = []
    stratum_keys: list[tuple[int, ...]] = []
    if count > int(ALTERNATIVE_CAP):
        generator = np.random.default_rng(seed)
        layers = np.array_split(np.arange(count), int(ALTERNATIVE_CAP))
        for layer in layers:
            if len(layer) == 0:
                raise RuntimeError("matched-mask stratum is empty")
            offset = int(generator.integers(0, len(layer)))
            layer_sizes.append(int(len(layer)))
            offsets.append(offset)
            stratum_keys.append(alternatives[int(layer[offset])])
        chosen = stratum_keys
    else:
        chosen = list(alternatives)
        stratum_keys = list(alternatives)
    if not np.array_equal(np.random.get_state()[1], prior_numpy[1]):
        raise RuntimeError("matched-mask selector consumed the global numpy stream")
    if any(active_mask(key) != mask for key in chosen):
        raise RuntimeError("selector returned a different activity mask")
    if len(set(chosen)) != len(chosen):
        raise RuntimeError("selector repeated an alternative")
    selected_keys = [baseline, *chosen]
    return {
        "baseline_key": baseline,
        "baseline_pool_index": index,
        "baseline_mask": [int(item) for item in mask],
        "pool_size": int(len(pool)),
        "unique_count": int(len(unique)),
        "duplicate_count": int(len(pool) - len(unique)),
        "dedup_first_index": dedup,
        "pool_sha256": pool_sha256(pool),
        "matched_alternative_count": int(count),
        "candidate_seed": int(seed),
        "layer_sizes": layer_sizes,
        "stratum_offsets": offsets,
        "stratum_pool_indices": [int(first_index[key]) for key in stratum_keys],
        "selected": _selected_rows(selected_keys, first_index, baseline),
        "no_same_mask_alternative": bool(count == 0),
    }


def activity_census(pool, baseline_index: int) -> dict:
    """Read-only pool inventory. Counts use unique assignments in generation order."""
    unique, first_index, _dedup_index = _dedup(pool)
    baseline = assignment_key(pool[int(baseline_index)])
    mask = active_mask(baseline)
    alternatives = [key for key in unique if key != baseline and active_mask(key) == mask]
    mask_counts: dict[str, int] = {}
    active_counts: dict[str, int] = {}
    for key in unique:
        item_mask = active_mask(key)
        mask_name = "".join(str(bit) for bit in item_mask)
        active_name = str(int(sum(item_mask)))
        mask_counts[mask_name] = mask_counts.get(mask_name, 0) + 1
        active_counts[active_name] = active_counts.get(active_name, 0) + 1
    return {
        "baseline_mask": [int(item) for item in mask],
        "matched_keys": alternatives,
        "matched_alternative_count": len(alternatives),
        "unique_active_count_hist": active_counts,
        "unique_active_mask_hist": mask_counts,
        "first_index": {key: int(first_index[key]) for key in alternatives},
    }


def dedup_sha256(dedup: list[int]) -> str:
    payload = np.asarray(dedup, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def public_record(pool, baseline_index: int, scenario_id: int, decision_time: int) -> dict:
    """Selection list without the full dedup vector. The hash covers this record only."""
    chosen = select_matched(pool, baseline_index, scenario_id, decision_time)
    return {
        "scenario_id": int(scenario_id),
        "decision_time": int(decision_time),
        "candidate_seed": int(chosen["candidate_seed"]),
        "pool_sha256": chosen["pool_sha256"],
        "dedup_sha256": dedup_sha256(chosen["dedup_first_index"]),
        "pool_size": int(chosen["pool_size"]),
        "unique_count": int(chosen["unique_count"]),
        "duplicate_count": int(chosen["duplicate_count"]),
        "baseline_pool_index": int(chosen["baseline_pool_index"]),
        "baseline_assignment": [int(item) for item in chosen["baseline_key"]],
        "baseline_mask": list(chosen["baseline_mask"]),
        "matched_alternative_count": int(chosen["matched_alternative_count"]),
        "no_same_mask_alternative": bool(chosen["no_same_mask_alternative"]),
        "layer_sizes": [int(item) for item in chosen["layer_sizes"]],
        "stratum_offsets": [int(item) for item in chosen["stratum_offsets"]],
        "stratum_pool_indices": [int(item) for item in chosen["stratum_pool_indices"]],
        "selected": chosen["selected"],
    }


def selection_sha256(records: list[dict]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
