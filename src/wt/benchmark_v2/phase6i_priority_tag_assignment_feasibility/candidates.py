"""Choose at most five unique pool assignments, always keeping the sampled baseline."""
from __future__ import annotations

import hashlib

import numpy as np


def assignment_key(assignment) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))


def pool_sha256(pool) -> str:
    payload = np.asarray([assignment_key(item) for item in pool], dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def select_registered(pool, baseline_index: int, limit: int = 5) -> dict:
    """Keep generation order. Do not pad when the pool is shorter than the cap."""
    if not pool:
        raise RuntimeError("candidate pool is empty")
    index = int(baseline_index)
    if index < 0 or index >= len(pool):
        raise RuntimeError("baseline index is outside the candidate pool")
    baseline = assignment_key(pool[index])
    unique: list[tuple[int, ...]] = []
    first_index: dict[tuple[int, ...], int] = {}
    dedup = []
    for position, assignment in enumerate(pool):
        key = assignment_key(assignment)
        if key not in first_index:
            first_index[key] = int(position)
            unique.append(key)
        dedup.append(int(first_index[key]))
    others = [key for key in unique if key != baseline]
    keep = set(others[: max(int(limit) - 1, 0)])
    keep.add(baseline)
    selected = [key for key in unique if key in keep]
    if len(selected) > int(limit):
        raise RuntimeError("registered candidate cap was exceeded")
    if len(unique) <= int(limit) and selected != unique:
        raise RuntimeError("a short pool was padded or reordered")
    return {
        "baseline_key": baseline,
        "baseline_index": index,
        "pool_size": len(pool),
        "unique_count": len(unique),
        "duplicate_count": len(pool) - len(unique),
        "selected": selected,
        "first_index": {key: int(first_index[key]) for key in selected},
        "dedup_first_index": dedup,
        "truncated": len(unique) > len(selected),
        "pool_sha256": pool_sha256(pool),
    }
