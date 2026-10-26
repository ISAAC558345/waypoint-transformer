"""Assigned-buoy head features. Tag is not a FIFO key and future arrivals are not read."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger, TagBucket
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.lowlevel_v2.contracts import env_target

from .protocol import AGE_SCALE_STEPS, FEATURE_NAMES


def fifo_key(bucket: TagBucket) -> tuple[int, int]:
    return (int(bucket.admission_step), int(bucket.insert_index))


def head_bucket(ledger: PriorityTagLedger, buoy: int) -> TagBucket | None:
    alive = [bucket for bucket in ledger.buckets[int(buoy)] if float(bucket.bits_remaining) > float(DUST_BITS)]
    if not alive:
        return None
    return min(alive, key=fifo_key)


def head_feature(ledger: PriorityTagLedger, time_step: int, buoy: int) -> np.ndarray:
    """Return age urgency, synthetic tag, and validity. Empty queues stay invalid."""
    head = head_bucket(ledger, int(buoy))
    if head is None:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    age = max(0, int(time_step) - int(head.admission_step))
    age_urgency = float(np.clip(age / float(AGE_SCALE_STEPS), 0.0, 1.0))
    return np.asarray([age_urgency, float(int(head.priority_tag)), 1.0], dtype=np.float64)


def assigned_features(ledger: PriorityTagLedger, time_step: int, association: np.ndarray, num_buoys: int) -> np.ndarray:
    rows = []
    for token in np.asarray(association, dtype=np.int64).reshape(-1):
        buoy = int(env_target(int(token), int(num_buoys)))
        if buoy < 0:
            rows.append(np.zeros(len(FEATURE_NAMES), dtype=np.float64))
        else:
            rows.append(head_feature(ledger, int(time_step), buoy))
    return np.stack(rows, axis=0)


def queue_age(ledger: PriorityTagLedger, time_step: int) -> tuple[float, float]:
    ages: list[float] = []
    weights: list[float] = []
    for buckets in ledger.buckets:
        for bucket in buckets:
            bits = float(bucket.bits_remaining)
            if bits <= float(DUST_BITS):
                continue
            ages.append(float(max(0, int(time_step) - int(bucket.admission_step))))
            weights.append(bits)
    if not weights:
        return 0.0, 0.0
    age = np.asarray(ages, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    return float(np.sum(weight * age) / np.sum(weight)), float(np.max(age))
