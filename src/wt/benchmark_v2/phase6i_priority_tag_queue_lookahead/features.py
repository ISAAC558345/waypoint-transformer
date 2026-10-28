"""Two pre-service queue fractions. They read only buckets already admitted."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger, TagBucket
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_reward.features import assigned_features, fifo_key
from wt.lowlevel_v2.contracts import env_target

from .protocol import ACTOR_FEATURE_COUNT, HEAD_FEATURE_COUNT


def admitted_buckets(buckets: list[TagBucket], time_step: int) -> list[TagBucket]:
    """Buckets accepted at or before the decision. Future admissions are ignored."""
    alive = []
    for bucket in buckets:
        if int(bucket.admission_step) > int(time_step):
            continue
        if float(bucket.bits_remaining) <= float(DUST_BITS):
            continue
        alive.append(bucket)
    alive.sort(key=fifo_key)
    return alive


def queue_fractions(buckets: list[TagBucket], time_step: int) -> tuple[float, float]:
    """Return T/Q and B/Q. A zero queue or a queue without tag 1 returns zeros."""
    alive = admitted_buckets(buckets, int(time_step))
    total = float(sum(float(bucket.bits_remaining) for bucket in alive))
    if total <= 0.0 or not alive:
        return 0.0, 0.0
    tagged = [bucket for bucket in alive if int(bucket.priority_tag) == 1]
    if not tagged:
        return 0.0, 0.0
    first = min(tagged, key=fifo_key)
    tagged_bits = float(sum(float(bucket.bits_remaining) for bucket in tagged))
    before = float(sum(float(bucket.bits_remaining) for bucket in alive if fifo_key(bucket) < fifo_key(first)))
    return tagged_bits / total, before / total


def actor_features(
    ledger: PriorityTagLedger,
    time_step: int,
    association: np.ndarray,
    num_buoys: int,
    *,
    use_lookahead: bool,
) -> np.ndarray:
    """Head features plus two lookahead columns. The control group zeros those columns."""
    head = assigned_features(ledger, int(time_step), association, int(num_buoys))
    extra = np.zeros((int(head.shape[0]), ACTOR_FEATURE_COUNT - HEAD_FEATURE_COUNT), dtype=np.float64)
    if use_lookahead:
        for index, token in enumerate(np.asarray(association, dtype=np.int64).reshape(-1)):
            buoy = int(env_target(int(token), int(num_buoys)))
            if buoy < 0:
                continue
            extra[index] = queue_fractions(ledger.buckets[buoy], int(time_step))
    features = np.concatenate([head, extra], axis=1)
    if int(features.shape[1]) != int(ACTOR_FEATURE_COUNT):
        raise RuntimeError("actor feature width drifted")
    if np.any(features[:, HEAD_FEATURE_COUNT:] < 0.0) or np.any(features[:, HEAD_FEATURE_COUNT:] > 1.0):
        raise RuntimeError("lookahead fractions left [0, 1]")
    return features
