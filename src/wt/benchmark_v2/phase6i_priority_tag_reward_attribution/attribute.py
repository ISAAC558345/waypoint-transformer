"""Split served bits into age urgency and the clipped tag boost.

The allocation reads bucket bits and does not change them. FIFO order is
admission step then insert index, the same key the ledger uses.
"""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import TagBucket
from wt.benchmark_v2.phase6i_priority_tag.protocol import AGE_SCALE_STEPS, BETA_TAG, DUST_BITS

from .protocol import BOOST_SHORTFALL_EPSILON, FULL_SATURATION_EPSILON


def _urgency_parts(bucket: TagBucket, time_step: int, beta_tag: float, age_scale: int) -> tuple[float, float, float, float]:
    age = float(max(0, int(time_step) - int(bucket.admission_step)))
    u_age = float(np.clip(age / float(age_scale), 0.0, 1.0))
    u_combined = float(np.clip(u_age + float(beta_tag) * int(bucket.priority_tag), 0.0, 1.0))
    return age, u_age, u_combined, float(u_combined - u_age)


def allocate_service(
    buckets: list[list[TagBucket]],
    served_bits,
    time_step: int,
    *,
    beta_tag: float = BETA_TAG,
    age_scale: int = AGE_SCALE_STEPS,
) -> dict:
    """Replay the FIFO drain on a shadow copy of the remaining bits."""
    served = np.asarray(served_bits, dtype=np.float64).reshape(-1)
    if int(served.size) != len(buckets):
        raise RuntimeError("served-bit vector does not match the ledger buoys")
    ordinary_by_buoy = np.zeros(len(buckets), dtype=np.float64)
    tagged_by_buoy = np.zeros(len(buckets), dtype=np.float64)
    totals = {
        "age_service_bits": 0.0,
        "age_service_ordinary_bits": 0.0,
        "age_service_tagged_bits": 0.0,
        "tag_increment_service_bits": 0.0,
        "served_ordinary_bits": 0.0,
        "served_tagged_bits": 0.0,
        "tagged_served_below_full_boost_bits": 0.0,
        "tagged_served_fully_saturated_bits": 0.0,
        "ordinary_age_mass": 0.0,
        "tagged_age_mass": 0.0,
        "ordinary_age_max": 0.0,
        "tagged_age_max": 0.0,
        "ordinary_served_steps": 0.0,
        "tagged_served_steps": 0.0,
    }
    for buoy, amount in enumerate(served):
        remaining = float(amount)
        if remaining <= float(DUST_BITS):
            continue
        order = sorted(
            range(len(buckets[buoy])),
            key=lambda index: (int(buckets[buoy][index].admission_step), int(buckets[buoy][index].insert_index)),
        )
        shadow = [float(bucket.bits_remaining) for bucket in buckets[buoy]]
        for index in order:
            if remaining <= float(DUST_BITS):
                break
            take = min(shadow[index], remaining)
            if take <= 0.0:
                continue
            bucket = buckets[buoy][index]
            age, u_age, _u_combined, increment = _urgency_parts(bucket, int(time_step), float(beta_tag), int(age_scale))
            shadow[index] -= take
            remaining -= take
            totals["age_service_bits"] += u_age * take
            if int(bucket.priority_tag) == 1:
                tagged_by_buoy[buoy] += take
                totals["served_tagged_bits"] += take
                totals["age_service_tagged_bits"] += u_age * take
                totals["tag_increment_service_bits"] += increment * take
                totals["tagged_age_mass"] += age * take
                totals["tagged_age_max"] = max(float(totals["tagged_age_max"]), age if take > float(DUST_BITS) else 0.0)
                totals["tagged_served_steps"] += take
                if increment < float(BETA_TAG) - float(BOOST_SHORTFALL_EPSILON):
                    totals["tagged_served_below_full_boost_bits"] += take
                if increment <= float(FULL_SATURATION_EPSILON):
                    totals["tagged_served_fully_saturated_bits"] += take
            else:
                ordinary_by_buoy[buoy] += take
                totals["served_ordinary_bits"] += take
                totals["age_service_ordinary_bits"] += u_age * take
                totals["ordinary_age_mass"] += age * take
                totals["ordinary_age_max"] = max(float(totals["ordinary_age_max"]), age if take > float(DUST_BITS) else 0.0)
                totals["ordinary_served_steps"] += take
    totals["combined_urgency_served_bits"] = float(totals["age_service_bits"] + totals["tag_increment_service_bits"])
    totals["served_ordinary_by_buoy"] = ordinary_by_buoy
    totals["served_tagged_by_buoy"] = tagged_by_buoy
    return totals


def service_origin_split(
    ordinary_before: np.ndarray,
    tagged_before: np.ndarray,
    ordinary_after: np.ndarray,
    tagged_after: np.ndarray,
) -> dict[str, float]:
    """Split a tagged-service change into buoy entry, shared volume, and shared mix."""
    before_o = np.asarray(ordinary_before, dtype=np.float64).reshape(-1)
    before_t = np.asarray(tagged_before, dtype=np.float64).reshape(-1)
    after_o = np.asarray(ordinary_after, dtype=np.float64).reshape(-1)
    after_t = np.asarray(tagged_after, dtype=np.float64).reshape(-1)
    if before_o.shape != after_o.shape or before_t.shape != after_t.shape:
        raise RuntimeError("buoy service vectors do not match")
    served_before = before_o + before_t
    served_after = after_o + after_t
    ratio_before = np.divide(before_t, served_before, out=np.zeros_like(before_t), where=served_before > float(DUST_BITS))
    ratio_after = np.divide(after_t, served_after, out=np.zeros_like(after_t), where=served_after > float(DUST_BITS))
    shared = (served_before > float(DUST_BITS)) & (served_after > float(DUST_BITS))
    entered = (served_before <= float(DUST_BITS)) & (served_after > float(DUST_BITS))
    exited = (served_before > float(DUST_BITS)) & (served_after <= float(DUST_BITS))
    entry_exit = float(np.sum(after_t[entered]) - np.sum(before_t[exited]))
    shared_volume = float(np.sum(ratio_before[shared] * (served_after[shared] - served_before[shared])))
    shared_mix = float(np.sum(served_after[shared] * (ratio_after[shared] - ratio_before[shared])))
    tagged_delta = float(np.sum(after_t - before_t))
    return {
        "tagged_delta_bits": tagged_delta,
        "tagged_delta_buoy_entry_exit": entry_exit,
        "tagged_delta_shared_volume": shared_volume,
        "tagged_delta_shared_mix": shared_mix,
        "tagged_delta_split_gap": float(entry_exit + shared_volume + shared_mix - tagged_delta),
    }
