"""Attribute FIFO service to snapshot-known buckets. Snap and drops are not service."""
from __future__ import annotations

from wt.benchmark_v2.phase6i_priority_tag.protocol import DRAIN_FAIL_BITS, DUST_BITS


def _key(bucket: dict) -> tuple[int, int]:
    return (int(bucket["admission_step"]), int(bucket["insert_index"]))


def attribute_served(
    buckets_by_buoy: list[list[dict]],
    served_bits,
    time_step: int,
    known_ids: set[int],
    urgency,
) -> dict[str, float]:
    """Replay the ledger drain on a copy. The live buckets are not modified."""
    known_tagged = 0.0
    known_ordinary = 0.0
    future_tagged = 0.0
    future_ordinary = 0.0
    urgency_served = 0.0
    for buoy, amount in enumerate(served_bits):
        remaining = float(amount)
        if remaining <= float(DUST_BITS):
            continue
        order = sorted(buckets_by_buoy[int(buoy)], key=_key)
        for bucket in order:
            if remaining <= float(DUST_BITS):
                break
            take = min(float(bucket["bits_remaining"]), remaining)
            if take <= 0.0:
                continue
            remaining -= take
            weight = float(urgency(bucket, int(time_step)))
            urgency_served += weight * take
            known = int(bucket["insert_index"]) in known_ids
            tagged = int(bucket["priority_tag"]) == 1
            if known and tagged:
                known_tagged += take
            elif known:
                known_ordinary += take
            elif tagged:
                future_tagged += take
            else:
                future_ordinary += take
        if remaining > float(DRAIN_FAIL_BITS):
            raise RuntimeError(f"buoy {buoy} service could not be attributed to existing buckets")
    return {
        "known_tagged": float(known_tagged),
        "known_ordinary": float(known_ordinary),
        "future_tagged": float(future_tagged),
        "future_ordinary": float(future_ordinary),
        "urgency_served": float(urgency_served),
    }


def copy_buckets(ledger) -> list[list[dict]]:
    copied = []
    for buckets in ledger.buckets:
        copied.append(
            [
                {
                    "admission_step": int(bucket.admission_step),
                    "priority_tag": int(bucket.priority_tag),
                    "bits_remaining": float(bucket.bits_remaining),
                    "insert_index": int(bucket.insert_index),
                }
                for bucket in buckets
            ]
        )
    return copied
