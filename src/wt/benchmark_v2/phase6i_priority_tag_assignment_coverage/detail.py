"""Per-bucket service rows. Drops and ledger snaps are not successful transmission."""
from __future__ import annotations

from wt.benchmark_v2.phase6i_priority_tag.protocol import DRAIN_FAIL_BITS, DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.attribution import attribute_served

from .protocol import BUCKET_TOL


def _key(bucket: dict) -> tuple[int, int]:
    return (int(bucket["admission_step"]), int(bucket["insert_index"]))


def bucket_service_rows(buckets_by_buoy, served_bits, time_step: int, known_ids: set[int]) -> list[dict]:
    """Replay the FIFO drain on the supplied copy and name each taken bucket."""
    rows = []
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
            known = int(bucket["insert_index"]) in known_ids
            tagged = int(bucket["priority_tag"]) == 1
            if known and tagged:
                kind = "known_tagged"
            elif known:
                kind = "known_ordinary"
            elif tagged:
                kind = "future_tagged"
            else:
                kind = "future_ordinary"
            rows.append(
                {
                    "step_t": int(time_step),
                    "buoy": int(buoy),
                    "insert_index": int(bucket["insert_index"]),
                    "admission_step": int(bucket["admission_step"]),
                    "priority_tag": int(bucket["priority_tag"]),
                    "known_at_snapshot": bool(known),
                    "bits_taken": float(take),
                    "service_class": kind,
                }
            )
        if remaining > float(DRAIN_FAIL_BITS):
            raise RuntimeError(f"buoy {buoy} service could not be attributed to existing buckets")
    return rows


def detail_matches_aggregate(rows: list[dict], served_bits, buckets_by_buoy, time_step: int, known_ids: set[int], urgency) -> bool:
    totals = attribute_served(buckets_by_buoy, served_bits, time_step, known_ids, urgency)
    summed = {name: 0.0 for name in ("known_tagged", "known_ordinary", "future_tagged", "future_ordinary")}
    for row in rows:
        summed[str(row["service_class"])] += float(row["bits_taken"])
    return all(abs(summed[name] - float(totals[name])) <= float(BUCKET_TOL) for name in summed)
