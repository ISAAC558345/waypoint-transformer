"""Priority buckets that follow a formal transition without changing it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import AGE_SCALE_STEPS, BETA_TAG, BLOCK_BITS, DRAIN_FAIL_BITS, DUST_BITS, SCHEMA_VERSION, TOL_BITS


class LedgerError(RuntimeError):
    """Raised when tag buckets cannot follow the formal accounting."""


@dataclass
class TagBucket:
    admission_step: int
    age_origin: str
    priority_tag: int
    bits_remaining: float
    insert_index: int


class PriorityTagLedger:
    """Per-buoy FIFO buckets. Tag is stored and is not a service-order key."""

    def __init__(self, num_buoys: int, beta_tag: float = BETA_TAG, age_scale_steps: int = AGE_SCALE_STEPS) -> None:
        if int(age_scale_steps) != int(AGE_SCALE_STEPS):
            raise LedgerError("age scale is fixed at 48")
        self.num_buoys = int(num_buoys)
        self.beta_tag = float(beta_tag)
        self.age_scale_steps = int(age_scale_steps)
        self.schema_version = SCHEMA_VERSION
        self.buckets: list[list[TagBucket]] = [[] for _ in range(self.num_buoys)]
        self._next_insert = 0

    def reset_from_queue(self, q_bits: np.ndarray) -> None:
        values = _vector(q_bits, self.num_buoys)
        self.buckets = [[] for _ in range(self.num_buoys)]
        self._next_insert = 0
        for buoy, bits in enumerate(values):
            if float(bits) > DUST_BITS:
                self.buckets[buoy].append(self._make(0, "reset_convention", 0, float(bits)))

    def view(self, time_step: int) -> dict[str, np.ndarray]:
        """Urgency of buckets that already exist. This does not read arrivals or service."""
        head_u = np.zeros(self.num_buoys, dtype=np.float64)
        head_tag = np.zeros(self.num_buoys, dtype=np.int64)
        valid = np.zeros(self.num_buoys, dtype=np.int64)
        tagged_fraction = np.zeros(self.num_buoys, dtype=np.float64)
        for buoy, buckets in enumerate(self.buckets):
            alive = [bucket for bucket in buckets if float(bucket.bits_remaining) > DUST_BITS]
            total = float(sum(float(bucket.bits_remaining) for bucket in alive))
            tagged = float(sum(float(bucket.bits_remaining) for bucket in alive if int(bucket.priority_tag) == 1))
            if total > DUST_BITS:
                valid[buoy] = 1
                tagged_fraction[buoy] = tagged / total
                head = min(alive, key=self._fifo_key)
                head_u[buoy] = self.urgency(head, int(time_step))
                head_tag[buoy] = int(head.priority_tag)
        return {
            "head_u": head_u,
            "head_tag": head_tag,
            "urgency_valid": valid,
            "tagged_fraction": tagged_fraction,
        }

    def urgency(self, bucket: TagBucket, time_step: int) -> float:
        age = max(0, int(time_step) - int(bucket.admission_step))
        u_age = float(np.clip(age / float(self.age_scale_steps), 0.0, 1.0))
        return float(np.clip(u_age + self.beta_tag * int(bucket.priority_tag), 0.0, 1.0))

    def apply_transition(
        self,
        served_bits: np.ndarray,
        generated_arrival_bits: np.ndarray,
        admitted_arrival_bits: np.ndarray,
        dropped_bits: np.ndarray,
        snap_bits: np.ndarray,
        admission_step: int,
        arrival_tags: list[int | None],
        tol_bits: float = TOL_BITS,
    ) -> dict[str, np.ndarray]:
        """Drain oldest bits, drop only this step's arrival, then apply snap."""
        served = _vector(served_bits, self.num_buoys)
        generated = _vector(generated_arrival_bits, self.num_buoys)
        admitted = _vector(admitted_arrival_bits, self.num_buoys)
        dropped = _vector(dropped_bits, self.num_buoys)
        snap = _vector(snap_bits, self.num_buoys)
        if len(arrival_tags) != self.num_buoys:
            raise LedgerError("arrival tag vector length does not match buoys")
        served_ordinary = np.zeros(self.num_buoys, dtype=np.float64)
        served_tagged = np.zeros(self.num_buoys, dtype=np.float64)
        dropped_ordinary = np.zeros(self.num_buoys, dtype=np.float64)
        dropped_tagged = np.zeros(self.num_buoys, dtype=np.float64)
        admitted_ordinary = np.zeros(self.num_buoys, dtype=np.float64)
        admitted_tagged = np.zeros(self.num_buoys, dtype=np.float64)
        snap_ordinary = np.zeros(self.num_buoys, dtype=np.float64)
        snap_tagged = np.zeros(self.num_buoys, dtype=np.float64)
        urgency_served = np.zeros(self.num_buoys, dtype=np.float64)
        limit = float(tol_bits)
        for buoy in range(self.num_buoys):
            if abs(float(generated[buoy]) - float(admitted[buoy]) - float(dropped[buoy])) > limit:
                raise LedgerError(f"buoy {buoy} generated bits do not split into admitted plus dropped")
            taken = self._drain(buoy, float(served[buoy]), int(admission_step), count_service=True)
            served_ordinary[buoy] = taken["ordinary"]
            served_tagged[buoy] = taken["tagged"]
            urgency_served[buoy] = taken["urgency_served"]
            tag = arrival_tags[buoy]
            if float(generated[buoy]) > DUST_BITS or float(dropped[buoy]) > DUST_BITS:
                if tag not in (0, 1):
                    raise LedgerError(f"buoy {buoy} arrival has no single binary tag")
                if abs(float(generated[buoy]) - float(BLOCK_BITS)) > limit:
                    raise LedgerError(f"buoy {buoy} arrival is not one {BLOCK_BITS:g}-bit block")
                fresh = self._make(int(admission_step), "environment_arrival", int(tag), float(generated[buoy]))
                if float(dropped[buoy]) - float(fresh.bits_remaining) > limit:
                    raise LedgerError(f"buoy {buoy} drop exceeds the new arrival bucket")
                fresh.bits_remaining -= float(dropped[buoy])
                if int(tag) == 1:
                    dropped_tagged[buoy] = float(dropped[buoy])
                    admitted_tagged[buoy] = float(fresh.bits_remaining)
                else:
                    dropped_ordinary[buoy] = float(dropped[buoy])
                    admitted_ordinary[buoy] = float(fresh.bits_remaining)
                if fresh.bits_remaining > DUST_BITS:
                    self.buckets[buoy].append(fresh)
            elif tag is not None:
                raise LedgerError(f"buoy {buoy} drew a tag without an arrival")
            snap_take = self._apply_snap(buoy, float(snap[buoy]), int(admission_step))
            snap_ordinary[buoy] = snap_take["ordinary"]
            snap_tagged[buoy] = snap_take["tagged"]
            self._prune(buoy)
            if abs(float(served_ordinary[buoy] + served_tagged[buoy] - served[buoy])) > limit:
                raise LedgerError(f"buoy {buoy} served tag split does not match served bits")
        return {
            "served_ordinary_bits": served_ordinary,
            "served_tagged_bits": served_tagged,
            "dropped_ordinary_bits": dropped_ordinary,
            "dropped_tagged_bits": dropped_tagged,
            "admitted_ordinary_bits": admitted_ordinary,
            "admitted_tagged_bits": admitted_tagged,
            "snap_ordinary_bits": snap_ordinary,
            "snap_tagged_bits": snap_tagged,
            "urgency_served_bits": urgency_served,
        }

    def bits_sum(self) -> np.ndarray:
        totals = np.zeros(self.num_buoys, dtype=np.float64)
        for buoy, buckets in enumerate(self.buckets):
            totals[buoy] = float(sum(float(bucket.bits_remaining) for bucket in buckets))
        return totals

    def bits_by_tag(self) -> tuple[np.ndarray, np.ndarray]:
        ordinary = np.zeros(self.num_buoys, dtype=np.float64)
        tagged = np.zeros(self.num_buoys, dtype=np.float64)
        for buoy, buckets in enumerate(self.buckets):
            for bucket in buckets:
                if int(bucket.priority_tag) == 1:
                    tagged[buoy] += float(bucket.bits_remaining)
                else:
                    ordinary[buoy] += float(bucket.bits_remaining)
        return ordinary, tagged

    def assert_matches(self, q_bits: np.ndarray, tol_bits: float = TOL_BITS) -> np.ndarray:
        gap = self.bits_sum() - _vector(q_bits, self.num_buoys)
        if float(np.max(np.abs(gap))) > float(tol_bits):
            raise LedgerError(f"ledger drifted from q_bits by {float(np.max(np.abs(gap)))} bits")
        ordinary, tagged = self.bits_by_tag()
        split_gap = ordinary + tagged - self.bits_sum()
        if float(np.max(np.abs(split_gap))) > float(tol_bits):
            raise LedgerError("tag split does not add up to the ledger")
        return gap

    def capture(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "beta_tag": self.beta_tag,
            "age_scale_steps": self.age_scale_steps,
            "next_insert": int(self._next_insert),
            "buckets": [
                [
                    {
                        "admission_step": int(bucket.admission_step),
                        "age_origin": str(bucket.age_origin),
                        "priority_tag": int(bucket.priority_tag),
                        "bits_remaining": float(bucket.bits_remaining),
                        "insert_index": int(bucket.insert_index),
                    }
                    for bucket in buckets
                ]
                for buckets in self.buckets
            ],
        }

    def restore(self, snap: dict) -> None:
        if str(snap["schema_version"]) != self.schema_version:
            raise LedgerError("ledger schema version does not match")
        if int(snap["age_scale_steps"]) != self.age_scale_steps:
            raise LedgerError("ledger age scale does not match")
        restored: list[list[TagBucket]] = []
        for buckets in snap["buckets"]:
            restored.append(
                [
                    TagBucket(
                        int(item["admission_step"]),
                        str(item["age_origin"]),
                        int(item["priority_tag"]),
                        float(item["bits_remaining"]),
                        int(item["insert_index"]),
                    )
                    for item in buckets
                ]
            )
        if len(restored) != self.num_buoys:
            raise LedgerError("ledger restore buoy count does not match")
        self.buckets = restored
        self._next_insert = int(snap["next_insert"])

    def _make(self, admission_step: int, origin: str, tag: int, bits: float) -> TagBucket:
        bucket = TagBucket(int(admission_step), str(origin), int(tag), float(bits), int(self._next_insert))
        self._next_insert += 1
        return bucket

    def _fifo_key(self, bucket: TagBucket) -> tuple[int, int]:
        return (int(bucket.admission_step), int(bucket.insert_index))

    def _drain(self, buoy: int, amount: float, time_step: int, *, count_service: bool) -> dict[str, float]:
        remaining = float(amount)
        ordinary = 0.0
        tagged = 0.0
        urgency_served = 0.0
        if remaining <= DUST_BITS:
            return {"ordinary": 0.0, "tagged": 0.0, "urgency_served": 0.0}
        order = sorted(range(len(self.buckets[buoy])), key=lambda index: self._fifo_key(self.buckets[buoy][index]))
        for index in order:
            if remaining <= DUST_BITS:
                break
            bucket = self.buckets[buoy][index]
            take = min(float(bucket.bits_remaining), remaining)
            if take <= 0.0:
                continue
            weight = self.urgency(bucket, int(time_step))
            bucket.bits_remaining -= take
            remaining -= take
            if int(bucket.priority_tag) == 1:
                tagged += take
            else:
                ordinary += take
            if count_service:
                urgency_served += weight * take
        if remaining > DRAIN_FAIL_BITS:
            raise LedgerError(f"buoy {buoy} could not drain {float(amount)} bits from oldest buckets")
        return {"ordinary": ordinary, "tagged": tagged, "urgency_served": urgency_served}

    def _apply_snap(self, buoy: int, amount: float, admission_step: int) -> dict[str, float]:
        if abs(float(amount)) <= DUST_BITS:
            return {"ordinary": 0.0, "tagged": 0.0}
        if float(amount) > 0.0:
            taken = self._drain(buoy, float(amount), int(admission_step), count_service=False)
            return {"ordinary": taken["ordinary"], "tagged": taken["tagged"]}
        credit = float(-amount)
        if self.buckets[buoy]:
            newest = max(range(len(self.buckets[buoy])), key=lambda index: self._fifo_key(self.buckets[buoy][index]))
            bucket = self.buckets[buoy][newest]
            bucket.bits_remaining += credit
            if int(bucket.priority_tag) == 1:
                return {"ordinary": 0.0, "tagged": -credit}
            return {"ordinary": -credit, "tagged": 0.0}
        self.buckets[buoy].append(self._make(int(admission_step), "snap_credit", 0, credit))
        return {"ordinary": -credit, "tagged": 0.0}

    def _prune(self, buoy: int) -> None:
        self.buckets[buoy] = [bucket for bucket in self.buckets[buoy] if float(bucket.bits_remaining) > DUST_BITS]


def _vector(value, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if int(array.size) != int(count):
        raise LedgerError(f"expected {int(count)} buoy values, received {int(array.size)}")
    return array
