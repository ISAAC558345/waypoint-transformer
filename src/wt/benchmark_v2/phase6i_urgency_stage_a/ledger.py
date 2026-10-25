"""Episode admission-age ledger. It does not call or replace the queue transition."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import AGE_ORIGIN_ARRIVAL, AGE_ORIGIN_RESET, AGE_SCALE_STEPS, DUST_BITS, SCHEMA_VERSION


class LedgerError(RuntimeError):
    """Raised when the age buckets cannot follow the environment's accounting."""


@dataclass
class AgeBucket:
    admission_step: int
    age_origin: str
    bits_remaining: float


class AdmissionAgeLedger:
    """Per-buoy buckets of bits admitted in this episode.

    Reset stock uses admission_step 0 and age_origin reset_convention.
    That age starts at zero when t is 0. It is not a measured wait from before the episode.
    """

    def __init__(self, num_buoys: int, age_scale_steps: int = AGE_SCALE_STEPS) -> None:
        if int(age_scale_steps) != int(AGE_SCALE_STEPS):
            raise LedgerError("age_scale_steps is fixed at the audited episode horizon")
        self.num_buoys = int(num_buoys)
        self.age_scale_steps = int(age_scale_steps)
        self.schema_version = SCHEMA_VERSION
        self.buckets: list[list[AgeBucket]] = [[] for _ in range(self.num_buoys)]

    def reset_from_queue(self, q_bits: np.ndarray) -> None:
        values = _vector(q_bits, self.num_buoys)
        self.buckets = [[] for _ in range(self.num_buoys)]
        for buoy, bits in enumerate(values):
            if float(bits) > DUST_BITS:
                self.buckets[buoy].append(AgeBucket(0, AGE_ORIGIN_RESET, float(bits)))

    def view(self, time_step: int, q_bits: np.ndarray, tol_bits: float) -> dict[str, np.ndarray]:
        """Ages before service. Reading the view does not change any bucket."""
        queue = _vector(q_bits, self.num_buoys)
        oldest = np.zeros(self.num_buoys, dtype=np.float64)
        mean = np.zeros(self.num_buoys, dtype=np.float64)
        backlog = np.zeros(self.num_buoys, dtype=np.float64)
        valid = np.zeros(self.num_buoys, dtype=np.int64)
        urgency = np.zeros(self.num_buoys, dtype=np.float64)
        for buoy, buckets in enumerate(self.buckets):
            weighted = 0.0
            total = 0.0
            oldest_age = 0.0
            for bucket in buckets:
                if float(bucket.bits_remaining) <= DUST_BITS:
                    continue
                age = float(max(0, int(time_step) - int(bucket.admission_step)))
                total += float(bucket.bits_remaining)
                weighted += age * float(bucket.bits_remaining)
                oldest_age = max(oldest_age, age)
            backlog[buoy] = weighted
            if total > DUST_BITS:
                mean[buoy] = weighted / total
                oldest[buoy] = oldest_age
            valid[buoy] = int(float(queue[buoy]) > 0.0)
            if valid[buoy]:
                urgency[buoy] = float(np.clip(oldest[buoy] / float(self.age_scale_steps), 0.0, 1.0))
        return {
            "oldest_wait_age": oldest,
            "mean_wait_age": mean,
            "age_weighted_backlog": backlog,
            "urgency_valid": valid,
            "urgency": urgency,
            "tol_bits": np.asarray([float(tol_bits)], dtype=np.float64),
        }

    def apply_transition(
        self,
        served_bits: np.ndarray,
        generated_arrival_bits: np.ndarray,
        admitted_arrival_bits: np.ndarray,
        dropped_bits: np.ndarray,
        snap_bits: np.ndarray,
        admission_step: int,
        tol_bits: float,
    ) -> None:
        """Drain oldest bits, then admit this step's arrivals, then apply the environment snap."""
        served = _vector(served_bits, self.num_buoys)
        generated = _vector(generated_arrival_bits, self.num_buoys)
        admitted = _vector(admitted_arrival_bits, self.num_buoys)
        dropped = _vector(dropped_bits, self.num_buoys)
        snap = _vector(snap_bits, self.num_buoys)
        limit = float(tol_bits)
        for buoy in range(self.num_buoys):
            if abs(float(generated[buoy]) - float(admitted[buoy]) - float(dropped[buoy])) > limit:
                raise LedgerError(f"buoy {buoy} arrival split does not match admitted plus dropped")
            self._drain(buoy, float(served[buoy]), limit)
            if float(generated[buoy]) > DUST_BITS or float(dropped[buoy]) > DUST_BITS:
                fresh = AgeBucket(int(admission_step), AGE_ORIGIN_ARRIVAL, float(generated[buoy]))
                if float(dropped[buoy]) - float(fresh.bits_remaining) > limit:
                    raise LedgerError(f"buoy {buoy} drop exceeds the new arrival bucket")
                fresh.bits_remaining -= float(dropped[buoy])
                if fresh.bits_remaining > DUST_BITS:
                    self.buckets[buoy].append(fresh)
            if float(snap[buoy]) > DUST_BITS:
                self._drain(buoy, float(snap[buoy]), limit)
            elif float(snap[buoy]) < -DUST_BITS:
                self._credit_newest(buoy, float(-snap[buoy]), int(admission_step))
            self._prune(buoy)

    def bits_sum(self) -> np.ndarray:
        totals = np.zeros(self.num_buoys, dtype=np.float64)
        for buoy, buckets in enumerate(self.buckets):
            totals[buoy] = float(sum(float(bucket.bits_remaining) for bucket in buckets))
        return totals

    def mismatch(self, q_bits: np.ndarray) -> np.ndarray:
        return self.bits_sum() - _vector(q_bits, self.num_buoys)

    def assert_matches(self, q_bits: np.ndarray, tol_bits: float) -> np.ndarray:
        gap = self.mismatch(q_bits)
        if float(np.max(np.abs(gap))) > float(tol_bits):
            raise LedgerError(f"ledger drifted from q_bits by {float(np.max(np.abs(gap)))} bits")
        return gap

    def capture(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "age_scale_steps": self.age_scale_steps,
            "buckets": [
                [
                    {
                        "admission_step": int(bucket.admission_step),
                        "age_origin": str(bucket.age_origin),
                        "bits_remaining": float(bucket.bits_remaining),
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
        restored: list[list[AgeBucket]] = []
        for buckets in snap["buckets"]:
            restored.append(
                [
                    AgeBucket(int(item["admission_step"]), str(item["age_origin"]), float(item["bits_remaining"]))
                    for item in buckets
                ]
            )
        if len(restored) != self.num_buoys:
            raise LedgerError("ledger restore buoy count does not match")
        self.buckets = restored

    def _drain(self, buoy: int, amount: float, _tol_bits: float) -> None:
        remaining = float(amount)
        if remaining <= DUST_BITS:
            return
        order = sorted(range(len(self.buckets[buoy])), key=lambda index: (self.buckets[buoy][index].admission_step, index))
        for index in order:
            if remaining <= DUST_BITS:
                break
            bucket = self.buckets[buoy][index]
            take = min(float(bucket.bits_remaining), remaining)
            bucket.bits_remaining -= take
            remaining -= take
        if remaining > 1.0e-6:
            raise LedgerError(f"buoy {buoy} could not drain {float(amount)} bits from oldest buckets")

    def _credit_newest(self, buoy: int, amount: float, admission_step: int) -> None:
        if self.buckets[buoy]:
            newest = max(range(len(self.buckets[buoy])), key=lambda index: (self.buckets[buoy][index].admission_step, index))
            self.buckets[buoy][newest].bits_remaining += float(amount)
            return
        if float(amount) > DUST_BITS:
            self.buckets[buoy].append(AgeBucket(int(admission_step), AGE_ORIGIN_ARRIVAL, float(amount)))

    def _prune(self, buoy: int) -> None:
        self.buckets[buoy] = [bucket for bucket in self.buckets[buoy] if float(bucket.bits_remaining) > DUST_BITS]


def snap_vector(env, snap_before: np.ndarray, info: dict) -> np.ndarray:
    """Use the snap already produced by this step. Do not recompute arrivals."""
    for key in ("float_snap_bits", "snap_bits"):
        if key in info:
            return _vector(info[key], int(env.config.num_buoys))
    if not hasattr(env, "cumulative_snap_bits"):
        raise LedgerError("float snap is not exposed by the step result or the cumulative counter")
    return _vector(env.cumulative_snap_bits, int(env.config.num_buoys)) - _vector(snap_before, int(env.config.num_buoys))


def _vector(value, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if int(array.size) != int(count):
        raise LedgerError(f"expected {int(count)} buoy values, received {int(array.size)}")
    return array
