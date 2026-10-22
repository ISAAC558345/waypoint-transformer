"""Benchmark-V2 bit queue. The legacy L3-B transition is not modified."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.l3b.env import L3BEnv, active_links, sinr_rates_for_links


def accounting_tolerance(block_bits: float) -> float:
    """Absolute bit tolerance. float64 values near 1e8 do not need a looser bound."""
    return float(max(1.0e-6 * float(block_bits), 1.0e-9))


def apply_service_then_arrival(
    q_before: np.ndarray,
    raw_capacity_bits: np.ndarray,
    generated_arrival_bits: np.ndarray,
    q_max_bits: np.ndarray,
    remainder_bits: np.ndarray,
    block_bits: float,
    tol: float,
) -> dict[str, np.ndarray]:
    """Service the queue that exists before this slot's arrivals, then admit arrivals.

    completed blocks are counted from the buoy remainder. They do not subtract again.
    """
    before = np.maximum(np.asarray(q_before, dtype=np.float64), 0.0)
    raw = np.maximum(np.asarray(raw_capacity_bits, dtype=np.float64), 0.0)
    generated = np.maximum(np.asarray(generated_arrival_bits, dtype=np.float64), 0.0)
    limit = np.maximum(np.asarray(q_max_bits, dtype=np.float64), 0.0)
    served = np.minimum(before, raw)
    after_service = before - served
    space = np.maximum(limit - after_service, 0.0)
    admitted = np.minimum(generated, space)
    dropped = generated - admitted
    updated = after_service + admitted
    snap = np.where(np.abs(updated) <= float(tol), updated, 0.0)
    updated = updated - snap
    if float(np.min(updated)) < -float(tol):
        raise RuntimeError("bit queue became negative")
    completed, remainder = _advance_remainder(
        np.asarray(remainder_bits, dtype=np.float64),
        served,
        float(block_bits),
        float(tol),
    )
    return {
        "q_after": updated,
        "served_bits": served,
        "generated_arrival_bits": generated,
        "admitted_arrival_bits": admitted,
        "dropped_bits": dropped,
        "float_snap_bits": snap,
        "completed_blocks": completed,
        "remainder_bits": remainder,
    }


def _advance_remainder(
    remainder: np.ndarray,
    served: np.ndarray,
    block_bits: float,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray(remainder, dtype=np.float64) + np.asarray(served, dtype=np.float64)
    completed = np.floor(total / float(block_bits))
    rest = total - completed * float(block_bits)
    carry = rest >= float(block_bits) - float(tol)
    completed = completed + carry.astype(np.float64)
    rest = np.where(carry, rest - float(block_bits), rest)
    rest = np.where(np.abs(rest) <= float(tol), 0.0, rest)
    return completed, rest


class L3BBitQueueEnvV2(L3BEnv):
    """Same geometry, channel, and arrival clock as L3-B. Queue state is bits."""

    def _reset_state(self, seed: int) -> None:
        super()._reset_state(seed)
        self.block_bits = float(self.config.block_payload_bits)
        self.tol_bits = accounting_tolerance(self.block_bits)
        self.q_bits = np.asarray(self.queue, dtype=np.float64) * self.block_bits
        self.q_max_bits = np.full(
            self.config.num_buoys,
            float(self.config.queue_capacity_blocks) * self.block_bits,
            dtype=np.float64,
        )
        self.service_remainder_bits = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.completed_block_equivalent = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.cumulative_served_bits = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.cumulative_dropped_bits = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.cumulative_generated_arrival_bits = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.cumulative_snap_bits = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.initial_q_bits = self.q_bits.copy()
        self.legacy_progress_completions = np.zeros(self.config.num_uavs, dtype=np.float64)
        self.legacy_task_progress = self.progress
        self._skip_graph = True
        self._sync_block_view()

    def _sync_block_view(self) -> None:
        self.queue = self.q_bits / self.block_bits
        self.payload_bits = float(np.sum(self.cumulative_served_bits))
        self.dropped = float(np.sum(self.cumulative_dropped_bits) / self.block_bits)

    def step(
        self,
        movement: np.ndarray,
        association: np.ndarray,
        rate_bps_override: np.ndarray | None = None,
    ) -> dict[str, Any]:
        move = np.asarray(movement, dtype=np.float64).reshape(self.config.num_uavs, 2)
        norms = np.linalg.norm(move, axis=-1, keepdims=True)
        move = np.where(norms > 1.0, move / np.clip(norms, 1.0e-9, None), move)
        self.uav_positions = np.clip(
            self.uav_positions + move * self.config.step_length_m,
            [0.0, 0.0],
            [self.config.area_width, self.config.area_height],
        )
        requested = np.asarray(association, dtype=np.int64).reshape(self.config.num_uavs)
        k_count = int(self.config.num_buoys)
        for uav, buoy in enumerate(requested):
            if buoy < -1 or buoy >= k_count:
                requested[uav] = -1
        self.assignment = requested
        for uav in range(self.config.num_uavs):
            target = int(self.assignment[uav])
            if target != int(self.served_k[uav]):
                self.progress[uav] = 0.0
                self.served_k[uav] = target
        self._sync_block_view()
        links = active_links(self)
        sinr, rate = sinr_rates_for_links(self, links)
        if rate_bps_override is not None:
            injected = np.asarray(rate_bps_override, dtype=np.float64).reshape(self.config.num_uavs)
            mask = np.zeros(self.config.num_uavs, dtype=np.float64)
            for uav, _buoy in links:
                mask[int(uav)] = 1.0
            rate = injected * mask
        self.last_sinr = sinr
        self.last_rate = rate
        raw = np.zeros(k_count, dtype=np.float64)
        delta_t = float(self.config.delta_t)
        for uav, buoy in links:
            raw[int(buoy)] += float(rate[uav]) * delta_t
        generated = self.arrivals() * self.block_bits
        accounted = apply_service_then_arrival(
            self.q_bits,
            raw,
            generated,
            self.q_max_bits,
            self.service_remainder_bits,
            self.block_bits,
            self.tol_bits,
        )
        self.q_bits = accounted["q_after"]
        self.service_remainder_bits = accounted["remainder_bits"]
        self.completed_block_equivalent += accounted["completed_blocks"]
        self.cumulative_served_bits += accounted["served_bits"]
        self.cumulative_dropped_bits += accounted["dropped_bits"]
        self.cumulative_generated_arrival_bits += accounted["generated_arrival_bits"]
        self.cumulative_snap_bits += accounted["float_snap_bits"]
        self._update_legacy_progress(links, rate)
        self._sync_block_view()
        self.t += 1
        done = self.t >= int(self.config.episode_horizon)
        info = self.info()
        info["terminated"] = bool(done)
        info["served_bits"] = accounted["served_bits"].copy()
        info["generated_arrival_bits"] = accounted["generated_arrival_bits"].copy()
        info["admitted_arrival_bits"] = accounted["admitted_arrival_bits"].copy()
        info["dropped_bits_step"] = accounted["dropped_bits"].copy()
        info["payload_increment_bits"] = float(np.sum(accounted["served_bits"]))
        info["completed_block_equivalent_increment"] = accounted["completed_blocks"].copy()
        info["legacy_progress_does_not_move_queue"] = True
        return info

    def _update_legacy_progress(self, links: list[tuple[int, int]], rate: np.ndarray) -> None:
        """Diagnostic dwell timer. It does not change Q_bits or payload."""
        ref = max(float(self.reference_rate_bps), 1.0e-9)
        duration = float(self.config.service_slots)
        q_before_progress = self.q_bits.copy()
        payload_before = float(self.payload_bits)
        for uav, _buoy in links:
            increment = (float(rate[uav]) / ref) / duration
            self.progress[uav] += min(increment, 1.0)
            if self.progress[uav] >= 1.0 - 1.0e-9:
                self.progress[uav] = 0.0
                self.legacy_progress_completions[uav] += 1.0
        if not np.array_equal(q_before_progress, self.q_bits) or float(self.payload_bits) != payload_before:
            raise RuntimeError("legacy progress moved the bit queue")

    def info(self) -> dict[str, Any]:
        payload = super().info()
        payload["q_bits"] = self.q_bits.copy()
        payload["queue_blocks_equivalent"] = self.queue.copy()
        payload["queue_blocks"] = self.queue.copy()
        payload["cumulative_payload_bits"] = float(self.payload_bits)
        payload["dropped_bits"] = float(np.sum(self.cumulative_dropped_bits))
        payload["dropped_block_equivalent"] = float(self.dropped)
        payload["service_remainder_bits"] = self.service_remainder_bits.copy()
        payload["completed_block_equivalent"] = self.completed_block_equivalent.copy()
        payload["legacy_task_progress"] = self.progress.copy()
        payload["progress_controls_payload"] = False
        payload["progress_controls_queue"] = False
        return payload

    def conservation_error_bits(self) -> np.ndarray:
        predicted = (
            self.initial_q_bits
            + self.cumulative_generated_arrival_bits
            - self.cumulative_served_bits
            - self.cumulative_dropped_bits
            - self.cumulative_snap_bits
        )
        return self.q_bits - predicted


__all__ = [
    "L3BBitQueueEnvV2",
    "accounting_tolerance",
    "apply_service_then_arrival",
]
