"""Level-3-A environment: UAV-UAV competition with conflict wipe.

No SINR, no cooperative service, no energy. Does not edit L2-A, HGNN, or PPO.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import L3AConfig
from wt.l2a.env import legal_assignment_count


def _cluster_buoys(config: L3AConfig) -> np.ndarray:
    # Three tightly packed hot buoys; three spread cold near buoys; six far.
    hot = np.array(((30.0, 40.0), (32.0, 41.5), (28.0, 38.5)), dtype=np.float64)
    cold_near = np.array(((18.0, 54.0), (36.0, 54.0), (42.0, 28.0)), dtype=np.float64)
    far = np.stack(
        (
            np.linspace(118.0, 142.0, 6),
            np.array((28.0, 40.0, 52.0, 28.0, 40.0, 52.0), dtype=np.float64),
        ),
        axis=-1,
    )
    _ = config
    return np.concatenate((hot, cold_near, far), axis=0)


def _start_uavs(config: L3AConfig) -> np.ndarray:
    xs = np.linspace(27.0, 33.0, config.num_uavs)
    ys = np.full(config.num_uavs, 40.0)
    return np.stack((xs, ys), axis=-1)


def conflict_mask(assignment: np.ndarray, num_buoys: int) -> np.ndarray:
    """True for UAVs that share a buoy with at least one other UAV."""
    assigned = np.asarray(assignment, dtype=np.int64)
    blocked = np.zeros(assigned.shape[0], dtype=bool)
    for buoy in range(int(num_buoys)):
        owners = np.flatnonzero(assigned == buoy)
        if owners.size > 1:
            blocked[owners] = True
    return blocked


def competition_pairs(assignment: np.ndarray, num_buoys: int) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    assigned = np.asarray(assignment, dtype=np.int64)
    for buoy in range(int(num_buoys)):
        owners = np.flatnonzero(assigned == buoy)
        for i in range(owners.size):
            for j in range(i + 1, owners.size):
                pairs.append((int(owners[i]), int(owners[j]), int(buoy)))
    return pairs


class L3AEnv:
    """Solo buoys with explicit competition: overlapping x_mk yields zero service."""

    def __init__(self, config: L3AConfig | None = None) -> None:
        self.config = config or L3AConfig()
        if self.config.num_uavs != 6 or self.config.num_buoys != 12:
            raise ValueError("Level-3-A is fixed at M=6, K=12")
        self.buoy_positions = _cluster_buoys(self.config)
        self._reset_state(self.config.seed)

    def _reset_state(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.uav_positions = _start_uavs(self.config)
        jitter = self.rng.uniform(-2.0, 2.0, size=self.config.num_uavs)
        self.uav_positions[:, 1] = np.clip(self.uav_positions[:, 1] + jitter, 8.0, self.config.area_height - 8.0)
        self.queue = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.queue[: int(self.config.num_hot)] = 4.0
        self.queue[int(self.config.num_hot) : 6] = 1.0
        self.progress = np.zeros(self.config.num_uavs, dtype=np.float64)
        self.served_k = np.full(self.config.num_uavs, -1, dtype=np.int64)
        self.assignment = np.full(self.config.num_uavs, -1, dtype=np.int64)
        self.payload_bits = 0.0
        self.dropped = 0.0
        self.last_blocked = np.zeros(self.config.num_uavs, dtype=bool)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._reset_state(self.config.seed if seed is None else int(seed))
        return self.info()

    def arrivals_at(self, time_step: int) -> np.ndarray:
        incoming = np.zeros(self.config.num_buoys, dtype=np.float64)
        if 4 <= int(time_step) <= 36 and int(time_step) % 2 == 0:
            incoming[6:] = 1.0
        return incoming

    def arrivals(self) -> np.ndarray:
        return self.arrivals_at(self.t)

    def remaining_arrivals(self) -> np.ndarray:
        leftover = np.zeros(self.config.num_buoys, dtype=np.float64)
        for time_step in range(self.t, int(self.config.episode_horizon)):
            leftover += self.arrivals_at(time_step)
        return leftover

    def hot_mask(self) -> np.ndarray:
        mask = np.zeros(self.config.num_buoys, dtype=bool)
        mask[: int(self.config.num_hot)] = True
        return mask

    def info(self) -> dict[str, Any]:
        eta = np.linalg.norm(self.uav_positions[:, None, :] - self.buoy_positions[None, :, :], axis=-1) / max(
            self.config.step_length_m, 1.0e-9
        )
        blocked = conflict_mask(self.assignment, self.config.num_buoys)
        pairs = competition_pairs(self.assignment, self.config.num_buoys)
        return {
            "time_step": int(self.t),
            "num_uavs": int(self.config.num_uavs),
            "num_buoys": int(self.config.num_buoys),
            "uav_positions_m": self.uav_positions.copy(),
            "buoy_positions_m": self.buoy_positions.copy(),
            "queue_blocks": self.queue.copy(),
            "service_progress": self.progress.copy(),
            "assignment": self.assignment.copy(),
            "eta_slots": eta,
            "cumulative_payload_bits": float(self.payload_bits),
            "dropped_blocks": float(self.dropped),
            "hot_buoy_mask": self.hot_mask(),
            "conflict_uav_mask": blocked,
            "conflict_rate": float(np.mean(blocked)) if blocked.size else 0.0,
            "n_conflict_pairs": int(len(pairs)),
            "competition_pairs": pairs,
            "exclusive_ok": bool(len(pairs) == 0),
        }

    def _in_range(self, uav: int, buoy: int) -> bool:
        if buoy < 0:
            return False
        dist = float(np.linalg.norm(self.uav_positions[uav] - self.buoy_positions[buoy]))
        return dist <= float(self.config.service_radius_m)

    def step(self, movement: np.ndarray, association: np.ndarray) -> dict[str, Any]:
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
        blocked = conflict_mask(self.assignment, k_count)
        self.last_blocked = blocked
        for uav in range(self.config.num_uavs):
            target = int(self.assignment[uav])
            if target != int(self.served_k[uav]):
                self.progress[uav] = 0.0
                self.served_k[uav] = target
        incoming = self.arrivals()
        service = np.zeros(self.config.num_buoys, dtype=np.float64)
        duration = float(self.config.service_slots)
        for uav in range(self.config.num_uavs):
            buoy = int(self.assignment[uav])
            if bool(blocked[uav]) or buoy < 0 or not self._in_range(uav, buoy) or self.queue[buoy] <= 1.0e-9:
                continue
            self.progress[uav] += 1.0 / duration
            if self.progress[uav] >= 1.0 - 1.0e-9:
                self.progress[uav] = 0.0
                service[buoy] += 1.0
                self.payload_bits += float(self.config.block_payload_bits)
        residual = self.queue - service
        overflow = np.maximum(0.0, residual + incoming - float(self.config.queue_capacity_blocks))
        self.dropped += float(np.sum(overflow))
        self.queue = np.clip(residual + incoming, 0.0, float(self.config.queue_capacity_blocks))
        self.t += 1
        done = self.t >= int(self.config.episode_horizon)
        info = self.info()
        info["terminated"] = bool(done)
        info["step_service_blocks"] = service
        info["arrivals_blocks"] = incoming
        return info


__all__ = ["L3AEnv", "conflict_mask", "competition_pairs", "legal_assignment_count"]
