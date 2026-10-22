"""Level-2-A environment: duration, arrivals, exclusive association, spatial delay.

This module does not edit Round-19/30/35 physics or PPO.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import L2AConfig


def legal_assignment_count(num_uavs: int, num_buoys: int) -> int:
    total = 0
    choose = 1
    for active in range(0, min(num_uavs, num_buoys) + 1):
        if active > 0:
            choose = choose * (num_uavs - active + 1) // active
        perm = 1
        for index in range(active):
            perm *= num_buoys - index
        total += choose * perm
    return int(total)


def _cluster_buoys(config: L2AConfig) -> np.ndarray:
    left = np.stack(
        (
            np.linspace(18.0, 42.0, 6),
            np.array((28.0, 40.0, 52.0, 28.0, 40.0, 52.0), dtype=np.float64),
        ),
        axis=-1,
    )
    right = np.stack(
        (
            np.linspace(118.0, 142.0, 6),
            np.array((28.0, 40.0, 52.0, 28.0, 40.0, 52.0), dtype=np.float64),
        ),
        axis=-1,
    )
    return np.concatenate((left, right), axis=0)


def _start_uavs(config: L2AConfig) -> np.ndarray:
    xs = np.linspace(20.0, 44.0, config.num_uavs)
    ys = np.full(config.num_uavs, 40.0)
    return np.stack((xs, ys), axis=-1)


def project_exclusive(requested: np.ndarray, positions: np.ndarray, buoys: np.ndarray) -> np.ndarray:
    """Keep at most one UAV per buoy; closer UAV wins ties."""
    assignment = np.asarray(requested, dtype=np.int64).copy()
    k_count = int(buoys.shape[0])
    for uav, buoy in enumerate(assignment):
        if buoy < -1 or buoy >= k_count:
            assignment[uav] = -1
    for buoy in range(k_count):
        owners = np.flatnonzero(assignment == buoy)
        if owners.size <= 1:
            continue
        dist = np.linalg.norm(positions[owners] - buoys[buoy], axis=-1)
        order = np.argsort(dist, kind="stable")
        keep = owners[int(order[0])]
        for uav in owners:
            if int(uav) != int(keep):
                assignment[uav] = -1
    return assignment


class L2AEnv:
    def __init__(self, config: L2AConfig | None = None) -> None:
        self.config = config or L2AConfig()
        if self.config.num_uavs != 5 or self.config.num_buoys != 12:
            raise ValueError("Level-2-A is fixed at M=5, K=12")
        if int(self.config.service_slots) < 3:
            raise ValueError("service duration must be at least 3 slots")
        self.buoy_positions = _cluster_buoys(self.config)
        self._reset_state(self.config.seed)

    def _reset_state(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.uav_positions = _start_uavs(self.config)
        jitter = self.rng.uniform(-3.5, 3.5, size=self.config.num_uavs)
        self.uav_positions[:, 1] = np.clip(self.uav_positions[:, 1] + jitter, 8.0, self.config.area_height - 8.0)
        self.queue = np.zeros(self.config.num_buoys, dtype=np.float64)
        self.queue[:6] = 3.0
        self.progress = np.zeros(self.config.num_uavs, dtype=np.float64)
        self.served_k = np.full(self.config.num_uavs, -1, dtype=np.int64)
        self.assignment = np.full(self.config.num_uavs, -1, dtype=np.int64)
        self.payload_bits = 0.0
        self.dropped = 0.0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._reset_state(self.config.seed if seed is None else int(seed))
        return self.info()

    def arrivals_at(self, time_step: int) -> np.ndarray:
        incoming = np.zeros(self.config.num_buoys, dtype=np.float64)
        if 4 <= int(time_step) <= 36 and int(time_step) % 2 == 0:
            incoming[6:] = 1.0
        return incoming

    def arrivals(self) -> np.ndarray:
        """Current-slot arrivals. Future traces are not written into info()."""
        return self.arrivals_at(self.t)

    def remaining_arrivals(self) -> np.ndarray:
        """Oracle-only remaining arrival mass; not part of the policy observation."""
        leftover = np.zeros(self.config.num_buoys, dtype=np.float64)
        for time_step in range(self.t, int(self.config.episode_horizon)):
            leftover += self.arrivals_at(time_step)
        return leftover

    def info(self) -> dict[str, Any]:
        eta = np.linalg.norm(self.uav_positions[:, None, :] - self.buoy_positions[None, :, :], axis=-1) / max(self.config.step_length_m, 1.0e-9)
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
        }

    def _in_range(self, uav: int, buoy: int) -> bool:
        if buoy < 0:
            return False
        dist = float(np.linalg.norm(self.uav_positions[uav] - self.buoy_positions[buoy]))
        return dist <= float(self.config.service_radius_m)

    def _interference_scale(self) -> np.ndarray:
        scale = np.ones(self.config.num_uavs, dtype=np.float64)
        active = [uav for uav in range(self.config.num_uavs) if self.assignment[uav] >= 0 and self._in_range(uav, int(self.assignment[uav]))]
        for uav in active:
            count = 1
            for other in active:
                if other == uav:
                    continue
                if float(np.linalg.norm(self.uav_positions[uav] - self.uav_positions[other])) <= self.config.interference_radius_m:
                    count += 1
            scale[uav] = 1.0 / float(count)
        return scale

    def step(self, movement: np.ndarray, association: np.ndarray) -> dict[str, Any]:
        move = np.asarray(movement, dtype=np.float64).reshape(self.config.num_uavs, 2)
        norms = np.linalg.norm(move, axis=-1, keepdims=True)
        move = np.where(norms > 1.0, move / np.clip(norms, 1.0e-9, None), move)
        self.uav_positions = np.clip(self.uav_positions + move * self.config.step_length_m, [0.0, 0.0], [self.config.area_width, self.config.area_height])
        requested = np.asarray(association, dtype=np.int64).reshape(self.config.num_uavs)
        self.assignment = project_exclusive(requested, self.uav_positions, self.buoy_positions)
        for uav in range(self.config.num_uavs):
            target = int(self.assignment[uav])
            if target != int(self.served_k[uav]):
                self.progress[uav] = 0.0
                self.served_k[uav] = target
        incoming = self.arrivals()
        service = np.zeros(self.config.num_buoys, dtype=np.float64)
        scale = self._interference_scale()
        duration = float(self.config.service_slots)
        for uav in range(self.config.num_uavs):
            buoy = int(self.assignment[uav])
            if buoy < 0 or not self._in_range(uav, buoy) or self.queue[buoy] <= 1.0e-9:
                continue
            self.progress[uav] += scale[uav] / duration
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


__all__ = ["L2AEnv", "legal_assignment_count", "project_exclusive"]
