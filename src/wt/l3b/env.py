"""Level-3-B: competition plus SINR coupling. No cooperative service."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import L3BConfig
from wt.l3a.env import competition_pairs, conflict_mask, legal_assignment_count


def _cluster_buoys(config: L3BConfig) -> np.ndarray:
    # Tight near pack so exclusive matching creates strong co-channel interference.
    near = np.array(
        (
            (30.0, 40.0),
            (32.5, 41.0),
            (28.0, 38.5),
            (31.5, 37.0),
            (27.5, 42.0),
            (33.0, 39.0),
        ),
        dtype=np.float64,
    )
    far = np.array(
        (
            (128.0, 40.0),
            (130.5, 41.0),
            (126.0, 38.5),
            (129.5, 37.0),
            (125.5, 42.0),
            (131.0, 39.0),
        ),
        dtype=np.float64,
    )
    _ = config
    return np.concatenate((near, far), axis=0)


def _start_uavs(config: L3BConfig) -> np.ndarray:
    xs = np.linspace(27.0, 33.0, config.num_uavs)
    ys = np.full(config.num_uavs, 40.0)
    return np.stack((xs, ys), axis=-1)


def cluster_id(buoy: int) -> int:
    return 0 if int(buoy) < 6 else 1


def kappa_for(config: L3BConfig, buoy_a: int, buoy_b: int) -> float:
    if cluster_id(buoy_a) == cluster_id(buoy_b):
        return float(config.kappa_same_cluster)
    return float(config.kappa_cross_cluster)


def path_gain(config: L3BConfig, tx: np.ndarray, rx: np.ndarray) -> float:
    dist = max(float(np.linalg.norm(np.asarray(tx) - np.asarray(rx))), float(config.min_link_m))
    return float(dist ** (-float(config.path_loss_exp)))


def active_links(env: "L3BEnv") -> list[tuple[int, int]]:
    blocked = conflict_mask(env.assignment, env.config.num_buoys)
    links: list[tuple[int, int]] = []
    for uav in range(env.config.num_uavs):
        buoy = int(env.assignment[uav])
        if bool(blocked[uav]) or buoy < 0:
            continue
        if not env._in_range(uav, buoy) or env.queue[buoy] <= 1.0e-9:
            continue
        links.append((uav, buoy))
    return links


def sinr_rates_for_links(env: "L3BEnv", links: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    cfg = env.config
    sinr = np.zeros(cfg.num_uavs, dtype=np.float64)
    rate = np.zeros(cfg.num_uavs, dtype=np.float64)
    for uav, buoy in links:
        signal = path_gain(cfg, env.uav_positions[uav], env.buoy_positions[buoy]) * float(cfg.tx_power)
        interference = float(cfg.noise_power)
        for other, other_buoy in links:
            if int(other) == int(uav):
                continue
            gain = path_gain(cfg, env.uav_positions[other], env.buoy_positions[buoy])
            interference += gain * float(cfg.tx_power) * kappa_for(cfg, other_buoy, buoy)
        value = signal / max(interference, 1.0e-18)
        sinr[uav] = value
        rate[uav] = float(cfg.bandwidth_hz) * np.log2(1.0 + value)
    return sinr, rate


def reference_rate(config: L3BConfig) -> float:
    gain = float(config.min_link_m ** (-float(config.path_loss_exp)))
    sinr = (gain * float(config.tx_power)) / max(float(config.noise_power), 1.0e-18)
    return float(config.bandwidth_hz) * np.log2(1.0 + sinr)


def hypothetical_rates(env: "L3BEnv") -> np.ndarray:
    """Rate if UAV m used buoy k, with other UAVs' current exclusive links held fixed."""
    cfg = env.config
    out = np.zeros((cfg.num_uavs, cfg.num_buoys), dtype=np.float64)
    base = active_links(env)
    for uav in range(cfg.num_uavs):
        others = [(other, buoy) for other, buoy in base if int(other) != int(uav)]
        for buoy in range(cfg.num_buoys):
            trial = list(others)
            if env.queue[buoy] > 1.0e-9:
                trial.append((uav, buoy))
            _, rate = sinr_rates_for_links(env, trial)
            out[uav, buoy] = float(rate[uav])
    return out


def interference_matrix(env: "L3BEnv") -> np.ndarray:
    cfg = env.config
    matrix = np.zeros((cfg.num_uavs, cfg.num_uavs), dtype=np.float64)
    links = {uav: buoy for uav, buoy in active_links(env)}
    for i, buoy_i in links.items():
        for j, buoy_j in links.items():
            if int(i) == int(j):
                continue
            gain = path_gain(cfg, env.uav_positions[j], env.buoy_positions[buoy_i])
            matrix[i, j] = gain * float(cfg.tx_power) * kappa_for(cfg, buoy_j, buoy_i)
    return matrix


class L3BEnv:
    def __init__(self, config: L3BConfig | None = None) -> None:
        self.config = config or L3BConfig()
        if self.config.num_uavs != 6 or self.config.num_buoys != 12:
            raise ValueError("Level-3-B is fixed at M=6, K=12")
        self.buoy_positions = _cluster_buoys(self.config)
        self.reference_rate_bps = reference_rate(self.config)
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
        self.last_sinr = np.zeros(self.config.num_uavs, dtype=np.float64)
        self.last_rate = np.zeros(self.config.num_uavs, dtype=np.float64)

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

    def _in_range(self, uav: int, buoy: int) -> bool:
        if buoy < 0:
            return False
        dist = float(np.linalg.norm(self.uav_positions[uav] - self.buoy_positions[buoy]))
        return dist <= float(self.config.service_radius_m)

    def info(self) -> dict[str, Any]:
        eta = np.linalg.norm(self.uav_positions[:, None, :] - self.buoy_positions[None, :, :], axis=-1) / max(
            self.config.step_length_m, 1.0e-9
        )
        blocked = conflict_mask(self.assignment, self.config.num_buoys)
        pairs = competition_pairs(self.assignment, self.config.num_buoys)
        distance_ub = np.linalg.norm(self.uav_positions[:, None, :] - self.buoy_positions[None, :, :], axis=-1)
        distance_uu = np.linalg.norm(self.uav_positions[:, None, :] - self.uav_positions[None, :, :], axis=-1)
        pressure = self.queue / max(float(self.config.queue_capacity_blocks), 1.0e-9)
        skip_graph = bool(getattr(self, "_skip_graph", False))
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
            "conflict_uav_mask": blocked,
            "conflict_rate": float(np.mean(blocked)) if blocked.size else 0.0,
            "n_conflict_pairs": int(len(pairs)),
            "competition_pairs": pairs,
            "exclusive_ok": bool(len(pairs) == 0),
            "sinr": self.last_sinr.copy(),
            "rate_bps": self.last_rate.copy(),
            "distance_uav_buoy": distance_ub,
            "distance_uav_uav": distance_uu,
            "rate_if_assigned": np.zeros((self.config.num_uavs, self.config.num_buoys)) if skip_graph else hypothetical_rates(self),
            "interference_uav_uav": np.zeros((self.config.num_uavs, self.config.num_uavs)) if skip_graph else interference_matrix(self),
            "queue_pressure": pressure,
        }

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
        for uav in range(self.config.num_uavs):
            target = int(self.assignment[uav])
            if target != int(self.served_k[uav]):
                self.progress[uav] = 0.0
                self.served_k[uav] = target
        links = active_links(self)
        sinr, rate = sinr_rates_for_links(self, links)
        self.last_sinr = sinr
        self.last_rate = rate
        incoming = self.arrivals()
        service = np.zeros(self.config.num_buoys, dtype=np.float64)
        duration = float(self.config.service_slots)
        ref = max(float(self.reference_rate_bps), 1.0e-9)
        for uav, buoy in links:
            increment = (float(rate[uav]) / ref) / duration
            self.progress[uav] += min(increment, 1.0)
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


__all__ = [
    "L3BEnv",
    "active_links",
    "conflict_mask",
    "hypothetical_rates",
    "interference_matrix",
    "legal_assignment_count",
    "reference_rate",
    "sinr_rates_for_links",
]
