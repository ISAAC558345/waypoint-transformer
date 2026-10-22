"""Deterministic Benchmark-V2 reference assignments. No learned policy."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2

from .channel import geometry, link_rates, served_bits

EPS = 1.0e-9
SCORE_TIE_BITS = 1.0e-4


def random_legal(env: L3BBitQueueEnvV2, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(env.config.num_buoys)
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    for uav, buoy in enumerate(order[: env.config.num_uavs]):
        assignment[uav] = int(buoy)
    return assignment


def legalize(score: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """UAV-index priority. A lower index keeps a contested buoy; the other takes its next score."""
    n_uavs, n_buoys = score.shape
    ranking = np.argsort(-np.asarray(score, dtype=np.float64), axis=1, kind="mergesort")
    raw = np.full(n_uavs, -1, dtype=np.int64)
    for uav in range(n_uavs):
        for buoy in ranking[uav]:
            if float(score[uav, buoy]) <= 0.0:
                break
            raw[uav] = int(buoy)
            break
    taken: set[int] = set()
    legal = np.full(n_uavs, -1, dtype=np.int64)
    for uav in range(n_uavs):
        for buoy in ranking[uav]:
            buoy_id = int(buoy)
            if float(score[uav, buoy_id]) <= 0.0 or buoy_id in taken:
                continue
            legal[uav] = buoy_id
            taken.add(buoy_id)
            break
    seen: set[int] = set()
    duplicate_count = 0
    for buoy in raw:
        if int(buoy) < 0:
            continue
        if int(buoy) in seen:
            duplicate_count += 1
        seen.add(int(buoy))
    return legal, raw, int(duplicate_count)


def nearest_scores(env: L3BBitQueueEnvV2) -> np.ndarray:
    distance = np.linalg.norm(env.uav_positions[:, None, :] - env.buoy_positions[None, :, :], axis=-1)
    score = np.zeros((env.config.num_uavs, env.config.num_buoys), dtype=np.float64)
    available = env.q_bits > EPS
    score[:, available] = 1.0 / (distance[:, available] + 1.0)
    return score


def max_queue_scores(env: L3BBitQueueEnvV2) -> np.ndarray:
    score = np.zeros((env.config.num_uavs, env.config.num_buoys), dtype=np.float64)
    available = env.q_bits > EPS
    score[:, available] = env.q_bits[available]
    return score


def queue_rate_utility(q_bits: np.ndarray, rate: np.ndarray) -> np.ndarray:
    q_scale = float(np.max(q_bits)) + EPS
    r_scale = float(np.max(rate)) + EPS
    return (q_bits[None, :] / q_scale) * (rate / r_scale)


def _channel_kwargs(env: L3BBitQueueEnvV2, *, kappa_same: float | None = None, kappa_cross: float | None = None) -> dict:
    cfg = env.config
    return {
        "bandwidth_hz": float(cfg.bandwidth_hz),
        "tx_power": float(cfg.tx_power),
        "noise_power": float(cfg.noise_power),
        "kappa_same": float(cfg.kappa_same_cluster if kappa_same is None else kappa_same),
        "kappa_cross": float(cfg.kappa_cross_cluster if kappa_cross is None else kappa_cross),
    }


def evaluate_assignment(
    env: L3BBitQueueEnvV2,
    assignment: np.ndarray,
    geo: dict[str, np.ndarray] | None = None,
    *,
    kappa_same: float | None = None,
    kappa_cross: float | None = None,
    gain: np.ndarray | None = None,
    q_bits: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    view = geo if geo is not None else geometry(env)
    bits = env.q_bits if q_bits is None else np.asarray(q_bits, dtype=np.float64)
    blocks = bits / float(env.block_bits)
    used_gain = view["gain"] if gain is None else gain
    sinr, rate, interference = link_rates(
        used_gain,
        assignment,
        blocks,
        view["in_range"],
        **_channel_kwargs(env, kappa_same=kappa_same, kappa_cross=kappa_cross),
    )
    delivered, per_buoy = served_bits(rate, assignment, bits, float(env.config.delta_t))
    return {
        "sinr": sinr,
        "rate_bps": rate,
        "served_bits": delivered,
        "served_per_buoy": per_buoy,
        "capacity_bits": float(np.sum(rate) * float(env.config.delta_t)),
        "mean_sinr": float(np.mean(sinr)),
        "mean_interference": float(interference),
    }


def one_step_optima(
    env: L3BBitQueueEnvV2,
    geo: dict[str, np.ndarray] | None = None,
    *,
    kappa_same: float | None = None,
    kappa_cross: float | None = None,
    gain: np.ndarray | None = None,
    q_bits: np.ndarray | None = None,
    in_range: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Exact one-step optimum over exclusive in-range assignments.

    An out-of-range or empty-queue assignment is not an active link, so it
    matches idle for both objectives. Duplicate targets are illegal.
    """
    view = geo if geo is not None else geometry(env)
    bits = np.asarray(env.q_bits if q_bits is None else q_bits, dtype=np.float64)
    blocks = bits / float(env.block_bits)
    usable_gain = np.asarray(view["gain"] if gain is None else gain, dtype=np.float64)
    usable_range = np.asarray(view["in_range"] if in_range is None else in_range, dtype=bool)
    kwargs = _channel_kwargs(env, kappa_same=kappa_same, kappa_cross=kappa_cross)
    delta_t = float(env.config.delta_t)
    options: list[list[int]] = []
    for uav in range(env.config.num_uavs):
        choices = [
            int(buoy)
            for buoy in range(env.config.num_buoys)
            if bool(usable_range[uav, buoy]) and float(blocks[buoy]) > 1.0e-9
        ]
        options.append(choices)
    current = np.full(env.config.num_uavs, -1, dtype=np.int64)
    used = np.zeros(env.config.num_buoys, dtype=bool)
    best_bit = -1.0
    best_capacity = -1.0
    best_bit_assignment = current.copy()
    best_capacity_assignment = current.copy()
    leaves = 0

    def consider() -> None:
        nonlocal best_bit, best_capacity, best_bit_assignment, best_capacity_assignment, leaves
        leaves += 1
        _sinr, rate, _interference = link_rates(usable_gain, current, blocks, usable_range, **kwargs)
        capacity = float(np.sum(rate) * delta_t)
        delivered, _per_buoy = served_bits(rate, current, bits, delta_t)
        identity = tuple(int(item) for item in current)
        if delivered > best_bit + SCORE_TIE_BITS or (
            abs(delivered - best_bit) <= SCORE_TIE_BITS and identity < tuple(int(item) for item in best_bit_assignment)
        ):
            best_bit = float(delivered)
            best_bit_assignment = current.copy()
        if capacity > best_capacity + SCORE_TIE_BITS or (
            abs(capacity - best_capacity) <= SCORE_TIE_BITS and identity < tuple(int(item) for item in best_capacity_assignment)
        ):
            best_capacity = float(capacity)
            best_capacity_assignment = current.copy()

    def rec(uav: int) -> None:
        if uav == env.config.num_uavs:
            consider()
            return
        current[uav] = -1
        rec(uav + 1)
        for buoy in options[uav]:
            if bool(used[buoy]):
                continue
            used[buoy] = True
            current[uav] = int(buoy)
            rec(uav + 1)
            used[buoy] = False

    rec(0)
    return {
        "true_bit_assignment": best_bit_assignment,
        "capacity_assignment": best_capacity_assignment,
        "true_bit_served_bits": float(best_bit),
        "capacity_bits": float(best_capacity),
        "leaves": int(leaves),
    }


__all__ = [
    "evaluate_assignment",
    "legalize",
    "max_queue_scores",
    "nearest_scores",
    "one_step_optima",
    "queue_rate_utility",
    "random_legal",
]
