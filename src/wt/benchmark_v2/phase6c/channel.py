"""Fast SINR evaluator. It follows L3BEnv formulas and does not change them."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv, active_links, hypothetical_rates, sinr_rates_for_links


def geometry(env: L3BEnv) -> dict[str, np.ndarray]:
    cfg = env.config
    delta = env.uav_positions[:, None, :] - env.buoy_positions[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    clipped = np.maximum(dist, float(cfg.min_link_m))
    gain = clipped ** (-float(cfg.path_loss_exp))
    in_range = dist <= float(cfg.service_radius_m)
    return {"distance": dist, "gain": gain, "in_range": in_range}


def link_rates(
    gain: np.ndarray,
    assignment: np.ndarray,
    queue_blocks: np.ndarray,
    in_range: np.ndarray,
    *,
    bandwidth_hz: float,
    tx_power: float,
    noise_power: float,
    kappa_same: float,
    kappa_cross: float,
    ignore_range: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return per-UAV SINR, rate in bit/s, and mean interference above noise."""
    chosen = np.asarray(assignment, dtype=np.int64)
    rate = np.zeros(chosen.shape[0], dtype=np.float64)
    sinr = np.zeros(chosen.shape[0], dtype=np.float64)
    active: list[tuple[int, int]] = []
    counts: dict[int, int] = {}
    for uav, buoy in enumerate(chosen):
        buoy_id = int(buoy)
        if buoy_id < 0:
            continue
        if (not ignore_range) and (not bool(in_range[uav, buoy_id])):
            continue
        if float(queue_blocks[buoy_id]) <= 1.0e-9:
            continue
        counts[buoy_id] = counts.get(buoy_id, 0) + 1
        active.append((uav, buoy_id))
    active = [(uav, buoy) for uav, buoy in active if counts[buoy] == 1]
    interference_sum = 0.0
    for uav, buoy in active:
        signal = float(gain[uav, buoy]) * float(tx_power)
        interference = float(noise_power)
        buoy_cluster = 0 if buoy < 6 else 1
        for other, other_buoy in active:
            if other == uav:
                continue
            other_cluster = 0 if other_buoy < 6 else 1
            kappa = float(kappa_same) if other_cluster == buoy_cluster else float(kappa_cross)
            added = float(gain[other, buoy]) * float(tx_power) * kappa
            interference += added
            interference_sum += added
        value = signal / max(interference, 1.0e-18)
        sinr[uav] = value
        rate[uav] = float(bandwidth_hz) * np.log2(1.0 + value)
    mean_interference = interference_sum / max(len(active), 1)
    return sinr, rate, float(mean_interference)


def served_bits(
    rate_bps: np.ndarray,
    assignment: np.ndarray,
    q_bits: np.ndarray,
    delta_t: float,
) -> tuple[float, np.ndarray]:
    per_buoy = np.zeros(q_bits.shape[0], dtype=np.float64)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0 or float(rate_bps[uav]) <= 0.0:
            continue
        per_buoy[int(buoy)] += float(rate_bps[uav]) * float(delta_t)
    delivered = np.minimum(np.maximum(q_bits, 0.0), per_buoy)
    return float(np.sum(delivered)), delivered


def rates_match_environment(env: L3BEnv, assignment: np.ndarray, geo: dict[str, np.ndarray] | None = None) -> bool:
    """Used by tests. The rollout does not need it."""
    snapshot = np.asarray(env.assignment, dtype=np.int64).copy()
    env.assignment = np.asarray(assignment, dtype=np.int64).copy()
    try:
        links = active_links(env)
        sinr_env, rate_env = sinr_rates_for_links(env, links)
    finally:
        env.assignment = snapshot
    view = geo if geo is not None else geometry(env)
    cfg = env.config
    sinr, rate, _interference = link_rates(
        view["gain"],
        assignment,
        env.queue,
        view["in_range"],
        bandwidth_hz=float(cfg.bandwidth_hz),
        tx_power=float(cfg.tx_power),
        noise_power=float(cfg.noise_power),
        kappa_same=float(cfg.kappa_same_cluster),
        kappa_cross=float(cfg.kappa_cross_cluster),
    )
    return bool(np.allclose(sinr, sinr_env) and np.allclose(rate, rate_env))


def hypothetical_match(env: L3BEnv) -> bool:
    cfg = env.config
    view = geometry(env)
    base_sinr, base_rate, _noise = link_rates(
        view["gain"],
        env.assignment,
        env.queue,
        view["in_range"],
        bandwidth_hz=float(cfg.bandwidth_hz),
        tx_power=float(cfg.tx_power),
        noise_power=float(cfg.noise_power),
        kappa_same=float(cfg.kappa_same_cluster),
        kappa_cross=float(cfg.kappa_cross_cluster),
    )
    _ = base_sinr, base_rate
    env._skip_graph = False
    try:
        reference = hypothetical_rates(env)
    finally:
        env._skip_graph = True
    fast = hypothetical_rate(env, view)
    return bool(np.allclose(reference, fast))


def explicit_link_rates(
    gain: np.ndarray,
    links: list[tuple[int, int]],
    n_uavs: int,
    *,
    bandwidth_hz: float,
    tx_power: float,
    noise_power: float,
    kappa_same: float,
    kappa_cross: float,
) -> np.ndarray:
    """SINR over an explicit link list. No range check and no conflict wipe."""
    rate = np.zeros(int(n_uavs), dtype=np.float64)
    for uav, buoy in links:
        signal = float(gain[uav, buoy]) * float(tx_power)
        interference = float(noise_power)
        for other, other_buoy in links:
            if int(other) == int(uav):
                continue
            same = (int(buoy) < 6) == (int(other_buoy) < 6)
            kappa = float(kappa_same) if same else float(kappa_cross)
            interference += float(gain[other, buoy]) * float(tx_power) * kappa
        value = signal / max(interference, 1.0e-18)
        rate[int(uav)] = float(bandwidth_hz) * np.log2(1.0 + value)
    return rate


def hypothetical_rate(env: L3BEnv, geo: dict[str, np.ndarray]) -> np.ndarray:
    """Match hypothetical_rates: other active links stay, the trial link ignores range."""
    cfg = env.config
    out = np.zeros((cfg.num_uavs, cfg.num_buoys), dtype=np.float64)
    current = np.asarray(env.assignment, dtype=np.int64)
    _sinr, _rate, _interference = link_rates(
        geo["gain"],
        current,
        env.queue,
        geo["in_range"],
        bandwidth_hz=float(cfg.bandwidth_hz),
        tx_power=float(cfg.tx_power),
        noise_power=float(cfg.noise_power),
        kappa_same=float(cfg.kappa_same_cluster),
        kappa_cross=float(cfg.kappa_cross_cluster),
    )
    base: list[tuple[int, int]] = []
    counts: dict[int, int] = {}
    for uav, buoy in enumerate(current):
        buoy_id = int(buoy)
        if buoy_id < 0:
            continue
        counts[buoy_id] = counts.get(buoy_id, 0) + 1
    for uav, buoy in enumerate(current):
        buoy_id = int(buoy)
        if buoy_id < 0 or counts.get(buoy_id, 0) != 1:
            continue
        if not bool(geo["in_range"][uav, buoy_id]) or float(env.queue[buoy_id]) <= 1.0e-9:
            continue
        base.append((uav, buoy_id))
    kwargs = {
        "bandwidth_hz": float(cfg.bandwidth_hz),
        "tx_power": float(cfg.tx_power),
        "noise_power": float(cfg.noise_power),
        "kappa_same": float(cfg.kappa_same_cluster),
        "kappa_cross": float(cfg.kappa_cross_cluster),
    }
    for uav in range(cfg.num_uavs):
        others = [(other, buoy) for other, buoy in base if int(other) != int(uav)]
        for buoy in range(cfg.num_buoys):
            if float(env.queue[buoy]) <= 1.0e-9:
                continue
            rate = explicit_link_rates(geo["gain"], [*others, (uav, int(buoy))], cfg.num_uavs, **kwargs)
            out[uav, buoy] = float(rate[uav])
    return out


__all__ = [
    "geometry",
    "hypothetical_match",
    "hypothetical_rate",
    "link_rates",
    "rates_match_environment",
    "served_bits",
]
