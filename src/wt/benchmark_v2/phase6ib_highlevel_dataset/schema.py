"""Current-state tensors. Labels and future realizations are not stored here."""
from __future__ import annotations

import hashlib

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry, link_rates
from wt.benchmark_v2.phase6e.generators import feature_matrices


def state_arrays(env: L3BBitQueueEnvV2) -> dict[str, np.ndarray]:
    """Observables already exposed by Benchmark-V2. No future queue, SINR, or served bits."""
    features = feature_matrices(env)
    geo = geometry(env)
    positions = np.asarray(env.uav_positions, dtype=np.float64)
    delta = positions[:, None, :] - positions[None, :, :]
    association = np.asarray(env.assignment, dtype=np.int64)
    _sinr, rate, mean_interference = link_rates(
        geo["gain"],
        association,
        env.queue,
        geo["in_range"],
        bandwidth_hz=float(env.config.bandwidth_hz),
        tx_power=float(env.config.tx_power),
        noise_power=float(env.config.noise_power),
        kappa_same=float(env.config.kappa_same_cluster),
        kappa_cross=float(env.config.kappa_cross_cluster),
    )
    return {
        "uav_position_m": positions,
        "buoy_position_m": np.asarray(env.buoy_positions, dtype=np.float64),
        "q_bits": np.asarray(env.q_bits, dtype=np.float64),
        "association": association,
        "uav_uav_distance_m": np.linalg.norm(delta, axis=-1),
        "uav_buoy_distance_m": np.asarray(features["distance"], dtype=np.float64),
        "uav_buoy_gain": np.asarray(features["gain"], dtype=np.float64),
        "uav_buoy_hypothetical_rate": np.asarray(features["rate"], dtype=np.float64),
        "uav_buoy_in_range": np.asarray(features["in_range"], dtype=np.uint8),
        "current_sinr": np.asarray(_sinr, dtype=np.float64),
        "current_rate_bps": np.asarray(rate, dtype=np.float64),
        "current_mean_interference": np.asarray([mean_interference], dtype=np.float64),
        "time_index": np.asarray([int(env.t)], dtype=np.int64),
    }


def state_hash(scenario_id: int, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(scenario_id)).encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()
