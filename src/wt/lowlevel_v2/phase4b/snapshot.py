"""Env snapshot restore. Does not change L3-B physics."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv


def snapshot_env(env: L3BEnv) -> dict:
    return {
        "t": int(env.t),
        "uav_positions": np.asarray(env.uav_positions, dtype=np.float64).copy(),
        "queue": np.asarray(env.queue, dtype=np.float64).copy(),
        "progress": np.asarray(env.progress, dtype=np.float64).copy(),
        "served_k": np.asarray(env.served_k, dtype=np.int64).copy(),
        "assignment": np.asarray(env.assignment, dtype=np.int64).copy(),
        "payload_bits": float(env.payload_bits),
        "dropped": float(env.dropped),
        "last_sinr": np.asarray(env.last_sinr, dtype=np.float64).copy(),
        "last_rate": np.asarray(env.last_rate, dtype=np.float64).copy(),
    }


def restore_env(env: L3BEnv, snap: dict) -> None:
    env.t = int(snap["t"])
    env.uav_positions = np.asarray(snap["uav_positions"], dtype=np.float64).copy()
    env.queue = np.asarray(snap["queue"], dtype=np.float64).copy()
    env.progress = np.asarray(snap["progress"], dtype=np.float64).copy()
    env.served_k = np.asarray(snap["served_k"], dtype=np.int64).copy()
    env.assignment = np.asarray(snap["assignment"], dtype=np.int64).copy()
    env.payload_bits = float(snap["payload_bits"])
    env.dropped = float(snap["dropped"])
    env.last_sinr = np.asarray(snap["last_sinr"], dtype=np.float64).copy()
    env.last_rate = np.asarray(snap["last_rate"], dtype=np.float64).copy()


__all__ = ["restore_env", "snapshot_env"]
