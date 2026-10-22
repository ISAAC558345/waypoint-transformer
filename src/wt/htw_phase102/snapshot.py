"""Capture and restore L3-B + controller clocks. Does not edit env.step."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv


def capture_env(env: L3BEnv) -> dict[str, Any]:
    return {
        "rng": env.rng.bit_generator.state,
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


def restore_env(env: L3BEnv, snap: dict[str, Any]) -> None:
    env.rng.bit_generator.state = snap["rng"]
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


def capture_controller(pending: int, option: TaskOption, last_arrival: np.ndarray, last_positions: np.ndarray, high_decisions: int) -> dict[str, Any]:
    return {
        "pending": int(pending),
        "option": option.copy(),
        "last_arrival": np.asarray(last_arrival, dtype=np.float32).copy(),
        "last_positions": np.asarray(last_positions, dtype=np.float32).copy(),
        "high_decisions": int(high_decisions),
    }


def restore_controller(ctrl: dict[str, Any]) -> dict[str, Any]:
    return {
        "pending": int(ctrl["pending"]),
        "option": ctrl["option"].copy(),
        "last_arrival": np.asarray(ctrl["last_arrival"], dtype=np.float32).copy(),
        "last_positions": np.asarray(ctrl["last_positions"], dtype=np.float32).copy(),
        "high_decisions": int(ctrl["high_decisions"]),
    }


__all__ = ["capture_controller", "capture_env", "restore_controller", "restore_env"]
