"""Frozen fresh Round-19 interference diagnostic scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wt.maritime.config import MaritimeEnvConfig

from .multi_env import SharedSpectrumMultiUAVEnv, SharedSpectrumOptions


SCENARIO_NAMESPACE = "round19_interference_coordination_diagnostic"
QUEUE_LEVELS = {"low": 7.5e5, "medium": 2.5e6, "high": 4.5e6}
INTERFERENCE_TIERS = (("orthogonal", "orthogonal", 0.0), ("low", "shared", 0.25), ("medium", "shared", 0.50), ("full", "shared", 1.0))
PATTERNS = ("compact_close_symmetric", "spread_far_asymmetric", "interference_stress_cross")


@dataclass(frozen=True)
class InterferenceScenario:
    scenario_id: str
    seed: int
    num_buoys: int
    queue_tier: str
    pattern: str
    geometry_group: str
    initialization_group: str
    interference_stress: bool
    spectrum_tier: str
    spectrum_mode: str
    interference_overlap_factor: float
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(num_uavs=2, num_buoys=self.num_buoys, task_semantics_version="v1", buoy_drift_mode="static", channel_variation_mode="static", episode_horizon=12, terminate_on_task_success=False, seed=self.seed, initial_uav_positions_m=self.uav_positions_m, initial_queue_bits_by_buoy=self.initial_queue_bits, arrival_mode="constant", constant_arrival_bits_per_step=4.5e5, multi_uav_min_separation_m=15.0)

    def options(self) -> SharedSpectrumOptions:
        return SharedSpectrumOptions(self.spectrum_mode, self.interference_overlap_factor, "legacy", 0.25)


def _positions(num_buoys: int, pattern: str) -> tuple[tuple[float, float], ...]:
    if pattern == "spread_far_asymmetric":
        values = ((30.0, 40.0), (100.0, 165.0), (170.0, 40.0), (35.0, 165.0), (165.0, 165.0))
    elif pattern == "interference_stress_cross":
        values = ((88.0, 100.0), (112.0, 100.0), (100.0, 126.0), (78.0, 126.0), (122.0, 126.0))
    else:
        values = ((65.0, 85.0), (100.0, 105.0), (135.0, 85.0), (80.0, 130.0), (120.0, 130.0))
    return values[:num_buoys]


def _uavs(pattern: str) -> tuple[tuple[float, float], ...]:
    return {"compact_close_symmetric": ((90.0, 65.0), (110.0, 65.0)), "spread_far_asymmetric": ((35.0, 155.0), (165.0, 30.0)), "interference_stress_cross": ((92.0, 80.0), (108.0, 80.0))}[pattern]


def _queues(num_buoys: int, level: float, pattern: str) -> tuple[float, ...]:
    values = np.full(num_buoys, level, dtype=np.float64)
    if pattern == "spread_far_asymmetric":
        values[0] = min(5.0e6, level * 1.30); values[-1] = max(1.0e5, level * .55)
    elif pattern == "interference_stress_cross":
        values[0] = min(5.0e6, level * 1.10); values[1] = min(5.0e6, level * 1.10)
        if num_buoys > 2: values[2:] *= .60
    return tuple(float(value) for value in values)


def planning_scenarios() -> tuple[InterferenceScenario, ...]:
    """The 24-row protocol covers every registered factor without selection."""
    rows: list[InterferenceScenario] = []; seed = 19000
    pattern_by_queue = {"low": "compact_close_symmetric", "medium": "spread_far_asymmetric", "high": "interference_stress_cross"}
    for count in (3, 5):
        for queue_tier, level in QUEUE_LEVELS.items():
            pattern = pattern_by_queue[queue_tier]
            geometry = "spread" if pattern == "spread_far_asymmetric" else "compact"
            initial = "far_asymmetric" if pattern == "spread_far_asymmetric" else "close_symmetric" if pattern == "compact_close_symmetric" else "close_asymmetric"
            for tier, mode, eta in INTERFERENCE_TIERS:
                rows.append(InterferenceScenario(f"r19_k{count}_{queue_tier}_{pattern}_eta_{tier}", seed, count, queue_tier, pattern, geometry, initial, pattern == "interference_stress_cross", tier, mode, eta, _positions(count, pattern), _uavs(pattern), _queues(count, level, pattern)))
                seed += 1
    return tuple(rows)


def materialize_scenario(scenario: InterferenceScenario) -> tuple[SharedSpectrumMultiUAVEnv, np.ndarray, dict[str, Any]]:
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    observation, info = env.reset(seed=scenario.seed)
    return env, observation, info


def scenario_manifest_rows() -> list[dict[str, Any]]:
    return [{"namespace": SCENARIO_NAMESPACE, "scenario_id": item.scenario_id, "seed": item.seed, "num_uavs": 2, "num_buoys": item.num_buoys, "queue_tier": item.queue_tier, "pattern": item.pattern, "geometry_group": item.geometry_group, "initialization_group": item.initialization_group, "interference_stress": item.interference_stress, "spectrum_tier": item.spectrum_tier, "spectrum_mode": item.spectrum_mode, "interference_overlap_factor": item.interference_overlap_factor, "buoy_positions_m": [list(value) for value in item.buoy_positions_m], "uav_positions_m": [list(value) for value in item.uav_positions_m], "initial_queue_bits": list(item.initial_queue_bits), "buoy_drift_mode": "static", "channel_variation_mode": "static"} for item in planning_scenarios()]


__all__ = ["INTERFERENCE_TIERS", "InterferenceScenario", "PATTERNS", "QUEUE_LEVELS", "SCENARIO_NAMESPACE", "materialize_scenario", "planning_scenarios", "scenario_manifest_rows"]
