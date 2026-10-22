"""Pre-registered fresh diagnostic scenarios for Round-18."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wt.maritime.config import MaritimeEnvConfig

from .multi_env import MultiUAVMaritimeEnv


SCENARIO_NAMESPACE = "round18_multi_uav_planning_diagnostic"
QUEUE_LEVELS = {"low": 7.5e5, "medium": 2.5e6, "high": 4.5e6}
PATTERNS = (
    "compact_symmetric", "compact_asymmetric", "spread_symmetric",
    "spread_asymmetric", "stress_remote_split", "stress_crossing",
    "stress_duplicate",
)


@dataclass(frozen=True)
class MultiUAVScenario:
    scenario_id: str
    seed: int
    num_buoys: int
    queue_tier: str
    pattern: str
    geometry_group: str
    initialization_group: str
    coordination_stress: bool
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(
            num_uavs=2, num_buoys=self.num_buoys, task_semantics_version="v1",
            buoy_drift_mode="static", channel_variation_mode="static",
            episode_horizon=12, terminate_on_task_success=False, seed=self.seed,
            initial_uav_positions_m=self.uav_positions_m,
            initial_queue_bits_by_buoy=self.initial_queue_bits,
            arrival_mode="constant", constant_arrival_bits_per_step=4.5e5,
            multi_uav_min_separation_m=15.0,
        )


def _positions(num_buoys: int, geometry: str) -> tuple[tuple[float, float], ...]:
    compact = ((65.0, 85.0), (100.0, 105.0), (135.0, 85.0), (80.0, 130.0), (120.0, 130.0))
    spread = ((30.0, 40.0), (100.0, 165.0), (170.0, 40.0), (35.0, 165.0), (165.0, 165.0))
    return (compact if geometry == "compact" else spread)[:num_buoys]


def _uavs(pattern: str) -> tuple[tuple[float, float], ...]:
    values = {
        "compact_symmetric": ((85.0, 60.0), (115.0, 60.0)),
        "compact_asymmetric": ((45.0, 160.0), (150.0, 55.0)),
        "spread_symmetric": ((85.0, 95.0), (115.0, 95.0)),
        "spread_asymmetric": ((35.0, 155.0), (165.0, 30.0)),
        # All close starts remain initially feasible under the frozen 15 m rule.
        "stress_remote_split": ((90.0, 80.0), (110.0, 80.0)),
        "stress_crossing": ((55.0, 95.0), (145.0, 95.0)),
        "stress_duplicate": ((90.0, 75.0), (110.0, 75.0)),
    }
    return values[pattern]


def _queues(num_buoys: int, level: float, pattern: str) -> tuple[float, ...]:
    values = np.full(num_buoys, level, dtype=np.float64)
    if pattern in ("compact_asymmetric", "spread_asymmetric"):
        values[0] = min(5.0e6, level * 1.35)
        values[-1] = max(1.0e5, level * 0.55)
    elif pattern == "stress_remote_split":
        values[0] = min(5.0e6, level * 1.25)
        values[-1] = min(5.0e6, level * 1.2)
    elif pattern == "stress_crossing":
        values[0] = min(5.0e6, level * 1.3)
        values[1] = min(5.0e6, level * 1.1)
    elif pattern == "stress_duplicate":
        values[0] = min(5.0e6, level * 1.45)
        values[1:] = np.maximum(1.0e5, values[1:] * 0.70)
    return tuple(float(value) for value in values)


def planning_scenarios() -> tuple[MultiUAVScenario, ...]:
    """Return the frozen 42-scenario M=2 pool without data-dependent selection."""
    rows: list[MultiUAVScenario] = []
    seed = 18000
    for num_buoys in (3, 5):
        for queue_tier, level in QUEUE_LEVELS.items():
            for pattern in PATTERNS:
                geometry = "compact" if pattern.startswith("compact") else "spread" if pattern.startswith("spread") else "compact" if pattern in ("stress_remote_split", "stress_duplicate") else "spread"
                initialization = "symmetric" if pattern.endswith("symmetric") else "asymmetric" if pattern.endswith("asymmetric") else "close" if pattern in ("stress_remote_split", "stress_duplicate") else "far"
                rows.append(MultiUAVScenario(
                    scenario_id=f"r18_k{num_buoys}_{queue_tier}_{pattern}", seed=seed,
                    num_buoys=num_buoys, queue_tier=queue_tier, pattern=pattern,
                    geometry_group=geometry, initialization_group=initialization,
                    coordination_stress=pattern.startswith("stress_"),
                    buoy_positions_m=_positions(num_buoys, geometry),
                    uav_positions_m=_uavs(pattern),
                    initial_queue_bits=_queues(num_buoys, level, pattern),
                ))
                seed += 1
    return tuple(rows)


def materialize_scenario(scenario: MultiUAVScenario) -> tuple[MultiUAVMaritimeEnv, np.ndarray, dict[str, Any]]:
    env = MultiUAVMaritimeEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64))
    observation, info = env.reset(seed=scenario.seed)
    return env, observation, info


def scenario_manifest_rows() -> list[dict[str, Any]]:
    return [
        {
            "namespace": SCENARIO_NAMESPACE, "scenario_id": scenario.scenario_id,
            "seed": scenario.seed, "num_uavs": 2, "num_buoys": scenario.num_buoys,
            "queue_tier": scenario.queue_tier, "pattern": scenario.pattern,
            "geometry_group": scenario.geometry_group,
            "initialization_group": scenario.initialization_group,
            "coordination_stress": scenario.coordination_stress,
            "buoy_positions_m": [list(value) for value in scenario.buoy_positions_m],
            "uav_positions_m": [list(value) for value in scenario.uav_positions_m],
            "initial_queue_bits": list(scenario.initial_queue_bits),
            "buoy_drift_mode": "static", "channel_variation_mode": "static",
        }
        for scenario in planning_scenarios()
    ]


__all__ = ["MultiUAVScenario", "PATTERNS", "QUEUE_LEVELS", "SCENARIO_NAMESPACE", "materialize_scenario", "planning_scenarios", "scenario_manifest_rows"]
