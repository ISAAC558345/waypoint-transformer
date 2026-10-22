"""New isolated physical scenarios for the Round-16 drift diagnostic."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv


SCENARIO_SCHEMA = "round16_dynamic_planning_diagnostic_v1"
_TIER_CONFIG = {
    "static": ("static", 0.0, 0.95, 0.0),
    "low": ("constant", 0.05, 0.95, 0.0),
    "medium": ("constant", 0.10, 0.95, 0.0),
    "high": ("constant", 0.20, 0.95, 0.0),
    "slowly_varying_secondary": ("slowly_varying", 0.10, 0.95, 0.01),
}


@dataclass(frozen=True)
class DynamicPlanningScenario:
    base_scenario_id: str
    scenario_id: str
    seed: int
    num_buoys: int
    difficulty: str
    queue_pressure: str
    geometry: str
    drift_tier: str

    def config(self) -> MaritimeEnvConfig:
        queue_scale = {"low": 0.55e6, "medium": 2.35e6, "high": 4.10e6}[self.queue_pressure]
        arrival = {"low": 0.12e6, "medium": 0.45e6, "high": 0.72e6}[self.queue_pressure]
        area = 120.0 if self.geometry == "compact" else 300.0
        multiplier = 0.85 + 0.30 * np.linspace(0.0, 1.0, self.num_buoys)
        queue = tuple(float(np.clip(queue_scale * item, 0.0, 5.0e6)) for item in multiplier)
        mode, fraction, rho, innovation = _TIER_CONFIG[self.drift_tier]
        return MaritimeEnvConfig(
            area_width=area, area_height=area, num_buoys=self.num_buoys,
            episode_horizon=64, delta_t=1.0, initial_uav_x=area / 2.0,
            initial_uav_y=area / 2.0, initial_energy=1000.0,
            randomize_buoys=True, buoy_margin=15.0, seed=self.seed,
            task_semantics_version="v1", queue_capacity_bits=5.0e6,
            initial_queue_bits_by_buoy=queue, arrival_mode="constant",
            constant_arrival_bits_per_step=arrival, terminate_on_task_success=False,
            buoy_drift_mode=mode, buoy_drift_speed_fraction=fraction,
            buoy_drift_rho=rho, buoy_drift_innovation_fraction=innovation,
        )


def planning_scenarios() -> tuple[DynamicPlanningScenario, ...]:
    combinations = (
        ("easy", "low", "compact"), ("easy", "medium", "spread"),
        ("medium", "medium", "compact"), ("medium", "high", "spread"),
        ("hard", "high", "compact"), ("hard", "low", "spread"),
    )
    result: list[DynamicPlanningScenario] = []
    for buoy_count in (3, 5):
        for index, (difficulty, pressure, geometry) in enumerate(combinations):
            seed = 16000 + (buoy_count - 3) * 100 + index
            base = f"round16_dynamic_planning_diagnostic__base-seed-{seed:05d}__buoys-{buoy_count}__{difficulty}__{pressure}__{geometry}"
            for tier in _TIER_CONFIG:
                result.append(DynamicPlanningScenario(
                    base_scenario_id=base, scenario_id=f"{base}__drift-{tier}", seed=seed,
                    num_buoys=buoy_count, difficulty=difficulty,
                    queue_pressure=pressure, geometry=geometry, drift_tier=tier,
                ))
    return tuple(result)


def materialize_scenario(scenario: DynamicPlanningScenario) -> tuple[MaritimeEnv, np.ndarray, Mapping[str, Any]]:
    env = MaritimeEnv(scenario.config())
    observation, info = env.reset(seed=scenario.seed)
    return env, observation, info


def scenario_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in planning_scenarios():
        env, observation, info = materialize_scenario(scenario)
        rows.append({
            "schema_version": SCENARIO_SCHEMA, **asdict(scenario),
            "config": asdict(scenario.config()), "initial_observation_dim": int(observation.size),
            "initial_uav_position_m": np.asarray(info["uav_position_m"], dtype=float).tolist(),
            "initial_buoy_positions_m": np.asarray(info["buoy_positions_m"], dtype=float).tolist(),
            "initial_buoy_drift_velocity_mps": np.asarray(info["buoy_drift_velocity_mps"], dtype=float).tolist(),
            "initial_queue_bits": np.asarray(info["raw_queue_bits"], dtype=float).tolist(),
            "uses_historical_dataset": False, "old_development_test_accessed": False,
            "untouched_holdout_new_accessed": False, "environment_state_dim": env.state_dim,
            "environment_action_dim": env.action_dim,
        })
    return rows


__all__ = ["DynamicPlanningScenario", "materialize_scenario", "planning_scenarios", "scenario_manifest_rows"]
