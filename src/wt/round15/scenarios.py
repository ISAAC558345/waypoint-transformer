"""New, self-contained Round-15 planning diagnostic scenarios.

The scenarios do not read any historical trajectory, development-test, or
holdout file.  They vary existing maritime configuration values only; they do
not alter the environment implementation or introduce new physics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv


SCENARIO_SCHEMA = "round15_planning_diagnostic_v1"


@dataclass(frozen=True)
class PlanningScenario:
    scenario_id: str
    seed: int
    num_buoys: int
    difficulty: str
    queue_pressure: str
    geometry: str

    def config(self) -> MaritimeEnvConfig:
        queue_scale = {"low": 0.55e6, "medium": 2.35e6, "high": 4.10e6}[self.queue_pressure]
        arrival = {"low": 0.12e6, "medium": 0.45e6, "high": 0.72e6}[self.queue_pressure]
        area = 120.0 if self.geometry == "compact" else 300.0
        multiplier = 0.85 + 0.30 * np.linspace(0.0, 1.0, self.num_buoys)
        queue = tuple(float(np.clip(queue_scale * item, 0.0, 5.0e6)) for item in multiplier)
        return MaritimeEnvConfig(
            area_width=area,
            area_height=area,
            num_buoys=self.num_buoys,
            episode_horizon=64,
            delta_t=1.0,
            initial_uav_x=area / 2.0,
            initial_uav_y=area / 2.0,
            initial_energy=1000.0,
            randomize_buoys=True,
            buoy_margin=15.0,
            seed=self.seed,
            task_semantics_version="v1",
            queue_capacity_bits=5.0e6,
            initial_queue_bits_by_buoy=queue,
            arrival_mode="constant",
            constant_arrival_bits_per_step=arrival,
            terminate_on_task_success=False,
        )


def planning_scenarios() -> tuple[PlanningScenario, ...]:
    """Return the pre-registered 3/5-buoy x easy/medium/hard diagnostic set."""
    combinations = (
        ("easy", "low", "compact"),
        ("easy", "medium", "spread"),
        ("medium", "medium", "compact"),
        ("medium", "high", "spread"),
        ("hard", "high", "compact"),
        ("hard", "low", "spread"),
    )
    result: list[PlanningScenario] = []
    for buoy_count in (3, 5):
        for index, (difficulty, pressure, geometry) in enumerate(combinations):
            seed = 15000 + buoy_count * 100 + index
            result.append(
                PlanningScenario(
                    scenario_id=(
                        f"round15_planning_diagnostic__seed-{seed:05d}__"
                        f"buoys-{buoy_count}__{difficulty}__{pressure}__{geometry}"
                    ),
                    seed=seed,
                    num_buoys=buoy_count,
                    difficulty=difficulty,
                    queue_pressure=pressure,
                    geometry=geometry,
                )
            )
    return tuple(result)


def materialize_scenario(scenario: PlanningScenario) -> tuple[MaritimeEnv, np.ndarray, Mapping[str, Any]]:
    env = MaritimeEnv(scenario.config())
    observation, info = env.reset(seed=scenario.seed)
    return env, observation, info


def scenario_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in planning_scenarios():
        env, _, info = materialize_scenario(scenario)
        rows.append(
            {
                "schema_version": SCENARIO_SCHEMA,
                **asdict(scenario),
                "config": asdict(scenario.config()),
                "initial_uav_position_m": np.asarray(info["uav_position_m"], dtype=float).tolist(),
                "initial_buoy_positions_m": np.asarray(info["buoy_positions_m"], dtype=float).tolist(),
                "initial_queue_bits": np.asarray(info["raw_queue_bits"], dtype=float).tolist(),
                "arrival_mode": scenario.config().arrival_mode,
                "uses_historical_dataset": False,
                "old_development_test_accessed": False,
                "untouched_holdout_accessed": False,
                "environment_state_dim": env.state_dim,
                "environment_action_dim": env.action_dim,
            }
        )
    return rows


__all__ = [
    "PlanningScenario",
    "SCENARIO_SCHEMA",
    "materialize_scenario",
    "planning_scenarios",
    "scenario_manifest_rows",
]
