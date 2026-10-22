"""Fresh, isolated diagnostic scenarios for the Round-17 channel audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.env import MaritimeEnv


SCENARIO_SCHEMA = "round17_channel_planning_diagnostic_v1"
_TIER = {
    "static": ("static", 0.00),
    "low": ("slow_oscillation", 0.15),
    "medium": ("slow_oscillation", 0.35),
    "high": ("slow_oscillation", 0.55),
}


@dataclass(frozen=True)
class ChannelPlanningScenario:
    base_scenario_id: str
    scenario_id: str
    seed: int
    num_buoys: int
    difficulty: str
    queue_pressure: str
    geometry: str
    channel_tier: str
    phase_relationship: str

    def phases(self) -> tuple[float, ...]:
        rng = np.random.default_rng(self.seed + 17017)
        phases = rng.uniform(-math.pi, math.pi, size=self.num_buoys)
        if self.phase_relationship == "channel_phase_conflict":
            # Pre-registered peak/trough pair: their channel ranking reverses
            # after ten slots under the frozen 24-slot oscillator.
            phases[0], phases[1] = math.pi / 2.0, -math.pi / 2.0
        return tuple(float(value) for value in phases)

    def config(self) -> MaritimeEnvConfig:
        queue_scale = {"low": 0.55e6, "medium": 2.35e6, "high": 4.10e6}[self.queue_pressure]
        arrival = {"low": 0.12e6, "medium": 0.45e6, "high": 0.72e6}[self.queue_pressure]
        area = 120.0 if self.geometry == "compact" else 300.0
        weights = 0.85 + 0.30 * np.linspace(0.0, 1.0, self.num_buoys)
        queue = tuple(float(np.clip(queue_scale * item, 0.0, 5.0e6)) for item in weights)
        mode, amplitude = _TIER[self.channel_tier]
        return MaritimeEnvConfig(
            area_width=area, area_height=area, num_buoys=self.num_buoys,
            episode_horizon=64, delta_t=1.0, initial_uav_x=area / 2.0,
            initial_uav_y=area / 2.0, initial_energy=1000.0,
            randomize_buoys=True, buoy_margin=15.0, seed=self.seed,
            task_semantics_version="v1", queue_capacity_bits=5.0e6,
            initial_queue_bits_by_buoy=queue, arrival_mode="constant",
            constant_arrival_bits_per_step=arrival, terminate_on_task_success=False,
            buoy_drift_mode="static", buoy_drift_speed_fraction=0.0,
            channel_variation_mode=mode, channel_log_amplitude=amplitude,
            channel_period_slots=24,
            channel_frequency_normalization_max_rad_per_slot=2.0 * math.pi / 20.0,
            channel_initial_phase_by_buoy=self.phases(),
        )


def planning_scenarios() -> tuple[ChannelPlanningScenario, ...]:
    combinations = (
        ("easy", "low", "compact"), ("easy", "medium", "spread"),
        ("medium", "medium", "compact"), ("medium", "high", "spread"),
        ("hard", "high", "compact"), ("hard", "low", "spread"),
    )
    result: list[ChannelPlanningScenario] = []
    for buoy_count in (3, 5):
        for index, (difficulty, pressure, geometry) in enumerate(combinations):
            seed = 17000 + (buoy_count - 3) * 100 + index
            base = (
                f"round17_channel_planning_diagnostic__base-seed-{seed:05d}"
                f"__buoys-{buoy_count}__{difficulty}__{pressure}__{geometry}"
            )
            relationship = "channel_phase_conflict" if index % 2 == 0 else "seeded_async"
            for tier in _TIER:
                result.append(ChannelPlanningScenario(
                    base_scenario_id=base, scenario_id=f"{base}__channel-{tier}",
                    seed=seed, num_buoys=buoy_count, difficulty=difficulty,
                    queue_pressure=pressure, geometry=geometry, channel_tier=tier,
                    phase_relationship=relationship,
                ))
    return tuple(result)


def materialize_scenario(scenario: ChannelPlanningScenario) -> tuple[MaritimeEnv, np.ndarray, Mapping[str, Any]]:
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
            "initial_queue_bits": np.asarray(info["raw_queue_bits"], dtype=float).tolist(),
            "initial_channel_phase_rad": list(scenario.phases()),
            "initial_channel_multiplier": np.asarray(info.get("channel_multipliers", np.ones(env.num_buoys)), dtype=float).tolist(),
            "primary_buoy_drift_mode": "static", "uses_historical_dataset": False,
            "old_development_test_accessed": False, "fresh_development_accessed": False,
            "untouched_holdout_new_accessed": False, "environment_state_dim": env.state_dim,
            "environment_action_dim": env.action_dim,
        })
    return rows


__all__ = ["ChannelPlanningScenario", "materialize_scenario", "planning_scenarios", "scenario_manifest_rows"]
