"""Pre-registered fresh policy-development scenario metadata for Round-22.

This module deliberately materializes configuration only.  It never builds an
environment or reads a rollout outcome, so registration is not fresh-dev
access.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round19.multi_env import SharedSpectrumOptions


FRESH_NAMESPACE = "round22_fresh_multi_uav_policy_dev_v1"
INTERFERENCE_TIERS = (("orthogonal", "orthogonal", 0.0), ("low", "shared", 0.25), ("medium", "shared", 0.5), ("full", "shared", 1.0))
EPISODE_HORIZON = 12


@dataclass(frozen=True)
class FreshPolicyScenario:
    scenario_id: str
    seed: int
    num_uavs: int
    num_buoys: int
    spectrum_tier: str
    spectrum_mode: str
    interference_overlap_factor: float
    queue_tier: str
    geometry_variant: str
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(num_uavs=2, num_buoys=self.num_buoys, task_semantics_version="v1", buoy_drift_mode="static", channel_variation_mode="static", episode_horizon=EPISODE_HORIZON, terminate_on_task_success=False, seed=self.seed, initial_uav_positions_m=self.uav_positions_m, initial_queue_bits_by_buoy=self.initial_queue_bits, arrival_mode="constant", constant_arrival_bits_per_step=4.5e5, multi_uav_min_separation_m=15.0)

    def options(self) -> SharedSpectrumOptions:
        return SharedSpectrumOptions(self.spectrum_mode, self.interference_overlap_factor, "legacy", 0.25)

    def manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.update({"namespace": FRESH_NAMESPACE, "episode_horizon": EPISODE_HORIZON, "buoy_drift_mode": "static", "channel_variation_mode": "static", "rollout_outcome_read": False})
        return row


def _base_layout(count: int, geometry: int) -> tuple[np.ndarray, np.ndarray]:
    buoy_sets = (
        ((58., 94.), (100., 118.), (144., 91.), (76., 140.), (125., 143.)),
        ((25., 50.), (105., 156.), (176., 47.), (44., 170.), (157., 173.)),
        ((85., 96.), (115., 96.), (100., 134.), (71., 128.), (130., 128.)),
    )
    uav_sets = (
        ((84., 68.), (116., 68.)), ((44., 148.), (156., 40.)), ((88., 76.), (112., 76.)),
    )
    return np.asarray(buoy_sets[geometry][:count], dtype=np.float64), np.asarray(uav_sets[geometry], dtype=np.float64)


def fresh_policy_development_scenarios() -> tuple[FreshPolicyScenario, ...]:
    """Return sixty new, seed-disjoint M=2/K={3,5} configurations."""
    rows: list[FreshPolicyScenario] = []
    for template in range(30):
        tier, mode, eta = INTERFERENCE_TIERS[template % len(INTERFERENCE_TIERS)]
        geometry = template % 3
        queue_level = (0.9e6, 2.4e6, 3.9e6)[template % 3]
        offset = np.array(((template % 5 - 2) * 1.7, ((template * 2) % 5 - 2) * 1.3), dtype=np.float64)
        for k_index, count in enumerate((3, 5)):
            buoys, uavs = _base_layout(count, geometry)
            buoys = np.clip(buoys + offset, 5.0, 195.0)
            uavs = np.clip(uavs - offset * 0.5, 5.0, 195.0)
            queues = np.full(count, queue_level, dtype=np.float64)
            if template % 2:
                queues[0] *= 1.15; queues[-1] *= 0.68
            if geometry == 2:
                queues[:2] *= 1.08
                if count > 2: queues[2:] *= 0.65
            seed = 22000 + 2 * template + k_index
            rows.append(FreshPolicyScenario(f"r22_fresh_k{count}_{tier}_{template:02d}", seed, 2, count, tier, mode, eta, ("low", "medium", "high")[template % 3], ("compact_offset", "spread_offset", "conflict_offset")[geometry], tuple(map(tuple, buoys)), tuple(map(tuple, uavs)), tuple(float(value) for value in queues)))
    if len(rows) != 60 or len({row.seed for row in rows}) != 60:
        raise RuntimeError("Round-22 fresh manifest must have 60 uniquely seeded scenarios")
    if {row.interference_overlap_factor for row in rows} != {0.0, 0.25, 0.5, 1.0}:
        raise RuntimeError("Round-22 fresh manifest lost an interference tier")
    return tuple(rows)


__all__ = ["EPISODE_HORIZON", "FRESH_NAMESPACE", "FreshPolicyScenario", "fresh_policy_development_scenarios"]
