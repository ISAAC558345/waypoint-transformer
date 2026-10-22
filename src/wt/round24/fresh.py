"""Pre-training registration of the Round-24 fresh stability scenarios."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round19.multi_env import SharedSpectrumOptions
from wt.round20.scenarios import INTERFERENCE_TIERS, TEMPLATES, _layout, _queues, _uav_layout


FRESH_NAMESPACE = "round24_seed_stability_dev_v1"
EPISODE_HORIZON = 12
BASE_SEEDS = tuple(range(24000, 24010))


@dataclass(frozen=True)
class FreshStabilityScenario:
    scenario_id: str
    seed: int
    base_seed: int
    num_uavs: int
    num_buoys: int
    spectrum_tier: str
    spectrum_mode: str
    interference_overlap_factor: float
    queue_tier: str
    geometry_variant: str
    coordination_stress: bool
    template_index: int
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(
            num_uavs=2, num_buoys=self.num_buoys, task_semantics_version="v1",
            buoy_drift_mode="static", channel_variation_mode="static",
            episode_horizon=EPISODE_HORIZON, terminate_on_task_success=False,
            seed=self.seed, initial_uav_positions_m=self.uav_positions_m,
            initial_queue_bits_by_buoy=self.initial_queue_bits, arrival_mode="constant",
            constant_arrival_bits_per_step=4.5e5, multi_uav_min_separation_m=15.0,
        )

    def options(self) -> SharedSpectrumOptions:
        return SharedSpectrumOptions(self.spectrum_mode, self.interference_overlap_factor, "legacy", 0.25)

    def manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.update({"namespace": FRESH_NAMESPACE, "episode_horizon": EPISODE_HORIZON,
                    "buoy_drift_mode": "static", "channel_variation_mode": "static",
                    "rollout_outcome_read": False,
                    "bridge_definition": "not_available_without_future_labels"})
        return row


def fresh_stability_scenarios() -> tuple[FreshStabilityScenario, ...]:
    """Return the pre-registered 10 base-seed x K x eta = 80 configurations."""
    rows: list[FreshStabilityScenario] = []
    for base_index, base_seed in enumerate(BASE_SEEDS):
        rng = np.random.default_rng(base_seed)
        shift = rng.uniform(-4.5, 4.5, size=2)
        for count in (3, 5):
            for tier_index, (tier, mode, eta) in enumerate(INTERFERENCE_TIERS):
                template_index = (base_index * 3 + tier_index + (1 if count == 5 else 0)) % len(TEMPLATES)
                template = TEMPLATES[template_index]
                buoys = np.clip(np.asarray(_layout(template.buoy_layout, count), dtype=np.float64) + shift, 5.0, 195.0)
                uavs = np.clip(np.asarray(_uav_layout(template.uav_layout), dtype=np.float64) - .4 * shift, 5.0, 195.0)
                scenario_seed = 24000 + base_index * 8 + (0 if count == 3 else 4) + tier_index
                rows.append(FreshStabilityScenario(
                    scenario_id=f"r24_fresh_base{base_index:02d}_k{count}_{tier}", seed=scenario_seed,
                    base_seed=base_seed, num_uavs=2, num_buoys=count, spectrum_tier=tier,
                    spectrum_mode=mode, interference_overlap_factor=eta, queue_tier=template.queue_tier,
                    geometry_variant=template.geometry_group, coordination_stress=template.coordination_stress,
                    template_index=template_index, buoy_positions_m=tuple(map(tuple, buoys)),
                    uav_positions_m=tuple(map(tuple, uavs)), initial_queue_bits=_queues(template, count),
                ))
    if len(rows) != 80 or len({row.scenario_id for row in rows}) != 80:
        raise RuntimeError("Round-24 fresh stability registration must contain exactly 80 unique scenarios")
    if {row.base_seed for row in rows} != set(BASE_SEEDS) or {row.interference_overlap_factor for row in rows} != {0.0, .25, .5, 1.0}:
        raise RuntimeError("Round-24 fresh stability strata are incomplete")
    return tuple(rows)


__all__ = ["BASE_SEEDS", "FRESH_NAMESPACE", "FreshStabilityScenario", "fresh_stability_scenarios"]
