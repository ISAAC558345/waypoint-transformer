"""Pre-registered, physically disjoint Round-33 long-episode scenarios.

Only ``episode_horizon`` is a Round-33 environment degree of freedom.  Every
other ``MaritimeEnvConfig`` and ``SharedSpectrumOptions`` field is inherited
from the frozen Round-19/30 shared-spectrum contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round19.multi_env import SharedSpectrumOptions


EPISODE_HORIZON = 20
DESIGN_NAMESPACE = "round33_long_episode_design_v1"
CONFIRMATION_NAMESPACE = "round33_long_episode_confirmation_v1"
ETA_VALUES = (0.25, 0.5)
QUEUE_PRESSURES = ("medium", "high")
GEOMETRIES = ("compact", "spread")
SEEDS_PER_STRATUM = 4
DESIGN_SEED_START = 53000
CONFIRMATION_SEED_START = 54000


@dataclass(frozen=True)
class LongEpisodeScenario:
    scenario_id: str
    seed: int
    population: str
    num_buoys: int
    eta: float
    queue_pressure: str
    geometry: str
    replicate: int
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self, *, episode_horizon: int = EPISODE_HORIZON) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(
            num_uavs=2, num_buoys=self.num_buoys, task_semantics_version="v1",
            buoy_drift_mode="static", channel_variation_mode="static",
            episode_horizon=int(episode_horizon), terminate_on_task_success=False,
            arrival_mode="constant", constant_arrival_bits_per_step=4.5e5,
            multi_uav_min_separation_m=15.0, seed=self.seed,
            initial_uav_positions_m=self.uav_positions_m,
            initial_queue_bits_by_buoy=self.initial_queue_bits,
        )

    def options(self) -> SharedSpectrumOptions:
        return SharedSpectrumOptions("shared", self.eta, "legacy", 0.25)

    def manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.update({
            "namespace": DESIGN_NAMESPACE if self.population == "design" else CONFIRMATION_NAMESPACE,
            "num_uavs": 2, "spectrum_mode": "shared", "matching_mode": "legacy",
            "buoy_drift_mode": "static", "channel_variation_mode": "static",
            "episode_horizon": EPISODE_HORIZON, "outcome_accessed": False,
        })
        return row


def _layout(count: int, geometry: str, rng: np.random.Generator) -> np.ndarray:
    compact = np.array(((58., 82.), (100., 108.), (142., 82.), (78., 142.), (122., 142.)))
    spread = np.array(((22., 34.), (100., 178.), (178., 34.), (32., 172.), (168., 172.)))
    return np.clip((compact if geometry == "compact" else spread)[:count] + rng.uniform(-3., 3., (count, 2)), 6., 194.)


def _uavs(geometry: str, rng: np.random.Generator) -> np.ndarray:
    base = ((82., 58.), (118., 58.)) if geometry == "compact" else ((34., 156.), (166., 42.))
    return np.clip(np.asarray(base) + rng.uniform(-2., 2., (2, 2)), 5., 195.)


def _population(population: str, seed_start: int) -> tuple[LongEpisodeScenario, ...]:
    rows: list[LongEpisodeScenario] = []
    seed = seed_start
    levels = {"medium": 2.35e6, "high": 4.05e6}
    for num_buoys in (3, 5):
        for eta in ETA_VALUES:
            for pressure in QUEUE_PRESSURES:
                for geometry in GEOMETRIES:
                    for replicate in range(SEEDS_PER_STRATUM):
                        rng = np.random.default_rng(seed)
                        queues = levels[pressure] * rng.uniform(.82, 1.15, num_buoys)
                        # Fixed pre-outcome asymmetry makes relocation and queue
                        # balance observable without introducing a new variable.
                        queues[0] = min(5.e6, queues[0] * 1.13)
                        queues[-1] = max(3.e5, queues[-1] * .81)
                        rows.append(LongEpisodeScenario(
                            scenario_id=f"r33_{population}_k{num_buoys}_eta{str(eta).replace('.', '')}_{pressure}_{geometry}_r{replicate}",
                            seed=seed, population=population, num_buoys=num_buoys,
                            eta=float(eta), queue_pressure=pressure, geometry=geometry,
                            replicate=replicate,
                            buoy_positions_m=tuple(tuple(float(x) for x in row) for row in _layout(num_buoys, geometry, rng)),
                            uav_positions_m=tuple(tuple(float(x) for x in row) for row in _uavs(geometry, rng)),
                            initial_queue_bits=tuple(float(x) for x in np.clip(queues, 3.e5, 5.e6)),
                        ))
                        seed += 1
    if len(rows) != 64 or len({row.seed for row in rows}) != 64:
        raise RuntimeError("Round-33 population lost the exact 64-row design")
    return tuple(rows)


def design_scenarios() -> tuple[LongEpisodeScenario, ...]:
    return _population("design", DESIGN_SEED_START)


def confirmation_scenarios() -> tuple[LongEpisodeScenario, ...]:
    return _population("confirmation", CONFIRMATION_SEED_START)


def assert_disjoint(*groups: Iterable[LongEpisodeScenario]) -> None:
    sets = [set(row.seed for row in group) for group in groups]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1:]):
        raise RuntimeError("Round-33 design/confirmation seed collision")
    # All registered R20/R22/R24/R29/R30 multi-UAV scenario namespaces use
    # seeds below 53,000.  These two ranges are intentionally outside them.
    if any(seed < DESIGN_SEED_START for group in sets for seed in group):
        raise RuntimeError("Round-33 seed lies in a historical namespace")


__all__ = [
    "CONFIRMATION_NAMESPACE", "CONFIRMATION_SEED_START", "DESIGN_NAMESPACE",
    "DESIGN_SEED_START", "EPISODE_HORIZON", "ETA_VALUES", "GEOMETRIES",
    "LongEpisodeScenario", "QUEUE_PRESSURES", "SEEDS_PER_STRATUM",
    "assert_disjoint", "confirmation_scenarios", "design_scenarios",
]
