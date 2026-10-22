"""Pre-registered, disjoint Round-30 shared-spectrum scenario populations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round19.multi_env import SharedSpectrumOptions


EPISODE_HORIZON = 12
TRAINING_NAMESPACE = "round30_ppo_training_scenarios_v1"
VALIDATION_NAMESPACE = "round30_ppo_validation_v1"
FRESH_NAMESPACE = "round30_ppo_fresh_v1"
ETA_VALUES = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class PPOScenario:
    scenario_id: str
    seed: int
    population: str
    num_buoys: int
    eta: float
    queue_pressure: str
    geometry: str
    initial_uav_configuration: str
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(
            num_uavs=2, num_buoys=self.num_buoys, task_semantics_version="v1",
            buoy_drift_mode="static", channel_variation_mode="static",
            episode_horizon=EPISODE_HORIZON, terminate_on_task_success=False,
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
            "namespace": {"training": TRAINING_NAMESPACE, "validation": VALIDATION_NAMESPACE, "fresh": FRESH_NAMESPACE}[self.population],
            "num_uavs": 2, "spectrum_mode": "shared", "matching_mode": "legacy",
            "buoy_drift_mode": "static", "channel_variation_mode": "static",
            "episode_horizon": EPISODE_HORIZON, "outcome_accessed": False,
        })
        return row


def _layout(count: int, geometry: str, rng: np.random.Generator) -> np.ndarray:
    compact = np.array(((62., 86.), (100., 108.), (138., 86.), (78., 136.), (122., 136.)))
    spread = np.array(((28., 46.), (100., 164.), (172., 46.), (42., 170.), (158., 170.)))
    base = compact if geometry == "compact" else spread
    return np.clip(base[:count] + rng.uniform(-4.0, 4.0, size=(count, 2)), 8.0, 192.0)


def _uavs(kind: str, rng: np.random.Generator) -> np.ndarray:
    table = {
        "symmetric_close": ((88., 72.), (112., 72.)),
        "symmetric_far": ((66., 54.), (134., 54.)),
        "asymmetric_close": ((82., 68.), (120., 80.)),
        "asymmetric_far": ((48., 150.), (154., 40.)),
    }
    return np.clip(np.asarray(table[kind]) + rng.uniform(-2.0, 2.0, size=(2, 2)), 5.0, 195.0)


def _population(population: str, seed_start: int, count: int) -> tuple[PPOScenario, ...]:
    if count % 8:
        raise ValueError("Round-30 populations must balance every K x eta cell")
    rows: list[PPOScenario] = []
    pressures = ("low", "medium", "high")
    queue_levels = {"low": 8.5e5, "medium": 2.35e6, "high": 4.05e6}
    uav_kinds = ("symmetric_close", "symmetric_far", "asymmetric_close", "asymmetric_far")
    for index in range(count):
        seed = seed_start + index
        rng = np.random.default_rng(seed)
        num_buoys = 3 if index % 2 == 0 else 5
        eta = ETA_VALUES[(index // 2) % len(ETA_VALUES)]
        pressure = pressures[(index // 8) % len(pressures)]
        geometry = ("compact", "spread")[(index // 24) % 2]
        uav_kind = uav_kinds[(index // 6) % len(uav_kinds)]
        buoys = _layout(num_buoys, geometry, rng)
        uavs = _uavs(uav_kind, rng)
        queues = queue_levels[pressure] * rng.uniform(0.78, 1.17, size=num_buoys)
        if index % 2:
            queues[0] *= 1.18
            queues[-1] *= 0.72
        queues = np.clip(queues, 3.0e5, 5.0e6)
        rows.append(PPOScenario(
            scenario_id=f"r30_{population}_k{num_buoys}_eta{str(eta).replace('.', '')}_{index:03d}",
            seed=seed, population=population, num_buoys=num_buoys, eta=float(eta),
            queue_pressure=pressure, geometry=geometry, initial_uav_configuration=uav_kind,
            buoy_positions_m=tuple(tuple(float(value) for value in row) for row in buoys),
            uav_positions_m=tuple(tuple(float(value) for value in row) for row in uavs),
            initial_queue_bits=tuple(float(value) for value in queues),
        ))
    if len({row.seed for row in rows}) != count or {row.num_buoys for row in rows} != {3, 5} or {row.eta for row in rows} != set(ETA_VALUES):
        raise RuntimeError("Round-30 scenario population lost registered balance")
    return tuple(rows)


def training_scenarios() -> tuple[PPOScenario, ...]:
    """Ninety-six new reset worlds; 12 scenarios in every K x eta cell."""
    return _population("training", 40000, 96)


def validation_scenarios() -> tuple[PPOScenario, ...]:
    """Forty-eight new validation worlds; no training gradient may inspect them."""
    return _population("validation", 41000, 48)


def fresh_scenarios() -> tuple[PPOScenario, ...]:
    """Sixty-four deferred-access worlds, frozen before PPO training."""
    return _population("fresh", 42000, 64)


def assert_disjoint(*groups: Iterable[PPOScenario]) -> None:
    sets = [set(item.seed for item in group) for group in groups]
    if any(left & right for i, left in enumerate(sets) for right in sets[i + 1:]):
        raise RuntimeError("Round-30 scenario seed populations overlap")
    # Every historical multi-UAV namespace uses seeds below this range.
    if any(seed < 40000 for group in sets for seed in group):
        raise RuntimeError("Round-30 scenario seed overlaps a historical namespace")


__all__ = ["EPISODE_HORIZON", "ETA_VALUES", "FRESH_NAMESPACE", "PPOScenario", "TRAINING_NAMESPACE", "VALIDATION_NAMESPACE", "assert_disjoint", "fresh_scenarios", "training_scenarios", "validation_scenarios"]
