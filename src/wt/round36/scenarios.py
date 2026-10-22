"""Frozen, generated clean-scenario specification for Round-36."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

from wt.maritime.config import MaritimeEnvConfig


DESIGN_FAMILY = "round36_design_v1"
CONFIRMATION_FAMILY = "round36_confirmation_v1"
SCENARIOS_PER_NAMESPACE = 64
_CANONICAL_BALANCED_UAVS = ((30.0, 30.0), (170.0, 30.0), (30.0, 170.0), (170.0, 170.0))


@dataclass(frozen=True)
class CleanScenario:
    scenario_id: str
    family: str
    seed: int
    eta: float
    queue_pressure: str
    geometry: str
    replicate: int
    initial_queue_bits_by_buoy: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        arrival = 2.0e5 if self.queue_pressure == "medium" else 5.0e5
        return MaritimeEnvConfig(
            num_uavs=4,
            num_buoys=8,
            episode_horizon=20,
            task_semantics_version="v1",
            initial_uav_positions_m=_CANONICAL_BALANCED_UAVS,
            initial_queue_bits_by_buoy=self.initial_queue_bits_by_buoy,
            constant_arrival_bits_per_step=arrival,
            randomize_buoys=False,
            seed=self.seed,
            terminate_on_task_success=False,
        )

    def buoy_positions_m(self) -> np.ndarray:
        center, spacing = 100.0, (18.0 if self.geometry == "compact" else 55.0)
        return np.asarray([
            (center + (column - 1.5) * spacing, center + (row - 0.5) * spacing)
            for row, column in (divmod(index, 4) for index in range(8))
        ], dtype=np.float64)

    def metadata(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_family": self.family,
            "source_split": self.family,
            "provenance": "generated_clean_round36",
            "source_kind": "generated_clean_metadata",
            "generated": True,
            "protected_data_access_count": 0,
            "seed": self.seed,
            "eta": self.eta,
            "queue_pressure": self.queue_pressure,
            "geometry": self.geometry,
            "replicate": self.replicate,
            "initial_uav_geometry": "frozen_balanced_canonical_rule",
            "initial_queue_bits_by_buoy": list(self.initial_queue_bits_by_buoy),
            "config": asdict(self.config()),
        }


def _queues(seed: int, pressure: str) -> tuple[float, ...]:
    rng = np.random.default_rng(int(seed))
    if pressure == "medium":
        values = rng.uniform(2.1e6, 2.9e6, size=8)
    elif pressure == "high":
        values = rng.uniform(4.05e6, 4.75e6, size=8)
    else:
        raise ValueError("queue pressure must be medium or high")
    return tuple(float(value) for value in values)


def build_clean_scenarios(family: str) -> list[CleanScenario]:
    if family == DESIGN_FAMILY:
        seed_start = 36000
    elif family == CONFIRMATION_FAMILY:
        seed_start = 36100
    else:
        raise ValueError("unknown Round-36 clean scenario family")
    scenarios: list[CleanScenario] = []
    for offset, (eta, pressure, geometry, replicate) in enumerate(product((0.25, 0.5), ("medium", "high"), ("compact", "spread"), range(8))):
        seed = seed_start + offset
        scenarios.append(CleanScenario(
            scenario_id=f"{family}_{offset:03d}", family=family, seed=seed, eta=float(eta),
            queue_pressure=pressure, geometry=geometry, replicate=int(replicate),
            initial_queue_bits_by_buoy=_queues(seed, pressure),
        ))
    if len(scenarios) != SCENARIOS_PER_NAMESPACE:
        raise RuntimeError("Round-36 factorial scenario count changed")
    return scenarios


def require_clean_manifest(metadata: dict[str, Any]) -> None:
    """Metadata-only, fail-closed allowlist used before state construction."""
    allowed = {DESIGN_FAMILY, CONFIRMATION_FAMILY}
    forbidden = ("development", "development_test", "fresh", "untouched", "historical", "unknown", "train", "validation", "test")
    if metadata.get("scenario_family") not in allowed:
        raise PermissionError("Round-36 scenario family is not allowlisted")
    for key in ("scenario_family", "source_split", "provenance", "source_kind"):
        value = str(metadata.get(key, "unknown")).lower()
        if any(token in value for token in forbidden):
            raise PermissionError(f"Round-36 protected-split fail-closed: {key}={value}")
    if metadata.get("generated") is not True or metadata.get("protected_data_access_count") != 0:
        raise PermissionError("Round-36 requires generated clean scenarios with zero access")


__all__ = ["CleanScenario", "CONFIRMATION_FAMILY", "DESIGN_FAMILY", "SCENARIOS_PER_NAMESPACE", "build_clean_scenarios", "require_clean_manifest"]
