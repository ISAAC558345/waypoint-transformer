"""Pre-registered Round-20 development and untouched scenario manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round19.multi_env import SharedSpectrumOptions


SCENARIO_NAMESPACE = "round20_multi_uav_shared_spectrum_development_v1"
UNTOUCHED_NAMESPACE = "round20_multi_uav_untouched_eval_v1"
INTERFERENCE_TIERS = (
    ("orthogonal", "orthogonal", 0.0),
    ("low", "shared", 0.25),
    ("medium", "shared", 0.50),
    ("full", "shared", 1.0),
)
SPLITS = ("train", "validation", "development_test")
EPISODE_HORIZON = 12


@dataclass(frozen=True)
class ScenarioTemplate:
    name: str
    queue_tier: str
    geometry_group: str
    coordination_stress: bool
    buoy_layout: str
    uav_layout: str
    queue_scale: float
    asymmetric: bool


TEMPLATES = (
    ScenarioTemplate("low_compact_center", "low", "compact", False, "compact", "center", 0.75e6, False),
    ScenarioTemplate("low_spread_wide", "low", "spread", False, "spread", "wide", 0.75e6, False),
    ScenarioTemplate("medium_compact_asymmetric", "medium", "compact", False, "compact", "center", 2.50e6, True),
    ScenarioTemplate("medium_spread_asymmetric", "medium", "spread", False, "spread", "wide", 2.50e6, True),
    ScenarioTemplate("high_compact_conflict", "high", "compact", True, "stress", "stress", 4.10e6, True),
    ScenarioTemplate("high_spread_conflict", "high", "spread", True, "spread_stress", "wide", 4.10e6, True),
    ScenarioTemplate("low_compact_offset", "low", "compact", False, "compact_offset", "center_offset", 0.95e6, True),
    ScenarioTemplate("medium_spread_offset", "medium", "spread", False, "spread_offset", "wide_offset", 2.20e6, True),
    ScenarioTemplate("high_compact_offset_conflict", "high", "compact", True, "stress_offset", "stress_offset", 3.80e6, True),
    ScenarioTemplate("medium_compact_offset", "medium", "compact", False, "compact_offset", "center_offset", 2.80e6, True),
)


@dataclass(frozen=True)
class DevelopmentScenario:
    scenario_id: str
    seed: int
    split: str
    num_uavs: int
    num_buoys: int
    spectrum_tier: str
    spectrum_mode: str
    interference_overlap_factor: float
    template_index: int
    template_name: str
    queue_tier: str
    geometry_group: str
    coordination_stress: bool
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(
            num_uavs=2,
            num_buoys=self.num_buoys,
            task_semantics_version="v1",
            buoy_drift_mode="static",
            channel_variation_mode="static",
            episode_horizon=EPISODE_HORIZON,
            terminate_on_task_success=False,
            seed=self.seed,
            initial_uav_positions_m=self.uav_positions_m,
            initial_queue_bits_by_buoy=self.initial_queue_bits,
            arrival_mode="constant",
            constant_arrival_bits_per_step=4.5e5,
            multi_uav_min_separation_m=15.0,
        )

    def options(self) -> SharedSpectrumOptions:
        return SharedSpectrumOptions(
            self.spectrum_mode,
            self.interference_overlap_factor,
            "legacy",
            0.25,
        )

    def manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.update({
            "namespace": SCENARIO_NAMESPACE,
            "buoy_drift_mode": "static",
            "channel_variation_mode": "static",
            "episode_horizon": EPISODE_HORIZON,
        })
        return row


def _layout(name: str, count: int) -> tuple[tuple[float, float], ...]:
    layouts = {
        "compact": ((65.0, 85.0), (100.0, 105.0), (135.0, 85.0), (80.0, 130.0), (120.0, 130.0)),
        "compact_offset": ((58.0, 92.0), (99.0, 112.0), (142.0, 88.0), (78.0, 138.0), (126.0, 136.0)),
        "spread": ((30.0, 40.0), (100.0, 165.0), (170.0, 40.0), (35.0, 165.0), (165.0, 165.0)),
        "spread_offset": ((26.0, 48.0), (104.0, 158.0), (174.0, 48.0), (42.0, 172.0), (158.0, 172.0)),
        "stress": ((88.0, 100.0), (112.0, 100.0), (100.0, 126.0), (78.0, 126.0), (122.0, 126.0)),
        "stress_offset": ((84.0, 102.0), (116.0, 102.0), (100.0, 132.0), (74.0, 130.0), (126.0, 130.0)),
        "spread_stress": ((72.0, 76.0), (128.0, 76.0), (100.0, 126.0), (42.0, 142.0), (158.0, 142.0)),
    }
    return tuple(layouts[name][:count])


def _uav_layout(name: str) -> tuple[tuple[float, float], ...]:
    layouts = {
        "center": ((90.0, 65.0), (110.0, 65.0)),
        "center_offset": ((82.0, 70.0), (118.0, 70.0)),
        "wide": ((35.0, 155.0), (165.0, 30.0)),
        "wide_offset": ((42.0, 150.0), (158.0, 36.0)),
        "stress": ((92.0, 80.0), (108.0, 80.0)),
        "stress_offset": ((88.0, 78.0), (112.0, 78.0)),
    }
    return layouts[name]


def _queues(template: ScenarioTemplate, count: int) -> tuple[float, ...]:
    values = np.full(count, template.queue_scale, dtype=np.float64)
    if template.asymmetric:
        values[0] = min(5.0e6, values[0] * 1.18)
        values[-1] = max(1.0e5, values[-1] * 0.62)
    if template.coordination_stress:
        values[0] = min(5.0e6, values[0] * 1.08)
        values[1] = min(5.0e6, values[1] * 1.08)
        if count > 2:
            values[2:] *= 0.67
    return tuple(float(item) for item in values)


def _held_out_templates(group_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Rotate holdout strata across K/tier blocks without looking at outcomes."""
    choices = (
        ((6, 8), (7, 9)),
        ((1, 8), (4, 7)),
        ((3, 4), (6, 9)),
        ((0, 5), (2, 7)),
    )
    return choices[group_index % len(choices)]


def development_scenarios() -> tuple[DevelopmentScenario, ...]:
    rows: list[DevelopmentScenario] = []
    seed = 20000
    for k_index, count in enumerate((3, 5)):
        for tier_index, (tier, mode, eta) in enumerate(INTERFERENCE_TIERS):
            group_index = k_index * len(INTERFERENCE_TIERS) + tier_index
            validation, development_test = _held_out_templates(group_index)
            for template_index, template in enumerate(TEMPLATES):
                split = "validation" if template_index in validation else "development_test" if template_index in development_test else "train"
                rows.append(DevelopmentScenario(
                    scenario_id=f"r20_k{count}_{tier}_{template.name}",
                    seed=seed,
                    split=split,
                    num_uavs=2,
                    num_buoys=count,
                    spectrum_tier=tier,
                    spectrum_mode=mode,
                    interference_overlap_factor=eta,
                    template_index=template_index,
                    template_name=template.name,
                    queue_tier=template.queue_tier,
                    geometry_group=template.geometry_group,
                    coordination_stress=template.coordination_stress,
                    buoy_positions_m=_layout(template.buoy_layout, count),
                    uav_positions_m=_uav_layout(template.uav_layout),
                    initial_queue_bits=_queues(template, count),
                ))
                seed += 1
    if len(rows) != 80:
        raise RuntimeError("Round-20 protocol must contain exactly 80 scenarios")
    if {item.split for item in rows} != set(SPLITS):
        raise RuntimeError("Round-20 scenario split is incomplete")
    return tuple(rows)


def untouched_evaluation_scenarios() -> tuple[dict[str, Any], ...]:
    """Frozen snapshots only; this function never constructs an environment."""
    rows: list[dict[str, Any]] = []
    seed = 29000
    for count in (3, 5):
        for tier, mode, eta in INTERFERENCE_TIERS:
            for replica in range(2):
                template = TEMPLATES[(replica * 3 + (0 if count == 3 else 1)) % len(TEMPLATES)]
                row = {
                    "namespace": UNTOUCHED_NAMESPACE,
                    "scenario_id": f"r20_untouched_k{count}_{tier}_{replica}",
                    "seed": seed,
                    "num_uavs": 2,
                    "num_buoys": count,
                    "spectrum_tier": tier,
                    "spectrum_mode": mode,
                    "interference_overlap_factor": eta,
                    "episode_horizon": EPISODE_HORIZON,
                    "buoy_drift_mode": "static",
                    "channel_variation_mode": "static",
                    "buoy_positions_m": _layout(template.buoy_layout, count),
                    "uav_positions_m": _uav_layout(template.uav_layout),
                    "initial_queue_bits": _queues(template, count),
                    "policy_rollout_count": 0,
                }
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                row["metadata_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
                rows.append(row)
                seed += 1
    return tuple(rows)


def split_summary(rows: tuple[DevelopmentScenario, ...] | None = None) -> list[dict[str, Any]]:
    values = development_scenarios() if rows is None else rows
    result: list[dict[str, Any]] = []
    for split in SPLITS:
        for count in (3, 5):
            for tier, _, _ in INTERFERENCE_TIERS:
                group = [item for item in values if item.split == split and item.num_buoys == count and item.spectrum_tier == tier]
                result.append({
                    "split": split,
                    "num_buoys": count,
                    "spectrum_tier": tier,
                    "scenario_count": len(group),
                    "queue_tiers": sorted({item.queue_tier for item in group}),
                    "geometry_groups": sorted({item.geometry_group for item in group}),
                    "coordination_stress_count": sum(item.coordination_stress for item in group),
                })
    return result


__all__ = [
    "DevelopmentScenario", "EPISODE_HORIZON", "INTERFERENCE_TIERS", "SCENARIO_NAMESPACE",
    "SPLITS", "TEMPLATES", "UNTOUCHED_NAMESPACE", "development_scenarios", "split_summary",
    "untouched_evaluation_scenarios",
]
