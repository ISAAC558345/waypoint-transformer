"""Phase-6I-K low-level scale adaptation. High-level checkpoints stay untouched."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    DEV_POOL,
    HOLD,
    PPO,
    ROOT,
    TRAIN_POOL,
)
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.l3b.config import L3BConfig

PHASE_NAME = "phase6ik_lowlevel_opt"
OUTPUT = ROOT / "results" / PHASE_NAME
CHECKPOINT_DIR = OUTPUT / "checkpoints"
REPORT = ROOT / "docs" / "HTW_Phase6I_LowLevelMAPPOOptimization_CN.md"
SOURCE_ACTOR = primary_checkpoint()
MODEL_SEEDS = (0, 1, 2)
UPDATES = 40
EVAL_POOL = DEV_POOL
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0


def physical_scales(config: L3BConfig | None = None) -> dict[str, float]:
    """Unit scales taken from the environment. They are not fit on payload."""
    cfg = L3BConfig() if config is None else config
    return {
        "area_width_m": float(cfg.area_width),
        "area_height_m": float(cfg.area_height),
        "diagonal_m": float((cfg.area_width**2 + cfg.area_height**2) ** 0.5),
        "queue_capacity_blocks": float(cfg.queue_capacity_blocks),
        "bandwidth_hz": float(cfg.bandwidth_hz),
        "tx_power": float(cfg.tx_power),
        "close_range_sinr": float(cfg.tx_power) / float(cfg.noise_power),
        "hold": float(HOLD),
        "service_radius_m": float(cfg.service_radius_m),
    }


def assert_phase_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    source = SOURCE_ACTOR.resolve()
    if resolved == source:
        raise RuntimeError(f"phase-6ik refused to overwrite {resolved}")
    if "phase6g_lowlevel_mappo" in resolved.parts:
        raise RuntimeError(f"phase-6ik refused to write inside the frozen MAPPO directory: {resolved}")
    text = str(resolved)
    if any(str(scenario) in Path(resolved).name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6ik refused a reserved-test path: {resolved}")
    if "322064" in text or "322079" in text:
        raise RuntimeError(f"phase-6ik refused a holdout path: {resolved}")
    return resolved


def assert_eval_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or value not in EVAL_POOL:
        raise RuntimeError(f"phase-6ik eval refuses scenario {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_DIR",
    "EVAL_POOL",
    "HOLD",
    "MODEL_SEEDS",
    "OUTPUT",
    "PPO",
    "REPORT",
    "SOURCE_ACTOR",
    "TRAIN_POOL",
    "UPDATES",
    "assert_eval_scenario",
    "assert_phase_output",
    "physical_scales",
]
