"""LL-V2.2C protocol. Frozen audit only. No training."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase2b.protocol import (
    BETA_RESIDUAL,
    EPSILON_BASE_FRACTION,
    OUTPUT as PHASE2B_OUT,
    PHASE2B,
    PPO,
    ROOT,
    action_max,
    epsilon_base,
    load_pools,
    refuse_phase2_validation,
)

PHASE2C = Path(__file__).resolve().parent
OUTPUT = PHASE2C / "outputs"
CKPT = PHASE2B_OUT
UPDATES = (0, 20, 40, 60, 80, 100)

__all__ = [
    "BETA_RESIDUAL",
    "CKPT",
    "EPSILON_BASE_FRACTION",
    "OUTPUT",
    "PHASE2B",
    "PHASE2C",
    "PPO",
    "ROOT",
    "UPDATES",
    "action_max",
    "epsilon_base",
    "load_pools",
    "refuse_phase2_validation",
]
