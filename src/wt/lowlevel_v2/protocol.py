"""LL-V2.0 protocol. Does not train. Does not modify FINAL-HIER."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase941.protocol import sha256_bytes
from wt.htw_phase9.const import HOLD, IDLE_INDEX, MAX_BUOYS, MAX_UAVS

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "phase0"
FINAL_CKPT = ROOT / "final_artifacts" / "checkpoints" / "deepsets_util_seed0.pt"
EXPECTED_HIGH_SHA = "dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109"
PHASE106_UNTOUCHED = frozenset(range(106000, 106064))
PHASE105_INDEPENDENT = frozenset(range(105000, 105064))
PHASE104_INDEPENDENT = frozenset(range(104000, 104064))

# Historical maritime Level-2 used multi_uav_min_separation_m=15.0. That value is
# not an L3-B physical radius: L3-B spawn spacing is ~1.2 m, so 15 m is infeasible.
COLLISION_REQUIRES_PRETRAINING_FREEZE = True
UAV_PHYSICAL_RADIUS_PLACEHOLDER_M = 0.4
COLLISION_MARGIN_PLACEHOLDER_M = 0.2
D_SAFE_PLACEHOLDER_M = float(2 * UAV_PHYSICAL_RADIUS_PLACEHOLDER_M + COLLISION_MARGIN_PLACEHOLDER_M)
SUGGESTED_UAV_RADIUS_RANGE_M = (0.2, 1.0)
SUGGESTED_COLLISION_MARGIN_RANGE_M = (0.1, 0.5)
NEAR_COLLISION_FACTOR = 1.5

ALPHA_RESIDUAL_PLACEHOLDER = 0.25
ALPHA_RESIDUAL_REQUIRES_FREEZE = True
SERVICE_SAFETY_MARGIN_PLACEHOLDER_M = 0.5
SERVICE_SAFETY_MARGIN_REQUIRES_FREEZE = True

HOLD_STEPS = int(HOLD)
ENV_IDLE = -1
DECODER_IDLE = int(IDLE_INDEX)
MAX_STEP_NORM = 1.0

LLV2_TRAIN_BASE = 220000
LLV2_DESIGN_BASE = 221000
LLV2_TRAIN_N = 64
LLV2_DESIGN_N = 16

__all__ = [
    "ALPHA_RESIDUAL_PLACEHOLDER",
    "COLLISION_REQUIRES_PRETRAINING_FREEZE",
    "D_SAFE_PLACEHOLDER_M",
    "DECODER_IDLE",
    "ENV_IDLE",
    "EXPECTED_HIGH_SHA",
    "FINAL_CKPT",
    "HOLD_STEPS",
    "OUTPUT",
    "PHASE106_UNTOUCHED",
    "ROOT",
    "sha256_bytes",
]
