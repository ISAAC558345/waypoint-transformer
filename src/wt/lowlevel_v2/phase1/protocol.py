"""LL-V2.1 constants. Geometry frozen in LLV2_PHASE1_PREFREEZE.json before training."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase101.protocol import PILOT_UPDATES, PPO, SMOKE_UPDATES
from wt.htw_phase941.protocol import sha256_bytes
from wt.lowlevel_v2.protocol import EXPECTED_HIGH_SHA, FINAL_CKPT, HOLD_STEPS, PHASE106_UNTOUCHED, ROOT

PHASE1 = Path(__file__).resolve().parent
OUTPUT = PHASE1 / "outputs"
PREFREEZE_FILE = PHASE1 / "LLV2_PHASE1_PREFREEZE.json"
REGISTRY_FILE = ROOT / "results/root_exports/LLV2_DATA_REGISTRY.json"

D_SAFE_M = 1.0
ALPHA_RESIDUAL = 0.25
SERVICE_SAFETY_MARGIN_M = 0.5
MODEL_SEED = 0
VECTOR_SEED_SMOKE = 221100
VECTOR_SEED_PILOT = 221200
LOG_STD_INIT = -2.2
ACTION_SCALE = 1.0
LAMBDA_PROJECT = 0.05
LAMBDA_RESIDUAL = 0.01
PAYLOAD_SCALE = 1.0e6
EVAL_EVERY = 20
SMOKE_UPDATES_N = int(SMOKE_UPDATES)
PILOT_UPDATES_N = int(PILOT_UPDATES)


def load_prefreeze() -> dict:
    payload = json.loads(PREFREEZE_FILE.read_text(encoding="utf-8"))
    if float(payload["d_safe_m"]) != D_SAFE_M:
        raise RuntimeError("prefreeze d_safe mismatch")
    if float(payload["alpha_residual"]) != ALPHA_RESIDUAL:
        raise RuntimeError("prefreeze alpha mismatch")
    if float(payload["service_safety_margin_m"]) != SERVICE_SAFETY_MARGIN_M:
        raise RuntimeError("prefreeze margin mismatch")
    if payload.get("tuned_from_payload") is True:
        raise RuntimeError("prefreeze illegally tuned from payload")
    return payload


def load_pools() -> tuple[list[int], list[int]]:
    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    train = [int(s) for s in registry["LLV2_train"]]
    design = [int(s) for s in registry["LLV2_design"]]
    if set(train) & PHASE106_UNTOUCHED or set(design) & PHASE106_UNTOUCHED:
        raise RuntimeError("LLV2 pools overlap Phase-10.6 untouched")
    return train, design


def refuse_untouched(seed: int) -> None:
    if int(seed) in PHASE106_UNTOUCHED:
        raise RuntimeError(f"refused Phase-10.6 untouched seed {seed}")


def high_sha() -> str:
    return sha256_bytes(FINAL_CKPT.read_bytes())
