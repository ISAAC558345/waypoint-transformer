"""Phase-10.1 frozen pilot protocol. Low-level training only."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase9.const import EVAL_SEEDS, HOLD, TRAIN_SEEDS
from wt.htw_phase941.protocol import dump_frozen, sha256_bytes
from wt.htw_phase945.protocol import ROOT
from wt.htw_phase100.protocol import ARCHIVE
from wt.round30.ppo import PPOConfig

OUTPUT = Path(__file__).resolve().parent / "outputs" / "pilot"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase10_1_low_level_adaptation_pilot_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase10_1_low_level_adaptation_pilot_report.md"

HIGH_MODEL_SEED = 0
LOW_MODEL_SEED = 0
HIGH_CKPT = ARCHIVE / f"deepsets_util_seed{HIGH_MODEL_SEED}.pt"
FORBIDDEN_M5_CKPT = Path(__file__).resolve().parents[1] / "htw_phase63" / "outputs" / "checkpoints" / "low_level_c05_d005.pt"

DESIGN_TRAIN_SEEDS = tuple(int(s) for s in TRAIN_SEEDS)  # 0-7
DESIGN_VAL_SEEDS = tuple(int(s) for s in EVAL_SEEDS)  # 8-15, design validation only

SMOKE_UPDATES = 2
PILOT_UPDATES = 100
CHECKPOINT_EVERY = 20
AUX_IN_RANGE = 0.5
AUX_DISTANCE = 0.05

PPO = PPOConfig(rollout_length=48, num_envs=4, num_minibatches=4, update_epochs=4, checkpoint_interval_steps=192)

VECTOR_SEED_SMOKE = 101100
VECTOR_SEED_PILOT = 101200

__all__ = [
    "ARCHIVE",
    "AUX_DISTANCE",
    "AUX_IN_RANGE",
    "CHECKPOINT_EVERY",
    "DESIGN_TRAIN_SEEDS",
    "DESIGN_VAL_SEEDS",
    "FORBIDDEN_M5_CKPT",
    "HIGH_CKPT",
    "HIGH_MODEL_SEED",
    "HOLD",
    "LOW_MODEL_SEED",
    "OUTPUT",
    "PACK_REPORT",
    "PILOT_UPDATES",
    "PPO",
    "REPORT",
    "ROOT",
    "SMOKE_UPDATES",
    "VECTOR_SEED_PILOT",
    "VECTOR_SEED_SMOKE",
    "dump_frozen",
    "sha256_bytes",
]
