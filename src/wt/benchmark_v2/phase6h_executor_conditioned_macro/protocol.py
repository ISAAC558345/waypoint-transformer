"""Phase-6H rules. Thresholds are fixed before any MAPPO macro label is computed."""
from __future__ import annotations

from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6H_ExecutorConditionedMacroValueRecalibration_CN.md"
CHECKPOINT_DIR = ROOT / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo" / "checkpoints"
PRIMARY_SEED = 0
ROBUSTNESS_SEEDS = (0, 1, 2, 3, 4)
LABEL_SHIFT_POOL = tuple(range(320000, 320016))
CLOSED_LOOP_POOL = tuple(range(320016, 320048))
ROBUSTNESS_POOL = tuple(range(320048, 320064))
RESERVED_POOL = tuple(range(320100, 320132))
HOLD = 15
TOP1_DISAGREEMENT_MATERIAL = 0.20
ALIGNMENT_SPEARMAN_GAIN = 0.05
SEED_CV_LIMIT = 0.50
POSITIVE_BIT = 1.0
MIN_POSITIVE_FRACTION = 0.60
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
ASSIGNMENT_DISAGREE_MATERIAL = 0.25


def primary_checkpoint() -> Path:
    return CHECKPOINT_DIR / f"phase6g_seed{PRIMARY_SEED}.pt"


def assert_allowed(scenario_id: int, *, split: str) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or 321000 <= value <= 321127 or 310100 <= value <= 310125:
        raise RuntimeError(f"phase-6h refuses scenario {value}")
    pools = {"label": LABEL_SHIFT_POOL, "closed": CLOSED_LOOP_POOL, "robust": ROBUSTNESS_POOL}
    if value not in pools[split]:
        raise RuntimeError(f"phase-6h {split} split does not include {value}")


def choose_gate(
    *,
    failures: int,
    seed_cv: float,
    c2_c0_pass: bool,
    c2_c1_pass: bool,
    label_shift_material: bool,
    alignment_improved: bool,
) -> dict:
    if int(failures) > 0:
        gate, name = "ALIGN-E", "SYSTEM_OR_ACCOUNTING_REGRESSION"
    elif float(seed_cv) > SEED_CV_LIMIT:
        gate, name = "ALIGN-D", "LOW_LEVEL_SEED_SENSITIVITY_TOO_HIGH"
    elif bool(c2_c0_pass) and bool(c2_c1_pass):
        gate, name = "ALIGN-A", "EXECUTOR_CONDITIONED_MACRO_PLANNING_CONFIRMED"
    elif bool(label_shift_material) and bool(alignment_improved) and not bool(c2_c0_pass):
        gate, name = "ALIGN-B", "EXECUTOR_LABEL_SHIFT_CONFIRMED_BUT_NO_SYSTEM_GAIN"
    else:
        gate, name = "ALIGN-C", "EXECUTOR_CONDITIONING_NOT_MATERIAL"
    return {
        "gate": gate,
        "gate_name": name,
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "mappo_v2_frozen": True,
        "high_level_trained": False,
        "trained_hgnn": False,
        "retrained_mappo": False,
        "candidate_generator_modified": False,
        "fresh_confirmation_untouched": True,
        "next_phase_builds_high_level_dataset": gate == "ALIGN-A",
        "primary_seed": PRIMARY_SEED,
        "executor_value_alignment_improved": bool(alignment_improved),
        "label_shift_material": bool(label_shift_material),
    }
