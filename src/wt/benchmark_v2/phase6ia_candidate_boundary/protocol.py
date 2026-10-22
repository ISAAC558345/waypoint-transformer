"""Phase-6I-A rules. Thresholds are fixed before any k=6 or k=7 macro label."""
from __future__ import annotations

from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6IA_MAPPOConditionedCandidateBoundaryClosure_CN.md"
CHECKPOINT_DIR = ROOT / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo" / "checkpoints"
PRIMARY_SEED = 0
ROBUSTNESS_SEEDS = (0, 1, 2, 3, 4)
DESIGN_POOL = tuple(range(320000, 320032))
CONFIRM_POOL = (320032, 320036, 320040, 320044, 320048, 320052, 320056, 320060)
SANITY_POOL = tuple(range(320032, 320040))
DEVELOPMENT_POOL = DESIGN_POOL + CONFIRM_POOL
RESERVED_POOL = tuple(range(320100, 320132))
HOLD = 15
PRIMARY_TIMES = (0, 15, 30)
SECONDARY_TIME = 45
KS = (5, 6)
K7_SKIP_REASON = "design primary states have mean NGain_7_6 = 0 and P95 NGain_7_6 = 0, so k=7 is not evaluated again"
NGAIN_MEAN_MAX = 0.02
NGAIN_P95_MAX = 0.05
COVERAGE_MIN = 0.90
RANK7_HEAVY = 0.50
EPS_BITS = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
PROTOCOL_STAMP = "6IA-PRE-J15-1"
STATE_SOURCE = "MAX_RATE_INDEPENDENT held for 15 steps and executed by frozen MAPPO-V2 seed0"
INTERFERENCE_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)


def assert_allowed(scenario_id: int, *, split: str) -> None:
    value = int(scenario_id)
    if value in RESERVED_POOL or 321000 <= value <= 321127 or 310100 <= value <= 310125:
        raise RuntimeError(f"phase-6ia refuses scenario {value}")
    pools = {"design": DESIGN_POOL, "confirm": CONFIRM_POOL, "sanity": SANITY_POOL}
    if value not in pools[split]:
        raise RuntimeError(f"phase-6ia {split} split does not include {value}")


def evaluate_flags(stats: dict) -> dict:
    """Apply the pre-registered sufficiency rules. This does not look at a new threshold."""

    def immaterial(mean_key: str, tail_key: str) -> bool:
        return float(stats[mean_key]) <= NGAIN_MEAN_MAX and float(stats[tail_key]) <= NGAIN_P95_MAX

    def stable(mean_key: str, low_key: str) -> bool:
        return float(stats[mean_key]) > NGAIN_MEAN_MAX and float(stats[low_key]) > 0.0

    confirm_k7 = stats["confirm_mean_ngain_7_6"] is not None and stats["confirm_p95_ngain_7_6"] is not None
    closed_significant = float(stats["closed_mean_bits"]) > 0.0 and float(stats["closed_ci_low_bits"]) > 0.0
    k5_sufficient = bool(
        immaterial("design_mean_ngain_6_5", "design_p95_ngain_6_5")
        and immaterial("confirm_mean_ngain_6_5", "confirm_p95_ngain_6_5")
        and not closed_significant
        and float(stats["design_coverage_k5"]) >= COVERAGE_MIN
        and float(stats["confirm_coverage_k5"]) >= COVERAGE_MIN
    )
    return {
        "k5_sufficient": k5_sufficient,
        "k6_stable": bool(stable("design_mean_ngain_6_5", "design_ngain_6_5_ci_low") and stable("confirm_mean_ngain_6_5", "confirm_ngain_6_5_ci_low")),
        "k7_stable": bool(confirm_k7 and stable("design_mean_ngain_7_6", "design_ngain_7_6_ci_low") and stable("confirm_mean_ngain_7_6", "confirm_ngain_7_6_ci_low")),
        "k7_immaterial": bool(immaterial("design_mean_ngain_7_6", "design_p95_ngain_7_6") and ((not confirm_k7) or immaterial("confirm_mean_ngain_7_6", "confirm_p95_ngain_7_6"))),
        "rank7_heavy": bool(float(stats["design_rank7_fraction"]) >= RANK7_HEAVY or (confirm_k7 and float(stats["confirm_rank7_fraction"]) >= RANK7_HEAVY)),
        "closed_significant": bool(closed_significant),
        "c7_needed": bool(float(stats["design_mean_ngain_7_6"]) > NGAIN_MEAN_MAX or float(stats["design_p95_ngain_7_6"]) > NGAIN_P95_MAX),
    }


def choose_gate(flags: dict, *, failures: int) -> dict:
    if int(failures) > 0:
        gate, name, selected_k = "BOUND-E", "SYSTEM_OR_ACCOUNTING_FAILURE", None
    elif bool(flags["k5_sufficient"]):
        gate, name, selected_k = "BOUND-A", "K5_MACRO_CANDIDATE_SPACE_CONFIRMED", 5
    elif bool(flags["k7_stable"]) and bool(flags["rank7_heavy"]):
        gate, name, selected_k = "BOUND-D", "CANDIDATE_SPACE_NOT_CLOSED", None
    elif bool(flags["k7_stable"]):
        gate, name, selected_k = "BOUND-C", "K7_REQUIRED", 7
    elif bool(flags["k6_stable"]) and bool(flags["k7_immaterial"]):
        gate, name, selected_k = "BOUND-B", "K6_REQUIRED", 6
    else:
        gate, name, selected_k = "BOUND-D", "CANDIDATE_SPACE_NOT_CLOSED", None
    frozen = gate in {"BOUND-A", "BOUND-B", "BOUND-C"}
    return {
        "gate": gate,
        "gate_name": name,
        "selected_generator": "G1_RATE" if frozen else None,
        "selected_k": selected_k,
        "candidate_rule_frozen_for_highlevel_dataset": frozen,
        "k5_sufficient": bool(flags.get("k5_sufficient", False)),
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "mappo_v2_frozen": True,
        "high_level_not_trained": True,
        "trained_hgnn": False,
        "retrained_mappo": False,
        "fresh_confirmation_untouched": True,
        "next_phase_builds_high_level_dataset": frozen,
        "primary_seed": PRIMARY_SEED,
    }
