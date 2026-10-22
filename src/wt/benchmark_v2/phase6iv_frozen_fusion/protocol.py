"""Phase-6I-V fusion rules. Thresholds are fixed before scoring."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT
from wt.benchmark_v2.phase6iu_pairwise_head.protocol import HEAD_OUTPUT, OUTPUT as PAIR_OUTPUT

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase6iv_frozen_fusion"
RULES = ("B", "C", "D", "E")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
BIT_TIE = 1.0
MIN_POSITIVE_SCENARIOS = 12
CONCENTRATION = 0.50
PUBLISHED_TOP1_COUNTS = {
    "epoch6": 27,
    "scheme_c": 32,
    "conditioned_0": 32,
    "conditioned_1": 28,
    "pair_0": 27,
    "pair_1": 27,
    "pair_2": 29,
    "pair_3": 28,
    "pair_4": 28,
}
EXPECTED_EPOCH = {"conditioned_0": 3, "conditioned_1": 3, "pair_0": 1, "pair_1": 1, "pair_2": 1, "pair_3": 1, "pair_4": 1}


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (EPOCH6_CKPT.resolve(), SCHEME_C_CKPT.resolve(), HEAD_OUTPUT.resolve(), PAIR_OUTPUT.resolve())
    if resolved in forbidden or any(resolved.is_relative_to(item) for item in forbidden if item.exists()):
        raise RuntimeError(f"phase-6iv refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iv refused a reserved-test path: {resolved}")
    return resolved


def assign_gate(blocked: bool, checks_passed: bool, rules: list[dict]) -> str:
    """One pre-registered rule must clear every decision bar.

    Mean J15 is not an input. Raw-score mixtures are not rules.
    """
    if blocked or not checks_passed or len(rules) != len(RULES):
        return "FROZEN_FUSION_AUDIT_BLOCKED"
    if any(bool(rule["supported"]) for rule in rules):
        return "FROZEN_FUSION_GAIN_SUPPORTED"
    return "FROZEN_FUSION_GAIN_INCONCLUSIVE"


def rule_supported(report: dict) -> bool:
    required = (
        "top1_above_scheme_c",
        "regret_not_worse",
        "mean_ci_above_scheme_c",
        "scenarios_consistent",
        "gain_not_concentrated",
    )
    return all(bool(report[name]) for name in required)


__all__ = [
    "BIT_TIE",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CONCENTRATION",
    "EXPECTED_EPOCH",
    "HEAD_OUTPUT",
    "MIN_POSITIVE_SCENARIOS",
    "OUTPUT",
    "PAIR_OUTPUT",
    "PUBLISHED_TOP1_COUNTS",
    "RESERVED_TEST",
    "RULES",
    "SCHEME_C_CKPT",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "assert_output_path",
    "assign_gate",
    "rule_supported",
]
