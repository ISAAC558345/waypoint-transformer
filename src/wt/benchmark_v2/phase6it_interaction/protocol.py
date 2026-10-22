"""Phase-6I-T gate. Existing checkpoints stay read-only."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase6it_interaction"
MATERIAL_BITS = 1.0e6
BIT_TIE = 1.0
SCORE_DUST = 1.0e-3
AGREEMENT_BAR = 0.75
DRAWS = 10000
SEED = 0
MIN_MATERIAL_SWAPS = 100
MIN_CONTEXT_STATES = 12


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6it refused a reserved-test path: {resolved}")
    return resolved


def assign_gate(
    blocked: bool,
    material_swaps: int,
    additive_flat: bool,
    context_states: int,
    sign_tracks: bool,
) -> str:
    """Support needs additive scores to miss a large disjoint interaction.

    A non-additive head that already tracks the interaction sign keeps the
    result inconclusive. Mean J15 is not an input to this rule.
    """
    if blocked:
        return "ASSIGNMENT_INTERACTION_AUDIT_BLOCKED"
    if int(material_swaps) < MIN_MATERIAL_SWAPS or not additive_flat:
        return "ASSIGNMENT_INTERACTION_LIMIT_INCONCLUSIVE"
    if int(context_states) < MIN_CONTEXT_STATES or sign_tracks:
        return "ASSIGNMENT_INTERACTION_LIMIT_INCONCLUSIVE"
    return "ASSIGNMENT_INTERACTION_LIMIT_SUPPORTED"


__all__ = [
    "AGREEMENT_BAR",
    "BIT_TIE",
    "DRAWS",
    "MATERIAL_BITS",
    "MIN_CONTEXT_STATES",
    "MIN_MATERIAL_SWAPS",
    "OUTPUT",
    "SCORE_DUST",
    "SEED",
    "assert_output_path",
    "assign_gate",
]
