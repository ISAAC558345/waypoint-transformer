"""Rules locked before any diagnostic trajectory is rolled out."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_priority_tag_signal_audit"
REPORT = ROOT / "docs" / "research" / "HTW_UrgencyAware_Tag_Signal_Audit_Report_CN.md"
E10_OUTPUT = ROOT / "results" / "phase6i_priority_tag_direct_service_reward"
E10_ROWS = E10_OUTPUT / "scenario_rows.csv"
E10_SUMMARY = E10_OUTPUT / "summary.json"
INSTRUCTION_COMMIT = "2aa614df997d766dbb4820fe37a4ac1f7c42e0ed"
E10_RESULT_COMMIT = "ddd397e03a9c02ed4e43022671e115531e842dd2"
E10_SOURCE_COMMIT = "ac1545a2070aab767d5d609b12fd75c2e1ee4d2b"

GATE_COMPLETE = "DIRECT_TAG_SIGNAL_AUDIT_COMPLETE"
GATE_BLOCKED = "DIRECT_TAG_SIGNAL_AUDIT_BLOCKED"
SCHEMA_VERSION = "priority_tag_signal_audit_v1"

DIAGNOSTIC_SCENARIOS = (321000, 321001)
SAFETY_CASES = (
    {"variant": "L3", "model_seed": 1, "scenario_id": 321123, "baseline_variant": "L0", "baseline_seed": 0},
    {"variant": "L3", "model_seed": 2, "scenario_id": 321104, "baseline_variant": "L0", "baseline_seed": 0},
)
L3_SEEDS = (0, 1, 2)
TAG_COLUMN = 1
SNAPSHOTS_PER_TRAJECTORY = 8
COUNTERFACTUAL_STEPS = 4
# Added to every UAV residual in the unit-step frame. Registered before the run.
PERTURBATION = 0.1
CANDIDATE_OFFSETS = (
    (0.0, 0.0),
    (PERTURBATION, 0.0),
    (-PERTURBATION, 0.0),
    (0.0, PERTURBATION),
    (0.0, -PERTURBATION),
)
# A short-horizon screen, not a performance claim.
OPPORTUNITY_TAG_BITS = 1.0
OPPORTUNITY_PAYLOAD_BITS = 1.0
SAFETY_DISTANCE_TOLERANCE_M = 1.0e-9
PROJECTOR_DISTANCE_TOLERANCE_M = 1.0e-6
BIT_TOLERANCE = 1.0
PATH_TOLERANCE_M = 1.0e-9

L3_CHECKPOINT_SHA256 = {
    0: "d47290cc750b6397bdab6e1ce6856139b5da461166ea0e946ebe7026f27d7257",
    1: "6072f7a5de388e27494dcbda3ba59ce2ed5c5575b1b5e52964730f88af641d80",
    2: "d47b371462c31cef2b216f8851cef0de567ccc197df97c76fbea4858fd64258a",
}


def snapshot_indices(horizon: int, count: int = SNAPSHOTS_PER_TRAJECTORY) -> tuple[int, ...]:
    """Left edges of equal time bins. The rule does not look at reward."""
    width = int(horizon)
    bins = int(count)
    if width < bins or bins < 1:
        raise RuntimeError("snapshot grid does not fit the episode")
    return tuple((index * width) // bins for index in range(bins))
