"""Deployment-requirement contract. Unspecified limits stay unnamed numbers."""
from __future__ import annotations

import json
from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6iae_deployment_contract"
REQUIREMENTS_FILE = OUTPUT / "deployment_requirements.json"
REGRESSION_SCENARIO = 322002
BASELINE_SCENARIO = 322000
BASELINE_STOP = 15
# Full-episode payload recorded for 322002 before this fault-handling change.
REGRESSION_PAYLOAD_BITS = 69252475.119
REGRESSION_T0_INDEX = 7736
REGRESSION_T0_ASSIGNMENT = (2, 0, 1, 3, 5, 4)
PAYLOAD_TOLERANCE_BITS = 1.0


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (
        EPOCH6_CKPT.resolve(),
        SCHEME_C_CKPT.resolve(),
        primary_checkpoint().resolve(),
        POLICY_FILE.resolve(),
    )
    if resolved in forbidden:
        raise RuntimeError(f"phase-6iae refused to write {resolved}")
    return resolved


def load_requirements(path: Path | None = None) -> dict:
    file = REQUIREMENTS_FILE if path is None else Path(path)
    return json.loads(file.read_text(encoding="utf-8"))


def pending_requirements(document: dict | None = None) -> list[str]:
    payload = load_requirements() if document is None else document
    return [str(item) for item in payload["pending_confirmation"]]


def assign_gate(*, pending: int, fault_failures: int, regression_ok: bool) -> str:
    """A numeric limit that is still unnamed keeps the contract incomplete."""
    if int(fault_failures) > 0 or not bool(regression_ok):
        return "DEPLOYMENT_FAULT_HANDLING_BLOCKED"
    if int(pending) > 0:
        return "DEPLOYMENT_REQUIREMENTS_INCOMPLETE"
    return "DEPLOYMENT_CONTRACT_READY"


def status_flags(*, gate: str, planner_ready: bool) -> dict[str, bool]:
    return {
        "planner_ready": bool(planner_ready) and gate != "DEPLOYMENT_FAULT_HANDLING_BLOCKED",
        "deployment_verified": False,
    }
