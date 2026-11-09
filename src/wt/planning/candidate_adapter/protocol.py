"""Phase-6I-Y adapter rules. The Phase-6F planner constant stays at k=5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wt.evaluation.closed_loop.protocol import CANDIDATE_K
from wt.control.mappo.protocol import ROOT
from wt.control.executor.protocol import primary_checkpoint
from wt.data.highlevel_graphs.protocol import GENERATOR, HOLD, SELECTED_K
from wt.models.hgnn_assets.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.models.assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT
from wt.legacy.candidate_alignment.protocol import PROBE

OUTPUT = ROOT / "results/phase6iy_candidate_adapter"
CANDIDATE_COUNT = 13327
RATE_ATOL = 1.0e-6
FORMAL_ENTRY_K = int(CANDIDATE_K)


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (EPOCH6_CKPT.resolve(), SCHEME_C_CKPT.resolve(), primary_checkpoint().resolve())
    if resolved in forbidden:
        raise RuntimeError(f"phase-6iy refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iy refused a reserved-test path: {resolved}")
    return resolved


def config_payload() -> dict:
    if GENERATOR != "G1_RATE" or int(SELECTED_K) != 6:
        raise RuntimeError("frozen dataset protocol is not G1_RATE k=6")
    return {
        "generator": GENERATOR,
        "k": int(SELECTED_K),
        "hold": int(HOLD),
        "candidate_count": CANDIDATE_COUNT,
        "builder": "wt.planning.candidate_generation.generators.build_pools",
        "order": "jobs._score_current sorted unique _key",
        "idle": "enumerate_legal allows idle and rejects duplicate buoys",
        "tie_break": "np.argmax of normalized G1_RATE sums on the sorted keys",
        "formal_planner_k_unchanged": FORMAL_ENTRY_K,
    }


def config_sha256() -> str:
    raw = json.dumps(config_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assign_gate(*, comparable: bool, identical: bool) -> str:
    """Supported only when set, order, and score-index binding all pass."""
    if not comparable:
        return "CANDIDATE_POOL_ALIGNMENT_INCONCLUSIVE"
    if identical:
        return "CANDIDATE_POOL_ALIGNMENT_SUPPORTED"
    return "CANDIDATE_POOL_ALIGNMENT_BLOCKED"


__all__ = [
    "CANDIDATE_COUNT",
    "EPOCH6_CKPT",
    "FORMAL_ENTRY_K",
    "OUTPUT",
    "PROBE",
    "RATE_ATOL",
    "RESERVED_TEST",
    "SCHEME_C_CKPT",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "assert_output_path",
    "assign_gate",
    "config_payload",
    "config_sha256",
]