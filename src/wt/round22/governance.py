"""Strict one-file overlay verification for the approved Round-21 amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wt.governance_state_sync import approved_lifecycle_overlay_sha256


ROOT = Path(".")
R21 = ROOT / "round_21"
TEST_PATH = Path("tests/test_round21_hgnn_task_waypoint.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_round21_amendment_overlay() -> dict[str, Any]:
    """Verify original evidence plus precisely the approved lifecycle change."""
    lock_path, amendment_path = R21 / "ROUND_LOCK.json", R21 / "ROUND_LOCK_AMENDMENT.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")); amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    expected_lock = "F754D441B0A8A3DF77EDAD7EAF2E2CA3062A32025444524270795897E9958BD6"
    required = {
        "amendment_type": "lifecycle_test_scope_fix",
        "original_round21_lock_sha256": expected_lock,
        "original_test_path": TEST_PATH.as_posix(),
        "original_semantic_intent": "Round-22 did not exist when Round-21 was completed.",
        "amended_semantic_intent": "Round-22 must not have existed before Round-21 lock; later legal Round creation is permitted.",
        "scientific_artifacts_modified": False,
        "dataset_modified": False,
        "checkpoints_modified": False,
        "metrics_modified": False,
        "reports_modified": False,
        "neural_training_performed": False,
        "protected_data_accessed": False,
        "original_lock_modified": False,
    }
    if any(amendment.get(key) != value for key, value in required.items()):
        raise RuntimeError("Round-21 amendment fields do not match the approved narrow scope")
    if sha256(lock_path) != expected_lock:
        raise RuntimeError("Round-21 original lock was modified")
    original = lock["source_tree"].get(TEST_PATH.as_posix())
    if original != amendment["original_test_sha256"]:
        raise RuntimeError("amendment original test hash is not the hash registered by the original lock")
    if sha256(ROOT / TEST_PATH) != approved_lifecycle_overlay_sha256(TEST_PATH.as_posix(), amendment["amended_test_sha256"]):
        raise RuntimeError("current lifecycle test is not the explicitly approved amendment")
    checks: list[tuple[Path, str]] = []
    manifest = json.loads((R21 / "round_manifest.json").read_text(encoding="utf-8"))
    checks.extend((R21 / path, digest) for path, digest in lock["configuration_hashes"].items())
    checks.extend((ROOT / path, digest) for path, digest in lock["source_tree"].items() if path != TEST_PATH.as_posix())
    checks.extend((ROOT / manifest["formal_output"] / path, digest) for path, digest in lock["output_hashes"].items())
    checks.extend((ROOT / manifest["figure_directory"] / path, digest) for path, digest in lock["figure_hashes"].items())
    checks.append((ROOT / lock["report"], lock["report_sha256"]))
    mismatches = [{"path": path.as_posix(), "actual": sha256(path) if path.exists() else "MISSING", "expected": digest} for path, digest in checks if not path.exists() or sha256(path) != digest]
    if mismatches:
        raise RuntimeError(f"unapproved Round-21 frozen-evidence mismatch: {mismatches[:3]}")
    return {"passed": True, "original_lock_sha256": expected_lock, "registered_hash_count": len(checks) + 1, "overlaid_path": TEST_PATH.as_posix(), "original_test_sha256": original, "amended_test_sha256": amendment["amended_test_sha256"], "mismatch_count": 0}


__all__ = ["sha256", "verify_round21_amendment_overlay"]
