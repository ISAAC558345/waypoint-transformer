"""Narrow lifecycle-compatibility regression tests for the Round-33 seal."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.governance_state_sync import (
    EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256,
    EXPECTED_R32_LOCK_SHA256,
    sha256,
    validate_round32_round33_lifecycle_amendment,
    verify_round32_locked_scientific_artifacts,
    verify_round32_round33_lifecycle,
    verify_round33_scientific_artifact_hashes,
)


COMPATIBILITY_PATH = ROOT / "round_33" / "GOVERNANCE_COMPATIBILITY_AMENDMENT.json"


def compatibility_amendment() -> dict:
    return json.loads(COMPATIBILITY_PATH.read_text(encoding="utf-8"))


class Round33GovernanceCompatibilityTests(unittest.TestCase):
    def test_01_round32_original_lock_sha_is_unchanged(self) -> None:
        result = verify_round32_round33_lifecycle()
        self.assertTrue(result["passed"])
        self.assertEqual(result["original_round32_lock_sha256"], EXPECTED_R32_LOCK_SHA256)

    def test_02_round33_without_compatibility_amendment_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "compatibility amendment is required"):
            validate_round32_round33_lifecycle_amendment(
                None,
                original_lock_sha256=EXPECTED_R32_LOCK_SHA256,
                lifecycle_amendment_sha256=EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256,
                round33_exists=True,
                round34_exists=False,
            )

    def test_03_exact_amendment_authorizes_only_round33(self) -> None:
        result = validate_round32_round33_lifecycle_amendment(
            compatibility_amendment(),
            original_lock_sha256=EXPECTED_R32_LOCK_SHA256,
            lifecycle_amendment_sha256=EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256,
            round33_exists=True,
            round34_exists=False,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["authorized_successor_round"], "round_33")

    def test_04_tampered_original_lock_binding_fails(self) -> None:
        amended = copy.deepcopy(compatibility_amendment())
        amended["original_round32_lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not narrow and exact"):
            validate_round32_round33_lifecycle_amendment(
                amended,
                original_lock_sha256=EXPECTED_R32_LOCK_SHA256,
                lifecycle_amendment_sha256=EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256,
                round33_exists=True,
                round34_exists=False,
            )

    def test_05_round34_is_rejected_even_with_an_amendment(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Round-34 is not authorized"):
            validate_round32_round33_lifecycle_amendment(
                compatibility_amendment(),
                original_lock_sha256=EXPECTED_R32_LOCK_SHA256,
                lifecycle_amendment_sha256=EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256,
                round33_exists=True,
                round34_exists=True,
            )

    def test_06_round32_scientific_artifact_tampering_fails(self) -> None:
        target = ROOT / "round_32" / "outputs" / "round32_horizon_semantics_audit_20260830_2231" / "round32_horizon_semantics_gate.json"

        def tampered_digest(path: Path) -> str:
            return "0" * 64 if path == target else sha256(path)

        with self.assertRaisesRegex(RuntimeError, "locked scientific evidence was modified"):
            verify_round32_locked_scientific_artifacts(digest=tampered_digest)

    def test_07_existing_round33_scientific_outputs_are_unchanged(self) -> None:
        result = verify_round33_scientific_artifact_hashes(compatibility_amendment())
        self.assertTrue(result["passed"])
        self.assertGreater(result["checked_artifact_count"], 0)


if __name__ == "__main__":
    unittest.main()
