"""Lifecycle and clean-restart guards for the Round-34 governance exception."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.governance_state_sync import EXPECTED_R33_LOCK_SHA256, verify_round35_round36_lifecycle


class Round34GovernanceLifecycleTests(unittest.TestCase):
    def test_01_round33_original_lock_remains_unchanged(self) -> None:
        self.assertEqual(hashlib.sha256((ROOT / "round_33/ROUND_LOCK.json").read_bytes()).hexdigest().upper(), EXPECTED_R33_LOCK_SHA256)

    def test_02_successor_amendment_verifies(self) -> None:
        result = verify_round35_round36_lifecycle()
        self.assertTrue(result["passed"])
        self.assertEqual(result["authorized_successor_round"], "round_36")

    def test_03_exception_is_exactly_one_record(self) -> None:
        exception = json.loads((ROOT / "round_34/ROUND34_PROTECTED_ACCESS_EXCEPTION.json").read_text(encoding="utf-8"))
        self.assertEqual(exception["event_type"], "accidental_preflight_protected_split_decode")
        self.assertEqual(exception["records_decoded"], 1)
        self.assertEqual(exception["protected_split"], "development_test")
        self.assertEqual(exception["first_detection_action"], "immediate_stop")

    def test_04_exception_has_no_scientific_use(self) -> None:
        exception = json.loads((ROOT / "round_34/ROUND34_PROTECTED_ACCESS_EXCEPTION.json").read_text(encoding="utf-8"))
        for key in ("scientific_metric_computed_from_record", "scientific_selection_influenced", "gate_influenced"):
            self.assertFalse(exception[key])

    def test_05_clean_manifest_preflight_is_metadata_only(self) -> None:
        for stem in ("sample", "transition"):
            preflight = json.loads((ROOT / f"round_34/round34_clean_{stem}_preflight.json").read_text(encoding="utf-8"))
            self.assertTrue(preflight["metadata_only"])
            self.assertFalse(preflight["json_payload_decoded"])
            self.assertEqual(preflight["unknown_split_count"], 0)
            self.assertEqual(preflight["untouched_in_formal_input"], 0)

    def test_06_clean_manifest_locks_are_stable(self) -> None:
        for stem in ("sample", "transition"):
            manifest = ROOT / f"round_34/round34_clean_{stem}_manifest.json"
            lock = json.loads((ROOT / f"round_34/round34_clean_{stem}_manifest.lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest().upper())

    def test_07_formal_access_audit_is_clean_after_exception(self) -> None:
        audit = json.loads((ROOT / "round_34/outputs/round34_benchmark_complexity_audit_20260901_1540/access_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["formal_scientific_protected_records_used"], 0)
        self.assertFalse(audit["second_protected_access"])
        self.assertEqual(audit["untouched_eval_access_count"], 0)
        self.assertTrue(audit["round34_scientific_analysis_clean_after_exception"])

    def test_08_root_and_round_manifest_agree(self) -> None:
        root_manifest = json.loads((ROOT / "round_manifest.json").read_text(encoding="utf-8"))
        round_manifest = json.loads((ROOT / "round_34/round_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(root_manifest["current_round"], "round_36")
        self.assertEqual(round_manifest["previous_round_lock_sha256"], EXPECTED_R33_LOCK_SHA256)
        self.assertTrue(round_manifest["protected_data_accessed"])

    def test_09_authorized_successor_chain_exists(self) -> None:
        self.assertTrue((ROOT / "round_35" / "ROUND_LOCK.json").is_file())
        self.assertTrue((ROOT / "round_36" / "round_manifest.json").is_file())

    def test_10_quarantine_is_excluded(self) -> None:
        boundary = json.loads((ROOT / "round_34/quarantine/CONTAMINATION_BOUNDARY.json").read_text(encoding="utf-8"))
        self.assertTrue(boundary["excluded_from_scientific_analysis"])


if __name__ == "__main__":
    unittest.main()
