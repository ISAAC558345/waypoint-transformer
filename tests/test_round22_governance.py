"""Round-22 initialization and approved Round-21 lifecycle-amendment guards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.governance_state_sync import verify_round22_governance_state_sync_amendment
from wt.round22.governance import verify_round21_amendment_overlay


OUTPUT = ROOT / "round_22" / "outputs" / "round22_deepsets_task_waypoint_policy_20260829_1153"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class Round22GovernanceTests(unittest.TestCase):
    def test_round21_original_lock_is_byte_stable(self) -> None:
        self.assertEqual(digest(ROOT / "round_21" / "ROUND_LOCK.json"), "F754D441B0A8A3DF77EDAD7EAF2E2CA3062A32025444524270795897E9958BD6")

    def test_round21_amendment_overlay_is_narrow_and_exact(self) -> None:
        result = verify_round21_amendment_overlay()
        self.assertTrue(result["passed"])
        self.assertEqual(result["overlaid_path"], "tests/test_round21_hgnn_task_waypoint.py")
        self.assertEqual(result["mismatch_count"], 0)

    def test_frozen_predictor_is_unchanged(self) -> None:
        selected = json.loads((OUTPUT / "selected_deepsets_task_waypoint_predictor.json").read_text(encoding="utf-8"))
        self.assertEqual(selected["seed"], 1)
        self.assertEqual(selected["checkpoint_sha256_before_policy_training"], selected["checkpoint_sha256_after_policy_training"])
        self.assertEqual(digest(ROOT / selected["checkpoint"]), selected["checkpoint_sha256_before_policy_training"])

    def test_fresh_manifest_is_registered_and_unaccessed(self) -> None:
        fresh = json.loads((OUTPUT / "fresh_multi_uav_policy_dev_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(fresh["registered_before_policy_training"])
        self.assertEqual(fresh["scenario_count"], 60)
        self.assertEqual(fresh["access_count"], 0)
        self.assertFalse(fresh["outcomes_read"])
        self.assertFalse(fresh["environment_constructed"])

    def test_protected_accesses_follow_the_frozen_gate(self) -> None:
        audit = json.loads((OUTPUT / "access_audit.json").read_text(encoding="utf-8"))
        self.assertIn(audit["fresh_dev_access_count"], (0, 1))
        if audit["fresh_dev_access_count"] == 1:
            gate = json.loads((OUTPUT / "round22_validation_policy_gate.json").read_text(encoding="utf-8"))
            fresh = json.loads((OUTPUT / "fresh_dev_access_audit.json").read_text(encoding="utf-8"))
            self.assertIn(gate["gate"], ("V-A", "V-B"))
            self.assertEqual(fresh["fresh_dev_access_count"], 1)
            self.assertTrue(fresh["access_is_single_use"])
        self.assertEqual(audit["untouched_eval_model_access_count"], 0)
        self.assertEqual(audit["untouched_eval_policy_rollout_count"], 0)
        self.assertEqual(audit["round20_development_test_reuse_count"], 0)

    def test_round23_was_not_created_at_round22_lock_time(self) -> None:
        lock = json.loads((ROOT / "round_22" / "ROUND_LOCK.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "round_22" / "round_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(lock["round23_created"])
        self.assertFalse(manifest["round23_created"])

    def test_round22_governance_state_sync_amendment_is_narrow_and_exact(self) -> None:
        result = verify_round22_governance_state_sync_amendment()
        self.assertTrue(result["passed"])
        self.assertEqual(result["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
