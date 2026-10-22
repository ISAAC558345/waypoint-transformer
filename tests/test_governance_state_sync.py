"""Regression guard for completed-round root-governance synchronization."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.governance_state_sync import verify_round22_governance_state_sync_amendment


def current_fields() -> tuple[str, dict[str, str]]:
    lines = (ROOT / "CURRENT_ROUND.txt").read_text(encoding="utf-8").splitlines()
    return lines[0], dict(line.split("=", 1) for line in lines[1:] if "=" in line)


class GovernanceStateSyncTests(unittest.TestCase):
    def test_round22_original_lock_and_narrow_amendment_verify(self) -> None:
        result = verify_round22_governance_state_sync_amendment()
        self.assertTrue(result["passed"])
        self.assertEqual(result["mismatch_count"], 0)

    def test_current_state_preserves_round22_evidence_and_legal_successor_accesses(self) -> None:
        current_round, current = current_fields()
        lock = json.loads((ROOT / "round_22" / "ROUND_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "locked")
        self.assertIn(current_round, ("round_22", "round_23", "round_24", "round_25", "round_26", "round_27", "round_28", "round_29", "round_30", "round_31", "round_32", "round_33", "round_34", "round_35", "round_36"))
        self.assertIn(current["status"], {"locked"} if current_round == "round_22" else {"in_progress", "locked"})
        if current_round == "round_31":
            self.assertEqual(current["previous_lock_file"], "round_30/ROUND_LOCK.json")
            self.assertEqual(current["previous_lock_sha256"], "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125")
            self.assertEqual(current["round31_fresh_new_rollout_count"], "0")
            self.assertEqual(current["untouched_eval_access_count"], "0")
        elif current_round == "round_32":
            self.assertEqual(current["previous_lock_file"], "round_31/ROUND_LOCK.json")
            self.assertEqual(current["previous_lock_sha256"], "9B26D6129264A05A6BD7198FE283D839A70085C7A8D798D4470AD9302CB6E2D3")
            self.assertEqual(current["fresh_new_rollout_count"], "0")
            self.assertEqual(current["untouched_eval_accessed"], "false")
        elif current_round == "round_33":
            self.assertEqual(current["previous_lock_file"], "round_32/ROUND_LOCK.json")
            self.assertEqual(current["previous_lock_sha256"], "8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F")
            self.assertEqual(current["development_test_reuse_count"], "0")
            self.assertEqual(current["untouched_eval_access_count"], "0")
        elif current_round == "round_34":
            self.assertEqual(current["previous_lock_file"], "round_33/ROUND_LOCK.json")
            self.assertEqual(current["previous_lock_sha256"], "40A4C92A3E085661D01B2D112B4525E2DDE2CC68D63D8718ED2F5BEF83DE8075")
            self.assertEqual(current["protected_data_accessed"], "true")
            self.assertEqual(current["development_test_access_count"], "1")
            self.assertEqual(current["formal_scientific_protected_records_used"], "0")
            self.assertEqual(current["untouched_eval_access_count"], "0")
        elif current_round == "round_35":
            self.assertEqual(current["previous_lock_file"], "round_34/ROUND_LOCK.json")
            self.assertEqual(current["previous_lock_sha256"], "59A472D367BEE8364C2D5CC4468759685795E6CB7E2176DFFD2B12F56C489AF0")
            self.assertEqual(current["protected_data_access_count"], "0")
            self.assertEqual(current["untouched_eval_access_count"], "0")
            self.assertEqual(current["second_protected_access"], "false")
        elif current_round == "round_36":
            self.assertEqual(current["previous_lock_file"], "round_35/ROUND_LOCK.json")
            self.assertEqual(current["previous_lock_sha256"], "8506179FF0F8E450125FF97FA42FC822D05DDFB9CD1F39CACC655079F972CD82")
            self.assertEqual(current["protected_access_count"], "0")
            self.assertEqual(current["untouched_access_count"], "0")
            self.assertEqual(current["round37_created"], "false")
        else:
            self.assertEqual(current["fresh_dev_access_count"], str(lock["fresh_dev_access_count"]))
            self.assertEqual(current["fresh_dev_access_count"], "1")
            self.assertEqual(current["untouched_eval_model_access_count"], str(lock["untouched_eval_model_access_count"]))
            self.assertEqual(current["untouched_eval_model_access_count"], "0")

    def test_round23_absence_is_historical_round22_lock_evidence(self) -> None:
        lock = json.loads((ROOT / "round_22" / "ROUND_LOCK.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "round_22" / "round_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(lock["round23_created"])
        self.assertFalse(manifest["round23_created"])

    def test_governance_amendment_allows_only_future_authorized_round23_creation(self) -> None:
        amendment = json.loads((ROOT / "round_22" / "ROUND_LOCK_AMENDMENT.json").read_text(encoding="utf-8"))
        self.assertTrue(amendment["resume_round23_creation_allowed"])


if __name__ == "__main__":
    unittest.main()
