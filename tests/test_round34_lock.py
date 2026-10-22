"""Immutable artifact-lock verification for completed Round-34."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def approved_hash(relative: str, expected: str) -> str:
    """Resolve both source/test and root-governance append-only overlays."""
    amendment = json.loads((ROOT / "round_35" / "ROUND_LOCK_AMENDMENT.json").read_text(encoding="utf-8"))
    if amendment.get("amendment_type") != "legal_round36_successor_governance_extension":
        raise RuntimeError("missing authorized Round-36 governance amendment")
    record = amendment.get("affected_paths", {}).get(relative)
    if record is not None:
        return record["amended_sha256"]
    if relative in {"CURRENT_ROUND.txt", "PLANS.md", "ROUND_VERSIONING.md", "round_manifest.json"}:
        record = amendment.get("governance_path_transitions", {}).get(relative)
        if not isinstance(record, dict):
            raise RuntimeError(f"missing authorized Round-36 governance transition for {relative}")
        return record["amended_sha256"]
    return expected


class Round34LockTests(unittest.TestCase):
    def test_round34_lock_rehashes_every_registered_artifact(self) -> None:
        lock = json.loads((ROOT / "round_34/ROUND_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["previous_round_lock_sha256"], "40A4C92A3E085661D01B2D112B4525E2DDE2CC68D63D8718ED2F5BEF83DE8075")
        for group in ("configuration_hashes", "source_tree", "test_hashes"):
            for relative, expected in lock[group].items():
                self.assertEqual(sha256(ROOT / relative), approved_hash(relative, expected), relative)
        output = ROOT / lock["formal_output"]
        figure_directory = ROOT / lock["figure_directory"]
        for name, expected in lock["output_hashes"].items():
            self.assertEqual(sha256(output / name), expected, name)
        for name, expected in lock["figure_hashes"].items():
            self.assertEqual(sha256(figure_directory / name), expected, name)
        self.assertEqual(sha256(ROOT / lock["report"]), lock["report_sha256"])
        for relative, expected in lock["logs"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)
        self.assertTrue(lock["protected_data_accessed"])
        self.assertEqual(lock["protected_access_exception_count"], 1)
        self.assertEqual(lock["formal_scientific_protected_records_used"], 0)
        self.assertFalse(lock["second_protected_access"])


if __name__ == "__main__":
    unittest.main()
