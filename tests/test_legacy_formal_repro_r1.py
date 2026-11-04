"""R1 checks the config copy and the saved 192-step output. It does not open another episode."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE, read_policy
from wt.legacy_formal_repro_r1.protocol import DECISION_TIMES, HORIZON, OUTPUT, ROOT, SCENARIOS, STEP_BUDGET
from wt.legacy_formal_repro_r1.runner import prepare_policy_copy


class LegacyFormalReproR1Tests(unittest.TestCase):
    def test_policy_copy_leaves_the_official_file_unchanged(self) -> None:
        before = POLICY_FILE.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            scheme = prepare_policy_copy("scheme_c", Path(folder) / "scheme_c.json")
            baseline = prepare_policy_copy("g1_rate", Path(folder) / "g1_rate.json")
            self.assertEqual(read_policy(scheme), "scheme_c")
            self.assertEqual(read_policy(baseline), "g1_rate")
            self.assertNotEqual(scheme.resolve(), POLICY_FILE.resolve())
            self.assertNotEqual(baseline.resolve(), POLICY_FILE.resolve())
            with self.assertRaises(RuntimeError):
                prepare_policy_copy("scheme_c", POLICY_FILE)
        self.assertEqual(POLICY_FILE.read_bytes(), before)

    def test_saved_run_matches_the_registered_contract(self) -> None:
        manifest_path = OUTPUT / "run_manifest.json"
        self.assertTrue(manifest_path.is_file(), "the registered run manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["scenarios"], list(SCENARIOS))
        self.assertEqual(int(manifest["horizon"]), HORIZON)
        self.assertEqual(int(manifest["step_budget"]), STEP_BUDGET)
        self.assertEqual(int(manifest["environment_steps"]), STEP_BUDGET)
        self.assertEqual(int(manifest["training_updates"]), 0)
        self.assertTrue(manifest["official_policy_unchanged"])
        self.assertEqual(
            manifest["official_policy_sha256"],
            hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest(),
        )
        self.assertEqual(len(manifest["arms"]), 4)
        for arm in manifest["arms"]:
            self.assertEqual([int(item) for item in arm["decision_times"]], list(DECISION_TIMES))
            self.assertEqual(int(arm["payload_matches_served"]), 1)
            self.assertTrue(arm["step_served_matches_episode"])
            self.assertEqual(int(arm["environment_steps"]), HORIZON)
            copy = Path(arm["policy_path"])
            self.assertNotEqual(copy.resolve(), POLICY_FILE.resolve())
            self.assertTrue(str(copy.resolve()).startswith(str((ROOT / "results" / "legacy_formal_repro_r1").resolve())))
            self.assertEqual(hashlib.sha256(copy.read_bytes()).hexdigest(), arm["policy_sha256_after"])
            served = sum(float(step["served_bits"]) for step in arm["steps"])
            self.assertLessEqual(abs(served - float(arm["payload_bits"])), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
