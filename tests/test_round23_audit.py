"""Regression guards for the locked-artifact Round-23 mechanism audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.round23.audit import EXPECTED_PREDICTOR, EXPECTED_R22_LOCK, FIGURES, OUTPUT, REQUIRED_OUTPUTS, sha256, verify_frozen_state


def rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


class Round23AuditTests(unittest.TestCase):
    def test_round22_frozen_lock_amendment_predictor_and_checkpoints_verify(self) -> None:
        audit = verify_frozen_state()
        self.assertEqual(sha256(ROOT / "round_22" / "ROUND_LOCK.json"), EXPECTED_R22_LOCK)
        self.assertTrue(audit["round22_governance_amendment_verified"])
        self.assertEqual(audit["predictor_sha256"], EXPECTED_PREDICTOR)
        self.assertEqual(audit["formal_best_last_checkpoint_failures"], [])
        self.assertEqual(audit["formal_policy_training"], "12/12")

    def test_round23_is_a_frozen_inference_audit_with_protected_accesses_zero(self) -> None:
        manifest = json.loads((ROOT / "round_23" / "round_manifest.json").read_text(encoding="utf-8"))
        access = json.loads((OUTPUT / "access_audit.json").read_text(encoding="utf-8"))
        self.assertIn(manifest["status"], {"in_progress", "locked"})
        self.assertFalse(manifest["training_performed"])
        self.assertFalse(manifest["policy_training_performed"])
        self.assertEqual(access["fresh_dev_new_rollout_count"], 0)
        self.assertEqual(access["untouched_eval_access_count"], 0)
        self.assertTrue(access["validation_only_new_inference"])
        self.assertFalse(access["uses_true_phi"])
        # The lock records historical Round-23 evidence: no Round-24 existed
        # *at lock time*.  A later explicitly authorized successor may exist
        # without changing that evidence (see ROUND_LOCK_AMENDMENT.json).
        self.assertFalse(manifest["round24_created"])

    def test_round23_source_has_no_training_or_checkpoint_creation_path(self) -> None:
        source = (ROOT / "src" / "wt" / "round23" / "audit.py").read_text(encoding="utf-8").lower()
        for forbidden in ("torch.optim", ".backward(", "torch.save(", "train_one("):
            self.assertNotIn(forbidden, source)
        for forbidden in ("hgnn", "ppo", "mappo", "marl"):
            self.assertIsNone(re.search(rf"\\b{forbidden}\\b", source))

    def test_all_required_outputs_figures_and_completion_records_exist(self) -> None:
        self.assertEqual([name for name in REQUIRED_OUTPUTS if not (OUTPUT / name).is_file()], [])
        figures = [path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"}]
        self.assertEqual(len(figures), 40)
        self.assertTrue((ROOT / "round_23" / "report" / "round23_completion_report.md").is_file())
        self.assertTrue((ROOT / "round_23" / "logs" / "implementation_log.md").is_file())

    def test_seed_identity_and_fresh_pairing_are_derived_from_locked_rows(self) -> None:
        identity = json.loads((OUTPUT / "seed_identity_audit.json").read_text(encoding="utf-8"))
        summary = rows("seed_level_policy_summary.csv")
        self.assertEqual(identity["positive_seed_id"], 2)
        self.assertEqual(identity["negative_seed_ids"], [0, 1])
        self.assertEqual(len(summary), 3)
        matrix = rows("seed_scenario_delta_matrix.csv")
        self.assertEqual(len(matrix), 60)
        self.assertTrue(all({"seed_0_delta_payload_bits", "seed_1_delta_payload_bits", "seed_2_delta_payload_bits"} <= set(row) for row in matrix))
        self.assertEqual(len(rows("seed_win_loss_audit.csv")), 3)
        self.assertEqual(len(rows("seed_tail_contribution_audit.csv")), 3)

    def test_bootstrap_leave_one_seed_out_and_frozen_groups_are_registered(self) -> None:
        bootstrap = rows("hierarchical_seed_bootstrap.csv")
        methods = {row["method"] for row in bootstrap}
        self.assertEqual(methods, {"original_scenario_bootstrap", "seed_cluster_bootstrap", "hierarchical_seed_then_scenario_bootstrap", "leave_one_seed_out"})
        original = next(row for row in bootstrap if row["method"] == "original_scenario_bootstrap")
        self.assertEqual((int(original["bootstrap_draws"]), int(original["bootstrap_seed"])), (2000, 22022))
        excluded = {int(row["excluded_seed"]) for row in bootstrap if row["method"] == "leave_one_seed_out"}
        self.assertEqual(excluded, {0, 1, 2})
        action_groups = {row["group"] for row in rows("seed_grouped_action_error.csv")}
        self.assertTrue({"interference_conflict", "coordination_bridge", "K3", "K5", "eta_0.25", "eta_0.5", "eta_1", "high_queue"} <= action_groups)

    def test_head_ablations_are_validation_paired_and_inference_only(self) -> None:
        utilization = rows("seed_head_utilization.csv")
        conditions = {row["condition"] for row in utilization}
        self.assertEqual(conditions, {"normal", "zero_full", "shuffle_full", "zero_service", "zero_progress", "zero_return", "shuffle_service", "shuffle_progress", "shuffle_return"})
        causal = rows("validation_head_causal_ablation.csv")
        self.assertEqual(len(causal), 288)
        self.assertTrue(all(row["validation_only"] == "True" and row["inference_only"] == "True" and row["same_registered_validation_snapshot"] == "True" for row in causal))
        gates = json.loads((OUTPUT / "task_waypoint_head_causal_gate.json").read_text(encoding="utf-8"))
        self.assertEqual(set(gates["heads"]), {"service", "progress", "return"})

    def test_no_nonfinite_numeric_results(self) -> None:
        for path in OUTPUT.glob("*.csv"):
            for row in rows(path.name):
                for value in row.values():
                    try:
                        numeric = float(value)
                    except ValueError:
                        continue
                    self.assertTrue(math.isfinite(numeric), f"{path.name} contains non-finite {value}")
        for path in OUTPUT.glob("*.json"):
            self.assertTrue(finite(json.loads(path.read_text(encoding="utf-8"))), path.name)

    def test_round23_lock_hashes_its_sources_outputs_figures_and_report(self) -> None:
        path = ROOT / "round_23" / "ROUND_LOCK.json"
        if not path.is_file():
            self.skipTest("Round-23 lock is written only after the pre-lock full regression")
        lock = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["previous_round_lock_sha256"], EXPECTED_R22_LOCK)
        self.assertEqual(lock["final_verification"]["full_regression_tests"]["count"], 857)
        self.assertTrue(lock["final_verification"]["compileall_passed"])
        self.assertEqual(lock["final_verification"]["required_output_missing"], [])
        for name, digest in lock["output_hashes"].items():
            self.assertEqual(sha256(OUTPUT / name), digest)
        for name, digest in lock["figure_hashes"].items():
            self.assertEqual(sha256(FIGURES / name), digest)
        self.assertEqual(sha256(ROOT / lock["report"]), lock["report_sha256"])


if __name__ == "__main__":
    unittest.main()
