"""Regression guards for the authorized Round-24 seed extension."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_24"
OUTPUT = ROUND / "outputs" / "round24_multiseed_stability_20260829_1515"
EXPECTED_R23 = "948E89F9197E8204B147FA2F3B9543CD878D4380680254DCB1190E7E9015F22A"
FAMILIES = {"predicted_task_waypoint_transformer", "state_only_transformer", "predicted_task_waypoint_mlp"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round24SeedExtensionTests(unittest.TestCase):
    def test_successor_chain_and_frozen_round23_lock(self) -> None:
        self.assertEqual(digest(ROOT / "round_23" / "ROUND_LOCK.json"), EXPECTED_R23)
        amendment = json.loads((ROOT / "round_23" / "ROUND_LOCK_AMENDMENT.json").read_text(encoding="utf-8"))
        self.assertEqual(amendment["original_round23_lock_sha256"], EXPECTED_R23)
        self.assertFalse(amendment["scientific_artifacts_modified"])
        local_manifest = json.loads((ROUND / "round_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(local_manifest["round25_created"])
        current = (ROOT / "CURRENT_ROUND.txt").read_text(encoding="utf-8")
        self.assertIn(current.splitlines()[0], {"round_24", "round_25", "round_26", "round_27", "round_28", "round_29", "round_30", "round_31", "round_32", "round_33", "round_34", "round_35", "round_36"})
        if current.startswith("round_24\n"):
            self.assertIn("previous_lock_sha256=" + EXPECTED_R23, current)
        elif current.startswith("round_31\n"):
            self.assertIn("previous_lock_sha256=7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125", current)
            self.assertFalse((ROOT / "round_32").exists())
        elif current.startswith("round_32\n"):
            self.assertIn("previous_lock_sha256=9B26D6129264A05A6BD7198FE283D839A70085C7A8D798D4470AD9302CB6E2D3", current)
        elif current.startswith("round_33\n"):
            self.assertIn("previous_lock_sha256=8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F", current)
        elif current.startswith("round_36\n"):
            self.assertIn("previous_lock_sha256=8506179FF0F8E450125FF97FA42FC822D05DDFB9CD1F39CACC655079F972CD82", current)
        elif current.startswith("round_30\n"):
            self.assertIn("previous_lock_sha256=1F4CCAECE01F4B3F127CE4B922A9AC07B0E7E01AC9A34B204192EF8651AC51FB", current)
        elif current.startswith("round_29\n"):
            self.assertIn("previous_lock_sha256=9E3F10B893891B6E0E0415D886E00AEC234CD1376B3146E5704093A210384C91", current)
        elif current.startswith("round_28\n"):
            self.assertIn("previous_lock_sha256=84B1F2993F2DA19C88874C92F33C287E174C3CD9830A7E43E78C73818B8A7267", current)
        else:
            self.assertIn("previous_lock_sha256=B14A668DB825590C6E850C168008DBCB8FAEE4BDB5C56C82B5C906241887043B", current)

    def test_exact_pretraining_fresh_registration_and_disjointness(self) -> None:
        manifest = json.loads((OUTPUT / "round24_fresh_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["namespace"], "round24_seed_stability_dev_v1")
        self.assertTrue(manifest["pre_training_registration"])
        self.assertFalse(manifest["rollout_outcomes_read"])
        self.assertEqual(manifest["scenario_count"], 80)
        self.assertEqual(manifest["disjointness_audit"]["scenario_id_overlap"], [])
        self.assertEqual(manifest["disjointness_audit"]["seed_overlap"], [])
        self.assertEqual({row["base_seed"] for row in manifest["scenarios"]}, set(range(24000, 24010)))

    def test_exact_new_training_matrix_and_frozen_provenance(self) -> None:
        training = json.loads((OUTPUT / "round24_new_seed_training_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(training["formal_training_completed"], "15/15")
        self.assertEqual(len(training["runs"]), 15)
        self.assertEqual({row["family"] for row in training["runs"]}, FAMILIES)
        self.assertEqual({int(row["seed"]) for row in training["runs"]}, {3, 4, 5, 6, 7})
        self.assertEqual(training["old_seeds_retrained"], [])
        self.assertEqual(training["predictor_sha256_before"], training["predictor_sha256_after"])
        self.assertTrue(all(row["finite"] and row["checkpoint_reload_exact"] for row in training["runs"]))

    def test_protocol_and_checkpoint_integrity(self) -> None:
        audit = json.loads((OUTPUT / "round24_protocol_equivalence_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["all_scientific_fields_equal"])
        self.assertTrue(audit["no_architecture_loss_sampler_or_optimizer_change"])
        checkpoints = json.loads((OUTPUT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(checkpoints["old_round22_checkpoints"]), 9)
        self.assertEqual(len(checkpoints["new_round24_checkpoints"]), 15)
        self.assertFalse(checkpoints["old_checkpoints_retrained"])
        self.assertTrue(checkpoints["dataset_hash_unchanged"])

    def test_eight_seed_coverage_uncertainty_and_conditional_availability(self) -> None:
        validation = rows(OUTPUT / "eight_seed_validation_stability.csv")
        fresh = rows(OUTPUT / "round24_fresh_seed_summary.csv")
        self.assertEqual({int(row["seed"]) for row in validation}, set(range(8)))
        self.assertEqual({int(row["seed"]) for row in fresh}, set(range(8)))
        uncertainty = rows(OUTPUT / "eight_seed_uncertainty_audit.csv")
        methods = {row["method"] for row in uncertainty if row["population"] == "round24_fresh_stability"}
        self.assertEqual(methods, {"scenario_bootstrap", "seed_cluster_bootstrap", "hierarchical_seed_then_scenario_bootstrap", "median_of_seeds", "leave_one_seed_out"})
        self.assertEqual({int(row["excluded_seed"]) for row in uncertainty if row["population"] == "round24_fresh_stability" and row["method"] == "leave_one_seed_out"}, set(range(8)))
        conditional = rows(OUTPUT / "eight_seed_conditional_value.csv")
        self.assertTrue(any(row["population"] == "round24_fresh_stability" and row["condition"] == "bridge" and row["available"] == "False" for row in conditional))

    def test_protected_accesses_gates_and_figures(self) -> None:
        access = json.loads((OUTPUT / "access_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(access["round24_fresh_access_count"], 1)
        self.assertEqual(access["round22_fresh_new_rollout_count"], 0)
        self.assertEqual(access["round20_development_test_reuse_count"], 0)
        self.assertEqual(access["untouched_eval_access_count"], 0)
        gate = json.loads((OUTPUT / "round24_seed_stability_gate.json").read_text(encoding="utf-8"))
        readiness = json.loads((OUTPUT / "round24_readiness_gate.json").read_text(encoding="utf-8"))
        self.assertIn(gate["gate"], {"MS-A", "MS-B", "MS-C"})
        self.assertEqual(gate["candidate_ready_for_untouched_eval"], gate["gate"] == "MS-A")
        self.assertFalse(readiness["round25_created"])
        figures = [path for path in (ROUND / "figures" / "round24_multiseed_stability_20260829_1515").iterdir() if path.suffix in {".png", ".svg"}]
        self.assertEqual(len(figures), 34)

    def test_no_nonfinite_csv_values_and_lock_coverage(self) -> None:
        for path in OUTPUT.glob("*.csv"):
            for row in rows(path):
                for value in row.values():
                    try: numeric = float(value)
                    except ValueError: continue
                    self.assertTrue(math.isfinite(numeric), f"{path.name}: {value}")
        lock_path = ROUND / "ROUND_LOCK.json"
        if not lock_path.is_file():
            self.skipTest("Round-24 lock is written only after the pre-lock full regression")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["previous_round_lock_sha256"], EXPECTED_R23)
        self.assertEqual(lock["final_verification"]["required_output_missing"], [])
        for name, value in lock["output_hashes"].items():
            self.assertEqual(digest(OUTPUT / name), value)


if __name__ == "__main__":
    unittest.main()
