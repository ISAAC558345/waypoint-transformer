"""Regression guards for the frozen Round-25 ambiguity/planner audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_25"
OUTPUT = ROUND / "outputs" / "round25_supervision_ambiguity_alignment_20260830_1203"
FIGURES = ROUND / "figures" / "round25_supervision_ambiguity_alignment_20260830_1203"
EXPECTED_R24 = "B14A668DB825590C6E850C168008DBCB8FAEE4BDB5C56C82B5C906241887043B"

REQUIRED_OUTPUTS = {
    "task_waypoint_supervision_ambiguity_spec.md", "initial_state_target_ambiguity.csv",
    "behavior_target_divergence.csv", "quality_conditioned_target_audit.csv",
    "knn_target_ambiguity.csv", "bridge_supervision_ambiguity.csv",
    "conflict_supervision_ambiguity.csv", "predictor_mean_regression_audit.csv",
    "service_entropy_averaging_audit.csv", "progress_mean_shrinkage_audit.csv",
    "return_mean_regression_audit.csv", "planner_task_waypoint_spec.md",
    "planner_task_waypoint_samples.csv", "planner_behavior_target_alignment.csv",
    "behavior_planner_alignment.csv", "quality_planner_alignment.csv",
    "predictor_planner_alignment.csv", "policy_seed_planner_action_alignment.csv",
    "predictability_vs_planning_utility.csv", "planner_target_horizon_audit.csv",
    "task_waypoint_supervision_ambiguity_gate.json", "planner_task_waypoint_target_gate.json",
    "round25_readiness_gate.json", "access_audit.json", "run_manifest.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Round25SupervisionAmbiguityTests(unittest.TestCase):
    def test_round24_lock_and_frozen_preconditions(self) -> None:
        self.assertEqual(digest(ROOT / "round_24" / "ROUND_LOCK.json"), EXPECTED_R24)
        from wt.round25.audit import verify_frozen_state
        result = verify_frozen_state()
        self.assertTrue(result["passed"])
        self.assertEqual(result["round24_mismatch_count"], 0)
        self.assertEqual(result["round20_development_test_reuse_count"], 0)

    def test_required_outputs_and_exact_state_grouping(self) -> None:
        self.assertEqual({path.name for path in OUTPUT.iterdir() if path.is_file()}, REQUIRED_OUTPUTS)
        manifest = read_json(OUTPUT / "run_manifest.json")
        self.assertEqual(manifest["formal_sample_counts"], {"h5_train_validation": 5120, "h10_train_validation": 1920, "exact_state_groups": 64, "knn_rows": 5120})
        ambiguity = rows(OUTPUT / "initial_state_target_ambiguity.csv")
        self.assertEqual(len(ambiguity), 64)
        self.assertTrue(all(row["behavior_count"] == "10" and row["pair_count"] == "45" for row in ambiguity))

    def test_no_training_or_checkpoint_write_path(self) -> None:
        source = (ROOT / "src" / "wt" / "round25" / "audit.py").read_text(encoding="utf-8")
        for forbidden in ("torch.optim", ".backward(", "torch.save(", "save_checkpoint("):
            self.assertNotIn(forbidden, source)
        manifest = read_json(OUTPUT / "run_manifest.json")
        self.assertFalse(manifest["training_performed"])
        self.assertFalse(manifest["predictor_training_performed"])
        self.assertFalse(manifest["policy_training_performed"])

    def test_metadata_are_diagnostic_and_knn_is_cross_episode(self) -> None:
        source = (ROOT / "src" / "wt" / "round25" / "audit.py").read_text(encoding="utf-8")
        self.assertIn('distance[local, episode == episode[index]] = np.inf', source)
        self.assertIn('"current_graph_state"', source)
        self.assertNotIn("quality_tier_input", source)
        knn = rows(OUTPUT / "knn_target_ambiguity.csv")
        self.assertEqual(len(knn), 5120)
        self.assertTrue(all(row["neighbor_count"] == "5" for row in knn))
        self.assertTrue(all(row["split"] in {"train", "validation"} for row in knn))

    def test_frozen_bridge_conflict_and_causal_planner_protocol(self) -> None:
        source = (ROOT / "src" / "wt" / "round25" / "audit.py").read_text(encoding="utf-8")
        self.assertIn("bridge_flags(dataset), conflict_flags(dataset)", source)
        planner = rows(OUTPUT / "planner_task_waypoint_samples.csv")
        self.assertEqual(len(planner), 128)
        h5 = [row for row in planner if row["planner_horizon"] == "5"]
        self.assertEqual(len(h5), 64)
        self.assertTrue(all(row["causal_current_state_only"] == "True" and row["future_trace_accessed"] == "False" and row["direct_replay"] == "True" and float(row["first_action_max_abs_error"]) <= 1.0e-9 for row in h5))
        self.assertTrue(all(math.isfinite(float(row["team_return_normalized"])) for row in planner))

    def test_planner_targets_are_finite_nonnegative_and_horizon_is_recorded(self) -> None:
        planner = rows(OUTPUT / "planner_task_waypoint_samples.csv")
        for row in planner:
            responsibility = json.loads(row["service_responsibility"])
            progress = json.loads(row["gross_progress_normalized"])
            self.assertTrue(all(math.isfinite(float(value)) and float(value) >= 0.0 for matrix in (responsibility, progress) for vector in matrix for value in vector))
        horizon = rows(OUTPUT / "planner_target_horizon_audit.csv")
        self.assertEqual(len(horizon), 64)
        self.assertEqual(sum(row["secondary_h10_direct_replay"] == "True" for row in horizon), 16)

    def test_predictor_and_policy_are_inference_only(self) -> None:
        access = read_json(OUTPUT / "access_audit.json")
        self.assertEqual(access["predictor_sha256_before"], access["predictor_sha256_after"])
        self.assertTrue(access["policy_checkpoint_inference_only"])
        policy = rows(OUTPUT / "policy_seed_planner_action_alignment.csv")
        self.assertEqual({int(row["seed"]) for row in policy}, set(range(8)))
        self.assertTrue(all(row["inference_only"] == "True" for row in policy))

    def test_access_controls_gates_figures_and_finite_outputs(self) -> None:
        access = read_json(OUTPUT / "access_audit.json")
        self.assertEqual(access["allowed_splits_decoded"], ["train", "validation"])
        self.assertEqual(access["development_test_rows_decoded"], 0)
        self.assertFalse(access["fresh_or_untouched_rollout_performed"])
        self.assertTrue(all(int(access[name]) == 0 for name in ("round22_fresh_new_rollout_count", "round20_development_test_reuse_count", "round24_fresh_new_rollout_count", "untouched_eval_access_count", "untouched_eval_model_access_count", "untouched_eval_policy_rollout_count")))
        self.assertEqual(read_json(OUTPUT / "task_waypoint_supervision_ambiguity_gate.json")["gate"], "SA-TW-C")
        self.assertEqual(read_json(OUTPUT / "planner_task_waypoint_target_gate.json")["gate"], "PT-B")
        self.assertEqual(len([path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"}]), 40)
        for path in OUTPUT.glob("*.csv"):
            for row in rows(path):
                for value in row.values():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    self.assertTrue(math.isfinite(number), f"{path.name}: {value}")

    def test_lock_covers_completed_evidence_when_present(self) -> None:
        lock_path = ROUND / "ROUND_LOCK.json"
        if not lock_path.is_file():
            self.skipTest("Round-25 lock is written only after pre-lock full regression")
        lock = read_json(lock_path)
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["previous_round_lock_sha256"], EXPECTED_R24)
        self.assertEqual(lock["final_verification"]["required_output_missing"], [])
        for name, value in lock["output_hashes"].items():
            self.assertEqual(digest(OUTPUT / name), value)


if __name__ == "__main__":
    unittest.main()
