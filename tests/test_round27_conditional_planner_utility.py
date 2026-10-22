"""Regression guards for the locked Round-27 causal utility audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_27"
OUTPUT = ROUND / "outputs" / "round27_conditional_planner_utility_20260830_1447"
FIGURES = ROUND / "figures" / "round27_conditional_planner_utility_20260830_1447"
LOGS = ROUND / "logs"
EXPECTED_R26 = "606558C28019A0B4C49DB0CE1B09B10CCC49750CAEF8648B1648D3BC66666FEE"

REQUIRED_OUTPUTS = {
    "planner_utility_definition.md", "planner_state_utility_labels.csv",
    "planner_utility_by_group.csv", "causal_trigger_feature_spec.md",
    "planner_utility_distribution.csv", "oracle_conditional_routing_upper_bound.csv",
    "causal_trigger_metrics.csv", "trigger_payload_utility.csv",
    "trigger_generalization_audit.csv", "conditional_planner_behavior_alignment.csv",
    "conditional_distillation_design.md", "planner_utility_taskwp_relation.csv",
    "planner_utility_seed_disagreement.csv", "conditional_planner_action_gate.json",
    "causal_planning_trigger_gate.json", "round27_readiness_gate.json", "run_manifest.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round27ConditionalPlannerUtilityTests(unittest.TestCase):
    def test_round26_lock_and_frozen_preconditions(self) -> None:
        self.assertEqual(digest(ROOT / "round_26" / "ROUND_LOCK.json"), EXPECTED_R26)
        from wt.round27.audit import verify_frozen_state
        result = verify_frozen_state()
        self.assertTrue(result["passed"])
        self.assertEqual(result["fresh_new_rollout_count"], 0)
        self.assertEqual(result["untouched_eval_access_count"], 0)

    def test_required_delivery_and_registered_state_pool(self) -> None:
        self.assertEqual({path.name for path in OUTPUT.iterdir() if path.is_file()}, REQUIRED_OUTPUTS)
        manifest = read_json(OUTPUT / "run_manifest.json")
        self.assertEqual(manifest["formal_sample_counts"], {
            "state_pool": 64, "train": 48, "validation": 16, "counterfactual_branches": 256,
        })
        labels = rows(OUTPUT / "planner_state_utility_labels.csv")
        self.assertEqual(len(labels), 64)
        self.assertEqual(sum(row["split"] == "train" for row in labels), 48)
        self.assertEqual(sum(row["split"] == "validation" for row in labels), 16)
        self.assertTrue(all(row["t"] == "0" and row["same_snapshot_verified"] == "True" and row["only_first_action_differs"] == "True" for row in labels))

    def test_direct_utility_formula_and_three_way_label(self) -> None:
        for row in rows(OUTPUT / "planner_state_utility_labels.csv"):
            utility = float(row["utility_payload"])
            self.assertAlmostEqual(utility, float(row["planner_payload_bits"]) - float(row["reference_payload_bits"]), places=6)
            expected = "useful" if utility > 1.0e-6 else "harmful" if utility < -1.0e-6 else "neutral"
            self.assertEqual(row["utility_class"], expected)
            self.assertEqual(row["continuation_policy"], "frozen_causal_joint_h1")
            self.assertEqual(row["continuation_horizon"], "1")

    def test_no_training_and_no_future_trigger_inputs(self) -> None:
        source = (ROOT / "src" / "wt" / "round27" / "audit.py").read_text(encoding="utf-8")
        for forbidden in ("torch.optim", ".backward(", "torch.save(", "save_checkpoint("):
            self.assertNotIn(forbidden, source)
        specification = (OUTPUT / "causal_trigger_feature_spec.md").read_text(encoding="utf-8")
        self.assertIn("Bridge, behavior, quality, Task-Waypoint", specification)
        trigger = read_json(OUTPUT / "causal_planning_trigger_gate.json")
        self.assertTrue(trigger["deployable_features_only"])
        self.assertFalse(trigger["bridge_used_as_trigger"])
        self.assertFalse(read_json(OUTPUT / "run_manifest.json")["training_performed"])

    def test_trigger_threshold_is_train_only_and_oracle_is_nondeployable(self) -> None:
        trigger = read_json(OUTPUT / "causal_planning_trigger_gate.json")
        threshold = float(trigger["train_queue_competition_p75"])
        metrics = rows(OUTPUT / "causal_trigger_metrics.csv")
        self.assertTrue(all(math.isclose(float(row["queue_competition_threshold"]), threshold, rel_tol=0.0, abs_tol=1.0e-15) for row in metrics))
        oracle = rows(OUTPUT / "oracle_conditional_routing_upper_bound.csv")
        self.assertTrue(all(row["oracle_trigger"] == "True" and row["deployable"] == "False" for row in oracle))
        self.assertTrue(all(row["threshold_source"] in {"none", "train_p75"} for row in metrics))

    def test_frozen_policy_access_controls_and_expected_gates(self) -> None:
        access = read_json(LOGS / "access_audit.json")
        self.assertEqual(access["allowed_splits_decoded"], ["train", "validation"])
        self.assertTrue(access["policy_checkpoint_inference_only"])
        self.assertFalse(access["bridge_used_as_trigger"])
        self.assertTrue(all(int(access[name]) == 0 for name in (
            "development_test_rows_decoded", "fresh_new_rollout_count",
            "round20_development_test_reuse_count", "round24_fresh_access_count",
            "untouched_eval_access_count", "future_trigger_feature_access_count",
            "new_scenarios_created",
        )))
        self.assertEqual(read_json(OUTPUT / "conditional_planner_action_gate.json")["gate"], "CPA-B")
        self.assertEqual(read_json(OUTPUT / "causal_planning_trigger_gate.json")["gate"], "TR-B")
        self.assertEqual(read_json(OUTPUT / "round27_readiness_gate.json")["recommendation"], "R27-B_continue_trigger_representation_study_no_training")

    def test_figures_finite_outputs_and_historical_no_round28(self) -> None:
        self.assertEqual(len([path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"}]), 32)
        self.assertFalse(read_json(ROUND / "ROUND_LOCK.json")["round28_created"])
        self.assertTrue((ROOT / "round_28" / "ROUND_LOCK.json").is_file())
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
            self.skipTest("Round-27 lock is written only after pre-lock full regression")
        lock = read_json(lock_path)
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["previous_round_lock_sha256"], EXPECTED_R26)
        self.assertEqual(lock["final_verification"]["required_output_missing"], [])
        for name, value in lock["output_hashes"].items():
            self.assertEqual(digest(OUTPUT / name), value)


if __name__ == "__main__":
    unittest.main()
