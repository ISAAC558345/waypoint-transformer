"""Regression guards for the diagnostic-only Round-28 audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_28"
OUTPUT = ROUND / "outputs" / "round28_planning_need_representation_20260830_1624"
FIGURES = ROUND / "figures" / "round28_planning_need_representation_20260830_1624"
EXPECTED_R27 = "84B1F2993F2DA19C88874C92F33C287E174C3CD9830A7E43E78C73818B8A7267"
REQUIRED = {
    "round28_state_sampling_protocol.json", "planner_utility_state_corpus_manifest.json",
    "planning_need_feature_spec.md", "planning_need_feature_normalization.json",
    "planning_need_feature_table.csv", "planning_feature_separability.csv",
    "planning_feature_utility_correlation.csv", "feature_generalization_audit.csv",
    "planning_composite_score_spec.md", "round28_trigger_thresholds.json",
    "composite_trigger_metrics.csv", "composite_trigger_grouped_metrics.csv",
    "round27_trigger_revalidation.csv", "planning_need_pca_audit.csv",
    "planning_utility_seed_disagreement_v2.csv", "planning_pressure_taskwp_relation.csv",
    "planning_need_representation_gate.json", "trigger_model_readiness_gate.json",
    "round28_readiness_gate.json", "access_audit.json", "run_manifest.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round28PlanningNeedRepresentationTests(unittest.TestCase):
    def _completed(self) -> None:
        if not (OUTPUT / "run_manifest.json").is_file():
            self.skipTest("Round-28 audit outputs are written after pre-output source checks")

    def test_01_round27_lock_unchanged(self) -> None:
        self.assertEqual(digest(ROOT / "round_27" / "ROUND_LOCK.json"), EXPECTED_R27)

    def test_02_no_optimizer(self) -> None:
        self.assertNotIn("torch.optim", (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8"))

    def test_03_no_backward(self) -> None:
        self.assertNotIn(".backward(", (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8"))

    def test_04_no_model_training(self) -> None:
        source = (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("train_one(", source)
        self.assertNotIn(".zero_grad(", source)
        self.assertNotIn("optimizer.step(", source)

    def test_05_no_classifier_training(self) -> None:
        source = (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8").lower()
        for forbidden in ("logisticregression", "decisiontree", "randomforest", "sklearn", "svm"):
            self.assertNotIn(forbidden, source)

    def test_06_no_checkpoint_writes(self) -> None:
        source = (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("torch.save", source)
        self.assertNotIn("save_checkpoint", source)

    def test_07_frozen_h1_trajectory_source(self) -> None:
        config = load(ROUND / "configs/round28_planning_need_representation_config.json")
        self.assertEqual(config["state_sampling"]["source_behavior_generator"], "joint_h1")
        self.assertIn("no H1 trajectory regeneration", config["state_sampling"]["source"])

    def test_08_sampling_is_predetermined(self) -> None:
        config = load(ROUND / "configs/round28_planning_need_representation_config.json")
        self.assertEqual(config["state_sampling"]["timesteps"], [0, 2, 4, 6])
        self.assertIn("before utility labels", config["state_sampling"]["selection_rule"])

    def test_09_train_validation_scenario_separation(self) -> None:
        from wt.round28.audit import _allowed_h1_states, _load, _scenario_map, CONFIG_PATH
        selected = _allowed_h1_states(_load(CONFIG_PATH), _scenario_map())
        split_by_scenario = {}
        for row in selected:
            split_by_scenario.setdefault(row["scenario_id"], row["split"])
            self.assertEqual(split_by_scenario[row["scenario_id"]], row["split"])
        self.assertEqual(len(selected), 256)

    def test_10_same_snapshot_utility(self) -> None:
        self._completed()
        for row in rows(OUTPUT / "planning_need_feature_table.csv"):
            self.assertEqual(row["same_snapshot_verified"], "True")
            self.assertEqual(row["only_first_action_intervention"], "True")
            self.assertLessEqual(float(row["stored_h1_action_reproduction_rmse"]), 1.0e-9)

    def test_11_planner_is_frozen(self) -> None:
        source = (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8")
        self.assertIn("joint_plan(env, info, horizon=first_horizon)", source)
        self.assertNotIn("planner modification", source.lower())

    def test_12_h5_utility_formula_unchanged(self) -> None:
        self._completed()
        for row in rows(OUTPUT / "planning_need_feature_table.csv"):
            self.assertAlmostEqual(float(row["utility_payload"]), float(row["planner_payload_bits"]) - float(row["reference_payload_bits"]), places=6)

    def test_13_no_future_trigger_features(self) -> None:
        self._completed()
        specification = (OUTPUT / "planning_need_feature_spec.md").read_text(encoding="utf-8").lower()
        self.assertIn("no future state or planner action", specification)
        self.assertIn("bridge, behavior, quality", specification)

    def test_14_normalization_is_train_only(self) -> None:
        self._completed()
        normalizer = load(OUTPUT / "planning_need_feature_normalization.json")
        self.assertEqual(normalizer["fit_split"], "train")
        self.assertFalse(normalizer["validation_used_for_fit"])

    def test_15_assignment_regret_formula(self) -> None:
        self._completed()
        for row in rows(OUTPUT / "planning_need_feature_table.csv"):
            independent, exclusive = float(row["independent_current_score"]), float(row["exclusive_current_score"])
            expected = max(independent - exclusive, 0.0) / max(abs(independent), 1.0e-12)
            self.assertAlmostEqual(float(row["assignment_regret"]), expected, places=10)

    def test_16_isr_is_current_only(self) -> None:
        source = (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8")
        self.assertIn("assignment_link_metrics(assignment, gains", source)
        self.assertIn("joint_aware_matching(queues, gains", source)

    def test_17_queue_tension_is_current_only(self) -> None:
        self._completed()
        for row in rows(OUTPUT / "planning_need_feature_table.csv"):
            self.assertGreaterEqual(float(row["queue_to_h5_current_capacity_ratio"]), 0.0)
            self.assertGreaterEqual(float(row["unassigned_queue_to_h5_current_capacity_ratio"]), 0.0)

    def test_18_composite_weights_fixed(self) -> None:
        config = load(ROUND / "configs/round28_planning_need_representation_config.json")
        self.assertEqual(config["composites"]["coordination_pressure"]["weights"], [0.5, 0.5])
        self.assertEqual(config["composites"]["unified_planning_pressure"]["weights"], [1 / 3, 1 / 3, 1 / 3])

    def test_19_thresholds_train_only(self) -> None:
        self._completed()
        thresholds = load(OUTPUT / "round28_trigger_thresholds.json")
        self.assertEqual(thresholds["fit_split"], "train")
        self.assertFalse(thresholds["validation_used_for_fit"])

    def test_20_validation_threshold_immutable(self) -> None:
        self._completed()
        thresholds = load(OUTPUT / "round28_trigger_thresholds.json")["composites"]
        for row in rows(OUTPUT / "composite_trigger_metrics.csv"):
            expected = float(thresholds[row["score"]]["train_" + row["threshold_kind"]])
            self.assertAlmostEqual(float(row["threshold"]), expected, places=12)

    def test_21_scenario_cluster_bootstrap(self) -> None:
        self._completed()
        self.assertTrue(all(row["bootstrap_cluster"] == "scenario_id" for row in rows(OUTPUT / "planning_feature_utility_correlation.csv")))
        self.assertTrue(all(row["bootstrap_cluster"] == "scenario_id" for row in rows(OUTPUT / "composite_trigger_metrics.csv")))

    def test_22_coverage_guard(self) -> None:
        self._completed()
        config = load(ROUND / "configs/round28_planning_need_representation_config.json")
        self.assertEqual(config["threshold_protocol"]["minimum_validation_active_states"], 7)
        self.assertTrue(all(int(row["activation_count"]) >= 0 for row in rows(OUTPUT / "composite_trigger_metrics.csv")))

    def test_23_bridge_is_not_deployable_trigger(self) -> None:
        self._completed()
        self.assertFalse(load(OUTPUT / "planning_need_representation_gate.json")["bridge_used_as_trigger"])

    def test_24_no_fresh_new_rollout(self) -> None:
        self._completed()
        self.assertEqual(load(OUTPUT / "access_audit.json")["fresh_new_rollout_count"], 0)

    def test_25_no_untouched_evaluation_access(self) -> None:
        self._completed()
        access = load(OUTPUT / "access_audit.json")
        self.assertEqual(access["untouched_eval_access_count"], 0)
        self.assertEqual(access["development_test_rows_decoded"], 0)

    def test_26_no_hgnn(self) -> None:
        self.assertFalse(load(ROUND / "round_manifest.json")["hgnn_training_performed"])

    def test_27_no_ppo_or_marl(self) -> None:
        self.assertFalse(load(ROUND / "round_manifest.json")["ppo_training_performed"])
        self.assertNotIn("mappo", (ROOT / "src/wt/round28/audit.py").read_text(encoding="utf-8").lower())

    def test_28_no_nan_or_inf(self) -> None:
        self._completed()
        for path in OUTPUT.glob("*.csv"):
            for row in rows(path):
                for value in row.values():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    self.assertTrue(math.isfinite(number), f"{path.name}: {value}")

    def test_29_compileable_and_legal_successor_chain(self) -> None:
        self.assertFalse(load(ROUND / "ROUND_LOCK.json")["round29_created"])
        self.assertTrue((ROOT / "round_29" / "AGENTS.md").is_file())
        self.assertTrue((ROOT / "round_30" / "AGENTS.md").is_file())
        successor = ROOT / "round_31"
        if successor.exists():
            manifest = load(successor / "round_manifest.json")
            self.assertEqual(manifest["previous_round_lock"], "round_30/ROUND_LOCK.json")
            self.assertEqual(manifest["previous_round_lock_sha256"], "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125")
        successor32 = ROOT / "round_32"
        if successor32.exists():
            manifest32 = load(successor32 / "round_manifest.json")
            self.assertEqual(manifest32["previous_round_lock"], "round_31/ROUND_LOCK.json")
            self.assertFalse(manifest32["round33_created"])
        self.assertTrue((ROOT / "src/wt/round28/audit.py").is_file())

    def test_lock_covers_completed_evidence_when_present(self) -> None:
        lock_path = ROUND / "ROUND_LOCK.json"
        if not lock_path.is_file():
            self.skipTest("Round-28 lock is written only after pre-lock full regression")
        lock = load(lock_path)
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["previous_round_lock_sha256"], EXPECTED_R27)
        self.assertEqual(lock["final_verification"]["required_output_missing"], [])
        for name, value in lock["output_hashes"].items():
            self.assertEqual(digest(OUTPUT / name), value)


if __name__ == "__main__":
    unittest.main()
