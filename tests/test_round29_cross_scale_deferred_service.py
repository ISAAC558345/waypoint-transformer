"""Regression guards for diagnostic-only Round-29 cross-scale DSP evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_29"
OUTPUT = ROUND / "outputs" / "round29_cross_scale_dsp_20260830_1745"
FIGURES = ROUND / "figures" / "round29_cross_scale_dsp_20260830_1745"
EXPECTED_R28 = "9E3F10B893891B6E0E0415D886E00AEC234CD1376B3146E5704093A210384C91"
REQUIRED = {
    "existing_dsp_scale_audit.md", "dsp_K_scale_audit.csv", "dsp_activation_decomposition.csv",
    "round29_dsp_variant_spec.json", "round29_trigger_thresholds.json", "round29_variant_selection_protocol.json",
    "round29_train_variant_selection.csv", "round28_replay_dsp_variants.csv", "selected_scale_normalized_dsp.json",
    "round29_preregistration_lock.json", "round29_confirmation_manifest.json", "round29_confirmation_manifest_lock.json",
    "round29_confirmation_raw_metrics.csv", "round29_confirmation_trigger_metrics.csv", "round29_final_integrity_audit.json", "cross_K_trigger_interaction.csv",
    "cross_K_mechanism_audit.csv", "planner_useful_prevalence_K.csv", "eta_confirmation_audit.csv",
    "cross_scale_planning_pressure_gate.json", "round29_trigger_training_readiness.json", "round29_readiness_gate.json",
    "access_audit.json", "run_manifest.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round29CrossScaleDeferredServiceTests(unittest.TestCase):
    def _completed(self) -> None:
        if not (OUTPUT / "run_manifest.json").is_file():
            self.skipTest("Round-29 evidence is written after source-only guards pass")

    def test_01_round28_lock_unchanged(self) -> None:
        self.assertEqual(digest(ROOT / "round_28" / "ROUND_LOCK.json"), EXPECTED_R28)

    def test_02_no_optimizer(self) -> None:
        self.assertNotIn("torch.optim", (ROOT / "src/wt/round29/audit.py").read_text(encoding="utf-8"))

    def test_03_no_backward(self) -> None:
        self.assertNotIn(".backward(", (ROOT / "src/wt/round29/audit.py").read_text(encoding="utf-8"))

    def test_04_no_model_or_policy_training(self) -> None:
        source = (ROOT / "src/wt/round29/audit.py").read_text(encoding="utf-8")
        for forbidden in ("train_one(", ".zero_grad(", "optimizer.step(", "torch.save"):
            self.assertNotIn(forbidden, source)

    def test_05_no_classifier_training(self) -> None:
        source = (ROOT / "src/wt/round29/audit.py").read_text(encoding="utf-8").lower()
        for forbidden in ("logisticregression", "decisiontree", "randomforest", "sklearn", "svm"):
            self.assertNotIn(forbidden, source)

    def test_06_frozen_round28_labels_unchanged(self) -> None:
        lock = load(ROOT / "round_28" / "ROUND_LOCK.json")
        self.assertEqual(digest(ROOT / lock["formal_output"] / "planning_need_feature_table.csv"), lock["output_hashes"]["planning_need_feature_table.csv"])

    def test_07_original_dsp_reproduced(self) -> None:
        self._completed()
        for row in rows(OUTPUT / "round28_replay_dsp_variants.csv"):
            self.assertAlmostEqual(float(row["original_dsp"]), float(row["original_dsp_locked"]), places=12)

    def test_08_v1_formula(self) -> None:
        self._completed()
        for row in rows(OUTPUT / "round28_replay_dsp_variants.csv"):
            expected = .5 * float(row["backlog_v1_queue_share"]) + .5 * float(row["norm_distance"])
            self.assertAlmostEqual(float(row["DSP-V1"]), expected, places=12)

    def test_09_v2_formula(self) -> None:
        self._completed()
        for row in rows(OUTPUT / "round28_replay_dsp_variants.csv"):
            expected = float(row["unserved_high_queue_mass"]) / float(row["num_buoys"])
            self.assertAlmostEqual(float(row["backlog_v2_per_buoy_fraction"]), expected, places=12)
            self.assertAlmostEqual(float(row["DSP-V2"]), .5 * expected + .5 * float(row["norm_distance"]), places=12)

    def test_10_v3_formula(self) -> None:
        self._completed()
        spec = load(OUTPUT / "round29_dsp_variant_spec.json")
        scaler = spec["v3_train_scaler"]
        for row in rows(OUTPUT / "round28_replay_dsp_variants.csv"):
            raw = float(row["backlog_v3_capacity_tension_raw"])
            normalized = min(max((raw - float(scaler["train_min"])) / max(float(scaler["range"]), 1e-12), 0.), 1.)
            self.assertAlmostEqual(float(row["backlog_v3_capacity_tension_norm"]), normalized, places=12)
            self.assertAlmostEqual(float(row["DSP-V3"]), .5 * normalized + .5 * float(row["norm_distance"]), places=12)

    def test_11_physical_or_train_only_normalization(self) -> None:
        self._completed()
        spec = load(OUTPUT / "round29_dsp_variant_spec.json")
        self.assertTrue(spec["weights_frozen"])
        self.assertTrue(spec["current_observable_only"])
        self.assertIn("round28_train", spec["variants"]["DSP-V3"]["backlog_normalization"])

    def test_12_uniform_threshold_across_k(self) -> None:
        self._completed()
        thresholds = load(OUTPUT / "round29_trigger_thresholds.json")
        metrics = rows(OUTPUT / "round29_confirmation_trigger_metrics.csv")
        selected = load(OUTPUT / "selected_scale_normalized_dsp.json")["confirmation_primary_variant"]
        values = {row["threshold"] for row in metrics if row["score"] == selected and row["group_type"] == "num_buoys"}
        self.assertEqual(values, {str(thresholds[selected])})

    def test_13_selection_never_uses_validation_or_confirmation_utility(self) -> None:
        self._completed()
        selected = load(OUTPUT / "selected_scale_normalized_dsp.json")
        self.assertEqual(selected["selection_source"], "round28_train_only")
        self.assertFalse(selected["validation_utility_used"])
        self.assertFalse(selected["confirmation_utility_used"])

    def test_14_confirmation_seeds_are_disjoint(self) -> None:
        self._completed()
        manifest = load(OUTPUT / "round29_confirmation_manifest.json")
        seeds = {item["seed"] for item in manifest["scenarios"]}
        self.assertEqual(len(seeds), 40)
        self.assertFalse(seeds & set(range(20000, 20080)))
        self.assertFalse(seeds & set(range(29000, 29016)))

    def test_15_confirmation_manifest_is_frozen_before_utility(self) -> None:
        self._completed()
        lock = load(OUTPUT / "round29_confirmation_manifest_lock.json")
        self.assertEqual(lock["manifest_sha256"], digest(OUTPUT / "round29_confirmation_manifest.json"))
        self.assertFalse(lock["confirmation_outcomes_accessed"])

    def test_16_sampling_is_outcome_independent(self) -> None:
        config = load(ROUND / "configs/round29_cross_scale_dsp_config.json")
        self.assertEqual(config["confirmation"]["state_timesteps"], [0, 2, 4, 6])
        self.assertIn("before DSP or utility", config["confirmation"]["selection_rule"])

    def test_17_planner_and_environment_are_frozen(self) -> None:
        source = (ROOT / "src/wt/round29/audit.py").read_text(encoding="utf-8")
        self.assertIn("joint_plan(env, info, horizon=first_horizon)", source)
        self.assertIn("matching_mode=\"joint_aware\"", source)

    def test_18_same_snapshot_utility(self) -> None:
        self._completed()
        raw = rows(OUTPUT / "round29_confirmation_raw_metrics.csv")
        self.assertEqual(len(raw), 160)
        self.assertTrue(all(row["same_snapshot_verified"] == "True" and row["only_first_action_intervention"] == "True" for row in raw))

    def test_19_scenario_cluster_bootstrap(self) -> None:
        self._completed()
        self.assertTrue(all(row["bootstrap_cluster"] == "scenario_id" for row in rows(OUTPUT / "round29_confirmation_trigger_metrics.csv")))

    def test_20_k_specific_thresholds_excluded_from_gate(self) -> None:
        self._completed()
        thresholds = load(OUTPUT / "round29_trigger_thresholds.json")
        self.assertTrue(thresholds["k_specific_threshold_excluded_from_gate"])
        self.assertTrue(all(row["threshold_scope"] == "single_train_p75" for row in rows(OUTPUT / "round29_confirmation_trigger_metrics.csv")))

    def test_21_protected_sets_untouched(self) -> None:
        self._completed()
        access = load(OUTPUT / "access_audit.json")
        for field in ("round20_development_test_reuse_count", "round22_fresh_access_count", "round24_fresh_access_count", "untouched_eval_access_count"):
            self.assertEqual(access[field], 0)
        self.assertEqual(access["confirmation_outcome_access_count"], 1)

    def test_22_no_hgnn(self) -> None:
        self.assertFalse(load(ROUND / "round_manifest.json")["hgnn_training_performed"])

    def test_23_no_ppo_or_marl(self) -> None:
        source = (ROOT / "src/wt/round29/audit.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("mappo", source)
        self.assertFalse(load(ROUND / "round_manifest.json")["ppo_training_performed"])

    def test_24_no_nan_or_inf(self) -> None:
        self._completed()
        for path in OUTPUT.glob("*.csv"):
            for row in rows(path):
                for value in row.values():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    self.assertTrue(math.isfinite(number), f"{path.name}: {value}")

    def test_25_compileable_figures_and_legal_successor_chain(self) -> None:
        self.assertTrue((ROOT / "src/wt/round29/audit.py").is_file())
        self.assertFalse(load(ROUND / "ROUND_LOCK.json")["round30_created"])
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
        if (OUTPUT / "run_manifest.json").is_file():
            figures = [path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"}]
            self.assertEqual(len(figures), 30)

    def test_26_inventory_and_lock_when_present(self) -> None:
        self._completed()
        actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
        self.assertIn(actual, (REQUIRED - {"round29_final_integrity_audit.json"}, REQUIRED))
        lock_path = ROUND / "ROUND_LOCK.json"
        if lock_path.is_file():
            self.assertEqual(actual, REQUIRED)
            lock = load(lock_path)
            self.assertEqual(lock["status"], "locked")
            for name, value in lock["output_hashes"].items():
                self.assertEqual(digest(OUTPUT / name), value)


if __name__ == "__main__":
    unittest.main()
