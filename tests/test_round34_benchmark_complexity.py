"""Round-34 non-learning benchmark-complexity guards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.round34.audit import (
    ProtectedSplitSecondViolation,
    _metadata_from_raw,
    association_count,
    classify_edge_derivability,
    decode_allowed_record,
    extract_decision_ownership,
)


class Round34BenchmarkComplexityTests(unittest.TestCase):
    def test_01_round33_original_lock_is_unchanged(self) -> None:
        self.assertEqual(hashlib.sha256((ROOT / "round_33" / "ROUND_LOCK.json").read_bytes()).hexdigest().upper(), "40A4C92A3E085661D01B2D112B4525E2DDE2CC68D63D8718ED2F5BEF83DE8075")

    def test_02_no_training_library_is_imported(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("tensorflow", source)

    def test_03_no_fitting_construct_is_present(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)

    def test_04_environment_is_not_imported(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("SharedSpectrumMultiUAVEnv", source)
        self.assertNotIn("run_episode(", source)

    def test_05_planner_is_not_imported(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("joint_plan(", source)
        self.assertNotIn("from wt.round19", source)

    def test_06_decision_ownership_is_exact(self) -> None:
        rows = extract_decision_ownership()
        self.assertEqual([row["decision"] for row in rows], ["UAV movement", "association", "service priority", "power", "bandwidth", "sensing", "interference coordination", "safety correction"])
        self.assertEqual(sum(bool(row["neural_policy_direct"]) for row in rows), 1)

    def test_07_matching_ownership_is_exact(self) -> None:
        rows = {row["decision"]: row for row in extract_decision_ownership()}
        self.assertTrue(rows["association"]["deterministic_matching"])
        self.assertTrue(rows["service priority"]["deterministic_matching"])
        self.assertTrue(rows["interference coordination"]["deterministic_matching"])

    def test_08_legal_association_counts_are_correct(self) -> None:
        self.assertEqual(association_count(2, 3), 13)
        self.assertEqual(association_count(2, 5), 31)
        self.assertEqual(association_count(4, 8), 3393)
        self.assertEqual(association_count(4, 16), 58625)

    def test_09_edge_derivability_is_deterministic(self) -> None:
        ub, uu = classify_edge_derivability()
        self.assertEqual({row["feature"] for row in ub if row["class"] == "C"}, {"current_assignment_relation"})
        self.assertEqual({row["feature"] for row in uu if row["class"] == "C"}, {"both_assigned_distinct", "receiver_cross_interference_power_w", "interference_to_signal_ratio"})

    def test_10_current_state_graph_is_the_only_graph_input(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertIn('record["current_graph_state"]', source)
        self.assertNotIn("graph_state_t_plus_1", source)

    def test_11_protected_data_markers_are_rejected_before_decode(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertIn("metadata is checked before JSON parsing", source)
        decoder = source.index("def decode_allowed_record")
        self.assertLess(source.index("if not metadata.get(\"allowed\")", decoder), source.index("return json.loads(raw.decode", decoder))

    def test_12_no_task_waypoint_fitting(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("TaskWaypointPredictor", source)

    def test_13_no_hgnn_fitting(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("HeterogeneousTaskWaypointPredictor", source)

    def test_14_no_ppo(self) -> None:
        self.assertNotIn("PPO", (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8"))

    def test_15_no_new_rollout(self) -> None:
        self.assertNotIn("env.step(", (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8"))

    def test_16_historical_gate_is_locked(self) -> None:
        lock = json.loads((ROOT / "round_33" / "ROUND_LOCK.json").read_text(encoding="utf-8"))
        relative = "horizon_separation_gate.json"
        expected = lock["output_hashes"][relative]
        actual = hashlib.sha256((ROOT / lock["formal_output"] / relative).read_bytes()).hexdigest().upper()
        self.assertEqual(actual, expected)

    def test_17_config_prohibits_training(self) -> None:
        config = json.loads((ROOT / "round_34/configs/round34_benchmark_complexity_audit_config.json").read_text(encoding="utf-8"))
        self.assertTrue(config["audit_only"])
        self.assertFalse(config["training_performed"])
        self.assertEqual(config["fresh_rollout_count"], 0)

    def test_18_compile_target_is_standard_library_only(self) -> None:
        self.assertTrue((ROOT / "src/wt/round34/audit.py").is_file())

    def test_19_development_test_metadata_is_detected_without_deserialization(self) -> None:
        raw = b'{"sample_id":"synthetic-dev","split":"development_test","t":0}'
        metadata = _metadata_from_raw(raw, 1, "samples")
        self.assertTrue(metadata["protected"])
        self.assertFalse(metadata["allowed"])

    def test_20_development_test_cannot_enter_decoder(self) -> None:
        raw = b'{"sample_id":"synthetic-dev","split":"development_test","t":0}'
        metadata = _metadata_from_raw(raw, 1, "samples")
        with self.assertRaises(ProtectedSplitSecondViolation):
            decode_allowed_record(raw, metadata, metadata)

    def test_21_untouched_cannot_enter_decoder(self) -> None:
        raw = b'{"sample_id":"synthetic-untouched","split":"untouched","t":0}'
        metadata = _metadata_from_raw(raw, 1, "samples")
        with self.assertRaises(ProtectedSplitSecondViolation):
            decode_allowed_record(raw, metadata, metadata)

    def test_22_unknown_split_fails_closed(self) -> None:
        with self.assertRaises(ProtectedSplitSecondViolation):
            _metadata_from_raw(b'{"sample_id":"synthetic-unknown","split":"other","t":0}', 1, "samples")

    def test_23_train_record_can_decode_after_allowlisting(self) -> None:
        raw = b'{"sample_id":"synthetic-train","split":"train","t":0}'
        metadata = _metadata_from_raw(raw, 1, "samples")
        self.assertEqual(decode_allowed_record(raw, metadata, metadata)["sample_id"], "synthetic-train")

    def test_24_validation_record_can_decode_after_allowlisting(self) -> None:
        raw = b'{"sample_id":"synthetic-validation","split":"validation","t":0}'
        metadata = _metadata_from_raw(raw, 1, "samples")
        self.assertEqual(decode_allowed_record(raw, metadata, metadata)["split"], "validation")

    def test_25_development_test_alias_is_protected(self) -> None:
        raw = b'{"sample_id":"synthetic-alias","split":"development-test","t":0}'
        self.assertTrue(_metadata_from_raw(raw, 1, "samples")["protected"])

    def test_26_contamination_boundary_is_outside_formal_output(self) -> None:
        boundary = json.loads((ROOT / "round_34/quarantine/CONTAMINATION_BOUNDARY.json").read_text(encoding="utf-8"))
        self.assertTrue(boundary["excluded_from_scientific_analysis"])
        self.assertFalse(boundary["directories_contained_files"])

    def test_27_exception_count_is_exactly_one(self) -> None:
        exception = json.loads((ROOT / "round_34/ROUND34_PROTECTED_ACCESS_EXCEPTION.json").read_text(encoding="utf-8"))
        config = json.loads((ROOT / "round_34/configs/round34_benchmark_complexity_audit_config.json").read_text(encoding="utf-8"))
        self.assertEqual(exception["records_decoded"], 1)
        self.assertEqual(config["protected_access_exception_count"], 1)

    def test_28_governance_access_state_is_truthful(self) -> None:
        config = json.loads((ROOT / "round_34/configs/round34_benchmark_complexity_audit_config.json").read_text(encoding="utf-8"))
        self.assertTrue(config["protected_data_accessed"])
        self.assertEqual(config["development_test_access_count"], 1)
        self.assertEqual(config["untouched_eval_access_count"], 0)

    def test_29_second_protected_attempt_hard_stops_before_decode(self) -> None:
        raw = b'{"sample_id":"synthetic-second","split":"development_test","t":0}'
        metadata = _metadata_from_raw(raw, 1, "samples")
        with self.assertRaisesRegex(ProtectedSplitSecondViolation, "before decode"):
            decode_allowed_record(raw, metadata, metadata)

    def test_30_preflight_uses_allowlist_not_blacklist(self) -> None:
        source = (ROOT / "src/wt/round34/audit.py").read_text(encoding="utf-8")
        self.assertIn('ALLOWED_SPLITS = frozenset(("train", "validation"))', source)
        self.assertIn('"protected_split_fail_closed": True', source)

    def test_31_clean_sample_preflight_has_no_formal_protected_input(self) -> None:
        preflight = json.loads((ROOT / "round_34/round34_clean_sample_preflight.json").read_text(encoding="utf-8"))
        self.assertTrue(preflight["metadata_only"])
        self.assertFalse(preflight["json_payload_decoded"])
        self.assertEqual(preflight["development_test_in_formal_input"], 0)
        self.assertEqual(preflight["untouched_in_formal_input"], 0)
        self.assertEqual(preflight["unknown_split_in_formal_input"], 0)

    def test_32_clean_manifests_are_hash_locked(self) -> None:
        for stem in ("sample", "transition"):
            manifest = ROOT / f"round_34/round34_clean_{stem}_manifest.json"
            lock = json.loads((ROOT / f"round_34/round34_clean_{stem}_manifest.lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest().upper())
            parsed = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(parsed["formal_protected_records"], 0)
            self.assertEqual(parsed["formal_unknown_records"], 0)

    def test_33_formal_outputs_exclude_quarantine(self) -> None:
        manifest = json.loads((ROOT / "round_34/outputs/round34_benchmark_complexity_audit_20260901_1540/run_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("quarantine", json.dumps(manifest))
        self.assertTrue(manifest["round34_scientific_analysis_clean_after_exception"])

    def test_34_formal_access_audit_has_zero_protected_scientific_records(self) -> None:
        access = json.loads((ROOT / "round_34/outputs/round34_benchmark_complexity_audit_20260901_1540/access_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(access["formal_scientific_protected_records_used"], 0)
        self.assertEqual(access["untouched_eval_access_count"], 0)
        self.assertFalse(access["second_protected_access"])

    def test_35_all_required_formal_artifacts_are_present(self) -> None:
        output = ROOT / "round_34/outputs/round34_benchmark_complexity_audit_20260901_1540"
        required = {"current_benchmark_structure.md", "decision_control_decomposition.csv", "matching_relational_reasoning_audit.md", "graph_decision_ownership.csv", "association_combinatorial_complexity.csv", "scale_complexity_projection.csv", "edge_feature_redundancy.csv", "uu_edge_information_audit.csv", "edge_incremental_information_audit.csv", "deepsets_pooling_collision.csv", "decision_relevant_pooling_collision.csv", "task_waypoint_control_interface_audit.md", "task_to_action_causal_chain.md", "association_action_interface_design.md", "task_waypoint_control_directness.csv", "effective_decision_horizon.csv", "why_h5_saturates.md", "benchmark_method_suitability_comparison.md", "hgnn_suitability_criteria.md", "task_waypoint_suitability_criteria.md", "candidate_benchmark_redesign.md", "benchmark_complexity_gate.json", "hgnn_future_reentry_gate.json", "task_waypoint_future_gate.json", "round34_method_strategy_gate.json", "round34_readiness_gate.json", "access_audit.json", "run_manifest.json"}
        self.assertEqual({path.name for path in output.iterdir() if path.is_file()}, required)

    def test_36_twelve_png_and_svg_figures_are_present(self) -> None:
        figures = ROOT / "round_34/figures/round34_benchmark_complexity_audit_20260901_1540"
        self.assertEqual(len(list(figures.glob("*.png"))), 12)
        self.assertEqual(len(list(figures.glob("*.svg"))), 12)


if __name__ == "__main__":
    unittest.main()
