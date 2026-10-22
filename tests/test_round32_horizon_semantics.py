"""Guards for the authorized read-only Round-32 horizon semantics audit."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
R32 = ROOT / "round_32"
OUTPUT = R32 / "outputs" / "round32_horizon_semantics_audit_20260830_2231"
FIGURES = R32 / "figures" / "round32_horizon_semantics_audit_20260830_2231"
R31_LOCK_SHA = "9B26D6129264A05A6BD7198FE283D839A70085C7A8D798D4470AD9302CB6E2D3"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round32HorizonSemanticsTests(unittest.TestCase):
    def test_01_predecessor_lock_is_byte_stable(self) -> None:
        self.assertEqual(digest(ROOT / "round_31/ROUND_LOCK.json"), R31_LOCK_SHA)

    def test_02_only_audit_source_has_no_learning_operations(self) -> None:
        source = (ROOT / "src/wt/round32/audit.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("backward", calls); self.assertNotIn("step", calls)
        for forbidden in ("import torch", "torch.optim", ".backward(", "torch.save(", "optimizer.step(", "env.step(", "joint_plan("):
            self.assertNotIn(forbidden, source)

    def test_03_no_environment_planner_or_dataset_modification(self) -> None:
        manifest = json.loads((R32 / "round_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["environment_modified"])
        self.assertFalse(manifest["planner_modified"])
        self.assertFalse(manifest["dataset_modified"])

    def test_04_source_horizons_are_recovered_exactly(self) -> None:
        matrix = {row["round"]: row for row in rows("cross_round_horizon_semantics.csv")}
        self.assertEqual([int(matrix[name]["environment_episode_max_steps"]) for name in ("round_15", "round_16", "round_17")], [64, 64, 64])
        self.assertEqual([int(matrix[name]["environment_episode_max_steps"]) for name in ("round_18", "round_19", "round_20", "round_22", "round_30", "round_31")], [12] * 6)
        self.assertEqual(matrix["round_22"]["policy_context_Hc"], "10")

    def test_05_effective_horizon_formula_and_terminal_audit(self) -> None:
        data = rows("planner_effective_horizon_audit.csv")
        self.assertGreater(len(data), 0)
        for row in data[::max(1, len(data) // 1000)]:
            nominal, remaining = int(row["nominal_planning_horizon"]), int(row["remaining_episode_steps"])
            self.assertEqual(int(row["episode_constrained_effective_future_steps"]), min(nominal, remaining))
            self.assertEqual(row["truncated_by_real_episode"], str(int(row["source_shadow_simulated_steps"]) > remaining))

    def test_06_utilization_statistics_are_exact(self) -> None:
        table = {(row["round"], row["nominal_H"]): row for row in rows("planner_horizon_utilization.csv")}
        self.assertEqual(table[("round_18", "10")]["mean_HUR"], "0.625")
        self.assertEqual(table[("round_19", "5")]["fraction_HUR_eq_1"], "0.66666667")
        self.assertEqual(table[("round_15", "20")]["fraction_HUR_eq_1"], "0.703125")

    def test_07_h5_h10_equality_is_exact_and_not_bug_claim(self) -> None:
        data = rows("h5_h10_equivalence_audit.csv")
        self.assertEqual(len(data), 792)
        self.assertTrue(all(row["h5_h10_action_equal"] == "True" for row in data))
        self.assertEqual({row["planning_score_equality"] for row in data}, {"not_assessable_from_locked_trajectory_trace"})
        self.assertTrue(all("EQ-C" not in row["equality_reason"] for row in data))

    def test_08_h5_h10_prefixes_are_identical(self) -> None:
        prefixes = rows("r18_r19_trajectory_prefix_equality.csv")
        self.assertEqual(len(prefixes), 66)
        self.assertTrue(all(row["all_shared_actions_equal"] == "True" for row in prefixes))

    def test_09_task_waypoint_labels_are_complete_windows(self) -> None:
        semantic = json.loads((OUTPUT / "task_waypoint_label_semantics.json").read_text(encoding="utf-8"))
        self.assertEqual(semantic["full_horizon_label_count"], 7040)
        self.assertEqual(semantic["truncated_or_terminal_label_count"], 0)
        data = rows("task_waypoint_effective_horizon.csv")
        self.assertTrue(all(row["full_horizon_label"] == "True" for row in data))

    def test_10_task_waypoint_coverage_and_protected_split(self) -> None:
        coverage = rows("task_waypoint_horizon_coverage.csv")
        train_h5 = [row for row in coverage if row["split"] == "train" and row["H_task"] == "5"]
        validation_h10 = [row for row in coverage if row["split"] == "validation" and row["H_task"] == "10"]
        self.assertEqual(sum(int(row["sample_count"]) for row in train_h5), 3840)
        self.assertEqual(sum(int(row["sample_count"]) for row in validation_h10), 480)
        self.assertTrue(all(row["sample_count"] == "protected_not_accessed" for row in coverage if row["split"] == "development_test"))

    def test_11_terminated_and_truncated_semantics_are_stated(self) -> None:
        matrix = rows("cross_round_horizon_semantics.csv")
        self.assertTrue(all("energy depleted" in row["terminated_semantics"] for row in matrix))
        self.assertTrue(all(row["remaining_horizon_state_field"].startswith("remaining_horizon_fraction") for row in matrix))

    def test_12_ppo_gae_endpoint_and_rollout_are_correct(self) -> None:
        summary = json.loads((OUTPUT / "ppo_credit_horizon_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["episode_horizon"], 12)
        self.assertEqual(summary["transitions_per_ppo_batch"], 2048)
        self.assertFalse(summary["terminal_or_truncated_bootstrap"])
        self.assertEqual(summary["mean_maximum_future_lag_steps"], 5.5)

    def test_13_historical_gate_is_reassessed_without_mutation(self) -> None:
        value = json.loads((OUTPUT / "historical_gate_horizon_reassessment.json").read_text(encoding="utf-8"))
        self.assertEqual(value["round_19_original_gate"], "IPN-A")
        self.assertFalse(value["original_gate_mutated"])
        self.assertEqual(value["interpretation_validity"], "effective-horizon-limited")

    def test_14_timescale_and_redesign_gate_are_data_derived(self) -> None:
        scales = json.loads((OUTPUT / "physical_queue_coordination_timescales.json").read_text(encoding="utf-8"))
        gate = json.loads((OUTPUT / "episode_horizon_sufficiency_gate.json").read_text(encoding="utf-8"))
        self.assertEqual((scales["recommended_minimum_episode_horizon"], scales["recommended_primary_episode_horizon"]), (11, 20))
        self.assertEqual(gate["gate"], "EH-C"); self.assertTrue(gate["episode_horizon_redesign_required"])

    def test_15_no_fresh_development_or_untouched_access(self) -> None:
        access = json.loads((OUTPUT / "access_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(access["fresh_new_rollout_count"], 0)
        self.assertEqual(access["round20_development_test_decoded_count"], 0)
        self.assertFalse(access["untouched_eval_accessed"])

    def test_16_required_outputs_figures_and_compile(self) -> None:
        required = {"horizon_semantics_spec.md", "cross_round_horizon_semantics.csv", "planner_effective_horizon_audit.csv", "planner_horizon_utilization.csv", "h5_h10_equivalence_audit.csv", "dynamic_round_effective_horizon.csv", "task_waypoint_effective_horizon.csv", "task_waypoint_horizon_coverage.csv", "remaining_horizon_semantics.md", "round30_episode_horizon_provenance.md", "ppo_rollout_vs_episode_horizon_audit.md", "ppo_effective_credit_horizon.csv", "historical_planning_horizon_validity.csv", "historical_task_waypoint_horizon_validity.csv", "historical_gate_horizon_reassessment.csv", "long_horizon_definition_proposal.md", "physical_episode_timescale_audit.csv", "queue_service_timescale_audit.csv", "coordination_timescale_audit.csv", "episode_horizon_sufficiency_gate.json", "round32_horizon_semantics_gate.json", "round32_readiness_gate.json", "access_audit.json", "run_manifest.json"}
        self.assertTrue(all((OUTPUT / name).is_file() for name in required))
        self.assertEqual(len(list(FIGURES.glob("*.png"))), 12); self.assertEqual(len(list(FIGURES.glob("*.svg"))), 12)
        for path in (ROOT / "src/wt/round32").glob("*.py"):
            compile(path.read_text(encoding="utf-8"), path.as_posix(), "exec")

    def test_17_round33_requires_a_narrow_successor_amendment(self) -> None:
        self.assertFalse(json.loads((R32 / "round_manifest.json").read_text(encoding="utf-8"))["round33_created"])
        successor = ROOT / "round_33"
        if successor.exists():
            amendment = json.loads((R32 / "ROUND_LOCK_AMENDMENT.json").read_text(encoding="utf-8"))
            self.assertEqual(amendment["amendment_type"], "legal_round33_successor_governance_extension")
            self.assertTrue(amendment["h20_support_explicitly_authorized"])
            self.assertEqual(json.loads((successor / "round_manifest.json").read_text(encoding="utf-8"))["previous_round_lock_sha256"], digest(R32 / "ROUND_LOCK.json"))


if __name__ == "__main__":
    unittest.main()
