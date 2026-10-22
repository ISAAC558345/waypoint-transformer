"""Acceptance guards for Round-33's non-learning long-episode benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_33"
OUTPUT = ROUND / "outputs" / "round33_long_episode_benchmark_20260830_2320"
FIGURES = ROUND / "figures" / "round33_long_episode_benchmark_20260830_2320"
R32 = "8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round33LongEpisodeTests(unittest.TestCase):
    def test_01_round32_lock_is_immutable(self) -> None:
        self.assertEqual(digest(ROOT / "round_32/ROUND_LOCK.json"), R32)

    def test_02_snapshot_preserves_preextension_planner(self) -> None:
        self.assertEqual(digest(ROOT / "backup/snapshots/before_round_33_20260830_2320/src/wt/round19/planner.py"), "C7B7A8F78E787BC5DECC77676985F5864E01AE9DE6C198E45D5A96CBCA419FA5")

    def test_03_round_agent_contract_prohibits_learning(self) -> None:
        text = (ROUND / "AGENTS.md").read_text(encoding="utf-8").lower()
        self.assertIn("no neural training", text); self.assertIn("do not create round-34", text)

    def test_04_configuration_is_exactly_t20(self) -> None:
        self.assertEqual(load("round33_long_episode_config.json")["episode_horizon"], 20)

    def test_05_h20_is_a_supported_frozen_search_depth(self) -> None:
        from wt.round19.planner import PLANNING_HORIZONS
        self.assertEqual(PLANNING_HORIZONS, (1, 5, 10, 20))

    def test_06_planner_integrity_limits_the_authorized_change(self) -> None:
        integrity = load("round33_planner_integrity.json")
        self.assertTrue(integrity["H1_H5_H10_exact_T12_regression"])
        self.assertIn("no candidates", integrity["authorized_change"])

    def test_07_t12_parameterization_regression_passes(self) -> None:
        self.assertTrue(load("round33_horizon_parameterization_regression.json")["exact_passed"])

    def test_08_each_t12_regression_dimension_is_exact(self) -> None:
        for row in load("round33_horizon_parameterization_regression.json")["rows"]:
            self.assertTrue(all(value for key, value in row.items() if key.endswith("_exact")))

    def test_09_design_manifest_is_preoutcome_and_has_64_scenarios(self) -> None:
        data = load("round33_design_manifest.json")
        self.assertEqual((data["scenario_count"], data["episode_horizon"]), (64, 20)); self.assertFalse(data["outcome_accessed"])

    def test_10_confirmation_manifest_is_preoutcome_and_has_64_scenarios(self) -> None:
        data = load("round33_confirmation_manifest.json")
        self.assertEqual((data["scenario_count"], data["episode_horizon"]), (64, 20)); self.assertFalse(data["outcome_accessed"])

    def test_11_design_and_confirmation_seeds_are_disjoint(self) -> None:
        design = {row["seed"] for row in load("round33_design_manifest.json")["scenarios"]}
        confirmation = {row["seed"] for row in load("round33_confirmation_manifest.json")["scenarios"]}
        self.assertFalse(design & confirmation)

    def test_12_every_new_stratum_has_four_seeds(self) -> None:
        data = load("round33_design_manifest.json")["scenarios"]
        counts = Counter((row["num_buoys"], row["eta"], row["queue_pressure"], row["geometry"]) for row in data)
        self.assertEqual(len(counts), 16); self.assertEqual(set(counts.values()), {4})

    def test_13_historical_population_collision_audit_is_empty(self) -> None:
        self.assertEqual(load("round33_confirmation_manifest.json")["historical_disjointness"]["overlapping_seeds"], [])

    def test_14_h5_t20_full_nominal_coverage_is_80_percent(self) -> None:
        row = next(row for row in rows("round33_horizon_coverage.csv") if row["nominal_horizon"] == "5")
        self.assertEqual((row["full_nominal_start_count"], row["coverage_fraction"]), ("16", "0.8"))

    def test_15_h10_t20_full_nominal_coverage_is_55_percent(self) -> None:
        row = next(row for row in rows("round33_horizon_coverage.csv") if row["nominal_horizon"] == "10")
        self.assertEqual((row["full_nominal_start_count"], row["coverage_fraction"]), ("11", "0.55"))

    def test_16_h20_is_full_only_at_the_earliest_slot(self) -> None:
        row = next(row for row in rows("round33_horizon_coverage.csv") if row["nominal_horizon"] == "20")
        self.assertEqual((row["full_nominal_start_count"], row["earliest_only"]), ("1", "True"))

    def test_17_primary_common_anchors_are_t0_through_t10(self) -> None:
        self.assertEqual(load("round33_common_anchor_protocol.json")["anchors"], list(range(11)))

    def test_18_primary_action_audit_covers_both_populations(self) -> None:
        self.assertEqual(len(rows("h5_h10_action_separation.csv")), 128 * 11)

    def test_19_every_primary_action_pair_shares_a_snapshot(self) -> None:
        self.assertTrue(all(row["same_snapshot_verified"] == "True" for row in rows("h5_h10_action_separation.csv")))

    def test_20_primary_action_rows_execute_the_requested_shadow_depths(self) -> None:
        for row in rows("h5_h10_action_separation.csv"):
            self.assertEqual((row["h5_actual_simulated_horizon"], row["h10_actual_simulated_horizon"]), ("5", "10"))

    def test_21_same_snapshot_utility_is_branch_difference(self) -> None:
        for row in rows("h10_vs_h5_same_snapshot_utility.csv"):
            self.assertAlmostEqual(float(row["utility_h10_minus_h5_bits"]), float(row["h10_payload_bits"]) - float(row["h5_payload_bits"]), places=5)

    def test_22_counterfactual_changes_only_the_first_action(self) -> None:
        self.assertTrue(all(row["only_first_action_intervention"] == "True" and row["continuation_policy"] == "frozen_joint_h1_receding_horizon" for row in rows("h10_vs_h5_same_snapshot_utility.csv")))

    def test_23_closed_loop_h1_h5_h10_cover_every_t20_scenario(self) -> None:
        data = rows("round33_closed_loop_planner_metrics.csv")
        for horizon in ("1", "5", "10"):
            self.assertEqual(sum(row["planner_horizon"] == horizon for row in data), 128)

    def test_24_h20_closed_loop_rows_are_prespecified_descriptions(self) -> None:
        data = [row for row in rows("round33_closed_loop_planner_metrics.csv") if row["planner_horizon"] == "20"]
        self.assertEqual(len(data), 4); self.assertTrue(all(row["descriptive_only"] == "True" and row["receding_horizon_every_state"] == "True" for row in data))

    def test_25_pairwise_statistics_use_scenario_cluster_bootstrap(self) -> None:
        self.assertTrue(all(row["bootstrap_cluster"] == "scenario_id" for row in rows("round33_planner_pairwise_statistics.csv")))

    def test_26_k_eta_robustness_is_reported(self) -> None:
        data = rows("round33_K_eta_robustness.csv")
        self.assertEqual({(row["dimension"], row["value"]) for row in data}, {("num_buoys", "3"), ("num_buoys", "5"), ("eta", "0.25"), ("eta", "0.5")})

    def test_27_physical_timescale_validation_uses_only_existing_variables(self) -> None:
        self.assertTrue(all(row["episode_horizon"] == "20" and row["no_new_physical_variable"] == "True" for row in rows("round33_physical_timescale_validation.csv")))

    def test_28_access_audit_has_no_protected_reuse(self) -> None:
        access = load("access_audit.json")
        self.assertEqual((access["development_test_reuse_count"], access["untouched_eval_access_count"]), (0, 0))

    def test_29_required_figure_pairs_exist(self) -> None:
        self.assertEqual(len(list(FIGURES.glob("*.png"))), 16); self.assertEqual(len(list(FIGURES.glob("*.svg"))), 16)

    def test_30_gates_are_consistent_and_round34_is_absent(self) -> None:
        gate, task, benchmark = load("horizon_separation_gate.json"), load("task_waypoint_reentry_gate.json"), load("round33_long_episode_benchmark_gate.json")
        self.assertIn(gate["gate"], {"HS-A", "HS-B", "HS-C"})
        self.assertEqual(task["task_waypoint_reentry"], gate["h10_specific"])
        self.assertEqual(benchmark["long_episode_benchmark_valid"], gate["gate"] != "HS-C")
        self.assertTrue((ROOT / "round_34" / "ROUND_LOCK.json").is_file())
        self.assertTrue((ROOT / "round_35" / "ROUND_LOCK.json").is_file())
        self.assertTrue((ROOT / "round_36" / "round_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
