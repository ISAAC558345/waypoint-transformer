"""Regression guards for the locked Round-26 executed-action audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_26"
OUTPUT = ROUND / "outputs" / "round26_action_ambiguity_20260830_1322"
FIGURES = ROUND / "figures" / "round26_action_ambiguity_20260830_1322"
LOGS = ROUND / "logs"
EXPECTED_R25 = "120A417582F314CDB8B227104DE096C39340D27646F6BF33308CEC018AC72027"

REQUIRED_OUTPUTS = {
    "action_supervision_semantics_audit.md", "action_supervision_ambiguity_spec.md",
    "initial_state_action_ambiguity.csv", "action_vs_taskwaypoint_ambiguity.csv",
    "behavior_action_divergence.csv", "quality_conditioned_action_audit.csv",
    "knn_action_ambiguity.csv", "action_multimodality_audit.csv",
    "mean_action_averaging_risk.csv", "conflict_action_ambiguity.csv",
    "bridge_action_ambiguity.csv", "action_ambiguity_grouped.csv",
    "task_waypoint_action_disambiguation.csv", "head_action_disambiguation_audit.csv",
    "planner_action_coherence.csv", "behavior_planner_action_alignment.csv",
    "planner_action_utility_alignment.csv", "quality_planner_action_alignment.csv",
    "policy_seed_planner_action_alignment.csv", "taskwp_state_planner_alignment.csv",
    "bc_supervision_conflict_score.csv", "bc_conflict_seed_instability.csv",
    "planner_action_distillation_gate.json", "action_supervision_ambiguity_gate.json",
    "task_waypoint_action_disambiguation_gate.json", "round26_readiness_gate.json",
    "run_manifest.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round26ActionSupervisionAmbiguityTests(unittest.TestCase):
    def test_round25_lock_and_frozen_preconditions(self) -> None:
        self.assertEqual(digest(ROOT / "round_25" / "ROUND_LOCK.json"), EXPECTED_R25)
        from wt.round26.audit import verify_frozen_state
        result = verify_frozen_state()
        self.assertTrue(result["passed"])
        self.assertEqual(result["round20_development_test_reuse_count"], 0)
        self.assertEqual(result["untouched_eval_access_count"], 0)

    def test_exact_delivery_set_and_sample_coverage(self) -> None:
        self.assertEqual({path.name for path in OUTPUT.iterdir() if path.is_file()}, REQUIRED_OUTPUTS)
        manifest = read_json(OUTPUT / "run_manifest.json")
        self.assertEqual(manifest["formal_sample_counts"], {
            "h5_train_validation": 5120, "allowed_transitions": 7680,
            "exact_state_groups": 64, "knn_rows": 5120,
        })
        exact = rows(OUTPUT / "initial_state_action_ambiguity.csv")
        self.assertEqual(len(exact), 64)
        self.assertTrue(all(row["behavior_count"] == "10" and row["pair_count"] == "45" for row in exact))
        self.assertEqual({row["behavior_pair"] for row in rows(OUTPUT / "behavior_action_divergence.csv")}, {
            "h10_vs_independent", "h10_vs_h1", "h10_vs_h5",
            "h10_vs_noisy_h10_noise015", "h10_vs_noisy_h10_noise035",
            "h10_vs_random", "h1_vs_independent",
        })

    def test_executed_action_only_and_no_training_path(self) -> None:
        source = (ROOT / "src" / "wt" / "round26" / "audit.py").read_text(encoding="utf-8")
        self.assertIn('row["joint_action_applied_t"]', source)
        self.assertIn("component clipping and norm projection", (OUTPUT / "action_supervision_semantics_audit.md").read_text(encoding="utf-8"))
        for forbidden in ("torch.optim", ".backward(", "torch.save(", "save_checkpoint("):
            self.assertNotIn(forbidden, source)
        manifest = read_json(OUTPUT / "run_manifest.json")
        self.assertFalse(manifest["training_performed"])
        self.assertFalse(manifest["predictor_training_performed"])
        self.assertFalse(manifest["policy_training_performed"])

    def test_knn_and_conditional_audits_are_cross_episode_and_finite(self) -> None:
        source = (ROOT / "src" / "wt" / "round26" / "audit.py").read_text(encoding="utf-8")
        self.assertIn("episodes == episodes[index]", source)
        self.assertIn('"current_graph_state"', source)
        self.assertEqual(len(rows(OUTPUT / "knn_action_ambiguity.csv")), 5120)
        self.assertTrue(all(row["neighbor_count"] == "5" for row in rows(OUTPUT / "knn_action_ambiguity.csv")))
        for path in (OUTPUT / "task_waypoint_action_disambiguation.csv", OUTPUT / "head_action_disambiguation_audit.csv"):
            self.assertTrue(all(math.isfinite(float(row["ambiguity_reduction_ratio"])) for row in rows(path)))

    def test_causal_planner_frozen_policy_and_access_controls(self) -> None:
        planner = rows(OUTPUT / "planner_action_coherence.csv")
        self.assertEqual(len(planner), 64)
        self.assertTrue(all(row["causal_current_state_only"] == "True" and row["future_trace_accessed"] == "False" and row["direct_replay"] == "True" and float(row["first_action_max_abs_error"]) <= 1e-9 for row in planner))
        policy = rows(OUTPUT / "policy_seed_planner_action_alignment.csv")
        self.assertEqual({int(row["seed"]) for row in policy}, set(range(8)))
        self.assertTrue(all(row["inference_only"] == "True" for row in policy))
        access = read_json(LOGS / "access_audit.json")
        self.assertEqual(access["allowed_splits_decoded"], ["train", "validation"])
        self.assertFalse(access["fresh_or_untouched_rollout_performed"])
        self.assertTrue(all(int(access[name]) == 0 for name in (
            "fresh_new_rollout_count", "round20_development_test_reuse_count",
            "round25_fresh_new_rollout_count", "untouched_eval_access_count",
            "untouched_eval_model_access_count", "untouched_eval_policy_rollout_count",
        )))

    def test_preregistered_gates_figures_and_finite_csv_values(self) -> None:
        self.assertEqual(read_json(OUTPUT / "action_supervision_ambiguity_gate.json")["gate"], "AS-B")
        self.assertEqual(read_json(OUTPUT / "task_waypoint_action_disambiguation_gate.json")["gate"], "TD-C")
        self.assertEqual(read_json(OUTPUT / "planner_action_distillation_gate.json")["gate"], "PAD-B")
        self.assertEqual(read_json(OUTPUT / "round26_readiness_gate.json")["recommendation"], "R26-D_conditional_planner_action_distillation_design_only")
        self.assertEqual(len([path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"}]), 36)
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
            self.skipTest("Round-26 lock is written only after pre-lock full regression")
        lock = read_json(lock_path)
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["previous_round_lock_sha256"], EXPECTED_R25)
        self.assertEqual(lock["final_verification"]["required_output_missing"], [])
        for name, value in lock["output_hashes"].items():
            self.assertEqual(digest(OUTPUT / name), value)


if __name__ == "__main__":
    unittest.main()
