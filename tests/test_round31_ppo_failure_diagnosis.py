"""Guards for the inference-only Round-31 PPO failure diagnosis."""
from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import unittest

import numpy as np

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.ppo import compute_gae
from wt.round30.scenarios import validation_scenarios
from wt.round31 import audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "round_31" / "outputs" / "round31_ppo_failure_diagnosis_20260830_2116"
FIGURES = ROOT / "round_31" / "figures" / "round31_ppo_failure_diagnosis_20260830_2116"
R30_LOCK_SHA = "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Round31PPOFailureDiagnosisTests(unittest.TestCase):
    def test_01_round30_lock_is_unchanged(self) -> None:
        self.assertEqual(digest(ROOT / "round_30" / "ROUND_LOCK.json"), R30_LOCK_SHA)
        self.assertTrue(json.loads((OUTPUT / "access_audit.json").read_text(encoding="utf-8"))["round30_hashes_before_after_match"])

    def test_02_no_optimizer_update_or_backward(self) -> None:
        tree = ast.parse((ROOT / "src/wt/round31/audit.py").read_text(encoding="utf-8"))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("backward", calls)
        self.assertNotIn("zero_grad", calls)
        self.assertNotIn("torch.optim", (ROOT / "src/wt/round31/audit.py").read_text(encoding="utf-8"))
        run = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((run["optimizer_update_count"], run["backward_pass_count"]), (0, 0))

    def test_03_no_training_or_checkpoint_write(self) -> None:
        source = (ROOT / "src/wt/round31/audit.py").read_text(encoding="utf-8")
        for forbidden in ("torch.save(", "optimizer.step(", ".backward("):
            self.assertNotIn(forbidden, source)
        run = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(run["training_performed"]); self.assertFalse(run["ppo_training_performed"])
        self.assertEqual(run["checkpoint_write_count"], 0)

    def test_04_objective_audit_is_exact(self) -> None:
        text = (OUTPUT / "ppo_objective_semantics_audit.md").read_text(encoding="utf-8")
        for item in ("clipped_surrogate_minimum_ratio", "old_logprob_is_stored_behavior_density", "tanh_logprob_jacobian_correction", "terminated_and_time_limit_are_finite_horizon_terminal", "gradient_clipping", "deterministic_evaluation_uses_mu_then_tanh"):
            self.assertIn(item, text)
        self.assertIn("implementation_bug_found=false", text)

    def test_05_gae_reconstruction_exact(self) -> None:
        rewards = np.asarray([[1.0], [2.0]], dtype=np.float32); values = np.asarray([[.5], [.25]], dtype=np.float32)
        advantage, returns = compute_gae(rewards, values, np.asarray([[False], [True]]), np.asarray([0.0], dtype=np.float32), 1.0, 1.0)
        self.assertTrue(np.allclose(advantage[:, 0], [2.5, 1.75])); self.assertTrue(np.allclose(returns[:, 0], [3.0, 2.0]))
        self.assertGreater(len(read_rows(OUTPUT / "ppo_advantage_statistics.csv")), 0)

    def test_06_actor_representation_is_exact_raw_input_vector(self) -> None:
        scenario = validation_scenarios()[0]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        state = {"scenario": scenario, "info": info}
        vector = audit.actor_raw_input(state, 0)
        self.assertEqual(vector.ndim, 1); self.assertTrue(np.isfinite(vector).all())
        source = inspect.getsource(audit.actor_raw_input)
        self.assertIn("feature_dict", source); self.assertIn("Canonical lexicographic", source)

    def test_07_knn_excludes_same_episode_and_temporal_neighbors(self) -> None:
        rows = read_rows(OUTPUT / "actor_representation_action_ambiguity.csv")
        self.assertTrue(all(row["same_episode_excluded"] == "True" and row["temporal_neighbor_excluded"] == "True" for row in rows))

    def test_08_full_relational_features_are_current_only(self) -> None:
        source = inspect.getsource(audit.full_relational_input).lower()
        self.assertNotIn("future", source)
        self.assertIn("gain_matrix", source); self.assertIn("rate", source)
        rows = read_rows(OUTPUT / "relational_information_loss_audit.csv")
        self.assertTrue(all(row["current_state_only"] == "True" for row in rows))

    def test_09_k_and_eta_grouping_are_present(self) -> None:
        rils = read_rows(OUTPUT / "relational_information_loss_audit.csv")
        groups = {row["group"] for row in rils}
        self.assertTrue({"K3", "K5", "eta_025", "eta_05"}.issubset(groups))
        self.assertEqual({row["group"] for row in read_rows(OUTPUT / "relation_aliasing_by_eta.csv")}, {"eta_00", "eta_025", "eta_05", "eta_10"})

    def test_10_frozen_planner_references_are_used(self) -> None:
        references = read_rows(OUTPUT / "ppo_action_reference_alignment.csv")
        self.assertEqual({row["reference"] for row in references}, {"independent", "joint_h1", "joint_h5"})
        self.assertTrue(all(int(row["states"]) == 48 for row in references))

    def test_11_no_fresh_new_rollout_or_untouched_access(self) -> None:
        access = json.loads((OUTPUT / "access_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(access["fresh_new_rollout_count"], 0); self.assertEqual(access["untouched_eval_access_count"], 0)
        self.assertFalse(access["untouched_eval_accessed"])
        source = (ROOT / "src/wt/round31/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("fresh_scenarios", source)

    def test_12_no_hgnn_transformer_actor_or_task_waypoint(self) -> None:
        source = (ROOT / "src/wt/round31/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("torch_geometric", source)
        self.assertNotIn("build_policy(", source)
        self.assertNotIn("Task-Waypoint", (ROOT / "round_31" / "AGENTS.md").read_text(encoding="utf-8").replace("Task-Waypoint", ""))

    def test_13_outputs_have_no_nan_or_inf(self) -> None:
        for path in OUTPUT.glob("*.csv"):
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for value in row.values():
                        try: number = float(value)
                        except (TypeError, ValueError): continue
                        self.assertTrue(math.isfinite(number), f"{path.name}: {value}")

    def test_14_required_artifacts_and_figures_exist(self) -> None:
        required = {"ppo_objective_semantics_audit.md", "ppo_optimization_health.csv", "ppo_policy_change_audit.csv", "ppo_advantage_statistics.csv", "advantage_payload_alignment.csv", "ppo_reward_scale_audit.csv", "ppo_credit_horizon_audit.md", "ppo_distribution_shift_audit.csv", "ppo_train_validation_gap.csv", "ppo_actor_observation_audit.md", "deepsets_actor_aliasing_audit.csv", "actor_representation_action_ambiguity.csv", "relational_information_loss_audit.csv", "relation_aliasing_by_eta.csv", "ppo_action_reference_alignment.csv", "ppo_action_geometry_audit.csv", "ppo_coordination_outcome_audit.csv", "ppo_K5_failure_audit.csv", "critic_actor_asymmetry_audit.csv", "ppo_gradient_signal_audit.csv", "ppo_evaluation_mode_audit.csv", "ppo_failure_mechanism_evidence.json", "round31_failure_diagnosis_gate.json", "round31_readiness_gate.json", "access_audit.json", "run_manifest.json"}
        self.assertTrue(all((OUTPUT / name).is_file() for name in required))
        self.assertGreaterEqual(len(list(FIGURES.glob("*.png"))), 18); self.assertGreaterEqual(len(list(FIGURES.glob("*.svg"))), 18)

    def test_15_round31_sources_compile_and_legal_round32_successor(self) -> None:
        for path in (ROOT / "src/wt/round31").glob("*.py"):
            compile(path.read_text(encoding="utf-8"), path.as_posix(), "exec")
        successor = ROOT / "round_32"
        if successor.exists():
            manifest = json.loads((successor / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["previous_round_lock"], "round_31/ROUND_LOCK.json")
            self.assertFalse(manifest["round33_created"])


if __name__ == "__main__":
    unittest.main()
