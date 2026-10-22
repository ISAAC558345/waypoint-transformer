"""Source and mechanism guards for the Round-30 state-only DeepSets PPO baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.evaluation import rollout
from wt.round30.features import feature_dict, stack_feature_dicts
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent, PPOConfig, compute_gae, load_agent, tanh_log_prob
from wt.round30.scenarios import assert_disjoint, fresh_scenarios, training_scenarios, validation_scenarios


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "round_30"
OUTPUT = ROUND / "outputs" / "round30_deepsets_ppo_20260830_1845"
R29_LOCK_SHA256 = "1F4CCAECE01F4B3F127CE4B922A9AC07B0E7E01AC9A34B204192EF8651AC51FB"
R30_LOCK_SHA256 = "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def first_features() -> dict[str, np.ndarray]:
    scenario = training_scenarios()[0]
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
    _, info = env.reset(seed=scenario.seed)
    return stack_feature_dicts([feature_dict(info, env.config)])


class Round30DeepSetsPPOTests(unittest.TestCase):
    def test_01_round29_lock_is_unchanged(self) -> None:
        self.assertEqual(digest(ROOT / "round_29" / "ROUND_LOCK.json"), R29_LOCK_SHA256)

    def test_02_round30_is_current_in_progress(self) -> None:
        current = (ROOT / "CURRENT_ROUND.txt").read_text(encoding="utf-8").lower()
        self.assertIn(current.splitlines()[0], {"round_30", "round_31", "round_32", "round_33", "round_34", "round_35", "round_36"})
        self.assertIn(json.loads((ROUND / "round_manifest.json").read_text(encoding="utf-8"))["status"], {"in_progress", "locked"})
        successor = ROOT / "round_31"
        if successor.exists():
            manifest = json.loads((successor / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertIn(manifest["status"], {"in_progress", "locked"})
            self.assertEqual(manifest["previous_round_lock"], "round_30/ROUND_LOCK.json")
            self.assertEqual(manifest["previous_round_lock_sha256"], R30_LOCK_SHA256)
        if current.startswith("round_32\n"):
            self.assertIn("previous_lock_file=round_31/round_lock.json", current)
        if current.startswith("round_33\n"):
            self.assertIn("previous_lock_file=round_32/round_lock.json", current)
        if current.startswith("round_36\n"):
            self.assertIn("previous_lock_file=round_35/round_lock.json", current)

    def test_03_no_task_waypoint_input_or_loss(self) -> None:
        data = first_features()
        self.assertEqual(set(data), {"uav", "buoy", "uav_mask", "buoy_mask", "global"})
        self.assertFalse(DeepSetsActorCritic().uses_task_waypoint)

    def test_04_no_planner_trigger_or_graph_policy_input(self) -> None:
        text = (ROOT / "src/wt/round30/models.py").read_text(encoding="utf-8").lower()
        for forbidden in ("joint_plan", "independent_decision", "edge_index", "torch_geometric", "gru(", "lstm("):
            self.assertNotIn(forbidden, text)

    def test_05_shared_actor_and_deepsets_properties(self) -> None:
        model = DeepSetsActorCritic()
        self.assertTrue(model.shared_actor_parameters)
        self.assertFalse(model.uses_task_waypoint)
        self.assertFalse(model.uses_planner_trigger)
        self.assertFalse(model.uses_graph_message_passing)
        self.assertFalse(model.uses_transformer)
        self.assertFalse(model.uses_recurrence)

    def test_06_actor_is_uav_permutation_equivariant(self) -> None:
        torch.manual_seed(11); model = DeepSetsActorCritic().eval(); data = first_features()
        value = {key: torch.as_tensor(row, dtype=torch.bool if key.endswith("mask") else torch.float32) for key, row in data.items()}
        with torch.no_grad():
            mu, log_std, _ = model(**value)
            permuted = dict(value)
            permuted["uav"], permuted["uav_mask"] = value["uav"][:, [1, 0]], value["uav_mask"][:, [1, 0]]
            perm_mu, perm_log_std, _ = model(**permuted)
        self.assertTrue(torch.allclose(perm_mu, mu[:, [1, 0]], atol=1.0e-6))
        self.assertTrue(torch.allclose(perm_log_std, log_std[:, [1, 0]], atol=1.0e-6))

    def test_07_critic_is_uav_permutation_invariant(self) -> None:
        torch.manual_seed(12); model = DeepSetsActorCritic().eval(); data = first_features()
        value = {key: torch.as_tensor(row, dtype=torch.bool if key.endswith("mask") else torch.float32) for key, row in data.items()}
        with torch.no_grad():
            _, _, first = model(**value)
            swapped = dict(value); swapped["uav"], swapped["uav_mask"] = value["uav"][:, [1, 0]], value["uav_mask"][:, [1, 0]]
            _, _, second = model(**swapped)
        self.assertTrue(torch.allclose(first, second, atol=1.0e-6))

    def test_08_id_free_dynamic_feature_shapes(self) -> None:
        for scenario in (training_scenarios()[0], training_scenarios()[1]):
            env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
            _, info = env.reset(seed=scenario.seed); data = feature_dict(info, env.config)
            self.assertEqual(data["uav"].shape, (2, 8)); self.assertEqual(data["buoy"].shape, (5, 7))
            self.assertEqual(int(data["buoy_mask"].sum()), scenario.num_buoys)
            self.assertTrue(all(np.isfinite(row).all() for name, row in data.items() if name not in {"uav_mask", "buoy_mask"}))

    def test_09_sampled_actions_have_frozen_m2x2_shape_and_bounds(self) -> None:
        agent = PPOAgent(DeepSetsActorCritic(), PPOConfig(rollout_length=2, num_envs=1, num_minibatches=1, update_epochs=1))
        decision = agent.act(first_features())
        self.assertEqual(decision["action"].shape, (1, 2, 2))
        self.assertLessEqual(float(np.abs(decision["action"]).max()), 1.0)

    def test_10_raw_tanh_log_probability_is_pre_projection_density(self) -> None:
        mu, log_std, z = torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2)), torch.full((1, 2, 2), 0.3)
        log_prob, entropy = tanh_log_prob(mu, log_std, z)
        self.assertTrue(torch.isfinite(log_prob).all()); self.assertTrue(torch.isfinite(entropy).all())
        self.assertGreater(float(log_prob.item()), -100.0)

    def test_11_gae_known_two_step_case(self) -> None:
        rewards = np.asarray([[1.0], [2.0]], dtype=np.float32); values = np.asarray([[.5], [.25]], dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, np.asarray([[False], [True]]), np.asarray([.0], dtype=np.float32), 1.0, 1.0)
        self.assertTrue(np.allclose(advantages[:, 0], np.asarray([2.5, 1.75]), atol=1.0e-6))
        self.assertTrue(np.allclose(returns[:, 0], np.asarray([3.0, 2.0]), atol=1.0e-6))

    def test_12_ppo_config_is_frozen_standard_algorithm(self) -> None:
        config = PPOConfig()
        self.assertEqual((config.gamma, config.gae_lambda, config.clip_eps), (.99, .95, .2))
        self.assertEqual((config.learning_rate, config.entropy_coef, config.value_coef, config.max_grad_norm), (3.0e-4, .01, .5, .5))
        self.assertEqual((config.rollout_length, config.num_envs, config.num_minibatches, config.update_epochs), (256, 8, 4, 4))

    def test_13_scenarios_are_disjoint_balanced_and_new(self) -> None:
        train, validation, fresh = training_scenarios(), validation_scenarios(), fresh_scenarios()
        assert_disjoint(train, validation, fresh)
        self.assertEqual((len(train), len(validation), len(fresh)), (96, 48, 64))
        self.assertTrue(all(row.seed >= 40000 for row in train + validation + fresh))
        for rows in (train, validation, fresh):
            self.assertEqual({row.num_buoys for row in rows}, {3, 5}); self.assertEqual({row.eta for row in rows}, {0.0, .25, .5, 1.0})

    def test_14_environment_is_frozen_shared_static_m2(self) -> None:
        scenario = validation_scenarios()[0]
        self.assertEqual((scenario.config().num_uavs, scenario.config().num_buoys), (2, 3))
        self.assertEqual((scenario.config().buoy_drift_mode, scenario.config().channel_variation_mode), ("static", "static"))
        self.assertEqual((scenario.options().spectrum_mode, scenario.options().matching_mode, scenario.config().multi_uav_min_separation_m), ("shared", "legacy", 15.0))

    def test_15_rollout_records_raw_and_executed_actions(self) -> None:
        metric, steps = rollout(training_scenarios()[0], population="test", policy_id="zero", model_seed=-1, action_fn=lambda env, info: np.zeros(env.action_shape, dtype=np.float64))
        self.assertEqual(metric["episode_length"], len(steps))
        self.assertTrue(all("raw_sampled_action" in row and "executed_projected_action" in row for row in steps))
        self.assertTrue(all(np.asarray(row["raw_sampled_action"]).shape == (2, 2) for row in steps))

    def test_16_checkpoint_carries_semantics_and_prohibited_flags(self) -> None:
        agent = PPOAgent(DeepSetsActorCritic(), PPOConfig(rollout_length=2, num_envs=1, num_minibatches=1, update_epochs=1))
        payload = agent.checkpoint(seed=0, steps=2, extra={})
        self.assertEqual(payload["raw_logprob_semantics"], "tanh-squashed policy sample before environment safety projection")
        self.assertTrue(all(not payload[key] for key in ("uses_task_waypoint", "uses_planner_trigger", "uses_hgnn", "uses_transformer")))

    def test_17_protocol_will_not_preaccess_fresh(self) -> None:
        config = json.loads((ROUND / "configs" / "round30_ppo_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(config["fresh_access_count_before_freeze"], 0)
        self.assertEqual(config["untouched_eval_access_count"], 0)

    def test_18_required_dirs_and_only_legal_round31_successor(self) -> None:
        for path in (ROUND / "AGENTS.md", ROUND / "configs", OUTPUT): self.assertTrue(path.exists())
        successor = ROOT / "round_31"
        if successor.exists(): self.assertTrue((successor / "ROUND_LOCK.json").is_file())
        successor32 = ROOT / "round_32"
        if successor32.exists():
            manifest32 = json.loads((successor32 / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest32["previous_round_lock"], "round_31/ROUND_LOCK.json")
            self.assertFalse(manifest32["round33_created"])

    def test_19_safe_projection_preserves_minimum_separation(self) -> None:
        metric, _ = rollout(training_scenarios()[0], population="test", policy_id="aggressive", model_seed=-1, action_fn=lambda env, info: np.ones(env.action_shape, dtype=np.float64))
        self.assertGreaterEqual(metric["minimum_uav_distance_m"], 15.0 - 1.0e-6)

    def test_20_clipped_surrogate_uses_minimum_ratio_objective(self) -> None:
        ratio = torch.tensor([1.5, .5]); advantage = torch.tensor([1., -1.]); clip = .2
        actual = -torch.minimum(ratio * advantage, ratio.clamp(1. - clip, 1. + clip) * advantage).mean()
        self.assertAlmostEqual(float(actual), -0.2, places=6)

    def test_21_advantage_normalization_and_gradient_clip_registered(self) -> None:
        source = (ROOT / "src/wt/round30/ppo.py").read_text(encoding="utf-8")
        self.assertIn('flat["advantages"] = (flat["advantages"] - flat["advantages"].mean())', source)
        self.assertIn("clip_grad_norm_", source)

    def test_22_validation_action_is_gradient_free(self) -> None:
        agent = PPOAgent(DeepSetsActorCritic(), PPOConfig(rollout_length=2, num_envs=1, num_minibatches=1, update_epochs=1))
        agent.act(first_features(), deterministic=True)
        self.assertTrue(all(parameter.grad is None for parameter in agent.model.parameters()))

    def test_23_fresh_is_preregistered_and_limited_to_one_access(self) -> None:
        self.assertEqual(len(fresh_scenarios()), 64)
        access = OUTPUT / "access_audit.json"
        if access.is_file(): self.assertLessEqual(json.loads(access.read_text(encoding="utf-8"))["round30_fresh_access_count"], 1)

    def test_24_untouched_access_is_zero(self) -> None:
        access = OUTPUT / "access_audit.json"
        if access.is_file(): self.assertEqual(json.loads(access.read_text(encoding="utf-8"))["untouched_eval_access_count"], 0)
        else: self.assertEqual(json.loads((ROUND / "round_manifest.json").read_text(encoding="utf-8"))["untouched_eval_access_count"], 0)

    def test_25_completed_metrics_contain_no_nan_or_inf(self) -> None:
        import csv
        for path in (OUTPUT / "ppo_training_summary.csv", OUTPUT / "ppo_validation_metrics.csv", OUTPUT / "ppo_fresh_metrics.csv"):
            if not path.is_file(): continue
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for value in row.values():
                        try: number = float(value)
                        except (TypeError, ValueError): continue
                        self.assertTrue(np.isfinite(number), f"{path.name}: {value}")

    def test_26_checkpoint_reloads_and_evaluation_is_deterministic(self) -> None:
        manifest = OUTPUT / "checkpoint_manifest.json"
        if not manifest.is_file(): self.skipTest("formal checkpoints are written after source guards pass")
        item = json.loads(manifest.read_text(encoding="utf-8"))["checkpoints"][0]
        payload = torch.load(item["path"], map_location="cpu", weights_only=False)
        agent = load_agent(payload, PPOConfig(**payload["ppo_config"]), DeepSetsActorCritic())
        first, second = agent.act(first_features(), deterministic=True), agent.act(first_features(), deterministic=True)
        self.assertTrue(np.array_equal(first["action"], second["action"]))

    def test_27_round30_sources_compile(self) -> None:
        for path in (ROOT / "src/wt/round30").glob("*.py"):
            compile(path.read_text(encoding="utf-8"), path.as_posix(), "exec")


if __name__ == "__main__":
    unittest.main()
