"""Full-budget clip-group checks. The official 300-update runs are not started here."""
from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import low_level_reward
from wt.benchmark_v2.phase6i_priority_tag_clip_benchmark.criteria import choose_gate, score_full
from wt.benchmark_v2.phase6i_priority_tag_clip_benchmark.evaluate import assert_checkpoint_identity
from wt.benchmark_v2.phase6i_priority_tag_clip_benchmark.protocol import (
    CLIP_RULE,
    E14_OUTPUT,
    SNAPSHOT_UPDATES,
    expected_adam_steps,
    schema,
)
from wt.benchmark_v2.phase6i_priority_tag_clip_benchmark.reconcile import reconcile_e14
from wt.benchmark_v2.phase6i_priority_tag_clip_benchmark.safety import closest_pair
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.reward import split_bonus
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.clip import apply_clip, assert_disjoint
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.protocol import J_MATCH_ABS, joint_limit, separate_limit
from wt.benchmark_v2.phase6i_priority_tag_reward.mappo import ExperimentMAPPO
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import EVAL_ASSIGNMENT_SEED
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import critic_dim
from wt.lowlevel_v2.phase1.protocol import ACTION_SCALE, LAMBDA_PROJECT, LAMBDA_RESIDUAL
from wt.round30.ppo import seed_everything


class _Tiny(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))


def _storage() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    steps, envs, uavs = 4, 1, 6
    width = int(critic_dim())

    def noise(shape):
        return rng.normal(size=shape).astype(np.float32)

    return {
        "self": noise((steps, envs, uavs, 5)),
        "task": noise((steps, envs, uavs, 8)),
        "communication": noise((steps, envs, uavs, 4)),
        "neighbor_set": noise((steps, envs, uavs, 5, 11)),
        "d_nearest": noise((steps, envs, uavs)),
        "priority": noise((steps, envs, uavs, 5)),
        "global_state": noise((steps, envs, width)),
        "pre_tanh": noise((steps, envs, uavs, 2)) * 0.1,
        "log_prob_uav": noise((steps, envs, uavs)) * 0.1,
        "policy_active_mask": np.ones((steps, envs, uavs), dtype=np.float32),
        "values": noise((steps, envs)) * 0.1,
        "rewards": noise((steps, envs)) * 0.1,
        "dones": np.zeros((steps, envs), dtype=np.float32),
        "last_values": np.zeros((envs,), dtype=np.float32),
    }


def _row(variant: str, seed: int, scenario: int, tagged: float, payload: float, safety: float = 0.0) -> dict:
    return {
        "variant": variant,
        "model_seed": int(seed),
        "scenario_id": int(scenario),
        "served_tagged_bits": float(tagged),
        "payload_bits": float(payload),
        "sinr_mean": 1.0,
        "interference_mean": 1.0,
        "safety_violations": float(safety),
        "constraint_failures": 0.0,
    }


def _l0() -> list[dict]:
    return [_row("L0", 0, scenario, 100.0, 1000.0) for scenario in (1, 2, 3, 4)]


def _paired(seed: int, tagged_gain: float, *, safety: float = 0.0) -> list[dict]:
    rows = []
    for scenario in (1, 2, 3, 4):
        rows.append(_row("FULL_J", seed, scenario, 100.0, 1000.0))
        rows.append(_row("FULL_S", seed, scenario, 100.0 + tagged_gain, 1000.0, safety))
    return rows


class ClipBenchmarkRules(unittest.TestCase):
    def test_budget_and_clip_mapping_stay_registered(self) -> None:
        body = schema()
        self.assertEqual(body["expected_ppo_updates"], 1800)
        self.assertEqual(body["expected_adam_steps"], 28800)
        self.assertEqual(expected_adam_steps(), 28800)
        self.assertEqual(SNAPSHOT_UPDATES, (1, 150, 300))
        self.assertFalse(body["seamless_resume"])
        self.assertEqual(CLIP_RULE, {"FULL_J": "PILOT_J", "FULL_S": "PILOT_S"})
        self.assertEqual(float(joint_limit()), 0.5)
        self.assertAlmostEqual(float(separate_limit()), 0.5 / math.sqrt(2.0))
        self.assertNotEqual(float(separate_limit()), 0.5)
        self.assertEqual(int(EVAL_ASSIGNMENT_SEED), 7000000)

    def test_group_clip_keeps_the_joint_bound(self) -> None:
        actor = _Tiny(8)
        critic = _Tiny(8)
        actor.weight.grad = torch.full((8,), 3.0)
        critic.weight.grad = torch.full((8,), 4.0)
        separate = apply_clip(actor, critic, "PILOT_S")
        self.assertLessEqual(float(separate["joint_post_norm"]), 0.5 + 1.0e-6)
        self.assertAlmostEqual(float(separate["actor_limit"]), 0.5 / math.sqrt(2.0))
        actor.weight.grad = torch.full((8,), 3.0)
        critic.weight.grad = torch.full((8,), 4.0)
        joint = apply_clip(actor, critic, "PILOT_J")
        self.assertLessEqual(float(joint["joint_post_norm"]), 0.5 + 1.0e-6)
        self.assertEqual(float(joint["actor_limit"]), 0.5)
        self.assertEqual(float(joint["critic_limit"]), 0.5)

    def test_overlapping_parameters_are_rejected(self) -> None:
        actor = _Tiny(4)
        with self.assertRaises(RuntimeError):
            assert_disjoint(actor, actor)

    def test_reward_identity_and_positive_tag_bonus(self) -> None:
        executed = np.zeros((6, 2), dtype=np.float64)
        desired = np.ones((6, 2), dtype=np.float64)
        delta = np.ones((6, 2), dtype=np.float64)
        self.assertEqual(float(LAMBDA_PROJECT), 0.05)
        self.assertEqual(float(LAMBDA_RESIDUAL), 0.01)
        base, _payload = low_level_reward(2.0e6, executed, desired, delta, 1.0e6)
        age, tag = split_bonus(1.0e6, 5.0e5, 2.0e6)
        scale = float(ACTION_SCALE) ** 2 + 1.0e-8
        penalty_project = float(np.mean(np.sum((executed - desired) ** 2, axis=-1) / scale))
        penalty_residual = float(np.mean(np.sum(delta ** 2, axis=-1) / scale))
        self.assertGreater(tag, 0.0)
        self.assertAlmostEqual(base + age + tag, 2.0 - 0.05 * penalty_project - 0.01 * penalty_residual + age + tag)

    def test_e14_reconciliation_and_nonresumable_pilots(self) -> None:
        body = reconcile_e14()
        self.assertEqual(body["training_rows"], 192)
        self.assertEqual(body["shadow_rows"], 18)
        self.assertEqual(body["effect_comparisons"], 12)
        self.assertEqual(body["safety_events"], 4)
        self.assertEqual(body["safety_geometry"], "unknown")
        self.assertEqual(body["repeat_max_abs"], 0.0)
        self.assertTrue(body["fresh_start_required"])
        self.assertEqual(len(body["checkpoints"]), 6)
        pilot = torch.load(E14_OUTPUT / "checkpoints" / "pilot_j_seed0.pt", map_location="cpu", weights_only=False)
        self.assertNotIn("env", pilot)
        self.assertNotIn("worker", pilot)
        with self.assertRaises(RuntimeError):
            assert_checkpoint_identity(pilot, "FULL_J", 0)
        fresh = {
            "variant": "FULL_J",
            "model_seed": 0,
            "ppo_updates": 32,
            "warm_start_from_pilot": False,
            "seamless_resume": False,
        }
        with self.assertRaises(RuntimeError):
            assert_checkpoint_identity(fresh, "FULL_J", 0)

    def test_gate_cases_stay_fixed(self) -> None:
        passing = _l0() + _paired(0, 20.0) + _paired(1, 20.0) + _paired(2, 0.0)
        score = score_full(passing)
        self.assertEqual(score["passing_seeds"], 2)
        self.assertEqual(choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, preflight_ok=True, score=score), "CLIP_GROUP_BENCHMARK_PASS")
        unsafe = _l0() + _paired(0, 20.0) + _paired(1, 20.0) + _paired(2, 20.0, safety=1.0)
        unsafe_score = score_full(unsafe)
        self.assertEqual(choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, preflight_ok=True, score=unsafe_score), "CLIP_GROUP_BENCHMARK_FAIL")
        negative = _l0() + _paired(0, -5.0) + _paired(1, -5.0) + _paired(2, 20.0)
        self.assertEqual(
            choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, preflight_ok=True, score=score_full(negative)),
            "CLIP_GROUP_BENCHMARK_FAIL",
        )
        inconclusive = _l0() + _paired(0, 20.0) + _paired(1, 0.0) + _paired(2, 0.0)
        self.assertEqual(
            choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, preflight_ok=True, score=score_full(inconclusive)),
            "CLIP_GROUP_BENCHMARK_INCONCLUSIVE",
        )
        self.assertEqual(choose_gate(tests_ok=False, train_ok=True, hashes_ok=True, preflight_ok=True, score=score), "CLIP_GROUP_BENCHMARK_BLOCKED")

    def test_same_pair_distance_and_unknown_geometry(self) -> None:
        points = np.asarray([[0.0, 0.0], [0.4, 0.0], [3.0, 0.0]], dtype=np.float64)
        pair = closest_pair(points)
        self.assertEqual((pair["uav_i"], pair["uav_j"]), (0, 1))
        self.assertAlmostEqual(pair["distance_m"], 0.4)

    def test_new_sources_omit_forbidden_tokens(self) -> None:
        token = "deci" + "de("
        held = ("322" + "048", "322" + "064")
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_clip_benchmark"
        for path in list(root.glob("*.py")) + [Path(__file__)]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(token, text)
            for item in held:
                self.assertNotIn(item, text)


def _max_parameters(left, right) -> float:
    gap = 0.0
    for first, second in zip(left.parameters(), right.parameters()):
        gap = max(gap, float(torch.max(torch.abs(first.detach().cpu() - second.detach().cpu())).cpu()))
    return gap


def _moment_gap(left, right) -> float:
    gap = 0.0
    for first, second in zip(left.param_groups[0]["params"], right.param_groups[0]["params"]):
        for key in ("exp_avg", "exp_avg_sq"):
            gap = max(gap, float(torch.max(torch.abs(left.state[first][key].detach().cpu() - right.state[second][key].detach().cpu())).cpu()))
    return gap


class ClipBenchmarkEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.train import build_pair

        torch.set_num_threads(1)
        cls.bundle = build_pair(0)

    def test_initial_states_match_and_new_columns_are_zero(self) -> None:
        left = self.bundle["L4_Q"]
        right = self.bundle["L4_C"]
        self.assertTrue(all(torch.equal(a.detach(), b.detach()) for a, b in zip(left["actor"].parameters(), right["actor"].parameters())))
        self.assertTrue(all(torch.equal(a.detach(), b.detach()) for a, b in zip(left["critic"].parameters(), right["critic"].parameters())))
        weight = dict(left["actor"].named_parameters())["body.0.weight"]
        self.assertEqual(int(torch.count_nonzero(weight[:, -2:]).cpu()), 0)
        assert_disjoint(left["actor"], left["critic"])
        self.assertEqual(len(left["agent"].optimizer.state), 0)
        self.assertEqual(len(right["agent"].optimizer.state), 0)

    def test_joint_path_matches_the_frozen_update(self) -> None:
        storage = _storage()
        reference_actor = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        reference_critic = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        pilot_actor = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        pilot_critic = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        reference = ExperimentMAPPO(reference_actor, reference_critic, PPO)
        from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.agent import GroupedPilot

        pilot = GroupedPilot(pilot_actor, pilot_critic, CLIP_RULE["FULL_J"])
        seed_everything(1)
        reference.update(storage)
        seed_everything(1)
        pilot.update(storage, ppo_update=2)
        self.assertLessEqual(_max_parameters(reference_actor, pilot_actor), float(J_MATCH_ABS))
        self.assertLessEqual(_max_parameters(reference_critic, pilot_critic), float(J_MATCH_ABS))
        self.assertLessEqual(_moment_gap(reference.optimizer, pilot.optimizer), float(J_MATCH_ABS))

    def test_reseeded_first_rollout_matches_without_a_shadow_hook(self) -> None:
        from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.rollout import collect
        from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.train import _workers
        from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore

        actor_j = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        critic_j = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        actor_s = copy.deepcopy(self.bundle["L4_C"]["actor"])
        critic_s = copy.deepcopy(self.bundle["L4_C"]["critic"])
        torch.set_num_threads(1)
        seed_everything(0)
        rollout, original = _install()
        try:
            probe = _workers(0, actor_j, critic_j, use_lookahead=True)
            stepped = probe[0].step()
            self.assertFalse(bool(stepped["obs"].get("leaks_future_arrivals")))
            self.assertNotIn("future_arrivals", stepped["obs"])
            self.assertAlmostEqual(
                float(stepped["reward"]),
                float(stepped["base_reward"]) + float(stepped["bonus_age"]) + float(stepped["bonus_tag"]),
            )
        finally:
            _restore(rollout, original)
        seed_everything(0)
        rollout, original = _install()
        try:
            left, _metrics, samples = collect(_workers(0, actor_j, critic_j, use_lookahead=True), int(PPO.rollout_length))
            self.assertTrue(samples)
            self.assertTrue(
                all(
                    abs(float(sample["reward"]) - (float(sample["base_reward"]) + float(sample["bonus_age"]) + float(sample["bonus_tag"]))) <= 1.0e-5
                    for sample in samples
                )
            )
        finally:
            _restore(rollout, original)
        seed_everything(0)
        rollout, original = _install()
        try:
            right, _metrics, _samples = collect(_workers(0, actor_s, critic_s, use_lookahead=True), int(PPO.rollout_length))
        finally:
            _restore(rollout, original)
        self.assertEqual(float(np.max(np.abs(left["rewards"] - right["rewards"]))), 0.0)
