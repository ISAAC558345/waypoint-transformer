"""Optimizer-intervention checks. The official 32-update pilot is not run here."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import low_level_reward
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.reward import split_bonus
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.agent import GroupedPilot
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.assess import assess_effect, recommendation
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.clip import apply_clip, assert_disjoint
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.protocol import (
    EFFECT_MEASURABLE,
    EFFECT_MIXED,
    EFFECT_NONE,
    J_MATCH_ABS,
    joint_limit,
    separate_limit,
)
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.shadow import clone_pilot, run_shadow
from wt.benchmark_v2.phase6i_priority_tag_reward.mappo import ExperimentMAPPO
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import critic_dim
from wt.round30.ppo import seed_everything


class _Box(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))


def _effect_row(seed: int, update: int, source: str, ratio, *, zero: bool = False, repeat: bool = True) -> dict:
    return {
        "model_seed": seed,
        "ppo_update": update,
        "source_variant": source,
        "r": ratio,
        "repeat_ok": repeat,
        "j_norm_zero": zero,
    }


def _covered(ratio: float) -> list[dict]:
    rows = []
    for seed in (0, 1, 2):
        for update in (16, 32):
            for source in ("PILOT_J", "PILOT_S"):
                rows.append(_effect_row(seed, update, source, ratio))
    return rows


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


class InterventionRules(unittest.TestCase):
    def test_limits_keep_the_total_budget(self) -> None:
        self.assertEqual(joint_limit(), 0.5)
        self.assertAlmostEqual(separate_limit(), 0.5 / (2.0 ** 0.5))
        self.assertLess(separate_limit() * (2.0 ** 0.5), 0.5 + 1.0e-12)

    def test_parameter_sets_must_be_disjoint(self) -> None:
        shared = nn.Parameter(torch.zeros(2))

        class _Shared(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = shared

        with self.assertRaises(RuntimeError):
            assert_disjoint(_Shared(), _Shared())

    def test_separate_clip_respects_the_joint_bound(self) -> None:
        actor = _Box(8)
        critic = _Box(8)
        actor.weight.grad = torch.ones(8)
        critic.weight.grad = torch.ones(8)
        info = apply_clip(actor, critic, "PILOT_S")
        self.assertEqual(info["actor_limit"], separate_limit())
        self.assertNotEqual(info["actor_limit"], joint_limit())
        self.assertLessEqual(info["joint_post_norm"], joint_limit() + 1.0e-6)
        self.assertLessEqual(info["actor_post_norm"], separate_limit() + 1.0e-6)
        self.assertLessEqual(info["critic_post_norm"], separate_limit() + 1.0e-6)
        self.assertGreater(info["joint_post_norm"], 0.2)

    def test_joint_clip_respects_the_same_bound(self) -> None:
        actor = _Box(8)
        critic = _Box(8)
        actor.weight.grad = torch.full((8,), 3.0)
        critic.weight.grad = torch.full((8,), 4.0)
        info = apply_clip(actor, critic, "PILOT_J")
        self.assertLessEqual(info["joint_post_norm"], joint_limit() + 1.0e-6)
        self.assertEqual(info["actor_scale"], info["critic_scale"])

    def test_zero_gradient_stays_finite(self) -> None:
        actor = _Box(4)
        critic = _Box(4)
        actor.weight.grad = torch.zeros(4)
        critic.weight.grad = torch.zeros(4)
        for rule in ("PILOT_J", "PILOT_S"):
            info = apply_clip(actor, critic, rule)
            self.assertEqual(info["joint_post_norm"], 0.0)
            self.assertEqual(info["actor_post_norm"], 0.0)
            self.assertTrue(np.isfinite(info["actor_scale"]))

    def test_nonfinite_gradient_is_rejected(self) -> None:
        actor = _Box(4)
        critic = _Box(4)
        actor.weight.grad = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
        critic.weight.grad = torch.zeros(4)
        with self.assertRaises(RuntimeError):
            apply_clip(actor, critic, "PILOT_J")
        actor.weight.grad = torch.tensor([float("inf"), 0.0, 0.0, 0.0])
        with self.assertRaises(RuntimeError):
            apply_clip(actor, critic, "PILOT_S")

    def test_reward_identity_and_positive_tagged_service(self) -> None:
        from wt.lowlevel_v2.phase1.protocol import ACTION_SCALE, LAMBDA_PROJECT, LAMBDA_RESIDUAL

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
        expected = 2.0 - 0.05 * penalty_project - 0.01 * penalty_residual + age + tag
        self.assertGreater(tag, 0.0)
        self.assertGreater(age, 0.0)
        self.assertAlmostEqual(base + age + tag, expected)

    def test_shadow_adam_copy_does_not_share_storage(self) -> None:
        actor = _Box(6)
        critic = _Box(5)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3.0e-4)
        loss = actor.weight.sum() + critic.weight.sum()
        loss.backward()
        optimizer.step()
        source_weight = actor.weight.detach().clone()
        source_moment = optimizer.state[actor.weight]["exp_avg"].detach().clone()
        copied_actor, copied_critic, copied = clone_pilot(actor, critic, optimizer)
        copied_actor.weight.data.add_(1.0)
        copied.state[copied_actor.weight]["exp_avg"].add_(1.0)
        self.assertTrue(torch.equal(actor.weight.detach().cpu(), source_weight.cpu()))
        self.assertTrue(torch.equal(optimizer.state[actor.weight]["exp_avg"].detach().cpu(), source_moment.cpu()))
        self.assertNotEqual(actor.weight.data_ptr(), copied_actor.weight.data_ptr())
        self.assertNotEqual(critic.weight.data_ptr(), copied_critic.weight.data_ptr())
        self.assertEqual(int(optimizer.state[actor.weight]["step"]), int(copied.state[copied_actor.weight]["step"]))

    def test_effect_rule_is_fixed(self) -> None:
        measurable = _covered(0.001)
        for seed in (0, 1):
            for update, source in ((16, "PILOT_J"), (16, "PILOT_S"), (32, "PILOT_J")):
                for row in measurable:
                    if row["model_seed"] == seed and row["ppo_update"] == update and row["source_variant"] == source:
                        row["r"] = 0.02
        self.assertEqual(assess_effect(measurable)["effect_assessment"], EFFECT_MEASURABLE)
        quiet = _covered(0.009)
        quiet.extend(_effect_row(0, 1, "PILOT_J", 10.0) for _unused in range(1))
        self.assertEqual(assess_effect(quiet)["effect_assessment"], EFFECT_NONE)
        missing = [row for row in _covered(0.009) if not (row["model_seed"] == 2 and row["ppo_update"] == 32 and row["source_variant"] == "PILOT_S")]
        self.assertEqual(assess_effect(missing)["effect_assessment"], EFFECT_MIXED)
        zeros = _covered(0.001)
        zeros[0]["r"] = None
        zeros[0]["j_norm_zero"] = True
        self.assertEqual(assess_effect(zeros)["effect_assessment"], EFFECT_MIXED)
        one_seed = _covered(0.001)
        for row in one_seed:
            if row["model_seed"] == 0:
                row["r"] = 0.05
        self.assertEqual(assess_effect(one_seed)["effect_assessment"], EFFECT_MIXED)
        self.assertEqual(recommendation(EFFECT_MEASURABLE, trial_complete=True, constraint_failure_count=0), "register_one_full_budget_clip_group_comparison")
        self.assertEqual(recommendation(EFFECT_MEASURABLE, trial_complete=True, constraint_failure_count=1), "do_not_extend_the_pilot")
        self.assertEqual(recommendation(EFFECT_NONE, trial_complete=True, constraint_failure_count=0), "do_not_extend_the_pilot")

    def test_new_sources_omit_forbidden_tokens(self) -> None:
        token = "deci" + "de("
        held = ("322" + "048", "322" + "064")
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_optimizer_intervention"
        for path in list(root.glob("*.py")) + [Path(__file__)]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(token, text)
            for item in held:
                self.assertNotIn(item, text)


class PilotEquivalence(unittest.TestCase):
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
        assert_disjoint(right["actor"], right["critic"])

    def test_joint_path_matches_the_frozen_update(self) -> None:
        storage = _storage()
        reference_actor = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        reference_critic = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        pilot_actor = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        pilot_critic = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        reference = ExperimentMAPPO(reference_actor, reference_critic, PPO)
        pilot = GroupedPilot(pilot_actor, pilot_critic, "PILOT_J")
        seed_everything(1)
        reference.update(storage)
        seed_everything(1)
        pilot.update(storage, ppo_update=2)
        self.assertLessEqual(_max_parameters(reference_actor, pilot_actor), float(J_MATCH_ABS))
        self.assertLessEqual(_max_parameters(reference_critic, pilot_critic), float(J_MATCH_ABS))
        self.assertLessEqual(_moment_gap(reference.optimizer, pilot.optimizer), float(J_MATCH_ABS))

    def test_shadow_hook_leaves_the_real_update_unchanged(self) -> None:
        storage = _storage()
        plain_actor = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        plain_critic = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        hooked_actor = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        hooked_critic = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        plain = GroupedPilot(plain_actor, plain_critic, "PILOT_J")
        hooked = GroupedPilot(hooked_actor, hooked_critic, "PILOT_J")
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "shadow.pt"

            def hook(context) -> None:
                result = run_shadow(context, destination)
                self.assertTrue(result["repeat_ok"])
                self.assertLessEqual(float(result["repeat_max_abs"]), 1.0e-7)

            seed_everything(3)
            hooked.update(storage, ppo_update=1, before_clip=hook)
            seed_everything(3)
            plain.update(storage, ppo_update=1)
        self.assertLessEqual(_max_parameters(plain_actor, hooked_actor), 0.0)
        self.assertLessEqual(_max_parameters(plain_critic, hooked_critic), 0.0)
        self.assertLessEqual(_moment_gap(plain.optimizer, hooked.optimizer), 0.0)
        self.assertTrue(torch.equal(dict(self.bundle["L4_Q"]["actor"].named_parameters())["body.0.weight"].detach().cpu(), dict(self.bundle["L4_C"]["actor"].named_parameters())["body.0.weight"].detach().cpu()))

    def test_reseeded_first_rollout_matches_after_a_shadowed_update(self) -> None:
        from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO as live_ppo
        from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.rollout import collect
        from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.train import _workers
        from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore

        actor_j = copy.deepcopy(self.bundle["L4_Q"]["actor"])
        critic_j = copy.deepcopy(self.bundle["L4_Q"]["critic"])
        actor_s = copy.deepcopy(self.bundle["L4_C"]["actor"])
        critic_s = copy.deepcopy(self.bundle["L4_C"]["critic"])
        pilot = GroupedPilot(actor_j, critic_j, "PILOT_J")
        torch.set_num_threads(1)
        seed_everything(0)
        rollout, original = _install()
        try:
            first, _metrics, _samples = collect(_workers(0, actor_j, critic_j, use_lookahead=True), int(live_ppo.rollout_length))
            with tempfile.TemporaryDirectory() as folder:
                pilot.update(first, ppo_update=1, before_clip=lambda context: run_shadow(context, Path(folder) / "shadow.pt"))
        finally:
            _restore(rollout, original)
        seed_everything(0)
        rollout, original = _install()
        try:
            second, _metrics, _samples = collect(_workers(0, actor_s, critic_s, use_lookahead=True), int(live_ppo.rollout_length))
        finally:
            _restore(rollout, original)
        self.assertEqual(float(np.max(np.abs(first["rewards"] - second["rewards"]))), 0.0)


if __name__ == "__main__":
    unittest.main()
