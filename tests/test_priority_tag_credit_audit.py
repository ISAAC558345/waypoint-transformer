"""Credit-audit checks. The twelve registered rollouts stay outside this file."""
from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6i_priority_tag_credit_audit.credit import _cosine, clip_scale, decompose_advantages, static_gradients
from wt.benchmark_v2.phase6i_priority_tag_credit_audit.history import audit_history
from wt.benchmark_v2.phase6i_priority_tag_credit_audit.probe import capture_rng, install_update_guard, match_saved_slice, restore_rng, restore_update_guard, same_rng
from wt.benchmark_v2.phase6i_priority_tag_credit_audit.protocol import (
    BIT_ABS,
    E12_GATE,
    FLOAT32_ABS,
    GATE_BLOCKED,
    GATE_COMPLETE,
    LOGPROB_ABS,
    LOOKAHEAD_ABS,
    MAX_REGISTERED_BATCHES,
    RATIO_ABS,
    REWARD_ABS,
    ROOT,
    schema,
)
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.actor import LookaheadActor
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.protocol import ACTOR_FEATURE_COUNT
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import critic_dim
from wt.lowlevel_v2.observation import COMM_DIM, NEIGHBOR_EDGE_DIM, SELF_DIM, TASK_DIM
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.lowlevel_v2.residual_policy import CentralizedCritic


def _storage():
    steps, envs, uavs = 4, int(PPO.num_envs) // 2, 6
    if envs < 2:
        envs = 2
    storage = {
        "self": np.zeros((steps, envs, uavs, SELF_DIM), dtype=np.float32),
        "task": np.zeros((steps, envs, uavs, TASK_DIM), dtype=np.float32),
        "communication": np.zeros((steps, envs, uavs, COMM_DIM), dtype=np.float32),
        "neighbor_set": np.zeros((steps, envs, uavs, 5, NEIGHBOR_EDGE_DIM), dtype=np.float32),
        "d_nearest": np.zeros((steps, envs, uavs), dtype=np.float32),
        "priority": np.zeros((steps, envs, uavs, ACTOR_FEATURE_COUNT), dtype=np.float32),
        "global_state": np.zeros((steps, envs, critic_dim()), dtype=np.float32),
        "pre_tanh": np.zeros((steps, envs, uavs, 2), dtype=np.float32),
        "log_prob_uav": np.zeros((steps, envs, uavs), dtype=np.float32),
        "policy_active_mask": np.ones((steps, envs, uavs), dtype=np.float32),
        "values": np.full((steps, envs), 0.5, dtype=np.float32),
        "rewards": np.ones((steps, envs), dtype=np.float32),
        "dones": np.zeros((steps, envs), dtype=np.float32),
        "last_values": np.zeros(envs, dtype=np.float32),
    }
    storage["dones"][-1] = 1.0
    storage["pre_tanh"][..., 0] = 0.1
    storage["priority"][..., -1] = 0.25
    base = np.ones((steps, envs), dtype=np.float64)
    age = np.zeros_like(base)
    tag = np.zeros_like(base)
    age[0, 0] = 0.25
    tag[1, 0] = 0.5
    storage["rewards"] = np.asarray(base + age + tag, dtype=np.float32)
    return storage, {"base": base, "age": age, "tag": tag}


def _fill_old_logprob(actor, storage) -> None:
    count = int(storage["rewards"].shape[0] * storage["rewards"].shape[1])

    def flat(key):
        value = storage[key]
        return value.reshape((count,) + value.shape[2:])

    with torch.no_grad():
        mu, log_std = actor(
            torch.as_tensor(flat("self"), dtype=torch.float32),
            torch.as_tensor(flat("task"), dtype=torch.float32),
            torch.as_tensor(flat("communication"), dtype=torch.float32),
            torch.as_tensor(flat("neighbor_set"), dtype=torch.float32),
            torch.as_tensor(flat("d_nearest"), dtype=torch.float32),
            torch.as_tensor(flat("priority"), dtype=torch.float32),
        )
        pre = torch.as_tensor(flat("pre_tanh"), dtype=torch.float32)
        log_prob, _entropy = tanh_log_prob_uav(mu, log_std, pre)
    storage["log_prob_uav"] = log_prob.cpu().numpy().reshape(storage["rewards"].shape + (storage["self"].shape[2],)).astype(np.float32)


class CreditAuditTests(unittest.TestCase):
    def test_protocol_limits_are_locked(self):
        recorded = schema()
        self.assertEqual(int(MAX_REGISTERED_BATCHES), 12)
        self.assertEqual(recorded["registered_batches"], 12)
        self.assertEqual(recorded["verification_reruns"], 0)
        self.assertEqual(float(BIT_ABS), 1.0)
        self.assertEqual(float(REWARD_ABS), 1.0e-5)
        self.assertEqual(float(LOOKAHEAD_ABS), 1.0e-5)
        self.assertEqual(float(FLOAT32_ABS), 1.0e-6)
        self.assertEqual(float(LOGPROB_ABS), 1.0e-4)
        self.assertEqual(float(RATIO_ABS), 1.0e-4)
        self.assertEqual(GATE_COMPLETE, "TAG_CREDIT_AUDIT_COMPLETE")
        self.assertEqual(GATE_BLOCKED, "TAG_CREDIT_AUDIT_BLOCKED")

    def test_gae_decomposition_and_normalization(self):
        _storage_arrays, components = _storage()
        values = np.full(components["base"].shape, 0.5, dtype=np.float32)
        dones = np.zeros_like(values)
        dones[-1] = 1.0
        last = np.zeros(values.shape[1], dtype=np.float32)
        decomposed = decompose_advantages(components["base"], components["age"], components["tag"], values, dones, last)
        self.assertLess(decomposed["shared_scale_additive_gap"], 1.0e-6)
        self.assertLess(decomposed["float32_linear_gap"], 1.0e-4)
        self.assertIn("separate", decomposed["normalization_note"])
        self.assertGreater(decomposed["positive_tag_steps"], 0)
        self.assertGreater(decomposed["tag_advantage_positive_bonus"]["count"], 0)
        self.assertIsNotNone(decomposed["tag_advantage_zero_bonus"]["count"])

    def test_fixed_critic_gradient_repeat_mask_and_sum(self):
        torch.manual_seed(0)
        actor = LookaheadActor()
        critic = CentralizedCritic(critic_dim())
        storage, components = _storage()
        _fill_old_logprob(actor, storage)
        before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        first = static_gradients(actor, critic, storage, components)
        second = static_gradients(actor, critic, storage, components)
        after = actor.state_dict()
        self.assertTrue(first["parameters_unchanged"])
        self.assertTrue(all(torch.equal(after[key], before[key]) for key in before))
        self.assertLess(first["logprob_max_abs"], float(LOGPROB_ABS))
        self.assertLess(first["ratio_max_abs_from_one"], float(RATIO_ABS))
        self.assertLess(first["within_step_policy_term_max_abs"], 1.0e-5)
        self.assertLess(first["per_uav_gradient_sum_max_abs"], 1.0e-4)
        self.assertEqual(first["mask_min"], 1.0)
        self.assertEqual(first["mask_max"], 1.0)
        self.assertFalse(first["clip_applied"])
        self.assertFalse(first["optimizer_step_called"])
        self.assertAlmostEqual(first["policy_norm_full"], second["policy_norm_full"], places=5)
        self.assertAlmostEqual(first["tag_difference_separate_norm"], second["tag_difference_separate_norm"], places=5)
        self.assertAlmostEqual(first["joint_clip_scale"], second["joint_clip_scale"], places=5)

    def test_clip_scale_matches_pytorch_without_writing_parameters(self):
        parameter = torch.nn.Parameter(torch.zeros(10))
        parameter.grad = torch.ones(10)
        pre_norm = float(parameter.grad.detach().norm())
        data = parameter.detach().clone()
        clip_grad_norm_([parameter], float(PPO.max_grad_norm))
        applied = float(parameter.grad.detach().norm()) / pre_norm
        self.assertAlmostEqual(applied, clip_scale(pre_norm), places=6)
        self.assertTrue(torch.equal(parameter.detach(), data))
        self.assertIsNone(_cosine(torch.zeros(4), torch.ones(4)))
        self.assertEqual(clip_scale(0.0), 1.0)

    def test_float32_and_slice_localization(self):
        reward = 1.8565797313631995
        stored = float(np.float32(reward))
        self.assertLessEqual(abs(stored - float(np.float32(stored))), float(FLOAT32_ABS))
        saved = [{"step": 0, "reward": 1.0, "base_reward": 1.0, "bonus_age": 0.0, "bonus_tag": 0.0, "motion_penalty": 0.0, "served_bits": 10.0, "served_tagged_bits": 0.0, "lookahead_input": [0.0], "lookahead_true": [0.0], "reward_float32": float(np.float32(1.0))}]
        same = match_saved_slice(saved, saved)
        self.assertTrue(same["ok"])
        changed = [dict(saved[0], served_bits=20.0)]
        mismatch = match_saved_slice(changed, saved)
        self.assertFalse(mismatch["ok"])
        self.assertTrue(mismatch["localized"])
        self.assertEqual(mismatch["first"]["field"], "served_bits")

    def test_optimizer_step_guard(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.grad = torch.ones_like(parameter)
        originals = install_update_guard()
        try:
            with self.assertRaises(RuntimeError):
                optimizer.step()
            self.assertTrue(torch.equal(parameter.detach(), torch.zeros(3)))
        finally:
            restore_update_guard(originals)
        saved = capture_rng()
        torch.manual_seed(1)
        self.assertFalse(same_rng(saved))
        restore_rng(saved)
        self.assertTrue(same_rng(saved))

    def test_package_text_has_no_forbidden_tokens(self):
        roots = [
            ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_credit_audit",
            ROOT / "tests" / "test_priority_tag_credit_audit.py",
        ]
        texts = []
        for root in roots:
            if root.is_dir():
                texts.extend(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
            else:
                texts.append(root.read_text(encoding="utf-8"))
        package = "\n".join(path.read_text(encoding="utf-8") for path in roots[0].glob("*.py"))
        joined = "\n".join(texts)
        self.assertNotIn("deci" + "de(", joined)
        self.assertNotIn("322" + "048", joined)
        self.assertNotIn(".step(", package)
        self.assertNotIn("agent.update", package)

    def test_historical_gate_stays_fail(self):
        history = audit_history()
        self.assertTrue(history["ok"], history["gaps"])
        self.assertEqual(history["gate"]["gate"], E12_GATE)
        self.assertEqual(history["gate"]["passing_seeds"], 0)
        self.assertEqual(history["pairing"]["rows"], 320)
        self.assertTrue(history["checkpoints"]["critic_init_matches_within_seed"])
        self.assertIn("env 0", history["sample_rows"][0]["coverage_note"])


if __name__ == "__main__":
    unittest.main()
