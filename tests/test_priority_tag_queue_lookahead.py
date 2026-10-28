"""Queue-lookahead observation checks. Training stays outside this file."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag.ledger import TagBucket
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.actor import LookaheadActor, same_module, transplant_head_actor
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.criteria import choose_gate, score_l4
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.features import actor_features, queue_fractions
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.protocol import (
    ACTOR_FEATURE_COUNT,
    GATE_BLOCKED,
    GATE_FAIL,
    GATE_INCONCLUSIVE,
    GATE_PASS,
    SAMPLE_ENV_INDEX,
    SAMPLE_UPDATES,
    UPDATES,
)
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.train import build_pair
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import capture_bundle, critic_dim, restore_bundle
from wt.lowlevel_v2.observation import COMM_DIM, SELF_DIM, TASK_DIM


def _bucket(step: int, tag: int, bits: float, insert: int) -> TagBucket:
    return TagBucket(int(step), "unit", int(tag), float(bits), int(insert))


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


class QueueLookaheadTests(unittest.TestCase):
    def test_locked_widths_and_sample_rule(self) -> None:
        self.assertEqual(int(ACTOR_FEATURE_COUNT), 5)
        self.assertEqual(int(UPDATES), 300)
        self.assertEqual(tuple(SAMPLE_UPDATES), (1, 150, 300))
        self.assertEqual(int(SAMPLE_ENV_INDEX), 0)

    def test_queue_fraction_cases(self) -> None:
        self.assertEqual(queue_fractions([], 0), (0.0, 0.0))
        ordinary = [_bucket(0, 0, 40.0, 0), _bucket(1, 0, 10.0, 1)]
        self.assertEqual(queue_fractions(ordinary, 1), (0.0, 0.0))
        head = [_bucket(0, 1, 10.0, 0), _bucket(1, 0, 30.0, 1)]
        tagged_fraction, before_fraction = queue_fractions(head, 1)
        self.assertGreater(tagged_fraction, 0.0)
        self.assertEqual(before_fraction, 0.0)
        tail = [_bucket(0, 0, 30.0, 0), _bucket(1, 1, 10.0, 1)]
        tagged_fraction, before_fraction = queue_fractions(tail, 1)
        self.assertAlmostEqual(tagged_fraction, 0.25)
        self.assertAlmostEqual(before_fraction, 0.75)
        partial = [_bucket(0, 0, 5.0, 0), _bucket(1, 1, 15.0, 1)]
        tagged_fraction, before_fraction = queue_fractions(partial, 1)
        self.assertAlmostEqual(tagged_fraction, 0.75)
        self.assertAlmostEqual(before_fraction, 0.25)
        dusty = [_bucket(0, 0, float(DUST_BITS), 0), _bucket(1, 1, 8.0, 1)]
        tagged_fraction, before_fraction = queue_fractions(dusty, 1)
        self.assertEqual(before_fraction, 0.0)
        self.assertAlmostEqual(tagged_fraction, 1.0)
        leaked = [_bucket(0, 0, 10.0, 0), _bucket(5, 1, 90.0, 1)]
        self.assertEqual(queue_fractions(leaked, 1), (0.0, 0.0))

    def test_idle_assignment_and_dtype(self) -> None:
        from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger

        ledger = PriorityTagLedger(2)
        ledger.buckets[0] = [_bucket(0, 0, 10.0, 0), _bucket(1, 1, 10.0, 1)]
        features = actor_features(ledger, 1, np.asarray([-1, -1]), 2, use_lookahead=True)
        self.assertEqual(features.dtype, np.float64)
        self.assertEqual(tuple(features.shape), (2, 5))
        self.assertTrue(np.all(features == 0.0))
        features = actor_features(ledger, 1, np.asarray([0, 1]), 2, use_lookahead=False)
        self.assertTrue(np.all(features[:, 3:] == 0.0))
        self.assertGreater(features[0, 2], 0.0)

    def test_transplant_keeps_closed_columns_and_paired_init(self) -> None:
        from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
        from wt.benchmark_v2.phase6i_priority_tag_reward.actor import PriorityTagActor, load_formal_actor, transplant_legacy_actor
        from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import LEGACY_ACTOR

        head = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
        transplant_legacy_actor(load_formal_actor(LEGACY_ACTOR, float(LOG_STD_INIT)), head)
        five = LookaheadActor(log_std_init=float(LOG_STD_INIT))
        transplant_head_actor(head, five)
        weight = five.state_dict()["body.0.weight"]
        self.assertTrue(torch.equal(weight[:, -2:], torch.zeros_like(weight[:, -2:])))
        self_f = torch.zeros(6, SELF_DIM)
        task_f = torch.zeros(6, TASK_DIM)
        comm_f = torch.zeros(6, COMM_DIM)
        edges = torch.zeros(6, 5, 11)
        near = torch.zeros(6)
        priority3 = torch.zeros(6, 3)
        priority5 = torch.zeros(6, 5)
        with torch.no_grad():
            mu3, _log3 = head(self_f, task_f, comm_f, edges, near, priority3)
            mu5, _log5 = five(self_f, task_f, comm_f, edges, near, priority5)
        self.assertTrue(torch.allclose(mu3, mu5))
        pair = build_pair(0)
        self.assertTrue(same_module(pair["L4_C"]["actor"], pair["L4_Q"]["actor"]))
        self.assertTrue(same_module(pair["L4_C"]["critic"], pair["L4_Q"]["critic"]))
        self.assertEqual(int(pair["L4_C"]["critic"].state_dict()["net.0.weight"].shape[1]), int(critic_dim()))

    def test_closed_feature_matches_parent_transition(self) -> None:
        from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
        from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.worker import DirectServiceWorker
        from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.worker import QueueLookaheadWorker
        from wt.benchmark_v2.phase6i_priority_tag_reward.actor import PriorityTagActor, load_formal_actor, transplant_legacy_actor
        from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import LEGACY_ACTOR
        from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore

        head = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
        transplant_legacy_actor(load_formal_actor(LEGACY_ACTOR, float(LOG_STD_INIT)), head)
        five = LookaheadActor(log_std_init=float(LOG_STD_INIT))
        transplant_head_actor(head, five)
        head.eval()
        five.eval()
        rollout, original = _install()
        try:
            assignment = 7_000_000 + 321000
            parent = DirectServiceWorker(assignment, 0, head, None, deterministic=True, scenario_id=321000)
            child = QueueLookaheadWorker(assignment, 0, five, None, deterministic=True, scenario_id=321000, use_lookahead=False)
            parent.reset()
            child.reset()
            parent.rng = np.random.default_rng(assignment)
            child.rng = np.random.default_rng(assignment)
            parent_step = parent.step()
            child_step = child.step()
            self.assertTrue(np.allclose(parent_step["action"], child_step["action"]))
            self.assertTrue(np.allclose(parent_step["priority"], child_step["priority"][:, :3]))
            self.assertEqual(float(np.max(np.abs(child_step["lookahead_input"]))), 0.0)
            self.assertAlmostEqual(float(parent_step["payload"]), float(child_step["payload"]))
            self.assertTrue(np.allclose(parent.env.uav_positions, child.env.uav_positions))
            self.assertEqual(int(child_step["global_state"].shape[-1]), int(critic_dim()))
            snap = capture_bundle(child)
            child.step()
            restore_bundle(child, snap)
            restored = child._priority()
            self.assertEqual(restored.dtype, np.float64)
            self.assertTrue(np.all(restored[:, 3:] >= 0.0))
            self.assertTrue(np.all(restored[:, 3:] <= 1.0))
            self.assertTrue(np.all(restored[:, 3:] == 0.0))
            opened = QueueLookaheadWorker(assignment, 0, five, None, deterministic=True, scenario_id=321000, use_lookahead=True)
            opened.reset()
            opened.rng = np.random.default_rng(assignment)
            opened.step()
            before = opened._priority().copy()
            snap = capture_bundle(opened)
            opened.step()
            restore_bundle(opened, snap)
            self.assertTrue(np.allclose(opened._priority(), before))
        finally:
            _restore(rollout, original)

    def test_training_reset_keeps_admitted_history_only(self) -> None:
        from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
        from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.worker import QueueLookaheadWorker
        from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
        from wt.lowlevel_v2.residual_policy import CentralizedCritic

        actor = LookaheadActor(log_std_init=float(LOG_STD_INIT))
        critic = CentralizedCritic(critic_dim())
        actor.eval()
        critic.eval()
        rollout, original = _install()
        try:
            worker = QueueLookaheadWorker(0, 0, actor, critic, deterministic=True, scenario_id=321000, use_lookahead=True)
            worker.reset()
            horizon = int(worker.env.config.episode_horizon)
            last = None
            for _ in range(horizon):
                last = worker.step()
            self.assertIsNotNone(last)
            self.assertTrue(bool(last["done"]))
            self.assertEqual(int(worker.env.t), horizon)
            worker.reset()
            self.assertEqual(int(worker.env.t), 0)
            for buckets in worker.ledger.buckets:
                for bucket in buckets:
                    self.assertLessEqual(int(bucket.admission_step), int(worker.env.t))
        finally:
            _restore(rollout, original)

    def test_gate_order(self) -> None:
        scenarios = list(range(321096, 321128))
        rows = []
        for scenario in scenarios:
            rows.append(_row("L0", 0, scenario, 100.0, 1.0e6))
            for seed in (0, 1, 2):
                rows.append(_row("L4_C", seed, scenario, 100.0, 1.0e6))
                rows.append(_row("L4_Q", seed, scenario, 1.0e6, 1.0e6))
        score = score_l4(rows)
        self.assertEqual(int(score["passing_seeds"]), 3)
        self.assertEqual(choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, feasibility="pass", score=score), GATE_PASS)
        negative = []
        for scenario in scenarios:
            negative.append(_row("L0", 0, scenario, 1.0e6, 1.0e6))
            for seed in (0, 1, 2):
                tagged = 1.0 if seed < 2 else 2.0e6
                negative.append(_row("L4_C", seed, scenario, 1.0e6, 1.0e6))
                negative.append(_row("L4_Q", seed, scenario, tagged, 1.0e6))
        failed = score_l4(negative)
        self.assertEqual(choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, feasibility="pass", score=failed), GATE_FAIL)
        unsafe = dict(score)
        unsafe["seeds"] = [dict(item) for item in score["seeds"]]
        unsafe["seeds"][0] = dict(unsafe["seeds"][0])
        unsafe["seeds"][0]["versus_l0"] = dict(unsafe["seeds"][0]["versus_l0"])
        unsafe["seeds"][0]["versus_l0"]["safety_diff_sum"] = 1.0
        self.assertEqual(choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, feasibility="pass", score=unsafe), GATE_FAIL)
        self.assertEqual(choose_gate(tests_ok=True, train_ok=False, hashes_ok=True, feasibility="inconclusive", score=None), GATE_INCONCLUSIVE)
        self.assertEqual(choose_gate(tests_ok=False, train_ok=True, hashes_ok=True, feasibility="pass", score=score), GATE_BLOCKED)

    def test_sources_do_not_embed_reserved_identifiers(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_queue_lookahead"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        text += "\n" + Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("deci" + "de(", text)
        self.assertNotIn("322" + "048", text)
        self.assertNotIn("322" + "064", text)


if __name__ == "__main__":
    unittest.main()
