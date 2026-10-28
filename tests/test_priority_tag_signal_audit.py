"""Locked checks for the frozen tag-signal audit. Fixtures are not a natural scenario distribution."""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import HOLD, LOG_STD_INIT
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.reward import split_bonus
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.worker import DirectServiceWorker
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import load_formal_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.attribute import allocate_service
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
from wt.benchmark_v2.phase6i_priority_tag.ledger import TagBucket
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.counterfactual import is_opportunity
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.protocol import PERTURBATION, snapshot_indices
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.safety import _labels
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.trace import capture_worker, restore_worker, take_step


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_signal_audit"
FIXTURE = "constructed_fixture_not_a_natural_scenario"


def _actor():
    actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
    actor.eval()
    return actor


def _zero_act(worker) -> None:
    def _act(obs, priority):
        pre, log_prob = DirectServiceWorker._act(worker, obs, priority)
        return np.zeros_like(pre), log_prob

    worker._act = _act


def _place(worker) -> None:
    assignment = sample_assignment(worker.env, worker.rng)
    worker.option = make_option(worker.env, assignment)
    worker.pending = int(HOLD)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        worker.env.uav_positions[uav] = np.asarray(worker.env.buoy_positions[int(buoy)], dtype=np.float64).copy()


def _flip_reset_tags(worker) -> None:
    for buckets in worker.ledger.buckets:
        for bucket in buckets:
            if bucket.age_origin == "reset_convention":
                bucket.priority_tag = 1
    worker.ledger.assert_matches(worker.env.q_bits, float(worker.env.tol_bits))


def _open(worker):
    rollout, original = _install()
    worker.reset()
    return rollout, original


class TagSignalAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.actor = _actor()

    def test_protocol_is_locked(self) -> None:
        self.assertEqual(float(PERTURBATION), 0.1)
        self.assertEqual(snapshot_indices(48), (0, 6, 12, 18, 24, 30, 36, 42))
        baseline = {"served_tagged_bits": 0.0, "payload_delta_bits": 10.0, "safety_violations": 0, "constraint_failures": 0}
        gain = {"served_tagged_bits": 1.1, "payload_delta_bits": 9.0, "safety_violations": 0, "constraint_failures": 0}
        self.assertTrue(is_opportunity(gain, baseline, reproduced=True))
        self.assertFalse(is_opportunity(gain, baseline, reproduced=False))
        tied = dict(gain)
        tied["served_tagged_bits"] = 1.0
        self.assertFalse(is_opportunity(tied, baseline, reproduced=True))
        unsafe = dict(gain)
        unsafe["safety_violations"] = 1
        self.assertFalse(is_opportunity(unsafe, baseline, reproduced=True))

    def test_forbidden_tokens(self) -> None:
        held = ("322" + "048", "322" + "064")
        for path in PACKAGE.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("deci" + "de(", text)
            self.assertNotIn("optimizer.step", text)
            for token in held:
                self.assertNotIn(token, text)
        own = Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("deci" + "de(", own)
        for token in held:
            self.assertNotIn(token, own)

    def test_zero_ordinary_bonus_cap_and_future_are_separate_from_the_natural_pool(self) -> None:
        self.assertEqual(FIXTURE, "constructed_fixture_not_a_natural_scenario")
        age, tag = split_bonus(0.0, 0.0, 0.0)
        storage = np.zeros((1, 1), dtype=np.float32)
        storage[0, 0] = float(age + tag)
        self.assertEqual(storage.dtype, np.dtype(np.float32))
        self.assertEqual(float(storage[0, 0]), 0.0)
        parameters = inspect.signature(allocate_service).parameters
        self.assertNotIn("generated_arrival_bits", parameters)
        older = TagBucket(0, "reset", 0, 10.0, 0)
        younger = TagBucket(36, "arrival", 1, 10.0, 1)
        before = float(older.bits_remaining)
        result = allocate_service([[older, younger]], np.asarray([10.0]), 48)
        self.assertAlmostEqual(float(result["served_ordinary_bits"]), 10.0, delta=1.0e-6)
        self.assertAlmostEqual(float(result["served_tagged_bits"]), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(float(older.bits_remaining), before)

    def test_tagged_fixture_reaches_bonus_and_float32_storage(self) -> None:
        worker = DirectServiceWorker(0, 0, self.actor, None, deterministic=True, scenario_id=321000)
        rollout, original = _open(worker)
        try:
            _place(worker)
            _flip_reset_tags(worker)
            _zero_act(worker)
            record = worker.step()
        finally:
            _restore(rollout, original)
        self.assertGreater(float(record["served_bits"]), 0.0)
        self.assertAlmostEqual(float(record["served_tagged_bits"]), float(record["served_bits"]), delta=1.0e-6)
        self.assertGreater(float(record["bonus_tag"]), 0.0)
        payload = float(record["served_bits"]) / 1.0e6
        self.assertLessEqual(float(record["bonus_tag"]), 0.00075 * payload + 1.0e-12)
        self.assertLessEqual(float(record["bonus_age"]), 0.00025 * payload + 1.0e-12)
        self.assertAlmostEqual(float(record["reward"]), float(record["base_reward"]) + float(record["bonus_age"]) + float(record["bonus_tag"]))
        storage = np.zeros((1, 1), dtype=np.float32)
        storage[0, 0] = float(record["reward"])
        self.assertEqual(float(storage[0, 0]), float(np.float32(record["reward"])))

    def test_ordinary_service_has_no_tag_bonus(self) -> None:
        worker = DirectServiceWorker(0, 0, self.actor, None, deterministic=True, scenario_id=321000)
        rollout, original = _open(worker)
        try:
            _place(worker)
            _zero_act(worker)
            record = worker.step()
        finally:
            _restore(rollout, original)
        self.assertGreater(float(record["served_bits"]), 0.0)
        self.assertEqual(float(record["served_tagged_bits"]), 0.0)
        self.assertEqual(float(record["bonus_tag"]), 0.0)
        self.assertGreater(float(record["served_ordinary_bits"]), 0.0)

    def test_disabled_bonus_matches_base_reward(self) -> None:
        worker = DirectServiceWorker(0, 0, self.actor, None, deterministic=True, scenario_id=321000, use_bonus=False)
        rollout, original = _open(worker)
        try:
            _place(worker)
            _flip_reset_tags(worker)
            _zero_act(worker)
            record = worker.step()
        finally:
            _restore(rollout, original)
        self.assertGreater(float(record["served_tagged_bits"]), 0.0)
        self.assertEqual(float(record["bonus_tag"]), 0.0)
        self.assertEqual(float(record["reward"]), float(record["base_reward"]))

    def test_terminal_reset_keeps_attribution(self) -> None:
        worker = DirectServiceWorker(0, 0, self.actor, None, deterministic=True)
        rollout, original = _open(worker)
        try:
            worker.env.t = int(worker.env.config.episode_horizon) - 1
            record = worker.step()
        finally:
            _restore(rollout, original)
        self.assertIn("bonus_tag", record)
        self.assertGreaterEqual(float(record["bonus_tag"]), 0.0)

    def test_zero_offset_matches_the_worker_step(self) -> None:
        worker = DirectServiceWorker(0, 0, self.actor, None, deterministic=True, scenario_id=321000)
        rollout, original = _open(worker)
        try:
            snap = capture_worker(worker)
            taken = take_step(worker, np.zeros(2, dtype=np.float64))
            restore_worker(worker, snap)
            stepped = worker.step()
        finally:
            _restore(rollout, original)
        self.assertEqual(float(taken["served_tagged_bits"]), float(stepped["served_tagged_bits"]))
        self.assertEqual(float(taken["payload"]), float(stepped["payload"]))
        self.assertLessEqual(float(np.max(np.abs(taken["action"] - stepped["action"]))), 1.0e-8)

    def test_labels_keep_multiple_evidence_classes(self) -> None:
        step = {
            "distance_after_m": 0.999999,
            "counter_threshold_m": 1.0,
            "predicted_distance_m": 1.0,
            "projector_threshold_m": 1.0,
            "projection_status": "ok",
            "constraint_failure": 0,
            "action": np.zeros((6, 2)),
            "positions_after": np.ones((6, 2)),
            "safety_violation": 1,
        }
        labels = _labels(step, None)
        self.assertIn("accepted_projection_then_realized_violation", labels)
        self.assertIn("predicted_safe_realized_violation", labels)
        self.assertGreater(len(labels), 1)


if __name__ == "__main__":
    unittest.main()
