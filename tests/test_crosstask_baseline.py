"""Guards, seed derivation, and a short dev-pool rollout. The 32-scenario run is separate."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6iaa_formal_planner.planner import _pool_hash
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import fly_action
from wt.benchmark_v2.phase6i_urgency_stage_b.analyze import choose_gate, paired_contrast
from wt.benchmark_v2.phase6i_urgency_stage_b.conditions import training_pool_decision
from wt.benchmark_v2.phase6i_urgency_stage_b.protocol import (
    dev_scenario_ids,
    pool_hash,
    replication_rng,
    require_dev_scenario,
)
from wt.benchmark_v2.phase6i_urgency_stage_b.rollout import environment_match, run_episode
from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info


class StageBGuardTest(unittest.TestCase):
    def test_dev_pool_is_the_phase6g_registry_and_refuses_other_splits(self):
        chosen = dev_scenario_ids()
        self.assertEqual(chosen, tuple(range(321096, 321128)))
        self.assertEqual(require_dev_scenario(321096), 321096)
        for scenario_id in (321000, 320100, 320000, 310100, 322000, 322064):
            with self.assertRaises(RuntimeError):
                require_dev_scenario(scenario_id)

    def test_replication_rng_depends_on_scenario_condition_and_seed(self):
        left = replication_rng(321096, 3, 0)
        again = replication_rng(321096, 3, 0)
        self.assertEqual(int(left.integers(0, 100000)), int(again.integers(0, 100000)))
        self.assertNotEqual(
            int(replication_rng(321096, 3, 0).integers(0, 100000)),
            int(replication_rng(321096, 3, 1).integers(0, 100000)),
        )
        self.assertNotEqual(
            int(replication_rng(321096, 3, 0).integers(0, 100000)),
            int(replication_rng(321096, 4, 0).integers(0, 100000)),
        )

    def test_pool_hash_matches_the_formal_planner(self):
        keys = [(0, 1, 2, 3, 4, 5), (-1, 0, 1, 2, 3, 4)]
        self.assertEqual(pool_hash(keys), _pool_hash(keys))

    def test_sources_do_not_bypass_guards_or_hardcode_the_pool_size(self):
        root = Path(__file__).resolve().parents[1]
        package = root / "src" / "wt" / "benchmark_v2" / "phase6i_urgency_stage_b"
        text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
        self.assertNotIn("allow_holdout", text)
        self.assertNotIn("13327", text)
        self.assertNotIn("de" + "cide(", text)
        self.assertNotIn("remaining_arrivals", text)

    def test_gate_follows_completion_and_the_primary_payload_interval(self):
        self.assertEqual(
            choose_gate(expected_episodes=256, complete_episodes=256, failures=0, log_mismatches=0, primary_payload_ci_excludes_zero=True),
            "CROSSTASK_BASELINE_COMPLETE",
        )
        self.assertEqual(
            choose_gate(expected_episodes=256, complete_episodes=256, failures=0, log_mismatches=0, primary_payload_ci_excludes_zero=False),
            "CROSSTASK_BASELINE_INCONCLUSIVE",
        )
        self.assertEqual(
            choose_gate(expected_episodes=256, complete_episodes=255, failures=1, log_mismatches=0, primary_payload_ci_excludes_zero=True),
            "CROSSTASK_BASELINE_BLOCKED",
        )

    def test_paired_payload_interval_uses_the_one_bit_tie(self):
        level = {}
        for scenario_id in range(321096, 321128):
            level[(scenario_id, "B")] = {"payload_bits": 10.0}
            level[(scenario_id, "A")] = {"payload_bits": 0.0}
        summary = paired_contrast(level, "B", "A", "payload_bits", tuple(range(321096, 321128)), np.random.default_rng(0))
        self.assertEqual(summary["scenarios"], 32)
        self.assertEqual(summary["positive_scenarios"], 32)
        self.assertEqual(summary["tie_scenarios"], 0)
        self.assertTrue(summary["ci_excludes_zero"])
        self.assertAlmostEqual(summary["mean"], 10.0)


class StageBRolloutTest(unittest.TestCase):
    def test_recording_does_not_change_a_short_dev_episode(self):
        scenario_id = 321096
        assignment = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)

        def decision(_env, _position_prev, _predecessor):
            return assignment, {
                "candidate_pool_hash": "unit",
                "candidate_index": 0,
                "candidate_count": 1,
                "unique_candidate_count": 1,
                "scheme_c_score": "",
                "scheme_c_called": 0,
            }

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "steps.csv"
            import csv

            from wt.benchmark_v2.phase6i_urgency_stage_b.rollout import ROW_FIELDS

            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore", lineterminator="\n")
                writer.writeheader()
                logged = run_episode(
                    scenario_id,
                    condition_id="C",
                    replication_seed=0,
                    movement=fly_action,
                    decision=decision,
                    record=True,
                    writer=writer,
                    max_steps=4,
                )
            silent = run_episode(
                scenario_id,
                condition_id="C",
                replication_seed=0,
                movement=fly_action,
                decision=decision,
                record=False,
                max_steps=4,
            )
        self.assertTrue(environment_match(logged, silent))
        self.assertEqual(logged.status, "ok")
        self.assertEqual(logged.final_t, 4)
        self.assertLessEqual(logged.max_ledger_mismatch_bits, 1.0)
        self.assertGreater(logged.rows_written, 0)
        self.assertEqual(silent.rows_written, 0)

    def test_k5_samples_are_legal_and_repeat_with_the_same_seed(self):
        env = L3BBitQueueEnvV2()
        install_light_info(env)
        env.reset(seed=321096)
        option_holder = {}

        def sample(seed: int):
            rng = replication_rng(321096, 3, seed)
            chosen, meta = training_pool_decision(rng)(env, None, None)
            option_holder["meta"] = meta
            make_option(env, np.asarray(chosen, dtype=np.int64))
            return tuple(int(item) for item in np.asarray(chosen).tolist()), int(meta["candidate_count"])

        left, count = sample(0)
        right, _count = sample(0)
        other, _other_count = sample(1)
        self.assertEqual(left, right)
        self.assertNotEqual(left, other)
        self.assertGreater(count, 1)
        self.assertEqual(option_holder["meta"]["scheme_c_called"], 0)
        self.assertEqual(option_holder["meta"]["scheme_c_score"], "")


if __name__ == "__main__":
    unittest.main()
