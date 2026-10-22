"""Fault-handling checks. They do not open the holdout or rewrite checkpoints."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.faults import (
    CheckpointError,
    PolicyError,
    RollbackError,
    file_sha256,
    handle_episode_fault,
    option_is_retainable,
    rollback_to_baseline,
    verify_checkpoint_file,
)
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, POLICY_FILE, SCHEME_C_CKPT, SCHEME_C_SHA256, read_policy, write_policy
from wt.benchmark_v2.phase6iae_deployment_contract.protocol import assign_gate, load_requirements, pending_requirements, status_flags
from wt.htw_phase100.contract import TaskOption


def _copy_policy(folder: Path) -> Path:
    path = folder / "active_policy.json"
    write_policy("scheme_c", path)
    return path


class Phase6IAEContractTests(unittest.TestCase):
    def test_gate_keeps_deployment_unverified_while_limits_are_unnamed(self) -> None:
        incomplete = assign_gate(pending=11, fault_failures=0, regression_ok=True)
        blocked = assign_gate(pending=0, fault_failures=1, regression_ok=True)
        ready = assign_gate(pending=0, fault_failures=0, regression_ok=True)
        self.assertEqual(incomplete, "DEPLOYMENT_REQUIREMENTS_INCOMPLETE")
        self.assertEqual(blocked, "DEPLOYMENT_FAULT_HANDLING_BLOCKED")
        self.assertEqual(ready, "DEPLOYMENT_CONTRACT_READY")
        self.assertEqual(assign_gate(pending=0, fault_failures=0, regression_ok=False), "DEPLOYMENT_FAULT_HANDLING_BLOCKED")
        flags = status_flags(gate=incomplete, planner_ready=True)
        self.assertTrue(flags["planner_ready"])
        self.assertFalse(flags["deployment_verified"])
        self.assertFalse(status_flags(gate=ready, planner_ready=True)["deployment_verified"])

    def test_unnamed_limits_remain_explicit(self) -> None:
        document = load_requirements()
        pending = pending_requirements(document)
        self.assertGreater(len(pending), 0)
        self.assertFalse(document["deployment_verified"])
        for name in pending:
            self.assertEqual(document["requirements"][name], "NOT_YET_SPECIFIED")
        self.assertEqual(document["requirements"]["episode_abort_and_recovery"]["status"], "SPECIFIED")
        self.assertEqual(document["requirements"]["rollback"]["policy"], "g1_rate")
        self.assertEqual(document["requirements"]["rollback"]["k"], 6)
        self.assertEqual(document["observed_evidence"]["note"].split(".")[0], "Phase-6I-AD measurements on one Windows process")

    def test_missing_malformed_and_illegal_policy_files_are_rejected(self) -> None:
        official = read_policy()
        self.assertEqual(official, "scheme_c")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            missing = root / "missing.json"
            with self.assertRaises(PolicyError):
                read_policy(missing)
            self.assertFalse(missing.exists())
            broken = root / "broken.json"
            broken.write_text("{", encoding="utf-8")
            with self.assertRaises(PolicyError):
                read_policy(broken)
            illegal = root / "illegal.json"
            illegal.write_text(json.dumps({"high_level_policy": "scheme_c", "generator": "OTHER", "k": 5, "hold": 1}), encoding="utf-8")
            with self.assertRaises(PolicyError):
                read_policy(illegal)
            self.assertEqual(read_policy(), "scheme_c")

    def test_checkpoint_missing_and_hash_mismatch_stop(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / "faults.jsonl"
            missing = root / "missing.pt"
            with self.assertRaises(CheckpointError):
                file_sha256(missing, log_path=log)
            wrong = root / "wrong.pt"
            wrong.write_bytes(b"not-a-checkpoint")
            with self.assertRaises(CheckpointError):
                verify_checkpoint_file(wrong, SCHEME_C_SHA256, log_path=log)
            text = log.read_text(encoding="utf-8")
            self.assertIn("checkpoint missing", text)
            self.assertIn("checkpoint hash mismatch", text)
        self.assertEqual(verify_checkpoint_file(primary_checkpoint(), MAPPO_SHA256), MAPPO_SHA256)
        self.assertEqual(file_sha256(SCHEME_C_CKPT), SCHEME_C_SHA256)

    def test_task_option_is_kept_only_while_it_matches_the_legal_decision(self) -> None:
        assignment = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        option = TaskOption(
            targets=assignment.copy(),
            decision_time=0,
            remaining_hold=15,
            uav_mask=np.ones(6, dtype=bool),
            buoy_mask=np.ones(12, dtype=bool),
            association=assignment.copy(),
        )
        self.assertTrue(option_is_retainable(option, assignment, 15))
        option.association = np.asarray([0, 0, 2, 3, 4, 5], dtype=np.int64)
        self.assertFalse(option_is_retainable(option, assignment, 15))
        option.association = assignment.copy()
        option.remaining_hold = 14
        self.assertFalse(option_is_retainable(option, assignment, 15))

    def test_rollback_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = _copy_policy(root)
            log = root / "faults.jsonl"
            record = rollback_to_baseline(path, log_path=log)
            self.assertEqual(record["rollback"], "g1_rate")
            self.assertEqual(record["mappo_sha256"], MAPPO_SHA256)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(read_policy(path), "g1_rate")
            self.assertEqual(payload["generator"], "G1_RATE")
            self.assertEqual(int(payload["k"]), 6)
            failed = _copy_policy(root)
            import wt.benchmark_v2.phase6iaa_formal_planner.faults as faults

            original = faults.file_sha256

            def _wrong(file, log_path=None):
                if Path(file) == primary_checkpoint():
                    return "0" * 64
                return original(file, log_path=log_path)

            faults.file_sha256 = _wrong
            try:
                with self.assertRaises(RollbackError):
                    handle_episode_fault(reason="injected rollback failure", scenario_id=322000, time_index=0, policy_path=failed, log_path=log)
            finally:
                faults.file_sha256 = original
            self.assertEqual(read_policy(failed), "scheme_c")
            self.assertIn("rollback_failed", log.read_text(encoding="utf-8"))

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2"
        for folder in ("phase6iaa_formal_planner", "phase6iae_deployment_contract"):
            for path in (root / folder).glob("*.py"):
                self.assertNotIn("decide" + "(", path.read_text(encoding="utf-8"))


class Phase6IAEFaultInjectionTests(unittest.TestCase):
    planner = None
    official = ""

    @classmethod
    def setUpClass(cls) -> None:
        from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner

        cls.official = POLICY_FILE.read_text(encoding="utf-8")
        cls.planner = FormalPlanner("scheme_c")

    @classmethod
    def tearDownClass(cls) -> None:
        if POLICY_FILE.read_text(encoding="utf-8") != cls.official:
            POLICY_FILE.write_text(cls.official, encoding="utf-8")
            raise RuntimeError("official policy changed during fault injection")

    def _run(self, folder: Path, **patches):
        from wt.benchmark_v2.phase6iaa_formal_planner.planner import run_closed_episode
        import wt.benchmark_v2.phase6iaa_formal_planner.planner as planner_mod

        path = _copy_policy(folder)
        log = folder / "faults.jsonl"
        originals = {}
        if "keys" in patches:
            originals["keys"] = planner_mod.frozen_candidate_keys
            planner_mod.frozen_candidate_keys = patches["keys"]
        if "infer" in patches:
            originals["infer"] = self.planner.scorer.infer
            self.planner.scorer.infer = patches["infer"]
        if "move" in patches:
            originals["move"] = planner_mod._actor_move
            planner_mod._actor_move = patches["move"]
        if "accounting" in patches:
            originals["accounting"] = L3BBitQueueEnvV2.conservation_error_bits
            L3BBitQueueEnvV2.conservation_error_bits = patches["accounting"]
        try:
            row = run_closed_episode(322000, self.planner, stop_time=15, policy_path=path, fault_log=log)
        finally:
            if "keys" in originals:
                planner_mod.frozen_candidate_keys = originals["keys"]
            if "infer" in originals:
                self.planner.scorer.infer = originals["infer"]
            if "move" in originals:
                planner_mod._actor_move = originals["move"]
            if "accounting" in originals:
                L3BBitQueueEnvV2.conservation_error_bits = originals["accounting"]
        return row, path, log

    def test_empty_pool_illegal_assignment_nan_and_execution_faults_stop(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty, empty_policy, empty_log = self._run(root / "empty", keys=lambda env: ([], 0))
            self.assertEqual(int(empty["terminal_time"]), 0)
            self.assertIn("no legal candidate", empty["anomalies"][0]["reason"])
            self.assertEqual(read_policy(empty_policy), "g1_rate")
            self.assertIn("no legal candidate", empty_log.read_text(encoding="utf-8"))

            illegal, illegal_policy, illegal_log = self._run(root / "illegal", keys=lambda env: ([(0, 0, 1, 2, 3, 4)], 1))
            self.assertEqual(int(illegal["terminal_time"]), 0)
            self.assertIn("illegal candidate", illegal["anomalies"][0]["reason"])
            self.assertEqual(read_policy(illegal_policy), "g1_rate")
            self.assertIn("illegal candidate", illegal_log.read_text(encoding="utf-8"))

            nan_row, nan_policy, nan_log = self._run(root / "nan", infer=_nan_infer)
            self.assertEqual(int(nan_row["terminal_time"]), 0)
            self.assertIn("scheme C score was rejected", nan_row["anomalies"][0]["reason"])
            self.assertEqual(read_policy(nan_policy), "g1_rate")
            self.assertIn("scheme C score was rejected", nan_log.read_text(encoding="utf-8"))

            moved, moved_policy, moved_log = self._run(root / "move", move=_raise_value_error)
            self.assertEqual(int(moved["terminal_time"]), 0)
            self.assertIn("ValueError: INJECTED", moved["anomalies"][0]["reason"])
            self.assertGreater(int(moved["accounting_failures"]), 0)
            self.assertEqual(read_policy(moved_policy), "g1_rate")
            self.assertIn("ValueError: INJECTED", moved_log.read_text(encoding="utf-8"))

            accounted, accounted_policy, accounted_log = self._run(root / "account", accounting=_bad_conservation)
            self.assertGreater(int(accounted["accounting_failures"]), 0)
            self.assertIn("accounting", accounted["anomalies"][0]["reason"])
            self.assertLess(int(accounted["terminal_time"]), 15)
            self.assertEqual(read_policy(accounted_policy), "g1_rate")
            self.assertIn("accounting", accounted_log.read_text(encoding="utf-8"))
            self.assertEqual(POLICY_FILE.read_text(encoding="utf-8"), self.official)


class _NanScore:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores
        self.chosen_index = 0
        self.chosen_assignment = [0, 1, 2, 3, 4, 5]
        self.seconds = 0.0


def _nan_infer(_graph, rows) -> _NanScore:
    scores = np.ones(len(rows), dtype=np.float64)
    scores[0] = np.nan
    return _NanScore(scores)


def _raise_value_error(*_args, **_kwargs):
    raise ValueError("INJECTED")


def _bad_conservation(env):
    return np.full(env.config.num_buoys, 10.0)


if __name__ == "__main__":
    unittest.main()
