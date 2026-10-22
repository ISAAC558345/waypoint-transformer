"""Phase-11 engineering tests. Does not run Phase-10.6 untouched seeds."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import numpy as np

from wt.final.constants import DEFAULT_DEMO_SEED, EXPECTED_HIGH_SHA, HOLD, IDLE_INDEX, UNTOUCHED_SEEDS
from wt.final.pipeline import load_final_model, refuse_untouched, run_final_hier
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.l3b.env import L3BEnv, conflict_mask
from wt.l3b.policies import fly_toward


ROOT = Path(__file__).resolve().parents[1]


class Phase11Tests(unittest.TestCase):
    def test_01_checkpoint_sha(self) -> None:
        model, digest = load_final_model()
        self.assertEqual(digest, EXPECTED_HIGH_SHA)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_02_demo_runs_and_shapes(self) -> None:
        result = run_final_hier(DEFAULT_DEMO_SEED, gated=True)
        self.assertEqual(result["n_steps"], 48)
        self.assertEqual(len(result["assignment"]), 6)
        self.assertEqual(np.asarray(result["uav_movement"]).shape[-1], 2)
        self.assertTrue(np.all(np.isfinite(result["payload"])))
        self.assertFalse(result["untouched_used"])

    def test_03_assignment_legal_and_idle_token(self) -> None:
        result = run_final_hier(DEFAULT_DEMO_SEED, gated=True)
        for assign in result["assignments"]:
            arr = np.asarray(assign, dtype=np.int64)
            self.assertFalse(bool(np.any(conflict_mask(arr, 12))))
            self.assertTrue(np.all((arr >= -1) & (arr < 12)))
        self.assertEqual(IDLE_INDEX, 12)
        self.assertEqual(result["task_option"]["idle_index"], 12)
        self.assertEqual(result["task_option"]["idle_env_code"], -1)

    def test_04_hold_and_decoder_interface(self) -> None:
        result = run_final_hier(DEFAULT_DEMO_SEED, gated=True)
        self.assertEqual(HOLD, 15)
        self.assertEqual(result["task_option"]["hold"], 15)
        model, _ = load_final_model()
        self.assertEqual(model.encoder_name, "deepsets")
        self.assertTrue(hasattr(model, "decoder"))

    def test_05_gate_and_fly_toward(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        assignment = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        move = fly_toward(env, assignment)
        self.assertEqual(move.shape, (6, 2))
        from wt.htw_phase100.contract import TaskOption

        option = TaskOption(assignment, 0, 15, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        raw = RuleExecutorAdapter().act(env, option)
        np.testing.assert_allclose(raw, fly_toward(env, assignment))
        gated, flags = InRangeZeroMotionGate().apply(env, option, raw)
        self.assertEqual(gated.shape, (6, 2))
        self.assertEqual(flags.dtype, bool)

    def test_06_gate_off_matches_rule_executor(self) -> None:
        on = run_final_hier(DEFAULT_DEMO_SEED, gated=True)
        off = run_final_hier(DEFAULT_DEMO_SEED, gated=False)
        self.assertNotEqual(on["payload"], off["payload"])
        env = L3BEnv()
        env.reset(seed=8)
        assignment = np.full(6, -1, dtype=np.int64)
        from wt.htw_phase100.contract import TaskOption

        option = TaskOption(assignment, 0, 15, np.ones(6, dtype=bool), np.ones(12, dtype=bool), assignment.copy())
        raw = RuleExecutorAdapter().act(env, option)
        gated, flags = InRangeZeroMotionGate().apply(env, option, raw)
        np.testing.assert_array_equal(gated, raw)
        self.assertFalse(bool(np.any(flags)))

    def test_07_plot_script_does_not_import_env(self) -> None:
        source = (ROOT / "scripts" / "plot_final_results.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any("l3b.env" in name or name.endswith("env") and "l3b" in name for name in imported))
        self.assertNotIn("wt.l3b.env", source)
        self.assertNotIn("L3BEnv", source)

    def test_08_demo_refuses_untouched(self) -> None:
        self.assertIn(106000, UNTOUCHED_SEEDS)
        with self.assertRaises(RuntimeError):
            refuse_untouched(106000)
        with self.assertRaises(RuntimeError):
            run_final_hier(106000)

    def test_09_config_matches_phase106(self) -> None:
        from wt.l3b.config import L3BConfig

        cfg = L3BConfig()
        self.assertEqual(cfg.num_uavs, 6)
        self.assertEqual(cfg.num_buoys, 12)
        self.assertEqual(cfg.episode_horizon, 48)
        self.assertEqual(HOLD, 15)
        self.assertEqual(cfg.option_hold, 15)
        self.assertAlmostEqual(cfg.service_radius_m, 16.0)
        arch = json.loads((ROOT / "src" / "wt" / "htw_phase105" / "outputs" / "benchmark" / "phase105_final_architecture_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(arch["environment"]["M"], 6)
        self.assertEqual(arch["high_level"]["file_sha256"], EXPECTED_HIGH_SHA)

    def test_10_order_free_decoder_forward(self) -> None:
        model, _ = load_final_model()
        self.assertEqual(model.decoder.__class__.__name__, "OrderFreeSetDecoder")
        self.assertTrue(callable(model.decoder))

    def test_11_hold_trace_uses_15(self) -> None:
        result = run_final_hier(DEFAULT_DEMO_SEED, gated=True)
        trace = result["task_option"]["remaining_hold_trace"]
        self.assertEqual(max(trace), 15)
        self.assertEqual(min(trace), 1)

    def test_12_fly_toward_nonzero_when_out_of_range(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        assignment = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        move = fly_toward(env, assignment)
        self.assertTrue(np.any(np.abs(move) > 0.0))

    def test_13_plot_script_is_postprocess_only(self) -> None:
        source = (ROOT / "scripts" / "plot_final_results.py").read_text(encoding="utf-8")
        self.assertNotIn("env.step", source)
        self.assertNotIn("run_final_hier", source)
        self.assertIn("phase106_final_results.csv", source)

    def test_14_default_demo_seed_is_not_untouched(self) -> None:
        self.assertNotIn(DEFAULT_DEMO_SEED, UNTOUCHED_SEEDS)
        self.assertEqual(DEFAULT_DEMO_SEED, 8)

    def test_15_no_nan_inf_and_packed_config(self) -> None:
        result = run_final_hier(DEFAULT_DEMO_SEED, gated=True)
        self.assertTrue(np.isfinite(result["payload"]))
        self.assertTrue(np.isfinite(result["flight_distance"]))
        self.assertTrue(np.all(np.isfinite(np.asarray(result["queue"]))))
        cfg_path = ROOT / "final_artifacts" / "configs" / "environment.json"
        self.assertTrue(cfg_path.is_file())
        dumped = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(dumped["M"], 6)
        self.assertEqual(dumped["K"], 12)
        self.assertEqual(dumped["T"], 48)
        self.assertEqual(dumped["hold"], 15)
        freeze = json.loads((ROOT / "FINAL_CODE_FREEZE.json").read_text(encoding="utf-8"))
        self.assertTrue(freeze["architecture_finalized"])
        self.assertTrue(freeze["code_development_complete"])
        self.assertEqual(freeze["checkpoint_sha"], EXPECTED_HIGH_SHA)


if __name__ == "__main__":
    unittest.main()
