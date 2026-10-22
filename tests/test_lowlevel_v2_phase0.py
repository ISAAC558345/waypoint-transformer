"""LL-V2.0 engineering tests. No training. No Phase-10.6 untouched eval."""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.l3b.env import L3BEnv, active_links, sinr_rates_for_links
from wt.lowlevel_v2.base_executor import BaseExecutor
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, env_target
from wt.lowlevel_v2.critic_state import build_critic_state, expected_critic_dim
from wt.lowlevel_v2.data_registry import build_registry, refuse_untouched
from wt.lowlevel_v2.metrics import collision_diagnostics
from wt.lowlevel_v2.neighbor_encoder import NeighborSetEncoder, permutation_invariance_error
from wt.lowlevel_v2.observation import build_actor_observation, neighbor_edges
from wt.lowlevel_v2.protocol import D_SAFE_PLACEHOLDER_M, EXPECTED_HIGH_SHA, FINAL_CKPT, HOLD_STEPS, PHASE106_UNTOUCHED, ROOT
from wt.lowlevel_v2.residual_action import residual_desired
from wt.lowlevel_v2.residual_policy import ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector
from wt.lowlevel_v2.service_safe_envelope import envelope_active, next_in_envelope


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _option(env: L3BEnv, assoc) -> TaskOption:
    assoc = np.asarray(assoc, dtype=np.int64)
    return TaskOption(assoc, int(env.t), HOLD_STEPS, np.ones(6, bool), np.ones(12, bool), assoc.copy())


class LowLevelV2Phase0Tests(unittest.TestCase):
    def test_01_final_sources_unmodified_and_sha(self) -> None:
        self.assertEqual(_sha(FINAL_CKPT), EXPECTED_HIGH_SHA)
        pipeline = ROOT / "src" / "wt" / "final" / "pipeline.py"
        self.assertTrue(pipeline.is_file())
        self.assertIn("InRangeZeroMotionGate", pipeline.read_text(encoding="utf-8"))

    def test_02_checkpoint_sha_frozen(self) -> None:
        self.assertEqual(_sha(FINAL_CKPT), "dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109")

    def test_03_untouched_refused(self) -> None:
        self.assertIn(106000, PHASE106_UNTOUCHED)
        with self.assertRaises(RuntimeError):
            refuse_untouched(106000)

    def test_04_task_option_matches_final_contract(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.array([0, 1, 2, 3, 4, -1]))
        self.assertEqual(option.remaining_hold, 15)
        np.testing.assert_array_equal(option.association, option.targets)
        raw = RuleExecutorAdapter().act(env, option)
        gated, _ = InRangeZeroMotionGate().apply(env, option, raw)
        base, flags = BaseExecutor(gated=True).act(env, option)
        np.testing.assert_allclose(base, gated)
        self.assertEqual(flags.dtype, bool)

    def test_05_idle_does_not_index_last_buoy(self) -> None:
        self.assertEqual(env_target(-1, 12), -1)
        self.assertEqual(env_target(12, 12), -1)
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.full(6, -1))
        obs = build_actor_observation(env, option)
        self.assertTrue(np.allclose(obs["task"][:, 0], 0.0))
        self.assertTrue(np.allclose(obs["task"][:, 1:6], 0.0))

    def test_06_neighbor_permutation_invariance(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.array([0, 1, 2, 3, 4, 5]))
        edges = torch.as_tensor(neighbor_edges(env, option, np.zeros((6, 2)), 0), dtype=torch.float32)
        err = permutation_invariance_error(NeighborSetEncoder(), edges, torch.randperm(edges.shape[0]))
        self.assertLess(err, 1.0e-5)

    def test_07_zero_residual_equals_base(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.array([0, 1, 2, 3, 4, -1]))
        a_base, _ = BaseExecutor(gated=True).act(env, option)
        a_des = residual_desired(a_base, np.zeros_like(a_base))
        np.testing.assert_allclose(a_des, clip_unit_step(a_base))

    def test_08_collision_distance(self) -> None:
        pos = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 0.0]])
        cfg = CollisionConfig(0.4, 0.2, True)
        self.assertAlmostEqual(cfg.d_safe, D_SAFE_PLACEHOLDER_M)
        stats = collision_diagnostics(pos, cfg)
        self.assertEqual(stats["collision_count"], 1)
        self.assertGreaterEqual(stats["near_collision_count"], 0)
        self.assertAlmostEqual(stats["minimum_pairwise_distance"], 0.5)

    def test_09_projector_finite(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.array([0, 1, 2, 3, 4, 5]))
        a_base, _ = BaseExecutor(gated=True).act(env, option)
        out = SafetyProjector().project(env, a_base, option)
        self.assertTrue(out.finite)
        self.assertTrue(np.all(np.isfinite(out.projected_actions)))

    def test_10_projector_enforces_d_safe_when_feasible(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.uav_positions[:] = np.array([[10.0, 10.0], [12.0, 10.0], [40.0, 10.0], [42.0, 10.0], [70.0, 10.0], [90.0, 10.0]])
        option = _option(env, np.full(6, -1))
        desired = np.zeros((6, 2))
        desired[0] = np.array([1.0, 0.0])
        desired[1] = np.array([-1.0, 0.0])
        out = SafetyProjector().project(env, desired, option)
        if out.constraint_status == "ok":
            self.assertGreaterEqual(out.minimum_predicted_distance, D_SAFE_PLACEHOLDER_M - 1.0e-6)
        else:
            self.assertEqual(out.constraint_status, "failure")

    def test_11_service_safe_envelope(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.uav_positions[0] = env.buoy_positions[0].copy()
        env.queue[0] = 4.0
        option = _option(env, np.array([0, -1, -1, -1, -1, -1]))
        self.assertTrue(bool(envelope_active(env, option)[0]))
        stay = env.uav_positions.copy()
        self.assertTrue(bool(next_in_envelope(env, option, stay)[0]))
        far = stay.copy()
        far[0] = env.buoy_positions[0] + np.array([100.0, 0.0])
        self.assertFalse(bool(next_in_envelope(env, option, far)[0]))

    def test_12_sinr_responds_to_position(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        env.uav_positions[0] = env.buoy_positions[0].copy()
        env.uav_positions[1] = env.buoy_positions[1].copy()
        env.assignment[:] = np.array([0, 1, -1, -1, -1, -1])
        env.queue[:] = 4.0
        links = active_links(env)
        sinr0, rate0 = sinr_rates_for_links(env, links)
        env.uav_positions[1] += np.array([0.8, 0.0])
        sinr1, rate1 = sinr_rates_for_links(env, links)
        self.assertTrue(float(np.max(np.abs(sinr1 - sinr0))) > 0.0 or float(np.max(np.abs(rate1 - rate0))) > 0.0)

    def test_13_actor_schema_m6(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.array([0, 1, 2, 3, 4, 5]))
        obs = build_actor_observation(env, option)
        self.assertEqual(obs["self"].shape, (6, 5))
        self.assertEqual(obs["task"].shape, (6, 8))
        self.assertEqual(obs["communication"].shape, (6, 4))
        self.assertEqual(obs["neighbor_set"].shape, (6, 5, 11))
        actor = ResidualActor()
        with torch.no_grad():
            mu, log_std = actor(
                torch.as_tensor(obs["self"], dtype=torch.float32),
                torch.as_tensor(obs["task"], dtype=torch.float32),
                torch.as_tensor(obs["communication"], dtype=torch.float32),
                torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
            )
        self.assertEqual(tuple(mu.shape), (6, 2))
        self.assertEqual(tuple(log_std.shape), (6, 2))
        self.assertLess(float(torch.max(torch.abs(mu))), 1.0e-5)

    def test_14_critic_dim(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.array([0, 1, 2, 3, 4, 5]))
        state = build_critic_state(env, option)
        self.assertEqual(state["dim"], expected_critic_dim(6, 12))
        self.assertTrue(state["finite"])

    def test_15_no_future_leak_and_registry(self) -> None:
        env = L3BEnv()
        env.reset(seed=8)
        option = _option(env, np.array([0, 1, 2, 3, 4, -1]))
        obs = build_actor_observation(env, option)
        self.assertNotIn("remaining_arrivals", obs)
        self.assertFalse(obs["leaks_future_arrivals"])
        registry = build_registry()
        self.assertFalse(registry["opened_phase106_result_csv"])
        self.assertFalse(registry["overlap_train_history"])
        self.assertFalse(set(registry["LLV2_train"]) & PHASE106_UNTOUCHED)
        self.assertIsNone(registry["LLV2_confirmation"])
        self.assertIsNone(registry["LLV2_final_untouched"])


if __name__ == "__main__":
    unittest.main()
