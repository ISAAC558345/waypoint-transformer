"""Stage C checks. The short scheme C rollout is the only test that loads the planner."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, low_level_reward, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.protocol import AGE_SCALE_STEPS
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import ScoreTap, fly_action
from wt.benchmark_v2.phase6i_urgency_stage_b.conditions import scheme_c_decision
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.observation import TASK_DIM, build_actor_observation
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor
from wt.round30.ppo import PPOConfig

from wt.benchmark_v2.phase6i_urgency_lowlevel.models import (
    UrgencyActor,
    assert_old_checkpoint_still_loads,
    load_checkpoint,
    save_checkpoint,
    warm_start_from_frozen,
)
from wt.benchmark_v2.phase6i_urgency_lowlevel.observation import build_urgency_actor_observation, build_urgency_critic_state
from wt.benchmark_v2.phase6i_urgency_lowlevel.protocol import ACTOR_IN_DIM, CRITIC_IN_DIM, LAMBDA_URGENCY, MODEL_SEEDS_STAGE_C
from wt.benchmark_v2.phase6i_urgency_lowlevel.reward import l1_reward, l2_reward
from wt.benchmark_v2.phase6i_urgency_lowlevel.train import train_variant
from wt.benchmark_v2.phase6i_urgency_lowlevel.worker import SchemeCWorker


FORBIDDEN = ("deci" + "de(", "remaining_arrivals", "arrivals_future", "future_trajectory")


class UrgencyLowLevelTests(unittest.TestCase):
    def test_package_has_no_future_leak_or_decision_helper(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "benchmark_v2" / "phase6i_urgency_lowlevel"
        texts = [path.read_text(encoding="utf-8") for path in root.glob("*.py")]
        self.assertTrue(texts)
        for text in texts:
            for token in FORBIDDEN:
                self.assertNotIn(token, text)

    def test_dimensions_and_zero_column_warm_start(self) -> None:
        self.assertEqual(int(TASK_DIM), 8)
        old = assert_old_checkpoint_still_loads(primary_checkpoint())
        self.assertEqual(int(old.body[0].in_features), 34)
        actor = UrgencyActor(log_std_init=-2.2)
        critic = CentralizedCritic(int(CRITIC_IN_DIM))
        report = warm_start_from_frozen(actor, critic, primary_checkpoint())
        actions = {row["name"]: row["action"] for row in report}
        self.assertEqual(actions["body.0.weight"], "copied_with_inserted_zero_columns")
        self.assertEqual(actions["net.0.weight"], "copied_prefix_new_columns_zero")
        self.assertEqual(int(actor.body[0].in_features), int(ACTOR_IN_DIM))
        self.assertTrue(torch.all(actor.body[0].weight[:, 13:15] == 0).item())
        self.assertTrue(torch.all(critic.net[0].weight[:, 188:] == 0).item())
        env = _reset(321000)
        option = make_option(env, np.arange(6, dtype=np.int64))
        formal = build_actor_observation(env, option, None)
        self.assertEqual(int(formal["task"].shape[-1]), 8)
        view = _view(env)
        urgent = build_urgency_actor_observation(env, option, view, None)
        self.assertEqual(tuple(urgent["task"].shape), (6, 10))
        self.assertTrue(np.allclose(urgent["task"][:, :8], formal["task"]))
        packed = _pack(old, formal)
        packed_new = _pack(actor, urgent)
        with torch.no_grad():
            mu_old, _ = old(*packed)
            mu_new, _ = actor(*packed_new)
        self.assertTrue(torch.allclose(mu_old, mu_new, atol=1e-6))

    def test_urgency_maps_to_the_current_target(self) -> None:
        env = _reset(321000)
        ledger = AdmissionAgeLedger(int(env.config.num_buoys), AGE_SCALE_STEPS)
        ledger.reset_from_queue(env.q_bits)
        view = ledger.view(0, env.q_bits, float(env.tol_bits))
        assignment = np.array([0, -1, 1, 2, 3, 4], dtype=np.int64)
        option = make_option(env, assignment)
        obs = build_urgency_actor_observation(env, option, view, None)
        critic = build_urgency_critic_state(env, option, view, None)
        self.assertEqual(int(critic["vector"].size), int(expected_critic_dim(6, 12)) + 12)
        self.assertEqual(float(obs["task"][0, 8]), 0.0)
        self.assertEqual(float(obs["task"][0, 9]), 1.0)
        self.assertEqual(float(obs["task"][1, 8]), 0.0)
        self.assertEqual(float(obs["task"][1, 9]), 0.0)
        self.assertEqual(float(critic["vector"][188]), 0.0)
        self.assertEqual(float(critic["vector"][189]), 1.0)
        self.assertEqual(float(critic["vector"][190]), 0.0)
        self.assertEqual(float(critic["vector"][191]), 0.0)
        aged = dict(view)
        aged["urgency"] = np.asarray(view["urgency"], dtype=np.float64).copy()
        aged["urgency"][0] = 0.25
        aged["urgency_valid"] = np.asarray(view["urgency_valid"], dtype=np.int64).copy()
        mapped = build_urgency_actor_observation(env, option, aged, None)
        self.assertEqual(float(mapped["task"][0, 8]), 0.25)
        self.assertEqual(float(mapped["task"][2, 8]), 0.0)

    def test_reward_checkpoint_and_conservation(self) -> None:
        served = np.array([2.0e6, 0.0, 1.0e6] + [0.0] * 9, dtype=np.float64)
        urgency = np.linspace(0.0, 1.0, 12)
        executed = np.zeros((6, 2))
        desired = np.zeros((6, 2))
        delta = np.zeros((6, 2))
        formal, communication = low_level_reward(float(np.sum(served)), executed, desired, delta, 1.0e6)
        first = l1_reward(served, executed, desired, delta, 1.0e6)
        second = l2_reward(served, urgency, executed, desired, delta, 1.0e6)
        self.assertAlmostEqual(first["reward"], formal)
        self.assertAlmostEqual(second["reward"], formal + float(LAMBDA_URGENCY) * float(np.sum(urgency * served)) / 1.0e6)
        self.assertEqual(first["urgency_reward"], 0.0)
        self.assertAlmostEqual(second["communication_reward"], communication)
        env = _reset(321000)
        ledger = AdmissionAgeLedger(int(env.config.num_buoys), AGE_SCALE_STEPS)
        ledger.reset_from_queue(env.q_bits)
        option = make_option(env, np.arange(6, dtype=np.int64))
        last = np.zeros((6, 2))
        for _ in range(4):
            before = int(env.t)
            queue_before = np.asarray(env.q_bits, dtype=np.float64).copy()
            snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
            action = fly_action(env, option, last)
            info = env.step(action.executed, option.association)
            ledger.apply_transition(
                info["served_bits"],
                info["generated_arrival_bits"],
                info["admitted_arrival_bits"],
                info["dropped_bits_step"],
                snap_vector(env, snap_before, info),
                before,
                float(env.tol_bits),
            )
            gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
            self.assertLessEqual(float(np.max(np.abs(gap))), float(env.tol_bits))
            self.assertLessEqual(float(np.max(np.abs(env.conservation_error_bits()))), float(env.tol_bits))
            last = action.executed
        actor = UrgencyActor(log_std_init=-2.2)
        critic = CentralizedCritic(200)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            save_checkpoint(path, actor, critic, {"variant": "L1", "model_seed": 0, "updates": 1})
            loaded, loaded_critic, payload = load_checkpoint(path)
            self.assertEqual(int(payload["actor_in_dim"]), 36)
            self.assertTrue(torch.equal(loaded.body[0].weight, actor.body[0].weight))
            self.assertTrue(torch.equal(loaded_critic.net[0].weight, critic.net[0].weight))
            blob = torch.load(path, map_location="cpu", weights_only=False)
            blob["actor_in_dim"] = 34
            bad = Path(directory) / "bad.pt"
            torch.save(blob, bad)
            with self.assertRaises(RuntimeError):
                load_checkpoint(bad)

    def test_scheme_c_smoke_update(self) -> None:
        self.assertEqual(MODEL_SEEDS_STAGE_C, (0, 1, 2))
        planner = FormalPlanner("scheme_c")
        tap = ScoreTap(planner)
        actor = UrgencyActor(log_std_init=-2.2)
        critic = CentralizedCritic(200)
        warm_start_from_frozen(actor, critic, primary_checkpoint())
        worker = SchemeCWorker(0, 0, actor, critic, planner, tap, "L1")
        worker.reset()
        first = worker.step()
        self.assertEqual(int(first["obs"]["task"].shape[-1]), 10)
        self.assertEqual(int(first["global_state"].shape[0]), 200)
        self.assertEqual(int(first["decision"]["scheme_c_called"]), 1)
        self.assertGreater(int(first["decision"]["candidate_count"]), 0)
        second = worker.step()
        self.assertIsNone(second["decision"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "L1_seed0.pt"
            result = train_variant(
                "L1",
                0,
                updates=1,
                ppo=PPOConfig(rollout_length=4, num_envs=1, num_minibatches=4, update_epochs=1, checkpoint_interval_steps=4),
                checkpoint_path=path,
            )
            self.assertEqual(result["updates"], 1)
            self.assertTrue(np.isfinite(result["history"][0]["gradient_norm_preclip"]))
            self.assertGreater(result["history"][0]["scheme_c_calls"], 0)
            loaded, _critic, payload = load_checkpoint(path)
            self.assertEqual(payload["variant"], "L1")
            self.assertEqual(int(loaded.body[0].in_features), 36)
        self.assertEqual(int(ResidualActor().body[0].in_features), 34)


def _reset(scenario: int) -> L3BBitQueueEnvV2:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    return env


def _view(env: L3BBitQueueEnvV2) -> dict:
    ledger = AdmissionAgeLedger(int(env.config.num_buoys), AGE_SCALE_STEPS)
    ledger.reset_from_queue(env.q_bits)
    return ledger.view(int(env.t), env.q_bits, float(env.tol_bits))


def _pack(module, obs: dict):
    args = (
        torch.as_tensor(obs["self"], dtype=torch.float32),
        torch.as_tensor(obs["task"], dtype=torch.float32),
        torch.as_tensor(obs["communication"], dtype=torch.float32),
        torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
        torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
    )
    if isinstance(module, ResidualActor):
        return args
    return args


if __name__ == "__main__":
    unittest.main()
