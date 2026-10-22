"""Phase-10.1 adapter tests. Does not run the 100-update pilot."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase9.const import HOLD, TRAIN_SEEDS, EVAL_SEEDS
from wt.l3b.env import L3BEnv

from wt.htw_phase101.features import attach_option, build_schema, encode_obs
from wt.htw_phase101.high_level import refuse_m5
from wt.htw_phase101.models import L3BGoalActorCritic, shape_from_schema
from wt.htw_phase101.protocol import DESIGN_TRAIN_SEEDS, DESIGN_VAL_SEEDS, FORBIDDEN_M5_CKPT, PPO
from wt.htw_phase101.rewards import auxiliary_reward


class Phase101Tests(unittest.TestCase):
    def test_01_schema_is_six_uav_not_5x21(self) -> None:
        schema = build_schema()
        self.assertEqual(schema["max_uavs"], 6)
        self.assertEqual(schema["max_buoys"], 12)
        self.assertEqual(schema["idle_index"], 12)
        self.assertNotEqual((schema["max_uavs"], schema["uav_dim"]), (5, 21))
        shape = shape_from_schema(schema)
        self.assertEqual(shape.max_uavs, 6)
        self.assertEqual(shape.uav_dim, schema["uav_dim"])

    def test_02_idle_does_not_read_last_buoy(self) -> None:
        env = L3BEnv()
        info = env.reset(seed=0)
        schema = build_schema(env)
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        option = TaskOption(
            targets=assignment,
            decision_time=0,
            remaining_hold=HOLD,
            uav_mask=np.ones(schema["max_uavs"], dtype=bool),
            buoy_mask=np.ones(schema["max_buoys"], dtype=bool),
            association=assignment.copy(),
        )
        from wt.htw_phase9.features import feature_dict_l3b

        feat = attach_option(feature_dict_l3b(info, env), env, info, option, schema)
        extra = feat["uav"][:, schema["base_uav_dim"] :]
        onehot = extra[:, : schema["option_onehot_dim"]]
        rel = extra[:, schema["option_onehot_dim"] : schema["option_onehot_dim"] + 2]
        valid = extra[:, -1]
        self.assertTrue(np.allclose(onehot[:, -1], 1.0))
        self.assertTrue(np.allclose(onehot[:, :-1], 0.0))
        self.assertTrue(np.allclose(rel, 0.0))
        self.assertTrue(np.allclose(valid, 0.0))
        self.assertEqual(float(auxiliary_reward(env, option)["in_range_count"]), 0.0)

    def test_03_refuses_m5_checkpoint_path(self) -> None:
        with self.assertRaises(RuntimeError):
            refuse_m5(FORBIDDEN_M5_CKPT)

    def test_04_design_seeds_are_phase9_train_val(self) -> None:
        self.assertEqual(DESIGN_TRAIN_SEEDS, tuple(TRAIN_SEEDS))
        self.assertEqual(DESIGN_VAL_SEEDS, tuple(EVAL_SEEDS))
        self.assertTrue(all(s < 24 for s in DESIGN_VAL_SEEDS))
        self.assertTrue(all(s not in range(1321, 2000) for s in DESIGN_VAL_SEEDS))

    def test_05_ppo_config_is_the_live_object(self) -> None:
        self.assertEqual(PPO.rollout_length, 48)
        self.assertEqual(PPO.num_envs, 4)
        self.assertEqual(PPO.gamma, 0.99)
        self.assertEqual(PPO.gae_lambda, 0.95)
        self.assertEqual(PPO.clip_eps, 0.2)
        schema = build_schema()
        model = L3BGoalActorCritic(schema)
        dummy = {
            "uav": torch.zeros(2, schema["max_uavs"], schema["uav_dim"]),
            "buoy": torch.zeros(2, schema["max_buoys"], schema["buoy_dim"]),
            "uav_mask": torch.ones(2, schema["max_uavs"], dtype=torch.bool),
            "buoy_mask": torch.ones(2, schema["max_buoys"], dtype=torch.bool),
            "global": torch.zeros(2, schema["global_dim"]),
        }
        mu, log_std, value = model(**dummy)
        self.assertEqual(tuple(mu.shape), (2, 6, 2))
        self.assertEqual(tuple(value.shape), (2,))

    def test_06_hold_schedule(self) -> None:
        times = []
        pending = 0
        for t in range(48):
            if pending <= 0:
                times.append(t)
                pending = int(HOLD)
            pending -= 1
        self.assertEqual(times, [0, 15, 30, 45])


if __name__ == "__main__":
    unittest.main()
