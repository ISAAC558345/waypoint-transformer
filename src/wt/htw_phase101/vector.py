"""Closed-loop L3-B vector: frozen high-level association + low-level movement."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase100.high_level import decide, make_option
from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv

from .features import encode_obs, stack_feature_dicts
from .rewards import auxiliary_reward, env_reward


class L3BLowLevelVector:
    def __init__(self, seeds: tuple[int, ...], count: int, seed: int, *, high_model, schema: dict[str, Any]) -> None:
        self.seeds = tuple(int(s) for s in seeds)
        self.rng = np.random.default_rng(int(seed))
        self.environment_steps = 0
        self.high_model = high_model
        self.schema = schema
        self.max_uavs = int(schema["max_uavs"])
        self.reward_parts = {"r_env": [], "r_aux": [], "r_train": []}
        self.worlds = [self._new_world() for _ in range(int(count))]

    def _new_world(self) -> dict[str, Any]:
        scenario_seed = int(self.seeds[int(self.rng.integers(len(self.seeds)))])
        env = L3BEnv()
        info = env.reset(seed=scenario_seed)
        return {
            "env": env,
            "info": info,
            "seed": scenario_seed,
            "pending": 0,
            "option": None,
            "last_arrival": np.zeros(env.config.num_buoys, dtype=np.float32),
            "last_positions": np.asarray(env.uav_positions, dtype=np.float32).copy(),
            "return_env": 0.0,
            "high_decisions": 0,
            "terminal_obs": None,
        }

    def _refresh_option(self, world: dict[str, Any]) -> None:
        env: L3BEnv = world["env"]
        info = world["info"]
        if int(world["pending"]) <= 0:
            assignment = decide(self.high_model, env, info, world["last_arrival"], world["last_positions"], world["pending"])
            world["pending"] = int(HOLD)
            world["option"] = make_option(env, info, assignment, world["pending"])
            world["high_decisions"] += 1
        option: TaskOption = world["option"]
        option.remaining_hold = int(world["pending"])
        if not np.array_equal(option.association, option.targets):
            raise RuntimeError("vector refused to let the low-level rewrite association")

    def _row(self, world: dict[str, Any]) -> dict[str, np.ndarray]:
        self._refresh_option(world)
        return encode_obs(world["env"], world["info"], world["option"], self.schema, world["last_arrival"], world["last_positions"])

    def features(self) -> dict[str, np.ndarray]:
        return stack_feature_dicts([self._row(world) for world in self.worlds])

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        expected = (len(self.worlds), self.max_uavs, 2)
        if tuple(actions.shape) != expected:
            raise ValueError(f"six-UAV PPO action batch has invalid shape {actions.shape}, expected {expected}")
        rewards, dones, completed = [], [], []
        for index, (world, requested) in enumerate(zip(self.worlds, actions)):
            self._refresh_option(world)
            env: L3BEnv = world["env"]
            option: TaskOption = world["option"]
            raw = np.asarray(requested[: env.config.num_uavs], dtype=np.float64)
            before_payload = float(env.payload_bits)
            after = env.step(raw, option.association)
            r_env = env_reward(before_payload, float(after["cumulative_payload_bits"]), env.config.block_payload_bits)
            parts = auxiliary_reward(env, option)
            r_aux = float(parts["r_aux"])
            r_train = r_env + r_aux
            world["pending"] = int(world["pending"]) - 1
            option.remaining_hold = int(world["pending"])
            world["last_arrival"] = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            world["last_positions"] = np.asarray(env.uav_positions, dtype=np.float32).copy()
            world["return_env"] += float(r_env)
            self.environment_steps += 1
            self.reward_parts["r_env"].append(float(r_env))
            self.reward_parts["r_aux"].append(float(r_aux))
            self.reward_parts["r_train"].append(float(r_train))
            rewards.append(float(r_train))
            done = bool(after["terminated"])
            dones.append(done)
            if done:
                world["terminal_obs"] = encode_obs(env, after, option, self.schema, world["last_arrival"], world["last_positions"])
                completed.append(
                    {
                        "payload_bits": float(after["cumulative_payload_bits"]),
                        "return": float(world["return_env"]),
                        "dropped_blocks": float(after["dropped_blocks"]),
                        "residual_queue_blocks": float(np.sum(after["queue_blocks"])),
                        "high_decisions": float(world["high_decisions"]),
                    }
                )
                self.worlds[index] = self._new_world()
            else:
                world["info"] = after
        return self.features(), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed

    def parts_mean(self) -> dict[str, float]:
        out = {}
        for key, values in self.reward_parts.items():
            out[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
        return out


__all__ = ["L3BLowLevelVector"]
