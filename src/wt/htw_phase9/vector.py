"""Held high-level vector on frozen L3-B. Low-level is fly-toward; PPO weights are not loaded."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase64.vector import high_level_reward
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward

from .const import HOLD, MAX_UAVS, REWARD_MODE, indices_to_assignment
from .features import build_graph, conflict_rate, feature_dict_l3b, in_range_completion, interference_utility, service_success, stack_feature_dicts


class L3BHeldVector:
    def __init__(self, seeds: tuple[int, ...], count: int, seed: int, *, hold: int = HOLD, reward_mode: str = REWARD_MODE, config=None) -> None:
        self.seeds = seeds
        self.rng = np.random.default_rng(seed)
        self.environment_steps = 0
        self.hold = int(hold)
        self.reward_mode = str(reward_mode)
        self.config = config
        self.worlds = [self._new_world() for _ in range(count)]

    def _new_world(self) -> dict[str, Any]:
        scenario_seed = int(self.seeds[int(self.rng.integers(len(self.seeds)))])
        env = L3BEnv(self.config)
        info = env.reset(seed=scenario_seed)
        return {
            "env": env,
            "info": info,
            "return": 0.0,
            "seed": scenario_seed,
            "last_arrival": np.zeros(env.config.num_buoys, dtype=np.float32),
            "last_positions": np.asarray(env.uav_positions, dtype=np.float32).copy(),
        }

    def _row_features(self, world: dict[str, Any]) -> dict[str, np.ndarray]:
        return feature_dict_l3b(world["info"], world["env"], last_arrival=world["last_arrival"], last_positions=world["last_positions"])

    def features(self) -> dict[str, np.ndarray]:
        rows = [self._row_features(world) for world in self.worlds]
        batch = stack_feature_dicts(rows)
        edges = [build_graph(world["info"], world["env"]) for world in self.worlds]
        batch["edge"] = np.stack([item[0] for item in edges], axis=0)
        batch["edge_uu"] = np.stack([item[1] for item in edges], axis=0)
        return batch

    def step(self, z_indices: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        if z_indices.shape != (len(self.worlds), MAX_UAVS):
            raise ValueError("held high-level z batch has invalid shape")
        rewards, dones, completed = [], [], []
        for index, (world, z_row) in enumerate(zip(self.worlds, z_indices)):
            env: L3BEnv = world["env"]
            assignment = indices_to_assignment(z_row, env.config.num_uavs)
            delta = 0.0
            done = False
            after = world["info"]
            for _ in range(self.hold):
                before_payload = float(env.payload_bits)
                move = fly_toward(env, assignment)
                after = env.step(move, assignment)
                delta += (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
                world["last_arrival"] = np.asarray(after["arrivals_blocks"], dtype=np.float32)
                world["last_positions"] = np.asarray(world["env"].uav_positions, dtype=np.float32).copy()
                world["return"] += (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
                self.environment_steps += 1
                world["info"] = after
                if bool(after["terminated"]):
                    done = True
                    break
            reward = high_level_reward(self.reward_mode, delta, 0.0, 0.0, float(world["return"]), done)
            rewards.append(float(reward))
            dones.append(done)
            if done:
                completed.append({"payload_bits": float(after["cumulative_payload_bits"]), "return": float(world["return"])})
                self.worlds[index] = self._new_world()
        return self.features(), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


def collect_held(agent, vector: L3BHeldVector):
    feature_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")}
    records = {key: [] for key in ("z", "log_prob", "values", "rewards", "dones", "entropy")}
    events = []
    features = vector.features()
    for _ in range(agent.config.rollout_length):
        decision = agent.act(features)
        for key in feature_rows:
            feature_rows[key].append(features[key])
        next_features, rewards, dones, completed = vector.step(decision["z"])
        records["z"].append(decision["z"])
        records["log_prob"].append(decision["log_prob"])
        records["values"].append(decision["value"])
        records["rewards"].append(rewards)
        records["dones"].append(dones)
        records["entropy"].append(decision["entropy"])
        events.extend(completed)
        features = next_features
    storage = {key: np.stack(value, axis=0) for key, value in feature_rows.items()}
    storage.update(
        {
            "z": np.stack(records["z"], axis=0),
            "log_prob": np.stack(records["log_prob"], axis=0),
            "values": np.stack(records["values"], axis=0),
            "rewards": np.stack(records["rewards"], axis=0),
            "dones": np.stack(records["dones"], axis=0),
            "last_values": agent.act(features, deterministic=True)["value"],
        }
    )
    health = {"mean_raw_reward": float(storage["rewards"].mean()), "mean_entropy": float(np.mean(records["entropy"]))}
    return storage, events, health


__all__ = ["L3BHeldVector", "collect_held", "conflict_rate", "in_range_completion", "interference_utility", "service_success"]
