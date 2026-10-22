"""Hierarchical vector: learned or oracle z, frozen or live low-level movement."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase61.features import MAX_UAVS, attach_option, encode_option, feature_dict_l2a, in_range_completion, service_success, stack_feature_dicts
from wt.htw_phase62.rewards import low_level_reward, mean_target_distance_slots
from wt.l2a.env import L2AEnv, project_exclusive
from wt.l2a.policies import oracle_option_assignment
from wt.round30.ppo import PPOAgent

from .high_ppo import assignment_to_indices, indices_to_assignment


class HierarchicalVector:
    def __init__(
        self,
        seeds: tuple[int, ...],
        count: int,
        seed: int,
        low_agent: PPOAgent,
        *,
        lambda_c: float,
        lambda_d: float,
        high_mode: str,
        low_deterministic: bool = True,
    ) -> None:
        if high_mode not in {"oracle", "learned"}:
            raise ValueError("high_mode must be oracle or learned")
        self.seeds = seeds
        self.rng = np.random.default_rng(seed)
        self.environment_steps = 0
        self.low_agent = low_agent
        self.lambda_c = float(lambda_c)
        self.lambda_d = float(lambda_d)
        self.high_mode = high_mode
        self.low_deterministic = bool(low_deterministic)
        self.worlds = [self._new_world() for _ in range(count)]

    def _new_world(self) -> dict[str, Any]:
        scenario_seed = int(self.seeds[int(self.rng.integers(len(self.seeds)))])
        env = L2AEnv()
        info = env.reset(seed=scenario_seed)
        return {"env": env, "info": info, "return": 0.0, "seed": scenario_seed, "accuracies": []}

    def features(self) -> dict[str, np.ndarray]:
        rows = []
        positions = []
        buoys = []
        for world in self.worlds:
            rows.append(feature_dict_l2a(world["info"], world["env"]))
            positions.append(np.asarray(world["env"].uav_positions, dtype=np.float64))
            buoys.append(np.asarray(world["env"].buoy_positions, dtype=np.float64))
        batch = stack_feature_dicts(rows)
        batch["_uav_positions"] = np.stack(positions, axis=0)
        batch["_buoy_positions"] = np.stack(buoys, axis=0)
        return batch

    def _low_move(self, env: L2AEnv, info: dict[str, Any], assignment: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        features = attach_option(feature_dict_l2a(info, env), encode_option(assignment))
        stacked = stack_feature_dicts([features])
        decision = self.low_agent.act(stacked, deterministic=self.low_deterministic)
        return decision["action"][0], stacked, decision

    def step(self, z_indices: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        if z_indices.shape != (len(self.worlds), MAX_UAVS):
            raise ValueError("high-level z batch has invalid shape")
        rewards, low_rewards, dones, completed = [], [], [], []
        executed = np.zeros_like(z_indices)
        low_packs: list[dict[str, np.ndarray]] = []
        for index, (world, z_row) in enumerate(zip(self.worlds, z_indices)):
            env: L2AEnv = world["env"]
            oracle = oracle_option_assignment(env)
            if self.high_mode == "oracle":
                assignment = np.asarray(oracle, dtype=np.int64)
            else:
                assignment = project_exclusive(indices_to_assignment(z_row), env.uav_positions, env.buoy_positions)
            executed[index] = assignment_to_indices(assignment)
            before_payload = float(env.payload_bits)
            move, stacked, low_decision = self._low_move(env, world["info"], assignment)
            after = env.step(move, assignment)
            r_comm = (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
            complete = in_range_completion(env, assignment)
            dist_slots = mean_target_distance_slots(env, assignment)
            r_low = low_level_reward(r_comm, complete, dist_slots, self.lambda_c, self.lambda_d)
            accuracy = float(np.mean(np.asarray(assignment, dtype=np.int64) == np.asarray(oracle, dtype=np.int64)))
            world["return"] += float(r_comm)
            world["accuracies"].append(accuracy)
            self.environment_steps += 1
            rewards.append(float(r_comm))
            low_rewards.append(float(r_low))
            low_packs.append(
                {
                    "uav": stacked["uav"][0],
                    "buoy": stacked["buoy"][0],
                    "uav_mask": stacked["uav_mask"][0],
                    "buoy_mask": stacked["buoy_mask"][0],
                    "global": stacked["global"][0],
                    "pre_tanh": low_decision["pre_tanh"][0],
                    "log_prob": low_decision["log_prob"][0],
                    "value": low_decision["value"][0],
                }
            )
            done = bool(after["terminated"])
            dones.append(done)
            if done:
                completed.append(
                    {
                        "payload_bits": float(after["cumulative_payload_bits"]),
                        "return": float(world["return"]),
                        "option_selection_accuracy": float(np.mean(world["accuracies"])),
                    }
                )
                self.worlds[index] = self._new_world()
            else:
                world["info"] = after
        next_features = self.features()
        next_features["_executed_z"] = executed
        next_features["_low_rewards"] = np.asarray(low_rewards, dtype=np.float32)
        next_features["_low_packs"] = low_packs
        return next_features, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


def collect_high(agent, vector: HierarchicalVector) -> tuple[dict[str, np.ndarray], list[dict[str, float]], dict[str, float]]:
    feature_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}
    records = {key: [] for key in ("z", "log_prob", "values", "rewards", "dones")}
    events: list[dict[str, float]] = []
    features = vector.features()
    for _ in range(agent.config.rollout_length):
        decision = agent.act(features)
        for key in feature_rows:
            feature_rows[key].append(features[key])
        next_features, rewards, dones, completed = vector.step(decision["z"])
        records["z"].append(next_features.get("_executed_z", decision["z"]))
        records["log_prob"].append(decision["log_prob"])
        records["values"].append(decision["value"])
        records["rewards"].append(rewards)
        records["dones"].append(dones)
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
    return storage, events, {"mean_raw_reward": float(storage["rewards"].mean())}


def collect_both(high_agent, low_agent, vector: HierarchicalVector) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, float]], dict[str, float]]:
    feature_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}
    low_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}
    high_records = {key: [] for key in ("z", "log_prob", "values", "rewards", "dones")}
    low_records = {key: [] for key in ("pre_tanh", "log_prob", "values", "rewards", "dones")}
    events: list[dict[str, float]] = []
    features = vector.features()
    for _ in range(high_agent.config.rollout_length):
        decision = high_agent.act(features)
        for key in feature_rows:
            feature_rows[key].append(features[key])
        next_features, rewards, dones, completed = vector.step(decision["z"])
        packs = next_features["_low_packs"]
        for key in low_rows:
            low_rows[key].append(np.stack([pack[key] for pack in packs], axis=0))
        low_records["pre_tanh"].append(np.stack([pack["pre_tanh"] for pack in packs], axis=0))
        low_records["log_prob"].append(np.asarray([pack["log_prob"] for pack in packs], dtype=np.float32))
        low_records["values"].append(np.asarray([pack["value"] for pack in packs], dtype=np.float32))
        low_records["rewards"].append(next_features["_low_rewards"])
        low_records["dones"].append(dones)
        high_records["z"].append(next_features.get("_executed_z", decision["z"]))
        high_records["log_prob"].append(decision["log_prob"])
        high_records["values"].append(decision["value"])
        high_records["rewards"].append(rewards)
        high_records["dones"].append(dones)
        events.extend(completed)
        features = next_features
    high_storage = {key: np.stack(value, axis=0) for key, value in feature_rows.items()}
    high_storage.update(
        {
            "z": np.stack(high_records["z"], axis=0),
            "log_prob": np.stack(high_records["log_prob"], axis=0),
            "values": np.stack(high_records["values"], axis=0),
            "rewards": np.stack(high_records["rewards"], axis=0),
            "dones": np.stack(high_records["dones"], axis=0),
        }
    )
    last_high = high_agent.act(features, deterministic=True)
    high_storage["last_values"] = last_high["value"]
    low_storage = {key: np.stack(value, axis=0) for key, value in low_rows.items()}
    low_last = []
    for world, z_row in zip(vector.worlds, last_high["z"]):
        assignment = project_exclusive(indices_to_assignment(z_row), world["env"].uav_positions, world["env"].buoy_positions)
        stacked = stack_feature_dicts([attach_option(feature_dict_l2a(world["info"], world["env"]), encode_option(assignment))])
        low_last.append(low_agent.act(stacked, deterministic=True)["value"][0])
    low_storage.update(
        {
            "pre_tanh": np.stack(low_records["pre_tanh"], axis=0),
            "log_prob": np.stack(low_records["log_prob"], axis=0),
            "values": np.stack(low_records["values"], axis=0),
            "rewards": np.stack(low_records["rewards"], axis=0),
            "dones": np.stack(low_records["dones"], axis=0),
            "last_values": np.asarray(low_last, dtype=np.float32),
        }
    )
    return high_storage, low_storage, events, {"mean_raw_reward": float(high_storage["rewards"].mean())}


__all__ = ["HierarchicalVector", "collect_both", "collect_high"]
