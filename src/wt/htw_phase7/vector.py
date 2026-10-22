"""Held high-level vector that also emits UAV–Buoy edges. Hold and reward stay Phase-6.4."""
from __future__ import annotations

import numpy as np

from wt.htw_phase64.vector import HeldHighVector

from .graph import build_edge_features


class GraphHeldVector(HeldHighVector):
    def features(self) -> dict[str, np.ndarray]:
        batch = super().features()
        batch["edge"] = np.stack([build_edge_features(world["info"], world["env"]) for world in self.worlds], axis=0)
        return batch


def collect_graph_held(agent, vector: GraphHeldVector):
    feature_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge")}
    records = {key: [] for key in ("z", "log_prob", "values", "rewards", "dones", "entropy")}
    events = []
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


__all__ = ["GraphHeldVector", "collect_graph_held"]
