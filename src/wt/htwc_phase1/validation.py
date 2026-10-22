"""Shared PPO loop for the three TW-input baselines. Does not edit round30/ppo.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.evaluation import rollout
from wt.round30.features import feature_dict, stack_feature_dicts
from wt.round30.models import CentralDeepSetsCritic, DeepSetsActorCritic, DeepSetsPPOShape, SharedDeepSetsActor
from wt.round30.ppo import PPOAgent, PPOConfig, seed_everything
from wt.round30.runner import TrainingVector, collect
from wt.round30.scenarios import PPOScenario, training_scenarios

from .adapter import CONDITIONED_UAV_DIM, attach_task_waypoint
from .conditioned_actor import GoalConditionedActorCritic
from .contract import DEFAULT_HORIZON, TaskWaypointV2
from .labels import collect_planner_window
from .old_tw import OLD_CONDITIONED_UAV_DIM, attach_tw_old, predict_tw_old
from .predictor import TaskWaypointV2Predictor, TWPredictorConfig, task_waypoint_v2_loss
from .relevance import cosine, new_tw_direction, old_tw_direction, summarize_cosines

Conditioner = Callable[[Mapping[str, Any], SharedSpectrumMultiUAVEnv], dict[str, np.ndarray]]

SMOKE_CONFIG = PPOConfig(rollout_length=16, num_envs=2, num_minibatches=2, update_epochs=1, checkpoint_interval_steps=32)
MINI_UPDATES = 8
EVAL_COUNT = 4
TRAIN_SLICE = 16
SEED = 0


@dataclass
class BaselineSpec:
    name: str
    family: str
    uav_dim: int
    model: DeepSetsActorCritic
    conditioner: Conditioner


class ConditionedVector(TrainingVector):
    def __init__(self, scenarios: tuple[PPOScenario, ...], count: int, seed: int, conditioner: Conditioner) -> None:
        self.conditioner = conditioner
        super().__init__(scenarios, count, seed)

    def features(self) -> dict[str, np.ndarray]:
        return stack_feature_dicts([self.conditioner(world["info"], world["env"]) for world in self.worlds])


class OldTWActorCritic(DeepSetsActorCritic):
    def __init__(self) -> None:
        shape = DeepSetsPPOShape(uav_dim=OLD_CONDITIONED_UAV_DIM)
        super().__init__(shape)
        self.actor = SharedDeepSetsActor(shape)
        self.critic = CentralDeepSetsCritic(shape)

    @property
    def uses_task_waypoint(self) -> bool:
        return True


def state_only_conditioner(info: Mapping[str, Any], env: SharedSpectrumMultiUAVEnv) -> dict[str, np.ndarray]:
    return feature_dict(info, env.config)


def old_tw_conditioner(info: Mapping[str, Any], env: SharedSpectrumMultiUAVEnv) -> dict[str, np.ndarray]:
    features = feature_dict(info, env.config)
    return attach_tw_old(features, predict_tw_old(info, env.config))


def make_new_tw_conditioner(predictor: TaskWaypointV2Predictor) -> Conditioner:
    predictor.eval()

    def conditioner(info: Mapping[str, Any], env: SharedSpectrumMultiUAVEnv) -> dict[str, np.ndarray]:
        features = feature_dict(info, env.config)
        batch = {key: torch.as_tensor(value[None, ...], dtype=torch.bool if key.endswith("mask") else torch.float32) for key, value in features.items()}
        with torch.no_grad():
            output = predictor(batch["uav"], batch["buoy"], batch["uav_mask"], batch["buoy_mask"], batch["global"])
        goal = output["goal_distribution"][0].cpu().numpy().astype(np.float64)
        waypoint = output["waypoint_xy"][0].cpu().numpy().astype(np.float32)
        totals = goal.sum(axis=-1)
        valid = np.asarray(features["uav_mask"], dtype=bool) & (totals > 0.5)
        for index, ok in enumerate(valid):
            if ok:
                goal[index] = goal[index] / max(float(totals[index]), 1.0e-8)
            else:
                goal[index] = 0.0
        tw = TaskWaypointV2(goal.astype(np.float32), waypoint, DEFAULT_HORIZON, valid, features["uav_mask"], features["buoy_mask"])
        return attach_task_waypoint(features, tw)

    return conditioner


def warmup_new_predictor(scenarios: tuple[PPOScenario, ...], steps: int = 12) -> tuple[TaskWaypointV2Predictor, list[float]]:
    predictor = TaskWaypointV2Predictor(TWPredictorConfig())
    optimizer = torch.optim.Adam(predictor.parameters(), lr=3.0e-4)
    samples: list[tuple[dict[str, np.ndarray], TaskWaypointV2]] = []
    for scenario in scenarios[:8]:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        features = feature_dict(info, env.config)
        try:
            label, _ = collect_planner_window(env, info, horizon=DEFAULT_HORIZON, policy="independent_queue_rate")
        except RuntimeError:
            continue
        samples.append((features, label))
    if not samples:
        raise RuntimeError("TW_new warmup could not collect planner labels")
    history: list[float] = []
    predictor.train()
    for _ in range(int(steps)):
        losses = []
        for features, label in samples:
            batch = {key: torch.as_tensor(value[None, ...], dtype=torch.bool if key.endswith("mask") else torch.float32) for key, value in features.items()}
            pred = predictor(batch["uav"], batch["buoy"], batch["uav_mask"], batch["buoy_mask"], batch["global"])
            goal = torch.as_tensor(label.goal_distribution[None, ...])
            waypoint = torch.as_tensor(label.waypoint_xy[None, ...])
            uav_mask = batch["uav_mask"]
            valid = torch.as_tensor(label.goal_valid_uav_mask[None, ...])
            loss, _ = task_waypoint_v2_loss(pred, goal, waypoint, uav_mask, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if not np.isfinite(history[-1]):
            raise RuntimeError("TW_new predictor warmup produced NaN")
    predictor.eval()
    return predictor, history


def train_baseline(spec: BaselineSpec, scenarios: tuple[PPOScenario, ...], updates: int, seed: int = SEED) -> dict[str, Any]:
    seed_everything(int(seed))
    agent = PPOAgent(spec.model, SMOKE_CONFIG)
    vector = ConditionedVector(scenarios, agent.config.num_envs, 73000 + seed, spec.conditioner)
    curve: list[dict[str, float]] = []
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {"update": float(index), **health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"{spec.name} PPO update produced NaN/Inf")
        curve.append(row)
    return {"family": spec.family, "name": spec.name, "uav_dim": spec.uav_dim, "environment_steps": vector.environment_steps, "curve": curve, "agent": agent}


def evaluate_baseline(spec: BaselineSpec, agent: PPOAgent, scenarios: tuple[PPOScenario, ...]) -> dict[str, Any]:
    metrics, rho, distances, angles = [], [], [], []

    def action_fn(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any]) -> np.ndarray:
        features = spec.conditioner(info, env)
        requested = agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
        current_xy = features["uav"][:, :2]
        phi = np.asarray(features["uav"][:, 8:19], dtype=np.float64) if spec.family == "tw_old" else None
        for uav in range(requested.shape[0]):
            if not bool(features["uav_mask"][uav]):
                continue
            action = np.asarray(requested[uav], dtype=np.float64)
            if spec.family == "htw_new":
                extra = features["uav"][uav, 8:]
                direction = new_tw_direction(extra[5:7], current_xy[uav])
                distances.append(float(np.linalg.norm(direction)))
                value = cosine(action, direction)
            elif spec.family == "tw_old":
                direction = old_tw_direction(phi, info, uav)
                value = cosine(action, direction)
            else:
                continue
            rho.append(value)
            if np.isfinite(value):
                angles.append(float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))))
        return requested

    for scenario in scenarios:
        metric, _ = rollout(scenario, population="htw_phase1_eval", policy_id=spec.family, model_seed=SEED, action_fn=action_fn)
        metrics.append(metric)
    payload = [row["cumulative_payload_bits"] for row in metrics]
    returns = [row["episode_return"] for row in metrics]
    queues = [row["residual_queue_bits"] for row in metrics]
    fairness = [row["jain_fairness"] for row in metrics]
    energy = [row["energy"] for row in metrics]
    energy_eff = [float(row["cumulative_payload_bits"]) / max(float(row["energy"]), 1.0) for row in metrics]
    return {
        "family": spec.family,
        "episodes": len(metrics),
        "mean_payload_bits": float(np.mean(payload)),
        "mean_return": float(np.mean(returns)),
        "mean_residual_queue_bits": float(np.mean(queues)),
        "mean_jain_fairness": float(np.mean(fairness)),
        "mean_energy": float(np.mean(energy)),
        "mean_energy_efficiency": float(np.mean(energy_eff)),
        "action_relevance": summarize_cosines(rho),
        "mean_waypoint_distance": float(np.nanmean(distances)) if distances else float("nan"),
        "mean_waypoint_angle_deg": float(np.nanmean(angles)) if angles else float("nan"),
        "rows": metrics,
    }


def interface_shapes() -> dict[str, int]:
    return {
        "state_only": DeepSetsActorCritic().shape.uav_dim,
        "tw_old": OLD_CONDITIONED_UAV_DIM,
        "htw_new": CONDITIONED_UAV_DIM,
    }


def build_specs(new_predictor: TaskWaypointV2Predictor | None = None) -> dict[str, BaselineSpec]:
    new_model = GoalConditionedActorCritic()
    conditioner = make_new_tw_conditioner(new_predictor) if new_predictor is not None else state_only_conditioner
    return {
        "state_only": BaselineSpec("Baseline 1 State-only PPO", "state_only", DeepSetsActorCritic().shape.uav_dim, DeepSetsActorCritic(), state_only_conditioner),
        "tw_old": BaselineSpec("Baseline 2 Original TW PPO", "tw_old", OLD_CONDITIONED_UAV_DIM, OldTWActorCritic(), old_tw_conditioner),
        "htw_new": BaselineSpec("Baseline 3 HTW TW PPO", "htw_new", CONDITIONED_UAV_DIM, new_model, conditioner),
    }


def train_slice() -> tuple[PPOScenario, ...]:
    return training_scenarios()[:TRAIN_SLICE]


def eval_slice() -> tuple[PPOScenario, ...]:
    return training_scenarios()[TRAIN_SLICE:TRAIN_SLICE + EVAL_COUNT]


__all__ = [
    "BaselineSpec",
    "ConditionedVector",
    "MINI_UPDATES",
    "SMOKE_CONFIG",
    "build_specs",
    "eval_slice",
    "evaluate_baseline",
    "interface_shapes",
    "train_baseline",
    "train_slice",
    "warmup_new_predictor",
]
