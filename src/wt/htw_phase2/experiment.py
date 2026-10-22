"""Oracle hierarchical controller experiment. No HGNN, no predictor, no Round."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.htwc_phase1.conditioned_actor import GoalConditionedActorCritic
from wt.htwc_phase1.contract import DEFAULT_HORIZON
from wt.htwc_phase1.oracle import oracle_conditioner, tw_enters_actor
from wt.htwc_phase1.relevance import summarize_cosines
from wt.htwc_phase1.validation import (
    MINI_UPDATES,
    SEED,
    SMOKE_CONFIG,
    BaselineSpec,
    ConditionedVector,
    eval_slice,
    train_slice,
)
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.features import feature_dict, stack_feature_dicts
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent, seed_everything
from wt.round30.runner import collect
from wt.round30.scenarios import PPOScenario

from .tracking import (
    DEFAULT_TRACKING_LAMBDA,
    action_waypoint_cosine,
    normalize_positions_m,
    tracking_distance,
    waypoint_from_uav_features,
)
from .vector import HierarchicalTrackingVector


def oracle_low_level_conditioner(info, env):
    return oracle_conditioner(info, env, horizon=DEFAULT_HORIZON, include_goal=True, include_waypoint=True)


def state_only_conditioner(info, env):
    return feature_dict(info, env.config)


def train_controller(spec: BaselineSpec, scenarios: tuple[PPOScenario, ...], updates: int, tracking_lambda: float) -> dict[str, Any]:
    seed_everything(SEED)
    agent = PPOAgent(spec.model, SMOKE_CONFIG)
    if spec.family == "htw_oracle":
        vector = HierarchicalTrackingVector(scenarios, agent.config.num_envs, 73000 + SEED, spec.conditioner, tracking_lambda)
    else:
        vector = ConditionedVector(scenarios, agent.config.num_envs, 73000 + SEED, spec.conditioner)
    curve = []
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {"update": float(index), **health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"{spec.name} PPO update produced NaN/Inf")
        curve.append(row)
    return {
        "family": spec.family,
        "uav_dim": spec.uav_dim,
        "tracking_lambda": float(tracking_lambda if spec.family == "htw_oracle" else 0.0),
        "environment_steps": vector.environment_steps,
        "mean_ppo_reward_last": curve[-1]["mean_raw_reward"],
        "curve": curve,
        "agent": agent,
    }


def evaluate_controller(spec: BaselineSpec, agent: PPOAgent, scenarios: tuple[PPOScenario, ...]) -> dict[str, Any]:
    payloads, returns, queues = [], [], []
    distances, rhos = [], []
    entry_flags = []
    for scenario in scenarios:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        total_return, done = 0.0, False
        while not done:
            oracle_features = oracle_low_level_conditioner(info, env)
            policy_features = spec.conditioner(info, env)
            entry_flags.append(tw_enters_actor(policy_features) if policy_features["uav"].shape[-1] == 16 else {"uav_width_is_conditioned": False, "goal_nonzero": False, "waypoint_nonzero": False, "horizon_nonzero": False})
            requested = agent.act(stack_feature_dicts([policy_features]), deterministic=True)["action"][0]
            waypoint = waypoint_from_uav_features(oracle_features["uav"])
            now_xy = normalize_positions_m(info["uav_positions_m"], env.config)
            for uav in range(env.num_uavs):
                rhos.append(action_waypoint_cosine(requested[uav], now_xy[uav], waypoint[uav]))
            try:
                _, reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error):
                    raise
                _, reward, terminated, truncated, after = env.step(np.zeros_like(requested), matching_mode="legacy")
            after_xy = normalize_positions_m(after["uav_positions_m"], env.config)
            distances.append(tracking_distance(after_xy, waypoint, oracle_features["uav_mask"]))
            total_return += float(reward)
            info, done = after, bool(terminated or truncated)
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        queues.append(float(np.sum(info["raw_queue_bits"])))
    flags = {
        "any_conditioned_width": any(item["uav_width_is_conditioned"] for item in entry_flags) if entry_flags else False,
        "any_waypoint_nonzero": any(item["waypoint_nonzero"] for item in entry_flags) if entry_flags else False,
    }
    return {
        "family": spec.family,
        "episodes": len(payloads),
        "mean_payload_bits": float(np.mean(payloads)),
        "mean_return": float(np.mean(returns)),
        "mean_residual_queue_bits": float(np.mean(queues)),
        "mean_waypoint_tracking_error": float(np.mean(distances)) if distances else float("nan"),
        "action_waypoint": summarize_cosines(rhos),
        "tw_enters_actor": flags,
    }


def build_specs() -> dict[str, BaselineSpec]:
    return {
        "state_only": BaselineSpec("State-only PPO", "state_only", 8, DeepSetsActorCritic(), state_only_conditioner),
        "htw_oracle": BaselineSpec("HTW-Oracle-PPO", "htw_oracle", 16, GoalConditionedActorCritic(), oracle_low_level_conditioner),
    }


def run_experiment(updates: int = MINI_UPDATES, tracking_lambda: float = DEFAULT_TRACKING_LAMBDA) -> dict[str, Any]:
    train, evaluation = train_slice(), eval_slice()
    specs = build_specs()
    trained: dict[str, Any] = {}
    evaluated: dict[str, Any] = {}
    for key, spec in specs.items():
        trained[key] = train_controller(spec, train, updates, tracking_lambda)
        evaluated[key] = evaluate_controller(spec, trained[key]["agent"], evaluation)
        evaluated[key]["environment_steps"] = trained[key]["environment_steps"]
        evaluated[key]["tracking_lambda"] = trained[key]["tracking_lambda"]
        evaluated[key]["mean_ppo_reward_last"] = trained[key]["mean_ppo_reward_last"]
    return {
        "horizon": DEFAULT_HORIZON,
        "tracking_lambda": float(tracking_lambda),
        "updates": int(updates),
        "fresh_access": 0,
        "untouched_access": 0,
        "hgnn": False,
        "transformer_modified": False,
        "environment_reward_modified": False,
        "ppo_core_modified": False,
        "results": evaluated,
    }


__all__ = ["build_specs", "evaluate_controller", "run_experiment", "train_controller"]
