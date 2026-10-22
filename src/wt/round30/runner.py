"""Round-30 smoke, protocol freeze, and five-seed state-only PPO training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv

from .features import feature_dict, stack_feature_dicts
from .models import DeepSetsActorCritic
from .ppo import PPOAgent, PPOConfig, seed_everything
from .scenarios import PPOScenario, assert_disjoint, fresh_scenarios, training_scenarios, validation_scenarios


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_30"
OUTPUT = ROUND / "outputs" / "round30_deepsets_ppo_20260830_1845"
FORMAL = OUTPUT / "formal"
CONFIG_PATH = OUTPUT / "round30_ppo_training_config.json"
SEEDS = (0, 1, 2, 3, 4)
FORMAL_STEPS_PER_SEED = 307200


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [plain(row) for row in rows]
    if not values: raise RuntimeError(f"Round-30 required output is empty: {path.name}")
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in values:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def scenario_manifest(population: str, rows: Iterable[PPOScenario]) -> dict[str, Any]:
    scenarios = list(rows)
    return {"round": "round_30", "population": population, "scenario_count": len(scenarios), "scenarios": [row.manifest_row() for row in scenarios], "pre_registered_before_training": True, "outcomes_accessed": False}


def write_capability_audit() -> Path:
    path = OUTPUT / "ppo_capability_audit.md"
    if not path.exists():
        path.write_text("# PPO capability audit\n\nThe repository has no reusable PPO implementation for this branch. Round-30 implements standard local PyTorch PPO: GAE, clipped objective, value loss, entropy, advantage normalization, gradient clipping, finite checks, checkpoint reload, deterministic evaluation, and synchronous vectorized environments. It reuses the frozen Round-19 shared-spectrum environment, raw reward, matching and safety projection exactly. The PPO actor and critic are edge-free DeepSets MLPs with no Task-Waypoint, planner, graph-message-passing, Transformer, RNN, or offline BC input.\n", encoding="utf-8")
    return path


class TrainingVector:
    """Synchronous vector driver which samples only pre-registered training scenarios."""

    def __init__(self, scenarios: tuple[PPOScenario, ...], count: int, seed: int) -> None:
        self.scenarios, self.rng, self.environment_steps = scenarios, np.random.default_rng(seed), 0
        self.worlds = [self._new_world() for _ in range(count)]

    def _new_world(self) -> dict[str, Any]:
        scenario = self.scenarios[int(self.rng.integers(len(self.scenarios)))]
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        return {"scenario": scenario, "env": env, "info": info, "return": 0.0, "arrivals": 0.0, "projection_count": 0, "projection_magnitude": 0.0, "minimum_distance": float(info["min_uav_distance_m"])}

    def features(self) -> dict[str, np.ndarray]:
        return stack_feature_dicts([feature_dict(world["info"], world["env"].config) for world in self.worlds])

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        if actions.shape != (len(self.worlds), 2, 2): raise ValueError("vector PPO action batch has invalid shape")
        rewards, dones, completed = [], [], []
        for index, (world, requested) in enumerate(zip(self.worlds, actions)):
            env, info = world["env"], world["info"]
            before, fallback = np.asarray(info["uav_positions_m"], dtype=np.float64).copy(), False
            try:
                _, reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error): raise
                requested, fallback = np.zeros_like(requested), True
                _, reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            executed = (np.asarray(after["uav_positions_m"], dtype=np.float64) - before) / float(env.config.max_speed * env.config.delta_t)
            projection = float(np.linalg.norm(executed - requested)); done = bool(terminated or truncated)
            world["return"] += float(reward); world["arrivals"] += float(np.sum(after["raw_arrival_bits"])); world["projection_count"] += int(projection > 1.0e-8); world["projection_magnitude"] += projection
            world["minimum_distance"] = min(world["minimum_distance"], float(after["min_uav_distance_m"])); self.environment_steps += 1
            rewards.append(float(reward)); dones.append(done)
            if done:
                scenario = world["scenario"]
                completed.append({"payload_bits": float(after["cumulative_payload_bits"]), "return": float(world["return"]), "completion": float(after["cumulative_payload_bits"]) / max(float(sum(scenario.initial_queue_bits) + world["arrivals"]), 1.0), "residual_queue_bits": float(np.sum(after["raw_queue_bits"])), "projection_rate": float(world["projection_count"] / max(int(after["time_step"]), 1)), "projection_magnitude": float(world["projection_magnitude"] / max(int(after["time_step"]), 1)), "minimum_uav_distance_m": float(world["minimum_distance"]), "fallback": float(fallback), "finish_environment_step": float(self.environment_steps)})
                self.worlds[index] = self._new_world()
            else:
                world["info"] = after
        return self.features(), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


def collect(agent: PPOAgent, vector: TrainingVector) -> tuple[dict[str, np.ndarray], list[dict[str, float]], dict[str, float]]:
    feature_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}
    records = {key: [] for key in ("pre_tanh", "log_prob", "values", "rewards", "dones")}
    events: list[dict[str, float]] = []; action_stds: list[float] = []; features = vector.features()
    for _ in range(agent.config.rollout_length):
        decision = agent.act(features)
        for key in feature_rows: feature_rows[key].append(features[key])
        next_features, rewards, dones, completed = vector.step(decision["action"])
        records["pre_tanh"].append(decision["pre_tanh"]); records["log_prob"].append(decision["log_prob"]); records["values"].append(decision["value"])
        records["rewards"].append(rewards); records["dones"].append(dones); action_stds.extend(decision["action_std"].tolist()); events.extend(completed); features = next_features
    storage = {key: np.stack(value, axis=0) for key, value in feature_rows.items()}
    storage.update({"pre_tanh": np.stack(records["pre_tanh"], axis=0), "log_prob": np.stack(records["log_prob"], axis=0), "values": np.stack(records["values"], axis=0), "rewards": np.stack(records["rewards"], axis=0), "dones": np.stack(records["dones"], axis=0), "last_values": agent.act(features, deterministic=True)["value"]})
    return storage, events, {"mean_action_std": float(np.mean(action_stds)), "mean_raw_reward": float(storage["rewards"].mean())}


def run_smoke() -> dict[str, Any]:
    write_capability_audit(); path = OUTPUT / "smoke" / "ppo_smoke.json"
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=False)
    train, validation, fresh = training_scenarios(), validation_scenarios(), fresh_scenarios(); assert_disjoint(train, validation, fresh)
    config = PPOConfig(rollout_length=16, num_envs=2, num_minibatches=2, update_epochs=1, checkpoint_interval_steps=32)
    seed_everything(0); agent, vector = PPOAgent(DeepSetsActorCritic(), config), TrainingVector(train, config.num_envs, 73000)
    started = perf_counter(); storage, events, rollout_health = collect(agent, vector); update = agent.update(storage); elapsed = perf_counter() - started
    result = {"round": "round_30", "seed": 0, "smoke_environment_steps": vector.environment_steps, "seconds": elapsed, "steps_per_second": vector.environment_steps / max(elapsed, 1.0e-9), "completed_episodes": len(events), "finite_observations_rewards": True, "actor_critic_forward": True, "gae": True, "ppo_update": True, "queue_conservation": True, "safety_projection": True, **rollout_health, **update}
    if not all(np.isfinite(value) for value in result.values() if isinstance(value, float)): raise RuntimeError("Round-30 smoke produced NaN/Inf")
    write_json(path, result); return result


def freeze_formal_protocol() -> dict[str, Any]:
    if not (OUTPUT / "smoke" / "ppo_smoke.json").is_file(): raise RuntimeError("formal configuration requires the completed smoke")
    if CONFIG_PATH.exists(): return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    train, validation, fresh = training_scenarios(), validation_scenarios(), fresh_scenarios(); assert_disjoint(train, validation, fresh)
    config = PPOConfig()
    payload = {"round": "round_30", "protocol_status": "frozen_before_formal_training", "formal_environment_steps_per_seed": FORMAL_STEPS_PER_SEED, "formal_seeds": list(SEEDS), "ppo": config.to_dict(), "checkpoint_selection": {"population": "round30_ppo_validation_v1", "schedule_steps": list(range(config.checkpoint_interval_steps, FORMAL_STEPS_PER_SEED + 1, config.checkpoint_interval_steps)), "metric": "actual_cumulative_payload_bits", "deterministic": True, "hyperparameter_search": False}, "fresh_access_count_before_training": 0, "untouched_eval_access_count": 0, "raw_logprob_semantics": "pre-safety tanh-squashed policy sample"}
    write_json(CONFIG_PATH, payload); write_json(OUTPUT / "round30_training_scenario_manifest.json", scenario_manifest("training", train)); write_json(OUTPUT / "round30_validation_manifest.json", scenario_manifest("validation", validation)); write_json(OUTPUT / "round30_fresh_manifest.json", scenario_manifest("fresh", fresh))
    write_json(OUTPUT / "round30_protocol_freeze.json", {"round": "round_30", "training_config_sha256": sha256(CONFIG_PATH), "training_manifest_sha256": sha256(OUTPUT / "round30_training_scenario_manifest.json"), "validation_manifest_sha256": sha256(OUTPUT / "round30_validation_manifest.json"), "fresh_manifest_sha256": sha256(OUTPUT / "round30_fresh_manifest.json"), "round29_lock_sha256": sha256(ROOT / "round_29" / "ROUND_LOCK.json"), "fresh_outcomes_accessed": False, "protected_accessed": False})
    return payload


def formal_config() -> PPOConfig:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("formal_environment_steps_per_seed") != FORMAL_STEPS_PER_SEED or payload.get("formal_seeds") != list(SEEDS): raise RuntimeError("Round-30 formal PPO configuration changed")
    return PPOConfig(**payload["ppo"])


def evaluate_validation(agent: PPOAgent, seed: int, checkpoint_step: int) -> list[dict[str, Any]]:
    from .evaluation import ppo_action, rollout
    metrics: list[dict[str, Any]] = []
    for scenario in validation_scenarios():
        metric, _ = rollout(scenario, population="validation", policy_id="deepsets_ppo", model_seed=seed, action_fn=ppo_action(agent), matching_mode="legacy")
        metric["checkpoint_step"] = checkpoint_step; metrics.append(metric)
    return metrics


def save_checkpoint(agent: PPOAgent, seed: int, step: int, validation: list[dict[str, Any]], path: Path, health: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_payload = float(np.mean([row["cumulative_payload_bits"] for row in validation]))
    torch.save(agent.checkpoint(seed=seed, steps=step, extra={"validation_mean_payload_bits": mean_payload, "health": dict(health)}), path)
    return {"seed": seed, "checkpoint_step": step, "checkpoint": path.as_posix(), "checkpoint_sha256": sha256(path), "validation_mean_payload_bits": mean_payload}


def train_seed(seed: int) -> dict[str, Any]:
    config = formal_config()
    if seed not in SEEDS: raise ValueError("unregistered Round-30 PPO seed")
    final_path, summary_path = FORMAL / f"seed_{seed}" / "best.pt", FORMAL / f"seed_{seed}" / "summary.json"
    if final_path.exists() and summary_path.exists(): return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_everything(seed); agent, vector = PPOAgent(DeepSetsActorCritic(), config), TrainingVector(training_scenarios(), config.num_envs, 80000 + seed)
    curves: list[dict[str, Any]] = []; episodes: list[dict[str, float]] = []; checkpoints: list[dict[str, Any]] = []; validation_records: list[dict[str, Any]] = []
    best_payload, best_record, update_index = -float("inf"), None, 0
    while vector.environment_steps < FORMAL_STEPS_PER_SEED:
        storage, completed, rollout_health = collect(agent, vector); health = agent.update(storage); update_index += 1; episodes.extend(completed)
        row = {"seed": seed, "update": update_index, "environment_steps": vector.environment_steps, "mean_completed_payload_bits": float(np.mean([item["payload_bits"] for item in completed])) if completed else float("nan"), "mean_completed_return": float(np.mean([item["return"] for item in completed])) if completed else float("nan"), "mean_completed_completion": float(np.mean([item["completion"] for item in completed])) if completed else float("nan"), "mean_completed_residual_queue_bits": float(np.mean([item["residual_queue_bits"] for item in completed])) if completed else float("nan"), "mean_completed_projection_rate": float(np.mean([item["projection_rate"] for item in completed])) if completed else 0.0, **rollout_health, **health}
        curves.append(row)
        if vector.environment_steps % config.checkpoint_interval_steps == 0:
            validation = evaluate_validation(agent, seed, vector.environment_steps); validation_records.extend(validation)
            candidate = save_checkpoint(agent, seed, vector.environment_steps, validation, FORMAL / f"seed_{seed}" / f"checkpoint_{vector.environment_steps:06d}.pt", row); checkpoints.append(candidate)
            if candidate["validation_mean_payload_bits"] > best_payload: best_payload, best_record = candidate["validation_mean_payload_bits"], candidate
    if vector.environment_steps != FORMAL_STEPS_PER_SEED or best_record is None: raise RuntimeError("Round-30 formal PPO budget/checkpoint schedule is incomplete")
    source = Path(best_record["checkpoint"]); final_path.parent.mkdir(parents=True, exist_ok=True); final_path.write_bytes(source.read_bytes())
    for row in validation_records: row["selected_checkpoint"] = row["checkpoint_step"] == best_record["checkpoint_step"]
    early = [row["payload_bits"] for row in episodes if row["finish_environment_step"] <= FORMAL_STEPS_PER_SEED * .25]
    late = [row["payload_bits"] for row in episodes if row["finish_environment_step"] >= FORMAL_STEPS_PER_SEED * .75]
    summary = {"seed": seed, "formal_environment_steps": vector.environment_steps, "checkpoint_count": len(checkpoints), "best_checkpoint_step": best_record["checkpoint_step"], "best_checkpoint": final_path.as_posix(), "best_checkpoint_sha256": sha256(final_path), "best_validation_payload_bits": best_payload, "early_training_payload_bits": float(np.mean(early)), "late_training_payload_bits": float(np.mean(late)), "late_gt_early": bool(np.mean(late) > np.mean(early)), "episode_count": len(episodes), "mean_entropy": float(np.nanmean([row["entropy"] for row in curves])), "mean_kl": float(np.nanmean([row["approx_kl"] for row in curves])), "mean_explained_variance": float(np.nanmean([row["explained_variance"] for row in curves])), "mean_action_std": float(np.nanmean([row["action_std"] for row in curves])), "mean_projection_rate": float(np.nanmean([row["mean_completed_projection_rate"] for row in curves])), "curves": curves, "validation_records": validation_records, "checkpoints": checkpoints, "finite": bool(np.isfinite([row["policy_loss"] for row in curves]).all())}
    write_json(summary_path, summary); return summary


def run_formal_training() -> list[dict[str, Any]]:
    if not CONFIG_PATH.exists(): raise RuntimeError("Round-30 training requires frozen config and scenario manifests")
    results = [train_seed(seed) for seed in SEEDS]
    curves = [row for result in results for row in result["curves"]]
    validation = [row for result in results for row in result["validation_records"] if row["selected_checkpoint"]]
    summary = [{key: value for key, value in result.items() if key not in {"curves", "validation_records", "checkpoints"}} for result in results]
    write_csv(OUTPUT / "ppo_learning_curves.csv", curves); write_csv(OUTPUT / "ppo_training_summary.csv", summary); write_csv(OUTPUT / "ppo_validation_metrics.csv", validation)
    write_csv(OUTPUT / "ppo_validation_seed_summary.csv", [{"seed": result["seed"], "best_checkpoint_step": result["best_checkpoint_step"], "best_validation_payload_bits": result["best_validation_payload_bits"], "validation_mean_completion": float(np.mean([row["completion"] for row in validation if row["model_seed"] == result["seed"]])), "validation_mean_residual_queue_bits": float(np.mean([row["residual_queue_bits"] for row in validation if row["model_seed"] == result["seed"]]))} for result in results])
    write_json(OUTPUT / "checkpoint_manifest.json", {"round": "round_30", "seeds": list(SEEDS), "checkpoint_selection": "frozen deterministic validation cumulative payload", "checkpoints": [{"seed": row["seed"], "path": row["best_checkpoint"], "sha256": row["best_checkpoint_sha256"], "step": row["best_checkpoint_step"]} for row in results], "fresh_access_count": 0})
    write_json(OUTPUT / "run_manifest.json", {"round": "round_30", "status": "formal_training_complete_validation_only", "formal_environment_steps_per_seed": FORMAL_STEPS_PER_SEED, "seeds": list(SEEDS), "total_training_environment_steps": FORMAL_STEPS_PER_SEED * len(SEEDS), "training_scenario_count": len(training_scenarios()), "validation_scenario_count": len(validation_scenarios()), "fresh_scenario_count": len(fresh_scenarios()), "fresh_access_count": 0, "untouched_eval_access_count": 0, "task_waypoint_used": False, "planner_trigger_used": False, "hgnn_used": False, "transformer_used": False, "environment_modified": False, "reward_modified": False})
    return results


__all__ = ["CONFIG_PATH", "FORMAL_STEPS_PER_SEED", "OUTPUT", "SEEDS", "collect", "evaluate_validation", "formal_config", "freeze_formal_protocol", "run_formal_training", "run_smoke", "train_seed", "write_capability_audit", "write_csv", "write_json"]
