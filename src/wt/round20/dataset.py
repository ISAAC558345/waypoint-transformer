"""Deterministic Round-20 trajectory collection and Task-Waypoint labels."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round19.planner import independent_decision, joint_plan

from .scenarios import DevelopmentScenario, EPISODE_HORIZON, development_scenarios, split_summary


MAX_UAVS = 2
MAX_BUOYS = 5
GAMMA = 0.99
SAFETY_NUMERICAL_TOLERANCE_M = 1.0e-5
GRAPH_SCHEMA_VERSION = "multi_uav_graph_schema_v1"
DATASET_SCHEMA_VERSION = "multi_uav_shared_spectrum_dataset_v1"
TASK_SCHEMA_VERSION = "multi_uav_task_waypoint_v1"
UAV_FEATURE_NAMES = (
    "position_x_m", "position_y_m", "velocity_x_mps", "velocity_y_mps",
    "remaining_energy", "cumulative_path_length_m", "current_served_payload_bits",
    "has_current_assignment",
)
BUOY_FEATURE_NAMES = (
    "position_x_m", "position_y_m", "queue_bits", "arrival_bits",
    "completion_ratio", "cumulative_served_payload_bits", "has_current_assignment",
)
UB_FEATURE_NAMES = (
    "horizontal_distance_m", "three_dimensional_distance_m", "geometric_path_gain",
    "full_bandwidth_snr_linear", "direct_achievable_rate_bps",
    "queue_weighted_direct_rate_bps", "cross_interference_potential_w",
    "current_assignment_relation",
)
UU_FEATURE_NAMES = (
    "uav_distance_m", "relative_x_m", "relative_y_m", "relative_velocity_x_mps",
    "relative_velocity_y_mps", "safety_margin_m", "both_currently_assigned_distinct",
    "receiver_cross_interference_power_w", "receiver_interference_to_signal_ratio",
)
GLOBAL_FEATURE_NAMES = (
    "time_fraction", "interference_overlap_factor", "shared_spectrum_active",
    "queue_capacity_bits", "num_uavs", "num_buoys",
)


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    family: str
    base_horizon: int | None = None
    noise_level: float = 0.0


BEHAVIORS = (
    BehaviorSpec("independent_queue_rate", "independent"),
    BehaviorSpec("joint_h1", "joint", 1),
    BehaviorSpec("joint_h5", "joint", 5),
    BehaviorSpec("joint_h10", "joint", 10),
    BehaviorSpec("noisy_joint_h1_noise_0.15", "noisy_joint", 1, .15),
    BehaviorSpec("noisy_joint_h1_noise_0.35", "noisy_joint", 1, .35),
    BehaviorSpec("noisy_joint_h10_noise_0.15", "noisy_joint", 10, .15),
    BehaviorSpec("noisy_joint_h10_noise_0.35", "noisy_joint", 10, .35),
    BehaviorSpec("policy_switch_independent_h1_h10", "policy_switch"),
    BehaviorSpec("random_valid_joint", "random"),
)


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(plain(row), sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reconstruct_formal_result(output: Path) -> dict[str, Any]:
    """Rebuild an audit/report view from formal raw files without a rollout."""
    output = output.resolve()
    manifest = json.loads((output / "multi_uav_dataset_manifest.json").read_text(encoding="utf-8"))
    scenarios = [SimpleNamespace(**row) for row in manifest["scenarios"]]
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    behavior_by_name = {item.name: item for item in BEHAVIORS}
    with (output / "multi_uav_episode_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        metrics_by_id = {row["episode_id"]: row for row in csv.DictReader(handle)}
    transitions = read_jsonl(output / "multi_uav_transitions.jsonl")
    samples = read_jsonl(output / "multi_uav_task_waypoint_samples.jsonl")
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for transition in transitions:
        by_episode.setdefault(transition["episode_id"], []).append(transition)
    episodes: list[dict[str, Any]] = []
    for episode_id, steps in by_episode.items():
        steps.sort(key=lambda item: int(item["timestep"]))
        first, last = steps[0], steps[-1]
        scenario = scenario_by_id[first["scenario_id"]]
        metrics = metrics_by_id[episode_id]
        payload_bits = float(metrics["payload_bits"])
        offered = float(sum(scenario.initial_queue_bits) + sum(sum(item["arrival_bits"]) for item in steps))
        minimum_distance = min(float(item["min_uav_distance_m"]) for item in steps)
        episodes.append({
            "episode_id": episode_id, "scenario": scenario, "behavior": behavior_by_name[first["behavior_generator"]],
            "behavior_seed": int(first["behavior_seed"]), "transitions": steps, "payload_bits": payload_bits,
            "payload_mbit": payload_bits / 1.0e6, "completion": payload_bits / max(offered, 1.0),
            "residual_queue_bits": float(sum(last["queue_after_bits"])), "return": float(sum(float(item["reward_t"]) for item in steps)),
            "steps": len(steps), "nodes_expanded": int(metrics["nodes_expanded"]), "decision_runtime_seconds": float(metrics["decision_runtime_seconds"]),
            "minimum_uav_distance_m": minimum_distance,
            "safety_violation_count": sum(int(float(item["min_uav_distance_m"]) < 15.0 - SAFETY_NUMERICAL_TOLERANCE_M) for item in steps),
            "safety_projection_fallback_count": sum(int(item["safety_projection_fallback"]) for item in steps),
            "quality_tier": first["quality_tier"], "quality_rank_within_scenario": int(metrics["quality_rank_within_scenario"]),
        })
    return {"output": output, "scenarios": scenarios, "episodes": episodes, "transitions": transitions, "samples": samples, "metrics": list(metrics_by_id.values())}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(plain(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for key, value in row.items()})


def graph_schema() -> dict[str, Any]:
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "logical_schema": {
            "uav_nodes": ["M", len(UAV_FEATURE_NAMES)],
            "buoy_nodes": ["K", len(BUOY_FEATURE_NAMES)],
            "ub_edges": ["M", "K", len(UB_FEATURE_NAMES)],
            "uu_edges": ["M", "M", len(UU_FEATURE_NAMES)],
            "uav_mask": ["M"],
            "buoy_mask": ["K"],
            "global_features": [len(GLOBAL_FEATURE_NAMES)],
        },
        "serialized_padding": {"max_num_uavs": MAX_UAVS, "max_num_buoys": MAX_BUOYS},
        "features": {
            "uav_nodes": list(UAV_FEATURE_NAMES), "buoy_nodes": list(BUOY_FEATURE_NAMES),
            "ub_edges": list(UB_FEATURE_NAMES), "uu_edges": list(UU_FEATURE_NAMES),
            "global_features": list(GLOBAL_FEATURE_NAMES),
        },
        "units": {
            "position_x_m": "m", "position_y_m": "m", "velocity_x_mps": "m/s",
            "velocity_y_mps": "m/s", "remaining_energy": "environment energy units",
            "cumulative_path_length_m": "m", "current_served_payload_bits": "bits",
            "queue_bits": "bits", "arrival_bits": "bits", "cumulative_served_payload_bits": "bits",
            "horizontal_distance_m": "m", "three_dimensional_distance_m": "m",
            "geometric_path_gain": "linear gain", "full_bandwidth_snr_linear": "linear",
            "direct_achievable_rate_bps": "bits/s", "queue_weighted_direct_rate_bps": "bits/s",
            "cross_interference_potential_w": "W", "uav_distance_m": "m",
            "relative_x_m": "m", "relative_y_m": "m", "relative_velocity_x_mps": "m/s",
            "relative_velocity_y_mps": "m/s", "safety_margin_m": "m",
            "receiver_cross_interference_power_w": "W",
            "receiver_interference_to_signal_ratio": "linear",
        },
        "normalization": {
            "graph_input": "physical values are serialized in native units; normalized state-overlap vectors are audit-only",
            "metadata_excluded_from_default_model_input": ["behavior_generator", "quality_tier", "spectrum_tier", "split", "scenario_id", "seed"],
        },
        "future_variable_m_compatible": True,
        "fixed_agent_identity_feature_forbidden": True,
    }


def dataset_schema() -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "transition_required_fields": [
            "episode_id", "scenario_id", "seed", "split", "timestep", "behavior_generator",
            "behavior_seed", "quality_tier", "interference_tier", "num_uavs", "num_buoys",
            "joint_observation_t", "joint_action_t", "joint_action_applied_t", "reward_t",
            "joint_observation_t_plus_1", "terminated", "truncated", "graph_state_t",
        ],
        "graph_state_contains_online_observable_only": True,
        "offline_supervision_stored_separately": True,
        "quality_and_behavior_are_metadata_not_default_model_input": True,
    }


def task_waypoint_schema() -> dict[str, Any]:
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "current_input": "current_graph_state with current uav/buoy masks only",
        "offline_targets": {
            "per_uav_service_responsibility_target": ["M", "K_max"],
            "service_valid_uav_mask": ["M"],
            "per_uav_gross_service_progress_bits": ["M", "K_max"],
            "per_uav_gross_service_progress_normalized": ["M", "K_max"],
            "team_h_step_return_raw": [],
            "team_h_step_return_normalized": [],
        },
        "horizons": [5, 10], "discount_gamma": GAMMA,
        "padding_semantics": {"padded_buoy_probability": 0.0, "padded_buoy_loss_mask": 0.0},
        "shared_queue_conservation": "sum over UAV gross progress equals actual future buoy service within numerical tolerance",
        "future_values_are_supervision_only": True,
    }


def _pad(values: np.ndarray, target: tuple[int, ...], fill: float = 0.0, dtype: Any = np.float64) -> np.ndarray:
    result = np.full(target, fill, dtype=dtype)
    slices = tuple(slice(0, size) for size in values.shape)
    result[slices] = values
    return result


def build_graph_state(info: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """Build only current observable physical graph fields, with K padding."""
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64)
    velocities = np.asarray(info["uav_velocity_mps"], dtype=np.float64)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64)
    arrivals = np.asarray(info["raw_arrival_bits"], dtype=np.float64)
    energy = np.asarray(info["remaining_energy_per_uav"], dtype=np.float64)
    path = np.asarray(info["cumulative_distance_m_per_uav"], dtype=np.float64)
    payload_by_uav = np.asarray(info["transmitted_payload_bits_by_uav"], dtype=np.float64)
    payload_by_buoy = np.asarray(info["cumulative_payload_bits_per_buoy"], dtype=np.float64)
    assignment = np.asarray(info["uav_to_buoy_assignment"], dtype=np.int64)
    matched = np.asarray(info["matched_uav_by_buoy"], dtype=np.int64)
    gains = np.asarray(info["uav_buoy_channel_gains"], dtype=np.float64)
    cross_power = np.asarray(info["cross_interference_power_w"], dtype=np.float64)
    isr = np.asarray(info["interference_to_signal_ratio"], dtype=np.float64)
    m_count, k_count = positions.shape[0], buoys.shape[0]
    if m_count > MAX_UAVS or k_count > MAX_BUOYS:
        raise ValueError("Round-20 storage padding exceeded")
    if config.arrival_mode != "constant":
        raise ValueError("Round-20 graph completion feature requires the frozen constant arrival mode")
    cumulative_arrivals = np.full(k_count, float(info["time_step"]) * float(config.constant_arrival_bits_per_step), dtype=np.float64)
    offered = np.asarray(config.initial_queue_bits_by_buoy, dtype=np.float64) + cumulative_arrivals
    completion = np.divide(payload_by_buoy, offered, out=np.zeros_like(payload_by_buoy), where=offered > 1.0e-12)
    has_assignment = (assignment >= 0).astype(np.float64)
    uav_nodes = np.column_stack((positions, velocities, energy, path, payload_by_uav, has_assignment))
    buoy_nodes = np.column_stack((buoys, queues, arrivals, np.clip(completion, 0.0, 1.0), payload_by_buoy, (matched >= 0).astype(np.float64)))
    rel = buoys[None, :, :] - positions[:, None, :]
    horizontal = np.linalg.norm(rel, axis=-1)
    three_dimensional = np.sqrt(horizontal * horizontal + float(config.uav_altitude) ** 2)
    snr = float(config.buoy_tx_power_w) * gains / float(config.noise_power_w)
    direct_rate = float(config.total_bandwidth_hz) * np.log2(1.0 + snr)
    weighted_rate = queues[None, :] / float(config.queue_capacity_bits) * direct_rate
    potential = float(info["interference_overlap_factor"]) * float(config.buoy_tx_power_w) * gains
    current_relation = np.zeros((m_count, k_count), dtype=np.float64)
    for m, k in enumerate(assignment):
        if k >= 0:
            current_relation[m, k] = 1.0
    ub_edges = np.stack((horizontal, three_dimensional, gains, snr, direct_rate, weighted_rate, potential, current_relation), axis=-1)
    uu_edges = np.zeros((m_count, m_count, len(UU_FEATURE_NAMES)), dtype=np.float64)
    for receiver in range(m_count):
        for other in range(m_count):
            if receiver == other:
                continue
            delta = positions[other] - positions[receiver]
            rel_velocity = velocities[other] - velocities[receiver]
            distance = float(np.linalg.norm(delta))
            distinct = float(assignment[receiver] >= 0 and assignment[other] >= 0 and assignment[receiver] != assignment[other])
            uu_edges[receiver, other] = (distance, delta[0], delta[1], rel_velocity[0], rel_velocity[1], distance - float(config.multi_uav_min_separation_m), distinct, cross_power[receiver], isr[receiver])
    time_fraction = float(info["time_step"]) / float(config.episode_horizon)
    global_features = np.array((time_fraction, float(info["interference_overlap_factor"]), float(info["spectrum_mode"] == "shared"), float(config.queue_capacity_bits), float(m_count), float(k_count)), dtype=np.float64)
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "num_uavs": int(m_count), "num_buoys": int(k_count),
        "uav_nodes": _pad(uav_nodes, (MAX_UAVS, len(UAV_FEATURE_NAMES))),
        "buoy_nodes": _pad(buoy_nodes, (MAX_BUOYS, len(BUOY_FEATURE_NAMES))),
        "ub_edges": _pad(ub_edges, (MAX_UAVS, MAX_BUOYS, len(UB_FEATURE_NAMES))),
        "uu_edges": _pad(uu_edges, (MAX_UAVS, MAX_UAVS, len(UU_FEATURE_NAMES))),
        "uav_mask": np.arange(MAX_UAVS) < m_count,
        "buoy_mask": np.arange(MAX_BUOYS) < k_count,
        "global_features": global_features,
        "uav_to_buoy_assignment": _pad(assignment, (MAX_UAVS,), fill=-1, dtype=np.int64),
        "matched_uav_by_buoy": _pad(matched, (MAX_BUOYS,), fill=-1, dtype=np.int64),
    }


def _safe_random_action(rng: np.random.Generator, num_uavs: int) -> np.ndarray:
    action = rng.uniform(-1.0, 1.0, size=(num_uavs, 2))
    norms = np.linalg.norm(action, axis=1, keepdims=True)
    return action / np.maximum(norms, 1.0)


def _switch_base(timestep: int) -> str:
    if timestep < 4:
        return "independent_queue_rate"
    if timestep < 8:
        return "joint_h1"
    return "joint_h10"


def behavior_action(spec: BehaviorSpec, env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any], timestep: int, rng: np.random.Generator) -> tuple[np.ndarray, str, str, int, float]:
    """Return a legal command and current-only decision metadata."""
    active = spec.name
    if spec.family == "random":
        return _safe_random_action(rng, env.num_uavs), active, "joint_aware", 0, 0.0
    if spec.family == "policy_switch":
        active = _switch_base(timestep)
    if active == "independent_queue_rate":
        decision = independent_decision(env, info)
        return decision.action, active, "legacy", decision.nodes_expanded, decision.runtime_seconds
    horizon = int(spec.base_horizon) if spec.family == "noisy_joint" else 1 if active == "joint_h1" else 5 if active == "joint_h5" else 10
    decision = joint_plan(env, info, horizon=horizon)
    action = decision.action.copy()
    if spec.noise_level > 0.0:
        action = np.clip(action + rng.normal(0.0, spec.noise_level, size=action.shape), -1.0, 1.0)
        norms = np.linalg.norm(action, axis=1, keepdims=True)
        action = action / np.maximum(norms, 1.0)
    return action, active, "joint_aware", decision.nodes_expanded, decision.runtime_seconds


def _service_matrix(info: Mapping[str, Any], max_buoys: int = MAX_BUOYS) -> np.ndarray:
    assignment = np.asarray(info["uav_to_buoy_assignment"], dtype=np.int64)
    payload = np.asarray(info["transmitted_payload_bits_by_uav"], dtype=np.float64)
    matrix = np.zeros((assignment.size, max_buoys), dtype=np.float64)
    for uav, buoy in enumerate(assignment):
        if buoy >= 0:
            matrix[uav, buoy] = payload[uav]
    return matrix


def run_episode(scenario: DevelopmentScenario, spec: BehaviorSpec) -> dict[str, Any]:
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    _, info = env.reset(seed=scenario.seed)
    behavior_index = BEHAVIORS.index(spec)
    behavior_seed = int(scenario.seed * 100 + behavior_index)
    rng = np.random.default_rng(behavior_seed)
    transitions: list[dict[str, Any]] = []
    total_reward = 0.0
    nodes_expanded = 0
    runtime_seconds = 0.0
    done = False
    episode_id = f"{scenario.scenario_id}__{spec.name}"
    while not done:
        timestep = int(info["time_step"])
        graph_t = build_graph_state(info, env.config)
        observation_t = np.asarray(env._shared_observation(), dtype=np.float32)
        action, active_behavior, matching_mode, nodes, runtime = behavior_action(spec, env, info, timestep, rng)
        requested_action = np.asarray(action, dtype=np.float64).copy()
        safety_projection_fallback = False
        try:
            observation_next, reward, terminated, truncated, after = env.step(action, matching_mode=matching_mode)
        except RuntimeError as error:
            # The frozen Round-18 projection is the only safety authority.  A
            # rejected joint command is never stored as a transition; retry
            # the current state with its already valid zero movement instead.
            if "hard UAV separation projection did not converge" not in str(error):
                raise
            action = np.zeros_like(requested_action)
            observation_next, reward, terminated, truncated, after = env.step(action, matching_mode=matching_mode)
            safety_projection_fallback = True
        graph_next = build_graph_state(after, env.config)
        nodes_expanded += nodes
        runtime_seconds += runtime
        total_reward += float(reward)
        transition = {
            "episode_id": episode_id, "scenario_id": scenario.scenario_id, "seed": scenario.seed,
            "split": scenario.split, "timestep": timestep, "behavior_generator": spec.name,
            "behavior_family": spec.family, "active_behavior": active_behavior,
            "behavior_seed": behavior_seed, "quality_tier": "unassigned",
            "interference_tier": scenario.spectrum_tier, "num_uavs": scenario.num_uavs,
            "num_buoys": scenario.num_buoys, "joint_observation_t": observation_t,
            "joint_action_t": np.asarray(action, dtype=np.float64),
            "requested_joint_action_t": requested_action,
            "joint_action_applied_t": np.asarray(after["normalized_applied_action"], dtype=np.float64),
            "safety_projection_fallback": safety_projection_fallback,
            "reward_t": float(reward), "joint_observation_t_plus_1": np.asarray(observation_next, dtype=np.float32),
            "terminated": bool(terminated), "truncated": bool(truncated), "graph_state_t": graph_t,
            "graph_state_t_plus_1": graph_next,
            "service_by_uav_buoy_bits": _service_matrix(after),
            "assignment_after": np.asarray(after["uav_to_buoy_assignment"], dtype=np.int64),
            "payload_by_buoy_bits": np.asarray(after["transmitted_payload_bits"], dtype=np.float64),
            "payload_by_uav_bits": np.asarray(after["transmitted_payload_bits_by_uav"], dtype=np.float64),
            "queue_before_bits": np.asarray(after["queue_before_bits"], dtype=np.float64),
            "queue_after_bits": np.asarray(after["raw_queue_bits"], dtype=np.float64),
            "arrival_bits": np.asarray(after["raw_arrival_bits"], dtype=np.float64),
            "link_capacity_bits": np.asarray(after["link_capacity_bits"], dtype=np.float64),
            "sinr_linear": np.asarray(after["sinr_linear"], dtype=np.float64),
            "interference_to_signal_ratio": np.asarray(after["interference_to_signal_ratio"], dtype=np.float64),
            "interference_conflict": bool(after["interference_conflict"]),
            "min_uav_distance_m": float(after["min_uav_distance_m"]),
            "safety_correction_m": np.asarray(after["safety_correction_m"], dtype=np.float64),
            "queue_conservation_residual_bits": np.asarray(after["queue_conservation_residual_bits"], dtype=np.float64),
            "raw_collision_risk_pairs": int(after["raw_collision_risk_pairs"]),
        }
        transitions.append(transition)
        info = after
        done = bool(terminated or truncated)
    if len(transitions) != EPISODE_HORIZON:
        raise RuntimeError("Round-20 protocol requires fixed 12-step trajectories")
    payload_bits = float(info["cumulative_payload_bits"])
    offered = float(np.sum(scenario.initial_queue_bits) + np.sum([np.sum(item["arrival_bits"]) for item in transitions]))
    completion = payload_bits / max(offered, 1.0)
    return {
        "episode_id": episode_id, "scenario": scenario, "behavior": spec,
        "behavior_seed": behavior_seed, "transitions": transitions, "payload_bits": payload_bits,
        "payload_mbit": payload_bits / 1.0e6, "completion": completion,
        "residual_queue_bits": float(np.sum(info["raw_queue_bits"])), "return": total_reward,
        "steps": len(transitions), "nodes_expanded": nodes_expanded,
        "decision_runtime_seconds": runtime_seconds,
        "minimum_uav_distance_m": min(item["min_uav_distance_m"] for item in transitions),
        "safety_violation_count": sum(int(item["min_uav_distance_m"] < scenario.config().multi_uav_min_separation_m - SAFETY_NUMERICAL_TOLERANCE_M) for item in transitions),
        "safety_projection_fallback_count": sum(int(item["safety_projection_fallback"]) for item in transitions),
    }


def label_quality(episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        groups.setdefault(episode["scenario"].scenario_id, []).append(episode)
    for scenario_id, group in sorted(groups.items()):
        if len(group) != len(BEHAVIORS):
            raise RuntimeError(f"scenario {scenario_id} has incomplete behavior coverage")
        ranked = sorted(group, key=lambda item: (-item["payload_bits"], -item["completion"], item["residual_queue_bits"], -item["return"], item["episode_id"]))
        for rank, episode in enumerate(ranked):
            tier = "high" if rank < 4 else "medium" if rank < 8 else "low"
            episode["quality_tier"] = tier
            episode["quality_rank_within_scenario"] = rank + 1
            for transition in episode["transitions"]:
                transition["quality_tier"] = tier
            scenario = episode["scenario"]
            result.append({
                "episode_id": episode["episode_id"], "scenario_id": scenario.scenario_id,
                "split": scenario.split, "behavior_generator": episode["behavior"].name,
                "quality_tier": tier, "quality_rank_within_scenario": rank + 1,
                "payload_mbit": episode["payload_mbit"], "completion": episode["completion"],
                "residual_queue_bits": episode["residual_queue_bits"], "return": episode["return"],
                "safety_violation_count": episode["safety_violation_count"],
            })
    return result


def build_task_waypoint_samples(episodes: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_samples: list[dict[str, Any]] = []
    for episode in episodes:
        transitions = episode["transitions"]
        scenario = episode["scenario"]
        for horizon in (5, 10):
            for start in range(0, len(transitions) - horizon + 1):
                window = transitions[start:start + horizon]
                if any(item["terminated"] or item["truncated"] for item in window[:-1]):
                    continue
                service = np.sum(np.asarray([item["service_by_uav_buoy_bits"] for item in window], dtype=np.float64), axis=0)
                service_total = np.sum(service, axis=1)
                responsibility = np.divide(service, service_total[:, None], out=np.zeros_like(service), where=service_total[:, None] > 1.0e-12)
                valid = service_total > 1.0e-12
                return_raw = float(sum((GAMMA ** index) * float(item["reward_t"]) for index, item in enumerate(window)))
                sample = {
                    "sample_id": f"{episode['episode_id']}__t{start:02d}__h{horizon}",
                    "episode_id": episode["episode_id"], "scenario_id": scenario.scenario_id,
                    "split": scenario.split, "t": start, "H_task": horizon,
                    "current_graph_state": transitions[start]["graph_state_t"],
                    "uav_mask": transitions[start]["graph_state_t"]["uav_mask"],
                    "buoy_mask": transitions[start]["graph_state_t"]["buoy_mask"],
                    "per_uav_service_responsibility_target": responsibility,
                    "service_valid_uav_mask": valid,
                    "per_uav_gross_service_progress_bits": service,
                    "per_uav_gross_service_progress_normalized": service / float(scenario.config().queue_capacity_bits),
                    "future_actual_buoy_service_bits": np.sum(service, axis=0),
                    "team_h_step_return_raw": return_raw,
                    "quality_tier": episode["quality_tier"], "behavior_generator": episode["behavior"].name,
                    "interference_tier": scenario.spectrum_tier,
                    "future_assignment_signature": [item["assignment_after"] for item in window],
                }
                raw_samples.append(sample)
    train_returns = np.asarray([item["team_h_step_return_raw"] for item in raw_samples if item["split"] == "train"], dtype=np.float64)
    if train_returns.size == 0 or not np.all(np.isfinite(train_returns)):
        raise RuntimeError("Task-Waypoint normalization requires finite train samples")
    mean = float(train_returns.mean())
    std = float(train_returns.std())
    scale = std if std > 1.0e-12 else 1.0
    for item in raw_samples:
        item["team_h_step_return_normalized"] = (float(item["team_h_step_return_raw"]) - mean) / scale
    normalization = {
        "schema_version": TASK_SCHEMA_VERSION,
        "service_responsibility": "probability distribution; zero vector with service_valid_uav_mask=0 when a UAV has no future service",
        "gross_service_progress": {"normalization": "divide by configured queue_capacity_bits", "unit": "bits"},
        "team_h_step_return": {"normalization": "train_split_zscore", "train_sample_count": int(train_returns.size), "mean": mean, "std": std, "scale_used": scale, "validation_or_test_statistics_used": False},
        "discount_gamma": GAMMA,
    }
    return raw_samples, normalization


def _transition_hash(episode: Mapping[str, Any]) -> str:
    compact = [{"observation": item["joint_observation_t"], "action": item["joint_action_t"], "reward": item["reward_t"], "assignment": item["assignment_after"]} for item in episode["transitions"]]
    return hashlib.sha256(json.dumps(plain(compact), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()


def episode_metrics(episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        scenario = episode["scenario"]
        rows.append({
            "episode_id": episode["episode_id"], "scenario_id": scenario.scenario_id, "seed": scenario.seed,
            "split": scenario.split, "behavior_generator": episode["behavior"].name,
            "behavior_family": episode["behavior"].family, "behavior_seed": episode["behavior_seed"],
            "quality_tier": episode["quality_tier"], "quality_rank_within_scenario": episode["quality_rank_within_scenario"],
            "interference_tier": scenario.spectrum_tier, "num_uavs": scenario.num_uavs,
            "num_buoys": scenario.num_buoys, "queue_tier": scenario.queue_tier,
            "geometry_group": scenario.geometry_group, "coordination_stress": scenario.coordination_stress,
            "payload_bits": episode["payload_bits"], "payload_mbit": episode["payload_mbit"],
            "completion": episode["completion"], "residual_queue_bits": episode["residual_queue_bits"],
            "return": episode["return"], "steps": episode["steps"],
            "nodes_expanded": episode["nodes_expanded"], "decision_runtime_seconds": episode["decision_runtime_seconds"],
            "minimum_uav_distance_m": episode["minimum_uav_distance_m"],
            "safety_violation_count": episode["safety_violation_count"], "safety_projection_fallback_count": episode["safety_projection_fallback_count"], "trajectory_sha256": _transition_hash(episode),
        })
    return rows


def behavior_distribution(episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in BEHAVIORS:
        group = [item for item in episodes if item["behavior"].name == spec.name]
        rows.append({"behavior_generator": spec.name, "behavior_family": spec.family, "episode_count": len(group), "transition_count": sum(item["steps"] for item in group), "mean_payload_mbit": float(np.mean([item["payload_mbit"] for item in group])), "mean_completion": float(np.mean([item["completion"] for item in group])), "mean_residual_queue_bits": float(np.mean([item["residual_queue_bits"] for item in group])), "mean_return": float(np.mean([item["return"] for item in group]))})
    return rows


def generate_dataset(root: Path, output: Path, scenarios: Sequence[DevelopmentScenario] | None = None) -> dict[str, Any]:
    """Freeze a manifest, then generate deterministic non-learning development data."""
    output.mkdir(parents=True, exist_ok=False)
    active_scenarios = tuple(development_scenarios() if scenarios is None else scenarios)
    scenario_rows = [item.manifest_row() for item in active_scenarios]
    scenario_manifest = {
        "namespace": "round20_multi_uav_shared_spectrum_development_v1",
        "scenario_count": len(active_scenarios), "scenarios": scenario_rows,
        "pre_generation_frozen": True,
    }
    write_json(output / "multi_uav_dataset_manifest.json", scenario_manifest)
    split_rows = split_summary(tuple(active_scenarios))
    write_json(output / "multi_uav_dataset_split_manifest.json", {"split_rule": "scenario-level deterministic stratification before rollout", "scenario_count": len(active_scenarios), "rows": split_rows, "no_scenario_across_splits": True})
    episodes = [run_episode(scenario, behavior) for scenario in active_scenarios for behavior in BEHAVIORS]
    quality_rows = label_quality(episodes)
    transitions = [transition for episode in episodes for transition in episode["transitions"]]
    samples, normalization = build_task_waypoint_samples(episodes)
    metrics = episode_metrics(episodes)
    write_jsonl(output / "multi_uav_transitions.jsonl", transitions)
    write_jsonl(output / "multi_uav_task_waypoint_samples.jsonl", samples)
    write_csv(output / "multi_uav_episode_metrics.csv", metrics)
    write_csv(output / "multi_uav_trajectory_quality_audit.csv", quality_rows)
    write_csv(output / "multi_uav_behavior_distribution.csv", behavior_distribution(episodes))
    write_json(output / "multi_uav_dataset_schema_v1.json", dataset_schema())
    write_json(output / "multi_uav_graph_schema_v1.json", graph_schema())
    write_json(output / "multi_uav_task_waypoint_schema_v1.json", task_waypoint_schema())
    write_json(output / "multi_uav_task_waypoint_normalization.json", normalization)
    task_manifest = {"schema_version": TASK_SCHEMA_VERSION, "sample_file": "multi_uav_task_waypoint_samples.jsonl", "sample_count": len(samples), "counts_by_horizon": {str(h): sum(item["H_task"] == h for item in samples) for h in (5, 10)}, "metadata_not_default_input": ["quality_tier", "behavior_generator", "interference_tier"], "no_cross_episode_or_terminal_window": True}
    write_json(output / "multi_uav_task_waypoint_samples_manifest.json", task_manifest)
    return {"output": output, "scenarios": active_scenarios, "episodes": episodes, "transitions": transitions, "samples": samples, "metrics": metrics, "quality_rows": quality_rows, "normalization": normalization, "scenario_manifest_sha256": sha256(output / "multi_uav_dataset_manifest.json"), "split_rows": split_rows}


__all__ = [
    "BEHAVIORS", "BUOY_FEATURE_NAMES", "DATASET_SCHEMA_VERSION", "GAMMA", "GLOBAL_FEATURE_NAMES",
    "GRAPH_SCHEMA_VERSION", "MAX_BUOYS", "MAX_UAVS", "TASK_SCHEMA_VERSION", "UB_FEATURE_NAMES",
    "UU_FEATURE_NAMES", "UAV_FEATURE_NAMES", "BehaviorSpec", "behavior_action", "build_graph_state",
    "build_task_waypoint_samples", "dataset_schema", "episode_metrics", "generate_dataset", "graph_schema",
    "label_quality", "plain", "read_jsonl", "reconstruct_formal_result", "run_episode", "sha256", "task_waypoint_schema", "write_csv", "write_json", "write_jsonl",
]
