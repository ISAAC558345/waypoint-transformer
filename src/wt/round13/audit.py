"""Round-13 audit-only movement/Task-Waypoint controllability study.

No trainable module, optimizer, gradient, model checkpoint, or environment
action-interface change is defined here.  The prospective service decoder is
an external counterfactual calculation, not an ``MaritimeEnv`` implementation.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.maritime.baselines import create_baseline_policy
from wt.maritime.env import MaritimeEnv
from wt.round7.round7_runner import _case as round7_case, radial_projection
from wt.round12.contracts import PREDICTOR_SHA256, sha256

from .contracts import (
    EPSILON,
    current_state_service_heuristic,
    discrete_service_fractions,
    valid_masked_priority_fractions,
)


SCHEMA = "round13_controllability_audit_v1"
PROTOCOL_PATH = Path("round_13/configs/round13_controllability_protocol.json")
ROUND12_OUTPUT = Path("round_12/outputs/round12_formal_20260810_161000")
ROUND10_METADATA = Path("round_10/outputs/round10_task_waypoint_20260810_0110/task_waypoint_metadata.jsonl")
ROUND7_OUTPUT = Path("round_07/outputs/round07_formal_20260809_1740")
UNTOUCHED_HOLDOUT_MANIFEST_SHA256 = "B0D58BE57FD140E113B403E3C96955C56CF65DCDA9D97306E7178036A4A2218D"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)):
            raise ValueError("Round-13 output contains NaN or Inf")
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Round-13 output contains NaN or Inf")
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        raise ValueError(f"cannot write an empty audit table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_plain(row[key]), ensure_ascii=False, sort_keys=True) if isinstance(row.get(key), (dict, list, tuple, np.ndarray)) else _plain(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _source_tree(root: Path) -> dict[str, str]:
    source = root / "src/wt/round13"
    return {path.relative_to(source).as_posix(): sha256(path) for path in sorted(source.rglob("*.py"))}


def _untouched_holdout_digest(root: Path) -> str:
    """Hash the protected holdout as an integrity object, without parsing it."""
    digest = hashlib.sha256((root / "round_07/outputs/round07_formal_20260809_1740/untouched_holdout_v2_manifest.json").read_bytes()).hexdigest().upper()
    if digest != UNTOUCHED_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError("protected untouched-holdout manifest hash changed")
    return digest


def _check_round12_lock(root: Path) -> Mapping[str, Any]:
    lock = _read_json(root / "round_12/ROUND_LOCK.json")
    frozen = _read_json(root / ROUND12_OUTPUT / "formal_protocol_manifest.json")
    policies = _read_json(root / ROUND12_OUTPUT / "frozen_policy_checkpoint_manifest.json")
    if lock.get("status") != "locked" or not lock.get("acceptance_complete"):
        raise RuntimeError("Round-12 is not locked")
    if sha256(root / "round_11/checkpoints/H10/seed_2/best.pt") != PREDICTOR_SHA256:
        raise RuntimeError("Round-12 selected predictor hash changed")
    current_round12 = {path.relative_to(root / "src/wt/round12").as_posix(): sha256(path) for path in sorted((root / "src/wt/round12").rglob("*.py"))}
    if current_round12 != frozen.get("source_tree"):
        raise RuntimeError("Round-12 source tree changed after its formal freeze")
    for row in policies.get("checkpoints", []):
        if sha256(root / ROUND12_OUTPUT / str(row["path"])) != row["sha256"]:
            raise RuntimeError("a locked Round-12 policy checkpoint changed")
    return {
        "round12_lock": "PASS", "predictor_sha256": PREDICTOR_SHA256,
        "policy_checkpoint_count": int(policies["checkpoint_count"]),
        "round12_source_tree_digest": frozen["source_tree_digest"],
        "old_development_test_accessed": False, "untouched_holdout_accessed": False,
        "untouched_holdout_integrity_sha256": _untouched_holdout_digest(root),
    }


def _protocol(root: Path) -> dict[str, Any]:
    value = _read_json(root / PROTOCOL_PATH)
    expected = {"task_horizon": 10, "episode_horizon": 128, "formal_scenario_split": "validation", "continuation_policy": "queue_rate_energy", "bootstrap_unit": "scenario_id"}
    for key, item in expected.items():
        if value.get(key) != item:
            raise RuntimeError(f"Round-13 protocol mismatch at {key}")
    if int(value.get("bootstrap_replicates", 0)) < 1000:
        raise RuntimeError("Round-13 bootstrap count is too small")
    return value


def _validation_scenarios(root: Path) -> list[dict[str, Any]]:
    """Read only Round-10's permitted train/validation metadata package."""
    scenarios: set[str] = set()
    with (root / ROUND10_METADATA).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "validation" and int(row.get("H_task", -1)) == 10:
                scenarios.add(str(row["scenario_id"]))
    result = []
    for scenario_id in sorted(scenarios):
        tail = scenario_id.rsplit("seed-", 1)[-1]
        seed_text, buoy_text = tail.split("__buoys-")
        result.append({"scenario_id": scenario_id, "scenario_seed": int(seed_text), "num_buoys": int(buoy_text)})
    if len(result) != 10:
        raise RuntimeError("Round-13 requires exactly the ten permitted validation scenarios")
    return result


def _allowed_difficulty_labels(root: Path, scenarios: Sequence[Mapping[str, Any]]) -> Mapping[str, str]:
    allowed = {str(row["scenario_id"]) for row in scenarios}
    labels: dict[str, str] = {}
    # Only labels whose IDs were established above are materialized; no test
    # transition, action, outcome, or model-selection content is consumed.
    with (root / ROUND7_OUTPUT / "scenario_difficulty_audit.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["scenario_id"]) in allowed:
                labels[str(row["scenario_id"])] = str(row["difficulty_tier"])
    if set(labels) != allowed:
        raise RuntimeError("permitted validation difficulty labels are incomplete")
    return labels


def _geometry_group(seed: int) -> str:
    return ("central", "random_interior", "corner")[int(seed) % 3]


def _new_policy() -> Any:
    policy = create_baseline_policy("queue_rate_energy", seed=0)
    policy.reset(seed=0)
    return policy


def _snapshot(scenario: Mapping[str, Any], timestep: int, horizon: int) -> tuple[MaritimeEnv, np.ndarray, Mapping[str, Any]]:
    scenario_id, config, _, trace_id, _ = round7_case(int(scenario["scenario_seed"]), int(scenario["num_buoys"]), 128)
    if scenario_id != scenario["scenario_id"]:
        raise RuntimeError("permitted scenario reconstruction mismatch")
    env = MaritimeEnv(config)
    observation, info = env.reset(seed=int(scenario["scenario_seed"]))
    if info["arrival_trace_id"] != trace_id:
        raise RuntimeError("arrival trace reconstruction mismatch")
    policy = _new_policy()
    for _ in range(int(timestep)):
        action = radial_projection(policy.act(observation, info))
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("pre-registered diagnostic snapshot ended early")
    if env.time_step + horizon > config.episode_horizon:
        raise RuntimeError("diagnostic horizon crosses terminal boundary")
    return env, observation.copy(), dict(info)


def _action_set(observation: np.ndarray, info: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    qre = radial_projection(_new_policy().act(observation, info))
    value: dict[str, np.ndarray] = {"queue_rate_energy": qre, "zero": np.zeros(2, dtype=np.float64), "+x": np.array((1.0, 0.0)), "-x": np.array((-1.0, 0.0)), "+y": np.array((0.0, 1.0)), "-y": np.array((0.0, -1.0))}
    for degrees in range(0, 360, 45):
        radians = math.radians(degrees)
        value[f"radial_{degrees}"] = radial_projection(np.array((math.cos(radians), math.sin(radians))))
    return value


def _rollout_from_snapshot(env: MaritimeEnv, observation: np.ndarray, info: Mapping[str, Any], first_action: np.ndarray, horizon: int) -> Mapping[str, Any]:
    rollout = copy.deepcopy(env)
    obs, current = observation.copy(), dict(info)
    policy = _new_policy()
    payload = np.zeros(rollout.num_buoys, dtype=np.float64)
    rewards: list[float] = []
    legal = True
    for step in range(int(horizon)):
        action = radial_projection(first_action if step == 0 else policy.act(obs, current))
        if not rollout.action_space.contains(action):
            legal = False
        obs, reward, terminated, truncated, current = rollout.step(action)
        payload += np.asarray(current["transmitted_payload_bits"], dtype=np.float64)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    total = float(payload.sum())
    distribution = payload / total if total > EPSILON else np.zeros_like(payload)
    return {"service_distribution": distribution, "service_valid": bool(total > EPSILON), "gross_service_progress_bits": payload, "total_service_bits": total, "H_step_return": float(sum(rewards)), "horizon_steps": len(rewards), "all_actions_legal": legal, "final_completion": float(current["task_completion_ratio"]), "final_residual_queue_bits": float(np.asarray(current["raw_queue_bits"], dtype=float).sum())}


def _js(left: np.ndarray, right: np.ndarray) -> float:
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    midpoint = .5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        active = a > EPSILON
        return float(np.sum(a[active] * np.log(a[active] / np.maximum(b[active], EPSILON))))
    return .5 * (kl(p, midpoint) + kl(q, midpoint))


def _entropy(value: np.ndarray) -> float:
    p = np.asarray(value, dtype=float)
    active = p > EPSILON
    return float(-np.sum(p[active] * np.log(p[active])))


def _top(value: np.ndarray) -> int | None:
    return None if float(np.asarray(value).sum()) <= EPSILON else int(np.argmax(value))


def _state_hash(observation: np.ndarray, info: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in (observation, np.asarray(info["raw_queue_bits"]), np.asarray(info["uav_position_m"]), np.asarray([info["time_step"]])):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest().upper()


def controllability_rows(root: Path, protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = _validation_scenarios(root)
    difficulty = _allowed_difficulty_labels(root, scenarios)
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for scenario in scenarios:
        for timestep in protocol["snapshot_timesteps"]:
            env, observation, info = _snapshot(scenario, int(timestep), int(protocol["task_horizon"]))
            state_id = f"{scenario['scenario_id']}__t-{int(timestep):03d}"
            snap = {**scenario, "snapshot_id": state_id, "timestep": int(timestep), "difficulty": difficulty[scenario["scenario_id"]], "geometry_group": _geometry_group(int(scenario["scenario_seed"])), "queue_bits": float(np.asarray(info["raw_queue_bits"], dtype=float).sum()), "snapshot_sha256": _state_hash(observation, info)}
            snapshots.append(snap)
            result = {name: _rollout_from_snapshot(env, observation, info, action, int(protocol["task_horizon"])) for name, action in _action_set(observation, info).items()}
            baseline = result["queue_rate_energy"]
            for name, metrics in result.items():
                service = np.asarray(metrics["service_distribution"], dtype=float)
                progress = np.asarray(metrics["gross_service_progress_bits"], dtype=float)
                rows.append({**snap, "action_name": name, "raw_action": _action_set(observation, info)[name].tolist(), "service_distribution": service.tolist(), "gross_service_progress_bits": progress.tolist(), "total_service_bits": metrics["total_service_bits"], "H_step_return": metrics["H_step_return"], "action_to_service_js": _js(service, baseline["service_distribution"]), "top_served_buoy_switch": _top(service) != _top(baseline["service_distribution"]), "service_entropy_change": _entropy(service) - _entropy(baseline["service_distribution"]), "total_service_change_bits": metrics["total_service_bits"] - baseline["total_service_bits"], "progress_l2_change_bits": float(np.linalg.norm(progress - baseline["gross_service_progress_bits"])), "return_change": metrics["H_step_return"] - baseline["H_step_return"], "all_actions_legal": metrics["all_actions_legal"], "horizon_steps": metrics["horizon_steps"]})
    values = np.asarray([row["queue_bits"] for row in snapshots], dtype=float)
    low, high = np.quantile(values, (.3333333333333333, .6666666666666666))
    for row in rows:
        row["queue_level"] = "low" if row["queue_bits"] <= low else ("medium" if row["queue_bits"] <= high else "high")
    for row in snapshots:
        row["queue_level"] = "low" if row["queue_bits"] <= low else ("medium" if row["queue_bits"] <= high else "high")
    return rows, snapshots


def _score_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = {"overall": lambda r: "all", "difficulty": lambda r: str(r["difficulty"]), "buoy_scale": lambda r: str(r["num_buoys"]), "queue_level": lambda r: str(r["queue_level"]), "geometry_group": lambda r: str(r["geometry_group"])}
    result: list[dict[str, Any]] = []
    for dimension, labeler in groups.items():
        labels = sorted({labeler(row) for row in rows})
        for label in labels:
            subset = [row for row in rows if labeler(row) == label]
            by_snapshot: dict[str, list[Mapping[str, Any]]] = {}
            for row in subset:
                by_snapshot.setdefault(str(row["snapshot_id"]), []).append(row)
            for head, field in (("service_distribution", "service_distribution"), ("gross_service_progress", "gross_service_progress_bits"), ("H_step_return", "H_step_return")):
                within = []
                all_values = []
                for variants in by_snapshot.values():
                    values = np.asarray([row[field] for row in variants], dtype=float)
                    within.append(float(np.var(values, axis=0).mean()))
                    all_values.append(values.reshape(-1))
                denominator = float(np.var(np.concatenate(all_values)))
                numerator = float(np.mean(within))
                result.append({"head": head, "group_dimension": dimension, "group_value": label, "snapshot_count": len(by_snapshot), "action_count_per_snapshot": len(next(iter(by_snapshot.values()))), "between_action_variance": numerator, "total_target_variance": denominator, "controllability_score": numerator / (denominator + EPSILON), "definition": "mean_over_snapshots(var_over_legal_actions(response)) / (var_over_all_snapshot_action_responses(response)+epsilon)"})
    return result


def _counterfactual_scenario(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compare fixed 1/K service to current-state discrete priority, no future inputs."""
    scenario_id, config, _, trace_id, _ = round7_case(int(scenario["scenario_seed"]), int(scenario["num_buoys"]), 128)
    env = MaritimeEnv(config); observation, info = env.reset(seed=int(scenario["scenario_seed"]))
    if info["arrival_trace_id"] != trace_id or scenario_id != scenario["scenario_id"]:
        raise RuntimeError("headroom scenario reconstruction mismatch")
    policy = _new_policy()
    alternative_queue = np.asarray(info["raw_queue_bits"], dtype=float).copy()
    initial = alternative_queue.copy(); cumulative_arrival = np.zeros_like(initial); alternative_payload = np.zeros_like(initial)
    baseline_return = 0.0; alternative_return = 0.0; priorities: list[int] = []; violations = 0
    while True:
        action = radial_projection(policy.act(observation, info))
        observation, reward, terminated, truncated, after = env.step(action)
        baseline_return += float(reward)
        rates = np.asarray(after["rates_bps"], dtype=float)
        full_rate = rates * float(env.num_buoys)
        selected = current_state_service_heuristic(alternative_queue, full_rate, np.ones(env.num_buoys, dtype=bool))
        fractions = discrete_service_fractions(selected, np.ones(env.num_buoys, dtype=bool))
        capacity = full_rate * fractions * float(config.delta_t)
        queue_before = alternative_queue.copy()
        payload = np.minimum(queue_before, capacity)
        arrival = np.asarray(after["raw_arrival_bits"], dtype=float)
        alternative_queue = np.clip(alternative_queue - payload + arrival, 0.0, float(config.queue_capacity_bits))
        alternative_payload += payload; cumulative_arrival += arrival
        if np.any(payload > capacity + 1e-6) or np.any(payload > queue_before + 1e-6):
            violations += 1
        components = dict(after["reward_components"])
        non_service = float(reward) - float(components.get("payload_reward", 0.0)) - float(components.get("unfinished_queue_penalty", 0.0)) - float(components.get("task_success_bonus", 0.0))
        alternative_return += non_service + float(config.payload_reward_weight * np.clip(payload.sum() / max(float(after["total_rate_bps"]) * config.delta_t, EPSILON), 0.0, 1.0))
        priorities.append(-1 if selected is None else selected)
        info = after
        if terminated or truncated:
            break
    offered = initial + cumulative_arrival
    alternative_completion = float(np.sum(alternative_payload) / max(float(offered.sum()), EPSILON))
    alternative_residual = float(alternative_queue.sum())
    if not bool(info["task_success"]):
        alternative_return -= float(config.unfinished_queue_penalty_weight * alternative_residual / max(float(offered.sum()), EPSILON))
    return {"scenario_id": scenario_id, "scenario_seed": int(scenario["scenario_seed"]), "num_buoys": int(scenario["num_buoys"]), "baseline_payload_bits": float(info["cumulative_payload_bits"]), "heuristic_payload_bits": float(alternative_payload.sum()), "Delta_payload": float(alternative_payload.sum() - float(info["cumulative_payload_bits"])), "baseline_completion": float(info["task_completion_ratio"]), "heuristic_completion": alternative_completion, "Delta_completion": alternative_completion - float(info["task_completion_ratio"]), "baseline_residual_queue_bits": float(np.asarray(info["raw_queue_bits"], dtype=float).sum()), "heuristic_residual_queue_bits": alternative_residual, "Delta_residual_queue": float(np.asarray(info["raw_queue_bits"], dtype=float).sum() - alternative_residual), "baseline_return": baseline_return, "heuristic_return": alternative_return, "Delta_return": alternative_return - baseline_return, "priority_sequence": priorities, "resource_decoder_violations": violations, "uses_current_state_only": True, "same_movement_continuation": True}


def _bootstrap(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(protocol["bootstrap_seed"]))
    result = []
    for metric in ("Delta_payload", "Delta_completion", "Delta_residual_queue", "Delta_return"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        samples = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(int(protocol["bootstrap_replicates"]))])
        result.append({"comparison": "current_state_service_heuristic_minus_fixed_parallel_1_over_K", "metric": metric, "scenario_count": len(values), "paired_mean": float(values.mean()), "paired_median": float(np.median(values)), "paired_std": float(values.std(ddof=0)), "bootstrap_ci_low": float(np.quantile(samples, .025)), "bootstrap_ci_high": float(np.quantile(samples, .975)), "win_rate": float(np.mean(values > 0.0)), "tie_rate": float(np.mean(values == 0.0)), "loss_rate": float(np.mean(values < 0.0)), "effect_size_mean_over_std": float(values.mean() / values.std(ddof=0)) if values.std(ddof=0) > EPSILON else 0.0})
    return result


def _service_gate(paired: Sequence[Mapping[str, Any]], alignment: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_metric = {str(row["metric"]): row for row in paired}
    payload = by_metric["Delta_payload"]
    beneficial = payload["paired_mean"] > 0.0 and payload["bootstrap_ci_low"] > 0.0
    harmful = payload["paired_mean"] < 0.0 and payload["bootstrap_ci_high"] < 0.0
    secondary_safe = by_metric["Delta_completion"]["paired_mean"] >= 0.0 and by_metric["Delta_residual_queue"]["paired_mean"] >= 0.0 and by_metric["Delta_return"]["paired_mean"] >= 0.0
    if beneficial and secondary_safe:
        gate = "SA-A"
    elif harmful:
        gate = "SA-C"
    else:
        gate = "SA-B"
    mismatch = any(row["head"] == "service_distribution" and not row["directly_controllable"] for row in alignment)
    return {"service_action_headroom_gate": gate, "payload_mean_delta": payload["paired_mean"], "payload_95_CI": [payload["bootstrap_ci_low"], payload["bootstrap_ci_high"]], "scenario_consistency": {"win": payload["win_rate"], "tie": payload["tie_rate"], "loss": payload["loss_rate"]}, "secondary_directionally_non_degrading": secondary_safe, "structural_action_task_mismatch": mismatch, "service_action_extension_ready": bool(gate == "SA-A" and mismatch), "gate_definition": "SA-A: positive paired payload CI excluding zero plus non-degrading completion/residual/return; SA-C: negative payload CI excluding zero; otherwise SA-B. No fitted percentage threshold."}


def _alignment(root: Path, scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sensitivity = _read_json(root / ROUND12_OUTPUT / "task_waypoint_head_utilization_summary.json")
    gate = _read_json(root / ROUND12_OUTPUT / "final_policy_gate.json")
    names = (("service_distribution", "Sensitivity_service"), ("gross_service_progress", "Sensitivity_progress"), ("H_step_return", "Sensitivity_return"))
    result = []
    for head, sensitivity_name in names:
        overall = next(row for row in scores if row["head"] == head and row["group_dimension"] == "overall")
        result.append({"head": head, "predictable": True, "policy_sensitive": float(sensitivity[sensitivity_name]) > 0.0, "directly_controllable": False, "indirectly_controllable": True, "controllability_score": overall["controllability_score"], "closed_loop_gain_observed": False, "round12_task_waypoint_policy_gate": gate["task_waypoint_policy_gate"], "alignment_status": "movement-mediated only; Round-12 showed no reliable closed-loop gain"})
    return result


def _write_markdown(output: Path, protocol: Mapping[str, Any], scores: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> None:
    overall = {row["head"]: row["controllability_score"] for row in scores if row["group_dimension"] == "overall"}
    (output / "round13_action_semantics_audit.md").write_text("# Round-13 action semantics audit\n\nCurrent public action is `a_t=[delta_x, delta_y]`, dimension 2, component bounds [-1,1], with radial unit-disk projection and boundary clipping in `MaritimeEnv._step_v1`. It directly controls UAV displacement only.\n\n| field | physical meaning | directly controllable variable |\n|---|---|---|\n| delta_x | normalized horizontal displacement command | UAV x position |\n| delta_y | normalized vertical displacement command | UAV y position |\n\nNo action field selects a buoy, allocates bandwidth/resource fraction, allocates payload, controls power, beamforming, or association.\n", encoding="utf-8")
    (output / "current_resource_decoder_audit.md").write_text("# Current resource decoder audit\n\nThe existing v1 environment serves all physical buoys in parallel. `compute_channel_metrics` fixes orthogonal bandwidth at `B/K`, and `rates_bps` already contains this allocation. Each slot uses `payload_k=min(Q_k, rates_k*delta_t)` before arrivals. There is no policy-controlled association, schedule, resource fraction, power, or payload allocation. Thus the current mechanism is **B: fixed equal 1/K allocation**, not a policy action.\n", encoding="utf-8")
    comparison = "# Service-intent action design comparison\n\n| candidate | action | strengths | limitation |\n|---|---|---|---|\n| A | `[dx,dy,u]`, discrete valid buoy index | simple, interpretable, exact one-hot decoder | cannot express parallel service or graded priorities |\n| B | `[dx,dy,pi_1,...,pi_K]`, masked simplex | variable-K, keeps padding zero, can represent A and parallel priorities | requires a simplex decoder |\n\n**Recommended: Candidate B, continuous masked service priority.** This is a prospective specification only; it is not implemented in the environment or trained in Round-13. A future decoder must normalize valid nonnegative priorities, set padding fractions to zero, ensure fractions sum to one (or zero with no valid buoy), and enforce `payload_k <= min(queue_k, fraction_k * full-bandwidth link capacity_k)`.\n"
    (output / "service_action_design_comparison.md").write_text(comparison, encoding="utf-8")
    report = "# Round-13 completion report\n\n" + "\n".join([f"- {head}: controllability score {float(value):.8f}" for head, value in overall.items()]) + f"\n- Service-Action Gate: `{gate['service_action_headroom_gate']}`\n- Extension ready: `{gate['service_action_extension_ready']}`\n\nThis audit did not train or modify any model, policy, action interface, or environment resource decoder.\n"
    (output / "round13_completion_report.md").write_text(report, encoding="utf-8")


def _figures(output: Path, rows: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]], headroom: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(exist_ok=True)
    written: list[str] = []
    def save(name: str, labels: Sequence[str], values: Sequence[float], title: str) -> None:
        figure, axis = plt.subplots(figsize=(max(6, .55 * len(labels)), 4), facecolor="white")
        axis.bar(range(len(values)), values, color="#2676a1"); axis.axhline(0, color="black", linewidth=.8); axis.set_xticks(range(len(labels))); axis.set_xticklabels(labels, rotation=35, ha="right"); axis.set_title(title); figure.tight_layout()
        for suffix in ("png", "svg"):
            path = folder / f"{name}.{suffix}"; figure.savefig(path, dpi=140 if suffix == "png" else None); written.append(path.name)
        plt.close(figure)
    action_names = sorted(set(str(row["action_name"]) for row in rows))
    for field, name, title in (("action_to_service_js", "01_action_service_js", "Action perturbation vs service-distribution change"), ("progress_l2_change_bits", "02_action_progress", "Action perturbation vs gross progress"), ("return_change", "03_action_return", "Action perturbation vs H-step return")):
        save(name, action_names, [float(np.mean([float(row[field]) for row in rows if row["action_name"] == action])) for action in action_names], title)
    overall = [row for row in scores if row["group_dimension"] == "overall"]
    save("04_head_controllability", [str(row["head"]) for row in overall], [float(row["controllability_score"]) for row in overall], "Head controllability scores")
    for dimension, name, title in (("buoy_scale", "05_buoy_controllability", "3-buoy vs 5-buoy controllability"), ("geometry_group", "06_geometry_controllability", "Geometry vs controllability"), ("queue_level", "07_queue_controllability", "Queue level vs controllability")):
        subset = [row for row in scores if row["group_dimension"] == dimension and row["head"] == "service_distribution"]
        save(name, [str(row["group_value"]) for row in subset], [float(row["controllability_score"]) for row in subset], title)
    save("08_resource_payload", [str(row["scenario_seed"]) + ":" + str(row["num_buoys"]) for row in headroom], [float(row["Delta_payload"]) for row in headroom], "Fixed resource vs current-state service heuristic payload")
    payload = next(row for row in paired if row["metric"] == "Delta_payload")
    residual = next(row for row in paired if row["metric"] == "Delta_residual_queue")
    save("09_paired_payload_delta", ["mean", "CI low", "CI high"], [float(payload["paired_mean"]), float(payload["bootstrap_ci_low"]), float(payload["bootstrap_ci_high"])], "Paired payload delta")
    save("10_residual_queue_delta", ["mean", "CI low", "CI high"], [float(residual["paired_mean"]), float(residual["bootstrap_ci_low"]), float(residual["bootstrap_ci_high"])], "Paired residual-queue improvement")
    example = headroom[0]["priority_sequence"]
    save("11_service_priority_example", [str(index) for index in range(len(example))], [float(value) for value in example], "Current-state discrete service priorities")
    save("12_action_space_candidates", ["movement", "A discrete", "B continuous"], [2.0, 3.0, 7.0], "Prospective action-space candidates (dimensions at K=5)")
    return sorted(written)


def run_round13(root: Path = Path.cwd(), *, output: Path | None = None) -> Path:
    root = Path(root); protocol = _protocol(root)
    output = output or root / "round_13/outputs" / datetime.now(timezone.utc).strftime("round13_controllability_%Y%m%d_%H%M%S")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    lock_audit = _check_round12_lock(root)
    frozen = {"schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol, "protocol_sha256": sha256(root / PROTOCOL_PATH), "source_tree": _source_tree(root), "round12_integrity": lock_audit, "training_performed": False, "service_action_implemented": False, "policy_training_performed": False, "old_development_test_accessed": False, "untouched_holdout_accessed": False}
    _write_json(output / "round13_controllability_manifest.json", frozen)
    rows, snapshots = controllability_rows(root, protocol)
    if not all(bool(row["all_actions_legal"]) for row in rows):
        raise RuntimeError("an action perturbation was not legal")
    scores = _score_rows(rows)
    alignment = _alignment(root, scores)
    headroom = [_counterfactual_scenario(scenario) for scenario in _validation_scenarios(root)]
    if any(int(row["resource_decoder_violations"]) for row in headroom):
        raise RuntimeError("diagnostic resource decoder violated a physical bound")
    paired = _bootstrap(headroom, protocol)
    gate = _service_gate(paired, alignment)
    _write_csv(output / "task_waypoint_controllability_raw.csv", rows)
    _write_csv(output / "task_waypoint_controllability.csv", scores)
    _write_csv(output / "task_waypoint_action_alignment.csv", alignment)
    _write_csv(output / "service_action_headroom_metrics.csv", headroom)
    _write_csv(output / "service_action_headroom_paired.csv", paired)
    _write_json(output / "service_action_headroom_gate.json", gate)
    _write_json(output / "recommended_service_action_v1.json", {"recommended_service_action_v1": "continuous_masked_service_priority", "candidate": "B", "implemented_in_round13": False, "action": "[delta_x, delta_y, pi_1,...,pi_K]", "constraints": ["pi_k>=0", "sum_valid(pi)=1 when any valid buoy exists", "padding_pi=0", "fraction_k=masked_normalized_priority_k", "payload_k<=queue_k", "payload_k<=fraction_k*full_bandwidth_link_capacity_k"], "variable_buoy_count_supported": True})
    _write_markdown(output, protocol, scores, gate)
    figures = _figures(output, rows, scores, headroom, paired)
    _write_json(output / "round13_readiness_gate.json", {"action_task_mismatch_structural": gate["structural_action_task_mismatch"], "service_action_headroom_gate": gate["service_action_headroom_gate"], "service_action_extension_ready": gate["service_action_extension_ready"], "next_round_creation_allowed": False, "untouched_holdout_accessed": False})
    _write_json(output / "run_manifest.json", {"schema_version": SCHEMA, "controllability_audit_performed": True, "training_performed": False, "service_action_implemented": False, "policy_training_performed": False, "old_development_test_accessed": False, "untouched_holdout_accessed": False, "hgnn_training_performed": False, "ppo_training_performed": False, "snapshot_count": len(snapshots), "perturbation_count": len(rows), "headroom_scenario_count": len(headroom), "figure_files": figures, "gate": gate})
    return output


def record_final_verification(output: Path, *, test_count: int, compileall_passed: bool) -> None:
    if int(test_count) < 1 or not compileall_passed:
        raise ValueError("only a passed final verification can be recorded")
    _write_json(Path(output) / "final_verification.json", {"repository_tests_passed": int(test_count), "compileall_passed": True})


def finalize_round13(root: Path, output: Path) -> Mapping[str, Any]:
    root, output = Path(root), Path(output)
    manifest = _read_json(root / "round_13/round_manifest.json")
    if manifest.get("status") != "in_progress":
        raise RuntimeError("Round-13 is not eligible for locking")
    frozen = _read_json(output / "round13_controllability_manifest.json")
    if frozen["source_tree"] != _source_tree(root) or frozen["protocol_sha256"] != sha256(root / PROTOCOL_PATH):
        raise RuntimeError("Round-13 source or protocol changed after audit freeze")
    final_round12_audit = _check_round12_lock(root)
    if final_round12_audit["untouched_holdout_integrity_sha256"] != frozen["round12_integrity"]["untouched_holdout_integrity_sha256"]:
        raise RuntimeError("protected untouched-holdout integrity object changed")
    verification = _read_json(output / "final_verification.json")
    required = ["task_waypoint_controllability.csv", "task_waypoint_action_alignment.csv", "service_action_headroom_metrics.csv", "service_action_headroom_paired.csv", "service_action_headroom_gate.json", "recommended_service_action_v1.json", "round13_readiness_gate.json", "round13_completion_report.md", "run_manifest.json"]
    if not verification.get("compileall_passed") or any(not (output / item).is_file() for item in required):
        raise RuntimeError("Round-13 acceptance evidence is incomplete")
    gate = _read_json(output / "service_action_headroom_gate.json")
    lock = {"round": "round_13", "status": "locked", "locked_at_utc": datetime.now(timezone.utc).isoformat(), "formal_output": str(output.relative_to(root)), "round12_lock_unchanged": True, "predictor_sha256": PREDICTOR_SHA256, "round12_policy_checkpoint_count": final_round12_audit["policy_checkpoint_count"], "untouched_holdout_integrity_sha256": final_round12_audit["untouched_holdout_integrity_sha256"], "training_performed": False, "service_action_implemented": False, "policy_training_performed": False, "old_development_test_access_count": 0, "untouched_holdout_access_count": 0, "service_action_headroom_gate": gate["service_action_headroom_gate"], "service_action_extension_ready": gate["service_action_extension_ready"], "round_14_created": False, "repository_tests_passed": verification["repository_tests_passed"], "compileall_passed": True}
    _write_json(root / "round_13/ROUND_LOCK.json", lock)
    completed = dict(manifest); completed.update({"status": "completed", "controllability_audit_performed": True, "service_action_headroom_gate": gate["service_action_headroom_gate"], "service_action_extension_ready": gate["service_action_extension_ready"], "lock_file": "round_13/ROUND_LOCK.json"})
    _write_json(root / "round_13/round_manifest.json", completed)
    current = (root / "CURRENT_ROUND.txt").read_text(encoding="utf-8")
    if "round_13" not in current or "status=in_progress" not in current:
        raise RuntimeError("CURRENT_ROUND is not Round-13 in progress")
    (root / "CURRENT_ROUND.txt").write_text(current.replace("status=in_progress", "status=completed", 1) + "lock_file=round_13/ROUND_LOCK.json\n", encoding="utf-8")
    root_manifest = _read_json(root / "round_manifest.json")
    round13 = dict(root_manifest.get("round_13", {}))
    round13.update({"status": "completed", "lock": "round_13/ROUND_LOCK.json", "formal_output": str(output.relative_to(root)), "controllability_audit_performed": True, "service_action_headroom_gate": gate["service_action_headroom_gate"], "service_action_extension_ready": gate["service_action_extension_ready"], "next_round_created": False})
    root_manifest.update({"current_round": "round_13", "status": "completed", "round_13": round13, "next_round_created": False})
    _write_json(root / "round_manifest.json", root_manifest)
    return lock


__all__ = ["controllability_rows", "finalize_round13", "record_final_verification", "run_round13"]
