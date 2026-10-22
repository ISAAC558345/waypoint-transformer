"""Read-only cross-round finite-horizon semantics audit.

This module deliberately works from locked text/CSV/JSONL artifacts.  It
does not import an environment, planner, model, or optimizer; in particular,
it cannot create a rollout or a learning update.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
R32 = ROOT / "round_32"
OUTPUT = R32 / "outputs" / "round32_horizon_semantics_audit_20260830_2231"
FIGURES = R32 / "figures" / "round32_horizon_semantics_audit_20260830_2231"
R20_JSONL = ROOT / "round_20" / "outputs" / "round20_multi_uav_dataset_20260828_2100"

TRACE_SPECS = {
    "round_15": (64, (1, 5, 10, 20), ROOT / "round_15/outputs/round15_planning_headroom_audit_20260825_1245/planning_trajectory_steps.csv", "single_before"),
    "round_16": (64, (1, 5, 10, 20), ROOT / "round_16/outputs/round16_dynamic_planning_audit_20260826_2101/dynamic_trajectory_steps.csv", "single_before"),
    "round_17": (64, (1, 5, 10, 20), ROOT / "round_17/outputs/round17_channel_planning_audit_20260827_1928/channel_trajectory_steps.csv", "single_before"),
    "round_18": (12, (1, 5, 10), ROOT / "round_18/outputs/round18_multi_uav_audit_20260827_2055/multi_uav_trajectory_steps.csv", "multi_after"),
    "round_19": (12, (1, 5, 10), ROOT / "round_19/outputs/round19_shared_spectrum_audit_20260828_1625/interference_trajectory_steps.csv", "multi_after"),
}

SOURCE_PATHS = (
    "src/wt/maritime/config.py", "src/wt/maritime/env.py",
    "src/wt/round15/scenarios.py", "src/wt/round15/planner.py", "src/wt/round15/audit.py",
    "src/wt/round16/scenarios.py", "src/wt/round16/planner.py",
    "src/wt/round17/scenarios.py", "src/wt/round17/planner.py",
    "src/wt/round18/scenarios.py", "src/wt/round18/audit.py",
    "src/wt/round19/scenarios.py", "src/wt/round19/planner.py", "src/wt/round19/multi_env.py",
    "src/wt/round20/scenarios.py", "src/wt/round20/dataset.py",
    "src/wt/round21/data.py", "src/wt/round22/data.py", "src/wt/round24/runner.py",
    "src/wt/round30/scenarios.py", "src/wt/round30/ppo.py",
    "round_30/outputs/round30_deepsets_ppo_20260830_1845/round30_ppo_training_config.json",
    "round_31/report/round31_completion_report.md",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(name: str, value: Any) -> None:
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty required audit table: {name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError(f"inconsistent columns in {name}")
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trace_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first(row: dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in ("", None):
            return str(row[name])
    return default


def number(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def horizon_of(row: dict[str, str]) -> int | None:
    direct = first(row, ("planning_horizon", "horizon", "H", "plan_horizon"))
    if direct:
        candidate = number(direct, -1)
        if candidate > 0:
            return candidate
    policy = first(row, ("policy", "planner_policy", "active_policy")).lower()
    match = re.search(r"(?:^|[_-])h(?:orizon)?[_-]?(1|5|10|20)(?:$|[_-])", policy)
    return int(match.group(1)) if match else None


def action_of(row: dict[str, str]) -> str:
    raw = first(row, ("action", "joint_action", "joint_action_applied", "applied_action"), "")
    return re.sub(r"\s+", "", raw)


def state_time(row: dict[str, str], convention: str) -> int:
    value = number(first(row, ("time_step", "timestep", "step", "t")), 0)
    return max(0, value - 1) if convention == "multi_after" else value


def scenario_of(row: dict[str, str]) -> str:
    return first(row, ("scenario_id", "base_scenario_id", "episode_id"), "unknown")


def policy_of(row: dict[str, str]) -> str:
    return first(row, ("policy", "planner_policy", "active_policy"), "unknown")


def source_simulated_steps(horizon: int, remaining: int, policy: str, round_name: str) -> tuple[int, str]:
    """Return the precise shadow length the source code requests.

    R15--17 causal code advances a detached shadow for nominal H.  Their
    oracle branch is different: it steps a copied finite environment and
    breaks on termination/truncation.  R18/R19 beam search state has no clock
    and always expands exactly H levels.
    """
    if round_name in {"round_15", "round_16", "round_17"} and "oracle" in policy.lower():
        return min(horizon, remaining), "oracle_copied_environment_breaks_on_done"
    if round_name in {"round_15", "round_16", "round_17"}:
        return horizon, "causal_shadow_has_no_episode_clock"
    return horizon, "joint_shadow_has_no_episode_clock"


def collect_planner_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    detail: list[dict[str, Any]] = []
    raw_by_round: dict[str, list[dict[str, str]]] = {}
    for round_name, (episode_horizon, _, path, convention) in TRACE_SPECS.items():
        raw = trace_rows(path)
        raw_by_round[round_name] = raw
        for row in raw:
            horizon = horizon_of(row)
            if horizon is None:
                continue
            t = state_time(row, convention)
            remaining = max(0, episode_horizon - t)
            policy = policy_of(row)
            simulated, semantics = source_simulated_steps(horizon, remaining, policy, round_name)
            effective = min(horizon, remaining)
            detail.append({
                "round": round_name, "scenario_id": scenario_of(row), "policy": policy,
                "state_t": t, "episode_horizon": episode_horizon, "remaining_episode_steps": remaining,
                "nominal_planning_horizon": horizon, "source_shadow_simulated_steps": simulated,
                "episode_constrained_effective_future_steps": effective,
                "horizon_utilization_ratio": round(effective / horizon, 8),
                "source_shadow_to_nominal_ratio": round(simulated / horizon, 8),
                "future_beyond_real_episode_steps": max(0, horizon - remaining),
                "terminal_reached_within_source_shadow": simulated >= remaining,
                "truncated_by_real_episode": simulated > remaining,
                "source_semantics": semantics,
                "trace_path": path.relative_to(ROOT).as_posix(),
            })
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped[(str(row["round"]), int(row["nominal_planning_horizon"]))].append(row)
    summary = []
    for (round_name, horizon), rows in sorted(grouped.items()):
        full = [item for item in rows if int(item["episode_constrained_effective_future_steps"]) == horizon]
        summary.append({
            "round": round_name, "episode_horizon": rows[0]["episode_horizon"], "nominal_planning_horizon": horizon,
            "trace_state_count": len(rows), "mean_episode_constrained_effective_steps": round(mean(float(item["episode_constrained_effective_future_steps"]) for item in rows), 6),
            "mean_source_shadow_simulated_steps": round(mean(float(item["source_shadow_simulated_steps"]) for item in rows), 6),
            "mean_horizon_utilization_ratio": round(mean(float(item["horizon_utilization_ratio"]) for item in rows), 8),
            "full_horizon_state_fraction": round(len(full) / len(rows), 8),
            "any_source_shadow_exceeds_remaining": any(int(item["future_beyond_real_episode_steps"]) > 0 and float(item["source_shadow_to_nominal_ratio"]) == 1.0 for item in rows),
            "validity_class": "A" if len(full) == len(rows) else "B" if mean(float(item["horizon_utilization_ratio"]) for item in rows) >= 0.75 else "C",
        })
    return detail, summary, raw_by_round


def h5_h10_equality(raw_by_round: dict[str, list[dict[str, str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prefixes: list[dict[str, Any]] = []
    for round_name in ("round_18", "round_19"):
        _, _, _, convention = TRACE_SPECS[round_name]
        paired: dict[tuple[str, int], dict[int, dict[str, str]]] = defaultdict(dict)
        sequences: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
        for source in raw_by_round[round_name]:
            horizon = horizon_of(source)
            if horizon not in {5, 10}:
                continue
            key = (scenario_of(source), state_time(source, convention))
            paired[key][horizon] = source
            sequences[(scenario_of(source), horizon)].append((key[1], action_of(source)))
        for (scenario, t), pair in sorted(paired.items()):
            if set(pair) != {5, 10}:
                continue
            posthoc_equal = min(5, 12 - t) == min(10, 12 - t)
            equal = action_of(pair[5]) == action_of(pair[10])
            reason = (
                "EQ-D_posthoc_remaining_window_equal_but_source_shadow_is_not_terminal_aware" if equal and posthoc_equal
                else "EQ-B_same_first_action_with_distinct_nominal_shadows" if equal
                else "not_equal"
            )
            rows.append({
                "round": round_name, "scenario_id": scenario, "state_t": t, "remaining_episode_steps": 12 - t,
                "h5_nominal": 5, "h10_nominal": 10, "h5_episode_constrained": min(5, 12 - t),
                "h10_episode_constrained": min(10, 12 - t), "posthoc_effective_horizons_equal": posthoc_equal,
                "h5_h10_action_equal": equal, "equality_reason": reason,
                "h5_planning_score_persisted": False, "h10_planning_score_persisted": False,
                "planning_score_equality": "not_assessable_from_locked_trajectory_trace",
            })
        by_scenario: dict[str, dict[int, list[tuple[int, str]]]] = defaultdict(dict)
        for (scenario, horizon), seq in sequences.items():
            by_scenario[scenario][horizon] = sorted(seq)
        for scenario, pair in sorted(by_scenario.items()):
            if set(pair) != {5, 10}:
                continue
            first5, first10 = pair[5], pair[10]
            shared = min(len(first5), len(first10))
            identical = sum(a[1] == b[1] for a, b in zip(first5[:shared], first10[:shared]))
            prefix = 0
            for a, b in zip(first5[:shared], first10[:shared]):
                if a[1] != b[1]:
                    break
                prefix += 1
            prefixes.append({"round": round_name, "scenario_id": scenario, "shared_trace_steps": shared, "equal_action_steps": identical, "identical_action_prefix_steps": prefix, "all_shared_actions_equal": identical == shared})
    return rows, prefixes


def permitted_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Decode only train/validation lines; protected lines are token-screened."""
    rows: list[dict[str, Any]] = []
    counts = {"train": 0, "validation": 0, "development_test_screened_without_decode": 0, "other_screened_without_decode": 0}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            split = "train" if '"split":"train"' in line or '"split": "train"' in line else "validation" if '"split":"validation"' in line or '"split": "validation"' in line else None
            if split is None:
                if '"split":"development_test"' in line or '"split": "development_test"' in line:
                    counts["development_test_screened_without_decode"] += 1
                else:
                    counts["other_screened_without_decode"] += 1
                continue
            row = json.loads(line)
            if row.get("split") != split:
                raise RuntimeError("split token and decoded split disagree")
            rows.append(row)
            counts[split] += 1
    return rows, counts


def task_waypoint_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    samples, screening = permitted_jsonl(R20_JSONL / "multi_uav_task_waypoint_samples.jsonl")
    transitions, transition_screening = permitted_jsonl(R20_JSONL / "multi_uav_transitions.jsonl")
    if not samples or not transitions:
        raise RuntimeError("no permitted Round-20 rows found")
    coverage: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    label_rows: list[dict[str, Any]] = []
    for item in samples:
        split, h, t = str(item["split"]), int(item["H_task"]), int(item["t"])
        k = int(item["current_graph_state"]["num_buoys"])
        tier = str(item["interference_tier"])
        coverage[(split, h, k, tier)].append(item)
        effective = min(h, 12 - t)
        label_rows.append({"split": split, "H_task": h, "anchor_t": t, "episode_horizon": 12, "effective_future_steps": effective, "full_horizon_label": effective == h, "label_source": "future window of locked transitions", "terminal_or_truncated_label": False})
    result: list[dict[str, Any]] = []
    for (split, h, k, tier), items in sorted(coverage.items()):
        starts = sorted({int(item["t"]) for item in items})
        result.append({
            "split": split, "H_task": h, "num_buoys": k, "interference_tier": tier, "sample_count": len(items),
            "anchor_min": min(starts), "anchor_max": max(starts), "anchor_count": len(starts),
            "all_labels_full_horizon": all(int(item["t"]) + h <= 12 for item in items),
            "eligible_anchor_fraction_of_episode": round((12 - h + 1) / 12.0, 8),
            "truncated_or_terminal_label_count": 0,
        })
    for h in (5, 10):
        result.append({"split": "development_test", "H_task": h, "num_buoys": "protected_not_accessed", "interference_tier": "protected_not_accessed", "sample_count": "protected_not_accessed", "anchor_min": "protected_not_accessed", "anchor_max": "protected_not_accessed", "anchor_count": "protected_not_accessed", "all_labels_full_horizon": "protected_not_accessed", "eligible_anchor_fraction_of_episode": round((12 - h + 1) / 12.0, 8), "truncated_or_terminal_label_count": "protected_not_accessed"})
    semantics = {
        "episode_horizon": 12, "construction": "for start in range(0, len(transitions) - horizon + 1)",
        "observed_horizons": sorted({int(item["H_task"]) for item in samples}),
        "full_horizon_label_count": len(label_rows), "truncated_or_terminal_label_count": 0,
        "effective_future_definition": "min(H_task, episode_horizon - anchor_t)",
        "conclusion": "Locked Round-20 labels are full future windows only; they do not encode a shortened terminal label.",
    }
    screening.update({f"transition_{key}": value for key, value in transition_screening.items()})
    return result, label_rows, semantics, screening, transitions


def percentile(values: list[float], p: float) -> float:
    """Linear, deterministic percentile used consistently in every table."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def planner_hur_statistics(detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped[(str(row["round"]), int(row["nominal_planning_horizon"]))].append(row)
    result = []
    for (round_name, horizon), rows in sorted(grouped.items()):
        values = [float(row["horizon_utilization_ratio"]) for row in rows]
        result.append({
            "round": round_name, "nominal_H": horizon, "state_count": len(values),
            "mean_HUR": round(mean(values), 8), "median_HUR": round(median(values), 8),
            "p25_HUR": round(percentile(values, 0.25), 8), "p75_HUR": round(percentile(values, 0.75), 8),
            "fraction_HUR_eq_1": round(sum(value == 1.0 for value in values) / len(values), 8),
            "fraction_HUR_lt_1": round(sum(value < 1.0 for value in values) / len(values), 8),
            "fraction_HUR_le_0_5": round(sum(value <= 0.5 for value in values) / len(values), 8),
        })
    return result


def cross_round_semantics(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand the concise matrix into the full four-horizon contract table."""
    result = []
    for row in matrix:
        round_name, episode = str(row["round"]), int(row["episode_horizon"])
        planner = row["registered_horizon_or_context"] if "planning" in str(row["component"]) else "not_applicable"
        task = "H_task={5,10}" if round_name in {"round_20", "round_21"} else "H_task=5" if round_name in {"round_22", "round_24"} else "not_applicable"
        context = 10 if round_name in {"round_22", "round_24"} else "not_applicable"
        result.append({
            "round": round_name, "environment_episode_max_steps": episode,
            "nominal_episode_horizon": episode, "actual_episode_length_distribution": f"fixed finite horizon; terminal at <= {episode}, normal truncation at {episode}",
            "planner_horizon": planner, "task_waypoint_horizon": task, "policy_context_Hc": context,
            "ppo_rollout_buffer_length": 256 if round_name == "round_30" else "not_applicable",
            "ppo_parallel_envs": 8 if round_name == "round_30" else "not_applicable",
            "gamma": 0.99 if round_name in {"round_20", "round_30", "round_31"} else "not_applicable",
            "gae_lambda": 0.95 if round_name in {"round_30", "round_31"} else "not_applicable",
            "terminated_semantics": "energy depleted; task success only when enabled",
            "truncated_semantics": f"time_step >= {episode}",
            "remaining_horizon_state_field": "remaining_horizon_fraction=(episode_horizon-time_step)/episode_horizon",
            "registered_horizon_crosses_episode_terminal": bool(round_name in {"round_18", "round_19"}),
            "target_truncates_at_terminal": "no shortened target emitted" if round_name == "round_20" else "not_applicable",
            "effective_maximum_future_steps": episode,
        })
    return result


def task_validity_rows(coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in coverage:
        if row["split"] != "development_test":
            grouped[(str(row["split"]), int(row["H_task"]))].append(row)
    result = []
    for (split, horizon), rows in sorted(grouped.items()):
        full = all(item["all_labels_full_horizon"] is True for item in rows)
        coverage_fraction = float(rows[0]["eligible_anchor_fraction_of_episode"])
        classification = "TW-H-A" if full and coverage_fraction >= 0.5 else "TW-H-B" if full else "TW-H-C"
        result.append({"split": split, "H_task": horizon, "classification": classification, "full_horizon_samples_only": full, "potential_anchor_coverage": coverage_fraction, "interpretation": "complete labels, but only early finite-episode anchors are eligible"})
    for horizon in (5, 10):
        result.append({"split": "development_test", "H_task": horizon, "classification": "protected_not_accessed", "full_horizon_samples_only": "protected_not_accessed", "potential_anchor_coverage": round((12 - horizon + 1) / 12, 8), "interpretation": "split token screened before decode"})
    return result


def timescale_rows(scales: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    physical = [
        {"measure": "max_movement_per_slot_m", "value": scales["max_movement_per_slot_m"], "unit": "m", "comparison_to_12_step_episode": "one twelfth of maximum episode path"},
        {"measure": "max_movement_per_episode_m", "value": scales["max_movement_per_episode_m"], "unit": "m", "comparison_to_12_step_episode": f"{100.0 * scales['max_movement_per_episode_m'] / scales['default_map_diagonal_m']:.4f}% of default map diagonal"},
        {"measure": "buoy_pair_distance_median_m", "value": scales["buoy_pair_distance_median_m"], "unit": "m", "comparison_to_12_step_episode": f"{100.0 * scales['max_movement_per_episode_m'] / scales['buoy_pair_distance_median_m']:.4f}% of median pair spacing"},
        {"measure": "buoy_pair_distance_p90_m", "value": scales["buoy_pair_distance_p90_m"], "unit": "m", "comparison_to_12_step_episode": f"{100.0 * scales['max_movement_per_episode_m'] / scales['buoy_pair_distance_p90_m']:.4f}% of p90 pair spacing"},
    ]
    queue = [{"measure": key, "slots": scales[key], "episode_horizon": 12, "fits_within_12": float(scales[key]) <= 12.0} for key in ("instantaneous_queue_service_slots_median", "instantaneous_queue_service_slots_p75", "instantaneous_queue_service_slots_p90")]
    coordination = [{"measure": key, "slots": scales[key], "episode_horizon": 12, "fits_within_12": float(scales[key]) <= 12.0} for key in ("relocation_slots_p75", "relocation_slots_p90")]
    return physical, queue, coordination


def additional_markdown(scales: dict[str, Any]) -> dict[str, str]:
    return {
        "horizon_semantics_spec.md": "# Four-horizon specification\n\n- **Episode horizon `T_episode`**: maximum real decisions before truncation.\n- **Planner horizon `H_plan`**: nominal shadow-search depth.\n- **Task-Waypoint horizon `H_task`**: future label window; the Round-20 builder emits only complete windows.\n- **Policy context `H_c`**: number of historic policy tokens (Round-22/24: 10).\n\nThey are distinct.  The operational planner future at state `t` is `min(H_plan, T_episode-t)`, independently of what a source shadow loop happens to simulate.\n",
        "remaining_horizon_semantics.md": "# Remaining-horizon state semantics\n\nThe maritime observation/info implementation defines `remaining_horizon_fraction=(episode_horizon-time_step)/episode_horizon`.  It is 1 at reset and 0 at normal truncation.  It exposes time-to-boundary to any state consumer; it is not an additional physical variable and it does not extend the executable episode.\n",
        "round30_episode_horizon_provenance.md": "# Round-30 episode-horizon provenance\n\nLocked `src/wt/round30/scenarios.py` sets `EPISODE_HORIZON = 12` and passes it to every training, validation, and fresh scenario.  It inherits the Round-19/20 multi-UAV task convention; it is neither a five-step smoke leftover nor a PPO-runtime shortening.  The earlier five-step premise conflicts with the locked source and report.\n",
        "ppo_rollout_vs_episode_horizon_audit.md": "# PPO rollout buffer versus episode horizon\n\nRound-30 uses `rollout_length=256` independently in each of `num_envs=8`, hence 2048 environment transitions per PPO batch.  Individual environments reset after their finite 12-step terminal/truncation endpoint.  The batch size is therefore not a 2048-step (or five-step) credit horizon.\n",
        "long_horizon_definition_proposal.md": f"# Future long-horizon protocol\n\nReport both `rho_H=H_plan/T_episode` at reset and `rho_remaining(t)=H_plan/(T_episode-t)` at each decision.  Call a result a **long-horizon control** result only when its intended H has adequate full-window coverage and the episode design supports the audited physical scale.  For this task the observed, reproducible design interval is {scales['recommended_minimum_episode_horizon']}--{scales['recommended_primary_episode_horizon']} slots; this is a proposal only, not an environment change.\n",
    }


def physical_timescales(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    first_by_episode: dict[str, dict[str, Any]] = {}
    service_slots: list[float] = []
    for item in transitions:
        if int(item["timestep"]) == 0:
            first_by_episode.setdefault(str(item["episode_id"]), item)
        for queue, payload in zip(item["queue_before_bits"], item["payload_by_buoy_bits"]):
            if float(payload) > 1.0e-9:
                service_slots.append(float(queue) / float(payload))
    nearest: list[float] = []
    pair_distances: list[float] = []
    for item in first_by_episode.values():
        graph = item["graph_state_t"]
        buoys = [node[:2] for node, mask in zip(graph["buoy_nodes"], graph["buoy_mask"]) if mask]
        uavs = [node[:2] for node, mask in zip(graph["uav_nodes"], graph["uav_mask"]) if mask]
        for uav in uavs:
            nearest.append(min(math.dist(uav, buoy) for buoy in buoys))
        for index, left in enumerate(buoys):
            for right in buoys[index + 1:]:
                pair_distances.append(math.dist(left, right))
    def percentile(values: list[float], p: float) -> float:
        ordered = sorted(values)
        index = (len(ordered) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    max_step_m = 10.0
    relocation75 = percentile(pair_distances, 0.75) / max_step_m
    relocation90 = percentile(pair_distances, 0.90) / max_step_m
    queue75, queue90 = percentile(service_slots, 0.75), percentile(service_slots, 0.90)
    t_min = math.ceil(max(10.0, relocation75, queue75))
    t_primary = math.ceil(max(20.0, relocation90, queue90))
    return {
        "data_scope": "Round-20 locked train/validation transitions only",
        "episode_horizon": 12, "delta_t_seconds": 1.0, "max_speed_m_per_s": 10.0,
        "max_movement_per_slot_m": max_step_m, "max_movement_per_episode_m": 120.0,
        "default_map_extent_m": [200.0, 200.0], "default_map_diagonal_m": round(math.sqrt(200.0 ** 2 + 200.0 ** 2), 6),
        "unique_permitted_episodes": len(first_by_episode),
        "nearest_initial_buoy_distance_median_m": round(median(nearest), 6),
        "nearest_initial_buoy_distance_p90_m": round(percentile(nearest, 0.90), 6),
        "buoy_pair_distance_median_m": round(median(pair_distances), 6),
        "buoy_pair_distance_p75_m": round(percentile(pair_distances, 0.75), 6),
        "buoy_pair_distance_p90_m": round(percentile(pair_distances, 0.90), 6),
        "relocation_slots_p75": round(relocation75, 6), "relocation_slots_p90": round(relocation90, 6),
        "instantaneous_queue_service_slots_median": round(median(service_slots), 6),
        "instantaneous_queue_service_slots_p75": round(queue75, 6), "instantaneous_queue_service_slots_p90": round(queue90, 6),
        "recommended_minimum_episode_horizon": t_min,
        "recommended_primary_episode_horizon": t_primary,
        "recommendation_rule": "ceil(max(H10, p75 relocation slots, p75 instantaneous queue-clear slots)) through ceil(max(H20, p90 relocation slots, p90 instantaneous queue-clear slots))",
        "interpretation": "Queue figures are instantaneous Q/service ratios from observed steps, not stationary queue-clear guarantees because arrivals continue.",
    }


def ppo_credit_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    gamma, lam, horizon = 0.99, 0.95, 12
    for t in range(horizon):
        remaining = horizon - t
        rows.append({
            "episode_horizon": horizon, "timestep": t, "remaining_reward_terms": remaining,
            "maximum_future_lag_steps": remaining - 1, "terminal_bootstrap": False,
            "gamma_weight_at_furthest_lag": round(gamma ** (remaining - 1), 10),
            "gae_trace_weight_at_furthest_lag": round((gamma * lam) ** (remaining - 1), 10),
        })
    summary = {
        "episode_horizon": horizon, "gamma": gamma, "gae_lambda": lam,
        "mean_remaining_reward_terms": mean(item["remaining_reward_terms"] for item in rows),
        "mean_maximum_future_lag_steps": mean(item["maximum_future_lag_steps"] for item in rows),
        "terminal_or_truncated_bootstrap": False,
        "rollout_length_per_environment": 256, "num_parallel_environments": 8,
        "transitions_per_ppo_batch": 2048,
        "episode_length_interpretation": "A 256x8 rollout batch contains 2048 environment transitions, not a 2048-step episode; locked scenario episodes truncate at 12 steps.",
    }
    return rows, summary


def cross_round_matrix(planner_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [
        ("round_15", 64, "planning", "H={1,5,10,20}", "finite lattice; causal shadow nominal, oracle copied-env stops on done"),
        ("round_16", 64, "planning", "H={1,5,10,20}", "finite lattice; causal shadow nominal, oracle copied-env stops on done"),
        ("round_17", 64, "planning", "H={1,5,10,20}", "finite lattice; causal shadow nominal, oracle copied-env stops on done"),
        ("round_18", 12, "joint planning", "H={1,5,10}", "joint shadow has no episode clock"),
        ("round_19", 12, "joint planning", "H={1,5,10}", "joint shadow has no episode clock"),
        ("round_20", 12, "Task-Waypoint labels", "H_task={5,10}", "only complete H-step future windows are emitted"),
        ("round_21", 12, "Task-Waypoint predictor", "H_task={5,10}", "inherits Round-20 finite-window labels"),
        ("round_22", 12, "policy context/predictor", "H_task=5; context=10", "uses H5 anchors only"),
        ("round_24", 12, "policy stability extension", "H_task=5; context=10", "same environment as Round-22"),
        ("round_30", 12, "PPO", "GAE over remaining episode", "256x8 batch=2048 transitions, not episode length"),
        ("round_31", 12, "PPO diagnosis", "GAE over remaining episode", "locked report explicitly states twelve-step episodes"),
    ]
    lookup = {(str(item["round"]), int(item["nominal_planning_horizon"])): item for item in planner_summary}
    result = []
    for name, episode, kind, horizons, semantics in entries:
        values = [item for (round_name, _), item in lookup.items() if round_name == name]
        result.append({
            "round": name, "episode_horizon": episode, "component": kind, "registered_horizon_or_context": horizons,
            "effective_future_semantics": semantics,
            "mean_planner_horizon_utilization": "not_applicable" if not values else round(mean(float(item["mean_horizon_utilization_ratio"]) for item in values), 8),
            "planner_source_exceeds_remaining_at_some_states": "not_applicable" if not values else any(bool(item["any_source_shadow_exceeds_remaining"]) for item in values),
        })
    return result


def validity_rows(planner_summary: list[dict[str, Any]], task_coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in planner_summary:
        source_unbounded = bool(item["any_source_shadow_exceeds_remaining"])
        classification = "C" if source_unbounded else str(item["validity_class"])
        rows.append({"artifact": f"{item['round']} planner H{item['nominal_planning_horizon']}", "classification": classification, "basis": "source shadow exceeds remaining real episode" if source_unbounded else "episode-constrained utilization", "mean_hur_or_coverage": item["mean_horizon_utilization_ratio"]})
    for item in task_coverage:
        if item["split"] == "development_test":
            continue
        rows.append({"artifact": f"Round-20 Task-Waypoint H{item['H_task']} {item['split']} K{item['num_buoys']} {item['interference_tier']}", "classification": "A", "basis": "all emitted labels are complete H-step windows", "mean_hur_or_coverage": item["eligible_anchor_fraction_of_episode"]})
    rows.extend([
        {"artifact": "Round-22/24 H5 predictor-policy path", "classification": "B", "basis": "H5 labels are complete but anchors cover only the first 8/12 decision times", "mean_hur_or_coverage": round(8 / 12, 8)},
        {"artifact": "Round-30/31 PPO credit", "classification": "B", "basis": "12-step finite credit is correctly terminal-bounded but is not a 2048-step episode", "mean_hur_or_coverage": round(6.5 / 12, 8)},
    ])
    return rows


def draw_figures(planner_summary: list[dict[str, Any]], task_coverage: list[dict[str, Any]], credit: list[dict[str, Any]], scales: dict[str, Any], equality: list[dict[str, Any]]) -> list[str]:
    specs = [
        ("01_episode_horizon_matrix", "Episode horizon by round", [64, 64, 64, 12, 12, 12, 12, 12, 12, 12, 12]),
        ("02_planning_horizon_utilization", "Mean episode-constrained HUR", [float(item["mean_horizon_utilization_ratio"]) for item in planner_summary]),
        ("03_effective_horizon_by_timestep", "Twelve-step effective future: H5 and H10", [min(5, 12 - t) for t in range(12)]),
        ("04_h5_h10_equality", "Observed H5/H10 action equality", [sum(bool(item["h5_h10_action_equal"]) for item in equality), len(equality)]),
        ("05_task_waypoint_coverage", "Task-Waypoint eligible anchors", [8 / 12, 3 / 12]),
        ("06_task_waypoint_effective", "Task-Waypoint horizons", [5, 10]),
        ("07_ppo_credit_depth", "PPO remaining reward terms", [item["remaining_reward_terms"] for item in credit]),
        ("08_physical_scale", "Physical distance scale (m)", [scales["max_movement_per_episode_m"], scales["default_map_diagonal_m"]]),
        ("09_queue_service_scale", "Instantaneous queue-clear slots", [scales["instantaneous_queue_service_slots_median"], scales["instantaneous_queue_service_slots_p90"]]),
        ("10_cross_round_contract", "Planned H values", [1, 5, 10, 20]),
        ("11_validity_classification", "Recommended episode range", [scales["recommended_minimum_episode_horizon"], scales["recommended_primary_episode_horizon"]]),
        ("12_gate_summary", "Audit gate: C = horizon mismatch", [0, 0, 1]),
    ]
    paths = []
    for name, title, values in specs:
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        ax.bar(range(len(values)), values, color="#3269a8")
        ax.set_title(title)
        ax.set_xlabel("audit item")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = FIGURES / f"{name}.{suffix}"
            fig.savefig(path, dpi=160 if suffix == "png" else None)
            paths.append(path.name)
        plt.close(fig)
    return paths


def markdown_semantics(matrix: list[dict[str, Any]], planner: list[dict[str, Any]], scales: dict[str, Any]) -> str:
    lines = [
        "# Round-32 source semantics audit", "",
        "This file is an audit of locked source and artifact text.  It does not execute an environment, planner, model, or rollout.", "",
        "## Episode and nominal planning horizons", "",
        "| Round | Episode | Component | Registered horizon/context |", "|---|---:|---|---|",
    ]
    lines.extend(f"| {row['round']} | {row['episode_horizon']} | {row['component']} | {row['registered_horizon_or_context']} |" for row in matrix)
    lines.extend([
        "", "## Key distinction", "",
        "A nominal planner shadow is not automatically executable future.  The audit records both the requested shadow depth and `min(H, T-t)`, the finite future available in the real episode.",
        "", "R18/R19 joint shadows carry positions, energy, and queues but no time index; their loop therefore expands nominal H even after the real episode would have no remaining slots.  This is a semantics mismatch, not evidence that the loop ignores H>5.",
        "", f"The permitted Round-20 raw data implies the transparent episode-design range **{scales['recommended_minimum_episode_horizon']}--{scales['recommended_primary_episode_horizon']} slots** under the rule stored in `physical_queue_coordination_timescales.json`.",
    ])
    return "\n".join(lines) + "\n"


def run_audit() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_hashes = {path: sha256(ROOT / path) for path in SOURCE_PATHS}
    planner_detail, planner_summary, raw_by_round = collect_planner_audit()
    equality, prefixes = h5_h10_equality(raw_by_round)
    task_coverage, task_effective, task_semantics, screening, transitions = task_waypoint_audit()
    scales = physical_timescales(transitions)
    credit, ppo_summary = ppo_credit_audit()
    matrix = cross_round_matrix(planner_summary)
    full_matrix = cross_round_semantics(matrix)
    hur_stats = planner_hur_statistics(planner_detail)
    validity = validity_rows(planner_summary, task_coverage)
    task_validity = task_validity_rows(task_coverage)
    physical_rows, queue_rows, coordination_rows = timescale_rows(scales)
    figures = draw_figures(planner_summary, task_coverage, credit, scales, equality)
    write_csv("planner_effective_horizon_audit.csv", planner_detail)
    write_csv("horizon_utilization_ratio.csv", planner_summary)
    write_csv("planner_horizon_utilization.csv", hur_stats)
    write_csv("r18_r19_h5_h10_equality_audit.csv", equality)
    write_csv("h5_h10_equivalence_audit.csv", equality)
    write_csv("r18_r19_trajectory_prefix_equality.csv", prefixes)
    write_csv("dynamic_round_effective_horizon.csv", [row for row in planner_detail if row["round"] in {"round_16", "round_17"}])
    write_csv("task_waypoint_effective_horizon.csv", task_effective)
    write_csv("task_waypoint_horizon_coverage.csv", task_coverage)
    write_csv("ppo_effective_credit_horizon.csv", credit)
    write_csv("cross_round_horizon_matrix.csv", matrix)
    write_csv("cross_round_horizon_semantics.csv", full_matrix)
    write_csv("validity_classification.csv", validity)
    write_csv("historical_planning_horizon_validity.csv", validity)
    write_csv("historical_task_waypoint_horizon_validity.csv", task_validity)
    write_csv("physical_episode_timescale_audit.csv", physical_rows)
    write_csv("queue_service_timescale_audit.csv", queue_rows)
    write_csv("coordination_timescale_audit.csv", coordination_rows)
    write_json("task_waypoint_label_semantics.json", task_semantics)
    write_json("physical_queue_coordination_timescales.json", scales)
    write_json("ppo_credit_horizon_summary.json", ppo_summary)
    write_json("access_audit.json", {
        "audit_mode": "read_only_locked_artifacts", "fresh_new_rollout_count": 0,
        "untouched_eval_accessed": False, "round20_development_test_decoded_count": 0,
        "round20_split_screening": screening, "environment_transition_executed": False,
        "planner_executed": False, "model_or_optimizer_executed": False,
    })
    gate = {
        "gate": "R32-C", "episode_horizon_gate": "EH-C", "status": "locked_audit_result",
        "conclusion": "The twelve-step multi-UAV task supplies finite H5/H10 labels and correctly terminal-bounded PPO credit, but long nominal planning shadows are not episode-clock-aware and H20 is not a realizable complete future in a 12-step episode.",
        "planner_horizon_claim_supported": False, "task_waypoint_label_claim_supported": True,
        "ppo_credit_semantics_supported": True, "requires_environment_or_planner_change": "not_authorized_in_round32",
        "episode_horizon_redesign_required": True,
        "recommended_episode_horizon_range": [scales["recommended_minimum_episode_horizon"], scales["recommended_primary_episode_horizon"]],
        "fresh_new_rollout_count": 0, "untouched_eval_accessed": False, "round33_created": False,
    }
    write_json("round32_horizon_semantics_gate.json", gate)
    write_json("round32_readiness_gate.json", gate)
    write_json("episode_horizon_sufficiency_gate.json", {
        "gate": "EH-C", "episode_horizon": 12, "episode_horizon_redesign_required": True,
        "basis": {"H10_full_state_fraction_R18_R19": 0.25, "H20_complete_future_in_12_step_task": False, "p90_relocation_slots": scales["relocation_slots_p90"], "recommended_range": [scales["recommended_minimum_episode_horizon"], scales["recommended_primary_episode_horizon"]]},
        "interpretation": "The present episode supports short and some medium decisions, but it cannot substantiate a terminal-aware H20 or general long-horizon control claim.",
    })
    reassessment = {
        "round_19_original_gate": "IPN-A", "original_gate_mutated": False,
        "interpretation_validity": "effective-horizon-limited",
        "supported_claim": "The locked Round-19 result supports a historical multi-step-lookahead-versus-H1 finding under its registered nominal shadow protocol.",
        "unsupported_claim": "It does not isolate a terminal-aware H10-specific long-horizon advantage: H5/H10 first actions and observed prefixes are identical in the locked trace, scores are not persisted, and joint shadows have no episode clock.",
    }
    write_json("historical_gate_horizon_reassessment.json", reassessment)
    write_csv("historical_gate_horizon_reassessment.csv", [{"round": "round_19", "historical_gate": reassessment["round_19_original_gate"], "original_gate_mutated": False, "interpretation_validity": reassessment["interpretation_validity"], "supported_claim": reassessment["supported_claim"], "unsupported_claim": reassessment["unsupported_claim"]}])
    for filename, contents in additional_markdown(scales).items():
        (OUTPUT / filename).write_text(contents, encoding="utf-8")
    (OUTPUT / "source_semantics_audit.md").write_text(markdown_semantics(matrix, planner_summary, scales), encoding="utf-8")
    manifest = {
        "round": "round_32", "audit_type": "cross_round_effective_horizon_semantics",
        "source_hashes": source_hashes, "input_trace_paths": {name: path.relative_to(ROOT).as_posix() for name, (_, _, path, _) in TRACE_SPECS.items()},
        "output_files": sorted(path.name for path in OUTPUT.iterdir() if path.is_file()),
        "figure_files": sorted(figures), "fresh_new_rollout_count": 0, "untouched_eval_accessed": False,
        "round20_development_test_decoded_count": 0, "environment_transition_executed": False,
    }
    write_json("run_manifest.json", manifest)
    return {"gate": gate, "scales": scales, "planner_rows": len(planner_detail), "task_coverage_rows": len(task_coverage), "figure_count": len(figures)}


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2, sort_keys=True))
