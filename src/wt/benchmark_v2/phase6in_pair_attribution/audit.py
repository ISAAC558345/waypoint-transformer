"""Attribute saved local-swap traces. No new rollout and no training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import PRIMARY_SEED
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ij_executor_audit.audit import buoy_set
from wt.benchmark_v2.phase6im_local_swap.audit import (
    FLOORS,
    HIGHER_IS_BETTER,
    _label_table,
    single_swaps,
    transposition_distance,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/phase6im_local_swap" / "states.json"
PRIOR = ROOT / "results/phase6ik_execution_closure" / "states.json"
AUDIT_DIR = ROOT / "results/phase6in_pair_attribution"
BIT_TOLERANCE = 1.0
MATERIAL_BITS = 1.0e6
SINR_FLOOR = 0.05
INTERFERENCE_FLOOR = 0.01
ENTRY_DISTANCE_FLOOR_M = 0.5
AGREEMENT_BAR = 0.75
DRAWS = 10000
SEED = 0
DECISION_FEATURES = (
    "distance_m",
    "steps_to_radius",
    "hypothetical_rate_bps",
    "geometric_rate_bps",
    "geometric_interference_w",
    "service_overlap_s",
)


def assign_gate(blocked: bool, consistent_features: int) -> str:
    if blocked:
        return "PAIR_SWAP_ATTRIBUTION_BLOCKED"
    if int(consistent_features) > 0:
        return "PAIR_SWAP_ATTRIBUTION_SUPPORTED"
    return "PAIR_SWAP_ATTRIBUTION_INCONCLUSIVE"


def _key(scenario: int, assignment: list[int] | tuple[int, ...]) -> tuple:
    return (int(scenario), 15, tuple(int(item) for item in assignment), int(PRIMARY_SEED))


def _bootstrap_mean(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    draws = np.empty(DRAWS, dtype=np.float64)
    for index in range(DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, data.size, data.size)]))
    return float(np.mean(data)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _feature_agreement(rows: list[dict]) -> dict:
    report = {}
    for name in DECISION_FEATURES:
        higher = name in HIGHER_IS_BETTER
        floor = float(FLOORS[name])
        pooled = {"agree": 0, "disagree": 0, "unresolved": 0}
        per_state = []
        for row in rows:
            agree = disagree = 0
            for swap in row["swaps"]:
                if abs(float(swap["delta_bits"])) <= BIT_TOLERANCE:
                    continue
                delta = float(swap["features"][name]) - float(row["baseline_features"][name])
                if abs(delta) <= floor:
                    pooled["unresolved"] += 1
                    continue
                matches = (delta > 0.0) if higher else (delta < 0.0)
                if matches == (float(swap["delta_bits"]) > 0.0):
                    agree += 1
                    pooled["agree"] += 1
                else:
                    disagree += 1
                    pooled["disagree"] += 1
            if agree + disagree:
                per_state.append(agree / (agree + disagree))
        mean, low, high = _bootstrap_mean(per_state)
        report[name] = {
            "pooled": pooled,
            "states_with_movement": int(len(per_state)),
            "mean_state_agreement": mean,
            "agreement_ci95": [low, high],
            "ci_above_bar": int(np.isfinite(low) and low > AGREEMENT_BAR),
        }
    return report


def _mechanisms(swap: dict) -> dict[str, int]:
    bits = float(swap["delta_bits"])
    entries = np.asarray(swap["entry_delta"], dtype=np.int64)
    earlier = bool(np.any(entries <= -1))
    same_entry = bool(np.all(entries == 0))
    uav_bits = np.asarray(swap["uav_bit_delta"], dtype=np.float64)
    opposing = bool(np.any(uav_bits > 1.0e5) and np.any(uav_bits < -1.0e5))
    small_total = abs(bits) <= 1.0e5
    quiet_channel = (
        abs(float(swap["distance_delta_m"])) < 1.0
        and abs(float(swap["hypothetical_rate_delta_bps"])) < 1.0e4
        and abs(float(swap["sinr_delta"])) < SINR_FLOOR
        and abs(float(swap["interference_delta_w"])) < INTERFERENCE_FLOOR
    )
    service_moved = abs(int(swap["service_step_delta"])) >= 5
    return {
        "A_earlier_entry_and_more_bits": int(bits > BIT_TOLERANCE and earlier),
        "B_same_entry_channel_shift": int(
            same_entry
            and abs(bits) > BIT_TOLERANCE
            and (abs(float(swap["sinr_delta"])) > SINR_FLOOR or abs(float(swap["interference_delta_w"])) > INTERFERENCE_FLOOR)
        ),
        "C_uav_cancellation_small_total": int(opposing and small_total),
        "D_quiet_inputs_service_steps_move": int(quiet_channel and service_moved and abs(bits) > 1.0e5),
    }


def _quantile(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"n": 0}
    return {
        "n": int(data.size),
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p90": float(np.percentile(data, 90)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def _double_swap_labels(base: np.ndarray, labels: dict[tuple[int, ...], float]) -> dict:
    base_key = tuple(int(item) for item in base)
    base_bits = float(labels[base_key])
    singles = {(int(left), int(right)): row for left, right, row in single_swaps(base)}
    single_delta = {
        pair: float(labels[tuple(int(item) for item in row)]) - base_bits for pair, row in singles.items()
    }
    grouped: dict[tuple[int, ...], list[dict]] = {}
    missing = 0
    pairs = list(singles)
    for first in pairs:
        for second in pairs:
            if first == second:
                continue
            composed = np.asarray(base, dtype=np.int64).copy()
            composed[list(first)] = composed[list(first)][::-1]
            composed[list(second)] = composed[list(second)][::-1]
            key = tuple(int(item) for item in composed)
            if key not in labels:
                missing += 1
                continue
            joint = float(labels[key]) - base_bits
            additive = float(single_delta[first] + single_delta[second])
            involved = len({first[0], first[1], second[0], second[1]})
            grouped.setdefault(key, []).append(
                {
                    "kind": "disjoint" if involved == 4 else "three_cycle",
                    "joint_bits": joint,
                    "additive_bits": additive,
                    "interaction_bits": joint - additive,
                    "single_a_bits": float(single_delta[first]),
                    "single_b_bits": float(single_delta[second]),
                }
            )

    def _direction(kind: str) -> dict:
        groups = [items for items in grouped.values() if items[0]["kind"] == kind]
        interactions = [float(np.median([item["interaction_bits"] for item in items])) for items in groups]
        joints = [float(items[0]["joint_bits"]) for items in groups]
        compared = [items for items in groups if abs(items[0]["joint_bits"]) > BIT_TOLERANCE]
        same = 0
        for items in compared:
            matches = [
                int(np.sign(item["joint_bits"]) == np.sign(item["additive_bits"]) and abs(item["additive_bits"]) > BIT_TOLERANCE)
                for item in items
            ]
            same += int(sum(matches) > len(matches) / 2.0)
        low_singles_high_joint = 0
        for items in groups:
            if any(max(item["single_a_bits"], item["single_b_bits"]) < 1.0e5 and item["joint_bits"] > MATERIAL_BITS for item in items):
                low_singles_high_joint += 1
        return {
            "n": int(len(groups)),
            "missing_labels": int(missing),
            "interaction": _quantile(interactions),
            "joint": _quantile(joints),
            "direction_compared": int(len(compared)),
            "sum_sign_matches_joint": int(same),
            "both_singles_below_1e5_joint_above_1e6": int(low_singles_high_joint),
        }

    return {"disjoint": _direction("disjoint"), "three_cycles": _direction("three_cycle")}


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    if int(PRIMARY_SEED) != 0:
        raise RuntimeError("primary executor seed is no longer 0")
    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    prior = {int(row["scenario_id"]): row for row in json.loads(PRIOR.read_text(encoding="utf-8"))}
    keys = []
    coverage_gaps = []
    sum_errors = []
    prepared = []
    mechanism_counts = {"A_earlier_entry_and_more_bits": 0, "B_same_entry_channel_shift": 0, "C_uav_cancellation_small_total": 0, "D_quiet_inputs_service_steps_move": 0}
    class_bits = {name: [] for name in mechanism_counts}
    examples = {name: None for name in mechanism_counts}
    spill_shares = []
    cancellation = []
    entry_distance_missing = 0
    same_entry_material = 0
    changed_uav_entry_earlier = 0
    unchanged_uav_bit_dominates = 0
    material_swaps = 0
    for row in saved:
        scenario = int(row["scenario_id"])
        if scenario in RESERVED_TEST or scenario not in VALIDATION_POOL:
            raise RuntimeError("local swap file contains a scenario outside validation")
        labels = _label_table(scenario)
        base_assignment = [int(item) for item in row["assignments"]["scheme_c"]]
        base = next(arm for arm in row["arms"] if arm["role"] == "scheme_c")
        if [int(item) for item in base["assignment"]] != base_assignment:
            coverage_gaps.append({"scenario_id": scenario, "reason": "baseline assignment mismatch"})
        base_uav = base["rollout"]["per_uav"]
        swaps = []
        for arm in row["arms"]:
            assignment = [int(item) for item in arm["assignment"]]
            keys.append(_key(scenario, assignment))
            per_uav = arm["rollout"]["per_uav"]
            attributed = float(sum(float(item["delivered_bits"]) for item in per_uav))
            total = float(arm["rollout"]["delivered_bits"])
            sum_errors.append(abs(attributed - total))
            if any(item["entry_distance_m"] is None for item in per_uav):
                entry_distance_missing += 1
            if arm["role"] != "single_swap":
                continue
            bit_delta = [float(item["delivered_bits"]) - float(before["delivered_bits"]) for item, before in zip(per_uav, base_uav)]
            entry_delta = [int(item["entry"]) - int(before["entry"]) for item, before in zip(per_uav, base_uav)]
            changed = [int(arm["uav_i"]), int(arm["uav_j"])]
            unchanged = [index for index in range(6) if index not in changed]
            changed_abs = float(sum(abs(bit_delta[index]) for index in changed))
            unchanged_abs = float(sum(abs(bit_delta[index]) for index in unchanged))
            gross = changed_abs + unchanged_abs
            if gross > BIT_TOLERANCE:
                share = unchanged_abs / gross
                spill_shares.append(share)
                if share > 0.5:
                    unchanged_uav_bit_dominates += 1
            if gross > BIT_TOLERANCE:
                cancellation.append(1.0 - abs(float(arm["delta_bits"])) / gross)
            swap = {
                "scenario_id": scenario,
                "uav_i": int(arm["uav_i"]),
                "uav_j": int(arm["uav_j"]),
                "assignment": assignment,
                "delta_bits": float(arm["delta_bits"]),
                "uav_bit_delta": bit_delta,
                "entry_delta": entry_delta,
                "service_step_delta": int(arm["rollout"]["service_steps"]) - int(base["rollout"]["service_steps"]),
                "sinr_delta": float(arm["rollout"]["mean_sinr"]) - float(base["rollout"]["mean_sinr"]),
                "interference_delta_w": float(arm["rollout"]["mean_interference_w"]) - float(base["rollout"]["mean_interference_w"]),
                "distance_delta_m": float(arm["features"]["distance_m"]) - float(base["features"]["distance_m"]),
                "hypothetical_rate_delta_bps": float(arm["features"]["hypothetical_rate_bps"]) - float(base["features"]["hypothetical_rate_bps"]),
                "features": arm["features"],
                "buoy_set_unchanged": int(buoy_set(np.asarray(assignment)) == buoy_set(np.asarray(base_assignment))),
                "projector_steps": int(arm["rollout"]["projector_steps"]),
                "baseline_projector_steps": int(base["rollout"]["projector_steps"]),
                "conflict_steps": int(arm["rollout"]["conflict_steps"]),
            }
            swaps.append(swap)
            if abs(float(arm["delta_bits"])) > MATERIAL_BITS:
                material_swaps += 1
            if all(delta == 0 for delta in entry_delta) and abs(float(arm["delta_bits"])) > BIT_TOLERANCE:
                same_entry_material += 1
            if any(entry_delta[index] <= -1 for index in changed) and float(arm["delta_bits"]) > BIT_TOLERANCE:
                changed_uav_entry_earlier += 1
            flags = _mechanisms(swap)
            for name, flag in flags.items():
                mechanism_counts[name] += int(flag)
                if flag:
                    class_bits[name].append(float(arm["delta_bits"]))
                    current = examples[name]
                    if current is None or abs(float(arm["delta_bits"])) > abs(float(current["delta_bits"])):
                        examples[name] = {
                            "scenario_id": scenario,
                            "uavs": changed,
                            "assignment": assignment,
                            "baseline": base_assignment,
                            "delta_bits": float(arm["delta_bits"]),
                            "uav_bit_delta": [round(item, 3) for item in bit_delta],
                            "entry_delta": entry_delta,
                            "sinr_delta": float(swap["sinr_delta"]),
                            "interference_delta_w": float(swap["interference_delta_w"]),
                            "distance_delta_m": float(swap["distance_delta_m"]),
                            "service_step_delta": int(swap["service_step_delta"]),
                        }
        expected = {tuple(int(item) for item in row_assignment) for _left, _right, row_assignment in single_swaps(np.asarray(base_assignment))}
        observed = {tuple(item["assignment"]) for item in swaps}
        if observed != expected:
            coverage_gaps.append({"scenario_id": scenario, "missing": len(expected - observed), "extra": len(observed - expected)})
        prepared.append(
            {
                "scenario_id": scenario,
                "baseline_features": base["features"],
                "swaps": swaps,
                "doubles": _double_swap_labels(np.asarray(base_assignment, dtype=np.int64), labels),
                "transposition_distance": transposition_distance(np.asarray(base_assignment), np.asarray(row["assignments"]["oracle"])),
            }
        )
    unique_keys = set(keys)
    prior_keys = []
    for scenario, row in prior.items():
        for name in ("oracle", "scheme_c", "epoch6"):
            prior_keys.append(_key(scenario, row["assignments"][name]))
    repeated = [key for key in prior_keys if key in unique_keys]
    blocked = bool(coverage_gaps or max(sum_errors, default=0.0) > BIT_TOLERANCE or len(unique_keys) != 256 or len(keys) != 256)
    features = _feature_agreement(prepared)
    consistent = sum(int(item["ci_above_bar"]) for item in features.values())
    disjoint_interaction = [float(row["doubles"]["disjoint"]["interaction"]["median"]) for row in prepared]
    cycle_interaction = [float(row["doubles"]["three_cycles"]["interaction"]["median"]) for row in prepared]
    low_to_high = sum(int(row["doubles"]["disjoint"]["both_singles_below_1e5_joint_above_1e6"]) + int(row["doubles"]["three_cycles"]["both_singles_below_1e5_joint_above_1e6"]) for row in prepared)
    summary = {
        "gate": assign_gate(blocked, consistent),
        "executed_records": int(len(keys)),
        "unique_keys": int(len(unique_keys)),
        "states": int(len(prepared)),
        "single_swaps": int(sum(len(row["swaps"]) for row in prepared)),
        "double_swap_labels_per_state": [int(row["doubles"]["disjoint"]["n"] + row["doubles"]["three_cycles"]["n"]) for row in prepared],
        "double_swap_executions": 0,
        "repeated_keys_also_in_phase6ik": int(len(repeated)),
        "coverage_gaps": coverage_gaps,
        "uav_sum_abs_error_max": float(max(sum_errors, default=0.0)),
        "entry_distance_missing_arms": int(entry_distance_missing),
        "first_positive_service_step_stored": 0,
        "per_uav_projector_stored": 0,
        "idle_events_stored": 0,
        "material_swaps": int(material_swaps),
        "same_entry_and_nonzero_bits": int(same_entry_material),
        "changed_uav_earlier_and_more_bits": int(changed_uav_entry_earlier),
        "unchanged_uav_abs_share_above_half": int(unchanged_uav_bit_dominates),
        "unchanged_uav_abs_share": _quantile(spill_shares),
        "cancellation_ratio": _quantile(cancellation),
        "mechanisms": {name: {"n": int(mechanism_counts[name]), "delta_bits": _quantile(class_bits[name]), "example": examples[name]} for name in mechanism_counts},
        "decision_features": features,
        "double_swaps": {
            "disjoint_median_interaction_by_state": _quantile(disjoint_interaction),
            "three_cycle_median_interaction_by_state": _quantile(cycle_interaction),
            "low_singles_high_joint": int(low_to_high),
            "per_state": [
                {
                    "scenario_id": int(row["scenario_id"]),
                    "disjoint": row["doubles"]["disjoint"],
                    "three_cycles": row["doubles"]["three_cycles"],
                }
                for row in prepared
            ],
        },
        "agreement_bar": AGREEMENT_BAR,
        "consistent_features": int(consistent),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"gate": summary["gate"], "unique_keys": summary["unique_keys"], "uav_sum_abs_error_max": summary["uav_sum_abs_error_max"], "consistent_features": consistent}), flush=True)
    return summary


if __name__ == "__main__":
    run_audit()
