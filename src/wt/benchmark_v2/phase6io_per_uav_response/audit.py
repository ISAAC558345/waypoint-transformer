"""Minimal per-UAV traces for four saved diagnostic swaps. No training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.value import _zero_arrivals
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL, HOLD, SERVICE_SAFETY_MARGIN_M
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import PRIMARY_SEED
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ij_executor_audit.audit import _matches_saved, _reach
from wt.benchmark_v2.phase6ik_execution_closure.audit import _rng_token
from wt.benchmark_v2.phase6im_local_swap.audit import _label_table
from wt.l3a.env import conflict_mask
from wt.l3b.env import interference_matrix, path_gain
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.safety_projector import SafetyProjector


ROOT = Path(__file__).resolve().parents[4]
PRIOR_STEPS = ROOT / "results/phase6ik_execution_closure" / "states.json"
AUDIT_DIR = ROOT / "results/phase6io_per_uav_response"
BIT_TOLERANCE = 1.0
MAX_NEW_ROLLOUTS = 12
UNCHANGED_SHARE = 0.25
# UAV indices are the assignment vector positions used in Phase-6I-M/N.
CASES = (
    {
        "scenario_id": 322051,
        "name": "322051_uav1_uav2",
        "reuse": "phase6ik",
        "changed": (1, 2),
        "arms": {
            "baseline": (9, 6, 10, 8, 11, 7),
            "swap": (9, 10, 6, 8, 11, 7),
        },
    },
    {
        "scenario_id": 322049,
        "name": "322049_uav3_uav4",
        "reuse": "",
        "changed": (3, 4),
        "arms": {
            "baseline": (8, 7, 9, 11, 10, 6),
            "swap": (8, 7, 9, 10, 11, 6),
        },
    },
    {
        "scenario_id": 322062,
        "name": "322062_uav0_uav4",
        "reuse": "",
        "changed": (0, 4),
        "arms": {
            "baseline": (10, 11, 8, 6, 9, 7),
            "swap": (9, 11, 8, 6, 10, 7),
        },
    },
    {
        "scenario_id": 322062,
        "name": "322062_uav1_uav2",
        "reuse": "",
        "changed": (1, 2),
        "arms": {
            "baseline": (10, 11, 8, 6, 9, 7),
            "swap": (10, 8, 11, 6, 9, 7),
        },
    },
    {
        "scenario_id": 322053,
        "name": "322053_two_swap",
        "reuse": "",
        "changed": (0, 2, 4),
        "arms": {
            "baseline": (10, 6, 11, 8, 9, 7),
            "single_0_2": (11, 6, 10, 8, 9, 7),
            "single_2_4": (10, 6, 9, 8, 11, 7),
            "joint": (11, 6, 9, 8, 10, 7),
        },
    },
)


def assign_gate(blocked: bool, attributable: bool, simple_contrasts: int, large_contrasts: int) -> str:
    if blocked or not attributable:
        return "PER_UAV_ATTRIBUTION_BLOCKED"
    if large_contrasts > 0 and simple_contrasts == large_contrasts:
        return "PER_UAV_ATTRIBUTION_SUPPORTED"
    return "PER_UAV_ATTRIBUTION_INCONCLUSIVE"


def compose_swaps(base: tuple[int, ...], first: tuple[int, int], second: tuple[int, int]) -> tuple[int, ...]:
    row = list(base)
    row[first[0]], row[first[1]] = row[first[1]], row[first[0]]
    row[second[0]], row[second[1]] = row[second[1]], row[second[0]]
    return tuple(row)


def _move(env, actor, option, last: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    observation = build_actor_observation(env, option, last)
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(observation["self"], dtype=torch.float32),
            torch.as_tensor(observation["task"], dtype=torch.float32),
            torch.as_tensor(observation["communication"], dtype=torch.float32),
            torch.as_tensor(observation["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(observation["d_nearest"], dtype=torch.float32),
        )
    delta = float(ALPHA_RESIDUAL) * np.tanh(mu.detach().cpu().numpy().astype(np.float64))
    desired = clip_unit_step(fly_toward(env, option.association) + delta)
    projected = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=float(SERVICE_SAFETY_MARGIN_M)).project(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    movement = clip_unit_step(executed)
    if failure:
        touched = [1] * int(env.config.num_uavs)
    else:
        gap = np.linalg.norm(np.asarray(projected.projected_actions, dtype=np.float64) - desired, axis=-1)
        touched = [int(item > 1.0e-3) for item in gap]
    team = int(failure or float(projected.intervention_norm) > 1.0e-3)
    return movement, team, touched


def _empty_uav(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append({"uav": index, "entry": int(HOLD), "first_served": int(HOLD), "service_steps": 0, "delivered_bits": 0.0, "completed_blocks": 0.0, "sinr": [], "interference_w": [], "rate_bps": []})
    return rows


def _finish(per_uav: list[dict], steps: list[dict], delivered: float, rng_consumed: int, switches: list[int], position_stored: int) -> dict:
    summary = []
    for item in per_uav:
        summary.append(
            {
                "uav": int(item["uav"]),
                "entry": int(item["entry"]),
                "first_served": int(item["first_served"]),
                "service_steps": int(item["service_steps"]),
                "delivered_bits": float(item["delivered_bits"]),
                "completed_blocks": float(item["completed_blocks"]),
                "mean_sinr": float(np.mean(item["sinr"])) if item["sinr"] else 0.0,
                "mean_interference_w": float(np.mean(item["interference_w"])) if item["interference_w"] else 0.0,
                "mean_rate_bps": float(np.mean(item["rate_bps"])) if item["rate_bps"] else 0.0,
                "end_progress": float(steps[-1]["uavs"][item["uav"]]["progress"]) if steps else 0.0,
            }
        )
    return {
        "delivered_bits": float(delivered),
        "per_uav_bit_sum": float(sum(item["delivered_bits"] for item in summary)),
        "rng_consumed": int(rng_consumed),
        "switches": switches,
        "position_stored": int(position_stored),
        "per_uav": summary,
        "steps": steps,
    }


def _rollout(env, actor, snap: dict, assignment: np.ndarray) -> dict:
    restore_state(env, snap)
    _zero_arrivals(env)
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1).copy()
    option = make_option(env, chosen)
    if int(option.remaining_hold) != int(HOLD):
        raise RuntimeError("hold was not reset to 15")
    rng_before = _rng_token(env)
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    served_k = np.asarray(snap["served_k"], dtype=np.int64).reshape(-1)
    switches = [int(int(buoy) != int(previous)) for buoy, previous in zip(chosen, served_k)]
    per_uav = _empty_uav(chosen.shape[0])
    steps = []
    delivered = 0.0
    for step_index in range(int(HOLD)):
        movement, team, touched = _move(env, actor, option, last)
        info = env.step(movement, option.association)
        if not np.array_equal(np.asarray(env.assignment, dtype=np.int64), chosen):
            raise RuntimeError("assignment changed during the hold")
        served = np.asarray(info["served_bits"], dtype=np.float64)
        blocks = np.asarray(info["completed_block_equivalent_increment"], dtype=np.float64)
        matrix = interference_matrix(env)
        delivered += float(np.sum(served))
        row = []
        for uav, buoy in enumerate(chosen):
            target = int(buoy)
            idle = int(target < 0)
            position = [float(item) for item in env.uav_positions[uav]]
            distance = None if idle else float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[target]))
            inside = int((not idle) and distance <= float(env.config.service_radius_m))
            bits = 0.0 if idle else float(served[target])
            signal = 0.0 if idle else float(path_gain(env.config, env.uav_positions[uav], env.buoy_positions[target])) * float(env.config.tx_power)
            interference = float(np.sum(matrix[uav]))
            if inside and per_uav[uav]["entry"] == int(HOLD):
                per_uav[uav]["entry"] = int(step_index)
            if bits > 0.0 and per_uav[uav]["first_served"] == int(HOLD):
                per_uav[uav]["first_served"] = int(step_index)
            if float(env.last_rate[uav]) > 0.0:
                per_uav[uav]["service_steps"] += 1
                per_uav[uav]["sinr"].append(float(env.last_sinr[uav]))
                per_uav[uav]["interference_w"].append(interference)
                per_uav[uav]["rate_bps"].append(float(env.last_rate[uav]))
            per_uav[uav]["delivered_bits"] += bits
            per_uav[uav]["completed_blocks"] += 0.0 if idle else float(blocks[target])
            row.append(
                {
                    "uav": int(uav),
                    "target": target,
                    "position_m": position,
                    "distance_m": distance,
                    "in_range": inside,
                    "sinr": float(env.last_sinr[uav]),
                    "interference_w": interference,
                    "signal_w": signal,
                    "rate_bps": float(env.last_rate[uav]),
                    "progress": float(env.progress[uav]),
                    "served_bits": bits,
                    "completed_blocks": 0.0 if idle else float(blocks[target]),
                    "idle": idle,
                    "projector": int(touched[uav]),
                    "conflict": int(bool(conflict_mask(env.assignment, env.config.num_buoys)[uav])),
                }
            )
        steps.append({"delivered_bits": float(np.sum(served)), "projector": int(team), "uavs": row})
        option.remaining_hold -= 1
        last = movement
    return _finish(per_uav, steps, delivered, int(_rng_token(env) != rng_before), switches, 1)


def _from_saved_steps(trace_steps: list[dict], assignment: list[int], served_k: list[int]) -> dict:
    chosen = [int(item) for item in assignment]
    per_uav = _empty_uav(len(chosen))
    steps = []
    delivered = 0.0
    for step_index, step in enumerate(trace_steps):
        delivered += float(step["delivered_bits"])
        row = []
        for uav, item in enumerate(step["uavs"]):
            bits = float(item["served_bits"])
            inside = int(item["in_range"])
            if inside and per_uav[uav]["entry"] == int(HOLD):
                per_uav[uav]["entry"] = int(step_index)
            if bits > 0.0 and per_uav[uav]["first_served"] == int(HOLD):
                per_uav[uav]["first_served"] = int(step_index)
            if bits > 0.0:
                per_uav[uav]["service_steps"] += 1
                per_uav[uav]["sinr"].append(float(item["sinr"]))
                per_uav[uav]["interference_w"].append(float(item["interference_w"]))
                per_uav[uav]["rate_bps"].append(float(item["rate_bps"]))
            per_uav[uav]["delivered_bits"] += bits
            per_uav[uav]["completed_blocks"] += float(item["completed_blocks"])
            row.append(
                {
                    "uav": int(uav),
                    "target": int(item["target"]),
                    "position_m": None,
                    "distance_m": None if item["distance_m"] != item["distance_m"] else float(item["distance_m"]),
                    "in_range": inside,
                    "sinr": float(item["sinr"]),
                    "interference_w": float(item["interference_w"]),
                    "signal_w": float(item["signal_w"]),
                    "rate_bps": float(item["rate_bps"]),
                    "progress": float(item["progress"]),
                    "served_bits": bits,
                    "completed_blocks": float(item["completed_blocks"]),
                    "idle": int(item["idle"]),
                    "projector": None,
                    "conflict": int(item["conflict"]),
                }
            )
        steps.append({"delivered_bits": float(step["delivered_bits"]), "projector": int(step["projector"]), "uavs": row})
    switches = [int(int(buoy) != int(previous)) for buoy, previous in zip(chosen, served_k)]
    return _finish(per_uav, steps, delivered, 0, switches, 0)


def _contrast(base: dict, other: dict, changed: tuple[int, ...]) -> dict:
    base_bits = np.asarray([item["delivered_bits"] for item in base["per_uav"]], dtype=np.float64)
    other_bits = np.asarray([item["delivered_bits"] for item in other["per_uav"]], dtype=np.float64)
    delta = other_bits - base_bits
    total = float(np.sum(delta))
    changed_net = float(np.sum(delta[list(changed)]))
    unchanged_net = total - changed_net
    gross = float(np.sum(np.abs(delta)))
    simple = int(abs(total) > 1.0e6 and abs(unchanged_net) <= UNCHANGED_SHARE * abs(total) and np.sign(changed_net) == np.sign(total))
    rows = []
    for index, (before, after) in enumerate(zip(base["per_uav"], other["per_uav"])):
        rows.append(
            {
                "uav": index,
                "changed_target": int(index in changed),
                "target_before": int(before.get("target", -1)) if "target" in before else None,
                "entry_delta": int(after["entry"]) - int(before["entry"]),
                "first_served_delta": int(after["first_served"]) - int(before["first_served"]),
                "service_step_delta": int(after["service_steps"]) - int(before["service_steps"]),
                "sinr_delta": float(after["mean_sinr"]) - float(before["mean_sinr"]),
                "interference_delta_w": float(after["mean_interference_w"]) - float(before["mean_interference_w"]),
                "bit_delta": float(delta[index]),
                "end_progress_delta": float(after["end_progress"]) - float(before["end_progress"]),
            }
        )
    return {"total_bit_delta": total, "changed_uav_bit_delta": changed_net, "unchanged_uav_bit_delta": unchanged_net, "gross_abs_bit_delta": gross, "simple": simple, "uavs": rows}


def _with_targets(trace: dict, assignment: tuple[int, ...]) -> dict:
    for item, target in zip(trace["per_uav"], assignment):
        item["target"] = int(target)
    return trace


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    if compose_swaps((10, 6, 11, 8, 9, 7), (0, 2), (2, 4)) != (11, 6, 9, 8, 10, 7):
        raise RuntimeError("322053 two-swap composition does not match the saved joint")
    fresh_keys = []
    for case in CASES:
        if case["reuse"]:
            continue
        for assignment in case["arms"].values():
            key = (int(case["scenario_id"]), tuple(assignment))
            if key not in fresh_keys:
                fresh_keys.append(key)
    new_rollouts = len(fresh_keys)
    if new_rollouts > MAX_NEW_ROLLOUTS:
        raise RuntimeError("diagnostic rollout budget exceeded")
    limit_threads()
    actor = load_primary()
    token = parameter_token(actor)
    saved = {int(row["scenario_id"]): row for row in json.loads(PRIOR_STEPS.read_text(encoding="utf-8"))}
    traces = {}
    blocked = False
    label_errors = []
    sum_errors = []
    reached = {}
    for case in CASES:
        scenario = int(case["scenario_id"])
        labels = _label_table(scenario)
        arm_traces = {}
        if case["reuse"] == "phase6ik":
            source = saved[scenario]
            for name, assignment in case["arms"].items():
                key = "scheme_c" if name == "baseline" else "oracle"
                trace = _from_saved_steps(source["steps"][key], list(assignment), [-1] * len(assignment))
                trace["switches"] = [None] * len(assignment)
                arm_traces[name] = _with_targets(trace, assignment)
        else:
            if scenario not in reached:
                env = _reach(scenario, actor)
                matched = _matches_saved(env, scenario)
                if int(matched["matched"]) != 1:
                    blocked = True
                reached[scenario] = {"env": env, "snap": capture_state(env), "matched": matched, "traces": {}}
            bucket = reached[scenario]
            for name, assignment in case["arms"].items():
                key = tuple(assignment)
                if key not in bucket["traces"]:
                    bucket["traces"][key] = _rollout(bucket["env"], actor, bucket["snap"], np.asarray(assignment, dtype=np.int64))
                arm_traces[name] = _with_targets(bucket["traces"][key], assignment)
        for name, assignment in case["arms"].items():
            trace = arm_traces[name]
            label = float(labels[tuple(assignment)])
            error = abs(float(trace["delivered_bits"]) - label)
            attributed = abs(float(trace["per_uav_bit_sum"]) - float(trace["delivered_bits"]))
            label_errors.append(error)
            sum_errors.append(attributed)
            if error > BIT_TOLERANCE or attributed > BIT_TOLERANCE or int(trace["rng_consumed"]) == 1:
                blocked = True
            trace["label_abs_error"] = float(error)
            trace["attribution_abs_error"] = float(attributed)
        traces[case["name"]] = arm_traces
    if parameter_token(actor) != token:
        blocked = True
    comparisons = []
    simple = 0
    large = 0
    for case in CASES:
        if case["name"] == "322053_two_swap":
            continue
        contrast = _contrast(traces[case["name"]]["baseline"], traces[case["name"]]["swap"], case["changed"])
        if abs(contrast["total_bit_delta"]) > 1.0e6:
            large += 1
            simple += int(contrast["simple"])
        comparisons.append({"name": case["name"], "scenario_id": int(case["scenario_id"]), "changed": list(case["changed"]), **contrast})
    joint = traces["322053_two_swap"]
    base_bits = np.asarray([item["delivered_bits"] for item in joint["baseline"]["per_uav"]], dtype=np.float64)
    single_a = np.asarray([item["delivered_bits"] for item in joint["single_0_2"]["per_uav"]], dtype=np.float64) - base_bits
    single_b = np.asarray([item["delivered_bits"] for item in joint["single_2_4"]["per_uav"]], dtype=np.float64) - base_bits
    joint_delta = np.asarray([item["delivered_bits"] for item in joint["joint"]["per_uav"]], dtype=np.float64) - base_bits
    interaction = joint_delta - single_a - single_b
    two_swap = {
        "joint_total_delta": float(np.sum(joint_delta)),
        "single_sum_total_delta": float(np.sum(single_a + single_b)),
        "interaction_total": float(np.sum(interaction)),
        "per_uav_interaction": [float(item) for item in interaction],
        "contrasts": {
            "single_0_2": _contrast(joint["baseline"], joint["single_0_2"], (0, 2)),
            "single_2_4": _contrast(joint["baseline"], joint["single_2_4"], (2, 4)),
            "joint": _contrast(joint["baseline"], joint["joint"], (0, 2, 4)),
        },
    }
    attributable = bool(sum_errors) and max(sum_errors) <= BIT_TOLERANCE
    summary = {
        "gate": assign_gate(blocked, attributable, simple, large),
        "new_rollouts": int(new_rollouts),
        "max_new_rollouts": int(MAX_NEW_ROLLOUTS),
        "reused_322051_steps": 1,
        "position_missing_on_reused_case": 1,
        "label_abs_error_max": float(max(label_errors)),
        "attribution_abs_error_max": float(max(sum_errors)),
        "actor_unchanged": int(parameter_token(actor) == token),
        "saved_match": {str(scenario): bucket["matched"] for scenario, bucket in reached.items()},
        "simple_contrasts": int(simple),
        "large_contrasts": int(large),
        "comparisons": comparisons,
        "two_swap": two_swap,
        "primary_seed": int(PRIMARY_SEED),
    }
    stored = {}
    for case in CASES:
        stored[case["name"]] = {
            "scenario_id": int(case["scenario_id"]),
            "reuse": case["reuse"],
            "arms": {
                name: {
                    "assignment": list(assignment),
                    "delivered_bits": float(traces[case["name"]][name]["delivered_bits"]),
                    "label_abs_error": float(traces[case["name"]][name]["label_abs_error"]),
                    "position_stored": int(traces[case["name"]][name]["position_stored"]),
                    "switches": traces[case["name"]][name]["switches"],
                    "per_uav": traces[case["name"]][name]["per_uav"],
                    "steps": traces[case["name"]][name]["steps"],
                }
                for name, assignment in case["arms"].items()
            },
        }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (("summary", summary), ("cases", stored)):
        path = AUDIT_DIR / f"{name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"gate": summary["gate"], "new_rollouts": new_rollouts, "label_abs_error_max": summary["label_abs_error_max"], "simple": simple, "large": large}), flush=True)
    return summary


if __name__ == "__main__":
    run_audit()
