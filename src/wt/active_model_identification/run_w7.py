"""Compare a held action with one extra lookahead on the predicted state.

archive trajectory: the headwind replay leaves the W4 delivery or the three recorded 60-second gaps.

future isolated: a forecast step uses the scenario's true energy coefficient or updates the saved belief.

copy unchanged: forecasting moves the stored trajectory or its beliefs.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from wt.active_model_identification import world as world_mod
from wt.active_model_identification.completion import clear_completion_cache, clear_return_tail_cache
from wt.active_model_identification.policies import select_lookahead
from wt.active_model_identification.run_w1 import _label, _parse
from wt.active_model_identification.run_w4 import _condition, assess, load_w4
from wt.active_model_identification.run_w5 import ARCHIVED_BITS, ARCHIVED_ENERGY_J, _extrapolate, _replay
from wt.active_model_identification.segment import clear_rollout_cache
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import mean_wind_fn

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w7.json"
W1_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w1.json"
RUN_ID = "w7_20261118"


def _parent() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_added(scenarios: list[dict], archive_id: str) -> list[dict]:
    """First tight and first window after the archive id is removed and names are sorted."""
    remaining = sorted((row for row in scenarios if row["id"] != archive_id), key=lambda row: str(row["id"]))
    tight = next(row for row in remaining if row["condition"] == "tight")
    window = next(row for row in remaining if row["condition"] == "window")
    return [tight, window]


def _load_protocol() -> tuple[dict, str]:
    raw = CONFIG_PATH.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("status") != "registered_before_run":
        raise SystemExit("registration moved")
    return protocol, hashlib.sha256(raw).hexdigest()


def _case_for(scenario_id: str) -> tuple[dict, dict]:
    _cfg, archive = load_w4()
    manifest = json.loads((ROOT / "results" / "active_model_identification_m13" / "m13_20261114" / "scenario_manifest.json").read_text(encoding="utf-8"))
    scenario = next(row for row in manifest["scenarios"] if row["id"] == scenario_id)
    case = {
        "cfg": archive["cfg"],
        "spec": archive["spec"],
        "scenario": scenario,
        "factor": archive["factor"],
    }
    return case, manifest


def _wind(case: dict, prior_std: float):
    """Structural wind only. The registered mean ax, ay is not an argument."""
    return mean_wind_fn(None, case["cfg"]["field"], prior_std)


def _advance_recorded(world, actions: list[tuple], beliefs: list, spec: dict, factor: float, wind_at, interval_index: int) -> tuple[list[dict], list[float]]:
    frozen = [item.copy() for item in beliefs]
    theta = [float(item.mean()) for item in frozen]
    watched = {id(uav) for uav in world.uavs}
    trace: list[dict] = []
    original = world_mod._move

    def wrapped(uav, action, spec, theta_u, phase_s, wind_xy=(0.0, 0.0)):
        original(uav, action, spec, theta_u, phase_s, wind_xy)
        if id(uav) not in watched:
            return
        trace.append(
            {
                "uav": int(uav.index),
                "action": (str(action[0]), None if action[1] is None else int(action[1])),
                "phase": int(phase_s),
                "x": float(uav.x),
                "y": float(uav.y),
                "theta": float(theta_u),
            }
        )

    world_mod._move = wrapped
    try:
        world_mod.advance(
            world,
            list(actions),
            theta,
            frozen,
            spec,
            factor,
            None,
            planning=False,
            interval_index=interval_index,
            noise_j=lambda _uav, _interval: 0.0,
            wind_at=wind_at,
        )
    finally:
        world_mod._move = original
    return trace, theta


def forecast_hold(uavs: list, actions: list[tuple], beliefs: list, spec: dict, wind_at, t0: float, seconds: int = 60) -> tuple[dict, list[float]]:
    """Hold the committed action. This is the W5.1 extrapolation with the current belief and wind."""
    clones = [copy.deepcopy(uav) for uav in uavs]
    theta = [float(item.mean()) for item in beliefs]
    points = {int(uav.index): {} for uav in clones}
    for second in range(seconds):
        for uav, action, scale in zip(clones, actions, theta):
            wind = wind_at(uav.x, uav.y, t0 + second)
            world_mod._move(uav, action, spec, scale, second % 30, wind)
            points[int(uav.index)][second + 1] = (float(uav.x), float(uav.y))
    return points, theta


def forecast_replan(world, actions: list[tuple], beliefs: list, spec: dict, factor: float, method: str, wind_at, epoch: int, *, second_interval: bool) -> dict:
    """Execute the committed action for 30 seconds, then one existing high-level decision."""
    copy_world = world.snapshot()
    frozen = [item.copy() for item in beliefs]
    first, theta = _advance_recorded(copy_world, actions, frozen, spec, factor, wind_at, epoch)
    if not second_interval:
        return {"first": first, "second": [], "requested": [], "theta": theta}
    requested, _tables = select_lookahead(method, copy_world, [item.copy() for item in frozen], spec, factor, None)
    second, theta_again = _advance_recorded(copy_world, requested, frozen, spec, factor, wind_at, epoch + 1)
    if theta_again != theta:
        raise RuntimeError("belief mean changed inside the prediction window")
    return {"first": first, "second": second, "requested": list(requested), "theta": theta}


def _by_uav(trace: list[dict], count: int) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in trace:
        grouped.setdefault(int(row["uav"]), []).append(row)
    for uav, rows in grouped.items():
        if len(rows) != count:
            raise RuntimeError(f"uav {uav} recorded {len(rows)} seconds, expected {count}")
        if int(rows[0]["phase"]) != 0:
            raise RuntimeError("guarded action was not recorded at the first second")
    return grouped


def _points(rows: list[dict]) -> dict[int, tuple[float, float]]:
    return {index + 1: (float(row["x"]), float(row["y"])) for index, row in enumerate(rows)}


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _phase(airborne_now: bool, seen_air: bool) -> str:
    if airborne_now:
        return "airborne"
    if seen_air:
        return "landed"
    return "pre_takeoff"


def _fly(case: dict, condition: dict) -> tuple[dict, dict, dict]:
    clear_rollout_cache()
    clear_completion_cache()
    clear_return_tail_cache()
    played, poses, committed = _replay(case, condition)
    for uav in played["states"][0]["world"].uavs:
        poses.setdefault((0, int(uav.index)), (float(uav.x), float(uav.y)))
    return played, poses, committed


def _marker(played: dict) -> list[tuple]:
    rows = []
    for state in played["states"]:
        t_now = int(round(float(state["t_s"])))
        for uav in state["world"].uavs:
            rows.append((t_now, int(uav.index), float(uav.x), float(uav.y), float(uav.energy_j), bool(uav.airborne)))
        for belief in state["beliefs"]:
            mean, variance = belief.moments()
            rows.append((t_now, "belief", float(mean), float(variance)))
    return rows


def _archive_gaps(case: dict, played: dict, poses: dict, committed: dict, protocol: dict) -> list[dict]:
    spec = case["spec"]
    mission = float(spec["time"]["mission_s"])
    receiver = int(protocol["receiver_uav"])
    outcome = assess(played["world"], spec, set(), 3, 0.0)
    if abs(float(outcome["on_time_bits"]) - ARCHIVED_BITS) >= 1e-6 or abs(float(outcome["energy_j"]) - ARCHIVED_ENERGY_J) >= 1e-4:
        raise RuntimeError(f"frozen trajectory drifted bits={outcome['on_time_bits']} energy={outcome['energy_j']}")
    gaps = []
    large = []
    max_30 = 0.0
    for state in played["states"]:
        t_now = int(round(float(state["t_s"])))
        if t_now + 60 > mission + 1e-9:
            continue
        predicted = _extrapolate(state["world"].uavs[receiver], committed[(t_now, receiver)], spec, [30, 60])
        error_30 = _distance(predicted[30], poses[(t_now + 30, receiver)])
        error_60 = _distance(predicted[60], poses[(t_now + 60, receiver)])
        max_30 = max(max_30, error_30)
        if error_60 > 1.0:
            large.append(t_now)
            gaps.append({"t_s": t_now, "error_60_m": error_60})
    expected_t = [int(item) for item in protocol["known_position_mismatch_t_s"]]
    expected_m = [float(item) for item in protocol["known_position_mismatch_m"]]
    tolerance = float(protocol["known_tolerance_m"])
    if large != expected_t or max_30 > 1e-6:
        raise RuntimeError(f"recorded 60-second gaps changed times={large} max30={max_30}")
    for found, target in zip(gaps, expected_m):
        if abs(float(found["error_60_m"]) - target) > tolerance:
            raise RuntimeError(f"recorded gap at {found['t_s']} is {found['error_60_m']}")
    return gaps


def _score_block(rows: list[dict], prefix: str) -> dict:
    return {
        "n": len(rows),
        "mean_30_m": _mean([float(row[f"{prefix}_err_30_m"]) for row in rows]),
        "max_30_m": _max([float(row[f"{prefix}_err_30_m"]) for row in rows]),
        "mean_60_m": _mean([float(row[f"{prefix}_err_60_m"]) for row in rows]),
        "max_60_m": _max([float(row[f"{prefix}_err_60_m"]) for row in rows]),
        "mean_second_m": _mean([float(row[f"{prefix}_mean_second_m"]) for row in rows]),
        "action_accuracy": _mean([float(row[f"{prefix}_action_hit"]) for row in rows]),
    }


def _blank_score() -> dict:
    return {
        "n": 0,
        "mean_30_m": None,
        "max_30_m": None,
        "mean_60_m": None,
        "max_60_m": None,
        "mean_second_m": None,
        "action_accuracy": None,
    }


def _timing(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "median_a_s": None, "max_a_s": None, "median_b_s": None, "max_b_s": None, "median_b_over_a": None}
    per_t = {}
    for row in rows:
        per_t[float(row["t_s"])] = (float(row["forecast_a_s"]), float(row["forecast_b_s"]))
    a_times = [item[0] for item in per_t.values()]
    b_times = [item[1] for item in per_t.values()]
    median_a = float(statistics.median(a_times))
    return {
        "n": len(per_t),
        "median_a_s": median_a,
        "max_a_s": max(a_times),
        "median_b_s": float(statistics.median(b_times)),
        "max_b_s": max(b_times),
        "median_b_over_a": None if median_a <= 0.0 else float(statistics.median(b_times)) / median_a,
    }


def _classify(row: dict, truth_requested: tuple, frozen_requested: tuple) -> str:
    if int(row["b_action_hit"]) == 1:
        return "match"
    if row["b_requested"] == row["actual_requested_next"] and row["b_executed"] != row["actual_executed_next"]:
        return "guard"
    if int(row["first_action_match"]) != 1 or float(row["b_err_30_m"]) > 1e-3:
        return "guard"
    if truth_requested == _parse(row["actual_requested_next"]) and frozen_requested != truth_requested:
        return "future_belief"
    if frozen_requested == _parse(row["actual_requested_next"]):
        return "business_queue"
    return "action_selection"


def _forecast_scenario(case: dict, condition: dict, protocol: dict, wind_at) -> dict:
    spec = case["spec"]
    method = str(protocol["method"])
    mission = float(spec["time"]["mission_s"])
    step = int(protocol["decision_s"])
    scenario_id = str(case["scenario"]["id"])
    started = time.perf_counter()
    played, poses, committed = _fly(case, condition)
    replay_s = time.perf_counter() - started
    requested = {}
    for row in played["rows"]:
        requested[(int(row["epoch"]) * step, int(row["uav"]))] = _parse(row["requested"])
    before = _marker(played)
    gaps = _archive_gaps(case, played, poses, committed, protocol) if scenario_id == protocol["archive_scenario_id"] else []
    outcome = assess(played["world"], spec, set(), 3, 0.0)
    pairs = []
    tracks = []
    seen_air = [False, False]
    theta_gap = 0.0
    for state in played["states"]:
        t_now = int(round(float(state["t_s"])))
        if t_now + step > mission + 1e-9:
            continue
        full = t_now + 2 * step <= mission + 1e-9
        uavs = list(state["world"].uavs)
        beliefs = list(state["beliefs"])
        held = [committed[(t_now, index)] for index in range(len(uavs))]
        clock = time.perf_counter()
        hold_points, hold_theta = forecast_hold(uavs, held, beliefs, spec, wind_at, float(t_now), 60 if full else step)
        hold_s = time.perf_counter() - clock
        clock = time.perf_counter()
        replanned = forecast_replan(
            state["world"],
            held,
            beliefs,
            spec,
            float(case["factor"]),
            method,
            wind_at,
            int(round(t_now / step)),
            second_interval=full,
        )
        replan_s = time.perf_counter() - clock
        for index, scale in enumerate(hold_theta):
            theta_gap = max(theta_gap, abs(scale - float(beliefs[index].mean())))
        for index, scale in enumerate(replanned["theta"]):
            theta_gap = max(theta_gap, abs(scale - float(beliefs[index].mean())))
        first = _by_uav(replanned["first"], step)
        second = _by_uav(replanned["second"], step) if full else {}
        next_t = t_now + step
        for index, uav in enumerate(uavs):
            airborne = bool(uav.airborne)
            phase = _phase(airborne, seen_air[index])
            if airborne:
                seen_air[index] = True
            actual_30 = poses[(next_t, index)]
            hold_30 = hold_points[index][step]
            b_rows = first[index]
            b_30 = (float(b_rows[-1]["x"]), float(b_rows[-1]["y"]))
            actual_next = committed[(next_t, index)] if full else None
            actual_requested_next = requested[(next_t, index)] if full else None
            b_executed = second[index][0]["action"] if full else None
            b_requested = replanned["requested"][index] if full else None
            row = {
                "scenario_id": scenario_id,
                "t_s": t_now,
                "uav": index,
                "phase": phase,
                "full_60": int(full),
                "action_t": _label(held[index]),
                "actual_executed_next": _label(actual_next) if full else "",
                "actual_requested_next": _label(actual_requested_next) if full else "",
                "switched": int(_label(actual_next) != _label(held[index])) if full else "",
                "a_err_30_m": _distance(hold_30, actual_30),
                "b_err_30_m": _distance(b_30, actual_30),
                "a_action_hit": int(_label(held[index]) == _label(actual_next)) if full else "",
                "b_requested": _label(b_requested) if full else "",
                "b_executed": _label(b_executed) if full else "",
                "b_action_hit": int(_label(b_executed) == _label(actual_next)) if full else "",
                "first_action_match": int(_label(b_rows[0]["action"]) == _label(held[index])),
                "forecast_a_s": hold_s,
                "forecast_b_s": replan_s,
                "ring": "",
            }
            if full:
                actual_60 = poses[(t_now + 2 * step, index)]
                b_points = _points(b_rows + second[index])
                a_seconds = [_distance(hold_points[index][second_i], poses[(t_now + second_i, index)]) for second_i in range(1, 61)]
                b_seconds = [_distance(b_points[second_i], poses[(t_now + second_i, index)]) for second_i in range(1, 61)]
                row.update(
                    {
                        "a_err_60_m": _distance(hold_points[index][60], actual_60),
                        "b_err_60_m": _distance(b_points[60], actual_60),
                        "a_mean_second_m": _mean(a_seconds),
                        "b_mean_second_m": _mean(b_seconds),
                    }
                )
                for second_i in range(1, 61):
                    true_xy = poses[(t_now + second_i, index)]
                    tracks.append(
                        {
                            "scenario_id": scenario_id,
                            "t_s": t_now,
                            "uav": index,
                            "second": second_i,
                            "x_true": true_xy[0],
                            "y_true": true_xy[1],
                            "x_a": hold_points[index][second_i][0],
                            "y_a": hold_points[index][second_i][1],
                            "x_b": b_points[second_i][0],
                            "y_b": b_points[second_i][1],
                        }
                    )
            else:
                row.update({"a_err_60_m": "", "b_err_60_m": "", "a_mean_second_m": "", "b_mean_second_m": ""})
            pairs.append(row)
        if int(t_now) % 300 == 0:
            print(f"{scenario_id} t={t_now} replay={replay_s:.1f}s a={hold_s:.3f}s b={replan_s:.3f}s", flush=True)
    for row in pairs:
        if int(row["full_60"]) != 1 or row["phase"] != "airborne" or int(row["b_action_hit"]) == 1:
            row["ring"] = "match" if int(row["full_60"]) == 1 and int(row["b_action_hit"]) == 1 else ""
            continue
        if row["b_requested"] != row["actual_requested_next"] and int(row["first_action_match"]) == 1 and float(row["b_err_30_m"]) <= 1e-3:
            state = next(item for item in played["states"] if int(round(float(item["t_s"]))) == int(row["t_s"]) + step)
            frozen_beliefs = next(item["beliefs"] for item in played["states"] if int(round(float(item["t_s"]))) == int(row["t_s"]))
            truth_requested, _tables = select_lookahead(method, state["world"].snapshot(), [item.copy() for item in state["beliefs"]], spec, float(case["factor"]), None)
            frozen_requested, _tables = select_lookahead(method, state["world"].snapshot(), [item.copy() for item in frozen_beliefs], spec, float(case["factor"]), None)
            row["ring"] = _classify(row, truth_requested[int(row["uav"])], frozen_requested[int(row["uav"])])
        else:
            row["ring"] = _classify(row, _parse(row["actual_requested_next"]), _parse(row["b_requested"]))
    after = _marker(played)
    if before != after:
        raise RuntimeError("stored trajectory changed")
    if theta_gap > 1e-12:
        raise RuntimeError("forecast theta left the belief mean")
    actions = []
    for t_now in range(0, int(mission), step):
        actions.append(tuple(_label(committed[(t_now, index)]) for index in range(2)))
    positions = []
    for t_now in range(0, int(mission) + 1):
        if (t_now, 0) not in poses:
            continue
        positions.append((poses[(t_now, 0)], poses[(t_now, 1)]))
    return {
        "scenario_id": scenario_id,
        "seed": int(case["scenario"]["seed"]),
        "theta": [float(item) for item in case["scenario"]["theta"]],
        "battery_j": [float(item) for item in case["scenario"]["battery_j"]],
        "replay_s": replay_s,
        "bits": float(outcome["on_time_bits"]),
        "energy_j": float(outcome["energy_j"]),
        "pairs": pairs,
        "tracks": tracks,
        "gaps": gaps,
        "actions": actions,
        "positions": positions,
        "pre_takeoff_t_s": sorted({int(row["t_s"]) for row in pairs if row["phase"] == "pre_takeoff"}),
    }


def _same_track(left: dict, right: dict) -> dict:
    if len(left["positions"]) != len(right["positions"]) or len(left["actions"]) != len(right["actions"]):
        raise RuntimeError("scenario logs do not cover the same times")
    same_actions = left["actions"] == right["actions"]
    gap = 0.0
    for one, two in zip(left["positions"], right["positions"]):
        gap = max(gap, _distance(one[0], two[0]), _distance(one[1], two[1]))
    return {"same_actions": same_actions, "max_position_gap_m": gap, "same_track": same_actions and gap <= 1e-6}


def _group_scores(pairs: list[dict], scenario_id: str) -> list[dict]:
    scored = []
    usable = [row for row in pairs if row["scenario_id"] == scenario_id and int(row["full_60"]) == 1]
    for uav in (0, 1):
        for phase in ("airborne", "pre_takeoff", "landed"):
            base = [row for row in usable if int(row["uav"]) == uav and row["phase"] == phase]
            for kind, chosen in (
                ("all", base),
                ("switch", [row for row in base if int(row["switched"]) == 1]),
                ("sustain", [row for row in base if int(row["switched"]) == 0]),
            ):
                block = {"scenario_id": scenario_id, "uav": uav, "phase": phase, "kind": kind}
                if not chosen:
                    block.update({"a": _blank_score(), "b": _blank_score()})
                else:
                    block.update({"a": _score_block(chosen, "a"), "b": _score_block(chosen, "b")})
                scored.append(block)
    return scored


def _flat_scores(scores: list[dict]) -> list[dict]:
    rows = []
    for block in scores:
        for arm in ("a", "b"):
            item = block[arm]
            rows.append(
                {
                    "scenario_id": block["scenario_id"],
                    "uav": block["uav"],
                    "phase": block["phase"],
                    "kind": block["kind"],
                    "arm": arm,
                    "n": item["n"],
                    "mean_30_m": item["mean_30_m"],
                    "max_30_m": item["max_30_m"],
                    "mean_60_m": item["mean_60_m"],
                    "max_60_m": item["max_60_m"],
                    "mean_second_m": item["mean_second_m"],
                    "action_accuracy": item["action_accuracy"],
                }
            )
    return rows


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, digest = _load_protocol()
    out = out_dir or (ROOT / "results" / "wind_observation_w7" / RUN_ID)
    if "wind_observation_w7" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w7")
    registered = json.loads(W1_CONFIG_PATH.read_text(encoding="utf-8"))
    _cfg, archive = load_w4()
    manifest = json.loads((ROOT / str(protocol["scenario_manifest"])).read_text(encoding="utf-8"))
    added = [str(row["id"]) for row in select_added(manifest["scenarios"], str(protocol["archive_scenario_id"]))]
    if added != list(protocol["added_scenario_ids"]):
        raise RuntimeError(f"scenario rule returned {added}")
    if archive["cfg"]["field"] != registered["field"] or archive["cfg"]["conditions"] != registered["conditions"]:
        raise RuntimeError("wind registration changed")
    condition = _condition(archive, str(protocol["condition"]))
    wind_at = _wind(archive, float(protocol["prior_std_m_s"]))
    scenarios = []
    pairs = []
    tracks = []
    for scenario_id in [str(protocol["archive_scenario_id"]), *added]:
        case, _manifest = _case_for(scenario_id)
        if str(case["cfg"]["method"]) != str(protocol["method"]):
            raise RuntimeError("scheduler method changed")
        print(f"scenario {scenario_id}", flush=True)
        flown = _forecast_scenario(case, condition, protocol, wind_at)
        scenarios.append(flown)
        pairs.extend(flown["pairs"])
        tracks.extend(flown["tracks"])
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "pairs.csv", pairs)
    _write_csv(out / "tracks.csv", tracks)
    scores = []
    timings = []
    for flown in scenarios:
        scores.extend(_group_scores(flown["pairs"], flown["scenario_id"]))
        usable = [row for row in flown["pairs"] if int(row["full_60"]) == 1]
        timings.append({"scenario_id": flown["scenario_id"], **_timing(usable)})
    flat = _flat_scores(scores)
    _write_csv(out / "scores.csv", flat)
    rings: dict[str, int] = {}
    for row in pairs:
        if row["phase"] == "airborne" and int(row["full_60"]) == 1 and row["ring"] not in ("", "match"):
            rings[row["ring"]] = rings.get(row["ring"], 0) + 1
    summary = {
        "schema_version": 1,
        "parent_commit": _parent(),
        "config_sha256": digest,
        "config_status": protocol["status"],
        "archive_match": True,
        "future_isolated": True,
        "copy_unchanged": True,
        "default_changed": False,
        "added_scenario_ids": added,
        "same_track": _same_track(scenarios[1], scenarios[2]),
        "known_gaps": scenarios[0]["gaps"],
        "pre_takeoff_t_s": {item["scenario_id"]: item["pre_takeoff_t_s"] for item in scenarios},
        "scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "seed": item["seed"],
                "theta": item["theta"],
                "battery_j": item["battery_j"],
                "bits": item["bits"],
                "energy_j": item["energy_j"],
                "replay_s": item["replay_s"],
            }
            for item in scenarios
        ],
        "timing": timings,
        "mismatch_rings": rings,
        "scores": flat,
        "elapsed_s": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    airborne = [row for row in flat if row["phase"] == "airborne" and row["kind"] == "switch" and row["arm"] == "b"]
    print(
        "switch "
        + " ".join(f"{row['scenario_id']}/u{row['uav']} n={row['n']} mean60={row['mean_60_m']}" for row in airborne),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
