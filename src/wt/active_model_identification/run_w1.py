"""Run the pre-registered two-UAV wind-observation pilot.

Zero wind is checked before any wind arm. A mismatch stops the run.
Wind amplitudes are read from the registered config and are not revised.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.belief import prior_belief
from wt.active_model_identification.completion import clear_completion_cache, clear_return_tail_cache
from wt.active_model_identification.physics import CRUISE_M_S
from wt.active_model_identification.policies import candidates
from wt.active_model_identification.run_m13 import _play, _stats
from wt.active_model_identification.scenarios import observation_noise
from wt.active_model_identification.segment import clear_rollout_cache
from wt.active_model_identification.spec import ROOT, decision_count
from wt.active_model_identification.wind import (
    adverse_wind_fn,
    charge_message,
    fly_script,
    make_return_guard,
    mean_wind_fn,
    message_cost_j,
    observe,
    predict,
    true_wind,
)
from wt.active_model_identification.world import (
    advance,
    energy_used_j,
    make_world,
    mission_terminal,
    on_time_bits,
    utility,
)

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w1.json"
M16_DIR = ROOT / "results" / "active_model_identification_m16" / "m16_20261114"
M13_DIR = ROOT / "results" / "active_model_identification_m13" / "m13_20261114"
RUN_ID = "w1_20261114"
ENERGY_TOL = 1e-6


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse(label: str):
    kind, target = str(label).split(":")
    return (kind, None if target == "None" else int(target))


def _label(action) -> str:
    return f"{action[0]}:{action[1]}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_case() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads((M16_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((M13_DIR / "scenario_manifest.json").read_text(encoding="utf-8"))
    if spec["terminal"]["mode"] != cfg["terminal_mode"] or spec["selection"]["mode"] != cfg["selection_mode"]:
        raise SystemExit("frozen scheduler modes do not match the W1 registration")
    if int(spec["radio"]["command_bytes"]) != int(cfg["payload_bytes"]):
        raise SystemExit("message size does not match the command airtime")
    if float(spec["energy"]["reserve_j"]) != float(cfg["reserve_j"]):
        raise SystemExit("reserve does not match the frozen spec")
    scenario = next(row for row in manifest["scenarios"] if row["id"] == cfg["scenario_id"])
    if int(scenario["seed"]) != int(cfg["seed"]):
        raise SystemExit("scenario seed does not match the registration")
    if list(scenario["wind_xy_m_s"]) != [0.0, 0.0]:
        raise SystemExit("the frozen scenario is not the zero-wind case")
    rows = [
        row
        for row in _read_csv(M16_DIR / "decisions.csv")
        if row.get("branch") == "core" and row["scenario_id"] == cfg["scenario_id"] and row["method"] == cfg["method"]
    ]
    requests = {(int(row["epoch"]), int(row["uav"])): row for row in rows}
    episode = next(
        row
        for row in _read_csv(M16_DIR / "episodes.csv")
        if row.get("branch") == "core" and row["scenario_id"] == cfg["scenario_id"] and row["method"] == cfg["method"]
    )
    return {
        "cfg": cfg,
        "spec": spec,
        "scenario": scenario,
        "factor": float(manifest["budget"]["integration_factor"]),
        "requests": requests,
        "episode": episode,
    }


def _mismatch(found: list[str], key: str, actual, expected, tol: float | None = None) -> None:
    if tol is None:
        if str(actual) != str(expected):
            found.append(f"{key} {actual} != {expected}")
        return
    if abs(float(actual) - float(expected)) > tol:
        found.append(f"{key} {actual} != {expected}")


def replay_case() -> dict:
    """Replay the frozen requests at zero wind and keep the 180 s decision state."""
    case = load_case()
    cfg = case["cfg"]
    spec = case["spec"]
    scenario = case["scenario"]
    requests = case["requests"]
    limit = decision_count(spec)
    expected = {(epoch, uav) for epoch in range(limit) for uav in (0, 1)}
    if set(requests) != expected:
        raise SystemExit(f"frozen decision log has {len(requests)} rows, expected {len(expected)}")
    world = make_world(spec, scenario["battery_j"], f"{scenario['id']}-{cfg['method']}")
    beliefs = [prior_belief(spec), prior_belief(spec)]
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    mismatches: list[str] = []
    sample = None
    prefix = None
    prefix_beliefs = None
    for epoch in range(limit):
        actions = [_parse(requests[(epoch, uav)]["requested"]) for uav in (0, 1)]
        observations = advance(
            world,
            actions,
            list(scenario["theta"]),
            beliefs,
            spec,
            case["factor"],
            None,
            planning=False,
            interval_index=epoch,
            noise_j=noise,
        )
        for obs in observations:
            saved = requests[(epoch, int(obs["uav"]))]
            executed = f"{obs['executed'][0]}:{obs['executed'][1]}"
            _mismatch(mismatches, f"e{epoch}u{obs['uav']} executed", executed, saved["executed"])
            _mismatch(mismatches, f"e{epoch}u{obs['uav']} requested", f"{obs['requested'][0]}:{obs['requested'][1]}", saved["requested"])
            _mismatch(mismatches, f"e{epoch}u{obs['uav']} energy", obs["energy_j"], saved["energy_j"], ENERGY_TOL)
            _mismatch(mismatches, f"e{epoch}u{obs['uav']} x", obs["x"], saved["x"], ENERGY_TOL)
            _mismatch(mismatches, f"e{epoch}u{obs['uav']} y", obs["y"], saved["y"], ENERGY_TOL)
            _mismatch(mismatches, f"e{epoch}u{obs['uav']} airborne", obs["airborne"], saved["airborne"])
        if epoch == int(cfg["sample_epoch"]):
            uav = world.uavs[int(cfg["observer_uav"])]
            sample = {"x_m": uav.x, "y_m": uav.y, "t_s": world.t, "energy_j": uav.energy_j}
            _mismatch(mismatches, "sample_time", world.t, cfg["sample_time_s"], ENERGY_TOL)
        if epoch + 1 == int(cfg["decision_epoch"]):
            prefix = world.snapshot()
            prefix_beliefs = [item.copy() for item in beliefs]
            _mismatch(mismatches, "decision_time", world.t, cfg["decision_time_s"], ENERGY_TOL)
    episode = case["episode"]
    terminals = mission_terminal(world, spec)
    _mismatch(mismatches, "bits", on_time_bits(world), episode["on_time_bits"], ENERGY_TOL)
    _mismatch(mismatches, "energy", sum(energy_used_j(world)), episode["energy_j"], ENERGY_TOL)
    _mismatch(mismatches, "utility", utility(world, spec), episode["utility"], ENERGY_TOL)
    _mismatch(mismatches, "returned", sum(int(item["returned"]) for item in terminals), episode["returned"])
    _mismatch(mismatches, "not_returned", sum(int(item["not_returned"]) for item in terminals), episode["not_returned"])
    _mismatch(mismatches, "not_landed", sum(int(item["not_landed"]) for item in terminals), episode["not_landed"])
    _mismatch(mismatches, "below_reserve", sum(int(item["below_reserve"]) for item in terminals), episode["below_reserve"])
    _mismatch(mismatches, "energy_exhausted", sum(int(item["energy_exhausted"]) for item in terminals), episode["energy_exhausted"])
    return {
        "ok": not mismatches,
        "mismatches": mismatches[:12],
        "mismatch_count": len(mismatches),
        "bits": on_time_bits(world),
        "energy_j": sum(energy_used_j(world)),
        "utility": utility(world, spec),
        "returned": sum(int(item["returned"]) for item in terminals),
        "sample": sample,
        "prefix": prefix,
        "prefix_beliefs": prefix_beliefs,
        "case": case,
    }


def fresh_play(case: dict) -> dict:
    """Ask the M1.6 scheduler again. This catches a planner change the request replay cannot see."""
    clear_rollout_cache()
    clear_completion_cache()
    clear_return_tail_cache()
    spec = case["spec"]
    scenario = case["scenario"]
    cfg = case["cfg"]
    world = make_world(spec, scenario["battery_j"], f"{scenario['id']}-{cfg['method']}-fresh")
    beliefs = [prior_belief(spec), prior_belief(spec)]
    episode, decisions, _deliveries, _snapshot = _play(
        cfg["method"], scenario, spec, case["factor"], _stats(), "core", 0, world, beliefs, set()
    )
    mismatches: list[str] = []
    saved = case["episode"]
    _mismatch(mismatches, "fresh bits", episode["on_time_bits"], saved["on_time_bits"], ENERGY_TOL)
    _mismatch(mismatches, "fresh energy", episode["energy_j"], saved["energy_j"], ENERGY_TOL)
    _mismatch(mismatches, "fresh utility", episode["utility"], saved["utility"], ENERGY_TOL)
    _mismatch(mismatches, "fresh returned", episode["returned"], saved["returned"])
    _mismatch(mismatches, "fresh not_returned", episode["not_returned"], saved["not_returned"])
    _mismatch(mismatches, "fresh not_landed", episode["not_landed"], saved["not_landed"])
    _mismatch(mismatches, "fresh below_reserve", episode["below_reserve"], saved["below_reserve"])
    _mismatch(mismatches, "fresh exhausted", episode["energy_exhausted"], saved["energy_exhausted"])
    for row in decisions:
        key = (int(row["epoch"]), int(row["uav"]))
        reference = case["requests"][key]
        _mismatch(mismatches, f"fresh e{key[0]}u{key[1]} requested", row["requested"], reference["requested"])
        _mismatch(mismatches, f"fresh e{key[0]}u{key[1]} executed", row["executed"], reference["executed"])
    return {
        "ok": not mismatches,
        "mismatches": mismatches[:12],
        "mismatch_count": len(mismatches),
        "bits": episode["on_time_bits"],
        "energy_j": episode["energy_j"],
        "utility": episode["utility"],
        "returned": episode["returned"],
    }


def _feasible(spec: dict, world, decider: int) -> list:
    holding = any(packet.holder == f"uav:{decider}" for packet in world.packets)
    found = []
    for action in candidates(spec, world, decider, include_calibrate=False):
        if action[0] == "backhaul" and not holding:
            continue
        if action[0] == "service":
            pending = any(packet.buoy == int(action[1]) and packet.holder == "buoy" for packet in world.packets)
            if not pending:
                continue
        found.append(action)
    return found


def _incremental(bits: float, energy: float, spec: dict) -> float:
    block = spec["utility"]
    return bits / float(block["b_ref_bits"]) - float(block["lambda_e"]) * energy / float(block["e_ref_j"])


def _choose(scored: list[dict], frozen: str, tau: float) -> dict:
    admitted = [row for row in scored if row["admitted"]]
    if not admitted:
        fallback = next(row for row in scored if row["action"] == frozen)
        fallback["selected_by"] = "fallback_request"
        return fallback
    best = max(row["utility"] for row in admitted)
    tied = [row for row in admitted if abs(row["utility"] - best) <= tau]
    for row in tied:
        if row["action"] == frozen:
            row["selected_by"] = "tie_frozen_request"
            return row
    tied[0]["selected_by"] = "max_utility"
    return tied[0]


def _wind_error(message, points, field, condition, prior: float) -> dict:
    sse = 0.0
    count = 0
    rows = []
    for name, x, y, t in points:
        found = predict(message, x, y, t, field, prior)
        truth = true_wind(condition, x, y, t, field)
        dx = found["wind_xy"][0] - truth[0]
        dy = found["wind_xy"][1] - truth[1]
        sse += dx * dx + dy * dy
        count += 2
        rows.append(
            {
                "point": name,
                "t_s": t,
                "predicted_m_s": [found["wind_xy"][0], found["wind_xy"][1]],
                "truth_m_s": [truth[0], truth[1]],
                "error_m_s": [dx, dy],
                "std_m_s": found["std_m_s"],
                "used_message": found["used_message"],
            }
        )
    return {"rmse_m_s": (sse / count) ** 0.5, "points": rows}


def _arm(replay: dict, condition: dict, mode: str, message: dict, future: dict, actual_cache: dict) -> tuple[dict, list[dict]]:
    case = replay["case"]
    cfg = case["cfg"]
    spec = case["spec"]
    scenario = case["scenario"]
    factor = case["factor"]
    world = replay["prefix"]
    beliefs = replay["prefix_beliefs"]
    decider = int(cfg["decider_uav"])
    observer = int(cfg["observer_uav"])
    field = cfg["field"]
    prior = float(cfg["prior_std_m_s"])
    info = message if mode == "timely" else None
    noise = observation_noise(int(scenario["seed"]), float(spec["belief"]["sigma_j"]))
    actions = _feasible(spec, world, decider)
    frozen = case["requests"][(int(cfg["decision_epoch"]), decider)]["requested"]
    if _label(("return", None)) not in {_label(action) for action in actions}:
        raise SystemExit("return is missing from the registered candidate set")
    origin = (world.uavs[decider].x, world.uavs[decider].y)
    home = (world.uavs[decider].home_x, world.uavs[decider].home_y)
    mean_fn = mean_wind_fn(info, field, prior)
    harsh_fn, std = adverse_wind_fn(info, field, prior, origin, home)
    theta_mean = beliefs[decider].mean()
    theta_q = beliefs[decider].quantile(float(spec["belief"]["guard_probability"]))
    battery = world.uavs[decider].energy_j
    reserve = float(spec["energy"]["reserve_j"])
    true_thetas = list(scenario["theta"])
    scored = []
    candidate_rows = []
    info_key = "message" if info is not None else "prior"
    for action in actions:
        thetas = list(true_thetas)
        thetas[decider] = theta_mean
        quote = fly_script(
            world, beliefs, spec, factor, decider, action, thetas, mean_fn, noise, sentinel_battery=True
        )
        thetas_q = list(true_thetas)
        thetas_q[decider] = theta_q
        admission = fly_script(
            world, beliefs, spec, factor, decider, action, thetas_q, harsh_fn, noise, sentinel_battery=True
        )
        admitted = action[0] == "return" or (
            admission["landed"]
            and admission["on_time"]
            and not admission["failed"]
            and battery + ENERGY_TOL >= admission["focus_energy_j"] + reserve
            and (action[0] != "service" or admission["on_time_bits"] > 0.0)
        )
        value = _incremental(quote["on_time_bits"], quote["focus_energy_j"], spec)
        label = _label(action)
        pred_key = (info_key, label)
        if pred_key not in actual_cache["predicted"]:
            actual_cache["predicted"][pred_key] = fly_script(
                world, beliefs, spec, factor, decider, action, true_thetas, mean_fn, noise, sentinel_battery=True
            )
        if label not in actual_cache["actual"]:
            actual_cache["actual"][label] = fly_script(
                world,
                beliefs,
                spec,
                factor,
                decider,
                action,
                true_thetas,
                lambda x, y, t, condition=condition: true_wind(condition, x, y, t, field),
                noise,
                sentinel_battery=True,
            )
        predicted = actual_cache["predicted"][pred_key]
        actual = actual_cache["actual"][label]
        row = {
            "condition": condition["id"],
            "mode": mode,
            "action": label,
            "admitted": admitted,
            "utility": value,
            "score_bits": quote["on_time_bits"],
            "score_energy_j": quote["focus_energy_j"],
            "admit_energy_j": admission["focus_energy_j"],
            "admit_landed": admission["landed"],
            "admit_bits": admission["on_time_bits"],
            "posthoc_predicted_j": predicted["focus_energy_j"],
            "posthoc_actual_j": actual["focus_energy_j"],
            "posthoc_error_j": predicted["focus_energy_j"] - actual["focus_energy_j"],
            "chosen": False,
        }
        scored.append({"action": label, "admitted": admitted, "utility": value, "row": row})
        candidate_rows.append(row)
    chosen = _choose(scored, frozen, float(spec["integral"]["tie_abs"]))
    for row in candidate_rows:
        row["chosen"] = row["action"] == chosen["action"]
    distance = ((origin[0] - home[0]) ** 2 + (origin[1] - home[1]) ** 2) ** 0.5
    points = [
        ("sample", replay["sample"]["x_m"], replay["sample"]["y_m"], float(cfg["sample_time_s"])),
        ("decider", origin[0], origin[1], float(cfg["decision_time_s"])),
        ("home_eta", home[0], home[1], float(cfg["decision_time_s"]) + distance / CRUISE_M_S),
    ]
    error = _wind_error(info, points, field, condition, prior)
    message_view = predict(message, origin[0], origin[1], float(cfg["decision_time_s"]), field, prior)
    decision_view = predict(info, origin[0], origin[1], float(cfg["decision_time_s"]), field, prior)
    team = world.snapshot()
    team_beliefs = [item.copy() for item in beliefs]
    radio = message_cost_j(int(cfg["payload_bytes"]))
    charged = {"done": False, "cost": None}
    if mode == "timely":
        charged["cost"] = charge_message(team, observer, decider, int(cfg["payload_bytes"]))
    elif mode == "no_send":
        radio = {"payload_bytes": 0, "airtime_s": 0.0, "sender_j": 0.0, "receiver_j": 0.0}

    def on_interval(sim, _beliefs, _interval, mode=mode, charged=charged):
        if mode == "late" and (not charged["done"]) and sim.t + ENERGY_TOL >= float(cfg["late_time_s"]):
            charged["cost"] = charge_message(sim, observer, decider, int(cfg["payload_bytes"]))
            charged["done"] = True

    flown = fly_script(
        team,
        team_beliefs,
        spec,
        factor,
        decider,
        _parse(chosen["action"]),
        true_thetas,
        lambda x, y, t, condition=condition: true_wind(condition, x, y, t, field),
        noise,
        other_actions=future,
        return_guard=make_return_guard(info, field, prior, spec, factor, decider),
        on_interval=on_interval,
    )
    sim = flown.pop("sim")
    terminals = mission_terminal(sim, spec)
    if mode == "late" and not charged["done"]:
        raise SystemExit("late message was not charged")
    if mode == "timely" and charged["cost"] is None:
        raise SystemExit("timely message was not charged")
    if mode == "no_send":
        message_j = 0.0
        nominal_j = 0.0
    else:
        drawn = charged["cost"]
        message_j = float(drawn["sender_j_drawn"] + drawn["receiver_j_drawn"])
        nominal_j = float(radio["sender_j"] + radio["receiver_j"])
    arm = {
        "condition": condition["id"],
        "ax_m_s": condition["ax"],
        "ay_m_s": condition["ay"],
        "mode": mode,
        "decision_used_message": info is not None,
        "message_available_after_decision": mode != "no_send",
        "chosen": chosen["action"],
        "selected_by": chosen["selected_by"],
        "first_executed": flown["first_executed"],
        "wind_rmse_m_s": error["rmse_m_s"],
        "wind_std_m_s": std,
        "decision_ax_m_s": decision_view.get("ax_m_s", 0.0),
        "decision_ay_m_s": decision_view.get("ay_m_s", 0.0),
        "decision_ax_error_m_s": decision_view.get("ax_m_s", 0.0) - float(condition["ax"]),
        "message_ax_error_m_s": message_view["ax_m_s"] - float(condition["ax"]),
        "message_ay_error_m_s": message_view["ay_m_s"] - float(condition["ay"]),
        "on_time_bits": flown["total_on_time_bits"],
        "energy_j": flown["world_energy_j"],
        "utility": flown["utility"],
        "message_j": message_j,
        "message_j_nominal": nominal_j,
        "returned": sum(int(item["returned"]) for item in terminals),
        "not_returned": sum(int(item["not_returned"]) for item in terminals),
        "not_landed": sum(int(item["not_landed"]) for item in terminals),
        "below_reserve": sum(int(item["below_reserve"]) for item in terminals),
        "energy_exhausted": sum(int(item["energy_exhausted"]) for item in terminals),
        "interventions": flown["interventions"],
        "uav_energy_j": [item["energy_j"] for item in terminals],
        "prediction": error,
    }
    return arm, candidate_rows


def _delta(left: dict, right: dict) -> dict:
    keys = ("on_time_bits", "utility", "energy_j", "returned", "not_returned", "not_landed", "below_reserve", "energy_exhausted", "wind_rmse_m_s")
    return {key: left[key] - right[key] for key in keys}


def _judge(arms: list[dict]) -> tuple[list[dict], str]:
    reviews = []
    claims = []
    order = []
    for row in arms:
        if row["condition"] not in order:
            order.append(row["condition"])
    for condition in order:
        group = {row["mode"]: row for row in arms if row["condition"] == condition}
        timely = group["timely"]
        late = group["late"]
        silent = group["no_send"]
        pred_better = timely["wind_rmse_m_s"] + 1e-9 < silent["wind_rmse_m_s"]
        review = {
            "condition": condition,
            "chosen": {mode: group[mode]["chosen"] for mode in ("no_send", "timely", "late")},
            "first_executed": {mode: group[mode]["first_executed"] for mode in ("no_send", "timely", "late")},
            "prediction_improved": pred_better,
            "timely_minus_no_send": _delta(timely, silent),
            "timely_minus_late": _delta(timely, late),
        }
        same = len(set(review["chosen"].values())) == 1
        if same:
            review["level"] = "prediction" if pred_better else "no_difference"
            review["reason"] = "三种信息下选择的动作相同"
        elif timely["chosen"] == late["chosen"] or timely["chosen"] == silent["chosen"]:
            review["level"] = "no_mission_claim"
            review["reason"] = "及时报文没有单独改变这个风况下的选择"
        else:
            retention = (
                timely["on_time_bits"] + 1e-6 >= silent["on_time_bits"]
                and timely["utility"] + 1e-9 >= silent["utility"]
                and timely["returned"] >= silent["returned"]
                and timely["below_reserve"] <= silent["below_reserve"]
                and timely["energy_exhausted"] <= silent["energy_exhausted"]
                and timely["not_returned"] <= silent["not_returned"]
                and timely["not_landed"] <= silent["not_landed"]
            )
            improved = (
                timely["on_time_bits"] > silent["on_time_bits"] + 0.5
                or timely["utility"] > silent["utility"] + 1e-4
                or timely["returned"] > silent["returned"]
                or timely["below_reserve"] < silent["below_reserve"]
            )
            review["retention"] = retention
            review["improved"] = improved
            if retention and improved:
                review["level"] = "mission"
                review["reason"] = f"及时报文使 {condition} 的选择从 {silent['chosen']} 变为 {timely['chosen']}"
                claims.append(review)
            else:
                review["level"] = "prediction" if pred_better else "no_benefit"
                review["reason"] = "动作有变化，但按时交付、净效用、返航和备用电量没有同时保持并改善"
        reviews.append(review)
    if claims:
        conclusion = "。".join(item["reason"] for item in claims) + "。这只说明该显式实验里的这一条及时报文。默认调度保持 legacy。"
    elif any(item["prediction_improved"] for item in reviews):
        conclusion = "及时报文降低了决策时的风预测误差，但没有在固定对照里改善并保持任务结果。结论止于预测。默认调度保持 legacy。"
    else:
        conclusion = "这条预注册对照没有显示及时报文改善预测或任务结果。默认调度保持 legacy。"
    return reviews, conclusion


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _public_arm(row: dict) -> dict:
    copied = dict(row)
    copied.pop("prediction", None)
    copied["uav_energy_j"] = json.dumps(copied["uav_energy_j"])
    return copied


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    out = out_dir or (ROOT / "results" / "wind_observation_w1" / RUN_ID)
    if "wind_observation_w1" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w1")
    out.mkdir(parents=True, exist_ok=True)
    replay = replay_case()
    zero = {
        "saved_request_replay": {key: replay[key] for key in ("ok", "mismatches", "mismatch_count", "bits", "energy_j", "utility", "returned")},
    }
    if not replay["ok"]:
        (out / "zero_wind.json").write_text(json.dumps(zero, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("zero-wind request replay does not match M1.6")
    print("zero-wind request replay matches M1.6", flush=True)
    fresh = fresh_play(replay["case"])
    zero["fresh_scheduler"] = fresh
    (out / "zero_wind.json").write_text(json.dumps(zero, ensure_ascii=False, indent=2), encoding="utf-8")
    if not fresh["ok"]:
        raise SystemExit("zero-wind scheduler replay does not match M1.6")
    print("zero-wind scheduler matches M1.6", flush=True)
    case = replay["case"]
    cfg = case["cfg"]
    sealed = [uav.consumed_j for uav in replay["prefix"].uavs]
    observer = int(cfg["observer_uav"])
    future = {}
    for epoch in range(int(cfg["decision_epoch"]), decision_count(case["spec"])):
        future[epoch] = {observer: _parse(case["requests"][(epoch, observer)]["executed"])}
    arms = []
    candidates_out = []
    predictions = []
    for condition in cfg["conditions"]:
        rng = random.Random(f"w1-{condition['id']}-{cfg['seed']}")
        message = observe(
            condition,
            replay["sample"]["x_m"],
            replay["sample"]["y_m"],
            replay["sample"]["t_s"],
            cfg["field"],
            float(cfg["observation_sigma_m_s"]),
            rng,
        )
        cache = {"actual": {}, "predicted": {}}
        same_cost = message_cost_j(int(cfg["payload_bytes"]))
        if message_cost_j(int(cfg["payload_bytes"])) != same_cost:
            raise SystemExit("timely and late radio cost diverged")
        for mode in cfg["modes"]:
            print(f"arm {condition['id']} {mode}", flush=True)
            arm, rows = _arm(replay, condition, mode, message, future, cache)
            arms.append(arm)
            candidates_out.extend(rows)
            for point in arm["prediction"]["points"]:
                predictions.append({"condition": condition["id"], "mode": mode, **point})
    if [uav.consumed_j for uav in replay["prefix"].uavs] != sealed:
        raise SystemExit("counterfactual replay changed the decision-state ledger")
    timely_cost = message_cost_j(int(cfg["payload_bytes"]))
    for condition in cfg["conditions"]:
        group = {row["mode"]: row for row in arms if row["condition"] == condition["id"]}
        if abs(group["timely"]["message_j_nominal"] - group["late"]["message_j_nominal"]) > ENERGY_TOL:
            raise SystemExit("timely and late nominal message joules differ")
        if group["no_send"]["message_j"] != 0.0:
            raise SystemExit("no-send was charged for a message")
        if group["timely"]["decision_used_message"] is not True or group["late"]["decision_used_message"] is not False:
            raise SystemExit("message deadline leaked into the decision record")
    reviews, conclusion = _judge(arms)
    summary = {
        "schema_version": 1,
        "parent_commit": _git(),
        "config_sha256": _sha256(CONFIG_PATH),
        "result_dir": str(out.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_s": time.perf_counter() - started,
        "zero_wind": zero,
        "sample": replay["sample"],
        "message_j": timely_cost,
        "reviews": reviews,
        "conclusion": conclusion,
        "default_changed": False,
    }
    _write_csv(out / "scenarios.csv", [_public_arm(row) for row in arms])
    _write_csv(out / "candidates.csv", candidates_out)
    _write_csv(out / "predictions.csv", predictions)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(conclusion, flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
