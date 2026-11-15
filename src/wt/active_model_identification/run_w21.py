"""Separate a landed backhaul from the reserve left at the deadline.

The default guard stays the W2 guard. The explicit option quotes a finished
landing as ground idle through 1800 s plus the cache radio, with no second
flight and no second landing. It does not release the backhaul when that
full remainder already misses the reserve.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.run_w1 import _label, _parse, replay_case
from wt.active_model_identification.scenarios import observation_noise
from wt.active_model_identification.run_w2 import (
    ENERGY_TOL,
    _public_arm,
    _read_csv,
    load_w2,
    roll_arm,
    score_state,
    visible,
)
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import (
    _keep,
    charge_message,
    finished_landing,
    landed_ground_quote,
    make_finished_landing_guard,
    make_return_guard,
    observe,
    predict,
    return_energy_parts,
    true_wind,
)
from wt.active_model_identification.world import advance, flight_state, pad_distance_m

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w21.json"
W2_DIR = ROOT / "results" / "wind_observation_w2" / "w2_20261115"
RUN_ID = "w21_20261115"
SHARED = (
    "scenario_id",
    "method",
    "seed",
    "observer_uav",
    "decider_uav",
    "sample_time_s",
    "decision_time_s",
    "late_time_s",
    "reserve_j",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_w21() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = load_w2()
    for key in SHARED:
        if cfg[key] != parent[key]:
            raise SystemExit(f"W2.1 registration changed locked field {key}")
    if abs(float(cfg["ground_idle_w"]) - 1.0) > 1e-12:
        raise SystemExit("ground idle wattage is no longer the registered 1 W")
    return cfg


def _message(replay: dict, condition: dict, cfg: dict) -> dict:
    rng = random.Random(f"w1-{condition['id']}-{cfg['seed']}")
    sample = replay["sample"]
    return observe(condition, sample["x_m"], sample["y_m"], sample["t_s"], cfg["field"], float(cfg["observation_sigma_m_s"]), rng)


def _w2_condition(condition_id: str, cfg: dict) -> dict:
    return next(row for row in cfg["conditions"] if row["id"] == condition_id)


def trace_archived(replay: dict, condition: dict, mode: str, message: dict, rows: list[dict], cfg: dict) -> list[dict]:
    """Walk the archived executed actions and price idle at every decision."""
    case = replay["case"]
    spec = case["spec"]
    factor = case["factor"]
    decider = int(cfg["decider_uav"])
    observer = int(cfg["observer_uav"])
    world = replay["prefix"].snapshot()
    beliefs = [item.copy() for item in replay["prefix_beliefs"]]
    charged = None
    wind_at = lambda x, y, t, condition=condition: true_wind(condition, x, y, t, cfg["field"])
    points = []
    for row in rows:
        if abs(world.t - float(row["t_s"])) > 1e-6:
            raise SystemExit(f"archived time {row['t_s']} is not the replay time {world.t}")
        if visible(mode, world.t, cfg) and charged is None:
            charged = charge_message(world, observer, decider, int(cfg["payload_bytes"]))
        info = message if visible(mode, world.t, cfg) else None
        point = reserve_point(world, beliefs, info, spec, factor, cfg, decider)
        if abs(point["energy_j"] - float(row["energy_j"])) > 1e-3:
            raise SystemExit(f"{condition['id']} {mode} t={world.t} energy {point['energy_j']} != archived {row['energy_j']}")
        points.append(point)
        epoch = int(row["epoch"])
        actions = [("wait", None) for _ in world.uavs]
        actions[decider] = _parse(row["executed"])
        actions[observer] = _parse(case["requests"][(epoch, observer)]["executed"])
        advance(
            world,
            actions,
            list(case["scenario"]["theta"]),
            beliefs,
            spec,
            factor,
            None,
            planning=False,
            interval_index=epoch,
            noise_j=observation_noise(int(case["scenario"]["seed"]), float(spec["belief"]["sigma_j"])),
            wind_at=wind_at,
            admit=_keep,
            return_guard=_keep,
        )
    return points


def reserve_point(world, beliefs, info, spec: dict, factor: float, cfg: dict, decider: int) -> dict:
    uav = world.uavs[decider]
    remaining_s = max(0.0, float(spec["time"]["mission_s"]) - float(world.t))
    idle_w = float(spec["energy"]["ground_idle_w"])
    theta = beliefs[decider].quantile(float(spec["belief"]["guard_probability"]))
    theta_mean = beliefs[decider].mean()
    landed = finished_landing(uav, spec)
    if landed:
        flight_j = 0.0
        mean_flight_j = 0.0
        flight_s = 0.0
        cruise_j = 0.0
        landing_j = 0.0
    else:
        wind = predict(info, uav.x, uav.y, world.t, cfg["field"], float(cfg["prior_std_m_s"]))["wind_xy"]
        parts = return_energy_parts(uav, theta, spec, factor, wind)
        mean_parts = return_energy_parts(uav, theta_mean, spec, factor, wind)
        flight_j = float(parts["total_j"])
        mean_flight_j = float(mean_parts["total_j"])
        flight_s = float(parts["cruise_s"] + parts["landing_s"])
        cruise_j = float(parts["cruise_j"])
        landing_j = float(parts["landing_j"])
    idle_after_j = idle_w * max(0.0, remaining_s - flight_s)
    return {
        "t_s": float(world.t),
        "energy_j": float(uav.energy_j),
        "flight_state": flight_state(uav),
        "finished_landing": landed,
        "posthoc_idle_upper_j": float(uav.energy_j) - idle_w * remaining_s,
        "knowable_upper_j": float(uav.energy_j) - flight_j - idle_after_j,
        "knowable_mean_upper_j": float(uav.energy_j) - mean_flight_j - idle_after_j,
        "flight_j": flight_j,
        "flight_s": flight_s,
        "cruise_j": cruise_j,
        "landing_j": landing_j,
        "idle_after_j": idle_after_j,
        "source_posthoc": "archived battery minus registered 1 W times the seconds left; optimistic while still airborne",
        "source_knowable": "q95 return quote plus ground idle after that analytic flight; no truth wind",
    }


def first_below(points: list[dict], key: str, reserve: float) -> dict | None:
    return next((row for row in points if row[key] + 1e-6 < reserve), None)


def audit_landing(captured: dict, replay: dict, condition: dict, cfg: dict) -> dict:
    world = captured["world"]
    beliefs = captured["beliefs"]
    info = captured["info"]
    case = replay["case"]
    spec = case["spec"]
    decider = int(cfg["decider_uav"])
    uav = world.uavs[decider]
    theta = beliefs[decider].quantile(float(spec["belief"]["guard_probability"]))
    wind = predict(info, uav.x, uav.y, world.t, cfg["field"], float(cfg["prior_std_m_s"]))["wind_xy"]
    parts = return_energy_parts(uav, theta, spec, case["factor"], wind)
    ground = landed_ground_quote(world, decider, spec)
    frozen = case["requests"][(int(cfg["decision_epoch"]), decider)]["requested"]
    chosen, rows = score_state(
        world, beliefs, info, spec, case["factor"], case["scenario"], decider, cfg["field"], float(cfg["prior_std_m_s"]), condition, frozen
    )
    backhaul = next(row for row in rows if row["action"] == "backhaul:None")
    old = make_return_guard(info, cfg["field"], float(cfg["prior_std_m_s"]), spec, case["factor"], decider)
    option = make_finished_landing_guard(info, cfg["field"], float(cfg["prior_std_m_s"]), spec, case["factor"], decider)
    old_action, old_rewrote = old(uav, ("backhaul", None), beliefs[decider], world)
    new_action, new_rewrote = option(uav, ("backhaul", None), beliefs[decider], world)
    speed = math.hypot(uav.vx, uav.vy)
    return {
        "t_s": float(world.t),
        "x_m": float(uav.x),
        "y_m": float(uav.y),
        "speed_m_s": float(speed),
        "airborne": bool(uav.airborne),
        "flight_state": flight_state(uav),
        "land_timer_s": float(uav.land_timer_s),
        "pad_distance_m": float(pad_distance_m(uav)),
        "energy_j": float(uav.energy_j),
        "finished_landing": finished_landing(uav, spec),
        "guard_cruise_j": float(parts["cruise_j"]),
        "guard_landing_j": float(parts["landing_j"]),
        "guard_total_j": float(parts["total_j"]),
        "guard_need_j": float(parts["total_j"]) + float(spec["energy"]["reserve_j"]),
        "admission_energy_j": float(backhaul["admit_energy_j"]),
        "admission_bits": float(backhaul["score_bits"]),
        "admitted": bool(backhaul["admitted"]),
        "chosen": chosen["action"],
        "ground": ground,
        "old_guard_action": _label(old_action),
        "old_guard_rewrote": bool(old_rewrote),
        "option_guard_action": _label(new_action),
        "option_guard_rewrote": bool(new_rewrote),
        "can_backhaul_now_under_old_admission": bool(backhaul["admitted"]),
        "can_hold_reserve_to_deadline": bool(ground["holds_reserve_without_radio"]),
    }


def _same_executed(arm: dict, rows: list[dict]) -> dict | None:
    for left, right in zip(arm["decisions"], rows):
        if left["executed"] != right["executed"] or left["chosen"] != right["chosen"]:
            return {"t_s": left["t_s"], "option_chosen": left["chosen"], "option_executed": left["executed"], "w2_chosen": right["chosen"], "w2_executed": right["executed"]}
    return None


def _regressed(new: dict, old: dict) -> bool:
    return (
        new["on_time_bits"] + 1e-6 < old["on_time_bits"]
        or new["utility"] + 1e-9 < old["utility"]
        or new["not_landed"] > old["not_landed"]
        or new["below_reserve"] > old["below_reserve"]
        or new["energy_exhausted"] > old["energy_exhausted"]
    )


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    out = out_dir or (ROOT / "results" / "wind_observation_w21" / RUN_ID)
    if "wind_observation_w21" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w21")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_w21()
    w2 = load_w2()
    replay = replay_case()
    if not replay["ok"]:
        raise SystemExit("W1 prefix replay no longer matches M1.6")
    idle_w = float(replay["case"]["spec"]["energy"]["ground_idle_w"])
    if abs(idle_w - float(cfg["ground_idle_w"])) > 1e-12:
        raise SystemExit("spec ground idle does not match the registered 1 W")
    decisions = _read_csv(W2_DIR / "decisions.csv")
    scenarios = _read_csv(W2_DIR / "scenarios.csv")
    headwind = _w2_condition("headwind_return", w2)
    message = _message(replay, headwind, w2)
    timely_rows = [row for row in decisions if row["condition"] == "headwind_return" and row["mode"] == "timely"]
    trace = trace_archived(replay, headwind, "timely", message, timely_rows, w2)
    old = roll_arm(replay, headwind, "timely", message, w2, stop_after_s=float(cfg["audit_time_s"]), capture_s=float(cfg["audit_time_s"]))
    audit = audit_landing(old["captured"], replay, headwind, w2)
    archived_300 = next(row for row in timely_rows if abs(float(row["t_s"]) - float(cfg["audit_time_s"])) <= ENERGY_TOL)
    fresh_300 = next(row for row in old["decisions"] if abs(row["t_s"] - float(cfg["audit_time_s"])) <= ENERGY_TOL)
    if fresh_300["chosen"] != archived_300["chosen"] or fresh_300["executed"] != archived_300["executed"] or fresh_300["block"] != "return_guard":
        raise SystemExit("old guard does not reproduce the 300 s request and rewrite")
    if not audit["finished_landing"]:
        raise SystemExit("300 s is not a finished landing; the ground option stays unused")
    reserve = float(cfg["reserve_j"])
    posthoc = first_below(trace, "posthoc_idle_upper_j", reserve)
    knowable = first_below(trace, "knowable_upper_j", reserve)
    knowable_again = None
    if knowable is not None:
        knowable_again = next((row for row in trace if row["t_s"] > knowable["t_s"] and row["knowable_upper_j"] + 1e-6 >= reserve), None)
    lasting = posthoc is not None and all(row["posthoc_idle_upper_j"] + 1e-6 < reserve for row in trace if row["t_s"] + 1e-6 >= posthoc["t_s"])
    option_rows = []
    divergences = []
    comparisons = []
    for condition in w2["conditions"]:
        arm_message = _message(replay, condition, w2)
        for mode in w2["modes"]:
            print(f"option {condition['id']} {mode}", flush=True)
            arm = roll_arm(replay, condition, mode, arm_message, w2, guard_factory=make_finished_landing_guard)
            archived = [row for row in decisions if row["condition"] == condition["id"] and row["mode"] == mode]
            split = _same_executed(arm, archived)
            if split is not None:
                divergences.append({"condition": condition["id"], "mode": mode, **split})
            reference = next(row for row in scenarios if row["condition"] == condition["id"] and row["mode"] == mode)
            option_rows.append(arm)
            comparisons.append(
                {
                    "condition": condition["id"],
                    "mode": mode,
                    "on_time_bits": arm["on_time_bits"] - float(reference["on_time_bits"]),
                    "energy_j": arm["energy_j"] - float(reference["energy_j"]),
                    "utility": arm["utility"] - float(reference["utility"]),
                    "below_reserve": arm["below_reserve"] - int(float(reference["below_reserve"])),
                    "regressed": _regressed(arm, {key: float(reference[key]) for key in ("on_time_bits", "utility", "not_landed", "below_reserve", "energy_exhausted")}),
                }
            )
    holds = bool(audit["can_hold_reserve_to_deadline"])
    adopt = holds and not divergences and not any(row["regressed"] for row in comparisons)
    if not holds:
        conclusion = "300 秒仅待机已经到不了 4000 J 备用。显式地面报价去掉了虚构飞行，但完整剩余待机开销仍不放行回传。默认守卫不变。"
    elif divergences:
        conclusion = "落地报价改变了已归档轨迹。备用标准没有同时满足，默认守卫不变。"
    else:
        conclusion = "显式地面报价没有改变六臂，默认守卫不变。"
    summary = {
        "schema_version": 1,
        "parent_commit": _git(),
        "config_sha256": _sha256(CONFIG_PATH),
        "w2_decisions_sha256": _sha256(W2_DIR / "decisions.csv"),
        "result_dir": str(out.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_s": time.perf_counter() - started,
        "audit": audit,
        "first_posthoc_idle_shortfall": posthoc,
        "first_knowable_shortfall": knowable,
        "crossings": {
            "posthoc_t_s": None if posthoc is None else posthoc["t_s"],
            "posthoc_stays_below_reserve": lasting,
            "knowable_t_s": None if knowable is None else knowable["t_s"],
            "knowable_clears_again_t_s": None if knowable_again is None else knowable_again["t_s"],
            "prior_decision_still_clears": next(row["t_s"] for row in reversed(trace) if row["t_s"] < (posthoc or {"t_s": 0})["t_s"] and row["posthoc_idle_upper_j"] >= reserve),
        },
        "divergences": divergences,
        "option_minus_w2": comparisons,
        "conclusion": conclusion,
        "adopt": adopt,
        "default_changed": False,
    }
    (out / "ledger.json").write_text(json.dumps({"audit": audit, "trace_headwind_timely": trace}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out / "scenarios.csv", [_public_arm(row) for row in option_rows])
    print(conclusion, flush=True)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
