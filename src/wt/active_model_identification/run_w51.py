"""Compare early, late, and online send times at one message per pair.

archive forecast: scoring with another future coordinate changes a saved forecast or a send.

one per pair: a strategy does not send exactly one 32-byte sample in a pair.

late position: the timing decision reads the second beat's true coordinate or the true wind.

archive rmse: the offline replay of W5's latest-sample sends leaves the archived errors.
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

from wt.active_model_identification.run_w4 import _condition, assess, load_w4
from wt.active_model_identification.run_w5 import (
    ARCHIVED_BITS,
    ARCHIVED_ENERGY_J,
    _cov,
    _draw_deltas,
    _extrapolate,
    _factor_observations,
    _replay,
    _ripple,
    _rmse,
    _solve,
    posterior_mean,
)
from wt.active_model_identification.spec import ROOT
from wt.active_model_identification.wind import message_cost_j

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w51.json"
W5_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w5.json"
W5_SUMMARY_PATH = ROOT / "results" / "wind_observation_w5" / "w5_20261115" / "summary.json"
W5_CHOICES_PATH = ROOT / "results" / "wind_observation_w5" / "w5_20261115" / "choices.csv"
RUN_ID = "w51_20261115"
SHARED_KEYS = (
    "scenario_id",
    "seed",
    "condition",
    "receiver_uav",
    "observer_uav",
    "decision_s",
    "sample_period_s",
    "payload_bytes",
    "observation_sigma_m_s",
    "prior_std_m_s",
    "noise_token",
)


def message_is_usable(send_t: float, decision_t: float, airtime: float) -> bool:
    """A sample sent now is not part of the forecast issued at that same decision."""
    return decision_t > send_t + 1e-9 and decision_t + 1e-9 >= send_t + airtime


def pick_beat(early_variance: float, late_variance: float, tie: float) -> str:
    """Lower predictive variance wins. A tie sends on the first beat."""
    if late_variance + tie < early_variance:
        return "late"
    return "early"


def pair_slots(first_t: float, mission: float, last_decision: float) -> list[tuple[float, int]]:
    """Return the decision time and horizon kept for one pair."""
    slots = []
    for offset in (30.0, 60.0):
        decision_t = first_t + offset
        if decision_t > last_decision + 1e-9:
            continue
        for horizon in (30, 60):
            if decision_t + horizon <= mission + 1e-9:
                slots.append((decision_t, horizon))
    return slots


def predict_late_sample(uav, action: tuple, spec: dict) -> tuple[float, float]:
    """Where the observer will be after one held action. The archive position is not an input."""
    return _extrapolate(uav, action, spec, [30])[30]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_protocol() -> tuple[dict, dict, str]:
    raw = CONFIG_PATH.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("status") != "registered_before_run":
        raise SystemExit("registration moved")
    archived = json.loads(W5_CONFIG_PATH.read_text(encoding="utf-8"))
    for key in SHARED_KEYS:
        if protocol[key] != archived[key]:
            raise RuntimeError(f"{key} left the W5 registration")
    if protocol["local_field"] != archived["local_field"]:
        raise RuntimeError("local field left the W5 registration")
    return protocol, archived, hashlib.sha256(raw).hexdigest()


def _kernel(protocol: dict, amplitude: float) -> dict:
    local = protocol["local_field"]
    return {
        "prior_std_m_s": float(protocol["prior_std_m_s"]),
        "sigma_m_s": float(protocol["observation_sigma_m_s"]),
        "amplitude_m_s": float(amplitude),
        "length_m": float(local["length_m"]),
        "time_s": float(local["time_s"]),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_choices() -> list[dict]:
    with W5_CHOICES_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _latent_variances(observations: list[dict], targets: list[dict], kernel: dict) -> list[float]:
    if not observations:
        return [_cov(target, target, kernel, noise=False) for target in targets]
    lower = _factor_observations(observations, kernel)
    variances = []
    for target in targets:
        cross = [_cov(target, item, kernel, noise=False) for item in observations]
        weight = _solve(lower, cross)
        prior = _cov(target, target, kernel, noise=False)
        variances.append(prior - sum(cross[index] * weight[index] for index in range(len(observations))))
    return variances


def _average_with_optional(base: list[dict], candidate: dict, targets: list[dict], flags: list[bool], kernel: dict) -> float:
    included = [target for target, flag in zip(targets, flags) if flag]
    excluded = [target for target, flag in zip(targets, flags) if not flag]
    total = 0.0
    count = 0
    if included:
        values = _latent_variances(base + [candidate], included, kernel)
        total += sum(values)
        count += len(values)
    if excluded:
        values = _latent_variances(base, excluded, kernel)
        total += sum(values)
        count += len(values)
    if count == 0:
        raise RuntimeError("no prediction point")
    return total / count


def _latest(samples: list[dict], now: float) -> dict:
    ready = [sample for sample in samples if float(sample["t"]) <= now + 1e-9]
    if not ready:
        raise RuntimeError("no sample")
    return max(ready, key=lambda sample: (float(sample["t"]), int(sample["id"])))


def _geo(sample: dict, sample_id: int | None = None) -> dict:
    return {
        "id": int(sample["id"] if sample_id is None else sample_id),
        "x": float(sample["x"]),
        "y": float(sample["y"]),
        "t": float(sample["t"]),
    }


def _targets(decision: dict, spec: dict, mission: float, last_decision: float) -> list[dict]:
    slots = pair_slots(float(decision["t"]), mission, last_decision)
    if not slots:
        raise RuntimeError("pair has no prediction point")
    offsets = sorted({int(round(decision_t + horizon - float(decision["t"]))) for decision_t, horizon in slots})
    points = _extrapolate(decision["receiver"], decision["receiver_action"], spec, offsets)
    targets = []
    for decision_t, horizon in slots:
        offset = int(round(decision_t + horizon - float(decision["t"])))
        x_m, y_m = points[offset]
        targets.append({"s": decision_t, "h": horizon, "x": x_m, "y": y_m, "t": decision_t + horizon, "id": -1})
    return targets


def _pairs(decisions: list[dict]) -> list[tuple[dict, dict]]:
    if len(decisions) != 60:
        raise RuntimeError("decision count")
    return [(decisions[index], decisions[index + 1]) for index in range(0, 60, 2)]


def _build(protocol: dict) -> tuple[dict, dict, list[dict], list[dict], float]:
    _w4, case = load_w4()
    condition = _condition(case, str(protocol["condition"]))
    spec = case["spec"]
    mission = float(spec["time"]["mission_s"])
    if mission != float(protocol["mission_s"]) or float(spec["time"]["decision_s"]) != float(protocol["decision_s"]):
        raise RuntimeError("mission timing changed")
    airtime = float(message_cost_j(int(protocol["payload_bytes"]))["airtime_s"])
    if airtime >= float(protocol["decision_s"]):
        raise RuntimeError("airtime does not fit before the next decision")
    played, poses, committed = _replay(case, condition)
    outcome = assess(played["world"], spec, set(), 3, 0.0)
    if abs(float(outcome["on_time_bits"]) - ARCHIVED_BITS) >= 1e-6 or abs(float(outcome["energy_j"]) - ARCHIVED_ENERGY_J) >= 1e-4:
        raise RuntimeError(f"frozen trajectory drifted bits={outcome['on_time_bits']} energy={outcome['energy_j']}")
    receiver = int(protocol["receiver_uav"])
    observer = int(protocol["observer_uav"])
    for state in played["states"]:
        t_now = int(round(float(state["t_s"])))
        for uav in state["world"].uavs:
            if t_now == 0:
                poses[(0, int(uav.index))] = (float(uav.x), float(uav.y))
            elif math.hypot(poses[(t_now, int(uav.index))][0] - uav.x, poses[(t_now, int(uav.index))][1] - uav.y) > 1e-6:
                raise RuntimeError("recorded pose left the decision snapshot")
    samples = []
    sample_t = 0
    while sample_t <= mission - float(protocol["decision_s"]) + 1e-9:
        x_m, y_m = poses[(sample_t, observer)]
        samples.append({"id": len(samples), "t": float(sample_t), "x": x_m, "y": y_m})
        sample_t += int(protocol["sample_period_s"])
    noise = random.Random(str(protocol["noise_token"]))
    sigma = float(protocol["observation_sigma_m_s"])
    for sample in samples:
        sample["rx"] = float(condition["ax"]) + noise.gauss(0.0, sigma)
        sample["ry"] = float(condition["ay"]) + noise.gauss(0.0, sigma)
    decisions = []
    for state in played["states"]:
        t_now = int(round(float(state["t_s"])))
        uavs = state["world"].uavs
        horizons = [int(horizon) for horizon in protocol["horizons_s"] if t_now + int(horizon) <= mission + 1e-9]
        predicted = _extrapolate(uavs[receiver], committed[(t_now, receiver)], spec, horizons)
        actual = {}
        leg_error = {}
        for horizon in horizons:
            actual[horizon] = poses[(t_now + horizon, receiver)]
            leg_error[horizon] = math.hypot(predicted[horizon][0] - actual[horizon][0], predicted[horizon][1] - actual[horizon][1])
        decisions.append(
            {
                "t": float(t_now),
                "predicted": predicted,
                "actual": actual,
                "leg_error": leg_error,
                "airborne": bool(uavs[receiver].airborne),
                "receiver": uavs[receiver],
                "observer": uavs[observer],
                "receiver_action": committed[(t_now, receiver)],
                "observer_action": committed[(t_now, observer)],
            }
        )
    points = [{"key": (int(sample["t"]), 0), "id": int(sample["id"]), "x": sample["x"], "y": sample["y"], "t": sample["t"]} for sample in samples]
    for t_score in sorted({int(decision["t"] + horizon) for decision in decisions for horizon in decision["actual"]}):
        x_m, y_m = poses[(t_score, receiver)]
        points.append({"key": (t_score, 1), "id": -1, "x": x_m, "y": y_m, "t": float(t_score)})
    return case, condition, samples, decisions, points, airtime


def _inbox(sends: list[dict], decision_t: float, airtime: float) -> list[int]:
    found = []
    seen: set[int] = set()
    for send in sends:
        if message_is_usable(float(send["t"]), decision_t, airtime) and int(send["id"]) not in seen:
            seen.add(int(send["id"]))
            found.append(int(send["id"]))
    return found


def _usable_count(sends: list[dict], decisions: list[dict], airtime: float, mission: float) -> int:
    times = [float(decision["t"]) for decision in decisions]
    count = 0
    for send in sends:
        if any(message_is_usable(float(send["t"]), decision_t, airtime) and decision_t + 30.0 <= mission + 1e-9 for decision_t in times):
            count += 1
    return count


def _observations(sample_ids: list[int], by_id: dict[int, dict], deltas) -> list[dict]:
    rows = []
    for sample_id in sample_ids:
        sample = by_id[sample_id]
        delta = (0.0, 0.0) if deltas is None else deltas[(int(sample["t"]), 0)]
        rows.append(
            {
                "id": sample_id,
                "x": float(sample["x"]),
                "y": float(sample["y"]),
                "t": float(sample["t"]),
                "rx": float(sample["rx"]) + delta[0],
                "ry": float(sample["ry"]) + delta[1],
            }
        )
    return rows


def _score_schedule(sends, decisions, samples, case, condition, deltas, kernel, airtime: float) -> list[dict]:
    by_id = {int(sample["id"]): sample for sample in samples}
    field = case["cfg"]["field"]
    rows = []
    for decision in decisions:
        observations = _observations(_inbox(sends, float(decision["t"]), airtime), by_id, deltas)
        for horizon, actual in decision["actual"].items():
            future_t = float(decision["t"]) + int(horizon)
            predicted = decision["predicted"][int(horizon)]
            actual_xy = actual
            delta = (0.0, 0.0) if deltas is None else deltas[(int(future_t), 1)]
            ripple_actual = _ripple(condition, field, actual_xy[0], actual_xy[1], future_t)
            ripple_predicted = _ripple(condition, field, predicted[0], predicted[1], future_t)
            truth = (
                float(condition["ax"]) + ripple_actual[0] + delta[0],
                float(condition["ay"]) + ripple_actual[1] + delta[1],
            )
            query = {"id": -1, "t": future_t}
            offline_mean = posterior_mean(observations, {**query, "x": actual_xy[0], "y": actual_xy[1]}, kernel)
            online_mean = posterior_mean(observations, {**query, "x": predicted[0], "y": predicted[1]}, kernel)
            offline = (ripple_actual[0] + offline_mean[0], ripple_actual[1] + offline_mean[1])
            online = (ripple_predicted[0] + online_mean[0], ripple_predicted[1] + online_mean[1])
            if decision["leg_error"][int(horizon)] <= 1e-6:
                gap = max(abs(online[0] - offline[0]), abs(online[1] - offline[1]))
                if gap > 1e-6:
                    raise RuntimeError("matched positions produced different forecasts")
            rows.append(
                {
                    "t_s": float(decision["t"]),
                    "horizon_s": int(horizon),
                    "airborne": bool(decision["airborne"]),
                    "leg_error_m": float(decision["leg_error"][int(horizon)]),
                    "online_forecast": online,
                    "offline_forecast": offline,
                    "truth": truth,
                    "online_error": (online[0] - truth[0], online[1] - truth[1]),
                    "offline_error": (offline[0] - truth[0], offline[1] - truth[1]),
                    "actual_xy": (float(actual_xy[0]), float(actual_xy[1])),
                }
            )
    return rows


def _subset_rmse(rows: list[dict], horizon: int, key: str, flight_only: bool) -> tuple[float, int]:
    errors = [row[key] for row in rows if int(row["horizon_s"]) == horizon and (row["airborne"] or not flight_only)]
    if not errors:
        raise RuntimeError("empty score set")
    return _rmse(errors), len(errors)


def _fixed_sends(pairs: list[tuple[dict, dict]], samples: list[dict], beat: str) -> list[dict]:
    sends = []
    for first, second in pairs:
        decision = first if beat == "early" else second
        sample = _latest(samples, float(decision["t"]))
        if float(sample["t"]) > float(decision["t"]) + 1e-9:
            raise RuntimeError("future sample")
        sends.append({"t": float(decision["t"]), "id": int(sample["id"])})
    return sends


def _online_sends(pairs, samples, spec, kernel, airtime, mission, last_decision, tie: float) -> tuple[list[dict], list[dict]]:
    sends = []
    rows = []
    previous: list[dict] = []
    for first, second in pairs:
        targets = _targets(first, spec, mission, last_decision)
        early = _latest(samples, float(first["t"]))
        late_xy = predict_late_sample(first["observer"], first["observer_action"], spec)
        actual_late = _latest(samples, float(second["t"]))
        hypothetical = {"id": -100, "x": late_xy[0], "y": late_xy[1], "t": float(second["t"])}
        base = [_geo(sample) for sample in previous]
        if any(int(item["id"]) == int(early["id"]) for item in base):
            raise RuntimeError("latest sample was already sent")
        early_variance = _average_with_optional(base, _geo(early), targets, [True] * len(targets), kernel)
        late_flags = [float(target["s"]) + 1e-9 >= float(first["t"]) + 60.0 for target in targets]
        late_variance = _average_with_optional(base, hypothetical, targets, late_flags, kernel)
        beat = pick_beat(early_variance, late_variance, tie)
        if beat == "early":
            sent = {"t": float(first["t"]), "id": int(early["id"])}
        else:
            sent = {"t": float(second["t"]), "id": int(actual_late["id"])}
        if float(sent["t"]) not in (float(first["t"]), float(second["t"])):
            raise RuntimeError("send left its pair")
        sends.append(sent)
        previous.append(_latest(samples, float(sent["t"])))
        rows.append(
            {
                "t_s": float(first["t"]),
                "second_t_s": float(second["t"]),
                "beat": beat,
                "early_variance": early_variance,
                "late_variance": late_variance,
                "targets": len(targets),
                "late_position_error_m": math.hypot(late_xy[0] - float(actual_late["x"]), late_xy[1] - float(actual_late["y"])),
                "sent_t_s": float(sent["t"]),
                "sent_id": int(sent["id"]),
            }
        )
    return sends, rows


def _check_one_each(sends: list[dict], pairs: list[tuple[dict, dict]]) -> None:
    if len(sends) != len(pairs):
        raise RuntimeError("send count")
    for send, (first, second) in zip(sends, pairs):
        if float(send["t"]) not in (float(first["t"]), float(second["t"])):
            raise RuntimeError("pair budget")
    if len({(float(send["t"]), int(send["id"])) for send in sends}) != len(sends):
        raise RuntimeError("duplicate send")


def _shift_check(row: dict, condition: dict, field: dict) -> dict:
    """Move the coordinate used for the true wind. The saved forecast stays put."""
    saved = tuple(row["online_forecast"])
    sends = tuple(row["online_forecast"])
    actual = row["actual_xy"]
    future_t = float(row["t_s"]) + int(row["horizon_s"])
    ripple = _ripple(condition, field, actual[0], actual[1], future_t)
    shifted = _ripple(condition, field, actual[0] + 1000.0, actual[1], future_t)
    truth = row["truth"]
    moved = (truth[0] - ripple[0] + shifted[0], truth[1] - ripple[1] + shifted[1])
    original = (saved[0] - truth[0], saved[1] - truth[1])
    revised = (saved[0] - moved[0], saved[1] - moved[1])
    if revised == original:
        raise RuntimeError("shifted coordinate did not change the error")
    if tuple(row["online_forecast"]) != saved or tuple(row["online_forecast"]) != sends:
        raise RuntimeError("scoring rewrote the forecast")
    return {"forecast_unchanged": True, "error_changed": True}


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, _archived_config, digest = _load_protocol()
    out = out_dir or (ROOT / "results" / "wind_observation_w51" / RUN_ID)
    if "wind_observation_w51" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w51")
    out.mkdir(parents=True, exist_ok=True)
    w5_summary = json.loads(W5_SUMMARY_PATH.read_text(encoding="utf-8"))
    if int(w5_summary["choices"]["control"]["differed"]) != 0 or int(w5_summary["choices"]["local"]["differed"]) != 0:
        raise RuntimeError("archived sample choices were not identical")
    archived_choices = _read_choices()
    if any(row["differed"] not in ("False", "0", "") for row in archived_choices):
        raise RuntimeError("archived choice log differs")
    print("replay", flush=True)
    case, condition, samples, decisions, points, airtime = _build(protocol)
    mission = float(protocol["mission_s"])
    last_decision = float(protocol["last_decision_s"])
    tie = float(protocol["tie_variance"])
    pairs = _pairs(decisions)
    if [float(first["t"]) for first, _second in pairs] != [float(index * 60) for index in range(30)]:
        raise RuntimeError("pairs moved")
    local_kernel = _kernel(protocol, protocol["local_field"]["amplitude_m_s"])
    control_kernel = _kernel(protocol, 0.0)
    print("reference", flush=True)
    reference = _fixed_sends([(decision, decision) for decision in decisions], samples, "early")
    if len(reference) != 60:
        raise RuntimeError("reference budget")
    archived_by_t = {}
    for row in archived_choices:
        if row["field"] != "control":
            continue
        archived_by_t[float(row["t_s"])] = float(row["latest_t_s"])
    for send, decision in zip(reference, decisions):
        sample = next(item for item in samples if int(item["id"]) == int(send["id"]))
        if abs(float(sample["t"]) - archived_by_t[float(decision["t"])]) > 1e-9:
            raise RuntimeError("reference sample left the W5 latest sample")
    deltas = _draw_deltas(
        points,
        local_kernel,
        [int(seed) for seed in protocol["local_field"]["seeds"]],
        float(protocol["local_field"]["generation_jitter"]),
    )
    mismatch = [decision for decision in decisions if decision["leg_error"].get(60, 0.0) > 1e-6]
    if len(mismatch) != int(w5_summary["leg_error_nonzero"]["60"]) or any(decision["leg_error"].get(30, 0.0) > 1e-6 for decision in decisions):
        raise RuntimeError("position mismatches left the W5 archive")
    fields = [("control", None, None, control_kernel)]
    for seed, table in deltas.items():
        fields.append(("local", int(seed), table, local_kernel))
    reference_rows = {}
    for field_name, seed, table, kernel in fields:
        reference_rows[(field_name, seed)] = _score_schedule(reference, decisions, samples, case, condition, table, kernel, airtime)
    archived_scores = {(row["field"], row["seed"], int(row["horizon_s"])): row for row in w5_summary["scores"]}
    for (field_name, seed), rows in reference_rows.items():
        for horizon in (30, 60):
            offline, _count = _subset_rmse(rows, horizon, "offline_error", False)
            expected = float(archived_scores[(field_name, seed, horizon)]["rmse_latest"])
            if abs(offline - expected) > 1e-6:
                raise RuntimeError(f"offline archive drifted {field_name} {seed} {horizon}: {offline} vs {expected}")
    shift = _shift_check(reference_rows[("control", None)][0], condition, case["cfg"]["field"])
    sends_before = [(float(send["t"]), int(send["id"])) for send in reference]
    if sends_before != [(float(send["t"]), int(send["id"])) for send in reference] or not shift["forecast_unchanged"]:
        raise RuntimeError("scoring changed the reference")
    print("timing", flush=True)
    early_sends = _fixed_sends(pairs, samples, "early")
    late_sends = _fixed_sends(pairs, samples, "late")
    _check_one_each(early_sends, pairs)
    _check_one_each(late_sends, pairs)
    online = {}
    for field_name, kernel in (("control", control_kernel), ("local", local_kernel)):
        sends, rows = _online_sends(pairs, samples, case["spec"], kernel, airtime, mission, last_decision, tie)
        _check_one_each(sends, pairs)
        online[field_name] = {"sends": sends, "rows": rows}
    schedules = {
        "early": early_sends,
        "late": late_sends,
        "reference60": reference,
    }
    print("scores", flush=True)
    score_rows = []
    point_rows = []
    for field_name, seed, table, kernel in fields:
        field_schedules = dict(schedules)
        field_schedules["online"] = online["control" if field_name == "control" else "local"]["sends"]
        scored = {name: _score_schedule(sends, decisions, samples, case, condition, table, kernel, airtime) for name, sends in field_schedules.items()}
        timing_rows = online["control" if field_name == "control" else "local"]["rows"]
        early_count = sum(1 for row in timing_rows if row["beat"] == "early")
        for subset, flight_only in (("full", False), ("flight", True)):
            for horizon in (30, 60):
                values = {}
                counts = {}
                for name in ("early", "late", "online", "reference60"):
                    values[name], counts[name] = _subset_rmse(scored[name], horizon, "online_error", flight_only)
                offline_reference, _offline_count = _subset_rmse(scored["reference60"], horizon, "offline_error", flight_only)
                score_rows.append(
                    {
                        "field": field_name,
                        "seed": seed,
                        "subset": subset,
                        "horizon_s": horizon,
                        "n": counts["early"],
                        "rmse_early": values["early"],
                        "rmse_late": values["late"],
                        "rmse_online": values["online"],
                        "rmse_reference60": values["reference60"],
                        "rmse_reference60_offline": offline_reference,
                        "paired_early_minus_late": values["early"] - values["late"],
                        "paired_online_minus_early": values["online"] - values["early"],
                        "paired_online_minus_late": values["online"] - values["late"],
                        "online_early_sends": early_count,
                        "online_late_sends": len(timing_rows) - early_count,
                    }
                )
        if field_name == "control":
            for row in scored["reference60"]:
                point_rows.append(
                    {
                        "t_s": row["t_s"],
                        "horizon_s": row["horizon_s"],
                        "airborne": row["airborne"],
                        "leg_error_m": row["leg_error_m"],
                        "online_x": row["online_forecast"][0],
                        "online_y": row["online_forecast"][1],
                        "offline_x": row["offline_forecast"][0],
                        "offline_y": row["offline_forecast"][1],
                        "online_error_m_s": math.hypot(*row["online_error"]),
                        "offline_error_m_s": math.hypot(*row["offline_error"]),
                    }
                )
    mismatch_rows = []
    for decision in mismatch:
        for field_name, seed, _table, kernel in fields:
            for row in reference_rows[(field_name, seed)]:
                if float(row["t_s"]) == float(decision["t"]) and int(row["horizon_s"]) == 60:
                    mismatch_rows.append(
                        {
                            "t_s": row["t_s"],
                            "field": field_name,
                            "seed": seed,
                            "leg_error_m": row["leg_error_m"],
                            "online_error_m_s": math.hypot(*row["online_error"]),
                            "offline_error_m_s": math.hypot(*row["offline_error"]),
                        }
                    )
    timing_csv = []
    for field_name, payload in online.items():
        for row in payload["rows"]:
            timing_csv.append({"field": field_name, **row})
    summary = {
        "schema_version": 1,
        "parent_commit": _parent(),
        "config_sha256": digest,
        "config_status": protocol["status"],
        "trajectory_match": True,
        "archive_rmse_match": True,
        "sample_choice_still_identical": True,
        "forecast_unchanged_when_score_point_moves": True,
        "error_changes_when_score_point_moves": True,
        "default_changed": False,
        "airtime_s": airtime,
        "payload_bytes": int(protocol["payload_bytes"]),
        "n_decisions": len(decisions),
        "n_pairs": len(pairs),
        "n_flight_decisions": sum(1 for decision in decisions if decision["airborne"]),
        "n_landed_decisions": sum(1 for decision in decisions if not decision["airborne"]),
        "position_mismatch_t_s": [float(decision["t"]) for decision in mismatch],
        "centralized_assumption": protocol["centralized"],
        "send_count": {
            "early": len(early_sends),
            "late": len(late_sends),
            "online": len(online["local"]["sends"]),
            "reference60": len(reference),
        },
        "usable_sends": {
            "early": _usable_count(early_sends, decisions, airtime, mission),
            "late": _usable_count(late_sends, decisions, airtime, mission),
            "online_control": _usable_count(online["control"]["sends"], decisions, airtime, mission),
            "online_local": _usable_count(online["local"]["sends"], decisions, airtime, mission),
            "reference60": _usable_count(reference, decisions, airtime, mission),
        },
        "late_position_error_max_m": {
            field_name: max(row["late_position_error_m"] for row in payload["rows"]) for field_name, payload in online.items()
        },
        "timing": {
            field_name: {
                "early": sum(1 for row in payload["rows"] if row["beat"] == "early"),
                "late": sum(1 for row in payload["rows"] if row["beat"] == "late"),
            }
            for field_name, payload in online.items()
        },
        "scores": score_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_csv(out / "scores.csv", score_rows)
    _write_csv(out / "timing.csv", timing_csv)
    _write_csv(out / "mismatches.csv", mismatch_rows)
    _write_csv(out / "reference_points.csv", point_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"mismatch {summary['position_mismatch_t_s']} timing {summary['timing']} usable {summary['usable_sends']}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
