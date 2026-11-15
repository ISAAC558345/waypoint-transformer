"""Compare an onboard wind sensor with the remote messages already scored in W6.

same field: an archived coordinate is given a new latent value, or the arms do not share one measurement.

remote on time: a remote sample enters at its send decision, a same-beat onboard sample waits, or the delayed arm leaves the remote clock.

remote baseline: remote-only forecasts leave the W6 weighted errors.
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

from wt.active_model_identification.run_w5 import _cholesky, _draw_deltas, _ripple, _rmse, _solve, posterior_mean
from wt.active_model_identification.run_w51 import (
    _build,
    _fixed_sends,
    _kernel,
    _latent_variances,
    _pairs,
    message_is_usable,
)
from wt.active_model_identification.run_w6 import _loglik, _measurement, _normalize, _predict
from wt.active_model_identification.spec import ROOT

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w61.json"
W6_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w6.json"
W51_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w51.json"
RUN_ID = "w61_20261115"
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
    "horizons_s",
    "prior_weight_global",
    "prior_weight_local",
)
ARMS = ("A", "B", "C", "delayed")
TOLERANCE_M = 1e-6


def observation_is_usable(source: str, sample_t: float, decision_t: float, airtime: float) -> bool:
    """Onboard wind is usable in the beat it is taken. A radio sample waits out its airtime."""
    if source == "onboard":
        return decision_t + 1e-9 >= sample_t
    if source in ("remote", "onboard_delayed"):
        return message_is_usable(sample_t, decision_t, airtime)
    raise ValueError(source)


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
    archived = json.loads(W6_CONFIG_PATH.read_text(encoding="utf-8"))
    for key in SHARED_KEYS:
        if protocol[key] != archived[key]:
            raise RuntimeError(f"{key} left the W6 registration")
    if protocol["local_field"] != archived["local_field"]:
        raise RuntimeError("local field left the W6 registration")
    if protocol["onboard_noise_token"] == protocol["noise_token"]:
        raise RuntimeError("onboard noise reuses the remote stream")
    if not protocol.get("conditional_token") or not protocol.get("onboard_noise_token"):
        raise RuntimeError("noise token")
    return protocol, archived, hashlib.sha256(raw).hexdigest()


def _delta_cov(left: dict, right: dict, kernel: dict, jitter: float, *, diagonal: bool) -> float:
    distance = math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
    dt = abs(float(left["t"]) - float(right["t"]))
    value = (float(kernel["amplitude_m_s"]) ** 2) * math.exp(-distance / float(kernel["length_m"])) * math.exp(
        -dt / float(kernel["time_s"])
    )
    if diagonal:
        value += jitter
    return value


def _conditional_alpha(points: list[dict], new_point: dict, kernel: dict, jitter: float) -> tuple[list[float], float]:
    """Return K^{-1}k and the conditional standard deviation of one new latent point."""
    self_var = _delta_cov(new_point, new_point, kernel, jitter, diagonal=True)
    size = len(points)
    if size == 0:
        return [], math.sqrt(self_var)
    matrix = [
        [_delta_cov(points[row], points[col], kernel, jitter, diagonal=row == col) for col in range(size)]
        for row in range(size)
    ]
    lower = _cholesky(matrix)
    cross = [_delta_cov(new_point, point, kernel, jitter, diagonal=False) for point in points]
    alpha = _solve(lower, cross)
    variance = self_var - sum(cross[index] * alpha[index] for index in range(size))
    if variance < -1e-8:
        raise RuntimeError("conditional variance")
    return alpha, math.sqrt(max(variance, 0.0))


def _find_point(points: list[dict], site: dict) -> tuple[str, dict | None]:
    key = (int(round(float(site["t"]))), 1)
    exact = [point for point in points if point["key"] == key]
    if len(exact) > 1:
        raise RuntimeError("duplicate receiver point")
    if exact:
        point = exact[0]
        if math.hypot(float(point["x"]) - float(site["x"]), float(point["y"]) - float(site["y"])) > TOLERANCE_M:
            raise RuntimeError("archived receiver point moved")
        return "archived", point
    matches = []
    for point in points:
        same_t = abs(float(point["t"]) - float(site["t"])) <= TOLERANCE_M
        same_xy = math.hypot(float(point["x"]) - float(site["x"]), float(point["y"]) - float(site["y"])) <= TOLERANCE_M
        if same_t and same_xy:
            matches.append(point)
    if len(matches) > 1:
        raise RuntimeError("several coincident field points")
    if matches:
        return "coincident", matches[0]
    return "missing", None


def _extend_tables(points: list[dict], tables: dict, sites: list[dict], kernel: dict, jitter: float, token: str) -> tuple[dict, list[dict]]:
    seeds = list(tables)
    archived_keys = set(tables[seeds[0]])
    for seed in seeds[1:]:
        if set(tables[seed]) != archived_keys:
            raise RuntimeError("seeds do not share the archived field")
    extended = {seed: dict(table) for seed, table in tables.items()}
    known = list(points)
    rows = []
    for site in sites:
        status, point = _find_point(known, site)
        key = (int(round(float(site["t"]))), 1)
        if status == "missing":
            new_point = {"key": key, "id": -2, "x": float(site["x"]), "y": float(site["y"]), "t": float(site["t"])}
            alpha, std = _conditional_alpha(known, new_point, kernel, jitter)
            for seed in seeds:
                east = [extended[seed][item["key"]][0] for item in known]
                north = [extended[seed][item["key"]][1] for item in known]
                mean_east = sum(alpha[index] * east[index] for index in range(len(known)))
                mean_north = sum(alpha[index] * north[index] for index in range(len(known)))
                east_rng = random.Random(f"{token}-{seed}-x-{key[0]}")
                north_rng = random.Random(f"{token}-{seed}-y-{key[0]}")
                extended[seed][key] = (
                    mean_east + std * east_rng.gauss(0.0, 1.0),
                    mean_north + std * north_rng.gauss(0.0, 1.0),
                )
            known.append(new_point)
            rows.append({"t_s": float(site["t"]), "status": "drawn", "conditional_std_m_s": std})
            continue
        for seed in seeds:
            _seed_status, seed_point = _find_point(known, site)
            if _seed_status != status or seed_point is None:
                raise RuntimeError("seeds resolved a site differently")
            extended[seed][key] = extended[seed][seed_point["key"]]
        rows.append({"t_s": float(site["t"]), "status": status, "conditional_std_m_s": None})
    for seed in seeds:
        for key, value in tables[seed].items():
            if extended[seed][key] != value:
                raise RuntimeError("archived field changed")
    return extended, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _hyp(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = (len(ordered) - 1) / 2.0
    low = int(math.floor(mid))
    high = int(math.ceil(mid))
    return 0.5 * (ordered[low] + ordered[high])


def _onboard_noise(sites: list[dict], token: str, sigma: float) -> None:
    noise = random.Random(str(token))
    for site in sites:
        site["nx"] = noise.gauss(0.0, sigma)
        site["ny"] = noise.gauss(0.0, sigma)


def _onboard_measurement(site: dict, deltas, ax: float, ay: float, sample_id: int) -> dict:
    key = (int(round(float(site["t"]))), 1)
    delta = (0.0, 0.0) if deltas is None else deltas[key]
    return {
        "id": sample_id,
        "source": "onboard",
        "x": float(site["x"]),
        "y": float(site["y"]),
        "t": float(site["t"]),
        "rx": ax + float(site["nx"]) + delta[0],
        "ry": ay + float(site["ny"]) + delta[1],
    }


def _remote_measurement(sample: dict, deltas) -> dict:
    measurement = _measurement(sample, deltas)
    measurement["source"] = "remote"
    return measurement


def _mixture_variance(observations: list[dict], query: dict, kernels: dict[str, dict], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    means = {}
    variances = {}
    for name, kernel in kernels.items():
        means[name] = posterior_mean(observations, query, kernel)
        variance = _latent_variances(observations, [query], kernel)[0]
        if variance < -1e-6:
            raise RuntimeError("negative variance")
        variances[name] = max(variance, 0.0)
    mean_x = sum(weights[name] * means[name][0] for name in weights)
    mean_y = sum(weights[name] * means[name][1] for name in weights)
    var_x = 0.0
    var_y = 0.0
    for name in weights:
        var_x += weights[name] * (variances[name] + (means[name][0] - mean_x) ** 2)
        var_y += weights[name] * (variances[name] + (means[name][1] - mean_y) ** 2)
    wind = {}
    for name in means:
        wind[name] = means[name]
    return 0.5 * (var_x + var_y), variances | {"mean_global": means["global"], "mean_local": means["local"], "wind_x": mean_x, "wind_y": mean_y}


def _geometry(observations: list[dict], source: str, predicted: tuple[float, float], future_t: float) -> dict:
    chosen = [item for item in observations if item["source"] == source]
    if not chosen:
        return {}
    def distance(item: dict) -> float:
        return _hyp(float(item["x"]) - predicted[0], float(item["y"]) - predicted[1])
    latest = max(chosen, key=lambda item: (float(item["t"]), int(item["id"])))
    nearest = min(chosen, key=lambda item: (distance(item), float(item["t"]), int(item["id"])))
    prefix = "onboard" if source == "onboard" else "remote"
    return {
        f"latest_{prefix}_distance_m": distance(latest),
        f"latest_{prefix}_age_s": future_t - float(latest["t"]),
        f"latest_{prefix}_t_s": float(latest["t"]),
        f"nearest_{prefix}_distance_m": distance(nearest),
        f"nearest_{prefix}_age_s": future_t - float(nearest["t"]),
    }


def _assimilate(observations: list[dict], log_weights: dict[str, float], measurement: dict, kernels: dict[str, dict]) -> None:
    if any(int(item["id"]) == int(measurement["id"]) for item in observations):
        raise RuntimeError("duplicate sample")
    likelihoods = {}
    for name, kernel in kernels.items():
        mean, variance = _predict(observations, measurement, kernel)
        likelihoods[name] = _loglik(measurement, mean, variance)
    for name in log_weights:
        log_weights[name] += likelihoods[name]
    observations.append(measurement)


def _run_arm(arm: str, decisions, onboard, remote, kernels, case, condition, deltas, airtime: float, separations: dict[int, float], prior: dict[str, float]) -> tuple[list[dict], tuple]:
    arrivals: dict[float, list[dict]] = {}
    if arm in ("A", "B"):
        for measurement in onboard:
            if not observation_is_usable("onboard", float(measurement["t"]), float(measurement["t"]), airtime):
                raise RuntimeError("onboard sample is not usable in its own beat")
            arrivals.setdefault(float(measurement["t"]), []).append(measurement)
    if arm == "delayed":
        for measurement in onboard:
            arrival = next(
                float(decision["t"])
                for decision in decisions
                if observation_is_usable("onboard_delayed", float(measurement["t"]), float(decision["t"]), airtime)
            )
            arrivals.setdefault(arrival, []).append(measurement)
    if arm in ("B", "C"):
        for measurement in remote:
            arrival = next(
                float(decision["t"])
                for decision in decisions
                if observation_is_usable("remote", float(measurement["t"]), float(decision["t"]), airtime)
            )
            if abs(arrival - float(measurement["t"])) <= 1e-9:
                raise RuntimeError("message arrived at its send decision")
            arrivals.setdefault(arrival, []).append(measurement)
    observations: list[dict] = []
    log_weights = {name: math.log(float(prior[name])) for name in ("global", "local")}
    field = case["cfg"]["field"]
    rows = []
    ax = float(condition["ax"])
    ay = float(condition["ay"])
    for decision in decisions:
        new_items = arrivals.get(float(decision["t"]), [])
        onboard_first = [item for item in new_items if item["source"] == "onboard"]
        remote_now = [item for item in new_items if item["source"] == "remote"]
        for measurement in onboard_first + remote_now:
            _assimilate(observations, log_weights, measurement, kernels)
        weights = _normalize(log_weights)
        n_onboard = sum(item["source"] == "onboard" for item in observations)
        n_remote = sum(item["source"] == "remote" for item in observations)
        remote_times = tuple(sorted(float(item["t"]) for item in observations if item["source"] == "remote"))
        onboard_times = tuple(sorted(float(item["t"]) for item in observations if item["source"] == "onboard"))
        for horizon, actual in decision["actual"].items():
            future_t = float(decision["t"]) + int(horizon)
            predicted = decision["predicted"][int(horizon)]
            query = {"id": -1, "x": float(predicted[0]), "y": float(predicted[1]), "t": future_t}
            ripple_predicted = _ripple(condition, field, predicted[0], predicted[1], future_t)
            if decision["airborne"]:
                mixture, parts = _mixture_variance(observations, query, kernels, weights)
            else:
                parts = {
                    "mean_global": posterior_mean(observations, query, kernels["global"]),
                    "mean_local": posterior_mean(observations, query, kernels["local"]),
                }
                mixture = None
            wind = (
                weights["global"] * (ripple_predicted[0] + parts["mean_global"][0]) + weights["local"] * (ripple_predicted[0] + parts["mean_local"][0]),
                weights["global"] * (ripple_predicted[1] + parts["mean_global"][1]) + weights["local"] * (ripple_predicted[1] + parts["mean_local"][1]),
            )
            score_key = (int(round(future_t)), 1)
            if deltas is not None and score_key not in deltas:
                raise RuntimeError("score point left the archived field")
            delta = (0.0, 0.0) if deltas is None else deltas[score_key]
            ripple_actual = _ripple(condition, field, actual[0], actual[1], future_t)
            truth = (ax + ripple_actual[0] + delta[0], ay + ripple_actual[1] + delta[1])
            error = (wind[0] - truth[0], wind[1] - truth[1])
            row = {
                "t_s": float(decision["t"]),
                "horizon_s": int(horizon),
                "airborne": bool(decision["airborne"]),
                "n_onboard": n_onboard,
                "n_remote": n_remote,
                "n_new_remote": len(remote_now),
                "new_remote_t_s": None if not remote_now else float(max(remote_now, key=lambda item: float(item["t"]))["t"]),
                "remote_times": remote_times,
                "onboard_times": onboard_times,
                "weight_local": weights["local"],
                "forecast": wind,
                "error": error,
                "variance": mixture,
            }
            if decision["airborne"]:
                row.update(_geometry(observations, "onboard", predicted, future_t))
                row.update(_geometry(observations, "remote", predicted, future_t))
                latest_remote_t = row.get("latest_remote_t_s")
                if latest_remote_t is not None:
                    row["pair_separation_m"] = separations[int(round(latest_remote_t))]
            rows.append(row)
    signature = tuple(
        (item["source"], int(item["id"]), float(item["rx"]), float(item["ry"]), float(item["t"]), float(item["x"]), float(item["y"]))
        for item in observations
    )
    return rows, signature


def _subset(rows: list[dict], horizon: int, flight_only: bool) -> tuple[float, int]:
    errors = [row["error"] for row in rows if int(row["horizon_s"]) == horizon and (row["airborne"] or not flight_only)]
    if not errors:
        raise RuntimeError("empty score set")
    return _rmse(errors), len(errors)


def _blank(row: dict, name: str):
    return row.get(name)


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, _archived, digest = _load_protocol()
    out = out_dir or (ROOT / "results" / "wind_observation_w61" / RUN_ID)
    if "wind_observation_w61" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w61")
    out.mkdir(parents=True, exist_ok=True)
    archived_summary = json.loads((ROOT / str(protocol["w6_result"])).read_text(encoding="utf-8"))
    archived_scores = {
        (row["field"], None if row["seed"] is None else int(row["seed"]), row["subset"], int(row["horizon_s"])): float(row["rmse_online"])
        for row in archived_summary["scores"]
    }
    print("replay", flush=True)
    w51 = json.loads(W51_CONFIG_PATH.read_text(encoding="utf-8"))
    case, condition, samples, decisions, points, airtime = _build(w51)
    sends = _fixed_sends(_pairs(decisions), samples, "early")
    if len(sends) != 30 or int(protocol["payload_bytes"]) != 32:
        raise RuntimeError("send budget")
    by_id = {int(sample["id"]): sample for sample in samples}
    sites = []
    for send in sends:
        sample = by_id[int(send["id"])]
        if abs(float(sample["t"]) - float(send["t"])) > 1e-9:
            raise RuntimeError("early send is not the sample taken at that decision")
        decision = next(item for item in decisions if abs(float(item["t"]) - float(send["t"])) <= 1e-9)
        if int(decision["receiver"].index) != int(protocol["receiver_uav"]):
            raise RuntimeError("receiver index")
        sites.append(
            {
                "t": float(sample["t"]),
                "x": float(decision["receiver"].x),
                "y": float(decision["receiver"].y),
                "observer_x": float(sample["x"]),
                "observer_y": float(sample["y"]),
                "remote_id": int(sample["id"]),
            }
        )
    _onboard_noise(sites, protocol["onboard_noise_token"], float(protocol["observation_sigma_m_s"]))
    separations = {int(round(site["t"])): _hyp(site["x"] - site["observer_x"], site["y"] - site["observer_y"]) for site in sites}
    kernel_global = _kernel(protocol, 0.0)
    kernel_local = _kernel(protocol, protocol["local_field"]["amplitude_m_s"])
    kernels = {"global": kernel_global, "local": kernel_local}
    prior = {"global": protocol["prior_weight_global"], "local": protocol["prior_weight_local"]}
    deltas = _draw_deltas(
        points,
        kernel_local,
        [int(seed) for seed in protocol["local_field"]["seeds"]],
        float(protocol["local_field"]["generation_jitter"]),
    )
    print("extend", flush=True)
    extended, extension_rows = _extend_tables(
        points,
        deltas,
        sites,
        kernel_local,
        float(protocol["local_field"]["generation_jitter"]),
        str(protocol["conditional_token"]),
    )
    flight_times = [float(decision["t"]) for decision in decisions if decision["airborne"]]
    if flight_times != [float(item) for item in archived_summary["flight_t_s"]]:
        raise RuntimeError("flight decisions left W6")
    ax = float(condition["ax"])
    ay = float(condition["ay"])
    fields = [("control", None, None)] + [("local", int(seed), extended[int(seed)]) for seed in extended]
    score_rows = []
    moment_rows = []
    print("forecast", flush=True)
    first_decision_t = float(decisions[0]["t"])
    expected_arrivals = [
        next(float(decision["t"]) for decision in decisions if message_is_usable(float(send["t"]), float(decision["t"]), airtime))
        for send in sends
    ]
    for field_name, seed, table in fields:
        onboard = [
            _onboard_measurement(site, table, ax, ay, 100000 + index)
            for index, site in enumerate(sites)
        ]
        remote = [_remote_measurement(by_id[int(send["id"])], table) for send in sends]
        if field_name == "control" and any(item["rx"] != ax + site["nx"] or item["ry"] != ay + site["ny"] for item, site in zip(onboard, sites)):
            raise RuntimeError("control onboard latent left zero")
        arm_rows = {}
        arm_signatures = {}
        for arm in ARMS:
            arm_rows[arm], arm_signatures[arm] = _run_arm(
                arm, decisions, onboard, remote, kernels, case, condition, table, airtime, separations, prior
            )
        _check_same_measurements(arm_signatures, onboard, remote)
        _check_arm_contract(arm_rows, first_decision_t, expected_arrivals)
        true_name = "global" if field_name == "control" else "local"
        for subset, flight_only in (("full", False), ("flight", True)):
            for horizon in (30, 60):
                rmse = {}
                count = None
                for arm in ARMS:
                    value, arm_count = _subset(arm_rows[arm], horizon, flight_only)
                    rmse[arm] = value
                    if count is None:
                        count = arm_count
                    elif count != arm_count:
                        raise RuntimeError("arms scored different points")
                if (field_name, seed, subset, horizon) in archived_scores:
                    expected = archived_scores[(field_name, seed, subset, horizon)]
                    if abs(rmse["C"] - expected) > 1e-6:
                        raise RuntimeError(f"remote arm left W6 {field_name} {seed} {subset} {horizon}: {rmse['C']} vs {expected}")
                score_rows.append(
                    {
                        "field": field_name,
                        "seed": seed,
                        "subset": subset,
                        "horizon_s": horizon,
                        "n": count,
                        "true_model": true_name,
                        "rmse_a": rmse["A"],
                        "rmse_b": rmse["B"],
                        "rmse_c": rmse["C"],
                        "rmse_delayed": rmse["delayed"],
                        "paired_b_minus_a": rmse["B"] - rmse["A"],
                        "paired_delayed_minus_c": rmse["delayed"] - rmse["C"],
                    }
                )
        moment_rows.extend(_moments(field_name, seed, true_name, arm_rows, int(protocol["payload_bytes"])))
    matched_keys = {
        (row["field"], None if row["seed"] is None else int(row["seed"]), row["subset"], int(row["horizon_s"]))
        for row in score_rows
    }
    if not set(archived_scores).issubset(matched_keys):
        raise RuntimeError("W6 score key missing")
    geometry_rows = []
    for site, extension in zip(sites, extension_rows):
        geometry_rows.append(
            {
                "t_s": float(site["t"]),
                "receiver_x_m": float(site["x"]),
                "receiver_y_m": float(site["y"]),
                "observer_x_m": float(site["observer_x"]),
                "observer_y_m": float(site["observer_y"]),
                "separation_m": separations[int(round(site["t"]))],
                "field_status": extension["status"],
                "conditional_std_m_s": extension["conditional_std_m_s"],
            }
        )
    summary = {
        "schema_version": 1,
        "parent_commit": _parent(),
        "config_sha256": digest,
        "config_status": protocol["status"],
        "same_field": True,
        "remote_on_time": True,
        "archive_match": True,
        "default_changed": False,
        "n_onboard_samples": len(sites),
        "n_remote_messages": len(sends),
        "payload_bytes": int(protocol["payload_bytes"]),
        "extra_messages_vs_onboard": len(sends),
        "extra_bytes_vs_onboard": len(sends) * int(protocol["payload_bytes"]),
        "n_drawn_points": sum(row["status"] == "drawn" for row in extension_rows),
        "n_reused_points": sum(row["status"] != "drawn" for row in extension_rows),
        "separation_min_m": min(separations.values()),
        "separation_median_m": _median(list(separations.values())),
        "separation_max_m": max(separations.values()),
        "n_flight_decisions": len(flight_times),
        "flight_t_s": flight_times,
        "scores": score_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_csv(out / "scores.csv", score_rows)
    _write_csv(out / "moments.csv", moment_rows)
    _write_csv(out / "geometry.csv", geometry_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"flight {flight_times[0]}-{flight_times[-1]} onboard {len(sites)} drawn {summary['n_drawn_points']} "
        f"separation {summary['separation_min_m']:.1f}-{summary['separation_max_m']:.1f}",
        flush=True,
    )
    return summary


def _source_values(signature: tuple, source: str) -> tuple:
    return tuple(item[1:] for item in signature if item[0] == source)


def _built_values(measurements: list[dict]) -> tuple:
    return tuple(
        (int(item["id"]), float(item["rx"]), float(item["ry"]), float(item["t"]), float(item["x"]), float(item["y"]))
        for item in measurements
    )


def _check_same_measurements(signatures: dict[str, tuple], onboard: list[dict], remote: list[dict]) -> None:
    onboard_values = _built_values(onboard)
    remote_values = _built_values(remote)
    if _source_values(signatures["A"], "onboard") != onboard_values or _source_values(signatures["A"], "remote"):
        raise RuntimeError("onboard arm did not keep the shared onboard samples")
    if _source_values(signatures["B"], "onboard") != onboard_values or _source_values(signatures["B"], "remote") != remote_values:
        raise RuntimeError("shared arm split the measurements")
    if _source_values(signatures["C"], "remote") != remote_values or _source_values(signatures["C"], "onboard"):
        raise RuntimeError("remote arm did not keep the W6 messages")
    if _source_values(signatures["delayed"], "onboard") != onboard_values or _source_values(signatures["delayed"], "remote"):
        raise RuntimeError("delayed arm changed the onboard samples")


def _check_arm_contract(arm_rows: dict[str, list[dict]], first_t: float, expected_arrivals: list[float]) -> None:
    by_arm = {}
    for arm, rows in arm_rows.items():
        by_arm[arm] = {(float(row["t_s"]), int(row["horizon_s"])): row for row in rows}
    horizons = {horizon for _t, horizon in by_arm["A"]}
    for horizon in horizons:
        first_a = by_arm["A"][(first_t, horizon)]
        first_b = by_arm["B"][(first_t, horizon)]
        if first_a["n_remote"] != 0 or first_b["n_remote"] != 0 or first_a["n_onboard"] != 1:
            raise RuntimeError("remote message used before it arrived")
        if _hyp(first_a["forecast"][0] - first_b["forecast"][0], first_a["forecast"][1] - first_b["forecast"][1]) > 1e-9:
            raise RuntimeError("onboard arms diverged before any remote message")
        if by_arm["C"][(first_t, horizon)]["n_remote"] != 0 or by_arm["delayed"][(first_t, horizon)]["n_onboard"] != 0:
            raise RuntimeError("delayed clock moved")
    for key, remote_row in by_arm["C"].items():
        delayed_row = by_arm["delayed"][key]
        if remote_row["remote_times"] != delayed_row["onboard_times"] or delayed_row["n_remote"] != 0:
            raise RuntimeError("delayed onboard left the remote arrival clock")
        if remote_row["n_onboard"] != 0 or remote_row["n_remote"] != delayed_row["n_onboard"]:
            raise RuntimeError("equal-count clock")
        shared = by_arm["B"][key]
        if shared["onboard_times"] != by_arm["A"][key]["onboard_times"] or shared["remote_times"] != remote_row["remote_times"]:
            raise RuntimeError("shared arm did not keep both sources")
    arrived_at = [float(row["t_s"]) for row in arm_rows["C"] if int(row["horizon_s"]) == 30 and int(row["n_new_remote"]) > 0]
    if arrived_at != expected_arrivals:
        raise RuntimeError("remote arrival times")


def _moments(field_name: str, seed: int | None, true_name: str, arm_rows: dict[str, list[dict]], payload: int) -> list[dict]:
    index = {
        arm: {(float(row["t_s"]), int(row["horizon_s"])): row for row in rows if row["airborne"]}
        for arm, rows in arm_rows.items()
    }
    moments = []
    for key in index["A"]:
        row_a = index["A"][key]
        row_b = index["B"][key]
        row_c = index["C"][key]
        row_d = index["delayed"][key]
        error_a = _hyp(*row_a["error"])
        error_b = _hyp(*row_b["error"])
        moments.append(
            {
                "field": field_name,
                "seed": seed,
                "true_model": true_name,
                "t_s": row_a["t_s"],
                "horizon_s": row_a["horizon_s"],
                "n_onboard": row_a["n_onboard"],
                "n_remote": row_b["n_remote"],
                "extra_messages": row_b["n_remote"],
                "extra_bytes": row_b["n_remote"] * payload,
                "remote_arrived": int(row_b["n_new_remote"]) > 0,
                "new_remote_t_s": row_b["new_remote_t_s"],
                "weight_local_a": row_a["weight_local"],
                "weight_local_b": row_b["weight_local"],
                "weight_local_c": row_c["weight_local"],
                "var_a": row_a["variance"],
                "var_b": row_b["variance"],
                "var_drop": None if row_a["variance"] is None or row_b["variance"] is None else row_a["variance"] - row_b["variance"],
                "error_a": error_a,
                "error_b": error_b,
                "error_c": _hyp(*row_c["error"]),
                "error_delayed": _hyp(*row_d["error"]),
                "sq_a": row_a["error"][0] ** 2 + row_a["error"][1] ** 2,
                "sq_b": row_b["error"][0] ** 2 + row_b["error"][1] ** 2,
                "sq_c": row_c["error"][0] ** 2 + row_c["error"][1] ** 2,
                "sq_delayed": row_d["error"][0] ** 2 + row_d["error"][1] ** 2,
                "latest_onboard_distance_m": _blank(row_a, "latest_onboard_distance_m"),
                "latest_onboard_age_s": _blank(row_a, "latest_onboard_age_s"),
                "nearest_onboard_distance_m": _blank(row_a, "nearest_onboard_distance_m"),
                "nearest_onboard_age_s": _blank(row_a, "nearest_onboard_age_s"),
                "latest_remote_distance_m": _blank(row_b, "latest_remote_distance_m"),
                "latest_remote_age_s": _blank(row_b, "latest_remote_age_s"),
                "nearest_remote_distance_m": _blank(row_b, "nearest_remote_distance_m"),
                "nearest_remote_age_s": _blank(row_b, "nearest_remote_age_s"),
                "pair_separation_m": _blank(row_b, "pair_separation_m"),
            }
        )
    return moments


if __name__ == "__main__":
    run()
