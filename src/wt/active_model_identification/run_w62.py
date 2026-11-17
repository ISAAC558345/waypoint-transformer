"""Separate an extra wind measurement from a second sampling location.

archive ab: the onboard arm or the onboard-plus-remote arm leaves the W6.1 errors.

same latent once: the colocated sample draws another latent value, or one observation id is assimilated twice.

second on time: the second sample enters on its own beat, or it misses the remote sample's arrival.

fixed global control: equal control-field residuals produce different fixed-global forecasts.
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

from wt.active_model_identification.run_w5 import _draw_deltas, _ripple, posterior_mean
from wt.active_model_identification.run_w51 import _build, _fixed_sends, _kernel, _pairs, message_is_usable
from wt.active_model_identification.run_w6 import _measurement
from wt.active_model_identification.run_w61 import (
    _extend_tables,
    _hyp,
    _onboard_measurement,
    _onboard_noise,
    _run_arm,
    _subset,
)
from wt.active_model_identification.spec import ROOT

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w62.json"
W61_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w61.json"
W51_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w51.json"
RUN_ID = "w62_20261117"
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
    "onboard_noise_token",
    "conditional_token",
    "horizons_s",
    "prior_weight_global",
    "prior_weight_local",
)


def _parent() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_protocol() -> tuple[dict, str]:
    raw = CONFIG_PATH.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("status") != "registered_before_run":
        raise SystemExit("registration moved")
    archived = json.loads(W61_CONFIG_PATH.read_text(encoding="utf-8"))
    for key in SHARED_KEYS:
        if protocol[key] != archived[key]:
            raise RuntimeError(f"{key} left the W6.1 registration")
    if protocol["local_field"] != archived["local_field"]:
        raise RuntimeError("local field left the W6.1 registration")
    holdout = [int(seed) for seed in protocol["holdout_seeds"]]
    if len(holdout) != 4 or len(set(holdout)) != 4:
        raise RuntimeError("holdout seeds")
    if set(holdout) & {int(seed) for seed in protocol["local_field"]["seeds"]}:
        raise RuntimeError("holdout seed reuses an archived seed")
    if not protocol.get("holdout_noise_token"):
        raise RuntimeError("holdout noise token")
    return protocol, hashlib.sha256(raw).hexdigest()


def _remote_measurement(sample: dict, deltas) -> dict:
    measurement = _measurement(sample, deltas)
    measurement["source"] = "remote"
    return measurement


def _colocated(site: dict, sample: dict, deltas, ax: float, ay: float, sample_id: int, remote_noise: tuple[float, float] | None) -> dict:
    """Second measurement at the receiver. The latent value is the onboard one; the noise is the remote one."""
    key = (int(round(float(site["t"]))), 1)
    delta = (0.0, 0.0) if deltas is None else deltas[key]
    if remote_noise is None:
        noise = (float(sample["rx"]) - ax, float(sample["ry"]) - ay)
    else:
        noise = remote_noise
    return {
        "id": sample_id,
        "source": "remote",
        "x": float(site["x"]),
        "y": float(site["y"]),
        "t": float(site["t"]),
        "rx": ax + noise[0] + delta[0],
        "ry": ay + noise[1] + delta[1],
    }


def _holdout_noise(count: int, token: str, seed: int, sigma: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    onboard_rng = random.Random(f"{token}-{seed}-onboard")
    remote_rng = random.Random(f"{token}-{seed}-remote")
    onboard = [(onboard_rng.gauss(0.0, sigma), onboard_rng.gauss(0.0, sigma)) for _ in range(count)]
    remote = [(remote_rng.gauss(0.0, sigma), remote_rng.gauss(0.0, sigma)) for _ in range(count)]
    return onboard, remote


def _schedule(decisions, onboard: list[dict], second: list[dict], airtime: float) -> list[list[dict]]:
    arrivals: dict[float, list[dict]] = {}
    for measurement in onboard:
        arrivals.setdefault(float(measurement["t"]), []).append(measurement)
    for measurement in second:
        arrival = next(
            float(decision["t"])
            for decision in decisions
            if message_is_usable(float(measurement["t"]), float(decision["t"]), airtime)
        )
        if abs(arrival - float(measurement["t"])) <= 1e-9:
            raise RuntimeError("second sample arrived at its send decision")
        arrivals.setdefault(arrival, []).append(measurement)
    frames = []
    observations: list[dict] = []
    seen: set[int] = set()
    for decision in decisions:
        new_items = arrivals.get(float(decision["t"]), [])
        onboard_now = [item for item in new_items if item["source"] == "onboard"]
        second_now = [item for item in new_items if item["source"] == "remote"]
        for measurement in onboard_now + second_now:
            sample_id = int(measurement["id"])
            if sample_id in seen:
                raise RuntimeError("duplicate sample")
            seen.add(sample_id)
            observations.append(measurement)
        frames.append(list(observations))
    return frames


def _residuals(frame: list[dict]) -> tuple:
    return tuple(sorted((float(item["t"]), float(item["rx"]), float(item["ry"])) for item in frame))


def _fixed_global(decisions, frames: list[list[dict]], kernel: dict, case: dict, condition: dict) -> list[tuple[float, float]]:
    field = case["cfg"]["field"]
    winds = []
    for decision, frame in zip(decisions, frames):
        for horizon in decision["actual"]:
            future_t = float(decision["t"]) + int(horizon)
            predicted = decision["predicted"][int(horizon)]
            query = {"id": -1, "x": float(predicted[0]), "y": float(predicted[1]), "t": future_t}
            mean = posterior_mean(frame, query, kernel)
            ripple = _ripple(condition, field, predicted[0], predicted[1], future_t)
            winds.append((ripple[0] + mean[0], ripple[1] + mean[1]))
    return winds


def _same_receiver_latent(onboard: list[dict], colocated: list[dict], onboard_noise: list[tuple[float, float]], remote_noise: list[tuple[float, float]], ax: float, ay: float) -> None:
    for left, right, first_noise, second_noise in zip(onboard, colocated, onboard_noise, remote_noise, strict=True):
        first = (float(left["rx"]) - ax - first_noise[0], float(left["ry"]) - ay - first_noise[1])
        second = (float(right["rx"]) - ax - second_noise[0], float(right["ry"]) - ay - second_noise[1])
        if abs(first[0] - second[0]) > 1e-9 or abs(first[1] - second[1]) > 1e-9:
            raise RuntimeError("colocated sample drew a second latent")
        if abs(float(left["x"]) - float(right["x"])) > 1e-9 or abs(float(left["y"]) - float(right["y"])) > 1e-9:
            raise RuntimeError("colocated sample left the receiver")
        if abs(float(left["t"]) - float(right["t"])) > 1e-9 or int(left["id"]) == int(right["id"]):
            raise RuntimeError("colocated sample is not an independent observation")


def _check_pair(onboard: list[dict], remote: list[dict], colocated: list[dict], *, control: bool) -> None:
    if not (len(onboard) == len(remote) == len(colocated)):
        raise RuntimeError("sample count")
    gap = 0.0
    for left, middle, right in zip(onboard, remote, colocated):
        if int(left["id"]) == int(right["id"]) or int(middle["id"]) == int(right["id"]):
            raise RuntimeError("observation ids collided")
        if abs(float(left["t"]) - float(right["t"])) > 1e-9 or abs(float(left["x"]) - float(right["x"])) > 1e-9:
            raise RuntimeError("colocated sample left the receiver")
        if abs(float(left["y"]) - float(right["y"])) > 1e-9:
            raise RuntimeError("colocated sample left the receiver")
        gap = max(gap, _hyp(float(middle["x"]) - float(right["x"]), float(middle["y"]) - float(right["y"])))
        if control and (float(middle["rx"]) != float(right["rx"]) or float(middle["ry"]) != float(right["ry"])):
            raise RuntimeError("control residuals diverged")
    if gap <= 1.0:
        raise RuntimeError("the second location is not separated from the receiver")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _squared(error: tuple[float, float]) -> float:
    return error[0] ** 2 + error[1] ** 2


def _flight_pairs(field_name: str, seed: int | None, group: str, rows: dict[str, list[dict]]) -> list[dict]:
    index = {arm: {(float(row["t_s"]), int(row["horizon_s"])): row for row in arm_rows if row["airborne"]} for arm, arm_rows in rows.items()}
    pairs = []
    for key in index["A"]:
        row_a = index["A"][key]
        row_d = index["D"][key]
        row_b = index["B"][key]
        sq_a = _squared(row_a["error"])
        sq_d = _squared(row_d["error"])
        sq_b = _squared(row_b["error"])
        pairs.append(
            {
                "field": field_name,
                "seed": seed,
                "group": group,
                "t_s": row_a["t_s"],
                "horizon_s": row_a["horizon_s"],
                "n_second": row_b["n_remote"],
                "sq_a": sq_a,
                "sq_d": sq_d,
                "sq_b": sq_b,
                "paired_sq_d_minus_a": sq_d - sq_a,
                "paired_sq_b_minus_d": sq_b - sq_d,
                "weight_local_a": row_a["weight_local"],
                "weight_local_d": row_d["weight_local"],
                "weight_local_b": row_b["weight_local"],
            }
        )
    return pairs


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("empty average")
    return sum(values) / len(values)


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, digest = _load_protocol()
    out = out_dir or (ROOT / "results" / "wind_observation_w62" / RUN_ID)
    if "wind_observation_w62" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w62")
    out.mkdir(parents=True, exist_ok=True)
    archived_summary = json.loads((ROOT / str(protocol["w61_result"])).read_text(encoding="utf-8"))
    archived_scores = {
        (row["field"], None if row["seed"] is None else int(row["seed"]), row["subset"], int(row["horizon_s"])): row
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
        sites.append(
            {
                "t": float(sample["t"]),
                "x": float(decision["receiver"].x),
                "y": float(decision["receiver"].y),
                "remote_id": int(sample["id"]),
            }
        )
    _onboard_noise(sites, protocol["onboard_noise_token"], float(protocol["observation_sigma_m_s"]))
    separations = {
        int(round(site["t"])): _hyp(site["x"] - by_id[site["remote_id"]]["x"], site["y"] - by_id[site["remote_id"]]["y"])
        for site in sites
    }
    kernel_global = _kernel(protocol, 0.0)
    kernel_local = _kernel(protocol, protocol["local_field"]["amplitude_m_s"])
    kernels = {"global": kernel_global, "local": kernel_local}
    prior = {"global": protocol["prior_weight_global"], "local": protocol["prior_weight_local"]}
    old_seeds = [int(seed) for seed in protocol["local_field"]["seeds"]]
    holdout_seeds = [int(seed) for seed in protocol["holdout_seeds"]]
    deltas = _draw_deltas(points, kernel_local, old_seeds + holdout_seeds, float(protocol["local_field"]["generation_jitter"]))
    print("extend", flush=True)
    extended, _rows = _extend_tables(
        points,
        deltas,
        sites,
        kernel_local,
        float(protocol["local_field"]["generation_jitter"]),
        str(protocol["conditional_token"]),
    )
    ax = float(condition["ax"])
    ay = float(condition["ay"])
    sigma = float(protocol["observation_sigma_m_s"])
    print("forecast", flush=True)
    score_rows = []
    pair_rows = []
    mixture_gap = 0.0
    flight_mixture_gap = 0.0
    fixed_gap = 0.0
    first_t = float(decisions[0]["t"])
    holdout_noise = {
        seed: _holdout_noise(len(sites), str(protocol["holdout_noise_token"]), seed, sigma) for seed in holdout_seeds
    }
    fields = [("control", None, "archive", None)] + [("local", seed, "archive", extended[seed]) for seed in old_seeds]
    fields += [("local", seed, "holdout", extended[seed]) for seed in holdout_seeds]
    for field_name, seed, group, table in fields:
        if group == "holdout":
            onboard_draw, remote_draw = holdout_noise[int(seed)]
            onboard = []
            for index, site in enumerate(sites):
                site_copy = dict(site)
                site_copy["nx"], site_copy["ny"] = onboard_draw[index]
                onboard.append(_onboard_measurement(site_copy, table, ax, ay, 100000 + index))
            remote = []
            colocated = []
            for index, site in enumerate(sites):
                sample = by_id[site["remote_id"]]
                noise = remote_draw[index]
                delta_key = (int(sample["t"]), 0)
                delta = (0.0, 0.0) if table is None else table[delta_key]
                remote.append(
                    {
                        "id": int(sample["id"]),
                        "source": "remote",
                        "x": float(sample["x"]),
                        "y": float(sample["y"]),
                        "t": float(sample["t"]),
                        "rx": ax + noise[0] + delta[0],
                        "ry": ay + noise[1] + delta[1],
                    }
                )
                colocated.append(_colocated(site, sample, table, ax, ay, 200000 + index, noise))
        else:
            onboard = [_onboard_measurement(site, table, ax, ay, 100000 + index) for index, site in enumerate(sites)]
            remote = [_remote_measurement(by_id[int(send["id"])], table) for send in sends]
            colocated = [
                _colocated(site, by_id[site["remote_id"]], table, ax, ay, 200000 + index, None)
                for index, site in enumerate(sites)
            ]
        if group == "holdout":
            latent_onboard, latent_remote = holdout_noise[int(seed)]
        else:
            latent_onboard = [(float(site["nx"]), float(site["ny"])) for site in sites]
            latent_remote = [
                (float(by_id[site["remote_id"]]["rx"]) - ax, float(by_id[site["remote_id"]]["ry"]) - ay) for site in sites
            ]
        _same_receiver_latent(onboard, colocated, latent_onboard, latent_remote, ax, ay)
        _check_pair(onboard, remote, colocated, control=table is None)
        arm_rows = {}
        for arm, second in (("A", []), ("B", remote), ("D", colocated)):
            label = "A" if arm == "A" else "B"
            arm_rows[arm], signature = _run_arm(label, decisions, onboard, second, kernels, case, condition, table, airtime, separations, prior)
            identities = [item[1] for item in signature]
            if len(identities) != len(set(identities)):
                raise RuntimeError("an observation was assimilated twice")
        _check_clock(arm_rows, first_t)
        if field_name == "control":
            frames_b = _schedule(decisions, onboard, remote, airtime)
            frames_d = _schedule(decisions, onboard, colocated, airtime)
            for left, right in zip(frames_b, frames_d):
                if _residuals(left) != _residuals(right):
                    raise RuntimeError("control residuals diverged")
            winds_b = _fixed_global(decisions, frames_b, kernel_global, case, condition)
            winds_d = _fixed_global(decisions, frames_d, kernel_global, case, condition)
            for left, right in zip(winds_b, winds_d):
                fixed_gap = max(fixed_gap, abs(left[0] - right[0]), abs(left[1] - right[1]))
            for left, right in zip(arm_rows["B"], arm_rows["D"]):
                gap = max(abs(left["forecast"][0] - right["forecast"][0]), abs(left["forecast"][1] - right["forecast"][1]))
                mixture_gap = max(mixture_gap, gap)
                if left["airborne"]:
                    flight_mixture_gap = max(flight_mixture_gap, gap)
        if fixed_gap > 1e-6:
            raise RuntimeError("fixed global forecasts diverged on the control field")
        if group == "archive":
            for subset, flight_only in (("full", False), ("flight", True)):
                for horizon in (30, 60):
                    expected = archived_scores[(field_name, seed, subset, horizon)]
                    got_a, _count = _subset(arm_rows["A"], horizon, flight_only)
                    got_b, _count_b = _subset(arm_rows["B"], horizon, flight_only)
                    if abs(got_a - float(expected["rmse_a"])) > 1e-6 or abs(got_b - float(expected["rmse_b"])) > 1e-6:
                        raise RuntimeError(f"A/B left W6.1 {field_name} {seed} {subset} {horizon}")
        for subset, flight_only in (("full", False), ("flight", True)):
            for horizon in (30, 60):
                rmse_a, count = _subset(arm_rows["A"], horizon, flight_only)
                rmse_d, count_d = _subset(arm_rows["D"], horizon, flight_only)
                rmse_b, count_b = _subset(arm_rows["B"], horizon, flight_only)
                if count != count_d or count != count_b:
                    raise RuntimeError("arms scored different points")
                chosen = [row for row in _flight_pairs(field_name, seed, group, arm_rows) if int(row["horizon_s"]) == horizon] if flight_only else []
                mean_da = _mean([row["paired_sq_d_minus_a"] for row in chosen]) if chosen else None
                mean_bd = _mean([row["paired_sq_b_minus_d"] for row in chosen]) if chosen else None
                score_rows.append(
                    {
                        "field": field_name,
                        "seed": seed,
                        "group": group,
                        "subset": subset,
                        "horizon_s": horizon,
                        "n": count,
                        "rmse_a": rmse_a,
                        "rmse_d": rmse_d,
                        "rmse_b": rmse_b,
                        "paired_d_minus_a": rmse_d - rmse_a,
                        "paired_b_minus_d": rmse_b - rmse_d,
                        "paired_b_minus_a": rmse_b - rmse_a,
                        "mean_paired_sq_d_minus_a": mean_da,
                        "mean_paired_sq_b_minus_d": mean_bd,
                    }
                )
        pair_rows.extend(_flight_pairs(field_name, seed, group, arm_rows))
    if fixed_gap > 1e-6:
        raise RuntimeError("fixed global forecasts diverged on the control field")
    flight_times = [float(decision["t"]) for decision in decisions if decision["airborne"]]
    if flight_times != [float(item) for item in archived_summary["flight_t_s"]]:
        raise RuntimeError("flight decisions left W6.1")
    summary = {
        "schema_version": 1,
        "parent_commit": _parent(),
        "config_sha256": digest,
        "config_status": protocol["status"],
        "archive_match": True,
        "same_latent_once": True,
        "second_on_time": True,
        "fixed_global_match": True,
        "default_changed": False,
        "fixed_global_max_abs_m_s": fixed_gap,
        "control_mixture_max_abs_m_s": mixture_gap,
        "control_flight_mixture_max_abs_m_s": flight_mixture_gap,
        "n_archive_fields": 1 + len(old_seeds),
        "n_holdout_fields": len(holdout_seeds),
        "holdout_seeds": holdout_seeds,
        "n_flight_decisions": len(flight_times),
        "flight_t_s": flight_times,
        "payload_bytes": int(protocol["payload_bytes"]),
        "scores": score_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_csv(out / "scores.csv", score_rows)
    _write_csv(out / "pairs.csv", pair_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"flight {flight_times[0]}-{flight_times[-1]} holdout {len(holdout_seeds)} "
        f"fixed {fixed_gap:.3e} mixture {mixture_gap:.3e}",
        flush=True,
    )
    return summary


def _check_clock(arm_rows: dict[str, list[dict]], first_t: float) -> None:
    index = {arm: {(float(row["t_s"]), int(row["horizon_s"])): row for row in rows} for arm, rows in arm_rows.items()}
    for key, row_b in index["B"].items():
        row_d = index["D"][key]
        row_a = index["A"][key]
        if row_d["remote_times"] != row_b["remote_times"] or row_d["onboard_times"] != row_a["onboard_times"]:
            raise RuntimeError("second sample left the remote arrival clock")
        if row_d["n_remote"] != row_b["n_remote"] or row_a["n_remote"] != 0:
            raise RuntimeError("second sample count")
        if abs(float(key[0]) - first_t) <= 1e-9 and (row_d["n_remote"] != 0 or row_b["n_remote"] != 0 or row_a["n_onboard"] != 1):
            raise RuntimeError("second sample was used before it arrived")


if __name__ == "__main__":
    run()
