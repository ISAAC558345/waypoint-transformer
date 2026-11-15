"""Ask whether arrived wind samples identify the global or local model.

fixed model archive: a fixed global or fixed local forecast leaves the W5.1 early-send errors.

message once: the arriving sample is scored with itself already in the posterior, or the same id is appended twice.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from wt.active_model_identification.run_w5 import _draw_deltas, _ripple, _rmse, posterior_mean
from wt.active_model_identification.run_w51 import (
    _build,
    _fixed_sends,
    _kernel,
    _latent_variances,
    _pairs,
    message_is_usable,
)
from wt.active_model_identification.spec import ROOT

CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w6.json"
W5_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w5.json"
W51_CONFIG_PATH = ROOT / "configs" / "experiments" / "wind_observation_w51.json"
W51_SUMMARY_PATH = ROOT / "results" / "wind_observation_w51" / "w51_20261115" / "summary.json"
RUN_ID = "w6_20261115"
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


def _normalize(log_weights: dict[str, float]) -> dict[str, float]:
    peak = max(log_weights.values())
    raw = {name: math.exp(value - peak) for name, value in log_weights.items()}
    scale = sum(raw.values())
    return {name: value / scale for name, value in raw.items()}


def _predict(observations: list[dict], sample: dict, kernel: dict) -> tuple[tuple[float, float], float]:
    """Measurement law from samples already assimilated. This sample is not one of them."""
    if any(int(item["id"]) == int(sample["id"]) for item in observations):
        raise RuntimeError("sample already assimilated")
    target = {"id": -1, "x": float(sample["x"]), "y": float(sample["y"]), "t": float(sample["t"])}
    mean = posterior_mean(observations, target, kernel)
    variance = _latent_variances(observations, [target], kernel)[0] + float(kernel["sigma_m_s"]) ** 2
    if variance <= 1e-12:
        raise RuntimeError("predictive variance")
    return mean, variance


def _loglik(sample: dict, mean: tuple[float, float], variance: float) -> float:
    total = 0.0
    for value, center in ((float(sample["rx"]), mean[0]), (float(sample["ry"]), mean[1])):
        total += -0.5 * (math.log(2.0 * math.pi * variance) + (value - center) ** 2 / variance)
    return total


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
    if abs(float(protocol["prior_weight_global"]) + float(protocol["prior_weight_local"]) - 1.0) > 1e-12:
        raise RuntimeError("prior weights")
    return protocol, archived, hashlib.sha256(raw).hexdigest()


def _measurement(sample: dict, deltas) -> dict:
    delta = (0.0, 0.0) if deltas is None else deltas[(int(sample["t"]), 0)]
    return {
        "id": int(sample["id"]),
        "x": float(sample["x"]),
        "y": float(sample["y"]),
        "t": float(sample["t"]),
        "rx": float(sample["rx"]) + delta[0],
        "ry": float(sample["ry"]) + delta[1],
    }


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


def _run_field(sends, decisions, samples, case, condition, deltas, kernel_global, kernel_local, airtime, prior) -> tuple[list[dict], list[dict]]:
    by_id = {int(sample["id"]): sample for sample in samples}
    arrivals: dict[float, list[dict]] = {}
    for send in sends:
        arrival = next(float(decision["t"]) for decision in decisions if message_is_usable(float(send["t"]), float(decision["t"]), airtime))
        if abs(arrival - float(send["t"])) <= 1e-9:
            raise RuntimeError("message arrived at its send decision")
        arrivals.setdefault(arrival, []).append(by_id[int(send["id"])])
    observations: list[dict] = []
    seen: set[int] = set()
    log_weights = {name: math.log(float(prior[name])) for name in ("global", "local")}
    field = case["cfg"]["field"]
    weight_rows = []
    point_rows = []
    for decision in decisions:
        for sample in arrivals.get(float(decision["t"]), []):
            measurement = _measurement(sample, deltas)
            if int(measurement["id"]) in seen:
                raise RuntimeError("duplicate message")
            before = _normalize(log_weights)
            predictions = {}
            likelihoods = {}
            for name, kernel in (("global", kernel_global), ("local", kernel_local)):
                mean, variance = _predict(observations, measurement, kernel)
                predictions[name] = mean
                likelihoods[name] = _loglik(measurement, mean, variance)
            for name in log_weights:
                log_weights[name] += likelihoods[name]
            after = _normalize(log_weights)
            mixed = (
                before["global"] * predictions["global"][0] + before["local"] * predictions["local"][0],
                before["global"] * predictions["global"][1] + before["local"] * predictions["local"][1],
            )
            weight_rows.append(
                {
                    "arrival_t_s": float(decision["t"]),
                    "sample_t_s": float(measurement["t"]),
                    "sample_x_m": float(measurement["x"]),
                    "sample_y_m": float(measurement["y"]),
                    "n_before": len(observations),
                    "n_after": len(observations) + 1,
                    "weight_global_before": before["global"],
                    "weight_local_before": before["local"],
                    "weight_global_after": after["global"],
                    "weight_local_after": after["local"],
                    "pred_error_global_m_s": _hyp(measurement["rx"] - predictions["global"][0], measurement["ry"] - predictions["global"][1]),
                    "pred_error_local_m_s": _hyp(measurement["rx"] - predictions["local"][0], measurement["ry"] - predictions["local"][1]),
                    "pred_error_weighted_m_s": _hyp(measurement["rx"] - mixed[0], measurement["ry"] - mixed[1]),
                }
            )
            seen.add(int(measurement["id"]))
            observations.append(measurement)
        weights = _normalize(log_weights)
        for horizon, actual in decision["actual"].items():
            future_t = float(decision["t"]) + int(horizon)
            predicted = decision["predicted"][int(horizon)]
            query = {"id": -1, "x": float(predicted[0]), "y": float(predicted[1]), "t": future_t}
            ripple_predicted = _ripple(condition, field, predicted[0], predicted[1], future_t)
            ripple_actual = _ripple(condition, field, actual[0], actual[1], future_t)
            delta = (0.0, 0.0) if deltas is None else deltas[(int(future_t), 1)]
            truth = (
                float(condition["ax"]) + ripple_actual[0] + delta[0],
                float(condition["ay"]) + ripple_actual[1] + delta[1],
            )
            mean_global = posterior_mean(observations, query, kernel_global)
            mean_local = posterior_mean(observations, query, kernel_local)
            wind_global = (ripple_predicted[0] + mean_global[0], ripple_predicted[1] + mean_global[1])
            wind_local = (ripple_predicted[0] + mean_local[0], ripple_predicted[1] + mean_local[1])
            wind_online = (
                weights["global"] * wind_global[0] + weights["local"] * wind_local[0],
                weights["global"] * wind_global[1] + weights["local"] * wind_local[1],
            )
            point_rows.append(
                {
                    "t_s": float(decision["t"]),
                    "horizon_s": int(horizon),
                    "airborne": bool(decision["airborne"]),
                    "n_received": len(observations),
                    "weight_global": weights["global"],
                    "weight_local": weights["local"],
                    "error_global": (wind_global[0] - truth[0], wind_global[1] - truth[1]),
                    "error_local": (wind_local[0] - truth[0], wind_local[1] - truth[1]),
                    "error_online": (wind_online[0] - truth[0], wind_online[1] - truth[1]),
                }
            )
    if len(seen) != len(sends):
        raise RuntimeError("message count")
    return point_rows, weight_rows


def _subset(rows: list[dict], horizon: int, key: str, flight_only: bool) -> tuple[float, int]:
    errors = [row[key] for row in rows if int(row["horizon_s"]) == horizon and (row["airborne"] or not flight_only)]
    if not errors:
        raise RuntimeError("empty score set")
    return _rmse(errors), len(errors)


def _snapshot(rows: list[dict], predicate) -> dict | None:
    chosen = [row for row in rows if predicate(row) and int(row["horizon_s"]) == 30]
    if not chosen:
        return None
    row = chosen[-1]
    return {"t_s": row["t_s"], "n_received": row["n_received"], "weight_global": row["weight_global"], "weight_local": row["weight_local"]}


def run(out_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    protocol, _archived, digest = _load_protocol()
    out = out_dir or (ROOT / "results" / "wind_observation_w6" / RUN_ID)
    if "wind_observation_w6" not in set(out.parts):
        raise SystemExit("refusing to write outside results/wind_observation_w6")
    out.mkdir(parents=True, exist_ok=True)
    archived_summary = json.loads(W51_SUMMARY_PATH.read_text(encoding="utf-8"))
    archived_scores = {
        (row["field"], None if row["seed"] is None else int(row["seed"]), row["subset"], int(row["horizon_s"])): float(row["rmse_early"])
        for row in archived_summary["scores"]
    }
    print("replay", flush=True)
    w51 = json.loads(W51_CONFIG_PATH.read_text(encoding="utf-8"))
    case, condition, samples, decisions, points, airtime = _build(w51)
    sends = _fixed_sends(_pairs(decisions), samples, "early")
    if len(sends) != 30:
        raise RuntimeError("send budget")
    kernel_global = _kernel(protocol, 0.0)
    kernel_local = _kernel(protocol, protocol["local_field"]["amplitude_m_s"])
    deltas = _draw_deltas(
        points,
        kernel_local,
        [int(seed) for seed in protocol["local_field"]["seeds"]],
        float(protocol["local_field"]["generation_jitter"]),
    )
    prior = {"global": protocol["prior_weight_global"], "local": protocol["prior_weight_local"]}
    fields = [("control", None, None)] + [("local", int(seed), table) for seed, table in deltas.items()]
    print("weights", flush=True)
    score_rows = []
    weight_rows = []
    snapshots = []
    sent_samples = [next(sample for sample in samples if int(sample["id"]) == int(send["id"])) for send in sends]
    span = 0.0
    for left in sent_samples:
        for right in sent_samples:
            span = max(span, math.hypot(left["x"] - right["x"], left["y"] - right["y"]))
    for field_name, seed, table in fields:
        points_field, weights = _run_field(sends, decisions, samples, case, condition, table, kernel_global, kernel_local, airtime, prior)
        for row in weights:
            weight_rows.append({"field": field_name, "seed": seed, **row})
        true_key = "error_global" if field_name == "control" else "error_local"
        for subset, flight_only in (("full", False), ("flight", True)):
            for horizon in (30, 60):
                global_rmse, count = _subset(points_field, horizon, "error_global", flight_only)
                local_rmse, _local_count = _subset(points_field, horizon, "error_local", flight_only)
                online_rmse, _online_count = _subset(points_field, horizon, "error_online", flight_only)
                expected_key = "error_global" if field_name == "control" else "error_local"
                expected = archived_scores[(field_name, seed, subset, horizon)]
                matched = global_rmse if expected_key == "error_global" else local_rmse
                if abs(matched - expected) > 1e-6:
                    raise RuntimeError(f"fixed model left W5.1 {field_name} {seed} {subset} {horizon}: {matched} vs {expected}")
                score_rows.append(
                    {
                        "field": field_name,
                        "seed": seed,
                        "subset": subset,
                        "horizon_s": horizon,
                        "n": count,
                        "true_model": "global" if field_name == "control" else "local",
                        "rmse_fixed_global": global_rmse,
                        "rmse_fixed_local": local_rmse,
                        "rmse_online": online_rmse,
                        "rmse_true_model": global_rmse if true_key == "error_global" else local_rmse,
                        "paired_online_minus_global": online_rmse - global_rmse,
                        "paired_online_minus_local": online_rmse - local_rmse,
                        "paired_online_minus_true": online_rmse - (global_rmse if true_key == "error_global" else local_rmse),
                    }
                )
        flight_rows = [row for row in points_field if row["airborne"]]
        end_rows = [row for row in points_field if float(row["t_s"]) == float(decisions[-1]["t"])]
        snapshots.append(
            {
                "field": field_name,
                "seed": seed,
                "true_model": "global" if field_name == "control" else "local",
                "last_flight": _snapshot(flight_rows, lambda row: True),
                "mission_end": _snapshot(end_rows, lambda row: True),
            }
        )
    flight_times = [float(decision["t"]) for decision in decisions if decision["airborne"]]
    if len(flight_times) != int(archived_summary["n_flight_decisions"]):
        raise RuntimeError("flight decisions left W5.1")
    summary = {
        "schema_version": 1,
        "parent_commit": _parent(),
        "config_sha256": digest,
        "config_status": protocol["status"],
        "archive_match": True,
        "message_once": True,
        "default_changed": False,
        "n_messages": len(sends),
        "payload_bytes": int(protocol["payload_bytes"]),
        "n_flight_decisions": len(flight_times),
        "flight_t_s": flight_times,
        "last_flight_t_s": flight_times[-1],
        "sent_sample_span_m": span,
        "snapshots": snapshots,
        "scores": score_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_csv(out / "scores.csv", score_rows)
    _write_csv(out / "weights.csv", weight_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"flight {flight_times[0]}-{flight_times[-1]} messages {len(sends)} span {span:.3f}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
