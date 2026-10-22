"""Write the HGNN-input revision. Pilot first; full replay only after the pilot passes."""
from __future__ import annotations

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import (
    DIAGNOSTIC_TIME,
    REVISION_NAME,
    REVISION_ROOT,
    audit_scenario,
    replay_scenario,
    revision_complete,
    scenarios_to_revise,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import OPEN_POOL, PRIMARY_TIMES


WORKERS = 16


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _pilot_checks(replay: dict, audit: dict) -> dict:
    config = L3BBitQueueEnvV2().config
    limit = float(config.step_length_m) + 1.0e-8
    trace = {int(row["time_step"]): row for row in replay["trace"]}
    failures = []
    if int(audit["state_count"]) != len(PRIMARY_TIMES):
        failures.append("state count")
    for row in audit["states"]:
        if int(row["dimension_match"]) != 1 or int(row["finite"]) != 1 or float(row["j15_max_abs_error_vs_frozen"]) != 0.0:
            failures.append(f"state {row['time_step']} schema")
        if float(row["uav_step_displacement_stats"]["max"]) > limit or float(row["uav_step_displacement_stats"]["min"]) < -limit:
            failures.append(f"state {row['time_step']} displacement exceeds one step")
    saved = {int(row["time_step"]): row for row in audit["states"]}
    if saved[0]["velocity_predecessor"] != "reset_pose_before_any_step" or saved[0]["predecessor_time_step"] is not None:
        failures.append("t=0 predecessor")
    if float(saved[0]["uav_step_displacement_stats"]["max"]) != 0.0 or float(saved[0]["uav_step_displacement_stats"]["min"]) != 0.0:
        failures.append("t=0 displacement")
    for time_step in (15, 30, DIAGNOSTIC_TIME):
        row = trace.get(int(time_step))
        if row is None or int(row["predecessor_time_step"]) != int(time_step) - 1:
            failures.append(f"t={time_step} predecessor")
        elif float(row["max_displacement_m"]) > limit:
            failures.append(f"t={time_step} step length")
    progress_rows = [row["service_progress"] for row in replay["trace"]]
    progress = np.asarray(progress_rows, dtype=np.float64)
    if progress.ndim != 2 or not np.all(np.isfinite(progress)) or float(np.min(progress)) < -1.0e-12 or float(np.max(progress)) > 1.0:
        failures.append("service_progress range")
    changed = int(np.any(np.diff(progress, axis=0) != 0.0)) if progress.shape[0] > 1 else 0
    if changed != 1:
        failures.append("service_progress did not change across consecutive steps")
    chords = {}
    for left, right in ((0, 15), (15, 30)):
        step_norm = float(trace[right]["max_displacement_m"])
        chord = float(np.max(np.abs(np.asarray(trace[right]["service_progress"]) - np.asarray(trace[left]["service_progress"]))))
        chords[f"{left}_{right}_progress_abs"] = chord
        chords[f"{right}_one_step_max_m"] = step_norm
        if int(trace[right]["predecessor_time_step"]) != right - 1:
            failures.append(f"t={right} is not a one-step predecessor")
    return {"passed": int(len(failures) == 0), "failures": failures, "progress_changed": changed, "checks": chords, "trace_steps": len(trace)}


def run_pilot() -> dict:
    replay = replay_scenario(322000, trace_until=DIAGNOSTIC_TIME)
    audit = audit_scenario(322000)
    verdict = _pilot_checks(replay, audit)
    payload = {"revision": REVISION_NAME, "replay_states": [row["time_step"] for row in replay["states"]], "audit": audit, "verdict": verdict, "trace": replay["trace"]}
    _write(REVISION_ROOT / "pilot_322000_audit.json", payload)
    print(f"pilot passed={verdict['passed']} failures={verdict['failures']}", flush=True)
    return payload


def run_full() -> list[dict]:
    pending = scenarios_to_revise()
    if not pending:
        print("revision already complete", flush=True)
        return []
    done = []
    pool = ProcessPoolExecutor(max_workers=WORKERS)
    try:
        futures = {pool.submit(replay_scenario, scenario): scenario for scenario in pending}
        for future in as_completed(futures):
            scenario = futures[future]
            result = future.result()
            done.append({"scenario_id": int(scenario), "states": [int(row["time_step"]) for row in result["states"]]})
            print(f"revised {scenario} {len(done)}/{len(pending)}", flush=True)
    except Exception:
        (REVISION_ROOT / "revision_worker_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except OSError:
            pass
        raise
    else:
        pool.shutdown(wait=True)
    missing = [scenario for scenario in OPEN_POOL if not revision_complete(scenario)]
    if missing:
        raise RuntimeError(f"revision incomplete: {missing[:8]}")
    return done


if __name__ == "__main__":
    import sys

    if "--full" in sys.argv:
        run_full()
    else:
        run_pilot()
