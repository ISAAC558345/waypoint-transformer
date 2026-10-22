"""One-shot EXEC first-hold evaluation labels. Writes only to Phase-4C outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from wt.htw_phase945.candidate_pool import assignment_hash, generate_pool
from wt.lowlevel_v2.htw_phase4b_b2_shadow.shadow_scorer import extract_candidate_geometry
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker
from wt.lowlevel_v2.integration_phase3.protocol import exec_ckpt
from wt.lowlevel_v2.phase1.protocol import load_pools
from wt.lowlevel_v2.phase4b.snapshot import snapshot_env
from wt.lowlevel_v2.phase4c_independent_validation.protocol import ALIGN_ATOL, OUTPUT, P3E_CAND_KEYS
from wt.lowlevel_v2.traj_audit.protocol import LL_RUN, LOW_SEEDS
from wt.lowlevel_v2.traj_audit.run_audit import (
    auditor_env_reset,
    load_high,
    make_override_stack,
    run_episode,
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _state_key(state: dict, assignment: np.ndarray) -> str:
    env = state["env"]
    raw = np.asarray(env["uav_positions"], dtype=np.float64).tobytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"{int(state['scenario'])}|{int(state['t'])}|{float(env['payload_bits'])}|{digest}|{assignment_hash(assignment)}"


def reset_state(scenario: int) -> dict:
    env = auditor_env_reset(int(scenario))
    return {
        "scenario": int(scenario),
        "t": int(env.t),
        "env": snapshot_env(env),
        "last_arrival": np.zeros(env.config.num_buoys, dtype=np.float32),
        "last_positions": np.asarray(env.uav_positions, dtype=np.float32).copy(),
        "last_movement": np.zeros((env.config.num_uavs, 2), dtype=np.float64),
    }


class Phase4CLabeler:
    """5-seed U_single cache lives only under Phase-4C outputs."""

    def __init__(self) -> None:
        self.shadows = {int(s): load_ll_worker(int(s)) for s in LOW_SEEDS}
        self.cache_path = OUTPUT / "ull_cache.json"
        self.ull_cache: dict[str, float] = {}
        if self.cache_path.is_file():
            self.ull_cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def save_cache(self) -> None:
        _dump(self.cache_path, self.ull_cache)

    def u_ll(self, state: dict, assignment: np.ndarray) -> float:
        key = _state_key(state, assignment) + f"|n{len(LOW_SEEDS)}"
        if key in self.ull_cache:
            return float(self.ull_cache[key])
        vals = [
            float(shadow_h15(self.shadows[int(s)], state, assignment, zero_residual=False)["payload_increment"])
            for s in LOW_SEEDS
        ]
        value = float(np.mean(vals))
        self.ull_cache[key] = value
        if len(self.ull_cache) % 8 == 0:
            self.save_cache()
        return value


def load_existing_labels(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        item = dict(row)
        for key in ("scenario_id", "candidate_id"):
            item[key] = int(float(item[key]))
        for key in ("U_single", "U_traj", "C_abs") + P3E_CAND_KEYS:
            item[key] = float(item[key])
        out.append(item)
    return out


def generate_exec_labels(scenarios: list[int]) -> list[dict]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "independent_eval_labels.csv"
    rows = load_existing_labels(path)
    done = {(int(r["scenario_id"]), int(r["candidate_id"])) for r in rows}
    pending = [sc for sc in scenarios if any((sc, i) not in done for i in range(24))]
    if not pending:
        return rows
    ckpt = exec_ckpt(0)
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing EXEC high-level checkpoint: {ckpt}")
    labeler = Phase4CLabeler()
    high = load_high(ckpt)
    worker = make_override_stack(high, LL_RUN)
    for scenario in scenarios:
        need = [(scenario, i) for i in range(24) if (scenario, i) not in done]
        if not need:
            continue
        env = auditor_env_reset(int(scenario))
        pool = generate_pool(env, seed_key=int(scenario) * 1009 + 0)
        if len(pool) != 24:
            raise RuntimeError(f"pool size {len(pool)} != 24 for scenario {scenario}")
        state0 = reset_state(int(scenario))
        for index, gamma in enumerate(pool):
            if (scenario, index) in done:
                continue
            u_single = labeler.u_ll(state0, gamma)
            worker.set_first_assignment(gamma)
            rec = run_episode(worker, int(scenario), labeler, compute_ull=False)
            c_abs = float(rec["U_traj"]) - float(u_single)
            if abs(c_abs - (float(rec["U_traj"]) - float(u_single))) > ALIGN_ATOL:
                raise RuntimeError("C_abs identity failed")
            geom = extract_candidate_geometry(env, gamma)
            row = {
                "family": "EXEC",
                "scenario_id": int(scenario),
                "state_id": int(scenario),
                "candidate_id": int(index),
                "U_single": float(u_single),
                "U_traj": float(rec["U_traj"]),
                "C_abs": float(c_abs),
                "assignment_hash": assignment_hash(gamma),
                "gamma": json.dumps([int(x) for x in np.asarray(gamma, dtype=np.int64)[:6].tolist()]),
                "n_hold": int(rec["n_hold"]),
                "ll_run": int(LL_RUN),
                "high_family": "EXEC",
                "high_seed": 0,
                **geom,
            }
            rows.append(row)
            done.add((scenario, index))
            print(
                f"label EXEC sc={scenario} cand={index + 1}/24 U_single={u_single:.1f} U_traj={rec['U_traj']:.1f}",
                flush=True,
            )
            if len(rows) % 4 == 0:
                _csv_write(path, rows)
                labeler.save_cache()
        _csv_write(path, rows)
        labeler.save_cache()
    del worker
    _csv_write(path, rows)
    labeler.save_cache()
    return rows


def unseen_train_scenarios() -> list[int]:
    train, design = load_pools()
    used = set(int(s) for s in design)
    return [int(s) for s in train if int(s) not in used]


__all__ = ["generate_exec_labels", "load_existing_labels", "unseen_train_scenarios"]
