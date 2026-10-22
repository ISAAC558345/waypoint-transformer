"""Join Phase-5E labels to the same decision states. No new scenario is generated."""
from __future__ import annotations

import csv

import numpy as np

from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch, observable_exclusive_assignment
from wt.lowlevel_v2.phase5e_utility_dataset.build import _legal_buoys
from wt.lowlevel_v2.phase5e_utility_dataset.state import candidate_features, current_observation

from .protocol import FORBIDDEN_INPUTS, LABELS, MODEL_INPUT_KEYS, assert_phase5f_seed


def load_label_table() -> dict[tuple[int, int, int, int], dict[str, float]]:
    if not LABELS.is_file():
        raise FileNotFoundError(str(LABELS))
    table = {}
    with LABELS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_phase5f_seed(scenario)
            key = (scenario, int(row["time_step"]), int(row["uav_id"]), int(row["candidate_id"]))
            table[key] = {
                "queue": float(row["queue"]),
                "channel": float(row["channel"]),
                "distance": float(row["distance"]),
                "ETA": float(row["ETA"]),
                "interference": float(row["interference"]),
                "urgency": float(row["urgency"]),
                "payload_gain": float(row["payload_gain"]),
                "utility_rank": float(row["utility_rank"]),
            }
    return table


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1.0e-5, atol=1.0e-4))


def _advance(env: L3BEnv, target: int, last_positions: np.ndarray | None, last_arrival: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    while int(env.t) < int(target):
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        assignment = observable_exclusive_assignment(env)
        info = env.step(fly_toward(env, assignment), assignment)
        last_positions = positions
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
    if int(env.t) != int(target):
        raise RuntimeError(f"phase-5f stopped at t={env.t}, expected {target}")
    return last_positions, last_arrival


def collect_scenario(seed: int, times: tuple[int, ...], labels: dict) -> list[dict]:
    assert_phase5f_seed(int(seed))
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_positions = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    rows = []
    for moment in times:
        last_positions, last_arrival = _advance(env, int(moment), last_positions, last_arrival)
        info = env.info()
        obs = current_observation(env, last_positions)
        base = observable_exclusive_assignment(env)
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        buoys = np.asarray(env.buoy_positions, dtype=np.float64).copy()
        graph_last = positions if last_positions is None else np.asarray(last_positions, dtype=np.float32)
        batch = graph_batch(info, env, last_arrival, graph_last, urgency=None)
        urgency_batch = graph_batch(info, env, last_arrival, graph_last, urgency=obs["urgency"])
        _assert_inputs(batch)
        _assert_inputs(urgency_batch)
        for focal in range(int(env.config.num_uavs)):
            legal = _legal_buoys(base, focal, int(env.config.num_buoys))
            ranks = []
            payloads = []
            assignment_scores = []
            for buoy in legal:
                key = (int(seed), int(moment), int(focal), int(buoy))
                if key not in labels:
                    raise RuntimeError(f"phase-5e label missing for {key}")
                label = labels[key]
                features = candidate_features(obs, focal, buoy)
                for name in ("queue", "channel", "distance", "ETA", "interference", "urgency"):
                    if not _close(float(features[name]), float(label[name])):
                        raise RuntimeError(f"replay drifted from phase-5e on {key} {name}")
                trial = np.asarray(base, dtype=np.int64).copy()
                trial[focal] = int(buoy)
                payload = float(shadow_h15_payload(env, trial))
                if abs(payload - float(label["payload_gain"])) > 1.0:
                    raise RuntimeError(f"shadow payload drifted for {key}")
                ranks.append(int(label["utility_rank"]))
                payloads.append(payload)
                assignment_scores.append(float(features["queue"]) * float(features["channel"]) / (float(features["distance"]) + 1.0))
            rows.append(
                {
                    "scenario_id": int(seed),
                    "time_step": int(moment),
                    "uav_id": int(focal),
                    "batch": batch,
                    "urgency_batch": urgency_batch,
                    "uav_positions": positions,
                    "buoy_positions": buoys,
                    "context": np.asarray(base, dtype=np.int64).copy(),
                    "candidate_ids": [int(item) for item in legal],
                    "utility_rank": ranks,
                    "payload_gain": payloads,
                    "assignment_score": assignment_scores,
                }
            )
    return rows


def _assert_inputs(batch: dict) -> None:
    keys = set(batch)
    leaked = keys.intersection(FORBIDDEN_INPUTS)
    if leaked:
        raise RuntimeError(f"model input contains label fields: {sorted(leaked)}")
    missing = [key for key in MODEL_INPUT_KEYS if key not in batch]
    if missing:
        raise RuntimeError(f"model input missing {missing}")


def load_splits() -> dict[str, list[dict]]:
    labels = load_label_table()
    from .protocol import TEST_SCENARIO, TEST_TIMES, TRAIN_SCENARIO, TRAIN_TIMES, VAL_TIMES

    return {
        "train": collect_scenario(TRAIN_SCENARIO, TRAIN_TIMES, labels),
        "validation": collect_scenario(TRAIN_SCENARIO, VAL_TIMES, labels),
        "test": collect_scenario(TEST_SCENARIO, TEST_TIMES, labels),
    }


__all__ = ["collect_scenario", "load_label_table", "load_splits"]
