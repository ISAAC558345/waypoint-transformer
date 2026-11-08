"""Assemble the 28 existing training states. This module does not step or score."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wt.htw_phase9.const import EDGE_UB_DIM
from wt.legacy_arrival_aware_ab.loader import TRAIN_SCENARIOS, load_package, resolve_graph


GRAPH_KEYS = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
L8_SCENARIOS = tuple(range(322024, 322032))
COMPARISON_SCENARIOS = tuple(range(322032, 322040))
EXCLUDED = (322022, 322023)
TIMES = (15, 30)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_graph(graph: dict, label: str) -> dict:
    if set(graph) != set(GRAPH_KEYS):
        raise RuntimeError(f"{label} graph keys are {tuple(graph)}")
    ordered = {key: np.asarray(graph[key]) for key in GRAPH_KEYS}
    edge = ordered["edge"]
    if int(edge.shape[-1]) != int(EDGE_UB_DIM):
        raise RuntimeError(f"{label} edge width is {edge.shape[-1]}")
    for key, array in ordered.items():
        if str(key).endswith("mask"):
            continue
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{label} graph field {key} is non-finite")
    return ordered


def _candidate_block(rows: list[dict], payload: np.ndarray, label: str) -> dict:
    if len(rows) != 128 or int(payload.shape[0]) != 128:
        raise RuntimeError(f"{label} does not have 128 aligned candidates")
    local = [int(row["local_position"]) for row in rows]
    if local != list(range(128)):
        raise RuntimeError(f"{label} local order is not 0..127")
    identities = [int(row["candidate_id"]) for row in rows]
    if len(set(identities)) != 128:
        raise RuntimeError(f"{label} candidate ids are not unique")
    if identities == local:
        raise RuntimeError(f"{label} collapsed candidate ids onto local positions")
    assignments = []
    for row in rows:
        parts = str(row["assignment"]).split("|")
        if len(parts) != 6:
            raise RuntimeError(f"{label} assignment does not describe six UAVs")
        assignments.append([int(part) for part in parts])
    if not np.all(np.isfinite(payload)):
        raise RuntimeError(f"{label} payload is non-finite")
    return {
        "assignments": np.asarray(assignments, dtype=np.int64),
        "candidate_id": np.asarray(identities, dtype=np.int64),
        "row_index": np.asarray([int(row["row_index"]) for row in rows], dtype=np.int64),
        "local_position": np.asarray(local, dtype=np.int64),
        "assignment_text": [str(row["assignment"]) for row in rows],
        "real_payload": np.asarray(payload, dtype=np.float64),
    }


def _l2_states(repo: Path) -> tuple[list[dict], list[dict]]:
    package = repo / "results" / "legacy_arrival_aware_ab_l2"
    loaded = load_package(package, repo)
    expected = [(scenario, time_step) for scenario in TRAIN_SCENARIOS for time_step in TIMES]
    found = [(int(state["scenario_id"]), int(state["time_step"])) for state in loaded["train"]]
    if found != expected:
        raise RuntimeError("L2 training order changed")
    states = []
    sources = []
    for state in loaded["train"]:
        label = f"L2 {int(state['scenario_id'])} t={int(state['time_step'])}"
        graph = _finite_graph(state["graph"], label)
        rows = [
            {
                "local_position": int(state["local_position"][index]),
                "candidate_id": int(state["candidate_id"][index]),
                "row_index": int(state["row_index"][index]),
                "assignment": state["assignment_text"][index],
            }
            for index in range(128)
        ]
        block = _candidate_block(rows, state["real_payload"], label)
        graph_path = resolve_graph(package, state["graph_file"], repo)
        states.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "source": "l2",
                "graph": graph,
                **block,
            }
        )
        sources.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "source": "results/legacy_arrival_aware_ab_l2",
                "graph_file": state["graph_file"].replace("\\", "/"),
                "graph_sha256": _sha256(graph_path),
            }
        )
    return states, sources


def _l8_states(repo: Path) -> tuple[list[dict], list[dict]]:
    root = repo / "results" / "legacy_payload_objective_l8"
    payload = json.loads((root / "candidates.json").read_text(encoding="utf-8"))
    labels: dict[tuple[int, int, int, str], float] = {}
    for line in (root / "real_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["scenario_id"]), int(row["time_step"]), int(row["candidate_id"]), str(row["assignment"]))
        if key in labels:
            raise RuntimeError(f"L8 label key repeats: {key}")
        labels[key] = float(row["real_payload_bits"])
    grouped = {(int(state["scenario_id"]), int(state["time_step"])): state for state in payload["states"]}
    expected = [(scenario, time_step) for scenario in L8_SCENARIOS for time_step in TIMES]
    if sorted(grouped) != expected:
        raise RuntimeError("L8 candidate states are not the 16 fixed states")
    states = []
    sources = []
    for scenario, time_step in expected:
        item = grouped[(scenario, time_step)]
        label = f"L8 {scenario} t={time_step}"
        values = []
        for row in item["rows"]:
            key = (scenario, time_step, int(row["candidate_id"]), str(row["assignment"]))
            if key not in labels:
                raise RuntimeError(f"{label} candidate {key} has no real label")
            values.append(labels[key])
        graph_path = root / "comparison" / f"scenario_{scenario}" / f"t{time_step:03d}" / "graph.npz"
        meta_path = graph_path.with_name("state_meta.json")
        with np.load(graph_path) as stored:
            graph = {key: np.asarray(stored[key]) for key in stored.files}
        graph = _finite_graph(graph, label)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta["state_hash"]) != str(item["state_hash"]):
            raise RuntimeError(f"{label} state hash does not match its graph metadata")
        block = _candidate_block(item["rows"], np.asarray(values, dtype=np.float64), label)
        states.append(
            {
                "scenario_id": scenario,
                "time_step": time_step,
                "source": "l8",
                "graph": graph,
                **block,
            }
        )
        sources.append(
            {
                "scenario_id": scenario,
                "time_step": time_step,
                "source": "results/legacy_payload_objective_l8",
                "graph_file": graph_path.relative_to(repo).as_posix(),
                "graph_sha256": _sha256(graph_path),
                "state_meta_sha256": _sha256(meta_path),
                "state_hash": item["state_hash"],
                "subset_hash": item["subset_hash"],
            }
        )
    return states, sources


def load_training_states(repo: Path) -> tuple[list[dict], dict]:
    """Return 28 states in scenario/time order and the source index. No model is loaded."""
    l2_states, l2_sources = _l2_states(repo)
    l8_states, l8_sources = _l8_states(repo)
    states = l2_states + l8_states
    order = sorted(range(len(states)), key=lambda index: (states[index]["scenario_id"], states[index]["time_step"]))
    if order != list(range(len(states))):
        raise RuntimeError("training concatenation is not scenario/time order")
    keys = [(int(state["scenario_id"]), int(state["time_step"])) for state in states]
    if len(set(keys)) != 28:
        raise RuntimeError("training states are not 28 distinct decisions")
    comparison = {(scenario, time_step) for scenario in COMPARISON_SCENARIOS for time_step in TIMES}
    if set(keys) & comparison:
        raise RuntimeError("training states overlap the new comparison")
    if any(scenario in EXCLUDED for scenario, _time in keys):
        raise RuntimeError("excluded explore states entered training")
    if sum(int(state["real_payload"].shape[0]) for state in states) != 3584:
        raise RuntimeError("training candidate count is not 3584")
    root = repo / "results"
    index = {
        "states": 28,
        "candidates": 3584,
        "graphs": 28,
        "order": [{"scenario_id": key[0], "time_step": key[1]} for key in keys],
        "sources": l2_sources + l8_sources,
        "l2_rows_sha256": _sha256(root / "legacy_arrival_aware_ab_l2" / "paired" / "rows.csv"),
        "l2_manifest_sha256": _sha256(root / "legacy_arrival_aware_ab_l2" / "manifest.json"),
        "l8_candidates_sha256": _sha256(root / "legacy_payload_objective_l8" / "candidates.json"),
        "l8_labels_sha256": _sha256(root / "legacy_payload_objective_l8" / "real_labels.jsonl"),
        "train_comparison_disjoint": True,
        "excluded_explore_scenarios": list(EXCLUDED),
        "environment_steps": 0,
        "scoring_forwards": 0,
        "relabelled_old_states": 0,
    }
    return states, index


def access_schedule(states: list[dict]) -> dict:
    """Fix 120 visits before any new comparison label exists."""
    if len(states) != 28:
        raise RuntimeError("access schedule requires 28 states")
    visits = []
    counts: dict[tuple[int, int], int] = {}
    for update in range(120):
        state = states[update % 28]
        key = (int(state["scenario_id"]), int(state["time_step"]))
        counts[key] = counts.get(key, 0) + 1
        visits.append({"update": update, "scenario_id": key[0], "time_step": key[1], "visit": counts[key]})
    keys = [(int(state["scenario_id"]), int(state["time_step"])) for state in states]
    if [counts[key] for key in keys[:8]] != [5] * 8 or [counts[key] for key in keys[8:]] != [4] * 20:
        raise RuntimeError("fixed visit counts are not 5 for the first 8 states and 4 for the other 20")
    return {
        "updates": 120,
        "full_passes": 4,
        "tail_updates": 8,
        "rule": "update i uses sorted state i mod 28",
        "per_state": [{"scenario_id": key[0], "time_step": key[1], "visits": counts[key]} for key in keys],
        "visits": visits,
        "r12_visits_per_state": 10,
    }
