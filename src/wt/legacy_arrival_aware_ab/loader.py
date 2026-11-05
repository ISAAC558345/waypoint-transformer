"""Read the L2 paired tables without opening the historical dataset loader."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


GRAPH_KEYS = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
TRAIN_SCENARIOS = tuple(range(322016, 322022))
EXPLORE_SCENARIOS = (322022, 322023)
TIMES = (15, 30)


def resolve_graph(package_root: Path, graph_file: str, repo_root: Path) -> Path:
    """Resolve a package-relative path, or an existing L2 repository-relative path."""
    relative = Path(str(graph_file).replace("\\", "/"))
    if relative.is_absolute():
        raise RuntimeError(f"graph path must stay relative: {graph_file}")
    package_root = package_root.resolve()
    candidates = (package_root / relative, repo_root.resolve() / relative)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and package_root in resolved.parents:
            return resolved
    raise FileNotFoundError(f"{graph_file} is not a file inside {package_root}")


def _assignment(text: str) -> np.ndarray:
    parts = str(text).split("|")
    if len(parts) != 6:
        raise RuntimeError(f"assignment does not describe six UAVs: {text}")
    return np.asarray([int(part) for part in parts], dtype=np.int64)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _state_order(scenario: int, time_step: int) -> tuple[int, int]:
    return int(scenario), 0 if int(time_step) == 15 else 1


def load_package(package_root: Path, repo_root: Path) -> dict[str, list[dict]]:
    rows = _read_rows(package_root / "paired" / "rows.csv")
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((int(row["scenario_id"]), int(row["time_step"])), []).append(row)
    expected = {(scenario, time_step) for scenario in TRAIN_SCENARIOS + EXPLORE_SCENARIOS for time_step in TIMES}
    if set(grouped) != expected:
        raise RuntimeError(f"package states are {sorted(grouped)}, not the fixed 16 states")
    states = {"train": [], "explore_eval": []}
    for key in sorted(grouped, key=lambda item: _state_order(*item)):
        group_rows = grouped[key]
        if len(group_rows) != 128:
            raise RuntimeError(f"{key} has {len(group_rows)} rows")
        scenario, time_step = key
        split = group_rows[0]["group"]
        if scenario in TRAIN_SCENARIOS and split != "train":
            raise RuntimeError(f"{key} is not in the training group")
        if scenario in EXPLORE_SCENARIOS and split != "explore_eval":
            raise RuntimeError(f"{key} is not in the explore-eval group")
        if any(row["group"] != split for row in group_rows):
            raise RuntimeError(f"{key} mixes groups")
        graph_files = {row["graph_file"] for row in group_rows}
        if len(graph_files) != 1:
            raise RuntimeError(f"{key} does not share one graph")
        graph_path = resolve_graph(package_root, next(iter(graph_files)), repo_root)
        with np.load(graph_path) as payload:
            if tuple(payload.files) != GRAPH_KEYS:
                raise RuntimeError(f"{graph_path.name} keys are {payload.files}")
            graph = {name: np.asarray(payload[name]) for name in GRAPH_KEYS}
        candidate_ids = [int(row["candidate_id"]) for row in group_rows]
        if len(set(candidate_ids)) != 128:
            raise RuntimeError(f"{key} repeats a candidate id")
        state = {
            "scenario_id": scenario,
            "time_step": time_step,
            "group": split,
            "graph": graph,
            "graph_file": next(iter(graph_files)).replace("\\", "/"),
            "assignments": np.stack([_assignment(row["assignment"]) for row in group_rows]),
            "candidate_id": np.asarray(candidate_ids, dtype=np.int64),
            "row_index": np.asarray([int(row["row_index"]) for row in group_rows], dtype=np.int64),
            "local_position": np.arange(128, dtype=np.int64),
            "zero_payload": np.asarray([float(row["zero_payload_bits"]) for row in group_rows], dtype=np.float64),
            "real_payload": np.asarray([float(row["real_payload_bits"]) for row in group_rows], dtype=np.float64),
            "assignment_text": [str(row["assignment"]) for row in group_rows],
        }
        if np.array_equal(state["candidate_id"], state["local_position"]):
            raise RuntimeError(f"{key} collapsed candidate ids onto local positions")
        if not np.all(np.isfinite(state["zero_payload"])) or not np.all(np.isfinite(state["real_payload"])):
            raise RuntimeError(f"{key} has a non-finite payload label")
        states[split].append(state)
    expected_train = [(scenario, time_step) for scenario in TRAIN_SCENARIOS for time_step in TIMES]
    expected_explore = [(scenario, time_step) for scenario in EXPLORE_SCENARIOS for time_step in TIMES]
    if [(item["scenario_id"], item["time_step"]) for item in states["train"]] != expected_train:
        raise RuntimeError("training order is not scenario ascending with t15 before t30")
    if [(item["scenario_id"], item["time_step"]) for item in states["explore_eval"]] != expected_explore:
        raise RuntimeError("explore order is not scenario ascending with t15 before t30")
    if len(states["train"]) != 12 or len(states["explore_eval"]) != 4:
        raise RuntimeError("split sizes are not 12 training states and 4 explore states")
    return states
