"""One Phase-6I decision state for the unchanged JointUtilityHGNN training step.

The model batch is the Phase-5L graph before append_queue_edge. Labels stay
outside that batch. Candidate counts come from the revision shard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import hgnn_tensors
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import (
    GRAPH_KEYS,
    PRIMARY_TIMES,
    REVISION_DATASET,
    TRAIN_POOL,
    VALIDATION_POOL,
    assert_open,
)
from wt.htw_phase9.const import EDGE_UB_DIM
from wt.lowlevel_v2.phase5l_joint_encoder.model import append_queue_edge


def _split_of(scenario: int) -> str:
    if int(scenario) in TRAIN_POOL:
        return "train"
    if int(scenario) in VALIDATION_POOL:
        return "validation"
    raise RuntimeError(f"phase-6ic refuses scenario {scenario}")


def _paths(scenario: int, time_step: int) -> tuple[Path, Path, Path]:
    folder = REVISION_DATASET / _split_of(scenario) / f"scenario_{int(scenario)}"
    stem = f"t{int(time_step):03d}"
    return folder / f"{stem}_state.npz", folder / f"{stem}_candidates.npz", folder / f"{stem}_manifest.json"


def load_decision(scenario: int, time_step: int) -> dict:
    """Load one revision state and separate graph inputs from J15 labels."""
    assert_open(int(scenario), _split_of(scenario))
    if int(time_step) not in PRIMARY_TIMES:
        raise RuntimeError(f"phase-6ic refuses time {time_step}")
    state_path, candidate_path, manifest_path = _paths(scenario, time_step)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(state_path) as state_file:
        arrays = {key: np.asarray(state_file[key]) for key in state_file.files}
    with np.load(candidate_path) as candidate_file:
        assignments = np.asarray(candidate_file["assignment"], dtype=np.int64)
        labels = np.asarray(candidate_file["j15_bits"], dtype=np.float64)
        ranks = np.asarray(candidate_file["utility_rank"], dtype=np.int32)
    if int(assignments.shape[0]) != int(labels.shape[0]) or int(assignments.shape[0]) != int(manifest["candidate_count"]):
        raise RuntimeError("candidate rows and J15 labels are not aligned")
    if int(assignments.shape[1]) != 6:
        raise RuntimeError("assignment width is not the six-UAV joint action")
    built = hgnn_tensors(arrays, L3BBitQueueEnvV2().config)
    edge = np.asarray(built["edge"], dtype=np.float32)
    if int(edge.shape[-1]) != int(EDGE_UB_DIM) + 1:
        raise RuntimeError("assembled UAV-buoy edge is not the checkpoint width")
    graph = {
        "uav": np.asarray(built["uav"], dtype=np.float32)[None, ...],
        "buoy": np.asarray(built["buoy"], dtype=np.float32)[None, ...],
        "uav_mask": np.ones((1, built["uav"].shape[0]), dtype=bool),
        "buoy_mask": np.ones((1, built["buoy"].shape[0]), dtype=bool),
        "global": np.asarray(built["global"], dtype=np.float32)[None, ...],
        "edge": edge[None, ..., : int(EDGE_UB_DIM)],
        "edge_uu": np.asarray(built["edge_uu"], dtype=np.float32)[None, ...],
    }
    if tuple(graph) != GRAPH_KEYS:
        raise RuntimeError("graph batch contains a non-input field")
    conditioned = append_queue_edge(graph)
    if not np.allclose(conditioned["edge"], edge[None, ...], atol=1.0e-6, rtol=0.0):
        raise RuntimeError("Phase-5L queue-edge append does not match the assembled edge")
    if int(graph["uav"].shape[-1]) != 8 or int(graph["buoy"].shape[-1]) != 8:
        raise RuntimeError("node width does not match the checkpoint")
    if int(graph["edge_uu"].shape[-1]) != 2 or int(conditioned["edge"].shape[-1]) != 4:
        raise RuntimeError("edge width does not match the checkpoint")
    return {
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "split": _split_of(scenario),
        "graph": graph,
        "assignments": assignments,
        "j15_bits": labels,
        "utility_rank": ranks,
        "candidate_count": int(assignments.shape[0]),
    }


class DecisionDataset(Dataset):
    """Scenario-level index. Each item is one decision and all of its candidates."""

    def __init__(self, split: str) -> None:
        pool = {"train": TRAIN_POOL, "validation": VALIDATION_POOL}[split]
        self.split = split
        self.index = [(int(scenario), int(time_step)) for scenario in pool for time_step in PRIMARY_TIMES]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        scenario, time_step = self.index[int(item)]
        return load_decision(scenario, time_step)


def one_state_batch(rows: list[dict]) -> dict:
    """Keep the frozen decoder's one-decision contract. Do not stack states."""
    if len(rows) != 1:
        raise RuntimeError("JointAssignmentDecoder scores one decision at a time")
    return rows[0]


__all__ = ["DecisionDataset", "load_decision", "one_state_batch"]
