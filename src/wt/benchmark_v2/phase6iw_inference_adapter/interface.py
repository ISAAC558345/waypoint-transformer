"""Isolated scheme-C scorer. It does not import or replace the official planner."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6is_assignment_head.audit import _load_decoder, _load_hgnn, _score_matching
from wt.benchmark_v2.phase6is_assignment_head.protocol import SCHEME_C_CKPT
from wt.benchmark_v2.phase6iw_inference_adapter.protocol import REPEAT_ATOL
from wt.htw_phase9.const import EDGE_UB_DIM

FORBIDDEN_INPUTS = ("j15_bits", "utility_rank", "future_sinr", "mappo_value", "served_bits")


@dataclass
class InferenceResult:
    scores: np.ndarray
    ranks: np.ndarray
    order: np.ndarray
    chosen_index: int
    chosen_assignment: list[int]
    seconds: float
    model_version: str
    checkpoint_sha256: str


class SchemeCAdapter:
    """Score a caller-supplied legal pool with the frozen scheme-C checkpoint."""

    model_version = "phase6if_scheme_c_epoch_001"

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.model = _load_hgnn(SCHEME_C_CKPT, self.device)
        self.decoder = _load_decoder(self.device)
        self.checkpoint_sha256 = ""

    def infer(self, graph: dict, assignments: np.ndarray) -> InferenceResult:
        self._reject_forbidden(graph)
        rows = np.asarray(assignments, dtype=np.int64)
        if rows.ndim != 2 or int(rows.shape[1]) != 6:
            raise ValueError("assignments must be [count, 6]")
        self._require_raw_edge(graph)
        started = time.perf_counter()
        sample = {"graph": graph, "assignments": rows}
        scores = _score_matching(self.model, self.decoder, sample, self.device)
        seconds = time.perf_counter() - started
        if scores.shape != (int(rows.shape[0]),) or not np.all(np.isfinite(scores)):
            raise RuntimeError("scheme-C scores are misaligned or non-finite")
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(order.shape[0], dtype=np.int64)
        ranks[order] = np.arange(1, order.shape[0] + 1)
        chosen = int(order[0])
        return InferenceResult(
            scores=scores,
            ranks=ranks,
            order=order,
            chosen_index=chosen,
            chosen_assignment=[int(item) for item in rows[chosen]],
            seconds=float(seconds),
            model_version=self.model_version,
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def repeat_gap(self, graph: dict, assignments: np.ndarray) -> float:
        first = self.infer(graph, assignments)
        second = self.infer(graph, assignments)
        return float(np.max(np.abs(first.scores - second.scores)))

    def _require_raw_edge(self, graph: dict) -> None:
        edge = np.asarray(graph["edge"])
        if int(edge.shape[-1]) != int(EDGE_UB_DIM):
            raise ValueError(f"edge width {int(edge.shape[-1])} is not the raw scheme-C contract")

    def _reject_forbidden(self, graph: dict) -> None:
        found = [name for name in FORBIDDEN_INPUTS if name in graph]
        if found:
            raise RuntimeError(f"scheme-C inference refused inputs {found}")


def assignment_is_legal(assignment: np.ndarray, n_uavs: int = 6, n_buoys: int = 12) -> bool:
    return bool(is_legal(np.asarray(assignment, dtype=np.int64), int(n_uavs), int(n_buoys)))


def g1_rate_scores(rate: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Sum stored hypothetical rates. This does not rebuild the candidate pool."""
    matrix = np.asarray(rate, dtype=np.float64)
    scale = float(np.max(matrix)) if matrix.size else 0.0
    normalized = np.zeros_like(matrix) if scale <= 0.0 else matrix / scale
    rows = np.asarray(assignments, dtype=np.int64)
    valid = rows >= 0
    safe = np.where(valid, rows, 0)
    uav = np.arange(rows.shape[1], dtype=np.int64)[None, :]
    return np.where(valid, normalized[uav, safe], 0.0).sum(axis=1)


def stable_choice(scores: np.ndarray, assignments: np.ndarray) -> tuple[int, list[int]]:
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("choice scores are non-finite")
    index = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    return index, [int(item) for item in np.asarray(assignments, dtype=np.int64)[index]]


__all__ = [
    "FORBIDDEN_INPUTS",
    "InferenceResult",
    "REPEAT_ATOL",
    "SchemeCAdapter",
    "assignment_is_legal",
    "g1_rate_scores",
    "stable_choice",
]
