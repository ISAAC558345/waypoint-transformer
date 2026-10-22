"""Legal task sources. Uniform draws stay on the training path."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from wt.benchmark_v2.phase6e.generators import build_pools, feature_matrices, is_legal, proposal_scores
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _key
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import additive_assignment, frozen_candidate_keys

from .protocol import EVAL_POLICIES, HELD_OUT_SOURCE, TRAIN_SOURCES


def _assignment_from_selection(selection) -> np.ndarray:
    if selection.status != "selected" or selection.assignment is None or selection.option is None:
        raise RuntimeError(selection.reason or selection.status)
    return np.asarray(selection.assignment, dtype=np.int64).copy()


def select_g2_distance(env) -> np.ndarray:
    pools, _packed, illegal = build_pools(env, (("G2_DISTANCE", 6),))
    if int(illegal) != 0:
        raise RuntimeError("ILLEGAL")
    raw = pools[("G2_DISTANCE", 6)]
    keys = sorted({_key(item) for item in raw})
    if not keys:
        raise RuntimeError("G2_DISTANCE pool is empty")
    scores = proposal_scores(feature_matrices(env))["G2_DISTANCE"]
    chosen = np.asarray(additive_assignment(keys, scores), dtype=np.int64)
    if not is_legal(chosen, int(env.config.num_uavs), int(env.config.num_buoys)):
        raise RuntimeError("ILLEGAL")
    return chosen


class TaskSelector:
    """Scheme C and G1_RATE use the formal planner. G2 is evaluation only."""

    def __init__(self) -> None:
        self.scheme = FormalPlanner("scheme_c")
        self.baseline = FormalPlanner("g1_rate")
        startup = self.scheme.startup
        payload = {
            "alpha": 0.25,
            "candidate_config_sha256": startup["candidate_config_sha256"],
            "eval_policies": list(EVAL_POLICIES),
            "held_out_source": HELD_OUT_SOURCE,
            "mappo_sha256": startup["mappo_sha256"],
            "reward": "served_bits / 1e6 - 0.05 * c_project - 0.01 * c_residual",
            "scheme_c_sha256": startup["scheme_c_sha256"],
            "train_sources": list(TRAIN_SOURCES),
        }
        self.config = payload
        self.config_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def choose(self, env, source: str, rng: np.random.Generator | None, position_prev: np.ndarray, predecessor: int | None) -> dict:
        name = str(source)
        if name == "scheme_c":
            pred = None if predecessor is None else int(predecessor)
            assignment = _assignment_from_selection(self.scheme.select_assignment(env, position_prev, pred))
        elif name == "g1_rate":
            pred = None if predecessor is None else int(predecessor)
            assignment = _assignment_from_selection(self.baseline.select_assignment(env, position_prev, pred))
        elif name == "uniform_g1_k6":
            if rng is None:
                raise RuntimeError("uniform tasks are training-only")
            keys, _raw = frozen_candidate_keys(env)
            if not keys:
                raise RuntimeError("candidate pool is empty")
            assignment = np.asarray(keys[int(rng.integers(0, len(keys)))], dtype=np.int64)
        elif name == HELD_OUT_SOURCE:
            assignment = select_g2_distance(env)
        else:
            raise RuntimeError(f"unknown task source {name}")
        if not is_legal(assignment, int(env.config.num_uavs), int(env.config.num_buoys)):
            raise RuntimeError("ILLEGAL")
        option = make_option(env, assignment)
        if not np.array_equal(option.association, option.targets):
            raise RuntimeError("INTERFACE")
        if not np.array_equal(option.association, assignment):
            raise RuntimeError("INTERFACE")
        return {"assignment": assignment, "option": option, "source": name}


__all__ = ["TaskSelector", "select_g2_distance"]
