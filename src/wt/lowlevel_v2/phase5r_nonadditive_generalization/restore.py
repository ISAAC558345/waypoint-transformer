"""Restore Phase-5Q weights. Phase-5Q did not write checkpoints.

The restoration repeats the same seed, data, and ten-epoch protocol, then checks
the recorded 310100/310101 scores. Scenarios 310102-310109 are not used here.
"""
from __future__ import annotations

import csv

import numpy as np
import torch

from wt.htw_phase9.const import BUOY_DIM
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.head import JointAssignmentHead
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5p_executor_joint_planner.run_phase5p import _eval_head, _train, load_states
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features, _eval_scorer, _train_scorer
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.scorer import AssignmentConditionedSetScorer
from wt.round30.ppo import seed_everything

from .protocol import CHECKPOINTS, PHASE5Q_PREDICTIONS, PHASE5Q_STATES, SEEDS


def _vector_key(seed: int, scenario: int, moment: int, vector: str) -> tuple:
    return int(seed), int(scenario), int(moment), str(vector)


def _recorded_scores() -> dict[tuple, float]:
    recorded = {}
    with PHASE5Q_PREDICTIONS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "model_c":
                continue
            recorded[_vector_key(int(row["seed"]), int(row["scenario_id"]), int(row["time_step"]), row["assignment_vector"])] = float(row["score"])
    if not recorded:
        raise RuntimeError("Phase-5Q did not record Model-C scores")
    return recorded


def _recorded_choices() -> dict[tuple, str]:
    choices = {}
    with PHASE5Q_STATES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in {"model_a", "model_c"}:
                continue
            choices[(row["model"], int(row["seed"]), int(row["scenario_id"]), int(row["time_step"]))] = row["assignment_vector"]
    return choices


def _check(heads, scorers, states, cached, embedded) -> None:
    scores = _recorded_scores()
    choices = _recorded_choices()
    worst = 0.0
    for seed in SEEDS:
        for state in states:
            additive = _eval_head(heads[int(seed)], cached[id(state)], state)
            scored = _eval_scorer(scorers[int(seed)], embedded[id(state)], state)
            key = ("model_a", int(seed), int(state["scenario_id"]), int(state["time_step"]))
            if additive["assignment_vector"] != choices[key]:
                raise RuntimeError(f"restored Model-A disagrees with Phase-5Q at {key}")
            key = ("model_c", int(seed), int(state["scenario_id"]), int(state["time_step"]))
            if scored["assignment_vector"] != choices[key]:
                raise RuntimeError(f"restored Model-C disagrees with Phase-5Q at {key}")
            for vector, score in zip(state["vectors"], scored["scores"]):
                recorded = scores[_vector_key(int(seed), int(state["scenario_id"]), int(state["time_step"]), vector)]
                worst = max(worst, abs(float(score) - float(recorded)))
    if worst > 1.0e-4:
        raise RuntimeError(f"restored Model-C scores differ from Phase-5Q by {worst}")
    print(f"restored checkpoints match Phase-5Q, max score gap {worst:.3e}", flush=True)


def load_frozen_models() -> tuple[dict, dict, JointUtilityHGNN]:
    """Return Model-A heads, Model-C scorers, and the frozen Phase-5L encoder."""
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = load_states()
    cached = {id(state): _features(frozen, matching, state["urgency_batch"]) for state in states}
    embedded = {id(state): _edge_features(frozen, state["urgency_batch"]) for state in states}
    heads = {}
    scorers = {}
    missing = False
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        head_path = CHECKPOINTS / f"model_a_seed{seed}.pt"
        scorer_path = CHECKPOINTS / f"model_c_seed{seed}.pt"
        if not head_path.is_file() or not scorer_path.is_file():
            missing = True
            break
        head = JointAssignmentHead()
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=False)["state_dict"])
        scorer = AssignmentConditionedSetScorer()
        scorer.load_state_dict(torch.load(scorer_path, map_location="cpu", weights_only=False)["state_dict"])
        head.eval()
        scorer.eval()
        heads[int(seed)] = head
        scorers[int(seed)] = scorer
    if missing:
        print("phase-5q checkpoints were not saved; restoring from the original protocol", flush=True)
        heads = {}
        scorers = {}
        for seed in SEEDS:
            seed_everything(int(seed))
            head = JointAssignmentHead()
            _train(head, cached, states, "j_mappo")
            heads[int(seed)] = head
            seed_everything(int(seed))
            scorer = AssignmentConditionedSetScorer()
            _train_scorer(scorer, embedded, states, int(seed))
            scorers[int(seed)] = scorer
        _check(heads, scorers, states, cached, embedded)
        for seed in SEEDS:
            torch.save({"seed": int(seed), "state_dict": heads[int(seed)].state_dict(), "source": "phase5q_protocol_restore"}, CHECKPOINTS / f"model_a_seed{seed}.pt")
            torch.save({"seed": int(seed), "state_dict": scorers[int(seed)].state_dict(), "source": "phase5q_protocol_restore"}, CHECKPOINTS / f"model_c_seed{seed}.pt")
    else:
        _check(heads, scorers, states, cached, embedded)
    for model in list(heads.values()) + list(scorers.values()):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return heads, scorers, frozen


__all__ = ["load_frozen_models"]
